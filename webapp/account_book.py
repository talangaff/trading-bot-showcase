"""Who owns what on a shared Alpaca account.

Many bots trade on the same paper account (the whole fleet on the fleet
keys; every custom bot of one user on that user's keys), and two of them
can easily hold the very same contract at the same time — a bot only ever
sees its own ledger, but Alpaca sees one merged position per symbol. This
module is the layer in between:

  * claims()      — per account, per OCC symbol, how many contracts each bot
                    says it holds (a vertical spread package is expanded
                    into its long leg, +qty, and its short leg, -qty).
  * snapshot()    — what the account actually holds right now (cached for a
                    cycle; every real fill adjusts it in place via adjust()).
  * can_afford()  — before a bot buys, whether the account has the options
                    buying power to pay for it. Alpaca cannot be funded
                    through the API (there is no such endpoint — a paper
                    balance is only changed by the dashboard's Reset, which
                    wipes history and invalidates the keys), so running
                    short is handled by REFUSING the entry, deterministically
                    and visibly, rather than by firing an order that Alpaca
                    rejects. Left unchecked, those rejections land on
                    whichever bots happen to tick first, which would bias the
                    fitness data by pure ordering luck.
  * sell_cap()    — before a bot sells, the most it may sell: never more than
                    its own claim, never more than the account holds. So a
                    bot whose ledger has drifted can't sell another bot's
                    contracts out from under it.
  * reconcile()   — the sum of claims vs. the account, per symbol, saved in
                    fleet_meta and shown on the Bot Fleet / Custom Bots
                    pages, plus an engine event when something doesn't add
                    up. Never "fixes" a mismatch on its own except for the
                    one unambiguous case below.
  * resolve_vanished() — a bot's position whose contract the account no
                    longer holds AT ALL (closed by hand in the Alpaca UI,
                    expired away, or sold by a broker-side liquidation) is
                    closed in the ledger at its last mark, flagged, so the
                    bot isn't stuck managing a phantom forever. Only when
                    the account holds zero of it (with several bots holding
                    the same contract, a partial shortfall can't be
                    attributed to any one of them) and only once the
                    position is a few minutes old (a fill Alpaca hasn't
                    reflected in /positions yet is not a vanished position).
"""

import json
import time
from datetime import datetime, timedelta, timezone

from . import bots, broker, option_model

_SNAP: dict[str, tuple[float, dict]] = {}     # account key -> (ts, {symbol: qty})
_SNAP_TTL = 45.0                              # one cycle
VANISHED_MIN_AGE_MIN = 5.0

_BP: dict[str, tuple[float, float]] = {}      # account key -> (ts, options buying power)
_BP_TTL = 45.0                                # one cycle, like the position snapshot
# Never commit the last sliver of the account: an entry is priced at the ask
# the bot saw, and the real fill can come in slightly worse.
BP_BUFFER = 0.05


# ---------------------------------------------------------------------------
# ledger side
# ---------------------------------------------------------------------------
def legs(contract: str, qty: float) -> dict[str, float]:
    """{OCC symbol: signed contracts} a ledger position of `qty` of
    `contract` corresponds to on the account."""
    info = option_model.parse_occ(contract)
    if not info:
        return {contract.upper(): float(qty)}
    if info.get("short_strike") is None:
        return {contract.upper(): float(qty)}
    exp = info["expiry"]
    long_sym = option_model.occ_symbol(info["underlying"], exp, info["is_call"], info["strike"])
    short_sym = option_model.occ_symbol(info["underlying"], exp, info["is_call"], info["short_strike"])
    return {long_sym: float(qty), short_sym: -float(qty)}


def long_leg(contract: str) -> str:
    """The symbol whose account holding limits a sell of `contract`."""
    return next(iter(legs(contract, 1)))


def _account_owner(bot_list: list[dict]):
    """The account an event is about belongs to one user — every bot on a
    set of keys has the same owner, since keys are per account."""
    for b in bot_list or ():
        if b.get("user_id") is not None:
            return int(b["user_id"])
    return None


def accounts() -> dict[str, tuple[tuple[str, str], list[dict]]]:
    """{account key: (keys, [bots])} for every bot that trades for real on
    some account — grouped by broker.account_key so every bot on the same
    keys lands in the same bucket. Bots with modeled execution or no keys
    are left out (nothing of theirs is on any account).

    With a fleet per user this is many accounts, not one: a user's fleet and
    baseline bots under their fleet keys, their custom bots under their
    custom keys, and each other user's the same. Reconciliation walks them
    independently — nothing is ever compared across two people's accounts."""
    out: dict[str, tuple[tuple[str, str], list[dict]]] = {}
    for b in bots.list_bots():
        mode, keys = broker.execution(b)
        if mode != "broker" or not keys:
            continue
        ak = broker.account_key(keys)
        out.setdefault(ak, (keys, []))[1].append(b)
    return out


def bots_by_account() -> dict[str, list[dict]]:
    """{account key: [bots]} — accounts() without the keys."""
    return {ak: bl for ak, (_k, bl) in accounts().items()}


def keys_for_account(account_key: str) -> tuple[str, str] | None:
    entry = accounts().get(account_key)
    return entry[0] if entry else None


def claims(bot_list: list[dict]) -> dict[str, dict[int, float]]:
    """{OCC symbol: {bot id: signed contracts}} across `bot_list`'s open
    ledger positions."""
    out: dict[str, dict[int, float]] = {}
    for b in bot_list:
        for contract, p in bots._positions(b["id"]).items():
            for sym, q in legs(contract, p["qty"]).items():
                out.setdefault(sym, {})
                out[sym][b["id"]] = out[sym].get(b["id"], 0.0) + q
    return out


def claimed_total(cl: dict[str, dict[int, float]]) -> dict[str, float]:
    return {sym: sum(v.values()) for sym, v in cl.items()}


# ---------------------------------------------------------------------------
# account side
# ---------------------------------------------------------------------------
def snapshot(keys: tuple[str, str], fresh: bool = False) -> dict[str, float] | None:
    """{symbol: signed qty} actually held on the account, cached for about
    one cycle. None if Alpaca couldn't be asked — callers must treat that
    as "unknown", never as "flat"."""
    ak = broker.account_key(keys)
    if not ak:
        return None
    hit = _SNAP.get(ak)
    now = time.time()
    if not fresh and hit and now - hit[0] < _SNAP_TTL:
        return hit[1]
    try:
        pos = broker.positions(keys)
    except broker.BrokerError as e:
        print("account_book positions {}: {}".format(ak[:6], e), flush=True)
        return hit[1] if hit and now - hit[0] < 600 else None
    _SNAP[ak] = (now, pos)
    return pos


def adjust(keys: tuple[str, str], contract: str, qty_delta: float) -> None:
    """Apply a fill we just made to the cached snapshot, so later bots in
    the same cycle see the account as it is now rather than as it was."""
    ak = broker.account_key(keys)
    hit = _SNAP.get(ak) if ak else None
    if not hit:
        return
    pos = dict(hit[1])
    for sym, q in legs(contract, qty_delta).items():
        pos[sym] = pos.get(sym, 0.0) + q
        if abs(pos[sym]) < 1e-9:
            pos.pop(sym, None)
    _SNAP[ak] = (hit[0], pos)


def forget(keys: tuple[str, str] | None = None) -> None:
    if keys is None:
        _SNAP.clear()
        _BP.clear()
    else:
        ak = broker.account_key(keys) or ""
        _SNAP.pop(ak, None)
        _BP.pop(ak, None)


# ---------------------------------------------------------------------------
# buying power
# ---------------------------------------------------------------------------
def buying_power(keys: tuple[str, str], fresh: bool = False) -> float | None:
    """Options buying power on the account, cached for about one cycle.
    None if Alpaca couldn't be asked — treated as "unknown", never as zero,
    so a data hiccup can never halt the whole fleet.

    Long options (everything these bots trade — buy-to-open only, and a
    vertical is a debit spread) cost their full premium, so this is the
    number an entry has to fit inside."""
    ak = broker.account_key(keys)
    if not ak:
        return None
    hit = _BP.get(ak)
    now = time.time()
    if not fresh and hit and now - hit[0] < _BP_TTL:
        return hit[1]
    try:
        acct = broker.account(keys)
    except broker.BrokerError as e:
        print("account_book buying_power {}: {}".format(ak[:6], e), flush=True)
        return hit[1] if hit and now - hit[0] < 600 else None
    bp = acct.get("options_buying_power")
    if bp is None:                    # older API shape: fall back to plain cash
        bp = acct.get("cash") if acct.get("cash") is not None else acct.get("buying_power")
    if bp is None:
        return None
    _BP[ak] = (now, float(bp))
    return float(bp)


def can_afford(keys: tuple[str, str], cost: float) -> tuple[bool, float | None]:
    """(ok, available) — whether `cost` of premium fits in the account's
    options buying power, keeping BP_BUFFER back. Unknown buying power is
    always ok: Alpaca's own rejection is the backstop, and refusing to trade
    because a status call failed would be worse than a rejected order."""
    bp = buying_power(keys)
    if bp is None:
        return True, None
    return (cost <= bp * (1.0 - BP_BUFFER) + 1e-6), bp


def charge(keys: tuple[str, str], delta: float) -> None:
    """Apply a fill to the cached buying power (negative for a buy, positive
    for a sell), so later bots in the same cycle see what is actually left
    rather than what was there when the cycle started."""
    ak = broker.account_key(keys)
    hit = _BP.get(ak) if ak else None
    if not hit:
        return
    _BP[ak] = (hit[0], max(0.0, hit[1] + delta))


# ---------------------------------------------------------------------------
# capital headroom (for the fleet / custom pages)
# ---------------------------------------------------------------------------
def capital_summary(rows: list[dict], position_pct_limit: float,
                    account: dict | None) -> dict:
    """How the bots' virtual capital lines up against the real account.

    `rows` are leaderboard rows (each carrying `balance` and `open_cost`).
    Three different numbers, easily confused:

      * committed  — the sum of the bots' virtual balances. Pure bookkeeping;
        this never needs to exist as cash anywhere.
      * max_exposure — the most premium every bot could hold AT ONCE, since
        each bot's total open exposure is capped at `position_pct_limit` of
        its own balance. THIS is the number the real account has to fund.
      * open_cost — what they actually hold right now.
    """
    live = [r for r in rows if r.get("status") == "active"]
    committed = sum(r.get("balance") or 0.0 for r in live)
    # each bot's own cap if the row carries one (a custom bot may run up to 100%
    # of its balance), else the account-wide limit
    max_exposure = sum(
        (r.get("balance") or 0.0)
        * (r["position_pct"] if r.get("position_pct") is not None else position_pct_limit) / 100.0
        for r in live)
    open_cost = sum(r.get("open_cost") or 0.0 for r in live)
    equity = (account or {}).get("equity")
    bp = (account or {}).get("options_buying_power")
    out = {
        "bots": len(live), "committed": committed, "max_exposure": max_exposure,
        "open_cost": open_cost, "equity": equity, "options_buying_power": bp,
        "short_by": None, "oversubscribed": False,
        "utilization": None,
    }
    if bp is not None and bp > 0:
        out["utilization"] = open_cost / bp * 100.0
    fundable = bp if bp is not None else equity
    if fundable is not None and max_exposure > fundable:
        out["oversubscribed"] = True
        out["short_by"] = max_exposure - fundable
    return out


def sell_cap(keys: tuple[str, str], contract: str, qty: int) -> int:
    """The most a bot may sell of `contract` right now: its request, capped
    at what the account actually holds of the position's long leg. Unknown
    account state (Alpaca unreachable) leaves the request alone — Alpaca's
    own sell_to_close check is the final backstop against going short."""
    snap = snapshot(keys)
    if snap is None:
        return int(qty)
    held = snap.get(long_leg(contract), 0.0)
    return max(0, min(int(qty), int(held)))


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------
def _meta_key(account_key: str) -> str:
    return "reconcile:" + account_key


def reconcile(keys: tuple[str, str], bot_list: list[dict], *, fresh: bool = True) -> dict:
    """Compare every bot's claims on this account with what it holds.
    Returns and stores {ts, ok, error, mismatches:[{symbol, ledger,
    account, bots:{id: qty}}], unclaimed:[{symbol, account}], symbols}."""
    ak = broker.account_key(keys) or "?"
    cl = claims(bot_list)
    totals = claimed_total(cl)
    snap = snapshot(keys, fresh=fresh)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if snap is None:
        report = {"ts": stamp, "ok": False, "error": "could not read the account's positions",
                  "mismatches": [], "unclaimed": [], "symbols": len(totals), "bots": len(bot_list)}
        bots._meta_set(_meta_key(ak), json.dumps(report))
        return report
    mism, unclaimed = [], []
    for sym, ledger_q in sorted(totals.items()):
        acct_q = snap.get(sym, 0.0)
        if abs(acct_q - ledger_q) > 1e-9:
            mism.append({"symbol": sym, "ledger": ledger_q, "account": acct_q,
                         "bots": {str(bid): q for bid, q in cl[sym].items()}})
    for sym, acct_q in sorted(snap.items()):
        if sym not in totals:
            unclaimed.append({"symbol": sym, "account": acct_q})
    report = {"ts": stamp, "ok": not mism and not unclaimed, "error": None,
              "mismatches": mism, "unclaimed": unclaimed,
              "symbols": len(totals), "bots": len(bot_list)}
    bots._meta_set(_meta_key(ak), json.dumps(report))
    if mism or unclaimed:
        bots.log_event_once("reconcile", {"account": ak[:6] + "…",
                                          "user_id": _account_owner(bot_list),
                                          "mismatches": len(mism),
                                          "unclaimed": len(unclaimed),
                                          "detail": (mism + unclaimed)[:8]},
                            key="reconcile:{}:{}:{}".format(ak[:6], len(mism), len(unclaimed)))
    return report


def last_report(keys: tuple[str, str] | None) -> dict | None:
    ak = broker.account_key(keys)
    if not ak:
        return None
    raw = bots._meta_get(_meta_key(ak))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def resolve_vanished(keys: tuple[str, str], bot_list: list[dict], now: datetime | None = None,
                     min_age_min: float = VANISHED_MIN_AGE_MIN) -> int:
    """Close, in the ledger only, any position whose contract the account no
    longer holds at all (see the module docstring for why only that case).
    Booked at the last mark (else average cost), forced, flagged. Returns
    the number of positions resolved."""
    snap = snapshot(keys)
    if snap is None:
        return 0
    now = now or datetime.now(timezone.utc)
    n = 0
    for b in bot_list:
        state = bots.pos_state(b["id"])
        for contract, p in bots._positions(b["id"]).items():
            if any(abs(snap.get(sym, 0.0)) > 1e-9 for sym in legs(contract, 1)):
                continue
            ss = state.get(contract) or {}
            opened = ss.get("opened_ts")
            try:
                od = datetime.fromisoformat(opened) if opened else None
            except ValueError:
                od = None
            if od is None:
                # no scratch state: fall back to the fill's own timestamp
                try:
                    od = max(datetime.fromisoformat(t["ts"]) for t in bots._trades(b["id"])
                             if t["contract"] == contract and t["side"] == "buy_to_open")
                except (ValueError, TypeError):
                    od = None
            if od is not None:
                if od.tzinfo is None:
                    od = od.replace(tzinfo=timezone.utc)
                if now - od < timedelta(minutes=min_age_min):
                    continue
            px = ss.get("last_mark") or p["avg_price"]
            err = bots.record_trade(
                b["id"], contract, "sell_to_close", int(p["qty"]), max(0.01, float(px)),
                symbol=bots._contract_symbol(contract), ts=now.isoformat(), forced=True,
                fill_source="model",
                note="reconciled: no longer held on the account (closed outside the bot) — "
                     "booked at last mark")
            if not err:
                bots.clear_pos_state(b["id"], contract)
                bots.log_event_once("reconcile", {"bot": b["id"], "user_id": b.get("user_id"),
                                                  "contract": contract,
                                                  "qty": int(p["qty"]), "resolved": "vanished"},
                                    key="vanished:{}:{}".format(b["id"], contract), within_s=60)
                n += 1
    return n


ORPHAN_MIN_AGE_S = 50.0        # unclaimed for more than one 1-minute cycle before we act
ORPHAN_MAX_QTY = 100           # sanity cap: a bigger "orphan" is a human position, not residue
ORPHAN_MAX_PER_CYCLE = 5


def _bot_origin_symbols(bot_list: list[dict], prefix: str) -> set[str]:
    """Every OCC leg symbol (spread packages expanded) that any of these bots
    has ever traded and that starts with `prefix` (root + expiry + C/P)."""
    ids = [b["id"] for b in bot_list]
    if not ids:
        return set()
    q = ("SELECT DISTINCT contract FROM bot_trades WHERE bot_id IN ({}) AND contract LIKE ?"
         .format(",".join("?" * len(ids))))
    out: set[str] = set()
    with bots._conn() as c:
        for r in c.execute(q, ids + [prefix + "%"]).fetchall():
            out.update(legs(r["contract"], 1))
    return out


def close_orphans(keys: tuple[str, str], bot_list: list[dict], now: datetime | None = None) -> int:
    """Close, FOR REAL on the account, an option position that no bot claims.

    These are residue, e.g. the short leg of a spread whose long leg another
    bot sold from under it. Left alone they sit on the account unmanaged
    (and a short can be assigned at expiry). Deliberately conservative, since
    this places real orders on an account someone may also use by hand:

      * only while the regular session is open and the contract has not
        expired (a closed market or dead contract cannot be traded);
      * only when the engine is switched on and no bot has an order in flight
        (an unbooked fill must not be mistaken for an orphan);
      * only a position that has stayed unclaimed for ORPHAN_MIN_AGE_S, so a
        fill the ledger has not caught up with yet is left alone;
      * only a contract that one of THESE bots has traded before: anything
        else on the account is somebody else's and is never touched;
      * long -> sell_to_close, short -> buy_to_close, market, quantity as held.
    Returns the number of positions closed."""
    from . import bot_settings, market_calendar
    if not bot_settings.engine_enabled():
        return 0
    now = now or datetime.now(timezone.utc)
    now_et = now.astimezone(option_model.ET)
    bounds = market_calendar.session_bounds(now_et.date())
    ak = broker.account_key(keys) or "?"
    mkey = "orphans:" + ak
    if bounds is None:
        return 0
    t = now_et.hour * 60 + now_et.minute
    if not (bounds[0] + 1 <= t < bounds[1] - 1):
        return 0
    if bots.pending_orders():
        return 0
    snap = snapshot(keys, fresh=True)
    if snap is None:
        return 0
    totals = claimed_total(claims(bot_list))
    orphans = {sym: q for sym, q in snap.items()
               if abs(q) > 1e-9 and abs(totals.get(sym, 0.0)) < 1e-9}

    try:
        seen = json.loads(bots._meta_get(mkey) or "{}")
    except ValueError:
        seen = {}
    stamp = now.isoformat(timespec="seconds")
    kept = {}
    for sym, q in orphans.items():
        prev = seen.get(sym)
        kept[sym] = prev if prev and abs(prev.get("q", 0) - q) < 1e-9 else {"q": q, "ts": stamp}
    bots._meta_set(mkey, json.dumps(kept))

    closed = 0
    for sym, q in sorted(orphans.items()):
        if closed >= ORPHAN_MAX_PER_CYCLE:
            break
        try:
            age = (now - datetime.fromisoformat(kept[sym]["ts"])).total_seconds()
        except (ValueError, TypeError):
            continue
        if age < ORPHAN_MIN_AGE_S:
            continue
        info = option_model.parse_occ(sym)
        if not info or abs(q - round(q)) > 1e-9 or not (0 < abs(q) <= ORPHAN_MAX_QTY):
            continue
        if option_model.minutes_to_expiry(info["expiry"], now_et) < 1.0:
            continue
        if sym not in _bot_origin_symbols(bot_list, sym[:-8]):
            continue
        qty = int(abs(round(q)))
        side, intent = ("sell", "sell_to_close") if q > 0 else ("buy", "buy_to_close")
        detail = {"symbol": sym, "account_qty": q, "action": intent, "qty": qty,
                  "user_id": _account_owner(bot_list)}
        try:
            filled, px, oid, err = broker.submit_and_wait(
                sym, qty, side, "market", position_intent=intent, keys=keys,
                client_order_id=broker.new_client_order_id(0, side))
        except broker.BrokerError as e:
            filled, px, oid, err = 0, None, None, str(e)
        if filled:
            adjust(keys, sym, -int(filled) if q > 0 else int(filled))
            bots.log_event("orphan_close", {**detail, "filled": int(filled), "price": px, "order": oid})
            kept.pop(sym, None)
            closed += 1
        else:
            bots.log_event_once("orphan_close", {**detail, "failed": err},
                                key="orphan_fail:{}:{}".format(ak[:6], sym), within_s=600)
    bots._meta_set(mkey, json.dumps(kept))
    return closed


def sync_all(now: datetime | None = None) -> dict:
    """One pass over every account any bot trades on — every user's fleet
    account and every user's custom-bot account — reconciling each, then
    resolving vanished positions. Called once per engine cycle."""
    out = {"accounts": 0, "mismatches": 0, "unclaimed": 0, "vanished": 0,
           "orphans_closed": 0, "errors": 0}
    for ak, (keys, bot_list) in accounts().items():
        out["accounts"] += 1
        rep = reconcile(keys, bot_list)
        if not rep["ok"] and rep.get("error"):
            out["errors"] += 1
            continue
        out["mismatches"] += len(rep["mismatches"])
        out["unclaimed"] += len(rep["unclaimed"])
        out["vanished"] += resolve_vanished(keys, bot_list, now)
        try:
            out["orphans_closed"] += close_orphans(keys, bot_list, now)
        except Exception as e:      # cleanup must never break the reconcile pass
            out["errors"] += 1
            print("close_orphans {}: {}".format(ak[:6], e), flush=True)
    return out
