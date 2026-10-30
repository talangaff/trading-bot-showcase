"""Autonomous live-trading engine for the 0DTE/swing options bot fleet.

Runs on a systemd timer:

    python -m webapp.bot_engine cycle   # every minute during the trading day
    python -m webapp.bot_engine daily   # once pre-open: seed / evolve / prune
    python -m webapp.bot_engine once    # single tick (manual / testing)

One cycle serves every user. The engine pulls 1-minute bars ONCE (Alpaca on
the best entitlement anyone has connected — bot_settings.data_source — else
yfinance), builds one feature context per symbol, records it all into the
shared market history, and then walks each user's fleet in turn. Every bot
trades for REAL on an Alpaca paper account (see webapp/broker.py): a user's
fleet and baseline bots on that USER's own fleet keys, their custom bots
(pool='custom') on their own separate custom keys. No bot ever trades on
keys that are not its owner's, and there is no modeled-fill mode to run
without an account: a fleet with no keys does not trade. Only the ACTUAL
fill Alpaca reports is ever booked, never a modeled price, so every bot's
$10k ledger, and therefore the fitness evolution breeds and culls on, is
built from live paper fills — all of them, with no exception a setting can
introduce. (Modeled fills still exist offline, in the replay harness, on
its own in-memory ledger.) Every guardrail (Buy-to-Open/Sell-to-Close
only, min contracts, position cap, drawdown disqualification) applies
exactly the same, enforced inside bots.record_trade().

Many bots share one account, so a contract can be held by several bots at
once. Each bot's ledger tracks only its own contracts, a sell is capped at
that bot's own claim AND at what the account actually holds
(account_book.sell_cap), every real order is journaled before it goes out
(bots.pending_orders → _recover_pending) and the ledgers are reconciled
against the account once per cycle (account_book.sync_all).

A fixed set of pool='baseline' bots (see bots.ensure_baseline_bots()) run
alongside the fleet: a pure-random-signal bot and an always-long
buy-and-hold-to-close bot, per symbol. They are never bred, mutated, or
culled by evolve()/daily_maintenance() below — they exist purely as the
null-hypothesis floor the evolved fleet has to actually beat (see
bots.fleet_stats()['beats_baseline']) for its numbers to mean anything.

Each bot's genome carries a `style` gene: "0dte" (the original mode — expiry
is always today's close, force-flattened same day) or "swing" (a real listed
expiry weeks out, held across multiple days up to swing_max_hold_days).
_manage_position() only applies the hard 15:55 ET flatten to a position
whose OWN contract actually expires today — a swing position rides past
that; _settle_expired() runs at the top of every cycle regardless of market
phase or the engine on/off switch to force-settle at intrinsic value
anything (0DTE or swing) still open past ITS contract's real expiry, as a
hard backstop either way.

Evolution is the one thing that is not per-user. Genomes live in one shared
population (webapp/gene_pool.py): a genome's track record is the union of
every user's trades on it, so it is proven — or culled — on a sample that
grows with the number of fleets rather than with calendar days; and no two
fleets are handed the same candidate at once, so N users search N times as
much of the space per market day. What stays private is everything that
matters financially: each user's bots, ledgers, capital, keys and orders.

Lifecycle:
  * A bot that hits the 25% drawdown is force-flattened and flagged failing.
  * evolve() archives it if it had ever been up >= 50% (kept forever), then
    replaces it with a lightly-mutated copy of the current best genome.
  * daily() also spawns mutated children of the top performers to retest them,
    and retires chronic underperformers to hold the fleet near its target.
  * Every one of those spawns goes through bots.spawn_bot(), which refuses
    past the fleet's max_fleet_bots guardrail — so the target is where the
    fleet aims and the cap is what it can never pass. daily() therefore
    retires before it breeds: at the cap, children need the room first.
"""

import fcntl
import math
import random
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

from . import (account_book, accounts, bots, bot_settings, broker, config, features, gene_pool, incubator,
               genome, market_calendar, market_data, notify, option_chain, option_model,
               option_quotes, recorder, strategies, watchdog)

ET = option_model.ET
SYMBOLS = genome.SYMBOLS

# Superseded by last_0dte_entry_min(), which derives the cutoff from the
# session's real close. Kept as the documented fallback shape only.
NO_NEW_TRADES_AFTER = 15 * 60 + 30      # ET minutes — the old fixed 15:30
MIN_MINUTES_LEFT = 20                   # need at least this much time value
SIGNAL_FLOOR = 0.15
# a run_cycle() stops starting new ticks once this much wall-clock has gone
# by: real orders wait for their fills, and a burst of entries must never
# push a cycle into the timer's next firing (or the unit's timeout)
CYCLE_BUDGET_S = float(getattr(config, "BOT_CYCLE_BUDGET_S", 45.0))

# The fast lane: custom bots are stepped by their own long-running loop
# (`python -m webapp.bot_engine fast`) every few seconds on short-lived data
# caches and with quick order polling, instead of riding the fleet's
# once-a-minute cycle. The fleet cycle only steps custom bots itself when the
# lane has stopped beating — a position must never go unmanaged because a
# service died.
FAST_INTERVAL_S = float(getattr(config, "BOT_FAST_INTERVAL_S", 5.0))
FAST_DATA_TTL_S = float(getattr(config, "BOT_FAST_DATA_TTL_S", 3.0))
HOLD_INTERVAL_S = float(getattr(config, "BOT_HOLD_INTERVAL_S", 1.5))   # pass cadence while a custom bot holds a position
HOLD_DATA_TTL_S = float(getattr(config, "BOT_HOLD_DATA_TTL_S", 1.0))   # ...and how old its quotes may be then
FAST_POLL_S = 0.25                  # order-fill polling for custom bots (fleet: 1 s)
FAST_ALIVE_S = 60.0                 # a beat older than this and the cycle takes over
FAST_BEAT_KEY = "fast_lane_beat"

# Evolution knobs. Breeding from a pool of top bots (not just #1) and giving
# fresh bots a grace window before they can be culled keeps off-lineage
# variants — and whole alternative strategies — alive long enough to prove out.
BREED_POOL = 8                         # per FAMILY: pick a parent at random from its top N
NEWBORN_GRACE_HOURS = 24               # protect fresh bots from being culled
IMMIGRANT_RATE = 0.15                  # fraction of new bots that are fresh random genomes
CROSSOVER_RATE = 0.5                   # of bred children, share made by crossover of two parents
MAX_STRATEGY_SHARE = 0.40             # don't breed from a strategy already past this share


# ---------------------------------------------------------------------------
# ledger abstraction
# ---------------------------------------------------------------------------
class LiveBook:
    """The real SQLite ledger (webapp/bots.py) behind the live engine.

    _step_bot()/_manage_position() do all their bookkeeping through a
    `book` with this interface, so the replay harness (webapp/replay.py)
    can run the exact same entry / exit / position-management code against
    an in-memory ledger over recorded sessions. Anything the trading logic
    needs from the ledger goes through here — never straight to `bots`."""

    def positions(self, bot_id):
        return bots._positions(bot_id)

    def pos_state(self, bot_id):
        return bots.pos_state(bot_id)

    def touch_pos_state(self, bot_id, contract, price, opened_ts, entry_iv=None, entry_mid=None):
        bots.touch_pos_state(bot_id, contract, price, opened_ts, entry_iv, entry_mid)

    def clear_pos_state(self, bot_id, contract):
        bots.clear_pos_state(bot_id, contract)

    def mark_pos(self, bot_id, contract, **flags):
        bots.mark_pos(bot_id, contract, **flags)

    def balance(self, bot_id) -> float:
        return bots._equity(bot_id)["balance"]

    def guardrails(self, bot) -> dict:
        return bots.effective_guardrails(bot)

    def record_trade(self, bot_id, contract, side, qty, price, **kw):
        return bots.record_trade(bot_id, contract, side, qty, price, **kw)

    def execution(self, bot) -> tuple[str, tuple[str, str] | None]:
        """('broker', keys) — real order; ('model', None) — modeled fill.
        The replay harness's SimBook has no such method, so _record_fill
        treats it as modeled: replay never touches an account."""
        return broker.execution(bot)


LIVE = LiveBook()


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------
# minutes either side of the real close: the last stretch of the session is
# for flattening rather than opening, and the engine keeps managing briefly
# past the bell so an expiring position is settled rather than abandoned.
FLATTEN_BEFORE_CLOSE_MIN = 5
MANAGE_AFTER_CLOSE_MIN = 5


def _phase(now: datetime) -> str:
    """Where in the trading day we are, according to the actual NYSE
    calendar rather than the clock alone.

    A weekday check cannot see a holiday, and a hardcoded 16:00 cannot see
    an early close: on the ~3 half days a year the market shuts at 13:00,
    the old version kept the fleet trading until 15:55 against contracts
    that had already expired. All times are ET via ZoneInfo, so EST and EDT
    are handled by the zone rather than by an offset anyone has to remember
    to change twice a year."""
    bounds = market_calendar.session_bounds(now.date())
    if bounds is None:
        return "closed"                      # weekend or market holiday
    open_min, close_min = bounds
    t = now.hour * 60 + now.minute
    if t < open_min:
        return "closed"
    if t < close_min - FLATTEN_BEFORE_CLOSE_MIN:
        return "open"
    if t < close_min + MANAGE_AFTER_CLOSE_MIN:
        return "closing"          # flatten only
    return "closed"


def last_0dte_entry_min(d) -> int:
    """The ET minute after which no FRESH 0DTE position is opened.

    Derived from the session's real close rather than a second hardcoded
    clock time, so it is one rule and not two that can disagree: a 0DTE
    entry is allowed for as long as the contract still has MIN_MINUTES_LEFT
    of life in it. That extends the old fixed 15:30 cutoff to the last
    moment a same-day trade can actually work, and on an early-close day it
    moves itself rather than trading three hours past the bell."""
    return market_calendar.session_close_min(d) - MIN_MINUTES_LEFT


def _expiry_datetime(d) -> datetime:
    """A `date` -> the ET-aware moment that date's session ends — the same
    representation 0DTE's expiry (today's close) already uses, so
    option_model.occ_symbol()/parse_occ() and every downstream minutes-left
    calculation treat a swing expiry exactly like a 0DTE one. 16:00 on a
    normal day, 13:00 when that date is an early close."""
    return market_calendar.session_close_dt(d)


def _fresh(bars: list, now: datetime) -> bool:
    if not bars:
        return False
    try:
        last = datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
        return (now.astimezone(last.tzinfo) - last).total_seconds() < 900
    except (ValueError, KeyError, AttributeError):
        return True


# ---------------------------------------------------------------------------
# execution — every bot trades its account; only the replay harness models fills
# ---------------------------------------------------------------------------
def _headroom(bot: dict, book, grd: dict, balance: float) -> float:
    """What is left of this bot's position cap after its OPEN positions.

    The cap is a share of the bot's own account value, so it is spent by
    everything the bot is holding at once — not re-granted in full to every
    new entry. Book-agnostic so the replay harness enforces the same rule."""
    open_cost = sum(p["cost"] for p in book.positions(bot["id"]).values())
    return balance * grd["position_pct_limit"] / 100.0 - open_cost


def _record_fill(bot: dict, contract: str, side: str, qty, price, *,
                 symbol: str, order_type: str, limit_price=None, stop_price=None,
                 trail_pct=None, ts=None, note: str | None = None,
                 market_fallback: bool = False,
                 fill_source: str | None = None,
                 book: LiveBook | None = None, regime: str | None = None) -> str | None:
    """Book one fill for `bot`. Returns None on success, else an error string.

    The book decides how (LiveBook.execution): a bot whose execution is
    'broker' submits a REAL order on its account and books only the fill
    Alpaca actually reports — `qty`/`price` here are just the requested
    size and the price the bot decided at, never what ends up in the
    ledger. Only a book with no execution() at all — the replay harness's
    SimBook — records the modeled price directly; no live bot can reach
    that branch.

    A confirmed real fill is booked with forced=True: the guardrail checks
    that decide whether to trade at all (sizing, position cap, the 0DTE
    close-out cutoff) already ran before we submitted the order, and a fill
    that already happened on the real account can't be un-happened by a
    bookkeeping rule after the fact. record_trade's one check that is never
    skipped by forced — refusing to close more than is tracked open —
    still applies and remains a real safety net either way.

    A sell is additionally capped at what the account actually holds of
    the contract (account_book.sell_cap): on a shared account another bot
    may own the rest, and this bot must never sell it.

    `market_fallback`: if an exit's own order type doesn't fill, retry once
    as a plain market order rather than leave a 0DTE position open into
    expiry undefended.
    """
    book = book or LIVE
    exec_fn = getattr(book, "execution", None)
    mode, keys = exec_fn(bot) if exec_fn else ("model", None)

    if mode != "broker":
        fee = qty * config.BOT_COMMISSION_PER_CONTRACT
        return book.record_trade(
            bot["id"], contract, side, qty, price, symbol=symbol, fees=fee,
            order_type=order_type, limit_price=limit_price, stop_price=stop_price,
            trail_pct=trail_pct, ts=ts, note=note, fill_source=fill_source, regime=regime)

    if not keys:
        err = ("no Alpaca keys saved for this bot's account — "
               + ("its owner must add keys on the Custom Bots page"
                  if bot.get("pool") == "custom" else "add the fleet keys on the Bot Fleet page"))
        bots.log_event_once("order_error", {"bot": bot["id"], "user_id": bot.get("user_id"),
                                            "error": err},
                            key="nokeys:{}".format(bot.get("pool") or "fleet"))
        print("bot {} {} {}: {}".format(bot["id"], side, contract, err), flush=True)
        return err

    qty = int(qty)
    if side == "buy_to_open":
        # There is no way to add money to an Alpaca paper account through the
        # API (only the dashboard's Reset, which wipes history and invalidates
        # the keys), so a fleet that outgrows its account has to REFUSE
        # entries rather than fire orders Alpaca will reject. Refusing here is
        # not just tidier: a rejection would otherwise fall on whichever bots
        # happen to tick last, handing the earlier ones a survivorship edge
        # that has nothing to do with their genomes.
        # THE position cap, enforced where nothing can skip it. A confirmed
        # real fill is booked forced=True (see this function's docstring), so
        # bots.record_trade's cap check never runs on a live order — and the
        # sizing that ran upstream used a CHAIN price, which on a thin strike
        # can be an order of magnitude under what Alpaca actually charges.
        # Sending the order out as a limit at the cap's own price makes the
        # guardrail a property of the order rather than of the estimate: a
        # market that has run away past it simply does not fill.
        hr = bots.position_headroom(bot["id"], bot)
        qty = min(qty, bots.max_contracts_for(hr["headroom"], float(price)))
        ceiling = math.floor(hr["headroom"] / max(qty, 1) / bots.CONTRACT_MULT * 100.0) / 100.0
        if qty < 1 or ceiling < 0.01:
            err = ("position cap reached — {:g}% of a ${:,.0f} balance is ${:,.0f} and "
                   "${:,.0f} of that is already open"
                   .format(hr["pct"], hr["balance"], hr["cap"], hr["open_cost"]))
            bots.log_event_once("position_cap", {"bot": bot["id"], "user_id": bot.get("user_id"),
                                                 "cap": round(hr["cap"], 2),
                                                 "open": round(hr["open_cost"], 2)},
                                key="cap:{}".format(bot["id"]))
            print("bot {} entry skipped: {}".format(bot["id"], err), flush=True)
            return err
        if order_type == "market":
            order_type, limit_price = "limit", ceiling
        elif limit_price is None or limit_price > ceiling:
            limit_price = ceiling
        note = "{} (cap ${:,.0f}, limit {:.2f})".format(note or "auto", hr["cap"], limit_price)

        # buying power is checked against what the entry is EXPECTED to cost,
        # not the cap ceiling above it: the ceiling exists to bound a bad
        # surprise, and funding the account for every entry as if the surprise
        # always happens would refuse ordinary trades. A fill that does come in
        # high is Alpaca's to reject, as it was before the ceiling existed.
        cost = qty * float(price) * bots.CONTRACT_MULT
        ok, avail = account_book.can_afford(keys, cost)
        if not ok:
            err = ("not enough options buying power — this entry needs ${:,.0f}, but only "
                   "${:,.0f} of the account's ${:,.0f} is usable ({:.0%} held back for slippage)"
                   .format(cost, (avail or 0.0) * (1.0 - account_book.BP_BUFFER),
                           avail or 0.0, account_book.BP_BUFFER))
            bots.log_event_once("buying_power", {"bot": bot["id"], "user_id": bot.get("user_id"),
                                                 "needed": round(cost, 2),
                                                 "available": round(avail or 0.0, 2)},
                                key="bp:{}".format(broker.account_key(keys)))
            print("bot {} entry skipped: {}".format(bot["id"], err), flush=True)
            return err
    if side == "sell_to_close":
        # never more than THIS bot's own ledger says it holds — checked before
        # anything goes out, because on a shared account the surplus would be
        # another bot's contracts (bots.record_trade re-checks after the fact)
        own = int((book.positions(bot["id"]).get(contract) or {}).get("qty", 0))
        if qty > own:
            return "cannot close {} contracts — only {} open on {}".format(qty, own, contract)
        capped = account_book.sell_cap(keys, contract, qty)
        if capped <= 0:
            err = "contract is not held on the account — reconcile pending"
            print("bot {} sell {} skipped: {}".format(bot["id"], contract, err), flush=True)
            return err
        if capped < qty:
            note = "{} (capped to the {} held on the account)".format(note or "auto", capped)
            qty = capped

    side_ap = "buy" if side == "buy_to_open" else "sell"
    cid = broker.new_client_order_id(bot["id"], side)
    pid = bots.add_pending(bot["id"], contract, side, qty, cid, symbol=symbol, note=note,
                           regime=regime, order_type=order_type, limit_price=limit_price,
                           stop_price=stop_price, trail_pct=trail_pct)
    try:
        filled_qty, avg_px, oid, err = broker.submit_and_wait(
            contract, qty, side_ap, order_type, limit_price, stop_price,
            trail_pct, position_intent=side, keys=keys, client_order_id=cid,
            poll=FAST_POLL_S if bot.get("pool") == "custom" else 1.0)
    except broker.BrokerError as e:
        filled_qty, avg_px, oid, err = 0, None, None, str(e)

    if not filled_qty and market_fallback and side == "sell_to_close":
        # the first order is terminal (cancelled/rejected): retire its journal
        # row and journal the retry in its place, so the retry's fill can
        # be recovered too if this process dies before booking it
        bots.clear_pending(pid)
        cid = broker.new_client_order_id(bot["id"], side)
        pid = bots.add_pending(bot["id"], contract, side, qty, cid, symbol=symbol,
                               note=note, regime=regime)
        try:
            filled_qty, avg_px, oid, err = broker.submit_and_wait(
                contract, qty, side_ap, "market", position_intent=side, keys=keys,
                client_order_id=cid, poll=FAST_POLL_S if bot.get("pool") == "custom" else 1.0)
        except broker.BrokerError as e:
            filled_qty, avg_px, oid, err = 0, None, None, str(e)

    if not filled_qty:
        bots.clear_pending(pid)
        print("bot {} {} {} not filled: {}".format(bot["id"], side, contract, err), flush=True)
        bots.log_event_once("order_error", {"bot": bot["id"], "user_id": bot.get("user_id"),
                                            "side": side, "contract": contract,
                                            "error": (err or "order not filled")[:200]},
                            key="order:{}:{}".format(bot["id"], (err or "")[:40]))
        return err or "order not filled"

    lat = ""
    if bot.get("pool") == "custom" and isinstance(ts, str):
        try:      # pass start -> fill: the whole data-to-fill delay, not just the order's
            lat = " [{:.1f}s pass-to-fill]".format(
                (datetime.now(ET) - datetime.fromisoformat(ts)).total_seconds())
        except ValueError:
            pass
    res = bots.record_trade(
        bot["id"], contract, side, filled_qty, avg_px, symbol=symbol,
        order_type=order_type, limit_price=limit_price, stop_price=stop_price,
        trail_pct=trail_pct, ts=ts,
        note="{} [alpaca #{}]{}".format(note or "auto", oid, lat), forced=True,
        fill_source="broker", regime=regime)   # a real Alpaca fill, not a quote/model price
    bots.clear_pending(pid)
    signed = filled_qty if side == "buy_to_open" else -filled_qty
    account_book.adjust(keys, contract, signed)
    account_book.charge(keys, -signed * (avg_px or 0.0) * bots.CONTRACT_MULT)
    if accounts.is_live_key(keys[0]):
        notify.send("LIVE order filled — bot {}: {} {} x{} {} @ {:.2f}".format(
            bot["id"], side.replace("_", " "), contract, filled_qty, symbol, avg_px or 0.0))
    return res


def _recover_pending(now: datetime | None = None) -> int:
    """Book the fills of real orders a previous cycle journaled but never
    booked (the process died between the submit and the record). Looks each
    one up on its account by client_order_id: filled → booked exactly as it
    would have been; terminal-and-unfilled or unknown to Alpaca → dropped;
    still open → cancelled so it can't fill unmanaged later, then booked
    if anything filled first. Returns the number of fills booked."""
    booked = 0
    for po in bots.pending_orders():
        bot = bots.get_bot(po["bot_id"])
        if not bot:
            bots.clear_pending(po["id"])
            continue
        mode, keys = broker.execution(bot)
        if mode != "broker" or not keys:
            bots.clear_pending(po["id"])
            continue
        try:
            order = broker.get_order_by_client_id(po["client_order_id"], keys=keys)
        except broker.BrokerError as e:
            print("recover pending #{}: {}".format(po["id"], e), flush=True)
            continue                        # try again next cycle
        if order is None:
            bots.clear_pending(po["id"])    # the submit never reached Alpaca
            continue
        if order.get("status") not in broker._TERMINAL:
            broker.cancel_order(order.get("id"), keys=keys)
            try:
                order = broker.get_order(order.get("id"), keys=keys) or order
            except broker.BrokerError:
                pass
        filled_qty, avg_px = broker.fill_of(order)
        if filled_qty > 0:
            err = bots.record_trade(
                bot["id"], po["contract"], po["side"], filled_qty, avg_px or 0.01,
                symbol=po.get("symbol"), order_type=po.get("order_type") or "market",
                limit_price=po.get("limit_price"), stop_price=po.get("stop_price"),
                trail_pct=po.get("trail_pct"), ts=(now or datetime.now(ET)).isoformat(),
                note="{} [alpaca #{}] (recovered fill)".format(po.get("note") or "auto",
                                                             order.get("id")),
                forced=True, fill_source="broker", regime=po.get("regime"))
            if err:
                print("recover pending #{} could not book: {}".format(po["id"], err), flush=True)
            else:
                booked += 1
                if po["side"] == "sell_to_close" and not bots._positions(bot["id"]).get(po["contract"]):
                    bots.clear_pos_state(bot["id"], po["contract"])
        bots.clear_pending(po["id"])
    if booked:
        bots.log_event("recovered_fills", {"booked": booked})
        account_book.forget()
    return booked


# ---------------------------------------------------------------------------
# one bot, one tick
# ---------------------------------------------------------------------------
def _step_bot(bot: dict, bars_by_sym: dict, feats_by_sym: dict, now: datetime,
              phase: str, book: LiveBook | None = None) -> None:
    book = book or LIVE
    g = bot.get("_genome") or bots.genome_of(bot)     # replay pre-parses once
    sym = g["symbol"]
    bars = bars_by_sym.get(sym) or []
    if len(bars) < 20:
        return
    S = bars[-1]["c"]
    minutes_left = option_model.minutes_to_close(now)
    state = book.pos_state(bot["id"])
    positions = book.positions(bot["id"])
    feat = feats_by_sym.get(sym) or {}
    regime = bots.regime_label(feat.get("vix"), feat.get("data_impact"))

    # ---- manage an open position -----------------------------------------
    if positions:
        for contract, p in positions.items():
            info = option_model.parse_occ(contract)
            if not info:
                continue
            # priced against THIS contract's own real expiry, not today's
            # close — for a 0DTE contract the two are identical; for a
            # swing contract weeks out, today's close would badly
            # understate its remaining time value
            pos_minutes_left = option_model.minutes_to_expiry(info["expiry"], now)
            ss = state.get(contract, {})
            iv_use = ss.get("entry_iv") or (0.20 * g.get("iv_bias", 1.0))
            if info.get("short_strike") is not None:
                # a vertical: each leg marked off its own usable quote or the
                # model; the package's bid/ask pay both legs' spreads
                live_q = option_chain.spread_mark(sym, S, info, pos_minutes_left, iv_use,
                                                  info["expiry"], now=now)
                cur = live_q["price"]
            else:
                cur = option_model.bs_price(S, info["strike"], pos_minutes_left,
                                            iv_use, info["is_call"])
                live_q = option_chain.live_quote(sym, S, info["expiry"], info["strike"],
                                                 info["is_call"], cur, now=now)
                if live_q:
                    cur = round((live_q["bid"] + live_q["ask"]) / 2.0, 3)
            book.touch_pos_state(bot["id"], contract, cur,
                                 ss.get("opened_ts") or now.isoformat(),
                                 ss.get("entry_iv"))
            try:
                _manage_position(bot, g, sym, contract, p, ss, cur,
                                 pos_minutes_left, phase, now, live_q,
                                 expiry=info["expiry"], book=book, regime=regime)
            except Exception as e:  # noqa: BLE001
                print("bot {} manage {}: {}".format(bot["id"], contract, e), flush=True)
        return

    # ---- flat: consider a new entry -------------------------------------
    if phase != "open" or bot["status"] != "active":
        return

    style = g.get("style", "0dte")
    # single-leg options only unless spreads are explicitly allowed (config.BOT_ALLOW_SPREADS)
    structure = g.get("structure", "single") if config.BOT_ALLOW_SPREADS else "single"

    if style == "0dte":
        if (now.hour * 60 + now.minute) >= last_0dte_entry_min(now.date()):
            return
        if minutes_left < MIN_MINUTES_LEFT:
            return

    ctx = {**feat, "closes": market_data.closes(bars), "bars": bars}
    direction, strength = strategies.evaluate(g, ctx)
    if not direction or strength < max(SIGNAL_FLOOR, g["signal_strength"]):
        return

    is_call = direction == "long"

    if style == "swing":
        target_dte = int(g.get("swing_target_dte", 14) or 14)
        exp_date = option_quotes.pick_expiry(sym, target_dte, today=now.astimezone(ET).date())
        if exp_date is None:      # no listed-expiration data available right now
            exp_date = (now.astimezone(ET) + timedelta(days=target_dte)).date()
        expiry = _expiry_datetime(exp_date)
    else:
        expiry = now.replace(hour=16, minute=0, second=0, microsecond=0)

    entry_minutes_left = option_model.minutes_to_expiry(expiry, now)
    if entry_minutes_left < MIN_MINUTES_LEFT:
        return          # safety net — e.g. a swing lookup returned something already expiring

    # --- the bot picks a strike off the chain (real quotes where available,
    # modeled Black-Scholes filling any gap — see option_chain.build_chain) --
    atm_iv = option_chain.realized_vol(market_data.closes(bars)) * g.get("iv_bias", 1.0)
    atm_iv = max(0.08, min(1.2, atm_iv))
    chain = option_chain.build_chain(sym, S, entry_minutes_left, atm_iv, is_call, expiry=expiry, now=now)

    balance = book.balance(bot["id"])
    grd = book.guardrails(bot)
    room = _headroom(bot, book, grd, balance)
    budget = min(balance * g["risk_pct"] / 100.0, room)
    if room <= 0:
        return                         # already holding a cap's worth

    pick = option_chain.pick_strike(chain, g, budget, grd["min_contracts"]) if budget > 0 else None
    # what the genome would buy if the per-trade budget (risk_pct) were not
    # the limit — only the position cap — so we can tell "nothing qualifies"
    # from "the qualifying contract costs more than risk_pct allows"
    ideal = option_chain.pick_strike(chain, g, room, grd["min_contracts"])
    min_size = False
    if (grd.get("min_size_over_risk") and structure == "single" and ideal
            and ideal["ask"] * grd["min_contracts"] * bots.CONTRACT_MULT > budget + 1e-6):
        # the owner opted this bot into buying the minimum size of the
        # contract it actually wants, even when risk_pct would not fund it;
        # the position cap (room) already covers it — pick_strike checked
        pick, min_size = ideal, True
    if not pick:                       # nothing clears the bot's filters
        _log_budget_too_small(bot, g, grd, chain, balance, budget, room, ideal, "no affordable contract")
        return
    if structure == "vertical":
        # the genome picked the LONG strike; the short leg sits spread_width
        # strikes further OTM and the package is what gets booked
        pick = option_chain.vertical_row(chain, pick, int(g.get("spread_width", 2) or 2),
                                         S, entry_minutes_left, atm_iv)
        if not pick:
            return
    # fleet-wide cost gate (not a gene): the planned hold must have enough
    # expected move in it to pay the spread, the theta and the commission
    if style == "swing":
        hold_min = float(g.get("swing_max_hold_days", 5) or 5) * 1440.0
    else:
        hold_min = float(g.get("hold_max_min", 60) or 60)
    gate = option_chain.cost_gate(pick, S, hold_min, entry_minutes_left,
                                  config.BOT_COMMISSION_PER_CONTRACT, config.BOT_COST_GATE_MULT)
    if not gate["ok"]:
        if not min_size:       # the budget may have steered it to a cheap strike that cannot pay its costs
            _log_budget_too_small(bot, g, grd, chain, balance, budget, room, ideal,
                                  "cheaper contract it could afford failed the cost gate")
        return
    K, entry_iv = pick["strike"], pick["iv"]
    fill_px = pick["ask"]              # what Buy-to-Open actually pays, not the mid
    # floored, and NOT floored up to min_contracts: when the cap cannot fund a
    # minimum-size entry the bot skips the trade rather than overspending on it
    qty = grd["min_contracts"] if min_size else min(grd["max_contracts"], bots.max_contracts_for(budget, fill_px))
    if qty < grd["min_contracts"]:
        return

    if structure == "vertical":
        contract = option_model.spread_symbol(sym, expiry, is_call, K, pick["short_strike"])
    else:
        contract = option_model.occ_symbol(sym, expiry, is_call, K)
    otype, lim, stop, trail = _entry_order(g, fill_px)
    err = _record_fill(
        bot, contract, "buy_to_open", qty, fill_px, symbol=sym,
        order_type=otype, limit_price=lim, stop_price=stop, trail_pct=trail,
        ts=now.isoformat(),
        note="auto {} {} {:+.1f}% {:.2f}d iv{:.0%} (sig {:.2f}) S/R {:.1f}↑/{:.1f}↓ATR "
             "{}{} exp {} edge {:.2f}x [{}]".format(
            g["strategy"], direction, pick["dist_pct"], abs(pick["delta"]),
            entry_iv, strength,
            ctx.get("res_dist_atr", 0.0), ctx.get("sup_dist_atr", 0.0),
            style, (" vertical {:g}w".format(pick["width"]) if structure == "vertical" else ""),
            expiry.date(), gate["ratio"], pick["source"]) + (
            " — min size, above risk_pct budget ${:,.0f}".format(budget) if min_size else ""),
        fill_source=pick["source"], book=book, regime=regime)
    if not err:
        # hwm/mark start at the fill — the price ACTUALLY booked (a real
        # bot's Alpaca fill, else the modeled ask); entry_mid is the MID at
        # entry so stops and targets are measured mid-to-mid, not ask-to-mid
        booked = (book.positions(bot["id"]).get(contract) or {}).get("avg_price") or fill_px
        book.touch_pos_state(bot["id"], contract, booked, now.isoformat(), entry_iv,
                             entry_mid=pick["price"])


def _log_budget_too_small(bot, g, grd, chain, balance, budget, room, ideal, why) -> None:
    """The bot had a signal and sat out because its entry budget could not
    fund the contract it wanted. Without this the skip is silent and looks
    like a quiet market. At most once per 30 min per bot.

    `ideal` is what the genome would buy if only the position cap limited it.
    Logged when that costs more than the per-trade budget (risk_pct), or —
    with nothing affordable at all — when the position cap itself is what
    rules out everything the filters allow."""
    mult = bots.CONTRACT_MULT
    if ideal:
        one = ideal["ask"] * grd["min_contracts"] * mult
        if one <= budget + 1e-6:
            return                     # the budget was not the obstacle
        cause = "risk_pct"
    else:
        wanted = option_chain.pick_strike(chain, g, 1e12, grd["min_contracts"])
        if not wanted:
            return                     # the filters (volume, IV…) rejected it — not a budget matter
        cause, one = "position_cap", wanted["ask"] * grd["min_contracts"] * mult
    detail = {"bot": bot["id"], "name": bot.get("name"), "user_id": bot.get("user_id"),
              "cause": cause, "why": why, "one_contract_cost": round(one, 2),
              "risk_budget": round(budget, 2), "risk_pct": g["risk_pct"],
              "position_room": round(room, 2), "balance": round(balance, 2),
              "min_size_override": bool(grd.get("min_size_over_risk"))}
    try:
        bots.log_event_once("budget_too_small", detail, "budget:{}".format(bot["id"]), within_s=1800.0)
    except Exception as e:  # noqa: BLE001 — logging must never stop a trade pass
        print("bot {} budget log: {}".format(bot["id"], e), flush=True)


def _manage_position(bot, g, sym, contract, p, ss, cur, minutes_left, phase, now,
                     live_q: dict | None = None, *, expiry: datetime,
                     book: LiveBook | None = None, regime: str | None = None):
    """Decide what to do with one open position this tick: hard stop, EOD,
    average-in, trim a partial, close the core (leaving runners), or manage the
    runners / post-trim remainder with a looser trail. Every behaviour past the
    hard exits is gated by a genome gene the bot can evolve to zero.

    `expiry` is THIS contract's own real expiry (0DTE: today; swing: weeks
    out) — the hard EOD flatten below only fires when it's actually today."""
    book = book or LIVE
    expires_today = expiry.astimezone(ET).date() == now.astimezone(ET).date()
    total = int(p["qty"])
    # Decision P&L is MID-to-MID: `cur` is the current mid, the reference
    # is the mid at entry (falling back to the ask actually paid only for
    # positions opened before entry_mid was recorded). Measuring ask-to-mid
    # started every trade 4-9% "down" and let the spread alone trip a stop
    # near the genome's 10% floor. Realized P&L in the ledger is still the
    # real fills — this only changes when the rules fire.
    entry = ss.get("entry_mid") or p["avg_price"]
    pnl_pct = ((cur - entry) / entry * 100.0) if entry else 0.0
    hwm = max(ss.get("hwm") or cur, cur)
    trimmed = int(ss.get("trimmed") or 0)
    core_closed = int(ss.get("core_closed") or 0)
    adds = int(ss.get("adds") or 0)

    held_min = 0.0
    opened = ss.get("opened_ts")
    if opened:
        try:
            od = datetime.fromisoformat(opened)
            if od.tzinfo is None:
                od = od.replace(tzinfo=ET)
            held_min = (now - od).total_seconds() / 60.0
        except ValueError:
            pass

    # `cur` (real quote mid when one exists, else the BS mid) drives every
    # decision below — pnl_pct, the high-water mark, trailing/take-profit/
    # stop triggers — since that's what the bot can actually observe live.
    # The ACTUAL fill uses the real bid/ask directly when live_q has this
    # contract; otherwise it falls back to a modeled spread around `cur`.
    exit_px = live_q["bid"] if live_q else option_model.exit_fill_price(cur, minutes_left)
    entry_px = live_q["ask"] if live_q else option_model.entry_fill_price(cur, minutes_left)
    fill_src = (live_q.get("source") or option_quotes.status(sym)) if live_q else "model"

    def _close(qty, reason):
        qty = int(qty)
        if qty <= 0:
            return
        otype, lim, stop, trail = _exit_order(g, exit_px)
        err = _record_fill(
            bot, contract, "sell_to_close", qty, exit_px, symbol=sym,
            order_type=otype, limit_price=lim, stop_price=stop, trail_pct=trail,
            ts=now.isoformat(), note="auto exit: " + reason, market_fallback=True,
            fill_source=fill_src, book=book, regime=regime)
        if not err and qty >= total:
            book.clear_pos_state(bot["id"], contract)

    # 1. hard exits — whole position, no discretion
    #    the 15:55 ET close-out flatten (and its minutes_left<=2 fail-safe)
    #    only apply when THIS contract actually expires today — always true
    #    for 0DTE, only true on a swing position's final day. Any other day,
    #    a swing position skips straight to the stop/trim/trail logic below
    #    and rides past the close.
    if expires_today:
        if phase == "closing":
            return _close(total, "eod")
        if minutes_left <= 2:
            return _close(total, "eod failsafe")
    if pnl_pct <= -g["stop_pct"]:
        return _close(total, "stop")

    # 2. average in (pyramid on strength / add on weakness) — only while the
    #    market is fully open; record_trade also hard-blocks any buy at/after
    #    15:55 ET on a contract expiring TODAY (0DTE, or a swing position's
    #    final day), so a late add on one simply falls through to exit
    #    management. A swing add on any other day isn't affected.
    when = g.get("scale_in_when", "never")
    max_adds = int(g.get("scale_in_max_adds", 0) or 0)
    if (phase == "open" and when != "never" and max_adds and adds < max_adds
            and not core_closed and not trimmed):
        trig = g.get("scale_in_trigger_pct", 40)
        fire = ((when in ("strength", "both") and pnl_pct >= trig) or
                (when in ("weakness", "both") and pnl_pct <= -trig))
        if fire:
            grd = book.guardrails(bot)
            # an add is more of the SAME capped position, so it spends the
            # room the open leg has left — not a fresh cap of its own
            room = _headroom(bot, book, grd, book.balance(bot["id"]))
            add_qty = min(grd["max_contracts"],
                          int(round(total * g.get("scale_in_add_frac", 0.5))),
                          bots.max_contracts_for(room, entry_px))
            if add_qty >= 1:
                otype, lim, stopp, trl = _entry_order(g, entry_px)
                err = _record_fill(
                    bot, contract, "buy_to_open", add_qty, entry_px, symbol=sym,
                    order_type=otype, limit_price=lim, stop_price=stopp, trail_pct=trl,
                    ts=now.isoformat(),
                    note="auto add #{} ({}, pnl {:+.0f}%)".format(adds + 1, when, pnl_pct),
                    fill_source=fill_src, book=book, regime=regime)
                if not err:
                    # re-base the mid-to-mid reference to the new weighted average
                    new_mid = (entry * total + cur * add_qty) / (total + add_qty)
                    book.mark_pos(bot["id"], contract, adds=adds + 1, entry_mid=new_mid)
                    return
            # no room, or the add was rejected — keep managing the position

    # 3. trim — one partial profit-take, keep the rest running
    if not trimmed and g.get("trim_frac", 0) > 0 and pnl_pct >= g.get("trim_trigger_pct", 40):
        tq = int(round(total * g["trim_frac"]))
        tq = max(1, min(total - 1, tq))
        if 1 <= tq < total:
            _close(tq, "trim {:.0f}%".format(g["trim_frac"] * 100))
            book.mark_pos(bot["id"], contract, trimmed=1)
            return

    # 4. core take-profit — sell the core, leave runners to hunt a big move
    if not core_closed and pnl_pct >= g["take_profit_pct"]:
        runners = int(round(total * g.get("runner_frac", 0.0)))
        core = total - runners
        if core >= 1:
            _close(core, "take_profit"
                   + ("" if not runners else " (hold {} runner)".format(runners)))
            book.mark_pos(bot["id"], contract, core_closed=1)
            return

    # 5. runners / remainder — looser trail once we've banked something
    banked = core_closed or trimmed
    trail_pct = g.get("runner_trail_pct", 30) if banked else g["trail_pct"]
    if hwm > 0 and cur <= hwm * (1 - trail_pct / 100.0) and pnl_pct > 0:
        return _close(total, "trail (runner)" if banked else "trail")
    # time stop applies only to an un-banked core; runners ride to the close.
    # 0dte's hold_max_min is intraday minutes; swing's own hold cap is in
    # days (multi-day holds are the whole point), so it reads a different
    # gene and compares against elapsed wall-clock days instead of minutes.
    if not banked:
        if g.get("style") == "swing":
            if (held_min / 1440.0) >= g.get("swing_max_hold_days", 5):
                return _close(total, "time")
        elif held_min >= g["hold_max_min"]:
            return _close(total, "time")


def _entry_order(g: dict, px: float):
    pref = g.get("order_pref", "market")
    off = g.get("limit_offset_pct", 0.0) / 100.0
    if pref == "limit":
        return "limit", round(px * (1 + off), 2), None, None
    if pref == "stop_limit":
        return "stop_limit", round(px * (1 + off + 0.01), 2), round(px * (1 + off), 2), None
    return "market", None, None, None          # trailing_stop is exit-only


def _exit_order(g: dict, px: float):
    pref = g.get("order_pref", "market")
    off = g.get("limit_offset_pct", 0.0) / 100.0
    if pref == "limit":
        return "limit", round(px * (1 - off), 2), None, None
    if pref == "stop_limit":
        return "stop_limit", round(px * (1 - off - 0.01), 2), round(px * (1 - off), 2), None
    if pref == "trailing_stop":
        return "trailing_stop", None, None, round(g.get("trail_pct", 15.0), 2)
    return "market", None, None, None


# ---------------------------------------------------------------------------
# cycle
# ---------------------------------------------------------------------------
def live_ctx(symbol: str) -> dict:
    """On-demand feature context for one symbol, built the same way _tick()
    builds it for the whole fleet — for the bot-detail page's live genome
    readout, not the trading loop. Bars/daily/vix are all cached upstream
    (market_data), so calling this per page view is cheap."""
    now = datetime.now(ET)
    bars = {s: market_data.recent_bars(s) for s in SYMBOLS}
    b = bars.get(symbol) or []
    if len(b) < 20:
        return {}
    try:
        ctx = features.build(symbol, b, bars, now,
                             minutes_left=option_model.minutes_to_close(now),
                             daily=market_data.daily_ctx(symbol), vix=market_data.vix())
    except Exception as e:  # noqa: BLE001
        print("live_ctx {}: {}".format(symbol, e), flush=True)
        ctx = {}
    return {**ctx, "closes": market_data.closes(b), "bars": b}


_LAST_FEATS: dict = {}      # features from the latest full pass — the held-position pass reuses them


def _market_context(now: datetime, phase: str, record: bool = True):
    """The bars, features and VIX for this minute — fetched and recorded
    ONCE per tick and then handed to every user's fleet.

    The market is the same for everyone, so paying for it once and sharing
    it is both the cheap thing and the fair thing: whoever's Alpaca
    entitlement served the fetch, the bars land in the one history every
    user's replay and evolution runs on. Returns None when there is nothing
    fresh to act on."""
    bars = {s: market_data.recent_bars(s) for s in SYMBOLS}
    if phase == "open" and not any(_fresh(b, now) for b in bars.values()):
        return None
    minutes_left = option_model.minutes_to_close(now)
    vix = market_data.vix()
    daily = {s: market_data.daily_ctx(s) for s in SYMBOLS}
    # one write, for everybody — see recorder.contributions()
    if record:      # the fast lane ticks every few seconds; the minute cycle records for everyone
        recorder.record_tick(now, bars, vix, daily,
                             user_id=bot_settings.data_source()["user_id"])
    feats = {}
    for s in SYMBOLS:
        try:
            feats[s] = features.build(s, bars.get(s) or [], bars, now,
                                      minutes_left=minutes_left,
                                      daily=daily.get(s), vix=vix)
        except Exception as e:  # noqa: BLE001
            print("features {}: {}".format(s, e), flush=True)
            feats[s] = {}
    _LAST_FEATS.clear()
    _LAST_FEATS.update(feats)
    return bars, feats


def fast_lane_alive() -> bool:
    """True while the fast-lane loop has beaten recently."""
    try:
        return time.time() - float(bots._meta_get(FAST_BEAT_KEY) or 0) < FAST_ALIVE_S
    except (TypeError, ValueError):
        return False


def fast_lane_age_s() -> float | None:
    try:
        raw = bots._meta_get(FAST_BEAT_KEY)
        return time.time() - float(raw) if raw else None
    except (TypeError, ValueError):
        return None


@contextmanager
def _engine_lock(timeout: float = 30.0):
    """A cross-process lock around anything that places or reconciles orders
    on an account, so the fast lane and the minute cycle's housekeeping
    (fill recovery, settlement, ledger reconcile) never interleave: an order
    the lane has in flight is not mistaken for an orphan. Yields whether
    the lock was obtained within `timeout`."""
    f = open(config.DATA_DIR / "engine.lock", "a+")
    deadline = time.monotonic() + timeout
    got = False
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            got = True
            break
        except OSError:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    try:
        yield got
    finally:
        if got:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _bots_to_step(user_id: int, scope: str = "all") -> list[dict]:
    """The bots of one account the engine should tick this minute.

    Their fleet and baseline bots only while that account's fleet can
    actually trade — switched on and connected; their custom bots
    regardless of any of that. A custom bot is a separate experiment on
    separate keys, so pausing (or never setting up) a fleet must not
    silently stop managing stops and trails on a live custom position."""
    fleet_on = int(user_id) in set(active_fleet_users())
    out = []
    for b in bots.live_bots(user_id=user_id):
        is_custom = (b.get("pool") or "fleet") == "custom"
        if (scope == "custom" and not is_custom) or (scope == "fleet" and is_custom):
            continue
        if fleet_on or is_custom:
            out.append(b)
    return out


def trading_users() -> list[int]:
    """Every account with something to tick: a fleet configured, or
    custom-bot keys saved. Sorted, so fleets are always stepped in the same
    order rather than one account systematically going first on some cycles
    and last on others."""
    out = set(fleet_users())
    for b in bots.custom_bots():
        if b.get("user_id") is not None:
            out.add(int(b["user_id"]))
    return sorted(out)


def _tick(now: datetime, user_ids: list | None = None, scope: str = "all",
          record: bool = True) -> dict:
    """One minute of trading for every account (or just `user_ids`').

    Accounts are stepped one after another off the SAME market snapshot, so
    nobody's bots are advantaged by being priced fresher than anybody
    else's; each account's bots then place their orders on that account's
    own keys. An account with nothing switched on is skipped without
    touching anyone else's cycle.

    Returns what it actually did — {"stepped": n, "idle": reason or None}.
    Every early return here silently stops the WHOLE fleet, position
    management included, so "did nothing" has to be a reportable outcome
    rather than a bare return: a stale feed once cost three and a half
    hours of unmanaged positions while the service stayed green and the
    cycle log kept printing phase=open, active=308."""
    phase = _phase(now)
    if phase == "closed":
        return {"stepped": 0, "idle": None}          # not idle — nothing to do
    ctx = _market_context(now, phase, record=record)
    if ctx is None:
        return {"stepped": 0, "idle": "no fresh market data"}
    bars, feats = ctx

    stepped = 0
    for uid in (user_ids if user_ids is not None else trading_users()):
        for b in _bots_to_step(uid, scope):
            stepped += 1
            try:
                _step_bot(b, bars, feats, now, phase)
            except Exception as e:  # noqa: BLE001 — isolate a bad bot
                print("bot {} (user {}) step failed: {}".format(b.get("id"), uid, e), flush=True)
    # a fleet-only tick with nothing to step is normal when an account has
    # only custom bots — the fast lane has them
    return {"stepped": stepped,
            "idle": None if (stepped or scope != "all") else "no bots to step"}


def _step_sims(now: datetime, bars: dict, feats: dict, phase: str, user_ids: list,
               budget_s: float | None = None) -> int:
    """One pass over the incubator's sim bots: the same _step_bot() the fleet
    runs, with modeled fills (broker.execution says 'model' for pool='sim'),
    inside `budget_s` (default incubator.TICK_BUDGET_S). Where the budget runs
    out the rest wait for the next pass; the start point rotates so the same
    bots are not the ones always left waiting."""
    budget_s = incubator.TICK_BUDGET_S if budget_s is None else budget_s
    todo = []
    for uid in user_ids:
        todo.extend(bots.active_bots("sim", int(uid)))
    if not todo:
        return 0
    off = int(time.time() // 15) % len(todo)
    todo = todo[off:] + todo[:off]
    t0 = time.monotonic()
    n = 0
    for b in todo:
        if time.monotonic() - t0 > budget_s:
            break
        try:
            _step_bot(b, bars, feats, now, phase)
        except Exception as e:  # noqa: BLE001 — isolate a bad bot
            print("sim bot {} step failed: {}".format(b.get("id"), e), flush=True)
        n += 1
    return n


def _sim_pass(users: list, budget_s: float) -> int:
    """Step the incubator off a fresh market snapshot. Called in the idle
    gap between the cycle's real ticks, so it spends time the cycle was going
    to sleep through anyway — and never before a real bot has been stepped.
    Any failure is swallowed: a sim problem must not stop the fleet."""
    if budget_s <= 0.5:
        return 0
    try:
        now = datetime.now(ET)
        phase = _phase(now)
        if phase == "closed":
            return 0
        ctx = _market_context(now, phase, record=False)
        if ctx is None:
            return 0
        return _step_sims(now, ctx[0], ctx[1], phase, users, budget_s)
    except Exception as e:  # noqa: BLE001
        print("incubator pass failed: {}".format(e), flush=True)
        return 0


def _report_idle(reason: str, now: datetime) -> None:
    """Say something when the engine ran but moved nothing during market
    hours. Deliberately noisy in the one place noise is warranted: a fleet
    that has silently stopped trading looks exactly like a quiet market,
    and the difference is worth a message. Rate-limited to one per 15
    minutes per reason, and the DB is what remembers that — each cycle is
    its own short-lived process, so an in-process cooldown would not hold."""
    fresh = bots.log_event_once("engine_idle", {"reason": reason},
                                key="idle:{}".format(reason), within_s=900.0)
    print("ENGINE IDLE during market hours: {}".format(reason), flush=True)
    if fresh and notify.enabled():
        notify.send(
            "\u26a0\ufe0f Bot fleet is not trading\n\n"
            "The engine ran at {} ET but moved nothing: {}.\n\n"
            "Open positions are not being managed either — stops, trails and "
            "take-profits all run from the same tick.".format(
                now.strftime("%H:%M"), reason),
            buttons=[notify._button("Open the fleet", "/bots")])


def _settle_expired(now: datetime) -> int:
    """Force-close every position whose 0DTE contract has expired — runners
    included. 0DTE means nothing may be held past the 16:00 ET close, so this
    runs on EVERY cycle regardless of market phase (and even with the engine
    switched off) and is the hard guarantee behind the graceful 15:55 flatten
    in _manage_position().

    A bot with real fills holds a REAL position, so this attempts a real
    market sell first; only if the broker call itself fails (unreachable,
    no keys, nothing left on the account) does it fall back to a synthetic
    settlement at modeled intrinsic value against the last known spot (if
    spot is unavailable the contract is worth ~nothing and settles at
    $0.01), clearly flagged as such so it's obvious the ledger and the real
    account may have diverged and need a manual check. A bot with modeled
    execution settles at intrinsic directly — pure bookkeeping.

    Covers retired bots too (retirement doesn't flatten), not just active ones.
    """
    now_et = now.astimezone(ET)
    today = now_et.date()
    late = (now_et.hour * 60 + now_et.minute) >= 15 * 60 + 58
    spot: dict[str, float | None] = {}
    n = 0
    for b in bots.list_bots():
        if b.get("status") == "failing":         # already force-flattened
            continue
        is_real = broker.execution(b)[0] == "broker"
        for contract, p in bots._positions(b["id"]).items():
            info = option_model.parse_occ(contract)
            if not info:
                continue
            exp = info["expiry"].astimezone(ET).date()
            if exp > today or (exp == today and not late):
                continue
            sym = info["underlying"]
            if sym not in spot:
                spot[sym] = market_data.last_price(sym)
            S = spot[sym]
            intrinsic = option_model.intrinsic(info, S) if S else 0.0   # a vertical caps at its width
            px = max(0.01, round(intrinsic, 2))
            qty = int(p["qty"])

            if is_real:
                err = _record_fill(
                    b, contract, "sell_to_close", qty, px, symbol=sym,
                    order_type="market", ts=now.isoformat(),
                    note="expiration settlement — real market close",
                    market_fallback=True)
                if err:
                    print("bot {} expiration close FAILED, settling synthetically "
                         "at intrinsic — reconcile against Alpaca manually: {}"
                         .format(b["id"], err), flush=True)
                    err = bots.record_trade(
                        b["id"], contract, "sell_to_close", qty, px, symbol=sym,
                        ts=now.isoformat(), forced=True,
                        note="expiration settlement @ intrinsic (broker close FAILED: {})"
                             .format(err))
            else:
                err = bots.record_trade(
                    b["id"], contract, "sell_to_close", qty, px,
                    symbol=sym, ts=now.isoformat(), forced=True,
                    note="expiration settlement @ intrinsic")
            if not err:
                bots.clear_pos_state(b["id"], contract)
                n += 1
    return n


def fleet_users() -> list[int]:
    """Every account with a fleet configured, whether or not it is running —
    what the admin views and what daily maintenance evolves."""
    return bot_settings.fleet_user_ids()


def active_fleet_users() -> list[int]:
    """Accounts whose fleet should actually trade this cycle: a fleet
    configured, that user's own engine switch on, and the installation
    master switch on."""
    return [u for u in fleet_users() if bot_settings.fleet_engine_enabled(u)]


def prepare(default_owner: int | None = None) -> int:
    """Schema, ownership migration and per-user baselines — everything that
    has to be true before a cycle can run. Returns the primary admin id.

    The migration is the upgrade path from the single shared fleet: the
    admin inherits the fleet that was already running AND the keys it was
    running on (bot_settings.migrate_fleet_to), so it keeps trading the
    same account without anything being re-entered, and it becomes that
    admin's own fleet rather than a house one nobody owns."""
    from . import auth
    owner = default_owner if default_owner is not None else auth.primary_admin_id()
    bots.init(default_owner=owner)
    bot_settings.migrate_fleet_to(owner)
    retired = bot_settings.retire_simulated_fleets()
    if retired:
        # these fleets were booking modeled fills and now place real orders
        # (or, with no keys, stop trading) — their ledgers mix two fill
        # rules, so say so somewhere durable rather than changing it silently
        bots.log_event("simulated_retired", {
            "user_ids": retired,
            "note": "simulated execution removed; these fleets now trade on their "
                    "Alpaca account. Their ledgers mix modeled and real fills — "
                    "reset the fleet to judge genomes under one set of rules."})
    for uid in fleet_users():
        bots.ensure_baseline_bots(uid)
    return owner


# One cycle at a time, process-wide. The systemd timer fires every minute and
# any user can also kick one from their fleet page; two cycles overlapping
# would step the same bot twice off two market snapshots and double-enter
# it. A caller that finds a cycle already running is told so rather than
# queueing behind it — the next tick is a minute away anyway.
_CYCLE_LOCK = threading.Lock()


def run_cycle(user_ids: list | None = None) -> dict:
    """One engine cycle.

    Housekeeping that must happen whatever anyone has switched on — booking
    fills a dead cycle left behind, settling expired contracts, reconciling
    every account's ledgers — runs for everybody. Trading and evolution run
    only for the accounts that are switched on, and only for `user_ids`
    when given (a user kicking a cycle from their own fleet page must not
    start placing orders on everyone else's accounts).
    """
    if not _CYCLE_LOCK.acquire(blocking=False):
        return {"skipped": "a cycle is already running"}
    stepped = 0
    sim_stepped = 0
    sim_s = 0.0
    phase = "?"
    try:
        prepare()
        now = datetime.now(ET)
        started = time.monotonic()
        # under the engine lock, so an order the fast lane has in flight is
        # never mistaken for an unbooked fill or an unclaimed position
        with _engine_lock(30.0) as locked:
            if locked:
                recovered = _recover_pending(now)   # fills a dead cycle left unbooked
                settled = _settle_expired(now)      # always — even when the engine is off,
                                                    # an expired 0DTE contract is not a position
                sync = account_book.sync_all(now)   # ledgers vs. what the accounts hold (after
                                                    # settlement, so an expired contract is
                                                    # settled at intrinsic, not "vanished")
            else:
                recovered, settled, sync = 0, 0, {"skipped": "engine lock busy"}
                print("housekeeping skipped: engine lock busy", flush=True)
        users = trading_users()
        if user_ids is not None:
            wanted = {int(u) for u in user_ids}
            users = [u for u in users if u in wanted]
        base = {"settled": settled, "recovered": recovered, "sync": sync,
                "accounts": len(users)}
        # "active: 308" says how many bots EXIST; "stepped" says how many the
        # engine actually moved. The outage on 2026-09-09 printed the first
        # number, unchanged, for three and a half hours.
        if not bot_settings.engine_enabled():
            return {"skipped": "engine disabled", **base}
        if not users:
            return {"skipped": "nothing switched on", **base}
        phase = _phase(now)
        if phase == "closed":
            return {"phase": "closed", "at": now.isoformat(), **base}

        ticks = 4 if phase == "open" else 1
        idle = None
        for i in range(ticks):
            # while the fast lane is beating it owns the custom bots; if it
            # is not, this cycle steps them itself (under the lock) so a
            # dead service never leaves a position unmanaged
            if fast_lane_alive():
                t = _tick(datetime.now(ET), users, scope="fleet")
            else:
                with _engine_lock(30.0) as locked:
                    t = _tick(datetime.now(ET), users, scope="all" if locked else "fleet")
            stepped += t["stepped"]
            idle = idle or t["idle"]
            if i < ticks - 1:
                # the budget covers every account, not each one: real orders
                # wait for their fills, and N accounts' worth of them must
                # never run a one-minute cycle into the next firing
                if time.monotonic() - started > CYCLE_BUDGET_S:
                    break                   # next timer fire continues
                # the simulated bots use the gap the cycle sleeps through
                gap = time.monotonic()
                sim_stepped += _sim_pass(users, min(incubator.TICK_BUDGET_S, 13.0))
                sim_s += time.monotonic() - gap
                time.sleep(max(0.0, 15 - (time.monotonic() - gap)))


        # a cycle that stepped nothing while the market was open is the shape
        # of a silent outage, not of a quiet market — every bot skipped means
        # no entries AND no stop/trail management
        if phase == "open" and not stepped and idle:
            _report_idle(idle, now)

        if sim_stepped:
            try:
                incubator.record_capacity(sim_stepped, sim_s, time.monotonic() - started,
                                          len(bots.active_bots("sim")))
            except Exception as e:  # noqa: BLE001 — telemetry must never stop a cycle
                print("capacity telemetry failed: {}".format(e), flush=True)

        ev = {"archived": 0, "replaced": 0, "topped_up": 0}
        for uid in users:
            if bot_settings.fleet_engine_enabled(uid):   # evolution is a fleet thing
                for k, v in evolve(uid).items():
                    ev[k] = ev.get(k, 0) + v
        return {"phase": phase, "at": now.isoformat(), "stepped": stepped,
                "sim_stepped": sim_stepped, "sim_s": round(sim_s, 1),
                "cycle_s": round(time.monotonic() - started, 1), **base,
                "active": len(bots.active_bots()), **ev}
    finally:
        # A LIVENESS signal, not a success one: the watchdog in the web app
        # is asking whether this process ran at all. A cycle that crashed
        # half way still ran; one that never started is the thing nothing
        # inside here could ever report.
        watchdog.beat(stepped, phase)
        _CYCLE_LOCK.release()


def fast_tick() -> str:
    """One pass of the fast lane: step only the custom bots, on data no more
    than FAST_DATA_TTL_S old, under the engine lock. Returns what it did
    (for the log), never raises past the caller's guard."""
    now = datetime.now(ET)
    if not bot_settings.engine_enabled():
        return "engine disabled"
    phase = _phase(now)
    if phase == "closed":
        return "closed"
    users = [u for u in trading_users() if _bots_to_step(u, "custom")]
    if not users:
        return "no custom bots"
    with _engine_lock(30.0) as locked:
        if not locked:
            return "engine lock busy"
        t = _tick(datetime.now(ET), users, scope="custom", record=False)
    return "stepped {}{}".format(t["stepped"], " — " + t["idle"] if t["idle"] else "")


def _held_custom_bots(users: list) -> list[dict]:
    return [b for u in users for b in _bots_to_step(u, "custom") if bots._positions(b["id"])]


def hold_tick() -> str | None:
    """The held-position pass: only custom bots that are holding something,
    only their underlying's bars, and the features of the last full pass.
    Entries are never scanned here, so it can run every HOLD_INTERVAL_S on
    very fresh quotes at the cost of a handful of calls — and only exists
    while something is held. Returns None when nothing is held."""
    now = datetime.now(ET)
    if not bot_settings.engine_enabled():
        return None
    phase = _phase(now)
    if phase == "closed":
        return None
    held = _held_custom_bots(trading_users())
    if not held:
        return None
    with _engine_lock(10.0) as locked:
        if not locked:
            return "engine lock busy"
        now = datetime.now(ET)
        syms = {bots.genome_of(b)["symbol"] for b in held}
        bars = {s: market_data.recent_bars(s) for s in syms}
        if phase == "open" and not any(_fresh(b, now) for b in bars.values()):
            return "held: no fresh market data"
        for b in held:
            try:
                _step_bot(b, bars, _LAST_FEATS, now, phase)
            except Exception as e:  # noqa: BLE001
                print("bot {} hold step failed: {}".format(b.get("id"), e), flush=True)
    return "held {}".format(len(held))


def _set_ttls(seconds: float) -> None:
    market_data._TTL = seconds          # this process only
    option_quotes._TTL = seconds


def fast_loop() -> None:
    """The long-running custom-bot loop (`python -m webapp.bot_engine fast`).
    Beats a heartbeat every pass — even when idle — so the minute cycle
    knows the lane is alive and leaves custom bots to it."""
    _set_ttls(FAST_DATA_TTL_S)
    prepare()
    print("fast lane up: every {}s (every {}s while holding), data ttl {}s ({}s holding)".format(
        FAST_INTERVAL_S, HOLD_INTERVAL_S, FAST_DATA_TTL_S, HOLD_DATA_TTL_S), flush=True)
    last_log = ""
    next_full = 0.0
    while True:
        started = time.monotonic()
        holding = False
        try:
            bots._meta_set(FAST_BEAT_KEY, repr(time.time()))
            holding = bool(bot_settings.engine_enabled() and _held_custom_bots(trading_users()))
            # quotes can be a second old while a position is open, a few seconds otherwise
            _set_ttls(HOLD_DATA_TTL_S if holding else FAST_DATA_TTL_S)
            if started >= next_full:
                what = fast_tick()                  # entries and exits, every FAST_INTERVAL_S
                next_full = started + FAST_INTERVAL_S
            elif holding:
                what = hold_tick() or "idle"        # exits only, every HOLD_INTERVAL_S
            else:
                what = last_log
            if what != last_log and not what.startswith(("stepped", "held ")):
                print("fast lane: " + what, flush=True)
            last_log = what
            bots._meta_set(FAST_BEAT_KEY, repr(time.time()))
        except Exception as e:  # noqa: BLE001 — the loop must outlive a bad pass
            print("fast lane pass failed: {}".format(e), flush=True)
        idle = last_log in ("closed", "engine disabled", "no custom bots")
        gap = HOLD_INTERVAL_S if holding else (15.0 if idle else FAST_INTERVAL_S)
        time.sleep(max(0.25, gap - (time.monotonic() - started)))


# ---------------------------------------------------------------------------
# evolution
# ---------------------------------------------------------------------------
# how often a new bot ADOPTS an under-tested genome from the shared pool
# instead of breeding a brand-new one. This is what turns several fleets
# into one deeper experiment rather than several shallow ones: a candidate
# another user started and abandoned half-tested gets picked up and its
# pooled record finished, instead of the population forgetting it.
ADOPT_RATE = 0.25


def _over_quota_families() -> set:
    """Families already past MAX_STRATEGY_SHARE of the ACTIVE population —
    counted across every user's fleet, since the population is what is
    being kept diverse, not any one user's slice of it."""
    return {f for f, share in gene_pool.family_shares().items()
            if share >= MAX_STRATEGY_SHARE}


def _breeding_parents() -> dict:
    """{family: [pool genomes proven on the POOLED record]} minus any family
    already over quota.

    Parents are judged on every user's trades on the genome at once, so a
    genome three fleets each gave twenty-five trades is a legitimate parent
    on seventy-five — which is the entire point of running one population
    across several accounts. Custom bots' hand-tuned genomes still never
    feed back into fleet evolution."""
    over = _over_quota_families()
    fams = gene_pool.proven_by_family(BREED_POOL)
    return {f: v for f, v in fams.items() if f not in over and v}


def _pick_parent(parents: dict):
    """Family-balanced: pick a family uniformly among those with a proven
    member, then a random proven genome within it — so a family with one
    proven genome gets the same shot at children as a family with eight."""
    fams = [f for f, v in parents.items() if v]
    if not fams:
        return None
    return random.choice(parents[random.choice(fams)])


class _Breeder:
    """The state one breeding pass shares across every slot it fills.

    Reading the pooled scoreboard, the pending list and the set of running
    fingerprints is a pass over the whole population, and a top-up can fill
    sixty slots at once — recomputing all three per slot made evolve()
    quadratic in the pool size. They are gathered once here instead, and
    `claimed` is updated as slots are filled so a single pass never hands
    two of its own new bots the same genome either.
    """

    def __init__(self):
        self.parents = _breeding_parents()
        self.claimed = gene_pool.claimed_fingerprints()
        self._waiting = None
        # set by _replacement_genome when the child's parent is a sim-bot elite:
        # that parent is not in the gene pool, so the pool cannot supply the
        # child's generation and the caller has to pass it
        self.child_generation: int | None = None

    @property
    def waiting(self) -> list:
        if self._waiting is None:
            self._waiting = gene_pool.pending(20, claimed=self.claimed)
        return self._waiting

    def take(self, fp: str) -> None:
        self.claimed.add(fp)
        if self._waiting is not None:
            self._waiting = [e for e in self._waiting if e["fingerprint"] != fp]


def _replacement_genome(user_id: int | None = None, breeder: "_Breeder | None" = None):
    """Pick a genome for a new bot in `user_id`'s fleet, and return
    (genome, parent_fp, origin).

      * IMMIGRANT_RATE of the time: a fresh random genome, no matter how
        good the leaders look — keeps the search exploring.
      * ADOPT_RATE of the time: an under-tested genome already in the shared
        pool that nobody is currently running (gene_pool.pending) — this
        fleet finishes the trial another one started, and the pooled record
        gets there sooner than either could alone.
      * otherwise: a mutated child of a random genome from the pooled
        fitness-ranked top-`BREED_POOL` of PROVEN genomes only, skipping any
        family already over-represented.
      * with no proven genome yet, or every eligible family over quota: a
        fresh random genome rather than an unproven parent.

    Whatever the route, the result is checked against what every other
    fleet is running right now (gene_pool.distinct_child) and re-rolled if
    it collides. Two users testing the same candidate on the same minutes
    would give the pool two correlated copies of one observation; two users
    testing different candidates give it twice the search. Only that second
    thing actually makes the population smarter, so it is the one the
    breeder is built to produce.
    """
    b = breeder or _Breeder()
    b.child_generation = None
    claimed = b.claimed
    # exploration (random seeds, adopted generation-0 genomes) is capped at
    # bots.EXPLORE_SHARE of the fleet; past it, every new bot is a descendant
    explore_ok = bots.exploration_room(user_id) > 0

    if explore_ok and random.random() < IMMIGRANT_RATE:
        g, _fp, _n = gene_pool.distinct_child(genome.random_genome, claimed=claimed)
        return g, None, "seed"

    waiting = b.waiting if explore_ok else [e for e in b.waiting if int(e.get("generation") or 0) > 0]
    if random.random() < ADOPT_RATE and waiting:
        pick = random.choice(waiting)
        b.take(pick["fingerprint"])
        return genome.loads(pick["genome"]), pick.get("parent_fp"), "adopted"

    parents = b.parents
    if not parents:
        # nothing is PROVEN yet (the strict bar takes weeks), but the incubator
        # has already found which genomes are working: breed from those rather
        # than starting from scratch. Falls back to a random genome only while
        # there are no elites at all.
        elites = incubator.cached_elites()
        if elites:
            for _try in range(12):
                g, pel, origin = incubator.child_of(elites)
                if gene_pool.fingerprint(g) not in claimed:
                    break
            claimed.add(gene_pool.fingerprint(g))
            if pel:           # a sim elite is not in the pool: its depth is carried here
                b.child_generation = int(pel.get("generation") or 0) + 1
            return g, (pel or {}).get("fp"), origin
        g, _fp, _n = gene_pool.distinct_child(genome.random_genome, claimed=claimed)
        return g, None, "seed"

    parent = _pick_parent(parents)
    pg = genome.loads(parent["genome"])
    if len(parents) > 1 and random.random() < CROSSOVER_RATE:
        other = _pick_parent({f: [e for e in v if e["fingerprint"] != parent["fingerprint"]]
                              for f, v in parents.items()})
        if other is not None:
            og = genome.loads(other["genome"])
            g, _fp, _n = gene_pool.distinct_child(
                lambda: genome.mutate(genome.crossover(pg, og), rate=0.15), claimed=claimed)
            return g, parent["fingerprint"], "crossover"
    g, _fp, _n = gene_pool.distinct_child(lambda: genome.mutate(pg), claimed=claimed)
    return g, parent["fingerprint"], "mutation"


def _spawn_replacement(user_id: int | None, breeder: "_Breeder | None" = None) -> int | None:
    """One fresh candidate for `user_id`'s fleet, or None if that fleet is
    already at its max_fleet_bots guardrail (bots.spawn_bot decides)."""
    # a sim bot that has already earned a real-fill trial beats any fresh guess
    if user_id is not None:
        try:
            promoted = incubator.promote_next(user_id)
        except Exception as e:  # noqa: BLE001 — never block the refill on the incubator
            print("incubator promote failed: {}".format(e), flush=True)
            promoted = None
        if promoted is not None:
            return promoted
    breeder = breeder if breeder is not None else _Breeder()
    g, parent_fp, origin = _replacement_genome(user_id, breeder)
    return bots.spawn_bot(g, origin=origin, user_id=user_id, parent_fp=parent_fp,
                          generation=breeder.child_generation)


def _fill_fleet(user_id: int | None, breeder: "_Breeder | None" = None) -> int:
    """Spawn until `user_id`'s fleet is back at its max_fleet_bots size.

    Every pass that retires bots calls this afterwards, so a fleet is only
    ever short between a retirement and the refill in the same run: the cap
    is the fleet size, not a limit it drifts below. Returns how many were
    spawned."""
    breeder = breeder if breeder is not None else _Breeder()
    spawned = 0
    while bots.fleet_headroom(user_id) > 0:
        if _spawn_replacement(user_id, breeder) is None:
            break               # refused at the door — stop rather than spin
        spawned += 1
    return spawned


def evolve(user_id: int | None = None) -> dict:
    """Breeding/culling for ONE user's fleet, drawing on the shared pool.

    The bots that get replaced and the slots that get filled are that
    user's; the genomes that fill them come from the population, and the
    evidence behind every choice is the population's too. Custom bots
    (pool='custom') are never read from here as parents and never touched
    by retire/archive below — they run their own live simulation entirely
    outside this loop."""
    sql = ("SELECT id FROM bots WHERE status='failing' "
           "AND COALESCE(pool,'fleet')='fleet'")
    params: list = []
    if user_id is not None:
        sql += " AND user_id=?"
        params.append(int(user_id))
    with bots._conn() as c:
        failing = [r["id"] for r in c.execute(sql, params).fetchall()]

    breeder = _Breeder()
    archived = replaced = 0
    for bid in failing:
        st = bots.stats(bid)
        # require an actual proven trade count, not just a peak — a bot that
        # spiked and blew up in a handful of trades is noise, not a genome
        # worth keeping around forever
        if (st["peak_pl_pct"] >= bots.ARCHIVE_PEAK_PCT
                and st["closed_count"] >= bots.MIN_ELIGIBLE_TRADES):
            bots.archive_bot(bid, "peak{:.0f}_then_fail".format(bots.ARCHIVE_PEAK_PCT))
            archived += 1
        # retire first, THEN refill its slot: with a hard fleet-size cap a
        # fleet sitting at its ceiling has no room for the replacement until
        # the bot it replaces is out
        bots.retire_bot(bid, "retired")
        if _spawn_replacement(user_id, breeder) is not None:
            replaced += 1

    spawned = _fill_fleet(user_id, breeder)

    ev = {"archived": archived, "replaced": replaced, "topped_up": spawned}
    if archived or replaced:      # skip the (common) empty cycle — see log_event()'s docstring
        bots.log_event("evolve", {**ev, "user_id": user_id})
    return ev


def _daily_for_user(user_id: int) -> dict:
    """One user's fleet: reseed / breed / cull, on the shared pool's
    evidence. Their bots, everyone's genomes."""
    baseline_reset = bots.reset_failed_baseline_bots(user_id)
    baseline_seeded = bots.ensure_baseline_bots(user_id)

    if not bots.active_bots(pool="fleet", user_id=user_id):
        seeded = _fill_fleet(user_id)
        return {"seeded": seeded, "baseline_reset": baseline_reset,
                "baseline_seeded": baseline_seeded, "killed_idle": 0,
                "children": 0, "retired": 0}

    # kill bots idle for 3+ days (they keep their ledger day to day, so no
    # trades in 3 days means the genome just isn't firing)
    killed = 0
    for b in bots.stale_active_bots(3, pool="fleet", user_id=user_id):
        bots.retire_bot(b["id"], "retired")
        killed += 1

    # kill bots that trade but go nowhere (no edge, balance flat): they would
    # otherwise hold a slot until the fleet happens to be over its target.
    # Close at the mark first — a retired bot is no longer position-managed.
    for b in bots.stagnant_active(pool="fleet", user_id=user_id):
        bots.flatten_bot(b["id"], "flattened at mark: stagnant")
        bots.retire_bot(b["id"], "retired")
        killed += 1

    # keep exploration (untested random seeds) to its share of the fleet:
    # ease the weakest out past their fair trial; the refill below makes
    # them descendants of winners (bots.EXPLORE_SHARE)
    for b in bots.excess_explorers(user_id):
        bots.flatten_bot(b["id"], "flattened at mark: exploration over budget")
        bots.retire_bot(b["id"], "retired")
        killed += 1

    # kill persistent losers before they bleed to the drawdown floor: a real,
    # lasting negative edge is not going to turn, and it holds a slot all the way
    for b in bots.losing_active(pool="fleet", user_id=user_id):
        bots.flatten_bot(b["id"], "flattened at mark: persistent loser")
        bots.retire_bot(b["id"], "retired")
        killed += 1

    # retest winners: a lightly-mutated child per top POOLED-proven genome
    # (never an unproven one), but not from a family that already dominates
    # the population — and never a candidate another fleet is already on.
    # Pick the parents first; the children are actually born below, after
    # room has been made for them.
    target = bots.fleet_cap(user_id)        # the fleet's size, ceiling and setpoint
    breeder = _Breeder()
    planned = []
    if breeder.parents:
        budget = max(3, target // 10)
        per_family = max(1, budget // len(breeder.parents))   # every proven family gets its share
        for members in breeder.parents.values():
            planned.extend(members[:per_family])

    # hold near target: retire the weakest ESTABLISHED bots — gated on the
    # same proven track record as breeding, and judged on the POOLED record
    # where one exists, so a genome another fleet has already shown to be
    # bad dies here too instead of burning this user's days rediscovering it.
    #
    # This runs BEFORE the children are spawned, not after. max_fleet_bots
    # is a hard ceiling, so a fleet already at it has to bury its weakest
    # before it can breed; culling afterwards would just mean every child
    # was refused at the door. Same steady-state population either way.
    over = len(bots.active_bots(pool="fleet", user_id=user_id)) + len(planned) - target
    retired = 0
    if over > 0:
        for b in bots.worst_active(over, pool="fleet", user_id=user_id, pooled=True,
                                   min_trades=bots.MIN_ELIGIBLE_TRADES,
                                   min_age_hours=bots.MIN_ELIGIBLE_DAYS * 24):
            # ranked on fitness_marked, so a bot hiding behind an open loser
            # is fair game — close it at the mark first, never leave it open
            bots.flatten_bot(b["id"])
            bots.retire_bot(b["id"], "retired")
            retired += 1

    children = 0
    for e in planned:
        pg = genome.loads(e["genome"])
        g, _fp, _n = gene_pool.distinct_child(lambda: genome.mutate(pg),
                                              claimed=breeder.claimed)
        if bots.spawn_bot(g, origin="mutation", user_id=user_id,
                          parent_fp=e["fingerprint"]) is not None:
            children += 1

    # every slot the idle-cull and the weakest-cull above freed and the
    # children did not take goes back to a fresh candidate: the day starts
    # with a FULL fleet, always
    seeded = _fill_fleet(user_id, breeder)

    return {"killed_idle": killed, "children": children, "retired": retired,
            "baseline_reset": baseline_reset, "baseline_seeded": baseline_seeded,
            "seeded": seeded}


def daily_maintenance() -> dict:
    """Pre-open pass over every configured fleet. Custom bots keep running
    on their own schedule (see run_cycle/_tick via bots.live_bots()) and are
    never seeded, retired, bred from, or pruned by this pass."""
    prepare()
    _settle_expired(datetime.now(ET))     # sweep anything a broken session left open

    ev_never = bots.retire_never_traded()      # first trade within 24h or out

    users = fleet_users()      # every configured fleet, paused ones included:
                               # maintenance is bookkeeping, not trading
    ev = {"fleets": len(users), "seeded": 0, "killed_idle": 0, "children": 0,
          "retired": 0, "baseline_reset": 0, "baseline_seeded": 0,
          "never_traded": sum(ev_never.values())}
    for uid in users:
        for k, v in _daily_for_user(uid).items():
            ev[k] = ev.get(k, 0) + v
    try:
        ev["incubator"] = incubator.maintain(force=True)
    except Exception as e:  # noqa: BLE001 — the fleet's daily pass must still complete
        print("incubator maintain failed: {}".format(e), flush=True)
    ev["pruned"] = bots.prune_retired()
    ev["active"] = len(bots.active_bots(pool="fleet"))
    ev["pool"] = gene_pool.size()
    bots.log_event("daily", ev)
    return ev


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "cycle"
    if cmd == "cycle":
        print(run_cycle(), flush=True)
    elif cmd == "fast":
        fast_loop()
    elif cmd == "once":
        prepare()
        now = datetime.now(ET)
        settled = _settle_expired(now)
        _tick(now)
        print({"active": len(bots.active_bots()), "settled": settled}, flush=True)
    elif cmd == "retire-spreads":
        prepare()
        with _engine_lock(60.0) as locked:
            if not locked:
                print("engine busy — try again", flush=True)
                return 1
            print(bots.retire_spread_bots(), flush=True)
    elif cmd == "retire-untraded":
        prepare()
        with _engine_lock(60.0) as locked:
            if not locked:
                print("engine busy — try again", flush=True)
                return 1
            print(bots.retire_never_traded(), flush=True)
    elif cmd == "dedupe-names":
        prepare()
        print({"renamed": bots.dedupe_generated_names()}, flush=True)
    elif cmd == "daily":
        print(daily_maintenance(), flush=True)
    elif cmd == "manager-fill":
        # adds a bot to any manager's empty slot that has one ready; places no
        # orders, but still takes the lock so it can't race a full pass
        from . import manager
        prepare()
        with _engine_lock(20.0) as locked:
            if locked:
                done = manager.fill_empty()
                if done:
                    print(done, flush=True)
    elif cmd == "manager":
        # a manager pass can close positions and place orders, so it runs with
        # the market open (timer: 10:00 and 15:30 ET) and under the engine lock
        from . import manager
        prepare()
        with _engine_lock(60.0) as locked:
            if not locked:
                print("engine busy — try again", flush=True)
                return 1
            print(manager.run_all(), flush=True)
    elif cmd == "reset":
        # `reset` on its own resets EVERY fleet and restarts the shared
        # multiple-comparison count — the right thing after a rules change,
        # which applies to the whole population. `reset <user-id>` resets
        # just that account's fleet and leaves everyone else alone.
        owner = prepare()
        if len(argv) > 1:
            uid = int(argv[1])
            n = bots.reset_fleet("cli_reset", user_id=uid)
            print({"user_id": uid, "retired": n,
                   "spawned": len(bots.active_bots(pool="fleet", user_id=uid))}, flush=True)
        else:
            total = 0
            users = fleet_users() or [owner]
            for i, uid in enumerate(users):
                total += bots.reset_fleet("cli_reset", user_id=uid, reset_pool=(i == 0))
            print({"fleets": len(users), "retired": total,
                   "spawned": sum(len(bots.active_bots(pool="fleet", user_id=u))
                                  for u in users)}, flush=True)
    else:
        print("usage: python -m webapp.bot_engine [cycle|fast|once|daily|manager|manager-fill|retire-spreads|retire-untraded|dedupe-names|reset [USER_ID]]",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
