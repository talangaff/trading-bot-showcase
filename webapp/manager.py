"""Managers: keep N custom bots on ONE account, chosen and replaced for you.

A manager owns a fixed number of slots on one saved account (paper or live).
Every slot gets the same money — total_budget / slots — and every slot bot is
held to the same per-trade limit (position_pct, applied as the bot's
`position_pct_limit` guardrail). The manager clones the best fleet bots into
the slots, retires one that starts failing, and swaps one out when a clearly
better candidate shows up. Spreading the account across several bots is the
point: one bad bot can only lose its own slot.

How a pass decides (run_manager):
  * candidates   — the owner's fleet bots with a real record (trades, days,
                   regimes, positive P&L and robust fitness, bounded drawdown)
                   whose priciest historical entry fits a slot at position_pct
                   (bots.min_balance_for) — a bot that could never afford one
                   contract would sit idle in its slot.
  * diversified  — no two slots share strategy + symbol + style, no genome
                   already running in a slot, and a genome retired recently
                   stays out for a cooldown.
  * replacement  — an incumbent is replaced when it breaches its own limits
                   (drawdown, loss, fading recent fitness), or when a candidate
                   beats its SOURCE bot's robust fitness by swap_margin after
                   min_hold_days, at most max_swaps_per_week times a week.
  * mode         — dry_run logs and messages only; propose waits for approval
                   on the site; auto applies. Auto on a live account needs
                   LIVE_AUTO_MIN_DAYS of earlier passes first.

Orders: retiring or flattening a bot sends real orders, so a pass that touches
a held position only does so while the market is open (a close attempted
after hours would be booked at the mark and leave the ledger and the account
disagreeing). The caller takes the engine lock around anything that applies.
"""

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import account_book, accounts, bot_settings, bots, market_calendar, notify

ET = ZoneInfo("America/New_York")
MODES = ("dry_run", "propose", "auto")
MAX_SLOTS = 10
MIN_SLOT_DOLLARS = 50.0
LIVE_AUTO_MIN_DAYS = 5          # distinct days of earlier passes before auto may run on a live account
PROPOSAL_TTL_DAYS = 3

DEFAULT_RULES = {
    # who may be picked
    "min_trades": 30,
    "min_days": 15,
    "min_regimes": 2,
    "min_profit_factor": 1.2,
    "max_candidate_drawdown_pct": 20.0,
    # when a held slot bot is thrown out
    "max_drawdown_pct": 20.0,
    "stop_loss_pct": 20.0,
    "min_trades_after": 10,
    "flatten_on_breach": False,   # False: a bot holding a position finishes it, then is replaced
    # when a better option replaces a healthy one
    "swap_margin": 0.75,
    "min_hold_days": 5,
    "max_swaps_per_week": 2,
    "cooldown_days": 14,
}
_RULE_TYPES = {k: (int if isinstance(v, int) and not isinstance(v, bool) else
                   bool if isinstance(v, bool) else float)
               for k, v in DEFAULT_RULES.items()}


class ManagerError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# config + storage
# ---------------------------------------------------------------------------
def clean_rules(raw: dict | None) -> dict:
    """DEFAULT_RULES with any recognised, parseable override applied."""
    out = dict(DEFAULT_RULES)
    for k, v in (raw or {}).items():
        if k not in out or v in (None, ""):
            continue
        t = _RULE_TYPES[k]
        try:
            if t is bool:
                out[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
            else:
                out[k] = t(float(v))
        except (TypeError, ValueError):
            continue
    return out


def rules_of(m: dict) -> dict:
    try:
        raw = json.loads(m.get("rules") or "{}")
    except (TypeError, ValueError):
        raw = {}
    return clean_rules(raw)


def slot_size(m: dict) -> float:
    """Equal money per slot, rounded DOWN to the cent so the slots can never add
    up to more than the budget (3 x $1,666.67 would be a cent over $5,000)."""
    return int(float(m["total_budget"]) / int(m["slots"]) * 100 + 1e-9) / 100.0


def get(mid: int) -> dict | None:
    with bots._conn() as c:
        r = c.execute("SELECT * FROM managers WHERE id=?", (int(mid),)).fetchone()
    return dict(r) if r else None


def list_for(uid: int | None = None) -> list[dict]:
    with bots._conn() as c:
        if uid is None:
            rows = c.execute("SELECT * FROM managers ORDER BY id").fetchall()
        else:
            rows = c.execute("SELECT * FROM managers WHERE user_id=? ORDER BY id",
                             (int(uid),)).fetchall()
    return [dict(r) for r in rows]


def slot_bots(mid: int, active_only: bool = True) -> list[dict]:
    sql = "SELECT * FROM bots WHERE manager_id=?"
    if active_only:
        sql += " AND status='active'"
    with bots._conn() as c:
        rows = c.execute(sql + " ORDER BY slot, id", (int(mid),)).fetchall()
    return [dict(r) for r in rows]


def managed_capital(mid: int) -> float:
    return sum(b["start_balance"] or 0.0 for b in slot_bots(mid))


def validate(uid: int, *, account_id, slots, total_budget, position_pct, mode,
             confirm_name: str | None = None, existing_id: int | None = None,
             check_pool: bool = True) -> tuple[dict | None, str | None]:
    """(clean config, None) or (None, why not)."""
    try:
        slots = int(slots)
        total_budget = round(float(total_budget), 2)
        position_pct = round(float(position_pct), 2)
    except (TypeError, ValueError):
        return None, "slots, budget and per-trade % must be numbers"
    if not 1 <= slots <= MAX_SLOTS:
        return None, "slots must be between 1 and {}".format(MAX_SLOTS)
    if total_budget < slots * MIN_SLOT_DOLLARS:
        return None, "budget must be at least ${:,.0f} per slot (${:,.0f} for {} slots)".format(
            MIN_SLOT_DOLLARS, slots * MIN_SLOT_DOLLARS, slots)
    if not 1.0 <= position_pct <= 100.0:
        return None, "per-trade % must be between 1 and 100"
    if mode not in MODES:
        return None, "unknown mode"
    acct = accounts.get(uid, account_id)
    if not acct:
        return None, "pick one of your saved accounts"
    old = get(existing_id) if existing_id else None
    changed_account = not old or old["account_id"] != acct["id"]
    if acct["type"] == "live":
        if not accounts.live_enabled():
            return None, "live trading is switched off"
        if changed_account and (confirm_name or "").strip() != acct["name"]:
            return None, "type the live account's exact name to confirm"
        if mode == "auto" and not (old and live_auto_ready(old["id"])):
            return None, ("auto on a live account needs {} days of earlier dry-run/propose "
                          "passes first".format(LIVE_AUTO_MIN_DAYS))
    if check_pool:
        pool_total = bot_settings.load_user(uid).get("custom_pool_total", 0.0) or 0.0
        free = pool_total - bots.custom_capital_used(uid)
        if old:
            free += managed_capital(old["id"])
        if total_budget > free + 1e-6:
            return None, ("budget ${:,.2f} exceeds the ${:,.2f} left in your custom bot capital "
                          "pool — raise the pool total on Custom Bots".format(total_budget, max(0.0, free)))
    return {"account_id": acct["id"], "slots": slots, "total_budget": total_budget,
            "position_pct": position_pct, "mode": mode}, None


def create(uid: int, name: str, cfg: dict, rules: dict | None = None) -> int:
    with bots._conn() as c:
        cur = c.execute(
            "INSERT INTO managers(user_id, name, account_id, slots, total_budget, position_pct, "
            "mode, enabled, rules, created_at) VALUES (?,?,?,?,?,?,?,1,?,?)",
            (int(uid), (name or "").strip() or "Manager", cfg["account_id"], cfg["slots"],
             cfg["total_budget"], cfg["position_pct"], cfg["mode"],
             json.dumps(clean_rules(rules)), _now()))
        return cur.lastrowid


def update(mid: int, cfg: dict, name: str | None = None, rules: dict | None = None) -> None:
    sets, params = [], []
    for k in ("account_id", "slots", "total_budget", "position_pct", "mode"):
        if k in cfg:
            sets.append("{}=?".format(k))
            params.append(cfg[k])
    if name is not None and name.strip():
        sets.append("name=?")
        params.append(name.strip())
    if rules is not None:
        sets.append("rules=?")
        params.append(json.dumps(clean_rules(rules)))
    if sets:
        with bots._conn() as c:
            c.execute("UPDATE managers SET {} WHERE id=?".format(", ".join(sets)),
                      params + [int(mid)])


def set_enabled(mid: int, on: bool) -> None:
    with bots._conn() as c:
        c.execute("UPDATE managers SET enabled=? WHERE id=?", (1 if on else 0, int(mid)))


def delete(mid: int) -> None:
    """Remove the manager. Its bots are kept, released to run by hand —
    nothing is closed and no order is sent."""
    with bots._conn() as c:
        c.execute("UPDATE bots SET manager_id=NULL, slot=NULL WHERE manager_id=?", (int(mid),))
        c.execute("DELETE FROM manager_proposals WHERE manager_id=?", (int(mid),))
        c.execute("DELETE FROM manager_events WHERE manager_id=?", (int(mid),))
        c.execute("DELETE FROM managers WHERE id=?", (int(mid),))


# ---------------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------------
def _event(mid: int, kind: str, reason: str = "", *, slot=None, old=None, new=None,
           detail: dict | None = None) -> None:
    with bots._conn() as c:
        c.execute("INSERT INTO manager_events(manager_id, ts, kind, slot, old_bot_id, new_bot_id, "
                  "reason, detail) VALUES (?,?,?,?,?,?,?,?)",
                  (int(mid), _now(), kind, slot, old, new, reason,
                   json.dumps(detail, default=str) if detail else None))


def events(mid: int, limit: int = 60) -> list[dict]:
    with bots._conn() as c:
        rows = c.execute("SELECT * FROM manager_events WHERE manager_id=? ORDER BY id DESC LIMIT ?",
                         (int(mid), int(limit))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["detail"] = json.loads(d["detail"]) if d["detail"] else {}
        except ValueError:
            d["detail"] = {}
        out.append(d)
    return out


def _seen(mid: int, kind: str, key: str, hours: float = 20.0) -> bool:
    """Has this exact thing been logged lately? Keeps a twice-daily dry run
    from repeating itself."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    with bots._conn() as c:
        rows = c.execute("SELECT detail FROM manager_events WHERE manager_id=? AND kind=? AND ts>=?",
                         (int(mid), kind, since)).fetchall()
    for r in rows:
        try:
            if (json.loads(r["detail"] or "{}") or {}).get("key") == key:
                return True
        except ValueError:
            continue
    return False


def _applied_since(mid: int, days: float) -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with bots._conn() as c:
        return c.execute("SELECT COUNT(*) FROM manager_events WHERE manager_id=? AND kind='replace' "
                         "AND ts>=?", (int(mid), since)).fetchone()[0]


def _recently_retired_fps(mid: int, days: float) -> set[str]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with bots._conn() as c:
        rows = c.execute("SELECT detail FROM manager_events WHERE manager_id=? AND kind='replace' "
                         "AND ts>=?", (int(mid), since)).fetchall()
    out = set()
    for r in rows:
        try:
            fp = (json.loads(r["detail"] or "{}") or {}).get("old_fp")
        except ValueError:
            fp = None
        if fp:
            out.add(fp)
    return out


def live_auto_ready(mid: int) -> bool:
    """Enough earlier passes (on distinct days, in dry_run/propose) to trust auto with real money."""
    days = set()
    for e in events(mid, 500):
        if e["kind"] == "run" and (e["detail"] or {}).get("mode") in ("dry_run", "propose"):
            days.add(e["ts"][:10])
    return len(days) >= LIVE_AUTO_MIN_DAYS


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def _style_key(r: dict) -> tuple:
    return (r.get("strategy"), r.get("symbol"), r.get("style"))


def candidate_check(r: dict, rules: dict, size: float, pct: float) -> str | None:
    """Why a fleet bot can't be picked, or None when it can."""
    if (r.get("closed_count") or 0) < rules["min_trades"]:
        return "only {} closed trades".format(r.get("closed_count") or 0)
    if (r.get("active_days") or 0) < rules["min_days"]:
        return "only {} active days".format(r.get("active_days") or 0)
    if (r.get("regime_count") or 0) < rules["min_regimes"]:
        return "only {} regimes".format(r.get("regime_count") or 0)
    if (r.get("pl_dollar") or 0.0) <= 0:
        return "not profitable"
    if (r.get("robust_fitness") or 0.0) <= 0:
        return "robust fitness not positive"
    if (r.get("profit_factor") or 0.0) < rules["min_profit_factor"]:
        return "profit factor {}".format(r.get("profit_factor"))
    if (r.get("max_drawdown_pct") or 0.0) > rules["max_candidate_drawdown_pct"]:
        return "drawdown {:.1f}%".format(r.get("max_drawdown_pct") or 0.0)
    need = bots.min_balance_for(bots.entry_cost_history(r["id"]), 1, pct)
    if need is None:
        return "no entry history to size from"
    if need > size + 1e-9:
        return "needs a ${:,.0f} slot at {:g}% per trade".format(need, pct)
    return None


def ranked_candidates(uid: int, rules: dict, size: float, pct: float, *, exclude_fps=(),
                      fresh: bool = False) -> tuple[list[dict], dict]:
    """(eligible fleet rows best-first, {bot id: reason it was left out})."""
    rows = bots.leaderboard("active", pool="fleet", fresh=fresh, user_id=uid)
    ok, why = [], {}
    skip = set(exclude_fps)
    for r in rows:
        if r.get("genome_fp") and r["genome_fp"] in skip:
            why[r["id"]] = "genome already used or in cooldown"
            continue
        reason = candidate_check(r, rules, size, pct)
        if reason:
            why[r["id"]] = reason
        else:
            ok.append(r)
    ok.sort(key=lambda r: (r.get("robust_fitness") or 0.0, r.get("pl_pct") or 0.0), reverse=True)
    return ok, why


def _pick(cands: list[dict], taken_keys: set, taken_fps: set, n: int) -> list[dict]:
    """Best-first, no repeated strategy+symbol+style; falls back to merely
    distinct genomes only if the strict pass can't fill the slots."""
    out, keys, fps = [], set(taken_keys), set(taken_fps)
    for strict in (True, False):
        for r in cands:
            if len(out) >= n:
                return out
            if r in out or (r.get("genome_fp") and r["genome_fp"] in fps):
                continue
            if strict and _style_key(r) in keys:
                continue
            out.append(r)
            keys.add(_style_key(r))
            if r.get("genome_fp"):
                fps.add(r["genome_fp"])
    return out


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def _breach(row: dict, start: float, rules: dict) -> str | None:
    dd = row.get("drawdown_pct") or 0.0
    if dd >= rules["max_drawdown_pct"]:
        return "drawdown {:.1f}% (limit {:g}%)".format(dd, rules["max_drawdown_pct"])
    pl = row.get("pl_pct")
    if pl is not None and pl <= -rules["stop_loss_pct"]:
        return "down {:.1f}% (limit {:g}%)".format(-pl, rules["stop_loss_pct"])
    fr = row.get("fitness_recent")
    if ((row.get("closed_count") or 0) >= rules["min_trades_after"] and fr is not None
            and fr < 0 and (row.get("pl_dollar") or 0.0) < 0):
        return "recent fitness {:.2f} with a losing record".format(fr)
    return None


def _source_score(inc: dict, fleet_by_id: dict) -> float:
    """How good the incumbent's SOURCE bot looks now — a like-for-like
    comparison against candidates (fleet ledger vs fleet ledger). A source that
    has been retired falls back to its own stats; one that can't be read counts
    as worthless, which is what lets a stale slot be replaced."""
    pid = inc.get("parent_id")
    r = fleet_by_id.get(pid)
    if r is None and pid:
        try:
            r = bots.stats(pid)
        except Exception:  # noqa: BLE001
            r = None
    return float((r or {}).get("robust_fitness") if r and r.get("robust_fitness") is not None else -9.0)


def _held_days(b: dict, now: datetime) -> float:
    try:
        t = datetime.fromisoformat(b["created_at"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return (now.astimezone(timezone.utc) - t).total_seconds() / 86400.0


def plan(m: dict, now: datetime | None = None, *, fresh: bool = False) -> dict:
    """What a pass would do right now, without doing any of it:
    {size, pct, slots:[...], actions:[...], candidates:[...], skipped:{...}, notes:[...]}."""
    now = now or datetime.now(ET)
    uid, mid = m["user_id"], m["id"]
    rules, size, pct = rules_of(m), slot_size(m), float(m["position_pct"])
    n_slots = int(m["slots"])
    held = [b for b in slot_bots(mid) if b.get("slot") and b["slot"] <= n_slots]
    by_slot = {b["slot"]: b for b in held}
    fleet = bots.leaderboard("active", pool="fleet", fresh=fresh, user_id=uid)
    fleet_by_id = {r["id"]: r for r in fleet}
    crows = {r["id"]: r for r in bots.leaderboard("active", pool="custom", fresh=fresh, user_id=uid)}

    used_fps = {b["genome_fp"] for b in held if b.get("genome_fp")}
    cooldown = _recently_retired_fps(mid, rules["cooldown_days"])
    cands, skipped = ranked_candidates(uid, rules, size, pct, exclude_fps=used_fps | cooldown,
                                       fresh=fresh)
    notes, actions, slots_view = [], [], []
    chosen_ids: set[int] = set()

    # 1) incumbents that breached their own limits go regardless of margin
    keep = {}
    finishing: set[int] = set()       # slots whose bot is working off its last position
    for s in range(1, n_slots + 1):
        b = by_slot.get(s)
        if not b:
            continue
        row = crows.get(b["id"], {})
        if b.get("drain"):
            # already told to go: once flat it is replaced at once, until then it
            # only manages the position it holds
            slots_view.append({"slot": s, "bot": b, "row": row, "breach": None})
            if bots.has_open_positions(b["id"]):
                finishing.add(s)
                notes.append("slot {}: {} is finishing its position, then it is replaced".format(s, b["name"]))
            else:
                actions.append({"type": "replace", "slot": s, "old_bot_id": b["id"],
                                "old_fp": b.get("genome_fp"), "flatten": False, "drain_done": True,
                                "drain": int(b["drain"]),
                                "reason": "finished its last position — replacing it"})
            continue
        why = _breach(row, size, rules)
        slots_view.append({"slot": s, "bot": b, "row": row, "breach": why})
        if why:
            actions.append({"type": "replace", "slot": s, "old_bot_id": b["id"],
                            "old_fp": b.get("genome_fp"), "reason": why, "drain": 1,
                            "flatten": bool(rules["flatten_on_breach"])})
        else:
            keep[s] = (b, row)

    # 2) fill the empty slots and the ones just vacated, best + diversified
    open_slots = [s for s in range(1, n_slots + 1) if s not in keep and s not in finishing]
    taken_keys = {_style_key(fleet_by_id[b["parent_id"]]) for b, _ in keep.values()
                  if b.get("parent_id") in fleet_by_id}
    picks = _pick(cands, taken_keys, {b.get("genome_fp") for b, _ in keep.values() if b.get("genome_fp")},
                  len(open_slots))
    for s, r in zip(open_slots, picks):
        chosen_ids.add(r["id"])
        a = next((x for x in actions if x["slot"] == s), None)
        if a:
            a.update({"source_id": r["id"], "source_name": r["name"],
                      "source_score": r.get("robust_fitness")})
        else:
            actions.append({"type": "fill", "slot": s, "old_bot_id": None, "source_id": r["id"],
                            "source_name": r["name"], "source_score": r.get("robust_fitness"),
                            "reason": "empty slot"})
    # a breached slot with nobody to put in it: still leave it (retire only
    # when there is a replacement, so a quiet candidate pool never empties the account)
    undrain = []
    for a in list(actions):
        if a["type"] == "replace" and "source_id" not in a:
            if a.get("drain_done") and a.get("drain") == 2:
                undrain.append(a["old_bot_id"])       # it was only being swapped for a better bot that is gone
                actions.remove(a)
                continue
            notes.append("slot {}: {} — no eligible replacement yet, keeping it".format(a["slot"], a["reason"]))
            actions.remove(a)

    # 3) healthy incumbents beaten by a clearly better candidate
    left = max(0, rules["max_swaps_per_week"] - _applied_since(mid, 7)
               - sum(1 for a in actions if a["type"] == "replace"))
    spare = [r for r in cands if r["id"] not in chosen_ids]
    if left and spare and keep:
        kept_keys = {_style_key(fleet_by_id[b["parent_id"]]) for b, _ in keep.values()
                     if b.get("parent_id") in fleet_by_id}
        order = sorted(keep.items(), key=lambda kv: _source_score(kv[1][0], fleet_by_id))
        for s, (b, row) in order:
            if left <= 0:
                break
            if _held_days(b, now) < rules["min_hold_days"]:
                continue
            mine = _source_score(b, fleet_by_id)
            own_key = _style_key(fleet_by_id[b["parent_id"]]) if b.get("parent_id") in fleet_by_id else None
            others = kept_keys - {own_key}
            used_now = {x["genome_fp"] for x, _ in keep.values() if x.get("genome_fp")}
            better = _pick([r for r in spare if (r.get("robust_fitness") or 0.0) >= mine + rules["swap_margin"]],
                           others, used_now, 1)
            if not better:
                continue
            r = better[0]
            spare.remove(r)
            chosen_ids.add(r["id"])
            kept_keys = others | {_style_key(r)}
            actions.append({"type": "replace", "slot": s, "old_bot_id": b["id"],
                            "old_fp": b.get("genome_fp"), "source_id": r["id"],
                            "source_name": r["name"], "source_score": r.get("robust_fitness"),
                            "reason": "{} beats its source ({:.2f} vs {:.2f}, margin {:g})".format(
                                r["name"], r.get("robust_fitness") or 0.0, mine, rules["swap_margin"]),
                            "flatten": False, "drain": 2})
            left -= 1
    elif not left and keep:
        notes.append("weekly swap limit reached")

    actions.sort(key=lambda a: a["slot"])
    return {"size": size, "pct": pct, "per_trade_cap": round(size * pct / 100.0, 2),
            "slots": slots_view, "actions": actions, "candidates": cands[:15],
            "n_candidates": len(cands), "skipped": skipped, "notes": notes, "undrain": undrain}


def preview(uid: int, slots: int, total_budget: float, position_pct: float,
            rules: dict | None = None) -> dict:
    """What a configuration would look like against today's fleet, for the form."""
    rules = clean_rules(rules)
    size = int(total_budget / max(1, slots) * 100 + 1e-9) / 100.0
    cands, skipped = ranked_candidates(uid, rules, size, position_pct)
    picks = _pick(cands, set(), set(), slots)
    anywhere, _ = ranked_candidates(uid, rules, 1e12, position_pct)    # eligible at any slot size
    needs = sorted(v for v in (bots.min_balance_for(bots.entry_cost_history(r["id"]), 1, position_pct)
                               for r in anywhere) if v)
    return {"size": size, "per_trade_cap": round(size * position_pct / 100.0, 2),
            "n_candidates": len(cands),
            "picks": [{"name": r["name"], "strategy": r.get("strategy"), "symbol": r.get("symbol"),
                       "robust_fitness": r.get("robust_fitness"), "pl_pct": r.get("pl_pct")}
                      for r in picks],
            "min_slot_median": needs[len(needs) // 2] if needs else None,
            "min_slot_lowest": needs[0] if needs else None,
            "enough": len(picks) >= slots}


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------
def market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(ET)
    b = market_calendar.session_bounds(now.astimezone(ET).date())
    if not b:
        return False
    mins = now.astimezone(ET).hour * 60 + now.astimezone(ET).minute
    return b[0] + 5 <= mins < b[1] - 5


def _assign(bot_id: int, mid: int, slot: int, account_id: int) -> None:
    with bots._conn() as c:
        c.execute("UPDATE bots SET manager_id=?, slot=?, account_id=?, enabled=1 WHERE id=?",
                  (int(mid), int(slot), int(account_id), int(bot_id)))


def apply_action(m: dict, act: dict, now: datetime | None = None) -> tuple[bool, str]:
    """Carry out one planned fill/replace. (ok, message). Never partial in a
    way that loses money: the old bot is only retired when it is flat."""
    now = now or datetime.now(ET)
    uid, mid, slot = m["user_id"], m["id"], int(act["slot"])
    acct = accounts.get(uid, m["account_id"])
    if not acct:
        return False, "the manager's account no longer exists"
    size, pct = slot_size(m), float(m["position_pct"])
    src = bots.get_bot(act.get("source_id")) if act.get("source_id") else None
    if not src or src.get("user_id") != uid or src.get("pool") != "fleet" or src.get("status") != "active":
        return False, "the chosen fleet bot is no longer available"
    if any(b["slot"] == slot and b["id"] != act.get("old_bot_id") for b in slot_bots(mid)):
        return False, "slot {} is already filled".format(slot)

    old = bots.get_bot(act["old_bot_id"]) if act.get("old_bot_id") else None
    if act.get("old_bot_id") and (not old or old.get("manager_id") != mid or old.get("status") != "active"):
        return False, "slot {}'s bot has already changed".format(slot)
    if old and bots.has_open_positions(old["id"]):
        if not act.get("flatten"):
            # let it finish: no new trades, exits still managed, replaced the moment it is flat
            if not old.get("drain"):
                with bots._conn() as c:
                    c.execute("UPDATE bots SET drain=?, enabled=0 WHERE id=?",
                              (int(act.get("drain") or 1), old["id"]))
                _event(mid, "drain", act.get("reason", ""), slot=slot, old=old["id"],
                       detail={"source_name": act.get("source_name")})
            return False, "slot {}'s bot is holding a position — it takes no new trades and is replaced once flat".format(slot)
        if not market_open(now):
            return False, "slot {}'s bot is holding a position and the market is closed".format(slot)
        bots.flatten_bot(old["id"], note="closed — manager replacing this bot")
        if bots.has_open_positions(old["id"]):
            return False, "slot {}'s bot could not be fully closed — will retry".format(slot)

    new_id = bots.clone_bot(
        src["id"], name="{} · {} s{}".format(src["name"], m["name"], slot)[:80],
        start_balance=size, user_id=uid,
        guardrail_overrides={"position_pct_limit": pct, "min_contracts": 1})
    if not new_id:
        return False, "could not clone {}".format(src["name"])
    _assign(new_id, mid, slot, acct["id"])
    err = bots.live_gate(bots.get_bot(new_id), acct)
    if err:
        bots.delete_bot(new_id)
        return False, err
    if old:
        bots.retire_bot(old["id"], "retired")
    bots.invalidate_caches()
    kind = "replace" if old else "fill"
    _event(mid, kind, act.get("reason", ""), slot=slot, old=old["id"] if old else None, new=new_id,
           detail={"source_id": src["id"], "source_name": src["name"],
                   "old_fp": old.get("genome_fp") if old else None, "size": size, "pct": pct})
    return True, "{} → {} ({})".format("slot {}".format(slot), src["name"], kind)


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------
def proposals(mid: int, status: str = "pending") -> list[dict]:
    with bots._conn() as c:
        rows = c.execute("SELECT * FROM manager_proposals WHERE manager_id=? AND status=? ORDER BY id",
                         (int(mid), status)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["action"] = json.loads(d["action"])
        except ValueError:
            d["action"] = {}
        out.append(d)
    return out


def _propose(mid: int, act: dict) -> int | None:
    for p in proposals(mid):
        a = p["action"]
        if (a.get("slot"), a.get("source_id"), a.get("old_bot_id")) == (
                act.get("slot"), act.get("source_id"), act.get("old_bot_id")):
            return None
    with bots._conn() as c:
        cur = c.execute("INSERT INTO manager_proposals(manager_id, ts, action) VALUES (?,?,?)",
                        (int(mid), _now(), json.dumps(act, default=str)))
        return cur.lastrowid


def _expire_proposals(mid: int) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=PROPOSAL_TTL_DAYS)).isoformat(timespec="seconds")
    with bots._conn() as c:
        c.execute("UPDATE manager_proposals SET status='expired' WHERE manager_id=? "
                  "AND status='pending' AND ts<?", (int(mid), cutoff))


def resolve_proposal(pid: int, approve: bool, now: datetime | None = None) -> tuple[bool, str]:
    with bots._conn() as c:
        r = c.execute("SELECT * FROM manager_proposals WHERE id=?", (int(pid),)).fetchone()
    if not r or r["status"] != "pending":
        return False, "that proposal is no longer pending"
    if not approve:
        with bots._conn() as c:
            c.execute("UPDATE manager_proposals SET status='rejected' WHERE id=?", (int(pid),))
        return True, "rejected"
    m = get(r["manager_id"])
    if not m:
        return False, "manager not found"
    ok, msg = apply_action(m, json.loads(r["action"]), now)
    if ok:
        with bots._conn() as c:
            c.execute("UPDATE manager_proposals SET status='approved' WHERE id=?", (int(pid),))
    return ok, msg


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------
def _describe(a: dict) -> str:
    if a["type"] == "fill":
        return "slot {}: add {}".format(a["slot"], a.get("source_name"))
    return "slot {}: replace bot #{} with {} — {}".format(
        a["slot"], a["old_bot_id"], a.get("source_name"), a.get("reason"))


def _tell(m: dict, lines: list[str], head: str) -> None:
    if not lines:
        return
    try:
        notify.send("🤖 {} — {}\n".format(m["name"], head) + "\n".join(lines),
                    buttons=[notify._button("Open manager", "/bots/managers/{}".format(m["id"]))])
    except Exception as e:  # noqa: BLE001 — a failed message never fails the pass
        print("[manager] notify failed: {}".format(e), flush=True)


def run_manager(mid: int, now: datetime | None = None, *, dry: bool = False,
                fills_only: bool = False) -> dict:
    """One pass. `dry=True` never changes anything whatever the manager's mode
    (the page's "Run now" button). `fills_only=True` is the quick pass that puts
    a bot into an EMPTY slot as soon as one is ready: it never replaces or closes
    anything, so it places no orders. The caller holds the engine lock when a
    pass can apply."""
    m = get(mid)
    if not m:
        return {"error": "no such manager"}
    if not m["enabled"] and not dry:
        return {"skipped": "manager paused"}
    now = now or datetime.now(ET)
    uid = m["user_id"]
    acct = accounts.get(uid, m["account_id"])
    if not acct:
        _event(mid, "error", "the manager's account is gone")
        return {"error": "account missing"}
    if acct["type"] == "live" and not accounts.live_enabled() and not dry:
        _event(mid, "error", "live trading is switched off — nothing done")
        return {"error": "live trading off"}
    mode = "dry_run" if dry else m["mode"]
    if mode == "auto" and acct["type"] == "live" and not live_auto_ready(mid):
        mode = "propose"
    _expire_proposals(mid)
    try:
        p = plan(m, now, fresh=not dry and not fills_only)
    except Exception as e:  # noqa: BLE001
        _event(mid, "error", "planning failed: {}".format(e))
        return {"error": str(e)}
    if mode == "auto" and p.get("undrain"):
        with bots._conn() as c:
            for bid in p["undrain"]:
                c.execute("UPDATE bots SET drain=0, enabled=1 WHERE id=?", (bid,))
        _event(mid, "run", "no better bot left to swap in — kept the bot that was being phased out")
    if fills_only:
        p["actions"] = [a for a in p["actions"] if a["type"] == "fill" or a.get("drain_done")]
    if p["actions"] or not fills_only:       # the quick pass stays out of the history when it found nothing
        _event(mid, "run", "{} action(s) planned{}".format(len(p["actions"]),
                                                          " (filling empty slots)" if fills_only else ""),
               detail={"mode": mode, "actions": len(p["actions"]), "candidates": p["n_candidates"]})
    result = {"mode": mode, "actions": p["actions"], "applied": [], "waiting": [], "notes": p["notes"]}
    lines = []
    keys = accounts.keys_of(acct)
    for a in p["actions"]:
        key = "{}:{}:{}:{}".format(a["type"], a["slot"], a.get("old_bot_id"), a.get("source_id"))
        if mode == "dry_run":
            if not _seen(mid, "would_" + a["type"], key):
                _event(mid, "would_" + a["type"], a.get("reason", ""), slot=a["slot"],
                       old=a.get("old_bot_id"), detail={"key": key, **a})
                lines.append("would " + _describe(a))
            continue
        if mode == "propose":
            pid = _propose(mid, a)
            if pid:
                _event(mid, "propose", a.get("reason", ""), slot=a["slot"], old=a.get("old_bot_id"),
                       detail={"key": key, "proposal": pid, **a})
                lines.append("proposal: " + _describe(a))
            continue
        # auto
        rep = account_book.last_report(keys) if keys else None
        if rep and not rep.get("ok") and not rep.get("error"):
            _event(mid, "skip", "account ledgers don't match the account — not changing bots today",
                   slot=a["slot"])
            result["waiting"].append("ledger mismatch")
            break
        ok, msg = apply_action(m, a, now)
        (result["applied"] if ok else result["waiting"]).append(msg)
        if ok:
            lines.append("✅ " + _describe(a))
        elif not _seen(mid, "skip", msg):
            _event(mid, "skip", msg, slot=a["slot"], detail={"key": msg})
            lines.append("⏳ " + msg)
    with bots._conn() as c:
        c.execute("UPDATE managers SET last_run_at=? WHERE id=?", (_now(), mid))
    if lines:
        _tell(m, lines, {"dry_run": "dry run", "propose": "needs approval", "auto": "changes"}[mode])
    return result


def run_all(now: datetime | None = None) -> dict:
    """Every enabled manager. One manager failing never stops the others."""
    out = {}
    for m in list_for():
        if not m["enabled"]:
            continue
        try:
            out[m["id"]] = run_manager(m["id"], now)
        except Exception as e:  # noqa: BLE001
            try:
                _event(m["id"], "error", "pass failed: {}".format(e))
            except Exception:  # noqa: BLE001
                pass
            out[m["id"]] = {"error": str(e)}
    return out


FILL_RECHECK_S = 600       # with nothing ready for an empty slot, look again after this long


def _drained_ready(m: dict) -> bool:
    """A bot being phased out has closed its last position — its slot can be refilled."""
    return any(b.get("drain") and (b.get("slot") or 0) <= int(m["slots"]) and not bots.has_open_positions(b["id"])
               for b in slot_bots(m["id"]))


def empty_slots(m: dict) -> int:
    return max(0, int(m["slots"]) - len([b for b in slot_bots(m["id"]) if (b.get("slot") or 0) <= int(m["slots"])]))


def fill_empty(mid: int | None = None, *, force: bool = False, now: datetime | None = None) -> dict:
    """Put a ready bot into every empty slot, now. Called when a manager is created,
    edited or resumed (`force`), and every couple of minutes while the market is
    open for any manager that still has an empty slot. A manager whose last look
    found nobody ready is only re-checked every FILL_RECHECK_S unless forced.
    Holds no orders: it only adds bots, so it is safe at any hour."""
    import time as _t
    out = {}
    for m in ([get(mid)] if mid else list_for()):
        if not m or not m["enabled"] or not (empty_slots(m) or _drained_ready(m)):
            continue
        key = "mgr_fill_{}".format(m["id"])
        last = bots._meta_get(key)
        if not force and last and _t.time() - float(last) < FILL_RECHECK_S:
            continue
        try:
            r = run_manager(m["id"], now, fills_only=True)
        except Exception as e:  # noqa: BLE001
            _event(m["id"], "error", "fill pass failed: {}".format(e))
            r = {"error": str(e)}
        # a pass that filled something looks again straight away; one that didn't waits
        bots._meta_set(key, "0" if r.get("applied") else str(_t.time()))
        out[m["id"]] = r
    return out
