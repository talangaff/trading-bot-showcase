"""Trade journal — SQLite-backed.

Five trade types:
  * calendar       — 2-leg calendar spread (the earnings strategy)
  * credit_spread  — 2-leg vertical sold for a credit (calls or puts)
  * single_option  — one long/short call or put
  * stock          — long/short shares
  * stock_swing    — shares sized off a stop and a max loss, scaled out in
                     equal trims across one or more take-profit prices
                     (see swing_plan())

A journal carries ONE strategy (journals.strategy) and every trade in it is
that type — the trade form offers no other, so logging a trade is a
question of numbers rather than of which shape to pick. `strategy='any'`
is the exception, and the default every pre-existing journal is migrated
to: those keep the full picker, so nothing already logged is stranded.
The lock is enforced here, in create/update/patch, not only in the form.

Multiple journals (paper / live), each owned by one user (journals.user_id —
see webapp/auth.py). Raw inputs are stored; P/L, position %, days-to-earnings,
win flag and the per-journal running aggregates are computed on read in
_enrich().
"""

import math
import sqlite3
from datetime import date, datetime, timezone

from . import config

DB_PATH = config.DATA_DIR / "journal.db"

POSITION_PCT_LIMIT = 6.0
CHECKPOINT_TRADES = 15
MAX_DRAWDOWN_LIMIT = 25.0

TRADE_TYPES = ("calendar", "credit_spread", "single_option", "stock", "stock_swing")

# trade types whose size is in shares rather than contracts
SHARE_TYPES = ("stock", "stock_swing")

# stock swings size in fractional shares, to this many decimal places
SWING_SHARE_DECIMALS = 2

# What a journal may be marked as: one trade type, or "any" — the
# everything-goes setting old journals migrate to.
ANY_STRATEGY = "any"
JOURNAL_STRATEGIES = (ANY_STRATEGY,) + TRADE_TYPES
STRATEGY_LABELS = {
    ANY_STRATEGY: "Any strategy",
    "calendar": "Calendar spread",
    "credit_spread": "Credit spread",
    "single_option": "Single option",
    "stock": "Stock",
    "stock_swing": "Stock Swings",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journals (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'live',
    strategy   TEXT NOT NULL DEFAULT 'any',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY,
    journal_id INTEGER NOT NULL REFERENCES journals(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    trade_type        TEXT DEFAULT 'single_option',
    direction         TEXT,
    option_type       TEXT,

    ticker            TEXT NOT NULL,
    earnings_date     TEXT,
    earnings_session  TEXT,
    entry_at          TEXT,
    exit_at           TEXT,

    ts_slope          REAL,
    avg_volume        REAL,
    iv30_rv30         REAL,
    recommendation    TEXT,
    flipped_earlier   INTEGER DEFAULT 0,

    strike            REAL,
    short_strike      REAL,
    long_strike       REAL,
    expiration        TEXT,
    front_expiration  TEXT,
    back_expiration   TEXT,
    contracts         INTEGER,
    shares            REAL,
    short_leg_open    REAL,
    long_leg_open     REAL,
    front_leg_open    REAL,
    back_leg_open     REAL,
    front_leg_close   REAL,
    back_leg_close    REAL,
    open_price        REAL,
    close_price       REAL,

    account_value     REAL,

    deviation_flag    INTEGER DEFAULT 0,
    deviation_notes   TEXT,
    notes             TEXT
);
CREATE TABLE IF NOT EXISTS trims (
    id          INTEGER PRIMARY KEY,
    trade_id    INTEGER NOT NULL REFERENCES trades(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    qty         REAL NOT NULL,
    close_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS adds (
    id         INTEGER PRIMARY KEY,
    trade_id   INTEGER NOT NULL REFERENCES trades(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    qty        REAL NOT NULL,
    price      REAL NOT NULL
);
"""

_ADD_COLUMNS = [
    ("trade_type", "TEXT"), ("direction", "TEXT"), ("option_type", "TEXT"),
    ("expiration", "TEXT"), ("shares", "REAL"),
    ("open_price", "REAL"), ("close_price", "REAL"),
    ("entry_reason", "TEXT"),
    ("draft", "INTEGER"),
    # credit spreads: two strikes and two entry legs of their own rather
    # than the calendar's front/back, which mean something different
    ("short_strike", "REAL"), ("long_strike", "REAL"),
    ("short_leg_open", "REAL"), ("long_leg_open", "REAL"),
    # stock swings: the stop, the dollars you're willing to lose there, and
    # the take-profit prices (comma-separated, nearest first)
    ("stop_loss", "REAL"), ("max_loss", "REAL"), ("take_profits", "TEXT"),
]

TRADE_FIELDS = [
    "trade_type", "direction", "option_type",
    "ticker", "earnings_date", "earnings_session", "entry_at", "exit_at",
    "ts_slope", "avg_volume", "iv30_rv30", "recommendation", "flipped_earlier",
    "strike", "short_strike", "long_strike",
    "expiration", "front_expiration", "back_expiration",
    "contracts", "shares",
    "short_leg_open", "long_leg_open",
    "front_leg_open", "back_leg_open", "front_leg_close", "back_leg_close",
    "open_price", "close_price",
    "stop_loss", "max_loss", "take_profits",
    "account_value",
    "entry_reason",
    "deviation_flag", "deviation_notes", "notes",
]


# Fields the trade form no longer asks for. Logging a trade is meant to be
# quick — a symbol, the position, the money — so the earnings-play context
# and the screener readings were taken off the form. The COLUMNS stay: old
# trades keep what they recorded, and since every save rewrites all of
# TRADE_FIELDS, update/patch carry these forward from the stored row rather
# than nulling them out the first time an old trade is edited.
FORM_RETIRED_FIELDS = (
    "earnings_date", "earnings_session", "entry_at", "exit_at",
    "ts_slope", "avg_volume", "iv30_rv30", "recommendation", "flipped_earlier",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


DEFAULT_JOURNALS = (("Paper Trading", "paper"), ("Live Trading", "live"))


def init(default_owner: int = 1) -> None:
    """Create / migrate the schema. Journals that pre-date accounts (no
    user_id) are handed to `default_owner` — the primary admin."""
    with _conn() as c:
        c.executescript(_SCHEMA)
        for col, decl in _ADD_COLUMNS:
            try:
                c.execute("ALTER TABLE trades ADD COLUMN {} {}".format(col, decl))
            except sqlite3.OperationalError:
                pass
        try:
            c.execute("ALTER TABLE journals ADD COLUMN user_id INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            # every journal that pre-dates per-journal strategies keeps the
            # full picker: they hold whatever was logged before the rule
            c.execute("ALTER TABLE journals ADD COLUMN strategy TEXT NOT NULL DEFAULT 'any'")
        except sqlite3.OperationalError:
            pass
        c.execute("UPDATE journals SET strategy=? WHERE strategy IS NULL OR TRIM(strategy)=''",
                  (ANY_STRATEGY,))
        # tickers used to be stored as "?" when blank, which then came back
        # up in the form's own ticker box
        c.execute("UPDATE trades SET ticker='' WHERE ticker='?'")
        c.execute("CREATE INDEX IF NOT EXISTS ix_journals_user ON journals(user_id)")
        # every pre-existing trade was a calendar spread
        c.execute("UPDATE trades SET trade_type='calendar' WHERE trade_type IS NULL")
        c.execute("UPDATE journals SET kind='paper' WHERE kind='backtest'")
        c.execute("UPDATE journals SET name='Paper Trading' WHERE name='Backtesting' AND kind='paper'")
        c.execute("UPDATE journals SET user_id=? WHERE user_id IS NULL", (int(default_owner),))
    ensure_defaults(default_owner)


def ensure_defaults(user_id: int) -> None:
    """Give a user who has no journals at all the two starter journals."""
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM journals WHERE user_id=?",
                      (int(user_id),)).fetchone()[0]
        if n == 0:
            for name, kind in DEFAULT_JOURNALS:
                c.execute("INSERT INTO journals(name, kind, created_at, user_id) VALUES (?,?,?,?)",
                          (name, kind, _now(), int(user_id)))


# ---------------------------------------------------------------------------
# journals
# ---------------------------------------------------------------------------
def list_journals(user_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM journals WHERE user_id=? "
            "ORDER BY CASE kind WHEN 'live' THEN 0 ELSE 1 END, id", (int(user_id),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["trade_count"] = _count(d["id"])
        out.append(d)
    return out


def get_journal(jid: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM journals WHERE id=?", (jid,)).fetchone()
    return dict(r) if r else None


def create_journal(name: str, kind: str, user_id: int,
                   strategy: str = ANY_STRATEGY) -> int:
    kind = "paper" if kind == "paper" else "live"
    strategy = strategy if strategy in JOURNAL_STRATEGIES else ANY_STRATEGY
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO journals(name, kind, strategy, created_at, user_id) VALUES (?,?,?,?,?)",
            (name.strip() or "Untitled", kind, strategy, _now(), int(user_id)))
        return cur.lastrowid


def strategy_of(jid: int) -> str:
    """The one trade type this journal holds, or "any"."""
    with _conn() as c:
        r = c.execute("SELECT strategy FROM journals WHERE id=?", (jid,)).fetchone()
    s = (r["strategy"] if r else None) or ANY_STRATEGY
    return s if s in JOURNAL_STRATEGIES else ANY_STRATEGY


def forced_type(jid: int) -> str | None:
    """The trade type every trade in `jid` must be, or None if it takes any.

    This is what makes the rule a rule rather than a default in a form:
    create/update/patch all run their values through it, so a trade cannot
    enter a journal as the wrong type no matter where the request came
    from."""
    s = strategy_of(jid)
    return None if s == ANY_STRATEGY else s


def _forced_type_for_trade(tid: int) -> str | None:
    with _conn() as c:
        r = c.execute("SELECT journal_id FROM trades WHERE id=?", (tid,)).fetchone()
    return forced_type(r["journal_id"]) if r else None


def strategy_conflicts(jid: int, strategy: str) -> list[str]:
    """Trade types already logged in `jid` that `strategy` would outlaw.

    Drafts are ignored — an unsaved trade is retyped when it is saved."""
    if strategy == ANY_STRATEGY:
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT COALESCE(trade_type, 'single_option') tt FROM trades "
            "WHERE journal_id=? AND COALESCE(draft,0)=0", (jid,)).fetchall()
    return sorted(r["tt"] for r in rows if r["tt"] != strategy)


def set_strategy(jid: int, strategy: str) -> str | None:
    """Mark an existing journal. Returns None on success, else an error.

    Refused when the journal already holds trades of another type: those
    entries are real history and silently relabelling them would falsify
    it. "Any strategy" is always allowed — that is the setting for a
    journal with a mixed past."""
    if strategy not in JOURNAL_STRATEGIES:
        return "Unknown strategy."
    clash = strategy_conflicts(jid, strategy)
    if clash:
        names = ", ".join(STRATEGY_LABELS.get(t, t).lower() for t in clash)
        return ("This journal already holds trades of another kind ({}). "
                "Leave it on \"Any strategy\", or move those trades to their "
                "own journal first.".format(names))
    with _conn() as c:
        c.execute("UPDATE journals SET strategy=? WHERE id=?", (strategy, jid))
    return None


def delete_journal(jid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE journal_id=?", (jid,))
        c.execute("DELETE FROM journals WHERE id=?", (jid,))


def delete_user_journals(user_id: int) -> int:
    """Remove every journal (and trade) a user owns — used when the account
    itself is deleted. Returns how many journals went."""
    with _conn() as c:
        ids = [r["id"] for r in c.execute(
            "SELECT id FROM journals WHERE user_id=?", (int(user_id),)).fetchall()]
    for jid in ids:
        delete_journal(jid)
    return len(ids)


def count_journals(user_id: int) -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM journals WHERE user_id=?",
                         (int(user_id),)).fetchone()[0]


def _count(jid: int) -> int:
    with _conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM trades WHERE journal_id=? AND COALESCE(draft,0)=0",
            (jid,)).fetchone()[0]


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------
def _coerce(values: dict, force_type: str | None = None) -> dict:
    floats = {"ts_slope", "avg_volume", "iv30_rv30", "strike", "shares",
              "short_strike", "long_strike", "short_leg_open", "long_leg_open",
              "front_leg_open", "back_leg_open", "front_leg_close", "back_leg_close",
              "open_price", "close_price", "account_value",
              "stop_loss", "max_loss"}
    ints = {"contracts"}
    bools = {"flipped_earlier", "deviation_flag"}
    out = {}
    for k in TRADE_FIELDS:
        v = values.get(k)
        if isinstance(v, str):
            v = v.strip()
        if k in bools:
            out[k] = 1 if v in (True, "1", "on", "true", "yes") else 0
        elif k in floats:
            try:
                out[k] = float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                out[k] = None
        elif k in ints:
            try:
                out[k] = int(float(v)) if v not in (None, "") else None
            except (TypeError, ValueError):
                out[k] = None
        else:
            out[k] = (v or None) if v != "" else None

    out["ticker"] = (out.get("ticker") or "").upper()
    out["trade_type"] = out.get("trade_type") if out.get("trade_type") in TRADE_TYPES else "single_option"
    if force_type:                      # the journal's strategy wins, always
        out["trade_type"] = force_type
    if out.get("direction") not in ("long", "short"):
        out["direction"] = None if out["trade_type"] in ("calendar", "credit_spread") else "long"
    if out["trade_type"] in ("calendar", "credit_spread"):
        # neither is "long" or "short" as a whole: a calendar's direction is
        # meaningless and a credit spread is short by construction
        out["direction"] = None
    if out.get("option_type") not in ("call", "put"):
        out["option_type"] = None
    if out.get("earnings_session") not in ("BMO", "AMC"):
        out["earnings_session"] = None
    if out.get("recommendation") not in ("Recommended", "Consider", "Avoid", "Passed"):
        out["recommendation"] = out.get("recommendation") or None
    if out["trade_type"] == "stock_swing":
        _coerce_swing(out)
    else:
        out["stop_loss"] = out["max_loss"] = out["take_profits"] = None
    return out


def _coerce_swing(out: dict) -> None:
    """A swing's share count is not typed in — it is what the stop and the
    max loss say it has to be. Whenever the plan works out, it overwrites
    whatever `shares` arrived with, so the stored size can never drift from
    the risk it was sized for. Direction follows the stop: below the entry
    is a long, above it a short."""
    op, sl = out.get("open_price"), out.get("stop_loss")
    if op is not None and sl is not None and op != sl:
        out["direction"] = "long" if sl < op else "short"
    tps = sort_targets(parse_targets(out.get("take_profits")), op)
    out["take_profits"] = ",".join(_n(p) for p in tps) or None
    plan = swing_plan(op, sl, out.get("max_loss"), tps)
    if plan["shares"]:
        out["shares"] = float(plan["shares"])


# ---------------------------------------------------------------------------
# stock swings — position sizing off the stop
# ---------------------------------------------------------------------------
def parse_targets(raw) -> list[float]:
    """Take-profit prices from a list or a comma/space-separated string.
    Blanks and anything that isn't a positive number are dropped."""
    if raw is None:
        return []
    parts = raw if isinstance(raw, (list, tuple)) else str(raw).replace(",", " ").split()
    out = []
    for p in parts:
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if v > 0 and math.isfinite(v):
            out.append(v)
    return out


def sort_targets(tps: list[float], open_price=None) -> list[float]:
    """Nearest target first — the order the trims will actually fill in."""
    if open_price is None:
        return list(tps)
    return sorted(tps, key=lambda p: abs(p - open_price))


def swing_plan(open_price, stop_loss, max_loss, take_profits) -> dict:
    """How many shares a stock swing buys, and how it scales out.

    risk per share = |entry - stop|. The raw size is max_loss / that, but
    every take-profit sells the SAME number of shares — fractional, to
    SWING_SHARE_DECIMALS places — so each trim is rounded DOWN to that
    precision and the total is that times the number of targets:

        per_trim = floor(max_loss / risk_per_share / targets, to 0.01)
        shares   = per_trim * targets

    Rounding down means the loss at the stop is never more than the max loss
    you gave — it may be a little under. Max profit assumes every target is
    hit, each one selling `per_trim` shares.

    Returns a dict that always has the same keys; `error` explains what is
    missing or wrong when the plan can't be made (shares is then None)."""
    tps = parse_targets(take_profits)
    plan = {
        "direction": None, "risk_per_share": None, "raw_shares": None,
        "targets": len(tps), "per_trim": None, "shares": None,
        "loss_at_stop": None, "unused_risk": None, "max_profit": None,
        "reward_risk": None, "trims": [], "error": None, "warnings": [],
    }
    if open_price is None or stop_loss is None or max_loss is None:
        plan["error"] = "Enter the open price, stop loss and max loss."
        return plan
    if open_price <= 0 or stop_loss <= 0:
        plan["error"] = "Prices must be above zero."
        return plan
    if max_loss <= 0:
        plan["error"] = "Max loss must be above zero."
        return plan
    risk = abs(open_price - stop_loss)
    if risk < 1e-9:
        plan["error"] = "The stop loss can't be the open price."
        return plan
    long = stop_loss < open_price
    m = 1.0 if long else -1.0
    plan["direction"] = "long" if long else "short"
    plan["risk_per_share"] = risk
    plan["raw_shares"] = max_loss / risk
    if not tps:
        plan["error"] = "Add at least one take-profit price."
        return plan
    tps = sort_targets(tps, open_price)
    n = len(tps)
    # the epsilon keeps float noise (e.g. 300/1.5/2 = 99.99999...) from
    # costing a hundredth of a share
    scale = 10 ** SWING_SHARE_DECIMALS
    per_trim = round(math.floor(max_loss / risk / n * scale + 1e-6) / scale, SWING_SHARE_DECIMALS)
    if per_trim <= 0:
        plan["error"] = ("A ${:,.2f} max loss buys {:.4f} shares at ${:,.2f} risk per share "
                         "— not enough for {} of a share per take-profit ({}). Raise the "
                         "max loss, tighten the stop, or use fewer targets."
                         .format(max_loss, max_loss / risk, risk, 1 / scale, n))
        return plan
    shares = round(per_trim * n, SWING_SHARE_DECIMALS)
    plan["per_trim"] = per_trim
    plan["shares"] = shares
    plan["loss_at_stop"] = shares * risk
    plan["unused_risk"] = max_loss - shares * risk
    trims = []
    for i, p in enumerate(tps, 1):
        move = (p - open_price) * m
        # % gain on what that trim's shares cost at entry — the same for any
        # trim size, since profit and cost both scale with the share count
        cost = open_price * per_trim
        trims.append({"n": i, "price": p, "qty": per_trim, "cost": cost,
                      "profit": move * per_trim, "pct": move * per_trim / cost * 100.0,
                      "r": move / risk})
    plan["trims"] = trims
    plan["max_profit"] = sum(x["profit"] for x in trims)
    plan["reward_risk"] = plan["max_profit"] / plan["loss_at_stop"]
    wrong = [x["price"] for x in trims if x["profit"] <= 0]
    if wrong:
        plan["warnings"].append(
            "Take-profit {} {} on the wrong side of the entry for a {} — it loses money."
            .format(", ".join("${:g}".format(p) for p in wrong),
                    "is" if len(wrong) == 1 else "are", plan["direction"]))
    return plan


def create_trade(jid: int, values: dict, draft: bool = False) -> int:
    v = _coerce(values, force_type=forced_type(jid))
    cols = ", ".join(TRADE_FIELDS)
    ph = ", ".join("?" for _ in TRADE_FIELDS)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO trades(journal_id, created_at, updated_at, draft, {}) VALUES (?,?,?,?,{})".format(cols, ph),
            [jid, _now(), _now(), 1 if draft else 0] + [v[k] for k in TRADE_FIELDS],
        )
        return cur.lastrowid


def _keep_retired(tid: int, values: dict) -> dict:
    """Fill in the fields the form no longer posts from the stored row.

    Without this, saving a trade logged before those fields were retired
    would blank them — the write covers every column in TRADE_FIELDS, and
    absent means None. A caller that DOES pass one (an import, a test)
    still wins."""
    missing = [f for f in FORM_RETIRED_FIELDS if f not in values]
    if not missing:
        return values
    with _conn() as c:
        r = c.execute("SELECT {} FROM trades WHERE id=?".format(", ".join(missing)),
                      (tid,)).fetchone()
    if not r:
        return values
    out = dict(values)
    for f in missing:
        if r[f] is not None:
            out[f] = r[f]
    return out


def update_trade(tid: int, values: dict) -> None:
    """Saving a trade always finalizes it — clears the draft flag, if any."""
    v = _coerce(_keep_retired(tid, values), force_type=_forced_type_for_trade(tid))
    sets = ", ".join("{}=?".format(k) for k in TRADE_FIELDS)
    with _conn() as c:
        c.execute("UPDATE trades SET updated_at=?, draft=0, {} WHERE id=?".format(sets),
                  [_now()] + [v[k] for k in TRADE_FIELDS] + [tid])


def delete_trade(tid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM trades WHERE id=?", (tid,))


def patch_trade_fields(tid: int, values: dict) -> None:
    """Persist the editable trade fields without finalizing a draft.

    Used when an add-on / exit is recorded from the trade form before the
    trade has been explicitly saved, so the position size etc. is captured.
    """
    v = _coerce(_keep_retired(tid, values), force_type=_forced_type_for_trade(tid))
    sets = ", ".join("{}=?".format(k) for k in TRADE_FIELDS)
    with _conn() as c:
        c.execute("UPDATE trades SET updated_at=?, {} WHERE id=?".format(sets),
                  [_now()] + [v[k] for k in TRADE_FIELDS] + [tid])


def size_reduction_error(tid: int, values: dict) -> str | None:
    """Reject shrinking a position below what has already been exited."""
    exited = trimmed_qty(tid)
    if exited <= 1e-9:
        return None
    tt = values.get("trade_type") or ""
    if tt == "stock_swing":
        # the size a swing will be saved with is the one its plan works out
        raw = _coerce(values, force_type="stock_swing").get("shares")
    else:
        raw = values.get("shares") if tt in SHARE_TYPES else values.get("contracts")
    unit = "shares" if tt in SHARE_TYPES else "contracts"
    try:
        new_size = float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        new_size = None
    total_held = (new_size or 0.0) + added_qty(tid)
    if new_size is None or total_held + 1e-9 < exited:
        return ("You've already exited {:g} {}. Delete an exit before reducing "
                "the position below that.".format(exited, unit))
    return None


def entry_reasons(user_id: int) -> list[str]:
    """Every distinct entry reason used across a user's journals, most-used first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT entry_reason, COUNT(*) n FROM trades t "
            "JOIN journals j ON j.id = t.journal_id "
            "WHERE j.user_id=? AND entry_reason IS NOT NULL AND TRIM(entry_reason) <> '' "
            "GROUP BY entry_reason ORDER BY n DESC, entry_reason", (int(user_id),)).fetchall()
    return [r["entry_reason"] for r in rows]


def get_trade(tid: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["trims"] = list_trims(tid)
    d["adds"] = list_adds(tid)
    return _enrich(d)


def list_trades(jid: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trades WHERE journal_id=? AND COALESCE(draft,0)=0 "
            "ORDER BY COALESCE(entry_at, created_at), id", (jid,)).fetchall()
        trims_by_trade: dict[int, list] = {}
        for tr in c.execute(
            "SELECT t.* FROM trims t JOIN trades d ON d.id = t.trade_id "
            "WHERE d.journal_id=? ORDER BY t.id", (jid,)).fetchall():
            trims_by_trade.setdefault(tr["trade_id"], []).append(dict(tr))
        adds_by_trade: dict[int, list] = {}
        for ad in c.execute(
            "SELECT a.* FROM adds a JOIN trades d ON d.id = a.trade_id "
            "WHERE d.journal_id=? ORDER BY a.id", (jid,)).fetchall():
            adds_by_trade.setdefault(ad["trade_id"], []).append(dict(ad))
    out = []
    for r in rows:
        d = dict(r)
        d["trims"] = trims_by_trade.get(d["id"], [])
        d["adds"] = adds_by_trade.get(d["id"], [])
        out.append(_enrich(d))
    return out


# ---------------------------------------------------------------------------
# trims — partial exits. Each trim has a quantity and a close price; the sum of
# trim quantities may not exceed the position's initial contracts (or shares).
# ---------------------------------------------------------------------------
def list_trims(tid: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trims WHERE trade_id=? ORDER BY id", (tid,)).fetchall()
    return [dict(r) for r in rows]


def trimmed_qty(tid: int) -> float:
    with _conn() as c:
        v = c.execute("SELECT COALESCE(SUM(qty), 0) FROM trims WHERE trade_id=?",
                      (tid,)).fetchone()[0]
    return float(v or 0.0)


def _base_position_size(t: dict) -> float | None:
    cap = t.get("shares") if (t.get("trade_type") in SHARE_TYPES) else t.get("contracts")
    try:
        return float(cap) if cap not in (None, "") else None
    except (TypeError, ValueError):
        return None


def position_size(tid: int, row: dict | None = None) -> float | None:
    """Total contracts/shares held = initial + every add-on."""
    if row is None:
        with _conn() as c:
            r = c.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
        if not r:
            return None
        row = dict(r)
    base = _base_position_size(row)
    added = added_qty(tid)
    if base is None:
        return added or None
    return base + added


def add_trim(tid: int, qty, close_price) -> str | None:
    """Returns None on success, or an error message string."""
    with _conn() as c:
        r = c.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
    if not r:
        return "trade not found"
    try:
        qty = float(qty)
        close_price = float(close_price)
    except (TypeError, ValueError):
        return "quantity and close price must be numbers"
    if qty <= 0:
        return "quantity must be greater than zero"
    cap = position_size(tid, dict(r))
    if not cap:
        return "set the number of contracts on the trade first"
    if trimmed_qty(tid) + qty > cap + 1e-9:
        return "total trim quantity would exceed the {:g} contracts held".format(cap)
    with _conn() as c:
        c.execute("INSERT INTO trims(trade_id, created_at, qty, close_price) VALUES (?,?,?,?)",
                  (tid, _now(), qty, close_price))
        c.execute("UPDATE trades SET updated_at=? WHERE id=?", (_now(), tid))
    return None


def delete_trim(trim_id: int, tid: int | None = None) -> None:
    """Delete a trim; with `tid` given, only if it belongs to that trade."""
    with _conn() as c:
        if tid is None:
            c.execute("DELETE FROM trims WHERE id=?", (trim_id,))
        else:
            c.execute("DELETE FROM trims WHERE id=? AND trade_id=?", (trim_id, tid))


# ---------------------------------------------------------------------------
# adds — scale-ins. Each add-on has a quantity and a fill price; they raise the
# total position and the weighted-average buy cost. No cap on how many.
# ---------------------------------------------------------------------------
def list_adds(tid: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM adds WHERE trade_id=? ORDER BY id", (tid,)).fetchall()
    return [dict(r) for r in rows]


def added_qty(tid: int) -> float:
    with _conn() as c:
        v = c.execute("SELECT COALESCE(SUM(qty), 0) FROM adds WHERE trade_id=?",
                      (tid,)).fetchone()[0]
    return float(v or 0.0)


def add_addon(tid: int, qty, price) -> str | None:
    """Returns None on success, or an error message string."""
    with _conn() as c:
        r = c.execute("SELECT id FROM trades WHERE id=?", (tid,)).fetchone()
    if not r:
        return "trade not found"
    try:
        qty = float(qty)
        price = float(price)
    except (TypeError, ValueError):
        return "quantity and price must be numbers"
    if qty <= 0:
        return "quantity must be greater than zero"
    with _conn() as c:
        c.execute("INSERT INTO adds(trade_id, created_at, qty, price) VALUES (?,?,?,?)",
                  (tid, _now(), qty, price))
        c.execute("UPDATE trades SET updated_at=? WHERE id=?", (_now(), tid))
    return None


def delete_addon(add_id: int, tid: int | None = None) -> None:
    """Delete an add-on; with `tid` given, only if it belongs to that trade."""
    with _conn() as c:
        if tid is None:
            c.execute("DELETE FROM adds WHERE id=?", (add_id,))
        else:
            c.execute("DELETE FROM adds WHERE id=? AND trade_id=?", (add_id, tid))


# ---------------------------------------------------------------------------
# per-trade calculated fields
# ---------------------------------------------------------------------------
def _credit_spread_risk(t: dict, credit_per_contract, qty):
    """What a credit spread can actually lose: (width - credit) per
    contract, in dollars.

    This is the number that belongs in position sizing and in the P/L
    percentage, NOT the credit taken in. A 5-wide spread sold for $1.20
    collects $120 and risks $380 — sizing off the $120 would understate
    the position by a factor of three. Falls back to None when either
    strike is missing, and the caller then uses the credit."""
    lo, hi = t.get("short_strike"), t.get("long_strike")
    if lo is None or hi is None or credit_per_contract is None or not qty:
        return None
    width = abs(float(hi) - float(lo))
    return max(0.0, width - float(credit_per_contract)) * float(qty) * 100.0


def _spread_value(long_price, short_price, contracts):
    if long_price is None or short_price is None or not contracts:
        return None
    return (long_price - short_price) * contracts * 100.0


def _days_between(d1, d2):
    try:
        return (date.fromisoformat(d1[:10]) - date.fromisoformat(d2[:10])).days
    except (TypeError, ValueError):
        return None


def _mmdd(d):
    return (d or "")[5:10].replace("-", "/")


def _n(x):
    """A number for display, or "" when there isn't one.

    Deliberately empty rather than "?": a half-filled trade should read as
    short, not as a row of question marks demanding to be answered."""
    if x is None:
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _join(*parts):
    return " ".join(str(p) for p in parts if p not in (None, ""))


def _qty(q):
    return "{}x".format(_n(q)) if q is not None else ""


def _strikes(*values):
    """The strikes that are actually set, "/"-joined."""
    return "/".join(_n(v) for v in values if v is not None)


def _type_label(t, tt):
    """The short badge for the trades table, and its tooltip.

    A credit spread names WHICH one it is: a put credit spread is bullish
    and a call credit spread is bearish, so labelling both "CrSp" hides the
    one thing that separates them. Falls back to the generic label only
    when the option type was never recorded."""
    if tt == "credit_spread":
        ot = t.get("option_type")
        if ot == "put":
            return "Bull put", "Put credit spread (bullish / neutral)"
        if ot == "call":
            return "Bear call", "Call credit spread (bearish / neutral)"
        return "CrSp", "Credit spread"
    label, title = {
        "calendar": ("Cal", "Calendar spread"),
        "single_option": ("Opt", "Single option"),
        "stock": ("Stk", "Stock"),
        "stock_swing": ("Swing", "Stock swing"),
    }.get(tt, (tt, tt))
    return label, title


def _type_summary(t, tt, qty):
    d = (t.get("direction") or "long").capitalize()
    ot = (t.get("option_type") or "")[:1].upper()
    if tt == "stock":
        return _join(d, _n(qty), "sh" if qty is not None else "")
    if tt == "stock_swing":
        n = len(parse_targets(t.get("take_profits")))
        return _join(d, _n(qty), "sh" if qty is not None else "",
                     "SL " + _n(t.get("stop_loss")) if t.get("stop_loss") is not None else "",
                     "{} TP".format(n) if n else "")
    if tt == "single_option":
        return _join(d, _qty(qty), _strikes(t.get("strike")) + ot,
                     _mmdd(t.get("expiration")))
    if tt == "credit_spread":
        return _join(_qty(qty), _strikes(t.get("short_strike"), t.get("long_strike")) + ot,
                     "cr", _mmdd(t.get("expiration")))
    legs = _mmdd(t.get("front_expiration")), _mmdd(t.get("back_expiration"))
    return _join(_qty(qty), _n(t.get("strike")), "cal",
                 "->".join(x for x in legs if x))


def _enrich(t: dict) -> dict:
    """Every derived value. Fees/slippage ignored on purpose."""
    tt = t.get("trade_type") or "calendar"
    is_long = t.get("direction") != "short"
    m = 1.0 if is_long else -1.0

    if tt in SHARE_TYPES:
        qty = t.get("shares")
        op, cl = t.get("open_price"), t.get("close_price")
        mult, pdir = 1.0, m
        entry_unit, close_unit = op, cl
        net_debit = (op * qty) if (op is not None and qty) else None
        net_credit = (cl * qty) if (cl is not None and qty) else None
        pl = ((cl - op) * qty * m) if (op is not None and cl is not None and qty) else None
        per_unit = op
    elif tt == "credit_spread":
        qty = t.get("contracts")
        so, lo = t.get("short_leg_open"), t.get("long_leg_open")
        # A credit spread runs the other way round: the money arrives at
        # entry and every exit is a debit paid to buy it back. pdir=-1 is
        # what makes the shared add-on/exit arithmetic below come out with
        # the right sign — profit is credit taken minus debit paid.
        mult, pdir = 100.0, -1.0
        entry_unit = (so - lo) if (so is not None and lo is not None) else None
        # no close-price field: exits are recorded below, at the net debit
        close_unit = None
        # `net_debit` is the money that changed hands at entry, which for
        # this type is a credit received. The risk figure is computed after
        # add-ons, once the average credit is known.
        net_debit = (entry_unit * qty * 100.0) if (entry_unit is not None and qty) else None
        net_credit = None
        pl = None
        per_unit = (entry_unit * 100.0) if entry_unit is not None else None
    elif tt == "single_option":
        qty = t.get("contracts")
        # single options have no close-price field: every exit is recorded below
        op, cl = t.get("open_price"), None
        mult, pdir = 100.0, m
        entry_unit, close_unit = op, None
        net_debit = (op * qty * 100.0) if (op is not None and qty) else None
        net_credit = None
        pl = None
        per_unit = (op * 100.0) if op is not None else None
    else:  # calendar
        qty = t.get("contracts")
        mult, pdir = 100.0, 1.0
        fo, bo = t.get("front_leg_open"), t.get("back_leg_open")
        fc, bc = t.get("front_leg_close"), t.get("back_leg_close")
        entry_unit = (bo - fo) if (fo is not None and bo is not None) else None
        close_unit = (bc - fc) if (fc is not None and bc is not None) else None
        net_debit = _spread_value(bo, fo, qty)
        net_credit = _spread_value(bc, fc, qty)
        pl = (net_credit - net_debit) if (net_credit is not None and net_debit is not None) else None
        per_unit = (net_debit / qty) if (net_debit is not None and qty) else None

    # --- add-ons (scale-ins) -------------------------------------------------
    adds = t.get("adds") or []
    init_qty = qty
    add_qty = sum((a.get("qty") or 0.0) for a in adds)
    buys = []
    if entry_unit is not None and init_qty:
        buys.append((float(init_qty), entry_unit))
    for a in adds:
        if a.get("qty") and a.get("price") is not None:
            buys.append((a["qty"], a["price"]))
    tot_buy_qty = sum(q for q, _ in buys)
    avg_cost = (sum(q * p for q, p in buys) / tot_buy_qty) if tot_buy_qty else entry_unit

    if init_qty is not None:
        qty = init_qty + add_qty
    elif add_qty:
        qty = add_qty

    if adds:
        entry_unit = avg_cost
        net_debit = (sum(q * p for q, p in buys) * mult) if buys else None
        net_credit = (close_unit * qty * mult) if (close_unit is not None and qty) else None
        pl = ((close_unit - avg_cost) * qty * mult * pdir) \
            if (close_unit is not None and avg_cost is not None and qty) else None
        per_unit = (avg_cost * mult) if avg_cost is not None else per_unit

    t["add_qty"] = add_qty
    t["avg_buy_cost"] = avg_cost
    t["total_qty"] = qty

    # --- trims (partial exits) -------------------------------------------------
    trims = t.get("trims") or []
    trim_qty = sum((x.get("qty") or 0.0) for x in trims)
    remaining_qty = (qty - trim_qty) if qty is not None else None
    realized_pl = None
    if trims and entry_unit is not None:
        realized_pl = sum((x["close_price"] - entry_unit) * x["qty"] * mult * pdir
                          for x in trims)
    if trims:
        has_remainder = bool(remaining_qty and remaining_qty > 1e-9)
        remainder_closed = (not has_remainder) or (close_unit is not None)
        # total proceeds taken out of the position via exits
        proceeds = sum(x["close_price"] * x["qty"] * mult for x in trims)
        if has_remainder and close_unit is not None:
            proceeds += close_unit * remaining_qty * mult

        if not has_remainder:
            # every contract has been exited — P/L is the actual money out minus
            # the actual money in (buys), not an average-price estimate
            pl = ((proceeds - net_debit) * pdir) if net_debit is not None else None
        else:
            rem_pl = 0.0
            if close_unit is not None and entry_unit is not None:
                rem_pl = (close_unit - entry_unit) * remaining_qty * mult * pdir
            pl = ((realized_pl or 0.0) + rem_pl) if realized_pl is not None else None

        net_credit = proceeds if (remainder_closed and (realized_pl is not None or not has_remainder)) else None
        if not (bool(t.get("exit_at")) or remainder_closed):
            # position still partly open — don't let it count as a closed trade
            net_credit = None
    # the form no longer asks when a trade was closed, so fall back to when
    # its last exit was recorded — that IS the moment it was closed out
    t["exit_recorded_at"] = max((x.get("created_at") or "") for x in trims) if trims else None
    t["trim_qty"] = trim_qty
    t["remaining_qty"] = remaining_qty
    t["realized_pl"] = realized_pl
    t["fully_exited"] = bool(trims) and not bool(remaining_qty and remaining_qty > 1e-9)

    basis = abs(net_debit) if net_debit is not None else None
    if tt == "credit_spread":
        # sized and scored on what it can lose, not on what it took in
        risk = _credit_spread_risk(t, entry_unit, qty)
        t["credit_received"] = basis
        t["max_risk"] = risk
        if risk is not None:
            basis = risk
            per_unit = (risk / qty) if qty else per_unit
    acct = t.get("account_value")

    t["qty"] = qty
    t["net_debit_calc"] = net_debit
    t["net_credit_calc"] = net_credit
    t["position_amount_calc"] = basis
    t["position_pct"] = (basis / acct * 100.0) if (basis and acct) else None
    t["position_over_limit"] = bool(t["position_pct"] and t["position_pct"] > POSITION_PCT_LIMIT)
    t["is_closed"] = bool(t.get("exit_at") or net_credit is not None)
    t["pl_dollar"] = pl
    t["pl_pct"] = (pl / basis * 100.0) if (pl is not None and basis) else None
    t["is_win"] = (pl > 0) if pl is not None else None

    exp = t.get("front_expiration") if tt == "calendar" else t.get("expiration")
    t["days_front_exp_to_earnings"] = _days_between(exp, t.get("earnings_date"))
    t["contract_cost"] = per_unit if tt in ("calendar", "single_option", "credit_spread") else None
    t["type_label"], t["type_title"] = _type_label(t, tt)
    t["type_summary"] = _type_summary(t, tt, qty)
    if tt == "stock_swing":
        t["swing"] = swing_plan(t.get("open_price"), t.get("stop_loss"),
                                t.get("max_loss"), t.get("take_profits"))
    return t


# ---------------------------------------------------------------------------
# per-journal running aggregates
# ---------------------------------------------------------------------------
def symbol_breakdown(jid: int) -> list[dict]:
    """Per-ticker P/L for a journal. Realized P/L and W/L come from closed
    trades; trade_count is every trade on that symbol."""
    by_sym: dict[str, dict] = {}
    for t in list_trades(jid):
        s = by_sym.setdefault(t["ticker"], {
            "ticker": t["ticker"], "trades": 0, "closed": 0,
            "wins": 0, "losses": 0, "open": 0, "pl_dollar": 0.0,
        })
        s["trades"] += 1
        if t["is_closed"] and t["pl_dollar"] is not None:
            s["closed"] += 1
            s["pl_dollar"] += t["pl_dollar"]
            if t["is_win"]:
                s["wins"] += 1
            else:
                s["losses"] += 1
        else:
            s["open"] += 1
    out = []
    for s in by_sym.values():
        decided = s["wins"] + s["losses"]
        s["win_rate"] = (s["wins"] / decided * 100.0) if decided else None
        out.append(s)
    out.sort(key=lambda s: s["pl_dollar"], reverse=True)
    return out


def aggregates(jid: int) -> dict:
    j = get_journal(jid) or {}
    is_paper = j.get("kind") == "paper"
    trades = list_trades(jid)
    closed = [t for t in trades if t["is_closed"] and t["pl_dollar"] is not None]

    wins = sum(1 for t in closed if t["is_win"])
    cum_pl = sum(t["pl_dollar"] for t in closed)

    ordered = sorted(
        closed,
        key=lambda t: (t.get("exit_at") or t.get("entry_at") or t.get("created_at") or "", t["id"]),
    )
    baseline = next((t["account_value"] for t in trades if t.get("account_value")), None)
    max_dd_pct = 0.0
    max_dd_dollar = 0.0
    if ordered:
        equity = baseline if baseline else 0.0
        peak = equity if equity else None
        running = 0.0
        for t in ordered:
            running += t["pl_dollar"]
            equity = (baseline + running) if baseline else running
            if peak is None or equity > peak:
                peak = equity
            dd_dollar = (peak - equity)
            max_dd_dollar = max(max_dd_dollar, dd_dollar)
            if peak and peak > 0:
                max_dd_pct = max(max_dd_pct, dd_dollar / peak * 100.0)

    costs = [t["contract_cost"] for t in trades if t["contract_cost"] is not None]

    return {
        "trade_count": len(trades),
        "closed_count": len(closed),
        "open_count": len(trades) - len(closed),
        "checkpoint_applies": is_paper,
        "checkpoint_target": CHECKPOINT_TRADES,
        "checkpoint_remaining": max(0, CHECKPOINT_TRADES - len(trades)),
        "checkpoint_pct": min(100.0, len(trades) / CHECKPOINT_TRADES * 100.0),
        "win_rate": (wins / len(closed) * 100.0) if closed else None,
        "wins": wins,
        "losses": len(closed) - wins,
        "cum_pl": cum_pl,
        "cum_pl_pct": (cum_pl / baseline * 100.0) if baseline else None,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollar": max_dd_dollar,
        "drawdown_limit": MAX_DRAWDOWN_LIMIT,
        "drawdown_breached": max_dd_pct > MAX_DRAWDOWN_LIMIT,
        "avg_contract_cost": (sum(costs) / len(costs)) if costs else None,
        "baseline_account": baseline,
    }
