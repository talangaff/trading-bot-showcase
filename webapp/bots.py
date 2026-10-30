"""0DTE/swing options bot fleets — SQLite-backed per-bot bookkeeping.

Every user runs their OWN fleet of autonomous index-ETF options bots, all
of that user's bots together on that user's own Alpaca paper account (the
fleet keys they entered — webapp/bot_settings.py). Each bot is a
bookkeeping construct: it
gets a virtual starting balance (default $10k) and its own independent
ledger of fills — the ACTUAL fills Alpaca reported for its own orders —
from which its balance, equity curve, drawdown and stats are derived. The
real account's P&L is the sum of the bots'; each bot's P&L is its own.

Because many bots share the account, two of them can hold the very same
contract at once. Each ledger tracks only its own contracts (FIFO-matched
below), a sell can never exceed what that bot itself has open, and
webapp/account_book.py reconciles the sum of every bot's claims against
what the account actually holds so nobody ever sells another bot's
contracts.

Hard rules enforced here (not by convention):
  * Buy-to-Open and Sell-to-Close only. Sell-to-Open is rejected outright.
    A vertical debit spread (genome structure='vertical') is booked as ONE
    defined-risk package — option_model.spread_symbol — whose price is the
    net debit and whose max loss is that debit, so the rule keeps its
    meaning (no undefined risk is ever opened) while the package itself
    carries far less theta and spread cost than a naked long option.
  * Minimum entry 2 contracts.
  * A single position may not exceed 10% of the bot's current balance
    (scale-ins count toward the same cap).
  * Balance / drawdown are computed from REALIZED (closed) P&L only — open
    positions are not marked to market.
  * When realized drawdown from the running peak hits 25%, the bot is
    force-closed (all open positions flattened) and flagged "failing";
    after that it can take no new trades.

The autonomous simulation engine (webapp/bot_engine.py) books fills through
record_trade() on a systemd timer; the dashboard can also record fills by
hand for debugging. Evolution (spawn / mutate / archive / retire) also lives
here — fleet-only; see pool='custom' below.

What is shared between users is the SEARCH, never the money: the genomes
every fleet runs come from one population (webapp/gene_pool.py), a
genome's track record is the union of every user's trades on it, and no
two fleets are handed the same candidate at the same time. So N users
explore N times as many genomes per market day and prove (or kill) each
one N times faster, while each user's bots, ledgers, capital and orders
stay entirely their own.

Custom bots (pool='custom', see clone_bot()) are a second kind of bot:
cloned from any bot, hand-tunable, and run on a DIFFERENT Alpaca paper
account — their owner's own keys — so one genome can be tested by itself
without the rest of the fleet trading around it. Each gets its own dollar
allocation as its starting balance, checked against a capital pool the user
sets (webapp/bot_settings.py) so allocations can never overlap. They are
excluded from every fleet evolution/culling function and never
auto-retired, however they perform.

EVERY bot belongs to one account (bots.user_id — webapp/auth.py). A custom
bot's real orders go to its owner's custom-bot keys
(bot_settings.execution_keys); a fleet or baseline bot's go to the same
owner's FLEET keys (bot_settings.fleet_keys) — a different account, so a
user's fleet and their single-genome test bots never share a book. A bot
with no owner cannot trade at all.
"""

import json
import sqlite3
from datetime import datetime, time as _time, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config, genome as _genome, guardrails as _guardrails, market_calendar, option_model

_ET = ZoneInfo("America/New_York")
# Hard close-out cutoff: no Buy-to-Open (entry OR average-in) at/after this ET
# time on a contract expiring that day. Derived from the session's real close,
# so an early-close half day moves the cutoff with it rather than letting a bot
# buy something that expired three hours earlier.
_NO_BUY_MARGIN_MIN = 5


def no_buy_after_et(d):
    """Cutoff time for same-day-expiry buys on date `d`."""
    from . import market_calendar
    m = market_calendar.session_close_min(d) - _NO_BUY_MARGIN_MIN
    return _time(m // 60, m % 60)


NO_BUY_AFTER_ET = _time(15, 55)          # the regular-session value, for display

DB_PATH = config.DATA_DIR / "bots.db"

START_BALANCE = float(getattr(config, "BOT_START_BALANCE", 10_000.0))
MAX_DRAWDOWN_PCT = float(getattr(config, "BOT_MAX_DRAWDOWN_PCT", 25.0))
POSITION_PCT_LIMIT = float(getattr(config, "BOT_POSITION_PCT_LIMIT", 10.0))
MIN_CONTRACTS = int(getattr(config, "BOT_MIN_CONTRACTS", 2))
MAX_CONTRACTS_PER_TRADE = int(getattr(config, "BOT_MAX_CONTRACTS", 250))
ARCHIVE_PEAK_PCT = float(getattr(config, "BOT_ARCHIVE_PEAK_PCT", 50.0))

# A bot must clear this much of a track record before it counts as a top
# performer / breeding parent — stops a lucky one-session run from dominating
# evolution before it's proven across regimes.
MIN_ELIGIBLE_TRADES = int(getattr(config, "BOT_MIN_ELIGIBLE_TRADES", 60))
MIN_ELIGIBLE_DAYS = int(getattr(config, "BOT_MIN_ELIGIBLE_DAYS", 15))
MIN_ELIGIBLE_REGIMES = int(getattr(config, "BOT_MIN_ELIGIBLE_REGIMES", 2))
PROVEN_ALPHA = float(getattr(config, "BOT_PROVEN_ALPHA", 0.05))
# "Promising": passes the same sample gates and is making money, with a fitness
# t-stat well clear of noise (but short of the multiple-comparison bar). Such
# genomes may breed; they are not called proven.
PROMISING_T = float(getattr(config, "BOT_PROMISING_T", 2.0))
STAGNANT_MIN_AGE_DAYS = float(getattr(config, "BOT_STAGNANT_MIN_AGE_DAYS", 14.0))
STAGNANT_MIN_TRADES = int(getattr(config, "BOT_STAGNANT_MIN_TRADES", 40))
STAGNANT_FITNESS = float(getattr(config, "BOT_STAGNANT_FITNESS", 0.5))
STAGNANT_PL_PCT = float(getattr(config, "BOT_STAGNANT_PL_PCT", 2.0))
LOSER_MIN_TRADES = int(getattr(config, "BOT_LOSER_MIN_TRADES", 40))
LOSER_FITNESS = float(getattr(config, "BOT_LOSER_FITNESS", -1.5))
EXPLORE_SHARE = float(getattr(config, "BOT_EXPLORE_SHARE", 0.20))
EXPLORE_TRIM_PER_DAY = int(getattr(config, "BOT_EXPLORE_TRIM_PER_DAY", 15))

SIDES = ("buy_to_open", "sell_to_close")

# The bot's own logic picks the order type per trade. `trailing_stop` is
# accepted for bookkeeping even though Alpaca does not natively support it for
# options yet — the execution engine will emulate it when it is built.
ORDER_TYPES = ("market", "limit", "stop", "stop_limit", "trailing_stop")

CONTRACT_MULT = 100.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    strategy      TEXT,
    symbol        TEXT,
    risk_pct      REAL,
    start_balance REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL,
    flagged_at    TEXT
);
CREATE TABLE IF NOT EXISTS bot_trades (
    id         INTEGER PRIMARY KEY,
    bot_id     INTEGER NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    ts         TEXT NOT NULL,
    symbol     TEXT,
    contract   TEXT NOT NULL,
    side       TEXT NOT NULL,
    qty         INTEGER NOT NULL,
    price       REAL NOT NULL,
    fees        REAL NOT NULL DEFAULT 0,
    order_type  TEXT NOT NULL DEFAULT 'market',
    limit_price REAL,
    stop_price  REAL,
    trail_pct   REAL,
    forced      INTEGER NOT NULL DEFAULT 0,
    note        TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_bot_trades_bot ON bot_trades(bot_id, ts, id);
CREATE INDEX IF NOT EXISTS ix_bots_status ON bots(status);
CREATE TABLE IF NOT EXISTS bot_pos_state (
    bot_id      INTEGER NOT NULL,
    contract    TEXT NOT NULL,
    hwm         REAL,
    opened_ts   TEXT,
    entry_iv    REAL,
    trimmed     INTEGER DEFAULT 0,
    core_closed INTEGER DEFAULT 0,
    adds        INTEGER DEFAULT 0,
    PRIMARY KEY (bot_id, contract)
);
CREATE TABLE IF NOT EXISTS archived_bots (
    id            INTEGER PRIMARY KEY,
    source_bot_id INTEGER,
    user_id       INTEGER,
    name          TEXT,
    genome        TEXT,
    reason        TEXT,
    archived_at   TEXT NOT NULL,
    peak_pl_pct   REAL,
    final_pl_pct  REAL,
    final_balance REAL,
    generation    INTEGER,
    stats_json    TEXT,
    equity_json   TEXT,
    trades_json   TEXT
);
CREATE TABLE IF NOT EXISTS engine_events (
    id      INTEGER PRIMARY KEY,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS ix_engine_events_ts ON engine_events(ts);
CREATE TABLE IF NOT EXISTS managers (
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    name          TEXT NOT NULL,
    account_id    INTEGER NOT NULL,
    slots         INTEGER NOT NULL,
    total_budget  REAL NOT NULL,
    position_pct  REAL NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'dry_run',
    enabled       INTEGER NOT NULL DEFAULT 1,
    rules         TEXT,
    created_at    TEXT NOT NULL,
    last_run_at   TEXT
);
CREATE TABLE IF NOT EXISTS manager_events (
    id          INTEGER PRIMARY KEY,
    manager_id  INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    slot        INTEGER,
    old_bot_id  INTEGER,
    new_bot_id  INTEGER,
    reason      TEXT,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS ix_manager_events ON manager_events(manager_id, ts);
CREATE TABLE IF NOT EXISTS manager_proposals (
    id          INTEGER PRIMARY KEY,
    manager_id  INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    action      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS fleet_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS pending_orders (
    id              INTEGER PRIMARY KEY,
    bot_id          INTEGER NOT NULL,
    contract        TEXT NOT NULL,
    side            TEXT NOT NULL,
    qty             INTEGER NOT NULL,
    client_order_id TEXT NOT NULL UNIQUE,
    submitted_at    TEXT NOT NULL,
    symbol          TEXT,
    note            TEXT,
    regime          TEXT,
    order_type      TEXT,
    limit_price     REAL,
    stop_price      REAL,
    trail_pct       REAL
);
"""

# columns added after a table first shipped — applied on init()
_ADD_COLUMNS = [
    ("order_type", "TEXT DEFAULT 'market'"),
    ("limit_price", "REAL"),
    ("stop_price", "REAL"),
    ("trail_pct", "REAL"),
    # 'alpaca'/'yfinance' (a real quote priced this fill), 'model' (no real
    # quote available, BS spread used instead), 'broker' (a custom bot's
    # REAL Alpaca order fill — see bot_engine._record_fill), or NULL (older
    # trades recorded before this column existed).
    ("fill_source", "TEXT"),
    # volatility regime the fill happened in (bots.regime_label) — eligibility
    # requires a bot's record to span MIN_ELIGIBLE_REGIMES distinct ones
    ("regime", "TEXT"),
]
_ADD_BOT_COLUMNS = [
    ("genome", "TEXT"),
    ("generation", "INTEGER DEFAULT 0"),
    ("parent_id", "INTEGER"),
    ("origin", "TEXT DEFAULT 'manual'"),
    ("archived", "INTEGER DEFAULT 0"),
    # custom bots: cloned from a fleet (or another custom) bot, tunable by
    # hand, run live on their own capital, and never touched by evolution.
    ("pool", "TEXT DEFAULT 'fleet'"),
    ("enabled", "INTEGER DEFAULT 1"),
    ("guardrails", "TEXT"),
    # owning account — every bot has one now (fleet and baseline included:
    # they trade on that user's own fleet keys). NULL only on rows written
    # by the pre-multi-user build, which init() hands to the primary admin.
    ("user_id", "INTEGER"),
    # fingerprint of the bot's genome in the shared pool (gene_pool.py) —
    # what ties one user's bot to every other user's evidence on the same
    # genome. Backfilled for pre-existing rows by init().
    ("genome_fp", "TEXT"),
    # custom bots: the saved broker account (accounts.py) the bot trades on.
    # NULL = not assigned, and an unassigned bot does not trade.
    ("account_id", "INTEGER"),
    # custom bots run by a manager (manager.py): which manager owns the slot
    # and which slot (1..N) it fills. NULL on every hand-made bot.
    ("manager_id", "INTEGER"),
    ("slot", "INTEGER"),
    # a managed bot being phased out while it still holds a position: 1 = it
    # breached its limits, 2 = a better bot was found. It takes no new trades
    # (enabled=0) but keeps managing its exits; the manager retires it and
    # fills the slot the moment it is flat. 0/NULL otherwise.
    ("drain", "INTEGER DEFAULT 0"),
]
_ADD_POS_COLUMNS = [
    ("entry_iv", "REAL"),
    ("trimmed", "INTEGER DEFAULT 0"),
    ("core_closed", "INTEGER DEFAULT 0"),
    ("adds", "INTEGER DEFAULT 0"),
    # the most recent mid the engine observed for this contract — what an
    # open position is marked to (stats()['unrealized_pl'] / fitness_marked)
    ("last_mark", "REAL"),
    # the MID at entry (the ledger's avg_price is the ask actually paid).
    # Stops / targets / trails are measured mid-to-mid against this, so a
    # stop can't be tripped by the spread alone the moment a fill is booked.
    ("entry_mid", "REAL"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA busy_timeout = 15000")
    return c


def _f(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def init(default_owner: int = 1) -> None:
    """Create / migrate the schema.

    Bots that pre-date multi-user fleets have no owner: the shared fleet,
    its baselines and any custom bot older than accounts are all handed to
    `default_owner` — the primary admin — so the fleet that was already
    running becomes that admin's own and keeps trading on the same keys
    (bot_settings.migrate_fleet_to does the matching half).

    Fleet bots that pre-date the shared gene pool have no fingerprint
    either; they are registered into the pool here, so the history they
    have already accumulated counts as pooled evidence from the first
    cycle instead of being thrown away."""
    from . import api_meter, gene_pool
    with _conn() as c:
        c.executescript(_SCHEMA)
        c.executescript(gene_pool.SCHEMA)
        c.executescript(api_meter.SCHEMA)
        try:
            c.execute("ALTER TABLE archived_bots ADD COLUMN user_id INTEGER")
        except sqlite3.OperationalError:
            pass
        for col, decl in _ADD_COLUMNS:
            try:
                c.execute("ALTER TABLE bot_trades ADD COLUMN {} {}".format(col, decl))
            except sqlite3.OperationalError:
                pass
        for col, decl in _ADD_BOT_COLUMNS:
            try:
                c.execute("ALTER TABLE bots ADD COLUMN {} {}".format(col, decl))
            except sqlite3.OperationalError:
                pass
        for col, decl in _ADD_POS_COLUMNS:
            try:
                c.execute("ALTER TABLE bot_pos_state ADD COLUMN {} {}".format(col, decl))
            except sqlite3.OperationalError:
                pass
        c.execute("CREATE INDEX IF NOT EXISTS ix_bots_owner "
                  "ON bots(user_id, pool, status)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_bots_fp ON bots(genome_fp)")
        c.execute("UPDATE bots SET user_id=? WHERE user_id IS NULL", (int(default_owner),))
        c.execute("UPDATE archived_bots SET user_id=? WHERE user_id IS NULL",
                  (int(default_owner),))
        orphans = [dict(r) for r in c.execute(
            "SELECT id, genome, generation, origin FROM bots "
            "WHERE genome_fp IS NULL AND COALESCE(pool,'fleet')='fleet'").fetchall()]
    for b in orphans:
        g = _genome.loads(b["genome"])
        fp = gene_pool.register(g, generation=int(b["generation"] or 0),
                                origin=b["origin"] or "legacy", user_id=default_owner)
        with _conn() as c:
            c.execute("UPDATE bots SET genome_fp=? WHERE id=?", (fp, b["id"]))


# ---------------------------------------------------------------------------
# bots CRUD
# ---------------------------------------------------------------------------
def list_bots() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM bots ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_bot(bot_id: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM bots WHERE id=?", (bot_id,)).fetchone()
    return dict(r) if r else None


def create_bot(name: str, strategy: str = "", symbol: str = "",
               risk_pct=None, start_balance=None, genome=None,
               generation: int = 0, parent_id=None, origin: str = "manual",
               pool: str = "fleet", enabled: bool = True,
               guardrail_overrides: dict | None = None,
               user_id: int | None = None, genome_fp: str | None = None,
               max_active_fleet: int | None = None) -> int | None:
    """Create a bot owned by `user_id`. Every bot has an owner now — it is
    the account whose Alpaca keys its real orders go out on (fleet keys for
    fleet/baseline, custom-bot keys for custom), so a bot without one can
    never trade.

    `max_active_fleet` (passed only by spawn_bot, which owns the fleet-size
    guardrail) refuses the insert and returns None if the owner already has
    that many active fleet bots. The count and the insert run in ONE
    immediate transaction on purpose: the timer's cycle and a user pressing
    Spawn can land at the same instant, and two writers must not both see
    the same last free slot and both take it."""
    g = genome if isinstance(genome, dict) else None
    pool = pool if pool in ("custom", "baseline", "sim") else "fleet"
    owner = int(user_id) if user_id is not None else None
    with _conn() as c:
        if max_active_fleet is not None:
            c.execute("BEGIN IMMEDIATE")
            sql, params = _scope("SELECT COUNT(*) AS n FROM bots WHERE status='active'",
                                 [], "fleet", owner)
            if c.execute(sql, params).fetchone()["n"] >= int(max_active_fleet):
                return None
        cur = c.execute(
            "INSERT INTO bots(name, strategy, symbol, risk_pct, start_balance, "
            "status, created_at, genome, generation, parent_id, origin, "
            "pool, enabled, guardrails, user_id, genome_fp) "
            "VALUES (?,?,?,?,?,'active',?,?,?,?,?,?,?,?,?,?)",
            ((name or "").strip() or "Unnamed bot",
             (strategy or (g or {}).get("strategy") or "").strip() or None,
             (symbol or (g or {}).get("symbol") or "").upper().strip() or None,
             _f(risk_pct), _f(start_balance) or START_BALANCE, _now(),
             _genome.dumps(g) if g else None,
             int(generation or 0), parent_id, origin, pool,
             1 if enabled else 0,
             json.dumps(guardrail_overrides) if guardrail_overrides else None,
             owner, genome_fp))
        return cur.lastrowid


def clone_bot(source_id: int, name: str | None = None,
              gene_overrides: dict | None = None,
              guardrail_overrides: dict | None = None,
              start_balance=None, user_id: int | None = None) -> int | None:
    """Clone any bot (fleet or custom) into a new custom bot owned by `user_id`.

    The clone gets its own genome (a copy of the source's, with any
    `gene_overrides` merged in), its own guardrail overrides, and its own
    capital — a fresh ledger fully separate from the source bot's. It runs
    live like any other bot but sits in the 'custom' pool: excluded from
    breeding/culling/evolution, and never auto-retired.
    """
    src = get_bot(source_id)
    if not src:
        return None
    # a full copy of the source's rules — style (0dte / swing) and structure
    # (single / vertical) included, since the clone runs the exact same
    # live execution path as the bot it was cloned from
    g = dict(genome_of(src))
    if gene_overrides:
        g.update(gene_overrides)
        _genome._repair(g)
    return create_bot(
        name or "{} (custom)".format(src["name"]),
        strategy=g.get("strategy", ""), symbol=g.get("symbol", ""),
        risk_pct=src.get("risk_pct"), start_balance=start_balance, genome=g,
        parent_id=source_id, origin="clone", pool="custom", enabled=True,
        guardrail_overrides=guardrail_overrides, user_id=user_id,
        genome_fp=src.get("genome_fp"))


def clone_genome(fingerprint: str, name: str | None = None,
                 start_balance=None, user_id: int | None = None) -> int | None:
    """Clone a genome straight out of the SHARED pool into a custom bot.

    The shared population is what every fleet is collectively learning, so
    a user must be able to take the best thing anybody has found and test
    it by itself on their own account — even when the bot that proved it
    belongs to someone else and is (rightly) invisible to them. The genome
    and its pooled record are public; the other user's ledger is not.
    """
    from . import gene_pool
    e = gene_pool.entry(fingerprint)
    if not e:
        return None
    g = _genome.loads(e["genome"])
    return create_bot(
        name or "{} g{} (custom)".format(_genome.family_of(g), e.get("generation") or 0),
        strategy=g.get("strategy", ""), symbol=g.get("symbol", ""),
        start_balance=start_balance, genome=g, generation=int(e.get("generation") or 0),
        origin="clone", pool="custom", enabled=True, user_id=user_id,
        genome_fp=fingerprint)


def owner_of(bot: dict | None) -> int | None:
    """The account a bot belongs to — for any pool. Its fills, its ledger
    and its real orders are that user's; None only for a row written
    before ownership existed and not yet migrated by init()."""
    if not bot:
        return None
    uid = bot.get("user_id")
    return int(uid) if uid is not None else None


# --- fleet / evolution ------------------------------------------------------
_ADJ = (
    "swift", "iron", "lunar", "vivid", "quiet", "amber", "delta", "nova",
    "onyx", "pulse", "rogue", "zen", "flux", "atlas", "cobalt", "ember",
    "arctic", "bold", "brisk", "crimson", "dusk", "echo", "frost", "gilded",
    "hollow", "indigo", "jade", "keen", "lucid", "mellow", "nimble", "opal",
    "polar", "quartz", "radiant", "silent", "tidal", "umber", "velvet", "wild",
    "azure", "binary", "copper", "dawn", "electric", "feral", "granite", "hazel",
    "ivory", "jolly", "kinetic", "lively", "mighty", "noble", "obsidian", "plucky",
    "quick", "rapid", "solar", "tawny", "ultra", "vector", "wired", "zephyr")
_NOUN = (
    "falcon", "otter", "comet", "yak", "lynx", "moth", "raven", "koi",
    "bison", "wren", "shark", "ibex", "crane", "gecko", "orca", "puma",
    "badger", "condor", "dingo", "egret", "fox", "gazelle", "heron", "iguana",
    "jackal", "kestrel", "lemur", "marten", "narwhal", "osprey", "panther", "quail",
    "robin", "stork", "tiger", "urchin", "viper", "walrus", "xerus", "zebra",
    "adder", "beetle", "cobra", "drake", "eagle", "finch", "grouse", "hawk",
    "impala", "jaguar", "kiwi", "llama", "mantis", "newt", "ocelot", "parrot",
    "quokka", "rhino", "stoat", "toucan", "unicorn", "vole", "wolf", "yeti")


def _name_taken(name: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM bots WHERE name=? LIMIT 1",
                         (name,)).fetchone() is not None


def _fleet_name(gen: int) -> str:
    """A fresh 'adjective-noun gN' name no bot (live or retired) already has.
    4,000+ pairs times the generation suffix; if random draws keep colliding
    the name gets a numeric tag so it is unique regardless."""
    import random
    for _ in range(40):
        name = "{}-{} g{}".format(random.choice(_ADJ), random.choice(_NOUN), gen)
        if not _name_taken(name):
            return name
    base = "{}-{} g{}".format(random.choice(_ADJ), random.choice(_NOUN), gen)
    n = 2
    while _name_taken("{} {}".format(base, n)):
        n += 1
    return "{} {}".format(base, n)


def dedupe_generated_names() -> int:
    """Rename duplicate auto-generated names (fleet/sim/baseline, not bots a
    person named or cloned) so every name is unique. The oldest bot keeps its
    name. Returns how many were renamed."""
    n = 0
    with _conn() as c:
        rows = c.execute(
            "SELECT id, name, origin, pool, generation FROM bots ORDER BY id").fetchall()
    seen: set[str] = set()
    todo = []
    for r in rows:
        if r["name"] in seen:
            if r["pool"] in ("fleet", "sim", "baseline") and r["origin"] not in ("manual", "clone"):
                todo.append(r)
        else:
            seen.add(r["name"])
    for r in todo:
        new = _fleet_name(int(r["generation"] or 0))
        with _conn() as c:
            c.execute("UPDATE bots SET name=? WHERE id=?", (new, r["id"]))
        n += 1
    return n


def fleet_cap(user_id: int | None) -> int:
    """The size of one account's fleet: both the hard ceiling AND the
    number every maintenance pass fills back up to.

    A guardrail like the risk limits: the house default, that account's own
    override, clamped to config's absolute range. Enforced in spawn_bot()
    below — the single door every fleet bot comes through, so no caller
    (evolution, daily breeding, a fleet reset, the Spawn button) can get
    past it — and aimed at by evolve(), _daily_for_user() and reset_fleet(),
    so a fleet that has just retired bots is refilled to it rather than
    left short. There is no separate lower target: an empty slot is a
    genome not being tested."""
    return int(_guardrails.load(user_id)["max_fleet_bots"])


def fleet_headroom(user_id: int | None) -> int:
    """How many more bots this fleet may spawn right now (never negative)."""
    return max(0, fleet_cap(user_id) - len(active_bots(pool="fleet", user_id=user_id)))


def spawn_bot(genome: dict, parent_id=None, origin: str = "mutation",
              user_id: int | None = None, generation: int | None = None,
              parent_fp: str | None = None) -> int | None:
    """Give one genome a live trial in `user_id`'s fleet.

    Returns the new bot's id, or None if the fleet is already at its
    `max_fleet_bots` guardrail — this is the one place the cap is enforced,
    so callers must treat None as "not spawned" rather than assume a row.

    The genome is registered in the shared pool first, so its generation
    and lineage belong to the population rather than to whichever fleet
    happened to run it first, and every user's trades on it accumulate
    against the same fingerprint."""
    from . import gene_pool
    cap = fleet_cap(user_id)
    if fleet_headroom(user_id) <= 0:
        # refuse before touching the shared pool: a genome registered for a
        # bot that was never created would be marked claimed by a fleet that
        # is not running it, and hidden from every other fleet's search
        log_event("spawn_blocked", {"user_id": user_id, "cap": cap, "origin": origin})
        return None
    gen = generation
    if gen is None:
        # a child's depth comes from its PARENT's place in the population —
        # by fingerprint, since breeding names a genome rather than a bot
        # (bot_engine._replacement_genome), and the parent may be running in
        # another user's fleet or in none at all.
        if parent_id and not parent_fp:
            parent_fp = (get_bot(parent_id) or {}).get("genome_fp")
        parent = gene_pool.entry(parent_fp) if parent_fp else None
        gen = int(parent["generation"]) + 1 if parent else 0
    fp = gene_pool.register(genome, generation=int(gen), origin=origin,
                            parent_fp=parent_fp, user_id=user_id)
    known = gene_pool.entry(fp) or {}
    gen = int(known.get("generation") or gen)      # the pool's number, not ours
    # re-checked inside the insert's own transaction — the pre-check above
    # keeps a blocked spawn from claiming a pool genome, this makes the cap
    # hold even if another writer took the last slot in between
    bid = create_bot(_fleet_name(gen), genome=genome, generation=gen,
                     parent_id=parent_id, origin=origin,
                     user_id=user_id, genome_fp=fp, max_active_fleet=cap)
    if bid is None:
        log_event("spawn_blocked", {"user_id": user_id, "cap": cap, "origin": origin})
    return bid


def _meta_get(key: str, default: str | None = None) -> str | None:
    with _conn() as c:
        r = c.execute("SELECT value FROM fleet_meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def _meta_set(key: str, value) -> None:
    with _conn() as c:
        c.execute("INSERT INTO fleet_meta(key, value) VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def _meta_incr(key: str) -> int:
    n = int(_meta_get(key, "0") or 0) + 1
    _meta_set(key, n)
    return n


def bots_tested() -> int:
    """The N in the multiple-comparison correction (proven_threshold): how
    many DISTINCT genomes the whole population has given a live trial since
    the last reset.

    Distinct genomes, not bot rows — ten fleets running the same sixty
    genomes have searched sixty candidates, not six hundred, so counting
    rows would inflate the bar every time a user joined and quietly make
    every genome un-provable. Falls back to the old per-row counter only
    if the gene pool is unreachable."""
    from . import gene_pool
    try:
        return gene_pool.distinct_tested()
    except sqlite3.Error:
        return max(1, int(_meta_get("bots_tested", "0") or 0))


def proven_threshold(n_tested: int | None = None, alpha: float | None = None) -> float:
    """The fitness t-stat a bot must clear to count as PROVEN: the z such
    that the best of N null (no-edge) bots would exceed it with probability
    <= alpha. Bonferroni: z = Phi^-1(1 - alpha / N). Searching more genomes
    raises the bar, as it must — the best of 150 coin-flippers always looks
    good."""
    from statistics import NormalDist
    n = max(1, int(n_tested if n_tested is not None else bots_tested()))
    a = alpha if alpha is not None else PROVEN_ALPHA
    return round(NormalDist().inv_cdf(1.0 - min(0.5, a / n)), 3)


def reset_fleet(reason: str, user_id: int, reset_pool: bool = False) -> int:
    """Retire every active fleet bot of ONE user and reseed a FULL fleet —
    fleet_cap() fresh candidates — for them. Their custom and baseline bots are
    untouched, and so is every other user's fleet.

    Use this after a change to the fill model, guardrails, or breeding
    rules: every currently-active fleet bot was selected (or its lineage
    was bred) under whatever rules were in effect before the change, so its
    numbers are not a fair read of the new rules. A one-time reset gives
    the new rules a clean cohort to actually be judged by, instead of
    silently grading them against genomes shaped by the old ones.

    The shared gene pool is deliberately NOT cleared: a genome's record is
    the population's, and one user resetting their own fleet must not
    erase what everyone else learned about it. `reset_pool=True` (admin,
    installation-wide) additionally restarts the multiple-comparison count
    — do that only when the rules changed for everybody.
    """
    from . import gene_pool
    user_id = int(user_id)          # never "every fleet": that is not one button
    retired = 0
    for b in active_bots(pool="fleet", user_id=user_id):
        flatten_bot(b["id"], "flattened at mark on fleet reset")
        retire_bot(b["id"], reason)
        retired += 1
    if reset_pool:
        gene_pool.reset_pool(reason)
    claimed = gene_pool.claimed_fingerprints()
    # refill every slot the retirements above just freed: a reset ends with
    # a FULL fleet, never a partial one
    spawned = 0
    while fleet_headroom(user_id) > 0:
        g, _fp, _n = gene_pool.distinct_child(_genome.random_genome, claimed=claimed)
        if spawn_bot(g, origin="seed", user_id=user_id) is None:
            break               # at the cap (or it moved under us) — stop
        spawned += 1
    log_event("reset", {"reason": reason, "user_id": user_id,
                        "retired": retired, "spawned": spawned})
    return retired


def genome_of(bot: dict) -> dict:
    return _genome.loads(bot.get("genome"))


def _pool_of(bot: dict) -> str:
    return bot.get("pool") or "fleet"


# A custom bot's own overrides are NOT held to the fleet's absolute range: the
# two percentages may be anything from 0 to 100, the contract counts anything
# from 1 up. (The owner's fleet-wide guardrails, which a custom bot inherits
# where it has no override, keep their absolute clamp — see guardrails.py.)
CUSTOM_BOUNDS = {"max_drawdown_pct": (0.0, 100.0), "position_pct_limit": (0.0, 100.0),
                 "min_contracts": (1, None), "max_contracts": (1, None)}
_CUSTOM_INT = ("min_contracts", "max_contracts")


def _custom_value(key: str, v):
    """A custom-bot override bounded only by CUSTOM_BOUNDS, or None if blank/invalid."""
    v = _f(v)
    if v is None:
        return None
    lo, hi = CUSTOM_BOUNDS[key]
    v = max(lo, v if hi is None else min(hi, v))
    return int(round(v)) if key in _CUSTOM_INT else round(v, 2)


def effective_guardrails(bot: dict) -> dict:
    """The guardrails a given bot actually trades under.

    The baseline is its OWNER's guardrails: their own account, their own
    risk appetite, layered over the installation-wide defaults an admin
    sets (webapp/guardrails.py). A custom bot may override any of the four
    limits on top of that — but every override, and every per-user setting,
    is clamped to the absolute floor/ceiling in config.py, so no edit
    anywhere can trade with looser risk control than the hard safety limit
    allows.
    """
    base = _guardrails.load(owner_of(bot))
    if _pool_of(bot) != "custom":
        return base
    try:
        override = json.loads(bot.get("guardrails") or "{}")
    except (ValueError, TypeError):
        override = {}
    out = dict(base)
    for k in CUSTOM_BOUNDS:
        v = _custom_value(k, override.get(k))
        if v is not None:
            out[k] = v
    if out["min_contracts"] > out["max_contracts"]:
        out["min_contracts"] = out["max_contracts"]
    # custom-bot opt-in, not a number: buy the minimum size even when the
    # per-trade budget (the genome's risk_pct) would not fund it
    out["min_size_over_risk"] = bool(override.get("min_size_over_risk"))
    return out


def guardrail_overrides_of(bot: dict) -> dict:
    try:
        return json.loads(bot.get("guardrails") or "{}")
    except (ValueError, TypeError):
        return {}


def update_guardrails(bot_id: int, overrides: dict) -> None:
    """Set (or clear, with an empty dict) a custom bot's guardrail overrides."""
    clean = {k: _custom_value(k, v) for k, v in (overrides or {}).items()
             if k in CUSTOM_BOUNDS and _custom_value(k, v) is not None}
    if (overrides or {}).get("min_size_over_risk"):
        clean["min_size_over_risk"] = True
    with _conn() as c:
        c.execute("UPDATE bots SET guardrails=? WHERE id=?",
                  (json.dumps(clean) if clean else None, bot_id))


def update_genome(bot_id: int, gene_updates: dict) -> None:
    """Merge `gene_updates` into a bot's genome (custom bots: hand-edited
    rules). Values are repaired/clamped to each gene's valid range."""
    bot = get_bot(bot_id)
    if not bot:
        return
    g = genome_of(bot)
    g.update(gene_updates or {})
    _genome._repair(g)
    with _conn() as c:
        c.execute(
            "UPDATE bots SET genome=?, strategy=?, symbol=? WHERE id=?",
            (_genome.dumps(g), g.get("strategy"), g.get("symbol"), bot_id))


MIN_PAPER_TRADES = 5      # closed trades a bot needs before it may go on a live account


def set_account(bot_id: int, account_id: int | None, pause: bool = True) -> None:
    """Point a custom bot at a saved account. It is left paused: starting it
    is a separate, deliberate step (start_bot)."""
    with _conn() as c:
        if pause:
            c.execute("UPDATE bots SET account_id=?, enabled=0 WHERE id=?", (account_id, bot_id))
        else:
            c.execute("UPDATE bots SET account_id=? WHERE id=?", (account_id, bot_id))


def closed_trade_count(bot_id: int) -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM bot_trades WHERE bot_id=? AND side='sell_to_close'",
                         (bot_id,)).fetchone()[0]


def has_open_positions(bot_id: int) -> bool:
    return bool(_positions(bot_id))


def live_gate(bot: dict, account: dict, confirm_name: str | None = None) -> str | None:
    """Why this bot may not go (or keep going) on this account, or None.
    Every paper account passes; a live one needs live trading switched on,
    the account's name typed back, and a track record on paper first."""
    from . import accounts
    if account["type"] != "live":
        return None
    if not accounts.live_enabled():
        return "live trading is switched off"
    if confirm_name is not None and confirm_name.strip() != account["name"]:
        return "type the account's exact name to confirm live trading"
    n = closed_trade_count(bot["id"])
    if bot.get("manager_id") and bot.get("parent_id"):
        # a managed bot is a fresh ledger cloned from a fleet bot that has
        # already proven itself on paper — the source's record counts
        n = max(n, closed_trade_count(bot["parent_id"]))
    if n < MIN_PAPER_TRADES:
        return "a bot needs {} closed trades on paper before going live (this one has {})".format(
            MIN_PAPER_TRADES, n)
    return None


def start_bot(bot_id: int) -> tuple[bool, str, list[int]]:
    """Switch a custom bot on. One bot at a time per account: any other
    custom bot on the same account is paused first. Returns (ok, message,
    ids paused). The one door a bot goes live through."""
    from . import accounts
    bot = get_bot(bot_id)
    if not bot:
        return False, "no such bot", []
    acct = accounts.get(bot.get("user_id"), bot.get("account_id"))
    if not acct:
        return False, "assign the bot to an account first", []
    err = live_gate(bot, acct)
    if err:
        return False, err, []
    # one bot at a time per account — except bots a manager runs, which are
    # meant to share an account (equal slots) and are neither paused by nor
    # pause a hand-started bot
    paused = [] if bot.get("manager_id") else [
        b["id"] for b in custom_bots(bot.get("user_id"))
        if b["id"] != bot_id and b.get("account_id") == acct["id"] and b.get("enabled", 1)
        and not b.get("manager_id")]
    with _conn() as c:
        for pid in paused:
            c.execute("UPDATE bots SET enabled=0 WHERE id=?", (pid,))
        c.execute("UPDATE bots SET enabled=1 WHERE id=?", (bot_id,))
    return True, "started", paused


def set_enabled(bot_id: int, enabled: bool) -> None:
    """Pause/resume a custom bot without touching its status or ledger — it
    keeps running (and being visible) but stops taking new trades while
    disabled. This is the intended on/off switch for custom bots; they are
    never auto-retired regardless of this flag."""
    with _conn() as c:
        c.execute("UPDATE bots SET enabled=? WHERE id=?", (1 if enabled else 0, bot_id))


def _scope(sql: str, params: list, pool: str | None, user_id: int | None) -> tuple[str, list]:
    """Append the pool / owner filters every listing shares. `user_id=None`
    means "every user" — the population-wide view evolution and the admin
    pages use; a number narrows it to one account's own bots."""
    if pool:
        sql += " AND COALESCE(pool,'fleet')=?"
        params.append(pool)
    if user_id is not None:
        sql += " AND user_id=?"
        params.append(int(user_id))
    return sql, params


def active_bots(pool: str | None = None, user_id: int | None = None) -> list[dict]:
    sql, params = _scope("SELECT * FROM bots WHERE status='active'", [], pool, user_id)
    with _conn() as c:
        rows = c.execute(sql + " ORDER BY id", params).fetchall()
    return [dict(r) for r in rows]


def custom_capital_used(user_id: int | None = None, exclude_id: int | None = None) -> float:
    """Sum of every custom bot's starting allocation — its own real capital,
    carved out of the owner's paper account. Counts every custom bot
    regardless of status (a failing/disabled one still holds its allocation
    until deleted), so a user's pool never gets double-allocated. A bot a
    manager has retired is the exception: its slot's money goes to the
    replacement. With
    `user_id` given, only that account's bots count."""
    with _conn() as c:
        if user_id is None:
            rows = c.execute(
                "SELECT id, start_balance FROM bots WHERE COALESCE(pool,'fleet')='custom' "
                "AND NOT (manager_id IS NOT NULL AND status!='active')"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, start_balance FROM bots "
                "WHERE COALESCE(pool,'fleet')='custom' AND user_id=? "
                "AND NOT (manager_id IS NOT NULL AND status!='active')", (int(user_id),)
            ).fetchall()
    return sum(r["start_balance"] or 0.0 for r in rows if r["id"] != exclude_id)


def custom_bots(user_id: int | None = None) -> list[dict]:
    """Every custom bot, or just one account's with `user_id`."""
    with _conn() as c:
        if user_id is None:
            rows = c.execute(
                "SELECT * FROM bots WHERE COALESCE(pool,'fleet')='custom' ORDER BY id"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM bots WHERE COALESCE(pool,'fleet')='custom' AND user_id=? "
                "ORDER BY id", (int(user_id),)).fetchall()
    return [dict(r) for r in rows]


def user_bots(user_id: int) -> list[dict]:
    """Every bot an account owns, in every pool."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM bots WHERE user_id=? ORDER BY id",
                         (int(user_id),)).fetchall()
    return [dict(r) for r in rows]


def delete_user_bots(user_id: int) -> int:
    """Delete every bot an account owns — its fleet, its baselines and its
    custom bots (the account is being removed). Only the bookkeeping goes:
    nothing is sent to the broker, so any position still open on that
    user's Alpaca account stays open there.

    The genomes the fleet was running stay in the shared pool with the
    record they built — that evidence belongs to the population, and other
    users are still learning from it. The bots that produced it going away
    does shrink the pooled sample, which is the honest outcome: the
    trades are gone with the ledger they lived in."""
    ids = [b["id"] for b in user_bots(user_id)]
    for bid in ids:
        delete_bot(bid)
    _LB_CACHE.clear()
    return len(ids)


def live_bots(user_id: int | None = None) -> list[dict]:
    """Every bot the engine should actually tick this cycle: the whole active
    fleet, the fixed null-hypothesis baseline bots, active custom bots the
    user hasn't disabled, PLUS any fleet bot retired/failed in roughly the
    last month that's still holding an open position.

    That last group matters because of swing: a bot's automatic retirement
    paths already avoid touching one mid-hold (see worst_active()/
    stale_active_bots()), but a human can still retire one by hand from the
    UI while it's holding a position that's weeks from its own real expiry.
    Dropping out of live_bots() the instant status flips would mean no more
    stop/trim/trail management until _settle_expired's real-expiry backstop
    — real risk left unmanaged for a long stretch. So a retired/failing bot
    stays ticked purely for exit management until it's flat, then drops out
    of this set on its own (it stops matching "has an open position") with
    no extra bookkeeping; bot_engine._step_bot's `if positions:` branch
    means it's never eligible to open a NEW position once retired either
    way — status=='active' is required to reach the entry logic at all.

    With `user_id`, only that account's — the engine ticks one fleet at a
    time so each user's bots trade on their own keys and their own account
    snapshot (see bot_engine._tick).

    Runs completely separately from evolution/training — see
    bot_engine.evolve()."""
    sql = ("SELECT * FROM bots WHERE ("
           "  (status='active' AND COALESCE(pool,'fleet')='fleet')"
           "  OR (status='active' AND COALESCE(pool,'fleet')='baseline')"
           "  OR (status='active' AND COALESCE(pool,'fleet')='custom' AND COALESCE(enabled,1)=1)"
           "  OR (status IN ('retired','failing') AND COALESCE(pool,'fleet')='fleet'"
           "      AND flagged_at >= ?))")
    params = [(datetime.now(timezone.utc) - timedelta(days=30)).isoformat()]
    if user_id is not None:
        sql += " AND user_id=?"
        params.append(int(user_id))
    with _conn() as c:
        rows = c.execute(sql + " ORDER BY id", params).fetchall()
    out = []
    for r in rows:
        b = dict(r)
        if b["status"] == "active" or _positions(b["id"]):
            out.append(b)
    return out


def baseline_bots(user_id: int | None = None) -> list[dict]:
    sql, params = _scope("SELECT * FROM bots WHERE 1=1", [], "baseline", user_id)
    with _conn() as c:
        rows = c.execute(sql + " ORDER BY id", params).fetchall()
    return [dict(r) for r in rows]


BASELINE_SPECS = [("random", s) for s in _genome.SYMBOLS] + \
                 [("buy_hold", s) for s in _genome.SYMBOLS]


def ensure_baseline_bots(user_id: int) -> int:
    """Seed one account's fixed random/buy-hold reference bots, if missing.

    Per user, because a baseline has to be a null hypothesis for THAT
    fleet: it trades the same account, under the same keys, in the same
    execution mode, so the fleet's numbers are compared against a floor
    that paid the same costs. They are never bred, mutated, or culled by
    evolution (which only ever operates on pool='fleet') and never enter
    the shared gene pool — they exist purely so the evolved fleet's numbers
    can be judged against a null-signal and a direction-only floor instead
    of taken on faith."""
    with _conn() as c:
        existing = {(r["strategy"], r["symbol"]) for r in c.execute(
            "SELECT strategy, symbol FROM bots WHERE COALESCE(pool,'fleet')='baseline' "
            "AND status='active' AND user_id=?", (int(user_id),)).fetchall()}
    spawned = 0
    for strat, sym in BASELINE_SPECS:
        if (strat, sym) in existing:
            continue
        g = _genome.baseline_genome(strat, sym)
        create_bot("{} baseline ({})".format(strat.replace("_", "-"), sym),
                  strategy=strat, symbol=sym, genome=g, pool="baseline",
                  origin="baseline", user_id=int(user_id))
        spawned += 1
    return spawned


def reset_failed_baseline_bots(user_id: int | None = None) -> int:
    """A baseline bot that hits its own drawdown guardrail gets force-flagged
    'failing' like anything else (see _check_drawdown) — but unlike a fleet
    bot it isn't bred from or replaced by evolve(), so left alone it would
    just go stale. Retire it and let ensure_baseline_bots() spawn a fresh one
    so the comparison keeps running."""
    n = 0
    for b in baseline_bots(user_id):
        if b["status"] == "failing":
            retire_bot(b["id"], "baseline_reset")
            n += 1
    return n


def baseline_stats(user_id: int | None = None) -> list[dict]:
    out = []
    for b in active_bots(pool="baseline", user_id=user_id):
        s = stats(b["id"])
        out.append({**b, **s, "genome_desc": _genome.describe(genome_of(b))})
    return out


def counts(pool: str | None = None, user_id: int | None = None) -> dict:
    sql, params = _scope("SELECT status, COUNT(*) n FROM bots WHERE 1=1", [], pool, user_id)
    with _conn() as c:
        rows = c.execute(sql + " GROUP BY status", params).fetchall()
        if user_id is None:
            arch = c.execute("SELECT COUNT(*) FROM archived_bots").fetchone()[0]
        else:
            arch = c.execute("SELECT COUNT(*) FROM archived_bots WHERE user_id=?",
                             (int(user_id),)).fetchone()[0]
    d = {r["status"]: r["n"] for r in rows}
    d["archived"] = arch
    d["active"] = d.get("active", 0)
    return d


def strategy_mix(pool: str | None = None, user_id: int | None = None) -> list[dict]:
    """Active head count and share per strategy, biggest first.

    Cheap (one grouped query) so the bots page can show whether evolution is
    keeping several strategies in play or collapsing onto one.
    """
    sql, params = _scope(
        "SELECT COALESCE(NULLIF(strategy,''),'?') s, COUNT(*) n FROM bots "
        "WHERE status='active'", [], pool, user_id)
    with _conn() as c:
        rows = c.execute(sql + " GROUP BY s", params).fetchall()
    total = sum(r["n"] for r in rows) or 1
    return sorted(
        ({"strategy": r["s"], "count": r["n"], "share": r["n"] / total * 100.0}
         for r in rows),
        key=lambda d: d["count"], reverse=True)


def _ranked_active(min_trades: int = 3, pool: str | None = None,
                   user_id: int | None = None) -> list[tuple]:
    """(fitness, eligible, bot) for every active bot with a minimal sample,
    best first. Eligible bots (proven across trades + days) always sort ahead
    of not-yet-eligible ones regardless of raw fitness."""
    out = []
    for b in active_bots(pool, user_id):
        s = stats(b["id"])
        if s["closed_count"] >= min_trades:
            out.append((s["fitness"], s["eligible"], b))
    out.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return out


def top_active(k: int, min_trades: int = 3, pool: str | None = None,
               user_id: int | None = None) -> list[dict]:
    ranked = _ranked_active(min_trades, pool, user_id)
    elig = [b for f, e, b in ranked if e]
    if elig:
        return elig[:k]
    return [b for f, e, b in ranked[:k]]        # fall back before anyone qualifies


def top_eligible(k: int, pool: str | None = None, user_id: int | None = None,
                 pooled: bool = False) -> list[dict]:
    """Bots that count as proven: past MIN_ELIGIBLE_TRADES/DAYS/REGIMES
    (never a fallback to an unproven bot just because nothing qualifies
    yet — see top_active(), which the UI leaderboard still uses and which
    DOES fall back), not currently souring (excluded outright if
    fitness_recent has gone negative), and ranked by `robust_fitness` — the
    worst of lifetime, recent and marked-to-market fitness — rather than
    raw lifetime fitness, so a genome has to be good over its whole life
    AND right now to win a breeding slot, not just once.

    With `pooled=True` the judgement is made on the genome's record across
    EVERY user's fleet (gene_pool.pooled_stats) instead of this one bot's
    ledger. That is what makes multiple fleets worth having: a genome two
    users each gave thirty trades is proven — or culled — on sixty, in half
    the calendar days it would have taken either of them alone.
    """
    from . import gene_pool
    ranked = []
    for b in active_bots(pool, user_id):
        s = gene_pool.pooled_stats(b.get("genome_fp")) if pooled else stats(b["id"])
        if not s["proven"]:          # sample gates AND the multiple-comparison threshold
            continue
        if s.get("fitness_recent") is not None and s["fitness_recent"] < 0:
            continue
        ranked.append((s["robust_fitness"], b))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in ranked[:k]]


def top_eligible_by_family(k: int, pool: str | None = "fleet",
                           user_id: int | None = None, pooled: bool = False) -> dict:
    """{family: [proven bots, best first]} — top_eligible() split by
    strategy family so breeding can give every family with a proven member
    an equal shot, instead of letting whichever family happens to hold the
    top-8 slots this week monopolise the children."""
    out: dict = {}
    for b in top_eligible(10_000, pool, user_id, pooled=pooled):
        fam = _genome.family_of(genome_of(b))
        if len(out.setdefault(fam, [])) < k:
            out[fam].append(b)
    return out


def family_leaderboard(pool: str = "fleet", user_id: int | None = None) -> list[dict]:
    """Per-strategy-family standings for the fleet page, off the cached
    leaderboard: head count, how many are proven, mean robust fitness, and
    the family's best bot."""
    rows = leaderboard("active", pool=pool, user_id=user_id)
    fams: dict = {}
    for r in rows:
        fams.setdefault(r.get("strategy") or "?", []).append(r)
    out = []
    for fam, members in fams.items():
        best = max(members, key=lambda r: r.get("robust_fitness") or -1e9)
        fits = [r.get("robust_fitness") or 0.0 for r in members]
        out.append({
            "family": fam, "count": len(members),
            "proven": sum(1 for r in members if r.get("proven")),
            "eligible": sum(1 for r in members if r.get("eligible")),
            "avg_fitness": sum(fits) / len(fits),
            "avg_pl_pct": sum((r.get("pl_pct") or 0.0) for r in members) / len(members),
            "best_id": best["id"], "best_name": best["name"],
            "best_fitness": best.get("robust_fitness") or 0.0,
            "best_pl_pct": best.get("pl_pct"),
        })
    out.sort(key=lambda d: (d["proven"], d["avg_fitness"]), reverse=True)
    return out


def best_genome(min_trades: int = 3) -> dict | None:
    """Genome of the fittest proven bot — active first, then the archive."""
    top = top_active(1, min_trades)
    if top:
        return genome_of(top[0])
    with _conn() as c:
        r = c.execute("SELECT genome FROM archived_bots "
                      "ORDER BY final_pl_pct DESC LIMIT 1").fetchone()
    return _genome.loads(r["genome"]) if r else None


def worst_active(k: int, min_trades: int = 5, min_age_hours: float = 24.0,
                 pool: str | None = None, user_id: int | None = None,
                 pooled: bool = False) -> list[dict]:
    """The k weakest active bots by P&L %%.

    Bots younger than `min_age_hours` are skipped: a fresh off-lineage variant
    (or a whole new strategy just introduced by mutation) needs a fair trial
    window before it competes for its life against established bots.

    Ranks by fitness_marked, so a bot sitting on an unrealized loser is
    judged on that loser at today's price rather than sheltered by it. The
    caller (bot_engine.daily_maintenance) flattens any open position at its
    mark via flatten_bot() before retiring — a retired bot drops out of
    live_bots()' position management, so it must never be left holding.

    With `pooled=True`, a genome that another user's fleet has already
    gathered enough evidence on is judged on that POOLED record rather than
    on this bot's own thin one — the point of a shared population is that a
    bad genome only has to be discovered once, by anybody, and dies
    everywhere. A genome the pool knows nothing about yet still falls back
    to the bot's own numbers, so nothing is culled on hearsay.
    """
    from . import gene_pool
    cutoff = datetime.now(timezone.utc) - timedelta(hours=min_age_hours)
    ranked = []
    for b in active_bots(pool, user_id):
        s = stats(b["id"])
        if pooled:
            ps = gene_pool.pooled_stats(b.get("genome_fp"))
            if ps["closed_count"] >= min_trades:
                s = {**s, "closed_count": ps["closed_count"],
                     "fitness_marked": ps["fitness_marked"]}
        if s["closed_count"] < min_trades:
            continue
        try:
            born = datetime.fromisoformat(b["created_at"])
        except (TypeError, ValueError):
            born = None
        if born is not None:
            if born.tzinfo is None:
                born = born.replace(tzinfo=timezone.utc)
            if born > cutoff:
                continue
        ranked.append((s["fitness_marked"], b))
    ranked.sort(key=lambda t: t[0])
    return [b for _, b in ranked[:k]]


def real_close(bot: dict, contract: str, qty: int) -> tuple[float, float | None, str | None, str | None]:
    """Sell `qty` of `contract` for real on the bot's account (market, waits
    for the fill). Returns (filled_qty, avg_price, order_id, error) exactly
    like broker.submit_and_wait; (0, None, None, error) if the bot's fills
    are modeled or it has no keys. Capped at what the account actually
    holds of this bot's contract (account_book.sell_cap) so a bot never
    sells another bot's contracts on a shared account."""
    from . import account_book, broker as _broker
    mode, keys = _broker.execution(bot)
    if mode != "broker":
        return 0, None, None, "modeled execution"
    if not keys:
        return 0, None, None, "no Alpaca keys saved for this bot's account"
    qty = account_book.sell_cap(keys, contract, int(qty))
    if qty <= 0:
        return 0, None, None, "contract is not held on the account"
    cid = _broker.new_client_order_id(bot["id"], "sell_to_close")
    pid = add_pending(bot["id"], contract, "sell_to_close", qty, cid,
                      symbol=_contract_symbol(contract), note="close")
    try:
        filled, px, oid, err = _broker.submit_and_wait(
            contract, qty, "sell", "market", position_intent="sell_to_close",
            keys=keys, client_order_id=cid)
    except _broker.BrokerError as e:
        filled, px, oid, err = 0, None, None, str(e)
    clear_pending(pid)
    if filled:
        account_book.adjust(keys, contract, -int(filled))
    return filled, px, oid, err


def flatten_bot(bot_id: int, note: str = "flattened at mark on retirement") -> int:
    """Close every open position so a bot can be retired without leaving a
    position unmanaged. A bot whose fills are real sells for real (market)
    and books the actual fill; only if that fails — or the bot's execution
    is modeled — is the close booked at its last mark (modeled bid side),
    flagged in the note so the ledger and the account can be reconciled.
    Returns the number of positions closed."""
    b = get_bot(bot_id)
    if not b:
        return 0
    mk = marks(bot_id)
    n = 0
    now = datetime.now(_ET)
    for contract, p in _positions(bot_id).items():
        mid = mk.get(contract, p["avg_price"])
        info = option_model.parse_occ(contract)
        minutes = option_model.minutes_to_expiry(info["expiry"], now) if info else 60.0
        qty = int(p["qty"])
        filled, real_px, oid, err = real_close(b, contract, qty)
        if filled:
            err = record_trade(bot_id, contract, "sell_to_close", int(filled), real_px,
                               symbol=info["underlying"] if info else None,
                               ts=now.isoformat(), forced=True, fill_source="broker",
                               note="{} — real market close [alpaca #{}]".format(note, oid))
            if not err and int(filled) < qty:
                # partial: the rest is settled at the mark, flagged
                px = option_model.exit_fill_price(mid, minutes)
                record_trade(bot_id, contract, "sell_to_close", qty - int(filled), px,
                             symbol=info["underlying"] if info else None,
                             ts=now.isoformat(), forced=True, fill_source="model",
                             note="{} (only {} of {} filled on the account — rest settled at mark)"
                                  .format(note, int(filled), qty))
        else:
            px = option_model.exit_fill_price(mid, minutes)
            flag = "" if err == "modeled execution" else " (broker close FAILED: {})".format(err)
            err = record_trade(bot_id, contract, "sell_to_close", qty, px,
                               symbol=info["underlying"] if info else None,
                               ts=now.isoformat(), forced=True, note=note + flag, fill_source="model")
        if not err:
            clear_pos_state(bot_id, contract)
            n += 1
    return n


def archive_bot(bot_id: int, reason: str = "manual") -> int | None:
    b = get_bot(bot_id)
    if not b:
        return None
    s = stats(bot_id)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO archived_bots(source_bot_id, user_id, name, genome, reason, "
            "archived_at, peak_pl_pct, final_pl_pct, final_balance, generation, "
            "stats_json, equity_json, trades_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (bot_id, owner_of(b), b["name"], b.get("genome"), reason, _now(),
             s["peak_pl_pct"], s["pl_pct"], s["balance"], b.get("generation") or 0,
             json.dumps(s, default=str),
             json.dumps(equity_series(bot_id), default=str),
             json.dumps(_trades(bot_id), default=str)))
        c.execute("UPDATE bots SET archived=1 WHERE id=?", (bot_id,))
        return cur.lastrowid


def list_archived(user_id: int | None = None) -> list[dict]:
    """Archived bots, or just one account's. An archived bot carries its
    owner's final balance and P&L, so it is as private as the live one was
    — the page scopes to the viewer (admins see all)."""
    sql = ("SELECT id, source_bot_id, user_id, name, reason, archived_at, peak_pl_pct, "
           "final_pl_pct, final_balance, generation, genome FROM archived_bots")
    params: list = []
    if user_id is not None:
        sql += " WHERE user_id=?"
        params.append(int(user_id))
    with _conn() as c:
        rows = c.execute(sql + " ORDER BY final_pl_pct DESC", params).fetchall()
    return [dict(r) for r in rows]


def retire_bot(bot_id: int, status: str = "retired") -> None:
    # A retired bot is no longer managed, so it must not be left holding
    # anything: close its positions first (for real, via flatten_bot).
    # "failing" is the pre-retirement flag and "active" is a revival, so
    # neither flattens. A failed close never blocks the retirement itself.
    if status not in ("active", "failing") and _positions(bot_id):
        try:
            flatten_bot(bot_id)
        except Exception as e:
            print("retire_bot {}: flatten failed: {}".format(bot_id, e), flush=True)
    with _conn() as c:
        c.execute("UPDATE bots SET status=?, flagged_at=? WHERE id=?",
                  (status, _now(), bot_id))


FIRST_TRADE_HOURS = 24.0


def retire_never_traded(hours: float = FIRST_TRADE_HOURS) -> dict:
    """A fresh bot must place its first trade within `hours` of being created
    or it is retired, so the population keeps turning over instead of
    carrying genomes that never fire. Only fleet and sim bots with NO fill
    at all are touched: a bot that has traded even once is exempt, and
    custom (user-owned) and baseline (control) bots are never auto-retired.
    The clock is wall time since created_at."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with _conn() as c:
        rows = c.execute(
            "SELECT b.id, b.created_at, COALESCE(b.pool,'fleet') AS pool FROM bots b "
            "WHERE b.status='active' AND COALESCE(b.pool,'fleet') IN ('fleet','sim') "
            "AND NOT EXISTS (SELECT 1 FROM bot_trades t WHERE t.bot_id=b.id)").fetchall()
    n: dict = {}
    for r in rows:
        try:
            born = datetime.fromisoformat(r["created_at"])
        except (TypeError, ValueError):
            continue
        if born.tzinfo is None:
            born = born.replace(tzinfo=timezone.utc)
        if born > cutoff:
            continue
        retire_bot(r["id"], "retired")
        n[r["pool"]] = n.get(r["pool"], 0) + 1
    if n:
        log_event("never_traded", {"retired": n, "hours": hours})
    return n


def retire_spread_bots(reason: str = "spreads_removed") -> dict:
    """Retire every live bot (any pool) whose stored genome names a spread
    structure — single-leg options only. Reads the RAW stored genome, since
    genome.loads() now repairs a spread to 'single' and would hide them.
    Retirement closes open positions first (retire_bot -> flatten_bot: a real
    market close for a bot with real fills)."""
    n: dict = {}
    for b in list_bots():
        if b.get("status") not in ("active", "failing"):
            continue
        try:
            raw = json.loads(b.get("genome") or "{}")
        except (ValueError, TypeError):
            continue
        if (raw.get("structure") or "single") == "single":
            continue
        retire_bot(b["id"], reason)
        key = _pool_of(b)
        n[key] = n.get(key, 0) + 1
    if n:
        log_event("spreads_removed", {"retired": n})
    return n


def prune_retired(keep_days: int = 5) -> int:
    """Drop rows (and cascade trades) for long-dead, non-archived bots so the
    DB stays bounded. Anything worth keeping is already in archived_bots.

    Fleet only — a custom bot is never automatically retired, archived or
    pruned, however it performs; the user removes one by hand if they want
    it gone. Also skips (never deletes) a bot still holding an open
    position — automatic retirement already avoids this (see
    worst_active()/stale_active_bots()), but a manually-retired swing bot
    could still be mid-hold with weeks left on a real contract; deleting
    its ledger out from under an open position would be a real loss of
    state, not just tidying."""
    cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400
    n = 0
    with _conn() as c:
        rows = c.execute(
            "SELECT id, flagged_at FROM bots "
            "WHERE status IN ('retired','failing') AND COALESCE(archived,0)=0 "
            "AND COALESCE(pool,'fleet') IN ('fleet','sim')"
        ).fetchall()
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["flagged_at"]).timestamp()
            except (TypeError, ValueError):
                continue
            if ts < cutoff and not _positions(r["id"]):
                c.execute("DELETE FROM bots WHERE id=?", (r["id"],))
                n += 1
    return n


def update_bot(bot_id: int, name: str, strategy: str = "", symbol: str = "",
               risk_pct=None) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE bots SET name=?, strategy=?, symbol=?, risk_pct=? WHERE id=?",
            ((name or "").strip() or "Unnamed bot",
             (strategy or "").strip() or None,
             (symbol or "").upper().strip() or None,
             _f(risk_pct), bot_id))


def set_status(bot_id: int, status: str) -> None:
    if status not in ("active", "failing", "retired"):
        status = "active"
    if status == "retired" and _positions(bot_id):
        try:
            flatten_bot(bot_id)
        except Exception as e:
            print("set_status {}: flatten failed: {}".format(bot_id, e), flush=True)
    with _conn() as c:
        if status == "active":
            c.execute("UPDATE bots SET status='active', flagged_at=NULL WHERE id=?",
                      (bot_id,))
        else:
            c.execute("UPDATE bots SET status=?, flagged_at=? WHERE id=?",
                      (status, _now(), bot_id))


def delete_bot(bot_id: int) -> None:
    fp = (get_bot(bot_id) or {}).get("genome_fp")
    with _conn() as c:
        c.execute("DELETE FROM bot_trades WHERE bot_id=?", (bot_id,))
        c.execute("DELETE FROM bots WHERE id=?", (bot_id,))
    _invalidate_pooled(fp)


# ---------------------------------------------------------------------------
# trades + FIFO position matching
# ---------------------------------------------------------------------------
def _trades(bot_id: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM bot_trades WHERE bot_id=? ORDER BY ts, id",
            (bot_id,)).fetchall()
    return [dict(r) for r in rows]


def _match(trades: list[dict]):
    """FIFO-match sells against buys.

    Returns (open_positions, closed_results).
      open_positions: {contract: {qty, avg_price, cost}}
      closed_results: [{contract, qty, buy_price, sell_price, pl, ts, sell_id}]
    """
    lots: dict[str, list[list]] = {}          # contract -> [[qty, price, fee, regime], ...]
    closed: list[dict] = []
    for t in trades:
        c_ = t["contract"]
        if t["side"] == "buy_to_open":
            lots.setdefault(c_, []).append([t["qty"], t["price"], t["fees"] or 0.0, t.get("regime")])
            continue
        remaining = t["qty"]
        sell_fee_per = (t["fees"] / t["qty"]) if t["qty"] else 0.0
        queue = lots.get(c_, [])
        while remaining > 1e-9 and queue:
            lot = queue[0]
            take = min(lot[0], remaining)
            buy_fee_per = (lot[2] / lot[0]) if lot[0] else 0.0
            pl = ((t["price"] - lot[1]) * take * CONTRACT_MULT
                  - take * (buy_fee_per + sell_fee_per))
            closed.append({"contract": c_, "qty": take, "buy_price": lot[1],
                           "sell_price": t["price"], "pl": pl, "ts": t["ts"],
                           "sell_id": t["id"], "regime": t.get("regime") or lot[3]})
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-9:
                queue.pop(0)
    open_pos: dict[str, dict] = {}
    for c_, queue in lots.items():
        q = sum(l[0] for l in queue)
        if q > 1e-9:
            cost = sum(l[0] * l[1] for l in queue)
            open_pos[c_] = {"qty": q, "avg_price": cost / q,
                            "cost": cost * CONTRACT_MULT}
    return open_pos, closed


def _positions(bot_id: int) -> dict:
    return _match(_trades(bot_id))[0]


def _equity(bot_id: int) -> dict:
    bot = get_bot(bot_id)
    start = bot["start_balance"] if bot else START_BALANCE
    _, closed = _match(_trades(bot_id))
    points = [{"ts": bot["created_at"] if bot else None, "balance": start}]
    bal = start
    for r in closed:
        bal += r["pl"]
        points.append({"ts": r["ts"], "balance": bal})
    return {"balance": bal, "start": start, "points": points, "closed": closed}


def position_headroom(bot_id: int, bot: dict | None = None,
                      positions: dict | None = None) -> dict:
    """How much MORE money this bot may commit to open positions right now.

    The cap is a percentage of the BOT's own account value — its start
    balance plus the P&L of every trade it has closed — and never of the
    Alpaca account behind the fleet, which belongs to every bot at once.
    Open positions already spend against it, so what a new entry may cost
    is the cap LESS what is already committed: a bot holding one capped
    position has no room for a second one.
    """
    bot = bot if bot is not None else get_bot(bot_id)
    lim = effective_guardrails(bot)
    balance = _equity(bot_id)["balance"]
    pos = positions if positions is not None else _positions(bot_id)
    open_cost = sum(p["cost"] for p in pos.values())
    cap = balance * lim["position_pct_limit"] / 100.0
    return {"balance": balance, "pct": lim["position_pct_limit"], "cap": cap,
            "open_cost": open_cost, "headroom": cap - open_cost}


def max_contracts_for(headroom: float, price: float,
                      mult: float = CONTRACT_MULT) -> int:
    """The most contracts `headroom` dollars buys at `price`. Floored, never
    rounded up and never floored up to a minimum size: one contract over the
    cap is still over the cap."""
    if headroom <= 0 or price <= 0:
        return 0
    return int((headroom + 1e-9) // (price * mult))


def lineage_ids(bot_id: int) -> list[int]:
    """The bot itself plus every ancestor it was cloned from (parent_id
    chain), nearest first. A custom bot starts life as a clone with a fresh
    ledger, so what it did while it was a fleet bot lives on its parents."""
    out, seen, cur = [], set(), bot_id
    while cur is not None and cur not in seen:
        b = get_bot(cur)
        if not b:
            break
        out.append(cur)
        seen.add(cur)
        cur = b.get("parent_id")
    return out


def entry_cost_history(bot_id: int) -> dict:
    """What one contract has cost this bot to open, across every entry it —
    or any bot it descends from, fleet history included — has made. The
    basis of min_balance_for(): real trades say how expensive entries get,
    so the balance it needs is read off them rather than guessed.
    Per-contract cost = fill x multiplier plus the fees per contract; a
    spread's net fill is already per-package."""
    ids = lineage_ids(bot_id)
    if not ids:
        return {"n": 0, "n_own": 0, "max": 0.0, "median": 0.0, "max_contract": None,
                "max_ts": None}
    with _conn() as c:
        rows = c.execute(
            "SELECT bot_id, contract, ts, qty, price, fees FROM bot_trades "
            "WHERE bot_id IN ({}) AND side='buy_to_open' AND qty>0 AND price>0 "
            "ORDER BY ts, id".format(",".join("?" * len(ids))), ids).fetchall()
    costs = [(r["price"] * CONTRACT_MULT + (r["fees"] or 0.0) / r["qty"], r) for r in rows]
    n_own = sum(1 for _, r in costs if r["bot_id"] == bot_id)
    if not costs:
        return {"n": 0, "n_own": 0, "max": 0.0, "median": 0.0, "max_contract": None,
                "max_ts": None}
    ordered = sorted(x[0] for x in costs)
    top = max(costs, key=lambda x: x[0])
    return {"n": len(costs), "n_own": n_own, "max": top[0], "median": ordered[len(ordered) // 2],
            "max_contract": top[1]["contract"], "max_ts": top[1]["ts"]}


def min_balance_for(hist: dict, min_contracts: float, position_pct: float) -> float | None:
    """The smallest bot balance at which its priciest historical entry still
    fits the guardrails: the cheapest legal entry (min_contracts of that
    contract) must come in under position_pct of the balance, because
    position_headroom() rejects anything over the cap. None with no history
    to base it on."""
    if not hist or not hist.get("n") or position_pct <= 0:
        return None
    return hist["max"] * max(1, min_contracts) / (position_pct / 100.0)


def _drawdown(points: list[dict]) -> dict:
    peak = points[0]["balance"]
    max_dd = 0.0
    for p in points:
        peak = max(peak, p["balance"])
        if peak > 0:
            max_dd = max(max_dd, (peak - p["balance"]) / peak * 100.0)
    cur = ((peak - points[-1]["balance"]) / peak * 100.0) if peak > 0 else 0.0
    return {"peak": peak, "current_pct": cur, "max_pct": max_dd}


def _invalidate_pooled(fp: str | None) -> None:
    """A genome's pooled record just changed — drop the cached view of it so
    the next reader (another user's fleet page, the next breeding decision)
    sees this fill rather than a stale summary."""
    if not fp:
        return
    try:
        from . import gene_pool
        gene_pool.invalidate(fp)
    except ImportError:
        pass


def _pretty_contract(contract: str) -> str:
    info = option_model.parse_occ(contract)
    if not info:
        return contract
    return "{} {:%b %d} ${:g} {}".format(info["underlying"], info["expiry"], info["strike"],
                                         "call" if info["is_call"] else "put")


def _dispatch(fn, *args) -> None:
    import threading
    threading.Thread(target=fn, args=args, daemon=True).start()


def notify_custom_fill(bot: dict, contract: str, side: str, qty: int, price: float,
                       avg_before: float | None, fees: float = 0.0, note: str | None = None) -> None:
    """Telegram a custom bot's entry or exit — custom bots only, and only the
    primary admin's (the Telegram chat is theirs; another account's custom bot
    must never post into it). An exit says what the trade made or lost.

    Sent from a short-lived thread so a slow Telegram call can never delay the
    next order of the fast lane, and any failure here is swallowed: a missing
    notification must never undo or block a fill that has already happened."""
    if (bot or {}).get("pool") != "custom":
        return
    try:
        from . import accounts, auth, notify
        if not notify.enabled() or bot.get("user_id") != auth.primary_admin_id():
            return
        label = _pretty_contract(contract)
        acct = accounts.get(bot.get("user_id"), bot.get("account_id")) if bot.get("account_id") else None
        who = "Bot: {}\nAccount: {}".format(
            bot["name"], "{} ({})".format(acct["name"], acct["type"]) if acct else "none assigned")
        if side == "buy_to_open":
            text = "\U0001F535 Position opened\n{}\n{}\nBought {} @ ${:.2f}  (cost ${:,.2f})".format(
                who, label, qty, price, qty * price * CONTRACT_MULT)
        else:
            pl = ((price - avg_before) * qty * CONTRACT_MULT - (fees or 0.0)
                  if avg_before else None)
            pct = ((price / avg_before - 1.0) * 100.0) if avg_before else None
            icon = "\U0001F7E2" if (pl or 0) >= 0 else "\U0001F534"
            text = "{} Position closed\n{}\n{}\nSold {} @ ${:.2f}".format(icon, who, label, qty, price)
            if pl is not None:
                text += "\nP/L: {}${:,.2f} ({:+.1f}%)".format("-" if pl < 0 else "+", abs(pl), pct)
            if note and ("force" in note or "settle" in note or "flatten" in note):
                text += "\n(" + note.split(" [")[0][:80] + ")"
        _dispatch(notify.send, text)
    except Exception as e:  # noqa: BLE001
        print("custom-bot notify failed: {}".format(e), flush=True)


def record_trade(bot_id: int, contract: str, side: str, qty, price, *,
                 symbol: str | None = None, fees=0.0, ts: str | None = None,
                 note: str | None = None, forced: bool = False,
                 order_type: str = "market", limit_price=None,
                 stop_price=None, trail_pct=None,
                 fill_source: str | None = None, regime: str | None = None) -> str | None:
    """Record one fill. Returns None on success or an error string.

    `price` is the actual fill price; `order_type` and the limit/stop/trail
    fields record the order the bot's logic chose to submit. Enforces the
    fleet guardrails. `forced=True` is only for the internal disqualification
    flatten and bypasses the sizing checks.
    """
    bot = get_bot(bot_id)
    if not bot:
        return "bot not found"
    if side not in SIDES:
        return "side must be buy_to_open or sell_to_close — sell-to-open is prohibited"
    try:
        qty = int(qty)
        price = float(price)
        fees = float(fees or 0.0)
    except (TypeError, ValueError):
        return "quantity, price and fees must be numbers"
    if qty <= 0:
        return "quantity must be a positive whole number of contracts"
    if price <= 0:
        return "price must be greater than zero"
    contract = (contract or "").strip().upper()
    if not contract:
        return "contract symbol is required"

    order_type = (order_type or "market").strip().lower()
    if order_type not in ORDER_TYPES:
        return "order type must be one of: " + ", ".join(ORDER_TYPES)
    limit_price = _f(limit_price)
    stop_price = _f(stop_price)
    trail_pct = _f(trail_pct)
    if order_type in ("limit", "stop_limit") and limit_price is None:
        return "{} orders need a limit price".format(order_type)
    if order_type in ("stop", "stop_limit") and stop_price is None:
        return "{} orders need a stop price".format(order_type)
    if order_type == "trailing_stop" and trail_pct is None:
        return "trailing_stop orders need a trail percent"

    positions = _positions(bot_id)
    open_qty = positions.get(contract, {}).get("qty", 0)
    lim = effective_guardrails(bot)

    if side == "buy_to_open":
        if bot["status"] != "active":
            return "bot is flagged failing — it can only close existing positions"
        if not forced and bot.get("pool") == "custom" and not bot.get("enabled", 1):
            return "bot is disabled — enable it to open new positions"
        if not forced:
            try:
                when_et = datetime.fromisoformat(ts or _now()).astimezone(_ET)
                info = option_model.parse_occ(contract)
                # only a contract expiring TODAY is 0DTE close-out risk — a
                # swing contract weeks out isn't affected by the same-day cutoff
                expires_today = bool(info) and info["expiry"].astimezone(_ET).date() == when_et.date()
                cutoff = no_buy_after_et(when_et.date())
                if expires_today and (when_et.hour, when_et.minute) >= (cutoff.hour, cutoff.minute):
                    return "no new positions at/after {:%H:%M} ET — 0DTE close-out window".format(cutoff)
            except (ValueError, TypeError):
                pass
        if not forced and qty < lim["min_contracts"]:
            return "minimum entry is {} contracts".format(lim["min_contracts"])
        if not forced and qty > lim["max_contracts"]:
            return "max {} contracts per trade".format(lim["max_contracts"])
        hr = position_headroom(bot_id, bot, positions)
        add_cost = qty * price * CONTRACT_MULT
        if not forced and add_cost > hr["headroom"] + 1e-6:
            return ("position cap exceeded — this fill takes open exposure to "
                    "${:,.0f}, over the {:g}% limit (${:,.0f} of a ${:,.0f} balance)"
                    .format(hr["open_cost"] + add_cost, hr["pct"], hr["cap"],
                            hr["balance"]))
    else:  # sell_to_close
        if qty > open_qty:
            return ("cannot close {} contracts — only {} open on {}"
                    .format(qty, open_qty, contract))

    with _conn() as c:
        c.execute(
            "INSERT INTO bot_trades(bot_id, ts, symbol, contract, side, qty, "
            "price, fees, order_type, limit_price, stop_price, trail_pct, "
            "forced, note, created_at, fill_source, regime) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (bot_id, ts or _now(),
             (symbol or bot["symbol"] or "").upper() or None,
             contract, side, qty, price, fees, order_type, limit_price,
             stop_price, trail_pct, 1 if forced else 0, note, _now(), fill_source, regime))

    # this fill is now part of the whole population's evidence on this genome
    _invalidate_pooled(bot.get("genome_fp"))
    notify_custom_fill(bot, contract, side, qty, price,
                       (positions.get(contract) or {}).get("avg_price"), fees, note)
    _check_drawdown(bot_id)
    return None


def delete_trade(bot_id: int, trade_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM bot_trades WHERE id=? AND bot_id=?",
                  (trade_id, bot_id))


# ---------------------------------------------------------------------------
# pending real orders — the journal that survives a cycle dying mid-order
# ---------------------------------------------------------------------------
def add_pending(bot_id: int, contract: str, side: str, qty: int, client_order_id: str, *,
                symbol: str | None = None, note: str | None = None, regime: str | None = None,
                order_type: str = "market", limit_price=None, stop_price=None,
                trail_pct=None) -> int:
    """Record that a real order is about to go out. Written BEFORE the
    submit and cleared right after the fill is booked, so if the engine
    process is killed in between, the next cycle can find the order by its
    client_order_id and book whatever actually filled
    (bot_engine._recover_pending) instead of leaving a position on the
    account that no ledger knows about."""
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO pending_orders(bot_id, contract, side, qty, client_order_id, "
            "submitted_at, symbol, note, regime, order_type, limit_price, stop_price, trail_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (int(bot_id), contract, side, int(qty), client_order_id, _now(), symbol, note,
             regime, order_type, _f(limit_price), _f(stop_price), _f(trail_pct)))
        return cur.lastrowid


def clear_pending(pending_id: int | None) -> None:
    if pending_id is None:
        return
    with _conn() as c:
        c.execute("DELETE FROM pending_orders WHERE id=?", (int(pending_id),))


def pending_orders() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM pending_orders ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# disqualification
# ---------------------------------------------------------------------------
def _check_drawdown(bot_id: int) -> None:
    bot = get_bot(bot_id)
    if not bot or bot["status"] != "active":
        return
    limit = effective_guardrails(bot)["max_drawdown_pct"]
    dd = _drawdown(_equity(bot_id)["points"])
    if dd["current_pct"] + 1e-9 >= limit:
        _flag_failing(bot_id, limit)


def _flag_failing(bot_id: int, limit: float | None = None) -> None:
    """Flatten every open position and flag the bot failing. This is a
    safety trip, not a retirement — even a custom bot (never auto-retired)
    gets force-flattened here if its own drawdown guardrail is breached.

    A bot with real fills holds a REAL position, so this submits a real
    market sell first and books the actual fill; only if the broker call
    itself fails (or the bot's execution is modeled) does it fall back to
    recording the exit at average cost (so the forced exit itself is
    P&L-neutral — the drawdown that triggered it is what's charged),
    clearly flagged so the ledger vs. the real account can be reconciled
    by hand."""
    bot = get_bot(bot_id)
    if limit is None:
        limit = effective_guardrails(bot)["max_drawdown_pct"] if bot else MAX_DRAWDOWN_PCT
    reason = "force-closed: realized drawdown hit {:g}% from peak".format(limit)
    stamp = _now()

    fills: dict[str, tuple[float, str, str | None]] = {}   # contract -> (price, note, source)
    for contract, p in _positions(bot_id).items():
        px, note, src = p["avg_price"], reason, "model"
        if bot:
            filled_qty, avg_px, oid, err = real_close(bot, contract, int(p["qty"]))
            if filled_qty and not err:
                px, src = avg_px, "broker"
                note = reason + " — real market close [alpaca #{}]".format(oid)
                if int(filled_qty) < int(p["qty"]):
                    note += " (only {} of {} filled on the account)".format(int(filled_qty), int(p["qty"]))
            elif err != "modeled execution":
                note = reason + " (broker close FAILED, settled at avg cost: {})".format(err)
        fills[contract] = (px, note, src)

    _positions_before = _positions(bot_id)
    with _conn() as c:
        for contract, p in _positions(bot_id).items():
            px, note, src = fills.get(contract, (p["avg_price"], reason, "model"))
            c.execute(
                "INSERT INTO bot_trades(bot_id, ts, symbol, contract, side, qty, "
                "price, fees, forced, note, created_at, fill_source) "
                "VALUES (?,?,?,?,?,?,?,?,1,?,?,?)",
                (bot_id, stamp, _contract_symbol(contract), contract, "sell_to_close", int(p["qty"]),
                 px, 0.0, note, stamp, src))
        c.execute("UPDATE bots SET status='failing', flagged_at=? WHERE id=?",
                  (stamp, bot_id))
    if bot:
        for contract, p in _positions_before.items():
            px = fills.get(contract, (p["avg_price"],))[0]
            notify_custom_fill(bot, contract, "sell_to_close", int(p["qty"]), px,
                               p["avg_price"], 0.0, reason)


# ---------------------------------------------------------------------------
# stats / leaderboard / chart
# ---------------------------------------------------------------------------
# Fitness = a t-statistic of per-trade returns, not a P&L ratio.
#
# Each closed trade's return is measured in units of the premium it risked
# (pl / (buy_price * qty * 100)), then DAY-ADJUSTED: the buy-and-hold
# baseline bot's return on the same symbol and day is subtracted, so a
# trend day that lifts every long-call bot at once doesn't bless any of
# them — a bot only scores for beating "just be long today". The mean of
# those adjusted returns is shrunk toward the fleet-wide average with a
# pseudo-count of SHRINK_K trades (a 10-trade hot streak counts for a
# third of what a 20-trade record does), then divided by its standard
# error. Bigger samples, steadier edges, and edges that exist AFTER the
# day's drift is removed all score higher; trade count, win rate, and raw
# P&L% no longer do on their own.
SHRINK_K = 20              # pseudo-trades of the fleet prior mixed into a bot's mean
SHRINK_K_RECENT = 5        # the recent-window veto is meant to be sensitive, so far less shrinkage
MIN_RETURN_SD = 0.02       # floor on per-trade return spread (2% of premium)


# Volatility regime a fill happened in. Eligibility requires a record that
# spans MIN_ELIGIBLE_REGIMES of these, so a genome tuned to one calm
# fortnight can't breed until it has also traded a nervous tape (or a
# scheduled macro event day, which behaves like one).
VIX_LOW, VIX_HIGH = 14.0, 20.0


def regime_label(vix: float | None, data_impact: float | None = 0.0) -> str | None:
    if (data_impact or 0.0) >= 0.8:
        return "event"
    if not vix or vix <= 0:
        return None
    if vix < VIX_LOW:
        return "low_vol"
    if vix < VIX_HIGH:
        return "mid_vol"
    return "high_vol"


def trade_return(r: dict) -> float:
    """Return on premium risked for one closed lot (net of fees)."""
    risked = (r.get("buy_price") or 0.0) * (r.get("qty") or 0.0) * CONTRACT_MULT
    return (r.get("pl") or 0.0) / risked if risked > 1e-9 else 0.0


def _contract_symbol(contract: str) -> str | None:
    info = option_model.parse_occ(contract or "")
    return info["underlying"] if info else None


def adjusted_returns(closed: list[dict], baseline_returns: dict | None = None) -> list[float]:
    """Per-trade premium returns, each minus the buy_hold baseline's return
    for the same symbol on the same ET day (0 when there is none)."""
    out = []
    for r in closed:
        adj = trade_return(r)
        if baseline_returns:
            d = _trade_date(r)
            sym = _contract_symbol(r.get("contract"))
            adj -= (baseline_returns.get(sym) or {}).get(d, 0.0) if sym else 0.0
        out.append(adj)
    return out


def fitness_from_returns(returns: list[float], prior_mean: float = 0.0,
                         k: float = SHRINK_K) -> float:
    """Shrunk t-statistic: (shrunk mean) / (sd / sqrt(n))."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    shrunk = (n * mean + k * prior_mean) / (n + k)
    var = sum((x - mean) ** 2 for x in returns) / (n - 1)
    sd = max(MIN_RETURN_SD, var ** 0.5)
    return round(shrunk / (sd / n ** 0.5), 4)


_PRIOR_CACHE: dict = {"ts": 0.0, "value": 0.0}
_PRIOR_TTL = 60.0


def fleet_prior_return(fresh: bool = False) -> float:
    """Mean baseline-adjusted per-trade return across EVERY user's active
    fleet bots — the prior each bot's own mean is shrunk toward. (A
    typically NEGATIVE number, given friction: a thin sample is pulled
    toward "you probably lose a little", not toward zero.)

    Population-wide on purpose. The prior is a statement about what a
    no-edge genome trading these symbols earns, which is a property of the
    market and the cost structure, not of whose account it happened on — so
    every fleet that joins makes it a better estimate for everyone, and a
    brand-new user's first bots are shrunk toward something real from their
    very first trade instead of toward zero."""
    import time as _t
    now = _t.time()
    if not fresh and now - _PRIOR_CACHE["ts"] < _PRIOR_TTL:
        return _PRIOR_CACHE["value"]
    base = baseline_day_returns()
    rets: list[float] = []
    for b in active_bots(pool="fleet"):
        rets.extend(adjusted_returns(_match(_trades(b["id"]))[1], base))
    value = (sum(rets) / len(rets)) if rets else 0.0
    _PRIOR_CACHE.update(ts=now, value=value)
    return value


_BASE_RET_CACHE: dict = {"ts": 0.0, "value": {}}


def baseline_day_returns(fresh: bool = False) -> dict:
    """{symbol: {date: buy_hold return on premium}} — the always-long
    reference bots' realized per-trade return per session, used to
    day-adjust every fleet bot's trades (see adjusted_returns).

    Pooled across every user's baseline bots: "what did simply being long
    SPY calls return on 3 March" is one fact about that day, and averaging
    all the accounts that measured it is a better estimate of it than any
    one account's fill. It is also what makes trades from different users
    comparable at all — every fleet's trades are adjusted against the same
    per-day yardstick, so pooled fitness compares genomes, not accounts."""
    import time as _t
    now = _t.time()
    if not fresh and now - _BASE_RET_CACHE["ts"] < _PRIOR_TTL:
        return _BASE_RET_CACHE["value"]
    out: dict = {}
    with _conn() as c:
        rows = c.execute("SELECT id, symbol FROM bots WHERE COALESCE(pool,'fleet')='baseline' "
                         "AND strategy='buy_hold'").fetchall()
    per_symbol: dict = {}
    for r in rows:
        for cl in _match(_trades(r["id"]))[1]:
            d = _trade_date(cl)
            if d is None:
                continue
            per_symbol.setdefault((r["symbol"] or "").upper(), {}) \
                      .setdefault(d, []).append(trade_return(cl))
    for sym, per_day in per_symbol.items():
        out[sym] = {d: sum(v) / len(v) for d, v in per_day.items()}
    _BASE_RET_CACHE.update(ts=now, value=out)
    return out


HOLDOUT_DAYS = int(getattr(config, "BOT_HOLDOUT_DAYS", 3))


def _trade_date(r: dict):
    try:
        d = datetime.fromisoformat(r["ts"])
    except (ValueError, TypeError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(_ET).date()


def _fitness_recent(closed: list[dict], start: float,
                    days: int = HOLDOUT_DAYS, baseline_returns: dict | None = None):
    """Fitness computed only from a bot's most recent `days` distinct
    trading days of closed trades, isolated from everything earlier.

    This isn't a true out-of-sample backtest (the fleet only ever runs
    forward, live/paper) — but it does directly answer the overfitting
    question a lifetime-only fitness number can't: a genome that got bred
    from because of an early hot streak, whose *recent* trades don't hold
    up, shows a negative fitness_recent here even while its lifetime
    fitness still looks great. Returns None when there isn't enough history
    to leave a distinct "earlier" era to compare against (needs more than
    `days` distinct trading days total).
    """
    if not closed:
        return None
    dated = [(_trade_date(r), r) for r in closed]
    all_days = sorted({d for d, r in dated if d is not None})
    if len(all_days) <= days:
        return None            # no "earlier" era left to compare recent trades against
    cutoff_days = set(all_days[-days:])
    recent = [r for d, r in dated if d in cutoff_days]
    earlier = [r for d, r in dated if d not in cutoff_days]
    if not recent or not earlier:
        return None
    # a veto, so it is deliberately less shrunk (toward 0, not the fleet
    # prior) than the lifetime score — see SHRINK_K_RECENT
    return fitness_from_returns(adjusted_returns(recent, baseline_returns), 0.0, SHRINK_K_RECENT)


def stats(bot_id: int) -> dict:
    bot = get_bot(bot_id) or {}
    eq = _equity(bot_id)
    dd_limit = effective_guardrails(bot)["max_drawdown_pct"] if bot else MAX_DRAWDOWN_PCT
    is_fleet = _pool_of(bot) in ("fleet", "sim")   # sim is judged by the fleet's own prior
    return stats_from_ledger(eq, _positions(bot_id), dd_limit,
                             status=bot.get("status"), trade_count=len(_trades(bot_id)),
                             baseline_returns=baseline_day_returns(),
                             prior_mean=fleet_prior_return() if is_fleet else 0.0,
                             marks=marks(bot_id), n_tested=bots_tested())


def _marked_as_closed(open_pos: dict, marks: dict | None) -> list[dict]:
    """Open positions expressed as if closed right now at their last mark —
    so an unrealized loser counts against a bot's fitness immediately
    instead of hiding until it is finally sold."""
    out = []
    for contract, p in (open_pos or {}).items():
        m = (marks or {}).get(contract)
        if m is None:
            continue
        pl = (m - p["avg_price"]) * p["qty"] * CONTRACT_MULT
        out.append({"contract": contract, "qty": p["qty"], "buy_price": p["avg_price"],
                    "sell_price": m, "pl": pl, "ts": None, "marked": True})
    return out


def _open_pos_row(contract: str, p: dict, marks: dict | None) -> dict:
    """One open position, marked to market when a mark is known.

    Uses the same formula as _marked_as_closed(), so the per-position P&L
    shown on a bot page adds up to the `unrealized_pl` the fitness numbers
    are already built from — two different answers to "what is this worth
    right now" on the same page would be worse than none at all.

    A position with no mark yet reports None, not 0.0: "not marked" and
    "flat on the trade" are different states and must not render alike.
    """
    m = (marks or {}).get(contract)
    row = {"contract": contract, **p, "mark": m, "pl": None, "pl_pct": None}
    if m is not None:
        row["pl"] = (m - p["avg_price"]) * p["qty"] * CONTRACT_MULT
        cost = p.get("cost") or 0.0
        row["pl_pct"] = (row["pl"] / cost * 100.0) if cost else None
    return row


def stats_from_ledger(eq: dict, open_pos: dict, dd_limit: float, *,
                      status=None, trade_count: int | None = None,
                      baseline_returns: dict | None = None, prior_mean: float = 0.0,
                      marks: dict | None = None, n_tested: int | None = None,
                      proven_bar: float | None = None) -> dict:
    """The pure part of stats(): everything derived from an equity record
    ({balance, start, points, closed} as _equity() builds it) plus the open
    positions. Shared with the replay harness's in-memory ledger so an
    offline evaluation scores a genome with exactly the live formula.

    `marks` ({contract: last mid}) lets open positions be marked to market:
    they feed `unrealized_pl` / `equity` and `fitness_marked` (fitness as if
    every open position were closed at its mark), which is what culling
    ranks by. The realized-only balance still drives the drawdown
    guardrail — that trip is a safety mechanism, not a score."""
    closed = eq["closed"]
    dd = _drawdown(eq["points"])
    win_lots = [r for r in closed if r["pl"] > 1e-9]
    wins = [r["pl"] for r in win_lots]
    losses = [r["pl"] for r in closed if r["pl"] < -1e-9]
    decided = len(wins) + len(losses)
    # the biggest single win as a RETURN, not just a dollar amount: $400 on a
    # $200 lot and $400 on a $4,000 one are not the same result, and only the
    # percentage says which genome actually did something
    best_lot = max(win_lots, key=lambda r: r["pl"], default=None)

    streak = 0
    for r in reversed(closed):
        if r["pl"] > 1e-9:
            if streak < 0:
                break
            streak += 1
        elif r["pl"] < -1e-9:
            if streak > 0:
                break
            streak -= 1
        else:
            break

    start = eq["start"]

    # distinct ET trading days the bot has closed a trade on
    days = set()
    for r in closed:
        try:
            d = datetime.fromisoformat(r["ts"])
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            days.add(d.astimezone(_ET).date())
        except (ValueError, TypeError):
            pass
    active_days = len(days)

    pl_pct = ((eq["balance"] - start) / start * 100.0) if start else 0.0
    win_rate = (len(wins) / decided * 100.0) if decided else None
    gross_w = sum(wins)
    gross_l = -sum(losses)
    profit_factor = (gross_w / gross_l) if gross_l > 1e-9 else (
        999.0 if gross_w > 1e-9 else 0.0)

    adj = adjusted_returns(closed, baseline_returns)
    fitness = fitness_from_returns(adj, prior_mean)
    fitness_recent = _fitness_recent(closed, start, baseline_returns=baseline_returns)
    marked = _marked_as_closed(open_pos, marks)
    unrealized = sum(r["pl"] for r in marked)
    fitness_marked = (fitness_from_returns(adj + adjusted_returns(marked, None), prior_mean)
                      if marked else fitness)
    # the worst of lifetime, recent, and marked-to-market — a genome only
    # ranks as "robust" if it's holding up over its whole life, right now,
    # AND once whatever it is still sitting on is counted at today's price.
    robust_fitness = min(x for x in (fitness, fitness_recent, fitness_marked) if x is not None)
    regimes = sorted({r["regime"] for r in closed if r.get("regime")})
    eligible = (len(closed) >= MIN_ELIGIBLE_TRADES
                and active_days >= MIN_ELIGIBLE_DAYS
                and len(regimes) >= MIN_ELIGIBLE_REGIMES)
    threshold = proven_threshold(n_tested) if proven_bar is None else proven_bar
    # a t-stat is per-trade and ignores size, so it can't see an account losing
    # money: proven and promising both also demand real dollar profit
    profitable = eq["balance"] > start and profit_factor >= 1.0
    proven = bool(eligible and profitable and robust_fitness >= threshold)
    promising = bool(eligible and profitable and not proven and robust_fitness >= PROMISING_T)

    return {
        "status": status,
        "start_balance": start,
        "balance": eq["balance"],
        "pl_dollar": eq["balance"] - start,
        "pl_pct": pl_pct if start else None,
        "closed_count": len(closed),
        "trade_count": trade_count if trade_count is not None else len(closed) * 2,
        "active_days": active_days,
        "profit_factor": round(profit_factor, 2),
        "fitness": fitness,
        "fitness_recent": fitness_recent,
        "fitness_marked": fitness_marked,
        "robust_fitness": robust_fitness,
        "avg_trade_return": (sum(adj) / len(adj)) if adj else None,   # baseline-adjusted, on premium
        "unrealized_pl": unrealized,
        "equity": eq["balance"] + unrealized,
        "holdout_days": HOLDOUT_DAYS,
        "eligible": eligible,
        "regimes": regimes,
        "regime_count": len(regimes),
        "proven": proven,
        "promising": promising,
        "promising_threshold": PROMISING_T,
        "proven_threshold": threshold,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "avg_win": (sum(wins) / len(wins)) if wins else None,
        "avg_loss": (sum(losses) / len(losses)) if losses else None,
        "largest_win": max(wins) if wins else None,
        "largest_win_pct": (trade_return(best_lot) * 100.0) if best_lot else None,
        "largest_loss": min(losses) if losses else None,
        "streak": streak,
        "peak_balance": dd["peak"],
        "peak_pl_pct": ((dd["peak"] - start) / start * 100.0) if start else 0.0,
        "peak_pl_dollar": dd["peak"] - start,
        "drawdown_pct": dd["current_pct"],
        "max_drawdown_pct": dd["max_pct"],
        "drawdown_limit": dd_limit,
        "dq_line": dd["peak"] * (1.0 - dd_limit / 100.0),
        "dq_headroom_pct": max(0.0, dd_limit - dd["current_pct"]),
        "open_positions": [_open_pos_row(k, v, marks) for k, v in sorted(open_pos.items())],
        "open_count": len(open_pos),
        "open_cost": sum(p["cost"] for p in open_pos.values()),
    }


_LB_CACHE: dict[str, tuple[float, list]] = {}
_LB_TTL = 45.0
WARMER_ON = False           # set by the web process once its background warmer starts


def invalidate_caches() -> None:
    """Drop every derived-view cache in this module: the leaderboard rows and
    the two fitness inputs (the fleet prior and the buy-and-hold day
    returns) that stats() shrinks and day-adjusts against.

    These exist so a page load is one pass over the ledgers rather than one
    per row, at the cost of showing numbers up to their TTL old. The
    Refresh button on the fleet and Custom Bots pages calls this so a user
    who wants the current state can have it without waiting the TTL out."""
    _LB_CACHE.clear()
    _PRIOR_CACHE.update(ts=0.0, value=0.0)
    _BASE_RET_CACHE.update(ts=0.0, value={})


def leaderboard(scope: str = "active", pool: str = "fleet", *, fresh: bool = False,
                user_id: int | None = None) -> list[dict]:
    """Ranked rows for one pool. `user_id` narrows a custom-pool board to
    one account's bots (a user only ever sees their own)."""
    import time as _t
    key = "{}:{}:{}".format(scope, pool, user_id)
    hit = _LB_CACHE.get(key)
    if not fresh and hit and _t.time() - hit[0] < _LB_TTL:
        return hit[1]
    # stale-while-revalidate: once the page-view warmer is running it keeps
    # these fresh, so a reader that finds an expired row serves it rather than
    # paying seconds to rebuild it (and queuing behind the warmer doing the same)
    if not fresh and hit and WARMER_ON and _t.time() - hit[0] < 600.0:
        return hit[1]
    out = _leaderboard(scope, pool, user_id)
    _LB_CACHE[key] = (_t.time(), out)
    return out


def _leaderboard(scope: str = "active", pool: str = "fleet",
                 user_id: int | None = None) -> list[dict]:
    from . import gene_pool
    if scope == "active":
        rows_pool = active_bots(pool, user_id)
    else:
        sql, params = _scope("SELECT * FROM bots WHERE 1=1", [], pool, user_id)
        if scope != "all":
            sql += " AND status=?"
            params.append(scope)
        with _conn() as c:
            rows_pool = [dict(r) for r in c.execute(sql + " ORDER BY id", params).fetchall()]
    out = []
    for b in rows_pool:
        s = stats(b["id"])
        g = genome_of(b)
        # what the whole population knows about this bot's genome, alongside
        # what this bot's own ledger says. The row shows both: your bot's
        # P&L is yours, the evidence behind it is everyone's.
        pooled = gene_pool.pooled_stats(b.get("genome_fp")) if pool == "fleet" else None
        out.append({**b, **s, "genome_desc": _genome.describe(g),
                    "style": g.get("style", "0dte"),
                    "pooled": pooled,
                    "pooled_trades": (pooled or {}).get("closed_count"),
                    "pooled_users": (pooled or {}).get("users"),
                    "pooled_proven": (pooled or {}).get("proven"),
                    "pooled_fitness": (pooled or {}).get("robust_fitness"),
                    **_sparks(equity_series(b["id"]), s.get("unrealized_pl"))})
    out.sort(key=lambda r: r["balance"], reverse=True)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# Sort options for the Bot Fleet list. (label, key fn, reverse).
BOT_SORTS = {
    "profit_desc":   ("Most profit",       lambda r: r.get("pl_dollar") or 0.0,     True),
    "profit_asc":    ("Least profit",      lambda r: r.get("pl_dollar") or 0.0,     False),
    "alive_desc":    ("Longest alive",     lambda r: r.get("created_at") or "9999", False),
    "alive_asc":     ("Shortest alive",    lambda r: r.get("created_at") or "0000", True),
    "fitness_desc":  ("Best fitness",      lambda r: r.get("fitness") or 0.0,       True),
    "fitness_asc":   ("Worst fitness",     lambda r: r.get("fitness") or 0.0,       False),
    "robust_desc":   ("Best robust fitness", lambda r: r.get("robust_fitness") or 0.0, True),
    "robust_asc":    ("Worst robust fitness", lambda r: r.get("robust_fitness") or 0.0, False),
    "winrate_desc":  ("Best win rate",     lambda r: r.get("win_rate") or 0.0,      True),
    "drawdown_desc": ("Deepest drawdown",  lambda r: r.get("drawdown_pct") or 0.0,  True),
    "trades_desc":   ("Most trades",       lambda r: r.get("closed_count") or 0,    True),
    "days_desc":     ("Most days active",  lambda r: r.get("active_days") or 0,      True),
    "gen_desc":      ("Newest generation", lambda r: r.get("generation") or 0,      True),
    "gen_asc":       ("Oldest generation", lambda r: r.get("generation") or 0,      False),
}


def filter_sort_rows(rows: list[dict], *, sort: str = "profit_desc",
                     strategy: str = "all", symbol: str = "all",
                     proven: str = "all", style: str = "all",
                     q: str = "", marked: bool = False) -> list[dict]:
    """Apply the list-page filters + sort to a leaderboard result (a copy).

    `marked` ranks the profit sorts on P&L *with* open positions counted at
    their latest mark, so the grid's order matches the value its cards are
    showing. Only the profit sorts move: every other sort key (fitness,
    drawdown, trade count) means the same thing in either view.

    `q` is a free-text search on the bot NAME: case-insensitive substring,
    so "moth" finds "pulse-moth g0" without the user having to know the
    generation suffix. It is deliberately a substring rather than a prefix
    match — generated names are two words plus a suffix, and the half a
    user remembers is as often the second word as the first. Names are not
    unique (two bots with different genomes can be generated with the same
    name), so a search is expected to return several rows, not one.
    """
    out = list(rows)
    needle = (q or "").strip().lower()
    if needle:
        out = [r for r in out if needle in (r.get("name") or "").lower()]
    if strategy != "all":
        out = [r for r in out if (r.get("strategy") or "") == strategy]
    if symbol != "all":
        out = [r for r in out if (r.get("symbol") or "").upper() == symbol.upper()]
    if style != "all":
        out = [r for r in out if (r.get("style") or "0dte") == style]
    if proven == "proven":
        out = [r for r in out if r.get("proven")]
    elif proven == "unproven":
        out = [r for r in out if not r.get("proven")]
    _, keyfn, rev = BOT_SORTS.get(sort, BOT_SORTS["profit_desc"])
    if marked and sort in ("profit_desc", "profit_asc"):
        keyfn = lambda r: (r.get("pl_dollar") or 0.0) + (r.get("unrealized_pl") or 0.0)
    out = sorted(out, key=keyfn, reverse=rev)
    return [{**r, "rank": i} for i, r in enumerate(out, 1)]


def fleet_stats(user_id: int | None = None) -> dict:
    """Aggregate scoreboard for one user's active fleet (built off the
    cached leaderboard, so it is cheap to call on every page load).
    `user_id=None` aggregates every fleet — the installation-wide view the
    Shared Pool page shows, where the headline tiles name a bot only if the
    viewer is allowed to open it (its owner, or an admin)."""
    rows = leaderboard("active", user_id=user_id)
    n = len(rows)

    base_rows = baseline_stats(user_id)
    base_fit = [r["fitness"] for r in base_rows if r.get("closed_count")]
    base_pl = [r["pl_pct"] for r in base_rows if r.get("pl_pct") is not None]
    avg_baseline_fitness = (sum(base_fit) / len(base_fit)) if base_fit else None
    avg_baseline_pl_pct = (sum(base_pl) / len(base_pl)) if base_pl else None

    if not n:
        return {"bots": 0, "fleets": 0, "total_unrealized": 0.0,
                "total_marked_equity": 0.0, "bots_holding": 0,
                "avg_fleet_fitness": None,
                "avg_baseline_fitness": avg_baseline_fitness,
                "avg_baseline_pl_pct": avg_baseline_pl_pct, "beats_baseline": False}

    start_sum = sum(r["start_balance"] or 0.0 for r in rows)
    bal_sum = sum(r["balance"] or 0.0 for r in rows)
    pl_sum = bal_sum - start_sum
    traded = [r for r in rows if (r["closed_count"] or 0) > 0]
    decided = [r for r in rows if r["win_rate"] is not None]

    by_pl = max(rows, key=lambda r: r["pl_pct"] if r["pl_pct"] is not None else -1e9)
    wins = [r for r in rows if r["largest_win"] is not None]
    big_win = max(wins, key=lambda r: r["largest_win"]) if wins else None
    by_peak = max(rows, key=lambda r: r["peak_pl_pct"] or 0.0)
    # each headline names one bot — carry that bot's other number too, so the
    # tile reads as a result rather than a bare ratio or a bare amount

    # best strategy = highest mean P&L %% among strategies with >= 3 rated bots
    buckets: dict[str, list[float]] = {}
    for r in rows:
        if r["pl_pct"] is None:
            continue
        buckets.setdefault(r["strategy"] or "?", []).append(r["pl_pct"])
    rated = {k: sum(v) / len(v) for k, v in buckets.items() if len(v) >= 3}
    best_strat = max(rated, key=rated.get) if rated else (
        max(buckets, key=lambda k: sum(buckets[k]) / len(buckets[k])) if buckets else None)

    avg_fleet_fitness = sum(r["fitness"] or 0.0 for r in rows) / n

    # what the fleet is still holding, at its latest marks. Kept beside the
    # realized total rather than folded into it: a booked P&L and a price
    # nobody has sold at are different claims, and the tiles say which.
    unreal_sum = sum(r.get("unrealized_pl") or 0.0 for r in rows)
    holding = sum(1 for r in rows if (r.get("open_count") or 0) > 0)

    return {
        "bots": n,
        "fleets": len({r.get("user_id") for r in rows if r.get("user_id") is not None}),
        "in_profit": sum(1 for r in rows if (r["pl_dollar"] or 0) > 0),
        "total_pl_dollar": pl_sum,
        "total_pl_pct": (pl_sum / start_sum * 100.0) if start_sum else 0.0,
        "total_equity": bal_sum,
        "total_unrealized": unreal_sum,
        "total_unrealized_pct": (unreal_sum / start_sum * 100.0) if start_sum else 0.0,
        "total_marked_equity": bal_sum + unreal_sum,
        "bots_holding": holding,
        "closed_trades": sum(r["closed_count"] or 0 for r in rows),
        "open_positions": sum(r["open_count"] or 0 for r in rows),
        "avg_win_rate": (sum(r["win_rate"] for r in decided) / len(decided)) if decided else None,
        "avg_drawdown_pct": (sum(r["drawdown_pct"] or 0.0 for r in rows) / n),
        "best_pl_pct": by_pl["pl_pct"],
        "best_pl_dollar": by_pl["pl_dollar"],
        "best_pl_name": by_pl["name"],
        "best_pl_id": by_pl["id"],
        "best_pl_user": by_pl.get("user_id"),
        "highest_win": big_win["largest_win"] if big_win else None,
        "highest_win_pct": big_win["largest_win_pct"] if big_win else None,
        "highest_win_name": big_win["name"] if big_win else None,
        "highest_win_id": big_win["id"] if big_win else None,
        "highest_win_user": big_win.get("user_id") if big_win else None,
        "best_peak_pct": by_peak["peak_pl_pct"],
        "best_peak_dollar": by_peak["peak_pl_dollar"],
        "best_peak_id": by_peak["id"],
        "best_peak_name": by_peak["name"],
        "best_peak_user": by_peak.get("user_id"),
        "best_strategy": best_strat,
        "best_strategy_pl": rated.get(best_strat) if best_strat in (rated or {}) else None,
        "rated_bots": len(decided),
        "avg_fleet_fitness": avg_fleet_fitness,
        "avg_baseline_fitness": avg_baseline_fitness,
        "avg_baseline_pl_pct": avg_baseline_pl_pct,
        "beats_baseline": (avg_baseline_fitness is not None
                           and avg_fleet_fitness > avg_baseline_fitness),
    }


def last_activity(bot_id: int) -> str | None:
    with _conn() as c:
        r = c.execute("SELECT MAX(ts) m FROM bot_trades WHERE bot_id=?",
                      (bot_id,)).fetchone()
    return r["m"] if r and r["m"] else None


def stagnant_active(pool: str | None = "fleet", user_id: int | None = None) -> list[dict]:
    """Established bots that trade but go nowhere: old enough, enough pooled
    trades, no measurable edge (lifetime fitness within +/-STAGNANT_FITNESS)
    and a balance within +/-STAGNANT_PL_PCT of where it started — and neither
    proven nor promising. Complements stale_active_bots (no trades at all) and
    worst_active (only runs when the fleet is over its size target): a flat
    bot otherwise holds its slot indefinitely. The caller flattens and
    retires, and the daily top-up replaces them with fresh genomes.

    Judged on the genome's POOLED record where there is one, so a genome is
    retired only after the whole population has given it a fair trial."""
    from . import gene_pool
    cutoff = datetime.now(timezone.utc) - timedelta(days=STAGNANT_MIN_AGE_DAYS)
    out = []
    for b in active_bots(pool, user_id):
        try:
            born = datetime.fromisoformat(b["created_at"])
        except (TypeError, ValueError):
            continue
        if born.tzinfo is None:
            born = born.replace(tzinfo=timezone.utc)
        if born > cutoff:
            continue
        s = stats(b["id"])
        ps = gene_pool.pooled_stats(b.get("genome_fp"))
        pooled = ps["closed_count"] >= s["closed_count"]
        fit = ps["fitness"] if pooled else s["fitness"]
        n = ps["closed_count"] if pooled else s["closed_count"]
        if n < STAGNANT_MIN_TRADES or abs(fit) > STAGNANT_FITNESS:
            continue
        if abs(s["pl_pct"] or 0.0) > STAGNANT_PL_PCT:
            continue
        if s["proven"] or s["promising"] or ps["proven"] or ps["promising"]:
            continue
        out.append(b)
    return out


def losing_active(pool: str | None = "fleet", user_id: int | None = None) -> list[dict]:
    """Bots with a persistent, statistically real loss: enough trades, lifetime
    fitness at or below LOSER_FITNESS, recent fitness negative too (still
    losing now, not an old hole it is climbing out of), and a negative dollar
    balance. These bleed toward the drawdown floor slowly and hold a slot the
    whole way; a genome with a real edge never looks like this.

    Judged on the genome's POOLED record where it has more trades than the
    bot's own, like stagnant_active — a bad genome dies everywhere once the
    population has seen enough of it. The caller flattens and retires."""
    from . import gene_pool
    out = []
    for b in active_bots(pool, user_id):
        s = stats(b["id"])
        if s["balance"] >= s["start_balance"]:
            continue
        ps = gene_pool.pooled_stats(b.get("genome_fp"))
        pooled = ps["closed_count"] >= s["closed_count"]
        src = ps if pooled else s
        if src["closed_count"] < LOSER_MIN_TRADES or src["fitness"] > LOSER_FITNESS:
            continue
        rec = src.get("fitness_recent")
        if rec is not None and rec >= 0:
            continue
        out.append(b)
    return out


_EXPLORER_SQL = ("origin='seed' OR (origin='adopted' AND COALESCE(generation,0)=0)")


def exploration_count(user_id: int | None) -> int:
    """Active fleet bots that are untested random genomes — seeds, and adopted
    generation-0 pool genomes (those are seeds another fleet started). The
    fleet's descendants of winners are everything else."""
    sql = ("SELECT COUNT(*) FROM bots WHERE status='active' AND COALESCE(pool,'fleet')='fleet' "
           "AND (" + _EXPLORER_SQL + ")")
    params: list = []
    if user_id is not None:
        sql += " AND user_id=?"
        params.append(int(user_id))
    with _conn() as c:
        return int(c.execute(sql, params).fetchone()[0])


def exploration_room(user_id: int | None) -> int:
    """How many more exploration bots this fleet may take before the
    EXPLORE_SHARE budget is used up (negative when it is already over)."""
    return int(EXPLORE_SHARE * fleet_cap(user_id)) - exploration_count(user_id)


def excess_explorers(user_id: int | None, max_n: int | None = None) -> list[dict]:
    """The weakest exploration bots beyond the budget, worst first, capped at
    `max_n` (default EXPLORE_TRIM_PER_DAY) so a fleet far over budget is eased
    down rather than purged in one pass. Only bots past a fair trial window and
    neither proven nor promising are eligible — a seed that is working keeps
    its slot, and so does one that has not had time yet."""
    from . import gene_pool
    over = -exploration_room(user_id)
    if over <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=STAGNANT_MIN_AGE_DAYS)
    ranked = []
    for b in active_bots("fleet", user_id):
        if not (b.get("origin") == "seed"
                or (b.get("origin") == "adopted" and int(b.get("generation") or 0) == 0)):
            continue
        try:
            born = datetime.fromisoformat(b["created_at"])
        except (TypeError, ValueError):
            continue
        if born.tzinfo is None:
            born = born.replace(tzinfo=timezone.utc)
        if born > cutoff:
            continue
        s = stats(b["id"])
        ps = gene_pool.pooled_stats(b.get("genome_fp"))
        if s["proven"] or s["promising"] or ps["proven"] or ps["promising"]:
            continue
        fit = ps["fitness_marked"] if ps["closed_count"] >= s["closed_count"] else s["fitness_marked"]
        ranked.append((fit, b))
    ranked.sort(key=lambda t: t[0])
    n = min(over, EXPLORE_TRIM_PER_DAY if max_n is None else max_n)
    return [b for _, b in ranked[:n]]


def stale_active_bots(days: int = 3, pool: str | None = None,
                      user_id: int | None = None) -> list[dict]:
    """Active bots with no fill (and no first trade) in the last `days`
    market days (weekends and NYSE holidays don't count against them).

    Skips any bot currently holding an open position — a swing bot can
    legitimately go many days between fills while genuinely mid-hold
    (that's the whole point of swing), and killing it there would discard
    a live position's whole thesis rather than a genome that just isn't
    firing."""
    now_et = datetime.now(_ET)
    cutoff_date = market_calendar.market_days_before(now_et.date(), days)
    cutoff = datetime.combine(cutoff_date, now_et.time(), tzinfo=_ET).astimezone(timezone.utc)
    out = []
    for b in active_bots(pool, user_id):
        if _positions(b["id"]):
            continue
        ref = last_activity(b["id"]) or b["created_at"]
        try:
            ts = datetime.fromisoformat(ref)
        except (TypeError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            out.append(b)
    return out


def window_return_pct(bot_id: int, days: float = 5.0) -> float:
    """Realized return, as a percent of the bot's start balance, over trades
    it closed in the last `days` days. A bot that closed nothing is at 0."""
    eq = _equity(bot_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    pl = 0.0
    for r in eq["closed"]:
        try:
            ts = datetime.fromisoformat(r["ts"])
        except (TypeError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            pl += r["pl"]
    return pl / eq["start"] * 100.0 if eq["start"] else 0.0


def equity_series(bot_id: int) -> list[dict]:
    return _equity(bot_id)["points"]


def _sparks(points: list[dict], unrealized: float | None) -> dict:
    """The same equity curve counted two ways, on one shared scale.

    There is no per-day mark history to replay for an open position — only
    its latest mark — so the marked curve is honestly the realized curve
    plus today's mark-to-market as one final leg, not a re-simulation of
    what the bot was worth each day it held. A bot holding nothing, or
    holding only positions with no mark yet, gets two identical curves.
    """
    marked = points
    if abs(unrealized or 0.0) > 1e-9:
        marked = points + [{"ts": None, "balance": points[-1]["balance"] + unrealized}]
    vals = [p["balance"] for p in marked]
    lo, hi = min(vals), max(vals)
    return {"spark": _spark(points, lo=lo, hi=hi),
            "spark_marked": _spark(marked, lo=lo, hi=hi)}


def _spark(points: list[dict], w: int = 240, h: int = 48, pad: int = 3,
           lo: float | None = None, hi: float | None = None) -> str:
    """Polyline points for an equity curve, normalised to fill the box.

    `lo`/`hi` override the curve's own range. Two curves of the same bot —
    realized and marked-to-market — are drawn on one shared scale that way,
    so toggling between them adds the open leg instead of silently redrawing
    the whole history at a different zoom.
    """
    vals = [p["balance"] for p in points]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi - lo < 1e-9:
        hi, lo = hi + 1.0, lo - 1.0
    n = len(points)
    xs = (lambda i: pad + (i / (n - 1) * (w - 2 * pad))) if n > 1 else (lambda i: w / 2)
    return " ".join("{:.1f},{:.1f}".format(xs(i), pad + (hi - p["balance"]) / (hi - lo) * (h - 2 * pad))
                    for i, p in enumerate(points))


# --- per-position scratch state (trailing high-water mark, open time) --------
def pos_state(bot_id: int) -> dict:
    with _conn() as c:
        rows = c.execute("SELECT * FROM bot_pos_state WHERE bot_id=?",
                         (bot_id,)).fetchall()
    return {r["contract"]: dict(r) for r in rows}


def touch_pos_state(bot_id: int, contract: str, price: float,
                    opened_ts: str, entry_iv: float | None = None,
                    entry_mid: float | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO bot_pos_state(bot_id, contract, hwm, opened_ts, entry_iv, last_mark, entry_mid) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(bot_id, contract) DO UPDATE SET "
            "hwm=MAX(hwm, excluded.hwm), "
            "entry_iv=COALESCE(bot_pos_state.entry_iv, excluded.entry_iv), "
            "last_mark=excluded.last_mark, "
            "entry_mid=COALESCE(bot_pos_state.entry_mid, excluded.entry_mid)",
            (bot_id, contract, price, opened_ts, entry_iv, price, entry_mid))


def marks(bot_id: int) -> dict:
    """{contract: last observed mid} for every open position with a mark."""
    with _conn() as c:
        rows = c.execute("SELECT contract, last_mark FROM bot_pos_state WHERE bot_id=? "
                         "AND last_mark IS NOT NULL", (bot_id,)).fetchall()
    return {r["contract"]: float(r["last_mark"]) for r in rows}


def clear_pos_state(bot_id: int, contract: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM bot_pos_state WHERE bot_id=? AND contract=?",
                  (bot_id, contract))


def mark_pos(bot_id: int, contract: str, *, trimmed=None, core_closed=None,
             adds=None, entry_mid=None) -> None:
    """Flag a position as trimmed / core-closed, bump its average-in count,
    or re-base its entry mid after an average-in."""
    sets, vals = [], []
    for col, v in (("trimmed", trimmed), ("core_closed", core_closed), ("adds", adds)):
        if v is not None:
            sets.append("{}=?".format(col))
            vals.append(int(v))
    if entry_mid is not None:
        sets.append("entry_mid=?")
        vals.append(float(entry_mid))
    if not sets:
        return
    vals += [bot_id, contract]
    with _conn() as c:
        c.execute("UPDATE bot_pos_state SET {} WHERE bot_id=? AND contract=?"
                  .format(", ".join(sets)), vals)


def trade_log(bot_id: int) -> list[dict]:
    """All fills, newest first, with realized P&L attached to each close."""
    trades = _trades(bot_id)
    _, closed = _match(trades)
    pl_by_sell: dict[int, float] = {}
    for r in closed:
        pl_by_sell[r["sell_id"]] = pl_by_sell.get(r["sell_id"], 0.0) + r["pl"]
    rows = []
    for t in trades:
        d = dict(t)
        d["pl"] = pl_by_sell.get(t["id"]) if t["side"] == "sell_to_close" else None
        rows.append(d)
    rows.reverse()
    return rows


def fill_mix(bot_id: int) -> dict:
    """Count of this bot's fills by fill_source ('alpaca'/'yfinance'/'model'/
    'broker'/unknown) — for the bot detail page, so "is this bot actually
    pricing off live quotes" is a fact you can check per-bot instead of
    inferred from the fleet-wide badge on the Bot Fleet page."""
    with _conn() as c:
        rows = c.execute(
            "SELECT COALESCE(fill_source, 'unknown') src, COUNT(*) n "
            "FROM bot_trades WHERE bot_id=? GROUP BY src", (bot_id,)).fetchall()
    counts = {r["src"]: r["n"] for r in rows}
    total = sum(counts.values()) or 1
    live = counts.get("alpaca", 0) + counts.get("yfinance", 0) + counts.get("broker", 0)
    return {"counts": counts, "total": sum(counts.values()),
           "live_pct": live / total * 100.0}


# ---------------------------------------------------------------------------
# engine event log — for the Engine History page
# ---------------------------------------------------------------------------
_MAX_EVENTS = 500


def log_event(kind: str, detail: dict | None = None) -> None:
    """Record one notable engine event (a fleet reset, a daily-maintenance
    pass, an evolve() cycle that actually did something). NOT called on
    every run_cycle()/evolve() — most cycles do nothing worth a row, and
    this table would otherwise grow unbounded on a 1-minute engine timer."""
    with _conn() as c:
        c.execute("INSERT INTO engine_events(ts, kind, detail) VALUES (?,?,?)",
                  (_now(), kind, json.dumps(detail or {}, default=str)))
        c.execute(
            "DELETE FROM engine_events WHERE id NOT IN "
            "(SELECT id FROM engine_events ORDER BY id DESC LIMIT ?)",
            (_MAX_EVENTS,))


def log_event_once(kind: str, detail: dict | None, key: str, within_s: float = 3600.0) -> bool:
    """log_event(), but at most once per `within_s` for the same `key` —
    for things that would otherwise repeat every minute (an order rejected
    for the same reason on every tick, a reconcile mismatch that persists
    until someone looks at it). Returns True if a row was written."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=within_s)).isoformat(timespec="seconds")
    with _conn() as c:
        r = c.execute(
            "SELECT 1 FROM engine_events WHERE kind=? AND ts>=? AND detail LIKE ? LIMIT 1",
            (kind, cutoff, '%"key": "{}"%'.format(key.replace('"', "")))).fetchone()
    if r:
        return False
    log_event(kind, {**(detail or {}), "key": key.replace('"', "")})
    return True


def recent_events(limit: int = 200, user_id: int | None = None) -> list[dict]:
    """The engine log. With `user_id`, only rows that are that account's own
    business: events it tagged with their id, plus the untagged
    installation-wide ones (data feed, quote source, settlement sweeps). An
    event carrying somebody else's user id describes their fleet's bots and
    is filtered out — the same rule the bot pages follow."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM engine_events ORDER BY id DESC LIMIT ?",
            (limit * 4 if user_id is not None else limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["detail"] = json.loads(d["detail"]) if d["detail"] else {}
        except (ValueError, TypeError):
            d["detail"] = {}
        if user_id is not None:
            owner = (d["detail"] or {}).get("user_id")
            if owner is not None and int(owner) != int(user_id):
                continue
        out.append(d)
        if len(out) >= limit:
            break
    return out


def chart_geometry(bot_id: int, width: int = 720, height: int = 220,
                   pad: int = 6) -> dict:
    """Pre-compute an SVG polyline for the realized equity curve plus the
    25%-below-peak disqualification reference line."""
    pts = equity_series(bot_id)
    st = stats(bot_id)
    vals = [p["balance"] for p in pts]
    lo = min(vals + [st["dq_line"]])
    hi = max(vals + [st["peak_balance"]])
    if hi - lo < 1e-9:
        hi, lo = hi + 1.0, lo - 1.0
    span = hi - lo
    n = len(pts)

    def x(i):
        return pad + (i / (n - 1) * (width - 2 * pad)) if n > 1 else width / 2

    def y(v):
        return pad + (hi - v) / span * (height - 2 * pad)

    line = " ".join("{:.1f},{:.1f}".format(x(i), y(p["balance"]))
                    for i, p in enumerate(pts))
    return {
        "width": width, "height": height,
        "line": line,
        "start_y": y(st["start_balance"]),
        "dq_y": y(st["dq_line"]),
        "last_y": y(vals[-1]),
        "up": st["pl_dollar"] >= 0,
        # what the hover readout shows: one entry per vertex, in drawing order
        "hover": [{"x": round(x(i), 1), "y": round(y(p["balance"]), 1),
                   "ts": p.get("ts"), "bal": round(p["balance"], 2)}
                  for i, p in enumerate(pts)],
        "start_balance": st["start_balance"],
    }
