"""Market history recorder — the data the replay harness (webapp/replay.py)
runs on.

The live engine already pulls 1-minute bars, the VIX term structure, prior-
day levels and real option quotes every minute of every session, then
throws them away. This module persists all of it to one SQLite file
(``market_history.db`` in the data dir) so a genome can be evaluated over
months of recorded sessions offline instead of costing a live market day
per test.

One history, shared by everybody. Every user's fleet reads it, and
whichever account's Alpaca entitlement served a fetch, the result lands in
the same tables (``contrib`` records who contributed what, so the Shared
Pool page can show it). That is the point: an installation with five
connected accounts records the same session five times over only if it is
wasteful — instead it fetches once on the best entitlement anyone has
(``bot_settings.data_source``) and every fleet replays and evolves against
the union. More users therefore means a deeper history per user, not a
divided one, and a genome any of them tests offline is tested against
everything all of them have ever seen.

What is recorded, and by whom:
  * bars     — every 1-min OHLCV bar the engine sees (bot_engine._tick).
  * vix      — the VIX / VIX9D / VIX open snapshot per minute (same place).
  * daily    — prior-session close/high/low per symbol per date (same place).
  * quotes   — every real option chain fetch, per contract, per minute
               (option_quotes.get_chain hooks in here on a successful fetch).

``backfill()`` fills the bars table from Alpaca (years of 1-min history on
the free IEX feed) or yfinance (last ~30 days) so the harness is useful
before the recorder has run for months. Backfilled days have no recorded
quotes, so a replay over them prices every fill off the modeled chain —
the same fallback the live engine uses on a bad quote day.

Everything here is best-effort: a recording failure is logged and never
allowed to stall the trading loop.
"""

import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone

from . import api_meter, config, market_calendar, option_model

ET = option_model.ET
DB_PATH = config.DATA_DIR / "market_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    t      TEXT NOT NULL,          -- UTC ISO minute, e.g. 2026-09-04T13:30:00+00:00
    day    TEXT NOT NULL,          -- ET session date, YYYY-MM-DD
    o REAL NOT NULL, h REAL NOT NULL, l REAL NOT NULL, c REAL NOT NULL,
    v REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, t)
);
CREATE INDEX IF NOT EXISTS ix_bars_day ON bars(symbol, day, t);
CREATE TABLE IF NOT EXISTS vix (
    t        TEXT PRIMARY KEY,     -- UTC ISO minute
    vix      REAL, vix9d REAL, vix_open REAL
);
CREATE TABLE IF NOT EXISTS daily (
    symbol TEXT NOT NULL, day TEXT NOT NULL,
    prev_close REAL, prev_high REAL, prev_low REAL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS contrib (
    user_id INTEGER NOT NULL,      -- whose Alpaca entitlement served the fetch
    day     TEXT NOT NULL,         -- ET session date
    kind    TEXT NOT NULL,         -- 'bars' | 'quotes'
    n       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day, kind)
);
CREATE TABLE IF NOT EXISTS quotes (
    symbol TEXT NOT NULL, expiry TEXT NOT NULL,
    t      TEXT NOT NULL,          -- UTC ISO minute the snapshot was taken
    occ    TEXT NOT NULL,
    bid REAL, ask REAL, last REAL, volume INTEGER,
    source TEXT,
    spot   REAL,                   -- underlying at (about) the quote's time
    qt     TEXT,                   -- the provider's own quote timestamp, if any
    PRIMARY KEY (symbol, expiry, t, occ)
);
"""


_initialized: set = set()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA busy_timeout = 15000")
    if str(DB_PATH) not in _initialized:       # lazy schema creation, once per file
        c.executescript(_SCHEMA)
        _initialized.add(str(DB_PATH))
    return c


def init() -> None:
    with _conn():
        pass


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------
def _parse_ts(t) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def minute_key(t) -> str | None:
    """Canonical UTC ISO string truncated to the minute, for table keys."""
    d = _parse_ts(t) if not isinstance(t, datetime) else t
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()


def _session_day(t) -> str | None:
    d = _parse_ts(t) if not isinstance(t, datetime) else t
    return d.astimezone(ET).date().isoformat() if d else None


def _in_session(t) -> bool:
    d = _parse_ts(t) if not isinstance(t, datetime) else t
    if d is None:
        return False
    et = d.astimezone(ET)
    m = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= m < 16 * 60


# ---------------------------------------------------------------------------
# recording (called from the live engine)
# ---------------------------------------------------------------------------
def _credit(user_id, day: str | None, kind: str, n: int) -> None:
    """Note that `user_id`'s connected account is what pulled `n` new rows
    everybody now has. Best-effort and never raises: attribution is a
    reporting nicety, the data itself is already saved."""
    if not n or not day:
        return
    uid = int(user_id) if user_id is not None else 0      # 0 = yfinance / no account
    try:
        with _conn() as c:
            c.execute("INSERT INTO contrib(user_id, day, kind, n) VALUES (?,?,?,?) "
                      "ON CONFLICT(user_id, day, kind) DO UPDATE SET n = n + excluded.n",
                      (uid, day, kind, int(n)))
    except sqlite3.Error:
        pass


def contributions() -> dict:
    """{"by_user": {uid: {bars, quotes, days}}, "totals": {...}} — who has
    fed the shared history. Rendered on the Shared Pool page so it is
    visible that connecting an account gives something back to everyone
    else, not just to yourself."""
    init()
    by_user: dict = {}
    with _conn() as c:
        for r in c.execute("SELECT user_id, kind, SUM(n) n, COUNT(DISTINCT day) days "
                           "FROM contrib GROUP BY user_id, kind").fetchall():
            u = by_user.setdefault(int(r["user_id"]), {"bars": 0, "quotes": 0, "days": 0})
            u[r["kind"]] = int(r["n"] or 0)
            u["days"] = max(u["days"], int(r["days"] or 0))
    totals = {"bars": sum(u["bars"] for u in by_user.values()),
              "quotes": sum(u["quotes"] for u in by_user.values()),
              "contributors": len([u for u in by_user if u])}
    return {"by_user": by_user, "totals": totals}


def record_bars(symbol: str, bars: list[dict], user_id=None) -> int:
    """INSERT OR IGNORE every bar — safe to call four times a minute with
    overlapping windows. Only regular-session bars are kept."""
    if not bars:
        return 0
    rows = []
    for b in bars:
        k = minute_key(b.get("t"))
        if not k or not _in_session(k):
            continue
        try:
            rows.append((symbol.upper(), k, _session_day(k),
                         float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]),
                         float(b.get("v") or 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return 0
    with _conn() as c:
        cur = c.executemany(
            "INSERT OR IGNORE INTO bars(symbol, t, day, o, h, l, c, v) VALUES (?,?,?,?,?,?,?,?)", rows)
        n = cur.rowcount if cur.rowcount is not None else 0
    _credit(user_id, rows[-1][2], "bars", n)
    return n


def record_tick(now: datetime, bars_by_sym: dict, vix: dict | None,
                daily_by_sym: dict | None, user_id=None) -> None:
    """Persist one engine tick's worth of context — once per cycle, for
    every fleet at once. Never raises."""
    try:
        init()
        for sym, bars in (bars_by_sym or {}).items():
            record_bars(sym, bars, user_id=user_id)
        k = minute_key(now)
        with _conn() as c:
            if vix and vix.get("vix"):
                c.execute("INSERT OR IGNORE INTO vix(t, vix, vix9d, vix_open) VALUES (?,?,?,?)",
                          (k, vix.get("vix"), vix.get("vix9d"), vix.get("vix_open")))
            day = _session_day(now)
            for sym, d in (daily_by_sym or {}).items():
                if d and d.get("prev_close"):
                    c.execute("INSERT OR IGNORE INTO daily(symbol, day, prev_close, prev_high, prev_low) "
                              "VALUES (?,?,?,?,?)",
                              (sym.upper(), day, d.get("prev_close"), d.get("prev_high"), d.get("prev_low")))
    except Exception as e:  # noqa: BLE001 — recording must never stall the engine
        print("recorder tick: {}".format(e), flush=True)


def record_quotes(symbol: str, expiry: date, quotes: dict, source: str,
                  now: datetime | None = None, user_id=None) -> None:
    """Persist one real chain fetch (option_quotes.get_chain). Never raises."""
    if not quotes:
        return
    try:
        init()
        k = minute_key(now or datetime.now(timezone.utc))
        rows = [(symbol.upper(), expiry.isoformat(), k, occ,
                 q.get("bid"), q.get("ask"), q.get("last"), q.get("volume"), source,
                 q.get("spot"), q.get("t"))
                for occ, q in quotes.items()]
        with _conn() as c:
            cur = c.executemany(
                "INSERT OR IGNORE INTO quotes(symbol, expiry, t, occ, bid, ask, last, volume, source, spot, qt) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            n = cur.rowcount if cur.rowcount is not None else 0
        _credit(user_id, _session_day(k), "quotes", n)
    except Exception as e:  # noqa: BLE001
        print("recorder quotes: {}".format(e), flush=True)


# ---------------------------------------------------------------------------
# reading (the replay harness)
# ---------------------------------------------------------------------------
def days(symbol: str, start: date | None = None, end: date | None = None,
         min_bars: int = 300) -> list[date]:
    """Session dates with a reasonably complete bar record for `symbol`."""
    init()
    q = "SELECT day, COUNT(*) n FROM bars WHERE symbol=?"
    args: list = [symbol.upper()]
    if start:
        q += " AND day>=?"
        args.append(start.isoformat())
    if end:
        q += " AND day<=?"
        args.append(end.isoformat())
    q += " GROUP BY day HAVING n>=? ORDER BY day"
    args.append(min_bars)
    with _conn() as c:
        return [date.fromisoformat(r["day"]) for r in c.execute(q, args).fetchall()]


def bars_for(symbol: str, day: date) -> list[dict]:
    init()
    with _conn() as c:
        rows = c.execute("SELECT t, o, h, l, c, v FROM bars WHERE symbol=? AND day=? ORDER BY t",
                         (symbol.upper(), day.isoformat())).fetchall()
    return [dict(r) for r in rows]


def vix_series(day: date) -> dict[str, dict]:
    """{minute_key: {vix, vix9d, vix_open}} for one session."""
    init()
    lo = datetime(day.year, day.month, day.day, 0, 0, tzinfo=ET).astimezone(timezone.utc).isoformat()
    hi = (datetime(day.year, day.month, day.day, 23, 59, tzinfo=ET)).astimezone(timezone.utc).isoformat()
    with _conn() as c:
        rows = c.execute("SELECT * FROM vix WHERE t>=? AND t<=? ORDER BY t", (lo, hi)).fetchall()
    return {r["t"]: {"vix": r["vix"], "vix9d": r["vix9d"], "vix_open": r["vix_open"]} for r in rows}


def daily_for(symbol: str, day: date) -> dict:
    """Prior-session levels for `day`: the recorded row if there is one,
    else derived from the previous recorded session's bars (which is what
    a backfilled history has)."""
    init()
    with _conn() as c:
        first = c.execute("SELECT o FROM bars WHERE symbol=? AND day=? ORDER BY t LIMIT 1",
                          (symbol.upper(), day.isoformat())).fetchone()
        today_open = {"today_open": first["o"]} if first else {}
        r = c.execute("SELECT * FROM daily WHERE symbol=? AND day=?",
                      (symbol.upper(), day.isoformat())).fetchone()
        if r and r["prev_close"]:
            return {"prev_close": r["prev_close"], "prev_high": r["prev_high"],
                    "prev_low": r["prev_low"], **today_open}
        p = c.execute("SELECT day, MAX(h) hi, MIN(l) lo FROM bars WHERE symbol=? AND day<? "
                      "GROUP BY day ORDER BY day DESC LIMIT 1", (symbol.upper(), day.isoformat())).fetchone()
        if not p:
            return today_open
        last = c.execute("SELECT c FROM bars WHERE symbol=? AND day=? ORDER BY t DESC LIMIT 1",
                         (symbol.upper(), p["day"])).fetchone()
    return {"prev_close": last["c"] if last else None, "prev_high": p["hi"], "prev_low": p["lo"],
            **today_open}


def quotes_at(symbol: str, expiry: date, t) -> dict:
    """{occ: {bid, ask, last, volume, source}} recorded at minute `t`, or
    the most recent snapshot within the prior 5 minutes. Empty if none."""
    init()
    k = minute_key(t)
    if not k:
        return {}
    lo = minute_key(_parse_ts(k) - timedelta(minutes=5))
    with _conn() as c:
        snap = c.execute("SELECT MAX(t) t FROM quotes WHERE symbol=? AND expiry=? AND t<=? AND t>=?",
                         (symbol.upper(), expiry.isoformat(), k, lo)).fetchone()
        if not snap or not snap["t"]:
            return {}
        rows = c.execute("SELECT * FROM quotes WHERE symbol=? AND expiry=? AND t=?",
                         (symbol.upper(), expiry.isoformat(), snap["t"])).fetchall()
    return {r["occ"]: {"bid": r["bid"], "ask": r["ask"], "last": r["last"],
                       "volume": r["volume"] or 0, "source": r["source"],
                       "t": r["qt"] or r["t"], "spot": r["spot"],
                       "delayed": (r["source"] == "yfinance")}
            for r in rows}


_STATUS_CACHE: dict = {"ts": 0.0, "value": None}


def status(*, fresh: bool = False) -> dict:
    """`_status()` behind a one-minute cache: it runs GROUP BY over every
    recorded bar (hundreds of MB) and the numbers move once a day."""
    import time as _t
    if not fresh and _STATUS_CACHE["value"] is not None and _t.time() - _STATUS_CACHE["ts"] < 600.0:
        return _STATUS_CACHE["value"]
    value = _status()
    _STATUS_CACHE.update(ts=_t.time(), value=value)
    return value


def _status() -> dict:
    """Summary for the UI / CLI: recorded day counts and ranges per symbol."""
    init()
    out = {}
    with _conn() as c:
        for r in c.execute("SELECT symbol, COUNT(DISTINCT day) days, MIN(day) first, MAX(day) last, "
                           "COUNT(*) bars FROM bars GROUP BY symbol").fetchall():
            out[r["symbol"]] = {"days": r["days"], "first": r["first"], "last": r["last"], "bars": r["bars"]}
        q = c.execute("SELECT COUNT(DISTINCT substr(t,1,10)) d FROM quotes").fetchone()
        out["quote_days"] = q["d"] if q else 0
    out["contributions"] = contributions()
    return out


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------
_ALPACA_DATA = "https://data.alpaca.markets/v2/stocks/{}/bars"


def _backfill_alpaca(symbol: str, start: date, end: date, key: str, secret: str,
                     feed: str, user_id=None) -> int:
    import requests
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "timeframe": "1Min", "limit": 10000, "feed": feed or "iex", "sort": "asc",
        "start": datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat(),
        "end": (datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)).isoformat(),
    }
    n = 0
    page = None
    for _ in range(500):
        if page:
            params["page_token"] = page
        r = api_meter.get(_ALPACA_DATA.format(symbol), keys=(key, secret),
                          params=params, headers=hdr, timeout=30)
        r.raise_for_status()
        j = r.json()
        bars = [{"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b.get("v", 0)}
                for b in (j.get("bars") or [])]
        n += record_bars(symbol, bars, user_id=user_id)
        page = j.get("next_page_token")
        if not page:
            break
        time.sleep(0.2)
    return n


def _backfill_yfinance(symbol: str, start: date, end: date, user_id=None) -> int:
    import yfinance as yf
    n = 0
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=6))
        try:
            df = yf.download(symbol, start=cur.isoformat(), end=(stop + timedelta(days=1)).isoformat(),
                             interval="1m", progress=False, auto_adjust=False)
        except Exception as e:  # noqa: BLE001
            print("backfill yfinance {} {}..{}: {}".format(symbol, cur, stop, e), flush=True)
            df = None
        if df is not None and not df.empty:
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            bars = []
            for idx, row in df.iterrows():
                bars.append({"t": idx.isoformat(), "o": float(row["Open"]), "h": float(row["High"]),
                             "l": float(row["Low"]), "c": float(row["Close"]),
                             "v": float(row.get("Volume", 0) or 0)})
            n += record_bars(symbol, bars, user_id=user_id)
        cur = stop + timedelta(days=1)
    return n


def backfill(symbols, days_back: int = 30, end: date | None = None) -> dict:
    """Fill the bars table for the last `days_back` calendar days. Alpaca
    (if keys are saved) reaches back years; yfinance only ~30 days."""
    from . import bot_settings
    init()
    end = end or datetime.now(ET).date()
    start = end - timedelta(days=days_back)
    source = bot_settings.data_source()
    out = {}
    for sym in symbols:
        try:
            if source["keys"]:
                n = _backfill_alpaca(sym, start, end, source["keys"][0], source["keys"][1],
                                     source["feed"], user_id=source["user_id"])
                src = "alpaca"
            else:
                n = _backfill_yfinance(sym, start, end)
                src = "yfinance"
        except Exception as e:  # noqa: BLE001
            print("backfill {}: {}".format(sym, e), flush=True)
            n, src = 0, "error"
        out[sym] = {"inserted": n, "source": src}
    return out


def session_minutes(day: date) -> list[datetime]:
    """Every minute of the regular session on `day` (09:30 .. 16:05 ET),
    the clock the replay harness ticks on."""
    if not market_calendar.is_market_day(day):
        return []
    t0 = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    return [t0 + timedelta(minutes=i) for i in range(0, 6 * 60 + 36)]


def main(argv=None) -> int:
    from . import genome
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        print(status(), flush=True)
    elif cmd == "backfill":
        days_back = 30
        if "--days" in argv:
            days_back = int(argv[argv.index("--days") + 1])
        print(backfill(genome.SYMBOLS, days_back), flush=True)
        print(status(), flush=True)
    else:
        print("usage: python -m webapp.recorder [status|backfill --days N]", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
