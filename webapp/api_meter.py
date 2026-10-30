"""How hard each Alpaca account is being worked, per minute.

Alpaca rate-limits per API key, and meters its two APIs separately:

  * trading  — paper-api.alpaca.markets: orders, fills, positions, account,
               the options-contracts reference. Every bot entry is at least
               one POST plus a poll per second until it fills, so this is
               the one that scales with bot count.
  * data     — data.alpaca.markets: 1-minute bars and option-chain
               snapshots. Fetched ONCE per cycle for the whole
               installation (bot_settings.data_source), so it scales with
               symbols and cycles, not with how many bots anyone runs.

Counted per (account, api, minute) in SQLite rather than in memory,
because the engine cycle is a different PROCESS from the web app — an
in-process counter on the web side would miss essentially every call the
bots make. The table is the only place both can see.

What it is for: the peak minute tells you the headroom before adding bots.
`stats()` reports peak, when it happened, and the fleet size at the time,
so calls-per-bot-per-minute can be extrapolated to a bigger fleet instead
of guessed at.

Metering must never break a trade: every function here swallows its own
errors and returns something harmless.
"""

from datetime import datetime, timedelta, timezone

# hostname -> which of Alpaca's two separately-metered APIs it belongs to
_TRADING_HOST = "paper-api.alpaca.markets"
_LIVE_TRADING_HOST = "//api.alpaca.markets"
_DATA_HOST = "data.alpaca.markets"
APIS = ("trading", "data")

# rows older than this are pruned on read — enough to see a busy week
KEEP_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    account TEXT NOT NULL,          -- Alpaca key ID the call was signed with
    api     TEXT NOT NULL,          -- 'trading' | 'data'
    minute  TEXT NOT NULL,          -- UTC minute, e.g. 2026-09-08T20:14
    n       INTEGER NOT NULL DEFAULT 0,
    bots    INTEGER NOT NULL DEFAULT 0,   -- live bots on that account at the time
    PRIMARY KEY (account, api, minute)
);
CREATE INDEX IF NOT EXISTS ix_api_calls_peak ON api_calls(account, api, n);
"""


def classify(url: str) -> str | None:
    """Which meter a URL counts against, or None if it isn't Alpaca."""
    if _TRADING_HOST in url or _LIVE_TRADING_HOST in url:
        return "trading"
    if _DATA_HOST in url:
        return "data"
    return None


def _minute(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")


# --- live bot count per account, cached: one query per minute, not per call
_BOTS: dict = {"ts": "", "by_account": {}}


def _bots_on(account: str) -> int:
    minute = _minute()
    if _BOTS["ts"] != minute:
        try:
            from . import account_book
            # accounts() is {key id: (keys, [bots])} — count the bots, not the pair
            _BOTS["by_account"] = {ak: len(bl) for ak, (_k, bl) in account_book.accounts().items()}
        except Exception:                          # noqa: BLE001 — never block a call
            _BOTS["by_account"] = {}
        _BOTS["ts"] = minute
    return int(_BOTS["by_account"].get(account, 0))


def record(keys, url: str, now: datetime | None = None) -> None:
    """Count one outbound Alpaca request. Best effort, never raises."""
    api = classify(url)
    account = keys[0] if keys and keys[0] else None
    if not api or not account:
        return
    try:
        from . import bots
        with bots._conn() as c:
            c.execute(
                "INSERT INTO api_calls(account, api, minute, n, bots) VALUES (?,?,?,1,?) "
                "ON CONFLICT(account, api, minute) DO UPDATE SET "
                "  n = n + 1, bots = MAX(bots, excluded.bots)",
                (account, api, _minute(now), _bots_on(account)))
    except Exception as e:      # noqa: BLE001 — a broken meter must never cost a trade
        print("api_meter record: {}".format(e), flush=True)


def request(method: str, url: str, *, keys=None, **kw):
    """`requests.request`, metered. Every Alpaca call in the app goes
    through here so a new call site cannot silently escape the count."""
    import requests
    record(keys, url)
    return requests.request(method, url, **kw)


def get(url: str, *, keys=None, **kw):
    return request("GET", url, keys=keys, **kw)


def post(url: str, *, keys=None, **kw):
    return request("POST", url, keys=keys, **kw)


def delete(url: str, *, keys=None, **kw):
    return request("DELETE", url, keys=keys, **kw)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _prune(c) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M")
    c.execute("DELETE FROM api_calls WHERE minute < ?", (cutoff,))


def stats(keys, *, prune: bool = True) -> dict:
    """{api: {peak, peak_at, peak_bots, per_bot, now, last, total, minutes}}
    for one account, plus the configured limit and the headroom left.

    `per_bot` is the number that answers "can I run more bots?": calls per
    minute per bot at the busiest minute observed. Multiply it by a target
    fleet size to see whether that would fit under the limit. It is only
    meaningful for the trading API — data calls are made once per cycle for
    the whole installation regardless of anybody's bot count."""
    account = keys[0] if keys and keys[0] else None
    out = {"account": account, "limit": _limit(), "apis": {}}
    if not account:
        return out
    try:
        from . import bots
        with bots._conn() as c:
            if prune:
                _prune(c)
            rows = c.execute(
                "SELECT api, MAX(n) peak, COUNT(*) minutes, SUM(n) total FROM api_calls "
                "WHERE account=? GROUP BY api", (account,)).fetchall()
            summary = {r["api"]: dict(r) for r in rows}
            for api in APIS:
                s = summary.get(api)
                if not s:
                    out["apis"][api] = _empty(api)
                    continue
                top = c.execute(
                    "SELECT minute, n, bots FROM api_calls WHERE account=? AND api=? "
                    "ORDER BY n DESC, minute DESC LIMIT 1", (account, api)).fetchone()
                cur = c.execute(
                    "SELECT n FROM api_calls WHERE account=? AND api=? AND minute=?",
                    (account, api, _minute())).fetchone()
                prev = c.execute(
                    "SELECT n FROM api_calls WHERE account=? AND api=? AND minute=?",
                    (account, api,
                     _minute(datetime.now(timezone.utc) - timedelta(minutes=1)))).fetchone()
                peak, peak_bots = int(top["n"]), int(top["bots"] or 0)
                out["apis"][api] = {
                    "api": api,
                    "peak": peak,
                    "peak_at": top["minute"],
                    "peak_bots": peak_bots,
                    "per_bot": (peak / peak_bots) if peak_bots else None,
                    "now": int(cur["n"]) if cur else 0,
                    "last": int(prev["n"]) if prev else 0,
                    "total": int(s["total"] or 0),
                    "minutes": int(s["minutes"] or 0),
                    "headroom": max(0, _limit() - peak),
                    "pct_of_limit": (peak / _limit() * 100.0) if _limit() else None,
                    # what the peak implies for a bigger fleet: at the same
                    # calls-per-bot rate, the bot count that would fill the
                    # limit. None until a peak with a known fleet size exists.
                    "bots_at_limit": (int(_limit() / (peak / peak_bots))
                                      if peak_bots and peak else None),
                }
    except Exception as e:      # noqa: BLE001 — a page must render without the meter
        print("api_meter stats: {}".format(e), flush=True)
    for api in APIS:
        out["apis"].setdefault(api, _empty(api))
    out["busiest"] = max(out["apis"].values(), key=lambda d: d["peak"])
    return out


def _empty(api: str) -> dict:
    return {"api": api, "peak": 0, "peak_at": None, "peak_bots": 0, "per_bot": None,
            "now": 0, "last": 0, "total": 0, "minutes": 0, "bots_at_limit": None,
            "headroom": _limit(), "pct_of_limit": 0.0}


def _limit() -> int:
    from . import config
    return int(getattr(config, "ALPACA_RATE_LIMIT_PER_MIN", 200))


def headroom_bots(keys, target_per_bot: float | None = None) -> int | None:
    """How many bots the trading limit would allow at the observed
    calls-per-bot rate. None until a peak with a known fleet size exists."""
    t = stats(keys, prune=False)["apis"]["trading"]
    rate = target_per_bot if target_per_bot is not None else t["per_bot"]
    if not rate:
        return None
    return int(_limit() / rate)


def reset(keys=None) -> None:
    """Forget the history for one account (or all). For after a change that
    makes the old peak meaningless — a different fleet size, new keys."""
    try:
        from . import bots
        with bots._conn() as c:
            if keys and keys[0]:
                c.execute("DELETE FROM api_calls WHERE account=?", (keys[0],))
            else:
                c.execute("DELETE FROM api_calls")
    except Exception as e:      # noqa: BLE001
        print("api_meter reset: {}".format(e), flush=True)
