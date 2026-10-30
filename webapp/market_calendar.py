"""Best-effort "is today a macro data day?" flag for the bot fleet, plus the
real NYSE trading-day calendar.

0DTE index options behave very differently on FOMC / CPI / jobs days, so the
feature layer exposes this as a signal. It is a hardcoded list plus a couple
of monthly rules — good enough to bias behaviour, not a substitute for a real
economic calendar. Update FOMC_DATES once a year.

Trading-day-ness (is_market_day / market_days_before) is NOT hand-maintained
— it's backed by pandas_market_calendars' NYSE calendar, which tracks
holidays (including one-off/observed shifts) without us updating a date list
by hand every year.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as _mcal

_ET = ZoneInfo("America/New_York")

# Precomputed once per process into a plain dict for O(1) lookups — building
# the calendar is the expensive part (~0.2s), not querying a range. The full
# SCHEDULE costs the same as the bare valid_days list it replaced but carries
# each session's real open and close, which is what makes early-close half
# days (1pm, ~3 a year) visible instead of being traded as full sessions.
_NYSE = _mcal.get_calendar("NYSE")


def _build_sessions() -> dict:
    sched = _NYSE.schedule(start_date="2015-01-01", end_date="2035-12-31")
    out = {}
    for day, row in sched.iterrows():
        o = row["market_open"].tz_convert(_ET)
        c = row["market_close"].tz_convert(_ET)
        out[day.date()] = (o.hour * 60 + o.minute, c.hour * 60 + c.minute)
    return out


_SESSIONS = _build_sessions()
_TRADING_DAYS = set(_SESSIONS)

# The close assumed for a trading date outside the precomputed range. Never
# used to decide whether the market is OPEN — session_bounds answers None for
# a day it does not know, so nothing has to guess that.
REGULAR_CLOSE_MIN = 16 * 60

# FOMC decision days (the second day of each meeting), US Eastern.
FOMC_DATES = {
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (schedule as published; refresh when the Fed confirms)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
}

# Known high-impact releases that don't follow a clean rule (CPI/PPI shift a
# day or two month to month). Extend as the BLS calendar is published.
FIXED_DATA_DATES = {
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-13", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-13", "2026-11-13", "2026-12-10",
}


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    shift = (weekday - d.weekday()) % 7
    return date(y, m, 1 + shift + 7 * (n - 1))


def classify(d: date) -> tuple[float, str]:
    """Return (impact 0..1, label). 0 = ordinary session."""
    iso = d.isoformat()
    if iso in FOMC_DATES:
        return 1.0, "FOMC"
    if iso in FIXED_DATA_DATES:
        return 0.8, "CPI/PPI"
    # Non-farm payrolls: first Friday of the month
    if d == _nth_weekday(d.year, d.month, 4, 1):
        return 0.8, "NFP"
    # Weekly jobless claims: every Thursday (mild)
    if d.weekday() == 3:
        return 0.3, "claims"
    # Triple-witching: third Friday of Mar/Jun/Sep/Dec
    if d.month in (3, 6, 9, 12) and d == _nth_weekday(d.year, d.month, 4, 3):
        return 0.6, "OpEx"
    return 0.0, ""


def is_data_day(d: date | None = None) -> bool:
    return classify(d or date.today())[0] >= 0.5


def is_market_day(d: date) -> bool:
    """True if the NYSE is open for a regular session on d (early-close half
    days still count — they trade, just shorter)."""
    return d in _TRADING_DAYS


def market_days_before(d: date, n: int) -> date:
    """Walk back n market days from d (exclusive of d itself)."""
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if is_market_day(d):
            count += 1
    return d


def is_market_open_today() -> bool:
    """Is the NYSE open for a regular session today (US Eastern date)?"""
    return is_market_day(datetime.now(_ET).date())


def session_bounds(d: date) -> tuple[int, int] | None:
    """(open, close) for `d` as minutes since ET midnight, or None when the
    market is shut that day — a weekend, or one of the holidays a plain
    weekday check cannot see.

    Minutes-since-midnight rather than `time` objects because every caller is
    comparing against a clock position, and because it makes an early close
    arithmetic rather than a special case: 1pm is simply a smaller number.
    """
    return _SESSIONS.get(d)


def session_close_min(d: date) -> int:
    """When trading ends on `d`, in ET minutes since midnight. Falls back to
    the regular 16:00 for a date outside the precomputed calendar — callers
    that need to know whether the market is open at all ask session_bounds,
    which answers None rather than guessing."""
    b = _SESSIONS.get(d)
    return b[1] if b else REGULAR_CLOSE_MIN


def session_close_dt(d: date) -> datetime:
    """The ET-aware moment `d`'s session ends — 16:00 normally, 13:00 on an
    early-close half day. This is when a 0DTE contract on `d` expires, so it
    is what every minutes-to-expiry calculation has to measure against."""
    m = session_close_min(d)
    return datetime(d.year, d.month, d.day, m // 60, m % 60, tzinfo=_ET)


def is_early_close(d: date) -> bool:
    b = _SESSIONS.get(d)
    return bool(b) and b[1] < REGULAR_CLOSE_MIN
