"""Server-side 'This Week' cache.

The screened results for each of the next N days are stored on disk so every
client gets them instantly. A background refresh (hourly systemd timer, or the
'Update All' button) repopulates the cache. An exclusive file lock makes sure
only one refresh runs at a time across processes.
"""

import errno
import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import engine
from . import config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


WEEK_SCAN_LIMIT = 45   # never look more than this many days ahead


def _probe_earning_days() -> list[str]:
    """Forward-scan from today and return the first WEEK_DAYS dates that actually
    have earnings on the Nasdaq calendar (weekends / holidays / empty days are
    skipped). A date whose fetch fails is kept, not skipped, so the failure is
    visible rather than silently dropping a day."""
    base = _today().date()
    out: list[str] = []
    i = 0
    while len(out) < config.WEEK_DAYS and i < WEEK_SCAN_LIMIT:
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        i += 1
        try:
            listing = engine._retry_on_rate_limit(lambda d=d: engine.fetch_nasdaq_earnings(d))
        except Exception:  # noqa: BLE001
            out.append(d)
            continue
        if listing:
            out.append(d)
    return out


def displayed_dates() -> list[str]:
    """The dates to show as tabs — taken from the cache (no network). If the
    cache holds fewer than WEEK_DAYS future days, pad forward with weekdays so
    there are always WEEK_DAYS tabs until the next refresh fills them in."""
    today = _today().strftime("%Y-%m-%d")
    dates = sorted(d for d in load_cache() if d >= today)[: config.WEEK_DAYS]
    if len(dates) < config.WEEK_DAYS:
        base = _today().date()
        i = 0
        while len(dates) < config.WEEK_DAYS and i < WEEK_SCAN_LIMIT:
            dt = base + timedelta(days=i)
            i += 1
            d = dt.strftime("%Y-%m-%d")
            if dt.weekday() < 5 and d not in dates:   # Mon–Fri
                dates.append(d)
        dates = sorted(dates)
    return dates


# ---------------------------------------------------------------------------
# cache file I/O (atomic writes, tolerant reads, prunes past dates)
# ---------------------------------------------------------------------------
def _read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.replace(tmp, path)


def load_cache() -> dict:
    cache = _read_json(config.WEEK_CACHE_FILE, {})
    today = _today().strftime("%Y-%m-%d")
    pruned = {d: v for d, v in cache.items() if d >= today}
    if pruned != cache:
        _write_json(config.WEEK_CACHE_FILE, pruned)
    return pruned


# ---------------------------------------------------------------------------
# manual BMO / AMC overrides (for symbols Nasdaq didn't tag)
# ---------------------------------------------------------------------------
SESSION_OVERRIDE_FILE = config.DATA_DIR / "session_overrides.json"


def session_overrides() -> dict:
    ov = _read_json(SESSION_OVERRIDE_FILE, {})
    today = _today().strftime("%Y-%m-%d")
    pruned = {d: v for d, v in ov.items() if d >= today}
    if pruned != ov:
        _write_json(SESSION_OVERRIDE_FILE, pruned)
    return pruned


def set_session_override(date_str: str, symbol: str, session: str) -> None:
    if session not in ("BMO", "AMC"):
        return
    ov = _read_json(SESSION_OVERRIDE_FILE, {})
    ov.setdefault(date_str, {})[symbol.strip().upper()] = session
    _write_json(SESSION_OVERRIDE_FILE, ov)


def clear_session_override(date_str: str, symbol: str) -> None:
    ov = _read_json(SESSION_OVERRIDE_FILE, {})
    day = ov.get(date_str) or {}
    if day.pop(symbol.strip().upper(), None) is not None:
        if not day:
            ov.pop(date_str, None)
        _write_json(SESSION_OVERRIDE_FILE, ov)


def apply_session_overrides(date_str: str, rows: list) -> list:
    """Return rows with `session` replaced by any manual override for that date;
    overridden rows are tagged session_manual=True."""
    ov = session_overrides().get(date_str, {})
    if not ov or not rows:
        return rows
    out = []
    for r in rows:
        if r.get("symbol") in ov:
            r = dict(r)
            r["session"] = ov[r["symbol"]]
            r["session_manual"] = True
        out.append(r)
    return out


def apply_passed(date_str: str, rows: list) -> list:
    """Mark 'Passed' any symbol whose earnings were released BMO today — the
    event is over, so it is no longer Recommended / Consider / Avoid.

    Run this AFTER apply_session_overrides so manual BMO tags count too.
    """
    if not rows or date_str != _today().strftime("%Y-%m-%d"):
        return rows
    out = []
    for r in rows:
        if r.get("session") == "BMO" and r.get("rating") not in ("Error", "Passed"):
            r = dict(r)
            r["rating"] = "Passed"
            r["avg_volume"] = r["iv30_rv30"] = r["ts_slope_0_45"] = "—"
        out.append(r)
    return out


def prepare_rows(date_str: str, rows: list) -> list:
    """Full display pipeline for a day's cached rows: session overrides ->
    Passed marking -> re-sort."""
    if not rows:
        return rows
    rows = apply_passed(date_str, apply_session_overrides(date_str, rows))
    return engine.sort_rows(rows)


def _store_day(date_str: str, rows: list, error: str | None = None) -> None:
    cache = _read_json(config.WEEK_CACHE_FILE, {})
    today = _today().strftime("%Y-%m-%d")
    cache = {d: v for d, v in cache.items() if d >= today}
    cache[date_str] = {
        "updated_at": _now_iso(),
        "rows": rows,
        "count": len(rows),
        "error": error,
    }
    _write_json(config.WEEK_CACHE_FILE, cache)


def get_day(date_str: str) -> dict | None:
    return load_cache().get(date_str)


# ---------------------------------------------------------------------------
# status file (so the UI can show refresh progress)
# ---------------------------------------------------------------------------
def get_status() -> dict:
    return _read_json(config.WEEK_STATUS_FILE, {"running": False})


def _set_status(**fields) -> None:
    status = _read_json(config.WEEK_STATUS_FILE, {})
    status.update(fields)
    _write_json(config.WEEK_STATUS_FILE, status)


# ---------------------------------------------------------------------------
# exclusive refresh lock (cross-process)
# ---------------------------------------------------------------------------
@contextmanager
def _refresh_lock():
    fh = open(config.WEEK_LOCK_FILE, "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES):
                raise RefreshBusy()
            raise
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


class RefreshBusy(RuntimeError):
    """Raised when a refresh is already running."""


def is_refreshing() -> bool:
    """True if the exclusive lock is currently held by someone."""
    try:
        fh = open(config.WEEK_LOCK_FILE, "w")
    except OSError:
        return False
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# the refresh itself
# ---------------------------------------------------------------------------
def _previous_details(date_str: str) -> dict:
    """symbol -> last known-good `detail` dict cached for this date, used so a
    freshly computed NaN doesn't overwrite a previously good value."""
    day = load_cache().get(date_str) or {}
    out = {}
    for row in day.get("rows") or []:
        detail = row.get("detail")
        if isinstance(detail, dict):
            out[row.get("symbol")] = detail
    return out


def _screen_one(date_str: str, top_n: int, use_cache: bool = True) -> tuple[list, str | None]:
    try:
        listing = engine._retry_on_rate_limit(lambda: engine.fetch_nasdaq_earnings(date_str))
    except Exception as e:  # noqa: BLE001
        return [], "Nasdaq fetch failed: {}".format(e)

    listing = listing[:top_n]
    if not listing:
        return [], None
    symbols = [x["symbol"] for x in listing]
    sessions = {x["symbol"]: x.get("session", "-") for x in listing}
    previous_details = _previous_details(date_str)
    rows = engine.compute_bulk(symbols, use_cache=use_cache, sessions=sessions,
                                previous_details=previous_details)
    return rows, None


def refresh_all(top_n: int | None = None, verbose: bool = False) -> dict:
    """Refresh every day in the window. Raises RefreshBusy if one is running."""
    top_n = top_n or config.WEEK_TOP_N

    with _refresh_lock():
        _set_status(running=True, started_at=_now_iso(), finished_at=None,
                    total=config.WEEK_DAYS, done=0, current_day="(finding earnings days)",
                    last_error=None, dates=displayed_dates())
        if verbose:
            print("{}  scanning Nasdaq calendar for earnings days".format(_now_iso()),
                  flush=True)
        dates = _probe_earning_days()
        if not dates:                       # total Nasdaq outage — keep old tabs
            dates = displayed_dates()
        _set_status(total=len(dates), current_day=None, dates=dates)
        old = load_cache()
        # keep showing already-cached days while they're re-screened
        cache = {d: old[d] for d in dates if d in old}
        errors = []
        for i, d in enumerate(dates):
            _set_status(current_day=d, done=i)
            if verbose:
                print("{}  screening {}".format(_now_iso(), d), flush=True)
            rows, err = _screen_one(d, top_n)
            cache[d] = {"updated_at": _now_iso(), "rows": rows,
                        "count": len(rows), "error": err}
            _write_json(config.WEEK_CACHE_FILE, cache)
            if err:
                errors.append("{}: {}".format(d, err))
            _set_status(done=i + 1)
        # final cache holds exactly the chosen days (stale future days dropped)
        _write_json(config.WEEK_CACHE_FILE, {d: cache[d] for d in dates})
        _set_status(running=False, current_day=None, finished_at=_now_iso(),
                    last_error="; ".join(errors) if errors else None)
    _notify_after_refresh()
    return get_status()


def refresh_day(date_str: str, top_n: int | None = None) -> dict:
    """Refresh a single day. Raises RefreshBusy if a refresh is running."""
    top_n = top_n or config.WEEK_TOP_N
    with _refresh_lock():
        _set_status(running=True, started_at=_now_iso(), finished_at=None,
                    total=1, done=0, current_day=date_str, last_error=None,
                    dates=[date_str])
        rows, err = _screen_one(date_str, top_n)
        _store_day(date_str, rows, err)
        _set_status(running=False, done=1, current_day=None,
                    finished_at=_now_iso(), last_error=err)
    _notify_after_refresh()
    return get_status()



def _next_market_day(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def near_market_dates() -> list[str]:
    """Today + the next market day — the two days that drive buy-today."""
    today = _today().date()
    return [today.strftime("%Y-%m-%d"),
            _next_market_day(today).strftime("%Y-%m-%d")]


def refresh_near(top_n: int | None = None) -> dict:
    """Refresh just today + the next market day with fresh Yahoo data (no
    per-symbol cache), merged into the existing cache. Runs frequently."""
    top_n = top_n or config.WEEK_TOP_N
    dates = near_market_dates()
    with _refresh_lock():
        _set_status(running=True, started_at=_now_iso(), finished_at=None,
                    total=len(dates), done=0, current_day=None, last_error=None,
                    dates=dates)
        errors = []
        for i, d in enumerate(dates):
            _set_status(current_day=d, done=i)
            rows, err = _screen_one(d, top_n, use_cache=False)
            _store_day(d, rows, err)
            if err:
                errors.append("{}: {}".format(d, err))
            _set_status(done=i + 1)
        _set_status(running=False, current_day=None, finished_at=_now_iso(),
                    last_error="; ".join(errors) if errors else None)
    _notify_after_refresh()
    return get_status()



def _notify_after_refresh() -> None:
    """Run the Telegram flip / no-time checks. Never let it break a refresh."""
    try:
        from . import notify
        notify.after_refresh()
    except Exception as e:  # noqa: BLE001
        print("[week] notify hook failed: {}".format(e), flush=True)
