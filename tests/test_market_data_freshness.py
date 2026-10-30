"""A market-data provider that answers with stale bars must not be treated
as a working provider.

bot_engine._market_context gates every tick on bar freshness and returns
None when nothing is fresh — which aborts the tick for EVERY account, so no
bot enters and, worse, no open position gets its stop, trail or take-profit
managed. On 2026-09-09 the Alpaca IEX feed froze at 11:27 ET while still
returning a full 120 bars on every call. recent_bars only fell back to
yfinance `if not bars`, so the stale list was cached and returned all
afternoon and the fleet sat dead for three and a half hours holding 23 live
positions.
"""

from datetime import datetime, timedelta, timezone

import pytest

from webapp import bot_engine, market_data


def _bars(last_ts: datetime, n=120, close=100.0, z=False):
    out = []
    for i in range(n):
        t = last_ts - timedelta(minutes=n - 1 - i)
        stamp = (t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if z
                 else t.isoformat())
        out.append({"t": stamp, "o": close, "h": close, "l": close, "c": close, "v": 1000})
    return out


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    market_data._CACHE.clear()
    monkeypatch.setattr(market_data.bot_settings, "data_source",
                        lambda: {"user_id": 1, "keys": ("K", "S"), "feed": "iex"})
    yield
    market_data._CACHE.clear()


def test_a_frozen_feed_is_not_a_working_feed(monkeypatch):
    """The regression: the primary returns a FULL list of hours-old bars.
    That is a miss, not a result — the fresh provider must win."""
    now = datetime.now(timezone.utc)
    frozen = _bars(now - timedelta(hours=3), z=True)
    live = _bars(now)
    monkeypatch.setattr(market_data, "_from_alpaca", lambda *a, **k: frozen)
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: live)

    bars = market_data.recent_bars("SPY")
    assert bars is live
    # and the engine's own gate — the thing that was aborting every tick — passes
    assert bot_engine._fresh(bars, datetime.now(market_data.ET)) is True
    assert bot_engine._fresh(frozen, datetime.now(market_data.ET)) is False


def test_a_fresh_primary_is_still_preferred_and_the_fallback_is_not_called():
    """The normal path is untouched: a current primary is used as-is, and
    the second provider is never even asked."""
    now = datetime.now(timezone.utc)
    fresh = _bars(now, z=True)
    called = []

    market_data._CACHE.clear()
    import unittest.mock as m
    with m.patch.object(market_data, "_from_alpaca", lambda *a, **k: fresh), \
         m.patch.object(market_data, "_from_yfinance",
                        lambda *a, **k: called.append(1) or []):
        assert market_data.recent_bars("SPY") is fresh
    assert called == []


def test_the_stale_one_is_kept_when_the_fallback_is_no_better(monkeypatch):
    """Outside market hours everything is legitimately old. The comparison
    must then keep the NEWER of the two, never blank the data out."""
    now = datetime.now(timezone.utc)
    older = _bars(now - timedelta(hours=20), z=True)
    newer = _bars(now - timedelta(hours=14))
    monkeypatch.setattr(market_data, "_from_alpaca", lambda *a, **k: older)
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: newer)
    assert market_data.recent_bars("SPY") is newer

    market_data._CACHE.clear()
    monkeypatch.setattr(market_data, "_from_alpaca", lambda *a, **k: newer)
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: older)
    assert market_data.recent_bars("SPY") is newer      # primary already newer

    market_data._CACHE.clear()
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: [])
    assert market_data.recent_bars("SPY") is newer      # nothing better on offer


def test_an_unparseable_stamp_keeps_its_benefit_of_the_doubt(monkeypatch):
    """_fresh() already treats a stamp it cannot read as fresh. _stale must
    agree, or a provider changing its timestamp format would silently become
    a dead feed — the exact failure this change exists to prevent."""
    junk = [{"t": "not-a-timestamp", "c": 100.0}] * 5
    assert market_data._stale(junk) is False
    assert market_data._last_ts(junk) is None
    monkeypatch.setattr(market_data, "_from_alpaca", lambda *a, **k: junk)
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: _bars(
        datetime.now(timezone.utc)))
    assert market_data.recent_bars("SPY") is junk


# ---------------------------------------------------------------------------
# why the feed looked frozen in the first place
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, bars):
        self._bars = bars

    def raise_for_status(self):
        pass

    def json(self):
        return {"bars": self._bars}


def test_the_bars_request_asks_for_the_NEWEST_window(monkeypatch):
    """The root cause of the freeze, pinned down.

    With sort=asc and no explicit start, Alpaca anchors at the earliest data
    in the window and `limit` truncates from the front — so the call returns
    the FIRST N bars of the session and its newest bar stops advancing once
    the session is N minutes old. It is a 200 with a full payload, which is
    why nothing looked wrong. Ask for newest-first instead.
    """
    seen = {}

    def fake_get(url, keys=None, params=None, headers=None, timeout=None):
        seen.update(params)
        # newest-first, the way the API answers sort=desc
        return _Resp([{"t": "2026-09-09T19:3{}:00Z".format(9 - i), "o": 1.0,
                       "h": 1.0, "l": 1.0, "c": 1.0, "v": 1} for i in range(5)])

    monkeypatch.setattr(market_data.api_meter, "get", fake_get)
    bars = market_data._from_alpaca("SPY", 120, "K", "S", "iex")

    assert seen["sort"] == "desc", seen
    # and the caller still gets them oldest-first, since everything downstream
    # reads bars[-1] as "now"
    assert [b["t"] for b in bars] == sorted(b["t"] for b in bars)
    assert bars[-1]["t"] == "2026-09-09T19:39:00Z"


# ---------------------------------------------------------------------------
# a provider that THROWS is the same event as one that returns stale bars
# ---------------------------------------------------------------------------
# 2026-09-11: Alpaca 504'd on every bars call for 48 minutes. The fallback
# lived after the Alpaca call inside one shared try, so `raise_for_status()`
# jumped straight over it into the except and yfinance — healthy the whole
# time — was never asked. `stepped: 0` on every cycle from 12:16 to 13:04,
# ~17 open positions with no stop, trail or take-profit management. The
# stale-feed fix of 2026-09-09 covered the returns-bad-data half of this and
# left the raises half wide open.
class _Boom(Exception):
    pass


def test_a_throwing_primary_still_reaches_the_fallback(monkeypatch):
    """THE regression. Alpaca raising must not cost us yfinance."""
    now = datetime.now(timezone.utc)
    live = _bars(now)

    def explode(*a, **k):
        raise _Boom("504 Server Error: Gateway Timeout")

    monkeypatch.setattr(market_data, "_from_alpaca", explode)
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: live)

    bars = market_data.recent_bars("SPY")
    assert bars is live
    # and the gate that aborted every tick that day now passes
    assert bot_engine._fresh(bars, datetime.now(market_data.ET)) is True


def test_a_throwing_fallback_cannot_discard_a_working_primary(monkeypatch):
    """The mirror image: the backup blowing up must not blank out bars we
    already hold, even stale ones — losing data is the expensive direction."""
    now = datetime.now(timezone.utc)
    stale = _bars(now - timedelta(hours=3), z=True)

    def explode(*a, **k):
        raise _Boom("yfinance is having a day")

    monkeypatch.setattr(market_data, "_from_alpaca", lambda *a, **k: stale)
    monkeypatch.setattr(market_data, "_from_yfinance", explode)
    assert market_data.recent_bars("SPY") is stale


def test_every_provider_down_falls_back_to_the_last_good_bars(monkeypatch):
    """With both providers throwing, serve the last good answer rather than
    an empty list. Note this only helps a long-lived process: the engine is
    Type=oneshot, so its _CACHE is empty every cycle and this pays out for
    the web app, never for the engine — which is exactly why the fallback
    above has to do the real work."""
    now = datetime.now(timezone.utc)
    good = _bars(now)

    monkeypatch.setattr(market_data, "_from_alpaca", lambda *a, **k: good)
    monkeypatch.setattr(market_data, "_from_yfinance", lambda *a, **k: [])
    assert market_data.recent_bars("SPY") is good       # warms the cache

    def explode(*a, **k):
        raise _Boom("everything is down")

    monkeypatch.setattr(market_data, "_from_alpaca", explode)
    monkeypatch.setattr(market_data, "_from_yfinance", explode)
    market_data._CACHE["SPY"] = (0.0, good)             # cached, but expired
    assert market_data.recent_bars("SPY") is good


def test_a_dead_provider_is_reported_on_its_own_terms(monkeypatch):
    """96 logged 504s raised no alarm of their own that day — only the
    downstream idle did. A provider going dark gets its own event."""
    logged = []

    def explode(*a, **k):
        raise _Boom("504 Server Error: Gateway Timeout")

    monkeypatch.setattr(market_data, "_from_alpaca", explode)
    monkeypatch.setattr(market_data, "_from_yfinance",
                        lambda *a, **k: _bars(datetime.now(timezone.utc)))
    from webapp import bots
    monkeypatch.setattr(bots, "log_event_once",
                        lambda kind, detail, key, within_s=0: logged.append((kind, detail["provider"], key)) or True)

    market_data.recent_bars("SPY")
    assert logged == [("data_provider_error", "alpaca", "data:alpaca:SPY")]


def test_reporting_a_failure_never_becomes_one(monkeypatch):
    """If the event log itself is broken, the data path must not care."""
    def explode(*a, **k):
        raise _Boom("504")

    monkeypatch.setattr(market_data, "_from_alpaca", explode)
    monkeypatch.setattr(market_data, "_from_yfinance",
                        lambda *a, **k: _bars(datetime.now(timezone.utc)))
    from webapp import bots
    monkeypatch.setattr(bots, "log_event_once",
                        lambda *a, **k: (_ for _ in ()).throw(_Boom("db locked")))
    assert market_data.recent_bars("SPY")       # still served
