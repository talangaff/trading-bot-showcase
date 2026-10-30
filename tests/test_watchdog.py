"""The engine is watched from outside itself.

webapp/watchdog.py exists because the in-cycle alarm (test_engine_idle.py)
has one blind spot it can never cover: it is code inside the cycle, so it
cannot fire when the cycle is not running. A disabled timer, a service
crash-looping on an import error, a hung box — all silence, and silence is
what three and a half hours of unmanaged positions looked like on
2026-09-09.

The cycle stamps a heartbeat into the database; the web app, which is a
different and longer-lived process, reads it from the other side.
"""

from datetime import datetime, timedelta, timezone

import pytest

from webapp import bot_engine, bot_settings, bots, notify, option_chain, watchdog

ET = bot_engine.ET


@pytest.fixture(autouse=True)
def engine_on(monkeypatch):
    monkeypatch.setattr(bot_settings, "engine_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(notify, "enabled", lambda: True)
    monkeypatch.setattr(notify, "send", lambda text, buttons=None: out.append(text) or True)
    return out


def _beat_at(when):
    bots._meta_set(watchdog.HEARTBEAT_KEY, when.isoformat(timespec="seconds"))


# ---------------------------------------------------------------------------
# the blind spot the in-cycle alarm cannot cover
# ---------------------------------------------------------------------------
def test_a_cycle_that_never_runs_is_caught():
    """The failure mode that went unnoticed: nothing is running, so nothing
    inside the engine is in a position to complain."""
    _beat_at(datetime.now(timezone.utc) - timedelta(minutes=42))
    problem = watchdog.check()
    assert problem and "has not completed a cycle in 42 minutes" in problem


def test_a_running_engine_is_quiet():
    _beat_at(datetime.now(timezone.utc) - timedelta(seconds=30))
    assert watchdog.check() is None


def test_an_engine_that_has_never_run_is_caught():
    """A fresh box where the timer was never enabled must not read as fine
    just because there is no heartbeat to be stale."""
    assert watchdog.check() == "the engine has never recorded a cycle"


def test_the_alarm_names_what_is_actually_at_stake(sent):
    _beat_at(datetime.now(timezone.utc) - timedelta(minutes=20))
    watchdog._alert(watchdog.check())
    assert len(sent) == 1
    # not "no trades" — the part a fleet owner needs to act on
    assert "open position is being managed" in sent[0]
    assert any(e["kind"] == "engine_down" for e in bots.recent_events(20))


def test_an_hour_down_is_one_message(sent):
    _beat_at(datetime.now(timezone.utc) - timedelta(minutes=20))
    for _ in range(30):
        watchdog._alert(watchdog.check())
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# things that are not outages
# ---------------------------------------------------------------------------
def test_a_closed_market_is_not_an_outage(monkeypatch):
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "closed")
    assert watchdog.check() is None            # no heartbeat at all, still fine


def test_an_engine_switched_off_on_purpose_is_not_an_outage(monkeypatch):
    monkeypatch.setattr(bot_settings, "engine_enabled", lambda: False)
    _beat_at(datetime.now(timezone.utc) - timedelta(hours=5))
    assert watchdog.check() is None


# ---------------------------------------------------------------------------
# the heartbeat itself
# ---------------------------------------------------------------------------
def test_the_heartbeat_is_liveness_not_success():
    """A cycle that crashed still ran. If a throwing cycle stopped beating,
    a crash-looping engine would look exactly like a dead one — and the two
    need different fixes."""
    watchdog.beat(0, "open", "boom")
    assert watchdog.check() is None
    assert bots._meta_get(watchdog.STEPPED_KEY) == "0"


def test_run_cycle_beats_even_when_it_raises(monkeypatch):
    """Wired into run_cycle's finally, not its happy path."""
    from webapp import account_book
    monkeypatch.setattr(bot_engine, "prepare", lambda *a, **k: 1)
    monkeypatch.setattr(bot_engine, "_recover_pending",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    bots._meta_set(watchdog.HEARTBEAT_KEY, "")
    with pytest.raises(RuntimeError):
        bot_engine.run_cycle()
    assert watchdog.last_beat() is not None


def test_a_normal_cycle_records_what_it_moved(monkeypatch):
    from webapp import account_book
    monkeypatch.setattr(bot_engine, "prepare", lambda *a, **k: 1)
    monkeypatch.setattr(bot_engine, "_recover_pending", lambda *a, **k: 0)
    monkeypatch.setattr(bot_engine, "_settle_expired", lambda *a, **k: 0)
    monkeypatch.setattr(account_book, "sync_all", lambda *a, **k: {})
    monkeypatch.setattr(bot_engine, "trading_users", lambda: [1])
    monkeypatch.setattr(bot_settings, "fleet_engine_enabled", lambda uid: False)
    monkeypatch.setattr(bot_engine, "CYCLE_BUDGET_S", -1.0)
    monkeypatch.setattr(bot_engine, "_tick",
                        lambda now, users=None, **kw: {"stepped": 99, "idle": None})
    bot_engine.run_cycle()
    assert bots._meta_get(watchdog.STEPPED_KEY) == "99"
    assert watchdog.last_beat() is not None


def test_start_is_idempotent():
    """The startup hook must not stack a thread per worker or per reload."""
    watchdog._started = False
    assert watchdog.start() is True
    assert watchdog.start() is False
