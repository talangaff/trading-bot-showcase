"""A cycle that moves nothing while the market is open must say so.

On 2026-09-09 the fleet stopped trading at 11:27 ET and nobody found out
for three and a half hours. Nothing was broken in a way anything checked:
the service was up, /healthz returned 200, the journal was clean, and the
cycle log kept printing `phase: open, active: 308` every minute — because
`active` counts the bots that EXIST, not the ones the engine moved. Every
tick was returning early on stale market data before a single bot was
stepped, which also meant no stop, trail or take-profit ran on any open
position.

So the cycle now reports what it actually moved, and shouts when that is
nothing during market hours.
"""

from datetime import datetime

import pytest

from webapp import bot_engine, bots, notify, option_model

ET = option_model.ET


@pytest.fixture
def open_market(monkeypatch):
    """A cycle with the housekeeping (and Alpaca) stubbed out, market open."""
    from webapp import account_book, bot_settings
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")
    monkeypatch.setattr(bot_engine, "prepare", lambda *a, **k: 1)
    monkeypatch.setattr(bot_engine, "_recover_pending", lambda *a, **k: 0)
    monkeypatch.setattr(bot_engine, "_settle_expired", lambda *a, **k: 0)
    monkeypatch.setattr(account_book, "sync_all", lambda *a, **k: {})
    monkeypatch.setattr(bot_engine, "trading_users", lambda: [1])
    monkeypatch.setattr(bot_settings, "engine_enabled", lambda: True)
    monkeypatch.setattr(bot_settings, "fleet_engine_enabled", lambda uid: False)
    monkeypatch.setattr(bot_engine, "CYCLE_BUDGET_S", -1.0)


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(notify, "enabled", lambda: True)
    monkeypatch.setattr(notify, "send", lambda text, buttons=None: out.append(text) or True)
    return out


def test_a_cycle_that_steps_nothing_raises_the_alarm(open_market, sent, monkeypatch):
    """The exact 2026-09-09 shape: market open, tick aborts on stale data."""
    monkeypatch.setattr(bot_engine, "_tick",
                        lambda now, users=None, **kw: {"stepped": 0, "idle": "no fresh market data"})
    res = bot_engine.run_cycle()

    assert res["stepped"] == 0
    assert len(sent) == 1
    assert "not trading" in sent[0]
    assert "no fresh market data" in sent[0]
    # and it is on the permanent record, not only in a chat message
    assert any(e["kind"] == "engine_idle" for e in bots.recent_events(20))


def test_a_working_cycle_says_how_many_bots_it_moved_and_stays_quiet(open_market, sent, monkeypatch):
    monkeypatch.setattr(bot_engine, "_tick",
                        lambda now, users=None, **kw: {"stepped": 77, "idle": None})
    res = bot_engine.run_cycle()

    # one tick here: the fixture sets CYCLE_BUDGET_S negative so the cycle
    # does not sleep 15s between the four an open market would normally run
    assert res["stepped"] == 77
    assert sent == []
    assert not any(e["kind"] == "engine_idle" for e in bots.recent_events(20))


def test_the_alarm_does_not_repeat_every_minute(open_market, sent, monkeypatch):
    """A fleet down for an hour is one problem, not sixty messages. The
    cooldown lives in the DB on purpose: every cycle is its own short-lived
    process, so an in-process one would reset on each firing and spam."""
    monkeypatch.setattr(bot_engine, "_tick",
                        lambda now, users=None, **kw: {"stepped": 0, "idle": "no fresh market data"})
    for _ in range(5):
        bot_engine.run_cycle()
    assert len(sent) == 1


def test_a_closed_market_is_not_an_outage(open_market, sent, monkeypatch):
    """Nothing to trade is not the same as unable to trade."""
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "closed")
    bot_engine.run_cycle()
    assert sent == []


def test_the_tick_reports_a_stale_feed_rather_than_returning_silently(monkeypatch):
    """The unit underneath: _market_context returning None used to be a bare
    `return`, indistinguishable from a tick that had simply found nothing."""
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")
    monkeypatch.setattr(bot_engine, "_market_context", lambda now, phase, **kw: None)
    assert bot_engine._tick(datetime.now(ET)) == {"stepped": 0,
                                                  "idle": "no fresh market data"}

    monkeypatch.setattr(bot_engine, "_market_context", lambda now, phase, **kw: ({}, {}))
    monkeypatch.setattr(bot_engine, "trading_users", lambda: [])
    assert bot_engine._tick(datetime.now(ET)) == {"stepped": 0, "idle": "no bots to step"}
