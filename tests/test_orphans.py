"""Positions no bot claims are closed for real on the account (cautiously),
and a bot that is retired closes everything it holds first.

Alpaca is never called: broker.positions / submit_and_wait are stubbed.
"""

from datetime import datetime, timedelta

import pytest

from tests.test_live_execution import (FLEET, FakeAlpaca, _connect_fleet, _contract, _fleet_bot,
                                       _ts, isolated_settings)  # noqa: F401  (autouse fixture)
from webapp import account_book, bot_settings, bots, broker, option_model

ET = option_model.ET
# a future Monday: a real trading day that cannot be expired by the real clock
DAY = datetime(2026, 10, 19, tzinfo=ET)
NOW = DAY.replace(hour=11, minute=30)
EXP = DAY.replace(hour=16, minute=0)
TS = DAY.replace(hour=10, minute=30).isoformat()


@pytest.fixture
def acct(monkeypatch):
    """A fleet with one bot and a fake account whose holdings the test edits."""
    _connect_fleet()
    state: dict[str, float] = {}
    fake = FakeAlpaca()
    monkeypatch.setattr(broker, "positions", lambda keys=None: dict(state))
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    return state, fake


def _spread():
    return option_model.spread_symbol("SPY", EXP, False, 774.0, 770.0)


def _long_leg():
    return option_model.occ_symbol("SPY", EXP, False, 774.0)


def _short_leg():
    return option_model.occ_symbol("SPY", EXP, False, 770.0)


def _orphaned_short(acct):
    """A bot opens then books out a put spread while the account keeps the short leg."""
    state, fake = acct
    a = _fleet_bot()
    bots.record_trade(a["id"], _spread(), "buy_to_open", 2, 1.00, ts=TS, fees=0)
    bots.record_trade(a["id"], _spread(), "sell_to_close", 2, 1.00, ts=TS, fees=0)
    assert not bots._positions(a["id"])
    state[_short_leg()] = -2.0
    return a, state, fake


def test_an_unclaimed_short_leg_is_bought_back_but_only_once_it_has_persisted(acct):
    a, state, fake = _orphaned_short(acct)
    assert account_book.close_orphans(FLEET, [a], now=NOW) == 0       # first sighting: wait
    assert fake.orders == []
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=30)) == 0
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 1
    o = fake.orders[-1]
    assert (o["symbol"], o["side"], o["intent"], o["qty"]) == (_short_leg(), "buy", "buy_to_close", 2)
    assert o["keys"] == FLEET
    assert any(e["kind"] == "orphan_close" for e in bots.recent_events())


def test_an_unclaimed_long_is_sold_to_close(acct):
    state, fake = acct
    a = _fleet_bot()
    bots.record_trade(a["id"], _long_leg(), "buy_to_open", 3, 1.00, ts=TS, fees=0)
    bots.record_trade(a["id"], _long_leg(), "sell_to_close", 3, 1.00, ts=TS, fees=0)
    state[_long_leg()] = 3.0
    account_book.close_orphans(FLEET, [a], now=NOW)
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 1
    o = fake.orders[-1]
    assert (o["side"], o["intent"], o["qty"]) == ("sell", "sell_to_close", 3)


def test_a_claimed_position_is_never_touched(acct):
    state, fake = acct
    a = _fleet_bot()
    bots.record_trade(a["id"], _spread(), "buy_to_open", 2, 1.00, ts=TS, fees=0)
    state.update({_long_leg(): 2.0, _short_leg(): -2.0})
    account_book.close_orphans(FLEET, [a], now=NOW)
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 0
    assert fake.orders == []


def test_something_no_bot_ever_traded_is_left_alone(acct):
    state, fake = acct
    a = _fleet_bot()
    state[option_model.occ_symbol("QQQ", EXP, False, 480.0)] = -1.0     # yours, by hand
    account_book.close_orphans(FLEET, [a], now=NOW)
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 0
    assert fake.orders == []


@pytest.mark.parametrize("when", [
    DAY.replace(hour=9, minute=0),            # before the open
    DAY.replace(hour=16, minute=30),          # after the close
    DAY + timedelta(days=5, hours=11),        # a Saturday
])
def test_nothing_is_ordered_when_the_market_is_shut(acct, when):
    a, state, fake = _orphaned_short(acct)
    account_book.close_orphans(FLEET, [a], now=when)
    assert account_book.close_orphans(FLEET, [a], now=when + timedelta(seconds=70)) == 0
    assert fake.orders == []


def test_an_order_in_flight_blocks_the_sweep(acct):
    a, state, fake = _orphaned_short(acct)
    bots.add_pending(a["id"], _spread(), "buy_to_open", 1, "cid-1", symbol="SPY")
    account_book.close_orphans(FLEET, [a], now=NOW)
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 0
    assert fake.orders == []


def test_engine_off_means_no_orders(acct, monkeypatch):
    a, state, fake = _orphaned_short(acct)
    monkeypatch.setattr(bot_settings, "engine_enabled", lambda: False)
    account_book.close_orphans(FLEET, [a], now=NOW)
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 0
    assert fake.orders == []


def test_a_failed_order_is_reported_not_raised(acct):
    a, state, fake = _orphaned_short(acct)
    fake.fail = "Alpaca rejected the order (HTTP 403)"
    account_book.close_orphans(FLEET, [a], now=NOW)
    assert account_book.close_orphans(FLEET, [a], now=NOW + timedelta(seconds=70)) == 0
    ev = [e for e in bots.recent_events() if e["kind"] == "orphan_close"]
    assert ev and "failed" in str(ev[0])


# ---------------------------------------------------------------------------
# retiring a bot closes its positions
# ---------------------------------------------------------------------------
def _holding(acct):
    state, fake = acct
    a = _fleet_bot()
    c = _contract(days=10)
    bots.record_trade(a["id"], c, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
    state[c] = 2.0
    return a, c, fake


def test_retire_bot_closes_every_position_for_real(acct):
    a, c, fake = _holding(acct)
    bots.retire_bot(a["id"], "retired")
    assert bots.get_bot(a["id"])["status"] == "retired"
    assert not bots._positions(a["id"])
    assert fake.orders[-1]["intent"] == "sell_to_close" and fake.orders[-1]["qty"] == 2


def test_set_status_retired_also_closes(acct):
    a, c, fake = _holding(acct)
    bots.set_status(a["id"], "retired")
    assert not bots._positions(a["id"]) and fake.orders


def test_failing_is_not_a_retirement_and_does_not_flatten(acct):
    a, c, fake = _holding(acct)
    bots.retire_bot(a["id"], "failing")
    assert bots._positions(a["id"]) and fake.orders == []


def test_a_broker_failure_never_blocks_the_retirement(acct):
    a, c, fake = _holding(acct)
    fake.fail = "Alpaca rejected the order (HTTP 403)"
    bots.retire_bot(a["id"], "retired")
    assert bots.get_bot(a["id"])["status"] == "retired"
