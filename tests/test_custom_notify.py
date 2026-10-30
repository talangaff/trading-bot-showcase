"""Custom bots (only) message Telegram when they enter and exit a position,
and the exit message carries the profit or loss."""

import pytest

from webapp import auth, bots, genome, notify


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(notify, "enabled", lambda: True)
    monkeypatch.setattr(notify, "send", lambda text, buttons=None: out.append(text) or True)
    monkeypatch.setattr(bots, "_dispatch", lambda fn, *a: fn(*a))
    monkeypatch.setattr(auth, "primary_admin_id", lambda: 1)
    return out


C = "SPY260904C00560000"
TS = "2026-06-02T15:00:00+00:00"


def _custom(user=1):
    return bots.create_bot("Scalper", genome=genome.random_genome(), pool="custom", user_id=user)


def test_entry_and_winning_exit_are_reported(sent):
    b = _custom()
    assert bots.record_trade(b, C, "buy_to_open", 3, 1.50, ts=TS, forced=True) is None
    assert bots.record_trade(b, C, "sell_to_close", 3, 2.00, ts=TS, forced=True) is None
    assert len(sent) == 2
    assert "opened" in sent[0] and "Bot: Scalper" in sent[0] and "Account: none assigned" in sent[0] and "SPY" in sent[0] and "$1.50" in sent[0]
    assert "closed" in sent[1] and "Bot: Scalper" in sent[1] and "P/L: +$150.00" in sent[1] and "+33.3%" in sent[1]


def test_losing_exit_shows_the_loss(sent):
    b = _custom()
    bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True)
    bots.record_trade(b, C, "sell_to_close", 2, 0.60, ts=TS, forced=True)
    assert "P/L: -$80.00" in sent[1] and "-40.0%" in sent[1]


def test_partial_exit_uses_the_average_entry_price(sent):
    b = _custom()
    bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True)
    bots.record_trade(b, C, "buy_to_open", 2, 2.00, ts=TS, forced=True)     # avg 1.50
    bots.record_trade(b, C, "sell_to_close", 2, 2.00, ts=TS, forced=True)
    assert "P/L: +$100.00" in sent[2]


def test_fleet_sim_and_other_accounts_stay_silent(sent):
    fleet = bots.create_bot("f", genome=genome.random_genome(), user_id=1)
    sim = bots.create_bot("s", genome=genome.random_genome(), pool="sim", user_id=1)
    other = _custom(user=2)
    for b in (fleet, sim, other):
        bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True)
        bots.record_trade(b, C, "sell_to_close", 2, 1.20, ts=TS, forced=True)
    assert sent == []


def test_a_telegram_failure_never_breaks_the_fill(monkeypatch):
    monkeypatch.setattr(notify, "enabled", lambda: True)
    monkeypatch.setattr(auth, "primary_admin_id", lambda: 1)
    def boom(*a):
        raise RuntimeError("telegram down")
    monkeypatch.setattr(bots, "_dispatch", boom)
    b = _custom()
    assert bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True) is None
    assert bots._positions(b)


def test_drawdown_force_close_is_reported_with_its_reason(sent):
    b = _custom()
    bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True)
    bots._flag_failing(b)
    assert any("closed" in m and "drawdown" in m for m in sent)


def test_messages_name_the_bot_and_its_account(sent, monkeypatch):
    from webapp import accounts
    monkeypatch.setattr(accounts, "get", lambda uid, aid: {"name": "Paper", "type": "paper"})
    b = _custom()
    with bots._conn() as c:
        c.execute("UPDATE bots SET account_id=3 WHERE id=?", (b,))
    bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True)
    bots.record_trade(b, C, "sell_to_close", 2, 1.20, ts=TS, forced=True)
    assert all("Bot: Scalper" in m and "Account: Paper (paper)" in m for m in sent) and len(sent) == 2


def test_emojis_blue_open_green_win_red_loss(sent):
    b = _custom()
    bots.record_trade(b, C, "buy_to_open", 2, 1.00, ts=TS, forced=True)
    bots.record_trade(b, C, "sell_to_close", 1, 1.50, ts=TS, forced=True)
    bots.record_trade(b, C, "sell_to_close", 1, 0.50, ts=TS, forced=True)
    assert sent[0].startswith("\U0001F535") and sent[1].startswith("\U0001F7E2") \
        and sent[2].startswith("\U0001F534")
