"""A custom bot's entry budget is the smaller of the genome's risk_pct share
of its balance and the room left under its position cap. On a small account
that budget can be too small to buy even one contract of the strike the bot
wants — which used to skip the trade silently.

Two things guard that now: a `budget_too_small` engine event, and a per-bot
opt-in (`min_size_over_risk`) to buy the minimum size anyway as long as the
position cap covers it.
"""

from datetime import datetime, timedelta

from webapp import bot_engine, bots, genome, market_data, option_model, strategies

UID = 1
ET = option_model.ET
S = 560.0


def _flat_bars(symbol, minutes=120):
    now = datetime.now(ET)
    return [{"t": (now - timedelta(minutes=i)).isoformat(), "o": S, "h": S + 0.02,
             "l": S - 0.02, "c": S, "v": 1000} for i in range(minutes, 0, -1)]


def _small_custom_bot(monkeypatch, **overrides):
    """A $1,000 custom bot whose risk_pct (2%) funds ~$20 — far less than one
    ATM contract — under a 62.5% position cap ($625)."""
    g = genome.random_genome()
    g.update(style="0dte", symbol="SPY", signal_strength=0.15, risk_pct=2, min_option_volume=0,
             max_iv=0.9, strike_style="delta", target_delta=0.5, hold_max_min=60)
    bid = bots.spawn_bot(g, origin="seed", user_id=UID)
    with bots._conn() as c:
        c.execute("UPDATE bots SET pool='custom', start_balance=1000 WHERE id=?", (bid,))
    bots.update_guardrails(bid, {"position_pct_limit": 62.5, "min_contracts": 1, **overrides})
    monkeypatch.setattr(market_data, "recent_bars", _flat_bars)
    monkeypatch.setattr(strategies, "evaluate", lambda genome_, ctx: ("long", 1.0))
    return bid


def _step(bid):
    now = datetime(2026, 9, 8, 11, 0, tzinfo=ET)
    bars = {s: market_data.recent_bars(s) for s in genome.SYMBOLS}
    feats = {s: {} for s in genome.SYMBOLS}
    bot_engine._step_bot(bots.get_bot(bid), bars, feats, now, "open")


def _events(kind):
    return [e for e in bots.recent_events(50) if e["kind"] == kind]


def test_a_budget_too_small_for_one_contract_is_logged_not_silent(monkeypatch, modeled_fills):
    bid = _small_custom_bot(monkeypatch)
    _step(bid)
    assert not bots._positions(bid)
    ev = _events("budget_too_small")
    assert len(ev) == 1
    d = ev[0]["detail"] if isinstance(ev[0]["detail"], dict) else __import__("json").loads(ev[0]["detail"])
    assert d["bot"] == bid and d["cause"] == "risk_pct"
    assert d["one_contract_cost"] > d["risk_budget"] and d["one_contract_cost"] <= d["position_room"]


def test_the_log_is_throttled_per_bot(monkeypatch, modeled_fills):
    bid = _small_custom_bot(monkeypatch)
    for _ in range(3):
        _step(bid)
    assert len(_events("budget_too_small")) == 1


def test_opting_in_buys_the_minimum_size_within_the_cap(monkeypatch, modeled_fills):
    bid = _small_custom_bot(monkeypatch, min_size_over_risk=True)
    _step(bid)
    pos = bots._positions(bid)
    assert pos, "opted-in bot should buy the minimum size"
    assert sum(p["qty"] for p in pos.values()) == 1
    cap = bots.position_headroom(bid)
    assert cap["open_cost"] <= cap["cap"] + 1e-6
    assert "min size" in bots.trade_log(bid)[0]["note"]
    assert not _events("budget_too_small")


def test_opting_in_never_exceeds_the_position_cap(monkeypatch, modeled_fills):
    # a 1% cap ($10) cannot fund any contract: the override must not overspend
    bid = _small_custom_bot(monkeypatch, position_pct_limit=1, min_size_over_risk=True)
    _step(bid)
    assert not bots._positions(bid)
    ev = _events("budget_too_small")
    assert ev and "position_cap" in str(ev[0]["detail"])


def test_the_flag_round_trips_and_clears():
    g = genome.random_genome()
    bid = bots.spawn_bot(g, origin="seed", user_id=UID)
    with bots._conn() as c:
        c.execute("UPDATE bots SET pool='custom' WHERE id=?", (bid,))
    assert bots.effective_guardrails(bots.get_bot(bid))["min_size_over_risk"] is False
    bots.update_guardrails(bid, {"min_size_over_risk": True})
    assert bots.effective_guardrails(bots.get_bot(bid))["min_size_over_risk"] is True
    bots.update_guardrails(bid, {})
    assert bots.effective_guardrails(bots.get_bot(bid))["min_size_over_risk"] is False
