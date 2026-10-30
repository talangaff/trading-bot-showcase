"""The fleet-wide entry cost gate (option_chain.cost_gate) is a hard rule,
not a gene: no entry unless the expected directional move over the planned
hold can pay the spread, the theta and the commission.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from webapp import bot_settings, bot_engine, bots, features, genome, market_data, option_chain, option_model, strategies

# every bot belongs to an account now (its fleet keys are what its real
# orders go out on) — one owner is enough for these mechanics tests
UID = 1

ET = option_model.ET
S = 560.0


def _row(delta_target, minutes_left, iv=0.15):
    chain = option_chain.build_chain("SPY", S, minutes_left, iv, True)
    return min(chain, key=lambda r: abs(abs(r["delta"]) - delta_target))


def test_atm_with_a_real_hold_passes_and_the_late_lottery_ticket_fails():
    ok = option_chain.cost_gate(_row(0.5, 300), S, 60, 300, 0.65)
    assert ok["ok"] and ok["ratio"] > 1.0
    bad = option_chain.cost_gate(_row(0.15, 45), S, 30, 45, 0.65)
    assert not bad["ok"] and bad["ratio"] < 1.0
    # a 10-minute hold can't outrun spread + commission on a cheap strike
    assert not option_chain.cost_gate(_row(0.15, 300), S, 10, 300, 0.65)["ok"]


def test_gate_components_are_all_charged():
    g = option_chain.cost_gate(_row(0.5, 300), S, 60, 300, 0.65)
    assert g["spread"] > 0 and g["theta"] > 0 and g["commission"] == 0.013
    assert abs(g["cost"] - (g["spread"] + g["theta"] + g["commission"])) < 1e-6
    # disabled by mult=0
    assert option_chain.cost_gate(_row(0.15, 45), S, 5, 45, 0.65, mult=0)["ok"]


def _flat_bars(symbol, minutes=120):
    now = datetime.now(ET)
    return [{"t": (now - timedelta(minutes=i)).isoformat(), "o": S, "h": S + 0.02,
             "l": S - 0.02, "c": S, "v": 1000} for i in range(minutes, 0, -1)]


def test_engine_refuses_an_entry_the_gate_rejects(monkeypatch, modeled_fills):
    g = genome.random_genome()
    g.update(style="0dte", symbol="SPY", signal_strength=0.15, risk_pct=8, min_option_volume=0,
             max_iv=0.9, strike_style="delta", target_delta=0.5, hold_max_min=60)
    bot_id = bots.spawn_bot(g, origin="seed", user_id=UID)
    bot = bots.get_bot(bot_id)
    monkeypatch.setattr(market_data, "recent_bars", _flat_bars)
    monkeypatch.setattr(strategies, "evaluate", lambda genome, ctx: ("long", 1.0))
    now = datetime(2026, 9, 8, 11, 0, tzinfo=ET)
    bars = {s: market_data.recent_bars(s) for s in genome.SYMBOLS}
    feats = {s: {} for s in genome.SYMBOLS}

    with patch.object(option_chain, "cost_gate", return_value={"ok": False, "ratio": 0.4}):
        bot_engine._step_bot(bot, bars, feats, now, "open")
    assert not bots._positions(bot_id), "a failed cost gate must block the entry"

    with patch.object(option_chain, "cost_gate", return_value={"ok": True, "ratio": 1.7}):
        bot_engine._step_bot(bot, bars, feats, now, "open")
    assert bots._positions(bot_id)
    assert "edge 1.70x" in bots.trade_log(bot_id)[0]["note"]


def test_session_open_is_the_0930_bar_not_the_window_start():
    now = datetime(2026, 9, 8, 13, 0, tzinfo=ET)
    t0 = now.replace(hour=9, minute=30)
    bars = []
    for i in range(211):            # 09:30 .. 13:00
        px = 500.0 + i * 0.1
        bars.append({"t": (t0 + timedelta(minutes=i)).isoformat(), "o": px, "h": px + 0.05,
                     "l": px - 0.05, "c": px + 0.02, "v": 1000})
    full = features.build("SPY", bars, {"SPY": bars}, now, minutes_left=180,
                          daily={"prev_close": 499.0, "prev_high": 501.0, "prev_low": 498.0}, vix={})
    window = features.build("SPY", bars[-120:], {"SPY": bars[-120:]}, now, minutes_left=180,
                            daily={"prev_close": 499.0, "prev_high": 501.0, "prev_low": 498.0,
                                   "today_open": 500.0}, vix={})
    assert abs(full["gap_pct"] - window["gap_pct"]) < 1e-9
    assert abs(full["move_so_far_pct"] - window["move_so_far_pct"]) < 1e-9
    assert full["move_so_far_pct"] > 3.0           # measured from the 09:30 open, not 11:00
