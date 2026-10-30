"""Stops, targets and trails are measured MID-to-MID (bot_pos_state.entry_mid),
not from the ask actually paid to the current mid — otherwise the spread
alone starts every trade several percent "down" and can trip a stop.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from webapp import bot_engine, bots, genome, option_model

ET = option_model.ET


def _closes_with(g, ss, cur, entry_ask=1.08):
    closed = {}

    def fake_record_fill(bot, contract, side, qty, price, **kw):
        closed["reason"] = kw.get("note")
        return None

    now = datetime(2026, 9, 8, 12, 0, tzinfo=ET)
    with patch.object(bot_engine, "_record_fill", fake_record_fill):
        bot_engine._manage_position(
            {"id": 1, "pool": "fleet"}, g, "SPY", "SPYFAKE", {"qty": 5, "avg_price": entry_ask},
            ss, cur=cur, minutes_left=200, phase="open", now=now, live_q=None,
            expiry=now.replace(hour=16))
    return closed.get("reason")


def test_spread_alone_no_longer_trips_the_stop():
    g = genome.random_genome()
    g.update(stop_pct=10, take_profit_pct=200, trail_pct=40, hold_max_min=240,
             trim_frac=0.0, runner_frac=0.0, scale_in_max_adds=0, style="0dte")
    # bought at the 1.08 ask when the mid was 1.00; the mid is still 1.00
    ss = {"opened_ts": datetime(2026, 9, 8, 11, 50, tzinfo=ET).isoformat(), "hwm": 1.00, "entry_mid": 1.00}
    assert _closes_with(g, ss, cur=1.00) is None
    # a real 12% move against does
    assert "stop" in _closes_with(g, ss, cur=0.88)
    # an old position without entry_mid falls back to the ask paid (legacy behaviour)
    legacy = {"opened_ts": ss["opened_ts"], "hwm": 1.00}
    assert "stop" in _closes_with(g, legacy, cur=0.96)


def test_take_profit_is_also_mid_to_mid():
    g = genome.random_genome()
    g.update(stop_pct=30, take_profit_pct=20, trail_pct=40, hold_max_min=240,
             trim_frac=0.0, runner_frac=0.0, scale_in_max_adds=0, style="0dte")
    ss = {"opened_ts": datetime(2026, 9, 8, 11, 50, tzinfo=ET).isoformat(), "hwm": 1.00, "entry_mid": 1.00}
    assert _closes_with(g, ss, cur=1.15) is None           # +15% mid-to-mid: not yet
    assert "take_profit" in _closes_with(g, ss, cur=1.21)


def test_entry_records_the_mid_and_average_in_rebases_it():
    bot_id = bots.spawn_bot(genome.random_genome(), origin="seed")
    ts = datetime(2026, 9, 8, 11, 0, tzinfo=ET).isoformat()
    bots.touch_pos_state(bot_id, "SPYX", 1.08, ts, 0.2, entry_mid=1.00)
    bots.touch_pos_state(bot_id, "SPYX", 1.20, ts, None, entry_mid=5.00)   # later touches never overwrite
    st = bots.pos_state(bot_id)["SPYX"]
    assert st["entry_mid"] == 1.00 and st["last_mark"] == 1.20 and st["hwm"] == 1.20
    bots.mark_pos(bot_id, "SPYX", adds=1, entry_mid=1.10)
    assert bots.pos_state(bot_id)["SPYX"]["entry_mid"] == 1.10
