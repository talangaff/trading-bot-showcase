"""Swing-style bots (genome 'style' == 'swing'): multi-day options positions
that must NOT be force-flattened by the daily 0DTE close-out, must still be
force-settled on their own real expiry day, and must never be abandoned
(retired/pruned) while genuinely still open. Custom bots stay 0DTE-only.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from webapp import bot_settings, bot_engine, bots, genome, market_data, option_chain, option_model, strategies

# every bot belongs to an account now (its fleet keys are what its real
# orders go out on) — one owner is enough for these mechanics tests
UID = 1

ET = option_model.ET


def _flat_bars(symbol, minutes=120):
    now = datetime.now(ET)
    return [{"t": (now - timedelta(minutes=i)).isoformat(), "o": 560.0, "h": 560.02,
            "l": 559.98, "c": 560.0, "v": 1000} for i in range(minutes, 0, -1)]


def test_style_gene_present_and_no_genes_deleted():
    g = genome.random_genome()
    assert "style" in g and g["style"] in genome.STYLES
    assert set(genome.NUM_GENES) <= set(g.keys())


def test_repair_clamps_max_hold_to_target_dte():
    g = genome.random_genome()
    g.update(style="swing", swing_target_dte=10, swing_max_hold_days=19)
    genome._repair(g)
    assert g["swing_max_hold_days"] <= g["swing_target_dte"]


def test_mutate_never_touches_0dte_only_gene_on_a_swing_bot():
    g = genome.random_genome()
    g["style"] = "swing"
    hold_before = g["hold_max_min"]
    for _ in range(100):
        child = genome.mutate(dict(g), rate=1.0)
        if child["style"] == "swing":
            assert child["hold_max_min"] == hold_before


def test_swing_entry_bypasses_0dte_cutoff_and_picks_a_real_dated_contract(monkeypatch, modeled_fills):
    g = genome.random_genome()
    # pin everything this test is NOT about (strike filters, structure, the
    # cost gate) so only the swing-vs-0DTE cutoff decides the outcome
    g.update(style="swing", swing_target_dte=14, swing_max_hold_days=10, structure="single",
             strike_style="delta", target_delta=0.5, min_option_volume=0, max_iv=0.9,
             risk_pct=8, signal_strength=0.15)
    bot_id = bots.spawn_bot(g, origin="seed", user_id=UID)
    bot = bots.get_bot(bot_id)

    monkeypatch.setattr(market_data, "recent_bars", _flat_bars)
    monkeypatch.setattr(market_data, "daily_ctx", lambda s: {})
    monkeypatch.setattr(market_data, "vix", lambda: {})
    monkeypatch.setattr(strategies, "evaluate", lambda genome, ctx: ("long", 1.0))
    monkeypatch.setattr("webapp.option_quotes.pick_expiry", lambda sym, dte, today=None: None)
    monkeypatch.setattr("webapp.option_quotes.get_chain", lambda *a, **k: {})   # no network
    monkeypatch.setattr(option_chain, "cost_gate", lambda *a, **k: {"ok": True, "ratio": 2.0})

    now = datetime(2026, 9, 8, 15, 56, tzinfo=ET)   # past the 0DTE 15:30 cutoff
    bars = {s: market_data.recent_bars(s) for s in genome.SYMBOLS}
    feats = {s: {} for s in genome.SYMBOLS}
    bot_engine._step_bot(bot, bars, feats, now, "open")

    positions = bots._positions(bot_id)
    assert positions, "swing entry must not be blocked by the 0DTE 15:30 cutoff"
    contract = next(iter(positions))
    info = option_model.parse_occ(contract)
    assert (info["expiry"].date() - now.date()).days >= 10


def test_swing_position_survives_the_daily_close_out(monkeypatch):
    g = genome.random_genome()
    g.update(style="swing", swing_target_dte=14, swing_max_hold_days=10)
    bot_id = bots.spawn_bot(g, origin="seed", user_id=UID)
    now = datetime(2026, 9, 8, 12, 0, tzinfo=ET)
    expiry = now + timedelta(days=14)
    contract = option_model.occ_symbol("SPY", expiry, True, 560.0)
    bots.record_trade(bot_id, contract, "buy_to_open", 5, 2.00, ts=now.isoformat(), fees=0)

    monkeypatch.setattr(market_data, "recent_bars", _flat_bars)
    bot = bots.get_bot(bot_id)
    bars = {s: market_data.recent_bars(s) for s in genome.SYMBOLS}
    feats = {s: {} for s in genome.SYMBOLS}
    closing_now = now + timedelta(days=1, hours=4)   # next day, in the 15:55-16:05 window
    bot_engine._step_bot(bot, bars, feats, closing_now, "closing")

    assert bots._positions(bot_id), "a swing position must not be flattened by the daily close-out"


def test_swing_time_stop_fires_after_max_hold_days():
    g = genome.random_genome()
    g.update(style="swing", swing_max_hold_days=10, stop_pct=90,
            take_profit_pct=900, trim_frac=0.0, runner_frac=0.0, trail_pct=90)
    now = datetime(2026, 9, 19, 12, 0, tzinfo=ET)
    opened = (now - timedelta(days=11)).isoformat()
    expiry = now + timedelta(days=3)      # real expiry still days away

    closed = {}
    def fake_record_fill(bot, contract, side, qty, price, **kw):
        closed["reason"] = kw.get("note")
        return None
    with patch.object(bot_engine, "_record_fill", fake_record_fill), \
         patch.object(bots, "clear_pos_state", lambda *a, **k: None):
        bot_engine._manage_position(
            {"id": 1}, g, "SPY", "SPYFAKE", {"qty": 5, "avg_price": 2.00},
            {"opened_ts": opened, "hwm": 2.00}, cur=2.00, minutes_left=4320,
            phase="open", now=now, live_q=None, expiry=expiry)

    assert closed.get("reason") and "time" in closed["reason"]


def test_settle_expired_ignores_swing_until_its_real_expiry(monkeypatch):
    monkeypatch.setattr(market_data, "last_price", lambda sym: 570.0)
    bot_id = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    expiry = datetime(2026, 9, 22, 16, 0, tzinfo=ET)
    contract = option_model.occ_symbol("SPY", expiry, True, 560.0)
    ts = (expiry - timedelta(days=10)).isoformat()
    bots.record_trade(bot_id, contract, "buy_to_open", 5, 2.00, ts=ts, fees=0)

    n = bot_engine._settle_expired(expiry - timedelta(days=2))
    assert n == 0 and bots._positions(bot_id)

    n2 = bot_engine._settle_expired(expiry.replace(hour=15, minute=59))
    assert n2 == 1 and not bots._positions(bot_id)
    assert bots.trade_log(bot_id)[0]["price"] == 10.0   # intrinsic = 570 - 560


def test_stale_skips_a_mid_hold_swing_bot_but_culling_marks_it_to_market():
    bot_id = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    expiry = datetime.now(ET) + timedelta(days=10)
    contract = option_model.occ_symbol("SPY", expiry, True, 560.0)
    ts = (datetime.now(ET) - timedelta(days=10)).isoformat()
    bots.record_trade(bot_id, contract, "buy_to_open", 2, 2.00, ts=ts, fees=0)
    bots.touch_pos_state(bot_id, contract, 2.00, ts)

    # genuinely mid-hold: not "idle", so the idle-kill must leave it alone
    assert bot_id not in [b["id"] for b in bots.stale_active_bots(3, pool="fleet")]

    # a modest realized record, plus the open position now marked 60% under water
    for i in range(6):
        c2 = option_model.occ_symbol("SPY", expiry, True, 561.0 + i)
        t = (datetime.now(ET) - timedelta(days=9 - i)).isoformat()
        assert bots.record_trade(bot_id, c2, "buy_to_open", 2, 1.00, ts=t, fees=0) is None
        assert bots.record_trade(bot_id, c2, "sell_to_close", 2, 1.02, ts=t, fees=0) is None
    bots.touch_pos_state(bot_id, contract, 0.80, ts)
    st = bots.stats(bot_id)
    assert st["unrealized_pl"] < 0 and st["equity"] < st["balance"]
    assert st["fitness_marked"] < st["fitness"], "the open loser must count against the bot now"
    assert st["robust_fitness"] == st["fitness_marked"]

    # culling sees it (ranked on the marked number) — no more hiding behind an open position
    assert bot_id in [b["id"] for b in
                      bots.worst_active(50, min_trades=1, min_age_hours=0, pool="fleet")]
    # and retirement flattens at the mark first so nothing is left unmanaged
    assert bots.flatten_bot(bot_id) == 1
    assert not bots._positions(bot_id)
    last = bots.trade_log(bot_id)[0]
    assert last["side"] == "sell_to_close" and last["price"] < 0.80 and last["forced"]


def test_manually_retired_bot_is_flattened_at_once_and_then_prunable():
    # retiring a bot closes everything it holds, so nothing is left unmanaged
    bot_id = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    expiry = datetime.now(ET) + timedelta(days=10)
    contract = option_model.occ_symbol("SPY", expiry, True, 560.0)
    ts = (datetime.now(ET) - timedelta(days=1)).isoformat()
    bots.record_trade(bot_id, contract, "buy_to_open", 5, 2.00, ts=ts, fees=0)

    bots.retire_bot(bot_id, "retired")
    assert not bots._positions(bot_id)
    assert bot_id not in [b["id"] for b in bots.live_bots()]
    assert bots.trade_log(bot_id)[0]["side"] == "sell_to_close"

    with bots._conn() as c:
        c.execute("UPDATE bots SET flagged_at=? WHERE id=?",
                  ((datetime.now(ET) - timedelta(days=30)).isoformat(), bot_id))
    bots.prune_retired(keep_days=5)
    assert bots.get_bot(bot_id) is None


def test_custom_bots_keep_their_style():
    # a clone is for testing ONE genome by itself on a separate account, so
    # it trades exactly as its source does — swing included
    src_g = genome.random_genome()
    src_g["style"] = "swing"
    src_id = bots.spawn_bot(src_g, origin="seed", user_id=UID)
    custom_id = bots.clone_bot(src_id, start_balance=10000.0)
    assert bots.genome_of(bots.get_bot(custom_id))["style"] == "swing"

    bots.update_genome(custom_id, {"style": "0dte"})
    assert bots.genome_of(bots.get_bot(custom_id))["style"] == "0dte"


def test_record_trade_cutoff_only_blocks_contracts_expiring_today():
    src_id = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    custom_id = bots.clone_bot(src_id, start_balance=10000.0)
    now = datetime.now(ET).replace(hour=15, minute=56)

    swing_contract = option_model.occ_symbol("SPY", now + timedelta(days=14), True, 560.0)
    assert bots.record_trade(custom_id, swing_contract, "buy_to_open", 2, 1.00,
                             ts=now.isoformat(), fees=0) is None

    today_contract = option_model.occ_symbol(
        "SPY", now.replace(hour=16, minute=0), True, 560.0)
    err = bots.record_trade(custom_id, today_contract, "buy_to_open", 2, 1.00,
                            ts=now.isoformat(), fees=0)
    assert err is not None and "close-out window" in err
