"""Vertical debit spreads as a genome option (structure='vertical'): one
defined-risk package per position, priced/marked/settled as long minus
short, never a naked sell-to-open. On a real account the package goes out
as one multi-leg order (see tests/test_live_execution.py).
"""

from datetime import datetime, timedelta

import pytest

from webapp import bot_settings, bot_engine, bots, genome, market_data, option_chain, option_model, replay, strategies

# every bot belongs to an account now (its fleet keys are what its real
# orders go out on) — one owner is enough for these mechanics tests
UID = 1

ET = option_model.ET
S = 560.0
EXPIRY = datetime(2026, 9, 8, 16, 0, tzinfo=ET)


def test_spread_symbol_round_trips_and_singles_are_unchanged():
    sym = option_model.spread_symbol("SPY", EXPIRY, True, 560.0, 563.0)
    assert sym == "SPY260908C00560000-00563000"
    info = option_model.parse_occ(sym)
    assert info["strike"] == 560.0 and info["short_strike"] == 563.0 and info["width"] == 3.0
    assert info["is_call"] and info["underlying"] == "SPY"
    single = option_model.parse_occ("SPY260908P00560000")
    assert single["short_strike"] is None and single["width"] is None
    assert option_model.parse_occ("SPY260908C00560000-junk") is None


def test_intrinsic_caps_at_the_width():
    call = option_model.parse_occ(option_model.spread_symbol("SPY", EXPIRY, True, 560.0, 563.0))
    assert option_model.intrinsic(call, 570.0) == 3.0
    assert option_model.intrinsic(call, 561.5) == 1.5
    assert option_model.intrinsic(call, 550.0) == 0.0
    put = option_model.parse_occ(option_model.spread_symbol("SPY", EXPIRY, False, 560.0, 557.0))
    assert option_model.intrinsic(put, 540.0) == 3.0
    assert option_model.intrinsic(option_model.parse_occ("SPY260908C00560000"), 570.0) == 10.0


def test_vertical_row_is_long_minus_short_and_cheaper_to_hold():
    chain = option_chain.build_chain("SPY", S, 300, 0.20, True)
    long_row = min(chain, key=lambda r: abs(abs(r["delta"]) - 0.5))
    v = option_chain.vertical_row(chain, long_row, 3, S, 300, 0.20)
    short = next(r for r in chain if r["strike"] == long_row["strike"] + 3)
    assert v["short_strike"] == long_row["strike"] + 3 and v["width"] == 3.0
    assert abs(v["price"] - (long_row["price"] - short["price"])) < 1e-6
    assert v["bid"] < v["price"] < v["ask"] <= 3.0
    assert 0 < v["delta"] < long_row["delta"]
    # the package bleeds less theta per share than the naked long over the same hold
    g_single = option_chain.cost_gate(long_row, S, 120, 300, 0.65)
    g_vert = option_chain.cost_gate(v, S, 120, 300, 0.65)
    assert g_vert["theta"] < g_single["theta"]
    # a put vertical shorts the strike BELOW
    pchain = option_chain.build_chain("SPY", S, 300, 0.20, False)
    plong = min(pchain, key=lambda r: abs(abs(r["delta"]) - 0.5))
    pv = option_chain.vertical_row(pchain, plong, 2, S, 300, 0.20)
    assert pv["short_strike"] == plong["strike"] - 2


def test_spread_mark_uses_both_legs_and_caps_at_width():
    info = option_model.parse_occ(option_model.spread_symbol("SPY", EXPIRY, True, 560.0, 562.0))
    now = EXPIRY.replace(hour=11)
    m = option_chain.spread_mark("SPY", S, info, 300, 0.20, EXPIRY, now=now)
    assert 0 < m["bid"] < m["price"] < m["ask"] <= 2.0 and m["source"] == "model"
    deep = option_chain.spread_mark("SPY", 600.0, info, 300, 0.20, EXPIRY, now=now)
    assert deep["price"] <= 2.0 and deep["ask"] <= 2.0


def test_spreads_are_off_so_every_genome_is_single_leg():
    """Single-leg options only: no genome is born, bred or repaired into a spread."""
    assert genome.STRUCTURES == ("single",)
    for _ in range(200):
        assert genome.random_genome()["structure"] == "single"
        assert genome.mutate(genome.random_genome(), rate=1.0)["structure"] == "single"
        assert genome.crossover(genome.random_genome(), genome.random_genome())["structure"] == "single"
    g = genome.random_genome()
    g["structure"] = "vertical"
    genome._repair(g)
    assert g["structure"] == "single"


def test_engine_never_builds_a_spread_even_for_a_stored_vertical_genome(monkeypatch, modeled_fills):
    g = genome.random_genome()
    g.update(style="0dte", structure="vertical", symbol="SPY", signal_strength=0.15, risk_pct=8,
             min_option_volume=0, max_iv=0.9, strike_style="delta", target_delta=0.5, hold_max_min=30)
    bot_id = bots.spawn_bot(g, origin="seed", user_id=UID)
    monkeypatch.setattr(bot_engine.strategies, "evaluate", lambda gg, ctx: ("long", 0.95))
    now = datetime(2026, 6, 2, 14, 0, tzinfo=ET)
    bars = _flat_bars("SPY")
    bot_engine._step_bot(bots.get_bot(bot_id), {"SPY": bars}, {"SPY": {"vix": 15.0}}, now, "open")
    for t in bots._trades(bot_id):
        assert "-" not in t["contract"] and len(t["contract"]) <= 21     # an OCC single, not a package


def _flat_bars(symbol, minutes=120):
    now = datetime.now(ET)
    return [{"t": (now - timedelta(minutes=i)).isoformat(), "o": S, "h": S + 0.02,
             "l": S - 0.02, "c": S, "v": 1000} for i in range(minutes, 0, -1)]


def test_simbook_settles_a_vertical_at_its_width():
    book = replay.SimBook(10_000.0)
    bot = book.add(genome.random_genome())
    contract = option_model.spread_symbol("SPY", EXPIRY, False, 560.0, 558.0)
    book.record_trade(bot["id"], contract, "buy_to_open", 2, 0.80, ts=EXPIRY.replace(hour=12).isoformat())
    assert book.settle_expired(EXPIRY.replace(hour=15, minute=59), {"SPY": 500.0}) == 1
    assert book.closed(bot["id"])[0]["sell_price"] == 2.0


def test_retire_spread_bots_kills_only_spread_genomes():
    import json
    single = bots.spawn_bot(genome.random_genome(), user_id=UID)
    spread = bots.spawn_bot(genome.random_genome(), user_id=UID)
    raw = json.loads(bots.get_bot(spread)["genome"])
    raw["structure"] = "vertical"
    with bots._conn() as c:
        c.execute("UPDATE bots SET genome=? WHERE id=?", (json.dumps(raw), spread))
    assert bots.retire_spread_bots() == {"fleet": 1}
    assert bots.get_bot(spread)["status"] == "spreads_removed"
    assert bots.get_bot(single)["status"] == "active"
