"""Fitness is a shrunk t-statistic of baseline-adjusted per-trade returns —
consistency and sample size score, a lucky outlier does not, a trend day
that lifts every long-call bot does not. See bots.fitness_from_returns().
"""

from datetime import datetime, timedelta

from webapp import bots, genome, option_model

# every bot belongs to an account now (its fleet keys are what its real
# orders go out on) — one owner is enough for these mechanics tests
UID = 1

ET = option_model.ET
D0 = datetime(2026, 8, 3, 15, 0, tzinfo=ET)


def _bot_with(rets, symbol="SPY", start_day=D0):
    """One round trip per entry, on consecutive market days, each with the
    given return on premium (fees 0)."""
    bot_id = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    day = start_day
    for i, r in enumerate(rets):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        contract = option_model.occ_symbol(symbol, day.replace(hour=16), True, 500.0 + i)
        bots.record_trade(bot_id, contract, "buy_to_open", 2, 1.00, ts=day.isoformat(), fees=0)
        bots.record_trade(bot_id, contract, "sell_to_close", 2, 1.00 + r, ts=day.isoformat(), fees=0)
        day += timedelta(days=1)
    return bot_id


def test_steady_edge_outranks_one_lucky_outlier_with_the_same_total():
    steady = _bot_with([0.10] * 20)                         # +200% of premium, evenly
    lucky = _bot_with([3.00] + [-0.05] * 19)                # +205% — but one lottery ticket
    assert bots.stats(steady)["fitness"] > bots.stats(lucky)["fitness"]
    assert bots.stats(lucky)["fitness"] < 1.0


def test_small_samples_are_shrunk_toward_the_fleet_prior():
    hot = _bot_with([0.50] * 5)
    long = _bot_with([0.15] * 40)
    fh, fl = bots.stats(hot)["fitness"], bots.stats(long)["fitness"]
    assert fl > fh, (fh, fl)
    # raw shrinkage math: 5 trades at +50% with a 0 prior and k=20 -> mean 0.10
    assert abs(bots.fitness_from_returns([0.5] * 5, 0.0) - (0.1 / (0.02 / 5 ** 0.5))) < 1e-3


def test_win_rate_no_longer_moves_fitness_on_its_own():
    # 40% hit rate but big winners vs 80% hit rate with tiny winners and one big loser
    low_wr = _bot_with([0.60, -0.20, -0.20, 0.60, -0.20] * 6)
    high_wr = _bot_with([0.05, 0.05, 0.05, 0.05, -0.30] * 6)
    assert bots.stats(low_wr)["win_rate"] < bots.stats(high_wr)["win_rate"]
    assert bots.stats(low_wr)["fitness"] > bots.stats(high_wr)["fitness"]


def test_returns_are_day_adjusted_against_the_buy_hold_baseline():
    bots.ensure_baseline_bots(UID)
    base = next(b for b in bots.baseline_bots() if b["strategy"] == "buy_hold" and b["symbol"] == "SPY")
    rets = [0.40] * 12
    # the always-long baseline made exactly the same on every one of those days
    day = D0
    for i, r in enumerate(rets):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        c = option_model.occ_symbol("SPY", day.replace(hour=16), True, 400.0)
        bots.record_trade(base["id"], c, "buy_to_open", 2, 1.00, ts=day.isoformat(), fees=0)
        bots.record_trade(base["id"], c, "sell_to_close", 2, 1.00 + r, ts=day.isoformat(), fees=0)
        day += timedelta(days=1)
    bot = _bot_with(rets)
    bots.baseline_day_returns(fresh=True)
    bots.fleet_prior_return(fresh=True)
    st = bots.stats(bot)
    assert abs(st["avg_trade_return"]) < 1e-9, "matching the market's drift is not an edge"
    assert abs(st["fitness"]) < 1e-6
    # a bot on QQQ those same days is not adjusted by SPY's baseline
    other = _bot_with(rets, symbol="QQQ")
    assert bots.stats(other)["fitness"] > 1.0


def test_recent_window_veto_stays_sensitive():
    bot = _bot_with([0.30] * 20 + [-0.20] * bots.HOLDOUT_DAYS)
    st = bots.stats(bot)
    assert st["fitness"] > 0 and st["fitness_recent"] < 0
    assert st["robust_fitness"] == st["fitness_recent"]
