"""Breeding/culling must draw only from proven, currently-robust bots, and
must never touch the fixed null-hypothesis baseline bots. See
bots.top_eligible(), bots.reset_fleet(), and bot_engine._replacement_genome().

"Proven" is three sample gates (MIN_ELIGIBLE_TRADES trades, MIN_ELIGIBLE_DAYS
distinct sessions, MIN_ELIGIBLE_REGIMES volatility regimes) PLUS a fitness
t-stat above bots.proven_threshold(), which rises with the number of fleet
genomes that have been tried (a Bonferroni bar for "best of N").
"""

from datetime import datetime, timedelta, timezone

from webapp import bot_engine, bots, genome

# every bot belongs to an account now (its fleet keys are what its real
# orders go out on) — one owner is enough for these mechanics tests
UID = 1

START = datetime(2026, 6, 1, 15, 0)     # a Monday
REGIMES = ("low_vol", "mid_vol")


def _sessions(n):
    """n consecutive NYSE-ish weekdays from START, as ISO timestamps."""
    out, d = [], START
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat() + "+00:00")
        d += timedelta(days=1)
    return out


def _add_trades(bot_id, rets, per_day=3, regimes=REGIMES, qty=5, first=0):
    """One round trip per return-on-premium in `rets`, `per_day` trades per
    session on consecutive sessions, regimes alternating by session."""
    days = _sessions((len(rets) + per_day - 1) // per_day)
    for i, r in enumerate(rets):
        di = i // per_day
        ts, regime = days[di], regimes[di % len(regimes)]
        contract = "SPY260904C{:08d}".format(500000 + first + i)
        bots.record_trade(bot_id, contract, "buy_to_open", qty, 1.00, ts=ts, fees=0, regime=regime)
        bots.record_trade(bot_id, contract, "sell_to_close", qty, 1.00 + r, ts=ts, fees=0, regime=regime)


N = bots.MIN_ELIGIBLE_TRADES


def test_unproven_bot_never_becomes_a_breeding_parent():
    bot_id = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    # a big win, but only 1 trade -> nowhere near MIN_ELIGIBLE_TRADES
    _add_trades(bot_id, [2.00])
    assert bots.top_eligible(5, pool="fleet") == []


def test_every_sample_gate_is_required():
    # enough trades and days, but all in ONE regime -> not eligible
    one_regime = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    _add_trades(one_regime, [0.30] * N, regimes=("mid_vol",))
    st = bots.stats(one_regime)
    assert st["closed_count"] >= N and st["active_days"] >= bots.MIN_ELIGIBLE_DAYS
    assert st["regime_count"] == 1 and not st["eligible"] and not st["proven"]

    # enough trades and regimes, but crammed into too few sessions -> not eligible
    few_days = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    _add_trades(few_days, [0.30] * N, per_day=N // 4)
    st = bots.stats(few_days)
    assert st["active_days"] < bots.MIN_ELIGIBLE_DAYS and not st["eligible"]
    assert bots.top_eligible(5, pool="fleet") == []


def test_eligible_but_below_the_multiple_comparison_threshold_is_not_proven():
    noisy = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    _add_trades(noisy, [0.30, -0.28] * (N // 2 + 1))     # clears every sample gate, no real edge
    st = bots.stats(noisy)
    assert st["eligible"] and not st["proven"]
    assert st["robust_fitness"] < st["proven_threshold"]
    assert bots.top_eligible(5, pool="fleet") == []


def test_proven_threshold_rises_with_the_number_of_genomes_tried():
    assert bots.proven_threshold(1) < bots.proven_threshold(60) < bots.proven_threshold(600)
    assert abs(bots.proven_threshold(1) - 1.645) < 0.01          # plain one-sided 5%
    bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    assert bots.bots_tested() == 2                       # every distinct genome is one trial


def test_the_bar_counts_distinct_genomes_not_bots_across_fleets():
    """The multiple-comparison bar must measure the SIZE OF THE SEARCH, not
    the size of the crowd. Two users running the same genome have tried one
    candidate between them; if that counted as two, every new user would
    silently raise the bar and make every genome harder to prove."""
    g = genome.random_genome()
    bots.spawn_bot(g, origin="seed", user_id=1)
    bots.spawn_bot(dict(g), origin="seed", user_id=2)     # same genome, other fleet
    assert bots.bots_tested() == 1

    bots.spawn_bot(genome.random_genome(), origin="seed", user_id=2)
    assert bots.bots_tested() == 2                        # a genuinely new candidate counts


def test_one_users_reset_keeps_the_shared_count_a_pool_reset_restarts_it():
    for _ in range(3):
        bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    assert bots.bots_tested() == 3

    # one user reseeding their own fleet must not erase what the population
    # has learned — the genomes stay in the pool and still count
    bots.reset_fleet("t", user_id=UID)
    assert bots.bots_tested() == 3 + bots.fleet_cap(UID)

    # an installation-wide rules change does restart the count
    bots.reset_fleet("t", user_id=UID, reset_pool=True)
    assert bots.bots_tested() == bots.fleet_cap(UID)


def test_recently_souring_bot_is_excluded_even_if_lifetime_looks_good():
    bot_id = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    # a long proven run, then the last HOLDOUT_DAYS sessions all losers —
    # small enough not to trip the drawdown guardrail
    _add_trades(bot_id, [0.30] * N + [-0.10] * (3 * bots.HOLDOUT_DAYS))

    st = bots.stats(bot_id)
    assert st["eligible"]
    assert st["fitness"] > 0                    # lifetime still looks good
    assert st["fitness_recent"] < 0              # recent trades tell the true story
    assert st["robust_fitness"] == st["fitness_recent"]
    assert bots.top_eligible(5, pool="fleet") == []


def test_robustly_good_bot_is_proven_and_ranked():
    bot_id = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    _add_trades(bot_id, [0.30, 0.25, 0.35] * (N // 3 + 1))     # steady wins throughout
    st = bots.stats(bot_id)
    assert st["eligible"] and st["proven"] and st["regime_count"] >= 2

    top = bots.top_eligible(5, pool="fleet")
    assert [b["id"] for b in top] == [bot_id]


def test_a_strong_t_stat_that_loses_dollars_is_neither_proven_nor_promising(monkeypatch):
    """Fitness is per-trade and size-blind: many small wins plus one oversized
    loss can score a huge t-stat while the account goes down."""
    monkeypatch.setattr(bots, "PROMISING_T", 0.5)
    monkeypatch.setattr(bots, "proven_threshold", lambda n=None, alpha=None: 0.5)
    bot_id = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    _add_trades(bot_id, [0.004, 0.005, 0.006] * (N // 3), qty=2)
    _add_trades(bot_id, [-0.10], qty=10, first=1000)      # one oversized loser
    st = bots.stats(bot_id)
    assert st["eligible"] and st["robust_fitness"] >= st["proven_threshold"]
    assert st["pl_dollar"] < 0
    assert not st["proven"] and not st["promising"]
    assert bots.top_eligible(5, pool="fleet") == []


def test_profitable_bot_between_the_bars_is_promising_not_proven(monkeypatch):
    monkeypatch.setattr(bots, "PROMISING_T", 0.8)      # the test fleet's proven bar is only 1.645
    bot_id = bots.create_bot("t", genome=genome.random_genome(), user_id=UID)
    _add_trades(bot_id, [0.28, -0.12] * (N // 2 + 1))
    st = bots.stats(bot_id)
    assert st["eligible"] and st["pl_dollar"] > 0
    assert bots.PROMISING_T <= st["robust_fitness"] < st["proven_threshold"]
    assert st["promising"] and not st["proven"]
    assert bots.top_eligible(5, pool="fleet") == []          # still not "proven"


def _age(bot_id, days):
    old = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with bots._conn() as c:
        c.execute("UPDATE bots SET created_at=? WHERE id=?", (old, bot_id))


def test_a_bot_that_trades_but_goes_nowhere_is_stagnant():
    flat = bots.create_bot("flat", genome=genome.random_genome(), user_id=UID)
    _add_trades(flat, [0.20, -0.20] * (bots.STAGNANT_MIN_TRADES // 2 + 1))
    young = bots.create_bot("young", genome=genome.random_genome(), user_id=UID)
    _add_trades(young, [0.20, -0.20] * (bots.STAGNANT_MIN_TRADES // 2 + 1))
    _age(flat, bots.STAGNANT_MIN_AGE_DAYS + 5)
    ids = [b["id"] for b in bots.stagnant_active(pool="fleet", user_id=UID)]
    assert flat in ids                 # old, plenty of trades, no edge, balance flat
    assert young not in ids            # no fair trial window yet


def test_a_persistent_loser_is_retired_but_a_small_dip_is_not():
    n = bots.LOSER_MIN_TRADES + 2
    loser = bots.create_bot("loser", genome=genome.random_genome(), user_id=UID)
    _add_trades(loser, [0.10, -0.20] * (n // 2))              # steady bleed, real negative edge
    dip = bots.create_bot("dip", genome=genome.random_genome(), user_id=UID)
    _add_trades(dip, [0.30, -0.28] * (n // 2) + [-0.28])     # down a hair, no persistent edge either way
    few = bots.create_bot("few", genome=genome.random_genome(), user_id=UID)
    _add_trades(few, [-0.20] * 10)                            # losing, but far too few trades
    ids = [b["id"] for b in bots.losing_active(pool="fleet", user_id=UID)]
    assert loser in ids
    assert dip not in ids and few not in ids


def test_exploration_is_capped_and_the_weakest_seeds_are_trimmed(monkeypatch):
    monkeypatch.setattr(bots, "EXPLORE_SHARE", 0.0)       # no room for any seed
    seeds = [bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) for _ in range(3)]
    kid = bots.spawn_bot(genome.random_genome(), origin="mutation", user_id=UID)
    assert bots.exploration_count(UID) == 3 and bots.exploration_room(UID) < 0
    assert bots.excess_explorers(UID) == []               # all still inside their fair trial
    for sid in seeds:
        _age(sid, bots.STAGNANT_MIN_AGE_DAYS + 3)
    out = {b["id"] for b in bots.excess_explorers(UID)}
    assert out == set(seeds) and kid not in out           # descendants are never trimmed
    assert len(bots.excess_explorers(UID, max_n=1)) == 1  # eased out, not purged
    monkeypatch.setattr(bots, "EXPLORE_SHARE", 1.0)
    assert bots.exploration_room(UID) > 0 and bots.excess_explorers(UID) == []


def test_a_child_of_a_sim_elite_carries_the_elites_generation_plus_one(monkeypatch):
    """Sim genomes are not in the gene pool, so the pool cannot number their
    children; the breeder must pass the depth itself."""
    from webapp import incubator
    g = genome.random_genome()
    elite = {"genome": g, "fp": "simparent00000001", "robust": 3.0,
             "family": genome.family_of(g), "generation": 4, "bot_id": 1, "source": "sim"}
    monkeypatch.setattr(incubator, "cached_elites", lambda: [elite])
    monkeypatch.setattr(incubator, "IMMIGRANT_RATE", 0.0)
    monkeypatch.setattr(bot_engine, "IMMIGRANT_RATE", 0.0)
    monkeypatch.setattr(bot_engine, "ADOPT_RATE", 0.0)
    monkeypatch.setattr(bot_engine, "_breeding_parents", lambda: {})
    bid = bot_engine._spawn_replacement(UID)
    b = bots.get_bot(bid)
    assert b["origin"] in ("mutation", "crossover") and int(b["generation"]) == 5


def test_baseline_bots_are_never_bred_from():
    bots.ensure_baseline_bots(UID)
    for b in bots.baseline_bots():
        _add_trades(b["id"], [4.00] * (N + 1))        # make every baseline bot look amazing

    # top_eligible/top_active are always called with pool="fleet" by the
    # breeding code paths (bot_engine._replacement_genome, daily_maintenance) —
    # confirm a baseline bot never shows up there even if it outperforms
    # every fleet bot.
    assert bots.top_eligible(10, pool="fleet") == []
    assert all(b["pool"] == "baseline" for b in bots.baseline_bots())


def test_baseline_bots_are_ticked_but_never_bred_by_evolve():
    bots.ensure_baseline_bots(UID)
    live_pools = {b["pool"] for b in bots.live_bots()}
    assert "baseline" in live_pools

    baseline_ids_before = {b["id"] for b in bots.baseline_bots()}
    bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    ev = bot_engine.evolve()
    assert isinstance(ev, dict)

    # baseline bots must still all exist, still in pool='baseline', still
    # active — evolve() only ever reads/writes status='failing'/'active'
    # fleet bots (see the WHERE clause in evolve()'s SQL and every
    # active_bots(pool="fleet") call inside it)
    baseline_ids_after = {b["id"] for b in bots.baseline_bots()}
    assert baseline_ids_before == baseline_ids_after
    for bid in baseline_ids_after:
        b = bots.get_bot(bid)
        assert b["pool"] == "baseline" and b["status"] == "active"


def test_reset_fleet_retires_old_cohort_and_reseeds_a_full_fleet():
    for _ in range(5):
        bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    before_ids = {b["id"] for b in bots.active_bots(pool="fleet")}

    n = bots.reset_fleet("test_reset", user_id=UID)

    after = bots.active_bots(pool="fleet")
    after_ids = {b["id"] for b in after}
    assert n == len(before_ids)
    assert before_ids.isdisjoint(after_ids)
    assert len(after_ids) == bots.fleet_cap(UID)
    for old_id in before_ids:
        old = bots.get_bot(old_id)
        assert old is not None and old["status"] == "test_reset"   # history kept, not deleted


def test_reset_fleet_never_touches_custom_or_baseline_bots():
    bots.ensure_baseline_bots(UID)
    custom_id = bots.clone_bot(
        bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID),
        start_balance=1000.0)
    baseline_ids_before = {b["id"] for b in bots.baseline_bots()}

    bots.reset_fleet("test_reset", user_id=UID)

    custom = bots.get_bot(custom_id)
    assert custom is not None and custom["status"] == "active" and custom["pool"] == "custom"
    baseline_ids_after = {b["id"] for b in bots.baseline_bots()}
    assert baseline_ids_before == baseline_ids_after
    assert all(bots.get_bot(bid)["status"] == "active" for bid in baseline_ids_after)


def _proven(strategy, user_id=UID):
    """A pooled-proven genome of `strategy`, run by `user_id`'s fleet.
    Returns its shared-pool fingerprint."""
    g = genome.random_genome()
    g["strategy"] = strategy
    bot_id = bots.spawn_bot(g, origin="seed", user_id=user_id)
    _add_trades(bot_id, [0.30, 0.25, 0.35] * (N // 3 + 1))
    return bots.get_bot(bot_id)["genome_fp"]


def test_breeding_is_balanced_across_strategy_families():
    import random
    from unittest.mock import patch
    random.seed(7)
    momentum = {_proven("momentum") for _ in range(5)}
    breakout = {_proven("breakout")}
    fams = bots.top_eligible_by_family(8, pool="fleet")
    assert set(fams) == {"momentum", "breakout"}
    assert len(fams["momentum"]) == 5 and len(fams["breakout"]) == 1

    picks = {"momentum": 0, "breakout": 0}
    origins = set()
    with patch.object(bot_engine, "IMMIGRANT_RATE", 0.0), \
         patch.object(bot_engine, "ADOPT_RATE", 0.0), \
         patch.object(bot_engine, "MAX_STRATEGY_SHARE", 1.01):
        for _ in range(200):
            child, parent_fp, origin = bot_engine._replacement_genome(UID)
            origins.add(origin)
            picks["breakout" if parent_fp in breakout else "momentum"] += 1
    # one proven breakout genome vs five momentum ones, yet the family gets ~half the children
    assert picks["breakout"] >= 60, picks
    assert {"mutation", "crossover"} <= origins

    fl = bots.family_leaderboard("fleet")
    assert {f["family"] for f in fl} == {"momentum", "breakout"}
    assert all(f["proven"] == f["count"] for f in fl)
