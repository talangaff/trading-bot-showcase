"""Shrinking the evolutionary search space (mutate() only touches genes
relevant to a bot's current strategy; 'weighted' bots stay sparse) must
never delete a gene from the schema — every gene stays settable and can
become live again the moment it's relevant.
"""

import random

from webapp import genome


def test_no_gene_removed_from_schema():
    g = genome.random_genome()
    assert set(genome.NUM_GENES) <= set(g.keys())
    assert len(genome.NUM_GENES) >= 50   # base genes + one per SIGNAL_KEY


def test_random_genome_weighted_starts_sparse():
    for _ in range(50):
        g = genome.random_genome()
        if g["strategy"] != "weighted":
            continue
        active = sum(1 for k in genome.WEIGHT_GENES if abs(g.get(k, 0.0)) >= 1e-6)
        assert active <= 10


def test_mutate_only_touches_genes_relevant_to_current_strategy():
    random.seed(1)
    base = genome.random_genome()
    base["strategy"] = "momentum"
    for k in genome.WEIGHT_GENES:
        base[k] = 0.0
    entry_z_before = base.get("entry_z")

    for _ in range(200):
        child = genome.mutate(dict(base), rate=1.0)
        if child["strategy"] != "momentum":
            continue   # a strategy-switch mutation fired this draw; skip it
        assert all(child[k] == 0.0 for k in genome.WEIGHT_GENES), \
            "weighted-only genes must not move for a momentum bot"
        assert child["entry_z"] == entry_z_before, \
            "mean_reversion/breakout-only gene must not move for a momentum bot"


def test_mutate_still_touches_universal_genes():
    random.seed(2)
    base = genome.random_genome()
    base["strategy"] = "momentum"
    changed = any(
        genome.mutate(dict(base), rate=1.0)["stop_pct"] != base["stop_pct"]
        for _ in range(50)
    )
    assert changed


def test_weighted_strategy_gets_sparsified_after_mutation():
    random.seed(3)
    g = genome.random_genome()
    g["strategy"] = "weighted"
    for k in genome.WEIGHT_GENES:
        g[k] = 1.0   # force every weight "on"
    child = genome.mutate(g, rate=0.0)   # rate=0 -> no random nudges, just the sparsify pass
    active = sum(1 for k in genome.WEIGHT_GENES if abs(child.get(k, 0.0)) >= 1e-6)
    assert active <= 10


def test_crossover_takes_every_gene_from_one_parent_and_repairs():
    random.seed(3)
    a = genome.random_genome()
    b = genome.random_genome()
    a["strategy"], b["strategy"] = "momentum", "breakout"
    a["mut_scale"], b["mut_scale"] = 0.10, 0.40
    child = genome.crossover(a, b)
    for name in list(genome.NUM_GENES) + list(genome.CATEGORICAL_GENES):
        if name in ("slow_ema", "rsi_high", "swing_max_hold_days"):      # _repair may adjust these
            continue
        assert child[name] in (a[name], b[name]), name
    assert abs(child["mut_scale"] - (0.10 * 0.40) ** 0.5) < 1e-3
    assert child["slow_ema"] > child["fast_ema"] and child["rsi_high"] > child["rsi_low"]
    assert child["strategy"] in ("momentum", "breakout")


def test_mutation_scale_is_self_adaptive_and_bounded():
    random.seed(5)
    g = genome.random_genome()
    g["mut_scale"] = 0.18
    scales = {genome.mutate(g)["mut_scale"] for _ in range(50)}
    assert len(scales) > 10, "each child carries its own perturbed step size"
    lo, hi = genome.MUT_SCALE_BOUNDS
    assert all(lo <= s <= hi for s in scales)
    # an explicit scale (the old API) leaves the stored step alone
    assert genome.mutate(g, scale=0.05)["mut_scale"] == 0.18
    # repair clamps a hand-set value into bounds
    g["mut_scale"] = 9.0
    genome._repair(g)
    assert g["mut_scale"] == hi
