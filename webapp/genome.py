"""A bot's "skills": the genome of tunable parameters that drives its strategy.

Stored as JSON on the bots row. Evolution = mutate() a good genome and spawn a
fresh bot with the result.
"""

import json
import math
import random

from . import config
from .features import SIGNAL_KEYS

# Self-adaptive mutation step (evolution-strategies style): every genome
# carries its own `mut_scale`, the relative size of a numeric nudge as a
# fraction of the gene's range. Each mutate() call first perturbs the scale
# log-normally, then uses the new scale — lineages that need fine-tuning
# drift toward small steps, lineages that need to escape drift toward big
# ones, and selection sorts out which. Not a behaviour gene: never read by
# the strategy, so it lives outside NUM_GENES.
MUT_SCALE_BOUNDS = (0.03, 0.40)
MUT_SCALE_DEFAULT = 0.18
MUT_SCALE_TAU = 0.3
CATEGORICAL_GENES = ("strategy", "symbol", "order_pref", "strike_style", "scale_in_when", "style",
                     "structure")

STRATEGIES = ("momentum", "mean_reversion", "breakout", "weighted")
# Fixed, never-evolved reference strategies (see baseline_genome() below) —
# kept out of STRATEGIES so random_genome()/mutate() never introduce them
# into the evolvable fleet population.
BASELINE_STRATEGIES = ("random", "buy_hold")
SYMBOLS = ("SPY", "QQQ")
ORDER_PREFS = ("market", "limit", "stop_limit", "trailing_stop")
# how the bot picks its strike off the (modeled) option chain
STRIKE_STYLES = ("delta", "distance", "premium", "cheapest")
# when (if ever) the bot averages more contracts into an open position
SCALE_IN_MODES = ("never", "strength", "weakness", "both")
# 0dte: today's close, force-flattened same day (the fleet's original mode).
# swing: a real listed expiry weeks out (option_quotes.pick_expiry), held
# across multiple days up to swing_max_hold_days — see bot_engine.py.
# Custom bots are forced to "0dte" regardless of this gene until swing
# execution against a real account has actually been observed running in
# the fleet simulation first.
STYLES = ("0dte", "swing")
# single: one long call/put (the original). vertical: a DEBIT spread — long
# the chosen strike, short `spread_width` strikes further OTM — booked as one
# defined-risk package (option_model.spread_symbol). Custom bots are pinned
# to "single": the real Alpaca account trades legs, not packages.
STRUCTURES = ("single", "vertical") if config.BOT_ALLOW_SPREADS else ("single",)

# name -> (low, high)  numeric genes, clamped to these bounds
NUM_GENES = {
    "fast_ema":          (3, 20),
    "slow_ema":          (15, 60),
    "rsi_period":        (5, 21),
    "rsi_low":           (15, 40),
    "rsi_high":          (60, 85),
    "entry_z":           (0.4, 3.0),
    "signal_strength":   (0.15, 0.9),
    "stop_pct":          (10, 60),
    "take_profit_pct":   (15, 200),
    "trail_pct":         (5, 40),
    "hold_max_min":      (5, 240),      # 0dte only — intraday time stop, in minutes
    "swing_target_dte":  (5, 45),       # swing only — target days-to-expiration at entry
    "swing_max_hold_days": (1, 20),     # swing only — force-exit after this many days if not banked
    "spread_width":      (1, 5),        # vertical only — short leg this many $1 strikes further OTM
    "risk_pct":          (2, 10),
    "limit_offset_pct":  (0.0, 2.0),
    # --- strike selection ---
    "iv_bias":           (0.6, 1.6),    # multiplier on realized-vol-derived ATM IV
    "target_delta":      (0.15, 0.65),  # for strike_style == "delta"
    "otm_pct":           (-0.6, 2.0),   # % distance for "distance" (neg = ITM)
    "premium_frac":      (0.2, 0.9),    # target premium as fraction of budget
    "min_option_volume": (0, 400),      # skip strikes thinner than this (modeled)
    "max_iv":            (0.20, 0.90),  # refuse strikes richer than this
    # --- position management (trim / runners / average-in) ---
    "trim_trigger_pct":     (15, 120),  # P&L%% at which to take a partial
    "trim_frac":            (0.0, 0.8), # fraction of the position sold at trim; 0 = never trim
    "runner_frac":          (0.0, 0.6), # fraction held back as runners at take-profit; 0 = none
    "runner_trail_pct":     (12, 60),   # trailing stop %% for runners / post-trim remainder
    "scale_in_trigger_pct": (10, 70),   # |P&L%%| move that triggers an average-in
    "scale_in_add_frac":    (0.25, 1.0),# add size as a fraction of current qty
    "scale_in_max_adds":    (0, 3),     # cap on average-ins; 0 = never
    # --- support / resistance ---
    "sr_veto":              (0.0, 1.5), # skip an entry within this many ATRs of
                                        # the level it would trade into; 0 = off
}
INT_GENES = {"fast_ema", "slow_ema", "rsi_period", "rsi_low", "rsi_high",
             "hold_max_min", "min_option_volume", "scale_in_max_adds",
             "swing_target_dte", "swing_max_hold_days", "spread_width"}

# One weight gene per feature the "weighted" strategy sums. Only bots whose
# strategy == "weighted" read these; everyone else ignores them.
WEIGHT_GENES = {"w_" + k: (-1.2, 1.2) for k in SIGNAL_KEYS}
NUM_GENES.update(WEIGHT_GENES)

# Which numeric genes each entry-signal strategy actually reads (see
# strategies._named()/_weighted()). Every other numeric gene — risk, exit,
# strike selection, position management, and the RSI/S-R vetoes that apply
# to every strategy's raw signal in strategies.evaluate() — is UNIVERSAL:
# it affects behaviour no matter which strategy a bot runs.
#
# mutate() uses this to only perturb genes that can actually change a bot's
# behaviour right now, instead of spending mutation budget nudging (say)
# w_* weights on a "momentum" bot that never reads them. No gene is ever
# removed from NUM_GENES over this — every gene keeps its stored value and
# is fully live again the moment a strategy-switch mutation makes it
# relevant, or a custom bot's owner hand-sets it directly.
_STRATEGY_GENES = {
    "momentum": {"fast_ema", "slow_ema"},
    "mean_reversion": {"slow_ema", "entry_z"},
    "breakout": {"entry_z"},
    "weighted": set(WEIGHT_GENES),
}
# Same idea, but keyed by STYLES (0dte vs swing) instead of STRATEGIES — see
# bot_engine.py for where hold_max_min (minutes) vs swing_target_dte/
# swing_max_hold_days (days) actually get read.
_STYLE_GENES = {
    "0dte": {"hold_max_min"},
    "swing": {"swing_target_dte", "swing_max_hold_days"},
}
_STRUCTURE_GENES = {"single": set(), "vertical": {"spread_width"}}
UNIVERSAL_GENES = (set(NUM_GENES) - set().union(*_STRATEGY_GENES.values())
                  - set().union(*_STYLE_GENES.values())
                  - set().union(*_STRUCTURE_GENES.values()))

# neutral defaults for genes missing from an older genome — so existing bots
# migrate with the new behaviours OFF and only pick them up through mutation.
_OFF_DEFAULTS = {
    "trim_frac": 0.0, "runner_frac": 0.0, "scale_in_max_adds": 0,
    "trim_trigger_pct": 40, "runner_trail_pct": 30,
    "scale_in_trigger_pct": 40, "scale_in_add_frac": 0.5,
    "sr_veto": 0.0,
}
_OFF_DEFAULTS.update({name: 0.0 for name in WEIGHT_GENES})


def _rand_num(name):
    lo, hi = NUM_GENES[name]
    v = random.uniform(lo, hi)
    return int(round(v)) if name in INT_GENES else round(v, 3)


def _sparsify_weights(g: dict, min_active: int = 4, max_active: int = 10) -> None:
    """Keep only the strongest few 'weighted'-strategy w_* genes nonzero,
    zero the rest — active feature selection so each bot searches a small
    combinatorial subset of SIGNAL_KEYS at a time instead of drifting
    toward all ~24 nonzero at once. No gene is deleted: a zeroed weight is
    exactly the "off" state strategies._weighted() already treats specially
    (skipped from the sum), and any later mutation can turn it back on —
    the full 24-gene vocabulary stays available to the population, just not
    all pinned "on" in any one genome."""
    weights = [(k, g.get(k, 0.0) or 0.0) for k in WEIGHT_GENES]
    active = [(k, w) for k, w in weights if abs(w) >= 1e-6]
    if len(active) <= max_active:
        return
    keep = random.randint(min_active, max_active)
    active.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for k, _ in active[keep:]:
        g[k] = 0.0


def random_genome() -> dict:
    g = {
        "strategy": random.choice(STRATEGIES),
        "symbol": random.choice(SYMBOLS),
        "order_pref": random.choice(ORDER_PREFS),
        "strike_style": random.choice(STRIKE_STYLES),
        "scale_in_when": random.choice(SCALE_IN_MODES),
        "style": random.choice(STYLES),
        "structure": random.choice(STRUCTURES),
    }
    for name in NUM_GENES:
        g[name] = _rand_num(name)
    g["mut_scale"] = round(random.uniform(0.08, 0.30), 3)
    if g["strategy"] == "weighted":
        _sparsify_weights(g)
    _repair(g)
    return g


def _repair(g: dict) -> None:
    for name, (lo, hi) in NUM_GENES.items():
        v = g.get(name, _OFF_DEFAULTS.get(name, _rand_num(name)))
        v = max(lo, min(hi, v))
        g[name] = int(round(v)) if name in INT_GENES else round(float(v), 3)
    try:
        ms = float(g.get("mut_scale", MUT_SCALE_DEFAULT))
    except (TypeError, ValueError):
        ms = MUT_SCALE_DEFAULT
    g["mut_scale"] = round(max(MUT_SCALE_BOUNDS[0], min(MUT_SCALE_BOUNDS[1], ms)), 4)
    if g["slow_ema"] <= g["fast_ema"]:
        g["slow_ema"] = min(NUM_GENES["slow_ema"][1], g["fast_ema"] + 5)
    if g["rsi_high"] <= g["rsi_low"]:
        g["rsi_high"] = min(NUM_GENES["rsi_high"][1], g["rsi_low"] + 15)
    if g["strategy"] not in STRATEGIES and g["strategy"] not in BASELINE_STRATEGIES:
        g["strategy"] = "momentum"
    if g["symbol"] not in SYMBOLS:
        g["symbol"] = "SPY"
    if g["order_pref"] not in ORDER_PREFS:
        g["order_pref"] = "market"
    if g.get("strike_style") not in STRIKE_STYLES:
        g["strike_style"] = "delta"
    if g.get("scale_in_when") not in SCALE_IN_MODES:
        g["scale_in_when"] = "never"
    if g.get("style") not in STYLES:
        g["style"] = "0dte"
    if g.get("structure") not in STRUCTURES:
        g["structure"] = "single"
    # never intentionally plan to hold past your own target DTE window —
    # the position still settles safely on its real expiry either way
    # (bot_engine._settle_expired), this just keeps the genome sensible
    if g["swing_max_hold_days"] > g["swing_target_dte"]:
        g["swing_max_hold_days"] = g["swing_target_dte"]


def baseline_genome(strategy: str, symbol: str) -> dict:
    """Fixed genome for a null-hypothesis reference bot — never bred,
    mutated, or culled (bot_engine excludes pool='baseline' from evolution
    entirely). 'random' fires a random long/short/flat signal every bar;
    'buy_hold' is always long a fresh ATM-ish call, held to the close, every
    day (0DTE has no real "buy and hold" since everything expires daily, so
    this is the closest analogue: always-long, no management). Both skip
    every genome-gated veto/management behaviour (see strategies.evaluate())
    so they measure pure signal-less / direction-only exposure — the floor
    the evolved fleet actually has to beat to mean anything.
    """
    g = {name: _rand_num(name) for name in NUM_GENES}
    g.update({
        "strategy": strategy, "symbol": symbol, "order_pref": "market",
        "strike_style": "delta", "target_delta": 0.5, "style": "0dte",
        "stop_pct": NUM_GENES["stop_pct"][1],
        "take_profit_pct": NUM_GENES["take_profit_pct"][1],
        "hold_max_min": NUM_GENES["hold_max_min"][1],
        "trim_frac": 0.0, "runner_frac": 0.0, "scale_in_max_adds": 0,
        "sr_veto": 0.0, "signal_strength": 0.0, "min_option_volume": 0,
        "max_iv": NUM_GENES["max_iv"][1], "limit_offset_pct": 0.0,
        "risk_pct": 5.0,
    })
    for name in INT_GENES:
        if name in g:
            g[name] = int(round(g[name]))
    return g


def mutate(genome: dict, rate: float = 0.35, scale: float | None = None) -> dict:
    """Return a slightly-changed copy. `rate` = chance to touch each gene,
    `scale` = relative size of a numeric nudge; None (the default) means
    SELF-ADAPTIVE — the genome's own mut_scale is log-normally perturbed
    first and both used for this call and stored on the child.

    Only perturbs UNIVERSAL_GENES plus whichever strategy-specific genes
    the genome's CURRENT strategy actually reads (see _STRATEGY_GENES) —
    shrinks the effective search space per mutation call without deleting
    any gene: an irrelevant gene simply isn't touched THIS call, but keeps
    whatever value it already had and becomes live again the moment a
    strategy-switch (below) makes it relevant, or a custom bot's owner
    hand-sets it directly.
    """
    g = dict(genome or random_genome())
    if scale is None:
        cur = float(g.get("mut_scale") or MUT_SCALE_DEFAULT)
        scale = cur * math.exp(MUT_SCALE_TAU * random.gauss(0.0, 1.0))
        scale = max(MUT_SCALE_BOUNDS[0], min(MUT_SCALE_BOUNDS[1], scale))
        g["mut_scale"] = round(scale, 4)
    live_genes = (UNIVERSAL_GENES
                 | _STRATEGY_GENES.get(g.get("strategy", "momentum"), set())
                 | _STYLE_GENES.get(g.get("style", "0dte"), set())
                 | _STRUCTURE_GENES.get(g.get("structure", "single"), set()))
    for name in live_genes:
        lo, hi = NUM_GENES[name]
        if random.random() < rate:
            span = hi - lo
            g[name] = g.get(name, _rand_num(name)) + random.uniform(-scale, scale) * span
    if random.random() < rate * 0.5:
        g["strategy"] = random.choice(STRATEGIES)
    if random.random() < rate * 0.4:
        g["symbol"] = random.choice(SYMBOLS)
    if random.random() < rate * 0.6:
        g["order_pref"] = random.choice(ORDER_PREFS)
    if random.random() < rate * 0.6:
        g["strike_style"] = random.choice(STRIKE_STYLES)
    if random.random() < rate * 0.6:
        g["scale_in_when"] = random.choice(SCALE_IN_MODES)
    if random.random() < rate * 0.3:
        g["style"] = random.choice(STYLES)
    if random.random() < rate * 0.3:
        g["structure"] = random.choice(STRUCTURES)
    if g.get("strategy") == "weighted":
        _sparsify_weights(g)
    _repair(g)
    return g


def crossover(a: dict, b: dict) -> dict:
    """Uniform crossover: a child that takes every gene — numeric,
    categorical, and the weighted-strategy weights — from parent `a` or
    parent `b` with equal chance, then repaired. Lets a lineage with good
    strike selection and exits combine with one that has a good entry
    signal, which mutation alone reaches only by luck. The child's
    mut_scale is the parents' geometric mean."""
    a = dict(a or random_genome())
    b = dict(b or random_genome())
    child = {}
    for name in list(NUM_GENES) + list(CATEGORICAL_GENES):
        src = a if random.random() < 0.5 else b
        child[name] = src.get(name, b.get(name, a.get(name)))
    ma = float(a.get("mut_scale") or MUT_SCALE_DEFAULT)
    mb = float(b.get("mut_scale") or MUT_SCALE_DEFAULT)
    child["mut_scale"] = round(math.sqrt(ma * mb), 4)
    if child.get("strategy") == "weighted":
        _sparsify_weights(child)
    _repair(child)
    return child


def family_of(genome: dict | None) -> str:
    """The strategy family a genome belongs to — what the per-family
    leaderboard and family-balanced breeding group on."""
    return (genome or {}).get("strategy") or "?"


def describe(genome: dict) -> str:
    if not genome:
        return "—"
    g = genome
    strike_style = g.get("strike_style", "delta")
    if strike_style == "delta":
        strike = "{:.2f}Δ".format(g.get("target_delta", 0.3))
    elif strike_style == "distance":
        strike = "{:+g}% OTM".format(g.get("otm_pct", 0))
    elif strike_style == "premium":
        strike = "{:g}% budget".format(g.get("premium_frac", 0.5) * 100)
    else:
        strike = "cheapest"
    horizon = ("swing {}d/hold≤{}d".format(g.get("swing_target_dte", "?"), g.get("swing_max_hold_days", "?"))
              if g.get("style") == "swing" else "0DTE")
    if g.get("structure") == "vertical":
        horizon += " · vertical {}w".format(g.get("spread_width", "?"))
    extras = []
    if g.get("scale_in_when", "never") != "never" and g.get("scale_in_max_adds", 0):
        extras.append("avg-in {}×{}".format(g["scale_in_when"][:4], g["scale_in_max_adds"]))
    if g.get("trim_frac", 0) > 0:
        extras.append("trim {:.0f}%@{:.0f}%".format(
            g["trim_frac"] * 100, g.get("trim_trigger_pct", 40)))
    if g.get("runner_frac", 0) > 0:
        extras.append("runners {:.0f}%".format(g["runner_frac"] * 100))
    tail = (" · " + " · ".join(extras)) if extras else ""
    if g.get("strategy") == "weighted":
        top = sorted(((k[2:], g[k]) for k in g if k.startswith("w_") and abs(g[k]) > 0.25),
                     key=lambda kv: abs(kv[1]), reverse=True)[:3]
        core = "{} · weighted [{}]".format(
            g.get("symbol", "?"),
            ", ".join("{}{:+.1f}".format(k, w) for k, w in top) or "unset")
    else:
        core = "{} · {} · EMA {}/{} · RSI {}".format(
            g.get("symbol", "?"), g.get("strategy", "?"),
            g.get("fast_ema", "?"), g.get("slow_ema", "?"), g.get("rsi_period", "?"))
    return "{} · {} · strike {} · stop {:g}% · TP {:g}% · {}{}".format(
        core, horizon, strike, g.get("stop_pct", 0), g.get("take_profit_pct", 0),
        g.get("order_pref", "?"), tail)


def dumps(genome: dict) -> str:
    return json.dumps(genome, separators=(",", ":"), sort_keys=True)


def loads(s: str | None) -> dict:
    if not s:
        return random_genome()
    try:
        g = json.loads(s)
        _repair(g)
        return g
    except (ValueError, TypeError):
        return random_genome()
