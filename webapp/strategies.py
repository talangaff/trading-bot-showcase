"""Entry signals, parameterised by a bot's genome.

evaluate(genome, ctx) -> (direction, strength)
  direction: 'long' | 'short' | None
  strength:  0..1 confidence (the engine also gates on genome['signal_strength'])

`ctx` is the per-symbol feature dict from features.build(), plus:
  ctx['closes'] — list[float] of 1-min closes
  ctx['bars']   — the raw OHLCV bars

The three named strategies (momentum / mean_reversion / breakout) work exactly
as before off closes + the genome's own periods. The "weighted" strategy sums
features.SIGNAL_KEYS, each scaled by a genome weight gene, so evolution can
discover which of the ~24 signals matter.
"""

import random

import numpy as np

from . import features
from .genome import BASELINE_STRATEGIES


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    span = max(2, int(span))
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _rsi(x: np.ndarray, period: int) -> float:
    period = max(2, int(period))
    if len(x) <= period:
        return 50.0
    d = np.diff(x[-(period + 1):])
    up = d[d > 0].sum() / period
    dn = -d[d < 0].sum() / period
    if dn == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + up / dn)


def _named(genome: dict, closes: list[float]):
    x = np.asarray(closes, dtype=float)
    strat = genome.get("strategy", "momentum")
    direction, strength = None, 0.0

    if strat == "momentum":
        f = _ema(x, genome.get("fast_ema", 8))
        s = _ema(x, genome.get("slow_ema", 25))
        gap = (f[-1] - s[-1]) / s[-1]
        slope = (f[-1] - f[-3]) / f[-3] if len(f) > 3 else 0.0
        if gap > 0 and slope > 0:
            direction, strength = "long", min(1.0, abs(gap) * 400 + abs(slope) * 200)
        elif gap < 0 and slope < 0:
            direction, strength = "short", min(1.0, abs(gap) * 400 + abs(slope) * 200)

    elif strat == "mean_reversion":
        s = _ema(x, genome.get("slow_ema", 25))
        window = x[-20:]
        sd = window.std() or 1e-9
        z = (x[-1] - s[-1]) / sd
        thr = genome.get("entry_z", 1.5)
        if z <= -thr:
            direction, strength = "long", min(1.0, abs(z) / (thr * 2))
        elif z >= thr:
            direction, strength = "short", min(1.0, abs(z) / (thr * 2))

    else:  # breakout
        look = x[-20:]
        hi, lo = look[:-1].max(), look[:-1].min()
        rng = (hi - lo) or 1e-9
        thr = genome.get("entry_z", 1.5)
        if x[-1] > hi:
            direction, strength = "long", min(1.0, (x[-1] - hi) / rng * thr * 5)
        elif x[-1] < lo:
            direction, strength = "short", min(1.0, (lo - x[-1]) / rng * thr * 5)

    return direction, strength


def _weighted(genome: dict, ctx: dict):
    score = 0.0
    active = 0
    for k in features.SIGNAL_KEYS:
        w = genome.get("w_" + k, 0.0) or 0.0
        if abs(w) < 1e-6:
            continue
        score += w * (ctx.get(k, 0.0) or 0.0)
        active += 1
    if active == 0:
        return None, 0.0
    norm = score / (active ** 0.5)             # keep strength comparable as weights grow
    if norm > 0:
        return "long", min(1.0, norm / 1.5)
    if norm < 0:
        return "short", min(1.0, -norm / 1.5)
    return None, 0.0


def evaluate_checks(genome: dict, ctx: dict) -> dict:
    """Diagnostic twin of evaluate(): same logic, but also reports which
    genome thresholds are *currently* satisfied against live data, for the
    bot-detail page's live genome readout. Never used by the trading engine.

    Returns {'direction', 'strength', 'ready', 'checks': [
        {'gene', 'met': bool, 'note': str}, ...
    ]}. A gene only appears when it's currently a live, checkable gate —
    e.g. rsi_high only applies once a raw long signal exists to veto.
    """
    checks: list[dict] = []
    closes = ctx.get("closes") or []
    if len(closes) < 20:
        return {"direction": None, "strength": 0.0, "ready": False, "checks": checks}

    x = np.asarray(closes, dtype=float)
    strat = genome.get("strategy", "momentum")

    if strat == "weighted":
        direction, strength = _weighted(genome, ctx)
    else:
        direction, strength = _named(genome, closes)
        if strat == "momentum":
            f = _ema(x, genome.get("fast_ema", 8))
            s = _ema(x, genome.get("slow_ema", 25))
            gap = (f[-1] - s[-1]) / (s[-1] or 1e-9)
            slope = (f[-1] - f[-3]) / (f[-3] or 1e-9) if len(f) > 3 else 0.0
            aligned = (gap > 0 and slope > 0) or (gap < 0 and slope < 0)
            checks.append({"gene": "fast_ema", "met": aligned,
                            "note": "EMA gap {:+.2%}, slope {:+.2%}".format(gap, slope)})
            checks.append({"gene": "slow_ema", "met": aligned,
                            "note": "EMA gap {:+.2%}, slope {:+.2%}".format(gap, slope)})
        elif strat == "mean_reversion":
            s = _ema(x, genome.get("slow_ema", 25))
            window = x[-20:]
            sd = window.std() or 1e-9
            z = (x[-1] - s[-1]) / sd
            thr = genome.get("entry_z", 1.5)
            checks.append({"gene": "entry_z", "met": abs(z) >= thr,
                            "note": "z {:+.2f} vs ±{:.2f} needed".format(z, thr)})
        else:  # breakout
            look = x[-20:]
            hi, lo = look[:-1].max(), look[:-1].min()
            checks.append({"gene": "entry_z", "met": x[-1] > hi or x[-1] < lo,
                            "note": "{:.2f} vs range {:.2f}-{:.2f}".format(x[-1], lo, hi)})

    req = genome.get("signal_strength", 0.3)
    checks.append({"gene": "signal_strength", "met": (strength or 0.0) >= req,
                    "note": "{:.2f} vs {:.2f} needed".format(strength or 0.0, req)})

    rsi = ctx.get("rsi")
    if rsi is None:
        rsi = _rsi(x, genome.get("rsi_period", 14))
    if direction == "long":
        hi = genome.get("rsi_high", 70)
        checks.append({"gene": "rsi_high", "met": rsi < hi,
                        "note": "RSI {:.1f} < {:.1f}".format(rsi, hi)})
    elif direction == "short":
        lo = genome.get("rsi_low", 30)
        checks.append({"gene": "rsi_low", "met": rsi > lo,
                        "note": "RSI {:.1f} > {:.1f}".format(rsi, lo)})

    guard = float(genome.get("sr_veto", 0.0) or 0.0)
    if guard > 0.05 and direction in ("long", "short"):
        d = ctx.get("res_dist_atr" if direction == "long" else "sup_dist_atr", 99.0)
        checks.append({"gene": "sr_veto", "met": d >= guard,
                        "note": "{:.2f} ATR clear vs {:.2f} needed".format(d, guard)})

    # apply the same vetoes evaluate() would, so 'direction'/'strength' here
    # match what the engine would actually act on
    if direction == "long" and rsi >= genome.get("rsi_high", 70):
        direction, strength = None, 0.0
    elif direction == "short" and rsi <= genome.get("rsi_low", 30):
        direction, strength = None, 0.0
    elif guard > 0.05:
        if direction == "long" and ctx.get("res_dist_atr", 99.0) < guard:
            direction, strength = None, 0.0
        elif direction == "short" and ctx.get("sup_dist_atr", 99.0) < guard:
            direction, strength = None, 0.0

    return {"direction": direction, "strength": round(float(strength or 0.0), 3),
            "ready": True, "checks": checks}


def _baseline(genome: dict):
    """'random'/'buy_hold' reference bots — see genome.baseline_genome(). No
    named-strategy logic, no RSI/S-R vetoes: pure signal-less or
    direction-only exposure, on purpose."""
    strat = genome.get("strategy")
    if strat == "buy_hold":
        return "long", 1.0
    if strat == "random":
        direction = random.choice(("long", "short", None))
        return (direction, random.uniform(0.5, 1.0)) if direction else (None, 0.0)
    return None, 0.0


def evaluate(genome: dict, ctx: dict) -> tuple[str | None, float]:
    closes = ctx.get("closes") or []
    if len(closes) < 20:
        return None, 0.0

    if genome.get("strategy") in BASELINE_STRATEGIES:
        return _baseline(genome)

    if genome.get("strategy") == "weighted":
        direction, strength = _weighted(genome, ctx)
    else:
        direction, strength = _named(genome, closes)

    # RSI veto — same as before; use the shared feature RSI if present
    rsi = ctx.get("rsi")
    if rsi is None:
        rsi = _rsi(np.asarray(closes, float), genome.get("rsi_period", 14))
    if direction == "long" and rsi >= genome.get("rsi_high", 70):
        return None, 0.0
    if direction == "short" and rsi <= genome.get("rsi_low", 30):
        return None, 0.0

    # support / resistance veto — don't buy straight into the wall. Gated by a
    # genome gene the bot can evolve to 0 (off).
    guard = float(genome.get("sr_veto", 0.0) or 0.0)
    if guard > 0.05:
        if direction == "long" and ctx.get("res_dist_atr", 99.0) < guard:
            return None, 0.0
        if direction == "short" and ctx.get("sup_dist_atr", 99.0) < guard:
            return None, 0.0

    return direction, round(float(strength or 0.0), 3)
