"""Per-symbol feature context, built once per tick and shared by the whole fleet.

build() turns the 1-minute bar history (plus spot, the other index, VIX, the
session clock and the macro-calendar flag) into a flat dict of ~40 indicators.
The rule-based strategies read a handful of them; the "weighted" strategy lets
evolution assign a weight to every entry in SIGNAL_KEYS and sum them.

Everything here is best-effort: short history or a missing feed yields 0 for
the affected keys, never an exception.
"""

import math
from datetime import date, datetime

import numpy as np

from . import market_calendar, option_chain, option_model

ET = option_model.ET

ET_OPEN_MIN = 9 * 60 + 30
SESSION_MIN = 390.0
OPENING_RANGE_MIN = 30

# Features the "weighted" strategy sums. Each is oriented so POSITIVE = bullish
# and scaled to roughly [-2, 2]. Genome carries a `w_<key>` weight for each.
SIGNAL_KEYS = (
    "ret_15", "ema_gap", "ema_slope", "macd_hist", "adx_dir", "linreg_slope",
    "rsi_dev", "stoch_dev", "bb_pctb", "z_vwap", "dist_vwap", "or_break",
    "gap_fade", "pdh_pdl_pos", "rvol_dir", "obv_slope", "vol_delta",
    "rs_spread", "vix_term", "vix_dir", "iv_rv", "move_budget", "data_day",
    "sr_room", "sr_break",
)

# how far a swing high/low may drift and still count as the same level
_SR_CLUSTER_ATR = 0.35
# distance (in ATR) treated as "no level in the way" when nothing is found
_SR_CAP_ATR = 5.0


def _f(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _clip(v, lo=-2.0, hi=2.0):
    return max(lo, min(hi, _f(v)))


def _ema(x, span):
    span = max(2, int(span))
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _rsi(x, period=14):
    if len(x) <= period:
        return 50.0
    d = np.diff(x[-(period + 1):])
    up = d[d > 0].sum() / period
    dn = -d[d < 0].sum() / period
    if dn == 0:
        return 100.0 if up > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + up / dn)


def _atr(h, l, c, period=14):
    if len(c) < 2:
        return 0.0
    pc = c[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    n = min(period, len(tr))
    return float(np.mean(tr[-n:])) if n else 0.0


def _adx(h, l, c, period=14):
    if len(c) < period + 2:
        return 0.0, 0.0, 0.0
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    n = period
    atr = np.convolve(tr, np.ones(n) / n, "valid")
    pdi = 100.0 * np.convolve(plus_dm, np.ones(n) / n, "valid") / np.where(atr == 0, 1e-9, atr)
    mdi = 100.0 * np.convolve(minus_dm, np.ones(n) / n, "valid") / np.where(atr == 0, 1e-9, atr)
    dx = 100.0 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, 1e-9, pdi + mdi)
    adx = float(np.mean(dx[-n:])) if len(dx) >= n else float(np.mean(dx)) if len(dx) else 0.0
    return adx, float(pdi[-1]), float(mdi[-1])


def _linreg(x):
    n = len(x)
    if n < 5:
        return 0.0, 0.0
    t = np.arange(n, dtype=float)
    t -= t.mean()
    xc = x - x.mean()
    denom = (t * t).sum() or 1e-9
    slope = (t * xc).sum() / denom
    pred = slope * t
    ss_res = ((xc - pred) ** 2).sum()
    ss_tot = (xc * xc).sum() or 1e-9
    return slope / (x[-1] or 1e-9), 1.0 - ss_res / ss_tot


def _session_vwap(h, l, c, v):
    tp = (h + l + c) / 3.0
    vv = np.where(v <= 0, 1.0, v)
    return float((tp * vv).sum() / vv.sum())


def _obv_slope(c, v, look=15):
    if len(c) < look + 1:
        return 0.0
    sign = np.sign(np.diff(c))
    obv = np.cumsum(sign * v[1:])
    seg = obv[-look:]
    t = np.arange(len(seg), dtype=float)
    t -= t.mean()
    denom = (t * t).sum() or 1e-9
    slope = (t * (seg - seg.mean())).sum() / denom
    scale = np.mean(np.abs(v[-look:])) or 1e-9
    return slope / scale


def _swing_levels(h, l, S, atr, k=3, lookback=150):
    """Fractal swing highs/lows over the recent bars, clustered into horizontal
    price levels. A pivot high is a bar whose high tops the `k` bars on each
    side (pivot low: bottoms them). Nearby pivots merge into one level whose
    strength is the number of pivots that fed it.

    Returns (resistances, supports): lists of (price, strength), each sorted by
    distance from spot. "Levels are levels" — a broken former resistance that
    now sits below price is treated as support, and vice versa.
    """
    n = len(h)
    if n < 2 * k + 6 or atr <= 0:
        return [], []
    lo = max(k, n - lookback)
    pivots = []
    for i in range(lo, n - k):
        left_h, right_h = h[i - k:i], h[i + 1:i + k + 1]
        left_l, right_l = l[i - k:i], l[i + 1:i + k + 1]
        if h[i] > left_h.max() and h[i] >= right_h.max():
            pivots.append(h[i])
        if l[i] < left_l.min() and l[i] <= right_l.min():
            pivots.append(l[i])
    if not pivots:
        return [], []
    tol = _SR_CLUSTER_ATR * atr
    levels = []
    for p in sorted(pivots):
        if levels and abs(p - levels[-1][0]) <= tol:
            px, ct = levels[-1]
            levels[-1] = ((px * ct + p) / (ct + 1), ct + 1)
        else:
            levels.append((p, 1))
    band = 0.05 * atr
    res = sorted((lv for lv in levels if lv[0] > S + band), key=lambda x: x[0] - S)
    sup = sorted((lv for lv in levels if lv[0] < S - band), key=lambda x: S - x[0])
    return res, sup


def _time_of_day(now: datetime):
    m = now.hour * 60 + now.minute
    since = max(0.0, m - ET_OPEN_MIN)
    frac = min(1.0, since / SESSION_MIN)
    if since <= 60:
        bucket = 0            # opening drive
    elif m >= 14 * 60 + 30:
        bucket = 2            # power hour
    else:
        bucket = 1            # midday
    return since, frac, bucket


def build(symbol: str, bars: list[dict], all_bars: dict, now: datetime,
          minutes_left: float | None = None, daily: dict | None = None,
          vix: dict | None = None) -> dict:
    ctx = {"symbol": symbol}
    if not bars or len(bars) < 15:
        return {**ctx, **{k: 0.0 for k in SIGNAL_KEYS}, "ready": False}

    o = np.array([b["o"] for b in bars], float)
    h = np.array([b["h"] for b in bars], float)
    l = np.array([b["l"] for b in bars], float)
    c = np.array([b["c"] for b in bars], float)
    v = np.array([b.get("v", 0) or 0 for b in bars], float)
    S = c[-1]

    since_open, day_frac, tod_bucket = _time_of_day(now)
    ml = minutes_left if minutes_left is not None else max(1.0, SESSION_MIN * (1 - day_frac))

    # --- trend / momentum ---
    def ret(n):
        return (c[-1] / c[-1 - n] - 1.0) * 100.0 if len(c) > n else 0.0
    ef, es = _ema(c, 9), _ema(c, 21)
    ema_gap = (ef[-1] - es[-1]) / (es[-1] or 1e-9) * 100.0
    ema_slope = (ef[-1] - ef[-4]) / (ef[-4] or 1e-9) * 100.0 if len(ef) > 4 else 0.0
    m_fast, m_slow = _ema(c, 12), _ema(c, 26)
    macd = m_fast - m_slow
    macd_sig = _ema(macd, 9)
    macd_hist = (macd[-1] - macd_sig[-1]) / (S or 1e-9) * 100.0
    lr_slope, lr_r2 = _linreg(c[-30:])
    adx, pdi, mdi = _adx(h, l, c, 14)

    # --- mean reversion / bands ---
    rsi = _rsi(c, 14)
    win = c[-20:]
    sd = win.std() or 1e-9
    bb_mid = win.mean()
    bb_up, bb_dn = bb_mid + 2 * sd, bb_mid - 2 * sd
    bb_pctb = (S - bb_dn) / ((bb_up - bb_dn) or 1e-9)
    bb_bw = (bb_up - bb_dn) / (bb_mid or 1e-9)
    lo14 = l[-14:].min()
    hi14 = h[-14:].max()
    stoch_k = (S - lo14) / ((hi14 - lo14) or 1e-9) * 100.0
    williams_r = -100.0 * (hi14 - S) / ((hi14 - lo14) or 1e-9)
    tp = (h + l + c) / 3.0
    cci = (tp[-1] - tp[-20:].mean()) / (0.015 * (np.abs(tp[-20:] - tp[-20:].mean()).mean() or 1e-9))

    # --- volatility ---
    atr = _atr(h, l, c, 14)
    atr_pct = atr / (S or 1e-9) * 100.0
    rv30 = option_chain.realized_vol(list(c[-40:]))
    rv10 = option_chain.realized_vol(list(c[-12:]))
    day_hi, day_lo = h.max(), l.min()
    range_pct = (day_hi - day_lo) / (S or 1e-9) * 100.0

    # --- volume ---
    vwap = _session_vwap(h, l, c, v)
    dist_vwap = (S - vwap) / (vwap or 1e-9) * 100.0
    z_vwap = (S - vwap) / (sd or 1e-9)
    rvol = v[-1] / (np.mean(v[-20:]) or 1e-9)
    obv_sl = _obv_slope(c, v, 15)
    rets = np.sign(np.diff(c[-11:]))
    vol_delta = float((rets * v[-10:]).sum() / (np.sum(v[-10:]) or 1e-9))

    # --- session structure ---
    or_bars = [b for b in bars
               if _session_min(b) is not None and _session_min(b) <= OPENING_RANGE_MIN]
    if len(or_bars) >= 2:
        or_hi = max(b["h"] for b in or_bars)
        or_lo = min(b["l"] for b in or_bars)
    else:
        or_hi, or_lo = h[:5].max(), l[:5].min()
    or_rng = (or_hi - or_lo) or 1e-9
    or_break = (S - or_hi) / or_rng if S > or_hi else ((S - or_lo) / or_rng if S < or_lo else 0.0)

    daily = daily or {}
    prev_close = daily.get("prev_close")
    # the SESSION open, not the first bar of a rolling window (which, two
    # hours in, is just a bar from two hours ago): prefer the daily context's
    # value, then an actual 09:30 bar if the window still contains it
    today_open = daily.get("today_open")
    if not today_open:
        for b in bars:
            sm = _session_min(b)
            if sm is not None and 0 <= sm <= 1:
                today_open = float(b["o"])
                break
    if not today_open:
        today_open = o[0]
    gap_pct = (today_open - prev_close) / prev_close * 100.0 if prev_close else 0.0
    prev_close_dist = (S - prev_close) / prev_close * 100.0 if prev_close else 0.0
    pdh, pdl = daily.get("prev_high"), daily.get("prev_low")
    if pdh and pdl and pdh > pdl:
        pdh_pdl_pos = (S - (pdh + pdl) / 2.0) / ((pdh - pdl) / 2.0)
    else:
        pdh_pdl_pos = 0.0

    # --- support / resistance -------------------------------------------------
    # Swing-pivot levels from today's 1-min action, plus the obvious session
    # anchors (prior day H/L/close, opening-range H/L) folded in at strength 2.
    _atr_safe = atr or (S * 0.001) or 1e-9
    res_lv, sup_lv = _swing_levels(h, l, S, _atr_safe)
    for lvl in (pdh, pdl, prev_close, or_hi, or_lo):
        if not lvl:
            continue
        (res_lv if lvl > S else sup_lv).append((float(lvl), 2))
    res_lv.sort(key=lambda x: x[0] - S)
    sup_lv.sort(key=lambda x: S - x[0])

    res_px, res_touch = (res_lv[0] if res_lv else (None, 0))
    sup_px, sup_touch = (sup_lv[0] if sup_lv else (None, 0))
    res_dist_atr = ((res_px - S) / _atr_safe) if res_px else _SR_CAP_ATR
    sup_dist_atr = ((S - sup_px) / _atr_safe) if sup_px else _SR_CAP_ATR

    # room to run: resistance far + support near => bullish
    sr_room = res_dist_atr - sup_dist_atr
    # level breaks in the last few bars, weighted by how well-tested the level was
    ref = float(c[-5]) if len(c) > 5 else float(c[0])
    up_break = sum(ct for px, ct in (res_lv + sup_lv) if ref < px <= S)
    dn_break = sum(ct for px, ct in (res_lv + sup_lv) if S <= px < ref)
    sr_break = up_break - dn_break

    # --- cross asset ---
    other = "QQQ" if symbol == "SPY" else "SPY"
    ob = all_bars.get(other) or []
    rs_spread = 0.0
    if len(ob) > 15:
        oc = np.array([b["c"] for b in ob], float)
        rs_spread = ret(15) - (oc[-1] / oc[-16] - 1.0) * 100.0

    vix = vix or {}
    vix_lvl = vix.get("vix")
    vix9d = vix.get("vix9d")
    vix_open = vix.get("vix_open")
    vix_term = (vix9d / vix_lvl - 1.0) * 100.0 if (vix_lvl and vix9d) else 0.0
    vix_dir = (vix_lvl / vix_open - 1.0) * 100.0 if (vix_lvl and vix_open) else 0.0

    # --- options / regime ---
    atm_iv = max(0.05, min(1.5, rv30))
    expiry = now.replace(hour=16, minute=0, second=0, microsecond=0) if isinstance(now, datetime) else None
    ch = option_chain.build_chain(symbol, S, ml, atm_iv, True, expiry=expiry, now=now)
    chp = option_chain.build_chain(symbol, S, ml, atm_iv, False, expiry=expiry, now=now)
    call25 = min(ch, key=lambda r: abs(abs(r["delta"]) - 0.25), default=None)
    put25 = min(chp, key=lambda r: abs(abs(r["delta"]) - 0.25), default=None)
    skew = (put25["iv"] - call25["iv"]) * 100.0 if (call25 and put25) else 0.0
    atmc = min(ch, key=lambda r: abs(abs(r["delta"]) - 0.5), default=None)
    atmp = min(chp, key=lambda r: abs(abs(r["delta"]) - 0.5), default=None)
    straddle = ((atmc["price"] if atmc else 0) + (atmp["price"] if atmp else 0))
    expected_move_pct = straddle / (S or 1e-9) * 100.0
    move_so_far = abs(S - today_open) / (today_open or 1e-9) * 100.0
    move_budget = (move_so_far / expected_move_pct) if expected_move_pct else 0.0
    iv_rv = atm_iv / (rv30 or 1e-9)

    impact, data_label = market_calendar.classify(now.date() if isinstance(now, datetime) else date.today())

    ctx.update({
        "ready": True,
        # raw (for the named strategies + display)
        "spot": S, "vwap": vwap, "rsi": rsi, "adx": adx, "atr": atr,
        "atr_pct": atr_pct, "rv30": rv30, "rv10": rv10, "rv_ratio": rv10 / (rv30 or 1e-9),
        "bb_bandwidth": bb_bw, "range_pct": range_pct, "stoch_k": stoch_k,
        "williams_r": williams_r, "cci": cci, "lr_r2": lr_r2, "rvol": rvol,
        "dist_vwap_pct": dist_vwap, "gap_pct": gap_pct, "prev_close_dist": prev_close_dist,
        "or_high": or_hi, "or_low": or_lo, "skew": skew, "iv_atm": atm_iv,
        "res_level": res_px, "sup_level": sup_px,
        "res_dist_atr": res_dist_atr, "sup_dist_atr": sup_dist_atr,
        "res_touch": res_touch, "sup_touch": sup_touch,
        "expected_move_pct": expected_move_pct, "move_so_far_pct": move_so_far,
        "since_open_min": since_open, "tod_bucket": tod_bucket, "day_frac": day_frac,
        "pdi": pdi, "mdi": mdi, "data_label": data_label, "data_impact": impact,
        "vix": vix_lvl or 0.0, "vix9d": vix9d or 0.0,
        # oriented signal features (bullish-positive, ~[-2,2]) -> SIGNAL_KEYS
        "ret_15": _clip(ret(15) / 0.5),
        "ema_gap": _clip(ema_gap / 0.15),
        "ema_slope": _clip(ema_slope / 0.15),
        "macd_hist": _clip(macd_hist / 0.05),
        "adx_dir": _clip((1.0 if pdi >= mdi else -1.0) * min(1.0, adx / 30.0) * 2.0),
        "linreg_slope": _clip(lr_slope * 400.0),
        "rsi_dev": _clip((rsi - 50.0) / 20.0),
        "stoch_dev": _clip((stoch_k - 50.0) / 30.0),
        "bb_pctb": _clip((bb_pctb - 0.5) * 3.0),
        "z_vwap": _clip(z_vwap),
        "dist_vwap": _clip(dist_vwap / 0.3),
        "or_break": _clip(or_break * 2.0),
        "gap_fade": _clip(-gap_pct / 0.4),           # fade the opening gap
        "pdh_pdl_pos": _clip(pdh_pdl_pos),
        "rvol_dir": _clip(np.sign(ret(3)) * min(2.0, rvol - 1.0)),
        "obv_slope": _clip(obv_sl * 3.0),
        "vol_delta": _clip(vol_delta * 3.0),
        "rs_spread": _clip(rs_spread / 0.15),
        "vix_term": _clip(vix_term / 4.0),           # contango (calm) -> positive
        "vix_dir": _clip(-vix_dir / 3.0),            # vix falling -> bullish
        "iv_rv": _clip((1.0 - iv_rv) * 1.5),         # cheap vol -> positive
        "move_budget": _clip((1.0 - move_budget) * 1.2),  # room left to move -> positive
        "data_day": _clip(-impact * 1.5),            # data day -> risk-off bias
        "sr_room": _clip(sr_room / 2.0),             # headroom above vs floor below
        "sr_break": _clip(sr_break * 0.6),           # fresh level break, touch-weighted
    })
    return ctx


def _session_min(bar: dict):
    """Minutes since the 09:30 ET open for a bar's timestamp (None if unparseable)."""
    try:
        dt = datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.astimezone(ET)
        return (dt.hour * 60 + dt.minute) - ET_OPEN_MIN
    except (ValueError, KeyError, TypeError):
        return None
