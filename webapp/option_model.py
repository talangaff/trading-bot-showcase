"""Lightweight 0DTE option pricing + OCC symbol helpers for the sim engine.

Black-Scholes with r = q = 0 and a fixed per-bot IV. Good enough to give the
fleet a consistent, directionally-correct P&L surface (delta / gamma / theta)
to evolve strategies against — it is NOT a real quote.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
_YEAR_MINUTES = 365.0 * 24.0 * 60.0
CLOSE_HHMM = (16, 0)          # 0DTE options settle at the 16:00 ET cash close


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def minutes_to_close(now: datetime | None = None) -> float:
    """Minutes until TODAY's session ends — 16:00 ET normally, 13:00 on an
    early-close half day. Reading the real close matters: on a half day a
    fixed 16:00 overstates a 0DTE contract's remaining life by three hours,
    which inflates every modelled price and every greek the fleet trades on."""
    from . import market_calendar
    now = now or datetime.now(ET)
    now = now.astimezone(ET)
    close = market_calendar.session_close_dt(now.date())
    return max(0.0, (close - now).total_seconds() / 60.0)


def minutes_to_expiry(expiry: datetime, now: datetime | None = None) -> float:
    """Minutes from `now` to `expiry`'s 16:00 ET close — the general form of
    minutes_to_close() for a contract whose expiry isn't necessarily today
    (a swing position). For a same-day expiry the two are identical."""
    now = now or datetime.now(ET)
    now = now.astimezone(ET)
    close = expiry.astimezone(ET).replace(hour=CLOSE_HHMM[0], minute=CLOSE_HHMM[1],
                                          second=0, microsecond=0)
    return max(0.0, (close - now).total_seconds() / 60.0)


def bs_price(S: float, K: float, minutes_left: float, iv: float,
             is_call: bool) -> float:
    """Per-share option price. Multiply by 100 for a contract."""
    S = max(S, 1e-6)
    K = max(K, 1e-6)
    iv = max(iv, 1e-4)
    T = max(minutes_left, 0.0) / _YEAR_MINUTES
    if T <= 0.0:
        intrinsic = (S - K) if is_call else (K - S)
        return max(0.0, intrinsic)
    vol = iv * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / vol
    d2 = d1 - vol
    if is_call:
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bid_ask(mid: float, minutes_left: float) -> tuple[float, float]:
    """Model a realistic 0DTE bid/ask spread around a theoretical mid price.

    There's no real quote feed in the sim (see option_chain.py), so every
    fleet fill was previously booked at the frictionless BS mid — flattering
    every bot's P&L versus what it would actually pay/receive. 0DTE spreads
    widen sharply as expiry nears and as premium gets cheap (a fixed minimum
    tick eats a bigger share of a thin option), so approximate both effects
    and pick the fill on the worse side for whoever is trading.
    """
    mid = max(mid, 0.01)
    time_frac = 0.06 + 0.14 * math.exp(-max(minutes_left, 0.0) / 120.0)   # ~6%..20%
    cheap_frac = 0.03 / mid                                              # extra for sub-$1 options
    spread_frac = min(0.35, time_frac + cheap_frac)
    half = mid * spread_frac / 2.0
    bid = max(0.01, round(mid - half, 2))
    ask = max(bid + 0.01, round(mid + half, 2))
    return bid, ask


def entry_fill_price(mid: float, minutes_left: float) -> float:
    """What a Buy-to-Open actually pays: the modeled ask."""
    return bid_ask(mid, minutes_left)[1]


def exit_fill_price(mid: float, minutes_left: float) -> float:
    """What a Sell-to-Close actually receives: the modeled bid."""
    return bid_ask(mid, minutes_left)[0]


def bs_delta(S: float, K: float, minutes_left: float, iv: float,
             is_call: bool) -> float:
    S = max(S, 1e-6)
    K = max(K, 1e-6)
    iv = max(iv, 1e-4)
    T = max(minutes_left, 0.0) / _YEAR_MINUTES
    if T <= 0.0:
        itm = (S > K) if is_call else (S < K)
        return (1.0 if is_call else -1.0) if itm else 0.0
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def implied_vol(price: float, S: float, K: float, minutes_left: float,
                is_call: bool, lo: float = 0.01, hi: float = 5.0) -> float | None:
    """Black-Scholes implied vol by bisection. None when `price` is at or
    below intrinsic (no time value to solve for), above the underlying, or
    the contract has expired."""
    T = max(minutes_left, 0.0) / _YEAR_MINUTES
    if T <= 0.0 or price <= 0.0 or price >= S:
        return None
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price <= intrinsic + 1e-4:
        return None
    if bs_price(S, K, minutes_left, hi, is_call) < price:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, minutes_left, mid, is_call) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return round(0.5 * (lo + hi), 5)


# --- OCC-style symbol: SPY260904C00560000 -------------------------------------
def occ_symbol(underlying: str, expiry: datetime, is_call: bool,
               strike: float) -> str:
    return "{}{}{}{:08d}".format(
        underlying.upper(),
        expiry.astimezone(ET).strftime("%y%m%d"),
        "C" if is_call else "P",
        int(round(strike * 1000)),
    )


def spread_symbol(underlying: str, expiry: datetime, is_call: bool,
                  long_strike: float, short_strike: float) -> str:
    """A vertical DEBIT spread as one bookable instrument: the long leg's OCC
    symbol plus '-' and the short leg's 8-digit strike, e.g.
    SPY260904C00560000-00563000 (long the 560 call, short the 563 call).
    The ledger books it as a single buy-to-open / sell-to-close package
    whose price is the net debit — max loss is that debit, so it keeps the
    fleet's no-sell-to-open rule's meaning (no undefined risk) while
    cutting the theta and spread paid on a naked long option."""
    return "{}-{:08d}".format(occ_symbol(underlying, expiry, is_call, long_strike),
                              int(round(short_strike * 1000)))


def parse_occ(sym: str) -> dict | None:
    """Parse an OCC symbol, or a spread_symbol(). The result always has
    underlying / expiry / is_call / strike (the long leg); a spread adds
    short_strike (else None) and width."""
    sym = (sym or "").strip().upper()
    # find the 6-digit date: underlying is the leading alpha run
    i = 0
    while i < len(sym) and sym[i].isalpha():
        i += 1
    if i == 0 or len(sym) - i < 15:
        return None
    try:
        under = sym[:i]
        date_s = sym[i:i + 6]
        cp = sym[i + 6]
        strike = int(sym[i + 7:i + 15]) / 1000.0
        exp = datetime.strptime(date_s, "%y%m%d").replace(tzinfo=ET)
        rest = sym[i + 15:]
        short = None
        if rest:
            if not (rest.startswith("-") and len(rest) == 9 and rest[1:].isdigit()):
                return None
            short = int(rest[1:]) / 1000.0
        return {"underlying": under, "expiry": exp, "is_call": cp == "C", "strike": strike,
                "short_strike": short,
                "width": (abs(short - strike) if short is not None else None)}
    except (ValueError, IndexError):
        return None


def spread_price(S: float, long_k: float, short_k: float, minutes_left: float,
                 iv_long: float, iv_short: float, is_call: bool) -> float:
    """Per-share value of a vertical debit spread (long_k long, short_k short)."""
    return max(0.0, bs_price(S, long_k, minutes_left, iv_long, is_call)
               - bs_price(S, short_k, minutes_left, iv_short, is_call))


def intrinsic(info: dict, S: float) -> float:
    """Settlement value per share at expiry for a parsed contract — a naked
    option's intrinsic, or a vertical's intrinsic capped at its width."""
    k = info["strike"]
    val = max(0.0, (S - k) if info["is_call"] else (k - S))
    if info.get("short_strike") is not None:
        val = min(val, info["width"] or 0.0)
    return val
