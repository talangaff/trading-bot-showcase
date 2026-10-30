"""A 0DTE option chain, real quotes blended over a modeled fallback, + the
bot's strike-selection logic.

Every strike is always synthesised around spot with plausible shapes first
(see option_quotes.py's docstring for why this fallback still matters even
with a live feed):
  * IV per strike: an equity-style skew (puts richer) plus a mild smile,
    anchored to an ATM IV the caller derives from recent realized vol.
  * volume per strike: a Gaussian in distance-from-ATM, bumped at round
    strikes — a liquidity signal the bot can filter on.
  * price / delta: Black-Scholes at that strike's IV.

When build_chain() is given an `expiry`, it then asks option_quotes.py for
real bid/ask/volume for that (symbol, expiry) and overwrites any strike a
provider actually quoted — the model only fills the gaps (illiquid/far
strikes, or a bad data day). Each row's 'source' says which happened.

pick_strike() is where a bot actually decides: it filters the chain by the
bot's liquidity / IV / affordability limits, then ranks what's left by the
genome's strike_style (target delta, % OTM, premium as a share of budget, or
just the most contracts it can buy).
"""

import math
from datetime import datetime, timezone

import numpy as np

from . import config, option_model, option_quotes

MAX_QUOTE_AGE_S = float(getattr(config, "BOT_MAX_QUOTE_AGE_S", 60.0))

STEP = 1.0                      # SPY / QQQ strikes are $1 apart
N_STRIKES = 9                   # each side of ATM
_YEAR_MIN = 365.0 * 24.0 * 60.0


def realized_vol(closes: list[float], bars_per_year: float = 252 * 390) -> float:
    """Annualised close-to-close vol from recent 1-min bars."""
    if not closes or len(closes) < 6:
        return 0.20
    x = np.asarray(closes[-40:], dtype=float)
    r = np.diff(np.log(x))
    sd = float(np.std(r))
    return max(0.05, min(1.5, sd * math.sqrt(bars_per_year)))


def _strike_iv(atm_iv: float, S: float, K: float, is_call: bool) -> float:
    m = (K - S) / S                       # + = OTM call side / ITM put side
    skew = -0.9 * m                       # puts (K<S) get higher iv
    smile = 2.5 * m * m
    return max(0.05, atm_iv * (1.0 + skew + smile))


def _volume(S: float, K: float) -> int:
    dist = abs(K - S) / S
    base = 6000.0 * math.exp(-(dist / 0.012) ** 2)
    if round(K) % 5 == 0:
        base *= 1.6
    return int(round(base))


def _plausible(q: dict, model_price: float) -> bool:
    """Guard against trusting a corrupted/garbage quote: real quotes are
    EXPECTED to differ from the model (that's the whole point of pulling
    them), so this band is deliberately generous — it only catches a quote
    that's structurally broken (crossed or inverted market, an unusably
    wide spread) or off from the model by an order of magnitude (a decimal
    error, a stale symbol collision, wrong contract multiplier, etc.), not
    ordinary real-vs-modeled disagreement."""
    bid, ask = q.get("bid"), q.get("ask")
    if not bid or not ask or bid <= 0 or ask <= bid:
        return False
    mid = (bid + ask) / 2.0
    if (ask - bid) / mid > 0.9:                          # spread eating ~all the premium
        return False
    if model_price > 0.02 and not (0.1 <= mid / model_price <= 10.0):
        return False
    return True


def quote_age_s(q: dict, now: datetime | None = None) -> float | None:
    """Seconds since the quote's own timestamp; None if it has none."""
    t = q.get("t")
    if not t:
        return None
    try:
        qt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if qt.tzinfo is None:
        qt = qt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - qt).total_seconds())


def _usable_quote(q: dict, S: float, K: float, minutes_left: float, is_call: bool,
                  model_price: float, symbol: str, now: datetime | None) -> dict | None:
    """Turn a real quote into something safe to fill against right now.

    FRESH (age <= MAX_QUOTE_AGE_S and not a delayed feed): the bid/ask are
    used as-is, and the quote's implied vol at the live spot replaces the
    modeled IV when it solves.

    STALE (older, a delayed feed, or no timestamp at all): a 15-minute-old
    mid against a spot that has since moved is a leak — a bot could "buy"
    a call at a price from before the rally. So the stale quote is used
    only to CALIBRATE: solve its implied vol against the spot it was quoted
    at (q['spot'], falling back to the live spot), keep its relative spread
    width, then reprice both at the live spot and the current time. Source
    is suffixed '-calibrated'. None when the quote is structurally broken
    or its IV doesn't solve — the caller falls back to the pure model."""
    if not _plausible(q, model_price):
        return None
    bid_q, ask_q = float(q["bid"]), float(q["ask"])
    mid_q = (bid_q + ask_q) / 2.0
    age = quote_age_s(q, now)
    fresh = (age is not None and age <= MAX_QUOTE_AGE_S and not q.get("delayed"))
    src = option_quotes.status(symbol)
    if fresh:
        iv = option_model.implied_vol(mid_q, S, K, minutes_left, is_call)
        return {"bid": bid_q, "ask": ask_q, "price": round(mid_q, 3), "iv": iv,
                "volume": q.get("volume") or 0, "source": src, "age_s": age}
    spot_q = float(q.get("spot") or 0.0) or S
    minutes_q = minutes_left + (age or 0.0) / 60.0
    iv = option_model.implied_vol(mid_q, spot_q, K, minutes_q, is_call)
    if iv is None:
        return None
    price = option_model.bs_price(S, K, minutes_left, iv, is_call)
    if price < 0.02:
        return None
    sf = (ask_q - bid_q) / mid_q
    bid = max(0.01, round(price * (1 - sf / 2.0), 2))
    ask = max(bid + 0.01, round(price * (1 + sf / 2.0), 2))
    return {"bid": bid, "ask": ask, "price": round(price, 3), "iv": iv,
            "volume": q.get("volume") or 0, "source": src + "-calibrated", "age_s": age}


def build_chain(symbol: str, S: float, minutes_left: float, atm_iv: float,
                is_call: bool, expiry: datetime | None = None,
                now: datetime | None = None) -> list[dict]:
    """`expiry` is optional so every existing caller/test keeps working with
    a purely modeled chain; bot_engine passes it (the 0DTE 16:00 ET close)
    so real quotes get a chance to override the model strike by strike.
    `now` is what a quote's age is measured against (the replay harness
    passes the recorded minute; live callers pass the tick time)."""
    atm = round(S / STEP) * STEP
    live = {}
    if expiry is not None:
        live = option_quotes.get_chain(
            symbol, expiry.date(), atm - N_STRIKES * STEP, atm + N_STRIKES * STEP)

    out = []
    for i in range(-N_STRIKES, N_STRIKES + 1):
        K = atm + i * STEP
        if K <= 0:
            continue
        iv = _strike_iv(atm_iv, S, K, is_call)
        price = option_model.bs_price(S, K, minutes_left, iv, is_call)
        if price < 0.02:
            continue
        bid, ask = option_model.bid_ask(price, minutes_left)
        volume = _volume(S, K)
        source = "model"

        occ = option_model.occ_symbol(symbol, expiry, is_call, K) if expiry is not None else None
        q = live.get(occ) if occ else None
        u = _usable_quote(q, S, K, minutes_left, is_call, price, symbol, now) if q else None
        # A real quote that _usable_quote refused (crossed, unusably wide, or
        # an order of magnitude off the model) is not safe to FILL against —
        # but it does not make the MODEL right either, and on a thin strike
        # the model is the half more likely to be wrong. Sizing off the model
        # there is how a bot bought 44 contracts of something it priced at
        # $0.06 and Alpaca charged $1.11 for.
        #
        # So the strike stays tradeable and keeps its modeled greeks, but the
        # ASK that affordability and sizing are computed from is raised to
        # whichever of the two readings is dearer. The bot then buys the
        # quantity the expensive reading affords: right if the market is
        # right, merely conservative if the quote was the corrupt one. It
        # cannot be unsafe either way, because _record_fill bounds the order
        # with a limit at the position cap whatever the sizing believed.
        conflict = bool(q) and not u
        if u:
            bid, ask, price = u["bid"], u["ask"], u["price"]   # real (or calibrated) mid replaces the BS mid
            volume = u["volume"] or volume           # a quote with no trade volume yet keeps the modeled estimate
            source = u["source"]                     # 'alpaca', 'yfinance-calibrated', ...
            if u["iv"]:
                iv = u["iv"]                         # the market's implied vol, when it solves
        elif conflict:
            q_ask = float(q.get("ask") or 0.0)
            if q_ask > ask:
                ask = q_ask                # size on the dearer of the two readings

        out.append({
            "strike": K,
            "is_call": is_call,
            "iv": round(iv, 4),             # modeled, or solved from a usable real quote
            "price": round(price, 3),
            "bid": bid,
            "ask": ask,                     # what a Buy-to-Open actually pays
            "delta": round(option_model.bs_delta(S, K, minutes_left, iv, is_call), 3),
            "volume": volume,
            "dist_pct": round((K - S) / S * 100.0, 3),
            "source": source,
            "quote_conflict": conflict,
        })
    return out


def live_quote(symbol: str, S: float, expiry: datetime, strike: float,
               is_call: bool, model_price: float,
               now: datetime | None = None) -> dict | None:
    """Real bid/ask/volume for one specific contract already in a position —
    the same per-strike lookup build_chain() does, for _manage_position()'s
    exit pricing. `model_price` is the BS price for this same contract, used
    only for the same sanity band build_chain() applies (see _plausible()) —
    real disagreement with the model is expected and fine, a structurally
    broken or wildly-off quote is not trusted. None if no provider has this
    strike, or what they have doesn't pass the sanity check (caller falls
    back to the modeled bid/ask around the BS price either way)."""
    atm = round(S / STEP) * STEP
    live = option_quotes.get_chain(
        symbol, expiry.date(), atm - N_STRIKES * STEP, atm + N_STRIKES * STEP)
    occ = option_model.occ_symbol(symbol, expiry, is_call, strike)
    q = live.get(occ)
    if not q:
        return None
    minutes_left = option_model.minutes_to_expiry(expiry, now)
    return _usable_quote(q, S, strike, minutes_left, is_call, model_price, symbol, now)


def _model_row(S: float, K: float, minutes_left: float, atm_iv: float, is_call: bool) -> dict:
    iv = _strike_iv(atm_iv, S, K, is_call)
    price = option_model.bs_price(S, K, minutes_left, iv, is_call)
    bid, ask = option_model.bid_ask(max(price, 0.01), minutes_left)
    return {"strike": K, "is_call": is_call, "iv": round(iv, 4), "price": round(price, 3),
            "bid": bid, "ask": ask, "delta": round(option_model.bs_delta(S, K, minutes_left, iv, is_call), 3),
            "volume": _volume(S, K), "dist_pct": round((K - S) / S * 100.0, 3), "source": "model"}


def vertical_row(chain: list[dict], long_row: dict, width_strikes: int, S: float,
                 minutes_left: float, atm_iv: float) -> dict | None:
    """The chain row for a vertical DEBIT spread built on `long_row`: short
    the strike `width_strikes` further out of the money (above for calls,
    below for puts). Net price = long mid - short mid; the package's bid is
    long bid - short ask and its ask is long ask - short bid, i.e. it pays
    both legs' spreads; delta/iv are the net / long-leg values. The short
    leg comes from the chain when it's there (real quote and all), else it
    is modeled. None if the spread would have no value."""
    is_call = long_row["is_call"]
    k_short = long_row["strike"] + (1 if is_call else -1) * max(1, int(width_strikes)) * STEP
    if k_short <= 0:
        return None
    short = next((r for r in chain if abs(r["strike"] - k_short) < 1e-9), None)
    if short is None:
        short = _model_row(S, k_short, minutes_left, atm_iv, is_call)
    price = max(0.0, long_row["price"] - short["price"])
    if price < 0.02:
        return None
    bid = max(0.01, round(long_row["bid"] - short["ask"], 2))
    ask = max(bid + 0.01, round(long_row["ask"] - short["bid"], 2))
    width = abs(k_short - long_row["strike"])
    return {
        "strike": long_row["strike"], "short_strike": k_short, "width": width,
        "is_call": is_call, "iv": long_row["iv"], "iv_short": short["iv"],
        "price": round(min(price, width), 3),
        "bid": min(bid, width), "ask": min(ask, width),
        "delta": round(long_row["delta"] - short["delta"], 3),
        "volume": min(long_row["volume"], short["volume"]),
        "dist_pct": long_row["dist_pct"],
        "source": long_row["source"] if long_row["source"] == short["source"]
                  else "{}+{}".format(long_row["source"], short["source"]),
    }


def _row_value(row: dict, S: float, minutes_left: float, iv: float) -> float:
    """Model value of a chain row (single or vertical) at `iv` for the long leg."""
    if row.get("short_strike") is not None:
        return option_model.spread_price(S, row["strike"], row["short_strike"], minutes_left,
                                         iv, row.get("iv_short") or iv, row["is_call"])
    return option_model.bs_price(S, row["strike"], minutes_left, iv, row["is_call"])


def spread_mark(symbol: str, S: float, info: dict, minutes_left: float, iv_long: float,
                expiry: datetime, now: datetime | None = None) -> dict:
    """Current mark + fill prices for an OPEN vertical (parse_occ info with a
    short_strike). Each leg uses its real quote when one is usable
    (option_chain.live_quote, fresh-or-calibrated) and the model otherwise;
    the package bid/ask pay both legs' spreads. Returns {price, bid, ask,
    source} — the shape _manage_position() consumes as `live_q`."""
    is_call = info["is_call"]
    k_l, k_s = info["strike"], info["short_strike"]
    skew = _strike_iv(1.0, S, k_s, is_call) / (_strike_iv(1.0, S, k_l, is_call) or 1e-9)
    iv_short = max(0.05, iv_long * skew)
    legs = []
    for k, iv in ((k_l, iv_long), (k_s, iv_short)):
        model = option_model.bs_price(S, k, minutes_left, iv, is_call)
        q = live_quote(symbol, S, expiry, k, is_call, model, now=now)
        if q:
            legs.append((q["price"], q["bid"], q["ask"], q["source"]))
        else:
            b, a = option_model.bid_ask(max(model, 0.01), minutes_left)
            legs.append((model, b, a, "model"))
    (pl, bl, al, sl), (ps, bs_, as_, ss) = legs
    width = abs(k_s - k_l)
    price = max(0.0, min(width, pl - ps))
    bid = max(0.01, min(width, round(bl - as_, 2)))
    ask = max(bid + 0.01, min(width, round(al - bs_, 2)))
    return {"price": round(price, 3), "bid": bid, "ask": ask,
            "source": sl if sl == ss else "{}+{}".format(sl, ss)}


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def cost_gate(row: dict, S: float, hold_minutes: float, minutes_left: float,
              commission_per_contract: float, mult: float = 1.0) -> dict:
    """Fleet-wide, non-evolvable entry filter: is there enough expected move
    in the planned hold to pay for the trade at all?

      expected gain = |delta| x E|dS| over the hold, with E|dS| = S x iv x
                      sqrt(T_hold) x sqrt(2/pi) at the strike's own IV
      certain cost  = (ask - bid)                       full spread, paid round trip
                    + BS(now) - BS(now - hold)          theta bleed over the hold
                    + 2 x commission / 100              per share, both legs

    Passes when expected gain >= mult x certain cost. This is a NECESSARY
    condition, not a sufficient one — it assumes the direction is right,
    so it only removes trades that lose even when the signal is correct:
    the last-hour lottery ticket, the sub-$0.20 strike whose spread is the
    whole premium, the 5-minute hold that can't outrun theta."""
    iv = max(1e-4, float(row.get("iv") or 0.0))
    hold = max(1.0, min(float(hold_minutes), float(minutes_left)))
    t_hold = hold / _YEAR_MIN
    exp_move = S * iv * math.sqrt(t_hold) * _SQRT_2_OVER_PI
    gain = abs(float(row.get("delta") or 0.0)) * exp_move

    spread = max(0.0, float(row["ask"]) - float(row["bid"]))
    now_px = _row_value(row, S, minutes_left, iv)
    later_px = _row_value(row, S, minutes_left - hold, iv)
    theta = max(0.0, now_px - later_px)      # a vertical's net theta is far smaller than a naked long's
    comm = 2.0 * float(commission_per_contract) / 100.0
    cost = spread + theta + comm
    ratio = (gain / cost) if cost > 1e-9 else float("inf")
    return {"ok": mult <= 0 or ratio >= mult, "ratio": round(ratio, 3),
            "gain": round(gain, 4), "cost": round(cost, 4), "spread": round(spread, 4),
            "theta": round(theta, 4), "commission": round(comm, 4),
            "hold_minutes": round(hold, 1)}


def pick_strike(chain: list[dict], g: dict, budget: float,
                min_contracts: int, mult: float = 100.0) -> dict | None:
    """Return the chosen strike row, or None if nothing clears the bot's
    filters (too illiquid / too rich / unaffordable at the minimum size)."""
    if not chain or budget <= 0:
        return None
    min_vol = g.get("min_option_volume", 0) or 0
    max_iv = g.get("max_iv", 1.0) or 1.0
    style = g.get("strike_style", "delta")

    # affordability is checked against the ask — what Buy-to-Open actually
    # pays — not the frictionless mid, so a bot can't size a position it
    # can't really afford once the spread is paid
    cands = [
        r for r in chain
        if r["volume"] >= min_vol
        and r["iv"] <= max_iv
        and r["ask"] * min_contracts * mult <= budget + 1e-6
    ]
    if not cands:
        return None

    if style == "delta":
        tgt = g.get("target_delta", 0.30)
        key = lambda r: (abs(abs(r["delta"]) - tgt), -r["volume"])
    elif style == "distance":
        tgt = g.get("otm_pct", 0.0)
        # for puts, "OTM" is below spot -> compare on signed distance in the
        # option's own OTM direction
        side = 1.0 if chain[0]["is_call"] else -1.0
        key = lambda r: (abs(side * r["dist_pct"] - tgt), -r["volume"])
    elif style == "premium":
        tgt = g.get("premium_frac", 0.5) * budget
        key = lambda r: (abs(r["ask"] * min_contracts * mult - tgt), -r["volume"])
    else:  # cheapest — most contracts for the money, tie-break liquidity
        key = lambda r: (r["ask"], -r["volume"])

    return sorted(cands, key=key)[0]
