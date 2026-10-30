"""Real option quotes (0DTE and swing) to blend into the modeled chain
(option_chain.py), plus real listed-expiration lookup for swing entries.

Every fill used to be priced off a pure Black-Scholes model with no link to
what the market was actually quoting. get_chain() pulls real bid/ask/volume
per contract — Alpaca's options data API first (uses the same paper key
entered on the Bot Fleet page), falling back to yfinance's (delayed) chain
when Alpaca has nothing for a symbol/expiry. Keyed by the exact OCC symbol
(option_model.occ_symbol()) so option_chain.build_chain() can drop real
quotes straight onto its modeled strike ladder, strike by strike.

list_expirations()/pick_expiry() answer a different question: not "what's
this contract quoted at" but "what expiration dates for this symbol
actually exist to trade" — needed because a swing bot's target DTE (say,
"14 days out") isn't automatically a real listed expiry the way 0DTE's
"today" always is for SPY/QQQ.

Alpaca's default 'indicative' options feed is Alpaca's own theoretical
quote, not true exchange NBBO, unless the account has an OPRA subscription
— we can't detect which one a given key has, so 'indicative' is requested
and used as reachable real-time data when it responds. yfinance's chain is
genuine exchange data but typically 15-20 minutes delayed. Either beats the
frictionless model on its own; if both fail, option_chain.py's synthetic
chain is the exhaust-all fallback so the engine still runs.
"""

import time
from datetime import date, datetime, timedelta, timezone

import requests

from . import api_meter, bot_settings, recorder

# yfinance's chain is exchange data but ~15-20 minutes delayed; it carries no
# quote timestamp, so every yfinance quote is stamped this far in the past
YF_DELAY_S = 15 * 60

# Replay override (webapp/replay.py): when set, get_chain()/list_expirations()
# answer from these callables instead of the network + cache, so a recorded
# session can be replayed through exactly the live code path.
_provider: dict | None = None


def set_provider(quotes_fn=None, expirations_fn=None) -> None:
    """quotes_fn(symbol, expiry, low, high) -> {occ: quote};
    expirations_fn(symbol) -> [date, ...]. Pass nothing to clear."""
    global _provider
    _provider = {"quotes": quotes_fn, "expirations": expirations_fn} if (quotes_fn or expirations_fn) else None


_CACHE: dict[tuple, tuple[float, dict, str]] = {}   # (symbol, expiry) -> (ts, quotes, source)
_TTL = 20.0                      # seconds — 0DTE quotes move fast, but shared across the whole fleet
_ALPACA_OPTIONS = "https://data.alpaca.markets/v1beta1/options/snapshots/{}"

_last_source: dict[str, str] = {}   # symbol -> human label of the last source that returned data


def _from_alpaca(symbol: str, expiry: date, low: float, high: float,
                 key: str, secret: str) -> dict:
    out = {}
    params = {
        "feed": "indicative",
        "limit": 1000,
        "expiration_date": expiry.isoformat(),
        "strike_price_gte": round(low, 2),
        "strike_price_lte": round(high, 2),
    }
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    page_token = None
    for _ in range(5):                     # hard cap — a 0DTE chain never needs 5 pages
        if page_token:
            params["page_token"] = page_token
        r = api_meter.get(_ALPACA_OPTIONS.format(symbol), keys=(key, secret),
                          params=params, headers=hdr, timeout=8)
        r.raise_for_status()
        j = r.json()
        for occ, snap in (j.get("snapshots") or {}).items():
            q = snap.get("latestQuote") or {}
            bid, ask = q.get("bp"), q.get("ap")
            if not bid or not ask or bid <= 0 or ask <= 0:
                continue
            trade = snap.get("latestTrade") or {}
            out[occ] = {"bid": float(bid), "ask": float(ask),
                       "last": float(trade.get("p") or 0.0) or None,
                       "volume": int(trade.get("s") or 0),
                       "t": q.get("t")}            # quote timestamp — its age decides fill vs calibrate
        page_token = j.get("next_page_token")
        if not page_token:
            break
    return out


def _from_yfinance(symbol: str, expiry: date) -> dict:
    import yfinance as yf
    tk = yf.Ticker(symbol)
    exp_s = expiry.isoformat()
    if exp_s not in (tk.options or ()):
        return {}
    chain = tk.option_chain(exp_s)
    out = {}
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=YF_DELAY_S)).isoformat()
    for df in (chain.calls, chain.puts):
        for _, row in df.iterrows():
            bid, ask = row.get("bid"), row.get("ask")
            if not bid or not ask or bid <= 0 or ask <= 0:
                continue
            out[row["contractSymbol"]] = {
                "bid": float(bid), "ask": float(ask),
                "last": float(row["lastPrice"]) if row.get("lastPrice") else None,
                "volume": int(row["volume"]) if row.get("volume") == row.get("volume") else 0,
                "t": stamp, "delayed": True,
            }
    return out


def get_chain(symbol: str, expiry: date, low: float, high: float) -> dict:
    """{occ_symbol: {bid, ask, last, volume}} for every real quote available
    between `low` and `high` strikes for `symbol`'s `expiry`. Empty (never
    raises) when neither provider has anything — callers fall back to the
    modeled chain."""
    symbol = symbol.upper()
    if _provider and _provider.get("quotes"):
        q = _provider["quotes"](symbol, expiry, low, high) or {}
        _last_source[symbol] = (next(iter(q.values())).get("source") or "recorded") if q else "none"
        return q

    key_ = (symbol, expiry)
    now = time.time()
    hit = _CACHE.get(key_)
    if hit and now - hit[0] < _TTL:
        return hit[1]

    quotes: dict = {}
    source = "none"
    # one fetch serves every user's fleet, on the best entitlement anyone
    # has connected, and is recorded once into the shared history below
    src = bot_settings.data_source()
    try:
        if src["keys"]:
            quotes = _from_alpaca(symbol, expiry, low, high, src["keys"][0], src["keys"][1])
            if quotes:
                source = "alpaca"
        if not quotes:
            quotes = _from_yfinance(symbol, expiry)
            if quotes:
                source = "yfinance"
    except Exception as e:  # noqa: BLE001 — a bad quote day must never stall the engine
        print("option_quotes {} {}: {}".format(symbol, expiry, e), flush=True)
        quotes = hit[1] if hit else {}
        source = _last_source.get(symbol, "none")

    if quotes:
        # the underlying at (about) the time of the quote — what a stale quote
        # gets calibrated against, so a 15-minute-old call mid isn't solved
        # for an implied vol at a spot that has since moved
        try:
            from . import market_data
            spot = market_data.last_price(symbol)
        except Exception:  # noqa: BLE001
            spot = None
        for q in quotes.values():
            q.setdefault("spot", spot)
    _last_source[symbol] = source
    _CACHE[key_] = (now, quotes, source)
    if quotes and source != "none":
        # feeds the shared replay history every user's fleet evolves against,
        # credited to whoever's entitlement served the fetch
        recorder.record_quotes(symbol, expiry, quotes, source, user_id=src["user_id"])
    return quotes


_EXP_CACHE: dict[str, tuple[float, list]] = {}      # symbol -> (ts, [date, ...])
_EXP_TTL = 3600.0                # listed expirations don't change intraday
_ALPACA_CONTRACTS = "https://paper-api.alpaca.markets/v2/options/contracts"


def _expirations_from_alpaca(symbol: str, key: str, secret: str) -> list[date]:
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    out = set()
    page_token = None
    for _ in range(10):
        params = {"underlying_symbols": symbol, "status": "active", "limit": 500}
        if page_token:
            params["page_token"] = page_token
        r = api_meter.get(_ALPACA_CONTRACTS, keys=(key, secret),
                          params=params, headers=hdr, timeout=8)
        r.raise_for_status()
        j = r.json()
        for c in j.get("option_contracts") or []:
            try:
                out.add(date.fromisoformat(c["expiration_date"]))
            except (KeyError, ValueError, TypeError):
                continue
        page_token = j.get("next_page_token")
        if not page_token:
            break
    return sorted(out)


def _expirations_from_yfinance(symbol: str) -> list[date]:
    import yfinance as yf
    out = []
    for s in (yf.Ticker(symbol).options or ()):
        try:
            out.append(date.fromisoformat(s))
        except ValueError:
            continue
    return sorted(out)


def list_expirations(symbol: str) -> list[date]:
    """Real listed expiration dates for `symbol`, soonest first — Alpaca's
    options-contracts reference endpoint first, yfinance's Ticker.options
    fallback. Cached an hour (listed expirations don't change intraday).
    Empty (never raises) if both fail — pick_expiry()/callers fall back to
    a synthetic date rather than blocking a swing entry entirely."""
    symbol = symbol.upper()
    if _provider and _provider.get("expirations"):
        return list(_provider["expirations"](symbol) or [])
    now = time.time()
    hit = _EXP_CACHE.get(symbol)
    if hit and now - hit[0] < _EXP_TTL:
        return hit[1]

    exps: list[date] = []
    src = bot_settings.data_source()
    try:
        if src["keys"]:
            exps = _expirations_from_alpaca(symbol, src["keys"][0], src["keys"][1])
        if not exps:
            exps = _expirations_from_yfinance(symbol)
    except Exception as e:  # noqa: BLE001 — a bad data day must never stall the engine
        print("option_quotes list_expirations {}: {}".format(symbol, e), flush=True)
        exps = hit[1] if hit else []

    _EXP_CACHE[symbol] = (now, exps)
    return exps


def pick_expiry(symbol: str, target_dte: int, today: date | None = None) -> date | None:
    """The real listed expiration closest to `target_dte` calendar days out
    (never today or in the past — a swing entry needs at least one full day
    of remaining life). None if no listed expirations are available at all;
    the caller (bot_engine._step_bot) falls back to a synthetic
    today + target_dte date so a data outage doesn't block swing entries,
    it just makes that one entry's contract a modeled approximation.
    `today` lets the replay harness ask as of a recorded session."""
    today = today or date.today()
    candidates = [e for e in list_expirations(symbol) if (e - today).days >= 1]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - today).days - target_dte))


def status(symbol: str) -> str:
    """Last known quote source for `symbol` — 'alpaca', 'yfinance', or
    'none' (nothing fetched yet, or every provider has failed/is empty).
    For option_chain.py; never triggers a fetch itself."""
    return _last_source.get(symbol.upper(), "none")


_LABELS = {"alpaca": "Alpaca (live)", "yfinance": "yfinance (~15-20min delayed)",
          "none": "modeled only"}


def status_label(symbol: str) -> str:
    """Human label for the Bot Fleet settings page."""
    return _LABELS.get(status(symbol), "modeled only")


def test_connection() -> tuple[bool, str]:
    """Probe both providers for today's SPY 0DTE chain right now, so a user
    can check this actually works against their account instead of waiting
    to notice the Bot Fleet page still says 'modeled only' during market
    hours. Returns (ok, human message); never raises."""
    today = date.today()
    key, secret = bot_settings.data_source()["keys"] or (None, None)

    alpaca_note = "no Alpaca key saved"
    alpaca_ok = False
    if key and secret:
        try:
            q = _from_alpaca("SPY", today, 1.0, 100000.0, key, secret)
            if q:
                sample = next(iter(q.values()))
                alpaca_ok = True
                alpaca_note = "{} contracts, e.g. bid {:.2f} / ask {:.2f}".format(
                    len(q), sample["bid"], sample["ask"])
            else:
                alpaca_note = ("reachable but returned no quotes for SPY today — "
                               "normal outside market hours, or the account's options "
                               "data plan may not include this feed")
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            if code in (401, 403):
                alpaca_note = ("Alpaca rejected the request (HTTP {}) — this key likely "
                               "isn't entitled to options data".format(code))
            else:
                alpaca_note = "Alpaca options API HTTP {}".format(code)
        except requests.RequestException as e:
            alpaca_note = "could not reach Alpaca options API: {}".format(e)

    yf_note = ""
    yf_ok = False
    try:
        q = _from_yfinance("SPY", today)
        if q:
            sample = next(iter(q.values()))
            yf_ok = True
            yf_note = "{} contracts, e.g. bid {:.2f} / ask {:.2f}".format(
                len(q), sample["bid"], sample["ask"])
        else:
            yf_note = "no SPY chain for today's date from yfinance right now"
    except Exception as e:  # noqa: BLE001 — this is a diagnostic probe, never fatal
        yf_note = "yfinance error: {}".format(e)

    ok = alpaca_ok or yf_ok
    msg = "Alpaca: {} · yfinance: {}".format(alpaca_note, yf_note)
    return ok, msg
