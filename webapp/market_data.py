"""Recent 1-minute bars for the sim engine.

Prefers Alpaca (keys entered on the Bot Fleet page); falls back to yfinance so
the fleet still runs before keys are set. One network fetch per symbol per
minute is shared across the whole fleet via a small in-process cache.
"""

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from . import api_meter, bot_settings

ET = ZoneInfo("America/New_York")

_CACHE: dict[str, tuple[float, list]] = {}
_TTL = 55.0                       # seconds — one fetch per engine minute
_ALPACA_DATA = "https://data.alpaca.markets/v2/stocks/{}/bars"


def _from_alpaca(symbol: str, minutes: int, key: str, secret: str,
                 feed: str) -> list[dict]:
    """The most recent `minutes` 1-minute bars, oldest first.

    Fetched with sort=DESC and reversed, which is the only ordering that
    reliably means "the latest N". With sort=asc and no explicit start,
    Alpaca anchors at the earliest data it has for the window and `limit`
    then truncates from the FRONT — so the request returns the first N bars
    of the session, not the last. That is not a visible failure: it is a
    200 with a full payload whose newest bar stops advancing once the
    session is N minutes old. It froze this fleet at ~11:20 ET every day
    (2026-09-09), because the engine gates each tick on bar freshness and
    quietly skipped every one after that.

    Anchoring on `start` instead would reintroduce the same class of bug on
    a quiet symbol, a half day or the first minutes after the open, where
    a fixed window can hold fewer bars than the engine's 20-bar minimum.
    Asking for the newest N cannot underfill for that reason.
    """
    params = {
        "timeframe": "1Min",
        "limit": max(minutes, 60),
        "feed": feed or "iex",
        "sort": "desc",
    }
    r = api_meter.get(
        _ALPACA_DATA.format(symbol),
        keys=(key, secret),
        params=params,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=8,
    )
    r.raise_for_status()
    bars = r.json().get("bars") or []
    bars.reverse()                      # newest-first off the wire, oldest-first out
    return [{"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"],
             "c": b["c"], "v": b.get("v", 0)} for b in bars]


def _from_yfinance(symbol: str, minutes: int) -> list[dict]:
    import yfinance as yf
    df = yf.download(symbol, period="1d", interval="1m", progress=False,
                     auto_adjust=False)
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    out = []
    for idx, row in df.tail(max(minutes, 60)).iterrows():
        out.append({"t": idx.isoformat(), "o": float(row["Open"]),
                    "h": float(row["High"]), "l": float(row["Low"]),
                    "c": float(row["Close"]), "v": float(row.get("Volume", 0) or 0)})
    return out


MAX_BAR_AGE_S = 900.0            # the engine's own freshness gate, mirrored here


def _last_ts(bars: list[dict]) -> datetime | None:
    """The timestamp of the newest bar, as an aware datetime. Handles both
    providers: Alpaca stamps UTC with a trailing Z, yfinance a local offset."""
    if not bars:
        return None
    try:
        t = str(bars[-1]["t"]).replace("Z", "+00:00")
        d = datetime.fromisoformat(t)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, KeyError, AttributeError, TypeError):
        return None


def _stale(bars: list[dict]) -> bool:
    """True when the newest bar is too old for the engine to act on. An
    unparseable stamp is NOT called stale — that is the existing benefit of
    the doubt, and this must not become a new way to lose a data feed."""
    last = _last_ts(bars)
    if last is None:
        return False
    return (datetime.now(timezone.utc) - last).total_seconds() > MAX_BAR_AGE_S


def _better(candidate: list[dict], incumbent: list[dict]) -> bool:
    """Is `candidate` a more current answer than the one already in hand?

    Empty is never better. Against an incumbent whose newest stamp cannot be
    parsed the answer is no: `_stale()` gives such a feed the benefit of the
    doubt and calls it fresh, so overruling it here would quietly take that
    benefit back. A candidate we cannot date is likewise never promoted over
    one we can — an unreadable timestamp is a reason to leave things alone,
    not a reason to swap.
    """
    if not candidate:
        return False
    if not incumbent:
        return True
    c, i = _last_ts(candidate), _last_ts(incumbent)
    if c is None or i is None:
        return False
    return c > i


def _provider_failed(symbol: str, name: str, err: Exception) -> None:
    """Record that one provider is down, separately from its consequences.

    On 2026-09-11 Alpaca returned 504 on every bars call for 48 minutes and
    printed 96 lines saying so, yet the only alarm that fired was the
    downstream `engine_idle` — which reports that the fleet HAS no data, not
    where the data went. A provider going dark is worth naming on its own
    terms, and rate-limiting lives in the DB because each cycle is its own
    short-lived process.
    """
    print("market_data {} via {}: {}".format(symbol, name, err), flush=True)
    try:
        from . import bots          # local import — keeps this hot module light
        bots.log_event_once(
            "data_provider_error",
            {"provider": name, "symbol": symbol, "error": str(err)[:300]},
            key="data:{}:{}".format(name, symbol), within_s=900.0)
    except Exception:  # noqa: BLE001 — reporting a failure must not become one
        pass


def recent_bars(symbol: str, minutes: int = 120) -> list[dict]:
    symbol = symbol.upper()
    now = time.time()
    hit = _CACHE.get(symbol)
    if hit and now - hit[0] < _TTL:
        return hit[1]

    # Bars are the same for everyone, so they are pulled ONCE per cycle on
    # the best entitlement anybody has connected (bot_settings.data_source)
    # and cached for every user's fleet. One user with a SIP subscription
    # raises the data quality of the whole installation.
    src = bot_settings.data_source()

    # Each provider is asked in its own right and the freshest answer wins.
    #
    # Until 2026-09-11 the fallback sat AFTER the Alpaca call inside one
    # shared `try`, which made it reachable only when Alpaca RETURNED
    # something unusable. `_from_alpaca` ends in `raise_for_status()`, so
    # when Alpaca itself went down — 504 on every bars call for 48 minutes —
    # the raise jumped clean over the fallback into the `except`, and
    # yfinance, healthy throughout, was never asked. The fleet stepped zero
    # bots for 48 minutes while ~17 open positions went unmanaged.
    #
    # A provider that returns STALE bars and a provider that THROWS are the
    # same event to the engine (no fresh bars), so they have to take the same
    # path here. Isolating each fetch is what makes that true: one provider
    # failing, in either manner, cannot stop the next one being consulted.
    # The guarded branch is gone — the fallback is now a property of this
    # function rather than of the primary's success path.
    providers = []
    if src["keys"]:
        providers.append(("alpaca", lambda: _from_alpaca(
            symbol, minutes, src["keys"][0], src["keys"][1], src["feed"])))
    providers.append(("yfinance", lambda: _from_yfinance(symbol, minutes)))

    bars: list[dict] = []
    for name, fetch in providers:
        # Stop at the first genuinely current answer, so the normal path
        # still costs exactly one network call and the backup provider is
        # not hit on every cycle for no reason.
        if bars and not _stale(bars):
            break
        try:
            got = fetch()
        except Exception as e:  # noqa: BLE001 — never let a data hiccup kill a cycle
            _provider_failed(symbol, name, e)
            continue
        if _better(got, bars):
            bars = got

    # Reached only when every provider came back empty-handed, which is a
    # real outage rather than routine staleness — outside market hours both
    # feeds are legitimately old and the loop above just keeps the newer.
    if not bars and hit:
        bars = hit[1]

    _CACHE[symbol] = (now, bars)
    return bars


_ALPACA_ACCOUNT = "https://paper-api.alpaca.markets/v2/account"


def test_connection(keys: tuple[str, str] | None = None,
                    feed: str | None = None) -> tuple[bool, str]:
    """Probe one Alpaca paper key pair: the credential pair against the
    trading API, then a 1-bar data pull on `feed` (which is what the engine
    actually relies on). Returns (ok, human message)."""
    src = bot_settings.data_source()
    key, secret = keys if keys else (src["keys"] or (None, None))
    if not key or not secret:
        return False, "No Alpaca key saved yet — enter the key ID and secret above, then Save."
    feed = feed or src["feed"]
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    acct_note = ""
    try:
        r = api_meter.get(_ALPACA_ACCOUNT, keys=(key, secret), headers=hdr, timeout=8)
        if r.status_code in (401, 403):
            return False, "Alpaca rejected the key / secret (HTTP {}). Double-check both values.".format(r.status_code)
        if r.status_code == 200:
            j = r.json()
            acct_note = "account {} · {}".format(j.get("account_number", "?"), j.get("status", "?"))
        else:
            acct_note = "account check returned HTTP {}".format(r.status_code)
    except requests.RequestException as e:
        return False, "Could not reach Alpaca trading API: {}".format(e)

    try:
        r = api_meter.get(
            _ALPACA_DATA.format("SPY"),
            keys=(key, secret),
            # sort=desc for the same reason _from_alpaca uses it: without it
            # this returns the FIRST bar of the window, so the "last bar" it
            # reports would be hours old and the check would cheerfully call a
            # frozen feed healthy — this is the diagnostic, it has to be honest
            params={"timeframe": "1Min", "limit": 1, "feed": feed, "sort": "desc"},
            headers=hdr, timeout=8)
        if r.status_code in (401, 403):
            return False, ("Key works for the account but the data API rejected it "
                           "(HTTP {}). The '{}' feed likely needs a higher Alpaca plan — "
                           "try IEX.".format(r.status_code, feed))
        if r.status_code != 200:
            return False, "Data API HTTP {}: {}".format(r.status_code, (r.text or "")[:140])
        bars = r.json().get("bars") or []
        if bars:
            return True, "Connected · {} · '{}' feed returning SPY bars (last {}).".format(
                acct_note, feed, bars[-1].get("t", "?"))
        return True, ("Connected · {} · '{}' feed reachable, no bars right now "
                      "(normal outside market hours).".format(acct_note, feed))
    except requests.RequestException as e:
        return False, "Could not reach Alpaca data API: {}".format(e)


def last_price(symbol: str) -> float | None:
    bars = recent_bars(symbol)
    return bars[-1]["c"] if bars else None


def closes(bars: list[dict]) -> list[float]:
    return [b["c"] for b in bars]


# --- extra context: prior-day levels + VIX term structure (yfinance) --------
_DAILY_CACHE: dict[str, tuple[float, dict]] = {}
_VIX_CACHE: tuple[float, dict] | None = None


def daily_ctx(symbol: str) -> dict:
    """Prior-session close / high / low for gap and PDH/PDL features."""
    global _DAILY_CACHE
    now = time.time()
    hit = _DAILY_CACHE.get(symbol)
    if hit and now - hit[0] < 3600:
        return hit[1]
    out = {}
    try:
        import yfinance as yf
        df = yf.download(symbol, period="5d", interval="1d", progress=False,
                         auto_adjust=False)
        if df is not None and not df.empty:
            if hasattr(df.columns, "get_level_values"):
                df.columns = df.columns.get_level_values(0)
            prev = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
            out = {"prev_close": float(prev["Close"]),
                   "prev_high": float(prev["High"]),
                   "prev_low": float(prev["Low"])}
            # during the session yfinance's last daily row IS today — its
            # Open is the session open the feature layer needs
            try:
                last_idx = df.index[-1]
                last_day = last_idx.date() if hasattr(last_idx, "date") else None
                if len(df) >= 2 and last_day == datetime.now(ET).date():
                    out["today_open"] = float(df.iloc[-1]["Open"])
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        print("daily_ctx {}: {}".format(symbol, e), flush=True)
        out = hit[1] if hit else {}
    _DAILY_CACHE[symbol] = (now, out)
    return out


def vix() -> dict:
    """{'vix':.., 'vix9d':.., 'vix_open':..} — VIX term structure. Empty on failure."""
    global _VIX_CACHE
    now = time.time()
    if _VIX_CACHE and now - _VIX_CACHE[0] < 300:
        return _VIX_CACHE[1]
    out = {}
    try:
        import yfinance as yf
        df = yf.download(["^VIX", "^VIX9D"], period="1d", interval="5m",
                         progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            close = df["Close"]
            v = float(close["^VIX"].dropna().iloc[-1])
            out["vix"] = v
            out["vix_open"] = float(close["^VIX"].dropna().iloc[0])
            try:
                out["vix9d"] = float(close["^VIX9D"].dropna().iloc[-1])
            except (KeyError, IndexError):
                out["vix9d"] = v
    except Exception as e:  # noqa: BLE001
        print("vix: {}".format(e), flush=True)
        out = _VIX_CACHE[1] if _VIX_CACHE else {}
    _VIX_CACHE = (now, out)
    return out
