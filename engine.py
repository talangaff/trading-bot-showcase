"""
Calculation engine for the earnings-volatility screener.

DISCLAIMER:
This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment
recommendations are made herein. The developers are not financial advisors and
accept no responsibility for any financial decisions or losses resulting from
the use of this software. Always consult a professional financial advisor
before making any investment decisions.

The calculation functions below (filter_dates, yang_zhang, build_term_structure,
get_current_price, compute_recommendation, get_rating, compute_bulk, and the
Nasdaq helpers) are copied verbatim from the original desktop calculator.py.
Do NOT modify the calculations.
"""

import math
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import requests
import yfinance as yf
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Calculations (verbatim from calculator.py)
# ---------------------------------------------------------------------------
def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)

    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break

    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    # interp1d can emit a benign "invalid value encountered in divide"
    # RuntimeWarning when two expirations map to the same day-to-expiry;
    # silence it without changing any results.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return float(spline(dte))

    return term_spline


def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    if todays_data.empty:
        return None
    return todays_data['Close'].iloc[-1]


def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."

        try:
            stock = yf.Ticker(ticker)
            if len(stock.options) == 0:
                raise KeyError()
        except KeyError:
            return f"Error: No options found for stock symbol '{ticker}'."

        exp_dates = list(stock.options)
        try:
            exp_dates = filter_dates(exp_dates)
        except:
            return "Error: Not enough option data."

        options_chains = {}
        for exp_date in exp_dates:
            options_chains[exp_date] = stock.option_chain(exp_date)

        try:
            underlying_price = get_current_price(stock)
            if underlying_price is None:
                raise ValueError("No market price found.")
        except Exception:
            return "Error: Unable to retrieve underlying stock price."

        atm_iv = {}
        straddle = None
        i = 0
        for exp_date, chain in options_chains.items():
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                continue

            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            if i == 0:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']

                if call_bid is not None and call_ask is not None:
                    call_mid = (call_bid + call_ask) / 2.0
                else:
                    call_mid = None

                if put_bid is not None and put_ask is not None:
                    put_mid = (put_bid + put_ask) / 2.0
                else:
                    put_mid = None

                if call_mid is not None and put_mid is not None:
                    straddle = (call_mid + put_mid)

            i += 1

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)

        term_spline = build_term_structure(dtes, ivs)

        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45-dtes[0])

        price_history = stock.history(period='3mo')
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)

        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

        expected_move = str(round(straddle / underlying_price * 100,2)) + "%" if straddle else None

        return {
            'avg_volume': avg_volume >= 1500000, 'iv30_rv30': iv30_rv30 >= 1.25,
            'ts_slope_0_45': ts_slope_0_45 <= -0.00406, 'expected_move': expected_move,  #Check that they are in our desired range (see video)
            # raw values, exposed for display only (not used in any calculation):
            'avg_volume_value': avg_volume, 'iv30_rv30_value': iv30_rv30,
            'ts_slope_0_45_value': ts_slope_0_45, 'underlying_price': underlying_price,
        }
    except Exception as e:
        raise Exception(f'Error occured processing: {e}')


def get_rating(result):
    """Map a compute_recommendation() result dict to Recommended / Consider / Avoid."""
    avg_volume_bool = result['avg_volume']
    iv30_rv30_bool  = result['iv30_rv30']
    ts_slope_bool   = result['ts_slope_0_45']

    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        return "Recommended"
    elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
        return "Consider"
    else:
        return "Avoid"


# ---------------------------------------------------------------------------
# Rate-limit resilience: retry-with-backoff + a short-lived per-symbol cache
# ---------------------------------------------------------------------------
_SYMBOL_CACHE = {}          # symbol -> (fetched_at, result)
_SYMBOL_CACHE_TTL = 900     # seconds (15 minutes)
_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "too many requests",
                       "yfratelimiterror", "429")
_TRANSIENT_RESULT_MARKER = "underlying stock price"   # usually a throttling symptom


def _looks_rate_limited(err):
    text = "{}: {}".format(type(err).__name__, err).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _retry_on_rate_limit(fn, max_retries=4, base_delay=2.0):
    """Call fn(); on an apparent rate-limit error, back off and retry."""
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if _looks_rate_limited(e) and attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise


def _is_nan(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


# raw value key -> (pass/fail key, threshold check used to recompute it when
# the value is replaced by a fallback)
_NAN_BOOL_RECOMPUTE = {
    'avg_volume_value': ('avg_volume', lambda v: v >= 1500000),
    'iv30_rv30_value': ('iv30_rv30', lambda v: v >= 1.25),
    'ts_slope_0_45_value': ('ts_slope_0_45', lambda v: v <= -0.00406),
}


def _fill_nan_from_previous(result, previous):
    """Replace any freshly computed NaN value with the last known-good one.

    yfinance can occasionally return data that makes a derived value compute
    as NaN (e.g. a zero-length window). Rather than caching/displaying "nan",
    fall back to the most recently available good value for that field. This
    checks every field in the result (not just avg_volume/iv30_rv30/
    ts_slope_0_45), so e.g. underlying_price or any future numeric field is
    covered too.
    """
    for key, value in list(result.items()):
        if not _is_nan(value):
            continue
        if key not in previous or _is_nan(previous.get(key)):
            continue
        result[key] = previous[key]
        if key in _NAN_BOOL_RECOMPUTE:
            bool_key, recompute_bool = _NAN_BOOL_RECOMPUTE[key]
            result[bool_key] = recompute_bool(previous[key])
    return result


def compute_recommendation_cached(symbol, use_cache=True):
    """compute_recommendation() with a 15-minute result cache and backoff retry."""
    key = (symbol or "").strip().upper()

    previous = _SYMBOL_CACHE.get(key)

    if use_cache and previous and (time.time() - previous[0]) < _SYMBOL_CACHE_TTL:
        return previous[1]

    def _call():
        return compute_recommendation(key)

    result = _retry_on_rate_limit(_call)

    if isinstance(result, str) and _TRANSIENT_RESULT_MARKER in result.lower():
        time.sleep(3.0)
        try:
            result = _retry_on_rate_limit(_call)
        except Exception:
            pass

    if isinstance(result, dict) and previous and isinstance(previous[1], dict):
        result = _fill_nan_from_previous(result, previous[1])

    transient = isinstance(result, str) and _TRANSIENT_RESULT_MARKER in result.lower()
    if not transient:
        _SYMBOL_CACHE[key] = (time.time(), result)
    return result


def _session_label(raw):
    """Map Nasdaq's earnings 'time' field to BMO / AMC / '?'."""
    r = (raw or "").lower()
    if "pre-market" in r or "premarket" in r or "pre_market" in r:
        return "BMO"
    if "after-hours" in r or "afterhours" in r or "after_hours" in r:
        return "AMC"
    return "?"


def compute_bulk(symbols, progress_cb=None, use_cache=True, sessions=None, previous_details=None):
    """Run compute_recommendation for each symbol and collect table rows.

    `previous_details`, if given, maps symbol -> the last known-good `detail`
    dict for that symbol (e.g. from a prior disk-cached refresh). It's used
    to fall back away from a freshly computed NaN value (see
    `_fill_nan_from_previous`), so a transient bad calculation doesn't get
    stored or displayed as "nan".

    Returns a list of dicts with keys: num, symbol, session, rating, avg_volume,
    iv30_rv30, ts_slope_0_45, expected_move, detail. Sorted Recommended ->
    Consider -> Avoid -> Error.
    """
    order = {"Recommended": 0, "Consider": 1, "Avoid": 2, "Error": 3}
    sessions = sessions or {}
    previous_details = previous_details or {}
    total = len(symbols)
    rows = []
    for idx, symbol in enumerate(symbols):
        session = sessions.get(symbol, "-")
        try:
            result = compute_recommendation_cached(symbol, use_cache=use_cache)
            if isinstance(result, dict):
                prev_detail = previous_details.get(symbol)
                if isinstance(prev_detail, dict):
                    result = _fill_nan_from_previous(result, prev_detail)
                rating = get_rating(result)
                rows.append({
                    "num": idx + 1,
                    "symbol": symbol,
                    "session": session,
                    "rating": rating,
                    "avg_volume": "PASS" if result['avg_volume'] else "FAIL",
                    "iv30_rv30": "PASS" if result['iv30_rv30'] else "FAIL",
                    "ts_slope_0_45": "PASS" if result['ts_slope_0_45'] else "FAIL",
                    "expected_move": result['expected_move'] if result['expected_move'] else "N/A",
                    "detail": _json_safe(result),
                })
            else:
                rows.append({
                    "num": idx + 1,
                    "symbol": symbol, "session": session, "rating": "Error",
                    "avg_volume": "-", "iv30_rv30": "-", "ts_slope_0_45": "-",
                    "expected_move": str(result),
                    "detail": None,
                })
        except Exception as e:
            rows.append({
                "num": idx + 1,
                "symbol": symbol, "session": session, "rating": "Error",
                "avg_volume": "-", "iv30_rv30": "-", "ts_slope_0_45": "-",
                "expected_move": str(e),
                "detail": None,
            })

        if progress_cb:
            progress_cb(idx + 1, total, symbol)

    rows.sort(key=lambda r: (order.get(r["rating"], 4), r["num"]))
    return rows


NASDAQ_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_date_input(text):
    """Accept common date formats, return an ISO 'YYYY-MM-DD' string."""
    text = (text or "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%b %d %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError("Enter a date as MM/DD/YYYY or YYYY-MM-DD")


def fetch_nasdaq_earnings(date_str):
    """Pull the nasdaq.com earnings calendar for date_str ('YYYY-MM-DD').

    Returns a list of {'symbol', 'name', 'market_cap', 'session'} sorted by
    market cap (largest first).
    """
    resp = requests.get(
        "https://api.nasdaq.com/api/calendar/earnings",
        params={"date": date_str}, headers=NASDAQ_HEADERS, timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json() or {}
    data = payload.get("data") or {}
    rows = data.get("rows") or []

    out = []
    for row in rows:
        symbol = (row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        digits = "".join(ch for ch in (row.get("marketCap") or "") if ch.isdigit())
        out.append({
            "symbol": symbol,
            "name": row.get("name") or symbol,
            "market_cap": int(digits) if digits else 0,
            "session": _session_label(row.get("time")),
        })

    out.sort(key=lambda x: x["market_cap"], reverse=True)
    return out


_NAME_CACHE = {}


def fetch_company_name(symbol):
    """Best-effort full company name lookup (display only), cached + backoff."""
    key = (symbol or "").strip().upper()
    if key in _NAME_CACHE:
        return _NAME_CACHE[key]
    try:
        info = _retry_on_rate_limit(lambda: yf.Ticker(key).info) or {}
        name = info.get("longName") or info.get("shortName") or key
    except Exception:
        name = key
    _NAME_CACHE[key] = name
    return name


# ---------------------------------------------------------------------------
# Display helpers (no calculation)
# ---------------------------------------------------------------------------
RATINGS = ("Recommended", "Consider", "Avoid", "Passed", "Error")

# "Passed" — earnings were released BMO earlier today, so there is nothing left
# to trade for this event; it is never Recommended/Consider/Avoid.
RATING_SORT = {"Recommended": 0, "Consider": 1, "Avoid": 2, "Passed": 3, "Error": 4}


def sort_rows(rows):
    return sorted(rows, key=lambda r: (RATING_SORT.get(r.get("rating"), 9), r.get("num", 0)))


def rating_summary(rows):
    counts = {}
    for r in rows:
        counts[r["rating"]] = counts.get(r["rating"], 0) + 1
    parts = ["{} {}".format(counts[n], n) for n in RATINGS if counts.get(n)]
    return "   ·   ".join(parts) if parts else "No data for this date"


def rating_counts(rows):
    counts = {n: 0 for n in RATINGS}
    for r in rows:
        counts[r["rating"]] = counts.get(r["rating"], 0) + 1
    return counts


def _json_safe(obj):
    """Recursively coerce numpy scalars / arrays to plain JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):        # numpy scalar
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def fmt_int(x):
    try:
        return "{:,}".format(int(round(float(x))))
    except (TypeError, ValueError):
        return str(x)


def fmt_num(x, places=4):
    try:
        return "{:.{p}f}".format(float(x), p=places)
    except (TypeError, ValueError):
        return str(x)
