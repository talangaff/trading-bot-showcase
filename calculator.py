"""
DISCLAIMER: 

This software is provided solely for educational and research purposes. 
It is not intended to provide investment advice, and no investment recommendations are made herein. 
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software. 
Always consult a professional financial advisor before making any investment decisions.
"""


import FreeSimpleGUI as sg
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import threading
import warnings
import requests
import json
import os
import time
import sys


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
        

    

# ----------------------------------------------------------------------------
# UI theme / styling
# ----------------------------------------------------------------------------
BG      = "#0f172a"   # app background   (slate-900)
SURFACE = "#1e293b"   # cards / inputs   (slate-800)
LINE    = "#334155"   # separators       (slate-700)
TEXT    = "#e2e8f0"   # primary text     (slate-200)
MUTED   = "#94a3b8"   # secondary text   (slate-400)
ACCENT  = "#3b82f6"   # primary action   (blue-500)
GREEN   = "#22c55e"
YELLOW  = "#f59e0b"
RED     = "#ef4444"
GREY    = "#64748b"

FONT_H1   = ("Helvetica", 20)
FONT_H2   = ("Helvetica", 14)
FONT_BODY = ("Helvetica", 11)
FONT_TINY = ("Helvetica", 9)

# (text_color, background_color) for each rating
RATING_COLORS = {
    "Recommended": ("#ffffff", "#15803d"),
    "Consider":    ("#111827", "#b45309"),
    "Avoid":       ("#ffffff", "#b91c1c"),
    "Error":       ("#ffffff", "#475569"),
}

RATING_ACCENT = {
    "Recommended": GREEN,
    "Consider":    YELLOW,
    "Avoid":       RED,
    "Error":       GREY,
}


def apply_theme():
    sg.theme_add_new("EarningsDark", {
        "BACKGROUND": BG,
        "TEXT": TEXT,
        "INPUT": SURFACE,
        "TEXT_INPUT": TEXT,
        "SCROLL": SURFACE,
        "BUTTON": ("#ffffff", ACCENT),
        "PROGRESS": (ACCENT, SURFACE),
        "BORDER": 0,
        "SLIDER_DEPTH": 0,
        "PROGRESS_DEPTH": 0,
    })
    sg.theme("EarningsDark")
    sg.set_options(
        font=FONT_BODY,
        element_padding=(6, 6),
        margins=(22, 22),
        button_element_size=(12, 1),
        border_width=0,
        input_elements_background_color=SURFACE,
        text_color=TEXT,
        background_color=BG,
        element_background_color=BG,
        text_element_background_color=BG,
    )


def _pill(text, fg, bg, font=None):
    return sg.Text(f"  {text}  ", text_color=fg, background_color=bg,
                   font=font or ("Helvetica", 10), pad=(0, 0))


def get_rating(result):
    """Map a compute_recommendation() result dict to Recommended / Consider / Avoid.

    Uses the exact same logic as the single-symbol result window; no calculations
    are changed here.
    """
    avg_volume_bool = result['avg_volume']
    iv30_rv30_bool  = result['iv30_rv30']
    ts_slope_bool   = result['ts_slope_0_45']

    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        return "Recommended"
    elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
        return "Consider"
    else:
        return "Avoid"


# ----------------------------------------------------------------------------
# Rate-limit resilience: retry-with-backoff + a short-lived per-symbol cache
# ----------------------------------------------------------------------------
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


def compute_recommendation_cached(symbol, use_cache=True):
    """compute_recommendation() with a 15-minute result cache and backoff retry.

    Does not touch any calculation - just wraps the call.
    """
    key = (symbol or "").strip().upper()

    if use_cache:
        hit = _SYMBOL_CACHE.get(key)
        if hit and (time.time() - hit[0]) < _SYMBOL_CACHE_TTL:
            return hit[1]

    def _call():
        return compute_recommendation(key)

    result = _retry_on_rate_limit(_call)

    # A "couldn't get underlying price" string is often just throttling - give
    # it one more backoff pass before accepting it.
    if isinstance(result, str) and _TRANSIENT_RESULT_MARKER in result.lower():
        time.sleep(3.0)
        try:
            result = _retry_on_rate_limit(_call)
        except Exception:
            pass

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


def compute_bulk(symbols, progress_cb=None, use_cache=True, sessions=None):
    """Run compute_recommendation for each symbol and collect table rows.

    Returns a list of dicts with keys: num, symbol, session, rating, avg_volume,
    iv30_rv30, ts_slope_0_45, expected_move, detail. Sorted Recommended ->
    Consider -> Avoid -> Error.

    sessions, if given, maps symbol -> "BMO"/"AMC"/"?" (from the Nasdaq calendar).
    progress_cb, if given, is called as progress_cb(done, total, symbol) after
    each symbol is processed.
    """
    order = {"Recommended": 0, "Consider": 1, "Avoid": 2, "Error": 3}
    sessions = sessions or {}
    total = len(symbols)
    rows = []
    for idx, symbol in enumerate(symbols):
        session = sessions.get(symbol, "-")
        try:
            result = compute_recommendation_cached(symbol, use_cache=use_cache)
            if isinstance(result, dict):
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
                    "detail": result,
                })
            else:
                # compute_recommendation returned an error/status string
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

    Returns a list of {'symbol', 'name', 'market_cap'} sorted by market cap
    (largest first).
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


# ----------------------------------------------------------------------------
# "This Week" — cached daily screens
# ----------------------------------------------------------------------------
WEEK_DAYS = 7
WEEK_TOP_N = 50
WEEK_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "week_cache.json")


def _json_default(obj):
    if hasattr(obj, "item"):        # numpy scalar -> python scalar
        return obj.item()
    return str(obj)


def load_week_cache():
    """Load the on-disk cache and drop any dates before today."""
    try:
        with open(WEEK_CACHE_FILE) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}

    today = datetime.today().strftime("%Y-%m-%d")
    pruned = {d: rows for d, rows in cache.items() if d >= today}
    if pruned != cache:
        save_week_cache(pruned)
    return pruned


def save_week_cache(cache):
    try:
        with open(WEEK_CACHE_FILE, "w") as f:
            json.dump(cache, f, default=_json_default)
    except OSError:
        pass


def refresh_week_cache(top_n=WEEK_TOP_N, days=WEEK_DAYS, verbose=True):
    """Headless refresh of the "This Week" cache (used by the cron job).

    For each of the next `days` days: pull the Nasdaq top-`top_n` earnings by
    market cap, screen every symbol, and write the results to WEEK_CACHE_FILE.
    Past dates are pruned. Returns the updated cache dict.
    """
    def _log(msg):
        if verbose:
            print("{}  {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg),
                  flush=True)

    cache = load_week_cache()          # also prunes past dates
    today = datetime.today().date()
    _log("refresh start -> {}".format(WEEK_CACHE_FILE))

    for i in range(days):
        d = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            listing = _retry_on_rate_limit(lambda d=d: fetch_nasdaq_earnings(d))
        except Exception as e:
            _log("[{}] Nasdaq fetch failed: {}".format(d, e))
            continue

        listing = listing[:top_n]
        symbols = [x["symbol"] for x in listing]
        sessions = {x["symbol"]: x.get("session", "-") for x in listing}
        if not symbols:
            cache[d] = []
            save_week_cache(cache)
            _log("[{}] no earnings listed".format(d))
            continue

        _log("[{}] screening {} symbols...".format(d, len(symbols)))
        rows = compute_bulk(symbols, use_cache=False, sessions=sessions)  # fresh screen
        cache[d] = rows
        save_week_cache(cache)
        _log("[{}] done: {}".format(d, _rating_summary(rows)))

    _log("refresh complete")
    return cache


def show_bulk_table(rows):
    headings = ["#", "Symbol", "When", "Rating", "Avg Volume", "IV30 / RV30",
                "Term Slope", "Exp. Move"]
    table_values = [
        [r["num"], r["symbol"], r.get("session", "-"), r["rating"], r["avg_volume"],
         r["iv30_rv30"], r["ts_slope_0_45"], r["expected_move"]]
        for r in rows
    ]
    row_colors = [
        [i, RATING_COLORS.get(r["rating"], (TEXT, SURFACE))[0],
            RATING_COLORS.get(r["rating"], (TEXT, SURFACE))[1]]
        for i, r in enumerate(rows)
    ]

    counts = {}
    for r in rows:
        counts[r["rating"]] = counts.get(r["rating"], 0) + 1
    chips = [sg.Text("Results", font=FONT_H1, pad=((0, 12), (0, 0)))]
    for name in ("Recommended", "Consider", "Avoid", "Error"):
        if counts.get(name):
            fg, bg = RATING_COLORS[name]
            chips.append(_pill(f"{counts[name]} {name}", fg, bg))
            chips.append(sg.Text(" ", pad=(0, 0)))

    layout = [
        chips,
        [sg.Text("Sorted best to worst  ·  double-click a row for details",
                 font=FONT_TINY, text_color=MUTED, pad=((0, 0), (2, 12)))],
        [sg.Table(
            values=table_values,
            headings=headings,
            auto_size_columns=False,
            col_widths=[4, 9, 6, 12, 12, 13, 12, 12],
            justification="center",
            num_rows=min(max(len(table_values), 3), 10),
            row_height=28,
            font=FONT_BODY,
            header_font=("Helvetica", 10),
            header_background_color=BG,
            header_text_color=MUTED,
            header_relief="flat",
            header_border_width=0,
            background_color=SURFACE,
            text_color=TEXT,
            selected_row_colors=(TEXT, LINE),
            row_colors=row_colors,
            border_width=0,
            hide_vertical_scroll=len(table_values) <= 10,
            expand_x=True,
            enable_events=True,
            key="bulk_table",
        )],
        [sg.Push(), sg.Button("Close", key="OK", button_color=(TEXT, LINE)), sg.Push()],
    ]

    win = sg.Window("Bulk Screen Results", layout, modal=True, finalize=True,
                    background_color=BG, margins=(24, 22))
    win["bulk_table"].bind("<Double-Button-1>", "+double")
    while True:
        ev, vals = win.read(timeout=100)
        if ev in (sg.WINDOW_CLOSED, "OK"):
            break
        if ev == "bulk_table+double":
            selected = vals.get("bulk_table", []) if vals else []
            if selected:
                show_detail(rows[selected[0]])
    win.close()


def _fmt_int(x):
    try:
        return "{:,}".format(int(round(float(x))))
    except (TypeError, ValueError):
        return str(x)


def _fmt_num(x, places=4):
    try:
        return "{:.{p}f}".format(float(x), p=places)
    except (TypeError, ValueError):
        return str(x)


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


def show_detail(row):
    symbol = row["symbol"]
    detail = row.get("detail")

    name, _ = run_with_loading(lambda: fetch_company_name(symbol),
                               message="Loading {} details...".format(symbol))
    if not name:
        name = symbol

    session = row.get("session", "-")
    session_text = {"BMO": "Reports before market open",
                    "AMC": "Reports after market close",
                    "?": "Earnings time not supplied by Nasdaq"}.get(session, "")

    header = [
        [sg.Text(symbol.upper(), font=FONT_H1, text_color=TEXT)],
        [sg.Text(name, font=FONT_BODY, text_color=MUTED, pad=(0, (0, 2)))],
        [sg.Text(session_text, font=FONT_TINY, text_color=MUTED, pad=(0, (0, 4)))],
    ]

    if not isinstance(detail, dict):
        body = [[sg.Text(row.get("expected_move", "No data available"),
                         font=FONT_BODY, text_color=RED)]]
    else:
        rating = get_rating(detail)
        accent = RATING_ACCENT[rating]

        def metric(title, value_str, threshold_str, passed):
            badge_bg = GREEN if passed else RED
            return sg.Column([
                [sg.Text(title, font=("Helvetica", 10), text_color=MUTED,
                         background_color=SURFACE)],
                [sg.Text(value_str, font=FONT_H2, text_color=TEXT,
                         background_color=SURFACE),
                 sg.Push(background_color=SURFACE),
                 sg.Text("PASS" if passed else "FAIL", text_color="#ffffff",
                         background_color=badge_bg, font=("Helvetica", 10),
                         size=(6, 1), justification="center")],
                [sg.Text("Criteria:  " + threshold_str, font=FONT_TINY,
                         text_color=MUTED, background_color=SURFACE)],
            ], background_color=SURFACE, expand_x=True, pad=(0, (0, 8)))

        av = detail.get("avg_volume_value")
        iv = detail.get("iv30_rv30_value")
        ts = detail.get("ts_slope_0_45_value")

        body = [
            [sg.Text(rating.upper(), font=("Helvetica", 18), text_color="#ffffff",
                     background_color=accent, justification="center",
                     expand_x=True, pad=(0, (2, 12)))],
            [metric("30-day average volume",
                    _fmt_int(av) + " shares", "≥ 1,500,000 shares",
                    detail["avg_volume"])],
            [metric("IV30 / RV30 ratio",
                    _fmt_num(iv, 3), "≥ 1.25", detail["iv30_rv30"])],
            [metric("Term-structure slope (0–45 DTE)",
                    _fmt_num(ts, 5), "≤ -0.00406", detail["ts_slope_0_45"])],
            [sg.Text("Expected move (straddle / price)", font=("Helvetica", 10),
                     text_color=MUTED, pad=(0, (6, 0)))],
            [sg.Text(detail.get("expected_move") or "N/A", font=FONT_H2,
                     text_color=ACCENT)],
        ]

    layout = header + body + [
        [sg.Button("Close", key="OK", expand_x=True, pad=(0, (16, 0)))],
    ]

    win = sg.Window("Details - " + symbol.upper(), layout, modal=True,
                    finalize=True, background_color=BG, margins=(22, 20))
    while True:
        ev, _ = win.read(timeout=100)
        if ev in (sg.WINDOW_CLOSED, "OK"):
            break
    win.close()


def run_with_loading(target_fn, message="Loading..."):
    """Run target_fn() in a worker thread while showing a modal loading window.

    Returns (result, error_string).
    """
    loading_window = sg.Window(
        "Loading",
        [
            [sg.Image(data=sg.DEFAULT_BASE64_LOADING_GIF, key="_gif_",
                      background_color=BG)],
            [sg.Text(message, key="loading", justification="center",
                     font=FONT_H2, text_color=TEXT, background_color=BG)],
        ],
        modal=True, finalize=True, no_titlebar=True, keep_on_top=True,
        element_justification="center", background_color=BG, margins=(34, 28),
    )
    holder = {}

    def worker():
        try:
            holder['result'] = target_fn()
        except Exception as e:
            holder['error'] = str(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        loading_window["_gif_"].update_animation(sg.DEFAULT_BASE64_LOADING_GIF,
                                                 time_between_frames=50)
        ev, _ = loading_window.read(timeout=50)
        if ev == sg.WINDOW_CLOSED:
            break
    thread.join(timeout=1)
    loading_window.close()
    return holder.get('result'), holder.get('error')


def run_bulk_with_progress(symbols, use_cache=True, sessions=None):
    """Run compute_bulk() in a worker thread while showing a progress bar.

    Returns (rows, error_string).
    """
    total = len(symbols)
    state = {"done": 0, "symbol": symbols[0] if symbols else ""}

    window = sg.Window(
        "Screening",
        [
            [sg.Text("Screening symbols", key="_title_", justification="center",
                     font=FONT_H2, text_color=TEXT, background_color=BG)],
            [sg.ProgressBar(total, orientation="h", size=(34, 18), key="_bar_",
                            bar_color=(ACCENT, SURFACE), border_width=0,
                            pad=(0, (12, 8)))],
            [sg.Text("", key="_status_", justification="center",
                     font=FONT_TINY, text_color=MUTED, background_color=BG)],
        ],
        modal=True, finalize=True, no_titlebar=True, keep_on_top=True,
        element_justification="center", background_color=BG, margins=(34, 26),
    )

    holder = {}

    def progress_cb(done, tot, symbol):
        state["done"] = done
        state["symbol"] = symbol

    def worker():
        try:
            holder['result'] = compute_bulk(symbols, progress_cb=progress_cb,
                                            use_cache=use_cache, sessions=sessions)
        except Exception as e:
            holder['error'] = str(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        done = state["done"]
        window["_bar_"].update(current_count=done)
        window["_status_"].update(
            "{} / {}   ·   last: {}".format(done, total, state["symbol"]))
        ev, _ = window.read(timeout=80)
        if ev == sg.WINDOW_CLOSED:
            break
    thread.join(timeout=1)
    window.close()
    return holder.get('result'), holder.get('error')


def _factor_row(label, passed):
    fg, bg = ("#ffffff", GREEN) if passed else ("#ffffff", RED)
    return [
        sg.Text(label, size=(18, 1), text_color=TEXT, background_color=SURFACE,
                font=FONT_BODY, pad=((14, 8), 6)),
        sg.Push(),
        sg.Text("PASS" if passed else "FAIL", text_color=fg, background_color=bg,
                font=("Helvetica", 10), pad=((0, 14), 6), size=(6, 1),
                justification="center"),
    ]


def show_result(symbol, result):
    rating = get_rating(result)
    accent = RATING_ACCENT[rating]

    card = sg.Column(
        [
            _factor_row("Avg volume", result['avg_volume']),
            _factor_row("IV30 / RV30", result['iv30_rv30']),
            _factor_row("Term structure slope", result['ts_slope_0_45']),
        ],
        background_color=SURFACE, pad=(0, 0), expand_x=True,
    )

    layout = [
        [sg.Text(symbol.upper(), font=FONT_TINY, text_color=MUTED)],
        [sg.Text(rating.upper(), font=("Helvetica", 24), text_color="#ffffff",
                 background_color=accent, justification="center", expand_x=True,
                 pad=(0, (2, 16)))],
        [card],
        [sg.Text("EXPECTED MOVE", font=FONT_TINY, text_color=MUTED, pad=(0, (16, 0)))],
        [sg.Text(result['expected_move'] or "N/A", font=FONT_H1, text_color=ACCENT)],
        [sg.Button("Close", key="OK", expand_x=True, pad=(0, (18, 0)))],
    ]

    win = sg.Window("Recommendation", layout, modal=True, finalize=True,
                    background_color=BG, margins=(24, 22),
                    element_justification="center")
    while True:
        ev, _ = win.read(timeout=100)
        if ev in (sg.WINDOW_CLOSED, "OK"):
            break
    win.close()


WEEK_HEADINGS = ["#", "Symbol", "When", "Rating", "Avg Volume", "IV30 / RV30",
                 "Term Slope", "Exp. Move"]


def _table_rows(rows):
    values = [
        [r["num"], r["symbol"], r.get("session", "-"), r["rating"], r["avg_volume"],
         r["iv30_rv30"], r["ts_slope_0_45"], r["expected_move"]]
        for r in rows
    ]
    colors = [
        [i] + list(RATING_COLORS.get(r["rating"], (TEXT, SURFACE)))
        for i, r in enumerate(rows)
    ]
    return values, colors


def _rating_summary(rows):
    counts = {}
    for r in rows:
        counts[r["rating"]] = counts.get(r["rating"], 0) + 1
    parts = ["{} {}".format(counts[n], n)
             for n in ("Recommended", "Consider", "Avoid", "Error") if counts.get(n)]
    return "   ·   ".join(parts) if parts else "No data for this date"


def show_week_window():
    """Tabbed 7-day view. Each tab lazily pulls & screens the Nasdaq top 50 for
    that date, caching the result to disk (past dates are pruned on open)."""
    cache = load_week_cache()
    today = datetime.today().date()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(WEEK_DAYS)]

    tabs = []
    for d in dates:
        label = datetime.strptime(d, "%Y-%m-%d").strftime("%a %m/%d")
        tab_layout = [
            [sg.Text("Not loaded yet", key="status_" + d, font=FONT_TINY,
                     text_color=MUTED, background_color=BG, pad=(0, (2, 6)))],
            [sg.Table(
                values=[], headings=WEEK_HEADINGS, auto_size_columns=False,
                col_widths=[4, 9, 6, 12, 12, 13, 12, 12], justification="center",
                num_rows=14, row_height=26, font=FONT_BODY,
                header_font=("Helvetica", 10), header_background_color=BG,
                header_text_color=MUTED, header_relief="flat",
                header_border_width=0, background_color=SURFACE, text_color=TEXT,
                selected_row_colors=(TEXT, LINE), border_width=0,
                expand_x=True, enable_events=True, key="table_" + d,
            )],
        ]
        tabs.append(sg.Tab(label, tab_layout, key="tab_" + d, background_color=BG))

    layout = [
        [sg.Text("This Week", font=FONT_H1, text_color=TEXT)],
        [sg.Text("Top {} earnings by market cap each day  ·  double-click a row "
                 "for details".format(WEEK_TOP_N), font=FONT_TINY,
                 text_color=MUTED, pad=(0, (0, 10)))],
        [sg.TabGroup([tabs], key="weektabs", enable_events=True,
                     background_color=BG, tab_background_color=SURFACE,
                     selected_background_color=ACCENT, title_color=MUTED,
                     selected_title_color="#ffffff", font=FONT_BODY,
                     border_width=0, tab_border_width=0)],
        [sg.Push(),
         sg.Button("Update All", key="update_all", button_color=("#ffffff", ACCENT)),
         sg.Button("Reload This Day", key="reload", button_color=(TEXT, LINE)),
         sg.Button("Close", key="OK", button_color=(TEXT, LINE))],
    ]

    win = sg.Window("This Week", layout, modal=True, finalize=True,
                    background_color=BG, margins=(22, 20))
    for d in dates:
        win["table_" + d].bind("<Double-Button-1>", "+double")

    loaded = {}
    bg_pause = threading.Event()   # set => background prefetch is paused
    bg_stop = threading.Event()    # set => background thread should exit

    def render(d):
        rows = loaded.get(d, [])
        values, colors = _table_rows(rows)
        win["table_" + d].update(values=values, row_colors=colors)
        win["status_" + d].update(
            _rating_summary(rows) if rows else "No earnings found for this date")

    def _screen_date(d, use_cache):
        """Fetch + screen one date. Returns (rows, error_string)."""
        listing = fetch_nasdaq_earnings(d)
        if not listing:
            return [], None
        listing = listing[:WEEK_TOP_N]
        symbols = [x["symbol"] for x in listing]
        sessions = {x["symbol"]: x.get("session", "-") for x in listing}
        return compute_bulk(symbols, use_cache=use_cache, sessions=sessions), None

    def ensure_loaded(d, force=False):
        """Foreground load (modal progress) - used when the user wants a day now."""
        if d in loaded and not force:
            return
        if not force and d in cache:
            loaded[d] = cache[d]
            render(d)
            return

        bg_pause.set()   # don't fight the background thread for Yahoo
        try:
            win["status_" + d].update("Pulling Nasdaq top {}...".format(WEEK_TOP_N))
            win.refresh()

            listing, err = run_with_loading(
                lambda: fetch_nasdaq_earnings(d),
                message="Fetching Nasdaq earnings for {}...".format(d))
            if err:
                win["status_" + d].update("Error: {}".format(err))
                return
            if not listing:
                loaded[d] = cache[d] = []
                save_week_cache(cache)
                render(d)
                return

            listing = listing[:WEEK_TOP_N]
            symbols = [x["symbol"] for x in listing]
            sessions = {x["symbol"]: x.get("session", "-") for x in listing}
            rows, error = run_bulk_with_progress(symbols, use_cache=not force,
                                                 sessions=sessions)
            if error:
                win["status_" + d].update("Error: {}".format(error))
                return
            loaded[d] = cache[d] = rows
            save_week_cache(cache)
            render(d)
        finally:
            bg_pause.clear()

    def bg_worker():
        """Silently fill in every day that isn't loaded yet, one at a time."""
        for d in list(dates):
            if bg_stop.is_set():
                return
            while bg_pause.is_set():
                if bg_stop.is_set():
                    return
                time.sleep(0.4)
            if d in loaded:
                continue
            if d in cache:
                win.write_event_value("_bg_rows_", (d, cache[d]))
                continue
            win.write_event_value("_bg_start_", d)
            try:
                rows, _ = _screen_date(d, use_cache=True)
                if bg_stop.is_set():
                    return
                win.write_event_value("_bg_rows_", (d, rows))
            except Exception as e:
                win.write_event_value("_bg_err_", (d, str(e)))

    # Mark the tabs that will fill in on their own
    for d in dates:
        if d not in cache:
            win["status_" + d].update("Waiting for background update...")

    ensure_loaded(dates[0])

    bg_thread = threading.Thread(target=bg_worker, daemon=True)
    bg_thread.start()

    while True:
        ev, vals = win.read(timeout=200)
        if ev in (sg.WINDOW_CLOSED, "OK"):
            bg_stop.set()
            break
        if ev == "weektabs":
            active = vals.get("weektabs") if vals else None
            if active and active.startswith("tab_"):
                ensure_loaded(active[4:])
        elif ev == "reload":
            active = vals.get("weektabs") if vals else None
            if active and active.startswith("tab_"):
                ensure_loaded(active[4:], force=True)
        elif ev == "update_all":
            for i, d in enumerate(dates):
                win["status_" + d].update("Queued for update ({}/{})...".format(
                    i + 1, len(dates)))
            win.refresh()
            for d in dates:
                ensure_loaded(d, force=True)
        elif ev == "_bg_start_":
            d = vals["_bg_start_"]
            if d not in loaded:
                win["status_" + d].update("Updating in background...")
        elif ev == "_bg_rows_":
            d, rows = vals["_bg_rows_"]
            if d not in loaded:
                loaded[d] = cache[d] = rows
                save_week_cache(cache)
                render(d)
        elif ev == "_bg_err_":
            d, msg = vals["_bg_err_"]
            if d not in loaded:
                win["status_" + d].update(
                    "Background update failed - use Reload This Day  ({})".format(
                        msg[:50]))
        elif isinstance(ev, str) and ev.startswith("table_") and ev.endswith("+double"):
            d = ev[len("table_"):-len("+double")]
            sel = vals.get("table_" + d, []) if vals else []
            if sel and loaded.get(d):
                show_detail(loaded[d][sel[0]])

    bg_stop.set()
    win.close()


def _section_label(text):
    return sg.Text(text, font=FONT_TINY, text_color=MUTED, pad=(0, (0, 2)))


def main_gui():
    default_date = (datetime.today() + timedelta(days=1)).strftime("%m/%d/%Y")
    main_layout = [
        [sg.Text("Earnings Position Checker", font=FONT_H1, text_color=TEXT)],
        [sg.Text("Screen tickers for earnings volatility plays",
                 font=FONT_BODY, text_color=MUTED, pad=(0, (0, 16)))],

        [_section_label("SINGLE SYMBOL")],
        [sg.Input(key="stock", size=(22, 1), focus=True,
                  background_color=SURFACE, text_color=TEXT),
         sg.Button("Analyze", key="Submit")],

        [sg.Text("", key="recommendation", size=(46, 2), text_color=YELLOW,
                 font=FONT_BODY, pad=(0, (8, 8)))],

        [sg.HorizontalSeparator(color=LINE)],

        [_section_label("BULK IMPORT  ·  comma-separated symbols")],
        [sg.Input(key="bulk", size=(36, 1), background_color=SURFACE, text_color=TEXT),
         sg.Button("Screen All", key="Bulk Import")],

        [sg.HorizontalSeparator(color=LINE)],

        [_section_label("PULL FROM NASDAQ  ·  screen the biggest earnings on a date")],
        [sg.Text("Date", text_color=MUTED, font=FONT_BODY, pad=((0, 4), 6)),
         sg.Input(default_date, key="nasdaq_date", size=(12, 1),
                  background_color=SURFACE, text_color=TEXT),
         sg.Text("Top #", text_color=MUTED, font=FONT_BODY, pad=((12, 4), 6)),
         sg.Input("50", key="nasdaq_top", size=(5, 1),
                  background_color=SURFACE, text_color=TEXT),
         sg.Button("Pull Earnings", key="Pull Earnings")],
        [sg.Button("This Week", key="This Week"),
         sg.Text("7-day tabbed view  ·  top {} per day, cached".format(WEEK_TOP_N),
                 font=FONT_TINY, text_color=MUTED, pad=((6, 0), 6))],

        [sg.Text("", pad=(0, 4))],
        [sg.Push(), sg.Button("Exit", button_color=(MUTED, BG))],
    ]

    window = sg.Window("Earnings Position Checker", main_layout,
                       background_color=BG, margins=(26, 24), finalize=True)

    # Pressing Enter in a field submits that field's action.
    window["stock"].bind("<Return>", "_Enter")
    window["bulk"].bind("<Return>", "_Enter")
    window["nasdaq_date"].bind("<Return>", "_Enter")
    window["nasdaq_top"].bind("<Return>", "_Enter")

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "stock_Enter":
            event = "Submit"
        elif event == "bulk_Enter":
            event = "Bulk Import"
        elif event in ("nasdaq_date_Enter", "nasdaq_top_Enter"):
            event = "Pull Earnings"

        if event == "This Week":
            window["recommendation"].update("")
            show_week_window()
            continue

        if event == "Pull Earnings":
            window["recommendation"].update("")
            try:
                date_str = parse_date_input(values.get("nasdaq_date", ""))
            except ValueError as e:
                window["recommendation"].update("Error: {}".format(e))
                continue
            try:
                top_n = int(str(values.get("nasdaq_top", "")).strip())
                if top_n < 1:
                    raise ValueError
            except ValueError:
                window["recommendation"].update("Error: Top # must be a positive whole number.")
                continue

            listing, error = run_with_loading(
                lambda: fetch_nasdaq_earnings(date_str),
                message="Fetching Nasdaq earnings for {}...".format(date_str),
            )
            if error:
                window["recommendation"].update("Error: {}".format(error))
                continue
            if not listing:
                window["recommendation"].update("No earnings listed on Nasdaq for {}.".format(date_str))
                continue

            top = listing[:top_n]
            symbols = [item["symbol"] for item in top]
            sessions = {item["symbol"]: item.get("session", "-") for item in top}
            window["recommendation"].update(
                "Screening top {} of {} symbols for {}...".format(
                    len(symbols), len(listing), date_str))

            rows, error = run_bulk_with_progress(symbols, sessions=sessions)
            if error:
                window["recommendation"].update("Error: {}".format(error))
            elif rows is not None:
                window["recommendation"].update(
                    "Nasdaq {}  ·  top {} by market cap".format(date_str, len(symbols)))
                show_bulk_table(rows)
            continue

        if event == "Bulk Import":
            window["recommendation"].update("")
            raw = values.get("bulk", "")
            symbols = []
            for part in raw.split(","):
                sym = part.strip().upper()
                if sym and sym not in symbols:
                    symbols.append(sym)

            if not symbols:
                window["recommendation"].update("No symbols provided.")
                continue

            rows, error = run_bulk_with_progress(symbols)
            if error:
                window["recommendation"].update("Error: {}".format(error))
            elif rows is not None:
                show_bulk_table(rows)
            continue

        if event == "Submit":
            window["recommendation"].update("")
            stock = values.get("stock", "")

            result, error = run_with_loading(
                lambda: compute_recommendation_cached(stock),
                message="Analyzing {}...".format(stock.strip().upper() or "symbol"),
            )

            if error:
                window["recommendation"].update("Error: {}".format(error))
            elif not isinstance(result, dict):
                window["recommendation"].update(str(result))
            else:
                show_result(stock, result)

    window.close()

def gui():
    apply_theme()
    main_gui()

if __name__ == "__main__":
    if "--refresh-week" in sys.argv:
        refresh_week_cache()
    else:
        gui()