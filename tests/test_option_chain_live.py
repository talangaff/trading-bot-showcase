"""option_chain.build_chain()/live_quote() blend real quotes onto the
modeled chain per strike, but must fail closed to the model on a bad quote
day or a garbage/implausible print — and must never FILL against a stale
quote: anything older than MAX_QUOTE_AGE_S (or from a delayed feed) is
only used to calibrate implied vol + spread width and is then repriced at
the live spot. See webapp/option_quotes.py and option_chain._usable_quote().
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from webapp import option_chain, option_model, option_quotes

ET = option_model.ET
NOW = datetime(2026, 9, 8, 14, 30, tzinfo=ET)
EXPIRY = NOW.replace(hour=16, minute=0, second=0, microsecond=0)
S = 560.0
MINUTES_LEFT = option_model.minutes_to_close(NOW)
ATM = round(S / option_chain.STEP) * option_chain.STEP
OCC = option_model.occ_symbol("SPY", EXPIRY, True, ATM)
FRESH_T = NOW.isoformat()
STALE_T = (NOW - timedelta(minutes=20)).isoformat()


def _chain_with_fake_quotes(quotes, source="alpaca"):
    with patch.object(option_quotes, "get_chain", return_value=quotes), \
         patch.object(option_quotes, "_last_source", {"SPY": source}):
        return option_chain.build_chain("SPY", S, MINUTES_LEFT, 0.20, True, expiry=EXPIRY, now=NOW)


def _atm(chain):
    return next(r for r in chain if r["strike"] == ATM)


def test_no_provider_data_falls_back_to_model():
    chain = _chain_with_fake_quotes({})
    assert chain
    assert {r["source"] for r in chain} == {"model"}


def test_fresh_plausible_quote_overrides_the_model():
    quotes = {OCC: {"bid": 1.10, "ask": 1.30, "last": 1.2, "volume": 500, "t": FRESH_T, "spot": S}}
    row = _atm(_chain_with_fake_quotes(quotes))
    assert row["source"] == "alpaca"
    assert row["bid"] == 1.10 and row["ask"] == 1.30
    assert abs(row["price"] - 1.20) < 1e-6
    assert row["volume"] == 500
    # the market's own implied vol replaced the modeled one
    assert abs(option_model.bs_price(S, ATM, MINUTES_LEFT, row["iv"], True) - 1.20) < 0.01
    assert all(r["source"] == "model" for r in _chain_with_fake_quotes(quotes) if r["strike"] != ATM)


def test_stale_quote_is_calibrated_and_repriced_not_filled():
    # quoted 20 minutes ago when spot was 559; spot is now 560. A call's
    # stale mid is too CHEAP for today's spot — filling at it would be a leak.
    model = _atm(_chain_with_fake_quotes({}))
    stale_mid = option_model.bs_price(559.0, ATM, MINUTES_LEFT + 20, 0.20, True)
    quotes = {OCC: {"bid": round(stale_mid * 0.95, 3), "ask": round(stale_mid * 1.05, 3),
                    "last": None, "volume": 500, "t": STALE_T, "spot": 559.0}}
    row = _atm(_chain_with_fake_quotes(quotes))
    assert row["source"] == "alpaca-calibrated"
    assert row["price"] > stale_mid, "repriced at the higher live spot"
    assert abs(row["iv"] - 0.20) < 0.01, "the stale quote's IV was recovered against its own spot"
    assert abs(row["price"] - model["price"]) < 0.05          # ...which is the modeled price at 560
    # relative spread width preserved
    assert abs((row["ask"] - row["bid"]) / row["price"] - 0.10) < 0.02


def test_delayed_feed_and_missing_timestamp_are_never_filled_directly():
    delayed = {OCC: {"bid": 1.10, "ask": 1.30, "last": None, "volume": 5, "t": FRESH_T,
                     "delayed": True, "spot": S}}
    assert _atm(_chain_with_fake_quotes(delayed, source="yfinance"))["source"] == "yfinance-calibrated"
    untimed = {OCC: {"bid": 1.10, "ask": 1.30, "last": None, "volume": 5}}
    assert _atm(_chain_with_fake_quotes(untimed))["source"] == "alpaca-calibrated"


def test_stale_quote_whose_iv_does_not_solve_falls_back_to_model():
    # a mid at intrinsic has no time value to solve for
    itm_occ = option_model.occ_symbol("SPY", EXPIRY, True, ATM - 5)
    quotes = {itm_occ: {"bid": 4.90, "ask": 5.10, "last": None, "volume": 5, "t": STALE_T, "spot": S}}
    chain = _chain_with_fake_quotes(quotes)
    assert next(r for r in chain if r["strike"] == ATM - 5)["source"] == "model"


def test_absurd_spread_is_rejected():
    quotes = {OCC: {"bid": 0.01, "ask": 20.0, "last": None, "volume": 500, "t": FRESH_T}}
    assert _atm(_chain_with_fake_quotes(quotes))["source"] == "model"


def test_quote_wildly_off_the_model_is_rejected():
    model_price = _atm(_chain_with_fake_quotes({}))["price"]
    quotes = {OCC: {"bid": model_price * 50, "ask": model_price * 51,
                    "last": None, "volume": 500, "t": FRESH_T}}
    assert _atm(_chain_with_fake_quotes(quotes))["source"] == "model"


def test_live_quote_lookup_for_open_position_respects_age_and_sanity():
    good = {OCC: {"bid": 1.10, "ask": 1.30, "last": 1.2, "volume": 500, "t": FRESH_T, "spot": S}}
    with patch.object(option_quotes, "get_chain", return_value=good), \
         patch.object(option_quotes, "_last_source", {"SPY": "alpaca"}):
        q = option_chain.live_quote("SPY", S, EXPIRY, ATM, True, model_price=1.20, now=NOW)
        assert q is not None and q["bid"] == 1.10 and q["source"] == "alpaca"
        q_rejected = option_chain.live_quote("SPY", S, EXPIRY, ATM, True, model_price=0.05, now=NOW)
        assert q_rejected is None  # 1.20 mid vs 0.05 model is >10x off -> rejected
    stale = {OCC: {"bid": 1.10, "ask": 1.30, "last": 1.2, "volume": 500, "t": STALE_T, "spot": S}}
    with patch.object(option_quotes, "get_chain", return_value=stale), \
         patch.object(option_quotes, "_last_source", {"SPY": "alpaca"}):
        q = option_chain.live_quote("SPY", S, EXPIRY, ATM, True, model_price=1.20, now=NOW)
        assert q is not None and q["source"] == "alpaca-calibrated"
        assert q["bid"] < q["ask"] and q["age_s"] > option_chain.MAX_QUOTE_AGE_S


def test_quote_age_helper():
    assert option_chain.quote_age_s({"t": FRESH_T}, NOW) == 0.0
    assert abs(option_chain.quote_age_s({"t": STALE_T}, NOW) - 1200) < 1e-6
    assert option_chain.quote_age_s({}, NOW) is None
    assert option_chain.quote_age_s({"t": "garbage"}, NOW) is None
