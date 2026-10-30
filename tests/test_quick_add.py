"""Quick add: one typed line becomes a credit spread, and asks for the rest.

The line is shorthand a trader already speaks — "AAL 10/16 $11/12 put
credit spread" — so the tests are written in the same shorthand, including
the sloppy versions (no $, strikes the other way round, lowercase).

The load-bearing rule is which leg is which: it follows from the strategy,
not from the order the strikes were typed. Get that backwards and the
journal records the opposite trade.
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from webapp import auth, config, journal, quick_add
import webapp.main as main

UID = 1
TODAY = date(2026, 9, 15)


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DB_PATH", tmp_path / "journal.db")
    monkeypatch.setattr(config, "AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr(auth, "_fails", {})
    auth.ensure_initialised()
    journal.init(default_owner=UID)
    yield


def _p(line, today=TODAY):
    values, err = quick_add.parse_credit_spread(line, today=today)
    assert err is None, err
    return values


# --- the line the feature was asked for ------------------------------------
def test_the_example_line():
    v = _p("AAL 10/16 $11/12 Put Credit Spread")
    assert v == {"trade_type": "credit_spread", "ticker": "AAL", "option_type": "put",
                 "expiration": "2026-10-16", "short_strike": 12.0, "long_strike": 11.0}


# --- which leg is which is decided by the strategy, not the typing ---------
def test_a_put_credit_spread_sells_the_higher_strike():
    for line in ("AAL 10/16 $11/12 put credit spread",
                 "AAL 10/16 $12/11 put credit spread"):
        v = _p(line)
        assert (v["short_strike"], v["long_strike"]) == (12.0, 11.0), line


def test_a_call_credit_spread_sells_the_lower_strike():
    for line in ("AAL 10/16 $11/12 call credit spread",
                 "AAL 10/16 $12/11 call credit spread"):
        v = _p(line)
        assert (v["short_strike"], v["long_strike"]) == (11.0, 12.0), line


def test_the_short_leg_is_always_the_one_nearer_the_money():
    """Put: sell high, buy lower. Call: sell low, buy higher. Same strikes,
    opposite assignment — this is the whole reason the parser exists."""
    put = _p("SPY 10/16 $640/635 put credit spread")
    call = _p("SPY 10/16 $640/635 call credit spread")
    assert put["short_strike"] == call["long_strike"] == 640.0
    assert put["long_strike"] == call["short_strike"] == 635.0


# --- reading the line loosely ---------------------------------------------
def test_the_dollar_sign_is_optional_and_the_date_comes_first():
    v = _p("NVDA 10/16 185/190 call credit spread")
    assert v["expiration"] == "2026-10-16"
    assert (v["short_strike"], v["long_strike"]) == (185.0, 190.0)


def test_an_explicit_year_is_honoured():
    assert _p("SPY 10/16/26 $640/635 put")["expiration"] == "2026-10-16"
    assert _p("SPY 10/16/2027 $640/635 put")["expiration"] == "2027-10-16"


def test_a_bare_date_rolls_forward_rather_than_landing_in_the_past():
    """Nobody quick-adds a trade into a month that has already ended."""
    assert _p("TSLA 10/16 $395/390 put")["expiration"] == "2026-10-16"
    assert _p("TSLA 9/1 $395/390 put")["expiration"] == "2027-09-01"
    assert _p("TSLA 9/15 $395/390 put")["expiration"] == "2026-09-15"   # today counts


def test_case_and_wording_are_loose():
    for line in ("aal 10/16 $11/12 put credit spread", "AAL 10/16 $11/12 PUT",
                 "AAL 10/16 $11/12 bull put spread", "AAL 10/16 $11/12 puts"):
        assert _p(line)["option_type"] == "put", line
    for line in ("AAL 10/16 $11/12 call", "AAL 10/16 $11/12 bear call spread"):
        assert _p(line)["option_type"] == "call", line


def test_decimal_strikes_survive():
    v = _p("SPY 10/16 $11.5/12.5 put")
    assert (v["short_strike"], v["long_strike"]) == (12.5, 11.5)


def test_the_ticker_is_not_confused_with_the_strategy_words():
    assert _p("AAL 10/16 $11/12 put credit spread")["ticker"] == "AAL"
    assert _p("sell AAL 10/16 $11/12 put credit spread")["ticker"] == "AAL"


# --- what it refuses, and how it says so -----------------------------------
@pytest.mark.parametrize("line, wants", [
    ("", "Type the spread"),
    ("AAL 10/16 $11/12", "put or a call"),
    ("AAL $11/12 put", "expiration"),
    ("AAL 10/16 $11/11 put", "different"),
    ("10/16 $11/12 put", "ticker"),
    ("AAL 13/45 $11/12 put", "Couldn't read 13/45 as an expiration"),
    ("AAL put credit spread", "strikes"),
])
def test_an_unreadable_line_says_what_is_missing(line, wants):
    values, err = quick_add.parse_credit_spread(line, today=TODAY)
    assert values is None
    assert wants in err, err


# --- through the web -------------------------------------------------------
@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    c = TestClient(main.app, follow_redirects=False)
    c.post("/login", data={"username": config.DEFAULT_USERNAME,
                           "password": config.DEFAULT_PASSWORD})
    c.post("/change-password", data={"new": "a-new-password", "confirm": "a-new-password"})
    return c


def _uid():
    return auth.get_user_by_username(config.DEFAULT_USERNAME)["id"]


def test_quick_adding_lands_on_the_form_with_the_rest_to_fill_in(web):
    jid = journal.create_journal("Spreads", "live", _uid(), "credit_spread")
    r = web.post(f"/journal/{jid}/quick-add",
                 data={"line": "AAL 10/16 $11/12 put credit spread"})
    assert r.status_code == 303 and "quick=1" in r.headers["location"]

    tid = int(r.headers["location"].split("/trade/")[1].split("?")[0])
    t = journal.get_trade(tid)
    assert t["trade_type"] == "credit_spread" and t["ticker"] == "AAL"
    assert (t["short_strike"], t["long_strike"]) == (12.0, 11.0)
    assert t["option_type"] == "put" and t["expiration"].startswith("2026-10-16")
    assert t["draft"] == 1                       # not a trade until it is finished
    assert t["entry_at"]                         # stamped like any other new trade
    # the numbers a line cannot carry are left for the form
    assert t["contracts"] is None
    assert t["short_leg_open"] is None and t["long_leg_open"] is None
    assert t["account_value"] is None

    page = web.get(r.headers["location"]).text
    assert "sell the 12 put, buy the 11 put" in page
    assert 'name="short_strike"' in page and 'value="12"' in page


def test_a_draft_does_not_count_as_a_trade_until_it_is_saved(web):
    jid = journal.create_journal("Spreads", "live", _uid(), "credit_spread")
    web.post(f"/journal/{jid}/quick-add", data={"line": "AAL 10/16 $11/12 put"})
    assert journal.aggregates(jid)["trade_count"] == 0


def test_an_unreadable_line_comes_back_to_the_journal_with_the_reason(web):
    jid = journal.create_journal("Spreads", "live", _uid(), "credit_spread")
    r = web.post(f"/journal/{jid}/quick-add", data={"line": "AAL 10/16 $11/12"})
    assert r.status_code == 303 and "quick_err=" in r.headers["location"]
    assert journal.list_trades(jid) == []
    assert "put or a call" in web.get(r.headers["location"]).text


def test_the_box_is_offered_on_credit_spread_and_mixed_journals(web):
    spreads = journal.create_journal("Spreads", "live", _uid(), "credit_spread")
    mixed = journal.create_journal("Mixed", "live", _uid(), "any")
    for jid in (spreads, mixed):
        assert "/quick-add" in web.get(f"/journal/{jid}").text


def test_a_journal_marked_for_another_strategy_neither_offers_nor_accepts_it(web):
    """Its strategy would retype the spread on the way in, quietly turning
    a parsed credit spread into a differently-shaped trade."""
    jid = journal.create_journal("Opts", "live", _uid(), "single_option")
    assert "/quick-add" not in web.get(f"/journal/{jid}").text

    r = web.post(f"/journal/{jid}/quick-add", data={"line": "AAL 10/16 $11/12 put"})
    assert "quick_err=" in r.headers["location"]
    assert journal.list_trades(jid) == []
    assert "only makes credit spreads" in web.get(r.headers["location"]).text


def test_another_users_journal_cannot_be_quick_added_to(web):
    other = auth.create_user("bob", "bobs-password")
    jid = journal.create_journal("Theirs", "live", other["id"], "credit_spread")
    web.post(f"/journal/{jid}/quick-add", data={"line": "AAL 10/16 $11/12 put"})
    assert journal.list_trades(jid) == []


def test_the_box_is_empty_not_pre_filled_with_an_example(web):
    """An example sitting in the field reads as a value that is already
    there. The help text under the box carries the explanation instead,
    and a line it can't parse answers with an example of its own."""
    jid = journal.create_journal("Spreads", "live", _uid(), "credit_spread")
    page = web.get(f"/journal/{jid}").text
    box = page.split('action="/journal/%d/quick-add"' % jid, 1)[1].split("</form>", 1)[0]
    assert 'name="line"' in box
    assert "placeholder" not in box
