"""A journal holds ONE strategy, and credit spreads are a strategy.

Two rules under test:
  * a marked journal accepts only its own trade type, enforced in
    webapp/journal.py rather than only in the form — so a hand-rolled POST
    cannot smuggle another shape in;
  * `strategy='any'` keeps the old behaviour, which is what every journal
    that pre-dates the rule is migrated to.

Plus the credit-spread arithmetic, which is the one type whose money moves
the other way round: it is paid at entry and bought back at exit.
"""

import pytest
from fastapi.testclient import TestClient

from webapp import auth, config, journal
import webapp.main as main

UID = 1


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DB_PATH", tmp_path / "journal.db")
    monkeypatch.setattr(config, "AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr(auth, "_fails", {})
    auth.ensure_initialised()
    journal.init(default_owner=UID)
    yield


def _spread(jid, **over):
    v = {"ticker": "spy", "option_type": "put", "contracts": "2",
         "short_strike": "450", "long_strike": "445",
         "short_leg_open": "1.50", "long_leg_open": "0.30",
         "trade_type": "credit_spread", "account_value": "50000"}
    v.update(over)
    return journal.create_trade(jid, v)


# --- the lock --------------------------------------------------------------
def test_a_marked_journal_retypes_whatever_it_is_handed():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = journal.create_trade(jid, {"ticker": "SPY", "trade_type": "stock",
                                     "shares": "100", "open_price": "5"})
    assert journal.get_trade(tid)["trade_type"] == "credit_spread"

    journal.update_trade(tid, {"ticker": "SPY", "trade_type": "single_option",
                               "contracts": "1", "open_price": "2"})
    assert journal.get_trade(tid)["trade_type"] == "credit_spread"

    journal.patch_trade_fields(tid, {"ticker": "SPY", "trade_type": "calendar",
                                     "contracts": "1"})
    assert journal.get_trade(tid)["trade_type"] == "credit_spread"


def test_an_any_journal_still_takes_every_type():
    jid = journal.create_journal("Mixed", "live", UID, "any")
    for tt in journal.TRADE_TYPES:
        tid = journal.create_trade(jid, {"ticker": "SPY", "trade_type": tt})
        assert journal.get_trade(tid)["trade_type"] == tt


def test_journals_that_pre_date_the_rule_are_marked_any():
    """The migration must not quietly outlaw anything already logged."""
    with journal._conn() as c:
        c.execute("INSERT INTO journals(name, kind, created_at, user_id) "
                  "VALUES ('Old', 'live', '2026-01-01T00:00:00+00:00', ?)", (UID,))
        jid = c.execute("SELECT id FROM journals WHERE name='Old'").fetchone()["id"]
    journal.init(default_owner=UID)
    assert journal.strategy_of(jid) == "any"
    assert journal.forced_type(jid) is None
    for name, kind in journal.DEFAULT_JOURNALS:
        pass
    assert all(j["strategy"] == "any" for j in journal.list_journals(UID))


def test_an_unknown_strategy_falls_back_to_any():
    jid = journal.create_journal("Weird", "live", UID, "iron_condor")
    assert journal.strategy_of(jid) == "any"


# --- marking a journal after the fact --------------------------------------
def test_an_empty_journal_can_be_marked_with_anything():
    jid = journal.create_journal("Fresh", "live", UID)
    assert journal.set_strategy(jid, "credit_spread") is None
    assert journal.strategy_of(jid) == "credit_spread"


def test_marking_is_refused_when_it_would_falsify_existing_trades():
    jid = journal.create_journal("Mixed", "live", UID, "any")
    journal.create_trade(jid, {"ticker": "SPY", "trade_type": "stock", "shares": "10"})
    err = journal.set_strategy(jid, "credit_spread")
    assert err and "stock" in err.lower()
    assert journal.strategy_of(jid) == "any"           # unchanged

    # ...and "any" is always available to a journal with a mixed past
    assert journal.set_strategy(jid, "any") is None


def test_a_draft_does_not_block_marking():
    jid = journal.create_journal("Fresh", "live", UID, "any")
    journal.create_trade(jid, {"ticker": "SPY", "trade_type": "stock"}, draft=True)
    assert journal.set_strategy(jid, "credit_spread") is None


# --- credit-spread arithmetic ----------------------------------------------
def test_credit_spread_is_sized_on_max_risk_not_on_the_credit():
    """A 5-wide put spread sold for $1.20 takes in $240 on 2 contracts and
    risks $760. Sizing off the credit would understate it 3x."""
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    t = journal.get_trade(_spread(jid))
    assert t["credit_received"] == pytest.approx(240.0)
    assert t["max_risk"] == pytest.approx(760.0)
    assert t["position_amount_calc"] == pytest.approx(760.0)
    assert t["position_pct"] == pytest.approx(760 / 50000 * 100)
    assert t["contract_cost"] == pytest.approx(380.0)


def test_a_credit_spread_is_open_until_it_is_bought_back():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    assert journal.get_trade(tid)["is_closed"] is False
    assert journal.get_trade(tid)["pl_dollar"] is None


def test_buying_a_credit_spread_back_cheaper_is_a_win():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    assert journal.add_trim(tid, 2, 0.40) is None      # paid $0.40 to close
    t = journal.get_trade(tid)
    assert t["is_closed"] is True
    assert t["pl_dollar"] == pytest.approx(160.0)      # $240 in - $80 out
    assert t["is_win"] is True
    assert t["pl_pct"] == pytest.approx(160 / 760 * 100)


def test_a_spread_that_expires_worthless_keeps_the_whole_credit():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    assert journal.add_trim(tid, 2, 0) is None
    assert journal.get_trade(tid)["pl_dollar"] == pytest.approx(240.0)


def test_buying_it_back_dearer_is_a_loss_bounded_by_the_width():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    journal.add_trim(tid, 2, 5.00)                     # max loss: 5-wide
    t = journal.get_trade(tid)
    assert t["pl_dollar"] == pytest.approx(-760.0)     # = -max_risk
    assert t["is_win"] is False


def test_a_partial_buy_back_realizes_only_that_part():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    journal.add_trim(tid, 1, 0.20)
    t = journal.get_trade(tid)
    assert t["realized_pl"] == pytest.approx(100.0)    # (1.20 - 0.20) * 1 * 100
    assert t["remaining_qty"] == pytest.approx(1.0)
    assert t["is_closed"] is False


def test_scaling_into_a_credit_spread_averages_the_credit():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    assert journal.add_addon(tid, 2, 0.80) is None     # 2 more at $0.80 credit
    t = journal.get_trade(tid)
    assert t["total_qty"] == pytest.approx(4.0)
    assert t["avg_buy_cost"] == pytest.approx(1.00)    # (1.20*2 + 0.80*2) / 4
    assert t["credit_received"] == pytest.approx(400.0)
    assert t["max_risk"] == pytest.approx(1600.0)      # (5 - 1.00) * 4 * 100

    journal.add_trim(tid, 4, 0.50)
    assert journal.get_trade(tid)["pl_dollar"] == pytest.approx(200.0)


def test_max_risk_falls_back_to_the_credit_when_a_strike_is_missing():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    t = journal.get_trade(_spread(jid, long_strike=""))
    assert t["max_risk"] is None
    assert t["position_amount_calc"] == pytest.approx(240.0)


def test_a_credit_spread_has_no_direction_to_get_wrong():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    t = journal.get_trade(_spread(jid, direction="long"))
    assert t["direction"] is None
    assert "cr" in t["type_summary"] and "450/445P" in t["type_summary"]


def test_calls_and_puts_live_in_the_same_credit_spread_journal():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    put = journal.get_trade(_spread(jid, option_type="put"))
    call = journal.get_trade(_spread(jid, option_type="call",
                                     short_strike="460", long_strike="465"))
    assert (put["option_type"], call["option_type"]) == ("put", "call")
    assert put["max_risk"] == call["max_risk"] == pytest.approx(760.0)
    assert journal.aggregates(jid)["trade_count"] == 2


# --- through the web ------------------------------------------------------
@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    c = TestClient(main.app, follow_redirects=False)
    c.post("/login", data={"username": config.DEFAULT_USERNAME,
                           "password": config.DEFAULT_PASSWORD})
    c.post("/change-password", data={"new": "a-new-password", "confirm": "a-new-password"})
    return c


def _uid(web):
    return auth.get_user_by_username(config.DEFAULT_USERNAME)["id"]


def test_the_form_offers_one_type_on_a_marked_journal(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    r = web.get(f"/journal/{jid}/trade/new", follow_redirects=True)
    assert r.status_code == 200
    body = r.text
    assert 'name="trade_type" id="trade_type" value="credit_spread"' in body
    assert "<select name=\"trade_type\"" not in body
    assert "Calendar spread" not in body


def test_the_form_still_offers_every_type_on_an_any_journal(web):
    jid = journal.create_journal("Mixed", "live", _uid(web), "any")
    body = web.get(f"/journal/{jid}/trade/new", follow_redirects=True).text
    assert "<select name=\"trade_type\"" in body
    for label in ("Single option", "Calendar spread", "Credit spread", "Stock"):
        assert label in body


def test_a_type_on_the_url_cannot_override_a_marked_journal(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    r = web.get(f"/journal/{jid}/trade/new?type=stock")
    tid = int(r.headers["location"].rsplit("/", 1)[1])
    assert journal.get_trade(tid)["trade_type"] == "credit_spread"


def test_posting_another_type_into_a_marked_journal_is_retyped(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    r = web.post(f"/journal/{jid}/trade", data={"ticker": "SPY", "trade_type": "stock",
                                               "shares": "100", "open_price": "10"})
    assert r.status_code == 303
    assert journal.list_trades(jid)[0]["trade_type"] == "credit_spread"


def test_marking_a_journal_from_its_page(web):
    jid = journal.create_journal("Fresh", "live", _uid(web))
    assert web.post(f"/journal/{jid}/strategy", data={"strategy": "credit_spread"}).status_code == 303
    assert journal.strategy_of(jid) == "credit_spread"

    journal.create_trade(jid, {"ticker": "SPY"})
    r = web.post(f"/journal/{jid}/strategy", data={"strategy": "stock"})
    assert "strategy_err=" in r.headers["location"]
    assert journal.strategy_of(jid) == "credit_spread"


def test_another_users_journal_cannot_be_marked(web):
    other = auth.create_user("bob", "bobs-password")
    jid = journal.create_journal("Theirs", "live", other["id"], "any")
    web.post(f"/journal/{jid}/strategy", data={"strategy": "stock"})
    assert journal.strategy_of(jid) == "any"


def test_a_credit_spread_renders_end_to_end(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    tid = _spread(jid)
    journal.add_trim(tid, 2, 0.40)
    page = web.get(f"/journal/{jid}").text
    assert "Bull put" in page and "450/445P" in page
    assert web.get(f"/journal/{jid}/trade/{tid}").status_code == 200
    assert web.get(f"/journal/{jid}/symbols").status_code == 200


# --- the form asks for as little as possible -------------------------------
def test_the_form_asks_for_the_symbol_and_the_money_only(web):
    """Logging a trade should be quick: no earnings context, no screener
    readings, no timestamps to type."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    body = web.get(f"/journal/{jid}/trade/new", follow_redirects=True).text
    for gone in ('name="earnings_date"', 'name="earnings_session"',
                 'name="entry_at"', 'name="exit_at"', 'name="ts_slope"',
                 'name="avg_volume"', 'name="iv30_rv30"', 'name="recommendation"',
                 'name="flipped_earlier"', "Pull screener data"):
        assert gone not in body, gone
    assert 'name="ticker"' in body
    assert 'name="short_leg_open"' in body


def test_a_new_trade_is_stamped_with_its_entry_time(web):
    """Nobody types it, so it has to be recorded — otherwise the journal
    loses the order its trades were taken in."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    r = web.get(f"/journal/{jid}/trade/new")
    tid = int(r.headers["location"].rsplit("/", 1)[1])
    assert journal.get_trade(tid)["entry_at"]


def test_saving_an_old_trade_keeps_what_the_form_no_longer_asks(web):
    """The write covers every column, so a retired field would be nulled
    the first time an old trade is edited. Twelve real trades say no."""
    jid = journal.create_journal("Old", "live", _uid(web), "any")
    tid = journal.create_trade(jid, {
        "ticker": "AAPL", "trade_type": "single_option", "contracts": "1",
        "open_price": "2.00", "earnings_date": "2026-01-29",
        "earnings_session": "AMC", "entry_at": "2026-01-28T15:30",
        "exit_at": "2026-01-29T10:00", "ts_slope": "-0.005",
        "avg_volume": "2000000", "iv30_rv30": "1.4",
        "recommendation": "Recommended", "flipped_earlier": "1"})

    # exactly what the slimmed-down form posts — nothing else
    web.post(f"/journal/{jid}/trade/{tid}", data={
        "ticker": "AAPL", "trade_type": "single_option", "direction": "long",
        "option_type": "call", "contracts": "2", "open_price": "2.50"})

    t = journal.get_trade(tid)
    assert t["contracts"] == 2 and t["open_price"] == 2.50      # the edit landed
    assert t["earnings_date"] == "2026-01-29"
    assert t["earnings_session"] == "AMC"
    assert t["entry_at"] == "2026-01-28T15:30"
    assert t["exit_at"] == "2026-01-29T10:00"
    assert t["ts_slope"] == -0.005
    assert t["avg_volume"] == 2000000
    assert t["iv30_rv30"] == 1.4
    assert t["recommendation"] == "Recommended"
    assert t["flipped_earlier"] == 1


def test_recording_an_exit_keeps_them_too():
    """add/exit posts carry a form snapshot through patch_trade_fields,
    which is the other path that rewrites every column."""
    jid = journal.create_journal("Old", "live", UID, "any")
    tid = journal.create_trade(jid, {"ticker": "AAPL", "trade_type": "single_option",
                                     "contracts": "1", "open_price": "2.00",
                                     "recommendation": "Consider",
                                     "entry_at": "2026-01-28T15:30"})
    journal.patch_trade_fields(tid, {"ticker": "AAPL", "trade_type": "single_option",
                                     "contracts": "1", "open_price": "2.00"})
    t = journal.get_trade(tid)
    assert t["recommendation"] == "Consider" and t["entry_at"] == "2026-01-28T15:30"


def test_a_caller_that_does_pass_a_retired_field_still_wins():
    jid = journal.create_journal("Old", "live", UID, "any")
    tid = journal.create_trade(jid, {"ticker": "AAPL", "recommendation": "Avoid"})
    journal.update_trade(tid, {"ticker": "AAPL", "recommendation": "Recommended"})
    assert journal.get_trade(tid)["recommendation"] == "Recommended"


def test_a_closed_out_trade_reports_when_its_last_exit_landed():
    """The form no longer asks for an exit time, so the table needs one."""
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    tid = _spread(jid)
    assert journal.get_trade(tid)["exit_recorded_at"] is None
    journal.add_trim(tid, 2, 0.40)
    t = journal.get_trade(tid)
    assert t["exit_recorded_at"] and t["is_closed"] is True


def test_the_trade_table_drops_columns_nothing_fills_any_more(web):
    jid = journal.create_journal("Fresh", "live", _uid(web), "credit_spread")
    journal.add_trim(_spread(jid), 2, 0.40)
    page = web.get(f"/journal/{jid}").text
    assert "<th>Rec</th>" not in page and "<th>Earnings</th>" not in page
    assert "<th>Exp→Earn</th>" not in page

    # ...but a journal whose history has them keeps them legible
    old = journal.create_journal("Old", "live", _uid(web), "any")
    journal.create_trade(old, {"ticker": "AAPL", "trade_type": "single_option",
                               "recommendation": "Recommended",
                               "earnings_date": "2026-01-29"})
    page = web.get(f"/journal/{old}").text
    assert "<th>Rec</th>" in page and "<th>Earnings</th>" in page
    assert "<th>Exp→Earn</th>" in page


def test_the_screener_log_link_carries_only_a_ticker(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    page = web.get("/journal/log?ticker=SPY").text
    assert f"/journal/{jid}/trade/new?ticker=SPY&type=credit_spread" in page
    assert "prefill" not in page and "earnings=" not in page


# --- nothing renders as a question mark ------------------------------------
def test_a_blank_ticker_stays_blank_instead_of_becoming_a_question_mark(web):
    """A new trade used to be stored with ticker "?", which came straight
    back up in the form's own ticker box."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    r = web.get(f"/journal/{jid}/trade/new")
    tid = int(r.headers["location"].rsplit("/", 1)[1])
    assert journal.get_trade(tid)["ticker"] == ""
    assert 'name="ticker" value=""' in web.get(f"/journal/{jid}/trade/{tid}").text


def test_existing_question_mark_tickers_are_cleared_by_the_migration():
    jid = journal.create_journal("Old", "live", UID, "any")
    tid = journal.create_trade(jid, {"ticker": "AAPL"})
    with journal._conn() as c:
        c.execute("UPDATE trades SET ticker='?' WHERE id=?", (tid,))
    journal.init(default_owner=UID)
    assert journal.get_trade(tid)["ticker"] == ""


def test_a_half_filled_trade_summarises_only_what_it_has():
    jid = journal.create_journal("Mixed", "live", UID, "any")
    for values, expected in [
        ({"trade_type": "single_option", "contracts": "2", "strike": "450",
          "option_type": "call", "expiration": "2026-10-17"},
         "Long 2x 450C 10/17"),
        ({"trade_type": "single_option", "contracts": "2"}, "Long 2x"),
        ({"trade_type": "single_option"}, "Long"),
        ({"trade_type": "stock", "direction": "short", "shares": "100"}, "Short 100 sh"),
        ({"trade_type": "stock"}, "Long"),
        ({"trade_type": "credit_spread", "contracts": "2", "short_strike": "450",
          "long_strike": "445", "option_type": "put", "expiration": "2026-10-17"},
         "2x 450/445P cr 10/17"),
        ({"trade_type": "credit_spread", "contracts": "2", "short_strike": "450"},
         "2x 450 cr"),
        ({"trade_type": "credit_spread"}, "cr"),
        ({"trade_type": "calendar", "contracts": "1", "strike": "450",
          "front_expiration": "2026-09-05", "back_expiration": "2026-10-17"},
         "1x 450 cal 09/05->10/17"),
        ({"trade_type": "calendar", "contracts": "1"}, "1x cal"),
    ]:
        t = journal.get_trade(journal.create_trade(jid, {"ticker": "SPY", **values}))
        assert t["type_summary"] == expected, values
        assert "?" not in t["type_summary"]


def test_an_empty_trade_row_carries_no_question_marks(web):
    jid = journal.create_journal("Mixed", "live", _uid(web), "any")
    journal.create_trade(jid, {})
    page = web.get(f"/journal/{jid}").text
    body = page.split("<tbody>")[1].split("</tbody>")[0]
    assert "?" not in body
    assert "—" in body


# --- the position is entered as an order ticket -----------------------------
import re


def _legs(html, tt):
    """The leg rows of one type's ticket: (row classes, row html)."""
    fs = html.split('data-type="%s"' % tt, 1)[1].split("</fieldset>", 1)[0]
    return re.findall(r'<tr class="leg ([a-z]+)">(.*?)</tr>', fs, re.S)


def test_a_credit_spread_is_two_rows_short_red_and_long_green(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    tid = _spread(jid)
    rows = _legs(web.get(f"/journal/{jid}/trade/{tid}").text, "credit_spread")
    assert [side for side, _ in rows] == ["sell", "buy"]

    short, long_ = rows[0][1], rows[1][1]
    assert "Short" in short and 'name="short_strike"' in short and 'name="short_leg_open"' in short
    assert "Long" in long_ and 'name="long_strike"' in long_ and 'name="long_leg_open"' in long_
    # the side of a spread leg is fixed by what the leg IS, not a dropdown
    assert "sidesel" not in short and "sidesel" not in long_


def test_a_calendar_sells_the_front_row_and_buys_the_back_one(web):
    jid = journal.create_journal("Cals", "live", _uid(web), "calendar")
    tid = journal.create_trade(jid, {"ticker": "SPY", "contracts": "1", "strike": "450"})
    rows = _legs(web.get(f"/journal/{jid}/trade/{tid}").text, "calendar")
    assert [side for side, _ in rows] == ["sell", "buy"]
    assert 'name="front_expiration"' in rows[0][1] and 'name="back_expiration"' in rows[1][1]


def test_a_one_legged_trade_takes_its_side_from_the_row_itself(web):
    """A single option or a stock position can be either way round, so its
    row carries a Buy/Sell select and is coloured from it."""
    jid = journal.create_journal("Opts", "live", _uid(web), "single_option")
    for direction, expected in (("long", "buy"), ("short", "sell")):
        tid = journal.create_trade(jid, {"ticker": "SPY", "direction": direction,
                                         "contracts": "1"})
        rows = _legs(web.get(f"/journal/{jid}/trade/{tid}").text, "single_option")
        assert len(rows) == 1
        assert rows[0][0] == expected
        assert "sidesel" in rows[0][1]


def test_both_rows_of_a_spread_carry_the_fields_they_share(web):
    """Quantity, call/put and a vertical's expiration are one value in the
    database but belong on every row of a ticket — they are marked for the
    sync that keeps them from disagreeing."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    tid = _spread(jid)
    rows = _legs(web.get(f"/journal/{jid}/trade/{tid}").text, "credit_spread")
    for _side, row in rows:
        for shared in ('name="contracts"', 'name="option_type"', 'name="expiration"'):
            assert shared in row, shared
            assert "data-sync" in row


def test_a_whole_number_is_not_shown_with_a_trailing_decimal(web):
    """395.0 is not a strike anybody types."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    tid = _spread(jid, account_value="50000")
    body = web.get(f"/journal/{jid}/trade/{tid}").text
    assert 'value="450"' in body and 'value="445"' in body
    assert 'value="50000"' in body
    assert "450.0" not in body and "50000.0" not in body
    assert 'value="1.5"' in body          # a real decimal is left alone


# --- call/put is a switch, and the two spreads are named --------------------
def test_the_type_cell_is_a_button_not_a_dropdown(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    tid = _spread(jid)
    body = web.get(f"/journal/{jid}/trade/{tid}").text
    rows = _legs(body, "credit_spread")
    for _side, row in rows:
        assert 'data-optype' in row                      # the switch
        assert 'name="option_type" value="put"' in row   # what actually posts
    assert "<select name=\"option_type\"" not in body


def test_the_same_switch_is_on_a_single_option(web):
    jid = journal.create_journal("Opts", "live", _uid(web), "single_option")
    tid = journal.create_trade(jid, {"ticker": "SPY", "option_type": "call",
                                     "contracts": "1"})
    row = _legs(web.get(f"/journal/{jid}/trade/{tid}").text, "single_option")[0][1]
    assert "data-optype" in row and 'name="option_type" value="call"' in row
    assert ">Call<" in row


def test_a_put_credit_spread_and_a_call_credit_spread_are_named_apart(web):
    """The two are opposite trades — bullish vs bearish — so the ticket
    says which one is being entered instead of leaving it to a Put/Call
    cell to imply."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    put = web.get(f"/journal/{jid}/trade/{_spread(jid, option_type='put')}").text
    assert "Bull put spread" in put
    assert "bullish / neutral" in put and "ABOVE your short strike" in put

    call = web.get(f"/journal/{jid}/trade/{_spread(jid, option_type='call')}").text
    assert "Bear call spread" in call
    assert "bearish / neutral" in call and "BELOW your short strike" in call


def test_the_switch_still_saves_the_option_type_it_is_left_on(web):
    """The button is only chrome — a hidden field is what posts."""
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    tid = _spread(jid, option_type="put")
    web.post(f"/journal/{jid}/trade/{tid}", data={
        "ticker": "SPY", "trade_type": "credit_spread", "option_type": "call",
        "contracts": "2", "short_strike": "460", "long_strike": "465",
        "short_leg_open": "1.5", "long_leg_open": "0.3"})
    t = journal.get_trade(tid)
    assert t["option_type"] == "call"
    assert "460/465C" in t["type_summary"]


# --- the trades table names the spread, not just its shape -----------------
def test_the_table_tells_a_bull_put_from_a_bear_call():
    """Both are credit spreads, but one is bullish and the other bearish —
    labelling both "CrSp" hides the only thing that separates them."""
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    put = journal.get_trade(_spread(jid, option_type="put"))
    call = journal.get_trade(_spread(jid, option_type="call",
                                     short_strike="460", long_strike="465"))
    assert (put["type_label"], call["type_label"]) == ("Bull put", "Bear call")
    assert "bullish" in put["type_title"] and "bearish" in call["type_title"]


def test_a_spread_with_no_option_type_keeps_the_generic_badge():
    jid = journal.create_journal("Spreads", "live", UID, "credit_spread")
    t = journal.get_trade(_spread(jid, option_type=""))
    assert t["type_label"] == "CrSp"


def test_the_other_types_keep_their_badges():
    jid = journal.create_journal("Mixed", "live", UID, "any")
    for tt, label in (("calendar", "Cal"), ("single_option", "Opt"), ("stock", "Stk")):
        t = journal.get_trade(journal.create_trade(jid, {"ticker": "SPY", "trade_type": tt}))
        assert t["type_label"] == label


def test_both_spread_names_show_up_on_the_page(web):
    jid = journal.create_journal("Spreads", "live", _uid(web), "credit_spread")
    _spread(jid, option_type="put")
    _spread(jid, option_type="call", short_strike="460", long_strike="465")
    page = web.get(f"/journal/{jid}").text
    assert "Bull put" in page and "Bear call" in page
    assert ">CrSp<" not in page
