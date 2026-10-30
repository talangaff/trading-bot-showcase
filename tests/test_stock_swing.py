"""Stock Swings: a journal type sized off the stop.

You give the entry, the stop, the dollars you're willing to lose there and
one or more take-profit prices. The journal works out how many shares to
buy so the stop costs (at most) that max loss, and every take-profit sells
the same number of shares — fractional, to 2 decimal places, rounded DOWN
so the loss at the stop never exceeds the max loss.
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


def _swing(jid, **over):
    v = {"ticker": "aapl", "open_price": "50", "stop_loss": "48",
         "max_loss": "500", "take_profits": "52,55,58", "account_value": "25000"}
    v.update(over)
    return journal.create_trade(jid, v)


# --- the plan ---------------------------------------------------------------
def test_shares_come_from_the_stop_and_split_evenly_across_targets():
    # $2 risk a share -> 250 shares raw; 3 targets -> 83.333.. each,
    # rounded down to 83.33, so 249.99 total
    p = journal.swing_plan(50, 48, 500, [52, 55, 58])
    assert p["direction"] == "long"
    assert p["per_trim"] == 83.33
    assert p["shares"] == 249.99
    assert p["loss_at_stop"] == pytest.approx(499.98)
    assert p["loss_at_stop"] <= 500
    assert p["max_profit"] == pytest.approx(83.33 * (2 + 5 + 8))
    assert [x["qty"] for x in p["trims"]] == [83.33, 83.33, 83.33]


def test_each_trim_shows_its_percent_gain_on_the_shares_it_sells():
    # 83.33 sh cost $4,166.50 at $50; selling at $52 makes $166.66 = +4%
    p = journal.swing_plan(50, 48, 500, [52, 55, 58])
    assert [x["cost"] for x in p["trims"]] == [pytest.approx(4166.5)] * 3
    assert [x["pct"] for x in p["trims"]] == [pytest.approx(v) for v in (4.0, 10.0, 16.0)]
    # a short gains as the price falls: $50 -> $45 is +10%
    assert journal.swing_plan(50, 52, 500, [45])["trims"][0]["pct"] == pytest.approx(10.0)


def test_fractional_shares_are_kept_to_two_decimals():
    # $0.70 risk, $100 max loss, 1 target -> 142.857.. -> 142.85 (never up)
    p = journal.swing_plan(10, 9.3, 100, [12])
    assert p["shares"] == 142.85
    assert p["loss_at_stop"] <= 100


def test_an_exact_split_loses_exactly_the_max_loss():
    p = journal.swing_plan(100, 95, 1000, [110, 120])   # 200 shares, 100 a trim
    assert p["shares"] == 200 and p["per_trim"] == 100
    assert p["loss_at_stop"] == pytest.approx(1000.0)
    assert p["unused_risk"] == pytest.approx(0.0)
    assert p["max_profit"] == pytest.approx(100 * 10 + 100 * 20)


def test_float_noise_does_not_cost_a_share():
    # 300 / 1.5 / 2 is 99.999... in floating point
    assert journal.swing_plan(10, 8.5, 300, [12, 13])["per_trim"] == 100
    # 0.29 / 0.01 is 28.999... — must stay 29 hundredths, not 28
    assert journal.swing_plan(10, 9, 0.29, [12])["per_trim"] == 0.29


def test_a_stop_above_the_entry_is_a_short():
    p = journal.swing_plan(50, 52, 500, [48, 45])
    assert p["direction"] == "short"
    assert p["shares"] == 250 and p["per_trim"] == 125
    assert p["max_profit"] == pytest.approx(125 * 2 + 125 * 5)


def test_targets_are_ordered_nearest_first():
    p = journal.swing_plan(50, 48, 500, [58, 52, 55])
    assert [x["price"] for x in p["trims"]] == [52, 55, 58]


def test_a_target_on_the_wrong_side_is_flagged():
    p = journal.swing_plan(50, 48, 500, [52, 47])
    assert p["warnings"] and "47" in p["warnings"][0]


@pytest.mark.parametrize("args, bit", [
    ((None, 48, 500, [52]), "open price"),
    ((50, 50, 500, [52]), "can't be the open price"),
    ((50, 48, 0, [52]), "Max loss"),
    ((50, 48, 500, []), "take-profit"),
    ((50, 48, 0.03, [52, 55]), "not enough"),         # 0.015 shares over 2 targets
])
def test_a_plan_that_cannot_be_made_says_why(args, bit):
    p = journal.swing_plan(*args)
    assert p["shares"] is None
    assert bit in p["error"]


# --- stored trades ----------------------------------------------------------
def test_a_swing_journal_stores_the_planned_share_count():
    jid = journal.create_journal("Swings", "live", UID, "stock_swing")
    # whatever `shares` arrives with, the plan wins
    t = journal.get_trade(_swing(jid, shares="9999"))
    assert t["trade_type"] == "stock_swing"
    assert t["shares"] == 249.99
    assert t["direction"] == "long"
    assert t["take_profits"] == "52,55,58"
    assert t["swing"]["per_trim"] == 83.33
    assert t["swing"]["max_profit"] == pytest.approx(1249.95)
    assert t["position_amount_calc"] == pytest.approx(50 * 249.99)
    assert "SL 48" in t["type_summary"] and "3 TP" in t["type_summary"]


def test_exits_at_every_target_realize_the_max_profit():
    jid = journal.create_journal("Swings", "live", UID, "stock_swing")
    tid = _swing(jid)
    for px in (52, 55, 58):
        assert journal.add_trim(tid, 83.33, px) is None
    t = journal.get_trade(tid)
    assert t["fully_exited"] and t["is_closed"]
    assert t["pl_dollar"] == pytest.approx(t["swing"]["max_profit"])


def test_stopping_out_loses_the_planned_amount():
    jid = journal.create_journal("Swings", "live", UID, "stock_swing")
    tid = _swing(jid)
    journal.add_trim(tid, 249.99, 48)
    assert journal.get_trade(tid)["pl_dollar"] == pytest.approx(-499.98)


def test_a_short_swing_profits_as_the_price_falls():
    jid = journal.create_journal("Swings", "live", UID, "stock_swing")
    tid = _swing(jid, stop_loss="52", take_profits="48,45")
    journal.add_trim(tid, 125, 48)
    journal.add_trim(tid, 125, 45)
    assert journal.get_trade(tid)["pl_dollar"] == pytest.approx(875.0)


def test_other_types_do_not_keep_swing_fields():
    jid = journal.create_journal("Mixed", "live", UID, "any")
    tid = journal.create_trade(jid, {"ticker": "SPY", "trade_type": "stock", "shares": "10",
                                     "stop_loss": "1", "take_profits": "5"})
    t = journal.get_trade(tid)
    assert t["stop_loss"] is None and t["take_profits"] is None and "swing" not in t


# --- through the web ------------------------------------------------------
@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    c = TestClient(main.app, follow_redirects=False)
    c.post("/login", data={"username": config.DEFAULT_USERNAME,
                           "password": config.DEFAULT_PASSWORD})
    c.post("/change-password", data={"new": "a-new-password", "confirm": "a-new-password"})
    return c


def test_a_swing_journal_can_be_created_and_logged_to(web):
    r = web.post("/journal/new", data={"name": "Swings", "kind": "live",
                                       "strategy": "stock_swing"})
    jid = int(r.headers["location"].rsplit("/", 1)[1])
    assert journal.strategy_of(jid) == "stock_swing"
    assert "Stock Swings" in web.get(f"/journal/{jid}").text

    r = web.get(f"/journal/{jid}/trade/new")
    tid = int(r.headers["location"].rsplit("/", 1)[1])
    body = web.get(f"/journal/{jid}/trade/{tid}").text
    for name in ('name="stop_loss"', 'name="max_loss"', 'name="take_profits"', "tp-input"):
        assert name in body

    web.post(f"/journal/{jid}/trade/{tid}", data={
        "trade_type": "stock_swing", "ticker": "AAPL", "open_price": "50",
        "stop_loss": "48", "max_loss": "500", "take_profits": "52,55,58"})
    t = journal.get_trade(tid)
    assert t["shares"] == 249.99 and not t["draft"]

    # the saved trade comes back with each target, and exits start at one trim
    body = web.get(f"/journal/{jid}/trade/{tid}").text
    for px in ("52", "55", "58"):
        assert 'value="{}"'.format(px) in body
    assert 'value="83.33"' in body
    assert "Swing" in web.get(f"/journal/{jid}").text
