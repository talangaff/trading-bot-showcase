"""A bot may never commit more than its position guardrail — 10% by default
— of ITS OWN account value to open positions.

"Its own account value" is the bot's start balance plus the P&L of every
trade it has closed: $10,000 to begin with, more once it has won, less once
it has lost. It is never the Alpaca account behind the fleet, which belongs
to a hundred and fifty bots at once and is orders of magnitude larger.

The rule is easy to state and was easy to leak, because a live fill is
booked forced=True — a fill that has already happened on a real account
cannot be un-happened by a bookkeeping rule — so bots.record_trade's cap
check, the one every offline path relies on, never runs on a live order.
Everything protecting a live bot therefore has to happen BEFORE the order
is submitted. These tests pin down each of those places.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from webapp import (account_book, bot_engine, bot_settings, bots, broker,
                    genome, option_chain, option_model, option_quotes)

ET = option_model.ET
UID = 1
KEYS = ("FLEETKEY", "FLEETSECRET")


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_settings, "_PATH", tmp_path / "bot_settings.json")
    monkeypatch.setattr(bot_settings, "_USER_PATH", tmp_path / "user_bot_settings.json")
    account_book.forget()
    yield


def _bot(**genes):
    g = genome.random_genome()
    g.update(genes)
    return bots.get_bot(bots.spawn_bot(g, origin="seed", user_id=UID))


def _contract(strike=560.0, days=0, is_call=True):
    exp = (datetime.now(ET) + timedelta(days=days)).replace(
        hour=16, minute=0, second=0, microsecond=0)
    return option_model.occ_symbol("SPY", exp, is_call, strike)


def _ts():
    return datetime.now(ET).replace(hour=11, minute=0, second=0,
                                    microsecond=0).isoformat()


class FakeAlpaca:
    """broker.submit_and_wait, recording every order.

    The market is at `fill_price` and the bot does not get a vote: a market
    order pays it, and a limit order fills only if it reaches — which is
    exactly the property the cap now leans on, so the fake has to honour it
    or the tests would be checking nothing."""

    def __init__(self, fill_price=1.11):
        self.fill_price = fill_price
        self.orders = []

    def __call__(self, symbol, qty, side, order_type="market", limit_price=None,
                 stop_price=None, trail_percent=None, position_intent=None,
                 timeout=8.0, poll=1.0, keys=None, client_order_id=None):
        self.orders.append({"qty": qty, "type": order_type, "limit": limit_price,
                            "side": side})
        if order_type == "limit" and limit_price is not None:
            if side == "buy" and limit_price < self.fill_price:
                return 0, None, "o1", "order not filled (canceled) — cancelled"
            return qty, min(self.fill_price, limit_price), "o1", None
        return qty, self.fill_price, "o1", None


@pytest.fixture
def live(monkeypatch):
    """A fleet with keys, an account with effectively unlimited buying power,
    and a stubbed Alpaca. Returns the FakeAlpaca so a test can read back the
    order that actually went out."""
    bot_settings.save_user(UID, fleet_alpaca_key_id=KEYS[0],
                           fleet_alpaca_secret_key=KEYS[1])
    fake = FakeAlpaca()
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "can_afford", lambda keys, cost: (True, 1e9))
    monkeypatch.setattr(account_book, "adjust", lambda *a, **k: None)
    monkeypatch.setattr(account_book, "charge", lambda *a, **k: None)
    return fake


def _spend(order) -> float:
    return order["qty"] * order["limit"] * bots.CONTRACT_MULT


# ---------------------------------------------------------------------------
# what the cap is a percentage OF
# ---------------------------------------------------------------------------
def test_the_cap_is_a_share_of_the_bots_own_balance_not_the_alpaca_account(live):
    """A fresh bot starts at $10,000 of its own, so 10% is $1,000 — however
    many millions the shared paper account happens to hold."""
    bot = _bot()
    hr = bots.position_headroom(bot["id"])
    assert hr["balance"] == bots.START_BALANCE == 10_000.0
    assert hr["cap"] == pytest.approx(1_000.0)
    assert hr["headroom"] == pytest.approx(1_000.0)


def test_the_cap_moves_with_what_the_bot_has_made_or_lost():
    """10% of the CURRENT value: a bot that has grown to $12,000 may commit
    $1,200; one that has lost its way down to $8,000 may commit $800."""
    for exit_px, expect in ((45.00, 1_200.0), (5.00, 800.0)):
        bot = _bot()
        c = _contract()
        # one round trip: in at $25.00, out at $45.00 (+$2,000) or $5.00 (-$2,000)
        assert not bots.record_trade(bot["id"], c, "buy_to_open", 1, 25.00,
                                     symbol="SPY", ts=_ts(), forced=True)
        assert not bots.record_trade(bot["id"], c, "sell_to_close", 1, exit_px,
                                     symbol="SPY", ts=_ts(), forced=True)
        pl = (exit_px - 25.00) * bots.CONTRACT_MULT
        hr = bots.position_headroom(bot["id"])
        assert hr["balance"] == pytest.approx(10_000.0 + pl)
        assert hr["cap"] == pytest.approx(expect)


def test_open_positions_spend_the_cap_they_do_not_each_get_one():
    """The cap is on exposure, not per trade — $700 open leaves $300, not
    another $1,000."""
    bot = _bot()
    assert not bots.record_trade(bot["id"], _contract(), "buy_to_open", 7, 1.00,
                                 symbol="SPY", ts=_ts(), forced=True)
    hr = bots.position_headroom(bot["id"])
    assert hr["open_cost"] == pytest.approx(700.0)
    assert hr["headroom"] == pytest.approx(300.0)
    # and the ledger check agrees: $400 more is refused, $300 is not
    assert "position cap exceeded" in bots.record_trade(
        bot["id"], _contract(561.0), "buy_to_open", 4, 1.00, symbol="SPY", ts=_ts())
    assert not bots.record_trade(bot["id"], _contract(561.0), "buy_to_open", 3, 1.00,
                                 symbol="SPY", ts=_ts())


def test_contracts_are_floored_never_rounded_up_to_the_cap():
    assert bots.max_contracts_for(1_000.0, 1.11) == 9      # 9 x $111 = $999
    assert bots.max_contracts_for(1_000.0, 1.00) == 10     # exactly 10 fits
    assert bots.max_contracts_for(100.0, 1.11) == 0        # not even one
    assert bots.max_contracts_for(-50.0, 1.11) == 0        # already over


# ---------------------------------------------------------------------------
# the live path: the order that goes to Alpaca is bounded
# ---------------------------------------------------------------------------
def test_a_wrong_chain_price_cannot_buy_more_than_the_cap(live):
    """The regression this file exists for. A thin strike modeled at $0.06
    that the market actually prices at $1.11 used to be sized as
    $1,000 / $6 = 166 contracts and sent out as a MARKET order, which
    Alpaca filled at the real price — $18,000 of a $1,000 cap.

    The bot may still be wrong about the price. It may not spend more than
    its cap being wrong: the order goes out as a limit at the cap's own
    price, so a market that far away simply does not fill."""
    bot = _bot()
    bot_engine._record_fill(bot, _contract(), "buy_to_open", 166, 0.06,
                            symbol="SPY", order_type="market", ts=_ts())
    order, = live.orders
    assert order["type"] == "limit"                      # never an unbounded market buy
    assert _spend(order) <= 1_000.0 + 1e-6, order
    assert order["limit"] >= 0.06                        # would still pay what it expected
    # the market was 18x away, so nothing filled — and nothing was booked
    assert bots.position_headroom(bot["id"])["open_cost"] == 0.0


def test_the_cap_does_not_stop_a_correctly_priced_entry(live):
    """The bound is the guardrail, not a brake: when the bot has the price
    right, it still buys every contract the cap affords."""
    live.fill_price = 0.06
    bot = _bot()
    assert not bot_engine._record_fill(bot, _contract(), "buy_to_open", 166, 0.06,
                                       symbol="SPY", order_type="market", ts=_ts())
    order, = live.orders
    assert order["qty"] == 166                           # 166 x $6 = $996
    assert bots.position_headroom(bot["id"])["open_cost"] == pytest.approx(996.0)


def test_a_limit_entry_is_clamped_to_the_cap_too(live):
    """A genome that asked for a limit does not get to name a price that
    breaches the cap either."""
    bot = _bot()
    bot_engine._record_fill(bot, _contract(), "buy_to_open", 166, 0.06,
                            symbol="SPY", order_type="limit",
                            limit_price=9.99, ts=_ts())
    order, = live.orders
    assert _spend(order) <= 1_000.0 + 1e-6, order


def test_a_bot_already_at_its_cap_is_refused_not_shrunk_to_nothing(live):
    bot = _bot()
    assert not bots.record_trade(bot["id"], _contract(), "buy_to_open", 10, 1.00,
                                 symbol="SPY", ts=_ts(), forced=True)
    err = bot_engine._record_fill(bot, _contract(561.0), "buy_to_open", 5, 1.00,
                                  symbol="SPY", order_type="market", ts=_ts())
    assert err and "position cap reached" in err
    assert live.orders == []                             # nothing was submitted


def test_the_live_guard_uses_the_bots_grown_balance(live):
    """A bot that has doubled its money may commit $2,000, not $1,000."""
    bot = _bot()
    c = _contract()
    assert not bots.record_trade(bot["id"], c, "buy_to_open", 1, 5.00,
                                 symbol="SPY", ts=_ts(), forced=True)
    assert not bots.record_trade(bot["id"], c, "sell_to_close", 1, 105.00,
                                 symbol="SPY", ts=_ts(), forced=True)   # +$10,000
    assert bots.position_headroom(bot["id"])["cap"] == pytest.approx(2_000.0)
    live.fill_price = 0.10
    assert not bot_engine._record_fill(bot, _contract(562.0), "buy_to_open", 500, 0.10,
                                       symbol="SPY", order_type="market", ts=_ts())
    order, = live.orders
    assert 1_000.0 < _spend(order) <= 2_000.0 + 1e-6, order


def test_every_real_fill_ever_booked_stays_inside_the_cap(live):
    """End to end over a run of entries and exits: whatever the chain said,
    the ledger's open exposure never exceeds 10% of the balance."""
    live.fill_price = 0.05
    bot = _bot()
    for i in range(6):
        c = _contract(560.0 + i)
        bot_engine._record_fill(bot, c, "buy_to_open", 400, 0.05, symbol="SPY",
                                order_type="market", ts=_ts())
        hr = bots.position_headroom(bot["id"])
        assert hr["open_cost"] <= hr["cap"] + 1e-6, (i, hr)


# ---------------------------------------------------------------------------
# the price the sizing is done on
# ---------------------------------------------------------------------------
NOW = datetime(2026, 9, 8, 14, 30, tzinfo=ET)
EXPIRY = NOW.replace(hour=16, minute=0, second=0, microsecond=0)
SPOT = 560.0
ATM = round(SPOT / option_chain.STEP) * option_chain.STEP


def _chain(quotes):
    with patch.object(option_quotes, "get_chain", return_value=quotes), \
         patch.object(option_quotes, "_last_source", {"SPY": "alpaca"}):
        return option_chain.build_chain("SPY", SPOT,
                                        option_model.minutes_to_close(NOW),
                                        0.20, True, expiry=EXPIRY, now=NOW)


def test_a_strike_the_market_and_the_model_disagree_about_is_sized_on_the_dearer():
    """When a real quote is an order of magnitude off the model, the model
    does not win by default — sizing off it there is what produced the 18x
    overspend. The strike stays tradeable (refusing it would idle half the
    fleet on a day the model is running cold), but the ask that sizing reads
    is the dearer of the two, so the bot buys the quantity the expensive
    reading affords."""
    occ = option_model.occ_symbol("SPY", EXPIRY, True, ATM)
    modeled = next(r for r in _chain({}) if r["strike"] == ATM)
    dear = modeled["price"] * 31
    row = next(r for r in _chain({occ: {"bid": modeled["price"] * 30, "ask": dear,
                                        "volume": 500, "t": NOW.isoformat()}})
               if r["strike"] == ATM)
    assert row["quote_conflict"] is True
    assert row["ask"] == pytest.approx(dear)          # not the modeled ask
    assert row["ask"] > modeled["ask"] * 10

    # sizing now reads the dear price, so the same budget buys far less
    g = {"strike_style": "cheapest", "min_option_volume": 0, "max_iv": 5.0}
    assert option_chain.pick_strike([row], g, 5_000.0, 1) is row
    assert bots.max_contracts_for(5_000.0, row["ask"]) \
        < bots.max_contracts_for(5_000.0, modeled["ask"]) / 10
    # and a budget that only the FANTASY price could afford now buys nothing,
    # instead of buying a position 30x the size the bot thought it was taking
    assert option_chain.pick_strike([row], g, 1_000.0, 1) is None
    assert option_chain.pick_strike([modeled], g, 1_000.0, 1) is modeled

    # a strike with no quote at all is still priced by the model, untouched
    assert modeled["quote_conflict"] is False


def test_a_quote_the_model_thinks_is_cheap_never_lowers_the_sizing_price():
    """The raise is one-directional: a conflicting quote can only make the
    bot buy LESS. A quote far BELOW the model leaves the modeled ask alone,
    so a corrupt cheap print cannot talk a bot into a bigger position."""
    occ = option_model.occ_symbol("SPY", EXPIRY, True, ATM)
    modeled = next(r for r in _chain({}) if r["strike"] == ATM)
    row = next(r for r in _chain({occ: {"bid": modeled["price"] / 60,
                                        "ask": modeled["price"] / 50,
                                        "volume": 500, "t": NOW.isoformat()}})
               if r["strike"] == ATM)
    assert row["quote_conflict"] is True
    assert row["ask"] == pytest.approx(modeled["ask"])
