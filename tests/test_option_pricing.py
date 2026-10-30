"""Fills must cost something: a real bid/ask spread plus commission, not a
frictionless Black-Scholes mid price. See webapp/option_model.py and the
`_record_fill` commission wiring in webapp/bot_engine.py.
"""

from webapp import bots, config, genome, option_model


def test_spread_widens_near_expiry():
    mid = 1.50
    bid_far, ask_far = option_model.bid_ask(mid, 60.0)
    bid_near, ask_near = option_model.bid_ask(mid, 2.0)
    assert bid_far < mid < ask_far
    assert (ask_near - bid_near) > (ask_far - bid_far)


def test_spread_widens_for_cheap_options():
    bid_a, ask_a = option_model.bid_ask(1.50, 60.0)
    bid_b, ask_b = option_model.bid_ask(0.05, 60.0)
    assert (ask_b - bid_b) / 0.05 > (ask_a - bid_a) / 1.50


def test_entry_exit_fill_prices_bracket_mid():
    mid = 2.00
    assert option_model.entry_fill_price(mid, 30.0) > mid
    assert option_model.exit_fill_price(mid, 30.0) < mid


def test_commission_reduces_realized_pnl():
    bot_id = bots.create_bot("t", genome=genome.random_genome())
    ts = "2026-08-01T15:00:00+00:00"
    fee = 5 * config.BOT_COMMISSION_PER_CONTRACT
    assert fee > 0
    bots.record_trade(bot_id, "SPYXXX", "buy_to_open", 5, 1.00, ts=ts, fees=fee)
    bots.record_trade(bot_id, "SPYXXX", "sell_to_close", 5, 1.00, ts=ts, fees=fee)
    st = bots.stats(bot_id)
    # bought and sold at the exact same price -> P&L should be purely negative,
    # equal to the two legs' commission
    assert st["pl_dollar"] < 0
    assert abs(st["pl_dollar"] + 2 * fee) < 1e-6
