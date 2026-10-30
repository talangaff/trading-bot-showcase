"""Peak Alpaca calls per minute, per account.

The number exists to answer one question: how much bot count is there room
for before the rate limit bites. Two things have to be true for it to answer
that honestly — it must count EVERY call that actually goes out, and it must
count per account and per API, since Alpaca meters trading and market data
separately and limits each key on its own.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from webapp import account_book, api_meter, bot_settings, bots, broker, genome

TRADING = "https://paper-api.alpaca.markets/v2/orders"
DATA = "https://data.alpaca.markets/v2/stocks/SPY/bars"
A, B = ("PKALICE", "S1"), ("PKBOB", "S2")


class _Resp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"id": "o1", "status": "filled", "filled_qty": "2",
                "filled_avg_price": "1.20"}

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _clean():
    api_meter._BOTS.update(ts="", by_account={})
    api_meter.reset()
    yield


def test_it_counts_every_call_that_actually_goes_out():
    """The meter wraps requests itself, so a call site cannot escape it by
    forgetting to increment a counter."""
    bot_settings.save_user(1, fleet_alpaca_key_id=A[0], fleet_alpaca_secret_key=A[1])
    for _ in range(4):
        bots.spawn_bot(genome.random_genome(), origin="seed", user_id=1)

    sent = []
    with patch("requests.request", side_effect=lambda m, u, **k: sent.append(u) or _Resp()):
        for i in range(3):
            broker.submit_and_wait("SPY260101C00560000", 2, "buy", "market",
                                   position_intent="buy_to_open", keys=A,
                                   client_order_id="c{}".format(i))
        broker.account(A)

    t = api_meter.stats(A)["apis"]["trading"]
    assert t["total"] == len(sent) > 0
    # one entry is a POST plus polls, so a fill is several calls, not one
    assert t["total"] > 3
    assert t["peak"] == t["total"] and t["now"] == t["total"]


def test_trading_and_data_are_metered_separately():
    """Alpaca rate-limits the two APIs independently, and only the trading
    one scales with bot count — merging them would misread the headroom."""
    for _ in range(9):
        api_meter.record(A, TRADING)
    for _ in range(2):
        api_meter.record(A, DATA)
    api_meter.record(A, "https://example.com/not-alpaca")   # ignored entirely

    apis = api_meter.stats(A)["apis"]
    assert apis["trading"]["peak"] == 9
    assert apis["data"]["peak"] == 2
    assert api_meter.stats(A)["busiest"]["api"] == "trading"
    assert api_meter.classify("https://example.com/x") is None


def test_each_account_is_metered_on_its_own():
    """The limit is per key. One global figure across several users' fleets
    would say 'near the limit' when no single account is close."""
    for _ in range(30):
        api_meter.record(A, TRADING)
    for _ in range(5):
        api_meter.record(B, TRADING)

    assert api_meter.stats(A)["apis"]["trading"]["peak"] == 30
    assert api_meter.stats(B)["apis"]["trading"]["peak"] == 5
    assert api_meter.stats(None)["account"] is None       # no keys, nothing to report


def test_the_peak_is_a_per_minute_high_water_mark_not_a_total():
    now = datetime.now(timezone.utc)
    for _ in range(4):
        api_meter.record(A, TRADING, now=now - timedelta(minutes=3))
    for _ in range(11):                                   # the busy minute
        api_meter.record(A, TRADING, now=now - timedelta(minutes=2))
    for _ in range(2):
        api_meter.record(A, TRADING, now=now - timedelta(minutes=1))

    t = api_meter.stats(A)["apis"]["trading"]
    assert t["peak"] == 11 and t["total"] == 17
    assert t["peak_at"] == api_meter._minute(now - timedelta(minutes=2))
    assert t["last"] == 2 and t["now"] == 0               # the current minute is quiet
    assert t["minutes"] == 3


def test_it_reports_the_headroom_for_adding_bots():
    """The point of the stat: calls per bot per minute at the busiest
    minute, extrapolated to how many bots the limit would allow."""
    bot_settings.save_user(1, fleet_alpaca_key_id=A[0], fleet_alpaca_secret_key=A[1])
    for _ in range(5):
        bots.spawn_bot(genome.random_genome(), origin="seed", user_id=1)
    assert {ak: len(bl) for ak, (_k, bl) in account_book.accounts().items()} == {A[0]: 5}

    for _ in range(20):
        api_meter.record(A, TRADING)

    st = api_meter.stats(A)
    t = st["apis"]["trading"]
    assert t["peak_bots"] == 5
    assert t["per_bot"] == 4.0                            # 20 calls / 5 bots
    assert t["headroom"] == st["limit"] - 20
    assert api_meter.headroom_bots(A) == st["limit"] // 4
    assert t["bots_at_limit"] == st["limit"] // 4         # what the tile renders
    # and a what-if at a different rate
    assert api_meter.headroom_bots(A, target_per_bot=2.0) == st["limit"] // 2


def test_metering_never_breaks_a_call():
    """A broken meter must cost a trade nothing."""
    with patch("webapp.api_meter._bots_on", side_effect=RuntimeError("boom")):
        api_meter.record(A, TRADING)                      # swallowed
    with patch("webapp.bots._conn", side_effect=RuntimeError("db down")):
        api_meter.record(A, TRADING)
        assert api_meter.stats(A)["apis"]["trading"]["peak"] == 0
