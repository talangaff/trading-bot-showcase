"""Every bot trades for REAL on an Alpaca paper account, and always on its
OWNER's keys: a user's fleet and baseline bots on that user's fleet keys,
their custom bots on their separate custom keys, never the other way round
and never on another user's account. Each bot books only the fill Alpaca
reports, tracks its own contracts on the account it shares with its
owner's other bots, never sells another bot's, journals every order so a
dead cycle can't lose a fill, and the ledgers are reconciled against what
the account actually holds.

Alpaca is never called here — broker.submit_and_wait / positions / order
lookups are stubbed at the module boundary.
"""

from datetime import datetime, timedelta

import pytest

from webapp import account_book, accounts, bot_engine, bot_settings, bots, broker, genome, option_model

ET = option_model.ET
# the fleet owner used throughout: one account, its fleet keys, and the
# separate keys its custom bots trade on
UID = 1
FLEET = ("FLEETKEY", "FLEETSECRET")
OWN = ("BOBKEY", "BOBSECRET")


def _connect_fleet(uid=UID, keys=FLEET, **extra):
    """Give `uid`'s fleet its own Alpaca paper keys."""
    return bot_settings.save_user(uid, fleet_alpaca_key_id=keys[0],
                                  fleet_alpaca_secret_key=keys[1], **extra)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_settings, "_PATH", tmp_path / "bot_settings.json")
    monkeypatch.setattr(bot_settings, "_USER_PATH", tmp_path / "user_bot_settings.json")
    account_book.forget()
    yield


def _fleet_bot(user_id=UID, **genes) -> dict:
    g = genome.random_genome()
    g.update(genes)
    return bots.get_bot(bots.spawn_bot(g, origin="seed", user_id=user_id))


def _custom_bot(user_id=7, **genes) -> dict:
    src = _fleet_bot(user_id=user_id, **genes)
    return bots.get_bot(bots.clone_bot(src["id"], start_balance=10000.0, user_id=user_id))


def _put_on_account(bot: dict, keys, kind="paper") -> dict:
    """Save `keys` as one of the bot owner's accounts and point the bot at it."""
    data = accounts._read()
    mine = data.setdefault(str(bot["user_id"]), {"accounts": [], "next": 1, "migrated": True})
    acct = {"id": mine["next"], "name": "acct{}".format(mine["next"]), "type": kind,
            "key_id": keys[0], "secret": keys[1]}
    mine["accounts"].append(acct)
    mine["next"] += 1
    accounts._write(data)
    bots.set_account(bot["id"], acct["id"], pause=False)
    return bots.get_bot(bot["id"])


def _contract(days=0, strike=560.0, is_call=True):
    exp = (datetime.now(ET) + timedelta(days=days)).replace(hour=16, minute=0, second=0, microsecond=0)
    return option_model.occ_symbol("SPY", exp, is_call, strike)


def _ts():
    """An entry timestamp inside today's regular session.

    Two constraints, and the old fixed 11:00 ET only satisfied one.

    It has to be BEFORE the 0DTE close-out cutoff, because record_trade
    refuses a Buy-to-Open at or after ~15:55 ET on a contract expiring that
    day — the reason a wall-clock-relative stamp used to fail the whole
    file every evening.

    It also has to be before the real clock: flatten_bot stamps its sell
    with datetime.now(), so an entry fixed at 11:00 sorts AFTER its own
    exit whenever the suite runs in the morning, and trade_log[0] is then
    the buy. That failed silently every day before 11:00 ET.

    Two minutes ago satisfies both, clamped under the cutoff for the
    evening case."""
    now = datetime.now(ET)
    stamp = min(now - timedelta(minutes=2),
                now.replace(hour=15, minute=54, second=0, microsecond=0))
    return stamp.isoformat()


class FakeAlpaca:
    """Stands in for broker.submit_and_wait: records every order, fills at
    a price of its own choosing (never the one the bot asked for), and can
    be told to fill partially or not at all."""

    def __init__(self, price=1.23, fill_frac=1.0, fail=None):
        self.price, self.fill_frac, self.fail = price, fill_frac, fail
        self.orders: list[dict] = []
        self.n = 0

    def __call__(self, symbol, qty, side, order_type="market", limit_price=None,
                 stop_price=None, trail_percent=None, position_intent=None,
                 timeout=8.0, poll=1.0, keys=None, client_order_id=None):
        self.n += 1
        self.orders.append({"symbol": symbol, "qty": qty, "side": side, "type": order_type,
                            "limit": limit_price, "intent": position_intent, "keys": keys,
                            "cid": client_order_id})
        if self.fail:
            raise broker.BrokerError(self.fail)
        filled = int(round(qty * self.fill_frac))
        if filled <= 0:
            return 0, None, "o{}".format(self.n), "order not filled (canceled) — cancelled"
        return filled, self.price, "o{}".format(self.n), None


# ---------------------------------------------------------------------------
# who trades where
# ---------------------------------------------------------------------------
def test_execution_routing_owner_fleet_keys_for_fleet_owner_custom_keys_for_custom():
    fleet = _fleet_bot(user_id=7)
    custom = _custom_bot(user_id=7)
    base = bots.get_bot(bots.create_bot("b", genome=genome.baseline_genome("random", "SPY"),
                                        pool="baseline", user_id=7))
    # nothing saved yet: real orders expected, no keys = an error, never a model fill
    assert broker.execution(fleet) == ("broker", None)
    assert broker.execution(base) == ("broker", None)
    assert broker.execution(custom) == ("broker", None)

    _connect_fleet(7)
    assert broker.execution(fleet) == ("broker", FLEET)
    assert broker.execution(base) == ("broker", FLEET)
    assert broker.execution(custom) == ("broker", None)       # no fallback to the fleet, ever

    custom = _put_on_account(custom, OWN)
    assert broker.execution(custom) == ("broker", OWN)

    # there is no setting that makes any of them stop trading for real
    for bot in (fleet, base, custom):
        assert broker.execution(bot)[0] == "broker"


def test_a_fleet_bot_never_trades_on_another_users_keys():
    """The financial guarantee behind sharing genomes: two fleets running
    the same population still place every order on their own account, and a
    fleet with no keys of its own does not trade at all — it never falls
    back to somebody else's."""
    _connect_fleet(1, ("AKEY", "ASECRET"))
    _connect_fleet(2, ("BKEY", "BSECRET"))
    a, b = _fleet_bot(user_id=1), _fleet_bot(user_id=2)
    assert broker.execution(a) == ("broker", ("AKEY", "ASECRET"))
    assert broker.execution(b) == ("broker", ("BKEY", "BSECRET"))

    stranger = _fleet_bot(user_id=3)              # no keys saved for user 3
    assert broker.execution(stranger) == ("broker", None)
    assert broker.keys_for_bot(stranger) is None


# ---------------------------------------------------------------------------
# real fills
# ---------------------------------------------------------------------------
def test_fleet_bot_books_the_actual_alpaca_fill_not_the_modeled_price(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=1.31)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    bot = _fleet_bot()
    c = _contract()

    err = bot_engine._record_fill(bot, c, "buy_to_open", 3, 1.10, symbol="SPY",
                                  order_type="stop_limit", limit_price=1.15, stop_price=1.12,
                                  ts=_ts(), note="auto entry", fill_source="alpaca", regime="mid_vol")
    assert err is None
    assert fake.orders[0]["keys"] == FLEET and fake.orders[0]["intent"] == "buy_to_open"
    assert fake.orders[0]["side"] == "buy" and fake.orders[0]["cid"].startswith("tm-b{}-bto-".format(bot["id"]))
    pos = bots._positions(bot["id"])[c]
    assert pos["qty"] == 3 and pos["avg_price"] == 1.31         # Alpaca's price, not 1.10
    t = bots.trade_log(bot["id"])[0]
    assert t["fill_source"] == "broker" and t["forced"] == 1 and "[alpaca #o1]" in t["note"]
    assert t["fees"] == 0.0 and t["regime"] == "mid_vol" and t["order_type"] == "stop_limit"
    assert bots.pending_orders() == []                          # journaled, then cleared


def test_a_fleet_bot_cannot_book_a_fill_without_sending_an_order(monkeypatch):
    """Simulated execution is gone: nothing a user can configure makes a
    fleet bot record a price the broker did not report. The fitness
    evolution breeds on is therefore always built from real fills."""
    _connect_fleet()
    fake = FakeAlpaca()
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    bot = _fleet_bot()
    c = _contract()
    assert bot_engine._record_fill(bot, c, "buy_to_open", 2, 1.10, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert len(fake.orders) == 1                      # an order really went out
    # ...and the ledger has the broker's price, not the one the bot asked for
    booked = bots.trade_log(bot["id"])[0]
    assert booked["price"] == fake.price != 1.10
    assert booked["fill_source"] == "broker"


def test_no_keys_is_an_error_never_a_modeled_fill(monkeypatch):
    fake = FakeAlpaca()
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    bot = _fleet_bot()
    err = bot_engine._record_fill(bot, _contract(), "buy_to_open", 2, 1.10, symbol="SPY",
                                  order_type="market", ts=_ts())
    assert err and "no Alpaca keys" in err and fake.orders == []
    assert bots.trade_log(bot["id"]) == []
    custom = _custom_bot(user_id=7)
    _connect_fleet()
    err = bot_engine._record_fill(custom, _contract(), "buy_to_open", 2, 1.10, symbol="SPY",
                                  order_type="market", ts=_ts())
    assert err and "Custom Bots page" in err and fake.orders == []
    # logged once for the fleet page, not once per bot per tick
    ev = [e for e in bots.recent_events() if e["kind"] == "order_error"]
    assert len(ev) == 2 and {e["detail"]["key"] for e in ev} == {"nokeys:fleet", "nokeys:custom"}
    bot_engine._record_fill(_fleet_bot(), _contract(), "buy_to_open", 2, 1.10, symbol="SPY",
                            order_type="market", ts=_ts())


def test_unfilled_order_books_nothing_and_clears_the_journal(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(fill_frac=0.0)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    bot = _fleet_bot()
    err = bot_engine._record_fill(bot, _contract(), "buy_to_open", 2, 1.10, symbol="SPY",
                                  order_type="limit", limit_price=1.10, ts=_ts())
    assert err and "not filled" in err
    assert bots.trade_log(bot["id"]) == [] and bots.pending_orders() == []

    # a rejected exit retries once as a market order before giving up
    c = _contract()
    bots.record_trade(bot["id"], c, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {c: 2.0})
    fake.orders.clear()
    err = bot_engine._record_fill(bot, c, "sell_to_close", 2, 1.10, symbol="SPY",
                                  order_type="limit", limit_price=1.10, ts=_ts(), market_fallback=True)
    assert err and [o["type"] for o in fake.orders] == ["limit", "market"]
    assert bots.pending_orders() == []


def test_custom_bot_trades_only_on_its_owners_account(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=0.77)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    custom = _put_on_account(_custom_bot(user_id=7), OWN)
    assert bot_engine._record_fill(custom, _contract(), "buy_to_open", 2, 1.10, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert fake.orders[0]["keys"] == OWN
    assert bots._positions(custom["id"])[_contract()]["avg_price"] == 0.77


# ---------------------------------------------------------------------------
# shared account: each bot only ever sells its own contracts
# ---------------------------------------------------------------------------
def test_sell_is_capped_at_the_bots_own_claim_and_the_account_holding(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=1.50)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    a, b = _fleet_bot(), _fleet_bot()
    c = _contract()
    bots.record_trade(a["id"], c, "buy_to_open", 3, 1.00, ts=_ts(), fees=0)
    bots.record_trade(b["id"], c, "buy_to_open", 5, 1.00, ts=_ts(), fees=0)

    # the account holds all 8; bot a may never sell more than its own 3
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {c: 8.0})
    err = bot_engine._record_fill(a, c, "sell_to_close", 5, 1.40, symbol="SPY",
                                  order_type="market", ts=_ts())
    assert err and "only 3 open" in err and fake.orders == []

    # the account is short of the ledgers (someone sold 6 by hand): a sell is
    # capped at what is actually there, and the shortfall is noted
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {c: 2.0})
    assert bot_engine._record_fill(a, c, "sell_to_close", 3, 1.40, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert fake.orders[-1]["qty"] == 2
    assert bots._positions(a["id"])[c]["qty"] == 1
    assert "capped to the 2 held" in bots.trade_log(a["id"])[0]["note"]

    # nothing on the account at all: no order goes out
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    n = len(fake.orders)
    err = bot_engine._record_fill(b, c, "sell_to_close", 5, 1.40, symbol="SPY",
                                  order_type="market", ts=_ts())
    assert err and "not held on the account" in err and len(fake.orders) == n
    # unknown account state leaves the request alone (Alpaca is the backstop)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: None)
    assert bot_engine._record_fill(b, c, "sell_to_close", 5, 1.40, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert fake.orders[-1]["qty"] == 5


def test_a_fill_updates_the_cached_account_view_for_the_rest_of_the_cycle(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=1.00)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(broker, "positions", lambda keys=None: {})
    a = _fleet_bot()
    c = _contract()
    assert account_book.snapshot(FLEET) == {}
    assert bot_engine._record_fill(a, c, "buy_to_open", 4, 1.00, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert account_book.snapshot(FLEET) == {c: 4.0}
    assert account_book.sell_cap(FLEET, c, 9) == 4
    assert bot_engine._record_fill(a, c, "sell_to_close", 4, 1.00, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert account_book.snapshot(FLEET) == {}


def test_claims_expand_spreads_into_legs_and_reconcile_flags_mismatches(monkeypatch):
    _connect_fleet()
    a, b = _fleet_bot(), _fleet_bot()
    exp = datetime.now(ET).replace(hour=16, minute=0, second=0, microsecond=0)
    single = option_model.occ_symbol("SPY", exp, True, 560.0)
    spread = option_model.spread_symbol("SPY", exp, True, 560.0, 563.0)
    short_leg = option_model.occ_symbol("SPY", exp, True, 563.0)
    bots.record_trade(a["id"], single, "buy_to_open", 3, 1.00, ts=_ts(), fees=0)
    bots.record_trade(b["id"], spread, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)

    cl = account_book.claims([a, b])
    assert cl == {single: {a["id"]: 3.0, b["id"]: 2.0}, short_leg: {b["id"]: -2.0}}
    assert account_book.claimed_total(cl) == {single: 5.0, short_leg: -2.0}
    assert account_book.long_leg(spread) == single

    monkeypatch.setattr(broker, "positions", lambda keys=None: {single: 5.0, short_leg: -2.0})
    rep = account_book.reconcile(FLEET, [a, b])
    assert rep["ok"] and rep["mismatches"] == [] and rep["unclaimed"] == []
    assert account_book.last_report(FLEET)["ok"]

    stray = option_model.occ_symbol("QQQ", exp, False, 480.0)
    monkeypatch.setattr(broker, "positions", lambda keys=None: {single: 4.0, short_leg: -2.0, stray: 1.0})
    rep = account_book.reconcile(FLEET, [a, b])
    assert not rep["ok"]
    assert rep["mismatches"] == [{"symbol": single, "ledger": 5.0, "account": 4.0,
                                  "bots": {str(a["id"]): 3.0, str(b["id"]): 2.0}}]
    assert rep["unclaimed"] == [{"symbol": stray, "account": 1.0}]
    assert any(e["kind"] == "reconcile" for e in bots.recent_events())
    # a partial shortfall is reported, never "fixed" — nobody's ledger is touched
    assert bots._positions(a["id"])[single]["qty"] == 3 and bots._positions(b["id"])[spread]["qty"] == 2

    def boom(keys=None):
        raise broker.BrokerError("down")
    monkeypatch.setattr(broker, "positions", boom)
    account_book.forget()
    rep = account_book.reconcile(FLEET, [a, b])
    assert not rep["ok"] and rep["error"] and rep["mismatches"] == []


def test_a_position_the_account_no_longer_holds_is_closed_in_the_ledger_only_once_old(monkeypatch):
    _connect_fleet()
    a = _fleet_bot()
    c = _contract()
    # anchored inside the session: a 0DTE Buy-to-Open is refused at/after
    # 15:55 ET, so wall-clock-relative timestamps break after hours
    session = datetime.now(ET).replace(hour=11, minute=30, second=0, microsecond=0)
    old_ts = (session - timedelta(minutes=30)).isoformat()
    new_ts = (session - timedelta(minutes=1)).isoformat()
    bots.record_trade(a["id"], c, "buy_to_open", 2, 1.00, ts=old_ts, fees=0)
    bots.touch_pos_state(a["id"], c, 0.60, old_ts)
    c2 = _contract(strike=565.0)
    bots.record_trade(a["id"], c2, "buy_to_open", 2, 1.00, ts=new_ts, fees=0)
    bots.touch_pos_state(a["id"], c2, 1.00, new_ts)

    monkeypatch.setattr(broker, "positions", lambda keys=None: {})
    assert account_book.resolve_vanished(FLEET, [a], now=session) == 1
    assert c not in bots._positions(a["id"]) and c2 in bots._positions(a["id"])
    t = bots.trade_log(a["id"])[0]
    assert t["contract"] == c and t["price"] == 0.60 and "no longer held" in t["note"]
    assert bots.pos_state(a["id"]).get(c) is None

    # still held (even partially): never resolved
    monkeypatch.setattr(broker, "positions", lambda keys=None: {c2: 1.0})
    account_book.forget()
    bots.touch_pos_state(a["id"], c2, 1.00, old_ts)
    with bots._conn() as db:
        db.execute("UPDATE bot_pos_state SET opened_ts=? WHERE bot_id=?", (old_ts, a["id"]))
    assert account_book.resolve_vanished(FLEET, [a], now=session) == 0
    # unknown account state: never resolved
    monkeypatch.setattr(broker, "positions", lambda keys=None: (_ for _ in ()).throw(broker.BrokerError("x")))
    account_book.forget()
    assert account_book.resolve_vanished(FLEET, [a], now=session) == 0


def test_sync_groups_bots_by_account(monkeypatch):
    _connect_fleet()
    fleet, custom, orphan = _fleet_bot(), _custom_bot(user_id=7), _custom_bot(user_id=8)
    custom = _put_on_account(custom, OWN)
    groups = account_book.bots_by_account()
    assert {b["id"] for b in groups[FLEET[0]]} >= {fleet["id"]}
    assert [b["id"] for b in groups[OWN[0]]] == [custom["id"]]
    assert all(orphan["id"] not in [b["id"] for b in v] for v in groups.values())
    assert account_book.keys_for_account(OWN[0]) == OWN
    seen = []
    monkeypatch.setattr(broker, "positions", lambda keys=None: seen.append(keys) or {})
    out = account_book.sync_all()
    assert out["accounts"] == 2 and set(seen) == {FLEET, OWN}


# ---------------------------------------------------------------------------
# the order journal survives a dead cycle
# ---------------------------------------------------------------------------
def test_pending_order_recovery_books_what_actually_filled(monkeypatch):
    _connect_fleet()
    a = _fleet_bot()
    c = _contract()
    # the previous cycle journaled three orders and died before booking them
    filled = bots.add_pending(a["id"], c, "buy_to_open", 2, "cid-filled", symbol="SPY",
                              note="auto entry", regime="low_vol", order_type="limit", limit_price=1.2)
    bots.add_pending(a["id"], c, "buy_to_open", 2, "cid-never", symbol="SPY")
    bots.add_pending(a["id"], c, "buy_to_open", 2, "cid-open", symbol="SPY")
    bots.add_pending(999, c, "buy_to_open", 2, "cid-ghost", symbol="SPY")   # bot since deleted

    orders = {"cid-filled": {"id": "A1", "status": "filled", "filled_qty": "2", "filled_avg_price": "1.17"},
              "cid-never": None,
              "cid-open": {"id": "A3", "status": "accepted", "filled_qty": "0"}}
    cancelled = []
    monkeypatch.setattr(broker, "get_order_by_client_id", lambda cid, keys=None: orders[cid])
    monkeypatch.setattr(broker, "cancel_order", lambda oid, keys=None: cancelled.append(oid))
    monkeypatch.setattr(broker, "get_order", lambda oid, keys=None:
                        {"id": oid, "status": "canceled", "filled_qty": "1", "filled_avg_price": "1.30"})

    assert bot_engine._recover_pending() == 2
    assert bots.pending_orders() == []
    assert cancelled == ["A3"]
    pos = bots._positions(a["id"])[c]
    assert pos["qty"] == 3 and abs(pos["avg_price"] - (2 * 1.17 + 1 * 1.30) / 3) < 1e-9
    log = bots.trade_log(a["id"])
    assert all(t["fill_source"] == "broker" and "recovered fill" in t["note"] for t in log)
    assert any(t["regime"] == "low_vol" and t["limit_price"] == 1.2 for t in log)
    assert any(e["kind"] == "recovered_fills" for e in bots.recent_events())

    # Alpaca unreachable: the journal is kept for the next cycle
    bots.add_pending(a["id"], c, "sell_to_close", 1, "cid-later", symbol="SPY")
    monkeypatch.setattr(broker, "get_order_by_client_id",
                        lambda cid, keys=None: (_ for _ in ()).throw(broker.BrokerError("down")))
    assert bot_engine._recover_pending() == 0
    assert len(bots.pending_orders()) == 1


# ---------------------------------------------------------------------------
# forced exits go to the account too
# ---------------------------------------------------------------------------
def test_drawdown_trip_and_flatten_sell_for_real_and_book_the_actual_fill(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=0.55)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    a = _fleet_bot()
    c = _contract(days=10)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {c: 2.0})
    bots.record_trade(a["id"], c, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
    assert bots.flatten_bot(a["id"], "retiring") == 1
    assert fake.orders[-1]["intent"] == "sell_to_close" and fake.orders[-1]["qty"] == 2
    t = bots.trade_log(a["id"])[0]
    assert t["price"] == 0.55 and t["fill_source"] == "broker" and "real market close" in t["note"]
    assert not bots._positions(a["id"])

    # the drawdown trip: a losing streak with one position left open — force-closed for real
    b = _fleet_bot()
    open_c = _contract(strike=500.0)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {c: 2.0, open_c: 2.0})
    bots.record_trade(b["id"], open_c, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
    n = 1
    while bots.get_bot(b["id"])["status"] == "active" and n < 40:
        c2 = _contract(strike=500.0 + n)
        bots.record_trade(b["id"], c2, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
        bots.record_trade(b["id"], c2, "sell_to_close", 2, 0.10, ts=_ts(), fees=0)
        n += 1
    assert bots.get_bot(b["id"])["status"] == "failing"
    last = bots.trade_log(b["id"])[0]
    assert last["contract"] == open_c and last["price"] == 0.55
    assert last["fill_source"] == "broker" and "real market close" in last["note"]

    # broker failure falls back to the flagged synthetic close
    fake.fail = "Alpaca rejected the order (HTTP 403)"
    d = _fleet_bot()
    bots.record_trade(d["id"], c, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
    assert bots.flatten_bot(d["id"]) == 1
    t = bots.trade_log(d["id"])[0]
    assert t["fill_source"] == "model" and "broker close FAILED" in t["note"]

    # a working broker: the flatten goes out as a real sell and books its fill
    fake.fail = None
    k = len(fake.orders)
    e = _fleet_bot()
    bots.record_trade(e["id"], c, "buy_to_open", 2, 1.00, ts=_ts(), fees=0)
    assert bots.flatten_bot(e["id"]) == 1 and len(fake.orders) == k + 1
    closed = bots.trade_log(e["id"])[0]
    assert "FAILED" not in closed["note"] and closed["price"] == fake.price


def test_expiry_settlement_closes_for_real_on_a_live_bot(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=2.05)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    exp = datetime.now(ET).replace(hour=16, minute=0, second=0, microsecond=0)
    c = option_model.occ_symbol("SPY", exp, True, 560.0)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {c: 2.0})
    monkeypatch.setattr(bot_engine.market_data, "last_price", lambda s: 570.0)
    a = _fleet_bot()
    bots.record_trade(a["id"], c, "buy_to_open", 2, 1.00, ts=exp.replace(hour=11).isoformat(), fees=0)
    assert bot_engine._settle_expired(exp.replace(hour=15, minute=59)) == 1
    t = bots.trade_log(a["id"])[0]
    assert t["price"] == 2.05 and t["fill_source"] == "broker"    # not the 10.00 intrinsic


# ---------------------------------------------------------------------------
# what goes over the wire
# ---------------------------------------------------------------------------
def test_order_bodies_follow_alpacas_options_rules():
    exp = datetime(2026, 9, 8, 16, 0, tzinfo=ET)
    single = option_model.occ_symbol("SPY", exp, True, 560.0)
    spread = option_model.spread_symbol("SPY", exp, True, 560.0, 563.0)

    b = broker.order_body(single, 2, "buy", "market", position_intent="buy_to_open", client_order_id="c1")
    assert b == {"qty": "2", "type": "market", "time_in_force": "day", "client_order_id": "c1",
                 "symbol": single, "side": "buy", "position_intent": "buy_to_open"}
    # only market and limit exist for options: stop-limit -> limit at its limit, trail -> market
    b = broker.order_body(single, 2, "buy", "stop_limit", limit_price=1.157, stop_price=1.10)
    assert b["type"] == "limit" and b["limit_price"] == "1.16" and "stop_price" not in b
    b = broker.order_body(single, 2, "sell", "trailing_stop", trail_percent=15)
    assert b["type"] == "market" and b["position_intent"] == "sell_to_close" and "trail_percent" not in b
    b = broker.order_body(single, 2, "sell", "stop", stop_price=0.9)
    assert b["type"] == "market"

    # a vertical is one multi-leg order: long the first strike, short the second, net debit limit
    b = broker.order_body(spread, 3, "buy", "limit", limit_price=1.05, position_intent="buy_to_open")
    assert b["order_class"] == "mleg" and b["qty"] == "3" and b["limit_price"] == "1.05" and "symbol" not in b
    assert b["legs"] == [
        {"symbol": single, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
        {"symbol": option_model.occ_symbol("SPY", exp, True, 563.0), "ratio_qty": "1",
         "side": "sell", "position_intent": "sell_to_open"}]
    # closing it reverses both legs; a net credit limit is negative
    b = broker.order_body(spread, 3, "sell", "limit", limit_price=1.40, position_intent="sell_to_close")
    assert b["limit_price"] == "-1.4"
    assert [(l["side"], l["position_intent"]) for l in b["legs"]] == [("sell", "sell_to_close"), ("buy", "buy_to_close")]

    assert broker.fill_of({"filled_qty": "3", "filled_avg_price": "-1.42"}) == (3.0, 1.42)
    assert broker.fill_of({"filled_qty": "0"}) == (0.0, None)


def test_positions_parse_signed_option_holdings_only(monkeypatch):
    class R:
        status_code = 200

        def json(self):
            return [{"symbol": "SPY260908C00560000", "qty": "3", "side": "long", "asset_class": "us_option"},
                    {"symbol": "SPY260908C00563000", "qty": "-3", "side": "short", "asset_class": "us_option"},
                    {"symbol": "QQQ260908P00480000", "qty": "1", "side": "short", "asset_class": "us_option"},
                    {"symbol": "AAPL", "qty": "10", "side": "long", "asset_class": "us_equity"}]
    # every Alpaca call goes out through api_meter's wrapper (so it can be
    # counted), which calls requests.request — patch that, not broker.requests
    monkeypatch.setattr("requests.request", lambda *a, **k: R())
    assert broker.positions(FLEET) == {"SPY260908C00560000": 3.0, "SPY260908C00563000": -3.0,
                                       "QQQ260908P00480000": -1.0}

    class Bad(R):
        status_code = 500
        text = "boom"
    monkeypatch.setattr("requests.request", lambda *a, **k: Bad())
    with pytest.raises(broker.BrokerError):
        broker.positions(FLEET)
    with pytest.raises(broker.BrokerError):
        broker.positions(("", ""))


# ---------------------------------------------------------------------------
# buying power — Alpaca has no funding API, so a short account refuses entries
# ---------------------------------------------------------------------------
def _account(bp=None, equity=100_000.0):
    return {"account_number": "PA1", "status": "ACTIVE", "equity": equity, "cash": equity,
            "buying_power": equity, "options_buying_power": bp,
            "options_trading_level": 3, "options_approved_level": 3}


def test_entry_is_refused_when_the_account_cannot_fund_it(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=1.00)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    monkeypatch.setattr(broker, "account", lambda keys=None: _account(bp=1_000.0))
    bot = _fleet_bot()

    # 10 contracts at $1.00 = $1,000 of premium, over the 5%-buffered $950
    err = bot_engine._record_fill(bot, _contract(), "buy_to_open", 10, 1.00, symbol="SPY",
                                  order_type="market", ts=_ts())
    assert err and "not enough options buying power" in err
    assert "needs $1,000" in err and "$950" in err        # the 5% buffer, stated
    assert fake.orders == []                              # nothing was sent
    assert bots.trade_log(bot["id"]) == []
    ev = [e for e in bots.recent_events() if e["kind"] == "buying_power"]
    assert len(ev) == 1 and ev[0]["detail"]["needed"] == 1000.0

    # what fits still goes through
    assert bot_engine._record_fill(bot, _contract(), "buy_to_open", 5, 1.00, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert len(fake.orders) == 1


def test_buying_power_is_spent_and_returned_across_a_cycle(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=1.00)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(broker, "positions", lambda keys=None: {})
    calls = []
    monkeypatch.setattr(broker, "account",
                        lambda keys=None: calls.append(1) or _account(bp=1_000.0))
    a, b = _fleet_bot(), _fleet_bot()
    c = _contract()
    account_book.snapshot(FLEET)        # a real cycle primes this in sync_all()

    assert account_book.buying_power(FLEET) == 1000.0
    # a spends $500 of the $1,000 — b must see $500 left, not the stale $1,000
    assert bot_engine._record_fill(a, c, "buy_to_open", 5, 1.00, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert account_book.buying_power(FLEET) == 500.0
    assert len(calls) == 1                       # cached, not re-fetched per bot
    err = bot_engine._record_fill(b, c, "buy_to_open", 5, 1.00, symbol="SPY",
                                  order_type="market", ts=_ts())
    assert err and "not enough options buying power" in err

    # closing gives the money back
    assert bot_engine._record_fill(a, c, "sell_to_close", 5, 1.00, symbol="SPY",
                                   order_type="market", ts=_ts()) is None
    assert account_book.buying_power(FLEET) == 1000.0
    assert bot_engine._record_fill(b, c, "buy_to_open", 5, 1.00, symbol="SPY",
                                   order_type="market", ts=_ts()) is None


def test_unknown_buying_power_never_halts_the_fleet(monkeypatch):
    _connect_fleet()
    fake = FakeAlpaca(price=1.00)
    monkeypatch.setattr(broker, "submit_and_wait", fake)
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    monkeypatch.setattr(broker, "account",
                        lambda keys=None: (_ for _ in ()).throw(broker.BrokerError("down")))
    assert account_book.buying_power(FLEET) is None
    assert account_book.can_afford(FLEET, 1e9) == (True, None)
    assert bot_engine._record_fill(_fleet_bot(), _contract(), "buy_to_open", 5, 1.00,
                                   symbol="SPY", order_type="market", ts=_ts()) is None
    assert len(fake.orders) == 1

    # an account with no options_buying_power field falls back to cash
    account_book.forget()
    monkeypatch.setattr(broker, "account", lambda keys=None: _account(bp=None, equity=250.0))
    assert account_book.buying_power(FLEET) == 250.0


def test_capital_summary_separates_bookkeeping_from_cash(monkeypatch):
    rows = [{"status": "active", "balance": 10_000.0, "open_cost": 400.0},
            {"status": "active", "balance": 10_000.0, "open_cost": 0.0},
            {"status": "retired", "balance": 10_000.0, "open_cost": 0.0}]   # not counted
    # $20k of virtual capital, but only 10% of it can ever be at risk at once
    cap = account_book.capital_summary(rows, 10.0, _account(bp=100_000.0))
    assert cap["bots"] == 2 and cap["committed"] == 20_000.0
    assert cap["max_exposure"] == 2_000.0 and cap["open_cost"] == 400.0
    assert not cap["oversubscribed"] and cap["short_by"] is None
    assert abs(cap["utilization"] - 0.4) < 1e-9

    # a small account can't cover a fully-loaded fleet — flagged with the gap
    cap = account_book.capital_summary(rows, 10.0, _account(bp=1_500.0))
    assert cap["oversubscribed"] and cap["short_by"] == 500.0

    # no account reading at all: reported, never guessed at
    cap = account_book.capital_summary(rows, 10.0, None)
    assert not cap["oversubscribed"] and cap["utilization"] is None
    assert account_book.capital_summary([], 10.0, None)["bots"] == 0
