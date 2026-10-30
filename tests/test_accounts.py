"""Paper/live accounts for custom bots: which host a key talks to, which keys
a bot resolves to, the one-bot-per-account rule and the live-trading gates."""

import pytest

from webapp import accounts, bot_settings, bots, broker, genome

UID = 1


def _bot(account_id=None, uid=UID):
    src = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=uid)
    cid = bots.clone_bot(src, start_balance=1000.0, user_id=uid)
    if account_id is not None:
        bots.set_account(cid, account_id)
    return cid


@pytest.fixture
def two_accounts(monkeypatch):
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"equity": "1000"})
    paper = accounts.add(UID, "Paper", "paper", "PKPAPER", "s1")
    live = accounts.add(UID, "Main live", "live", "AKLIVE", "s2")
    return paper, live


def test_live_keys_use_the_live_host_and_paper_keys_the_paper_host(two_accounts):
    assert accounts.trading_base(("AKLIVE", "s2")) == accounts.LIVE_BASE
    assert accounts.trading_base(("PKPAPER", "s1")) == accounts.PAPER_BASE
    assert accounts.trading_base(("unknown", "x")) == accounts.PAPER_BASE   # default is paper
    assert accounts.trading_base(None) == accounts.PAPER_BASE


def test_unassigned_bot_has_no_keys(two_accounts):
    cid = _bot()
    assert broker.execution(bots.get_bot(cid)) == ("broker", None)


def test_live_account_resolves_to_no_keys_until_live_trading_is_switched_on(two_accounts):
    _, live = two_accounts
    cid = _bot(live["id"])
    assert broker.execution(bots.get_bot(cid))[1] is None
    bot_settings.save(live_enabled=True)
    assert broker.execution(bots.get_bot(cid))[1] == ("AKLIVE", "s2")
    bot_settings.save(live_enabled=False)           # the kill switch
    assert broker.execution(bots.get_bot(cid))[1] is None


def test_starting_a_bot_pauses_the_others_on_that_account(two_accounts):
    paper, _ = two_accounts
    a, b = _bot(paper["id"]), _bot(paper["id"])
    assert bots.start_bot(a)[0]
    ok, _, paused = bots.start_bot(b)
    assert ok and paused == [a]
    assert not bots.get_bot(a)["enabled"] and bots.get_bot(b)["enabled"]


def test_bots_on_different_accounts_do_not_pause_each_other(two_accounts):
    paper, _ = two_accounts
    other = accounts.add(UID, "Paper 2", "paper", "PKTWO", "s3")
    a, b = _bot(paper["id"]), _bot(other["id"])
    bots.start_bot(a)
    assert bots.start_bot(b)[2] == []
    assert bots.get_bot(a)["enabled"]


def test_start_requires_an_account(two_accounts):
    ok, msg, _ = bots.start_bot(_bot())
    assert not ok and "account" in msg


def test_live_gate_needs_switch_name_and_paper_record(two_accounts):
    _, live = two_accounts
    cid = _bot()
    bot = bots.get_bot(cid)
    assert "switched off" in bots.live_gate(bot, live, "Main live")
    bot_settings.save(live_enabled=True)
    assert "exact name" in bots.live_gate(bot, live, "wrong")
    assert "closed trades on paper" in bots.live_gate(bot, live, "Main live")
    # paper accounts are never gated
    assert bots.live_gate(bot, two_accounts[0], None) is None


def test_live_gate_passes_with_a_paper_track_record(two_accounts, monkeypatch):
    _, live = two_accounts
    bot_settings.save(live_enabled=True)
    cid = _bot()
    monkeypatch.setattr(bots, "closed_trade_count", lambda _id: bots.MIN_PAPER_TRADES)
    assert bots.live_gate(bots.get_bot(cid), live, "Main live") is None


def test_start_rechecks_the_live_gate(two_accounts):
    _, live = two_accounts
    cid = _bot(live["id"])
    ok, msg, _ = bots.start_bot(cid)
    assert not ok and "switched off" in msg


def test_cannot_save_the_same_key_as_both_types(two_accounts):
    with pytest.raises(accounts.AccountError):
        accounts.add(UID, "Sneaky", "paper", "AKLIVE", "s2")


def test_keys_are_verified_against_the_type_claimed(monkeypatch):
    calls = []

    def probe(base, k, s):
        calls.append(base)
        return {"equity": "5"} if base == accounts.PAPER_BASE else None
    monkeypatch.setattr(accounts, "_probe", probe)
    with pytest.raises(accounts.AccountError, match="paper account, not live"):
        accounts.verify("live", "PK", "s")
    assert accounts.verify("paper", "PK", "s")["equity"] == "5"


def test_cannot_remove_an_account_with_a_bot_on_it(two_accounts):
    paper, _ = two_accounts
    _bot(paper["id"])
    with pytest.raises(accounts.AccountError):
        accounts.remove(UID, paper["id"])


def test_legacy_custom_keys_migrate_to_a_paper_account():
    bot_settings.save_user(UID, alpaca_key_id="PKOLD", alpaca_secret_key="sold")
    cid = bots.clone_bot(bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID),
                         start_balance=1000.0, user_id=UID)
    accts = accounts.list_accounts(UID)
    assert [(a["name"], a["type"], a["key_id"]) for a in accts] == [("Paper", "paper", "PKOLD")]
    assert bots.get_bot(cid)["account_id"] == accts[0]["id"]
    assert bots.get_bot(cid)["enabled"]            # migration does not pause a running bot
    assert broker.execution(bots.get_bot(cid))[1] == ("PKOLD", "sold")


def test_entry_history_includes_the_fleet_bot_it_was_cloned_from():
    src = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    err = bots.record_trade(src, "SPY260116C00560000", "buy_to_open", 1, 2.50,
                            ts="2026-01-05T10:00:00-05:00", fees=0, forced=True)
    assert err is None, err
    cid = bots.clone_bot(src, start_balance=1000.0, user_id=UID)
    hist = bots.entry_cost_history(cid)
    assert hist["n"] == 1 and hist["n_own"] == 0 and hist["max"] == 250.0
    # a clone of the clone sees the whole chain, and the minimum follows
    cid2 = bots.clone_bot(cid, start_balance=1000.0, user_id=UID)
    assert bots.entry_cost_history(cid2)["max"] == 250.0
    assert bots.min_balance_for(bots.entry_cost_history(cid2), 2, 10.0) == 5000.0


def test_custom_bot_guardrails_are_only_bounded_by_0_100_and_1_up():
    src = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    cid = bots.clone_bot(src, start_balance=1000.0, user_id=UID)
    bots.update_guardrails(cid, {"max_drawdown_pct": "90", "position_pct_limit": "100",
                                 "min_contracts": "3", "max_contracts": "5000"})
    eff = bots.effective_guardrails(bots.get_bot(cid))
    assert (eff["max_drawdown_pct"], eff["position_pct_limit"]) == (90.0, 100.0)
    assert (eff["min_contracts"], eff["max_contracts"]) == (3, 5000)
    # beyond the two ends is still pulled back: percentages stop at 0-100, counts at 1
    bots.update_guardrails(cid, {"position_pct_limit": "250", "max_drawdown_pct": "-5",
                                 "min_contracts": "0"})
    eff = bots.effective_guardrails(bots.get_bot(cid))
    assert eff["position_pct_limit"] == 100.0 and eff["max_drawdown_pct"] == 0.0
    assert eff["min_contracts"] == 1
    # the fleet's own clamp is untouched
    from webapp import guardrails
    assert guardrails.load(UID)["position_pct_limit"] <= 20.0


def test_peak_capital_uses_each_custom_bots_own_cap():
    from webapp import account_book
    rows = [{"status": "active", "balance": 1000.0, "open_cost": 0.0, "position_pct": 100.0},
            {"status": "active", "balance": 1000.0, "open_cost": 0.0}]          # falls back to 10%
    out = account_book.capital_summary(rows, 10.0, None)
    assert out["max_exposure"] == 1100.0


# --- the fast lane --------------------------------------------------------
def _fleet_and_custom(uid=UID):
    fleet = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=uid)
    custom = bots.clone_bot(fleet, start_balance=1000.0, user_id=uid)
    return fleet, custom


def test_scopes_split_fleet_and_custom_bots(monkeypatch):
    from webapp import bot_engine
    monkeypatch.setattr(bot_engine, "active_fleet_users", lambda: [UID])
    fleet, custom = _fleet_and_custom()
    ids = lambda scope: {b["id"] for b in bot_engine._bots_to_step(UID, scope)}   # noqa: E731
    assert fleet in ids("fleet") and custom not in ids("fleet")
    assert ids("custom") == {custom}
    assert {fleet, custom} <= ids("all")


def test_lane_liveness_follows_its_heartbeat():
    import time
    from webapp import bot_engine
    assert not bot_engine.fast_lane_alive()
    bots._meta_set(bot_engine.FAST_BEAT_KEY, repr(time.time()))
    assert bot_engine.fast_lane_alive()
    bots._meta_set(bot_engine.FAST_BEAT_KEY, repr(time.time() - 120))
    assert not bot_engine.fast_lane_alive()


def test_fast_tick_steps_only_custom_bots_under_the_lock(monkeypatch):
    from webapp import bot_engine
    _fleet_and_custom()
    seen = []
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")
    monkeypatch.setattr(bot_engine, "_tick",
                        lambda now, users, scope="all", record=True:
                        seen.append((scope, record)) or {"stepped": 1, "idle": None})
    assert bot_engine.fast_tick().startswith("stepped")
    assert seen == [("custom", False)]


def test_fast_tick_does_nothing_without_custom_bots(monkeypatch):
    from webapp import bot_engine
    bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")
    monkeypatch.setattr(bot_engine, "_tick", lambda *a, **k: pytest.fail("must not fetch data"))
    assert bot_engine.fast_tick() == "no custom bots"


def test_engine_lock_excludes_a_second_holder():
    from webapp import bot_engine
    with bot_engine._engine_lock(1.0) as first:
        assert first
        with bot_engine._engine_lock(0.3) as second:
            assert not second
    with bot_engine._engine_lock(1.0) as again:
        assert again


def test_custom_orders_poll_faster_than_fleet(monkeypatch):
    from webapp import account_book, bot_engine
    seen = []
    monkeypatch.setattr(broker, "submit_and_wait",
                        lambda *a, **k: seen.append(k.get("poll")) or (0, None, "o", "not filled"))
    monkeypatch.setattr(account_book, "snapshot", lambda keys, fresh=False: {})
    monkeypatch.setattr(account_book, "can_afford", lambda keys, cost: (True, 1e6))
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {})
    acct = accounts.add(UID, "P", "paper", "PKFAST", "s")
    fleet, custom = _fleet_and_custom()
    bots.set_account(custom, acct["id"], pause=False)
    bot_settings.save_user(UID, fleet_alpaca_key_id="FK", fleet_alpaca_secret_key="FS")
    c = "SPY260116C00560000"
    for bid in (fleet, custom):
        bot_engine._record_fill(bots.get_bot(bid), c, "buy_to_open", 1, 1.0, symbol="SPY",
                                order_type="limit", limit_price=1.0, ts="2026-01-05T10:00:00-05:00")
    assert seen == [1.0, bot_engine.FAST_POLL_S]


# --- the held-position pass ----------------------------------------------
def _hold_env(monkeypatch):
    from webapp import bot_engine, market_data
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")
    monkeypatch.setattr(bot_engine, "_fresh", lambda bars, now: True)
    monkeypatch.setattr(market_data, "recent_bars", lambda s, minutes=120: [{"c": 1.0}] * 30)
    stepped = []
    monkeypatch.setattr(bot_engine, "_step_bot",
                        lambda b, bars, feats, now, phase, book=None: stepped.append(b["id"]))
    return bot_engine, stepped


def test_hold_tick_steps_only_custom_bots_that_hold_something(monkeypatch):
    bot_engine, stepped = _hold_env(monkeypatch)
    fleet, flat = _fleet_and_custom()
    holder = bots.clone_bot(fleet, start_balance=1000.0, user_id=UID)
    err = bots.record_trade(holder, "SPY260116C00560000", "buy_to_open", 1, 1.0,
                            ts="2026-01-05T10:00:00-05:00", fees=0, forced=True)
    assert err is None, err
    assert bot_engine.hold_tick() == "held 1"
    assert stepped == [holder]                      # not the flat clone, not the fleet bot


def test_hold_tick_does_nothing_when_nothing_is_held(monkeypatch):
    bot_engine, stepped = _hold_env(monkeypatch)
    _fleet_and_custom()
    assert bot_engine.hold_tick() is None and stepped == []


def test_hold_tick_skips_on_stale_bars(monkeypatch):
    bot_engine, stepped = _hold_env(monkeypatch)
    monkeypatch.setattr(bot_engine, "_fresh", lambda bars, now: False)
    fleet, _ = _fleet_and_custom()
    holder = bots.clone_bot(fleet, start_balance=1000.0, user_id=UID)
    bots.record_trade(holder, "SPY260116C00560000", "buy_to_open", 1, 1.0,
                      ts="2026-01-05T10:00:00-05:00", fees=0, forced=True)
    assert bot_engine.hold_tick() == "held: no fresh market data" and stepped == []


def test_rename_and_replace_keys(two_accounts, monkeypatch):
    paper, live = two_accounts
    seen = []
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: seen.append((kind, k, s)) or {"equity": "1"})
    out = accounts.update(UID, paper["id"], name="Renamed")
    assert out["name"] == "Renamed" and out["key_id"] == "PKPAPER" and seen == []     # rename alone calls Alpaca not at all
    out = accounts.update(UID, paper["id"], key_id="PKNEW", secret="newsecret")
    assert (out["key_id"], out["secret"]) == ("PKNEW", "newsecret") and seen == [("paper", "PKNEW", "newsecret")]
    assert accounts.update(UID, paper["id"], secret="onlysecret")["key_id"] == "PKNEW"   # secret alone is fine
    assert accounts.get(UID, paper["id"])["type"] == "paper"                             # the type never changes


def test_update_refuses_bad_changes(two_accounts):
    paper, live = two_accounts
    with pytest.raises(accounts.AccountError, match="already have"):
        accounts.update(UID, paper["id"], name="main LIVE")
    with pytest.raises(accounts.AccountError, match="secret that goes with"):
        accounts.update(UID, paper["id"], key_id="PKELSE")
    with pytest.raises(accounts.AccountError, match="already saved as a live"):
        accounts.update(UID, paper["id"], key_id="AKLIVE", secret="x")
    with pytest.raises(accounts.AccountError, match="short name"):
        accounts.update(UID, paper["id"], name="bad/name!")
    with pytest.raises(accounts.AccountError, match="no such"):
        accounts.update(UID, 999, name="x")
    assert accounts.get(UID, paper["id"])["key_id"] == "PKPAPER"                          # nothing half-saved


def test_keys_cannot_be_swapped_under_a_bot_holding_a_position(two_accounts):
    from datetime import datetime
    from webapp import option_model
    paper, _ = two_accounts
    cid = _bot(paper["id"])
    contract = option_model.occ_symbol("SPY", datetime.now(option_model.ET).replace(hour=16), True, 500.0)
    bots.record_trade(cid, contract, "buy_to_open", 1, 1.0, forced=True, ts=datetime.now(option_model.ET).isoformat())
    with pytest.raises(accounts.AccountError, match="holding positions"):
        accounts.update(UID, paper["id"], key_id="PKNEW", secret="s")
    assert accounts.update(UID, paper["id"], name="Still renamable")["name"] == "Still renamable"


def test_cannot_remove_an_account_a_manager_runs_on(two_accounts):
    paper, live = two_accounts
    with bots._conn() as c:
        c.execute("INSERT INTO managers(user_id, name, account_id, slots, total_budget, position_pct, "
                  "mode, created_at) VALUES (?,?,?,?,?,?,?,?)",
                  (UID, "Steady", paper["id"], 4, 1000.0, 25.0, "dry_run", "2026-10-23T00:00:00+00:00"))
    with pytest.raises(accounts.AccountError, match="manager"):
        accounts.remove(UID, paper["id"])
    accounts.remove(UID, live["id"])                                  # an unused one still goes
    assert [a["id"] for a in accounts.list_accounts(UID)] == [paper["id"]]


# --- the fleet account: editable, paper-only, and the fleet's alone ---------------
@pytest.fixture
def fleet_account(monkeypatch):
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"equity": "1000", "account_number": "ACC1"})
    monkeypatch.setattr(accounts, "_probe", lambda base, k, s: {"account_number": "ACC1"})
    bot_settings.save_user(UID, fleet_alpaca_key_id="PKFLEET", fleet_alpaca_secret_key="fs")


def test_the_fleet_account_is_shown_as_type_fleet_but_never_listed_for_picking(fleet_account):
    a = accounts.fleet_account(UID)
    assert a["type"] == "fleet" and a["key_id"] == "PKFLEET"
    assert all(x["type"] != "fleet" for x in accounts.list_accounts(UID))
    assert accounts.get(UID, "fleet") is None


def test_fleet_keys_can_be_replaced_and_must_be_paper(fleet_account, monkeypatch):
    accounts.set_fleet_keys(UID, "PKNEW", "ns")
    assert bot_settings.fleet_keys(UID) == ("PKNEW", "ns")
    accounts.set_fleet_keys(UID, "", "")                       # blank keeps what is saved
    assert bot_settings.fleet_keys(UID) == ("PKNEW", "ns")
    seen = []
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: seen.append(kind) or {"account_number": "ACC1"})
    accounts.set_fleet_keys(UID, "PKNEWER", "ns2")
    assert seen == ["paper"]
    with pytest.raises(accounts.AccountError):
        accounts.set_fleet_keys(UID, "PKX", "")                # new key id without its secret


def test_a_saved_account_cannot_become_the_fleet_account(fleet_account):
    accounts.add(UID, "Mine", "paper", "PKMINE", "s")
    with pytest.raises(accounts.AccountError, match="separate paper account"):
        accounts.set_fleet_keys(UID, "PKMINE", "s")
    assert bot_settings.fleet_keys(UID) == ("PKFLEET", "fs")


def test_the_fleet_account_cannot_be_saved_or_edited_in_as_another_account(fleet_account):
    with pytest.raises(accounts.AccountError, match="fleet"):
        accounts.add(UID, "Sneaky", "paper", "PKFLEET", "fs")
    a = accounts.add(UID, "Mine", "paper", "PKMINE", "s")
    with pytest.raises(accounts.AccountError, match="fleet"):
        accounts.update(UID, a["id"], key_id="PKFLEET", secret="fs")


def test_a_bot_never_resolves_to_the_fleet_keys_even_if_an_account_has_them(fleet_account):
    a = accounts.add(UID, "Mine", "paper", "PKMINE", "s")
    cid = _bot(a["id"])
    assert accounts.keys_for_bot(bots.get_bot(cid)) == ("PKMINE", "s")
    bot_settings.save_user(UID, fleet_alpaca_key_id="PKMINE", fleet_alpaca_secret_key="s")   # legacy overlap
    assert accounts.keys_for_bot(bots.get_bot(cid)) is None


def test_swapping_to_a_different_alpaca_account_is_refused_while_the_fleet_holds_a_position(fleet_account, monkeypatch):
    src = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID)
    monkeypatch.setattr(bots, "has_open_positions", lambda bid: True)
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"account_number": "OTHER"})
    with pytest.raises(accounts.AccountError, match="different Alpaca account"):
        accounts.set_fleet_keys(UID, "PKELSE", "x")
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"account_number": "ACC1"})
    accounts.set_fleet_keys(UID, "PKREGEN", "x")               # same account, regenerated key: fine
    assert bot_settings.fleet_keys(UID) == ("PKREGEN", "x")
