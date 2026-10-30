"""Managers keep N equal-money custom bots on one account: who gets picked,
when a slot is replaced, and that nothing happens in dry-run.

The fleet's and the custom pool's leaderboards are stubbed (a real 40-trade,
15-day record per candidate would be pages of setup that test fitness, not the
manager); everything else — cloning, guardrails, accounts, retiring, the live
gate — runs against the real code and a throwaway database."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from webapp import account_book, accounts, bot_settings, bots, genome, manager, option_model

UID = 1
ET = option_model.ET


@pytest.fixture
def world(monkeypatch):
    w = SimpleNamespace(fleet={}, over={}, sent=[], cost=50.0)

    def lb(scope="active", pool="fleet", *, fresh=False, user_id=None):
        if pool == "fleet":
            return sorted(w.fleet.values(), key=lambda r: -r["robust_fitness"])
        rows = []
        for b in bots.custom_bots(user_id):
            if b["status"] == "active":
                rows.append({**b, "drawdown_pct": 0.0, "pl_pct": 0.0, "pl_dollar": 0.0,
                             "closed_count": 0, "fitness_recent": None, **w.over.get(b["id"], {})})
        return rows

    monkeypatch.setattr(bots, "leaderboard", lb)
    monkeypatch.setattr(bots, "entry_cost_history",
                        lambda bid: {"n": 5, "n_own": 5, "max": w.cost, "median": w.cost})
    monkeypatch.setattr(manager.notify, "send", lambda text, buttons=None: w.sent.append(text) or True)
    monkeypatch.setattr(account_book, "last_report", lambda keys: None)
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"equity": "1000"})
    monkeypatch.setattr(manager, "market_open", lambda now=None: True)
    bot_settings.save_user(UID, custom_pool_total=5000.0)
    return w


def add_fleet(w, strategy="weighted", symbol="SPY", style="0dte", fit=1.5, **over):
    bid = bots.create_bot("f{}".format(len(w.fleet) + 1), genome=genome.random_genome(), user_id=UID)
    with bots._conn() as c:
        c.execute("UPDATE bots SET genome_fp=? WHERE id=?", ("fp{}".format(bid), bid))
    row = dict(id=bid, name="f{}".format(bid), strategy=strategy, symbol=symbol, style=style,
               closed_count=60, active_days=20, regime_count=3, pl_dollar=500.0, pl_pct=5.0,
               robust_fitness=fit, profit_factor=1.6, max_drawdown_pct=5.0, genome_fp="fp{}".format(bid))
    row.update(over)
    w.fleet[bid] = row
    return bid


_N = [0]


def make_manager(slots=4, budget=1000.0, pct=25.0, mode="auto", kind="paper", rules=None, name="M"):
    _N[0] += 1
    acct = accounts.add(UID, "Acct {} {}".format(kind, _N[0]), kind, "PK{}{}".format(kind.upper(), _N[0]), "s")
    cfg, err = manager.validate(UID, account_id=acct["id"], slots=slots, total_budget=budget,
                                position_pct=pct, mode=mode, confirm_name=acct["name"],
                                check_pool=False)
    assert err is None, err
    return manager.create(UID, name, cfg, rules), acct


def held(mid):
    return manager.slot_bots(mid)


# --- configuration ----------------------------------------------------------
def test_every_slot_gets_the_same_money(world):
    mid, _ = make_manager(slots=4, budget=1000.0)
    assert manager.slot_size(manager.get(mid)) == 250.0


@pytest.mark.parametrize("kw,fragment", [
    ({"slots": 4, "total_budget": 100}, "at least"),
    ({"slots": 0, "total_budget": 500}, "between 1"),
    ({"slots": 4, "total_budget": 1000, "position_pct": 150}, "per-trade"),
    ({"slots": 4, "total_budget": 1000, "mode": "yolo"}, "mode"),
])
def test_bad_config_is_refused(world, kw, fragment):
    acct = accounts.add(UID, "P", "paper", "PKX", "s")
    args = dict(account_id=acct["id"], slots=4, total_budget=1000, position_pct=25, mode="dry_run",
                check_pool=False)
    args.update(kw)
    cfg, err = manager.validate(UID, **args)
    assert cfg is None and fragment in err


def test_budget_cannot_exceed_the_capital_pool(world):
    acct = accounts.add(UID, "P", "paper", "PKX", "s")
    cfg, err = manager.validate(UID, account_id=acct["id"], slots=4, total_budget=9000,
                                position_pct=25, mode="dry_run")
    assert cfg is None and "capital pool" in err


def test_live_account_needs_the_switch_the_typed_name_and_history_for_auto(world):
    live = accounts.add(UID, "Real", "live", "AKLIVE", "s")
    base = dict(account_id=live["id"], slots=4, total_budget=1000, position_pct=25, check_pool=False)
    _, err = manager.validate(UID, mode="dry_run", confirm_name="Real", **base)
    assert "switched off" in err
    bot_settings.save(live_enabled=True)
    _, err = manager.validate(UID, mode="dry_run", confirm_name="wrong", **base)
    assert "exact name" in err
    cfg, err = manager.validate(UID, mode="dry_run", confirm_name="Real", **base)
    assert err is None
    _, err = manager.validate(UID, mode="auto", confirm_name="Real", **base)
    assert "dry-run" in err


def test_live_auto_unlocks_after_enough_days_of_earlier_passes(world):
    bot_settings.save(live_enabled=True)
    mid, _ = make_manager(mode="dry_run", kind="live")
    assert not manager.live_auto_ready(mid)
    for d in range(manager.LIVE_AUTO_MIN_DAYS):
        with bots._conn() as c:
            c.execute("INSERT INTO manager_events(manager_id, ts, kind, detail) VALUES (?,?,?,?)",
                      (mid, "2026-10-{:02d}T14:00:00+00:00".format(10 + d), "run", '{"mode": "dry_run"}'))
    assert manager.live_auto_ready(mid)


# --- picking ----------------------------------------------------------------
def test_fills_empty_slots_with_the_best_and_no_repeated_style(world):
    best = add_fleet(world, "weighted", "SPY", fit=3.0)
    twin = add_fleet(world, "weighted", "SPY", fit=2.9)          # same style as best -> skipped
    other = add_fleet(world, "breakout", "SPY", fit=2.0)
    third = add_fleet(world, "weighted", "QQQ", fit=1.0)
    mid, _ = make_manager(slots=3)
    p = manager.plan(manager.get(mid))
    srcs = [a["source_id"] for a in p["actions"]]
    assert srcs == [best, other, third] and twin not in srcs
    assert [a["slot"] for a in p["actions"]] == [1, 2, 3]


def test_a_bot_whose_entries_cannot_fit_a_slot_is_not_picked(world):
    world.cost = 400.0                       # one contract; 25% of a $250 slot is only $62.50
    add_fleet(world)
    mid, _ = make_manager(slots=4, budget=1000.0, pct=25.0)
    p = manager.plan(manager.get(mid))
    assert p["actions"] == [] and p["n_candidates"] == 0
    assert "slot" in list(p["skipped"].values())[0]
    mid2, _ = make_manager(slots=1, budget=1000.0, pct=100.0, name="M2")  # $1000 slot, $1000 cap -> fits
    assert manager.plan(manager.get(mid2))["actions"]


def test_candidates_must_have_a_real_record(world):
    add_fleet(world, closed_count=10)
    add_fleet(world, pl_dollar=-5.0, strategy="breakout")
    add_fleet(world, regime_count=1, strategy="momentum")
    add_fleet(world, profit_factor=1.0, strategy="mean_reversion")
    mid, _ = make_manager()
    assert manager.plan(manager.get(mid))["actions"] == []


# --- dry run / propose / auto ----------------------------------------------
def test_dry_run_changes_nothing_and_does_not_repeat_itself(world):
    add_fleet(world)
    mid, _ = make_manager(mode="dry_run")
    r1 = manager.run_manager(mid)
    assert r1["mode"] == "dry_run" and held(mid) == []
    assert len(world.sent) == 1 and "would" in world.sent[0]
    manager.run_manager(mid)
    assert len(world.sent) == 1                                    # same thing, not re-announced
    kinds = [e["kind"] for e in manager.events(mid)]
    assert kinds.count("would_fill") == 1 and kinds.count("run") == 2


def test_run_now_is_always_a_dry_run(world):
    add_fleet(world)
    mid, _ = make_manager(mode="auto")
    manager.run_manager(mid, dry=True)
    assert held(mid) == []


def test_auto_fills_slots_with_equal_money_and_the_per_trade_limit(world):
    for i, s in enumerate(("weighted", "breakout", "momentum", "mean_reversion")):
        add_fleet(world, s, "SPY", fit=3.0 - i * 0.3)
    mid, acct = make_manager(slots=4, budget=1000.0, pct=25.0)
    r = manager.run_manager(mid)
    assert len(r["applied"]) == 4
    bs = held(mid)
    assert [b["slot"] for b in bs] == [1, 2, 3, 4]
    for b in bs:
        assert b["start_balance"] == 250.0
        assert b["manager_id"] == mid and b["account_id"] == acct["id"] and b["enabled"]
        g = bots.effective_guardrails(b)
        assert g["position_pct_limit"] == 25.0 and g["min_contracts"] == 1
        hr = bots.position_headroom(b["id"], b)
        assert hr["cap"] == 62.5                                   # never the whole $250
    assert bots.custom_capital_used(UID) == 1000.0


def test_managed_bots_share_an_account_without_pausing_each_other(world):
    for s in ("weighted", "breakout"):
        add_fleet(world, s)
    mid, acct = make_manager(slots=2, budget=500.0)
    manager.run_manager(mid)
    a, b = held(mid)
    for bot in (a, b):
        assert bots.start_bot(bot["id"])[0]
    assert all(x["enabled"] for x in held(mid))
    hand = bots.clone_bot(a["parent_id"], start_balance=100.0, user_id=UID)
    bots.set_account(hand, acct["id"])
    ok, _, paused = bots.start_bot(hand)
    assert ok and paused == [] and all(x["enabled"] for x in held(mid))


def test_propose_waits_for_approval(world):
    add_fleet(world)
    mid, _ = make_manager(mode="propose", slots=1, budget=300.0)
    manager.run_manager(mid)
    assert held(mid) == []
    [p] = manager.proposals(mid)
    manager.run_manager(mid)
    assert len(manager.proposals(mid)) == 1                        # not duplicated
    ok, _ = manager.resolve_proposal(p["id"], True)
    assert ok and len(held(mid)) == 1
    assert manager.proposals(mid) == []


def test_a_rejected_proposal_does_nothing(world):
    add_fleet(world)
    mid, _ = make_manager(mode="propose", slots=1, budget=300.0)
    manager.run_manager(mid)
    [p] = manager.proposals(mid)
    assert manager.resolve_proposal(p["id"], False) == (True, "rejected")
    assert held(mid) == []


def test_auto_on_live_falls_back_to_propose_until_it_has_history(world):
    bot_settings.save(live_enabled=True)
    add_fleet(world)
    mid, _ = make_manager(mode="dry_run", kind="live", slots=1, budget=300.0)
    with bots._conn() as c:
        c.execute("UPDATE managers SET mode='auto' WHERE id=?", (mid,))    # bypass the form's check
    r = manager.run_manager(mid)
    assert r["mode"] == "propose" and held(mid) == []


def test_a_pass_stops_when_ledgers_do_not_match_the_account(world, monkeypatch):
    add_fleet(world)
    monkeypatch.setattr(account_book, "last_report", lambda keys: {"ok": False, "error": None, "mismatches": [1]})
    mid, _ = make_manager(slots=1, budget=300.0)
    manager.run_manager(mid)
    assert held(mid) == []
    assert any(e["kind"] == "skip" for e in manager.events(mid))


# --- replacing --------------------------------------------------------------
def filled(world, n=2, **mk):
    for i in range(n):
        add_fleet(world, ("weighted", "breakout", "momentum", "mean_reversion")[i], fit=2.0)
    mid, acct = make_manager(slots=n, budget=500.0, **mk)
    manager.run_manager(mid)
    assert len(held(mid)) == n
    return mid


def test_a_bot_that_breaches_its_drawdown_is_replaced_and_its_money_goes_to_the_new_bot(world):
    mid = filled(world)
    spare = add_fleet(world, "momentum", "QQQ", fit=1.8)
    bad = held(mid)[0]
    world.over[bad["id"]] = {"drawdown_pct": 22.0}
    manager.run_manager(mid)
    now = held(mid)
    assert len(now) == 2 and bad["id"] not in [b["id"] for b in now]
    assert bots.get_bot(bad["id"])["status"] == "retired"
    assert spare in [b["parent_id"] for b in now]
    assert bots.custom_capital_used(UID) == 500.0                  # the retired bot's $250 was released
    assert any(e["kind"] == "replace" for e in manager.events(mid))


def test_a_breached_bot_stays_when_there_is_nobody_to_replace_it(world):
    mid = filled(world)
    bad = held(mid)[0]
    world.over[bad["id"]] = {"pl_pct": -30.0}
    p = manager.plan(manager.get(mid))
    assert p["actions"] == [] and any("no eligible replacement" in n for n in p["notes"])


def test_a_healthy_bot_is_not_swapped_for_a_marginal_improvement(world):
    mid = filled(world)
    for b in held(mid):                                    # old enough to be swapped
        with bots._conn() as c:
            c.execute("UPDATE bots SET created_at=? WHERE id=?",
                      ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(), b["id"]))
    add_fleet(world, "weighted", "QQQ", fit=2.5)           # +0.5 over a 2.0 source: under the 0.75 margin
    assert manager.plan(manager.get(mid))["actions"] == []
    add_fleet(world, "weighted", "IWM", fit=3.0)           # +1.0: clears it
    [a] = manager.plan(manager.get(mid))["actions"]
    assert a["type"] == "replace" and a["flatten"] is False


def test_a_new_bot_is_not_swapped_before_its_minimum_hold(world):
    mid = filled(world)
    add_fleet(world, "weighted", "IWM", fit=5.0)
    assert manager.plan(manager.get(mid))["actions"] == []


def test_the_weekly_swap_limit_holds(world):
    mid = filled(world, rules={"max_swaps_per_week": 1})
    for b in held(mid):
        with bots._conn() as c:
            c.execute("UPDATE bots SET created_at=? WHERE id=?",
                      ((datetime.now(timezone.utc) - timedelta(days=30)).isoformat(), b["id"]))
    add_fleet(world, "weighted", "IWM", fit=5.0)
    add_fleet(world, "breakout", "IWM", fit=4.9)
    manager._event(mid, "replace", "earlier this week")
    p = manager.plan(manager.get(mid))
    assert p["actions"] == [] and "weekly swap limit reached" in p["notes"]


def test_a_bot_holding_a_position_is_not_swapped_out_from_under_it(world, monkeypatch):
    mid = filled(world)
    add_fleet(world, "weighted", "IWM", fit=5.0)
    bad = held(mid)[0]
    world.over[bad["id"]] = {"drawdown_pct": 30.0}
    contract = option_model.occ_symbol("SPY", datetime.now(ET).replace(hour=16), True, 500.0)
    bots.record_trade(bad["id"], contract, "buy_to_open", 1, 1.0, forced=True,
                      ts=datetime.now(ET).isoformat())
    act = manager.plan(manager.get(mid))["actions"][0]
    act["flatten"] = False
    ok, msg = manager.apply_action(manager.get(mid), act)
    assert not ok and "flat" in msg and bots.get_bot(bad["id"])["status"] == "active"
    act["flatten"] = True
    monkeypatch.setattr(manager, "market_open", lambda now=None: False)   # shut: no real close is attempted
    ok, msg = manager.apply_action(manager.get(mid), act)
    assert not ok and "market is closed" in msg and bots.get_bot(bad["id"])["status"] == "active"


def test_a_retired_genome_is_not_brought_straight_back(world):
    mid = filled(world)
    bad = held(mid)[0]
    world.over[bad["id"]] = {"drawdown_pct": 22.0}
    add_fleet(world, "momentum", "QQQ", fit=1.8)
    manager.run_manager(mid)
    src_fp = "fp{}".format(bad["parent_id"])
    assert src_fp in manager._recently_retired_fps(mid, 14)
    p = manager.plan(manager.get(mid))
    assert bad["parent_id"] not in [a.get("source_id") for a in p["actions"]]


# --- the live gate and cleanup ----------------------------------------------
def test_a_managed_bot_inherits_its_sources_paper_record_for_the_live_gate(world):
    bot_settings.save(live_enabled=True)
    live = accounts.add(UID, "Real", "live", "AKLIVE", "s")
    src = add_fleet(world)
    for i in range(6):
        c = option_model.occ_symbol("SPY", datetime(2026, 9, 1 + i, 16, tzinfo=ET), True, 500.0)
        bots.record_trade(src, c, "buy_to_open", 1, 1.0, forced=True, ts="2026-09-0{}T10:00:00-04:00".format(1 + i))
        bots.record_trade(src, c, "sell_to_close", 1, 1.1, forced=True, ts="2026-09-0{}T11:00:00-04:00".format(1 + i))
    hand = bots.clone_bot(src, start_balance=100.0, user_id=UID)
    assert "closed trades on paper" in bots.live_gate(bots.get_bot(hand), live)
    mid, _ = make_manager(mode="propose", kind="live", slots=1, budget=300.0)
    with bots._conn() as c:
        c.execute("UPDATE managers SET account_id=? WHERE id=?", (live["id"], mid))
    ok, msg = manager.apply_action(manager.get(mid), manager.plan(manager.get(mid))["actions"][0])
    assert ok, msg


def test_deleting_a_manager_keeps_its_bots_and_closes_nothing(world):
    add_fleet(world)
    mid, _ = make_manager(slots=1, budget=300.0)
    manager.run_manager(mid)
    [b] = held(mid)
    manager.delete(mid)
    after = bots.get_bot(b["id"])
    assert after["status"] == "active" and after["manager_id"] is None and manager.get(mid) is None


def test_a_paused_manager_does_nothing(world):
    add_fleet(world)
    mid, _ = make_manager(slots=1, budget=300.0)
    manager.set_enabled(mid, False)
    assert manager.run_manager(mid) == {"skipped": "manager paused"} and held(mid) == []


def test_preview_reports_whether_the_fleet_can_fill_the_slots(world):
    world.cost = 150.0
    add_fleet(world, "weighted")
    add_fleet(world, "breakout")
    out = manager.preview(UID, 2, 500.0, 25.0)
    assert out["size"] == 250.0 and out["per_trade_cap"] == 62.5 and out["enough"] is False
    assert out["min_slot_lowest"] == 600.0                       # $150 contract / 25%
    ok = manager.preview(UID, 2, 2400.0, 25.0)
    assert ok["enough"] is True and len(ok["picks"]) == 2


# --- filling empty slots as soon as a bot is ready --------------------------
def test_fill_empty_fills_at_once_in_auto_and_only_empty_slots(world):
    for s in ("weighted", "breakout"):
        add_fleet(world, s)
    mid, _ = make_manager(slots=3, budget=900.0)
    r = manager.fill_empty(mid, force=True)
    assert len(r[mid]["applied"]) == 2 and len(held(mid)) == 2
    assert manager.empty_slots(manager.get(mid)) == 1
    # nobody left for the third slot -> nothing happens, and the quick pass leaves no noise in History
    runs = [e for e in manager.events(mid) if e["kind"] == "run"]
    manager.fill_empty(mid, force=True)
    assert len(held(mid)) == 2 and [e for e in manager.events(mid) if e["kind"] == "run"] == runs


def test_the_third_slot_fills_the_moment_a_new_bot_qualifies(world):
    for s in ("weighted", "breakout"):
        add_fleet(world, s)
    mid, _ = make_manager(slots=3, budget=900.0)
    manager.fill_empty(mid, force=True)
    assert len(held(mid)) == 2
    add_fleet(world, "momentum")
    manager.fill_empty(mid, force=True)
    assert len(held(mid)) == 3 and manager.empty_slots(manager.get(mid)) == 0


def test_fill_empty_rechecks_only_every_few_minutes_unless_forced(world):
    mid, _ = make_manager(slots=2, budget=500.0)
    manager.fill_empty(mid)                                   # nobody ready; remembers it looked
    add_fleet(world, "weighted")
    assert manager.fill_empty(mid) == {} and held(mid) == []  # too soon to look again
    assert len(manager.fill_empty(mid, force=True)[mid]["applied"]) == 1
    assert len(held(mid)) == 1


def test_a_pass_that_filled_looks_again_straight_away(world):
    for s in ("weighted", "breakout"):
        add_fleet(world, s)
    mid, _ = make_manager(slots=3, budget=900.0)
    manager.fill_empty(mid)
    add_fleet(world, "momentum")
    assert len(manager.fill_empty(mid)[mid]["applied"]) == 1  # no waiting: the last look found something


def test_the_quick_pass_never_replaces_or_closes_anything(world):
    mid = filled(world)
    bad = held(mid)[0]
    world.over[bad["id"]] = {"drawdown_pct": 40.0}
    add_fleet(world, "momentum", "QQQ", fit=2.5)
    assert manager.fill_empty(mid, force=True) == {}          # no slot is empty: nothing to do
    assert bots.get_bot(bad["id"])["status"] == "active" and len(held(mid)) == 2
    with bots._conn() as c:                                   # free a slot by hand
        c.execute("UPDATE bots SET status='retired' WHERE id=?", (held(mid)[1]["id"],))
    manager.fill_empty(mid, force=True)
    assert bots.get_bot(bad["id"])["status"] == "active"      # the breached bot is still left to the full pass


def test_fill_empty_respects_mode_and_pause(world):
    add_fleet(world)
    dry, _ = make_manager(mode="dry_run", slots=1, budget=300.0, name="D")
    manager.fill_empty(dry, force=True)
    assert held(dry) == [] and any(e["kind"] == "would_fill" for e in manager.events(dry))
    prop, _ = make_manager(mode="propose", slots=1, budget=300.0, name="P")
    manager.fill_empty(prop, force=True)
    assert held(prop) == [] and len(manager.proposals(prop)) == 1
    off, _ = make_manager(slots=1, budget=300.0, name="O")
    manager.set_enabled(off, False)
    assert manager.fill_empty(off, force=True) == {} and held(off) == []


def test_slots_never_add_up_to_more_than_the_budget(world):
    mid, _ = make_manager(slots=3, budget=5000.0)
    size = manager.slot_size(manager.get(mid))
    assert size == 1666.66 and size * 3 <= 5000.0
    assert manager.slot_size({"total_budget": 1000.0, "slots": 4}) == 250.0
    assert manager.preview(UID, 3, 5000.0, 25.0)["size"] == 1666.66


# --- a bot holding a position finishes it, then is replaced at once ---------------
def _hold(bot_id):
    contract = option_model.occ_symbol("SPY", datetime.now(ET).replace(hour=16), True, 500.0)
    bots.record_trade(bot_id, contract, "buy_to_open", 1, 1.0, forced=True, ts=datetime.now(ET).isoformat())
    return contract


def test_a_bot_holding_a_position_stops_trading_finishes_it_then_is_replaced_at_once(world):
    mid = filled(world)
    add_fleet(world, "momentum", "QQQ", fit=1.8)
    bad = held(mid)[0]
    world.over[bad["id"]] = {"drawdown_pct": 30.0}
    contract = _hold(bad["id"])
    r = manager.run_manager(mid)
    assert r["applied"] == [] and len(r["waiting"]) == 1
    b = bots.get_bot(bad["id"])
    assert b["status"] == "active" and b["drain"] == 1 and not b["enabled"]      # still there, takes no new trades
    assert manager.fill_empty(mid, force=True) == {}                              # still holding: nothing to do
    bots.record_trade(bad["id"], contract, "sell_to_close", 1, 1.2, forced=True, ts=datetime.now(ET).isoformat())
    r = manager.fill_empty(mid, force=True)                                       # flat -> replaced on the quick pass
    assert len(r[mid]["applied"]) == 1
    assert bots.get_bot(bad["id"])["status"] == "retired"
    assert len(held(mid)) == 2 and bad["id"] not in [x["id"] for x in held(mid)]


def test_a_breach_no_longer_force_closes_by_default(world):
    mid = filled(world)
    add_fleet(world, "momentum", "QQQ", fit=1.8)
    world.over[held(mid)[0]["id"]] = {"drawdown_pct": 30.0}
    [a] = manager.plan(manager.get(mid))["actions"]
    assert a["flatten"] is False


def test_a_phased_out_swap_is_cancelled_if_the_better_bot_is_gone_when_it_goes_flat(world):
    mid = filled(world)
    old = held(mid)[0]
    with bots._conn() as c:
        c.execute("UPDATE bots SET drain=2, enabled=0 WHERE id=?", (old["id"],))
    manager.run_manager(mid)
    b = bots.get_bot(old["id"])
    assert b["status"] == "active" and not b["drain"] and b["enabled"]
