"""Manager pages: owners only, creation through the form, dry-run button, delete."""

from test_accounts_web import admin_client, isolated_state, user_client  # noqa: F401  (fixture re-export)
from webapp import accounts, auth, bot_settings, bots, manager
import webapp.main as main


def _setup(monkeypatch):
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"equity": "1000"})
    bob_c = user_client("bob", "bob-password-1")
    bob = auth.get_user_by_username("bob")
    carol_c = user_client("carol", "carol-password-1")
    bot_settings.save_user(bob["id"], custom_pool_total=5000.0)
    acct = accounts.add(bob["id"], "Paper", "paper", "PKBOB", "s")
    return bob_c, carol_c, bob, acct


def test_create_view_run_toggle_delete_and_privacy(monkeypatch):
    bob_c, carol_c, bob, acct = _setup(monkeypatch)
    assert bob_c.get("/bots/managers").status_code == 200
    r = bob_c.post("/bots/managers/create", data={
        "name": "Zebra", "account_id": str(acct["id"]), "slots": "4", "total_budget": "1000",
        "position_pct": "25", "mode": "dry_run"})
    assert r.status_code == 303
    [m] = manager.list_for(bob["id"])
    assert r.headers["location"].startswith("/bots/managers/{}".format(m["id"]))
    assert manager.slot_size(m) == 250.0 and m["position_pct"] == 25.0

    page = bob_c.get("/bots/managers/{}".format(m["id"]))
    assert page.status_code == 200 and "Zebra" in page.text and "$250" in page.text
    assert "Zebra" in bob_c.get("/bots/managers").text
    assert "Zebra" not in carol_c.get("/bots/managers").text

    assert carol_c.get("/bots/managers/{}".format(m["id"])).status_code == 403
    for verb in ("run", "toggle", "delete", "edit"):
        assert carol_c.post("/bots/managers/{}/{}".format(m["id"], verb), data={}).status_code == 403

    r = bob_c.post("/bots/managers/{}/run".format(m["id"]))
    assert r.status_code == 303 and "err=" not in r.headers["location"]
    r = bob_c.post("/bots/managers/{}/toggle".format(m["id"]))
    assert manager.get(m["id"])["enabled"] == 0
    r = bob_c.post("/bots/managers/{}/edit".format(m["id"]), data={
        "account_id": str(acct["id"]), "slots": "5", "total_budget": "1500", "position_pct": "20",
        "mode": "propose", "name": "Zebra 5"})
    got = manager.get(m["id"])
    assert (got["slots"], got["total_budget"], got["position_pct"], got["mode"]) == (5, 1500.0, 20.0, "propose")
    assert bob_c.post("/bots/managers/{}/delete".format(m["id"])).status_code == 303
    assert manager.get(m["id"]) is None


def test_bad_input_is_explained_not_saved(monkeypatch):
    bob_c, _, bob, acct = _setup(monkeypatch)
    r = bob_c.post("/bots/managers/create", data={
        "name": "x", "account_id": str(acct["id"]), "slots": "4", "total_budget": "10",
        "position_pct": "25", "mode": "dry_run"})
    assert "err=" in r.headers["location"] and manager.list_for(bob["id"]) == []
    r = bob_c.post("/bots/managers/create", data={
        "name": "x", "account_id": str(acct["id"]), "slots": "4", "total_budget": "99999",
        "position_pct": "25", "mode": "dry_run"})
    assert "capital" in r.headers["location"] and manager.list_for(bob["id"]) == []


def test_preview_endpoint(monkeypatch):
    bob_c, *_ = _setup(monkeypatch)
    r = bob_c.get("/bots/managers/preview?slots=4&total_budget=1000&position_pct=25")
    assert r.status_code == 200 and r.json()["size"] == 250.0 and r.json()["per_trade_cap"] == 62.5
    assert bob_c.get("/bots/managers/preview?slots=0&total_budget=1000&position_pct=25").status_code == 400


def test_proposals_are_owner_only(monkeypatch):
    bob_c, carol_c, bob, acct = _setup(monkeypatch)
    bob_c.post("/bots/managers/create", data={
        "name": "P", "account_id": str(acct["id"]), "slots": "1", "total_budget": "300",
        "position_pct": "25", "mode": "propose"})
    [m] = manager.list_for(bob["id"])
    pid = manager._propose(m["id"], {"type": "fill", "slot": 1, "old_bot_id": None, "source_id": 99999,
                                     "source_name": "ghost", "reason": "empty slot"})
    assert carol_c.post("/bots/managers/proposals/{}/reject".format(pid)).status_code == 403
    assert bob_c.get("/bots/managers/{}".format(m["id"])).status_code == 200      # renders the pending row
    r = bob_c.post("/bots/managers/proposals/{}/reject".format(pid))
    assert r.status_code == 303 and manager.proposals(m["id"]) == []


def test_accounts_page_add_edit_delete_and_privacy(monkeypatch):
    bob_c, carol_c, bob, acct = _setup(monkeypatch)
    page = bob_c.get("/bots/accounts")
    assert page.status_code == 200 and "Paper" in page.text

    r = bob_c.post("/bots/accounts/add", data={"name": "Second", "kind": "paper", "key_id": "PKTWO", "secret": "sec2"})
    assert r.status_code == 303 and "ok=" in r.headers["location"]
    second = next(a for a in accounts.list_accounts(bob["id"]) if a["name"] == "Second")
    shown = bob_c.get("/bots/accounts").text
    assert "Second" in shown and "Second" not in carol_c.get("/bots/accounts").text
    assert "sec2" not in shown and "PKTWO" in shown                       # the key ID is shown, the secret never is

    assert carol_c.get("/bots/accounts/{}/status.json".format(second["id"])).status_code == 404
    for verb in ("edit", "delete", "test"):
        r = carol_c.post("/bots/accounts/{}/{}".format(second["id"], verb), data={"name": "hijack"})
        assert "err=" in r.headers["location"]
    assert accounts.get(bob["id"], second["id"])["name"] == "Second"

    r = bob_c.post("/bots/accounts/{}/edit".format(second["id"]), data={"name": "Second renamed", "key_id": "", "secret": ""})
    assert "ok=" in r.headers["location"] and accounts.get(bob["id"], second["id"])["name"] == "Second renamed"
    r = bob_c.post("/bots/accounts/{}/edit".format(second["id"]), data={"name": "Paper"})
    assert "err=" in r.headers["location"]                                           # duplicate name explained

    r = bob_c.post("/bots/accounts/{}/test".format(second["id"]))
    assert "ok=" in r.headers["location"]

    bob_c.post("/bots/managers/create", data={"name": "M", "account_id": str(second["id"]), "slots": "1",
                                              "total_budget": "300", "position_pct": "25", "mode": "dry_run"})
    r = bob_c.post("/bots/accounts/{}/delete".format(second["id"]))
    assert "manager" in r.headers["location"] and accounts.get(bob["id"], second["id"])
    [m] = manager.list_for(bob["id"])
    bob_c.post("/bots/managers/{}/delete".format(m["id"]))
    r = bob_c.post("/bots/accounts/{}/delete".format(second["id"]))
    assert "ok=" in r.headers["location"] and accounts.get(bob["id"], second["id"]) is None


def test_fleet_account_is_shown_and_editable_on_the_accounts_page(monkeypatch):
    bob_c, carol_c, bob, acct = _setup(monkeypatch)
    assert "not connected" in bob_c.get("/bots/accounts").text            # no fleet keys saved yet
    bot_settings.save_user(bob["id"], fleet_alpaca_key_id="PKFLEETXYZ123", fleet_alpaca_secret_key="fleetsecret")
    page = bob_c.get("/bots/accounts").text
    assert "Fleet account" in page and ">fleet<" in page and "fleetsecret" not in page
    assert "/bots/accounts/fleet/edit" in page
    assert "not connected" in carol_c.get("/bots/accounts").text
    monkeypatch.setattr(main, "_account_summary", lambda keys: {"status": "ACTIVE", "equity": "50000",
                                                                   "options_buying_power": "40000"})
    r = bob_c.get("/bots/accounts/fleet/status.json")
    assert r.status_code == 200 and r.json()["equity"] == "50000"
    assert carol_c.get("/bots/accounts/fleet/status.json").json() == {"error": "no fleet keys saved"}
    assert bob_c.get("/bots/accounts/{}/status.json".format(acct["id"])).status_code == 200   # real ids still route

    # editing: new keys go through the same checks as any account; a saved account's key is refused
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"account_number": "A1", "equity": "1"})
    r = bob_c.post("/bots/accounts/fleet/edit", data={"key_id": "PKREPLACED", "secret": "newsecret"})
    assert bot_settings.fleet_keys(bob["id"]) == ("PKREPLACED", "newsecret") and "ok=" in r.headers["location"]
    r = bob_c.post("/bots/accounts/fleet/edit", data={"key_id": acct["key_id"], "secret": "x"})
    assert "err=" in r.headers["location"] and bot_settings.fleet_keys(bob["id"]) == ("PKREPLACED", "newsecret")
    assert carol_c.post("/bots/accounts/fleet/edit", data={"key_id": "PKC", "secret": "x"}).status_code in (303, 200)
    assert bot_settings.fleet_keys(bob["id"]) == ("PKREPLACED", "newsecret")


def test_creating_a_manager_in_auto_fills_ready_slots_immediately(monkeypatch):
    from types import SimpleNamespace
    bob_c, _, bob, acct = _setup(monkeypatch)
    src = bots.create_bot("src", genome=__import__("webapp.genome", fromlist=["x"]).random_genome(), user_id=bob["id"])
    row = dict(id=src, name="src", strategy="weighted", symbol="SPY", style="0dte", closed_count=60, active_days=20,
               regime_count=3, pl_dollar=500.0, pl_pct=5.0, robust_fitness=1.5, profit_factor=1.6,
               max_drawdown_pct=5.0, genome_fp="fp1")
    real = bots.leaderboard
    monkeypatch.setattr(bots, "leaderboard", lambda scope="active", pool="fleet", **k:
                        [row] if pool == "fleet" else real(scope, pool, **k))
    monkeypatch.setattr(bots, "entry_cost_history", lambda bid: {"n": 3, "n_own": 3, "max": 50.0, "median": 50.0})
    monkeypatch.setattr(manager.notify, "send", lambda *a, **k: True)
    r = bob_c.post("/bots/managers/create", data={"name": "Now", "account_id": str(acct["id"]), "slots": "2",
                                                  "total_budget": "600", "position_pct": "30", "mode": "auto"})
    [m] = manager.list_for(bob["id"])
    assert "filled 1 slot" in r.headers["location"].replace("%20", " ") and len(manager.slot_bots(m["id"])) == 1
    assert "still empty" in r.headers["location"].replace("%20", " ")
