"""Route-level access: who can see what.

Walks an admin and two ordinary users through sign-in, the forced password
change, each user's OWN fleet (its own Alpaca keys, controls and
guardrails, invisible to everybody else), the genuinely admin-only pages,
private journals, and custom bots that belong to — and trade on the keys
of — exactly one account.
"""

import pytest
from fastapi.testclient import TestClient

from webapp import accounts, auth, bot_settings, bots, config, genome, journal
import webapp.bot_engine as bot_engine
import webapp.main as main

ADMIN_PW = "admin-new-password"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr(journal, "DB_PATH", tmp_path / "journal.db")
    monkeypatch.setattr(bot_settings, "_PATH", tmp_path / "bot_settings.json")
    monkeypatch.setattr(bot_settings, "_USER_PATH", tmp_path / "user_bot_settings.json")
    # the detail page's live genome read-out would hit market data
    monkeypatch.setattr(bot_engine, "live_ctx", lambda *_a, **_k: None)
    # "Run cycle now" starts a background thread; letting it outlive the test
    # would race the monkeypatched DB path (and hit the network)
    monkeypatch.setattr(bot_engine, "run_cycle", lambda *_a, **_k: {})
    # every test's failed sign-ins come from the same test-client IP — don't
    # let the brute-force throttle (or its 0.5 s penalty) stack up across tests
    monkeypatch.setattr(auth, "_fails", {})
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    auth.ensure_initialised()
    journal.init(default_owner=1)
    yield


@pytest.fixture
def client():
    return TestClient(main.app, follow_redirects=False)


def login(c: TestClient, username: str, password: str):
    return c.post("/login", data={"username": username, "password": password})


def admin_client() -> TestClient:
    """A signed-in admin who has replaced the default password (doing so on
    the first call in a test; later calls just sign in with the new one)."""
    c = TestClient(main.app, follow_redirects=False)
    if not auth.get_user_by_username(config.DEFAULT_USERNAME)["must_change"]:
        assert login(c, config.DEFAULT_USERNAME, ADMIN_PW).status_code == 303
        return c
    r = login(c, config.DEFAULT_USERNAME, config.DEFAULT_PASSWORD)
    assert r.status_code == 303 and r.headers["location"] == "/change-password"
    r = c.post("/change-password", data={"new": ADMIN_PW, "confirm": ADMIN_PW})
    assert r.status_code == 303
    return c


def user_client(username: str, password: str) -> TestClient:
    """An admin creates the account; the user signs in and sets a password."""
    a = admin_client()
    r = a.post("/admin/users/new", data={"username": username, "password": "temp-" + password, "role": "user"})
    assert r.status_code == 303
    c = TestClient(main.app, follow_redirects=False)
    r = login(c, username, "temp-" + password)
    assert r.headers["location"] == "/change-password"
    # locked to the change page until it's done
    assert c.get("/journal").headers["location"] == "/change-password"
    r = c.post("/change-password", data={"new": password, "confirm": password})
    assert r.status_code == 303
    return c


# ---------------------------------------------------------------------------
# sign-in
# ---------------------------------------------------------------------------
def test_anonymous_is_sent_to_login(client):
    r = client.get("/journal")
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert client.post("/journal/new", data={"name": "x"}).status_code == 401
    assert client.get("/healthz").status_code == 200


def test_bad_credentials_rejected(client):
    r = login(client, "admin", "nope")
    assert r.status_code == 200 and "Incorrect username or password" in r.text
    r = login(client, "ghost", config.DEFAULT_PASSWORD)
    assert r.status_code == 200 and "Incorrect username or password" in r.text


def test_forced_change_rejects_default_and_short():
    c = TestClient(main.app, follow_redirects=False)
    login(c, config.DEFAULT_USERNAME, config.DEFAULT_PASSWORD)
    r = c.post("/change-password", data={"new": "admin", "confirm": "admin"})
    assert r.status_code == 200 and "other than the default" in r.text
    r = c.post("/change-password", data={"new": "short", "confirm": "short"})
    assert r.status_code == 200 and "at least" in r.text


def test_logout_clears_session():
    a = admin_client()
    assert a.get("/journal").status_code == 200
    a.get("/logout")
    assert a.get("/journal").status_code == 303


# ---------------------------------------------------------------------------
# admin-only pages
# ---------------------------------------------------------------------------
def test_admin_sees_fleet_and_users_pages():
    a = admin_client()
    assert a.get("/bots").status_code == 200
    assert a.get("/guardrails").status_code == 200
    assert a.get("/admin/users").status_code == 200
    assert a.get("/bots/leaderboard.json").status_code == 200
    assert a.get("/pool").status_code == 200
    page = a.get("/journal").text
    assert 'href="/bots"' in page and 'href="/admin/users"' in page


def test_every_user_runs_their_own_fleet_with_its_own_controls(monkeypatch):
    """A fleet is no longer a shared thing an admin steers: it is the
    account's own bots on the account's own Alpaca keys, so the controls
    belong to the user — and the admin's keys and bots stay invisible."""
    _connect_fleet(1, "PKADMINKEY", data_feed="sip")
    admin_bot = _fleet_bot(1)
    u = user_client("bob", "bobs-password")
    bob = auth.get_user_by_username("bob")

    for path in ("/bots", "/bots/leaderboard.json", "/bots/archive", "/bots/history",
                 "/guardrails", "/pool"):
        assert u.get(path).status_code == 200, path
    for path in ("/bots/settings", "/bots/options-test",
                 "/bots/spawn", "/bots/run", "/bots/reset", "/guardrails"):
        assert u.post(path, data={"count": "1"}).status_code == 303, path

    # ...but the genuinely installation-wide things are still admin-only
    assert u.get("/admin/users").status_code == 403
    assert u.post("/admin/users/new").status_code == 403
    assert u.post("/guardrails/house").status_code == 403
    assert u.post("/pool/reset").status_code == 403
    r = u.get("/admin/users", headers={"accept": "text/html"})
    assert r.status_code == 403 and "admins only" in r.text

    # bob's fleet page is his own: his controls, and never the admin's key
    page = u.get("/bots").text
    assert "/bots/settings" in page and "/bots/reset" in page
    assert "PKADMINKEY" not in page

    # bob connects his own keys and gets his own fleet
    r = u.post("/bots/settings", data={"alpaca_key_id": "PKIGNORED", "alpaca_secret_key": "X",
                                       "engine_enabled": "1"})
    assert r.status_code == 303 and bot_settings.fleet_keys(bob["id"]) is None   # keys are not set here any more
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"account_number": "A", "equity": "1"})
    r = u.post("/bots/accounts/fleet/edit", data={"key_id": "PKBOBKEY", "secret": "BOBSECRET"})
    assert r.status_code == 303
    assert bot_settings.fleet_keys(bob["id"]) == ("PKBOBKEY", "BOBSECRET")
    assert bot_settings.fleet_keys(1) == ("PKADMINKEY", "S")     # untouched
    assert "PKBOBKEY" not in u.get("/bots").text and "alpaca_key_id" not in u.get("/bots").text
    assert "PKBOBKEY" in u.get("/bots/accounts").text

    # the admin's fleet bot is as private as their journal
    assert u.get(f"/bots/{admin_bot}").status_code == 403
    assert u.get(f"/bots/{admin_bot}/equity.json").status_code == 403
    for path, data in ((f"/bots/{admin_bot}/status", {"status": "retired"}),
                       (f"/bots/{admin_bot}/archive", {}),
                       (f"/bots/{admin_bot}/edit", {"name": "x"}),
                       (f"/bots/{admin_bot}/delete", {})):
        assert u.post(path, data=data).status_code == 403, path
    assert bots.get_bot(admin_bot)["status"] == "active"
    assert bots.get_bot(admin_bot)["name"] != "x"

    # bob's own fleet bot IS his to steer
    bobs_bot = _fleet_bot(bob["id"])
    assert u.get(f"/bots/{bobs_bot}").status_code == 200
    assert u.post(f"/bots/{bobs_bot}/edit", data={"name": "mine"}).status_code == 303
    assert bots.get_bot(bobs_bot)["name"] == "mine"

    page = u.get("/journal").text
    assert 'href="/bots"' in page and 'href="/admin/users"' not in page


def test_a_users_fleet_reset_leaves_other_fleets_alone():
    a = admin_client()
    u = user_client("bob", "bobs-password")
    bob = auth.get_user_by_username("bob")
    mine, theirs = _fleet_bot(bob["id"]), _fleet_bot(1)

    assert u.post("/bots/reset").status_code == 303
    assert bots.get_bot(mine)["status"] != "active"        # retired under the reset reason
    assert bots.get_bot(theirs)["status"] == "active"
    assert len(bots.active_bots(pool="fleet", user_id=bob["id"])) == bots.fleet_cap(bob["id"])


def test_admin_can_view_another_users_fleet_without_signing_in_as_them():
    _connect_fleet(1, "PKADMINKEY")
    a = admin_client()
    u = user_client("bob", "bobs-password")
    bob = auth.get_user_by_username("bob")
    _connect_fleet(bob["id"], "PKBOBKEY")
    bobs_bot = _fleet_bot(bob["id"])

    own = a.get("/bots/accounts").text
    assert "PKADMINKEY" in own and "PKBOBKEY" not in own          # keys are shown on Accounts only

    theirs = a.get("/bots?user={}".format(bob["id"])).text
    assert "PKBOBKEY" not in theirs and "PKADMINKEY" not in theirs and "bob" in theirs
    assert a.get(f"/bots/{bobs_bot}").status_code == 200

    # a non-admin asking for someone else's fleet just gets their own
    assert "PKADMINKEY" not in u.get("/bots?user=1").text


def test_guardrails_are_per_user_over_the_house_defaults():
    from webapp import guardrails
    a = admin_client()
    u = user_client("bob", "bobs-password")
    bob = auth.get_user_by_username("bob")

    a.post("/guardrails/house", data={"max_drawdown_pct": "20", "position_pct_limit": "8",
                                      "min_contracts": "2", "max_contracts": "100"})
    assert guardrails.load(bob["id"])["max_drawdown_pct"] == 20.0     # inherited

    u.post("/guardrails", data={"max_drawdown_pct": "12", "position_pct_limit": "",
                                "min_contracts": "", "max_contracts": ""})
    assert guardrails.load(bob["id"])["max_drawdown_pct"] == 12.0     # bob's own
    assert guardrails.load(bob["id"])["position_pct_limit"] == 8.0    # still inherited
    assert guardrails.load(1)["max_drawdown_pct"] == 20.0             # admin unaffected

    # the absolute ceiling in config.py is not negotiable at any layer
    u.post("/guardrails", data={"max_drawdown_pct": "9999", "position_pct_limit": "",
                                "min_contracts": "", "max_contracts": ""})
    assert guardrails.load(bob["id"])["max_drawdown_pct"] == config.BOT_ABS_MAX_DRAWDOWN_PCT

    # a blank clears the override and falls back to the house default
    u.post("/guardrails", data={"max_drawdown_pct": "", "position_pct_limit": "",
                                "min_contracts": "", "max_contracts": ""})
    assert guardrails.overrides_of(bob["id"]) == {}
    assert guardrails.load(bob["id"])["max_drawdown_pct"] == 20.0


# ---------------------------------------------------------------------------
# journals are per user (and private from admins too)
# ---------------------------------------------------------------------------
def test_journals_are_private_per_user():
    a = admin_client()
    u = user_client("bob", "bobs-password")
    bob = auth.get_user_by_username("bob")
    admin_j = journal.list_journals(1)
    bob_j = journal.list_journals(bob["id"])
    assert len(admin_j) == 2 and len(bob_j) == 2          # starter journals each
    assert {j["id"] for j in admin_j}.isdisjoint({j["id"] for j in bob_j})

    # bob sees his own, not the admin's
    assert u.get(f"/journal/{bob_j[0]['id']}").status_code == 200
    r = u.get(f"/journal/{admin_j[0]['id']}")
    assert r.status_code == 303 and r.headers["location"] == "/journal"
    assert a.get(f"/journal/{bob_j[0]['id']}").status_code == 303

    # a journal bob creates is his
    r = u.post("/journal/new", data={"name": "Swing", "kind": "paper"})
    jid = int(r.headers["location"].rsplit("/", 1)[1])
    assert journal.get_journal(jid)["user_id"] == bob["id"]
    assert len(journal.list_journals(bob["id"])) == 3

    # trades in someone else's journal are unreachable, even by id
    tid = journal.create_trade(admin_j[0]["id"], {"ticker": "AAPL"})
    assert u.get(f"/journal/{admin_j[0]['id']}/trade/{tid}").status_code == 303
    assert u.post(f"/journal/{jid}/trade/{tid}/delete").status_code == 303
    assert journal.get_trade(tid) is not None
    assert u.post(f"/journal/{admin_j[0]['id']}/delete").status_code == 303
    assert journal.get_journal(admin_j[0]["id"]) is not None


def test_existing_journals_migrate_to_admin(tmp_path):
    with journal._conn() as c:
        c.execute("UPDATE journals SET user_id=NULL")
    journal.init(default_owner=7)
    assert {j["user_id"] for j in journal.list_journals(7)} == {7}


# ---------------------------------------------------------------------------
# custom bots belong to one account
# ---------------------------------------------------------------------------
def _fleet_bot(user_id: int = 1) -> int:
    """One active fleet bot belonging to `user_id`.

    Spawns a fresh one, or hands back one they already have: a fleet that
    has been seeded is sitting at its max_fleet_bots size (the daily pass
    refills every slot), so there is no headroom for another."""
    bid = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=user_id)
    if bid is None:
        bid = bots.active_bots(pool="fleet", user_id=user_id)[0]["id"]
    return bid


def _connect_fleet(uid: int, key: str, secret: str = "S", **extra):
    return bot_settings.save_user(uid, fleet_alpaca_key_id=key,
                                  fleet_alpaca_secret_key=secret, **extra)


def test_custom_bots_are_per_user():
    a = admin_client()
    bob_c = user_client("bob", "bobs-password")
    carol_c = user_client("carol", "carols-password")
    bob = auth.get_user_by_username("bob")
    src = _fleet_bot(bob["id"])

    # bob funds his own pool and clones from his own fleet
    assert bob_c.post("/bots/custom/pool", data={"custom_pool_total": "5000"}).status_code == 303
    r = bob_c.post("/bots/custom/clone", data={"source_id": str(src), "allocation": "2000"})
    assert r.status_code == 303 and r.headers["location"].startswith("/bots/")
    bid = int(r.headers["location"].rsplit("/", 1)[1])
    bot = bots.get_bot(bid)
    assert bot["pool"] == "custom" and bots.owner_of(bot) == bob["id"]
    assert bots.custom_capital_used(bob["id"]) == 2000.0
    assert bots.custom_capital_used(auth.get_user_by_username("carol")["id"]) == 0.0

    # bob's fleet bot is his: carol cannot clone from it at all
    carol_c.post("/bots/custom/pool", data={"custom_pool_total": "5000"})
    assert carol_c.post("/bots/custom/clone",
                        data={"source_id": str(src), "allocation": "100"}).status_code == 403

    # ...but the shared GENE POOL is everyone's — carol can take the genome
    # itself, and gets her own bot on her own keys with a fresh ledger
    fp = bots.get_bot(src)["genome_fp"]
    r = carol_c.post("/bots/custom/clone-genome",
                     data={"fingerprint": fp, "allocation": "100"})
    assert r.status_code == 303 and r.headers["location"].startswith("/bots/")
    cid = int(r.headers["location"].rsplit("/", 1)[1])
    carol = auth.get_user_by_username("carol")
    assert bots.owner_of(bots.get_bot(cid)) == carol["id"]
    assert bots.get_bot(cid)["genome_fp"] == fp          # same genome, pooled evidence
    assert bots.get_bot(cid)["start_balance"] == 100.0   # her own fresh ledger
    carol_c.post(f"/bots/{cid}/delete")

    # her pool still gates the allocation
    r = carol_c.post("/bots/custom/clone-genome",
                     data={"fingerprint": fp, "allocation": "999999"})
    assert "exceeds" in r.headers["location"]

    # owner and admin can open it; carol cannot
    assert bob_c.get(f"/bots/{bid}").status_code == 200
    r = a.get(f"/bots/{bid}")
    assert r.status_code == 200 and "viewing it as an admin" in r.text
    assert carol_c.get(f"/bots/{bid}").status_code == 403
    assert carol_c.get(f"/bots/{bid}/equity.json").status_code == 403
    assert carol_c.post(f"/bots/{bid}/toggle").status_code == 403
    assert carol_c.post(f"/bots/{bid}/genome", data={"strategy": "x"}).status_code == 403
    assert carol_c.post(f"/bots/{bid}/delete").status_code == 403
    assert bots.get_bot(bid) is not None
    # ...nor clone from it
    carol_c.post("/bots/custom/pool", data={"custom_pool_total": "1000"})
    assert carol_c.post("/bots/custom/clone", data={"source_id": str(bid), "allocation": "100"}).status_code == 403

    # each Custom Bots page lists only its owner's bots
    assert bot["name"] in bob_c.get("/bots/custom").text
    assert bot["name"] not in carol_c.get("/bots/custom").text
    assert bot["name"] not in a.get("/bots/custom").text
    assert bots.custom_bots(bob["id"]) and not bots.custom_bots(1)

    # the owner can toggle and delete
    assert bob_c.post(f"/bots/{bid}/toggle").status_code == 303
    assert not bots.get_bot(bid)["enabled"]
    r = bob_c.post(f"/bots/{bid}/delete")
    assert r.headers["location"] == "/bots/custom" and bots.get_bot(bid) is None


def test_existing_custom_bots_migrate_to_admin():
    src = _fleet_bot()
    bid = bots.clone_bot(src, start_balance=500)      # no owner, as before accounts
    assert bots.get_bot(bid)["user_id"] is None
    bots.init(default_owner=1)
    assert bots.owner_of(bots.get_bot(bid)) == 1


def test_custom_bot_keys_are_the_owners(monkeypatch):
    from webapp import accounts, bots, broker, genome
    monkeypatch.setattr(accounts, "verify", lambda kind, k, s: {"equity": "1000"})
    bob_c = user_client("bob", "bobs-password")
    bob = auth.get_user_by_username("bob")
    _connect_fleet(1, "FLEETKEY", "FLEETSECRET")

    # a custom bot never falls back to its owner's FLEET keys — it exists to
    # test one genome on a different account
    assert accounts.list_accounts(bob["id"]) == []
    page = bob_c.get("/bots/custom").text
    assert "No Alpaca account saved" in page and "FLEETKEY" not in page

    assert "name=\"key_id\"" not in bob_c.get("/bots/custom").text           # no key entry outside Accounts
    assert bob_c.post("/bots/custom/accounts/add", data={}).status_code in (404, 405)
    r = bob_c.post("/bots/accounts/add", data={
        "name": "Bob paper", "kind": "paper", "key_id": "BOBKEY123", "secret": "BOBSECRET"})
    assert r.status_code == 303
    acct = accounts.list_accounts(bob["id"])[0]
    assert accounts.keys_of(acct) == ("BOBKEY123", "BOBSECRET")
    page = bob_c.get("/bots/accounts").text
    assert "BOBKEY" in page and "BOBSECRET" not in page          # secret is never shown again

    # a bot only trades once it is put on the account
    src = bots.spawn_bot(genome.random_genome(), origin="seed", user_id=bob["id"])
    cid = bots.clone_bot(src, start_balance=500.0, user_id=bob["id"])
    assert broker.keys_for_bot(bots.get_bot(cid)) is None
    r = bob_c.post("/bots/{}/account".format(cid), data={"account_id": str(acct["id"])})
    assert r.status_code == 303
    assert broker.keys_for_bot(bots.get_bot(cid)) == ("BOBKEY123", "BOBSECRET")
    detail = bob_c.get("/bots/{}".format(cid)).text
    assert "Move bot to this account" in detail and "Bob paper" in detail

    # starting it works; a live account is refused while live trading is off
    assert bob_c.post("/bots/{}/toggle".format(cid)).status_code == 303
    assert bots.get_bot(cid)["enabled"]
    live = accounts.add(bob["id"], "Bob live", "live", "AKBOBLIVE", "x")
    r = bob_c.post("/bots/{}/account".format(cid), data={
        "account_id": str(live["id"]), "confirm_name": "Bob live"})
    assert "switched%20off" in r.headers["location"]
    assert bots.get_bot(cid)["account_id"] == acct["id"]

    # another user cannot reach bob's account, even by id
    a = admin_client()
    assert accounts.get(1, acct["id"]) is None
    assert broker.keys_for_bot({"pool": "custom", "user_id": 1, "account_id": acct["id"]}) is None

    # fleet routing is untouched by any of this
    assert broker.keys_for_bot({"pool": "fleet", "user_id": 1}) == ("FLEETKEY", "FLEETSECRET")
    assert broker.keys_for_bot({"pool": "baseline", "user_id": 1}) == ("FLEETKEY", "FLEETSECRET")
    assert broker.keys_for_bot({"pool": "fleet", "user_id": bob["id"]}) is None
    assert broker.keys_for_bot({"pool": "fleet"}) is None          # no owner, no keys
    assert broker.connected(("k", "s")) and not broker.connected(("", ""))


def test_pool_total_migrates_to_primary_admin():
    bot_settings.save(custom_pool_total="12000")
    bot_settings.migrate_pool_to(1)
    assert bot_settings.load_user(1)["custom_pool_total"] == 12000.0
    assert bot_settings.load()["custom_pool_total"] == 0.0
    bot_settings.save(custom_pool_total="99")           # never overwrites a set pool
    bot_settings.migrate_pool_to(1)
    assert bot_settings.load_user(1)["custom_pool_total"] == 12000.0


# ---------------------------------------------------------------------------
# admin account management
# ---------------------------------------------------------------------------
def test_admin_creates_lists_disables_resets_and_deletes():
    a = admin_client()
    r = a.post("/admin/users/new", data={"username": "bob", "password": "", "role": "user"})
    assert r.status_code == 303
    page = a.get("/admin/users").text
    assert "Temporary password for <strong>bob</strong>" in page
    temp = page.split('user-select:all">')[1].split("<")[0]
    bob = auth.get_user_by_username("bob")
    assert bob["must_change"] and auth.authenticate("bob", temp)
    assert len(journal.list_journals(bob["id"])) == 2
    # shown once
    assert "Temporary password for" not in a.get("/admin/users").text

    # invalid input comes back as an error, not a 500
    r = a.post("/admin/users/new", data={"username": "bob", "password": "", "role": "user"})
    assert "already taken" in a.get("/admin/users").text

    # bob signs in and finishes setup; the admin then disables him
    bob_c = TestClient(main.app, follow_redirects=False)
    login(bob_c, "bob", temp)
    bob_c.post("/change-password", data={"new": "bobs-password", "confirm": "bobs-password"})
    assert bob_c.get("/journal").status_code == 200
    assert a.post(f"/admin/users/{bob['id']}/disable", data={"disabled": "1"}).status_code == 303
    assert bob_c.get("/journal").status_code == 303            # signed out immediately
    assert login(bob_c, "bob", "bobs-password").status_code == 200   # and can't get back in
    a.post(f"/admin/users/{bob['id']}/disable", data={"disabled": "0"})
    assert login(bob_c, "bob", "bobs-password").headers["location"] == "/"

    # a password reset kicks his session and hands out a temporary password
    a.post(f"/admin/users/{bob['id']}/reset-password")
    assert bob_c.get("/journal").status_code == 303
    temp2 = a.get("/admin/users").text.split('user-select:all">')[1].split("<")[0]
    assert login(bob_c, "bob", temp2).headers["location"] == "/change-password"

    # promote / demote
    a.post(f"/admin/users/{bob['id']}/role", data={"role": "admin"})
    assert auth.get_user(bob["id"])["role"] == "admin"
    a.post(f"/admin/users/{bob['id']}/role", data={"role": "user"})
    assert auth.get_user(bob["id"])["role"] == "user"

    # self-protection
    a.post("/admin/users/1/role", data={"role": "user"})
    assert auth.get_user(1)["role"] == "admin"
    a.post("/admin/users/1/disable", data={"disabled": "1"})
    assert not auth.get_user(1)["disabled"]
    a.post("/admin/users/1/delete")
    assert auth.get_user(1) is not None

    # delete takes the user's data with it
    src = _fleet_bot()
    bots.clone_bot(src, start_balance=100, user_id=bob["id"])
    assert a.post(f"/admin/users/{bob['id']}/delete").status_code == 303
    assert auth.get_user(bob["id"]) is None
    assert journal.list_journals(bob["id"]) == []
    assert bots.custom_bots(bob["id"]) == []
    assert bots.get_bot(src) is not None                 # the fleet is untouched


def test_own_password_change_keeps_session_but_drops_others():
    a = admin_client()
    other = TestClient(main.app, follow_redirects=False)
    assert login(other, config.DEFAULT_USERNAME, ADMIN_PW).headers["location"] == "/"
    r = a.post("/change-password", data={"current": ADMIN_PW, "new": "yet-another-pw", "confirm": "yet-another-pw"})
    assert r.status_code == 303
    assert a.get("/journal").status_code == 200
    assert other.get("/journal").status_code == 303
    r = a.post("/change-password", data={"current": "wrong", "new": "yet-another-pw2", "confirm": "yet-another-pw2"})
    assert "Current password is incorrect" in r.text


# ---------------------------------------------------------------------------
# Refresh: what a browser reload alone cannot do
# ---------------------------------------------------------------------------
def _fleet_pl(client, path="/bots") -> str:
    """The Fleet P&L tile's dollar figure as rendered."""
    import re
    # the label carries a "realized"/"unrealized" qualifier since the tile split
    m = re.search(r'Fleet P&amp;L\s*(?:<span[^>]*>[^<]*</span>)?</div>\s*'
                  r'<div class="stat-value[^"]*">\s*[^<]*</div>\s*'
                  r'<div class="muted small">([^<·]*)', client.get(path).text, re.S)
    return m.group(1).strip() if m else ""


def _book_a_win(bot_id: int, pl_per_contract: float = 2.50) -> None:
    from datetime import datetime, timedelta, timezone
    t = datetime.now(timezone.utc) - timedelta(days=2)
    bots.record_trade(bot_id, "SPY260101C00560000", "buy_to_open", 2, 1.00,
                      ts=t.isoformat(), fees=0, regime="low_vol")
    bots.record_trade(bot_id, "SPY260101C00560000", "sell_to_close", 2, 1.00 + pl_per_contract,
                      ts=(t + timedelta(hours=1)).isoformat(), fees=0, regime="low_vol")


def test_refresh_shows_what_a_browser_reload_still_caches():
    """The pages are served through 10-60 second caches, so pressing F5
    inside that window re-renders the same numbers. Refresh is the escape
    hatch, and this pins down that it actually is one."""
    a = admin_client()
    bid = _fleet_bot(1)
    assert a.get("/bots").status_code == 200          # warms the leaderboard cache

    _book_a_win(bid)                                   # an engine cycle books a fill
    assert _fleet_pl(a) == "$0.00"                     # a plain reload: still the cached view

    r = a.post("/bots/refresh")
    assert r.status_code == 303 and r.headers["location"] == "/bots"
    assert _fleet_pl(a) == "$500.00"                   # now it is the current state


def test_refresh_keeps_the_view_you_were_looking_at():
    a = admin_client()
    r = a.post("/bots/refresh?scope=all&sort=fitness_desc&strategy=momentum&user=1")
    assert r.status_code == 303
    loc = r.headers["location"]
    for kept in ("scope=all", "sort=fitness_desc", "strategy=momentum", "user=1"):
        assert kept in loc, loc
    # one-shot banners are not carried back into the refreshed view
    r = a.post("/bots/refresh?scope=all&atest=1ok&reset=7")
    assert "atest" not in r.headers["location"] and "reset" not in r.headers["location"]
    assert "scope=all" in r.headers["location"]


def test_both_pages_offer_refresh_and_say_when_they_were_built():
    u = user_client("bob", "bobs-password")
    for path, action in (("/bots", "/bots/refresh"), ("/bots/custom", "/bots/custom/refresh")):
        page = u.get(path).text
        assert action in page, path
        assert "Refresh" in page and "as of" in page, path

    r = u.post("/bots/custom/refresh")
    assert r.status_code == 303 and r.headers["location"] == "/bots/custom"


def test_bot_fleet_page_shows_the_incubator_panel():
    from webapp import incubator
    a = admin_client()
    uid = auth.get_user_by_username(config.DEFAULT_USERNAME)["id"]
    for _ in range(3):
        incubator.spawn(genome.random_genome(), user_id=uid)
    incubator.maintain([uid], force=True)
    page = a.get("/bots").text
    assert "Incubator" in page and "simulated bots" in page
    assert "Waiting to graduate" in page
    assert "Leading simulated bots" not in page or "sort by" in page
    by_pl = a.get("/bots?sim_sort=pl")
    assert by_pl.status_code == 200 and "Incubator" in by_pl.text
