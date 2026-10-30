"""The fleet-size guardrail: max_fleet_bots is a ceiling nothing may pass.

Every fleet bot is born in bots.spawn_bot(), so that is the one door the cap
is enforced at — and these tests come at that door from every direction the
app can: the Spawn button, a fleet reset, evolve()'s replacements and
top-up, and the daily breeding pass. See webapp/guardrails.py for the
layering (house default, per-account override, absolute clamp).
"""

import pytest
from fastapi.testclient import TestClient

from webapp import auth, bot_engine, bot_settings, bots, config, genome, guardrails, journal
import webapp.main as main

UID = 1
OTHER = 2


def _cap(n, user_id=UID):
    guardrails.save_user(user_id, max_fleet_bots=n)


def _active(user_id=UID):
    return len(bots.active_bots(pool="fleet", user_id=user_id))


def _fill_to_cap(n, user_id=UID):
    for _ in range(n):
        bots.spawn_bot(genome.random_genome(), origin="seed", user_id=user_id)


# --- the door itself -------------------------------------------------------
def test_spawn_bot_refuses_past_the_cap_and_says_so():
    _cap(3)
    _fill_to_cap(3)
    assert _active() == 3

    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None
    assert _active() == 3


def test_a_refused_spawn_does_not_claim_a_pool_genome():
    """A genome registered for a bot that was never created would look
    claimed by a fleet that is not running it, and hide from every other
    fleet's search."""
    from webapp import gene_pool

    _cap(1)
    _fill_to_cap(1)
    before = gene_pool.size()
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None
    assert gene_pool.size() == before


def test_retiring_a_bot_frees_exactly_one_slot():
    _cap(2)
    _fill_to_cap(2)
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None

    bots.retire_bot(bots.active_bots(pool="fleet", user_id=UID)[0]["id"], "t")
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is not None
    assert _active() == 2
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None


# --- the cap is one account's, not the installation's ----------------------
def test_one_accounts_cap_does_not_bind_another_fleet():
    _cap(1, UID)
    _fill_to_cap(1, UID)
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None

    # the other account inherits the (much larger) house default
    for _ in range(5):
        assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=OTHER) is not None
    assert _active(OTHER) == 5


def test_custom_and_baseline_bots_are_outside_the_cap():
    """The cap counts the evolving fleet. A user's hand-tuned custom bots
    and the fixed null-hypothesis baselines are neither blocked by it nor
    counted against it."""
    _cap(2)
    _fill_to_cap(2)

    src = bots.active_bots(pool="fleet", user_id=UID)[0]["id"]
    assert bots.clone_bot(src, user_id=UID) is not None
    assert bots.ensure_baseline_bots(UID) > 0
    # ... and the fleet cap still binds the fleet
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None
    assert _active() == 2


# --- the cap is also the setpoint: a fleet is always filled to it ----------
def test_the_cap_is_the_fleet_size_there_is_no_lower_target():
    """There is no separate house target any more — whatever the cap is,
    that is how many bots the fleet is refilled to."""
    _cap(4)
    assert bots.fleet_headroom(UID) == 4
    bot_engine.evolve(UID)
    assert _active() == 4

    _cap(9)                          # raise it and the next pass fills the new room
    assert bots.fleet_headroom(UID) == 5
    bot_engine.evolve(UID)
    assert _active() == 9


def test_reset_reseeds_a_full_fleet_at_the_cap():
    _cap(5)
    _fill_to_cap(5)
    retired = bots.reset_fleet("t", user_id=UID)
    assert retired == 5
    assert _active() == 5           # a full fleet again, not a partial one


# --- evolution ------------------------------------------------------------
def test_evolve_tops_up_to_the_cap_and_stops():
    _cap(4)
    _fill_to_cap(2)
    bot_engine.evolve(UID)
    assert _active() == 4

    bot_engine.evolve(UID)          # already there: nothing more gets through
    assert _active() == 4


def test_evolve_replaces_a_failing_bot_without_breaching_the_cap():
    _cap(3)
    _fill_to_cap(3)
    doomed = bots.active_bots(pool="fleet", user_id=UID)[0]["id"]
    with bots._conn() as c:
        c.execute("UPDATE bots SET status='failing' WHERE id=?", (doomed,))

    ev = bot_engine.evolve(UID)
    assert ev["replaced"] == 1
    assert _active() == 3
    assert doomed not in [b["id"] for b in bots.active_bots(pool="fleet", user_id=UID)]


def test_daily_maintenance_never_breaches_the_cap():
    _cap(6)
    bot_settings.save_user(UID, alpaca_key_id="k", alpaca_secret_key="s")
    bot_engine._daily_for_user(UID)      # seeds an empty fleet
    assert _active() == 6
    bot_engine._daily_for_user(UID)      # a second pass must not grow it
    assert _active() == 6


def test_daily_maintenance_refills_every_slot_it_retired():
    """The whole point of the daily pass: whatever it retires, it replaces.
    It must never leave the fleet short of its cap."""
    _cap(6)
    bot_settings.save_user(UID, alpaca_key_id="k", alpaca_secret_key="s")
    bot_engine._daily_for_user(UID)
    assert _active() == 6

    # retire a third of the fleet behind the pass's back, as a real day's
    # idle-cull and weakest-cull do
    for b in bots.active_bots(pool="fleet", user_id=UID)[:2]:
        bots.retire_bot(b["id"], "retired")
    assert _active() == 4

    ev = bot_engine._daily_for_user(UID)
    assert _active() == 6
    assert ev["seeded"] >= 2


def test_evolve_refills_a_fleet_left_short_by_retirements():
    _cap(5)
    _fill_to_cap(5)
    for b in bots.active_bots(pool="fleet", user_id=UID)[:3]:
        bots.retire_bot(b["id"], "retired")
    assert _active() == 2

    bot_engine.evolve(UID)
    assert _active() == 5


# --- absolute bounds ------------------------------------------------------
def test_cap_is_clamped_to_the_absolute_range_like_every_other_guardrail():
    _cap(10 ** 6)
    assert bots.fleet_cap(UID) == config.BOT_ABS_MAX_FLEET_BOTS
    _cap(0)
    assert bots.fleet_cap(UID) == config.BOT_ABS_MIN_FLEET_BOTS
    _cap("")                             # cleared — back to the house default
    assert bots.fleet_cap(UID) == guardrails.house()["max_fleet_bots"]


def test_house_default_applies_to_an_account_with_no_override():
    guardrails.save(max_fleet_bots=7)
    assert bots.fleet_cap(UID) == 7
    _fill_to_cap(7)
    assert bots.spawn_bot(genome.random_genome(), origin="seed", user_id=UID) is None


# --- the Spawn button -----------------------------------------------------
@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_FILE", tmp_path / "auth.json")
    monkeypatch.setattr(journal, "DB_PATH", tmp_path / "journal.db")
    monkeypatch.setattr(auth, "_fails", {})
    monkeypatch.setattr(main.time, "sleep", lambda *_a, **_k: None)
    auth.ensure_initialised()
    journal.init(default_owner=1)
    c = TestClient(main.app, follow_redirects=False)
    c.post("/login", data={"username": config.DEFAULT_USERNAME,
                           "password": config.DEFAULT_PASSWORD})
    c.post("/change-password", data={"new": "a-new-password", "confirm": "a-new-password"})
    return c


def test_the_spawn_button_cannot_push_a_fleet_past_its_cap(web):
    uid = auth.get_user_by_username(config.DEFAULT_USERNAME)["id"]
    guardrails.save_user(uid, max_fleet_bots=5)

    assert web.post("/bots/spawn", data={"count": "100"}).status_code == 303
    assert len(bots.active_bots(pool="fleet", user_id=uid)) == 5

    # and pressing it again at the cap adds nothing at all
    web.post("/bots/spawn", data={"count": "10"})
    assert len(bots.active_bots(pool="fleet", user_id=uid)) == 5


def test_lowering_the_cap_below_a_running_fleet_stops_growth_without_a_cull(web):
    uid = auth.get_user_by_username(config.DEFAULT_USERNAME)["id"]
    web.post("/bots/spawn", data={"count": "8"})
    assert len(bots.active_bots(pool="fleet", user_id=uid)) == 8

    r = web.post("/guardrails", data={"max_drawdown_pct": "", "position_pct_limit": "",
                                      "min_contracts": "", "max_contracts": "",
                                      "max_fleet_bots": "3"})
    assert r.status_code == 303
    # nobody is retired — a guardrail, not a cull. It just stops growth.
    assert len(bots.active_bots(pool="fleet", user_id=uid)) == 8
    assert bots.fleet_headroom(uid) == 0
    web.post("/bots/spawn", data={"count": "5"})
    assert len(bots.active_bots(pool="fleet", user_id=uid)) == 8
