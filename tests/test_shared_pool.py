"""The three things multi-user fleets are supposed to buy, tested directly.

Every user runs their own fleet on their own Alpaca keys, but there is one
shared population of genomes (webapp/gene_pool.py). What that has to deliver:

  1. **Isolation.** One user's bots, ledgers and keys never reach another's.
  2. **Pooled evidence.** A genome's record is the union of every fleet's
     trades on it, so it is proven — or culled — on a sample that grows with
     the number of fleets instead of only with calendar days.
  3. **Split search.** No two fleets are handed the same candidate, so N
     fleets cover N times as many distinct genomes per market day.

(3) is the load-bearing one. N fleets running the SAME genome on the same
minutes would give the pool N correlated copies of one observation, not N
times the information; the acceleration only exists if they test different
things and pool what comes back. These tests pin that down.
"""

from datetime import datetime, timedelta, timezone

import pytest

from webapp import bot_engine, bot_settings, bots, gene_pool, genome, recorder

ALICE, BOB = 1, 2
N = bots.MIN_ELIGIBLE_TRADES


def _mk_genome(**genes) -> dict:
    g = genome.random_genome()
    g.update({"symbol": "SPY", "strategy": "momentum"})
    g.update(genes)
    return g


def _add_trades(bot_id: int, returns, start_day: int = 0, regimes=("low_vol", "high_vol")):
    """Book `returns` as closed round-trips, spread over distinct sessions so
    the eligibility gates (trades / days / regimes) can actually be met."""
    base = datetime.now(timezone.utc) - timedelta(days=90)
    contract = "SPY260101C00560000"
    for i, r in enumerate(returns):
        ts = (base + timedelta(days=start_day + i, hours=1)).isoformat()
        out = (base + timedelta(days=start_day + i, hours=2)).isoformat()
        bots.record_trade(bot_id, contract, "buy_to_open", 2, 1.00, ts=ts, fees=0,
                          regime=regimes[i % len(regimes)])
        bots.record_trade(bot_id, contract, "sell_to_close", 2, 1.00 + r, ts=out, fees=0,
                          regime=regimes[i % len(regimes)])


# ---------------------------------------------------------------------------
# 1. isolation
# ---------------------------------------------------------------------------
def test_fleets_are_separate_but_the_population_is_one():
    a = bots.spawn_bot(_mk_genome(), origin="seed", user_id=ALICE)
    b = bots.spawn_bot(_mk_genome(), origin="seed", user_id=BOB)

    assert [x["id"] for x in bots.active_bots(pool="fleet", user_id=ALICE)] == [a]
    assert [x["id"] for x in bots.active_bots(pool="fleet", user_id=BOB)] == [b]
    assert len(bots.active_bots(pool="fleet")) == 2          # the population sees both
    assert bots.owner_of(bots.get_bot(a)) == ALICE

    # the engine ticks one fleet at a time, so a bot only ever appears in
    # its owner's cycle — and therefore only ever on its owner's account
    assert [x["id"] for x in bots.live_bots(user_id=ALICE)] == [a]
    assert [x["id"] for x in bots.live_bots(user_id=BOB)] == [b]


def test_only_fleets_with_their_own_keys_trade():
    """Keys are the whole requirement: every fleet places real orders, so
    an account without an Alpaca connection has nothing to tick — it never
    falls back to modeled fills, and never to anyone else's account."""
    assert bot_settings.fleet_user_ids() == []

    bot_settings.save_user(ALICE, fleet_alpaca_key_id="AK", fleet_alpaca_secret_key="AS")
    bot_settings.save_user(BOB, fleet_data_feed="sip")             # settings, but no keys
    assert bot_engine.fleet_users() == [ALICE]
    assert bot_engine.active_fleet_users() == [ALICE]

    bot_settings.save_user(BOB, fleet_alpaca_key_id="BK", fleet_alpaca_secret_key="BS")
    assert bot_engine.fleet_users() == [ALICE, BOB]
    assert bot_engine.active_fleet_users() == [ALICE, BOB]

    # a user's own switch pauses only their fleet
    bot_settings.save_user(BOB, fleet_engine_enabled=False)
    assert bot_engine.active_fleet_users() == [ALICE]

    # the house master switch stops everybody
    bot_settings.save_user(BOB, fleet_engine_enabled=True)
    bot_settings.save(engine_enabled=False)
    assert bot_engine.active_fleet_users() == []
    assert bot_engine.run_cycle()["skipped"] == "engine disabled"


def test_deleting_a_user_removes_their_whole_fleet_but_not_the_populations_memory():
    keep = bots.spawn_bot(_mk_genome(), origin="seed", user_id=ALICE)
    gone = bots.spawn_bot(_mk_genome(), origin="seed", user_id=BOB)
    fp = bots.get_bot(gone)["genome_fp"]

    assert bots.delete_user_bots(BOB) == 1
    assert bots.get_bot(gone) is None
    assert bots.get_bot(keep) is not None
    # the genome stays a known candidate — other fleets may still adopt it
    assert gene_pool.entry(fp) is not None


# ---------------------------------------------------------------------------
# 2. pooled evidence
# ---------------------------------------------------------------------------
def test_two_fleets_reach_the_proving_sample_in_half_the_sessions():
    """Neither fleet has enough trades on its own; together they do. That is
    the whole reason to share a population rather than run parallel ones."""
    g = _mk_genome()
    a = bots.spawn_bot(g, origin="seed", user_id=ALICE)
    b = bots.spawn_bot(dict(g), origin="seed", user_id=BOB)
    fp = bots.get_bot(a)["genome_fp"]
    assert bots.get_bot(b)["genome_fp"] == fp          # same genome, one record

    half = [0.30, 0.25, 0.35] * (N // 6 + 1)
    _add_trades(a, half, start_day=0)
    _add_trades(b, half, start_day=len(half))         # different sessions

    assert not bots.stats(a)["eligible"]               # thin on its own
    assert not bots.stats(b)["eligible"]

    pooled = gene_pool.pooled_stats(fp)
    assert pooled["users"] == 2 and pooled["bots"] == 2
    assert pooled["closed_count"] == len(half) * 2 >= N
    assert pooled["eligible"] and pooled["proven"]

    # ...and only the pooled view lets it breed
    assert bots.top_eligible(5, pool="fleet", user_id=ALICE) == []
    assert gene_pool.proven_by_family(5)


def test_a_genome_another_fleet_disproved_is_culled_here_too():
    """A bad genome should only have to be discovered once, by anybody.

    Alice's bot has barely traded, so on its own ledger it is not even
    eligible to be culled — it would sit in her fleet burning her market
    days rediscovering what Bob's fleet already established."""
    g = _mk_genome()
    mine = bots.spawn_bot(g, origin="seed", user_id=ALICE)
    theirs = bots.spawn_bot(dict(g), origin="seed", user_id=BOB)
    good = bots.spawn_bot(_mk_genome(strategy="breakout"), origin="seed", user_id=ALICE)

    _add_trades(mine, [0.02, -0.01])                          # barely any evidence here
    _add_trades(theirs, [-0.08, -0.06] * (N // 2 + 1))        # the other fleet found it out
    _add_trades(good, [0.30, 0.25, 0.35] * (N // 3 + 1))

    assert bots.stats(theirs)["status"] == "active"           # gentle enough not to trip drawdown
    assert gene_pool.pooled_stats(bots.get_bot(mine)["genome_fp"])["closed_count"] >= N

    # on Alice's own ledgers, her thin bot isn't even a candidate for culling
    own = [b["id"] for b in bots.worst_active(2, pool="fleet", user_id=ALICE,
                                              min_trades=bots.MIN_ELIGIBLE_TRADES,
                                              min_age_hours=0)]
    assert mine not in own

    # judged on the pooled record it is the obvious thing to retire, ahead
    # of the genome that is actually working
    pooled = [b["id"] for b in bots.worst_active(2, pool="fleet", user_id=ALICE, pooled=True,
                                                 min_trades=bots.MIN_ELIGIBLE_TRADES,
                                                 min_age_hours=0)]
    assert pooled[0] == mine and good in pooled

    # Bob's bot is Bob's: culling Alice's fleet never returns it
    assert theirs not in pooled


def test_the_multiple_comparison_bar_counts_the_search_not_the_crowd():
    g = _mk_genome()
    bots.spawn_bot(g, origin="seed", user_id=ALICE)
    bots.spawn_bot(dict(g), origin="seed", user_id=BOB)
    assert gene_pool.distinct_tested() == 1 == bots.bots_tested()
    bar_one = bots.proven_threshold()

    bots.spawn_bot(_mk_genome(strategy="breakout"), origin="seed", user_id=BOB)
    assert bots.bots_tested() == 2
    assert bots.proven_threshold() > bar_one       # a real new candidate does raise it


# ---------------------------------------------------------------------------
# 3. split search
# ---------------------------------------------------------------------------
def test_no_two_fleets_are_handed_the_same_candidate():
    g = _mk_genome()
    bots.spawn_bot(g, origin="seed", user_id=ALICE)
    fp = gene_pool.fingerprint(g)
    assert gene_pool.is_claimed(fp)
    assert not gene_pool.is_claimed(fp, exclude_user=ALICE)   # only Alice holds it

    # a breeder that keeps proposing what Alice is already running is made
    # to re-roll until it offers Bob something else
    proposals = [dict(g), dict(g), _mk_genome(strategy="breakout")]
    got, got_fp, tries = gene_pool.distinct_child(lambda: proposals.pop(0), exclude_user=BOB)
    assert got_fp != fp and tries == 3

    # ...and when every re-roll collides (a small or converged pool) it
    # still returns a genome rather than leaving the fleet a bot short
    same, same_fp, n = gene_pool.distinct_child(lambda: dict(g), exclude_user=BOB, tries=3)
    assert same_fp == fp and n == 3


def test_a_replacement_never_duplicates_a_running_genome():
    import random
    random.seed(11)
    for _ in range(12):
        bots.spawn_bot(_mk_genome(), origin="seed", user_id=ALICE)
    running = gene_pool.claimed_fingerprints()
    assert len(running) == 12

    for _ in range(20):
        g, _parent, _origin = bot_engine._replacement_genome(BOB)
        assert gene_pool.fingerprint(g) not in running


def test_an_abandoned_candidate_is_adopted_rather_than_forgotten():
    """A genome one fleet started and dropped is picked up by another, so
    its pooled record is completed instead of the trial being wasted."""
    from unittest.mock import patch
    started = bots.spawn_bot(_mk_genome(), origin="seed", user_id=ALICE)
    fp = bots.get_bot(started)["genome_fp"]
    _add_trades(started, [0.05, -0.02, 0.04])
    bots.retire_bot(started, "retired")               # nobody is running it now

    waiting = {e["fingerprint"] for e in gene_pool.pending()}
    assert fp in waiting

    with patch.object(bot_engine, "IMMIGRANT_RATE", 0.0), \
         patch.object(bot_engine, "ADOPT_RATE", 1.0):
        g, parent, origin = bot_engine._replacement_genome(BOB)
    assert origin == "adopted" and gene_pool.fingerprint(g) == fp

    picked = bots.spawn_bot(g, origin=origin, user_id=BOB)
    assert bots.get_bot(picked)["genome_fp"] == fp
    # Bob's new trades land on the same record Alice's did
    _add_trades(picked, [0.06, 0.03], start_day=10)
    assert gene_pool.pooled_stats(fp)["closed_count"] == 5


def test_a_genome_keeps_the_populations_generation_not_the_fleets():
    """Generations are the population's. A fleet that meets a genome late
    inherits its real lineage depth rather than restarting the count."""
    g = _mk_genome()
    gene_pool.register(g, generation=7, origin="mutation")
    late = bots.spawn_bot(g, origin="seed", user_id=BOB)
    assert bots.get_bot(late)["generation"] == 7


# ---------------------------------------------------------------------------
# shared market data
# ---------------------------------------------------------------------------
def test_market_data_uses_the_best_entitlement_anyone_has_connected(tmp_path, monkeypatch):
    assert bot_settings.data_source()["keys"] is None      # yfinance until somebody connects

    bot_settings.save_user(ALICE, fleet_alpaca_key_id="AK", fleet_alpaca_secret_key="AS",
                           fleet_data_feed="iex")
    assert bot_settings.data_source() == {"user_id": ALICE, "keys": ("AK", "AS"), "feed": "iex"}

    # one user upgrading their data plan upgrades the whole installation
    bot_settings.save_user(BOB, fleet_alpaca_key_id="BK", fleet_alpaca_secret_key="BS",
                           fleet_data_feed="sip")
    src = bot_settings.data_source()
    assert src["user_id"] == BOB and src["feed"] == "sip"


def test_recorded_history_is_one_shared_file_with_attribution(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "DB_PATH", tmp_path / "market_history.db")
    monkeypatch.setattr(recorder, "_initialized", set())
    recorder.init()

    day = datetime.now(recorder.ET).replace(hour=10, minute=0, second=0, microsecond=0)
    mk = lambda i: {"t": (day + timedelta(minutes=i)).astimezone(timezone.utc).isoformat(),
                    "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}

    assert recorder.record_bars("SPY", [mk(i) for i in range(5)], user_id=ALICE) == 5
    assert recorder.record_bars("SPY", [mk(i) for i in range(5, 9)], user_id=BOB) == 4
    # the same minutes again from another account add nothing: one history,
    # deduplicated, not one per user
    assert recorder.record_bars("SPY", [mk(i) for i in range(5)], user_id=BOB) == 0

    contrib = recorder.contributions()
    assert contrib["by_user"][ALICE]["bars"] == 5
    assert contrib["by_user"][BOB]["bars"] == 4
    assert contrib["totals"]["bars"] == 9 and contrib["totals"]["contributors"] == 2

    # and everyone reads all nine — the depth per user grows with the users
    assert len(recorder.bars_for("SPY", day.date())) == 9


# ---------------------------------------------------------------------------
# what must not regress
# ---------------------------------------------------------------------------
def test_pausing_a_fleet_does_not_abandon_that_users_custom_bots():
    """A custom bot is a separate experiment on separate keys. Pausing the
    fleet must not silently stop managing stops and trails on a live custom
    position — nor must having no fleet keys at all."""
    src = bots.spawn_bot(_mk_genome(), origin="seed", user_id=BOB)
    custom = bots.clone_bot(src, start_balance=1000.0, user_id=BOB)
    bot_settings.save_user(BOB, alpaca_key_id="CK", alpaca_secret_key="CS")

    # no fleet keys at all: the fleet cannot trade, the custom bot still does
    assert BOB in bot_engine.trading_users()
    assert [(b["id"], b["pool"]) for b in bot_engine._bots_to_step(BOB)] == [(custom, "custom")]

    bot_settings.save_user(BOB, fleet_alpaca_key_id="FK", fleet_alpaca_secret_key="FS")
    assert {b["id"] for b in bot_engine._bots_to_step(BOB)} >= {src, custom}

    bot_settings.save_user(BOB, fleet_engine_enabled=False)
    assert [b["id"] for b in bot_engine._bots_to_step(BOB)] == [custom]


def test_a_kicked_cycle_only_touches_the_account_that_kicked_it(monkeypatch):
    from webapp import account_book
    stepped = []
    monkeypatch.setattr(bot_engine, "_tick",
                        lambda now, users=None, **kw: stepped.append(list(users or []))
                        or {"stepped": 1, "idle": None})
    monkeypatch.setattr(bot_engine, "_phase", lambda now: "open")
    monkeypatch.setattr(bot_engine, "prepare", lambda *_a, **_k: ALICE)
    # the housekeeping half of a cycle would otherwise call Alpaca for real
    monkeypatch.setattr(bot_engine, "_recover_pending", lambda *_a, **_k: 0)
    monkeypatch.setattr(bot_engine, "_settle_expired", lambda *_a, **_k: 0)
    monkeypatch.setattr(account_book, "sync_all", lambda *_a, **_k: {})
    # an "open" cycle otherwise sleeps 15s between its four ticks
    monkeypatch.setattr(bot_engine, "CYCLE_BUDGET_S", -1.0)
    for uid in (ALICE, BOB):
        bot_settings.save_user(uid, fleet_alpaca_key_id="K{}".format(uid),
                               fleet_alpaca_secret_key="S")
        bots.spawn_bot(_mk_genome(), origin="seed", user_id=uid)

    bot_engine.run_cycle(user_ids=[BOB])
    assert stepped and all(u == [BOB] for u in stepped)

    stepped.clear()
    bot_engine.run_cycle()                      # the timer's cycle serves everyone
    assert stepped and stepped[0] == [ALICE, BOB]


def test_two_cycles_never_overlap():
    """Any user can kick a cycle from their fleet page while the timer's is
    still running; two of them would step the same bot twice off two market
    snapshots and double-enter it."""
    bot_engine._CYCLE_LOCK.acquire()
    try:
        assert bot_engine.run_cycle()["skipped"] == "a cycle is already running"
    finally:
        bot_engine._CYCLE_LOCK.release()


def test_a_bred_genome_keeps_its_parents_place_in_the_population():
    parent = bots.spawn_bot(_mk_genome(), origin="seed", user_id=ALICE)
    pfp = bots.get_bot(parent)["genome_fp"]
    assert bots.get_bot(parent)["generation"] == 0

    child = bots.spawn_bot(genome.mutate(bots.genome_of(bots.get_bot(parent))),
                           origin="mutation", user_id=BOB, parent_fp=pfp)
    assert bots.get_bot(child)["generation"] == 1

    grandchild = bots.spawn_bot(genome.mutate(bots.genome_of(bots.get_bot(child))),
                                origin="mutation", user_id=ALICE,
                                parent_fp=bots.get_bot(child)["genome_fp"])
    assert bots.get_bot(grandchild)["generation"] == 2


def test_the_archive_and_the_engine_log_stay_with_their_account():
    mine = bots.spawn_bot(_mk_genome(), origin="seed", user_id=ALICE)
    theirs = bots.spawn_bot(_mk_genome(), origin="seed", user_id=BOB)
    bots.archive_bot(mine, "test")
    bots.archive_bot(theirs, "test")

    assert [r["user_id"] for r in bots.list_archived(user_id=ALICE)] == [ALICE]
    assert len(bots.list_archived()) == 2            # the admin view
    assert bots.counts(user_id=ALICE)["archived"] == 1

    bots.log_event("evolve", {"user_id": BOB, "replaced": 1})
    bots.log_event("evolve", {"user_id": ALICE, "replaced": 2})
    bots.log_event("settled", {"n": 3})              # installation-wide, untagged

    kinds = [(e["kind"], e["detail"].get("user_id")) for e in bots.recent_events(50, user_id=ALICE)]
    assert ("evolve", BOB) not in kinds
    assert ("evolve", ALICE) in kinds and ("settled", None) in kinds


def test_resetting_a_fleet_requires_naming_one():
    """The old signature defaulted to "every fleet", which from a per-user
    button would have retired everybody's bots and spawned ownerless ones."""
    with pytest.raises(TypeError):
        bots.reset_fleet("oops")
