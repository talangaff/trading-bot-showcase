"""The incubator: a large population of simulated bots that earns fleet slots.

The fleet trades for real on Alpaca paper accounts, which caps it at a couple
of hundred bots per account and means every genome costs a real market day of
real orders to judge. The incubator is a second population (``pool='sim'``)
that runs the *same* entry / exit / position-management code
(bot_engine._step_bot) on the same live market snapshot, but books MODELED
fills (the ask on the way in, the bid on the way out, plus commission — see
broker.execution) and never sends an order. It costs CPU, not API calls, so it
can hold far more genomes than a fleet can, all being tested at once.

Three loops run on top of it, and between them they are the whole point:

  1. GRADUATE.  A sim bot that has traded enough, over enough sessions, with a
     positive baseline-adjusted fitness, a non-souring recent record and a
     drawdown well inside the guardrail is a *graduate*. Graduates are the
     first choice whenever a fleet has a free slot (bot_engine._spawn_replacement),
     and a fleet's worst bot is swapped out for a graduate that beat it
     (make_room). The fleet then re-tests the genome with real fills — it stays
     the final out-of-sample test, because a sim edge is partly selection luck
     (hundreds of genomes, a few will look good) and a modeled spread flatters
     a genome a little.

  2. CULL.  A sim bot that breaks the rules (the same drawdown guardrail the
     fleet has), goes idle, or builds a clearly negative record is retired and
     its slot is refilled. Sim slots are cheap, so this is aggressive.

  3. BREED.  Every sim slot that opens is refilled from the ELITE: the best
     genomes anywhere in the population — sim bots, promoted graduates and
     fleet bots — by robust fitness, picked by tournament so the very best
     breed most, family-balanced so one strategy cannot take over. Children are
     mutations (self-adaptive step, smaller for stronger parents) or crossovers
     of two elites, plus a few fresh random genomes to keep exploring. Each
     generation therefore starts from the previous generation's winners, and
     ``summary()`` reports fitness by generation so the climb is visible.

make_room() retires only the WORST flat fleet bot of the last SWAP_WINDOW_DAYS
days, and only when a graduate out-earned it over that window; the rest of the
fleet keeps running untouched.

Hard rules are untouched: sim bots go through bots.record_trade, so buy-to-open
only, min/max contracts, the position cap, the 15:55 0DTE cutoff and the
drawdown guardrail (which flags the bot 'failing') all apply to them exactly as
to the fleet.

    python -m webapp.incubator maintain     # cull / graduate / breed / swap (timer, every 5 min)
    python -m webapp.incubator status
"""

import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from . import bots, config, gene_pool, genome as _genome

POOL = "sim"

PER_FLEET = int(getattr(config, "BOT_SIM_PER_FLEET", 200))
TICK_BUDGET_S = float(getattr(config, "BOT_SIM_TICK_BUDGET_S", 10.0))

# --- graduation: the bar a sim bot clears to earn a real-fill trial --------
GRAD_MIN_TRADES = int(getattr(config, "BOT_SIM_GRAD_TRADES", 30))
GRAD_MIN_DAYS = int(getattr(config, "BOT_SIM_GRAD_DAYS", 5))
GRAD_MIN_FITNESS = float(getattr(config, "BOT_SIM_GRAD_FITNESS", 1.0))
GRAD_MAX_DD_FRAC = 0.6          # peak-to-trough must stay under this share of the guardrail

# --- culling ---------------------------------------------------------------
NEWBORN_GRACE_HOURS = 20        # a sim bot is not judged on a handful of early trades
CULL_MIN_TRADES = 20
CULL_FITNESS = -0.75
MAX_AGE_DAYS = 10               # a bot this old that never graduated has had its trial
IDLE_MARKET_DAYS = 3

# --- breeding --------------------------------------------------------------
ELITE_MIN_TRADES = 12           # the smallest record that can make a genome a parent
ELITE_K = 24                    # size of the breeding pool
IMMIGRANT_RATE = 0.10           # fresh random genomes — exploration
CROSSOVER_RATE = 0.30           # of the bred children
TOURNAMENT = 3                  # parent = best of this many random elites
MAX_FAMILY_SHARE = 0.40

# --- swapping losers out of the fleet for graduates ------------------------
SWAP_WINDOW_DAYS = 5.0          # a graduate that out-earned the worst fleet bot over this long takes its slot

MAINTAIN_EVERY_S = 240.0     # under the 5-min timer even after a slow pass (the stamp is written at the END)

_SUMMARY_KEY = "incubator_summary"
_ELITES_KEY = "incubator_elites"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_hours(b: dict) -> float:
    try:
        born = datetime.fromisoformat(b["created_at"])
    except (TypeError, ValueError, KeyError):
        return 1e9
    if born.tzinfo is None:
        born = born.replace(tzinfo=timezone.utc)
    return (_now() - born).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------
def active(user_id: int | None = None) -> list[dict]:
    return bots.active_bots(POOL, user_id)


def _rows(statuses: tuple, user_id: int | None = None) -> list[dict]:
    q = "SELECT * FROM bots WHERE COALESCE(pool,'fleet')='sim' AND status IN ({})".format(
        ",".join("?" * len(statuses)))
    params: list = list(statuses)
    if user_id is not None:
        q += " AND user_id=?"
        params.append(int(user_id))
    with bots._conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY id", params).fetchall()]


def seen_fingerprints() -> set:
    """Every genome the population is running or has recently run, fleet or
    sim — so a new sim bot is always a candidate nobody is testing and nobody
    just finished testing."""
    with bots._conn() as c:
        sim = {r["genome_fp"] for r in c.execute(
            "SELECT DISTINCT genome_fp FROM bots WHERE COALESCE(pool,'fleet')='sim' "
            "AND genome_fp IS NOT NULL").fetchall()}
    return sim | gene_pool.claimed_fingerprints()


def spawn(genome: dict, *, user_id: int, parent: dict | None = None,
          origin: str = "mutation") -> int | None:
    """One sim bot. No fleet cap applies (sim is bounded by fill()); the
    genome is NOT registered in the shared gene pool until it graduates, so
    the hundreds of genomes the incubator rejects do not inflate the
    multiple-comparison bar the fleet's 'proven' test is held to."""
    gen = int(parent["generation"]) + 1 if parent and parent.get("generation") is not None else 0
    return bots.create_bot(
        bots._fleet_name(gen), genome=genome, generation=gen,
        parent_id=(parent or {}).get("bot_id"), origin=origin, pool=POOL,
        user_id=int(user_id), genome_fp=gene_pool.fingerprint(genome))


# ---------------------------------------------------------------------------
# judging a sim bot
# ---------------------------------------------------------------------------
def verdict(bot: dict, s: dict | None = None) -> dict:
    """{'state': 'graduate'|'cull'|'keep', 'reason': str, 'stats': s}.

    The graduation bar is deliberately LOWER than the fleet's 'proven' bar
    (60 trades, 15 sessions, a Bonferroni t-stat): it only decides who gets a
    real-fill trial, and the real-fill record is what is held to the strict
    bar afterwards."""
    s = s or bots.stats(bot["id"])
    dd_limit = s.get("drawdown_limit") or bots.MAX_DRAWDOWN_PCT
    if bot.get("status") != "active":
        return {"state": "cull", "reason": "failed the drawdown guardrail", "stats": s}
    n = s["closed_count"]
    rf = s["robust_fitness"]
    recent_ok = s.get("fitness_recent") is None or s["fitness_recent"] >= 0
    if (n >= GRAD_MIN_TRADES and s["active_days"] >= GRAD_MIN_DAYS
            and rf >= GRAD_MIN_FITNESS and recent_ok and s["pl_dollar"] > 0
            and s["max_drawdown_pct"] <= dd_limit * GRAD_MAX_DD_FRAC):
        return {"state": "graduate", "reason": "cleared the graduation bar", "stats": s}
    if _age_hours(bot) >= NEWBORN_GRACE_HOURS:
        if n >= CULL_MIN_TRADES and rf <= CULL_FITNESS:
            return {"state": "cull", "reason": "negative record", "stats": s}
        if _age_hours(bot) >= MAX_AGE_DAYS * 24 and not (rf > 0 and recent_ok):
            return {"state": "cull", "reason": "trial expired", "stats": s}
    return {"state": "keep", "reason": "", "stats": s}


def _retire(bot: dict, status: str = "retired") -> None:
    bots.retire_bot(bot["id"], status)        # flattens at the mark first (modeled)


def _idle_ids(user_id: int | None) -> set:
    return {b["id"] for b in bots.stale_active_bots(IDLE_MARKET_DAYS, pool=POOL, user_id=user_id)}


# ---------------------------------------------------------------------------
# the elite — who gets to be a parent
# ---------------------------------------------------------------------------
def _elite_candidates(scored: list[dict]) -> list[dict]:
    """Best genomes anywhere: sim bots (active or promoted) from `scored`,
    plus fleet genomes on their POOLED real-fill record. Each is
    {genome, fp, robust, family, generation, bot_id, source}."""
    out: dict = {}

    def offer(fp, g, robust, generation, bot_id, source):
        if fp not in out or robust > out[fp]["robust"]:
            out[fp] = {"genome": g, "fp": fp, "robust": robust, "generation": generation,
                       "bot_id": bot_id, "source": source, "family": _genome.family_of(g)}

    for r in scored:
        s = r["stats"]
        if r["bot"]["status"] not in ("active", "promoted"):
            continue
        if s["closed_count"] < ELITE_MIN_TRADES or s["robust_fitness"] <= 0:
            continue
        if s.get("fitness_recent") is not None and s["fitness_recent"] < 0:
            continue
        g = _genome.loads(r["bot"].get("genome"))
        offer(r["bot"]["genome_fp"] or gene_pool.fingerprint(g), g, s["robust_fitness"],
              r["bot"].get("generation") or 0, r["bot"]["id"], "sim")
    try:
        for e in gene_pool.scoreboard():
            if e["closed_count"] < ELITE_MIN_TRADES or e["robust_fitness"] <= 0:
                continue
            if e.get("fitness_recent") is not None and e["fitness_recent"] < 0:
                continue
            offer(e["fingerprint"], _genome.loads(e["genome"]), e["robust_fitness"],
                  e.get("generation") or 0, None, "fleet")
    except Exception as ex:  # noqa: BLE001 — the pool is a bonus source, never a blocker
        print("incubator: fleet elite lookup failed: {}".format(ex), flush=True)
    return sorted(out.values(), key=lambda d: d["robust"], reverse=True)


def elite_parents(scored: list[dict] | None = None, k: int = ELITE_K) -> list[dict]:
    """The breeding pool: the top `k` elite genomes, family-balanced (no
    family holds more than MAX_FAMILY_SHARE of the slots, so a lineage that
    happens to lead cannot crowd out every other strategy)."""
    scored = _score_all() if scored is None else scored
    cands = _elite_candidates(scored)
    cap = max(1, int(math.ceil(k * MAX_FAMILY_SHARE)))
    per_family: dict = {}
    out = []
    for c in cands:
        if per_family.get(c["family"], 0) >= cap:
            continue
        per_family[c["family"]] = per_family.get(c["family"], 0) + 1
        out.append(c)
        if len(out) >= k:
            break
    return out


def cached_elites() -> list[dict]:
    """The breeding pool as of the last maintenance pass. The fleet's refill
    path (bot_engine._replacement_genome) runs inside the one-minute trading
    cycle, where re-scoring every ledger is not affordable — it reads this."""
    try:
        return json.loads(bots._meta_get(_ELITES_KEY) or "[]")
    except ValueError:
        return []


def _tournament(elites: list[dict]) -> dict:
    return max(random.sample(elites, min(TOURNAMENT, len(elites))), key=lambda d: d["robust"])


def child_of(elites: list[dict], *, other_quota: set | None = None):
    """(genome, parent_dict_or_None, origin). The generation loop: stronger
    parents breed more (tournament) and are nudged more gently (a hill-climb
    around what already works); weaker ones, and the random immigrants, keep
    the search wide."""
    if not elites or random.random() < IMMIGRANT_RATE:
        return _genome.random_genome(), None, "seed"
    pool = [e for e in elites if not (other_quota and e["family"] in other_quota)] or elites
    a = _tournament(pool)
    if len(pool) > 1 and random.random() < CROSSOVER_RATE:
        b = _tournament([e for e in pool if e["fp"] != a["fp"]] or pool)
        return (_genome.mutate(_genome.crossover(a["genome"], b["genome"]), rate=0.12),
                a, "crossover")
    top_quartile = a["robust"] >= elites[min(len(elites) - 1, max(0, len(elites) // 4))]["robust"]
    return _genome.mutate(a["genome"], rate=0.2 if top_quartile else 0.35), a, "mutation"


def fill(user_id: int, scored: list[dict] | None = None, elites: list[dict] | None = None,
         target: int | None = None, seen: set | None = None) -> dict:
    """Top `user_id`'s incubator up to `target` (default PER_FLEET) with new
    candidates bred from the elite. Every child is distinct from everything
    running anywhere (re-rolled up to a dozen times before a duplicate is
    tolerated)."""
    target = PER_FLEET if target is None else int(target)
    scored = _score_all() if scored is None else scored
    elites = elite_parents(scored) if elites is None else elites
    seen = seen_fingerprints() if seen is None else seen
    have = active(user_id)
    over = {f for f, share in _family_shares(have).items() if share >= MAX_FAMILY_SHARE}
    made = {"seed": 0, "mutation": 0, "crossover": 0}
    for _ in range(max(0, target - len(have))):
        for _try in range(12):
            g, parent, origin = child_of(elites, other_quota=over)
            fp = gene_pool.fingerprint(g)
            if fp not in seen:
                break
        seen.add(fp)
        if spawn(g, user_id=user_id, parent=parent, origin=origin) is not None:
            made[origin] += 1
    return made


def _family_shares(rows: list[dict]) -> dict:
    fams: dict = {}
    for b in rows:
        f = _genome.family_of(_genome.loads(b.get("genome")))
        fams[f] = fams.get(f, 0) + 1
    n = sum(fams.values())
    return {f: c / n for f, c in fams.items()} if n else {}


# ---------------------------------------------------------------------------
# scoring the whole population once per pass
# ---------------------------------------------------------------------------
def _score_all() -> list[dict]:
    """[{bot, stats}] for every sim row still in the DB (active, promoted and
    recently retired) — computed once per maintenance pass and shared by
    culling, breeding and the summary, since stats() reads a whole ledger."""
    out = []
    for b in _rows(("active", "promoted", "retired", "failing")):
        out.append({"bot": b, "stats": bots.stats(b["id"])})
    return out


# ---------------------------------------------------------------------------
# graduates -> fleet
# ---------------------------------------------------------------------------
_GRAD_CACHE: dict = {}
_GRAD_TTL = 30.0


def graduates(user_id: int, fresh: bool = False) -> list[dict]:
    """Active sim bots of `user_id` that clear the graduation bar, best
    first, minus any genome this fleet is already running."""
    hit = _GRAD_CACHE.get(user_id)
    if not fresh and hit and time.time() - hit[0] < _GRAD_TTL:
        return [g for g in hit[1] if g["bot"]["status"] == "active"]
    running = {b["genome_fp"] for b in bots.active_bots("fleet", user_id) if b.get("genome_fp")}
    out = []
    for b in active(user_id):
        if b["genome_fp"] in running:
            continue
        v = verdict(b)
        if v["state"] == "graduate":
            out.append({"bot": b, "stats": v["stats"]})
    out.sort(key=lambda r: r["stats"]["robust_fitness"], reverse=True)
    _GRAD_CACHE[user_id] = (time.time(), out)
    return out


def promote_next(user_id: int, fresh: bool = False) -> int | None:
    """Move the best waiting graduate into `user_id`'s fleet as a NEW fleet
    bot (own real-fill ledger from zero), if the fleet has a free slot.
    Returns the new fleet bot's id, else None. The sim bot is kept (status
    'promoted') as the lineage and as a parent for the next generation."""
    if bots.fleet_headroom(user_id) <= 0:
        return None
    for r in graduates(user_id, fresh=fresh):
        b = r["bot"]
        parent_fp = None
        if b.get("parent_id"):
            parent_fp = (bots.get_bot(b["parent_id"]) or {}).get("genome_fp")
        fleet_id = bots.spawn_bot(_genome.loads(b["genome"]), parent_id=b["id"],
                                  origin="graduate", user_id=user_id,
                                  generation=int(b.get("generation") or 0),
                                  parent_fp=parent_fp)
        if fleet_id is None:
            return None
        bots.retire_bot(b["id"], "promoted")
        _GRAD_CACHE.pop(user_id, None)
        s = r["stats"]
        bots.log_event("graduate", {
            "user_id": user_id, "sim_bot": b["id"], "fleet_bot": fleet_id,
            "generation": b.get("generation"), "trades": s["closed_count"],
            "days": s["active_days"], "fitness": round(s["robust_fitness"], 2),
            "pl_pct": round(s["pl_pct"] or 0.0, 2)})
        return fleet_id
    return None


def make_room(user_id: int, scored_graduates: list[dict] | None = None) -> int:
    """A full fleet swaps its WORST bot for a graduate that has done better.

    Compare realized return over the last SWAP_WINDOW_DAYS days: the graduate
    against each fleet bot. Fleet bots are ranked worst first and paired with
    graduates best first; a pair swaps while the graduate's return is strictly
    higher, and stops at the first pair where it is not. The loser is retired
    at once and the graduate takes the slot as a new fleet bot, trading real
    fills from then on.

    Protected: a fleet bot younger than the window (it has not had a full
    window to be judged — this also keeps a just-promoted graduate in place)
    and a bot holding an open position (retiring it would mean sending a close
    order from this process; it is judged again on the next pass once flat)."""
    grads = graduates(user_id, fresh=True) if scored_graduates is None else scored_graduates
    if not grads or bots.fleet_headroom(user_id) > 0:
        return 0
    scored_grads = sorted(((bots.window_return_pct(g["bot"]["id"], SWAP_WINDOW_DAYS), g)
                           for g in grads), key=lambda t: t[0], reverse=True)
    pool = []
    for b in bots.active_bots("fleet", user_id):
        if _age_hours(b) < SWAP_WINDOW_DAYS * 24 or bots._positions(b["id"]):
            continue
        pool.append((bots.window_return_pct(b["id"], SWAP_WINDOW_DAYS), b))
    pool.sort(key=lambda t: t[0])
    swapped = 0
    for (loser_ret, loser), (grad_ret, g) in zip(pool, scored_grads):
        if grad_ret <= loser_ret:
            break
        bots.retire_bot(loser["id"], "retired")
        bots.log_event("swap_out", {"user_id": user_id, "fleet_bot": loser["id"],
                                    "return_pct": round(loser_ret, 2),
                                    "graduate_return_pct": round(grad_ret, 2),
                                    "window_days": SWAP_WINDOW_DAYS,
                                    "replaced_by_sim": g["bot"]["id"]})
        if promote_next(user_id, fresh=True) is not None:
            swapped += 1
    return swapped


# ---------------------------------------------------------------------------
# one maintenance pass
# ---------------------------------------------------------------------------
def maintain(user_ids: list | None = None, *, force: bool = False) -> dict:
    """Cull -> promote -> swap -> breed -> publish the summary, for every
    fleet account (or `user_ids`). Throttled to once per MAINTAIN_EVERY_S
    unless forced, since it scores the whole population."""
    from . import bot_engine
    bots.init()
    last = bots._meta_get("incubator_last_run")
    if not force and last:
        try:
            if time.time() - float(last) < MAINTAIN_EVERY_S:
                return {"skipped": "ran recently"}
        except ValueError:
            pass
    users = [int(u) for u in (user_ids if user_ids is not None else bot_engine.fleet_users())]
    ev = {"users": len(users), "culled": 0, "promoted": 0, "swapped": 0,
          "seed": 0, "mutation": 0, "crossover": 0}

    ev["never_traded"] = sum(bots.retire_never_traded().values())

    scored = _score_all()
    idle = _idle_ids(None)
    for r in scored:
        b = r["bot"]
        if b["status"] not in ("active", "failing") or int(b["user_id"] or -1) not in users:
            continue
        v = verdict(b, r["stats"])
        if v["state"] == "cull" or (v["state"] == "keep" and b["id"] in idle
                                    and _age_hours(b) >= NEWBORN_GRACE_HOURS):
            _retire(b)
            b["status"] = "retired"
            ev["culled"] += 1

    fleet_on = set(bot_engine.active_fleet_users())
    for uid in users:
        if uid not in fleet_on:
            continue
        graduates(uid, fresh=True)
        while promote_next(uid) is not None:
            ev["promoted"] += 1
        ev["swapped"] += make_room(uid)

    # re-score after the culls and promotions so breeding sees the survivors
    scored = _score_all()
    elites = elite_parents(scored)
    seen = seen_fingerprints()
    for uid in users:
        for k, v in fill(uid, scored, elites, seen=seen).items():
            ev[k] += v

    ev["elite"] = len(elites)
    bots._meta_set(_ELITES_KEY, json.dumps(elites))
    _publish(scored, users)
    bots._meta_set("incubator_last_run", time.time())
    if ev["culled"] or ev["never_traded"] or ev["promoted"] or ev["swapped"] or ev["mutation"] or ev["crossover"]:
        bots.log_event("incubator", ev)
    return ev


# ---------------------------------------------------------------------------
# what the Bot Fleet page shows
# ---------------------------------------------------------------------------
def _publish(scored: list[dict], users: list[int]) -> None:
    """Store the numbers the page renders. Computed here, once per pass,
    because deriving them means scoring every sim ledger — far too much to
    do per page load."""
    per_user: dict = {}
    for uid in users:
        mine = [r for r in scored if int(r["bot"]["user_id"] or -1) == uid]
        act = [r for r in mine if r["bot"]["status"] == "active"]
        rated = [r for r in act if r["stats"]["closed_count"] >= ELITE_MIN_TRADES]
        gens: dict = {}
        for r in mine:                       # active + promoted + recently retired
            if r["stats"]["closed_count"] < 8:
                continue
            gens.setdefault(int(r["bot"].get("generation") or 0), []).append(
                r["stats"]["robust_fitness"])
        top = sorted(act, key=lambda r: r["stats"]["robust_fitness"], reverse=True)[:12]
        top_pl = sorted(act, key=lambda r: r["stats"]["pl_pct"] or 0.0, reverse=True)[:12]

        def row(r):
            return {
                "id": r["bot"]["id"], "name": r["bot"]["name"],
                "generation": r["bot"].get("generation") or 0,
                "origin": r["bot"].get("origin"),
                "desc": _genome.describe(_genome.loads(r["bot"].get("genome"))),
                "trades": r["stats"]["closed_count"], "days": r["stats"]["active_days"],
                "fitness": r["stats"]["robust_fitness"], "pl_pct": r["stats"]["pl_pct"],
                "graduate": verdict(r["bot"], r["stats"])["state"] == "graduate"}
        per_user[str(uid)] = {
            "active": len(act),
            "trading": sum(1 for r in act if r["stats"]["closed_count"] > 0),
            "rated": len(rated),
            "in_profit": sum(1 for r in rated if r["stats"]["pl_dollar"] > 0),
            "graduates": sum(1 for r in act if verdict(r["bot"], r["stats"])["state"] == "graduate"),
            "promoted": sum(1 for r in mine if r["bot"]["status"] == "promoted"),
            "retired_recent": sum(1 for r in mine if r["bot"]["status"] == "retired"),
            "trades": sum(r["stats"]["closed_count"] for r in act),
            "generations": [
                {"gen": g, "n": len(v), "mean": sum(v) / len(v), "best": max(v)}
                for g, v in sorted(gens.items())],
            "top": [row(r) for r in top],
            "top_pl": [row(r) for r in top_pl],
        }
    bots._meta_set(_SUMMARY_KEY, json.dumps({"at": _now().isoformat(), "users": per_user}))


_CAP_KEY = "incubator_capacity"
SIM_PASSES = 3                  # sim passes per cycle: the idle gaps between the 4 ticks
_EMA = 0.2


def record_capacity(sim_stepped: int, sim_s: float, cycle_s: float, sim_bots: int) -> None:
    """Fold one cycle's timing into a smoothed picture of how much headroom the
    box has for more sim bots. coverage = steps made / steps wanted (every sim
    bot stepped on every pass); pass_s = seconds one sim pass takes."""
    if sim_bots <= 0 or sim_s <= 0:
        return
    cov = min(1.0, sim_stepped / float(sim_bots * SIM_PASSES))
    pass_s = sim_s / SIM_PASSES
    try:
        old = json.loads(bots._meta_get(_CAP_KEY) or "{}")
    except ValueError:
        old = {}

    def ema(key, new):
        return new if old.get(key) is None else old[key] * (1 - _EMA) + new * _EMA

    bots._meta_set(_CAP_KEY, json.dumps({
        "coverage": ema("coverage", cov), "pass_s": ema("pass_s", pass_s),
        "cycle_s": ema("cycle_s", cycle_s), "sim_bots": sim_bots,
        "per_step_ms": ema("per_step_ms", 1000.0 * sim_s / max(1, sim_stepped)),
        "n": int(old.get("n", 0)) + 1, "at": time.time()}))


def capacity() -> dict:
    """Is there room for more sim bots? Verdict 'room' / 'tight' / 'full' from
    the smoothed cycle timings, plus a rough count of how many more sim bots
    (all fleets together) would still fit the per-pass budget at 90% coverage."""
    try:
        c = json.loads(bots._meta_get(_CAP_KEY) or "{}")
    except ValueError:
        c = {}
    if not c or c.get("n", 0) < 3:
        return {"verdict": "measuring", "detail": "waiting for a few cycles of data"}
    budget = TICK_BUDGET_S
    util = c["pass_s"] / budget
    cycle_cap = 58.0                      # a cycle that nears the minute starts running late
    if c["coverage"] >= 0.98 and util <= 0.5 and c["cycle_s"] <= 56:
        verdict = "room"
    elif c["coverage"] >= 0.9 and util <= 0.85 and c["cycle_s"] <= cycle_cap:
        verdict = "tight"
    else:
        verdict = "full"
    per_step_s = max(c["per_step_ms"], 0.1) / 1000.0
    fit = int(budget * 0.9 / per_step_s)          # steps per pass that fit 90% of the budget
    more = max(0, fit - int(c["sim_bots"]))
    return {"verdict": verdict, "coverage": c["coverage"], "pass_s": c["pass_s"],
            "budget_s": budget, "cycle_s": c["cycle_s"], "sim_bots": int(c["sim_bots"]),
            "more_fit": more, "util": util}


def summary(user_id: int) -> dict:
    """The stored numbers for one account, plus the recent graduations. An
    account the incubator has not run for yet gets zeros, not an error."""
    empty = {"active": 0, "trading": 0, "rated": 0, "in_profit": 0, "graduates": 0,
             "promoted": 0, "retired_recent": 0, "trades": 0, "generations": [], "top": [], "top_pl": []}
    try:
        blob = json.loads(bots._meta_get(_SUMMARY_KEY) or "{}")
    except ValueError:
        blob = {}
    mine = {**empty, **(blob.get("users", {}).get(str(user_id)) or {})}
    mine["at"] = blob.get("at")
    mine["active"] = len(active(user_id))       # live: the stored count lags a refill by one pass
    mine["target"] = PER_FLEET
    mine["capacity"] = capacity()
    mine["recent"] = [e for e in bots.recent_events(400, user_id=user_id)
                      if e.get("kind") in ("graduate", "swap_out")][:8]
    return mine


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "maintain"
    if cmd == "maintain":
        print(maintain(force="--force" in argv), flush=True)
    elif cmd == "status":
        bots.init()
        for u in sorted({int(b["user_id"]) for b in active() if b.get("user_id") is not None}):
            s = summary(u)
            print({k: s[k] for k in ("active", "trading", "rated", "in_profit", "graduates",
                                     "promoted", "retired_recent", "generations")}, flush=True)
    else:
        print("usage: python -m webapp.incubator [maintain [--force]|status]", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
