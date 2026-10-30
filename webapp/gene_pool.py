"""The shared gene pool — one evolutionary search, run by every user's fleet.

Each user owns a *fleet* of bots that trade on that user's own Alpaca paper
keys (webapp/bot_settings.py), but there is only ONE population of genomes,
kept here. Every genome a fleet has ever been given a live trial is
registered in the ``gene_pool`` table by a stable fingerprint of its
behaviour genes, and every bot row carries that fingerprint
(``bots.genome_fp``). Two things follow, and they are the whole point of the
module:

  * **Pooled evidence.** A genome's track record is the union of the closed
    trades of EVERY user's bot that ran it — see ``pooled_stats()``. A
    genome two users are running gathers its 60-trade proving sample in half
    the calendar days it would take one user alone, so "proven" (and
    "cull it") arrives sooner for everybody. Fitness itself is still the
    same day-adjusted, shrunk t-statistic bots.stats() computes; only the
    sample it is computed over is bigger.

  * **Split search.** Breeding never hands one user a genome another user's
    fleet is already running live (``is_claimed`` / ``distinct_child``).
    N users therefore explore N times as many distinct genomes per market
    day, instead of N correlated copies of the same one — which is where
    the real acceleration comes from, since calendar days, not CPU, are what
    this search is short of. A genome that was registered but never got (or
    has since lost) a live bot is left ``pending``: any user's fleet can
    adopt it and add its trades to the same pooled record.

A user's own dashboard still shows their own bot's own ledger — their money,
their fills, their P&L. Only the *decisions* evolution makes (who may breed,
who gets culled, where the multiple-comparison bar sits) read the pooled
numbers.

Custom bots (pool='custom') are hand-tuned and deliberately outside
evolution, exactly as before: they never register a genome here, never
contribute pooled evidence, and are never bred from.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from . import bots as _bots, genome as _genome

# genes that describe *behaviour*. `mut_scale` is the self-adaptive mutation
# step — bookkeeping for the lineage, never read by the strategy — so two
# genomes that trade identically must not fingerprint differently over it.
_NON_BEHAVIOUR = ("mut_scale",)

# how many of a mutation's re-rolls we will spend looking for a genome no
# other active fleet bot is running before accepting a duplicate
_DISTINCT_TRIES = 12


def _now() -> str:
    # microseconds, not seconds: a pool reset and the cohort it reseeds
    # happen in the same second, and `created_at >= reset` has to put the
    # new genomes after the marker and the old ones before it
    return datetime.now(timezone.utc).isoformat()


def _canonical(g: dict) -> str:
    """Stable JSON for the behaviour genes only, floats rounded so two
    genomes that differ by a rounding artefact share a fingerprint."""
    out = {}
    for k, v in (g or {}).items():
        if k in _NON_BEHAVIOUR:
            continue
        if isinstance(v, float):
            out[k] = round(v, 6)
        elif isinstance(v, dict):
            out[k] = {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                      for kk, vv in sorted(v.items())}
        else:
            out[k] = v
    return json.dumps(out, separators=(",", ":"), sort_keys=True)


def fingerprint(g: dict | None) -> str:
    return hashlib.sha1(_canonical(g or {}).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# storage (lives in bots.db — one file, one shared population)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS gene_pool (
    fingerprint TEXT PRIMARY KEY,
    genome      TEXT NOT NULL,
    generation  INTEGER NOT NULL DEFAULT 0,
    origin      TEXT,
    parent_fp   TEXT,
    family      TEXT,
    created_at  TEXT NOT NULL,
    first_user  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_gene_pool_created ON gene_pool(created_at);
"""


def register(g: dict, *, generation: int = 0, origin: str = "seed",
             parent_fp: str | None = None, user_id: int | None = None) -> str:
    """Put a genome in the shared pool (idempotent) and return its
    fingerprint. Re-registering a genome another user already introduced
    keeps the ORIGINAL row — its generation, lineage and pooled history are
    the population's, not one user's."""
    fp = fingerprint(g)
    with _bots._conn() as c:
        c.execute(
            "INSERT INTO gene_pool(fingerprint, genome, generation, origin, parent_fp, "
            "family, created_at, first_user) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(fingerprint) DO NOTHING",
            (fp, _genome.dumps(g), int(generation or 0), origin, parent_fp,
             _genome.family_of(g), _now(),
             int(user_id) if user_id is not None else None))
    return fp


def entry(fp: str) -> dict | None:
    with _bots._conn() as c:
        r = c.execute("SELECT * FROM gene_pool WHERE fingerprint=?", (fp,)).fetchone()
    return dict(r) if r else None


def entries() -> list[dict]:
    with _bots._conn() as c:
        rows = c.execute("SELECT * FROM gene_pool ORDER BY created_at, fingerprint").fetchall()
    return [dict(r) for r in rows]


def genome_of(fp: str) -> dict | None:
    e = entry(fp)
    return _genome.loads(e["genome"]) if e else None


def size() -> int:
    with _bots._conn() as c:
        return int(c.execute("SELECT COUNT(*) FROM gene_pool").fetchone()[0])


def distinct_tested() -> int:
    """The N in the multiple-comparison correction (bots.proven_threshold):
    how many DISTINCT genomes the whole population has given a live trial
    since the last pool reset.

    Counting distinct genomes rather than bot rows is what keeps the bar
    honest under multiple fleets: ten users running the same 60 genomes
    have searched 60 candidates, not 600, and the threshold should reflect
    the size of the search, not the size of the crowd."""
    since = _bots._meta_get("pool_reset_at") or ""
    with _bots._conn() as c:
        if since:
            n = c.execute("SELECT COUNT(*) FROM gene_pool WHERE created_at >= ?",
                          (since,)).fetchone()[0]
        else:
            n = c.execute("SELECT COUNT(*) FROM gene_pool").fetchone()[0]
    return max(1, int(n))


def reset_pool(reason: str = "reset") -> None:
    """Start the multiple-comparison count over. The genomes themselves are
    kept — their pooled history is the population's memory — but only trials
    from here on count toward the proven bar."""
    _bots._meta_set("pool_reset_at", _now())
    _bots.log_event("pool_reset", {"reason": reason, "pool_size": size()})


# ---------------------------------------------------------------------------
# who is running what
# ---------------------------------------------------------------------------
def _fleet_bots_by_fp(fp: str, *, active_only: bool = False) -> list[dict]:
    sql = ("SELECT * FROM bots WHERE genome_fp=? AND COALESCE(pool,'fleet')='fleet'"
           + (" AND status='active'" if active_only else ""))
    with _bots._conn() as c:
        rows = c.execute(sql, (fp,)).fetchall()
    return [dict(r) for r in rows]


def runners(fp: str) -> list[int]:
    """User ids with an ACTIVE fleet bot running this genome right now."""
    return sorted({int(b["user_id"]) for b in _fleet_bots_by_fp(fp, active_only=True)
                   if b.get("user_id") is not None})


def is_claimed(fp: str, *, exclude_user: int | None = None) -> bool:
    """True if some active fleet bot is already running this genome. Used to
    keep two users off the same candidate — see distinct_child()."""
    r = runners(fp)
    if exclude_user is not None:
        r = [u for u in r if u != int(exclude_user)]
    return bool(r)


def claimed_fingerprints() -> set:
    with _bots._conn() as c:
        rows = c.execute(
            "SELECT DISTINCT genome_fp FROM bots WHERE status='active' "
            "AND COALESCE(pool,'fleet')='fleet' AND genome_fp IS NOT NULL").fetchall()
    return {r["genome_fp"] for r in rows}


def distinct_child(make, *, exclude_user: int | None = None,
                   tries: int = _DISTINCT_TRIES, claimed: set | None = None):
    """Call `make()` until it returns a genome no active fleet bot is
    running, and return (genome, fingerprint, tries_used).

    This is the split-search rule: a user's new bot gets a candidate nobody
    else is currently testing, so the population covers N times as much of
    the search space per market day as one fleet could. If every re-roll
    collides (a small or converged pool) the last one is used anyway — a
    duplicate is worse than a distinct genome, but far better than no bot.

    A caller filling many slots at once (evolve, a fleet reset) passes the
    `claimed` set once and adds each accepted fingerprint to it, so a pass
    neither re-queries per re-roll nor hands two of its OWN new bots the
    same genome.
    """
    g = fp = None
    for i in range(max(1, tries)):
        g = make()
        fp = fingerprint(g)
        taken = (fp in claimed) if claimed is not None else is_claimed(
            fp, exclude_user=exclude_user)
        if not taken:
            if claimed is not None:
                claimed.add(fp)
            return g, fp, i + 1
    return g, fp, tries


_PENDING_CACHE: dict = {}


def _stale_ok(ts: float) -> bool:
    import time as _t
    return _bots.WARMER_ON and _t.time() - ts < 600.0


def pending_view(limit: int = 20, *, fresh: bool = False) -> list[dict]:
    """`pending()` behind the same 45s cache, for the page that lists it."""
    import time as _t
    hit = _PENDING_CACHE.get(limit)
    if not fresh and hit and (_t.time() - hit[0] < 45.0 or _stale_ok(hit[0])):
        return hit[1]
    value = pending(limit)
    _PENDING_CACHE[limit] = (_t.time(), value)
    return value


def pending(limit: int = 50, claimed: set | None = None) -> list[dict]:
    """Registered genomes with no active fleet bot anywhere — candidates any
    user's fleet can adopt so their pooled record keeps growing instead of
    being abandoned half-tested. Least-tested first."""
    claimed = claimed_fingerprints() if claimed is None else claimed
    out = []
    for e in entries():
        if e["fingerprint"] in claimed:
            continue
        s = pooled_stats(e["fingerprint"])
        if s["closed_count"] >= _bots.MIN_ELIGIBLE_TRADES:
            continue                     # already had its trial; nothing left to learn
        if s["closed_count"] >= 5 and s["fitness"] < 0:
            continue                     # tried and souring — don't spend a slot on it
        out.append({**e, "pooled": s})
    out.sort(key=lambda d: (d["pooled"]["closed_count"], d["created_at"]))
    return out[:limit]


# ---------------------------------------------------------------------------
# pooled evidence
# ---------------------------------------------------------------------------
def _pooled_ledger(fp: str) -> tuple[list[dict], list[dict]]:
    """(closed trades, open positions marked to their last mid) across every
    user's fleet bot that has ever run this genome."""
    closed: list[dict] = []
    marked: list[dict] = []
    for b in _fleet_bots_by_fp(fp):
        bid = b["id"]
        closed.extend(_bots._match(_bots._trades(bid))[1])
        marked.extend(_bots._marked_as_closed(_bots._positions(bid), _bots.marks(bid)))
    closed.sort(key=lambda r: r.get("ts") or "")
    return closed, marked


# Pooled stats read every ledger that has ever run a genome, and the fleet
# page asks for one per row while breeding asks for one per pool entry. A
# short TTL (the leaderboard's own) keeps a page load to a single pass over
# the trades rather than one per row; every write that changes a genome's
# record invalidates it (bots._invalidate_pooled), so nothing is ever served
# from a cycle that has already moved on.
_STATS_CACHE: dict = {}
_STATS_TTL = 10.0


def invalidate(fp: str | None = None) -> None:
    """Drop cached pooled stats — for one genome, or all of them."""
    if fp is None:
        _STATS_CACHE.clear()
        _COVERAGE_CACHE.update(ts=0.0, value=None)
        _SB_CACHE.update(ts=0.0, value=None)
        _PENDING_CACHE.clear()
    else:
        _STATS_CACHE.pop(fp, None)


def _cache(fp: str, value: dict) -> dict:
    import time as _t
    _STATS_CACHE[fp] = (_t.time(), value)
    return value


def pooled_stats(fp: str | None, *, fresh: bool = False) -> dict:
    """The genome's track record across the whole population.

    Same formulas bots.stats_from_ledger() uses for one bot — day-adjusted
    per-trade returns, shrunk toward the fleet prior, divided by their
    standard error — but computed over every user's trades on this genome
    at once, and with the eligibility gates (trades / sessions / regimes)
    counted over the union too.
    """
    empty = {"fingerprint": fp, "bots": 0, "users": 0, "user_ids": [],
             "closed_count": 0, "active_days": 0, "regimes": [], "regime_count": 0,
             "fitness": 0.0, "fitness_recent": None, "fitness_marked": 0.0,
             "robust_fitness": 0.0, "eligible": False, "proven": False, "promising": False,
             "proven_threshold": _bots.proven_threshold(), "pl_dollar": 0.0,
             "avg_trade_return": None}
    if not fp:
        return empty
    import time as _t
    hit = _STATS_CACHE.get(fp)
    if not fresh and hit and _t.time() - hit[0] < _STATS_TTL:
        return hit[1]
    rows = _fleet_bots_by_fp(fp)
    if not rows:
        return _cache(fp, empty)
    closed, marked = _pooled_ledger(fp)
    users = sorted({int(b["user_id"]) for b in rows if b.get("user_id") is not None})
    if not closed:
        return _cache(fp, {**empty, "bots": len(rows), "users": len(users),
                           "user_ids": users})

    base = _bots.baseline_day_returns()
    prior = _bots.fleet_prior_return()
    adj = _bots.adjusted_returns(closed, base)
    fitness = _bots.fitness_from_returns(adj, prior)
    fitness_recent = _bots._fitness_recent(closed, 0.0, baseline_returns=base)
    fitness_marked = (_bots.fitness_from_returns(adj + _bots.adjusted_returns(marked, None), prior)
                      if marked else fitness)
    robust = min(x for x in (fitness, fitness_recent, fitness_marked) if x is not None)

    days = {d for d in (_bots._trade_date(r) for r in closed) if d is not None}
    regimes = sorted({r["regime"] for r in closed if r.get("regime")})
    eligible = (len(closed) >= _bots.MIN_ELIGIBLE_TRADES
                and len(days) >= _bots.MIN_ELIGIBLE_DAYS
                and len(regimes) >= _bots.MIN_ELIGIBLE_REGIMES)
    threshold = _bots.proven_threshold()
    pl_total = sum(r.get("pl") or 0.0 for r in closed)
    gross_w = sum(r["pl"] for r in closed if (r.get("pl") or 0.0) > 1e-9)
    gross_l = -sum(r["pl"] for r in closed if (r.get("pl") or 0.0) < -1e-9)
    profitable = pl_total > 0 and gross_w >= gross_l
    proven = bool(eligible and profitable and robust >= threshold)
    return _cache(fp, {
        "fingerprint": fp,
        "bots": len(rows),
        "users": len(users),
        "user_ids": users,
        "closed_count": len(closed),
        "active_days": len(days),
        "regimes": regimes,
        "regime_count": len(regimes),
        "fitness": fitness,
        "fitness_recent": fitness_recent,
        "fitness_marked": fitness_marked,
        "robust_fitness": robust,
        "eligible": eligible,
        "proven": proven,
        "promising": bool(eligible and profitable and not proven and robust >= _bots.PROMISING_T),
        "promising_threshold": _bots.PROMISING_T,
        "proven_threshold": threshold,
        "pl_dollar": pl_total,
        "avg_trade_return": (sum(adj) / len(adj)) if adj else None,
    })


_SB_CACHE: dict = {"ts": 0.0, "value": None}


def scoreboard_view(*, fresh: bool = False) -> list[dict]:
    """`scoreboard()` behind a 45s cache for the pages that display it. The
    rows are shared, so callers copy before annotating; breeding keeps
    calling `scoreboard()` itself for current numbers."""
    import time as _t
    if not fresh and _SB_CACHE["value"] is not None and (_t.time() - _SB_CACHE["ts"] < _COVERAGE_TTL or _stale_ok(_SB_CACHE["ts"])):
        return _SB_CACHE["value"]
    value = scoreboard()
    _SB_CACHE.update(ts=_t.time(), value=value)
    return value


def scoreboard(*, proven_only: bool = False) -> list[dict]:
    """Every pool genome with a live trial, with its pooled record, best
    first. The Shared Pool page renders this; breeding reads it."""
    out = []
    claimed = claimed_fingerprints()
    for e in entries():
        s = pooled_stats(e["fingerprint"])
        if proven_only and not s["proven"]:
            continue
        if s["bots"] == 0:
            continue
        out.append({**e, **s, "running": e["fingerprint"] in claimed,
                    "genome_desc": _genome.describe(_genome.loads(e["genome"]))})
    out.sort(key=lambda d: (d["proven"], d["robust_fitness"]), reverse=True)
    return out


def proven_by_family(k: int) -> dict:
    """{family: [pooled proven or promising genomes, best first]} — the
    breeding parents. (Proven sorts ahead of promising; scoreboard() orders
    by proven, then fitness.)

    A parent qualifies on the POOLED record, so a genome that two fleets
    each gave thirty trades is a legitimate parent even though neither bot
    reached sixty on its own. Souring genomes (negative recent fitness) are
    excluded exactly as bots.top_eligible() excludes them per bot.
    """
    out: dict = {}
    for row in scoreboard():
        if not (row["proven"] or row["promising"]):
            continue
        if row.get("fitness_recent") is not None and row["fitness_recent"] < 0:
            continue
        fam = row.get("family") or _genome.family_of(_genome.loads(row["genome"]))
        if len(out.setdefault(fam, [])) < k:
            out[fam].append(row)
    return {f: v for f, v in out.items() if v}


def family_shares() -> dict:
    """{family: share of ACTIVE fleet bots across every user}. Breeding uses
    it to stop one family taking over the whole population."""
    with _bots._conn() as c:
        rows = c.execute(
            "SELECT genome_fp, COUNT(*) n FROM bots WHERE status='active' "
            "AND COALESCE(pool,'fleet')='fleet' AND genome_fp IS NOT NULL "
            "GROUP BY genome_fp").fetchall()
    fams: dict = {}
    total = 0
    for r in rows:
        e = entry(r["genome_fp"])
        fam = (e or {}).get("family") or "?"
        fams[fam] = fams.get(fam, 0) + int(r["n"])
        total += int(r["n"])
    if not total:
        return {}
    return {f: n / total for f, n in fams.items()}


# ---------------------------------------------------------------------------
# how much the sharing is actually buying
# ---------------------------------------------------------------------------
_COVERAGE_CACHE: dict = {"ts": 0.0, "value": None}
_COVERAGE_TTL = 45.0


def coverage(*, fresh: bool = False) -> dict:
    """`_coverage()` behind a short cache: it scans every genome's record
    (seconds on a big population) and is read on every page load."""
    import time as _t
    hit = _COVERAGE_CACHE
    if not fresh and hit["value"] is not None and (_t.time() - hit["ts"] < _COVERAGE_TTL or _stale_ok(hit["ts"])):
        return hit["value"]
    value = _coverage(fresh)
    _COVERAGE_CACHE.update(ts=_t.time(), value=value)
    return value


def _coverage(fresh: bool = False) -> dict:
    """Headline numbers for the Shared Pool page: how much wider and deeper
    the search is than any one fleet's would be on its own."""
    with _bots._conn() as c:
        fleets = c.execute(
            "SELECT user_id, COUNT(*) n FROM bots WHERE status='active' "
            "AND COALESCE(pool,'fleet')='fleet' AND user_id IS NOT NULL "
            "GROUP BY user_id").fetchall()
    per_user = {int(r["user_id"]): int(r["n"]) for r in fleets}
    rows = scoreboard_view(fresh=fresh)
    tested = [r for r in rows if r["closed_count"] > 0]
    shared = [r for r in tested if r["users"] > 1]
    trades = sum(r["closed_count"] for r in tested)
    biggest = max((r["users"] for r in tested), default=0)
    return {
        "fleets": len(per_user),
        "bots": sum(per_user.values()),
        "bots_per_fleet": per_user,
        "pool_size": size(),
        "distinct_tested": distinct_tested(),
        "with_trades": len(tested),
        "proven": sum(1 for r in rows if r["proven"]),
        "shared_genomes": len(shared),
        "max_users_on_a_genome": biggest,
        "pooled_trades": trades,
        # The honest headline: the mean number of fleets contributing to a
        # tested genome's record. That IS the speed-up — a genome three
        # fleets are running reaches its 60-trade proving sample in a third
        # of the calendar days. With one user it is 1.0, and the page says
        # so rather than dressing a single fleet up as a crowd.
        # None until something has actually traded — a multiplier of "0×"
        # would be a claim about pooling, and there is nothing to pool yet
        "evidence_multiplier": (sum(r["users"] for r in tested) / len(tested)) if tested else None,
        "pending": len(pending(1000)),
    }
