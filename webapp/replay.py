"""Replay harness: evaluate genomes over RECORDED sessions instead of live ones.

The live fleet can only test a genome at the cost of a real market day, one
generation per day, on a sample far too small to tell skill from luck. This
module runs the exact same entry / exit / position-management code
(bot_engine._step_bot and friends, via the `book` ledger abstraction) over
the 1-minute bars, VIX, prior-day levels and option quotes that
webapp/recorder.py has persisted, so evolution can run thousands of
genome-evaluations offline and walk-forward validate them before anything
is promoted into the live fleet.

    python -m webapp.recorder backfill --days 30          # get some history
    python -m webapp.replay backtest --bot 12 --days 20   # one live bot's genome
    python -m webapp.replay evolve --generations 20 --pop 60 --workers 4 --promote 5

Fidelity notes, so nobody over-trusts a replay number:
  * Fills use recorded real quotes where the recorder has them for that
    minute, else the modeled chain — the same fallback the live engine uses.
    Backfilled days (bars only) are therefore entirely model-priced.
  * The replay ticks once per minute; the live engine ticks four times.
  * Nothing here can see the future: each minute's feature context is built
    from bars up to that minute only, and quotes are looked up by minute.

The live fleet stays the FINAL out-of-sample test. Replay is the search.
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone

from . import (bot_engine, bots, config, features, genome as _genome, guardrails as _guardrails,
               market_calendar, option_model, option_quotes, recorder)

ET = option_model.ET
SYMBOLS = _genome.SYMBOLS
RUNS_DIR = config.DATA_DIR / "replay_runs"

# offline GA knobs
ELITE_FRAC = 0.2
IMMIGRANT_RATE = 0.15
CROSSOVER_RATE = 0.5
VALIDATE_TOP = 10           # how many train-winners get a validation run
MIN_VALIDATION_TRADES = 5   # a survivor must have actually traded out of sample


# ---------------------------------------------------------------------------
# in-memory ledger — same guardrails as bots.record_trade, no SQLite
# ---------------------------------------------------------------------------
class SimBook:
    """Implements bot_engine.LiveBook's interface for a population of
    simulated bots. Enforces the same hard rules bots.record_trade() does
    (buy-to-open/sell-to-close only, min/max contracts, position cap, the
    15:55 ET same-day cutoff, drawdown force-flatten + 'failing')."""

    def __init__(self, start_balance: float | None = None, grd: dict | None = None):
        self.start_balance = float(start_balance or bots.START_BALANCE)
        self.grd = dict(grd or _guardrails.load())
        self.bots: dict[int, dict] = {}
        self.trades: dict[int, list[dict]] = {}
        self.state: dict[int, dict[str, dict]] = {}
        self._match_cache: dict[int, tuple[int, tuple]] = {}
        self._next_trade_id = 1

    # --- population -----------------------------------------------------
    def add(self, g: dict, bot_id: int | None = None) -> dict:
        bot_id = bot_id if bot_id is not None else len(self.bots) + 1
        bot = {"id": bot_id, "name": "sim-{}".format(bot_id), "pool": "fleet",
               "status": "active", "start_balance": self.start_balance,
               "genome": _genome.dumps(g), "_genome": g,
               "created_at": None}
        self.bots[bot_id] = bot
        self.trades[bot_id] = []
        self.state[bot_id] = {}
        return bot

    # --- LiveBook interface ---------------------------------------------
    def _matched(self, bot_id):
        trades = self.trades[bot_id]
        hit = self._match_cache.get(bot_id)
        if hit and hit[0] == len(trades):
            return hit[1]
        m = bots._match(trades)
        self._match_cache[bot_id] = (len(trades), m)
        return m

    def positions(self, bot_id):
        return self._matched(bot_id)[0]

    def closed(self, bot_id):
        return self._matched(bot_id)[1]

    def pos_state(self, bot_id):
        return self.state[bot_id]

    def touch_pos_state(self, bot_id, contract, price, opened_ts, entry_iv=None, entry_mid=None):
        st = self.state[bot_id].setdefault(contract, {"hwm": price, "opened_ts": opened_ts,
                                                       "entry_iv": entry_iv, "trimmed": 0,
                                                       "core_closed": 0, "adds": 0,
                                                       "entry_mid": entry_mid})
        st["hwm"] = max(st.get("hwm") or price, price)
        st["last_mark"] = price
        if st.get("entry_iv") is None:
            st["entry_iv"] = entry_iv
        if st.get("entry_mid") is None:
            st["entry_mid"] = entry_mid

    def marks(self, bot_id) -> dict:
        return {c: st["last_mark"] for c, st in self.state[bot_id].items()
                if st.get("last_mark") is not None}

    def clear_pos_state(self, bot_id, contract):
        self.state[bot_id].pop(contract, None)

    def mark_pos(self, bot_id, contract, **flags):
        st = self.state[bot_id].get(contract)
        if st is None:
            return
        for k, v in flags.items():
            if v is not None:
                st[k] = float(v) if k == "entry_mid" else int(v)

    def equity(self, bot_id) -> dict:
        closed = self.closed(bot_id)
        points = [{"ts": None, "balance": self.start_balance}]
        bal = self.start_balance
        for r in closed:
            bal += r["pl"]
            points.append({"ts": r["ts"], "balance": bal})
        return {"balance": bal, "start": self.start_balance, "points": points, "closed": closed}

    def balance(self, bot_id) -> float:
        return self.equity(bot_id)["balance"]

    def guardrails(self, bot) -> dict:
        return self.grd

    def record_trade(self, bot_id, contract, side, qty, price, *, symbol=None, fees=0.0,
                     ts=None, note=None, forced=False, order_type="market", limit_price=None,
                     stop_price=None, trail_pct=None, fill_source=None, regime=None) -> str | None:
        bot = self.bots.get(bot_id)
        if not bot:
            return "bot not found"
        if side not in bots.SIDES:
            return "side must be buy_to_open or sell_to_close — sell-to-open is prohibited"
        qty = int(qty)
        price = float(price)
        if qty <= 0 or price <= 0:
            return "quantity and price must be positive"
        positions = self.positions(bot_id)
        lim = self.grd
        if side == "buy_to_open":
            if bot["status"] != "active":
                return "bot is flagged failing — it can only close existing positions"
            if not forced:
                try:
                    when_et = datetime.fromisoformat(ts).astimezone(ET)
                    info = option_model.parse_occ(contract)
                    expires_today = bool(info) and info["expiry"].astimezone(ET).date() == when_et.date()
                    if expires_today and (when_et.hour, when_et.minute) >= (bots.NO_BUY_AFTER_ET.hour,
                                                                            bots.NO_BUY_AFTER_ET.minute):
                        return "no new positions at/after 15:55 ET — 0DTE close-out window"
                except (ValueError, TypeError):
                    pass
                if qty < lim["min_contracts"]:
                    return "minimum entry is {} contracts".format(lim["min_contracts"])
                if qty > lim["max_contracts"]:
                    return "max {} contracts per trade".format(lim["max_contracts"])
                balance = self.balance(bot_id)
                open_cost = sum(p["cost"] for p in positions.values())
                if open_cost + qty * price * bots.CONTRACT_MULT > balance * lim["position_pct_limit"] / 100.0 + 1e-6:
                    return "position cap exceeded"
        else:
            if qty > positions.get(contract, {}).get("qty", 0):
                return "cannot close more than is open"
        self.trades[bot_id].append({
            "id": self._next_trade_id, "bot_id": bot_id, "ts": ts, "symbol": symbol,
            "contract": contract, "side": side, "qty": qty, "price": price,
            "fees": float(fees or 0.0), "order_type": order_type, "limit_price": limit_price,
            "stop_price": stop_price, "trail_pct": trail_pct, "forced": 1 if forced else 0,
            "note": note, "fill_source": fill_source, "regime": regime})
        self._next_trade_id += 1
        if side == "sell_to_close":
            self._check_drawdown(bot_id, ts)
        return None

    def _check_drawdown(self, bot_id, ts):
        bot = self.bots[bot_id]
        if bot["status"] != "active":
            return
        dd = bots._drawdown(self.equity(bot_id)["points"])
        if dd["current_pct"] + 1e-9 < self.grd["max_drawdown_pct"]:
            return
        for contract, p in list(self.positions(bot_id).items()):
            self.record_trade(bot_id, contract, "sell_to_close", int(p["qty"]), p["avg_price"],
                              ts=ts, forced=True, note="force-closed: drawdown guardrail")
            self.clear_pos_state(bot_id, contract)
        bot["status"] = "failing"

    def settle_expired(self, now: datetime, spot: dict) -> int:
        """Mirror of bot_engine._settle_expired for the sim ledger."""
        now_et = now.astimezone(ET)
        if (now_et.hour * 60 + now_et.minute) < 15 * 60 + 58:
            return 0
        n = 0
        for bot_id, bot in self.bots.items():
            for contract, p in list(self.positions(bot_id).items()):
                info = option_model.parse_occ(contract)
                if not info or info["expiry"].astimezone(ET).date() > now_et.date():
                    continue
                S = spot.get(info["underlying"])
                intrinsic = option_model.intrinsic(info, S) if S else 0.0
                err = self.record_trade(bot_id, contract, "sell_to_close", int(p["qty"]),
                                        max(0.01, round(intrinsic, 2)), symbol=info["underlying"],
                                        ts=now.isoformat(), forced=True,
                                        note="expiration settlement @ intrinsic")
                if not err:
                    self.clear_pos_state(bot_id, contract)
                    n += 1
        return n

    # --- scoring ----------------------------------------------------------
    def baseline_day_returns(self) -> dict:
        """Same shape as bots.baseline_day_returns(), from this book's own
        buy_hold reference bots (run() adds one per symbol)."""
        out: dict = {}
        for bot_id, bot in self.bots.items():
            g = bot["_genome"]
            if g.get("strategy") != "buy_hold":
                continue
            per_day: dict = {}
            for cl in self.closed(bot_id):
                d = bots._trade_date(cl)
                if d is not None:
                    per_day.setdefault(d, []).append(bots.trade_return(cl))
            out[g["symbol"]] = {d: sum(v) / len(v) for d, v in per_day.items()}
        return out

    def prior_return(self, base: dict) -> float:
        rets = []
        for bot_id, bot in self.bots.items():
            if bot["_genome"].get("strategy") in _genome.BASELINE_STRATEGIES:
                continue
            rets.extend(bots.adjusted_returns(self.closed(bot_id), base))
        return (sum(rets) / len(rets)) if rets else 0.0

    def stats(self, bot_id, base: dict | None = None, prior: float | None = None) -> dict:
        base = self.baseline_day_returns() if base is None else base
        prior = self.prior_return(base) if prior is None else prior
        is_baseline = self.bots[bot_id]["_genome"].get("strategy") in _genome.BASELINE_STRATEGIES
        return bots.stats_from_ledger(self.equity(bot_id), self.positions(bot_id),
                                      self.grd["max_drawdown_pct"], status=self.bots[bot_id]["status"],
                                      trade_count=len(self.trades[bot_id]),
                                      baseline_returns=base, prior_mean=0.0 if is_baseline else prior,
                                      marks=self.marks(bot_id), n_tested=len(self.bots))


# ---------------------------------------------------------------------------
# the session runner
# ---------------------------------------------------------------------------
class _Session:
    """One recorded day's data, pre-indexed for minute-by-minute stepping."""

    def __init__(self, day: date, symbols):
        self.day = day
        self.bars = {}
        self.minute_of = {}
        for s in symbols:
            rows = recorder.bars_for(s, day)
            self.bars[s] = rows
            self.minute_of[s] = [recorder.minute_key(r["t"]) for r in rows]
        self.daily = {s: recorder.daily_for(s, day) for s in symbols}
        self.vix = recorder.vix_series(day)
        self._vix_keys = sorted(self.vix)
        self._quote_cache: dict = {}

    def complete(self, min_bars: int = 300) -> bool:
        return all(len(b) >= min_bars for b in self.bars.values())

    def bars_upto(self, sym: str, now_key: str, window: int = 120) -> list[dict]:
        keys = self.minute_of[sym]
        # bars are sorted; find how many have key <= now_key
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] <= now_key:
                lo = mid + 1
            else:
                hi = mid
        return self.bars[sym][max(0, lo - window):lo]

    def vix_at(self, now_key: str) -> dict:
        best = None
        for k in self._vix_keys:
            if k <= now_key:
                best = k
            else:
                break
        return self.vix[best] if best else {}

    def quotes(self, symbol: str, expiry: date, now_key: str) -> dict:
        ck = (symbol, expiry, now_key)
        if ck not in self._quote_cache:
            self._quote_cache[ck] = recorder.quotes_at(symbol, expiry, now_key)
        return self._quote_cache[ck]


def _synthetic_expirations(day: date) -> list[date]:
    """SPY/QQQ list an expiry every trading day; for a replay that's exactly
    what a swing bot needs to pick a target DTE off."""
    out = []
    d = day
    for _ in range(90):
        d += timedelta(days=1)
        if market_calendar.is_market_day(d):
            out.append(d)
    return out


def run(genomes: list[dict], days: list[date], *, symbols=SYMBOLS,
        start_balance: float | None = None, grd: dict | None = None,
        progress=None) -> list[dict]:
    """Replay every genome through every recorded day, in one process.

    Returns one dict per genome: {genome, stats, closed, trades, days}."""
    book = SimBook(start_balance, grd)
    pop = [book.add(g) for g in genomes]
    # hidden buy_hold reference per symbol: every fleet trade's return is
    # day-adjusted against it, exactly as the live fleet's is against the
    # live baseline bots (see bots.adjusted_returns)
    refs = [book.add(_genome.baseline_genome("buy_hold", s)) for s in symbols]
    pop_all = pop + refs
    days_run = 0
    for day in days:
        sess = _Session(day, symbols)
        if not sess.complete():
            continue
        days_run += 1
        state = {"now_key": None}

        def quotes_fn(symbol, expiry, low, high, _s=sess, _st=state):
            q = _s.quotes(symbol, expiry, _st["now_key"])
            if q:
                # a recorded quote without the underlying's spot at its time
                # gets the bar close of its minute, so calibration is honest
                for occ, row in q.items():
                    if not row.get("spot"):
                        b = _s.bars_upto(symbol, recorder.minute_key(row.get("t")) or _st["now_key"], 1)
                        row["spot"] = b[-1]["c"] if b else None
            return q

        def exps_fn(symbol, _day=day):
            return _synthetic_expirations(_day)

        option_quotes.set_provider(quotes_fn, exps_fn)
        try:
            for now in recorder.session_minutes(day):
                now_key = recorder.minute_key(now)
                state["now_key"] = now_key
                phase = bot_engine._phase(now)
                bars = {s: sess.bars_upto(s, now_key) for s in symbols}
                if phase == "closed":
                    spot = {s: (bars[s][-1]["c"] if bars[s] else None) for s in symbols}
                    book.settle_expired(now, spot)
                    continue
                minutes_left = option_model.minutes_to_close(now)
                vix = sess.vix_at(now_key)
                feats = {}
                for s in symbols:
                    try:
                        feats[s] = features.build(s, bars.get(s) or [], bars, now,
                                                  minutes_left=minutes_left,
                                                  daily=sess.daily.get(s), vix=vix)
                    except Exception as e:  # noqa: BLE001
                        print("replay features {} {}: {}".format(s, now_key, e), flush=True)
                        feats[s] = {}
                for bot in pop_all:
                    try:
                        bot_engine._step_bot(bot, bars, feats, now, phase, book=book)
                    except Exception as e:  # noqa: BLE001
                        print("replay bot {} {}: {}".format(bot["id"], now_key, e), flush=True)
                spot = {s: (bars[s][-1]["c"] if bars[s] else None) for s in symbols}
                book.settle_expired(now, spot)
        finally:
            option_quotes.set_provider()
        if progress:
            progress(day, days_run)
    out = []
    base = book.baseline_day_returns()
    prior = book.prior_return(base)
    for bot in pop:
        out.append({"genome": bot["_genome"], "stats": book.stats(bot["id"], base, prior),
                    "closed": book.closed(bot["id"]), "trades": book.trades[bot["id"]],
                    "days": days_run, "status": bot["status"]})
    return out


def _run_chunk(args):
    genomes, days, kw = args
    random.seed(kw.pop("seed", None))
    return run(genomes, days, **kw)


def run_parallel(genomes: list[dict], days: list[date], workers: int = 1, **kw) -> list[dict]:
    """run() split across `workers` forked processes (each replays its own
    slice of the population through every day, so feature contexts are
    rebuilt per worker but wall time scales down with worker count)."""
    if workers <= 1 or len(genomes) <= 1:
        return run(genomes, days, **kw)
    workers = min(workers, len(genomes))
    chunks = [genomes[i::workers] for i in range(workers)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_run_chunk, [(c, days, dict(kw, seed=random.random())) for c in chunks])
    # re-interleave to the input order
    out = [None] * len(genomes)
    for w, part in enumerate(parts):
        for j, r in enumerate(part):
            out[w + j * workers] = r
    return out


# ---------------------------------------------------------------------------
# walk-forward evolution
# ---------------------------------------------------------------------------
def fitness_of(result: dict) -> float:
    return float(result["stats"].get("robust_fitness") or 0.0)


def _next_generation(ranked: list[dict], pop: int) -> list[dict]:
    """ranked: results best-first. Elites survive unchanged; the rest are
    mutated children of random elites, plus fresh immigrants."""
    n_elite = max(2, int(round(pop * ELITE_FRAC)))
    elites = [r["genome"] for r in ranked[:n_elite]]
    nxt = list(elites)
    while len(nxt) < pop:
        roll = random.random()
        if roll < IMMIGRANT_RATE:
            nxt.append(_genome.random_genome())
        elif roll < IMMIGRANT_RATE + (1 - IMMIGRANT_RATE) * CROSSOVER_RATE and len(elites) > 1:
            a, b = random.sample(elites, 2)
            nxt.append(_genome.mutate(_genome.crossover(a, b), rate=0.15))
        else:
            nxt.append(_genome.mutate(random.choice(elites)))
    return nxt


def evolve(train_days: list[date], validate_days: list[date], *, pop: int = 60,
           generations: int = 20, workers: int = 1, seed_genomes: list[dict] | None = None,
           log=print) -> dict:
    """Offline GA on `train_days`, then a single validation pass of the
    top VALIDATE_TOP train-winners over `validate_days` they never saw.

    Returns {"survivors": [...], "candidates": [...], ...}. A survivor is a
    genome whose VALIDATION fitness is positive with at least
    MIN_VALIDATION_TRADES closed trades — the only thing promote() will
    ever seed into the live fleet."""
    t0 = time.time()
    population = list(seed_genomes or [])[:pop]
    while len(population) < pop:
        population.append(_genome.random_genome())

    hall: dict[str, dict] = {}       # genome json -> best result on train
    history = []
    for gen in range(generations):
        results = run_parallel(population, train_days, workers=workers)
        ranked = sorted(results, key=fitness_of, reverse=True)
        for r in ranked:
            key = _genome.dumps(r["genome"])
            if key not in hall or fitness_of(r) > fitness_of(hall[key]):
                hall[key] = r
        best = ranked[0]
        history.append({"gen": gen, "best_fitness": fitness_of(best),
                        "best_trades": best["stats"]["closed_count"],
                        "mean_fitness": sum(fitness_of(r) for r in ranked) / len(ranked)})
        log("gen {:>3}  best {:+.3f} ({} trades, {:+.1f}%)  mean {:+.3f}  [{:.0f}s]".format(
            gen, fitness_of(best), best["stats"]["closed_count"], best["stats"]["pl_pct"] or 0.0,
            history[-1]["mean_fitness"], time.time() - t0))
        if gen < generations - 1:
            population = _next_generation(ranked, pop)

    # walk-forward validation of the best distinct genomes seen on train
    top = sorted(hall.values(), key=fitness_of, reverse=True)[:VALIDATE_TOP]
    top = [r for r in top if fitness_of(r) > 0]
    candidates = []
    if top and validate_days:
        val = run_parallel([r["genome"] for r in top], validate_days, workers=workers)
        for tr, vr in zip(top, val):
            candidates.append({
                "genome": tr["genome"],
                "train": {"fitness": fitness_of(tr), "trades": tr["stats"]["closed_count"],
                          "pl_pct": tr["stats"]["pl_pct"]},
                "validation": {"fitness": fitness_of(vr), "trades": vr["stats"]["closed_count"],
                               "pl_pct": vr["stats"]["pl_pct"]},
            })
    survivors = [c for c in candidates
                 if c["validation"]["fitness"] > 0
                 and c["validation"]["trades"] >= MIN_VALIDATION_TRADES]
    survivors.sort(key=lambda c: c["validation"]["fitness"], reverse=True)
    summary = {
        "train_days": [str(d) for d in train_days],
        "validate_days": [str(d) for d in validate_days],
        "pop": pop, "generations": generations, "history": history,
        "candidates": candidates, "survivors": survivors,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _save_run(summary)
    return summary


def _save_run(summary: dict) -> None:
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (RUNS_DIR / "{}.json".format(stamp)).write_text(json.dumps(summary, indent=1, default=str))
        bots.init()
        bots.log_event("replay", {
            "generations": summary["generations"], "pop": summary["pop"],
            "train_days": len(summary["train_days"]), "validate_days": len(summary["validate_days"]),
            "candidates": len(summary["candidates"]), "survivors": len(summary["survivors"]),
            "best_validation": (summary["survivors"][0]["validation"]["fitness"]
                                if summary["survivors"] else None),
            "elapsed_s": summary["elapsed_s"]})
    except Exception as e:  # noqa: BLE001
        print("replay save: {}".format(e), flush=True)


def promote(survivors: list[dict], n: int, user_id: int | None = None) -> list[int]:
    """Seed the top `n` walk-forward survivors into a live fleet as fresh
    bots (origin='replay'). They start with an empty live ledger and must
    prove themselves live like anyone else — replay earns a seat, not a
    breeding slot.

    Each survivor is registered in the SHARED pool on the way in, so the
    genome offline evolution found is a candidate the whole population
    knows about: whichever fleet runs it, its live trades accumulate
    against the same record, and any other user can clone it from /pool.
    Without `user_id` the survivors go to the primary admin's fleet.
    """
    from . import auth
    owner = int(user_id) if user_id is not None else auth.primary_admin_id()
    bots.init(default_owner=owner)
    ids = []
    for c in survivors[:n]:
        ids.append(bots.spawn_bot(c["genome"], origin="replay", user_id=owner))
    if ids:
        bots.log_event("promote", {"spawned": len(ids), "ids": ids, "user_id": owner})
    return ids


# ---------------------------------------------------------------------------
def _recorded_days(start: date | None, end: date | None) -> list[date]:
    per = [set(recorder.days(s, start, end)) for s in SYMBOLS]
    return sorted(set.intersection(*per)) if per else []


def _parse_range(text: str) -> tuple[date, date]:
    a, b = text.split(":")
    return date.fromisoformat(a), date.fromisoformat(b)


def _print_result(r: dict) -> None:
    st = r["stats"]
    print("{:>4} closed  P&L {:+.2f}%  maxDD {:.1f}%  win {}  fitness {:+.3f} (recent {})  over {} days"
          .format(st["closed_count"], st["pl_pct"] or 0.0, st["max_drawdown_pct"],
                  "{:.0f}%".format(st["win_rate"]) if st["win_rate"] is not None else "—",
                  st["fitness"], st["fitness_recent"], r["days"]), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m webapp.replay")
    sub = ap.add_subparsers(dest="cmd")

    bt = sub.add_parser("backtest", help="replay one genome over recorded days")
    bt.add_argument("--bot", type=int, help="live bot id whose genome to replay")
    bt.add_argument("--genome", help="genome JSON string")
    bt.add_argument("--days", type=int, default=20, help="last N recorded days")
    bt.add_argument("--range", help="YYYY-MM-DD:YYYY-MM-DD instead of --days")

    ev = sub.add_parser("evolve", help="offline GA with walk-forward validation")
    ev.add_argument("--train", help="YYYY-MM-DD:YYYY-MM-DD (default: first 75%% of recorded days)")
    ev.add_argument("--validate", help="YYYY-MM-DD:YYYY-MM-DD (default: last 25%%)")
    ev.add_argument("--pop", type=int, default=60)
    ev.add_argument("--generations", type=int, default=20)
    ev.add_argument("--workers", type=int, default=1)
    ev.add_argument("--seed-fleet", action="store_true", help="seed with the live fleet's current genomes")
    ev.add_argument("--promote", type=int, default=0, help="spawn the top N validated survivors into a live fleet")
    ev.add_argument("--promote-to", type=int, default=None, metavar="USER_ID",
                    help="which account's fleet the promoted survivors join (default: the primary admin)")

    sub.add_parser("status", help="recorded history available to replay")
    args = ap.parse_args(argv)

    if args.cmd == "status" or not args.cmd:
        print(recorder.status(), flush=True)
        print("replayable days (all symbols):", len(_recorded_days(None, None)), flush=True)
        return 0

    if args.cmd == "backtest":
        if args.range:
            a, b = _parse_range(args.range)
            days = _recorded_days(a, b)
        else:
            days = _recorded_days(None, None)[-args.days:]
        if not days:
            print("no recorded days — run `python -m webapp.recorder backfill` first", flush=True)
            return 1
        if args.bot:
            bots.init()
            b = bots.get_bot(args.bot)
            if not b:
                print("no bot {}".format(args.bot), flush=True)
                return 1
            g = bots.genome_of(b)
        elif args.genome:
            g = json.loads(args.genome)
            _genome._repair(g)
        else:
            g = _genome.random_genome()
            print("(no --bot/--genome given: replaying a random genome)", flush=True)
        print(_genome.describe(g), flush=True)
        print("{} days: {} .. {}".format(len(days), days[0], days[-1]), flush=True)
        _print_result(run([g], days)[0])
        return 0

    if args.cmd == "evolve":
        all_days = _recorded_days(None, None)
        if args.train and args.validate:
            train = _recorded_days(*_parse_range(args.train))
            validate = _recorded_days(*_parse_range(args.validate))
        else:
            cut = max(1, int(len(all_days) * 0.75))
            train, validate = all_days[:cut], all_days[cut:]
        if not train or not validate:
            print("need recorded days for both train and validation (have {})".format(len(all_days)), flush=True)
            return 1
        seeds = None
        if args.seed_fleet:
            bots.init()
            seeds = [bots.genome_of(b) for b in bots.active_bots(pool="fleet")]
        print("train {} days {}..{}, validate {} days {}..{}".format(
            len(train), train[0], train[-1], len(validate), validate[0], validate[-1]), flush=True)
        summary = evolve(train, validate, pop=args.pop, generations=args.generations,
                         workers=args.workers, seed_genomes=seeds)
        print("\ncandidates (train -> validation):", flush=True)
        for c in summary["candidates"]:
            print("  {:+.3f} ({:>3} tr) -> {:+.3f} ({:>3} tr, {:+.1f}%)  {}".format(
                c["train"]["fitness"], c["train"]["trades"], c["validation"]["fitness"],
                c["validation"]["trades"], c["validation"]["pl_pct"] or 0.0,
                _genome.describe(c["genome"])), flush=True)
        print("survivors:", len(summary["survivors"]), flush=True)
        if args.promote and summary["survivors"]:
            ids = promote(summary["survivors"], args.promote, args.promote_to)
            print("promoted into the live fleet as bots", ids, flush=True)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
