"""The replay harness must drive the SAME engine code the live fleet runs
(bot_engine._step_bot via the `book` abstraction), enforce the same ledger
guardrails in memory, and only ever promote walk-forward survivors into the
live fleet. See webapp/recorder.py and webapp/replay.py.
"""

import random
from datetime import date, datetime, timedelta

import pytest

from webapp import bots, genome, option_model, recorder, replay, strategies

ET = option_model.ET
DAY1 = date(2026, 9, 1)     # a Tuesday
DAY2 = date(2026, 9, 2)


def _synth_day(day: date, start: float, drift: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    t0 = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    px = start
    out = []
    for i in range(390):
        o = px
        px = px * (1 + drift + rng.gauss(0, 0.0004))
        h, l = max(o, px) * 1.0002, min(o, px) * 0.9998
        out.append({"t": (t0 + timedelta(minutes=i)).isoformat(), "o": o, "h": h, "l": l,
                    "c": px, "v": 1000 + rng.randint(0, 500)})
    return out


@pytest.fixture(autouse=True)
def recorded_history(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "DB_PATH", tmp_path / "market_history.db")
    recorder.init()
    for i, day in enumerate((DAY1, DAY2)):
        recorder.record_bars("SPY", _synth_day(day, 560.0, 0.00015, 10 + i))   # gentle uptrend
        recorder.record_bars("QQQ", _synth_day(day, 480.0, 0.0, 20 + i))
    yield


def test_recorder_round_trip_and_session_filtering():
    assert recorder.days("SPY") == [DAY1, DAY2]
    bars = recorder.bars_for("SPY", DAY1)
    assert len(bars) == 390
    # an after-hours bar is dropped on record
    n = recorder.record_bars("SPY", [{"t": datetime(2026, 9, 1, 17, 0, tzinfo=ET).isoformat(),
                                      "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}])
    assert n == 0
    # idempotent
    assert recorder.record_bars("SPY", _synth_day(DAY1, 560.0, 0.00015, 10)) == 0
    st = recorder.status()
    assert st["SPY"]["days"] == 2 and st["QQQ"]["days"] == 2
    # prior-day levels derive from the previous recorded session when no
    # daily row was recorded (a backfilled history)
    d = recorder.daily_for("SPY", DAY2)
    assert d["prev_close"] and d["prev_high"] >= d["prev_close"] >= d["prev_low"]


def test_replay_runs_the_shared_engine_code(monkeypatch):
    g = genome.random_genome()
    g.update(style="0dte", symbol="SPY", strategy="momentum", signal_strength=0.15,
             risk_pct=8, min_option_volume=0, max_iv=0.9, stop_pct=40, take_profit_pct=60,
             trail_pct=30, hold_max_min=90, scale_in_max_adds=0, trim_frac=0.0, runner_frac=0.0,
             strike_style="delta", target_delta=0.5)
    monkeypatch.setattr(strategies, "evaluate", lambda genome, ctx: ("long", 1.0))

    res = replay.run([g], [DAY1, DAY2])[0]
    st = res["stats"]
    assert res["days"] == 2
    assert st["closed_count"] >= 1, "a forced long signal must produce trades"
    assert all(t["fees"] > 0 for t in res["trades"]), "fleet fills pay commission in replay too"
    assert all(t["side"] in bots.SIDES for t in res["trades"])
    # nothing rides past the close: every position is flat at the end of the replay
    assert st["open_count"] == 0
    assert "robust_fitness" in st and "fitness_recent" in st
    # the replay provider was torn down so the live engine isn't affected
    from webapp import option_quotes
    assert option_quotes._provider is None


def test_simbook_enforces_the_ledger_guardrails():
    book = replay.SimBook(10_000.0, {"max_drawdown_pct": 25.0, "position_pct_limit": 10.0,
                                     "min_contracts": 2, "max_contracts": 250})
    bot = book.add(genome.random_genome())
    ts = datetime(2026, 9, 1, 12, 0, tzinfo=ET).isoformat()
    contract = option_model.occ_symbol("SPY", datetime(2026, 9, 1, 16, 0, tzinfo=ET), True, 560.0)
    assert book.record_trade(bot["id"], contract, "buy_to_open", 1, 1.00, ts=ts) is not None   # min contracts
    assert "cap" in book.record_trade(bot["id"], contract, "buy_to_open", 20, 1.00, ts=ts)     # 10% cap
    assert book.record_trade(bot["id"], contract, "buy_to_open", 5, 1.00, ts=ts) is None
    assert book.positions(bot["id"])[contract]["qty"] == 5
    late = datetime(2026, 9, 1, 15, 56, tzinfo=ET).isoformat()
    assert "close-out" in book.record_trade(bot["id"], contract, "buy_to_open", 2, 1.00, ts=late)
    # a 30% realized loss trips the drawdown trip and flags the bot failing
    assert book.record_trade(bot["id"], contract, "sell_to_close", 5, 1.00, ts=ts, fees=0) is None
    for i in range(6):
        c = contract[:-4] + "{:04d}".format(i)
        book.record_trade(bot["id"], c, "buy_to_open", 5, 1.00, ts=ts, fees=0)
        book.record_trade(bot["id"], c, "sell_to_close", 5, 0.05, ts=ts, fees=0)
    assert bot["status"] == "failing"
    assert book.record_trade(bot["id"], contract, "buy_to_open", 5, 1.00, ts=ts) is not None


def test_settle_expired_flattens_at_intrinsic_after_1558():
    book = replay.SimBook(10_000.0)
    bot = book.add(genome.random_genome())
    expiry = datetime(2026, 9, 1, 16, 0, tzinfo=ET)
    contract = option_model.occ_symbol("SPY", expiry, True, 560.0)
    book.record_trade(bot["id"], contract, "buy_to_open", 2, 1.00, ts=expiry.replace(hour=12).isoformat())
    assert book.settle_expired(expiry.replace(hour=15, minute=30), {"SPY": 565.0}) == 0
    assert book.settle_expired(expiry.replace(hour=15, minute=59), {"SPY": 565.0}) == 1
    assert not book.positions(bot["id"])
    assert book.closed(bot["id"])[0]["sell_price"] == 5.0


def test_evolve_only_promotes_walk_forward_survivors(monkeypatch):
    # stub the (slow) replay with canned results: genome A is great on train,
    # terrible on validation; genome B is modest on both; C never trades
    def fake_run_parallel(genomes, days, workers=1, **kw):
        out = []
        for g in genomes:
            tag = g.get("_tag")
            if days == ["train"]:
                fit, n = {"A": 5.0, "B": 1.0, "C": 0.5}.get(tag, -1.0), 30
            else:
                fit, n = {"A": -2.0, "B": 0.8, "C": 0.5}.get(tag, -1.0), (30 if tag != "C" else 0)
            out.append({"genome": g, "stats": {"robust_fitness": fit, "closed_count": n,
                                               "pl_pct": fit, "fitness": fit,
                                               "fitness_recent": None, "max_drawdown_pct": 0,
                                               "win_rate": None}, "days": 1,
                        "closed": [], "trades": [], "status": "active"})
        return out

    monkeypatch.setattr(replay, "run_parallel", fake_run_parallel)
    monkeypatch.setattr(replay, "_save_run", lambda s: None)
    seeds = []
    for tag in ("A", "B", "C"):
        g = genome.random_genome()
        g["_tag"] = tag
        seeds.append(g)
    summary = replay.evolve(["train"], ["val"], pop=3, generations=2, seed_genomes=seeds, log=lambda *a: None)
    tags = [c["genome"].get("_tag") for c in summary["survivors"]]
    assert tags and set(tags) == {"B"}, tags   # A fails validation, C never traded out of sample (mutated B-children keep the tag)

    before = {b["id"] for b in bots.active_bots(pool="fleet")}
    ids = replay.promote(summary["survivors"], 1)
    assert len(ids) == 1
    new = [b for b in bots.active_bots(pool="fleet") if b["id"] not in before]
    assert len(new) == 1 and new[0]["origin"] == "replay"
