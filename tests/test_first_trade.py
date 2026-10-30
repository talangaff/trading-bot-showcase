"""A fresh bot must place its first trade within 24h or it is retired."""

from datetime import datetime, timedelta, timezone

from webapp import bots, genome

C = "SPY260904C00560000"


def _age(bot_id, hours):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with bots._conn() as c:
        c.execute("UPDATE bots SET created_at=? WHERE id=?", (ts, bot_id))


def _mk(pool="fleet", hours=0.0, traded=False):
    b = bots.create_bot("x", genome=genome.random_genome(), pool=pool, user_id=1)
    if traded:
        bots.record_trade(b, C, "buy_to_open", 1, 1.0, forced=True)
    _age(b, hours)
    return b


def test_old_bot_that_never_traded_is_retired():
    f, s = _mk("fleet", 25), _mk("sim", 30)
    assert bots.retire_never_traded() == {"fleet": 1, "sim": 1}
    assert bots.get_bot(f)["status"] == "retired" and bots.get_bot(s)["status"] == "retired"


def test_young_bot_gets_its_full_24_hours():
    b = _mk("sim", 23)
    assert bots.retire_never_traded() == {}
    assert bots.get_bot(b)["status"] == "active"


def test_bot_that_has_traded_even_once_is_exempt():
    b = _mk("fleet", 100, traded=True)
    assert bots.retire_never_traded() == {}
    assert bots.get_bot(b)["status"] == "active"


def test_custom_and_baseline_bots_are_never_touched():
    c, bl = _mk("custom", 100), _mk("baseline", 100)
    assert bots.retire_never_traded() == {}
    assert bots.get_bot(c)["status"] == "active" and bots.get_bot(bl)["status"] == "active"
