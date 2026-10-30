"""A dead man's switch for the trading engine.

The engine already shouts when a cycle runs and moves nothing. That alarm
has one fatal blind spot: it is code inside the cycle, so it cannot fire
when the cycle is not running at all. A disabled timer, a service that
crash-loops on an import error, a box that hangs — every one of those is
silence, and silence is exactly what the 2026-09-09 outage looked like for
three and a half hours.

So the check lives somewhere the engine cannot take with it. The engine
runs as short-lived oneshot processes fired by a systemd timer once a
minute; this runs in the web app, which is a long-lived service with
Restart=on-failure. Each cycle stamps a heartbeat into the database, and
this thread reads it from the other side. Neither process can mask the
other's death, which is the entire point — a watchdog sharing a lifetime
with the thing it watches is decoration.

What is still NOT covered, and cannot be from in here: the whole box going
down, or Telegram being unreachable. Nothing running on the machine can
report that the machine is gone. That needs an external uptime check.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

from . import bots, notify

HEARTBEAT_KEY = "engine_last_cycle"
STEPPED_KEY = "engine_last_stepped"
DETAIL_KEY = "engine_last_detail"

# Cycles fire once a minute and finish in well under one. Five minutes of
# nothing is not a slow cycle, it is a stopped one — long enough that a
# single overrunning cycle never trips it, short enough to catch a real
# halt inside a coffee break rather than an afternoon.
STALE_S = 300.0
CHECK_EVERY_S = 60.0
ALERT_COOLDOWN_S = 900.0


def beat(stepped: int, phase: str, detail: str = "") -> None:
    """Record that a cycle completed. Called from run_cycle's finally, so a
    cycle that throws half way still counts as the process having run —
    this is a liveness signal, not a success signal, and conflating the two
    would mean a crashing engine looked identical to a dead one."""
    try:
        bots._meta_set(HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        bots._meta_set(STEPPED_KEY, int(stepped))
        bots._meta_set(DETAIL_KEY, "{}|{}".format(phase, detail)[:200])
    except Exception as e:  # noqa: BLE001 — a heartbeat must never break a cycle
        print("watchdog beat failed: {}".format(e), flush=True)


def last_beat() -> datetime | None:
    raw = bots._meta_get(HEARTBEAT_KEY)
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(raw)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def check(now: datetime | None = None) -> str | None:
    """The problem with the engine right now, or None if it is fine.

    Only speaks during market hours and only when the engine is switched
    on: an engine someone deliberately turned off is not an outage, and an
    alarm that cries wolf on a Sunday teaches its reader to ignore it."""
    from . import bot_engine, bot_settings          # lazy: bot_engine imports notify

    now = now or datetime.now(bot_engine.ET)
    if bot_engine._phase(now) == "closed":
        return None
    try:
        if not bot_settings.engine_enabled():
            return None
    except Exception:  # noqa: BLE001
        pass

    beat_at = last_beat()
    if beat_at is None:
        return "the engine has never recorded a cycle"
    age = (datetime.now(timezone.utc) - beat_at).total_seconds()
    if age > STALE_S:
        return ("the engine has not completed a cycle in {:.0f} minutes "
                "(last was {} UTC)".format(age / 60.0, beat_at.strftime("%H:%M")))
    return None


def _alert(problem: str) -> None:
    fresh = bots.log_event_once("engine_down", {"problem": problem},
                                key="down", within_s=ALERT_COOLDOWN_S)
    print("ENGINE DOWN: {}".format(problem), flush=True)
    if fresh and notify.enabled():
        notify.send(
            "\U0001f6d1 Bot fleet engine is not running\n\n"
            "{}.\n\n"
            "Nothing is being traded and no open position is being managed — "
            "stops, trails and take-profits all run from the cycle.\n\n"
            "Check: systemctl status trade-calculator-bots.timer".format(problem),
            buttons=[notify._button("Open the fleet", "/bots")])


def _loop() -> None:
    while True:
        try:
            problem = check()
            if problem:
                _alert(problem)
        except Exception as e:  # noqa: BLE001 — the watchdog outlives its own bugs
            print("watchdog check failed: {}".format(e), flush=True)
        time.sleep(CHECK_EVERY_S)


_started = False
_LOCK = threading.Lock()


def start() -> bool:
    """Start the watcher once per process. Returns whether it started here."""
    global _started
    with _LOCK:
        if _started:
            return False
        _started = True
    threading.Thread(target=_loop, name="engine-watchdog", daemon=True).start()
    print("engine watchdog started (stale after {:.0f}s)".format(STALE_S), flush=True)
    return True
