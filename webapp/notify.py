"""Telegram notifications for calendar-spread entry candidates.

"Buy today" (enter near today's close, T-1) means, under the BMO/AMC rule:
  * Recommended for tomorrow  AND earnings BMO  (report is before tomorrow's open)
  * Recommended for today     AND earnings AMC  (report is after today's close)

Sends exactly two Telegram messages a day, nothing in between:
  * a market-open summary of all buys for today
  * a 15:00 ET "buy before the close" summary of the same

Each summary also flags anything Recommended today/tomorrow that the Nasdaq
calendar didn't tag with a BMO/AMC time (fix it on the This Week page).

Credentials come from the environment (never stored in the repo):
  TELEGRAM_BOT_TOKEN   TELEGRAM_CHAT_ID
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from . import config, week

ET = ZoneInfo("America/New_York")
STATE_FILE = config.DATA_DIR / "notify_state.json"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SITE_URL = os.environ.get("TRADE_PUBLIC_URL", "http://localhost:5005").rstrip("/")


def enabled() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)


def _button(text: str, path: str) -> list:
    """One inline-keyboard row with a single URL button."""
    return [{"text": text, "url": SITE_URL + path}]


def send(text: str, buttons: list | None = None) -> bool:
    """buttons: list of rows, each row a list of {"text","url"} dicts."""
    if not enabled():
        print("[notify] Telegram not configured — would send:\n" + text, flush=True)
        return False
    payload = {"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": "true"}
    if buttons:
        payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})
    # Telegram answers in well under a second almost always, but a read can
    # stall for 15s+ while the engine is busy; one slow call used to drop the
    # message for good. Retry a couple of times before giving up.
    last = None
    for attempt, pause in enumerate((0, 2, 6), start=1):
        if pause:
            time.sleep(pause)
        try:
            r = requests.post(
                "https://api.telegram.org/bot{}/sendMessage".format(BOT_TOKEN),
                data=payload, timeout=(5, 25),
            )
            if r.ok:
                return True
            last = "Telegram {}: {}".format(r.status_code, r.text[:200])
            retry = r.status_code in (429, 500, 502, 503, 504)
        except Exception as e:  # noqa: BLE001
            last, retry = str(e), True
        print("[notify] send failed (attempt {}): {}".format(attempt, last), flush=True)
        if not retry:
            break                                      # a 4xx will not get better
    return False


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(st: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    os.replace(tmp, STATE_FILE)


def _roll_day(st: dict, today: str) -> dict:
    if st.get("snapshot_date") != today:
        return {
            "snapshot_date": today,
            "tracked": {},
            "notime_alerted": [],
            "open_sent_date": st.get("open_sent_date"),
        }
    st.setdefault("tracked", {})
    st.setdefault("notime_alerted", [])
    return st


# ---------------------------------------------------------------------------
# candidate detection
# ---------------------------------------------------------------------------
def _next_trading_day(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _in_trading_window(lo=(8, 0), hi=(16, 30)) -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return lo <= (now.hour, now.minute) <= hi


def tracked_candidates():
    """(tracked, no_time)

    tracked: {SYM: {rating, session, label, day}} — every symbol whose
             (day, session) makes it a today-entry candidate under BMO/AMC.
    no_time: {SYM: {label, day}} — Recommended today/tomorrow, no BMO/AMC tag.
    """
    today = datetime.now(ET).date()
    tomo = _next_trading_day(today)
    cache = week.load_cache()

    tracked, no_time = {}, {}
    plan = (
        (today.strftime("%Y-%m-%d"), "AMC", "tonight (AMC)"),
        (tomo.strftime("%Y-%m-%d"), "BMO", "tomorrow pre-market (BMO)"),
    )
    for day_str, want, label in plan:
        entry = cache.get(day_str)
        if not entry:
            continue
        rows = week.prepare_rows(day_str, entry.get("rows") or [])
        for r in rows:
            sym, sess, rec = r["symbol"], r.get("session", "-"), r["rating"] == "Recommended"
            if sess == want:
                tracked[sym] = {"rating": r["rating"], "session": sess,
                                "label": label, "day": day_str}
            elif rec and sess not in ("BMO", "AMC"):
                no_time[sym] = {"label": label, "day": day_str}
    return tracked, no_time


def buys_for_today(tracked=None) -> dict:
    tracked = tracked if tracked is not None else tracked_candidates()[0]
    return {s: v for s, v in tracked.items() if v["rating"] == "Recommended"}


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
def after_refresh() -> None:
    """Called after every This Week refresh.

    Intraday flip / no-time alerts are intentionally disabled — the only
    Telegram messages are the two daily summaries (market open and 15:00 ET).
    We still refresh the tracked snapshot so those summaries stay accurate.
    """
    today = datetime.now(ET).strftime("%Y-%m-%d")
    st = _roll_day(_load_state(), today)
    tracked, _ = tracked_candidates()
    st["tracked"] = tracked
    _save_state(st)


def market_open_message() -> None:
    """Called at the open — summarise every buy for today."""
    now = datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    st = _roll_day(_load_state(), today)

    tracked, no_time = tracked_candidates()
    buys = buys_for_today(tracked)
    header = now.strftime("%a %b %-d")

    if buys:
        lines = ["\U0001f514 Buys for today ({})".format(header), "",
                 "Enter near the close:"]
        for s in sorted(buys):
            lines.append("• {} — earnings {}".format(s, buys[s]["label"]))
        lines += ["", "{} to enter.".format(len(buys))]
        send("\n".join(lines), buttons=[_button("Open This Week", "/week")])
    else:
        send("\U0001f514 No buys for today ({}).".format(header))

    for sym, info in no_time.items():
        key = "{}@{}".format(sym, info["day"])
        if key not in st["notime_alerted"]:
            send("❓ {} is Recommended (earnings {}) but has no BMO/AMC time.".format(
                sym, info["label"]),
                buttons=[_button("Set {} to BMO/AMC".format(sym),
                                 "/week?day=" + info["day"])])
            st["notime_alerted"].append(key)

    st["tracked"] = tracked
    st["open_sent_date"] = today
    _save_state(st)


def afternoon_message() -> None:
    """Called at 15:00 ET — the "buy before the close" list.

    Same BMO/AMC rule as the open summary: a symbol is a buy for today when it
    is Recommended AND either
      * dated today with earnings AMC (reports after today's close), or
      * dated the next market day with earnings BMO (reports before that open).
    Only Recommended symbols are sent; nothing else.
    """
    now = datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    st = _roll_day(_load_state(), today)

    tracked, no_time = tracked_candidates()
    buys = buys_for_today(tracked)
    header = now.strftime("%a %b %-d")

    if buys:
        lines = ["\U0001f6d2 Buy before the close ({})".format(header), "",
                 "These meet the criteria and are Recommended:"]
        for s in sorted(buys):
            lines.append("• {} — earnings {}".format(s, buys[s]["label"]))
        lines += ["", "{} to enter near the close.".format(len(buys))]
        send("\n".join(lines), buttons=[_button("Open This Week", "/week")])
    else:
        send("\U0001f6d2 Nothing to buy before the close ({}).".format(header))

    for sym, info in no_time.items():
        send("❓ {} is Recommended (earnings {}) but has no BMO/AMC time.".format(
            sym, info["label"]),
            buttons=[_button("Set {} to BMO/AMC".format(sym),
                             "/week?day=" + info["day"])])

    st["tracked"] = tracked
    _save_state(st)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "check"
    if cmd == "open":
        market_open_message()
    elif cmd == "afternoon":
        afternoon_message()
    elif cmd == "check":
        after_refresh()
    elif cmd == "test":
        send("✅ Trade notifications are wired up.")
    else:
        print("usage: python -m webapp.notify [open|afternoon|check|test]", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
