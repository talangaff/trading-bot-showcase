"""Per-account fleet settings (Alpaca keys, data feed, execution mode), plus
each account's custom-bot settings.

Stored as JSON in the mutable data dir (git-ignored), like auth.json.

  * user_bot_settings.json — per account, keyed by user id:
      - the account's own FLEET: the Alpaca paper keys that user's fleet
        and baseline bots place their REAL orders with, its data feed and
        its engine on/off switch;
      - the account's CUSTOM bots: their own Alpaca paper keys and the
        capital pool they are allocated from.
  * bot_settings.json      — the house defaults (admin only): the master
        engine switch for the whole installation, and the legacy
        single-fleet keys, which are migrated onto the primary admin's own
        fleet on first start (migrate_fleet_to) and not used after that.

Every user runs their own fleet on their own account, so one user's bots
can never trade on another user's money, and a user without keys simply
does not trade. The genomes those fleets run are shared
(webapp/gene_pool.py) — the money never is.

A user's fleet keys and their custom-bot keys are still deliberately
separate accounts: the fleet trades as a crowd, and a custom bot is for
testing ONE genome on its own. A custom bot never falls back to the fleet
keys — no keys of its own means it does not trade, for anyone, admins
included.

There is no simulated execution mode. Every bot on every fleet sends real
orders to its owner's Alpaca paper account and books only the fill Alpaca
reports, so the fitness evolution breeds and culls on is built from fills
that really happened. A fleet with no keys does not trade — it never
trades modeled ones instead. (Offline evaluation still exists, and is what
modeled fills are for: webapp/replay.py, on its own in-memory ledger.)

Market DATA is the one thing that is not per-user: see data_source(). The
best entitlement anyone has connected serves the whole installation, and
everything it pulls is recorded once into the shared history every user's
replay and evolution runs on.
"""

import json
import os

from . import config

_PATH = config.DATA_DIR / "bot_settings.json"
_FIELDS = ("alpaca_key_id", "alpaca_secret_key", "data_feed", "engine_enabled",
          "custom_pool_total", "fleet_migrated_to", "simulated_retired",
          "live_enabled")
DATA_FEEDS = ("iex", "sip")

_USER_PATH = config.DATA_DIR / "user_bot_settings.json"
# per account: custom-bot keys + pool, and the account's own fleet settings
_USER_FIELDS = ("alpaca_key_id", "alpaca_secret_key", "custom_pool_total",
                "fleet_alpaca_key_id", "fleet_alpaca_secret_key",
                "fleet_data_feed", "fleet_engine_enabled")
_USER_SECRETS = ("alpaca_secret_key", "fleet_alpaca_secret_key")


def _write_json(path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _USER_CACHE.clear()


def load() -> dict:
    try:
        d = json.loads(_PATH.read_text())
    except (OSError, ValueError):
        d = {}
    out = {k: d.get(k) for k in _FIELDS}
    out["data_feed"] = out.get("data_feed") or "iex"
    # absent (a file that has never been written) means ON — the master
    # switch exists to stop the engine deliberately, not to leave a fresh
    # install silently dead. Nothing trades until a user connects keys.
    out["engine_enabled"] = True if out.get("engine_enabled") is None else bool(out["engine_enabled"])
    # live trading is OFF until deliberately switched on, and is the kill switch
    out["live_enabled"] = bool(out.get("live_enabled"))
    try:
        out["custom_pool_total"] = float(out.get("custom_pool_total") or 0.0)
    except (TypeError, ValueError):
        out["custom_pool_total"] = 0.0
    return out


def engine_enabled() -> bool:
    """The installation-wide master switch (admin, Guardrails page). Off
    means no fleet trades, whatever any individual user has set."""
    return bool(load().get("engine_enabled", True))


def save(**changes) -> dict:
    d = load()
    for k, v in changes.items():
        if k not in _FIELDS:
            continue
        if isinstance(v, str):
            v = v.strip()
        # blank string leaves an existing secret untouched
        if k == "alpaca_secret_key" and v == "":
            continue
        d[k] = v
    _write_json(_PATH, d)
    return load()


def public() -> dict:
    """Safe-to-render view of the house settings. Fleet keys live per user
    now (public_fleet), so nothing secret is left in here."""
    d = load()
    return {
        "engine_enabled": bool(d.get("engine_enabled", True)),
        "fleet_migrated_to": d.get("fleet_migrated_to"),
    }


# ---------------------------------------------------------------------------
# per-account custom-bot settings
# ---------------------------------------------------------------------------
# Every bot's execution routing goes through load_user() now (broker.execution
# is called per bot, several times per cycle, and once per row on the fleet
# page), so re-reading the JSON each time would be hundreds of file reads a
# cycle. Cached on the file's identity, invalidated by any write here — and by
# a write from another process, which changes the mtime/size.
_USER_CACHE: dict = {}


def _load_all_users() -> dict:
    try:
        st = _USER_PATH.stat()
        stamp = (str(_USER_PATH), st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    hit = _USER_CACHE.get("v")
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        d = json.loads(_USER_PATH.read_text())
    except (OSError, ValueError):
        d = {}
    d = d if isinstance(d, dict) else {}
    _USER_CACHE["v"] = (stamp, d)
    return d


def load_user(user_id: int) -> dict:
    d = _load_all_users().get(str(int(user_id))) or {}
    out = {k: d.get(k) for k in _USER_FIELDS}
    try:
        out["custom_pool_total"] = float(out.get("custom_pool_total") or 0.0)
    except (TypeError, ValueError):
        out["custom_pool_total"] = 0.0
    out["fleet_data_feed"] = (out.get("fleet_data_feed")
                              if out.get("fleet_data_feed") in DATA_FEEDS else "iex")
    # a fleet a user has never touched is ON: connecting keys is the
    # deliberate act, and the switch exists to pause a running fleet
    out["fleet_engine_enabled"] = (True if out.get("fleet_engine_enabled") is None
                                   else bool(out["fleet_engine_enabled"]))
    return out


def save_user(user_id: int, **changes) -> dict:
    allu = _load_all_users()
    d = allu.get(str(int(user_id))) or {}
    for k, v in changes.items():
        if k not in _USER_FIELDS:
            continue
        if isinstance(v, str):
            v = v.strip()
        if k in _USER_SECRETS and v == "":
            continue        # blank leaves an existing secret untouched
        d[k] = v
    allu[str(int(user_id))] = d
    _write_json(_USER_PATH, allu)
    return load_user(user_id)


def clear_user_keys(user_id: int) -> None:
    save_user(user_id, alpaca_key_id="", alpaca_secret_key=None)


def delete_user(user_id: int) -> None:
    allu = _load_all_users()
    if allu.pop(str(int(user_id)), None) is not None:
        _write_json(_USER_PATH, allu)


def migrate_pool_to(user_id: int) -> None:
    """One-time upgrade: the fleet-wide custom-bot capital pool from the
    single-admin build becomes the primary admin's own pool."""
    d = load()
    total = d.get("custom_pool_total") or 0.0
    if total <= 0:
        return
    allu = _load_all_users()
    mine = allu.get(str(int(user_id))) or {}
    if not mine.get("custom_pool_total"):
        mine["custom_pool_total"] = total
        allu[str(int(user_id))] = mine
        _write_json(_USER_PATH, allu)
    d["custom_pool_total"] = 0.0
    _write_json(_PATH, d)


def execution_keys(user_id: int | None) -> tuple[str, str] | None:
    """(key_id, secret) a given account's custom bots trade with, or None.

    Only the account's OWN keys, ever — a custom bot exists to test one
    genome by itself on a separate Alpaca account, so it never falls back
    to the fleet keys (that would drop it into the same account the whole
    fleet is trading on, admin or not)."""
    if user_id is None:
        return None
    u = load_user(user_id)
    if u.get("alpaca_key_id") and u.get("alpaca_secret_key"):
        return u["alpaca_key_id"], u["alpaca_secret_key"]
    return None


def same_as_fleet(user_id: int) -> bool:
    """True if this account's custom-bot key ID is the same account its own
    fleet trades on — its custom bots would then be sharing an account with
    sixty fleet bots, which defeats testing a genome on its own. Surfaced
    as a warning, never blocked."""
    u = load_user(user_id)
    return bool(u.get("alpaca_key_id")) and u.get("alpaca_key_id") == u.get("fleet_alpaca_key_id")


# ---------------------------------------------------------------------------
# per-account FLEET settings — every user runs their own fleet
# ---------------------------------------------------------------------------
def fleet_keys(user_id: int | None) -> tuple[str, str] | None:
    """(key_id, secret) this account's fleet and baseline bots trade with.

    Only that account's own keys, ever. There is no shared fallback: a
    fleet with no keys does not trade, which is the only way one user's
    bots can be guaranteed never to reach another user's money."""
    if user_id is None:
        return None
    u = load_user(user_id)
    if u.get("fleet_alpaca_key_id") and u.get("fleet_alpaca_secret_key"):
        return u["fleet_alpaca_key_id"], u["fleet_alpaca_secret_key"]
    return None


def fleet_engine_enabled(user_id: int | None) -> bool:
    """This account's own engine switch AND the installation master switch —
    an admin turning the engine off stops every fleet."""
    if user_id is None or not engine_enabled():
        return False
    return bool(load_user(user_id)["fleet_engine_enabled"])


def clear_fleet_keys(user_id: int) -> None:
    save_user(user_id, fleet_alpaca_key_id="", fleet_alpaca_secret_key=None)


def fleet_user_ids() -> list[int]:
    """Every account with a fleet the engine can actually run: keys saved.

    Keys are the whole requirement now — there is no modeled mode to run
    without them, so an account with none has nothing to tick."""
    out = []
    for uid in sorted(int(k) for k in _load_all_users() if str(k).isdigit()):
        u = load_user(uid)
        if u.get("fleet_alpaca_key_id") and u.get("fleet_alpaca_secret_key"):
            out.append(uid)
    return out


def data_source() -> dict:
    """The keys and feed the whole installation pulls MARKET DATA with.

    Data is the one thing nobody owns privately here: bars, the VIX
    snapshot and option chains are the same for everyone, they are fetched
    once per cycle, and everything fetched is recorded into the shared
    history (webapp/recorder.py) that every user's replay and evolution
    runs on. So we use the best entitlement anyone has connected — a single
    user with a SIP subscription upgrades the data quality, and therefore
    the fill realism, of every other user's fleet.

    Returns {"user_id", "keys", "feed"}; keys is None when nobody has
    connected any, and the engine falls back to yfinance as it always did.
    """
    best = {"user_id": None, "keys": None, "feed": "iex"}
    for uid in sorted(int(k) for k in _load_all_users() if str(k).isdigit()):
        u = load_user(uid)
        key, sec = u.get("fleet_alpaca_key_id"), u.get("fleet_alpaca_secret_key")
        if not (key and sec):
            key, sec = u.get("alpaca_key_id"), u.get("alpaca_secret_key")
            if not (key and sec):
                continue
        feed = u["fleet_data_feed"]
        if best["keys"] is None or (feed == "sip" and best["feed"] != "sip"):
            best = {"user_id": uid, "keys": (key, sec), "feed": feed}
    if best["keys"] is None:                      # legacy pre-migration install
        d = load()
        if d.get("alpaca_key_id") and d.get("alpaca_secret_key"):
            best = {"user_id": None,
                    "keys": (d["alpaca_key_id"], d["alpaca_secret_key"]),
                    "feed": d.get("data_feed") or "iex"}
    return best


def migrate_fleet_to(user_id: int) -> bool:
    """One-time upgrade from the single shared fleet: the house Alpaca keys,
    data feed and execution mode become the primary admin's OWN fleet
    settings, so the fleet that was already running keeps trading on the
    same account without anything being re-entered."""
    d = load()
    if d.get("fleet_migrated_to"):
        return False
    if not (d.get("alpaca_key_id") and d.get("alpaca_secret_key")):
        # nothing to carry over, but record that the migration has happened
        # so a later key entry on the house file is never picked up again
        d["fleet_migrated_to"] = int(user_id)
        _write_json(_PATH, d)
        return False
    save_user(user_id,
              fleet_alpaca_key_id=d.get("alpaca_key_id"),
              fleet_alpaca_secret_key=d.get("alpaca_secret_key"),
              fleet_data_feed=d.get("data_feed") or "iex",
              fleet_engine_enabled=bool(d.get("engine_enabled", True)))
    d["fleet_migrated_to"] = int(user_id)
    _write_json(_PATH, d)
    return True


def retire_simulated_fleets() -> list[int]:
    """One-time upgrade off the removed "simulated" execution mode.

    Returns the accounts that had it set, and strips the dead field. Those
    fleets now place REAL orders on their Alpaca account (or, with no keys
    saved, stop trading entirely). Worth surfacing rather than doing
    silently: their existing ledgers were built from modeled fills, so
    every number derived from them — balance, fitness, whether a genome
    looks proven — now mixes two different fill rules. A fleet reset gives
    the current rules a clean cohort.

    Idempotent: the house file records that it has run."""
    d = load()
    if d.get("simulated_retired"):
        return []
    allu = _load_all_users()
    affected = []
    for k, u in allu.items():
        if not str(k).isdigit():
            continue
        if u.pop("fleet_execution_mode", None) == "simulated":
            affected.append(int(k))
    u_house = d.pop("execution_mode", None)
    if affected or u_house:
        _write_json(_USER_PATH, allu)
    d["simulated_retired"] = True
    _write_json(_PATH, d)
    return sorted(affected)


def public_fleet(user_id: int) -> dict:
    """Safe-to-render view of one account's fleet settings."""
    u = load_user(user_id)
    connected = bool(u.get("fleet_alpaca_key_id") and u.get("fleet_alpaca_secret_key"))
    return {
        "alpaca_key_id": u.get("fleet_alpaca_key_id") or "",
        "has_secret": bool(u.get("fleet_alpaca_secret_key")),
        "data_feed": u["fleet_data_feed"],
        "engine_enabled": bool(u["fleet_engine_enabled"]),
        "house_engine_enabled": engine_enabled(),
        "running": bool(u["fleet_engine_enabled"]) and engine_enabled(),
        "connected": connected,
        # every fleet places real orders, so no keys means no trading at
        # all — surfaced on the page rather than failing silently
        "needs_keys": not connected,
    }


def public_user(user_id: int) -> dict:
    """Safe-to-render view of one account's custom-bot settings."""
    u = load_user(user_id)
    own = bool(u.get("alpaca_key_id") and u.get("alpaca_secret_key"))
    return {
        "alpaca_key_id": u.get("alpaca_key_id") or "",
        "has_secret": bool(u.get("alpaca_secret_key")),
        "own_keys": own,
        "connected": own,
        "same_as_fleet": same_as_fleet(user_id),
        "custom_pool_total": u.get("custom_pool_total") or 0.0,
    }
