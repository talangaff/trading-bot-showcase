"""Broker accounts a custom bot can be put to work on — paper or live.

Each signed-in user keeps a short list of named Alpaca accounts, each one
either 'paper' or 'live', stored as JSON in the mutable data dir (git-ignored,
0600) like the other key files. A custom bot points at ONE of them
(bots.account_id); the account decides both the keys its orders go out on and
which Alpaca trading host receives them.

Which host a key pair talks to is a property of the key, not of every call
site: trading_base(keys) looks the key ID up here, so the many places that
already pass `keys` around (broker, account_book, api_meter) keep their
signatures and a live account can never be reached by accident through a
code path that assumed paper.

Live trading is off until an admin switches it on (bot_settings
`live_enabled`). With it off, a bot assigned to a live account resolves to no
keys and does not trade — that switch is also the kill switch.

The fleet and baseline bots are not part of this list: they trade on the
owner's separate paper FLEET account (shown on the Accounts page as type
"fleet"; its keys live in bot_settings because the whole engine reads them
there). A fleet account is paper only, is edited through set_fleet_keys(), and
its key ID can never be saved as — or used by — a custom/manager account, nor a
custom account's key as the fleet's.
"""

import json
import os
import re

import requests

from . import bot_settings, config

_PATH = config.DATA_DIR / "accounts.json"
PAPER_BASE = "https://paper-api.alpaca.markets/v2"
LIVE_BASE = "https://api.alpaca.markets/v2"
TYPES = ("paper", "live")
_NAME_RE = re.compile(r"^[\w .\-]{1,40}$")
_CACHE: dict = {}


class AccountError(Exception):
    pass


def _read() -> dict:
    try:
        st = _PATH.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    hit = _CACHE.get("v")
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        d = json.loads(_PATH.read_text())
    except (OSError, ValueError):
        d = {}
    d = d if isinstance(d, dict) else {}
    _CACHE["v"] = (stamp, d)
    return d


def _write(data: dict) -> None:
    _PATH.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(_PATH, 0o600)
    except OSError:
        pass
    _CACHE.clear()


def _migrate(uid: int) -> None:
    """One-time: the custom-bot key pair a user saved before accounts existed
    becomes their first account, 'Paper', and their existing custom bots are
    put on it so nothing silently stops trading."""
    data = _read()
    mine = data.get(str(uid)) or {}
    if mine.get("accounts") or mine.get("migrated"):
        return
    legacy = bot_settings.load_user(uid)
    kid, sec = legacy.get("alpaca_key_id"), legacy.get("alpaca_secret_key")
    mine = {"accounts": [], "next": 1, "migrated": True}
    if kid and sec:
        mine["accounts"].append({"id": 1, "name": "Paper", "type": "paper",
                                 "key_id": kid, "secret": sec})
        mine["next"] = 2
    data[str(uid)] = mine
    _write(data)
    if mine["accounts"]:
        from . import bots
        for b in bots.custom_bots(uid):
            if b.get("account_id") is None:
                bots.set_account(b["id"], 1, pause=False)


def list_accounts(uid: int) -> list[dict]:
    _migrate(int(uid))
    return list((_read().get(str(int(uid))) or {}).get("accounts") or [])


def get(uid: int | None, account_id) -> dict | None:
    if uid is None or account_id in (None, ""):
        return None
    try:
        aid = int(account_id)
    except (TypeError, ValueError):
        return None                 # "fleet" is not a number: the fleet account is never handed out here
    for a in list_accounts(uid):
        if a["id"] == aid:
            return a
    return None


def fleet_account(uid: int | None) -> dict | None:
    """The user's fleet account as an account-shaped dict (type "fleet"), or None
    when no fleet keys are saved. For display and key management only — it is not
    in list_accounts(), so nothing that picks an account can pick it."""
    keys = bot_settings.fleet_keys(uid)
    if not keys:
        return None
    return {"id": "fleet", "name": "Fleet", "type": "fleet", "key_id": keys[0], "secret": keys[1]}


def is_fleet_key(key_id: str | None) -> bool:
    """True if this key ID is any user's fleet account."""
    if not key_id:
        return False
    for uid in bot_settings.fleet_user_ids():
        k = bot_settings.fleet_keys(uid)
        if k and k[0] == key_id:
            return True
    return False


def set_fleet_keys(uid: int, key_id: str = "", secret: str = "") -> None:
    """Replace the fleet account's keys (blank fields keep the saved ones). The
    keys must be a PAPER account on Alpaca, must not be a saved custom/manager
    account, and — if the fleet is holding positions — must be the same Alpaca
    account as the old keys (a regenerated key on the same account is fine;
    a different account would leave the ledgers describing positions that are
    not there)."""
    key_id, secret = (key_id or "").strip(), (secret or "").strip()
    old = bot_settings.fleet_keys(uid)
    new_key = key_id or (old[0] if old else "")
    new_secret = secret or (old[1] if old else "")
    if not new_key or not new_secret:
        raise AccountError("enter both the key ID and the secret")
    if key_id and not secret and old and key_id != old[0]:
        raise AccountError("enter the secret that goes with the new key ID")
    if old and (new_key, new_secret) == tuple(old):
        return
    for mine in _read().values():
        for a in (mine or {}).get("accounts") or []:
            if a.get("key_id") == new_key:
                raise AccountError("that key is a saved account (\"{}\") — the fleet needs its own "
                                   "separate paper account".format(a["name"]))
    new = verify("paper", new_key, new_secret)
    if old:
        from . import bots
        holding = [b for b in bots.active_bots(pool="fleet", user_id=uid) if bots.has_open_positions(b["id"])]
        if holding:
            try:
                prev = _probe(PAPER_BASE, old[0], old[1])
            except AccountError:
                prev = None
            if prev and prev.get("account_number") != new.get("account_number"):
                raise AccountError("{} fleet bot(s) are holding positions on the current account, and these keys "
                                   "are a different Alpaca account — let them close first".format(len(holding)))
    bot_settings.save_user(uid, fleet_alpaca_key_id=new_key, fleet_alpaca_secret_key=new_secret)


def public(a: dict) -> dict:
    """The account without its secret — what pages and templates get."""
    return {"id": a["id"], "name": a["name"], "type": a["type"],
            "key_id": a["key_id"], "live": a["type"] == "live"}


def keys_of(a: dict | None) -> tuple[str, str] | None:
    return (a["key_id"], a["secret"]) if a and a.get("key_id") and a.get("secret") else None


def is_live_key(key_id: str | None) -> bool:
    if not key_id:
        return False
    for mine in _read().values():
        for a in (mine or {}).get("accounts") or []:
            if a.get("type") == "live" and a.get("key_id") == key_id:
                return True
    return False


def trading_base(keys: tuple[str, str] | None) -> str:
    """The Alpaca trading API a key pair talks to. Paper unless the key ID is
    registered here as a live account."""
    return LIVE_BASE if keys and is_live_key(keys[0]) else PAPER_BASE


def live_enabled() -> bool:
    return bool(bot_settings.load().get("live_enabled"))


def keys_for_bot(bot: dict | None) -> tuple[str, str] | None:
    """The keys a custom bot trades with: its assigned account's, or None —
    no account assigned, or a live account while live trading is switched off."""
    b = bot or {}
    a = get(b.get("user_id"), b.get("account_id"))
    if not a:
        return None
    if a["type"] == "live" and not live_enabled():
        return None
    if is_fleet_key(a.get("key_id")):
        return None                 # the fleet's account is for the fleet only
    return keys_of(a)


def _probe(base: str, key_id: str, secret: str) -> dict | None:
    try:
        r = requests.get(base + "/account", timeout=8, headers={
            "APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})
    except requests.RequestException as e:
        raise AccountError("could not reach Alpaca: {}".format(e))
    return r.json() if r.status_code == 200 else None


def verify(kind: str, key_id: str, secret: str) -> dict:
    """Confirm the keys really are a `kind` account on Alpaca, so a live bot
    can't be pointed at a paper account (or the reverse) by mislabeling."""
    own = _probe(PAPER_BASE if kind == "paper" else LIVE_BASE, key_id, secret)
    if own is None:
        other = _probe(LIVE_BASE if kind == "paper" else PAPER_BASE, key_id, secret)
        if other is not None:
            raise AccountError("these keys are a {} account, not {}".format(
                "live" if kind == "paper" else "paper", kind))
        raise AccountError("Alpaca rejected these keys for a {} account".format(kind))
    return own


def add(uid: int, name: str, kind: str, key_id: str, secret: str) -> dict:
    name, key_id, secret = (name or "").strip(), (key_id or "").strip(), (secret or "").strip()
    if kind not in TYPES:
        raise AccountError("account type must be paper or live")
    if not _NAME_RE.match(name):
        raise AccountError("give the account a short name (letters, numbers, spaces)")
    if not key_id or not secret:
        raise AccountError("enter both the key ID and the secret")
    mine_list = list_accounts(uid)
    if any(a["name"].lower() == name.lower() for a in mine_list):
        raise AccountError("you already have an account with that name")
    for mine in _read().values():
        for a in (mine or {}).get("accounts") or []:
            if a.get("key_id") == key_id and a.get("type") != kind:
                raise AccountError("that key ID is already saved as a {} account".format(a["type"]))
    if is_fleet_key(key_id):
        raise AccountError("that key is the fleet account's — the fleet account can only be used by the fleet")
    verify(kind, key_id, secret)
    data = _read()
    mine = data.get(str(int(uid))) or {"accounts": [], "next": 1, "migrated": True}
    acct = {"id": int(mine.get("next") or 1), "name": name, "type": kind,
            "key_id": key_id, "secret": secret}
    mine["accounts"] = list(mine.get("accounts") or []) + [acct]
    mine["next"] = acct["id"] + 1
    data[str(int(uid))] = mine
    _write(data)
    return acct


def usage(uid: int, account_id: int) -> dict:
    """What is using an account: the custom bots assigned to it and the managers
    that run on it. A bot or manager keeps pointing at an account by id, so
    this is what stops the account being deleted out from under them."""
    from . import bots
    mine = [b for b in bots.custom_bots(uid) if b.get("account_id") == int(account_id)]
    with bots._conn() as c:
        mgrs = [dict(r) for r in c.execute(
            "SELECT id, name FROM managers WHERE user_id=? AND account_id=?",
            (int(uid), int(account_id))).fetchall()]
    return {"bots": mine, "managers": mgrs,
            "holding": [b for b in mine if bots.has_open_positions(b["id"])]}


def update(uid: int, account_id: int, name: str | None = None, key_id: str | None = None,
           secret: str | None = None) -> dict:
    """Rename an account and/or replace its keys. The type is fixed — it is what
    the keys were verified against, and a bot on a live account has passed the
    live gate; to change it, add a new account. Blank key fields keep the
    saved ones. New keys are verified against Alpaca for this account's type,
    and are refused while a bot on the account is holding a position (the
    positions live on the old Alpaca account, so the ledger and the account
    would disagree)."""
    acct = get(uid, account_id)
    if not acct:
        raise AccountError("no such account")
    new_name = acct["name"] if name is None or not name.strip() else name.strip()
    key_id, secret = (key_id or "").strip(), (secret or "").strip()
    new_key, new_secret = key_id or acct["key_id"], secret or acct["secret"]
    if not _NAME_RE.match(new_name):
        raise AccountError("give the account a short name (letters, numbers, spaces)")
    if any(a["id"] != acct["id"] and a["name"].lower() == new_name.lower()
           for a in list_accounts(uid)):
        raise AccountError("you already have an account with that name")
    keys_changed = (new_key, new_secret) != (acct["key_id"], acct["secret"])
    if keys_changed:
        if key_id and not secret and key_id != acct["key_id"]:
            raise AccountError("enter the secret that goes with the new key ID")
        for mine in _read().values():
            for a in (mine or {}).get("accounts") or []:
                if a.get("key_id") == new_key and a.get("type") != acct["type"]:
                    raise AccountError("that key ID is already saved as a {} account".format(a["type"]))
        if is_fleet_key(new_key):
            raise AccountError("that key is the fleet account's — the fleet account can only be used by the fleet")
        held = usage(uid, acct["id"])["holding"]
        if held:
            raise AccountError("bot(s) {} on this account are holding positions — close them before "
                               "changing its keys".format(", ".join("#{}".format(b["id"]) for b in held)))
        verify(acct["type"], new_key, new_secret)
    data = _read()
    mine = data.get(str(int(uid))) or {}
    for a in mine.get("accounts") or []:
        if a["id"] == acct["id"]:
            a.update({"name": new_name, "key_id": new_key, "secret": new_secret})
            out = dict(a)
    data[str(int(uid))] = mine
    _write(data)
    return out


def remove(uid: int, account_id: int) -> None:
    u = usage(uid, account_id)
    if u["bots"]:
        raise AccountError("a bot is still assigned to that account — move or delete it first")
    if u["managers"]:
        raise AccountError("manager \"{}\" runs on that account — delete the manager first".format(
            u["managers"][0]["name"]))
    data = _read()
    mine = data.get(str(int(uid))) or {}
    mine["accounts"] = [a for a in mine.get("accounts") or [] if a["id"] != int(account_id)]
    data[str(int(uid))] = mine
    _write(data)
