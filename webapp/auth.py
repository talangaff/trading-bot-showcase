"""Username + password accounts with roles, stored as JSON in data/auth.json.

Two roles:
  * admin — everything, plus the Users page (create / reset / disable /
    delete accounts, grant or revoke admin), Guardrails, and control of the
    fleet (engine settings + Alpaca connection, spawn / run / reset, edits).
  * user  — the screener pages, a read-only view of the fleet, their own
    journals and their own custom bots.

A fresh install has one admin (``admin`` / config.DEFAULT_PASSWORD) that
must change its password on first login. A data dir written by the old
single-password build (a bare ``password_hash`` at the top level) is
migrated in place: that hash becomes the ``admin`` account, so nobody is
locked out by the upgrade.

Sessions carry the user id plus a per-user ``session_version``; bumping the
version (an admin resetting a password, disabling the account) signs every
existing session for that user out on its next request.

CLI (``python -m webapp.auth``) for lockout recovery and scripting::

    python -m webapp.auth list
    python -m webapp.auth reset-admin            # admin back to the default, forced change
    python -m webapp.auth create NAME [--admin]  # prints the temporary password
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import time
from datetime import datetime, timezone

from fastapi import Request

from . import config

_PBKDF2_ROUNDS = 240_000
_SCHEMA_VERSION = 2

ROLES = ("admin", "user")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
MIN_PASSWORD_LEN = 8


# ---------------------------------------------------------------------------
# password hashing
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, dk_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# storage  (data/auth.json)
# ---------------------------------------------------------------------------
def _read_raw() -> dict:
    try:
        return json.loads(config.AUTH_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_raw(data: dict) -> None:
    tmp = config.AUTH_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, config.AUTH_FILE)


def _new_user(uid: int, username: str, password_hash: str, role: str,
              must_change: bool) -> dict:
    return {
        "id": uid,
        "username": username,
        "role": role,
        "password_hash": password_hash,
        "must_change": bool(must_change),
        "disabled": False,
        "session_version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "last_login_at": None,
    }


def _migrate_v1(old: dict) -> dict:
    """Old single-password layout -> one admin account carrying that hash."""
    admin = _new_user(1, config.DEFAULT_USERNAME, old["password_hash"], "admin",
                      old.get("must_change", False))
    admin["created_at"] = old.get("updated_at") or admin["created_at"]
    return {"version": _SCHEMA_VERSION, "next_id": 2, "users": [admin]}


def _load() -> dict:
    """The v2 document, migrating or creating it as needed."""
    raw = _read_raw()
    if raw.get("version") == _SCHEMA_VERSION and isinstance(raw.get("users"), list):
        return raw
    if raw.get("password_hash"):
        doc = _migrate_v1(raw)
    else:
        doc = {
            "version": _SCHEMA_VERSION, "next_id": 2,
            "users": [_new_user(1, config.DEFAULT_USERNAME,
                                _hash_password(config.DEFAULT_PASSWORD), "admin", True)],
        }
    _write_raw(doc)
    return doc


def ensure_initialised() -> None:
    """Create data/auth.json (default admin) or migrate an old one in place."""
    _load()


def _public(u: dict) -> dict:
    return {k: v for k, v in u.items() if k != "password_hash"}


def _find(doc: dict, uid: int) -> dict | None:
    for u in doc["users"]:
        if u["id"] == uid:
            return u
    return None


def normalise_username(name: str) -> str:
    return (name or "").strip().lower()


def username_error(name: str) -> str | None:
    if not USERNAME_RE.match(name):
        return ("Usernames are 2–32 characters: lowercase letters, digits, "
                "dots, dashes or underscores, starting with a letter or digit.")
    return None


def password_error(password: str) -> str | None:
    if (password or "").strip().lower() == config.DEFAULT_PASSWORD.lower():
        return "Pick something other than the default."
    if len(password or "") < MIN_PASSWORD_LEN:
        return "Use at least {} characters.".format(MIN_PASSWORD_LEN)
    return None


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
def list_users() -> list[dict]:
    return [_public(u) for u in sorted(_load()["users"], key=lambda u: u["id"])]


def get_user(uid: int) -> dict | None:
    u = _find(_load(), uid)
    return _public(u) if u else None


def get_user_by_username(name: str) -> dict | None:
    name = normalise_username(name)
    for u in _load()["users"]:
        if u["username"] == name:
            return _public(u)
    return None


def admin_ids() -> list[int]:
    return [u["id"] for u in _load()["users"] if u["role"] == "admin"]


def _primary_admin_id(doc: dict) -> int:
    ids = [u["id"] for u in doc["users"] if u["role"] == "admin"]
    return min(ids) if ids else 1


def primary_admin_id() -> int:
    """The lowest-numbered admin — the account pre-existing data (journals,
    custom bots, the custom-bot capital pool) is assigned to on upgrade."""
    return _primary_admin_id(_load())


def is_reserved_username(name: str, uid: int | None = None) -> bool:
    """Whether `name` is the default login name reserved for the primary
    admin. That account answers to config.DEFAULT_USERNAME forever (see
    authenticate), so letting anyone else hold the name would make the
    login ambiguous: two accounts, one way in."""
    doc = _load()
    return (normalise_username(name) == config.DEFAULT_USERNAME
            and uid != _primary_admin_id(doc))


def set_username(uid: int, new_username: str) -> dict:
    """Rename an account. The username is both the login and the name shown
    around the UI, so this changes what people see and what they type.

    Renaming the primary admin does NOT cost it the default login name:
    authenticate() still accepts config.DEFAULT_USERNAME for that account,
    so a rename can never lock the installation out of its own admin."""
    name = normalise_username(new_username)
    err = username_error(name)
    if err:
        raise ValueError(err)
    doc = _load()
    u = _find(doc, uid)
    if not u:
        raise ValueError("No such account.")
    if name == u["username"]:
        return _public(u)
    if any(o["username"] == name for o in doc["users"] if o["id"] != uid):
        raise ValueError("That username is already taken.")
    if name == config.DEFAULT_USERNAME and uid != _primary_admin_id(doc):
        raise ValueError('"{}" is reserved for the primary admin account.'
                         .format(config.DEFAULT_USERNAME))
    u["username"] = name
    u["updated_at"] = _now()
    _write_raw(doc)
    return _public(u)


def is_admin(user: dict | None) -> bool:
    return bool(user) and user.get("role") == "admin"


def authenticate(username: str, password: str) -> dict | None:
    """The user if the credentials match an enabled account, else None.
    Always runs the hash so a bad username costs the same as a bad password."""
    name = normalise_username(username)
    doc = _load()
    match = next((u for u in doc["users"] if u["username"] == name), None)
    if match is None and name == config.DEFAULT_USERNAME:
        # The primary admin answers to its default name for good, even after
        # being renamed: the rename is a display change, not a new way in, and
        # nobody should be able to lock the box out of its own admin by
        # picking a name they later forget. An account literally holding the
        # name still wins the exact match above, and no other account is
        # allowed to hold it (see is_reserved_username).
        match = _find(doc, _primary_admin_id(doc))
    stored = match["password_hash"] if match else _hash_password("")
    ok = _verify_password(password or "", stored)
    if not match or not ok or match.get("disabled"):
        return None
    match["last_login_at"] = _now()
    _write_raw(doc)
    return _public(match)


def check_password(uid: int, password: str) -> bool:
    u = _find(_load(), uid)
    return bool(u) and _verify_password(password or "", u["password_hash"])


def create_user(username: str, password: str, role: str = "user",
                must_change: bool = True) -> dict:
    name = normalise_username(username)
    err = username_error(name)
    if err:
        raise ValueError(err)
    if role not in ROLES:
        raise ValueError("Unknown role.")
    if len(password or "") < MIN_PASSWORD_LEN:
        raise ValueError("Use at least {} characters.".format(MIN_PASSWORD_LEN))
    doc = _load()
    if any(u["username"] == name for u in doc["users"]):
        raise ValueError("That username is already taken.")
    if name == config.DEFAULT_USERNAME:
        raise ValueError('"{}" is reserved for the primary admin account.'
                         .format(config.DEFAULT_USERNAME))
    uid = int(doc.get("next_id") or (max(u["id"] for u in doc["users"]) + 1))
    user = _new_user(uid, name, _hash_password(password), role, must_change)
    doc["users"].append(user)
    doc["next_id"] = uid + 1
    _write_raw(doc)
    return _public(user)


def set_password(uid: int, new_password: str, *, must_change: bool = False,
                 sign_out_elsewhere: bool = False) -> None:
    """Set a user's password. `must_change` marks it temporary (they are
    forced to a new one on next login); `sign_out_elsewhere` invalidates
    every existing session for that user."""
    doc = _load()
    u = _find(doc, uid)
    if not u:
        raise ValueError("No such user.")
    u["password_hash"] = _hash_password(new_password)
    u["must_change"] = bool(must_change)
    u["updated_at"] = _now()
    if sign_out_elsewhere:
        u["session_version"] = int(u.get("session_version") or 1) + 1
    _write_raw(doc)


def reset_password(uid: int) -> str:
    """Admin reset: a fresh random temporary password that must be changed
    on next login. Existing sessions are signed out. Returns the password."""
    temp = secrets.token_urlsafe(9)
    set_password(uid, temp, must_change=True, sign_out_elsewhere=True)
    return temp


def _assert_not_last_admin(doc: dict, uid: int) -> None:
    others = [u for u in doc["users"]
              if u["role"] == "admin" and not u.get("disabled") and u["id"] != uid]
    if not others:
        raise ValueError("That would leave no enabled admin account.")


def set_role(uid: int, role: str) -> None:
    if role not in ROLES:
        raise ValueError("Unknown role.")
    doc = _load()
    u = _find(doc, uid)
    if not u:
        raise ValueError("No such user.")
    if u["role"] == "admin" and role != "admin":
        _assert_not_last_admin(doc, uid)
    u["role"] = role
    u["updated_at"] = _now()
    _write_raw(doc)


def set_disabled(uid: int, disabled: bool) -> None:
    doc = _load()
    u = _find(doc, uid)
    if not u:
        raise ValueError("No such user.")
    if disabled and u["role"] == "admin":
        _assert_not_last_admin(doc, uid)
    u["disabled"] = bool(disabled)
    u["updated_at"] = _now()
    if disabled:
        u["session_version"] = int(u.get("session_version") or 1) + 1
    _write_raw(doc)


def delete_user(uid: int) -> None:
    """Remove the account itself. The caller is responsible for the user's
    data (journals, custom bots) — see main.admin_user_delete."""
    doc = _load()
    u = _find(doc, uid)
    if not u:
        raise ValueError("No such user.")
    if u["role"] == "admin":
        _assert_not_last_admin(doc, uid)
    doc["users"] = [x for x in doc["users"] if x["id"] != uid]
    _write_raw(doc)


# ---------------------------------------------------------------------------
# session helpers
# ---------------------------------------------------------------------------
def current_user(request: Request) -> dict | None:
    """The signed-in user for this request, re-read from the store so a
    disable / role change / password reset takes effect immediately."""
    uid = request.session.get("uid")
    if not uid:
        return None
    u = get_user(int(uid))
    if not u or u.get("disabled"):
        return None
    if int(request.session.get("sv") or 0) != int(u.get("session_version") or 1):
        return None
    return u


def is_logged_in(request: Request) -> bool:
    return current_user(request) is not None


def login_session(request: Request, user: dict) -> None:
    request.session.clear()
    request.session["uid"] = int(user["id"])
    request.session["sv"] = int(user.get("session_version") or 1)
    request.session["login_at"] = int(time.time())


def refresh_session(request: Request, uid: int) -> None:
    """Re-sync the session version after the user changed their own password."""
    u = get_user(uid)
    if u:
        request.session["sv"] = int(u.get("session_version") or 1)


def logout_session(request: Request) -> None:
    request.session.clear()


# ---------------------------------------------------------------------------
# very light brute-force throttle (per-process, best effort)
# ---------------------------------------------------------------------------
_fails: dict[str, list] = {}


def register_failure(ip: str) -> None:
    now = time.time()
    hist = [t for t in _fails.get(ip, []) if now - t < 300]
    hist.append(now)
    _fails[ip] = hist


def throttled_seconds(ip: str) -> float:
    now = time.time()
    hist = [t for t in _fails.get(ip, []) if now - t < 300]
    _fails[ip] = hist
    if len(hist) >= 5:
        # exponential-ish backoff after 5 failures in 5 minutes
        return min(2 ** (len(hist) - 4), 30)
    return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        for u in list_users():
            flags = [u["role"]]
            if u.get("disabled"):
                flags.append("disabled")
            if u.get("must_change"):
                flags.append("must-change-password")
            print("{:>3}  {:<20} {}".format(u["id"], u["username"], ", ".join(flags)))
        return 0
    if cmd == "reset-admin":
        doc = _load()
        admin = next((u for u in doc["users"] if u["username"] == config.DEFAULT_USERNAME), None)
        if admin is None:
            admin = _find(doc, primary_admin_id())
        if admin is None:
            print("no admin account found", file=sys.stderr)
            return 1
        admin["password_hash"] = _hash_password(config.DEFAULT_PASSWORD)
        admin["must_change"] = True
        admin["disabled"] = False
        admin["role"] = "admin"
        admin["session_version"] = int(admin.get("session_version") or 1) + 1
        admin["updated_at"] = _now()
        _write_raw(doc)
        print("'{}' reset to the default password; a change is forced on next login."
              .format(admin["username"]))
        return 0
    if cmd == "create" and len(argv) >= 2:
        role = "admin" if "--admin" in argv[2:] else "user"
        temp = secrets.token_urlsafe(9)
        try:
            u = create_user(argv[1], temp, role=role, must_change=True)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        print("created {} ({}) — temporary password: {}".format(u["username"], role, temp))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
