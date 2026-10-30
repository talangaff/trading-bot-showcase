"""Risk guardrails, in three layers.

  1. The installation-wide defaults an admin sets on the Guardrails page,
     stored as JSON in the mutable data dir (git-ignored).
  2. Each user's own overrides for their own fleet — their account, their
     money, their risk appetite — stored per user id alongside them.
  3. A custom bot's per-bot overrides on top of its owner's (see
     bots.effective_guardrails).

`clamp()` enforces the absolute floor/ceiling from config.py at every
layer, so loosening a guardrail anywhere — house default, a user's fleet,
or one custom bot — can never remove risk control outright.

Most fields bound what a bot may risk on one trade. `max_fleet_bots` is
the exception: it bounds how many active bots a fleet may have at all
(enforced in bots.spawn_bot, the one door every fleet bot comes through).
It is a property of the fleet, not of a bot, so — unlike the four risk
limits — a single custom bot cannot override it; bots.update_guardrails
does not accept it.
"""

import json
import os

from . import config

_PATH = config.DATA_DIR / "guardrails.json"
_USER_PATH = config.DATA_DIR / "user_guardrails.json"
_FIELDS = ("max_drawdown_pct", "position_pct_limit", "min_contracts", "max_contracts",
           "max_fleet_bots")

_DEFAULTS = {
    "max_drawdown_pct": config.BOT_MAX_DRAWDOWN_PCT,
    "position_pct_limit": config.BOT_POSITION_PCT_LIMIT,
    "min_contracts": config.BOT_MIN_CONTRACTS,
    "max_contracts": config.BOT_MAX_CONTRACTS,
    "max_fleet_bots": config.BOT_MAX_FLEET_BOTS,
}

# absolute bounds — never editable, never overridable per-bot
_FLOOR = {
    "max_drawdown_pct": (1.0, config.BOT_ABS_MAX_DRAWDOWN_PCT),
    "position_pct_limit": (1.0, config.BOT_ABS_POSITION_PCT_LIMIT),
    "min_contracts": (config.BOT_ABS_MIN_CONTRACTS, config.BOT_ABS_MAX_CONTRACTS),
    "max_contracts": (config.BOT_ABS_MIN_CONTRACTS, config.BOT_ABS_MAX_CONTRACTS),
    "max_fleet_bots": (config.BOT_ABS_MIN_FLEET_BOTS, config.BOT_ABS_MAX_FLEET_BOTS),
}

_INT_FIELDS = ("min_contracts", "max_contracts", "max_fleet_bots")


def clamp(d: dict) -> dict:
    out = dict(d)
    for k, (lo, hi) in _FLOOR.items():
        v = out.get(k, _DEFAULTS[k])
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = _DEFAULTS[k]
        v = max(lo, min(hi, v))
        out[k] = int(round(v)) if k in _INT_FIELDS else round(v, 2)
    if out["min_contracts"] > out["max_contracts"]:
        out["min_contracts"] = out["max_contracts"]
    return out


def _read(path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = {}
    return d if isinstance(d, dict) else {}


def _write(path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def house() -> dict:
    """The installation-wide defaults (admin, Guardrails page)."""
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in _read(_PATH).items() if k in _FIELDS})
    return clamp(out)


def load(user_id: int | None = None) -> dict:
    """The limits a given account's bots trade under: that account's own
    overrides layered on the house defaults, clamped. Without a user id
    (population-wide views, replay) the house defaults themselves."""
    out = house()
    if user_id is not None:
        mine = _read(_USER_PATH).get(str(int(user_id))) or {}
        out.update({k: v for k, v in mine.items() if k in _FIELDS and v not in (None, "")})
    return clamp(out)


def overrides_of(user_id: int) -> dict:
    """Only the fields this account has actually overridden — so the form
    can show a blank (= inherit the house default) rather than a copy."""
    return {k: v for k, v in (_read(_USER_PATH).get(str(int(user_id))) or {}).items()
            if k in _FIELDS and v not in (None, "")}


def save(**changes) -> dict:
    """Set the house defaults (admin)."""
    d = house()
    for k, v in changes.items():
        if k not in _FIELDS or v in (None, ""):
            continue
        d[k] = v
    d = clamp(d)
    _write(_PATH, d)
    return d


def save_user(user_id: int, **changes) -> dict:
    """Set one account's own fleet guardrails. A field passed blank is
    CLEARED, so that limit goes back to inheriting the house default."""
    allu = _read(_USER_PATH)
    mine = dict(allu.get(str(int(user_id))) or {})
    for k, v in changes.items():
        if k not in _FIELDS:
            continue
        if v in (None, ""):
            mine.pop(k, None)
        else:
            mine[k] = v
    # store the clamped value so nothing out of bounds is ever persisted
    if mine:
        clamped = clamp({**house(), **mine})
        mine = {k: clamped[k] for k in mine}
        allu[str(int(user_id))] = mine
    else:
        allu.pop(str(int(user_id)), None)
    _write(_USER_PATH, allu)
    return load(user_id)


def delete_user(user_id: int) -> None:
    allu = _read(_USER_PATH)
    if allu.pop(str(int(user_id)), None) is not None:
        _write(_USER_PATH, allu)


def bounds() -> dict:
    """Absolute min/max each field can ever be set to — for rendering the form."""
    return _FLOOR
