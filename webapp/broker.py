"""Real order execution against Alpaca paper trading accounts.

Every bot in the system places REAL orders (see execution() below), and
every bot has an owner (bots.user_id) whose keys they go out on:

  * a user's fleet and baseline bots trade on THAT USER's Alpaca paper
    fleet keys (entered on their Bot Fleet settings panel), all of that
    user's fleet together on that one account — each bot keeps its own $10k
    ledger in bots.bot_trades, and webapp/account_book.py tracks which bot
    owns how many contracts of each symbol on the shared account so a bot
    only ever sells its own;
  * custom bots trade on their owner's separate custom-bot keys (entered on
    the Custom Bots page) — a different account again, so one genome can be
    tested on its own.

There is no shared or fallback key anywhere: a bot with no owner, or an
owner with no keys saved, does not trade. That is the guarantee that one
user's bots can never reach another user's account, however the genomes
they run are shared.

Every fill recorded in bots.bot_trades is the ACTUAL fill Alpaca reported,
never a modeled price — there is no exception and no way to configure one.
The only modeled fills anywhere are in the offline replay harness
(webapp/replay.py), which runs entirely in memory and never touches an
account.

Alpaca options rules this module enforces on the way out:
  * only `market` and `limit` order types exist for options, time in force
    `day`. A bot's genome may prefer stop_limit / trailing_stop — those are
    the bot's own decision rules and the engine emulates them, so here a
    stop_limit goes out as a limit at its limit price and a trailing_stop
    exit goes out as a market order (see _options_order()).
  * a vertical debit spread (option_model.spread_symbol, one package in the
    ledger) goes out as ONE multi-leg (`order_class: mleg`) order, long the
    first strike and short the second, and comes back with one net fill
    price — so the account is never left holding a naked short leg.
  * every order carries a client_order_id so a cycle that dies between
    submitting and booking can find its fill again (bot_engine._recover_pending).
"""

import time
from datetime import datetime, timezone

import requests

from . import accounts, api_meter, bot_settings, option_model

_BASE = "https://paper-api.alpaca.markets/v2"
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "stopped"}


class BrokerError(Exception):
    pass


# ---------------------------------------------------------------------------
# which account a bot trades on
# ---------------------------------------------------------------------------
def keys_for_bot(bot: dict | None) -> tuple[str, str] | None:
    """The Alpaca keys a bot's orders go out on — always its OWNER's: their
    custom-bot keys for a custom bot, their fleet keys for a fleet or
    baseline bot. None if the bot has no owner or the owner has not saved
    those keys."""
    b = bot or {}
    uid = b.get("user_id")
    if (b.get("pool") or "fleet") == "custom":
        return accounts.keys_for_bot(b)
    return bot_settings.fleet_keys(uid)


def execution(bot: dict | None) -> tuple[str, tuple[str, str] | None]:
    """How this bot's fills are produced. Always by sending a real order:

      ('broker', keys) — send a real order on `keys`, book Alpaca's fill
      ('broker', None) — no keys saved for this bot's account: an error,
                         never a silent fallback to a modeled fill

    Every FLEET/custom bot in the live engine trades on an Alpaca account, with
    no modeled-fill mode to fall back on (the incubator's pool='sim' bots are
    the one deliberate exception, see below). That is the whole basis of the
    fitness numbers: evolution breeds and culls on fills that really
    happened, at prices a real book really gave, so a genome cannot be
    selected on the strength of a spread model that flatters it. A fleet
    with no keys does not trade at all — it never trades imaginary ones
    instead.

    (The offline replay harness still models fills, but it runs on its own
    in-memory ledger and reaches this module not at all — see
    bot_engine._record_fill, which treats a book with no execution() as
    modeled.)
    """
    b = bot or {}
    uid = b.get("user_id")
    if (b.get("pool") or "fleet") == "sim":
        # the incubator (webapp/incubator.py): the one population that never
        # touches an account. It is not a fallback for a fleet with no keys —
        # a sim bot is *defined* by modeled fills, and only graduates to a
        # fleet bot (a new row, with its own real ledger) after proving itself.
        return "model", None
    if (b.get("pool") or "fleet") == "custom":
        return "broker", accounts.keys_for_bot(b)
    return "broker", bot_settings.fleet_keys(uid)


def account_key(keys: tuple[str, str] | None) -> str | None:
    """Stable identifier for the account behind a key pair — the key ID.
    Used to group every bot trading on the same account (account_book)."""
    return keys[0] if keys and keys[0] else None


def _headers(keys: tuple[str, str] | None = None) -> dict | None:
    """Auth headers for one account's keys, or None if there are none.

    There is deliberately no fallback: with a fleet per user there is no
    "the" account to fall back TO, and guessing one would be the single
    mistake that could send a bot's order to somebody else's money. A
    caller with no keys gets None and every path above turns that into a
    visible BrokerError."""
    if keys is None:
        return None
    key, secret = keys
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def connected(keys: tuple[str, str] | None = None) -> bool:
    return _headers(keys) is not None


def _get(path: str, keys, params: dict | None = None, what: str = "request") -> dict | list:
    hdr = _headers(keys)
    if not hdr:
        raise BrokerError("no Alpaca keys saved")
    try:
        r = api_meter.get(accounts.trading_base(keys) + path, keys=keys, headers=hdr, params=params, timeout=8)
    except requests.RequestException as e:
        raise BrokerError("could not reach Alpaca: {}".format(e))
    if r.status_code != 200:
        raise BrokerError("Alpaca {} failed (HTTP {}): {}".format(
            what, r.status_code, (r.text or "")[:200]))
    return r.json()


# ---------------------------------------------------------------------------
# account state
# ---------------------------------------------------------------------------
def account(keys: tuple[str, str] | None = None) -> dict:
    """The account behind `keys`: number, status, equity, buying power and
    options level — what the Bot Fleet / Custom Bots pages show so the
    virtual per-bot ledgers can be sanity-checked against the real thing."""
    j = _get("/account", keys, what="account lookup")

    def f(k):
        try:
            return float(j.get(k)) if j.get(k) not in (None, "") else None
        except (TypeError, ValueError):
            return None

    return {
        "account_number": j.get("account_number"),
        "status": j.get("status"),
        "equity": f("equity"),
        "cash": f("cash"),
        "buying_power": f("buying_power"),
        "options_buying_power": f("options_buying_power"),
        "options_trading_level": j.get("options_trading_level"),
        "options_approved_level": j.get("options_approved_level"),
    }


def positions(keys: tuple[str, str] | None = None) -> dict[str, float]:
    """{OCC symbol: signed contracts} for every OPTION position actually held
    on the account right now (short legs of a spread come back negative).
    Raises BrokerError on any failure — a caller must never mistake "could
    not ask" for "holds nothing"."""
    rows = _get("/positions", keys, what="positions lookup")
    out: dict[str, float] = {}
    for p in rows or []:
        if (p.get("asset_class") or "") not in ("us_option", "option"):
            continue
        sym = (p.get("symbol") or "").upper()
        try:
            q = float(p.get("qty") or 0)
        except (TypeError, ValueError):
            continue
        if not sym or abs(q) < 1e-9:
            continue
        if q > 0 and (p.get("side") or "long") == "short":
            q = -q
        out[sym] = out.get(sym, 0.0) + q
    return out


# ---------------------------------------------------------------------------
# orders
# ---------------------------------------------------------------------------
def _options_order(order_type: str, limit_price, stop_price):
    """Map a bot's chosen order type onto what Alpaca accepts for options
    (market / limit, day). Returns (alpaca_type, limit_price)."""
    ot = (order_type or "market").lower()
    if ot == "limit":
        return "limit", limit_price
    if ot == "stop_limit":
        # the bot's own logic already decided the trigger fired; go out as a
        # marketable limit at its limit price
        return "limit", limit_price if limit_price is not None else stop_price
    return "market", None          # stop / trailing_stop are engine-emulated


def legs_of(contract: str, side: str) -> list[dict] | None:
    """The Alpaca legs for a ledger contract: None for a plain OCC symbol
    (single-leg order), else the two legs of a vertical debit spread."""
    info = option_model.parse_occ(contract)
    if not info or info.get("short_strike") is None:
        return None
    exp = info["expiry"]
    long_sym = option_model.occ_symbol(info["underlying"], exp, info["is_call"], info["strike"])
    short_sym = option_model.occ_symbol(info["underlying"], exp, info["is_call"], info["short_strike"])
    if side in ("buy_to_open", "buy"):
        return [{"symbol": long_sym, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
                {"symbol": short_sym, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"}]
    return [{"symbol": long_sym, "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_close"},
            {"symbol": short_sym, "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_close"}]


def order_body(symbol: str, qty: float, side: str, order_type: str = "market",
               limit_price=None, stop_price=None, trail_percent=None,
               position_intent: str | None = None, time_in_force: str = "day",
               client_order_id: str | None = None) -> dict:
    """The JSON Alpaca gets. `symbol` is a ledger contract — an OCC symbol
    or a spread_symbol(); `side` is 'buy'/'sell'."""
    atype, lim = _options_order(order_type, limit_price, stop_price)
    body = {"qty": str(int(qty)), "type": atype, "time_in_force": time_in_force}
    if client_order_id:
        body["client_order_id"] = client_order_id[:128]
    intent = position_intent or ("buy_to_open" if side == "buy" else "sell_to_close")
    legs = legs_of(symbol, intent)
    if legs:
        body["order_class"] = "mleg"
        body["legs"] = legs
        if lim is not None:
            # Alpaca's mleg convention: positive limit = net debit, negative = net credit
            body["limit_price"] = str(round(abs(float(lim)) * (1 if side == "buy" else -1), 2))
    else:
        body["symbol"] = symbol
        body["side"] = side
        body["position_intent"] = intent
        if lim is not None:
            body["limit_price"] = str(round(float(lim), 2))
    return body


def submit_order(symbol: str, qty: float, side: str, order_type: str = "market",
                 limit_price=None, stop_price=None, trail_percent=None,
                 position_intent: str | None = None,
                 time_in_force: str = "day",
                 keys: tuple[str, str] | None = None,
                 client_order_id: str | None = None) -> dict:
    hdr = _headers(keys)
    if not hdr:
        raise BrokerError("no Alpaca keys saved for this bot's account")
    body = order_body(symbol, qty, side, order_type, limit_price, stop_price, trail_percent,
                      position_intent, time_in_force, client_order_id)
    try:
        r = api_meter.post("{}/orders".format(accounts.trading_base(keys)), keys=keys, headers=hdr,
                           json=body, timeout=8)
    except requests.RequestException as e:
        raise BrokerError("could not reach Alpaca: {}".format(e))
    if r.status_code not in (200, 201):
        raise BrokerError("Alpaca rejected the order (HTTP {}): {}".format(
            r.status_code, (r.text or "")[:300]))
    return r.json()


def get_order(order_id: str, keys: tuple[str, str] | None = None) -> dict:
    return _get("/orders/{}".format(order_id), keys, what="order lookup")


def get_order_by_client_id(client_order_id: str, keys: tuple[str, str] | None = None) -> dict | None:
    """The order submitted with this client_order_id, or None if Alpaca has
    never seen it (the submit never got through)."""
    hdr = _headers(keys)
    if not hdr:
        raise BrokerError("no Alpaca keys saved")
    try:
        r = api_meter.get("{}/orders:by_client_order_id".format(accounts.trading_base(keys)), keys=keys,
                          headers=hdr, params={"client_order_id": client_order_id},
                          timeout=8)
    except requests.RequestException as e:
        raise BrokerError("could not reach Alpaca: {}".format(e))
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise BrokerError("Alpaca order lookup failed (HTTP {})".format(r.status_code))
    return r.json()


def cancel_order(order_id: str, keys: tuple[str, str] | None = None) -> None:
    hdr = _headers(keys)
    if not hdr or not order_id:
        return
    try:
        api_meter.delete("{}/orders/{}".format(accounts.trading_base(keys), order_id), keys=keys,
                         headers=hdr, timeout=8)
    except requests.RequestException:
        pass


def fill_of(order: dict) -> tuple[float, float | None]:
    """(filled contracts, average fill price per share) from an order record.
    An mleg fill's price is the net per spread; a credit comes back as a
    positive number so the ledger's sell price is always > 0."""
    try:
        filled_qty = float(order.get("filled_qty") or 0)
    except (TypeError, ValueError):
        filled_qty = 0.0
    avg = None
    if filled_qty > 0:
        try:
            avg = abs(float(order.get("filled_avg_price") or 0)) or None
        except (TypeError, ValueError):
            avg = None
    return filled_qty, avg


def new_client_order_id(bot_id: int, side: str) -> str:
    return "tm-b{}-{}-{}".format(int(bot_id), "bto" if side.startswith("buy") else "stc",
                                 int(time.time() * 1000))


def submit_and_wait(symbol: str, qty: float, side: str, order_type: str = "market",
                    limit_price=None, stop_price=None, trail_percent=None,
                    position_intent: str | None = None,
                    timeout: float = 8.0, poll: float = 1.0,
                    keys: tuple[str, str] | None = None,
                    client_order_id: str | None = None):
    """Submit an order and poll until it fills, is otherwise terminal, or
    `timeout` elapses — at which point it is cancelled so nothing is left
    resting unmanaged on the account. Returns (filled_qty, avg_fill_price,
    order_id, error). `error` is None only if filled_qty > 0."""
    order = submit_order(symbol, qty, side, order_type, limit_price, stop_price,
                         trail_percent, position_intent, keys=keys,
                         client_order_id=client_order_id)
    oid = order.get("id")
    deadline = time.monotonic() + timeout
    while True:
        status = order.get("status")
        filled_qty = float(order.get("filled_qty") or 0)
        if status in _TERMINAL or filled_qty >= qty:
            break
        if time.monotonic() >= deadline:
            cancel_order(oid, keys=keys)
            try:
                order = get_order(oid, keys=keys)     # pick up anything filled before the cancel
            except BrokerError:
                pass
            break
        time.sleep(poll)
        try:
            order = get_order(oid, keys=keys)
        except BrokerError:
            break
    filled_qty, avg_price = fill_of(order)
    if filled_qty > 0 and avg_price is None:
        avg_price = 0.01                    # a reported fill with no price — never lose the fill
    err = None if filled_qty > 0 else "order not filled ({}) — cancelled".format(order.get("status"))
    return filled_qty, avg_price, oid, err


def order_age_s(order: dict, now: datetime | None = None) -> float | None:
    try:
        sub = datetime.fromisoformat((order.get("submitted_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - sub).total_seconds()
