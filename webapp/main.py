"""FastAPI application: one page per section of the desktop screener."""

import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import engine
from contextlib import asynccontextmanager

from . import (account_book, accounts, api_meter, auth, bot_settings, bots, broker, config, gene_pool, incubator, manager,
               genome, guardrails, jobs, journal, market_data, option_quotes, quick_add, recorder,
               watchdog, week)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Watch the trading engine from OUTSIDE it.

    The engine runs as one-shot processes fired by a systemd timer; this is
    a long-lived service with Restart=on-failure. Hosting the check here is
    the whole point — an alarm living inside the cycle cannot fire when the
    cycle is what stopped, which is precisely how a halt went unnoticed for
    three and a half hours on 2026-09-09."""
    watchdog.start()
    _start_view_warmer()
    yield


def _warm_views_once() -> None:
    """Recompute what the fleet / pool pages read through TTL caches, so a
    page load finds them warm instead of paying for the scan itself."""
    uids = [u["id"] for u in auth.list_users() if not u.get("disabled")]
    for uid in uids:
        bots.leaderboard("active", pool="fleet", user_id=uid, fresh=True)
        bots.leaderboard("active", pool="custom", user_id=uid, fresh=True)
    bots.leaderboard("active", pool="fleet", user_id=None, fresh=True)   # the Shared Pool tiles
    gene_pool.pending_view(20, fresh=True)
    gene_pool.coverage(fresh=True)      # also refreshes the pool scoreboard
    recorder.status(fresh=True)


def _start_view_warmer(every: float = 30.0) -> None:
    import threading
    import time as _t

    def loop():
        while True:
            try:
                _warm_views_once()
            except Exception:
                pass            # a failed warm just leaves the page to compute it
            _t.sleep(every)

    def start():
        try:
            _warm_views_once()           # fill every cache before readers may serve stale ones
        except Exception:
            pass
        bots.WARMER_ON = True
        loop()

    threading.Thread(target=start, name="view-warmer", daemon=True).start()


app = FastAPI(title=config.APP_TITLE, docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=_lifespan)

_APP_ROOT = Path(__file__).resolve().parent          # .../webapp
app.mount("/static", StaticFiles(directory=str(_APP_ROOT / "static")), name="static")

templates = Jinja2Templates(directory=str(_APP_ROOT / "templates"))
templates.env.globals["APP_TITLE"] = config.APP_TITLE
templates.env.globals["WEEK_TOP_N"] = config.WEEK_TOP_N
try:
    templates.env.globals["ASSET_VER"] = str(int(
        (_APP_ROOT / "static" / "style.css").stat().st_mtime))
except OSError:
    templates.env.globals["ASSET_VER"] = "1"


_DISPLAY_TZ = ZoneInfo("America/New_York")   # EST / EDT


def _localstamp(iso: str) -> str:
    """ISO UTC timestamp -> e.g. '8:11pm 27/08/26 EDT' in US Eastern time."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso).astimezone(_DISPLAY_TZ)
    except ValueError:
        return iso
    clock = dt.strftime("%I:%M%p").lstrip("0").lower()
    return f"{clock} {dt.strftime('%d/%m/%y')} {dt.strftime('%Z')}"


def _money(v):
    if v is None:
        return "—"
    return "-${:,.2f}".format(abs(v)) if v < 0 else "${:,.2f}".format(v)


def _pct(v, places=1):
    return "—" if v is None else "{:.{p}f}%".format(float(v), p=places)


def _num(v, places=2):
    return "—" if v is None else "{:.{p}f}".format(float(v), p=places)


def _formval(v):
    """A stored value as it should sit in a form input: empty for nothing,
    and a whole number without the float's trailing ".0" — 395, not 395.0,
    which is what a strike or a share count actually is."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


templates.env.filters["localstamp"] = _localstamp
templates.env.filters["money"] = _money
templates.env.filters["pct"] = _pct
templates.env.filters["num"] = _num
templates.env.filters["formval"] = _formval

auth.ensure_initialised()
# Data that pre-dates per-user fleets (journals, the shared fleet, its
# baselines, its Alpaca keys, custom bots and the custom-bot capital pool)
# belongs to the original admin after the upgrade: the fleet that was
# already running becomes that admin's own and keeps trading the same
# account, with nothing to re-enter.
_ADMIN_ID = auth.primary_admin_id()
journal.init(default_owner=_ADMIN_ID)
bots.init(default_owner=_ADMIN_ID)
bot_settings.migrate_pool_to(_ADMIN_ID)
bot_settings.migrate_fleet_to(_ADMIN_ID)
_RETIRED_SIM = bot_settings.retire_simulated_fleets()
if _RETIRED_SIM:
    bots.log_event("simulated_retired", {
        "user_ids": _RETIRED_SIM,
        "note": "simulated execution removed; these fleets now trade on their Alpaca "
                "account. Their ledgers mix modeled and real fills — reset the fleet "
                "to judge genomes under one set of rules."})
for _uid_ in bot_settings.fleet_user_ids() or [_ADMIN_ID]:
    bots.ensure_baseline_bots(_uid_)

_OPEN_PATHS = ("/login", "/logout", "/static", "/healthz", "/favicon.ico")

# Every user runs their OWN fleet: /bots and everything under it acts on the
# signed-in account's fleet, its own Alpaca keys and its own guardrails, so
# none of it is admin-only any more — the routes scope by owner instead (see
# _fleet_uid). What stays admin-only is what is genuinely installation-wide:
# accounts, the house guardrail defaults, the master engine switch and the
# shared gene pool's controls.
_ADMIN_PATHS = frozenset((
    "/pool/reset", "/pool/engine",
))
_ADMIN_PREFIXES = ("/guardrails/house", "/admin")


def _admin_only(path: str) -> bool:
    return path in _ADMIN_PATHS or any(
        path == p or path.startswith(p + "/") for p in _ADMIN_PREFIXES)


def _forbidden(request: Request, what: str = "That page is for admins only."):
    if request.method == "GET" and "text/html" in (request.headers.get("accept") or "text/html"):
        return render(request, "forbidden.html", active="", message=what, status_code=403)
    return JSONResponse({"error": what}, status_code=403)


# NOTE: register the guard BEFORE SessionMiddleware so the session middleware
# ends up outermost and has populated request.session by the time guard runs.
@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/static"):
        return await call_next(request)

    user = auth.current_user(request)
    request.state.user = user
    if any(path == p or path.startswith(p + "/") for p in _OPEN_PATHS):
        return await call_next(request)

    if user is None:
        request.session.clear()      # a stale / revoked session cookie
        if request.method == "GET":
            return RedirectResponse("/login?next=" + quote(path), status_code=303)
        return JSONResponse({"error": "not authenticated"}, status_code=401)

    if user.get("must_change") and path != "/change-password":
        if request.method == "GET":
            return RedirectResponse("/change-password", status_code=303)
        return JSONResponse({"error": "password change required"}, status_code=403)

    if _admin_only(path) and not auth.is_admin(user):
        return _forbidden(request)

    return await call_next(request)


app.add_middleware(
    SessionMiddleware,
    secret_key=config.SECRET_KEY,
    session_cookie="trade_session",
    same_site="lax",
    https_only=config.SECURE_COOKIE,
    max_age=60 * 60 * 24 * 14,
)


def render(request: Request, _template: str, status_code: int = 200, **ctx):
    user = getattr(request.state, "user", None)
    # when this render happened, in the display timezone — pages that are
    # served through a cache show it next to Refresh, so it is obvious
    # whether what you are looking at is current
    ctx.setdefault("rendered_at",
                   datetime.now(_DISPLAY_TZ).strftime("%I:%M:%S%p").lstrip("0").lower())
    ctx.setdefault("active", "")
    ctx.setdefault("user", user)
    ctx.setdefault("is_admin", auth.is_admin(user))
    ctx.setdefault("must_change", bool(user and user.get("must_change")))
    return templates.TemplateResponse(request, _template, ctx, status_code=status_code)


def _user(request: Request) -> dict:
    return request.state.user


def _uid(request: Request) -> int:
    return int(request.state.user["id"])


def _fleet_uid(request: Request) -> int:
    """Whose fleet a /bots request is about: your own, or — for an admin
    passing ?user=N — that account's, so an admin can see and support any
    fleet without a second sign-in. A non-admin's ?user= is ignored rather
    than refused: there is nothing to steal, they simply get their own."""
    me = _uid(request)
    if not auth.is_admin(_user(request)):
        return me
    raw = request.query_params.get("user")
    try:
        uid = int(raw) if raw not in (None, "", "me") else me
    except ValueError:
        return me
    return uid if auth.get_user(uid) else me


def _fleet_view_uid(request: Request) -> int:
    """Whose fleet a READ-ONLY /bots view is about. Any signed-in user may
    look at any account's fleet: the fleets share one gene pool, so seeing
    what another cohort is doing is the point of the pool being shared.

    Deliberately separate from _fleet_uid, which stays narrow. _fleet_uid is
    the authorisation boundary for every write — /bots/settings, /bots/reset,
    /bots/run, /guardrails — none of which are admin-gated in the middleware;
    they are safe purely because a non-admin's ?user= resolves to their own
    id there. Widening that one function instead of adding this one would
    have handed every user the ability to overwrite another account's Alpaca
    keys or retire their fleet."""
    me = _uid(request)
    raw = request.query_params.get("user")
    try:
        uid = int(raw) if raw not in (None, "", "me") else me
    except ValueError:
        return me
    u = auth.get_user(uid)
    return uid if u and not u.get("disabled") else me


def _fleet_qs(uid: int, request: Request) -> str:
    """`?user=N` to carry an admin's fleet selection through a redirect."""
    return "?user={}".format(uid) if uid != _uid(request) else ""


# ---------------------------------------------------------------------------
# health / auth
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}


def _safe_next(next: str) -> str:
    return next if next.startswith("/") and not next.startswith("//") else "/"


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/", error: str = ""):
    if _user(request):
        return RedirectResponse("/", status_code=303)
    return render(request, "login.html", next=_safe_next(next), error=error,
                  username="", active="login")


@app.post("/login")
def login_submit(request: Request, username: str = Form(""),
                 password: str = Form(...), next: str = Form("/")):
    ip = request.client.host if request.client else "?"
    wait = auth.throttled_seconds(ip)
    if wait:
        time.sleep(wait)

    user = auth.authenticate(username, password)
    if not user:
        auth.register_failure(ip)
        time.sleep(0.5)
        return render(request, "login.html", next=_safe_next(next), active="login",
                      username=auth.normalise_username(username),
                      error="Incorrect username or password.")

    auth.login_session(request, user)
    if user.get("must_change"):
        return RedirectResponse("/change-password", status_code=303)
    return RedirectResponse(_safe_next(next), status_code=303)


@app.get("/logout")
def logout(request: Request):
    auth.logout_session(request)
    return RedirectResponse("/login", status_code=303)


@app.get("/change-password", response_class=HTMLResponse)
def change_pw_form(request: Request, error: str = "", done: str = ""):
    return render(request, "change_password.html", error=error, done=done,
                  active="", forced=bool(_user(request).get("must_change")))


@app.post("/change-password")
def change_pw_submit(request: Request,
                     current: str = Form(""),
                     new: str = Form(...),
                     confirm: str = Form(...)):
    user = _user(request)
    forced = bool(user.get("must_change"))
    if not forced and not auth.check_password(user["id"], current):
        return render(request, "change_password.html", forced=forced,
                      error="Current password is incorrect.")
    if new != confirm:
        return render(request, "change_password.html", forced=forced,
                      error="New passwords do not match.")
    err = auth.password_error(new)
    if err:
        return render(request, "change_password.html", forced=forced, error=err)
    # a new password signs out every other session for this account (a
    # temporary one an admin handed out, a forgotten browser) — then re-syncs
    # this one so the user stays put
    auth.set_password(user["id"], new, must_change=False, sign_out_elsewhere=True)
    auth.refresh_session(request, user["id"])
    return RedirectResponse("/?changed=1", status_code=303)


# ---------------------------------------------------------------------------
# admin: accounts
# ---------------------------------------------------------------------------
def _flash(request: Request, **kv) -> None:
    request.session["flash"] = kv


def _pop_flash(request: Request) -> dict:
    return request.session.pop("flash", None) or {}


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    rows = []
    for u in auth.list_users():
        rows.append({**u,
                     "journals": journal.count_journals(u["id"]),
                     "custom_bots": len(bots.custom_bots(u["id"]))})
    return render(request, "admin_users.html", active="admin", users=rows,
                  me=_uid(request), roles=auth.ROLES, flash=_pop_flash(request),
                  primary_admin_id=auth.primary_admin_id(),
                  default_username=config.DEFAULT_USERNAME,
                  min_password=auth.MIN_PASSWORD_LEN)


@app.post("/admin/users/new")
def admin_user_create(request: Request, username: str = Form(""),
                      password: str = Form(""), role: str = Form("user")):
    role = role if role in auth.ROLES else "user"
    generated = not password.strip()
    if generated:
        password = secrets.token_urlsafe(9)
    try:
        u = auth.create_user(username, password, role=role, must_change=True)
    except ValueError as e:
        _flash(request, error=str(e), username=auth.normalise_username(username))
        return RedirectResponse("/admin/users", status_code=303)
    journal.ensure_defaults(u["id"])
    _flash(request, ok="Created {} ({}).".format(u["username"], role),
           temp_for=u["username"], temp_password=password)
    return RedirectResponse("/admin/users", status_code=303)


def _admin_target(request: Request, uid: int) -> dict | None:
    return auth.get_user(uid)


@app.post("/admin/users/{uid}/reset-password")
def admin_user_reset(request: Request, uid: int):
    u = _admin_target(request, uid)
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    temp = auth.reset_password(uid)
    if uid == _uid(request):
        auth.refresh_session(request, uid)      # keep the admin's own session alive
    _flash(request, ok="Password reset for {}. They must choose a new one at next sign-in."
           .format(u["username"]), temp_for=u["username"], temp_password=temp)
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/username")
def admin_user_rename(request: Request, uid: int, username: str = Form("")):
    """Rename an account. The username is both the login and the name shown
    around the UI, so this changes both at once.

    Renaming the primary admin is safe: auth.authenticate() keeps accepting
    the default name for that account forever, so the old name still signs
    in and a forgotten new one cannot lock the installation out."""
    u = _admin_target(request, uid)
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    old_name = u["username"]
    try:
        renamed = auth.set_username(uid, username)
    except ValueError as e:
        _flash(request, error=str(e))
        return RedirectResponse("/admin/users", status_code=303)
    if renamed["username"] == old_name:
        return RedirectResponse("/admin/users", status_code=303)
    msg = "Renamed {} to {}.".format(old_name, renamed["username"])
    if uid == auth.primary_admin_id():
        msg += ' They can sign in as "{}" or as "{}".'.format(
            renamed["username"], config.DEFAULT_USERNAME)
    else:
        msg += " They sign in with the new name from now on."
    _flash(request, ok=msg)
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/role")
def admin_user_role(request: Request, uid: int, role: str = Form(...)):
    u = _admin_target(request, uid)
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    if uid == _uid(request):
        _flash(request, error="You can't change your own role — ask another admin.")
        return RedirectResponse("/admin/users", status_code=303)
    try:
        auth.set_role(uid, role)
        _flash(request, ok="{} is now {}.".format(u["username"], "an admin" if role == "admin" else "a user"))
    except ValueError as e:
        _flash(request, error=str(e))
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/disable")
def admin_user_disable(request: Request, uid: int, disabled: str = Form("1")):
    u = _admin_target(request, uid)
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    flag = disabled in ("1", "on", "true", "yes")
    if uid == _uid(request) and flag:
        _flash(request, error="You can't disable the account you're signed in with.")
        return RedirectResponse("/admin/users", status_code=303)
    try:
        auth.set_disabled(uid, flag)
        _flash(request, ok="{} {}.".format(u["username"], "disabled" if flag else "re-enabled"))
    except ValueError as e:
        _flash(request, error=str(e))
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/delete")
def admin_user_delete(request: Request, uid: int):
    u = _admin_target(request, uid)
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    if uid == _uid(request):
        _flash(request, error="You can't delete the account you're signed in with.")
        return RedirectResponse("/admin/users", status_code=303)
    try:
        auth.delete_user(uid)
    except ValueError as e:
        _flash(request, error=str(e))
        return RedirectResponse("/admin/users", status_code=303)
    nj = journal.delete_user_journals(uid)
    nb = bots.delete_user_bots(uid)
    bot_settings.delete_user(uid)
    guardrails.delete_user(uid)
    _flash(request, ok="Deleted {} along with {} journal{} and {} bot{} (their whole "
                       "fleet). The genomes their fleet was testing stay in the shared "
                       "pool.".format(u["username"], nj, "" if nj == 1 else "s",
                                      nb, "" if nb == 1 else "s"))
    return RedirectResponse("/admin/users", status_code=303)


@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/bots", status_code=303)


# ---------------------------------------------------------------------------
# 0DTE options bot fleet
# ---------------------------------------------------------------------------
_ACCOUNT_CACHE: dict[str, tuple[float, dict]] = {}
_ACCOUNT_TTL = 60.0


def _account_summary(keys) -> dict | None:
    """The real Alpaca account behind `keys` for a page header — cached a
    minute so page loads don't hammer the API. {"error": ...} on failure."""
    if not keys:
        return None
    ak = broker.account_key(keys)
    hit = _ACCOUNT_CACHE.get(ak)
    if hit and time.time() - hit[0] < _ACCOUNT_TTL:
        return hit[1]
    try:
        out = broker.account(keys)
    except broker.BrokerError as e:
        out = {"error": str(e)}
    _ACCOUNT_CACHE[ak] = (time.time(), out)
    return out


# Query parameters that describe WHAT the page is showing (as opposed to a
# one-shot banner like atest / reset), so Refresh lands back on the same
# view rather than resetting the user's filters.
_VIEW_PARAMS = ("user", "scope", "sort", "strategy", "symbol", "proven",
                "style", "q", "value")


def _same_view(path: str, request: Request) -> str:
    keep = [(k, v) for k in _VIEW_PARAMS
            for v in [request.query_params.get(k)] if v not in (None, "")]
    return path + ("?" + urlencode(keep) if keep else "")


def refresh_view_caches() -> None:
    """Drop every TTL cache the fleet / Custom Bots pages read through, so
    the next render is computed from the current database and a live call to
    Alpaca rather than from a snapshot up to a minute old.

    The caches are there because a page load would otherwise be one pass
    over every bot's ledger per row; this is the escape hatch for a user who
    has just changed something and wants to see it now. Deliberately NOT a
    reconcile: that writes to the account's ledger report and costs an
    Alpaca round trip per account, and it is the engine cycle's job."""
    bots.invalidate_caches()        # leaderboard rows + the fitness inputs
    gene_pool.invalidate()          # the pooled columns on every fleet row
    _ACCOUNT_CACHE.clear()          # the live Alpaca account tile


def _own_pending(uid: int) -> list[dict]:
    """Journaled-but-unbooked orders belonging to one account's bots. The
    banner is about that account's Alpaca account, so another user's
    unbooked fills must not be counted into it."""
    mine = {b["id"] for b in bots.user_bots(uid)}
    return [p for p in bots.pending_orders() if p.get("bot_id") in mine]


def _incubator_view(uid: int, sim_sort: str | None) -> dict:
    """The incubator summary with its leaderboard in the order asked for:
    'fitness' (default) or 'pl'."""
    view = incubator.summary(uid)
    by_pl = (sim_sort or "").lower() == "pl"
    view["sim_sort"] = "pl" if by_pl else "fitness"
    top = view.get("top") or []
    # a summary stored before the P&L list existed: reorder what it has until the next pass
    view["leaders"] = (view.get("top_pl") or sorted(top, key=lambda t: t.get("pl_pct") or 0.0,
                                                    reverse=True)) if by_pl else top
    return view


def _fleet_owner_ctx(request: Request, uid: int) -> dict:
    """Whose fleet is on screen, and whether that is somebody else's — the
    template shows a banner when an admin is looking at another account."""
    me = _uid(request)
    owner = auth.get_user(uid) or {}
    return {
        "fleet_uid": uid,
        "fleet_owner": owner.get("username") or "account {}".format(uid),
        "is_own_fleet": uid == me,
        "fleet_qs": _fleet_qs(uid, request),
        # everyone can pick a fleet to look at; only its owner (or an admin)
        # gets the controls that write to it
        "other_fleets": [{"id": u["id"], "username": u["username"]}
                         for u in auth.list_users() if not u.get("disabled")],
        "can_manage_fleet": uid == me or auth.is_admin(_user(request)),
    }


@app.get("/bots", response_class=HTMLResponse)
def bots_overview(request: Request):
    q = request.query_params
    scope = q.get("scope", "active")
    scope = scope if scope in ("active", "all", "retired") else "active"
    sort = q.get("sort", "profit_desc")
    sort = sort if sort in bots.BOT_SORTS else "profit_desc"
    fstrategy = q.get("strategy", "all")
    fsymbol = q.get("symbol", "all")
    fproven = q.get("proven", "all")
    fstyle = q.get("style", "all")
    # capped: this is echoed back into the filter form and every tab link
    fquery = (q.get("q") or "").strip()[:64]
    # which number each bot card shows. Purely a view over the same rows —
    # it never filters or reorders them, and the scoreboard above the grid
    # is unaffected, so the two views are always of the same fleet.
    fvalue = q.get("value", "realized")
    fvalue = fvalue if fvalue in ("realized", "marked") else "realized"

    uid = _fleet_view_uid(request)
    bots.ensure_baseline_bots(uid)
    all_rows = bots.leaderboard(scope, pool="fleet", user_id=uid)
    rows = bots.filter_sort_rows(all_rows, sort=sort, strategy=fstrategy,
                                 symbol=fsymbol, proven=fproven, style=fstyle,
                                 q=fquery, marked=fvalue == "marked")
    lim = guardrails.load(uid)
    settings = bot_settings.public_fleet(uid)
    fleet_keys = bot_settings.fleet_keys(uid)
    acct = _account_summary(fleet_keys)
    return render(request, "bots.html", active="bots", rows=rows, scope=scope,
                  **_fleet_owner_ctx(request, uid),
                  engine_on=settings["running"],
                  account=acct,
                  capital=account_book.capital_summary(
                      all_rows, lim["position_pct_limit"],
                      acct if acct and not acct.get("error") else None),
                  reconcile=account_book.last_report(fleet_keys),
                  api=api_meter.stats(fleet_keys),
                  pending_count=len(_own_pending(uid)),
                  total_rows=len(all_rows),
                  sort=sort, sort_options=[(k, v[0]) for k, v in bots.BOT_SORTS.items()],
                  filters={"strategy": fstrategy, "symbol": fsymbol, "proven": fproven,
                           "style": fstyle, "q": fquery, "value": fvalue},
                  strategy_names=("all",) + genome.STRATEGIES,
                  symbol_names=("all",) + genome.SYMBOLS,
                  style_names=("all",) + genome.STYLES,
                  counts=bots.counts(pool="fleet", user_id=uid),
                  strategy_mix=bots.strategy_mix(pool="fleet", user_id=uid),
                  families=bots.family_leaderboard("fleet", user_id=uid),
                  fleet=bots.fleet_stats(uid), baseline=bots.baseline_stats(uid),
                  # the fleet's own Alpaca connection — the owner's business
                  settings=settings,
                  house_engine_on=bot_settings.engine_enabled(),
                  pool_coverage=gene_pool.coverage(),
                  incubator=_incubator_view(uid, request.query_params.get("sim_sort")),
                  option_sources={s: option_quotes.status_label(s) for s in genome.SYMBOLS},
                  history=recorder.status(),
                  atest=request.query_params.get("atest"),
                  otest=request.query_params.get("otest"),
                  reset_count=request.query_params.get("reset"),
                  spawned=request.query_params.get("spawned"),
                  spawn_asked=request.query_params.get("spawn_asked"),
                  custom_count=len(bots.custom_bots(uid)),
                  limits={
                      "start": bots.START_BALANCE,
                      "drawdown": lim["max_drawdown_pct"],
                      "position": lim["position_pct_limit"],
                      "min_contracts": lim["min_contracts"],
                      "fleet_cap": bots.fleet_cap(uid),
                      "fleet_headroom": bots.fleet_headroom(uid),
                      "poll_seconds": config.BOT_POLL_SECONDS,
                      "archive_peak": bots.ARCHIVE_PEAK_PCT,
                      "elig_trades": bots.MIN_ELIGIBLE_TRADES,
                      "elig_days": bots.MIN_ELIGIBLE_DAYS,
                      "elig_regimes": bots.MIN_ELIGIBLE_REGIMES,
                      "bots_tested": bots.bots_tested(),
                      "proven_threshold": bots.proven_threshold(),
                  })


@app.post("/bots/refresh")
def bots_refresh(request: Request):
    """Rebuild the fleet page from the current state — every bot's ledger
    re-read, its stats recomputed, the pooled columns re-derived and the
    Alpaca account re-queried — then land back on the same view.

    A browser reload alone would not do this: the page is served through
    caches with 10-60 second lifetimes, so pressing F5 inside that window
    re-renders the same numbers."""
    refresh_view_caches()
    return RedirectResponse(_same_view("/bots", request), status_code=303)


@app.post("/bots/custom/refresh")
def custom_bots_refresh(request: Request):
    """The same, for the Custom Bots page."""
    refresh_view_caches()
    return RedirectResponse(_same_view("/bots/custom", request), status_code=303)


@app.get("/bots/leaderboard.json")
def bots_leaderboard_json(request: Request):
    q = request.query_params
    rows = bots.leaderboard(q.get("scope", "active"), user_id=_fleet_view_uid(request))
    rows = bots.filter_sort_rows(rows, sort=q.get("sort", "profit_desc"),
                                 strategy=q.get("strategy", "all"),
                                 symbol=q.get("symbol", "all"),
                                 proven=q.get("proven", "all"),
                                 style=q.get("style", "all"),
                                 q=(q.get("q") or "").strip()[:64])
    return JSONResponse(rows)


@app.get("/bots/archive", response_class=HTMLResponse)
def bots_archive(request: Request):
    """The viewer's own archived bots. An archived bot still carries its
    owner's final balance and P&L, so it stays as private as it was alive;
    what everyone can see about a genome is on /pool."""
    from . import genome as _g
    uid = _fleet_uid(request)
    rows = bots.list_archived(user_id=uid)
    for r in rows:
        r["genome_desc"] = _g.describe(_g.loads(r.get("genome")))
    return render(request, "bot_archive.html", active="bots", rows=rows,
                  **_fleet_owner_ctx(request, uid))


@app.get("/bots/history", response_class=HTMLResponse)
def bots_history(request: Request):
    """The engine log as it concerns this account: events tagged with their
    id, plus the installation-wide ones. Another account's fleet events are
    theirs."""
    uid = _fleet_uid(request)
    return render(request, "bot_history.html", active="bots",
                  events=bots.recent_events(200, user_id=uid),
                  max_events=bots._MAX_EVENTS,
                  **_fleet_owner_ctx(request, uid))


@app.get("/guardrails", response_class=HTMLResponse)
def guardrails_page(request: Request):
    """Your own fleet's risk limits. Every user sets their own — it is
    their account and their money — starting from the house defaults an
    admin sets below, and clamped either way by the absolute bounds in
    config.py that nothing in the UI can cross."""
    uid = _fleet_uid(request)
    return render(request, "guardrails.html", active="guardrails",
                  **_fleet_owner_ctx(request, uid),
                  lim=guardrails.load(uid), house=guardrails.house(),
                  overrides=guardrails.overrides_of(uid),
                  bounds=guardrails.bounds(),
                  fleet_size=len(bots.active_bots(pool="fleet", user_id=uid)))


@app.post("/guardrails")
def guardrails_save(request: Request, max_drawdown_pct: str = Form(""),
                    position_pct_limit: str = Form(""),
                    min_contracts: str = Form(""), max_contracts: str = Form(""),
                    max_fleet_bots: str = Form("")):
    """Save one account's own limits. A field left blank CLEARS that
    override, so the limit goes back to inheriting the house default.

    Lowering max_fleet_bots never retires anything: bots already running
    keep running and their ledgers stay intact. The cap stops new spawns,
    and the fleet drifts down to it as evolution retires bots normally —
    a guardrail, not a cull."""
    uid = _fleet_uid(request)
    guardrails.save_user(uid, max_drawdown_pct=max_drawdown_pct,
                         position_pct_limit=position_pct_limit,
                         min_contracts=min_contracts, max_contracts=max_contracts,
                         max_fleet_bots=max_fleet_bots)
    return RedirectResponse("/guardrails" + _fleet_qs(uid, request), status_code=303)


@app.post("/guardrails/house")
def guardrails_house_save(request: Request, max_drawdown_pct: str = Form(""),
                          position_pct_limit: str = Form(""),
                          min_contracts: str = Form(""), max_contracts: str = Form(""),
                          max_fleet_bots: str = Form("")):
    """Admin: the installation-wide defaults every account inherits until
    it overrides them."""
    guardrails.save(max_drawdown_pct=max_drawdown_pct,
                    position_pct_limit=position_pct_limit,
                    min_contracts=min_contracts, max_contracts=max_contracts,
                    max_fleet_bots=max_fleet_bots)
    return RedirectResponse("/guardrails", status_code=303)


# ---------------------------------------------------------------------------
# custom bots — cloned, hand-tuned, run live, never touched by evolution
# ---------------------------------------------------------------------------
def _bot_for(request: Request, bid: int) -> dict | None:
    """The bot if the signed-in user may CHANGE it: an admin may touch any
    bot; a user only their own — every bot has an owner now, fleet bots
    included. None otherwise — every mutating /bots/{id} route turns that
    into a 403 via _bot_denied."""
    bot = bots.get_bot(bid)
    if not bot:
        return None
    if auth.is_admin(_user(request)):
        return bot
    return bot if bots.owner_of(bot) == _uid(request) else None


def _bot_view(request: Request, bid: int) -> dict | None:
    """The bot if the signed-in user may SEE it — which is the same set.

    A fleet bot used to be public because there was one shared fleet; now
    it is somebody's own bot on their own Alpaca account, and its ledger,
    fills and balance are as private as a journal. What IS shared, and
    visible to everyone, is the genome it runs and that genome's pooled
    record — see /pool."""
    return _bot_for(request, bid)


def _bot_denied(request: Request):
    return _forbidden(request, "That bot isn't yours.")


def _custom_bot_for(request: Request, bid: int) -> dict | None:
    bot = _bot_for(request, bid)
    return bot if bot and bot.get("pool") == "custom" else None


@app.get("/bots/custom", response_class=HTMLResponse)
def custom_bots_page(request: Request):
    uid = _uid(request)
    rows = bots.leaderboard("all", pool="custom", user_id=uid)
    for r in rows:
        r["guardrail_overrides"] = bots.guardrail_overrides_of(r)
    # clone from your own fleet or your own custom bots. The shared search
    # reaches you through the fleet — its bots run the population's genomes
    # — and any genome in the pool can be cloned straight off /pool.
    clone_sources = bots.leaderboard("active", pool="fleet", user_id=uid)[:40] + rows
    pool_total = bot_settings.load_user(uid).get("custom_pool_total", 0.0)
    pool_used = bots.custom_capital_used(uid)
    from . import bot_engine
    fast_age = bot_engine.fast_lane_age_s()
    accts = accounts.list_accounts(uid)
    # the account the page's numbers describe: the one a running bot is on,
    # else the first saved
    running = next((b for b in rows if b.get("enabled", 1) and b.get("account_id") is not None), None)
    shown = accounts.get(uid, running["account_id"]) if running else (accts[0] if accts else None)
    keys = accounts.keys_of(shown)
    acct = _account_summary(keys)
    lim = guardrails.load(uid)          # the owner's own limits, not the house defaults
    for r in rows:
        r["position_pct"] = bots.effective_guardrails(r)["position_pct_limit"]
        a = accounts.get(uid, r.get("account_id"))
        r["account_name"] = a["name"] if a else None
        r["account_live"] = bool(a and a["type"] == "live")
    return render(request, "bots_custom.html", active="custom_bots", rows=rows,
                  clone_sources=clone_sources, lim=lim,
                  accounts=[accounts.public(a) for a in accts],
                  shown_account=accounts.public(shown) if shown else None,
                  live_enabled=accounts.live_enabled(),
                  fast_lane_ok=fast_age is not None and fast_age < bot_engine.FAST_ALIVE_S,
                  fast_lane_age=fast_age, fast_interval=bot_engine.FAST_INTERVAL_S,
                  settings={**bot_settings.public_user(uid), "connected": bool(accts),
                            "same_as_fleet": False, "own_keys": bool(accts)},
                  account=acct,
                  capital=account_book.capital_summary(
                      rows, lim["position_pct_limit"],
                      acct if acct and not acct.get("error") else None),
                  reconcile=account_book.last_report(keys),
                  api=api_meter.stats(keys),
                  pool_total=pool_total, pool_used=pool_used,
                  pool_available=max(0.0, pool_total - pool_used),
                  err=request.query_params.get("err"))


@app.post("/bots/custom/pool")
def custom_bots_pool_save(request: Request, custom_pool_total: str = Form("0")):
    bot_settings.save_user(_uid(request), custom_pool_total=custom_pool_total)
    return RedirectResponse("/bots/custom", status_code=303)


# ---------------------------------------------------------------------------
# managers — N equal-money custom bots on one account, picked and replaced for you
# ---------------------------------------------------------------------------
def _manager_for(request: Request, mid: int) -> dict | None:
    m = manager.get(mid)
    if not m:
        return None
    return m if (auth.is_admin(_user(request)) or m["user_id"] == _uid(request)) else None


def _rules_from_form(form) -> dict:
    return {k: form.get("rule_" + k) for k in manager.DEFAULT_RULES if form.get("rule_" + k) not in (None, "")}


def _mgr_back(mid=None, err: str | None = None, ok: str | None = None) -> RedirectResponse:
    base = "/bots/managers" + ("/{}".format(mid) if mid else "")
    q = []
    if err:
        q.append("err=" + quote(err))
    if ok:
        q.append("ok=" + quote(ok))
    return RedirectResponse(base + ("?" + "&".join(q) if q else ""), status_code=303)


def _fill_now(mid: int) -> str:
    """Put a ready bot into every empty slot right now (what the manager's mode allows),
    and say what happened. Takes the engine lock so it can't race a scheduled pass."""
    from . import bot_engine
    m = manager.get(mid)
    if not m or not m["enabled"]:
        return ""
    if not manager.empty_slots(m):
        return ""
    with bot_engine._engine_lock(20.0) as locked:
        if not locked:
            return " — the engine is busy; empty slots will be checked again within 2 minutes"
        r = manager.fill_empty(mid, force=True).get(mid) or {}
    if r.get("error"):
        return " — could not fill slots: {}".format(r["error"])
    done = len(r.get("applied") or [])
    left = manager.empty_slots(manager.get(mid))
    if r.get("mode") == "dry_run":
        return " — dry run: {} slot(s) would be filled now (see History)".format(len(r.get("actions") or []))
    if r.get("mode") == "propose":
        return " — {} proposal(s) waiting for your approval".format(len(manager.proposals(mid)))
    if done:
        return " — filled {} slot(s) now{}".format(done, "" if not left else
                                                   "; {} still empty (nobody qualifies yet — it keeps checking)".format(left))
    return " — no fleet bot qualifies for the empty slot(s) yet; it checks every 2 minutes while the market is open"


@app.get("/bots/managers", response_class=HTMLResponse)
def managers_page(request: Request):
    uid = _uid(request)
    mgrs = manager.list_for(uid)
    crows = {r["id"]: r for r in bots.leaderboard("active", pool="custom", user_id=uid)}
    cards = []
    for m in mgrs:
        a = accounts.get(uid, m["account_id"])
        held = manager.slot_bots(m["id"])
        rows = [crows[b["id"]] for b in held if b["id"] in crows]
        cards.append({
            "m": m, "account": accounts.public(a) if a else None, "size": manager.slot_size(m),
            "cap": round(manager.slot_size(m) * m["position_pct"] / 100.0, 2),
            "filled": len(held), "rows": rows,
            "pl": sum(r.get("pl_dollar") or 0.0 for r in rows),
            "pending": len(manager.proposals(m["id"]))})
    pool_total = bot_settings.load_user(uid).get("custom_pool_total", 0.0)
    return render(request, "managers.html", active="managers", cards=cards,
                  accounts=[accounts.public(a) for a in accounts.list_accounts(uid)],
                  live_enabled=accounts.live_enabled(), modes=manager.MODES,
                  defaults=manager.DEFAULT_RULES, max_slots=manager.MAX_SLOTS, auto_days=manager.LIVE_AUTO_MIN_DAYS,
                  pool_total=pool_total, pool_available=max(0.0, pool_total - bots.custom_capital_used(uid)),
                  err=request.query_params.get("err"), ok=request.query_params.get("ok"))


@app.get("/bots/managers/preview")
def managers_preview(request: Request, slots: int = 4, total_budget: float = 1000.0,
                     position_pct: float = 25.0):
    """What this setup would look like against today's fleet — drives the form's live hint."""
    if not (1 <= slots <= manager.MAX_SLOTS) or total_budget <= 0 or not (1 <= position_pct <= 100):
        return JSONResponse({"error": "check the numbers"}, status_code=400)
    return JSONResponse(manager.preview(_uid(request), slots, total_budget, position_pct))


@app.post("/bots/managers/create")
async def managers_create(request: Request):
    uid = _uid(request)
    f = await request.form()
    cfg, err = manager.validate(uid, account_id=f.get("account_id"), slots=f.get("slots"),
                                total_budget=f.get("total_budget"), position_pct=f.get("position_pct"),
                                mode=f.get("mode") or "dry_run", confirm_name=f.get("confirm_name"))
    if err:
        return _mgr_back(err=err)
    mid = manager.create(uid, f.get("name") or "", cfg, _rules_from_form(f))
    return _mgr_back(mid, ok="manager created — {} mode{}".format(cfg["mode"].replace("_", " "), _fill_now(mid)))


@app.get("/bots/managers/{mid}", response_class=HTMLResponse)
def manager_page(request: Request, mid: int):
    m = _manager_for(request, mid)
    if not m:
        return _forbidden(request, "That manager isn't yours.")
    uid = m["user_id"]
    a = accounts.get(uid, m["account_id"])
    crows = {r["id"]: r for r in bots.leaderboard("active", pool="custom", user_id=uid)}
    by_slot = {b["slot"]: b for b in manager.slot_bots(mid)}
    slot_cards = []
    for s in range(1, m["slots"] + 1):
        b = by_slot.get(s)
        slot_cards.append({"slot": s, "bot": b, "row": crows.get(b["id"]) if b else None,
                           "source": bots.get_bot(b["parent_id"]) if b and b.get("parent_id") else None})
    try:
        plan = manager.plan(m)
    except Exception as e:  # noqa: BLE001 — the page still renders its history
        plan = {"actions": [], "notes": ["could not plan: {}".format(e)], "n_candidates": 0,
                "candidates": [], "skipped": {}, "size": manager.slot_size(m), "pct": m["position_pct"],
                "per_trade_cap": 0}
    return render(request, "manager_detail.html", active="managers", m=m,
                  rules=manager.rules_of(m), defaults=manager.DEFAULT_RULES,
                  account=accounts.public(a) if a else None, size=manager.slot_size(m),
                  cap=round(manager.slot_size(m) * m["position_pct"] / 100.0, 2),
                  slot_cards=slot_cards, plan=plan, events=manager.events(mid),
                  proposals=manager.proposals(mid), modes=manager.MODES,
                  accounts=[accounts.public(x) for x in accounts.list_accounts(uid)],
                  auto_ready=manager.live_auto_ready(mid), auto_days=manager.LIVE_AUTO_MIN_DAYS,
                  market_open=manager.market_open(), live_enabled=accounts.live_enabled(),
                  err=request.query_params.get("err"), ok=request.query_params.get("ok"))


@app.post("/bots/managers/{mid}/edit")
async def manager_edit(request: Request, mid: int):
    m = _manager_for(request, mid)
    if not m:
        return _bot_denied(request)
    f = await request.form()
    cfg, err = manager.validate(m["user_id"], account_id=f.get("account_id"), slots=f.get("slots"),
                                total_budget=f.get("total_budget"), position_pct=f.get("position_pct"),
                                mode=f.get("mode") or m["mode"], confirm_name=f.get("confirm_name"),
                                existing_id=mid)
    if err:
        return _mgr_back(mid, err=err)
    top = max([b["slot"] or 0 for b in manager.slot_bots(mid)] or [0])
    if cfg["slots"] < top:
        return _mgr_back(mid, err="slot {} is in use — retire or release that bot before going down to {} slots"
                         .format(top, cfg["slots"]))
    if cfg["account_id"] != m["account_id"] and manager.slot_bots(mid):
        return _mgr_back(mid, err="this manager has bots on its current account — delete the manager "
                                  "(its bots are kept) or retire them before moving to another account")
    manager.update(mid, cfg, name=f.get("name"), rules=_rules_from_form(f))
    for b in manager.slot_bots(mid):          # the per-trade % applies to the bots already running too
        bots.update_guardrails(b["id"], {"position_pct_limit": cfg["position_pct"], "min_contracts": 1})
    bots.invalidate_caches()
    return _mgr_back(mid, ok="saved — a changed budget applies to bots added from now on; "
                             "bots already running keep their allocation" + _fill_now(mid))


@app.post("/bots/managers/{mid}/run")
def manager_run(request: Request, mid: int):
    """'Run now' is always a dry run: it shows what the next pass would do and changes nothing."""
    m = _manager_for(request, mid)
    if not m:
        return _bot_denied(request)
    r = manager.run_manager(mid, dry=True)
    return _mgr_back(mid, ok="dry run done — {} action(s) would be taken".format(len(r.get("actions", [])))
                     if "error" not in r else None, err=r.get("error"))


@app.post("/bots/managers/{mid}/toggle")
def manager_toggle(request: Request, mid: int):
    m = _manager_for(request, mid)
    if not m:
        return _bot_denied(request)
    manager.set_enabled(mid, not m["enabled"])
    return _mgr_back(mid, ok=("manager resumed" + _fill_now(mid)) if not m["enabled"] else
                     "manager paused — its bots keep trading, it just stops picking and replacing")


@app.post("/bots/managers/{mid}/delete")
def manager_delete(request: Request, mid: int):
    m = _manager_for(request, mid)
    if not m:
        return _bot_denied(request)
    manager.delete(mid)
    return _mgr_back(ok="manager deleted — its bots were kept and are now ordinary custom bots")


@app.post("/bots/managers/proposals/{pid}/{verb}")
def manager_proposal(request: Request, pid: int, verb: str):
    if verb not in ("approve", "reject"):
        return RedirectResponse("/bots/managers", status_code=303)
    with bots._conn() as c:
        row = c.execute("SELECT manager_id FROM manager_proposals WHERE id=?", (pid,)).fetchone()
    m = _manager_for(request, row["manager_id"]) if row else None
    if not m:
        return _bot_denied(request)
    from . import bot_engine
    with bot_engine._engine_lock(30.0) as locked:      # applying can place real orders
        if not locked:
            return _mgr_back(m["id"], err="the engine is busy — try again in a moment")
        ok, msg = manager.resolve_proposal(pid, verb == "approve")
    return _mgr_back(m["id"], ok=msg if ok else None, err=None if ok else msg)


# ---------------------------------------------------------------------------
# accounts — the saved Alpaca accounts (paper or live) bots and managers run on
# ---------------------------------------------------------------------------
def _acct_back(err: str | None = None, ok: str | None = None) -> RedirectResponse:
    q = []
    if err:
        q.append("err=" + quote(err))
    if ok:
        q.append("ok=" + quote(ok))
    return RedirectResponse("/bots/accounts" + ("?" + "&".join(q) if q else ""), status_code=303)


@app.get("/bots/accounts", response_class=HTMLResponse)
def accounts_page(request: Request):
    uid = _uid(request)
    rows = []
    for a in accounts.list_accounts(uid):
        u = accounts.usage(uid, a["id"])
        pub = accounts.public(a)
        pub["key_hint"] = (a["key_id"][:6] + "…" + a["key_id"][-3:]) if len(a["key_id"]) > 10 else a["key_id"]
        rows.append({**pub, "bots": u["bots"], "managers": u["managers"], "holding": len(u["holding"])})
    f = bot_settings.public_fleet(uid)
    kid = f["alpaca_key_id"]
    fleet = {"connected": f["connected"],
             "key_hint": (kid[:6] + "…" + kid[-3:]) if len(kid) > 10 else kid,
             "running": f["running"], "engine_enabled": f["engine_enabled"],
             "house_engine_enabled": f["house_engine_enabled"], "data_feed": f["data_feed"],
             "feeds": bot_settings.DATA_FEEDS,
             "bots": len(bots.active_bots(pool="fleet", user_id=uid))}
    return render(request, "accounts.html", active="accounts", rows=rows, fleet=fleet,
                  live_enabled=accounts.live_enabled(),
                  err=request.query_params.get("err"), ok=request.query_params.get("ok"))


@app.get("/bots/accounts/fleet/status.json")
def fleet_account_status(request: Request):
    """Same read-out as account_status, for the fleet's own paper account. Registered
    before the {aid} route so "fleet" is not read as an account id."""
    keys = bot_settings.fleet_keys(_uid(request))
    if not keys:
        return JSONResponse({"error": "no fleet keys saved"})
    s = _account_summary(keys) or {}
    if s.get("error"):
        return JSONResponse({"error": s["error"]})
    return JSONResponse({k: s.get(k) for k in ("status", "account_number", "equity", "cash",
                                               "options_buying_power", "options_trading_level")})


@app.post("/bots/accounts/fleet/edit")
def fleet_account_edit(request: Request, key_id: str = Form(""), secret: str = Form("")):
    uid = _fleet_uid(request)
    try:
        accounts.set_fleet_keys(uid, key_id, secret)
    except accounts.AccountError as e:
        return _acct_back(err="Fleet account: {}".format(e))
    _ACCOUNT_CACHE.clear()
    bots.ensure_baseline_bots(uid)
    return _acct_back(ok="saved the fleet account's keys")


@app.post("/bots/accounts/fleet/test")
def fleet_account_test(request: Request):
    uid = _fleet_uid(request)
    a = accounts.fleet_account(uid)
    if not a:
        return _acct_back(err="Fleet account: no keys saved")
    try:
        j = accounts.verify("paper", a["key_id"], a["secret"])
    except accounts.AccountError as e:
        return _acct_back(err="Fleet account: {}".format(e))
    ok, msg = market_data.test_connection(keys=(a["key_id"], a["secret"]),
                                          feed=bot_settings.load_user(uid)["fleet_data_feed"])
    note = "Fleet account: account {} · {} · equity ${:,.2f} · data: {}".format(
        j.get("account_number", "?"), j.get("status", "?"), float(j.get("equity") or 0), msg)
    return _acct_back(ok=note) if ok else _acct_back(err=note)


@app.post("/bots/accounts/fleet/feed")
def fleet_account_feed(request: Request, data_feed: str = Form("iex")):
    uid = _fleet_uid(request)
    bot_settings.save_user(uid, fleet_data_feed=(data_feed if data_feed in bot_settings.DATA_FEEDS else "iex"))
    return _acct_back(ok="saved the fleet's data feed")


@app.post("/bots/accounts/fleet/disconnect")
def fleet_account_disconnect(request: Request):
    bot_settings.clear_fleet_keys(_fleet_uid(request))
    _ACCOUNT_CACHE.clear()
    return _acct_back(ok="disconnected the fleet account — the fleet stops trading until keys are saved again")


@app.get("/bots/accounts/{aid}/status.json")
def account_status(request: Request, aid: int):
    """Live Alpaca read-out for one saved account. The page fetches these after it
    loads so a slow or unreachable Alpaca never holds the page itself up."""
    a = accounts.get(_uid(request), aid)
    if not a:
        return JSONResponse({"error": "not found"}, status_code=404)
    if a["type"] == "live" and not accounts.live_enabled():
        return JSONResponse({"error": "live trading is switched off"})
    s = _account_summary(accounts.keys_of(a)) or {}
    if s.get("error"):
        return JSONResponse({"error": s["error"]})
    return JSONResponse({k: s.get(k) for k in ("status", "account_number", "equity", "cash",
                                               "options_buying_power", "options_trading_level")})


@app.post("/bots/accounts/add")
def accounts_add(request: Request, name: str = Form(""), kind: str = Form("paper"),
                 key_id: str = Form(""), secret: str = Form("")):
    try:
        a = accounts.add(_uid(request), name, kind, key_id, secret)
    except accounts.AccountError as e:
        return _acct_back(err=str(e))
    _ACCOUNT_CACHE.clear()
    return _acct_back(ok="added {} ({})".format(a["name"], a["type"]))


@app.post("/bots/accounts/{aid}/edit")
def accounts_edit(request: Request, aid: int, name: str = Form(""), key_id: str = Form(""),
                  secret: str = Form("")):
    try:
        a = accounts.update(_uid(request), aid, name=name, key_id=key_id, secret=secret)
    except accounts.AccountError as e:
        return _acct_back(err=str(e))
    _ACCOUNT_CACHE.clear()
    return _acct_back(ok="saved {}".format(a["name"]))


@app.post("/bots/accounts/{aid}/test")
def accounts_test(request: Request, aid: int):
    a = accounts.get(_uid(request), aid)
    if not a:
        return _acct_back(err="no such account")
    try:
        j = accounts.verify(a["type"], a["key_id"], a["secret"])
    except accounts.AccountError as e:
        return _acct_back(err="{}: {}".format(a["name"], e))
    return _acct_back(ok="{} ({}): account {} · {} · equity ${:,.2f}".format(
        a["name"], a["type"], j.get("account_number", "?"), j.get("status", "?"), float(j.get("equity") or 0)))


@app.post("/bots/accounts/{aid}/delete")
def accounts_delete(request: Request, aid: int):
    uid = _uid(request)
    a = accounts.get(uid, aid)
    if not a:
        return _acct_back(err="no such account")
    try:
        accounts.remove(uid, aid)
    except accounts.AccountError as e:
        return _acct_back(err=str(e))
    _ACCOUNT_CACHE.clear()
    return _acct_back(ok="deleted {}".format(a["name"]))


@app.post("/admin/live")
def admin_live_switch(request: Request, live_enabled: str = Form("")):
    """Admin: the live-trading master switch. Off is also the kill switch —
    any bot on a live account resolves to no keys and stops trading."""
    bot_settings.save(live_enabled=live_enabled in ("1", "on", "true", "yes"))
    return RedirectResponse("/bots/accounts", status_code=303)


def _check_allocation(uid: int, allocation: str):
    """(alloc, None) or (None, redirect) — the capital-pool gate both clone
    routes share, so a custom bot can never be funded past its owner's
    declared pool."""
    try:
        alloc = float(allocation)
    except (TypeError, ValueError):
        return None, RedirectResponse(
            "/bots/custom?err=" + quote("enter a starting allocation in dollars"),
            status_code=303)
    if alloc <= 0:
        return None, RedirectResponse(
            "/bots/custom?err=" + quote("allocation must be greater than zero"),
            status_code=303)
    pool_total = bot_settings.load_user(uid).get("custom_pool_total", 0.0)
    used = bots.custom_capital_used(uid)
    available = pool_total - used
    if alloc > available + 1e-6:
        return None, RedirectResponse(
            "/bots/custom?err=" + quote(
                "allocation of ${:,.2f} exceeds the ${:,.2f} left in your custom bot capital pool "
                "(${:,.2f} of ${:,.2f} already allocated) — raise the pool total or allocate less"
                .format(alloc, max(0.0, available), used, pool_total)),
            status_code=303)
    return alloc, None


@app.post("/bots/custom/clone")
def custom_bots_clone(request: Request, source_id: str = Form(...),
                      name: str = Form(""), allocation: str = Form("")):
    uid = _uid(request)
    try:
        sid = int(source_id)
    except ValueError:
        return RedirectResponse("/bots/custom", status_code=303)
    if not _bot_for(request, sid):
        return _bot_denied(request)
    alloc, bad = _check_allocation(uid, allocation)
    if bad is not None:
        return bad

    new_id = bots.clone_bot(sid, name=name or None, start_balance=alloc, user_id=uid)
    if new_id:
        return RedirectResponse(f"/bots/{new_id}", status_code=303)
    return RedirectResponse("/bots/custom", status_code=303)


@app.post("/bots/custom/clone-genome")
def custom_bots_clone_genome(request: Request, fingerprint: str = Form(...),
                             name: str = Form(""), allocation: str = Form("")):
    """Clone any genome out of the SHARED pool into a custom bot of your
    own — including one that only another user's fleet has ever run. The
    genome and its pooled record are the population's; the bot you get is
    yours, on your keys, with its own fresh ledger."""
    uid = _uid(request)
    alloc, bad = _check_allocation(uid, allocation)
    if bad is not None:
        return bad
    new_id = bots.clone_genome(fingerprint, name=name or None,
                               start_balance=alloc, user_id=uid)
    if new_id:
        return RedirectResponse(f"/bots/{new_id}", status_code=303)
    return RedirectResponse(
        "/bots/custom?err=" + quote("that genome is not in the shared pool"), status_code=303)


@app.post("/bots/{bid}/toggle")
def custom_bot_toggle(request: Request, bid: int):
    bot = _custom_bot_for(request, bid)
    if not bot:
        return _bot_denied(request)
    if bot.get("enabled", 1):
        bots.set_enabled(bid, False)
        return RedirectResponse(f"/bots/{bid}", status_code=303)
    ok, msg, paused = bots.start_bot(bid)
    if not ok:
        return RedirectResponse(f"/bots/{bid}?err=" + quote(msg), status_code=303)
    note = "" if not paused else "?err=" + quote(
        "started — paused bot(s) {} (one bot at a time per account)".format(
            ", ".join("#{}".format(i) for i in paused)))
    return RedirectResponse(f"/bots/{bid}" + note, status_code=303)


@app.post("/bots/{bid}/account")
def custom_bot_account(request: Request, bid: int, account_id: str = Form(""),
                       confirm_name: str = Form("")):
    """Put a custom bot on one of your saved accounts. It is left paused —
    Start is a separate click. A live account needs live trading switched on,
    the account's name typed back, and the bot's track record on paper."""
    bot = _custom_bot_for(request, bid)
    if not bot:
        return _bot_denied(request)
    uid = bots.owner_of(bot)
    back = f"/bots/{bid}"
    if account_id == "":
        new = None
    else:
        new = accounts.get(uid, account_id)
        if not new:
            return RedirectResponse(back + "?err=" + quote("no such account"), status_code=303)
    if (new["id"] if new else None) == bot.get("account_id"):
        return RedirectResponse(back, status_code=303)
    if bots.has_open_positions(bid):
        return RedirectResponse(back + "?err=" + quote(
            "this bot has open positions — close them before moving it to another account"),
            status_code=303)
    if new:
        err = bots.live_gate(bot, new, confirm_name)
        if err:
            return RedirectResponse(back + "?err=" + quote(err), status_code=303)
    bots.set_account(bid, new["id"] if new else None)
    return RedirectResponse(back, status_code=303)


@app.post("/bots/{bid}/genome")
async def custom_bot_genome(request: Request, bid: int):
    """Hand-edit a custom bot's rules (its genome). Every field is optional —
    only genes present in the submitted form are changed; a blank field is
    left as-is. Numeric bounds/repair happen in bots.update_genome()."""
    bot = _custom_bot_for(request, bid)
    if not bot:
        return _bot_denied(request)
    form = await request.form()
    updates: dict = {}
    for name in genome.NUM_GENES:
        v = form.get(name)
        if v not in (None, ""):
            try:
                updates[name] = float(v)
            except ValueError:
                pass
    for name, choices in (("strategy", genome.STRATEGIES), ("symbol", genome.SYMBOLS),
                          ("order_pref", genome.ORDER_PREFS),
                          ("strike_style", genome.STRIKE_STYLES),
                          ("scale_in_when", genome.SCALE_IN_MODES),
                          ("style", genome.STYLES), ("structure", genome.STRUCTURES)):
        v = form.get(name)
        if v in choices:
            updates[name] = v
    bots.update_genome(bid, updates)
    return RedirectResponse(f"/bots/{bid}", status_code=303)


@app.post("/bots/{bid}/guardrails")
def custom_bot_guardrails(request: Request, bid: int,
                          max_drawdown_pct: str = Form(""),
                          position_pct_limit: str = Form(""),
                          min_contracts: str = Form(""),
                          max_contracts: str = Form(""),
                          min_size_over_risk: str = Form(""),
                          clear: str = Form("")):
    bot = _custom_bot_for(request, bid)
    if not bot:
        return _bot_denied(request)
    if clear:
        bots.update_guardrails(bid, {})
    else:
        bots.update_guardrails(bid, {
            "max_drawdown_pct": max_drawdown_pct, "position_pct_limit": position_pct_limit,
            "min_contracts": min_contracts, "max_contracts": max_contracts,
            "min_size_over_risk": bool(min_size_over_risk)})
    return RedirectResponse(f"/bots/{bid}", status_code=303)


def _fleet_redirect(uid: int, request: Request, **params) -> RedirectResponse:
    """Back to the fleet page, keeping an admin's ?user= selection."""
    qs = [("user", str(uid))] if uid != _uid(request) else []
    qs += [(k, v) for k, v in params.items() if v is not None]
    tail = ("?" + "&".join("{}={}".format(k, quote(str(v))) for k, v in qs)) if qs else ""
    return RedirectResponse("/bots" + tail, status_code=303)


@app.post("/bots/settings")
def bots_settings_save(request: Request, engine_enabled: str = Form("")):
    """The fleet's engine switch. Keys and the data feed are connection settings
    and live only on the Accounts page."""
    uid = _fleet_uid(request)
    bot_settings.save_user(uid, fleet_engine_enabled=engine_enabled in ("1", "on", "true", "yes"))
    bots.ensure_baseline_bots(uid)
    return _fleet_redirect(uid, request)


@app.post("/bots/spawn")
def bots_spawn(request: Request, count: str = Form("10")):
    """Add candidates to your own fleet, drawn from the shared pool the same
    way evolution draws them — so a hand-added bot still tests something no
    other fleet is currently running.

    The max_fleet_bots guardrail binds here exactly as it does on evolution:
    ask for more than the fleet has room for and you get the room, not the
    number you asked for. bots.spawn_bot() refuses past the cap regardless,
    so this only decides how honest the button looks before it is pressed."""
    from . import bot_engine
    try:
        n = max(1, min(100, int(count)))
    except ValueError:
        n = 10
    uid = _fleet_uid(request)
    asked, spawned = n, 0
    for _ in range(n):
        if bot_engine._spawn_replacement(uid) is None:
            break               # at the cap — every further attempt is refused too
        spawned += 1
    return _fleet_redirect(uid, request, spawned=spawned,
                           spawn_asked=(asked if spawned < asked else None))


@app.post("/bots/run")
def bots_run(request: Request):
    """Kick a cycle for THIS fleet now (for testing between timer fires).

    Scoped to the one account: a user pressing their own button must never
    start placing orders on anybody else's Alpaca account. The engine's own
    lock keeps it from overlapping the timer's cycle."""
    from . import bot_engine
    uid = _fleet_uid(request)
    threading.Thread(target=bot_engine.run_cycle, kwargs={"user_ids": [uid]},
                     daemon=True).start()
    return _fleet_redirect(uid, request)


@app.post("/bots/options-test")
def bots_options_test(request: Request):
    ok, msg = option_quotes.test_connection()
    return _fleet_redirect(_fleet_uid(request), request,
                           otest=("1" if ok else "0") + msg)


@app.post("/bots/reset")
def bots_reset(request: Request):
    """Retire YOUR active fleet and reseed fresh candidates — for use right
    after a change to fill pricing, guardrails, or breeding rules, so your
    fleet's numbers reflect only the current rules (see bots.reset_fleet()).

    Only your own bots: other users' fleets keep running, and the shared
    gene pool keeps everything it has learned. An admin resetting the pool's
    multiple-comparison count for everybody does that on the Shared Pool
    page, deliberately separately."""
    uid = _fleet_uid(request)
    n = bots.reset_fleet("web_reset", user_id=uid)
    return _fleet_redirect(uid, request, reset=str(n),
                           spawned=len(bots.active_bots(pool="fleet", user_id=uid)))


# ---------------------------------------------------------------------------
# the shared gene pool — one search, every fleet
# ---------------------------------------------------------------------------
@app.get("/pool", response_class=HTMLResponse)
def pool_page(request: Request):
    """What every fleet is collectively learning: the genome population,
    each genome's POOLED record across all users, and who is feeding the
    shared market history."""
    rows = [dict(r) for r in gene_pool.scoreboard_view()]
    users = {u["id"]: u["username"] for u in auth.list_users()}
    mine = _uid(request)
    for r in rows:
        r["running_here"] = mine in (r.get("user_ids") or [])
        r["user_names"] = [users.get(u, "#{}".format(u)) for u in (r.get("user_ids") or [])]
    return render(request, "pool.html", active="pool", rows=rows[:200],
                  total_rows=len(rows),
                  # every fleet's active bots as one scoreboard — same tiles the
                  # /bots page shows for a single fleet, summed across all of them
                  fleet=bots.fleet_stats(),
                  my_uid=mine,
                  coverage=gene_pool.coverage(),
                  contributions=recorder.contributions(),
                  usernames=users,
                  history=recorder.status(),
                  pending=gene_pool.pending_view(20),
                  house_engine_on=bot_settings.engine_enabled(),
                  proven_threshold=bots.proven_threshold(),
                  elig_trades=bots.MIN_ELIGIBLE_TRADES,
                  bots_tested=bots.bots_tested())


@app.post("/pool/engine")
def pool_engine(request: Request, engine_enabled: str = Form("")):
    """Admin: the installation-wide master engine switch. Off stops every
    fleet at once, whatever each user has set for their own."""
    bot_settings.save(engine_enabled=engine_enabled in ("1", "on", "true", "yes"))
    return RedirectResponse("/pool", status_code=303)


@app.post("/pool/reset")
def pool_reset(request: Request):
    """Admin: restart the multiple-comparison count after a rules change
    that applies to everybody. The genomes and their pooled records are
    kept — that is the population's memory — but only trials from here on
    count toward the proven bar, so the new rules are judged on genomes
    tried under them."""
    gene_pool.reset_pool("admin_reset")
    return RedirectResponse("/pool", status_code=303)


def _bots_home(request: Request) -> str:
    return "/bots"


@app.get("/bots/{bid}", response_class=HTMLResponse)
def bot_detail(request: Request, bid: int):
    if not bots.get_bot(bid):
        return RedirectResponse(_bots_home(request), status_code=303)
    bot = _bot_view(request, bid)
    if not bot:
        return _bot_denied(request)
    can_edit = _bot_for(request, bid) is not None
    from . import bot_engine, genome as _g, strategies
    parent = bots.get_bot(bot["parent_id"]) if bot.get("parent_id") else None
    is_custom = bot.get("pool") == "custom"
    owner = bots.owner_of(bot)
    owner_user = auth.get_user(owner) if owner is not None else None
    parent_visible = bool(parent) and _bot_view(request, parent["id"]) is not None
    eff = bots.effective_guardrails(bot)
    bot_genome = bots._genome.loads(bot.get("genome"))
    live = {}
    try:
        ctx = bot_engine.live_ctx(bot_genome.get("symbol", "SPY"))
        if ctx:
            live = strategies.evaluate_checks(bot_genome, ctx)
    except Exception as e:  # noqa: BLE001 — the readout is informational only
        print("bot_detail live check {}: {}".format(bid, e), flush=True)
    gene_status = {c["gene"]: c["met"] for c in live.get("checks", [])}
    exec_mode, exec_keys = broker.execution(bot)
    entry_hist = bots.entry_cost_history(bid)
    acct_summary = _account_summary(exec_keys) if is_custom and exec_keys else None
    acct_equity = (acct_summary or {}).get("equity") if acct_summary and not acct_summary.get("error") else None
    return render(request, "bot_detail.html",
                  exec_mode=exec_mode,
                  exec_account=("its owner's" if is_custom else "the fleet's"),
                  active="custom_bots" if is_custom else "bots", bot=bot,
                  is_custom=is_custom, can_edit=can_edit,
                  st=bots.stats(bid), log=bots.trade_log(bid),
                  chart=bots.chart_geometry(bid),
                  fill_mix=bots.fill_mix(bid),
                  genome=bot_genome,
                  genome_desc=_g.describe(bot_genome),
                  gene_status=gene_status, live=live,
                  parent=parent, parent_visible=parent_visible,
                  owner_user=owner_user,
                  # a bot reached from another account's fleet must go BACK
                  # to that fleet, not to the viewer's own. Derived from the
                  # bot's owner rather than a query param, so it is right for
                  # every link into this page (/pool's included).
                  back_qs=_fleet_qs(owner, request) if owner is not None else "",
                  min_contracts=eff["min_contracts"],
                  max_contracts=eff["max_contracts"],
                  position_limit=eff["position_pct_limit"],
                  order_types=bots.ORDER_TYPES,
                  elig_trades=bots.MIN_ELIGIBLE_TRADES,
                  elig_days=bots.MIN_ELIGIBLE_DAYS,
                  elig_regimes=bots.MIN_ELIGIBLE_REGIMES,
                  guardrail_bounds=bots.CUSTOM_BOUNDS if is_custom else guardrails.bounds(),
                  entry_hist=entry_hist,
                  bot_accounts=[accounts.public(a) for a in accounts.list_accounts(owner)]
                  if is_custom and owner is not None else [],
                  bot_account=(accounts.public(accounts.get(owner, bot.get("account_id")))
                               if accounts.get(owner, bot.get("account_id")) else None),
                  acct_equity=acct_equity,
                  live_enabled=accounts.live_enabled(),
                  min_paper_trades=bots.MIN_PAPER_TRADES,
                  paper_trades=bots.closed_trade_count(bid),
                  min_balance=bots.min_balance_for(entry_hist, eff["min_contracts"],
                                                   eff["position_pct_limit"]),
                  bot_balance=bots._equity(bid)["balance"],
                  guardrail_overrides=bots.guardrail_overrides_of(bot),
                  risk_pct=bot_genome.get("risk_pct"),
                  # what this bot's own overrides sit on top of: its owner's
                  # fleet guardrails, not the house defaults
                  fleet_guardrails=guardrails.load(bots.owner_of(bot)),
                  alpaca_connected=exec_keys is not None,
                  gene_bounds=_g.NUM_GENES, int_genes=_g.INT_GENES,
                  gene_choices={"strategy": _g.STRATEGIES, "symbol": _g.SYMBOLS,
                                "order_pref": _g.ORDER_PREFS,
                                "strike_style": _g.STRIKE_STYLES,
                                "scale_in_when": _g.SCALE_IN_MODES,
                                "style": _g.STYLES, "structure": _g.STRUCTURES},
                  err=request.query_params.get("err"))


@app.post("/bots/{bid}/archive")
def bot_archive(request: Request, bid: int):
    if not _bot_for(request, bid):
        return _bot_denied(request)
    bots.archive_bot(bid, "manual")
    return RedirectResponse(f"/bots/{bid}", status_code=303)


@app.get("/bots/{bid}/equity.json")
def bot_equity_json(request: Request, bid: int):
    if not bots.get_bot(bid):
        return JSONResponse({"error": "not found"}, status_code=404)
    if not _bot_view(request, bid):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return JSONResponse({"points": bots.equity_series(bid),
                         "stats": bots.stats(bid)})


@app.post("/bots/{bid}/edit")
def bot_edit(request: Request, bid: int, name: str = Form(...),
             strategy: str = Form(""), symbol: str = Form(""),
             risk_pct: str = Form("")):
    if not _bot_for(request, bid):
        return _bot_denied(request)
    bots.update_bot(bid, name, strategy, symbol, risk_pct)
    return RedirectResponse(f"/bots/{bid}", status_code=303)


@app.post("/bots/{bid}/status")
def bot_status(request: Request, bid: int, status: str = Form(...)):
    if not _bot_for(request, bid):
        return _bot_denied(request)
    bots.set_status(bid, status)
    return RedirectResponse(f"/bots/{bid}", status_code=303)


@app.post("/bots/{bid}/delete")
def bot_delete(request: Request, bid: int):
    bot = _bot_for(request, bid)
    if not bot:
        return _bot_denied(request)
    bots.delete_bot(bid)
    return RedirectResponse("/bots/custom" if bot.get("pool") == "custom" else "/bots",
                            status_code=303)


@app.post("/bots/{bid}/trade")
def bot_trade(request: Request, bid: int, contract: str = Form(...),
              side: str = Form(...), qty: str = Form(...), price: str = Form(...),
              fees: str = Form("0"), symbol: str = Form(""), note: str = Form(""),
              order_type: str = Form("market"), limit_price: str = Form(""),
              stop_price: str = Form(""), trail_pct: str = Form("")):
    if not _bot_for(request, bid):
        return _bot_denied(request)
    err = bots.record_trade(bid, contract, side, qty, price,
                            symbol=symbol, fees=fees, note=note,
                            order_type=order_type, limit_price=limit_price,
                            stop_price=stop_price, trail_pct=trail_pct)
    if err:
        return RedirectResponse(f"/bots/{bid}?err=" + quote(err), status_code=303)
    return RedirectResponse(f"/bots/{bid}", status_code=303)


@app.post("/bots/{bid}/trade/{tid}/delete")
def bot_trade_delete(request: Request, bid: int, tid: int):
    if not _bot_for(request, bid):
        return _bot_denied(request)
    bots.delete_trade(bid, tid)
    return RedirectResponse(f"/bots/{bid}", status_code=303)
