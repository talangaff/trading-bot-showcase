"""Runtime configuration, all overridable via environment variables."""

import os
import secrets
from pathlib import Path

# Where mutable state lives (auth file, week cache). Kept OUT of the repo.
DATA_DIR = Path(os.environ.get("TRADE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

AUTH_FILE = DATA_DIR / "auth.json"
WEEK_CACHE_FILE = DATA_DIR / "week_cache.json"
WEEK_STATUS_FILE = DATA_DIR / "week_status.json"
WEEK_LOCK_FILE = DATA_DIR / "week.lock"
SECRET_FILE = DATA_DIR / "secret_key"

# Session signing key. Prefer an explicit env var in production; otherwise
# persist a generated one so sessions survive restarts.
def _load_secret_key() -> str:
    env = os.environ.get("TRADE_SECRET_KEY")
    if env:
        return env
    if SECRET_FILE.exists():
        return SECRET_FILE.read_text().strip()
    key = secrets.token_hex(32)
    SECRET_FILE.write_text(key)
    try:
        SECRET_FILE.chmod(0o600)
    except OSError:
        pass
    return key


SECRET_KEY = _load_secret_key()

# "This Week" screen
WEEK_DAYS = int(os.environ.get("TRADE_WEEK_DAYS", "7"))
WEEK_TOP_N = int(os.environ.get("TRADE_WEEK_TOP_N", "50"))

# The first admin account on a brand-new install. Its password must be
# changed on first login. Other accounts are created from the Users page.
DEFAULT_USERNAME = os.environ.get("TRADE_DEFAULT_USERNAME", "admin").strip().lower() or "admin"
DEFAULT_PASSWORD = os.environ.get("TRADE_DEFAULT_PASSWORD", "admin")

# Cookie is https-only in production. Set TRADE_INSECURE_COOKIE=1 for local http.
SECURE_COOKIE = os.environ.get("TRADE_INSECURE_COOKIE", "") not in ("1", "true", "yes")

APP_TITLE = "Trade Manager"

# --- 0DTE options bot fleet ------------------------------------------------
# Per-bot bookkeeping on a shared Alpaca paper account: every bot places real
# orders there (fleet/baseline on the fleet keys, custom bots on their
# owner's own keys) and starts with its own virtual slice of this size.
BOT_START_BALANCE = float(os.environ.get("BOT_START_BALANCE", "10000"))
BOT_MAX_DRAWDOWN_PCT = float(os.environ.get("BOT_MAX_DRAWDOWN_PCT", "25"))
BOT_POSITION_PCT_LIMIT = float(os.environ.get("BOT_POSITION_PCT_LIMIT", "10"))
BOT_MIN_CONTRACTS = int(os.environ.get("BOT_MIN_CONTRACTS", "2"))
BOT_MAX_CONTRACTS = int(os.environ.get("BOT_MAX_CONTRACTS", "250"))

# Fleet sizing for the autonomous simulation engine (webapp/bot_engine.py).
# Pure simulation on shared 1-min bars, so bot count is bounded by CPU +
# SQLite, not Alpaca's request limit.
#
# ONE number, not two: max_fleet_bots is both the ceiling and the setpoint.
# Every account runs a FULL fleet — the daily pass retires the idle and the
# weakest and then refills every empty slot, so a fleet ends each
# maintenance run at exactly this many active bots. There is deliberately
# no separate, lower "target" any more: an empty slot is a genome not being
# tested. Nothing may exceed it, by any route, including the Spawn button.
# It is a guardrail like the risk limits below: the house default here,
# each account's own override on the Guardrails page, clamped to the
# absolute range further down.
BOT_MAX_FLEET_BOTS = int(os.environ.get("BOT_MAX_FLEET_BOTS", "100"))
BOT_POLL_SECONDS = int(os.environ.get("BOT_POLL_SECONDS", "15"))
# A run_cycle() stops starting new ticks after this many seconds — real
# orders wait for their fills, and a burst of entries must never run a
# one-minute timer cycle into the next firing.
BOT_CYCLE_BUDGET_S = float(os.environ.get("BOT_CYCLE_BUDGET_S", "45"))
# Single-leg options only (decided 2026-10-22): no vertical or any other spread.
# With this False the genome has no spread structure to choose, any stored genome
# that names one is repaired to "single", and the engine never builds a spread.
BOT_ALLOW_SPREADS = os.environ.get("BOT_ALLOW_SPREADS", "0") == "1"
# The incubator (webapp/incubator.py): a second population of bots (pool='sim')
# that run the SAME entry/exit code as the fleet on the live market feed, but
# with modeled fills and no orders, so many more genomes can be tested at once.
# Bots that clear the graduation bar are promoted into the real fleet; the best
# performers are the parents of the next generation of sim bots.
BOT_SIM_PER_FLEET = int(os.environ.get("BOT_SIM_PER_FLEET", "750"))
BOT_SIM_TICK_BUDGET_S = float(os.environ.get("BOT_SIM_TICK_BUDGET_S", "10"))
BOT_SIM_GRAD_TRADES = int(os.environ.get("BOT_SIM_GRAD_TRADES", "30"))
BOT_SIM_GRAD_DAYS = int(os.environ.get("BOT_SIM_GRAD_DAYS", "5"))
BOT_SIM_GRAD_FITNESS = float(os.environ.get("BOT_SIM_GRAD_FITNESS", "1.0"))
# Auto-archive a bot forever if it was ever up >= this %% and then failed.
BOT_ARCHIVE_PEAK_PCT = float(os.environ.get("BOT_ARCHIVE_PEAK_PCT", "50"))
# A bot must clear this track record before it can be a breeding parent —
# blocks a lucky one-session run from dominating evolution. Trades, distinct
# NYSE sessions, and distinct volatility regimes (see bots.regime_label) all
# have to be covered.
BOT_MIN_ELIGIBLE_TRADES = int(os.environ.get("BOT_MIN_ELIGIBLE_TRADES", "60"))
BOT_MIN_ELIGIBLE_DAYS = int(os.environ.get("BOT_MIN_ELIGIBLE_DAYS", "15"))
BOT_MIN_ELIGIBLE_REGIMES = int(os.environ.get("BOT_MIN_ELIGIBLE_REGIMES", "2"))
# On top of the sample gates, a bot is only PROVEN (may breed) if its fitness
# t-stat clears a Bonferroni-style threshold for the number of fleet bots
# tested since the last reset: z such that P(best of N null bots > z) <= alpha.
BOT_PROVEN_ALPHA = float(os.environ.get("BOT_PROVEN_ALPHA", "0.05"))
BOT_PROMISING_T = float(os.environ.get("BOT_PROMISING_T", "2.0"))
# stagnant fleet bots (trading, but going nowhere) are retired and replaced
BOT_STAGNANT_MIN_AGE_DAYS = float(os.environ.get("BOT_STAGNANT_MIN_AGE_DAYS", "14"))
BOT_STAGNANT_MIN_TRADES = int(os.environ.get("BOT_STAGNANT_MIN_TRADES", "40"))
BOT_STAGNANT_FITNESS = float(os.environ.get("BOT_STAGNANT_FITNESS", "0.5"))
BOT_STAGNANT_PL_PCT = float(os.environ.get("BOT_STAGNANT_PL_PCT", "2.0"))
# fleet bots with a statistically persistent loss are retired well before the drawdown floor
BOT_LOSER_MIN_TRADES = int(os.environ.get("BOT_LOSER_MIN_TRADES", "40"))
BOT_LOSER_FITNESS = float(os.environ.get("BOT_LOSER_FITNESS", "-1.5"))
# at most this share of a fleet may be untested random "exploration" genomes
# (seeds, and adopted generation-0 pool genomes); the rest descend from winners
BOT_EXPLORE_SHARE = float(os.environ.get("BOT_EXPLORE_SHARE", "0.20"))
BOT_EXPLORE_TRIM_PER_DAY = int(os.environ.get("BOT_EXPLORE_TRIM_PER_DAY", "15"))
# Recent-window (most recent trading days present in a bot's own closed-trade
# history) used to compute a secondary "holdout" fitness — a bot whose
# lifetime numbers were made by an early hot streak but whose recent trades
# are underwater doesn't get to keep breeding off the old streak.
BOT_HOLDOUT_DAYS = int(os.environ.get("BOT_HOLDOUT_DAYS", "5"))
# Fleet-wide entry cost gate (option_chain.cost_gate): the expected
# directional premium gain over the planned hold (|delta| x expected
# underlying move at the option's own IV) must be at least this multiple of
# the certain costs (full bid/ask spread + theta decay over the hold + round-
# trip commission). Not a gene — evolution shouldn't spend its budget
# rediscovering friction. 0 disables the gate.
BOT_COST_GATE_MULT = float(os.environ.get("BOT_COST_GATE_MULT", "1.0"))
# A real option quote older than this is never used as a fill price. It is
# still used to CALIBRATE (implied vol + spread width) and then repriced at
# the live spot — see option_chain._usable_quote. yfinance's chain is always
# 15-20 minutes delayed, so it is always in calibrate-only mode.
BOT_MAX_QUOTE_AGE_S = float(os.environ.get("BOT_MAX_QUOTE_AGE_S", "60"))
# Per-contract commission charged against MODELED fills only — which now
# means the offline replay harness alone, since every live fleet trades on
# an Alpaca account and books the real fill, reflecting whatever that
# account is actually charged.
BOT_COMMISSION_PER_CONTRACT = float(os.environ.get("BOT_COMMISSION_PER_CONTRACT", "0.65"))

# Absolute safety ceiling/floor on the fleet guardrails above. These bound
# what ANY bot may run with — including a custom bot with its own overridden
# guardrails — and are not editable from the UI. They exist so a bad custom
# guardrail edit can loosen risk controls but never remove them outright.
BOT_ABS_MAX_DRAWDOWN_PCT = float(os.environ.get("BOT_ABS_MAX_DRAWDOWN_PCT", "40"))
BOT_ABS_POSITION_PCT_LIMIT = float(os.environ.get("BOT_ABS_POSITION_PCT_LIMIT", "20"))
BOT_ABS_MIN_CONTRACTS = int(os.environ.get("BOT_ABS_MIN_CONTRACTS", "1"))
BOT_ABS_MAX_CONTRACTS = int(os.environ.get("BOT_ABS_MAX_CONTRACTS", "500"))
# The fleet-size cap is bounded too: at least one bot (a fleet of zero is a
# stopped fleet, which is what the engine switch is for) and never more than
# what one process can actually tick inside BOT_CYCLE_BUDGET_S.
BOT_ABS_MIN_FLEET_BOTS = int(os.environ.get("BOT_ABS_MIN_FLEET_BOTS", "1"))
BOT_ABS_MAX_FLEET_BOTS = int(os.environ.get("BOT_ABS_MAX_FLEET_BOTS", "500"))

# Alpaca's per-key rate limit, requests per minute. 200/min is the documented
# basic-plan figure for the trading API; raise it if the account is on a plan
# with a higher allowance. Used only to render headroom on the Bot Fleet page
# (webapp/api_meter.py) — nothing throttles against it, so a wrong value
# misleads the reader but changes no behaviour.
ALPACA_RATE_LIMIT_PER_MIN = int(os.environ.get("ALPACA_RATE_LIMIT_PER_MIN", "200"))

# Alpaca paper-trading credentials. Unused — the app reads the fleet's
# key/secret entered on the Bot Fleet settings panel (webapp/bot_settings.py)
# instead, for market data and for the fleet's real order execution
# (webapp/broker.py); each user's custom bots use the keys entered on their
# own Custom Bots page. Kept only in case an env-var-based deploy is wanted
# later. Set them in /var/www/trade-calculator-data/env:
#   ALPACA_API_KEY_ID=...
#   ALPACA_API_SECRET_KEY=...
#   ALPACA_PAPER=1
ALPACA_API_KEY_ID = os.environ.get("ALPACA_API_KEY_ID", "")
ALPACA_API_SECRET_KEY = os.environ.get("ALPACA_API_SECRET_KEY", "")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "1") not in ("0", "false", "no")
