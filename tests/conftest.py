import os
import sys
from pathlib import Path

# Make sure the app package resolves regardless of where pytest is invoked from,
# and point config.DATA_DIR somewhere disposable before any webapp module (which
# creates that directory at import time) gets imported.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TRADE_DATA_DIR", "/tmp/trade_calculator_test_data")
os.environ.setdefault("TRADE_INSECURE_COOKIE", "1")

import pytest

from webapp import (account_book as _account_book, accounts as _accounts, bot_settings as _bot_settings,
                    bots as _bots, gene_pool as _gene_pool, guardrails as _guardrails)


@pytest.fixture(autouse=True)
def isolated_bots_db(tmp_path, monkeypatch):
    """Every test gets its own empty bots.db and its own (empty) fleet /
    per-user settings and guardrails — nothing here should ever depend on,
    or leak into, another test's fleet/ledger state, saved Alpaca keys or
    risk limits. (Guardrails matter: a house default left behind by one
    test would silently shrink another test's position cap and make its
    fills disappear.)"""
    monkeypatch.setattr(_bots, "DB_PATH", tmp_path / "bots.db")
    monkeypatch.setattr(_bot_settings, "_PATH", tmp_path / "bot_settings.json")
    monkeypatch.setattr(_bot_settings, "_USER_PATH", tmp_path / "user_bot_settings.json")
    monkeypatch.setattr(_accounts, "_PATH", tmp_path / "accounts.json")
    _accounts._CACHE.clear()
    monkeypatch.setattr(_guardrails, "_PATH", tmp_path / "guardrails.json")
    monkeypatch.setattr(_guardrails, "_USER_PATH", tmp_path / "user_guardrails.json")
    _account_book.forget()
    # module-level TTL caches would otherwise leak one test's fleet into the next
    _bots.invalidate_caches()
    _gene_pool.invalidate()
    _bot_settings._USER_CACHE.clear()
    _bots.init()
    yield


@pytest.fixture
def modeled_fills(monkeypatch):
    """Book fills at the price the bot decided, with no Alpaca account.

    A live fleet always trades for real — there is no simulated execution
    mode — so a test that is only exercising ENGINE MECHANICS (the cost
    gate, spread packaging, swing expiry) patches broker.execution to the
    modeled branch the offline replay harness uses. That keeps the fill
    price the bot's own decision, which is what those assertions are
    about, without pretending a fleet can run without keys."""
    from webapp import broker as _broker
    monkeypatch.setattr(_broker, "execution", lambda bot: ("model", None))
    yield
