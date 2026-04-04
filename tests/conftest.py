"""
tests/conftest.py

Shared pytest fixtures for all test modules.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Provide minimal env vars before any import touches get_settings()
os.environ.setdefault("VAULT_KEY", "kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=")
os.environ.setdefault("EXECUTION_MODE", "ml_quant")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_bot_token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "123456789")


@pytest.fixture
def mock_settings():
    """Returns a minimal Settings with all required fields populated.

    No real env vars needed. Safe to use in CI without .env file.
    """
    from wallet_manager.config import Settings
    return Settings(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode="ml_quant",
        telegram_bot_token="test_bot_token",
        telegram_chat_id="123456789",
        garch_alpha=0.10,
        garch_beta=0.85,
    )


@pytest.fixture
def mock_wallet_manager(mocker):
    """sign_and_broadcast_tx -> returns mock TxReceipt(status=SUCCESS).

    Never makes real RPC calls.
    """
    from wallet_manager.core.models import TxReceipt, TxStatus
    manager = mocker.MagicMock()
    receipt = TxReceipt(
        tx_hash="0xabc123",
        wallet_id="test_wallet",
        nonce_used=42,
        status=TxStatus.SUCCESS,
        gas_used=21000,
        block_number=1000,
        chain_id=8453,
    )
    manager.sign_and_broadcast_tx = mocker.AsyncMock(return_value=receipt)
    return manager


@pytest.fixture
def mock_rpc(aioresponses):
    """Intercepts all aiohttp calls. Returns preset responses."""
    return aioresponses


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """500-step OHLCV DataFrame with realistic ETH price series.

    Includes all columns required by HistoricalDataPipeline.
    """
    rng = np.random.default_rng(42)
    n = 500
    base_price = 2000.0
    log_returns = rng.normal(0, 0.01, n)
    prices = base_price * np.exp(np.cumsum(log_returns))

    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")
    highs = prices * (1 + rng.uniform(0, 0.005, n))
    lows = prices * (1 - rng.uniform(0, 0.005, n))
    volumes = rng.uniform(10_000, 100_000, n)
    buy_vol = volumes * rng.uniform(0.4, 0.6, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume_usdc": volumes,
        "buy_volume_usdc": buy_vol,
        "sell_volume_usdc": volumes - buy_vol,
        "trade_count": rng.integers(10, 200, n),
        "gas_price_mean_gwei": rng.uniform(0.01, 1.0, n),
        "source": "test",
        "sample_weight": np.ones(n),
        "purge_zone": np.zeros(n, dtype=bool),
        "stale": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def sample_feature_vectors() -> np.ndarray:
    """100 rows x 30 columns of normalised feature vectors."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 30)).astype(np.float32)


@pytest.fixture
def sample_feature_vector() -> np.ndarray:
    """30-dim normalised feature vector (float64)."""
    np.random.seed(0)
    return np.random.randn(30).astype(np.float64)


@pytest.fixture
def sample_feature_matrix() -> np.ndarray:
    """200 × 30 feature matrix for model training."""
    np.random.seed(1)
    return np.random.randn(200, 30).astype(np.float64)


@pytest.fixture
def feature_names() -> list:
    """30 generic feature names — matches sample_feature_vector length."""
    return [f"feature_{i}" for i in range(30)]


@pytest.fixture
def sample_price_series() -> list:
    """100-step realistic WETH/USDC price series (~$3 000)."""
    np.random.seed(42)
    returns = np.random.normal(0.0, 0.002, 100)
    prices = 3_000.0 * np.cumprod(1.0 + returns)
    return prices.tolist()


@pytest.fixture
def mock_circuit_breaker(mocker):
    """CircuitBreaker that always allows requests."""
    from wallet_manager.resilience.circuit_breaker import CircuitState
    cb = mocker.MagicMock()
    cb.allow_request.return_value = True
    cb.state = CircuitState.CLOSED
    return cb


@pytest.fixture
def mock_vault(mocker):
    """Vault where encrypt/decrypt are identity operations (for testing)."""
    vault = mocker.MagicMock()
    vault.encrypt.side_effect = lambda x: f"encrypted:{x}".encode()
    vault.decrypt.side_effect = lambda x: x.decode().replace("encrypted:", "") if isinstance(x, bytes) else x.replace("encrypted:", "")
    return vault


@pytest.fixture(scope="function")
async def analytics_db(tmp_path):
    """Real AnalyticsDB backed by tmp_path SQLite — isolated per test."""
    from wallet_manager.resilience.analytics import AnalyticsDB
    db = AnalyticsDB(db_path=str(tmp_path / "analytics_test.db"))
    await db.initialize()
    yield db
    with contextlib.suppress(Exception):
        await asyncio.wait_for(db.shutdown(), timeout=5.0)


@pytest.fixture
def mock_aiohttp_session(mocker):
    """Mocked aiohttp.ClientSession that never makes real HTTP calls."""
    session = mocker.AsyncMock(spec=aiohttp.ClientSession)
    session.closed = False
    return session
