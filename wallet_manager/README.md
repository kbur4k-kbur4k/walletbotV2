# wallet_manager

Fully automated DeFi portfolio management system running on **Base** (chain ID 8453).
Built with Python 3.11+, asyncio, aiohttp, and web3.py.

## Canonical Project Docs

If multiple AI agents or operators continue this repository, start here:

- `docs/MASTER_ROADMAP.md`
- `docs/PROJECT_STATE.md`
- `docs/HANDOFF.md`
- `deployment/AZURE_RUNTIME.md`

## Portfolio Allocation

| Pillar   | Weight | Strategy |
|----------|--------|----------|
| YIELD    | 40%    | Aave V3 USDC passive supply |
| ALGO     | 35%    | QUANT-X Alpha Engine (ML ensemble) |
| SHORT    | 10%    | Delta-neutral synthetic short via Aave V3 borrow |
| AIRDROP  | 15%    | ETH ↔ USDC ping-pong protocol farming |

---

## Quick Start

```bash
# Step 1: Install
pip install -r requirements.txt

# Step 2: Bootstrap local config files
python scripts/setup_wizard.py

# Step 3: Edit .env and wallets.json with real values
# Generate VAULT_KEY:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Step 4: Fetch historical data (takes ~30-60 minutes on first run)
python scripts/data/historical_data_pipeline.py

# Step 5: Train ML models (requires Step 4 complete)
python scripts/train_models.py

# Step 6: Run
EXECUTION_MODE=robust_paper python -m wallet_manager.orchestrator   # safe first
EXECUTION_MODE=ml_quant     python -m wallet_manager.orchestrator   # production
```

## Docker Quick Start

```bash
# Bootstrap config, then edit .env and wallets.json
python scripts/setup_wizard.py

# Run with Docker
docker-compose up -d
docker-compose logs -f wallet_manager
```

## Monitoring

The system exposes Prometheus metrics on port 9090 (`/metrics`) and an
operator dashboard on `/dashboard`.

To start the full monitoring stack (Prometheus + Grafana):

```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
# Grafana: http://localhost:3000 (admin / admin)
# Prometheus: http://localhost:9091
```

The Grafana dashboard (`monitoring/grafana_dashboard.json`) is pre-configured
for all key metrics: portfolio value, pillar allocation, ML confidence,
concept drift, Aave APY, gas price, and RPC latency.

---

## Execution Modes

| Mode          | Description                                 | When to use              |
|---------------|---------------------------------------------|--------------------------|
| `legacy`      | Bollinger Bands only                        | Debug / baseline         |
| `quantx`      | Z-Score + OFI + VWAP                        | Statistical testing      |
| `ml_quant`    | 4-model ML ensemble                         | Production               |
| `robust_paper`| ML + shadow paper execution                 | Pre-production check     |
| `robust`      | Robust Alpha + Walk-Forward audit           | Overfit verification     |
| `robust_live` | Full shields + production data              | Production alternative   |

---

## Telegram Commands

| Command                           | Description                                              |
|-----------------------------------|----------------------------------------------------------|
| `/status`                         | Full portfolio dashboard (real on-chain balances)        |
| `/pnl`                            | Last 24h / 7d / 30d P&L summary                          |
| `/trades`                         | Last 10 trades (entry, exit, pnl, strategy, venue)       |
| `/regime`                         | Current HMM state + next economic calendar event         |
| `/paper`                          | Paper engine vs real execution (last 6h)                 |
| `/risk`                           | VaR, CVaR, IR tracker, alpha decay metrics               |
| `/setmode <mode>`                 | Change execution mode without restart                    |
| `/wallets`                        | List wallets with balances and circuit breaker status    |
| `/allocation`                     | Portfolio allocation vs targets with drift info          |
| `/reset_circuit <wallet_id> confirm` | Force reset a wallet's circuit breaker               |
| `/report <daily\|weekly\|monthly>` | P&L and performance report                              |
| `/gas`                            | Current gas price and wallet suspension status           |
| `/retrain`                        | Initiate manual model retraining (requires confirmation) |
| `/retrain confirm`                | Confirm and trigger model retraining subprocess          |
| `/pause`                          | Suspend execution after current TX                       |
| `/resume`                         | Resume paused execution / clear circuit breaker halt     |
| `/stop`                           | Graceful shutdown (finishes current TX)                  |
| `/help`                           | List all available commands                              |

**Rate limiting:** 10 commands/minute per user; heavy commands (e.g. `/retrain`) limited to 1/hour.

---

## Architecture

```
wallet_manager/
├── orchestrator.py          MasterOrchestrator — main event loop, strategy dispatch
├── portfolio_manager.py     Four-pillar allocation, sync_from_chain(), snapshots
├── config.py                Frozen Settings dataclass, get_settings() singleton
├── scheduler.py             Cron-style task scheduler (JSON-backed)
├── exceptions.py            Domain exception hierarchy
│
├── core/
│   ├── interfaces.py        ABCs: IVault, ICooldown, INotifier, ITxBroadcaster
│   ├── manager.py           WalletManager — sign, broadcast, nonce mgmt
│   └── models.py            TxRecord, TxReceipt, ChainConfig, TxStatus
│
├── strategies/
│   ├── alpha_engine.py      AdaptiveGARCH + Kalman + 4-model ensemble (XGB/LGB/LSTM/Transformer)
│   ├── deep_kelly_risk.py   PPO RL + VaR/CVaR + DrawdownController
│   ├── execution_logic.py   MLQuantExecutionEngine — full pipeline coordinator
│   ├── paper_engine.py      PaperTradingEngine + PaperExecutionBridge (shadow mode)
│   ├── wf_validator.py      CPCV + DeflatedSharpe + ConceptDriftDetector + RetrainController
│   ├── robust_alpha.py      FracDiff + FeatureImportanceStabilityMonitor
│   ├── regime_filter.py     HMM regime detection + news blackout calendar
│   ├── signal_intelligence.py  Z-Score 2.0 + OFI + VWAP alpha generation
│   ├── smart_order_router.py   Multi-DEX SOR (Uniswap V3, SushiSwap, Aerodrome)
│   ├── mev_shield.py        Gaussian jitter + RBF gas escalation
│   ├── kelly_criterion.py   Half-Kelly position sizing
│   └── flash_crash_detector.py  8%/5min crash detection + KILL_SWITCH
│
├── network/
│   ├── router.py            RPCRouter with startup health probe + latency scoring
│   ├── balance_provider.py  ETH/ERC-20 balance fetchers
│   ├── gas_monitor.py       GasMonitor — async gas price poller (eth_gasPrice)
│   ├── tx_broadcaster.py    Transaction signing and submission
│   └── tx_repair.py         RBF replace-by-fee loop
│
├── resilience/
│   ├── analytics.py         SQLite analytics DB with schema versioning (v3)
│   ├── circuit_breaker.py   Per-wallet circuit breaker FSM
│   └── retry.py             Full-Jitter exponential backoff decorator
│
└── security/
    ├── vault.py             Fernet AES-256 vault with key rotation
    ├── logging_filter.py    SensitiveDataFilter (private keys never in logs)
    └── telegram_manager.py  Secure Telegram command dispatch

scripts/
├── data/
│   ├── fetch_uniswap_subgraph.py  Uniswap V3 swap history (on-chain)
│   ├── fetch_defillama.py         DeFiLlama hourly prices (fallback)
│   ├── fetch_binance.py           Binance 5m OHLCV (CEX fallback)
│   └── historical_data_pipeline.py  Multi-source pipeline + regime labeling
└── train_models.py          XGBoost / LightGBM / LSTM / Transformer / PPO training

tests/
├── conftest.py              Shared fixtures (mock_settings, mock_rpc, sample data)
├── unit/                    Fast unit tests (no network, no DB)
└── integration/             Integration tests (SQLite, mocked RPC)
```

---

## Wallet Config Format

Copy `wallets.json.example` to `wallets.json` and fill in your data:

```json
[
  {
    "wallet_id": "wallet_01",
    "address": "0x4200000000000000000000000000000000000006",
    "encrypted_private_key": "PASTE_YOUR_FERNET_TOKEN_HERE",
    "label": "Main trading wallet",
    "rpc_url": "https://mainnet.base.org",
    "chain_id": 8453,
    "max_gas_price_gwei": 50.0,
    "enabled": true
  }
]
```

Use `encrypted_private_key` for Fernet-encrypted keys (recommended) or
`private_key` for testing. If `rpc_url` or `chain_id` are omitted, the loader
defaults to the first configured Base RPC and chain ID `8453`.

## Environment Variables

| Variable                      | Required    | Default      | Description                                    |
|-------------------------------|-------------|--------------|------------------------------------------------|
| `VAULT_KEY`                   | Yes         | —            | Fernet key for private key encryption          |
| `TELEGRAM_BOT_TOKEN`          | Yes         | —            | Telegram Bot API token                         |
| `TELEGRAM_CHAT_ID`            | Yes         | —            | Target Telegram chat/group ID                  |
| `INFURA_KEY`                  | Recommended | —            | Infura project ID (Base RPC)                   |
| `ALCHEMY_KEY`                 | Recommended | —            | Alchemy API key (Base RPC)                     |
| `EXECUTION_MODE`              | No          | `ml_quant`   | One of the six execution modes                 |
| `LOG_LEVEL`                   | No          | `INFO`       | Python logging level                           |
| `LOG_FORMAT`                  | No          | `text`       | `text` or `json` (structured logging)          |
| `YIELD_WEIGHT`                | No          | `0.40`       | YIELD pillar allocation weight                 |
| `ALGO_WEIGHT`                 | No          | `0.35`       | ALGO pillar allocation weight                  |
| `SHORT_WEIGHT`                | No          | `0.10`       | SHORT pillar allocation weight                 |
| `AIRDROP_WEIGHT`              | No          | `0.15`       | AIRDROP pillar allocation weight               |
| `MONTHLY_INJECTION_USD`       | No          | `150.0`      | Monthly capital injection amount (USD)         |
| `REBALANCE_DRIFT_THRESHOLD`   | No          | `0.05`       | Drift threshold before rebalancing triggers    |
| `METRICS_ENABLED`             | No          | `true`       | Enable Prometheus /metrics endpoint            |
| `METRICS_PORT`                | No          | `9090`       | Prometheus metrics server port                 |

---

## Development Setup

```bash
pip install -e ".[dev]"
```

---

## Running Tests

```bash
pytest tests/ -v --cov=wallet_manager --cov-report=term-missing --cov-fail-under=85
```

---

## Contributing

### Branching Model

- `main` — stable, production-ready code only. Direct commits are not allowed.
- `develop` — integration branch for completed features and fixes.
- Feature branches — branch from `develop` using the pattern `feature/<short-description>`.
- Hotfix branches — branch from `main` using the pattern `hotfix/<short-description>`, then merge back into both `main` and `develop`.
- All changes enter `main` via a pull request from `develop` or a hotfix branch.

### Test Requirement

All pull requests must include tests for new or modified code. The test suite must pass with at least 85% coverage (`--cov-fail-under=85`). Run tests locally before pushing:

```bash
pytest tests/ -v --cov=wallet_manager --cov-report=term-missing --cov-fail-under=85
```

### Lint Requirement

All code must pass `ruff` linting and formatting checks before a PR can be merged:

```bash
ruff check .
ruff format . --check
```

Type annotations are enforced via `mypy`:

```bash
mypy wallet_manager/ --ignore-missing-imports
```

---

## Hard Constraints

- Chain is fixed: **Base, chain ID 8453**. Never add other chains.
- Never modify ABC signatures in `core/interfaces.py`.
- All I/O is async/await. No blocking calls in the event loop.
- Private keys never reach logs. `SensitiveDataFilter` is non-negotiable.
- `legacy` execution mode works at all times.
- `psutil` is optional — always guarded with `try/except ImportError`.
