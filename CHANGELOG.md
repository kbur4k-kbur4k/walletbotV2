# Changelog

## [Unreleased] — Autonomous agent run (2026-03-26)

### Phase 1 — Stability & Correctness
- **fix**: Replaced 19 bare `except: pass` blocks with structured `log.warning()` / `log.error()` calls across 7 files
- **feat**: Completed Compound V3 USDC APY integration on Base (`0xb125E6687...`) in `actions.py`
- **fix**: Added full type annotations to `oci_deployer.py`, removed mypy exclusion from `pyproject.toml`

### Phase 2 — ML Excellence
- **feat**: `DynamicEnsembleWeighter` — rolling OOS Sharpe-based weight adaptation, persisted to `ml_models/ensemble_weights.json`
- **feat**: Feature set expanded from 30 to 36 dimensions (rsi_14, bb_position, volume_trend, price_acceleration, spread_ratio, microstructure_score)
- **feat**: Model versioning — `archive_models()` snapshots to `ml_models/archive/YYYYMMDD_HHMMSS/`, prunes to 3 most recent; `training_report.json` gains version/trained_at/archive_path
- **feat**: Online logistic regression meta-learner (`stacking_meta_learner.py`) replaces static weighted average; SGD updates every 100 obs over 500-obs window

### Phase 3 — New Capabilities
- **feat**: Backtesting engine (`scripts/backtest.py`) with Sharpe, drawdown, win-rate, profit-factor, pillar breakdown
- **feat**: 13 Prometheus alert rules (`monitoring/alert_rules.yml`) covering portfolio, ML, RPC, gas, and safety
- **feat**: Prometheus Alertmanager webhook (`alerting/prometheus_webhook.py`) → Telegram with 30s rate limiting

### Phase 4 — Refactoring
- **refactor**: `orchestrator.py` split into `orchestrator_gas.py` (`GasGuard`) + `orchestrator_lifecycle.py` (`LifecycleManager`)
- **refactor**: `task_factories.py` split into 4 per-pillar shim modules

### Phase 5 — Test Coverage
- **test**: Added `test_oci_deployer.py` (12 tests), `test_ensemble_weighter.py` (7), `test_feature_expansion.py` (8), `test_model_versioning.py` (3), `test_compound_integration.py` (5)
- **test**: Added `test_flash_crash_detector.py`, `test_stochastic_stress_test.py`, `test_algo_strategy.py`, `test_backtest.py`, `test_prometheus_webhook.py`, `test_log_schema.py`, `test_health_endpoints.py`
- **test**: Added `tests/integration/test_full_cycle.py`

### Phase 6 — Operational Excellence
- **feat**: Standardised JSON logging schema (`utils/log_schema.py` + `JsonFormatter`)
- **feat**: Health check endpoints `/ready`, `/live`, `/metrics/summary`
- **feat**: Multi-stage Dockerfile, docker-compose resource limits, Kubernetes manifests (`deployment/k8s/`)

## [Unreleased] — 2026-03-25

### Added
- Unit tests for core/manager.py (WalletManager)
- Unit tests for network/tx_broadcaster.py, balance_provider.py, session_factory.py
- Unit tests for security/logging_filter.py, vault_manager.py
- Unit tests for strategies/execution_logic.py, mev_shield.py, latency_shield.py
- Unit tests for strategies/liquidity_alpha.py, market_microstructure.py, short_strategy.py
- Unit tests for strategies/quant_analytics.py, quantx_engine.py
- Unit tests for utils/health_check.py, rate_limiter.py, cooldown.py
- Expanded coverage for airdrop_strategy, yield_strategy, robust_alpha, idle_sweep, rpc_router, tx_repair
- SessionFactory.get_session() and close_all() methods for per-wallet session management
- CooldownManager.trigger_cooldown(), is_in_cooldown(), clear(), clear_all() methods
- Docker and docker-compose support (`Dockerfile`, `docker-compose.yml`, `.dockerignore`)
- Monitoring stack (`monitoring/docker-compose.monitoring.yml`, `monitoring/prometheus.yml`)
- Grafana dashboard JSON (`monitoring/grafana_dashboard.json`) for all Prometheus metrics
- New Telegram commands: `/setmode`, `/wallets`, `/allocation`, `/reset_circuit`, `/report`, `/gas`
- Per-user rate limiting for Telegram commands (10 commands/minute, 1 heavy command/hour)
- Portfolio rebalancing threshold (`REBALANCE_DRIFT_THRESHOLD` env var)
- `needs_rebalancing()` method on `PortfolioManager` — rebalance only when drift exceeds threshold
- Configurable allocation weights via environment variables (`YIELD_WEIGHT`, `ALGO_WEIGHT`, `SHORT_WEIGHT`, `AIRDROP_WEIGHT`, `MONTHLY_INJECTION_USD`)
- Weight sum validation in `Settings.__post_init__` — raises `ConfigurationError` if weights ≠ 1.0
- `metrics_host` and `log_format` fields in `Settings`
- Wallet config JSON schema validation with full error collection before raising
- `SIGTERM` graceful shutdown support (`_setup_signal_handlers`)
- Structured JSON logging option (`LOG_FORMAT=json` env var) via `build_logger()`
- `wallets.json.example` template (no real keys)
- Orchestrator integration tests (`tests/integration/test_orchestrator.py`)
- Wallet config validation tests (`tests/unit/test_wallet_config_validation.py`)
- Extended Telegram command tests (`tests/unit/test_telegram_commands.py`)
- Portfolio manager rebalancing threshold tests

### Changed
- pyproject.toml: updated ruff lint rules, mypy strict=false with disallow_untyped_defs, coverage fail_under=75
- pytest-xdist added for parallel test execution

### Fixed
- SessionFactory missing get_session()/close_all() methods
- CooldownManager incomplete implementation (missing cooldown tracking methods)
- `deep_kelly_history.json` removed from version control (runtime-generated file)
- `allcode.txt` debugging artifact deleted
- `wallets.json` added to `.gitignore`
- Portfolio weights now loaded from config/env vars instead of hardcoded constants
- Telegram `/help` updated to list all new commands

### Removed
- `wallet_manager/allcode.txt` (34 KB debugging artifact)
- `wallet_manager/deep_kelly_history.json` (runtime state file)
