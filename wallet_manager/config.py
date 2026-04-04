"""
wallet_manager/config.py

Centralised, validated configuration loader for the wallet_manager package.
All modules import from here rather than calling os.getenv directly.

Usage::

    from wallet_manager.config import get_settings
    cfg = get_settings()
    print(cfg.retry_attempts)
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from wallet_manager.env_loader import load_project_env

load_project_env()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        from wallet_manager.exceptions import ConfigurationError
        raise ConfigurationError(
            f"Env var {name!r} must be a float; got {raw!r}."
        ) from exc


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        from wallet_manager.exceptions import ConfigurationError
        raise ConfigurationError(
            f"Env var {name!r} must be an int; got {raw!r}."
        ) from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes")


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else Path(default)


def _str_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

VALID_EXECUTION_MODES = frozenset({
    "legacy", "quantx", "ml_quant", "robust_paper", "robust", "robust_live",
    "shadow_live",
})
LIVE_GUARDED_EXECUTION_MODES = frozenset({"ml_quant", "robust", "robust_live"})

VALID_LOG_LEVELS = frozenset({
    "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
})


@dataclass(frozen=True)
class Settings:
    """Immutable, validated configuration snapshot for the whole package.

    Covers all environment variables used across every module.
    Construct via ``Settings.from_env()`` or inject directly in tests.
    """

    # --- Vault ---------------------------------------------------------
    vault_key: str = ""

    # --- Telegram ------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_rate_limit: float = 1.0
    telegram_rate_burst: int = 1

    # --- Retry ---------------------------------------------------------
    retry_attempts: int = 4
    retry_base_delay: float = 1.5

    # --- Circuit Breaker -----------------------------------------------
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0

    # --- Traffic regulation --------------------------------------------
    poisson_lambda: float = 2.0
    gaussian_jitter_sigma: float = 0.3

    # --- HTTP sessions -------------------------------------------------
    session_total_timeout: float = 30.0
    session_connect_timeout: float = 10.0

    # --- Orchestrator --------------------------------------------------
    max_concurrent_wallets: int = 5
    gas_poll_interval: float = 30.0
    gas_wait_timeout: float = 600.0
    loop_interval_seconds: float = 45.0
    algo_loop_interval_seconds: float = 300.0
    heartbeat_interval_seconds: float = 21600.0
    default_task_loop_cycles: int = 3
    use_master_orchestrator: bool = True

    # --- Transaction repair (RBF) --------------------------------------
    tx_poll_interval: float = 5.0
    tx_stuck_deadline: float = 180.0
    rbf_gas_bump_min: float = 0.15
    rbf_gas_bump_max: float = 0.25
    max_rbf_attempts: int = 3

    # --- Multi-chain router / MEV Shield RPC cluster -------------------
    infura_key: str = ""
    alchemy_key: str = ""
    private_rpc_1: str = ""
    private_rpc_2: str = ""
    private_rpc_3: str = ""
    base_rpc_1: str = ""
    base_rpc_2: str = ""
    base_rpc_3: str = ""

    # --- Analytics DB --------------------------------------------------
    analytics_db_path: Path = Path("wallet_analytics.db")
    quant_analytics_db_path: Path = Path("quant_analytics.db")
    historical_data_db_path: Path = Path("historical_data.db")

    # --- Logging -------------------------------------------------------
    log_level: str = "INFO"

    # --- Wallet config -------------------------------------------------
    wallet_config_path: Path = Path("wallets.json")

    # --- Signal Intelligence -------------------------------------------
    zscore_window_size: int = 288
    zscore_entry_threshold: float = -2.8
    zscore_short_threshold: float = 2.8
    ofi_tick_range: int = 100
    vwap_window_trades: int = 500

    # --- MEV Shield / SOR ----------------------------------------------
    sor_enabled_dexes: list[str] = field(
        default_factory=lambda: ["uniswap_v3", "sushiswap_v3", "aerodrome"]
    )
    sor_cache_ttl: float = 10.0

    # --- Kelly Criterion -----------------------------------------------
    kelly_fraction: float = 0.25          # quarter-Kelly until 500+ trades
    kelly_max_position_pct: float = 0.10  # max 10% per trade early on
    kelly_min_trades: int = 10
    min_trades_before_full_kelly: int = 500

    # --- Flash Crash Circuit Breaker -----------------------------------
    quantx_crash_threshold_pct: float = 8.0
    quantx_crash_window_seconds: float = 300.0

    # --- Gaussian Jitter (Ghost Protocol) ------------------------------
    jitter_mean_ms: int = 800
    jitter_std_ms: int = 400
    jitter_min_ms: int = 100
    jitter_max_ms: int = 2000

    # --- Dynamic RBF ---------------------------------------------------
    rbf_block_deadline: int = 2
    rbf_escalation_pct: int = 15

    # --- Engine mode ---------------------------------------------------
    quantx_engine_mode: str = "quantx"
    execution_mode: str = "ml_quant"

    # --- GARCH(1,1) ----------------------------------------------------
    garch_omega: float = 1e-6
    garch_alpha: float = 0.10
    garch_beta: float = 0.85

    # --- Kalman Filter -------------------------------------------------
    kalman_process_noise: float = 1e-5
    kalman_measurement_noise: float = 1e-3

    # --- ML Ensemble ---------------------------------------------------
    xgboost_weight: float = 0.00
    lightgbm_weight: float = 0.00
    lstm_weight: float = 0.00
    transformer_weight: float = 1.00
    ml_confidence_threshold: float = 0.65
    # Comma-separated list of active models: xgb, lgbm, lstm, transformer
    # Start with xgb,lgbm only. Add lstm,transformer after 1000+ real trades.
    ensemble_active_models: str = "xgb,lgbm,lstm,transformer"

    # --- PPO (Deep Kelly) ----------------------------------------------
    ppo_model_path: Path = Path("ml_models/ppo_risk_policy.pt")
    ppo_min_trades: int = 50
    enhanced_kelly_fraction: float = 0.5
    min_paper_trades_before_live: int = 50
    # Minimum completed paper trades required before switching to ml_quant/robust.
    # Set to 0 to disable this gate (not recommended).

    # --- Drawdown Controller -------------------------------------------
    drawdown_reduce_threshold: float = 0.05
    drawdown_halt_threshold: float = 0.10
    drawdown_recovery_threshold: float = 0.03

    # --- VaR -----------------------------------------------------------
    var_confidence: float = 0.99
    var_lookback: int = 500
    var_max_position_pct: float = 0.30

    # --- Robustness Shield ---------------------------------------------
    fracdiff_d_min: float = 0.1
    fracdiff_d_max: float = 1.0
    fracdiff_d_step: float = 0.1
    fracdiff_min_observations: int = 50
    adf_significance_level: float = 0.05
    robust_paper_adf_entry_pvalue_max: float = 0.35
    robust_paper_alpha_entry_threshold: float = 0.04

    # --- Paths ---------------------------------------------------------
    algo_price_history_path: Path = Path("")
    short_state_path: Path = Path(__file__).resolve().parent / "state" / "short_state.json"
    schedule_data_path: Path = Path(__file__).resolve().parent / "state" / "schedule_data.json"

    # --- Aave APY alert ------------------------------------------------
    aave_apy_alert_threshold_pct: float = 2.0

    # --- Cooldown ------------------------------------------------------
    cooldown_min_delay: float = 1.5

    # --- OCI -----------------------------------------------------------
    oci_region: str = "unknown"

    # --- Paper Trading -------------------------------------------------
    paper_trade: bool = False
    paper_initial_balance_usd: float = 10_000.0
    paper_slippage_model: str = "gaussian"
    paper_slippage_mean_bps: float = 5.0
    paper_slippage_std_bps: float = 2.0
    paper_gas_estimation: bool = True
    paper_max_gas_fraction_pct: float = 5.0
    paper_db_path: Path = Path("paper_trades.db")

    # --- Live Safety Guard ---------------------------------------------
    live_trading_enabled: bool = False
    live_tiny_cap_mode: bool = True
    live_max_total_capital_usd: float = 250.0
    live_max_wallets: int = 1

    # --- Audit ----------------------------------------------------------
    audit_snapshot_enabled: bool = True
    audit_recent_sessions_limit: int = 25

    # --- Monte Carlo ---------------------------------------------------
    mc_n_simulations: int = 1000
    mc_price_jitter_pct: float = 0.02
    mc_min_win_rate: float = 0.45
    rfe_min_features: int = 5
    rfe_importance_decay_threshold: float = 0.01

    # --- Latency Shield ------------------------------------------------
    latency_mean_ms: float = 50.0
    latency_std_ms: float = 20.0
    latency_min_ms: float = 10.0
    latency_max_ms: float = 500.0
    stale_quote_threshold_s: float = 30.0
    stale_quote_warning_s: float = 15.0
    price_divergence_threshold_pct: float = 0.5
    max_consecutive_stale: int = 3

    # --- Walk-Forward Validator ----------------------------------------
    wf_n_splits: int = 5
    wf_purge_pct: float = 0.05
    wf_embargo_pct: float = 0.01
    wf_retrain_interval: int = 100
    wf_min_oos_sharpe: float = 0.5
    wf_validator_db_path: Path = Path("wf_validator.db")

    # --- Regime Filter / HMM ------------------------------------------
    hmm_n_states: int = 3
    hmm_refit_interval: int = 288
    hmm_min_observations: int = 60
    hmm_news_blackout_before_hours: float = 0.5
    hmm_news_blackout_after_hours: float = 1.0

    # --- Liquidity Alpha -----------------------------------------------
    liquidity_thin_threshold: float = 30.0
    liquidity_split_volume_pct: float = 0.01
    liquidity_profile_cache_ttl: float = 30.0
    liquidity_split_weights: list[float] = field(
        default_factory=lambda: [0.40, 0.35, 0.25]
    )

    # --- Permutation Feature Importance --------------------------------
    permutation_importance_threshold: float = 0.01
    permutation_n_shuffles: int = 30
    permutation_min_samples: int = 100

    # --- OCI Deployer --------------------------------------------------
    watchdog_memory_limit_mb: int = 512
    watchdog_ws_timeout_s: float = 30.0
    watchdog_interval_s: float = 60.0
    health_api_port: int = 8080
    health_api_host: str = "0.0.0.0"
    max_restart_attempts: int = 3
    restart_cooldown_s: float = 30.0

    # --- Historical Data Pipeline (Section 2) --------------------------
    historical_data_min_candles: int = 105_120     # 2 years of 5-min candles
    historical_data_target_candles: int = 157_680  # 3 years target
    subgraph_url: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    defillama_enabled: bool = True
    binance_fallback_enabled: bool = True
    incremental_update_hour_utc: int = 2
    min_new_candles_for_retrain: int = 500
    min_days_between_retrains: int = 7

    # --- Pool Addresses ------------------------------------------------
    trade_execution_delay_ms: int = 500
    weth_usdc_pool: str = "0xd0b53d9277642d899df5c87a3966a349a798f224"
    quoter_address: str = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"
    # --- New Phase 2 config keys ----------------------------------------
    # C.2
    airdrop_max_gas_fee_ratio: float = 0.5
    airdrop_min_swap_usd: float = 10.0
    # C.3
    rpc_reprobe_interval_seconds: float = 300.0
    # C.4
    yield_min_apy_pct: float = 1.0
    aave_apy_poll_interval_seconds: float = 3600.0
    # C.5
    proactive_alert_interval_seconds: float = 300.0
    alert_dedup_window_seconds: float = 3600.0
    # C.6
    metrics_enabled: bool = True
    metrics_port: int = 9092
    # C.7
    trade_journal_path: Path = Path("trade_journal.jsonl")
    trade_journal_max_mb: float = 100.0
    # D.2
    feature_drift_kl_threshold: float = 0.1
    feature_drift_window: int = 288
    feature_drift_check_interval: int = 100
    # D.3
    wf_sliding_window_days: int = 90
    wf_use_dual_validation: bool = True
    # E.1
    vault_key_rotation_days: int = 90

    # --- Portfolio Allocation Weights ----------------------------------
    yield_weight: float = 0.40           # YIELD_WEIGHT env var
    algo_weight: float = 0.35            # ALGO_WEIGHT env var
    short_weight: float = 0.10           # SHORT_WEIGHT env var
    airdrop_weight: float = 0.15         # AIRDROP_WEIGHT env var
    monthly_injection_usd: float = 150.0 # MONTHLY_INJECTION_USD env var

    @property
    def is_paper_execution(self) -> bool:
        """Return True when this config is guaranteed not to broadcast live TXs."""
        return self.paper_trade or self.execution_mode == "robust_paper"

    @property
    def is_live_execution(self) -> bool:
        """Return True when the configured mode may broadcast real transactions."""
        return self.execution_mode in LIVE_GUARDED_EXECUTION_MODES and not self.paper_trade
    rebalance_drift_threshold: float = 0.05  # 5% drift before rebalance triggers

    # --- Metrics server ------------------------------------------------
    metrics_host: str = "0.0.0.0"
    log_format: str = "text"             # "text" or "json"

    # --- Trade journal -------------------------------------------------
    trade_journal_path_str: str = ""     # fallback string path (unused if trade_journal_path set)

    # --- Overfit Protection (Section 3) --------------------------------
    cpcv_n_splits: int = 10
    cpcv_n_test_splits: int = 2
    cpcv_p_value_threshold: float = 0.20
    deflated_sharpe_confidence: float = 0.95
    feature_stability_min_spearman: float = 0.6
    concept_drift_kl_warn: float = 0.15
    concept_drift_kl_retrain: float = 0.30
    concept_drift_kl_disable: float = 0.50

    # --- Alertmanager webhook ------------------------------------------
    alertmanager_webhook_enabled: bool = False
    alertmanager_webhook_port: int = 9093

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all fields after construction."""
        from wallet_manager.exceptions import ConfigurationError

        errors: list[str] = []

        if self.retry_attempts < 1:
            errors.append(f"retry_attempts must be >= 1; got {self.retry_attempts}.")
        if self.retry_base_delay <= 0:
            errors.append(f"retry_base_delay must be > 0; got {self.retry_base_delay}.")
        if self.circuit_failure_threshold < 1:
            errors.append(f"circuit_failure_threshold must be >= 1; got {self.circuit_failure_threshold}.")
        if self.circuit_recovery_timeout <= 0:
            errors.append(f"circuit_recovery_timeout must be > 0; got {self.circuit_recovery_timeout}.")
        if self.poisson_lambda <= 0:
            errors.append(f"poisson_lambda must be > 0; got {self.poisson_lambda}.")
        if self.gaussian_jitter_sigma < 0:
            errors.append(f"gaussian_jitter_sigma must be >= 0; got {self.gaussian_jitter_sigma}.")
        if self.session_total_timeout <= 0:
            errors.append(f"session_total_timeout must be > 0; got {self.session_total_timeout}.")
        if self.session_connect_timeout <= 0:
            errors.append(f"session_connect_timeout must be > 0; got {self.session_connect_timeout}.")
        if self.log_level.upper() not in VALID_LOG_LEVELS:
            errors.append(f"log_level must be one of {sorted(VALID_LOG_LEVELS)}; got {self.log_level!r}.")
        if self.execution_mode not in VALID_EXECUTION_MODES:
            errors.append(
                f"execution_mode must be one of {sorted(VALID_EXECUTION_MODES)}; "
                f"got {self.execution_mode!r}."
            )
        if self.max_concurrent_wallets < 1:
            errors.append(f"max_concurrent_wallets must be >= 1; got {self.max_concurrent_wallets}.")
        if self.gas_poll_interval <= 0:
            errors.append(f"gas_poll_interval must be > 0; got {self.gas_poll_interval}.")
        if self.gas_wait_timeout <= 0:
            errors.append(f"gas_wait_timeout must be > 0; got {self.gas_wait_timeout}.")
        if self.live_max_total_capital_usd <= 0:
            errors.append(
                f"live_max_total_capital_usd must be > 0; got {self.live_max_total_capital_usd}."
            )
        if self.paper_initial_balance_usd <= 0:
            errors.append(
                f"paper_initial_balance_usd must be > 0; got {self.paper_initial_balance_usd}."
            )
        if self.paper_max_gas_fraction_pct <= 0:
            errors.append(
                "paper_max_gas_fraction_pct must be > 0; "
                f"got {self.paper_max_gas_fraction_pct}."
            )
        if self.robust_paper_adf_entry_pvalue_max <= 0:
            errors.append(
                "robust_paper_adf_entry_pvalue_max must be > 0; "
                f"got {self.robust_paper_adf_entry_pvalue_max}."
            )
        if self.robust_paper_alpha_entry_threshold <= 0:
            errors.append(
                "robust_paper_alpha_entry_threshold must be > 0; "
                f"got {self.robust_paper_alpha_entry_threshold}."
            )
        if self.live_max_wallets < 1:
            errors.append(f"live_max_wallets must be >= 1; got {self.live_max_wallets}.")
        if self.audit_recent_sessions_limit < 1:
            errors.append(
                f"audit_recent_sessions_limit must be >= 1; got {self.audit_recent_sessions_limit}."
            )
        if not (0 < self.kelly_fraction <= 1.0):
            errors.append(f"kelly_fraction must be in (0, 1]; got {self.kelly_fraction}.")
        if not (0 < self.kelly_max_position_pct <= 1.0):
            errors.append(f"kelly_max_position_pct must be in (0, 1]; got {self.kelly_max_position_pct}.")
        if self.garch_alpha < 0 or self.garch_beta < 0:
            errors.append("garch_alpha and garch_beta must be >= 0.")
        if self.garch_alpha + self.garch_beta >= 1.0:
            errors.append(
                f"garch_alpha + garch_beta must be < 1.0; got {self.garch_alpha + self.garch_beta}."
            )
        if not (0 < self.var_confidence < 1.0):
            errors.append(f"var_confidence must be in (0, 1); got {self.var_confidence}.")
        if self.drawdown_halt_threshold <= self.drawdown_reduce_threshold:
            errors.append(
                "drawdown_halt_threshold must be > drawdown_reduce_threshold."
            )

        # Portfolio weight validation
        weight_sum = round(self.yield_weight + self.algo_weight + self.short_weight + self.airdrop_weight, 10)
        if abs(weight_sum - 1.0) > 1e-9:
            errors.append(
                f"Portfolio weights must sum to 1.0; got "
                f"{self.yield_weight}+{self.algo_weight}+{self.short_weight}+{self.airdrop_weight}={weight_sum}."
            )

        if self.log_format not in ("text", "json"):
            errors.append(f"log_format must be 'text' or 'json'; got {self.log_format!r}.")

        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> Settings:
        """Construct a ``Settings`` instance from environment variables.

        Returns:
            A validated, frozen :class:`Settings` instance.

        Raises:
            ConfigurationError: If any env var has an invalid type or value.
        """
        # Parse liquidity_split_weights as list of floats
        raw_lsw = os.getenv("LIQUIDITY_SPLIT_WEIGHTS", "").strip()
        liquidity_split_weights = [0.40, 0.35, 0.25]
        if raw_lsw:
            with contextlib.suppress(ValueError):
                liquidity_split_weights = [float(x.strip()) for x in raw_lsw.split(",") if x.strip()]

        return cls(
            # Vault
            vault_key=_str("VAULT_KEY", ""),
            # Telegram
            telegram_bot_token=_str("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=_str("TELEGRAM_CHAT_ID", ""),
            telegram_rate_limit=_float("TELEGRAM_RATE_LIMIT", 1.0),
            telegram_rate_burst=_int("TELEGRAM_RATE_BURST", 1),
            # Retry
            retry_attempts=_int("RETRY_ATTEMPTS", 4),
            retry_base_delay=_float("RETRY_BASE_DELAY", 1.5),
            # Circuit Breaker
            circuit_failure_threshold=_int("CIRCUIT_FAILURE_THRESHOLD", 5),
            circuit_recovery_timeout=_float("CIRCUIT_RECOVERY_TIMEOUT", 30.0),
            # Traffic
            poisson_lambda=_float("POISSON_LAMBDA", 2.0),
            gaussian_jitter_sigma=_float("GAUSSIAN_JITTER_SIGMA", 0.3),
            # HTTP sessions
            session_total_timeout=_float("SESSION_TOTAL_TIMEOUT", 30.0),
            session_connect_timeout=_float("SESSION_CONNECT_TIMEOUT", 10.0),
            # Orchestrator
            max_concurrent_wallets=_int("MAX_CONCURRENT_WALLETS", 5),
            gas_poll_interval=_float("GAS_POLL_INTERVAL", 30.0),
            gas_wait_timeout=_float("GAS_WAIT_TIMEOUT", 600.0),
            loop_interval_seconds=_float("LOOP_INTERVAL_SECONDS", 45.0),
            algo_loop_interval_seconds=_float("ALGO_LOOP_INTERVAL_SECONDS", 300.0),
            heartbeat_interval_seconds=_float("HEARTBEAT_INTERVAL_SECONDS", 21600.0),
            default_task_loop_cycles=_int("DEFAULT_TASK_LOOP_CYCLES", 3),
            use_master_orchestrator=_bool("USE_MASTER_ORCHESTRATOR", True),
            # Transaction repair (RBF)
            tx_poll_interval=_float("TX_POLL_INTERVAL", 5.0),
            tx_stuck_deadline=_float("TX_STUCK_DEADLINE", 180.0),
            rbf_gas_bump_min=_float("RBF_GAS_BUMP_MIN", 0.15),
            rbf_gas_bump_max=_float("RBF_GAS_BUMP_MAX", 0.25),
            max_rbf_attempts=_int("MAX_RBF_ATTEMPTS", 3),
            # Multi-chain router / MEV Shield RPC cluster
            infura_key=_str("INFURA_KEY", ""),
            alchemy_key=_str("ALCHEMY_KEY", ""),
            private_rpc_1=_str("PRIVATE_RPC_1", ""),
            private_rpc_2=_str("PRIVATE_RPC_2", ""),
            private_rpc_3=_str("PRIVATE_RPC_3", ""),
            base_rpc_1=_str("BASE_RPC_1", ""),
            base_rpc_2=_str("BASE_RPC_2", ""),
            base_rpc_3=_str("BASE_RPC_3", ""),
            # Analytics DB
            analytics_db_path=_path("ANALYTICS_DB_PATH", "wallet_analytics.db"),
            quant_analytics_db_path=_path("QUANT_ANALYTICS_DB_PATH", "quant_analytics.db"),
            # Logging
            log_level=_str("LOG_LEVEL", "INFO").upper(),
            # Wallet config
            wallet_config_path=_path("WALLET_CONFIG_PATH", "wallets.json"),
            # Signal Intelligence
            zscore_window_size=_int("ZSCORE_WINDOW_SIZE", 288),
            zscore_entry_threshold=_float("ZSCORE_ENTRY_THRESHOLD", -2.8),
            zscore_short_threshold=_float("ZSCORE_SHORT_THRESHOLD", 2.8),
            ofi_tick_range=_int("OFI_TICK_RANGE", 100),
            vwap_window_trades=_int("VWAP_WINDOW_TRADES", 500),
            # MEV Shield / SOR
            sor_enabled_dexes=_str_list("SOR_ENABLED_DEXES", ["uniswap_v3", "sushiswap_v3", "aerodrome"]),
            sor_cache_ttl=_float("SOR_CACHE_TTL", 10.0),
            # Kelly Criterion
            kelly_fraction=_float("KELLY_FRACTION", 0.25),
            kelly_max_position_pct=_float("KELLY_MAX_POSITION_PCT", 0.10),
            kelly_min_trades=_int("KELLY_MIN_TRADES", 10),
            min_trades_before_full_kelly=_int("MIN_TRADES_BEFORE_FULL_KELLY", 500),
            # Flash Crash
            quantx_crash_threshold_pct=_float("QUANTX_CRASH_THRESHOLD_PCT", 8.0),
            quantx_crash_window_seconds=_float("QUANTX_CRASH_WINDOW_SECONDS", 300.0),
            # Gaussian Jitter
            jitter_mean_ms=_int("JITTER_MEAN_MS", 800),
            jitter_std_ms=_int("JITTER_STD_MS", 400),
            jitter_min_ms=_int("JITTER_MIN_MS", 100),
            jitter_max_ms=_int("JITTER_MAX_MS", 2000),
            # Dynamic RBF
            rbf_block_deadline=_int("RBF_BLOCK_DEADLINE", 2),
            rbf_escalation_pct=_int("RBF_ESCALATION_PCT", 15),
            # Engine mode
            quantx_engine_mode=_str("QUANTX_ENGINE_MODE", "quantx").lower(),
            execution_mode=_str("EXECUTION_MODE", "ml_quant").lower(),
            # GARCH
            garch_omega=_float("GARCH_OMEGA", 1e-6),
            garch_alpha=_float("GARCH_ALPHA", 0.10),
            garch_beta=_float("GARCH_BETA", 0.85),
            # Kalman Filter
            kalman_process_noise=_float("KALMAN_PROCESS_NOISE", 1e-5),
            kalman_measurement_noise=_float("KALMAN_MEASUREMENT_NOISE", 1e-3),
            # ML Ensemble
            xgboost_weight=_float("XGBOOST_WEIGHT", 0.40),
            lightgbm_weight=_float("LIGHTGBM_WEIGHT", 0.25),
            lstm_weight=_float("LSTM_WEIGHT", 0.25),
            transformer_weight=_float("TRANSFORMER_WEIGHT", 0.10),
            ml_confidence_threshold=_float("ML_CONFIDENCE_THRESHOLD", 0.65),
            ensemble_active_models=_str("ENSEMBLE_ACTIVE_MODELS", "xgb,lgbm,lstm,transformer"),
            # PPO (Deep Kelly)
            ppo_model_path=_path("PPO_MODEL_PATH", "ml_models/ppo_risk_policy.pt"),
            ppo_min_trades=_int("PPO_MIN_TRADES", 50),
            enhanced_kelly_fraction=_float("ENHANCED_KELLY_FRACTION", 0.5),
            min_paper_trades_before_live=_int("MIN_PAPER_TRADES_BEFORE_LIVE", 50),
            # Drawdown Controller
            drawdown_reduce_threshold=_float("DRAWDOWN_REDUCE_THRESHOLD", 0.05),
            drawdown_halt_threshold=_float("DRAWDOWN_HALT_THRESHOLD", 0.10),
            drawdown_recovery_threshold=_float("DRAWDOWN_RECOVERY_THRESHOLD", 0.03),
            # VaR
            var_confidence=_float("VAR_CONFIDENCE", 0.99),
            var_lookback=_int("VAR_LOOKBACK", 500),
            var_max_position_pct=_float("VAR_MAX_POSITION_PCT", 0.30),
            # Robustness Shield
            fracdiff_d_min=_float("FRACDIFF_D_MIN", 0.1),
            fracdiff_d_max=_float("FRACDIFF_D_MAX", 1.0),
            fracdiff_d_step=_float("FRACDIFF_D_STEP", 0.1),
            fracdiff_min_observations=_int("FRACDIFF_MIN_OBSERVATIONS", 50),
            adf_significance_level=_float("ADF_SIGNIFICANCE_LEVEL", 0.05),
            robust_paper_adf_entry_pvalue_max=_float("ROBUST_PAPER_ADF_ENTRY_PVALUE_MAX", 0.35),
            robust_paper_alpha_entry_threshold=_float("ROBUST_PAPER_ALPHA_ENTRY_THRESHOLD", 0.04),
            # Paths
            algo_price_history_path=_path("ALGO_PRICE_HISTORY_PATH", ""),
            short_state_path=_path("SHORT_STATE_PATH", str(Path(__file__).resolve().parent / "state" / "short_state.json")),
            schedule_data_path=_path("SCHEDULE_DATA_PATH", str(Path(__file__).resolve().parent / "state" / "schedule_data.json")),
            # Aave APY alert
            aave_apy_alert_threshold_pct=_float("AAVE_APY_ALERT_THRESHOLD_PCT", 2.0),
            # Cooldown
            cooldown_min_delay=_float("COOLDOWN_MIN_DELAY", 1.5),
            # OCI
            oci_region=_str("OCI_REGION", "unknown"),
            # Paper Trading
            paper_trade=_bool("PAPER_TRADE", False),
            paper_initial_balance_usd=_float("PAPER_INITIAL_BALANCE_USD", 10_000.0),
            paper_slippage_model=_str("PAPER_SLIPPAGE_MODEL", "gaussian"),
            paper_slippage_mean_bps=_float("PAPER_SLIPPAGE_MEAN_BPS", 5.0),
            paper_slippage_std_bps=_float("PAPER_SLIPPAGE_STD_BPS", 2.0),
            paper_gas_estimation=_bool("PAPER_GAS_ESTIMATION", True),
            paper_max_gas_fraction_pct=_float("PAPER_MAX_GAS_FRACTION_PCT", 5.0),
            paper_db_path=_path("PAPER_DB_PATH", "paper_trades.db"),
            live_trading_enabled=_bool("LIVE_TRADING_ENABLED", False),
            live_tiny_cap_mode=_bool("LIVE_TINY_CAP_MODE", True),
            live_max_total_capital_usd=_float("LIVE_MAX_TOTAL_CAPITAL_USD", 250.0),
            live_max_wallets=_int("LIVE_MAX_WALLETS", 1),
            audit_snapshot_enabled=_bool("AUDIT_SNAPSHOT_ENABLED", True),
            audit_recent_sessions_limit=_int("AUDIT_RECENT_SESSIONS_LIMIT", 25),
            # Monte Carlo
            mc_n_simulations=_int("MC_N_SIMULATIONS", 1000),
            mc_price_jitter_pct=_float("MC_PRICE_JITTER_PCT", 0.02),
            mc_min_win_rate=_float("MC_MIN_WIN_RATE", 0.45),
            rfe_min_features=_int("RFE_MIN_FEATURES", 5),
            rfe_importance_decay_threshold=_float("RFE_IMPORTANCE_DECAY_THRESHOLD", 0.01),
            # Latency Shield
            latency_mean_ms=_float("LATENCY_MEAN_MS", 50.0),
            latency_std_ms=_float("LATENCY_STD_MS", 20.0),
            latency_min_ms=_float("LATENCY_MIN_MS", 10.0),
            latency_max_ms=_float("LATENCY_MAX_MS", 500.0),
            stale_quote_threshold_s=_float("STALE_QUOTE_THRESHOLD_S", 30.0),
            stale_quote_warning_s=_float("STALE_QUOTE_WARNING_S", 15.0),
            price_divergence_threshold_pct=_float("PRICE_DIVERGENCE_THRESHOLD_PCT", 0.5),
            max_consecutive_stale=_int("MAX_CONSECUTIVE_STALE", 3),
            # Walk-Forward Validator
            wf_n_splits=_int("WF_N_SPLITS", 5),
            wf_purge_pct=_float("WF_PURGE_PCT", 0.05),
            wf_embargo_pct=_float("WF_EMBARGO_PCT", 0.01),
            wf_retrain_interval=_int("WF_RETRAIN_INTERVAL", 100),
            wf_min_oos_sharpe=_float("WF_MIN_OOS_SHARPE", 0.5),
            wf_validator_db_path=_path("WF_VALIDATOR_DB_PATH", "wf_validator.db"),
            # Regime Filter / HMM
            hmm_n_states=_int("HMM_N_STATES", 3),
            hmm_refit_interval=_int("HMM_REFIT_INTERVAL", 288),
            hmm_min_observations=_int("HMM_MIN_OBSERVATIONS", 60),
            hmm_news_blackout_before_hours=_float("HMM_NEWS_BLACKOUT_BEFORE_HOURS", 0.5),
            hmm_news_blackout_after_hours=_float("HMM_NEWS_BLACKOUT_AFTER_HOURS", 1.0),
            # Liquidity Alpha
            liquidity_thin_threshold=_float("LIQUIDITY_THIN_THRESHOLD", 30.0),
            liquidity_split_volume_pct=_float("LIQUIDITY_SPLIT_VOLUME_PCT", 0.01),
            liquidity_profile_cache_ttl=_float("LIQUIDITY_PROFILE_CACHE_TTL", 30.0),
            liquidity_split_weights=liquidity_split_weights,
            # Permutation Feature Importance
            permutation_importance_threshold=_float("PERMUTATION_IMPORTANCE_THRESHOLD", 0.01),
            permutation_n_shuffles=_int("PERMUTATION_N_SHUFFLES", 30),
            permutation_min_samples=_int("PERMUTATION_MIN_SAMPLES", 100),
            # OCI Deployer
            watchdog_memory_limit_mb=_int("WATCHDOG_MEMORY_LIMIT_MB", 512),
            watchdog_ws_timeout_s=_float("WATCHDOG_WS_TIMEOUT_S", 30.0),
            watchdog_interval_s=_float("WATCHDOG_INTERVAL_S", 60.0),
            health_api_port=_int("HEALTH_API_PORT", 8080),
            health_api_host=_str("HEALTH_API_HOST", "0.0.0.0"),
            max_restart_attempts=_int("MAX_RESTART_ATTEMPTS", 3),
            restart_cooldown_s=_float("RESTART_COOLDOWN_S", 30.0),
            # Historical Data Pipeline
            historical_data_db_path=_path("HISTORICAL_DATA_DB_PATH", "historical_data.db"),
            historical_data_min_candles=_int("HISTORICAL_DATA_MIN_CANDLES", 105_120),
            historical_data_target_candles=_int("HISTORICAL_DATA_TARGET_CANDLES", 157_680),
            subgraph_url=_str("SUBGRAPH_URL", "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"),
            defillama_enabled=_bool("DEFILLAMA_ENABLED", True),
            binance_fallback_enabled=_bool("BINANCE_FALLBACK_ENABLED", True),
            incremental_update_hour_utc=_int("INCREMENTAL_UPDATE_HOUR_UTC", 2),
            min_new_candles_for_retrain=_int("MIN_NEW_CANDLES_FOR_RETRAIN", 500),
            min_days_between_retrains=_int("MIN_DAYS_BETWEEN_RETRAINS", 7),
            # Overfit Protection
            cpcv_n_splits=_int("CPCV_N_SPLITS", 10),
            cpcv_n_test_splits=_int("CPCV_N_TEST_SPLITS", 2),
            cpcv_p_value_threshold=_float("CPCV_P_VALUE_THRESHOLD", 0.20),
            deflated_sharpe_confidence=_float("DEFLATED_SHARPE_CONFIDENCE", 0.95),
            feature_stability_min_spearman=_float("FEATURE_STABILITY_MIN_SPEARMAN", 0.6),
            concept_drift_kl_warn=_float("CONCEPT_DRIFT_KL_WARN", 0.15),
            concept_drift_kl_retrain=_float("CONCEPT_DRIFT_KL_RETRAIN", 0.30),
            concept_drift_kl_disable=_float("CONCEPT_DRIFT_KL_DISABLE", 0.50),
            # Pool Addresses
            weth_usdc_pool=_str("WETH_USDC_POOL", "0xd0b53d9277642d899df5c87a3966a349a798f224"),
            # Phase 2 new keys
            airdrop_max_gas_fee_ratio=_float("AIRDROP_MAX_GAS_FEE_RATIO", 0.5),
            airdrop_min_swap_usd=_float("AIRDROP_MIN_SWAP_USD", 10.0),
            rpc_reprobe_interval_seconds=_float("RPC_REPROBE_INTERVAL_SECONDS", 300.0),
            yield_min_apy_pct=_float("YIELD_MIN_APY_PCT", 1.0),
            aave_apy_poll_interval_seconds=_float("AAVE_APY_POLL_INTERVAL_SECONDS", 3600.0),
            proactive_alert_interval_seconds=_float("PROACTIVE_ALERT_INTERVAL_SECONDS", 300.0),
            alert_dedup_window_seconds=_float("ALERT_DEDUP_WINDOW_SECONDS", 3600.0),
            metrics_enabled=_bool("METRICS_ENABLED", True),
            metrics_port=_int("METRICS_PORT", 9092),
            metrics_host=_str("METRICS_HOST", "0.0.0.0"),
            log_format=_str("LOG_FORMAT", "text").lower(),
            trade_journal_path=_path("TRADE_JOURNAL_PATH", "trade_journal.jsonl"),
            trade_journal_path_str=_str("TRADE_JOURNAL_PATH", ""),
            trade_journal_max_mb=_float("TRADE_JOURNAL_MAX_MB", 100.0),
            feature_drift_kl_threshold=_float("FEATURE_DRIFT_KL_THRESHOLD", 0.1),
            feature_drift_window=_int("FEATURE_DRIFT_WINDOW", 288),
            feature_drift_check_interval=_int("FEATURE_DRIFT_CHECK_INTERVAL", 100),
            wf_sliding_window_days=_int("WF_SLIDING_WINDOW_DAYS", 90),
            wf_use_dual_validation=_bool("WF_USE_DUAL_VALIDATION", True),
            vault_key_rotation_days=_int("VAULT_KEY_ROTATION_DAYS", 90),
            # Portfolio allocation weights
            yield_weight=_float("YIELD_WEIGHT", 0.40),
            algo_weight=_float("ALGO_WEIGHT", 0.35),
            short_weight=_float("SHORT_WEIGHT", 0.10),
            airdrop_weight=_float("AIRDROP_WEIGHT", 0.15),
            monthly_injection_usd=_float("MONTHLY_INJECTION_USD", 150.0),
            rebalance_drift_threshold=_float("REBALANCE_DRIFT_THRESHOLD", 0.05),
            # Alertmanager webhook
            alertmanager_webhook_enabled=_bool("ALERTMANAGER_WEBHOOK_ENABLED", False),
            alertmanager_webhook_port=_int("ALERTMANAGER_WEBHOOK_PORT", 9093),
        )


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return the module-level ``Settings`` singleton.

    On first call, ``Settings.from_env()`` is invoked and the result cached.
    In tests, construct ``Settings(...)`` directly and inject it instead.

    Returns:
        The process-wide :class:`Settings` instance.
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings.from_env()
    return _settings_cache


def set_settings(settings: Settings) -> None:
    """Override the cached settings singleton. Useful for tests.

    Args:
        settings: The Settings instance to use as the global singleton.
    """
    global _settings_cache
    _settings_cache = settings


def reset_settings_cache() -> None:
    """Clear the cached settings singleton. For use in tests only."""
    global _settings_cache
    _settings_cache = None


def validate_wallet_config(data: dict[str, object]) -> None:
    """Validate a single wallet config dict loaded from wallets.json.

    Args:
        data: Raw dict parsed from wallets.json.

    Raises:
        ConfigurationError: If any required field is missing or chain_id is not 8453.
    """
    from wallet_manager.exceptions import ConfigurationError

    required = ["wallet_id", "private_key", "rpc_url", "chain_id"]
    for key in required:
        if key not in data:
            raise ConfigurationError(f"wallets.json: missing '{key}'.")
    if data.get("chain_id") != 8453:
        raise ConfigurationError("wallets.json: chain_id must be 8453 (Base).")
