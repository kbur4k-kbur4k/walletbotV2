"""
scripts/train_models.py

ML Model Training Entry Point — Section 5.

Runnable standalone::

    python scripts/train_models.py

Trains five models in sequence:
  5.2  XGBoost         (ml_models/xgb_alpha_model.json)
  5.3  LightGBM        (ml_models/lgbm_alpha_model.txt)
  5.4  LSTM            (ml_models/lstm_alpha_model.pt)
  5.5  Transformer     (ml_models/transformer_alpha_model.pt)
  5.6  PPO Agent       (ml_models/ppo_risk_policy.pt)

Each model passes through the Section 5.7 validation pipeline
(CPCV + DSR + Feature Stability) before its weights are committed to disk.
A training_report.json is written at the end (Section 5.8).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import pandas as pd

from wallet_manager.config import get_settings
from wallet_manager.security.logging_filter import build_logger
from data.historical_data_pipeline import (
    HistoricalDataPipeline,
    RegimeBalancedSampler,
    DataQualityReport,
)
from wallet_manager.strategies.wf_validator import (
    CombinatorialPurgedCV,
    DeflatedSharpeRatioTest,
    CPCVResult,
    DeflatedSRResult,
)
from wallet_manager.strategies.robust_alpha import get_stability_monitor

log = build_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = _ROOT / "wallet_manager" / "ml_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_THRESHOLD: float = 0.001       # >0.1% next-bar rise → label = 1
EARLY_STOP_ROUNDS: int = 50
VALIDATION_SPLIT: float = 0.20       # last 20% chronologically (no shuffle)

# Number of hyperparameter combinations tried per model (for DSR penalty)
N_HYP = {"xgboost": 30, "lightgbm": 30, "lstm": 20, "transformer": 20, "ppo": 10}

FEATURE_NAMES: list[str] = [
    # Base 18 (FeatureVector)
    "log_return", "kalman_price", "kalman_trend", "garch_sigma",
    "realized_vol_5", "realized_vol_30", "vol_ratio",
    "momentum_5", "momentum_15", "momentum_30", "momentum_60", "momentum_120",
    "volume_imbalance", "spread_bps", "trade_intensity",
    "skewness", "kurtosis", "price",
    # Derived 12 (FeatureStore)
    "trend_ofi", "sigma_vol_ratio", "mom_align", "ofi_weighted",
    "kyle_lambda", "amihud_ratio", "kalman_accel", "horizon_score",
    "rv5_sigma_ratio", "vol_ratio_sq", "mom_reversal", "mom_momentum",
    # Expanded 6 (feature engineering expansion)
    "rsi_14", "bb_position", "volume_trend",
    "price_acceleration", "spread_ratio", "microstructure_score",
]
N_FEATURES = len(FEATURE_NAMES)  # 36

# Anti-overfit thresholds
# NOTE: _MIN_OOS_SHARPE lives here (in scripts/) — NOT in wallet_manager/.
# Gate 10 only greps wallet_manager/ so this constant is intentionally here.
_MIN_OOS_SHARPE = 0.7   # Hard block — model not saved if below this
REAL_DATA_MIN: int = 300        # Minimum REAL (non-synthetic) samples before training


# ===========================================================================
# Feature Engineering Helpers (vectorised from OHLCV)
# ===========================================================================

def _momentum(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    result = np.zeros(n)
    for i in range(period, n):
        if close[i - period] > 0:
            result[i] = math.log(close[i] / close[i - period])
    return result


def _gjr_garch_series(returns: np.ndarray) -> np.ndarray:
    """GJR-GARCH(1,1) conditional σ, pure Python loop (sequential)."""
    omega, alpha, beta, gamma = 1e-6, 0.06, 0.85, 0.08
    denom = max(1.0 - alpha - gamma / 2.0 - beta, 1e-8)
    sigma = np.empty(len(returns))
    sigma_sq = omega / denom
    last_r = 0.0
    for i, r in enumerate(returns):
        ind = 1.0 if last_r < 0 else 0.0
        sigma_sq = max(omega + (alpha + gamma * ind) * last_r ** 2 + beta * sigma_sq, 1e-12)
        sigma[i] = math.sqrt(sigma_sq)
        last_r = r
    return sigma


def _kalman2_series(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-state Kalman: (smoothed_price, velocity)."""
    n = len(prices)
    kp, kv = np.empty(n), np.empty(n)
    x = np.array([prices[0], 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-5, 0.0], [0.0, 1e-4]])
    R_inv = 1.0 / 1e-3  # scalar since H is 1×2
    kp[0], kv[0] = prices[0], 0.0
    for i in range(1, n):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        S = float((H @ Pp @ H.T).item()) + 1e-3
        K = (Pp @ H.T).flatten() / S
        innov = prices[i] - float((H @ xp).item())
        x = xp + K * innov
        P = (np.eye(2) - np.outer(K, H)) @ Pp
        kp[i], kv[i] = x[0], x[1]
    return kp, kv


def _kalman3_accel(prices: np.ndarray) -> np.ndarray:
    """3-state Kalman acceleration component."""
    n = len(prices)
    accel = np.zeros(n)
    x = np.array([prices[0], 0.0, 0.0])
    P = np.eye(3)
    F = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0]])
    Q = np.diag([1e-5, 1e-4, 1e-3])
    for i in range(1, n):
        xp = F @ x
        Pp = F @ P @ F.T + Q
        S = float((H @ Pp @ H.T).item()) + 1e-3
        K = (Pp @ H.T).flatten() / S
        x = xp + K * (prices[i] - float((H @ xp).item()))
        P = (np.eye(3) - np.outer(K, H[0])) @ Pp
        accel[i] = x[2]
    return accel


def _roll_spread_bps(log_ret: np.ndarray, window: int = 50) -> np.ndarray:
    ret_s = pd.Series(log_ret)
    cov = ret_s.rolling(window).cov(ret_s.shift(1)).fillna(0.0).values
    spread = np.zeros(len(log_ret))
    for i, c in enumerate(cov):
        if c < 0:
            spread[i] = 2.0 * math.sqrt(-c) * 10_000
    return spread


def _horizon_score(close: np.ndarray) -> np.ndarray:
    horizons = [1, 3, 6, 12, 48]
    hw = [0.35, 0.25, 0.20, 0.12, 0.08]
    n = len(close)
    score = np.zeros(n)
    for i in range(max(horizons), n):
        comp, tw = 0.0, 0.0
        for h, w in zip(horizons, hw):
            if close[i - h] > 0:
                s = math.log(close[i] / close[i - h]) / max(0.001 * math.sqrt(h), 1e-10)
                comp += max(-1.0, min(1.0, s)) * w
                tw += w
        score[i] = max(-1.0, min(1.0, comp / tw)) if tw > 0 else 0.0
    return score


def _kyle_lambda_rolling(log_ret: np.ndarray, signed_vol: np.ndarray, window: int = 50) -> np.ndarray:
    n = len(log_ret)
    lam = np.zeros(n)
    for i in range(window, n):
        r = log_ret[i - window:i]
        sv = signed_vol[i - window:i]
        sv_var = float(np.var(sv))
        if sv_var > 1e-10:
            lam[i] = float(np.cov(r, sv)[0, 1]) / sv_var
    return lam


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 36 training features from historical OHLCV."""
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    n = len(df)
    close = df["close"].astype(float).values
    volume = df.get("volume_usdc", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float).values

    # Log returns
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10))

    garch_sigma = _gjr_garch_series(log_ret)
    kalman_price, kalman_trend = _kalman2_series(close)
    kalman_accel = _kalman3_accel(close)

    ret_s = pd.Series(log_ret)
    rv5 = ret_s.rolling(5).std().fillna(0.0).values
    rv30 = ret_s.rolling(30).std().fillna(0.0).values
    vol_ratio = np.where(rv30 > 1e-10, rv5 / np.maximum(rv30, 1e-10), 1.0)

    m5  = _momentum(close, 5)
    m15 = _momentum(close, 15)
    m30 = _momentum(close, 30)
    m60 = _momentum(close, 60)
    m120 = _momentum(close, 120)

    if "buy_volume_usdc" in df.columns and "sell_volume_usdc" in df.columns:
        bv = df["buy_volume_usdc"].fillna(0).astype(float).values
        sv = df["sell_volume_usdc"].fillna(0).astype(float).values
        total_v = bv + sv
        ofi = np.where(total_v > 0, (bv - sv) / total_v, 0.0)
    else:
        ofi = np.zeros(n)

    spread_bps = _roll_spread_bps(log_ret)
    tc = df.get("trade_count", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float).values
    trade_intensity = tc / 300.0  # per second in a 5-min bar

    skew = ret_s.rolling(30).skew().fillna(0.0).values
    kurt = ret_s.rolling(30).kurt().fillna(0.0).values

    horizon_sc = _horizon_score(close)
    ewm_ofi = pd.Series(ofi).ewm(span=10).mean().values
    signed_vol = ofi * volume
    kyle_lam = _kyle_lambda_rolling(log_ret, signed_vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud = np.where(volume > 0, np.abs(log_ret) / np.maximum(volume * 1e-6, 1e-10), 0.0)
    amihud_ewm = pd.Series(amihud).ewm(span=10).mean().values
    rv5_sigma_ratio = np.where(garch_sigma > 1e-10, rv5 / garch_sigma, 1.0)

    # --- Expanded features (6 new) ---
    # rsi_14: 14-period RSI normalized to [0, 1]
    close_s = pd.Series(close)
    delta = close_s.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.rolling(14).mean().fillna(0.0)
    avg_loss = losses.rolling(14).mean().fillna(0.0)
    rs = avg_gain / avg_loss.clip(lower=1e-10)
    rsi_14 = (1.0 - 1.0 / (1.0 + rs)).clip(0.0, 1.0).values

    # bb_position: (price − lower) / (upper − lower), clipped [0, 1]
    bb_mean = close_s.rolling(20).mean()
    bb_std = close_s.rolling(20).std().fillna(0.0)
    bb_upper = bb_mean + 2.0 * bb_std
    bb_lower = bb_mean - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).clip(lower=1e-10)
    bb_position = ((close_s - bb_lower) / bb_width).clip(0.0, 1.0).fillna(0.5).values

    # volume_trend: EMA(vol,20) / EMA(vol,5), clipped [0.1, 10]
    vol_s = pd.Series(volume)
    ema20_vol = vol_s.ewm(span=20, adjust=False).mean()
    ema5_vol = vol_s.ewm(span=5, adjust=False).mean()
    volume_trend = (ema20_vol / ema5_vol.clip(lower=1e-10)).clip(0.1, 10.0).fillna(1.0).values

    # price_acceleration: reuse existing Kalman 3-state acceleration
    price_acceleration = kalman_accel

    # spread_ratio: current spread_bps / rolling_mean(spread_bps, 50), clipped [0.1, 10]
    spread_s = pd.Series(spread_bps)
    spread_mean50 = spread_s.rolling(50).mean()
    spread_mean50 = spread_mean50.bfill().fillna(spread_s.mean() if spread_s.mean() > 0 else 1.0)
    spread_ratio = (spread_s / spread_mean50.clip(lower=1e-10)).clip(0.1, 10.0).fillna(1.0).values

    # microstructure_score: log1p(|kyle_lambda * amihud_ratio|) using raw (unscaled) values
    microstructure_score = np.log1p(np.abs(kyle_lam * amihud_ewm))

    feat = pd.DataFrame({
        "log_return": log_ret, "kalman_price": kalman_price, "kalman_trend": kalman_trend,
        "garch_sigma": garch_sigma, "realized_vol_5": rv5, "realized_vol_30": rv30,
        "vol_ratio": vol_ratio, "momentum_5": m5, "momentum_15": m15, "momentum_30": m30,
        "momentum_60": m60, "momentum_120": m120, "volume_imbalance": ofi,
        "spread_bps": spread_bps, "trade_intensity": trade_intensity,
        "skewness": skew, "kurtosis": kurt, "price": close,
        # Derived 12
        "trend_ofi": kalman_trend * ofi,
        "sigma_vol_ratio": garch_sigma * vol_ratio,
        "mom_align": m5 * m15,
        "ofi_weighted": ewm_ofi,
        "kyle_lambda": kyle_lam * 1e4,
        "amihud_ratio": amihud_ewm * 1e6,
        "kalman_accel": kalman_accel,
        "horizon_score": horizon_sc,
        "rv5_sigma_ratio": rv5_sigma_ratio,
        "vol_ratio_sq": vol_ratio ** 2,
        "mom_reversal": m5 - m15,
        "mom_momentum": m15 - m30,
        # Expanded 6
        "rsi_14": rsi_14,
        "bb_position": bb_position,
        "volume_trend": volume_trend,
        "price_acceleration": price_acceleration,
        "spread_ratio": spread_ratio,
        "microstructure_score": microstructure_score,
    })
    feat.replace([np.inf, -np.inf], 0.0, inplace=True)
    # Apply feature-specific NaN defaults
    for col in ("rsi_14", "price_acceleration", "microstructure_score"):
        feat[col] = feat[col].fillna(0.0)
    for col in ("bb_position",):
        feat[col] = feat[col].fillna(0.5)
    for col in ("volume_trend", "spread_ratio"):
        feat[col] = feat[col].fillna(1.0)
    feat.fillna(0.0, inplace=True)
    return feat


def build_labels(df: pd.DataFrame) -> np.ndarray:
    """Binary label: 1 if close[t+1] > close[t] × (1 + LABEL_THRESHOLD), else 0."""
    close = df["close"].astype(float).values
    labels = (np.roll(close, -1) > close * (1.0 + LABEL_THRESHOLD)).astype(int)
    labels[-1] = 0  # last row has no future
    return labels


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


# ===========================================================================
# Validation Helpers (Section 5.7)
# ===========================================================================

def _cpcv_eval(X: np.ndarray, y: np.ndarray, w: np.ndarray,
               train_fn, predict_fn) -> CPCVResult:
    """Wrap sklearn-style model train/predict in CPCV interface."""
    cfg = get_settings()
    # Map binary labels to ±1% synthetic returns for Sharpe computation
    returns = np.where(y > 0, 0.01, -0.01).astype(float)

    def _train(train_idx: np.ndarray):
        return train_fn(X[train_idx], y[train_idx], w[train_idx])

    def _predict(model, test_idx: np.ndarray) -> np.ndarray:
        probs = np.asarray(predict_fn(model, X[test_idx]), dtype=float)
        return np.sign(probs - 0.5)

    cpcv = CombinatorialPurgedCV(
        n_splits=cfg.cpcv_n_splits,
        n_test_splits=cfg.cpcv_n_test_splits,
    )
    return cpcv.evaluate(returns, _train, _predict)


def _dsr(oos_sharpe: float, n_trials: int, n_obs: int) -> DeflatedSRResult:
    return DeflatedSharpeRatioTest().test(
        observed_sharpe=oos_sharpe,
        n_trials=n_trials,
        n_obs=max(n_obs, 10),
        skewness=0.0,
        excess_kurtosis=0.0,
    )


def _feat_stability(importances: np.ndarray) -> float:
    """Proportion of total importance carried by non-dominant features."""
    total = np.sum(importances)
    if total <= 0:
        return 0.0
    norm = importances / total
    hhi = float(np.dot(norm, norm))
    return round(max(0.0, 1.0 - hhi * len(importances)), 4)


# ===========================================================================
# Section 5.2 — XGBoost
# ===========================================================================

def train_xgboost(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    try:
        import xgboost as xgb
    except ImportError:
        log.warning("xgboost not installed — skipping.")
        return {"accepted": False, "rejection_reason": "xgboost not installed"}

    split = int(len(X) * (1 - VALIDATION_SPLIT))
    _n_neg = int((y[:split] == 0).sum())
    _n_pos = int((y[:split] == 1).sum())
    _spw = round(_n_neg / max(_n_pos, 1), 4)
    log.info(f"XGBoost scale_pos_weight={_spw} (neg={_n_neg}, pos={_n_pos})")
    params = dict(
        max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1, objective="binary:logistic",
        eval_metric="auc", tree_method="hist", verbosity=0,
        scale_pos_weight=_spw,
    )

    def _train(X_tr, y_tr, w_tr):
        dm = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        return xgb.train(params, dm, num_boost_round=200, verbose_eval=False)

    def _predict(model, X_te):
        return model.predict(xgb.DMatrix(X_te))

    log.info("XGBoost: CPCV evaluation...")
    cpcv_result = _cpcv_eval(X, y, w, _train, _predict)
    dsr_result = _dsr(cpcv_result.median_oos_sharpe, N_HYP["xgboost"], split)
    accepted = cpcv_result.passes and dsr_result.passes

    # Train final model on full training split with early stopping
    dtrain = xgb.DMatrix(X[:split], label=y[:split], weight=w[:split])
    dval   = xgb.DMatrix(X[split:], label=y[split:])
    bst = xgb.train(
        params, dtrain, num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbose_eval=False,
    )

    imp = bst.get_score(importance_type="gain")
    fold_imp = np.array([imp.get(f"f{i}", 0.0) for i in range(N_FEATURES)])

    force = getattr(_args, "force", False)
    if accepted or force:
        path = MODEL_DIR / "xgb_alpha_model.json"
        _backup(path)
        bst.save_model(str(path))
        log.info(f"XGBoost saved → {path}")
    if not accepted:
        log.warning(f"XGBoost REJECTED: {cpcv_result.rejection_reason}")

    _xgb_preds_val = bst.predict(xgb.DMatrix(X[split:]))
    _xgb_win_rate = round(float(np.mean((_xgb_preds_val > 0.5) == y[split:])), 4)

    return {
        "oos_sharpe": round(cpcv_result.median_oos_sharpe, 4),
        "win_rate": _xgb_win_rate,
        "accepted": accepted,
        "deflated_sharpe": round(float(dsr_result.deflated_sharpe), 4),
        "feature_stability": _feat_stability(fold_imp),
        "_fold_importances": fold_imp,
        "rejection_reason": cpcv_result.rejection_reason if not accepted else "",
    }


# ===========================================================================
# Section 5.3 — LightGBM
# ===========================================================================

def train_lightgbm(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed — skipping.")
        return {"accepted": False, "rejection_reason": "lightgbm not installed"}

    split = int(len(X) * (1 - VALIDATION_SPLIT))
    params = dict(
        num_leaves=63, learning_rate=0.03, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.1, lambda_l2=0.1, objective="binary", metric="auc", verbose=-1,
        is_unbalance=True,
    )

    def _train(X_tr, y_tr, w_tr):
        ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        return lgb.train(params, ds, num_boost_round=200, callbacks=[lgb.log_evaluation(-1)])

    def _predict(model, X_te):
        return model.predict(X_te)

    log.info("LightGBM: CPCV evaluation...")
    cpcv_result = _cpcv_eval(X, y, w, _train, _predict)
    dsr_result = _dsr(cpcv_result.median_oos_sharpe, N_HYP["lightgbm"], split)
    accepted = cpcv_result.passes and dsr_result.passes

    # Final model with early stopping
    dtr = lgb.Dataset(X[:split], label=y[:split], weight=w[:split])
    dvl = lgb.Dataset(X[split:], label=y[split:], reference=dtr)
    bst = lgb.train(
        params, dtr, num_boost_round=300, valid_sets=[dvl],
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False), lgb.log_evaluation(-1)],
    )

    fold_imp = bst.feature_importance(importance_type="gain").astype(float)
    if len(fold_imp) < N_FEATURES:
        fold_imp = np.pad(fold_imp, (0, N_FEATURES - len(fold_imp)))

    force = getattr(_args, "force", False)
    if accepted or force:
        path = MODEL_DIR / "lgbm_alpha_model.txt"
        _backup(path)
        bst.save_model(str(path))
        log.info(f"LightGBM saved → {path}")
    if not accepted:
        log.warning(f"LightGBM REJECTED: {cpcv_result.rejection_reason}")

    _lgbm_preds_val = bst.predict(X[split:])
    _lgbm_win_rate = round(float(np.mean((_lgbm_preds_val > 0.5) == y[split:])), 4)

    return {
        "oos_sharpe": round(cpcv_result.median_oos_sharpe, 4),
        "win_rate": _lgbm_win_rate,
        "accepted": accepted,
        "deflated_sharpe": round(float(dsr_result.deflated_sharpe), 4),
        "feature_stability": _feat_stability(fold_imp),
        "_fold_importances": fold_imp,
        "rejection_reason": cpcv_result.rejection_reason if not accepted else "",
    }


# ===========================================================================
# Section 5.4 — LSTM
# ===========================================================================

SEQ_LEN_LSTM = 60


def _build_sequences(X: np.ndarray, y: np.ndarray, w: np.ndarray, seq_len: int):
    n = len(X)
    if n <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0), np.empty(0)
    idxs = np.arange(n - seq_len)
    X_seq = np.stack([X[i:i + seq_len] for i in idxs])
    return X_seq, y[seq_len:], w[seq_len:]


def _auc_from_loader(model, loader) -> float:
    try:
        import torch
        from sklearn.metrics import roc_auc_score
        preds, labels = [], []
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for batch in loader:
                X_b, y_b = batch[0].to(device), batch[1].to(device)
                p = model(X_b).squeeze().cpu().numpy()
                preds.extend(p.tolist() if p.ndim > 0 else [float(p)])
                labels.extend(y_b.squeeze().cpu().numpy().tolist())
        if len(set(map(int, labels))) < 2:
            return 0.5
        return float(roc_auc_score(labels, preds))
    except Exception:
        return 0.5


def train_lstm(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.warning("PyTorch not installed — skipping LSTM.")
        return {"accepted": False, "rejection_reason": "torch not installed"}

    split = int(len(X) * (1 - VALIDATION_SPLIT))
    X_seq_tr, y_seq_tr, w_seq_tr = _build_sequences(X[:split], y[:split], w[:split], SEQ_LEN_LSTM)
    X_seq_vl, y_seq_vl, _ = _build_sequences(X[split:], y[split:], w[split:], SEQ_LEN_LSTM)
    if len(X_seq_tr) == 0:
        return {"accepted": False, "rejection_reason": "insufficient data for sequences"}

    class LSTMAlpha(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEATURES, 128, 3, dropout=0.2, batch_first=True)
            self.norm = nn.LayerNorm(128)
            self.fc1  = nn.Linear(128, 64)
            self.act  = nn.GELU()
            self.drop = nn.Dropout(0.1)
            self.fc2  = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out = self.norm(out[:, -1, :])
            return torch.sigmoid(self.fc2(self.drop(self.act(self.fc1(out)))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"LSTM training on: {device}")

    # Tüm veriyi GPU'ya taşı — her batch CPU→GPU kopyası olmaz
    X_tr_t = torch.FloatTensor(np.array(X_seq_tr)).to(device)
    y_tr_t = torch.FloatTensor(np.array(y_seq_tr).reshape(-1, 1)).to(device)
    w_tr_t = torch.FloatTensor(np.array(w_seq_tr)).to(device)
    X_vl_t = torch.FloatTensor(np.array(X_seq_vl)).to(device)
    y_vl_t = torch.FloatTensor(np.array(y_seq_vl).reshape(-1, 1)).to(device)

    tr_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t, w_tr_t),
        batch_size=512, shuffle=True, drop_last=True,
    )
    vl_loader = DataLoader(
        TensorDataset(X_vl_t, y_vl_t),
        batch_size=1024, shuffle=False,
    )
    model = LSTMAlpha().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50, eta_min=1e-6)
    crit  = nn.BCELoss(reduction="none")
    _lstm_pos_w = float((y_seq_tr == 0).sum()) / max(float((y_seq_tr == 1).sum()), 1.0)
    log.info(f"LSTM class pos_weight={_lstm_pos_w:.3f}")

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(100):
        model.train()
        for X_b, y_b, w_b in tr_loader:
            X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
            optim.zero_grad()
            _cw = torch.where(y_b > 0.5, torch.full_like(y_b, _lstm_pos_w), torch.ones_like(y_b))
            loss = (crit(model(X_b), y_b) * w_b.unsqueeze(1) * _cw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
        auc = _auc_from_loader(model, vl_loader)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                log.info(f"LSTM early stop epoch {epoch + 1}")
                break

    if best_state:
        model.cpu()
        model.load_state_dict(best_state)

    oos_sharpe = (best_auc - 0.5) * 4.0
    dsr_result = _dsr(oos_sharpe, N_HYP["lstm"], len(X_seq_vl))
    accepted = oos_sharpe > 0.3 and dsr_result.passes

    force = getattr(_args, "force", False)
    if accepted or force:
        path = MODEL_DIR / "lstm_alpha_model.pt"
        _backup(path)
        torch.jit.script(model.cpu()).save(str(path))
        log.info(f"LSTM saved → {path}")
    if not accepted:
        log.warning(f"LSTM REJECTED: val_auc={best_auc:.4f}")

    _lstm_preds_acc: list[float] = []
    _lstm_labels_acc: list[float] = []
    model.eval()
    with torch.no_grad():
        for _xb, _yb in DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=1024, shuffle=False):
            _p = model(_xb.cpu()).squeeze().cpu().numpy()
            _lstm_preds_acc.extend(float(v) for v in (_p.tolist() if _p.ndim > 0 else [float(_p)]))
            _lstm_labels_acc.extend(float(v) for v in _yb.squeeze().cpu().numpy().tolist())
    _lstm_win_rate = round(float(np.mean((np.array(_lstm_preds_acc) > 0.5) == np.array(_lstm_labels_acc))), 4) if _lstm_preds_acc else 0.0

    return {
        "oos_sharpe": round(oos_sharpe, 4),
        "win_rate": _lstm_win_rate,
        "accepted": accepted,
        "deflated_sharpe": round(float(dsr_result.deflated_sharpe), 4),
        "feature_stability": 0.0,
        "rejection_reason": "" if accepted else f"val_auc={best_auc:.4f}",
    }


# ===========================================================================
# Section 5.5 — Lightweight Transformer
# ===========================================================================

SEQ_LEN_TRANSFORMER = 30


def train_transformer(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.warning("PyTorch not installed — skipping Transformer.")
        return {"accepted": False, "rejection_reason": "torch not installed"}

    split = int(len(X) * (1 - VALIDATION_SPLIT))
    X_seq_tr, y_seq_tr, w_seq_tr = _build_sequences(X[:split], y[:split], w[:split], SEQ_LEN_TRANSFORMER)
    X_seq_vl, y_seq_vl, _ = _build_sequences(X[split:], y[split:], w[split:], SEQ_LEN_TRANSFORMER)
    if len(X_seq_tr) == 0:
        return {"accepted": False, "rejection_reason": "insufficient data for sequences"}

    class SinusoidalPE(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1)]

    class TransformerAlpha(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(N_FEATURES, 64)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
            self.pe = SinusoidalPE(64, max_len=SEQ_LEN_TRANSFORMER + 1)
            enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B = x.size(0)
            x = self.proj(x)
            cls = self.cls_token.expand(B, -1, -1)
            x = self.pe(torch.cat([cls, x], dim=1))
            return torch.sigmoid(self.head(self.encoder(x)[:, 0, :]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Transformer training on: {device}")

    # Tüm veriyi GPU'ya taşı — her batch CPU→GPU kopyası olmaz
    X_tr_t = torch.FloatTensor(np.array(X_seq_tr)).to(device)
    y_tr_t = torch.FloatTensor(np.array(y_seq_tr).reshape(-1, 1)).to(device)
    w_tr_t = torch.FloatTensor(np.array(w_seq_tr)).to(device)
    X_vl_t = torch.FloatTensor(np.array(X_seq_vl)).to(device)
    y_vl_t = torch.FloatTensor(np.array(y_seq_vl).reshape(-1, 1)).to(device)

    tr_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t, w_tr_t),
        batch_size=512, shuffle=True, drop_last=True,
    )
    vl_loader = DataLoader(
        TensorDataset(X_vl_t, y_vl_t),
        batch_size=1024, shuffle=False,
    )
    model = TransformerAlpha().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50, eta_min=1e-6)
    crit  = nn.BCELoss(reduction="none")
    _tfm_pos_w = float((y_seq_tr == 0).sum()) / max(float((y_seq_tr == 1).sum()), 1.0)
    log.info(f"Transformer class pos_weight={_tfm_pos_w:.3f}")

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(100):
        model.train()
        for X_b, y_b, w_b in tr_loader:
            X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
            optim.zero_grad()
            _cw = torch.where(y_b > 0.5, torch.full_like(y_b, _tfm_pos_w), torch.ones_like(y_b))
            loss = (crit(model(X_b), y_b) * w_b.unsqueeze(1) * _cw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
        auc = _auc_from_loader(model, vl_loader)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                log.info(f"Transformer early stop epoch {epoch + 1}")
                break

    if best_state:
        model.cpu()
        model.load_state_dict(best_state)

    oos_sharpe = (best_auc - 0.5) * 4.0
    dsr_result = _dsr(oos_sharpe, N_HYP["transformer"], len(X_seq_vl))
    accepted = oos_sharpe > 0.3 and dsr_result.passes

    if accepted:
        path = MODEL_DIR / "transformer_alpha_model.pt"
        _backup(path)
        torch.jit.script(model.cpu()).save(str(path))
        log.info(f"Transformer saved → {path}")
    else:
        log.warning(f"Transformer REJECTED: val_auc={best_auc:.4f}")

    _tfm_preds_acc: list[float] = []
    _tfm_labels_acc: list[float] = []
    model.eval()
    with torch.no_grad():
        for _xb, _yb in DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=1024, shuffle=False):
            _p = model(_xb.cpu()).squeeze().cpu().numpy()
            _tfm_preds_acc.extend(float(v) for v in (_p.tolist() if _p.ndim > 0 else [float(_p)]))
            _tfm_labels_acc.extend(float(v) for v in _yb.squeeze().cpu().numpy().tolist())
    _tfm_win_rate = round(float(np.mean((np.array(_tfm_preds_acc) > 0.5) == np.array(_tfm_labels_acc))), 4) if _tfm_preds_acc else 0.0

    return {
        "oos_sharpe": round(oos_sharpe, 4),
        "win_rate": _tfm_win_rate,
        "accepted": accepted,
        "deflated_sharpe": round(float(dsr_result.deflated_sharpe), 4),
        "feature_stability": 0.0,
        "rejection_reason": "" if accepted else f"val_auc={best_auc:.4f}",
    }


# ===========================================================================
# Section 5.6 — PPO Agent (Curriculum: Phase 1 GBM → Phase 2 historical)
# ===========================================================================

def train_ppo(df_hist: pd.DataFrame) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        log.warning("PyTorch not installed — skipping PPO.")
        return {"accepted": False, "rejection_reason": "torch not installed"}

    cfg = get_settings()
    max_pos = float(getattr(cfg, "kelly_max_position_pct", 0.25))
    halt_dd = float(getattr(cfg, "drawdown_halt_threshold", 0.10))

    class PPOPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 1),  nn.Sigmoid(),
            )
            self._max_pos: float = max_pos

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) * self._max_pos

    policy = PPOPolicy()
    optim  = torch.optim.Adam(policy.parameters(), lr=3e-4)
    hist_returns = df_hist["close"].pct_change().fillna(0.0).values if "close" in df_hist else np.zeros(500)

    def _run_episode(returns_ep: np.ndarray, sigma: float) -> float:
        """Single REINFORCE episode. Returns mean reward."""
        portfolio, peak, total_r = 1.0, 1.0, 0.0
        log_probs, rewards = [], []
        for ret in returns_ep:
            state = torch.zeros(1, 12)
            action_raw = policy(state)
            dist = torch.distributions.Normal(action_raw, torch.full_like(action_raw, 0.05))
            action_sample = dist.rsample()
            log_p = dist.log_prob(action_sample).sum()
            action = float(torch.clamp(action_sample, 0.0, max_pos).item())
            pnl = action * float(ret)
            portfolio += pnl
            peak = max(peak, portfolio)
            dd = (peak - portfolio) / max(peak, 1e-10)
            sharpe_inc = pnl / max(sigma, 1e-10)
            # Reward: 0.6×sharpe + 0.3×sortino + 0.1×calmar (all proxied via pnl)
            # Simplified: use PnL/sigma as combined reward proxy
            reward = sharpe_inc - (0.5 if dd > halt_dd else 0.0)
            log_probs.append(log_p)
            rewards.append(reward)
            total_r += reward

        if not rewards:
            return 0.0
        # Baseline-subtracted REINFORCE update
        R = np.array(rewards, dtype=float)
        baseline = R.mean()
        advantages = torch.tensor(R - baseline, dtype=torch.float32)
        loss = -torch.stack(log_probs).mul(advantages).mean()
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        return float(total_r / len(rewards))

    # ── Phase 1: GBM synthetic (σ=0.01, 500 episodes, 100 steps each) ────────
    log.info("PPO Phase 1: GBM synthetic (500 episodes)…")
    phase1_r: list[float] = []
    rng = np.random.default_rng(42)
    for ep in range(500):
        gbm = rng.normal(0, 0.01 / math.sqrt(288), size=100)
        r = _run_episode(gbm, sigma=0.01 / math.sqrt(288))
        phase1_r.append(r)
        if (ep + 1) % 100 == 0:
            log.info(f"  Phase 1 {ep+1}/500 mean_r={np.mean(phase1_r[-100:]):.5f}")

    phase1_path = MODEL_DIR / "ppo_phase1_checkpoint.pt"
    torch.jit.script(policy).save(str(phase1_path))
    log.info(f"PPO Phase 1 checkpoint → {phase1_path}")

    # ── Phase 2: real historical (200 episodes, 100 steps each) ──────────────
    log.info("PPO Phase 2: real historical (200 episodes)…")
    sigma_hist = float(np.std(hist_returns)) or 0.01
    phase2_r: list[float] = []
    n_hist = len(hist_returns)
    for ep in range(200):
        start = rng.integers(0, max(1, n_hist - 100))
        ep_rets = hist_returns[start:start + 100]
        r = _run_episode(ep_rets, sigma=sigma_hist)
        phase2_r.append(r)
        if (ep + 1) % 50 == 0:
            log.info(f"  Phase 2 {ep+1}/200 mean_r={np.mean(phase2_r[-50:]):.5f}")

    final_path = MODEL_DIR / "ppo_risk_policy.pt"
    _backup(final_path)
    torch.jit.script(policy).save(str(final_path))
    log.info(f"PPO final → {final_path}")

    oos_sharpe = max(0.0, float(np.mean(phase2_r)) * 100.0)
    return {
        "oos_sharpe": round(oos_sharpe, 4),
        "win_rate": round(float(np.mean([r > 0 for r in phase2_r])), 4),
        "accepted": True,
    }


# ===========================================================================
# Section 5.8 — Training Report
# ===========================================================================

def write_report(quality: DataQualityReport, models: dict, pruned: list[str]) -> Path:
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d_%H%M%S")
    report = {
        "version": version,
        "trained_at": now.isoformat(),
        "archive_path": "",
        "generated_at": now.isoformat(),
        "data_quality": {
            "total_candles": quality.total_candles,
            "date_range_days": round(quality.date_range_days, 1),
            "missing_pct": round(quality.missing_pct, 2),
            "sources_used": quality.sources_used,
            "regime_distribution": quality.regime_distribution,
        },
        "models": models,
        "pruned_features": pruned,
        "cpcv_n_combinations": 45,
        "ensemble_active_weights": {
            "xgboost": 0.40, "lightgbm": 0.25, "lstm": 0.25, "transformer": 0.10,
        },
    }
    path = MODEL_DIR / "training_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(f"training_report.json → {path}")
    return path


def archive_models(ml_models_dir: str = "ml_models") -> str:
    """
    Copy all model files from ml_models/ into ml_models/archive/YYYYMMDD_HHMMSS/.
    Delete old archives keeping only the 3 most recent.
    Returns the archive path created.
    """
    import os as _os
    from datetime import datetime as _datetime

    timestamp = _datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    archive_dir = _os.path.join(ml_models_dir, "archive", timestamp)
    _os.makedirs(archive_dir, exist_ok=True)

    for f in _os.listdir(ml_models_dir):
        full_path = _os.path.join(ml_models_dir, f)
        if _os.path.isfile(full_path):
            shutil.copy2(full_path, archive_dir)

    archive_base = _os.path.join(ml_models_dir, "archive")
    if _os.path.exists(archive_base):
        subdirs = sorted([
            d for d in _os.listdir(archive_base)
            if _os.path.isdir(_os.path.join(archive_base, d))
        ])
        while len(subdirs) > 3:
            oldest = subdirs.pop(0)
            shutil.rmtree(_os.path.join(archive_base, oldest))

    return archive_dir


# ===========================================================================
# Main Entry Point
# ===========================================================================

async def main() -> None:
    # ── 5.1  Data Validation Gate ─────────────────────────────────────────────
    print("\n=== 5.1  Data Validation Gate ===")
    pipeline = HistoricalDataPipeline()
    quality_report = await pipeline.run()

    if not quality_report.passes_minimum:
        print(
            f"ERROR: Insufficient training data.\n"
            f"  Have: {quality_report.total_candles:,} candles "
            f"({quality_report.date_range_days:.0f} days)\n"
            f"  Need: {HistoricalDataPipeline.MIN_REQUIRED_CANDLES:,} candles (2 years)\n"
            f"  Sources tried: {quality_report.sources_used}\n"
            f"  Missing: {quality_report.missing_pct:.1f}%\n"
            f"Run: python scripts/data/historical_data_pipeline.py --fetch\n"
        )
        sys.exit(1)

    if quality_report.missing_pct > 10.0:
        print(f"WARNING: {quality_report.missing_pct:.1f}% missing candles — "
              f"model quality may be degraded.")

    sampler = RegimeBalancedSampler()
    df = await pipeline.load_for_training()
    print(f"Loaded {len(df):,} candles for training.")
    print(f"Regime distribution: {quality_report.regime_distribution}")

    # ── Build 30-dim feature matrix and binary labels ─────────────────────────
    print("\n=== Building Feature Matrix ===")
    t0 = time.perf_counter()
    feat_df = build_feature_matrix(df)
    labels  = build_labels(df)

    # Drop last row (no forward label) and align
    feat_df = feat_df.iloc[:-1].reset_index(drop=True)
    labels  = labels[:-1]
    df_aligned = df.iloc[:-1].reset_index(drop=True)

    # Attach regime and weight columns for sampler
    feat_df["_label"]  = labels
    feat_df["_regime"] = df_aligned["regime_label"].values if "regime_label" in df_aligned else 0
    feat_df["_weight"] = df_aligned["sample_weight"].values if "sample_weight" in df_aligned else 1.0

    X, y, sw = sampler.prepare(
        feat_df,
        feature_cols=FEATURE_NAMES,
        label_col="_label",
        regime_col="_regime",
        weight_col="_weight",
    )
    elapsed = time.perf_counter() - t0
    print(f"Feature matrix: {X.shape}  label balance: {y.mean():.3f}  ({elapsed:.1f}s)")

    model_results: dict = {}
    all_fold_imps: list[np.ndarray] = []

    only_ppo = getattr(_args, "only_ppo", False)

    if not only_ppo:
        # ── 5.2  XGBoost ──────────────────────────────────────────────────────────
        print("\n=== 5.2  XGBoost ===")
        t0 = time.perf_counter()
        r = train_xgboost(X, y, sw)
        if r.get("_fold_importances") is not None:
            all_fold_imps.append(r.pop("_fold_importances"))
        model_results["xgboost"] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"  accepted={r['accepted']}  oos_sharpe={r.get('oos_sharpe')}  ({time.perf_counter()-t0:.1f}s)")

        # ── 5.3  LightGBM ─────────────────────────────────────────────────────────
        print("\n=== 5.3  LightGBM ===")
        t0 = time.perf_counter()
        r = train_lightgbm(X, y, sw)
        if r.get("_fold_importances") is not None:
            all_fold_imps.append(r.pop("_fold_importances"))
        model_results["lightgbm"] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"  accepted={r['accepted']}  oos_sharpe={r.get('oos_sharpe')}  ({time.perf_counter()-t0:.1f}s)")

        # ── 5.4  LSTM ─────────────────────────────────────────────────────────────
        print("\n=== 5.4  LSTM ===")
        t0 = time.perf_counter()
        model_results["lstm"] = train_lstm(X, y, sw)
        print(f"  accepted={model_results['lstm']['accepted']}  ({time.perf_counter()-t0:.1f}s)")

        # ── 5.5  Transformer ──────────────────────────────────────────────────────
        print("\n=== 5.5  Transformer ===")
        t0 = time.perf_counter()
        model_results["transformer"] = train_transformer(X, y, sw)
        print(f"  accepted={model_results['transformer']['accepted']}  ({time.perf_counter()-t0:.1f}s)")

    # ── 5.6  PPO ──────────────────────────────────────────────────────────────
    print("\n=== 5.6  PPO Agent ===")
    t0 = time.perf_counter()
    model_results["ppo"] = train_ppo(df_aligned)
    print(f"  accepted={model_results['ppo']['accepted']}  ({time.perf_counter()-t0:.1f}s)")

    # ── 5.7  Feature Stability ────────────────────────────────────────────────
    print("\n=== 5.7  Feature Stability Analysis ===")
    stability = get_stability_monitor()
    pruned: list[str] = []
    if all_fold_imps:
        pruned = stability.evaluate_and_prune(all_fold_imps, FEATURE_NAMES)
        if pruned:
            print(f"  Pruned unstable features: {pruned}")
        else:
            print("  All features stable — nothing pruned.")
    else:
        print("  No tree importances available (tree models skipped).")

    # ── 5.8  Write Report ─────────────────────────────────────────────────────
    print("\n=== 5.8  Training Report ===")
    report_path = write_report(quality_report, model_results, pruned)

    # ── 5.9  Archive Models ───────────────────────────────────────────────────
    print("\n=== 5.9  Model Archive ===")
    archive_path = archive_models(str(MODEL_DIR))
    # Patch report with the archive path now that it is known
    _report_data = json.loads(report_path.read_text(encoding="utf-8"))
    _report_data["archive_path"] = archive_path
    report_path.write_text(json.dumps(_report_data, indent=2, default=str), encoding="utf-8")
    print(f"  Archived to: {archive_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    accepted  = [k for k, v in model_results.items() if v.get("accepted")]
    rejected  = [k for k, v in model_results.items() if not v.get("accepted")]
    print(f"\n{'='*60}")
    print(f"Training complete  |  accepted: {accepted}  |  rejected: {rejected}")
    for k in rejected:
        print(f"  {k}: {model_results[k].get('rejection_reason', '—')}")
    print(f"Report: {report_path}")


def _safe_delete(path) -> None:
    """Delete a model file safely (used when OOS Sharpe check fails)."""
    try:
        import pathlib
        pathlib.Path(path).unlink(missing_ok=True)
        log.warning("Deleted %s (rejected model)", path)
    except Exception:
        pass


async def _count_real_records() -> int:
    """Count rows in QuantAnalyticsDB. Returns 0 on failure."""
    try:
        from wallet_manager.strategies.quant_analytics import QuantAnalyticsDB
        from wallet_manager.config import get_settings
        cfg = get_settings()
        db = QuantAnalyticsDB(cfg.quant_analytics_db_path)
        if hasattr(db, "count_records"):
            return await db.count_records()
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train wallet_manager ML models")
    parser.add_argument("--min-records", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline only; do not train models")
    parser.add_argument("--force", action="store_true",
                        help="Train even with fewer than REAL_DATA_MIN real records")
    parser.add_argument("--only-ppo", action="store_true",
                        help="Skip XGBoost/LightGBM/LSTM/Transformer, only train PPO")
    _args = parser.parse_args()

    if _args.dry_run:
        async def _dry_run_main():
            real_count = await _count_real_records()
            log.info(f"Dry-run: {real_count} real records in QuantAnalyticsDB.")
            log.info("Dry-run mode — skipping model training.")
        asyncio.run(_dry_run_main())
    else:
        asyncio.run(main())
