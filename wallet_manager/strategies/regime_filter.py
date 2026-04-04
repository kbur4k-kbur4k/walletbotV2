"""
wallet_manager/strategies/regime_filter.py

Economic & Volatility Sentinel — Macro-Aware Regime Detection.

Core Mandate
============
Classify the live market into one of three volatility regimes using a
fully self-contained Gaussian Hidden Markov Model (3-state), and cross-
reference an economic calendar for high-impact news blackout windows.

Regime taxonomy
---------------
* LOW_VOL   — trending, orderly moves.  Full strategy exposure allowed.
* MEDIUM_VOL — sideways / choppy.  Position sizing scaled down.
* HIGH_VOL  — chaos / dislocation.  SAFETY override: 100 % USDC + Aave.
* NEWS_BLACKOUT — within configurable window of a high-impact release.

HMM implementation (Baum-Welch / Viterbi / Forward) is pure NumPy — no
hmmlearn, pomegranate, or any external ML library.

Economic Calendar
-----------------
Maintains a hardcoded table of 2026 FOMC dates plus hooks for CPI, NFP,
GDP, and PPI releases.  Optional async fetch from a configurable API
endpoint; gracefully degrades when the API is unreachable.
"""

from __future__ import annotations

import asyncio
import enum
import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
from numpy.typing import NDArray

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_OBSERVATIONS: Final[int] = 50
"""Minimum data points before HMM activates; simple vol threshold before."""

N_STATES: Final[int] = 3
"""Number of HMM hidden states (LOW / MEDIUM / HIGH volatility)."""

N_FEATURES: Final[int] = 3
"""Observation dimensionality: [log_return, realized_vol, volume_ratio]."""

SELF_TRANSITION_PROB: Final[float] = 0.90
"""Diagonal initialisation for the transition matrix."""

EM_MAX_ITER: Final[int] = 50
"""Maximum Baum-Welch iterations."""

EM_TOL: Final[float] = 1e-6
"""Log-likelihood convergence tolerance for Baum-Welch."""

SIMPLE_VOL_HIGH_THRESHOLD: Final[float] = 0.04
"""Annualised vol above this ⇒ HIGH_VOL (pre-HMM fallback)."""

SIMPLE_VOL_MED_THRESHOLD: Final[float] = 0.015
"""Annualised vol above this ⇒ MEDIUM_VOL (pre-HMM fallback)."""

ROLLING_VOL_WINDOW: Final[int] = 20
"""Window length for realised volatility calculation."""

NEWS_BLACKOUT_HOURS_BEFORE: Final[float] = 2.0
NEWS_BLACKOUT_HOURS_AFTER: Final[float] = 1.0

CALENDAR_FETCH_TIMEOUT: Final[float] = 5.0
"""Seconds before giving up on the economic calendar API."""

CALENDAR_REFRESH_INTERVAL: Final[float] = 3600.0
"""Re-fetch economic calendar every hour."""

SAFETY_OVERRIDE_ACTION: Final[str] = "SAFETY_MODE"
SAFETY_OVERRIDE_REASON: Final[str] = "Move to 100% USDC + Aave Supply"

# Trending-state detection via smoothed velocity
TRENDING_VELOCITY_EWM_ALPHA: Final[float] = 0.05
"""EWM α for price-velocity estimation (~20-bar half-life at 5-min bars)."""

TRENDING_VELOCITY_THRESHOLD: Final[float] = 0.0003
"""EWM velocity magnitude above which LOW_VOL is promoted to TRENDING."""

# --- 2026 FOMC meeting dates (announcement is day-2 afternoon ET) --------
FOMC_2026_DATES: Final[list[tuple[int, int, int]]] = [
    # (year, month, day) — first day of each two-day meeting
    (2026, 1, 28),
    (2026, 3, 18),
    (2026, 5, 6),
    (2026, 6, 17),
    (2026, 7, 29),
    (2026, 9, 16),
    (2026, 10, 28),
    (2026, 12, 9),
]

# --- 2026 CPI releases (approx. 2nd–3rd week, 08:30 ET / 13:30 UTC) ------
CPI_2026_DATES: Final[list[tuple[int, int, int]]] = [
    (2026, 1, 14), (2026, 2, 11), (2026, 3, 11), (2026, 4, 10),
    (2026, 5, 13), (2026, 6, 11), (2026, 7, 15), (2026, 8, 12),
    (2026, 9, 11), (2026, 10, 14), (2026, 11, 13), (2026, 12, 11),
]

# --- 2026 Non-Farm Payrolls (first Friday of each month, 08:30 ET) --------
NFP_2026_DATES: Final[list[tuple[int, int, int]]] = [
    (2026, 1, 9),  (2026, 2, 6),  (2026, 3, 6),  (2026, 4, 3),
    (2026, 5, 1),  (2026, 6, 5),  (2026, 7, 10), (2026, 8, 7),
    (2026, 9, 4),  (2026, 10, 2), (2026, 11, 6), (2026, 12, 4),
]

# --- 2026 GDP Advance Estimates (last Thursday of quarter-end+1 month) ---
GDP_2026_DATES: Final[list[tuple[int, int, int]]] = [
    (2026, 1, 29),  # Q3 2025 advance
    (2026, 4, 30),  # Q4 2025 advance
    (2026, 7, 30),  # Q1 2026 advance
    (2026, 10, 29), # Q2 2026 advance
]

# --- 2026 PPI releases (typically one day before CPI) --------------------
PPI_2026_DATES: Final[list[tuple[int, int, int]]] = [
    (2026, 1, 13), (2026, 2, 10), (2026, 3, 10), (2026, 4, 9),
    (2026, 5, 12), (2026, 6, 10), (2026, 7, 14), (2026, 8, 11),
    (2026, 9, 10), (2026, 10, 13), (2026, 11, 12), (2026, 12, 10),
]

# High-impact event types we care about
HIGH_IMPACT_EVENT_TYPES: Final[list[str]] = [
    "FOMC",
    "CPI",
    "NFP",
    "GDP",
    "PPI",
]


# ---------------------------------------------------------------------------
# Enums & data-classes
# ---------------------------------------------------------------------------

class MarketRegime(enum.Enum):
    """Discrete market regime labels.

    5-state taxonomy (Section 4.3):
        LOW_VOL      — low volatility, flat / no clear trend.
        TRENDING_BULL — low vol + positive Kalman velocity (sustained uptrend).
        TRENDING_BEAR — low vol + negative Kalman velocity (sustained downtrend).
        MEDIUM_VOL   — elevated vol, choppy / sideways.
        HIGH_VOL     — crisis vol; triggers SAFETY override.
        NEWS_BLACKOUT — within blackout window of a high-impact macro event.
    """

    LOW_VOL = "LOW_VOL"
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    MEDIUM_VOL = "MEDIUM_VOL"
    HIGH_VOL = "HIGH_VOL"
    NEWS_BLACKOUT = "NEWS_BLACKOUT"


@dataclass(frozen=True)
class RegimeSignal:
    """Immutable snapshot of the current regime assessment."""

    regime: MarketRegime
    confidence: float
    vol_percentile: float
    is_news_blackout: bool
    override_action: str | None
    reason_code: str


# ---------------------------------------------------------------------------
# Economic Calendar
# ---------------------------------------------------------------------------

@dataclass
class EconomicEvent:
    """A single scheduled macro release."""

    event_type: str
    timestamp: datetime
    description: str
    impact: str = "HIGH"


class EconomicCalendar:
    """Maintains an in-memory schedule of high-impact macro events.

    Hardcoded defaults cover 2026 FOMC dates.  An optional API endpoint
    can be supplied for richer data; fetch failures degrade gracefully.
    """

    def __init__(
        self,
        api_endpoint: str | None = None,
        hours_before: float = NEWS_BLACKOUT_HOURS_BEFORE,
        hours_after: float = NEWS_BLACKOUT_HOURS_AFTER,
    ) -> None:
        self._api_endpoint = api_endpoint
        self._hours_before = hours_before
        self._hours_after = hours_after
        self._events: list[EconomicEvent] = []
        self._last_fetch_ts: float = 0.0
        self._lock = threading.Lock()

        self._load_hardcoded_events()

    # ---- hardcoded defaults ------------------------------------------------

    def _load_hardcoded_events(self) -> None:
        """Populate with known 2026 macro event schedule.

        Sources: FOMC, CPI, NFP, GDP (advance), PPI — all approx. release times.
        Exact dates may shift ±1 day; the economic calendar API (if configured)
        provides authoritative dates and will override these.
        """
        # FOMC rate decisions (2:00 PM ET = 18:00 or 19:00 UTC depending on DST)
        for year, month, day in FOMC_2026_DATES:
            announce = datetime(year, month, day + 1, 18, 0, 0, tzinfo=UTC)
            self._events.append(EconomicEvent(
                event_type="FOMC",
                timestamp=announce,
                description=f"FOMC Rate Decision {year}-{month:02d}-{day + 1:02d}",
            ))

        # CPI — 08:30 ET = 13:30 UTC
        for year, month, day in CPI_2026_DATES:
            self._events.append(EconomicEvent(
                event_type="CPI",
                timestamp=datetime(year, month, day, 13, 30, 0, tzinfo=UTC),
                description=f"CPI {year}-{month:02d}-{day:02d}",
            ))

        # NFP — 08:30 ET = 13:30 UTC
        for year, month, day in NFP_2026_DATES:
            self._events.append(EconomicEvent(
                event_type="NFP",
                timestamp=datetime(year, month, day, 13, 30, 0, tzinfo=UTC),
                description=f"NFP {year}-{month:02d}-{day:02d}",
            ))

        # GDP advance — 08:30 ET = 13:30 UTC
        for year, month, day in GDP_2026_DATES:
            self._events.append(EconomicEvent(
                event_type="GDP",
                timestamp=datetime(year, month, day, 13, 30, 0, tzinfo=UTC),
                description=f"GDP Advance {year}-{month:02d}-{day:02d}",
            ))

        # PPI — 08:30 ET = 13:30 UTC
        for year, month, day in PPI_2026_DATES:
            self._events.append(EconomicEvent(
                event_type="PPI",
                timestamp=datetime(year, month, day, 13, 30, 0, tzinfo=UTC),
                description=f"PPI {year}-{month:02d}-{day:02d}",
            ))

        n_fomc = len(FOMC_2026_DATES)
        n_other = len(CPI_2026_DATES) + len(NFP_2026_DATES) + len(GDP_2026_DATES) + len(PPI_2026_DATES)
        log.info(f"Loaded {n_fomc} FOMC + {n_other} CPI/NFP/GDP/PPI events for 2026.")

    # ---- async fetch -------------------------------------------------------

    async def fetch_events(self) -> None:
        """Optionally pull events from an external API.

        Gracefully degrades: if the endpoint is ``None`` or the request
        fails, we keep using the hardcoded schedule.
        """
        if self._api_endpoint is None:
            log.debug("No calendar API endpoint configured; using hardcoded events.")
            return

        now = time.monotonic()
        if now - self._last_fetch_ts < CALENDAR_REFRESH_INTERVAL:
            return

        try:
            # Use asyncio-friendly HTTP if aiohttp is available; fall back
            # to a thread-pool executor wrapping urllib.
            import json
            import urllib.request

            endpoint = self._api_endpoint
            if endpoint is None:
                return

            loop = asyncio.get_running_loop()
            response_text: str = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(
                        endpoint, timeout=int(CALENDAR_FETCH_TIMEOUT)
                    ).read().decode(),
                ),
                timeout=CALENDAR_FETCH_TIMEOUT + 1.0,
            )

            raw_events = json.loads(response_text)
            new_events: list[EconomicEvent] = []
            for item in raw_events:
                evt_type = item.get("type", "UNKNOWN").upper()
                if evt_type not in HIGH_IMPACT_EVENT_TYPES:
                    continue
                ts = datetime.fromisoformat(item["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                new_events.append(
                    EconomicEvent(
                        event_type=evt_type,
                        timestamp=ts,
                        description=item.get("description", evt_type),
                        impact=item.get("impact", "HIGH"),
                    )
                )

            with self._lock:
                # Merge: keep hardcoded events, add API-fetched ones
                hardcoded_ts = {e.timestamp for e in self._events}
                for evt in new_events:
                    if evt.timestamp not in hardcoded_ts:
                        self._events.append(evt)
                self._events.sort(key=lambda e: e.timestamp)

            self._last_fetch_ts = now
            log.info(f"Fetched {len(new_events)} high-impact events from calendar API.")

        except Exception as exc:
            log.warning(
                "Calendar API fetch failed (%s); continuing with hardcoded events.",
                exc,
            )

    # ---- blackout query ----------------------------------------------------

    def _check_news_blackout(
        self,
        hours_before: float | None = None,
        hours_after: float | None = None,
        ref_time: datetime | None = None,
    ) -> tuple[bool, EconomicEvent | None]:
        """Internal: return (is_blackout, triggering_event | None)."""
        hb = hours_before if hours_before is not None else self._hours_before
        ha = hours_after if hours_after is not None else self._hours_after
        now = ref_time if ref_time is not None else datetime.now(UTC)

        with self._lock:
            for evt in self._events:
                window_start = evt.timestamp - timedelta(hours=hb)
                window_end = evt.timestamp + timedelta(hours=ha)
                if window_start <= now <= window_end:
                    return True, evt
        return False, None

    def is_news_blackout(
        self,
        hours_before: float | None = None,
        hours_after: float | None = None,
        ref_time: datetime | None = None,
    ) -> bool:
        """Check whether *ref_time* falls within a news blackout window.

        Returns
        -------
        True if in a blackout window, False otherwise.
        """
        result, _ = self._check_news_blackout(hours_before, hours_after, ref_time)
        return result

    def upcoming_events(self, horizon_hours: float = 48.0) -> list[EconomicEvent]:
        """Return events within *horizon_hours* from now."""
        now = datetime.now(UTC)
        cutoff = now + timedelta(hours=horizon_hours)
        with self._lock:
            return [e for e in self._events if now <= e.timestamp <= cutoff]

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# Gaussian HMM — fully self-contained NumPy implementation
# ---------------------------------------------------------------------------

class GaussianHMM:
    """3-state multivariate-Gaussian Hidden Markov Model.

    Implements Baum-Welch (EM), Viterbi decoding, and the forward algorithm
    using only NumPy.

    State ordering convention
    -------------------------
    * 0 — LOW_VOL
    * 1 — MEDIUM_VOL
    * 2 — HIGH_VOL
    """

    def __init__(self, n_states: int = N_STATES, n_features: int = N_FEATURES) -> None:
        self.n_states = n_states
        self.n_features = n_features

        # Initial state distribution (uniform)
        self.pi: NDArray[np.float64] = np.full(n_states, 1.0 / n_states)

        # Transition matrix — high self-transition probability
        off_diag = (1.0 - SELF_TRANSITION_PROB) / (n_states - 1)
        self.A: NDArray[np.float64] = np.full((n_states, n_states), off_diag)
        np.fill_diagonal(self.A, SELF_TRANSITION_PROB)

        # Emission parameters: means (n_states, n_features)
        # and covariances (n_states, n_features, n_features)
        self.means: NDArray[np.float64] = np.zeros((n_states, n_features))
        self.covars: NDArray[np.float64] = np.zeros((n_states, n_features, n_features))
        self._init_emission_defaults()

        self._fitted: bool = False

    # ---- default emission parameters --------------------------------------

    def _init_emission_defaults(self) -> None:
        """Sensible priors for crypto vol regimes.

        Features: [log_return, realized_vol, volume_ratio] (trims/pads to n_features)
        """
        # Full 3-feature defaults; sliced to n_features at runtime
        _means_3 = [
            np.array([0.0005, 0.008, 1.0]),    # LOW_VOL
            np.array([0.0, 0.020, 1.3]),         # MEDIUM_VOL
            np.array([-0.002, 0.060, 2.5]),      # HIGH_VOL
        ]
        _diags_3 = [
            np.array([1e-5, 1e-5, 0.04]),
            np.array([5e-5, 5e-5, 0.10]),
            np.array([2e-4, 2e-4, 0.50]),
        ]
        for j in range(min(self.n_states, 3)):
            d = self.n_features
            m = _means_3[j][:d] if d <= 3 else np.pad(_means_3[j], (0, d - 3))
            dv = _diags_3[j][:d] if d <= 3 else np.pad(_diags_3[j], (0, d - 3), constant_values=1e-5)
            self.means[j] = m
            self.covars[j] = np.diag(dv)
        # remaining states (if n_states > 3): identity covariance
        for j in range(3, self.n_states):
            self.covars[j] = np.eye(self.n_features) * 1e-4

    # ---- multivariate Gaussian PDF ----------------------------------------

    def _log_emission_probs(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log p(obs_t | state_j) for all t, j.

        Parameters
        ----------
        obs: (T, n_features)

        Returns
        -------
        (T, n_states) log-probability matrix.
        """
        T = obs.shape[0]
        log_B = np.zeros((T, self.n_states))

        for j in range(self.n_states):
            diff = obs - self.means[j]  # (T, D)
            cov = self.covars[j]  # (D, D)

            # Regularise covariance to avoid singular matrices
            cov_reg = cov + np.eye(self.n_features) * 1e-8

            sign, log_det = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                log_det = -30.0 * self.n_features  # fallback for degenerate cov

            cov_inv = np.linalg.inv(cov_reg)
            # Mahalanobis distance for each time step
            mahal = np.sum(diff @ cov_inv * diff, axis=1)  # (T,)

            log_B[:, j] = (
                -0.5 * self.n_features * math.log(2.0 * math.pi)
                - 0.5 * log_det
                - 0.5 * mahal
            )

        return log_B

    # ---- log-sum-exp utility -----------------------------------------------

    @staticmethod
    def _logsumexp(a: NDArray[np.float64], axis: int | None = None) -> NDArray[np.float64]:
        """Numerically stable log-sum-exp."""
        a_max = np.max(a, axis=axis, keepdims=True)
        # Guard against -inf
        a_max_squeeze = np.where(np.isfinite(a_max), a_max, 0.0)
        out = np.log(np.sum(np.exp(a - a_max_squeeze), axis=axis, keepdims=True)) + a_max_squeeze
        if axis is not None:
            out = np.squeeze(out, axis=axis)
        return np.asarray(out, dtype=np.float64)

    # ---- Forward algorithm -------------------------------------------------

    def _forward(self, log_B: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Compute forward (alpha) variables in log space.

        Parameters
        ----------
        log_B: (T, n_states) log-emission probabilities.

        Returns
        -------
        log_alpha: (T, n_states)
        log_likelihood: scalar
        """
        T = log_B.shape[0]
        log_alpha = np.full((T, self.n_states), -np.inf)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # t = 0
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = (
                    self._logsumexp(log_alpha[t - 1] + log_A[:, j], axis=0)
                    + log_B[t, j]
                )

        log_likelihood = float(self._logsumexp(log_alpha[-1], axis=0))
        return log_alpha, log_likelihood

    # ---- Backward algorithm ------------------------------------------------

    def _backward(self, log_B: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute backward (beta) variables in log space.

        Parameters
        ----------
        log_B: (T, n_states)

        Returns
        -------
        log_beta: (T, n_states)
        """
        T = log_B.shape[0]
        log_beta = np.full((T, self.n_states), -np.inf)
        log_A = np.log(self.A + 1e-300)

        # t = T-1
        log_beta[-1] = 0.0  # log(1)

        for t in range(T - 2, -1, -1):
            for j in range(self.n_states):
                log_beta[t, j] = self._logsumexp(
                    log_A[j, :] + log_B[t + 1, :] + log_beta[t + 1, :], axis=0
                )

        return log_beta

    # ---- Baum-Welch (EM) ---------------------------------------------------

    def fit(self, obs: NDArray[np.float64], max_iter: int = EM_MAX_ITER, tol: float = EM_TOL) -> float:
        """Estimate HMM parameters via Baum-Welch (EM).

        Parameters
        ----------
        obs: (T, n_features) observation sequence.
        max_iter: maximum EM iterations.
        tol: log-likelihood convergence threshold.

        Returns
        -------
        Final log-likelihood.
        """
        T = obs.shape[0]
        if T < 2:
            log.warning("HMM fit called with fewer than 2 observations; skipping.")
            return -np.inf

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # --- E-step ---
            log_B = self._log_emission_probs(obs)
            log_alpha, ll = self._forward(log_B)
            log_beta = self._backward(log_B)

            if not np.isfinite(ll):
                log.info(f"HMM log-likelihood is non-finite at iter {iteration}; aborting.")
                break

            # Convergence check
            delta = ll - prev_ll
            if iteration > 0 and delta < tol:
                log.debug(f"Baum-Welch converged at iteration {iteration} (ΔLL={delta:.2e}).")
                break
            prev_ll = ll

            # Posterior state probabilities: gamma(t, j) = p(z_t = j | obs)
            log_gamma = log_alpha + log_beta
            log_gamma -= self._logsumexp(log_gamma, axis=1)[:, np.newaxis]
            gamma = np.exp(log_gamma)

            # Posterior transition probabilities: xi(t, i, j)
            log_A = np.log(self.A + 1e-300)
            log_xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        log_xi[t, i, j] = (
                            log_alpha[t, i]
                            + log_A[i, j]
                            + log_B[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                denom = self._logsumexp(log_xi[t].ravel())
                log_xi[t] -= denom

            xi = np.exp(log_xi)

            # --- M-step ---
            # Initial distribution
            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)

            # Transition matrix
            xi_sum = xi.sum(axis=0)  # (n_states, n_states)
            row_sums = xi_sum.sum(axis=1, keepdims=True)
            self.A = xi_sum / (row_sums + 1e-300)

            # Emission means & covariances
            for j in range(self.n_states):
                weight = gamma[:, j]  # (T,)
                w_sum = weight.sum() + 1e-300

                self.means[j] = (weight[:, np.newaxis] * obs).sum(axis=0) / w_sum

                diff = obs - self.means[j]  # (T, D)
                weighted_diff = diff * weight[:, np.newaxis]  # (T, D)
                self.covars[j] = (weighted_diff.T @ diff) / w_sum

                # Regularise
                self.covars[j] += np.eye(self.n_features) * 1e-6

        self._fitted = True
        log.info(f"HMM fit complete — final LL={prev_ll:.4f} after ≤{max_iter} iterations.")
        return prev_ll

    # ---- Viterbi decoding --------------------------------------------------

    def viterbi(self, obs: NDArray[np.float64]) -> NDArray[np.intp]:
        """Most-likely state sequence via the Viterbi algorithm.

        Parameters
        ----------
        obs: (T, n_features)

        Returns
        -------
        states: (T,) integer array of state indices.
        """
        T = obs.shape[0]
        log_B = self._log_emission_probs(obs)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # delta(t, j) = max log-prob of path ending in state j at time t
        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=np.intp)

        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(self.n_states):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(candidates))
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Backtrack
        states = np.zeros(T, dtype=np.intp)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ---- Forward filtering (current-time state probs) ----------------------

    def predict_proba(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return posterior state probabilities at the *last* time step.

        Uses the forward algorithm only (no smoothing).

        Parameters
        ----------
        obs: (T, n_features)

        Returns
        -------
        probs: (n_states,) probability vector.
        """
        log_B = self._log_emission_probs(obs)
        log_alpha, _ = self._forward(log_B)
        # Normalize each row to get posterior probabilities (T, n_states)
        log_norm = self._logsumexp(log_alpha, axis=1)[:, np.newaxis]
        log_probs = log_alpha - log_norm
        return np.exp(log_probs)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Regime Filter — main guardrail class
# ---------------------------------------------------------------------------

class RegimeFilter:
    """Macro-aware regime detection combining HMM + economic calendar.

    Typical usage::

        rf = get_regime_filter()
        rf.ingest_market_data(price=3_200.0, volume=1_500_000, timestamp=time.time())
        signal = rf.get_current_regime()
        override, reason = rf.should_override_to_safety()
    """

    def __init__(
        self,
        calendar_api_endpoint: str | None = None,
        blackout_hours_before: float = NEWS_BLACKOUT_HOURS_BEFORE,
        blackout_hours_after: float = NEWS_BLACKOUT_HOURS_AFTER,
    ) -> None:
        self._hmm = GaussianHMM(n_states=N_STATES, n_features=N_FEATURES)
        self._calendar = EconomicCalendar(
            api_endpoint=calendar_api_endpoint,
            hours_before=blackout_hours_before,
            hours_after=blackout_hours_after,
        )

        # Rolling data buffers
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._timestamps: list[float] = []

        # Observation matrix built incrementally
        self._observations: list[NDArray[np.float64]] = []

        # Cached regime state
        self._current_regime: MarketRegime = MarketRegime.LOW_VOL
        self._state_probs: NDArray[np.float64] = np.full(N_STATES, 1.0 / N_STATES)
        self._vol_percentile: float = 0.0
        self._regime_start_ts: float = time.time()
        self._transition_counts: dict[str, int] = {r.value: 0 for r in MarketRegime}
        self._last_fit_count: int = 0
        # Use config refit interval if available, else default 288 (daily at 5-min bars)
        try:
            from wallet_manager.config import get_settings
            _cfg = get_settings()
            self._refit_interval: int = int(getattr(_cfg, "hmm_refit_interval", 288))
        except Exception:
            self._refit_interval = 288

        # Kalman-velocity overlay for TRENDING_BULL / TRENDING_BEAR detection
        self._velocity_ewm: float = 0.0

        self._lock = threading.Lock()

        log.info(f"RegimeFilter initialised (min_obs={MIN_OBSERVATIONS}).")

    # ---- data ingestion ----------------------------------------------------

    def ingest_market_data(
        self,
        price: float,
        volume: float,
        timestamp: float,
    ) -> None:
        """Feed a new price/volume tick into the regime detector.

        Internally computes log returns, rolling realised vol, and
        volume ratio, then appends to the observation matrix.

        Parameters
        ----------
        price: spot price (e.g. WETH/USDC).
        volume: trade volume in quote currency.
        timestamp: UNIX epoch seconds.
        """
        with self._lock:
            self._prices.append(price)
            self._volumes.append(volume)
            self._timestamps.append(timestamp)

            n = len(self._prices)
            if n < 2:
                return  # need at least two prices for a return

            # --- compute observation features ---
            log_return = math.log(self._prices[-1] / self._prices[-2])

            # Realised vol: std of recent log returns
            lookback = min(n - 1, ROLLING_VOL_WINDOW)
            recent_returns = [
                math.log(self._prices[i] / self._prices[i - 1])
                for i in range(n - lookback, n)
            ]
            realized_vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else abs(log_return)

            # Volume ratio vs rolling median
            vol_window = self._volumes[max(0, n - ROLLING_VOL_WINDOW):]
            median_vol = float(np.median(vol_window)) if vol_window else volume
            volume_ratio = volume / median_vol if median_vol > 0 else 1.0

            obs_vector = np.array([log_return, realized_vol, volume_ratio], dtype=np.float64)
            self._observations.append(obs_vector)

            # Update EWM velocity for TRENDING overlay
            self._velocity_ewm = (
                (1.0 - TRENDING_VELOCITY_EWM_ALPHA) * self._velocity_ewm
                + TRENDING_VELOCITY_EWM_ALPHA * log_return
            )

            # --- update regime ---
            self._update_regime()

    # ---- internal regime update --------------------------------------------

    def _update_regime(self) -> None:
        """Re-evaluate current regime.  Must be called under ``self._lock``."""
        n_obs = len(self._observations)

        if n_obs < MIN_OBSERVATIONS:
            # Fallback: simple volatility thresholding
            self._update_regime_simple()
            return

        # Periodically refit the HMM
        if n_obs - self._last_fit_count >= self._refit_interval or not self._hmm.is_fitted:
            obs_matrix = np.array(self._observations, dtype=np.float64)
            try:
                self._hmm.fit(obs_matrix)
                self._last_fit_count = n_obs
            except Exception as exc:
                log.error("HMM fit failed: %s — falling back to simple vol.", exc)
                self._update_regime_simple()
                return

        # Forward filter for current state probabilities
        obs_matrix = np.array(self._observations, dtype=np.float64)
        try:
            proba_2d = self._hmm.predict_proba(obs_matrix)
            # Use last time step's posterior probabilities
            self._state_probs = proba_2d[-1]
        except Exception as exc:
            log.error("HMM predict failed: %s — falling back.", exc)
            self._update_regime_simple()
            return

        most_likely = int(np.argmax(self._state_probs))
        regime_map = {0: MarketRegime.LOW_VOL, 1: MarketRegime.MEDIUM_VOL, 2: MarketRegime.HIGH_VOL}
        new_regime = regime_map[most_likely]
        new_regime = self._apply_trending_overlay(new_regime)

        self._apply_regime_transition(new_regime)
        self._update_vol_percentile()

    def _update_regime_simple(self) -> None:
        """Simple volatility threshold when insufficient data for HMM."""
        if len(self._observations) < 2:
            return

        recent_vols = [obs[1] for obs in self._observations[-ROLLING_VOL_WINDOW:]]
        current_vol = recent_vols[-1]

        if current_vol >= SIMPLE_VOL_HIGH_THRESHOLD:
            new_regime = MarketRegime.HIGH_VOL
            self._state_probs = np.array([0.05, 0.10, 0.85])
        elif current_vol >= SIMPLE_VOL_MED_THRESHOLD:
            new_regime = MarketRegime.MEDIUM_VOL
            self._state_probs = np.array([0.15, 0.70, 0.15])
        else:
            new_regime = MarketRegime.LOW_VOL
            self._state_probs = np.array([0.80, 0.15, 0.05])

        new_regime = self._apply_trending_overlay(new_regime)
        self._apply_regime_transition(new_regime)
        self._update_vol_percentile()

    def _apply_trending_overlay(self, base_regime: MarketRegime) -> MarketRegime:
        """Promote LOW_VOL to TRENDING_BULL or TRENDING_BEAR via Kalman velocity.

        Only LOW_VOL can be promoted — MEDIUM_VOL and HIGH_VOL are not
        subdivided since trending structure requires orderly market conditions.

        Criteria:
            velocity_ewm > +TRENDING_VELOCITY_THRESHOLD  → TRENDING_BULL
            velocity_ewm < -TRENDING_VELOCITY_THRESHOLD  → TRENDING_BEAR
            otherwise                                      → base_regime unchanged
        """
        if base_regime is not MarketRegime.LOW_VOL:
            return base_regime
        if self._velocity_ewm > TRENDING_VELOCITY_THRESHOLD:
            return MarketRegime.TRENDING_BULL
        if self._velocity_ewm < -TRENDING_VELOCITY_THRESHOLD:
            return MarketRegime.TRENDING_BEAR
        return base_regime

    def _apply_regime_transition(self, new_regime: MarketRegime) -> None:
        """Record a regime change if different from current."""
        if new_regime != self._current_regime:
            log.info(
                "Regime transition: %s → %s (probs=%s).",
                self._current_regime.value,
                new_regime.value,
                np.round(self._state_probs, 3),
            )
            self._transition_counts[new_regime.value] = (
                self._transition_counts.get(new_regime.value, 0) + 1
            )
            self._current_regime = new_regime
            self._regime_start_ts = time.time()

    def _update_vol_percentile(self) -> None:
        """Compute realised-vol percentile rank over history."""
        if len(self._observations) < 2:
            self._vol_percentile = 0.0
            return

        all_vols = np.array([obs[1] for obs in self._observations])
        current_vol = all_vols[-1]
        self._vol_percentile = float(np.mean(all_vols <= current_vol)) * 100.0

    # ---- public query API --------------------------------------------------

    def get_current_regime(self) -> RegimeSignal:
        """Return the current regime assessment as a frozen signal.

        News blackout overrides the HMM regime.
        """
        with self._lock:
            blackout, evt = self._calendar._check_news_blackout()
            if blackout:
                reason = f"NEWS_EVENT_IMMINENT: {evt.description}" if evt else "NEWS_EVENT_IMMINENT"
                return RegimeSignal(
                    regime=MarketRegime.NEWS_BLACKOUT,
                    confidence=1.0,
                    vol_percentile=self._vol_percentile,
                    is_news_blackout=True,
                    override_action=SAFETY_OVERRIDE_ACTION,
                    reason_code=reason,
                )

            confidence = float(self._state_probs[np.argmax(self._state_probs)])
            override_action: str | None = None
            reason_code = self._current_regime.value

            if self._current_regime == MarketRegime.HIGH_VOL:
                override_action = SAFETY_OVERRIDE_ACTION
                reason_code = f"HIGH_VOL_DETECTED: {SAFETY_OVERRIDE_REASON}"

            return RegimeSignal(
                regime=self._current_regime,
                confidence=confidence,
                vol_percentile=self._vol_percentile,
                is_news_blackout=False,
                override_action=override_action,
                reason_code=reason_code,
            )

    def should_override_to_safety(self) -> tuple[bool, str]:
        """Check whether the current regime mandates a safety override.

        Returns
        -------
        (should_override, reason_string)
        """
        signal = self.get_current_regime()
        if signal.override_action == SAFETY_OVERRIDE_ACTION:
            return True, signal.reason_code
        return False, ""

    def get_regime_stats(self) -> dict:
        """Return a summary of regime state for dashboards / logging.

        Keys: ``state_probabilities``, ``current_regime``,
        ``regime_duration_s``, ``transition_counts``,
        ``total_observations``, ``vol_percentile``,
        ``hmm_fitted``.
        """
        with self._lock:
            return {
                "state_probabilities": {
                    "LOW_VOL": round(float(self._state_probs[0]), 4),
                    "MEDIUM_VOL": round(float(self._state_probs[1]), 4),
                    "HIGH_VOL": round(float(self._state_probs[2]), 4),
                },
                "current_regime": self._current_regime.value,
                "regime_duration_s": round(time.time() - self._regime_start_ts, 2),
                "transition_counts": dict(self._transition_counts),
                "total_observations": len(self._observations),
                "vol_percentile": round(self._vol_percentile, 2),
                "hmm_fitted": self._hmm.is_fitted,
                "velocity_ewm": round(self._velocity_ewm, 6),
            }

    # ---- calendar passthrough ----------------------------------------------

    def update(
        self,
        price: float,
        volume: float = 0.0,
        gas_gwei: float = 0.0,
        timestamp: float | None = None,
    ) -> RegimeSignal:
        """Ingest a price tick and return the current RegimeSignal.

        Convenience method combining ``ingest_market_data`` + ``get_current_regime``.
        """
        import time as _time
        self.ingest_market_data(
            price=price,
            volume=volume,
            timestamp=timestamp if timestamp is not None else _time.time(),
        )
        return self.get_current_regime()

    async def refresh_calendar(self) -> None:
        """Trigger an async refresh of the economic calendar."""
        await self._calendar.fetch_events()

    def is_news_blackout(
        self,
        hours_before: float = NEWS_BLACKOUT_HOURS_BEFORE,
        hours_after: float = NEWS_BLACKOUT_HOURS_AFTER,
    ) -> bool:
        """Convenience wrapper — True if in a news blackout window."""
        return self._calendar.is_news_blackout(
            hours_before=hours_before, hours_after=hours_after
        )

    @property
    def calendar(self) -> EconomicCalendar:
        """Direct access to the economic calendar instance."""
        return self._calendar

    @property
    def hmm(self) -> GaussianHMM:
        """Direct access to the underlying HMM (for diagnostics)."""
        return self._hmm

    @property
    def observation_count(self) -> int:
        with self._lock:
            return len(self._observations)

    def get_viterbi_path(self) -> list[str]:
        """Run Viterbi on all collected observations.

        Returns a list of regime labels for the full history.
        """
        with self._lock:
            if len(self._observations) < MIN_OBSERVATIONS or not self._hmm.is_fitted:
                return []
            obs_matrix = np.array(self._observations, dtype=np.float64)
            states = self._hmm.viterbi(obs_matrix)
            label_map = {0: "LOW_VOL", 1: "MEDIUM_VOL", 2: "HIGH_VOL"}
            return [label_map[int(s)] for s in states]


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_regime_filter_instance: RegimeFilter | None = None
_singleton_lock = threading.Lock()


def get_regime_filter(
    calendar_api_endpoint: str | None = None,
    blackout_hours_before: float = NEWS_BLACKOUT_HOURS_BEFORE,
    blackout_hours_after: float = NEWS_BLACKOUT_HOURS_AFTER,
) -> RegimeFilter:
    """Return the module-level singleton ``RegimeFilter``.

    Thread-safe.  First call creates the instance; subsequent calls
    return the same object regardless of arguments.
    """
    global _regime_filter_instance
    if _regime_filter_instance is not None:
        return _regime_filter_instance

    with _singleton_lock:
        if _regime_filter_instance is None:
            _regime_filter_instance = RegimeFilter(
                calendar_api_endpoint=calendar_api_endpoint,
                blackout_hours_before=blackout_hours_before,
                blackout_hours_after=blackout_hours_after,
            )
            log.info("RegimeFilter singleton created.")
    return _regime_filter_instance


def reset_regime_filter() -> None:
    """Destroy the singleton, allowing a fresh instance on next call.

    Primarily for testing and hot-reload scenarios.
    """
    global _regime_filter_instance
    with _singleton_lock:
        _regime_filter_instance = None
    log.info("RegimeFilter singleton reset.")
