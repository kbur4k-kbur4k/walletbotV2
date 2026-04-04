"""
wallet_manager/strategies/oci_deployer.py

Oracle Cloud (OCI) Production Prep — Self-Healing Watchdog & Health-Check API.

Responsibility
--------------
Provide production-grade infrastructure tooling for Oracle Cloud deployment:

1. Process Watchdog: Continuous memory and WebSocket liveness monitoring
   with automatic restart on resource exhaustion or stalled connections.
2. Health-Check HTTP API: Minimal ``aiohttp.web`` server exposing
   ``/health``, ``/vitals``, ``/events``, and ``/portfolio`` endpoints
   on localhost for OCI health-probe integration.
3. SystemD Unit Generator: Produces a drop-in ``.service`` template
   with memory caps, journal logging, and restart policies.

The watchdog enforces a 1.5 GB RSS ceiling and a 60-second WebSocket
heartbeat deadline.  When either threshold is breached the process is
restarted (Linux ``os.execv``) or flagged for manual intervention
(Windows).  A five-minute cooldown prevents restart storms.

Concurrency model
-----------------
All monitoring loops are ``asyncio.Task`` instances managed by
``OCIDeployer.start()`` / ``OCIDeployer.stop()``.  The health API
runs on a dedicated ``aiohttp.web.AppRunner`` bound to 127.0.0.1:8484.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency — psutil
# ---------------------------------------------------------------------------

try:
    import psutil as psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]  # psutil unavailable at runtime
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_LIMIT_MB: int = 1536                 # 1.5 GB RSS ceiling
WEBSOCKET_TIMEOUT_S: int = 60               # Max seconds without WS heartbeat
HEALTH_CHECK_INTERVAL_S: int = 30           # Interval for periodic health log
WATCHDOG_INTERVAL_S: int = 10               # Watchdog tick interval
MAX_RESTART_ATTEMPTS: int = 5               # Hard cap on automatic restarts
RESTART_COOLDOWN_S: int = 300               # 5 min between successive restarts
HEALTH_API_PORT: int = 8484                 # Health endpoint listen port
HEALTH_API_HOST: str = "127.0.0.1"          # Loopback only — no external access

# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

WATCHDOG_MEMORY_OK: str = "WATCHDOG_MEMORY_OK"
WATCHDOG_MEMORY_EXCEEDED: str = "WATCHDOG_MEMORY_EXCEEDED"
WATCHDOG_WS_OK: str = "WATCHDOG_WS_OK"
WATCHDOG_WS_LOST: str = "WATCHDOG_WS_LOST"
WATCHDOG_RESTART_TRIGGERED: str = "WATCHDOG_RESTART_TRIGGERED"
WATCHDOG_RESTART_COOLDOWN: str = "WATCHDOG_RESTART_COOLDOWN"
HEALTH_API_STARTED: str = "HEALTH_API_STARTED"
HEALTH_API_STOPPED: str = "HEALTH_API_STOPPED"

# ---------------------------------------------------------------------------
# Maximum number of watchdog events retained in memory
# ---------------------------------------------------------------------------

_MAX_EVENTS: int = 50


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemVitals:
    """Point-in-time snapshot of process health metrics."""

    memory_usage_mb: float
    memory_limit_mb: float
    memory_pct: float
    cpu_pct: float
    uptime_seconds: float
    last_ws_heartbeat: float
    ws_connected: bool
    total_restarts: int
    last_restart_timestamp: float | None
    active_tasks: int
    is_healthy: bool
    timestamp: float


@dataclass(frozen=True)
class WatchdogEvent:
    """Single watchdog lifecycle event."""

    event_type: str       # MEMORY_EXCEEDED | WS_LOST | PROCESS_RESTART | HEALTH_OK | HEALTH_DEGRADED
    details: str
    severity: str         # INFO | WARNING | CRITICAL
    timestamp: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_memory_rss_mb() -> float:
    """Return current process RSS in megabytes.

    Attempts ``psutil`` first.  Falls back to ``/proc/self/status`` on
    Linux.  Returns ``0.0`` when neither source is available (Windows
    without psutil).
    """
    if _HAS_PSUTIL:
        try:
            proc = psutil.Process(os.getpid())
            return float(proc.memory_info().rss / (1024 * 1024))
        except Exception:
            log.warning("psutil.Process().memory_info() failed; falling back")

    # Linux fallback — parse /proc/self/status
    if platform.system() == "Linux":
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except Exception:
            pass

    return 0.0


def _get_cpu_pct() -> float:
    """Return process CPU percent (0–100) or ``0.0`` on failure."""
    if _HAS_PSUTIL:
        try:
            proc = psutil.Process(os.getpid())
            return float(proc.cpu_percent(interval=0))
        except Exception:
            pass
    return 0.0


def _active_task_count() -> int:
    """Count currently running asyncio tasks (excluding the current one)."""
    try:
        all_tasks = asyncio.all_tasks()
        return max(0, len(all_tasks) - 1)
    except RuntimeError:
        return 0


# ---------------------------------------------------------------------------
# ProcessWatchdog
# ---------------------------------------------------------------------------

class ProcessWatchdog:
    """Continuous self-healing monitor for memory and WebSocket liveness.

    The watchdog runs a periodic loop (every ``WATCHDOG_INTERVAL_S`` seconds)
    that checks RSS memory against ``memory_limit_mb`` and verifies that
    a WebSocket heartbeat has been received within ``ws_timeout_s`` seconds.

    When a threshold is breached the watchdog attempts an automatic process
    restart (Linux only) subject to a cooldown window and a hard restart cap.
    """

    def __init__(
        self,
        memory_limit_mb: int = MEMORY_LIMIT_MB,
        ws_timeout_s: int = WEBSOCKET_TIMEOUT_S,
    ) -> None:
        self._memory_limit_mb: int = memory_limit_mb
        self._ws_timeout_s: int = ws_timeout_s

        # Internal state
        self._start_time: float = time.time()
        self._last_ws_heartbeat: float = time.time()
        self._total_restarts: int = 0
        self._last_restart_ts: float | None = None
        self._events: deque[WatchdogEvent] = deque(maxlen=_MAX_EVENTS)
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

        log.info(
            "ProcessWatchdog created  memory_limit=%dMB  ws_timeout=%ds",
            self._memory_limit_mb,
            self._ws_timeout_s,
        )

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the watchdog background loop."""
        if self._running:
            log.warning("ProcessWatchdog already running — ignoring start()")
            return
        self._running = True
        self._start_time = time.time()
        self._last_ws_heartbeat = time.time()
        self._task = asyncio.create_task(self._loop(), name="oci-watchdog")
        log.info("ProcessWatchdog started")

    async def stop(self) -> None:
        """Cancel the watchdog loop and wait for clean exit."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        log.info("ProcessWatchdog stopped")

    # -- main loop -----------------------------------------------------------

    async def _loop(self) -> None:
        """Periodic check loop — runs every ``WATCHDOG_INTERVAL_S`` seconds."""
        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(WATCHDOG_INTERVAL_S)
        except asyncio.CancelledError:
            log.debug("Watchdog loop cancelled")

    async def _tick(self) -> None:
        """Execute a single watchdog pass."""
        mem_ok = await self._check_memory()
        ws_ok = await self._check_websocket()

        if mem_ok and ws_ok:
            self._record_event(
                event_type="HEALTH_OK",
                details="All vitals nominal",
                severity="INFO",
            )
        else:
            degraded_parts: list[str] = []
            if not mem_ok:
                degraded_parts.append("memory")
            if not ws_ok:
                degraded_parts.append("websocket")
            self._record_event(
                event_type="HEALTH_DEGRADED",
                details=f"Degraded subsystems: {', '.join(degraded_parts)}",
                severity="WARNING",
            )

    # -- memory check --------------------------------------------------------

    async def _check_memory(self) -> bool:
        """Check RSS memory against the configured limit.

        Returns ``True`` when memory is within bounds.
        """
        rss_mb = _get_memory_rss_mb()
        if rss_mb <= 0:
            # Cannot determine memory — assume OK but log once
            log.debug("Memory check skipped — unable to read RSS")
            return True

        if rss_mb > self._memory_limit_mb:
            pct = (rss_mb / self._memory_limit_mb) * 100.0
            log.critical(
                "MEMORY EXCEEDED: %.1fMB / %dMB (%.1f%%) — triggering restart",
                rss_mb,
                self._memory_limit_mb,
                pct,
            )
            self._record_event(
                event_type="MEMORY_EXCEEDED",
                details=(
                    f"RSS {rss_mb:.1f}MB exceeds limit "
                    f"{self._memory_limit_mb}MB ({pct:.1f}%)"
                ),
                severity="CRITICAL",
            )
            await self._trigger_restart(
                reason=f"Memory exceeded: {rss_mb:.1f}MB > {self._memory_limit_mb}MB",
            )
            return False

        log.debug(
            "Memory OK: %.1fMB / %dMB",
            rss_mb,
            self._memory_limit_mb,
        )
        return True

    # -- websocket liveness check --------------------------------------------

    async def _check_websocket(self, rpc_url: str = "") -> bool:
        """Verify that a WebSocket heartbeat was received recently.

        Returns ``True`` when the last heartbeat is within the timeout
        window.  The optional *rpc_url* parameter is accepted for
        future logging/metrics but is not used for connectivity tests.
        """
        elapsed = time.time() - self._last_ws_heartbeat
        if elapsed > self._ws_timeout_s:
            log.critical(
                "WS heartbeat lost — %.1fs since last response (timeout %ds)",
                elapsed,
                self._ws_timeout_s,
            )
            self._record_event(
                event_type="WS_LOST",
                details=(
                    f"No WS heartbeat for {elapsed:.1f}s "
                    f"(timeout {self._ws_timeout_s}s)"
                ),
                severity="CRITICAL",
            )
            await self._trigger_restart(
                reason=f"WebSocket heartbeat lost for {elapsed:.1f}s",
            )
            return False

        log.debug("WS heartbeat OK — %.1fs ago", elapsed)
        return True

    # -- restart logic -------------------------------------------------------

    async def _trigger_restart(self, reason: str) -> None:
        """Attempt an automatic process restart.

        Respects ``RESTART_COOLDOWN_S`` between successive restarts and
        ``MAX_RESTART_ATTEMPTS`` as a hard cap.  On Linux the process is
        replaced in-place via ``os.execv``.  On Windows a warning is
        logged instructing the operator to restart manually.
        """
        now = time.time()

        # Cooldown guard
        if self._last_restart_ts is not None:
            since_last = now - self._last_restart_ts
            if since_last < RESTART_COOLDOWN_S:
                remaining = RESTART_COOLDOWN_S - since_last
                log.warning(
                    "Restart cooldown active — %.0fs remaining (reason: %s)",
                    remaining,
                    reason,
                )
                self._record_event(
                    event_type="PROCESS_RESTART",
                    details=(
                        f"Restart suppressed — cooldown {remaining:.0f}s "
                        f"remaining.  Reason: {reason}"
                    ),
                    severity="WARNING",
                )
                return

        # Hard cap guard
        if self._total_restarts >= MAX_RESTART_ATTEMPTS:
            log.error(
                "Restart cap reached (%d/%d) — manual intervention required",
                self._total_restarts,
                MAX_RESTART_ATTEMPTS,
            )
            self._record_event(
                event_type="PROCESS_RESTART",
                details=(
                    f"Restart cap {MAX_RESTART_ATTEMPTS} reached.  "
                    f"Manual intervention required.  Reason: {reason}"
                ),
                severity="CRITICAL",
            )
            return

        self._total_restarts += 1
        self._last_restart_ts = now
        log.warning(
            "Triggering restart %d/%d — reason: %s",
            self._total_restarts,
            MAX_RESTART_ATTEMPTS,
            reason,
        )
        self._record_event(
            event_type="PROCESS_RESTART",
            details=(
                f"Restart #{self._total_restarts} triggered.  Reason: {reason}"
            ),
            severity="CRITICAL",
        )

        if platform.system() == "Linux":
            log.info("Executing os.execv() to replace current process")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            log.warning(
                "Automatic restart not supported on %s — "
                "manual restart required",
                platform.system(),
            )

    # -- heartbeat recorder --------------------------------------------------

    def record_ws_heartbeat(self) -> None:
        """Record a successful WebSocket / RPC response timestamp."""
        self._last_ws_heartbeat = time.time()

    # -- event ring-buffer ---------------------------------------------------

    def _record_event(
        self,
        event_type: str,
        details: str,
        severity: str,
    ) -> None:
        """Append an event to the bounded ring-buffer."""
        evt = WatchdogEvent(
            event_type=event_type,
            details=details,
            severity=severity,
            timestamp=time.time(),
        )
        self._events.append(evt)

    def get_events(self) -> list[WatchdogEvent]:
        """Return the most recent watchdog events (up to 50)."""
        return list(self._events)

    # -- vitals snapshot -----------------------------------------------------

    def build_vitals(self) -> SystemVitals:
        """Construct a point-in-time ``SystemVitals`` snapshot."""
        rss_mb = _get_memory_rss_mb()
        limit = float(self._memory_limit_mb)
        pct = (rss_mb / limit * 100.0) if limit > 0 else 0.0
        uptime = time.time() - self._start_time
        ws_elapsed = time.time() - self._last_ws_heartbeat
        ws_connected = ws_elapsed <= self._ws_timeout_s
        cpu = _get_cpu_pct()
        active = _active_task_count()

        is_healthy = (
            rss_mb <= self._memory_limit_mb
            and ws_connected
            and self._total_restarts < MAX_RESTART_ATTEMPTS
        )

        return SystemVitals(
            memory_usage_mb=round(rss_mb, 2),
            memory_limit_mb=limit,
            memory_pct=round(pct, 2),
            cpu_pct=round(cpu, 2),
            uptime_seconds=round(uptime, 2),
            last_ws_heartbeat=round(self._last_ws_heartbeat, 3),
            ws_connected=ws_connected,
            total_restarts=self._total_restarts,
            last_restart_timestamp=self._last_restart_ts,
            active_tasks=active,
            is_healthy=is_healthy,
            timestamp=round(time.time(), 3),
        )


# ---------------------------------------------------------------------------
# HealthCheckAPI
# ---------------------------------------------------------------------------

class HealthCheckAPI:
    """Minimal ``aiohttp.web`` server exposing health probes on loopback.

    Endpoints
    ---------
    ``GET /health``    — compact status for OCI health probes
    ``GET /vitals``    — full ``SystemVitals`` snapshot
    ``GET /events``    — recent ``WatchdogEvent`` ring-buffer
    ``GET /portfolio`` — execution mode and engine status
    """

    def __init__(self, watchdog: ProcessWatchdog) -> None:
        self._watchdog: ProcessWatchdog = watchdog
        self._runner: object | None = None  # aiohttp.web.AppRunner
        self._site: object | None = None    # aiohttp.web.TCPSite
        self._started: bool = False

    # -- lifecycle -----------------------------------------------------------

    async def start(
        self,
        host: str = HEALTH_API_HOST,
        port: int = HEALTH_API_PORT,
    ) -> None:
        """Bind and start the health-check HTTP server."""
        if self._started:
            log.warning("HealthCheckAPI already running")
            return

        from aiohttp import web  # local import — avoids top-level cost

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/vitals", self._handle_vitals)
        app.router.add_get("/events", self._handle_events)
        app.router.add_get("/portfolio", self._handle_portfolio)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        self._runner = runner
        self._site = site
        self._started = True
        log.info(
            "HealthCheckAPI started on %s:%d",
            host,
            port,
        )

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if not self._started:
            return
        try:
            if self._runner is not None:
                from aiohttp import web

                runner: web.AppRunner = self._runner  # type: ignore[assignment]
                await runner.cleanup()
        except Exception:
            log.warning("HealthCheckAPI cleanup error", exc_info=True)
        finally:
            self._runner = None
            self._site = None
            self._started = False
            log.info("HealthCheckAPI stopped")

    # -- route handlers ------------------------------------------------------

    async def _handle_health(self, _request: Request) -> Response:
        """``GET /health`` — compact JSON for OCI health probes."""
        from aiohttp import web

        vitals = self._watchdog.build_vitals()
        if vitals.is_healthy:
            status_str = "healthy"
        elif vitals.ws_connected and vitals.memory_pct < 95.0:
            status_str = "degraded"
        else:
            status_str = "unhealthy"

        body = {
            "status": status_str,
            "uptime_s": vitals.uptime_seconds,
            "memory_mb": vitals.memory_usage_mb,
            "memory_pct": vitals.memory_pct,
            "ws_connected": vitals.ws_connected,
            "total_restarts": vitals.total_restarts,
            "timestamp": vitals.timestamp,
        }
        return web.json_response(body)

    async def _handle_vitals(self, _request: Request) -> Response:
        """``GET /vitals`` — full ``SystemVitals`` as JSON dict."""
        from aiohttp import web

        vitals = self._watchdog.build_vitals()
        return web.json_response(asdict(vitals))

    async def _handle_events(self, _request: Request) -> Response:
        """``GET /events`` — recent watchdog events."""
        from aiohttp import web

        events = self._watchdog.get_events()
        payload = [asdict(e) for e in events]
        return web.json_response(payload)

    async def _handle_portfolio(self, _request: Request) -> Response:
        """``GET /portfolio`` — execution mode and engine status."""
        from aiohttp import web

        execution_mode = os.environ.get("EXECUTION_MODE", "robust_paper")
        mode = execution_mode if execution_mode else "robust_paper"

        # Determine engine status from watchdog health
        vitals = self._watchdog.build_vitals()
        engine_status = "running" if vitals.is_healthy else "stopped"

        body = {
            "mode": mode,
            "execution_mode": execution_mode,
            "engine_status": engine_status,
        }
        return web.json_response(body)


# ---------------------------------------------------------------------------
# SystemD unit generator
# ---------------------------------------------------------------------------

def generate_systemd_unit(
    service_name: str = "wallet-manager",
    python_path: str = "/usr/bin/python3",
    working_dir: str = "/opt/wallet_manager",
    user: str = "ubuntu",
) -> str:
    """Return a complete systemd ``.service`` unit file as a string.

    The generated unit includes:
    * ``Restart=always`` with a 10-second back-off
    * ``MemoryMax=1536M`` cgroup limit matching ``MEMORY_LIMIT_MB``
    * Journal logging via ``StandardOutput=journal``
    * ``After=network-online.target`` ordering

    The template is returned as a plain string — nothing is written to disk.
    """
    unit = f"""\
[Unit]
Description={service_name} — QUANT-X Wallet Manager
Documentation=https://github.com/quant-x/wallet-manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={working_dir}
ExecStart={python_path} -m wallet_manager
Restart=always
RestartSec=10
WatchdogSec=120
StartLimitIntervalSec=600
StartLimitBurst=5

# Resource limits
MemoryMax={MEMORY_LIMIT_MB}M
MemoryHigh={int(MEMORY_LIMIT_MB * 0.9)}M
CPUQuota=200%
TasksMax=256

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier={service_name}

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true
ReadWritePaths={working_dir}/data {working_dir}/logs

# Environment
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-{working_dir}/.env

[Install]
WantedBy=multi-user.target
"""
    return unit


# ---------------------------------------------------------------------------
# OCIDeployer — main coordinator
# ---------------------------------------------------------------------------

class OCIDeployer:
    """Top-level coordinator for OCI production deployment tooling.

    Owns the ``ProcessWatchdog`` and ``HealthCheckAPI`` instances and
    exposes a unified ``start()`` / ``stop()`` lifecycle.
    """

    def __init__(self, rpc_url: str = "") -> None:
        self._rpc_url: str = rpc_url
        self._start_time: float = time.time()
        self._last_activity: float = time.time()

        self._watchdog: ProcessWatchdog = ProcessWatchdog(
            memory_limit_mb=MEMORY_LIMIT_MB,
            ws_timeout_s=WEBSOCKET_TIMEOUT_S,
        )
        self._health_api: HealthCheckAPI = HealthCheckAPI(self._watchdog)
        self._started: bool = False

        log.info(
            "OCIDeployer created  rpc_url=%s  platform=%s",
            self._rpc_url or "(none)",
            platform.system(),
        )

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the watchdog loop and health-check API."""
        if self._started:
            log.warning("OCIDeployer already running")
            return

        self._start_time = time.time()
        self._last_activity = time.time()

        await self._watchdog.start()
        try:
            await self._health_api.start(
                host=HEALTH_API_HOST,
                port=HEALTH_API_PORT,
            )
        except Exception:
            log.warning(
                "Health API failed to start — watchdog continues without it",
                exc_info=True,
            )

        self._started = True
        log.info("OCIDeployer started")

    async def stop(self) -> None:
        """Gracefully shut down health API and watchdog."""
        if not self._started:
            return

        log.info("OCIDeployer shutting down")
        await self._health_api.stop()
        await self._watchdog.stop()
        self._started = False
        log.info("OCIDeployer stopped")

    # -- public interface ----------------------------------------------------

    def get_vitals(self) -> SystemVitals:
        """Return a point-in-time ``SystemVitals`` snapshot."""
        return self._watchdog.build_vitals()

    def record_activity(self) -> None:
        """Record orchestrator liveness (also refreshes WS heartbeat)."""
        self._last_activity = time.time()
        self._watchdog.record_ws_heartbeat()

    def get_deployment_info(self) -> dict:
        """Return OCI deployment metadata.

        Includes region, instance shape, uptime, platform details, and
        environment configuration.
        """
        vitals = self._watchdog.build_vitals()
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "oci_region": os.environ.get("OCI_REGION", "unknown"),
            "oci_availability_domain": os.environ.get(
                "OCI_AVAILABILITY_DOMAIN", "unknown"
            ),
            "oci_instance_shape": os.environ.get(
                "OCI_INSTANCE_SHAPE", "VM.Standard.A1.Flex"
            ),
            "rpc_url": self._rpc_url or "(not set)",
            "uptime_seconds": vitals.uptime_seconds,
            "memory_usage_mb": vitals.memory_usage_mb,
            "memory_limit_mb": vitals.memory_limit_mb,
            "total_restarts": vitals.total_restarts,
            "is_healthy": vitals.is_healthy,
            "execution_mode": os.environ.get("EXECUTION_MODE", "robust_paper"),
            "health_api": f"http://{HEALTH_API_HOST}:{HEALTH_API_PORT}/health",
            "pid": os.getpid(),
            "started_at": self._start_time,
        }

    # -- properties ----------------------------------------------------------

    @property
    def watchdog(self) -> ProcessWatchdog:
        """Direct access to the watchdog (for advanced callers)."""
        return self._watchdog

    @property
    def health_api(self) -> HealthCheckAPI:
        """Direct access to the health API (for advanced callers)."""
        return self._health_api

    @property
    def is_running(self) -> bool:
        """Whether the deployer has been started."""
        return self._started


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_oci_deployer: OCIDeployer | None = None


def get_oci_deployer(rpc_url: str = "") -> OCIDeployer:
    """Return the process-wide OCIDeployer singleton."""
    global _oci_deployer
    if _oci_deployer is None:
        _oci_deployer = OCIDeployer(rpc_url=rpc_url)
    return _oci_deployer


def reset_oci_deployer() -> None:
    """Reset the singleton (for tests)."""
    global _oci_deployer
    _oci_deployer = None
