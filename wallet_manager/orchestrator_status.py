"""
wallet_manager/orchestrator_status.py

LiveStatusProvider — real-time Telegram /status dashboard.

Extracted from orchestrator.py to keep the main orchestrator focused on
lifecycle management.  All business logic for the /status command lives here.
"""
from __future__ import annotations

from wallet_manager import actions
from wallet_manager.core.models import WalletConfig
from wallet_manager.portfolio_manager import PortfolioManager
from wallet_manager.scheduler import SchedulerManager
from wallet_manager.security.logging_filter import build_logger
from wallet_manager.telegram_notifier import StatusProvider

log = build_logger(__name__)


def _get_wallet_address_local(config: WalletConfig) -> str:
    """Resolve the wallet address from config (local import avoids circular dep)."""
    from wallet_manager.task_factories import _get_wallet_address
    return _get_wallet_address(config)


class LiveStatusProvider(StatusProvider):
    """Real-time status for /status command: full on-chain portfolio snapshot.

    Fetches live ETH and USDC balances, Aave deposited USDC, and short
    position value, then formats them into a Telegram Markdown message.

    Args:
        configs: Ordered list of wallet configs (first entry used for balance fetch).
        scheduler: SchedulerManager to compute time-until-next-run.
        portfolio_manager: Optional PortfolioManager for total equity snapshot.
    """

    def __init__(
        self,
        configs: list[WalletConfig],
        scheduler: SchedulerManager,
        portfolio_manager: PortfolioManager | None = None,
    ) -> None:
        self._configs = configs
        self._scheduler = scheduler
        self._portfolio_manager = portfolio_manager

    async def build_status_message(self) -> str:
        """Build Markdown status from real on-chain balances via portfolio_manager."""
        if not self._configs:
            return "ℹ️ No wallets configured."

        config = self._configs[0]
        address = _get_wallet_address_local(config)
        chain_id = int(config.chain_id) if config.chain_id is not None else 0

        # --- Fetch all on-chain balances ---
        try:
            balance_wei = await actions.fetch_native_balance(config.rpc_url, address)
        except Exception:
            balance_wei = 0
        try:
            balance_usdc = await actions.fetch_usdc_balance(
                config.rpc_url, address, chain_id
            )
        except Exception:
            balance_usdc = 0
        try:
            eth_price = await actions.fetch_eth_price_usd()
        except Exception:
            eth_price = 3000.0

        # --- Fetch Aave deposited USDC ---
        try:
            from wallet_manager.strategies import yield_strategy
            aave_raw = await yield_strategy.get_aave_usdc_balance(config.rpc_url, address)
            aave_usdc = aave_raw / 1e6
        except Exception:
            aave_usdc = 0.0

        # --- Fetch short position value ---
        try:
            from wallet_manager.strategies import short_strategy
            short_state = short_strategy.get_short_position_state()
            short_usd = float(getattr(short_state, "position_value_usd", 0.0))
        except Exception:
            short_usd = 0.0

        # --- Sync portfolio manager and get snapshot ---
        if self._portfolio_manager is not None:
            await self._portfolio_manager.sync_from_chain(
                wallet_address=address,
                rpc_url=config.rpc_url,
                chain_id=chain_id,
                eth_price_usd=eth_price,
            )
            snapshot = self._portfolio_manager.get_snapshot()
            total_usd = snapshot.total_value_usd
        else:
            eth_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
            total_usd = eth_usd + (balance_usdc / 1e6) + aave_usdc + short_usd

        eth_val = balance_wei / 1e18
        eth_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = balance_usdc / 1e6

        secs = self._scheduler.seconds_until_due(config.wallet_id)
        if secs >= 3600:
            next_str = f"{secs / 3600:.1f} hours later"
        elif secs >= 60:
            next_str = f"{secs / 60:.0f} mins later"
        elif secs > 0:
            next_str = f"{secs:.0f} secs later"
        else:
            next_str = "due now"

        return (
            "📊 *Portfolio Live Status*\n"
            f"💳 Wallet: {address}\n"
            f"🔹 ETH: {eth_val:.4f} ETH (~${eth_usd:.2f})\n"
            f"💵 Spot USDC: {usdc_val:.2f} USDC\n"
            f"🏦 Aave USDC: {aave_usdc:.2f} USDC\n"
            f"📉 Short Pos: ${short_usd:.2f}\n"
            f"💼 *Total Portfolio: ${total_usd:.2f}*\n"
            f"⏱️ Next Swap: {next_str}"
        )
