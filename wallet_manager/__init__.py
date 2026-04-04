"""
wallet_manager
==============

Secure, async-based personal digital asset management.

Public surface
--------------
Import everything you need from this top-level package::

    from wallet_manager import WalletManager, WalletConfig, TxParams, TxReceipt

All other modules are considered private implementation details and may
change between versions without notice.
"""

from wallet_manager.core.manager import WalletManager
from wallet_manager.core.models import TxParams, TxReceipt, WalletConfig

__all__ = [
    "TxParams",
    "TxReceipt",
    "WalletConfig",
    "WalletManager",
]
