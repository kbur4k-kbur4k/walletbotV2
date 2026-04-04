"""
wallet_manager/task_factories_yield.py

Re-exports for YIELD-pillar task factory functions.
Actual implementations live in task_factories.py (file is too entangled to
split cleanly — each factory orchestrates multiple pillars together).
"""
from wallet_manager.task_factories import (  # noqa: F401
    _robust_paper_task_factory,
    _robust_task_factory,
)

__all__ = [
    "_robust_paper_task_factory",
    "_robust_task_factory",
]
