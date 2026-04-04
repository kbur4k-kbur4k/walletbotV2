"""
wallet_manager/task_factories_algo.py

Re-exports for ALGO-pillar task factory functions.
Actual implementations live in task_factories.py (file is too entangled to
split cleanly — each factory orchestrates multiple pillars together).
"""
from wallet_manager.task_factories import (  # noqa: F401
    _default_task_factory,
    _ml_quant_task_factory,
    _quantx_task_factory,
)

__all__ = [
    "_default_task_factory",
    "_ml_quant_task_factory",
    "_quantx_task_factory",
]
