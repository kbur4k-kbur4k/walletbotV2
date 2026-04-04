"""
wallet_manager/task_factories_short.py

Re-exports for SHORT-pillar task factory functions.
Actual implementations live in task_factories.py (file is too entangled to
split cleanly — each factory orchestrates multiple pillars together).
"""
from wallet_manager.task_factories import (  # noqa: F401
    _three_pillar_task_factory,
)

__all__ = [
    "_three_pillar_task_factory",
]
