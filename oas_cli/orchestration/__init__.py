"""Multi-agent orchestration for Open Agent Spec.

This package provides a runtime-agnostic orchestration layer that lets
OA Spec agents participate in multi-agent workflows.  A manager agent
decomposes objectives into tasks on a board; worker agents pull tasks
that match their role and deliver structured results.

The default implementation runs everything in-process with no external
dependencies.  The interfaces are deliberately thin so that the board,
runner, and registry can be backed by Redis, SQLite, Temporal, or
anything else.
"""

from .board import TaskBoard, Task, TaskStatus, TaskPriority
from .registry import AgentEntry, AgentRegistry
from .runner import AgentRunner
from .loop import OrchestrationLoop

__all__ = [
    "TaskBoard",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "AgentEntry",
    "AgentRegistry",
    "AgentRunner",
    "OrchestrationLoop",
]
