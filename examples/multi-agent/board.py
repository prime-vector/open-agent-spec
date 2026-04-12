"""Task board -- a runtime-agnostic data structure for agent work items.

The board is the central coordination point.  A manager posts tasks;
workers pull tasks that match their role.  The default implementation
is an in-memory dict.  Swap in Redis, SQLite, or Postgres by
subclassing ``TaskBoard`` and overriding the storage methods.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """A unit of work on the board."""

    id: str
    title: str
    description: str
    required_role: str
    input_data: dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    depends_on: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    assigned_at: float | None = None
    completed_at: float | None = None
    source: str = "manager"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "required_role": self.required_role,
            "input_data": self.input_data,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "depends_on": self.depends_on,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "assigned_at": self.assigned_at,
            "completed_at": self.completed_at,
            "source": self.source,
        }


class TaskBoard:
    """In-memory task board.  Override for persistent backends."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    # -- mutations --

    def post_task(
        self,
        title: str,
        description: str,
        required_role: str,
        input_data: dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        depends_on: list[str] | None = None,
        source: str = "manager",
    ) -> Task:
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            title=title,
            description=description,
            required_role=required_role,
            input_data=input_data,
            priority=priority,
            depends_on=depends_on or [],
            source=source,
        )
        self._tasks[task_id] = task
        return task

    def claim_task(self, task_id: str, agent_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        if not self._dependencies_met(task):
            return False
        task.status = TaskStatus.ASSIGNED
        task.assigned_to = agent_id
        task.assigned_at = time.time()
        return True

    def start_task(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.ASSIGNED:
            task.status = TaskStatus.IN_PROGRESS

    def complete_task(self, task_id: str, result: dict[str, Any]) -> None:
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()

    def fail_task(self, task_id: str, error: str) -> None:
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = time.time()

    # -- queries --

    def available_tasks(self, role: str | None = None) -> list[Task]:
        """Return pending tasks whose dependencies are met, optionally filtered by role."""
        out: list[Task] = []
        for t in self._tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            if not self._dependencies_met(t):
                continue
            if role and t.required_role != role:
                continue
            out.append(t)
        out.sort(key=lambda t: (-t.priority.value, t.created_at))
        return out

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for t in self._tasks.values():
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return {
            "total": len(self._tasks),
            "by_status": counts,
        }

    def is_complete(self) -> bool:
        """True when every task is completed or failed (and at least one exists)."""
        if not self._tasks:
            return False
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self._tasks.values()
        )

    # -- internals --

    def _dependencies_met(self, task: Task) -> bool:
        for dep_id in task.depends_on:
            dep = self._tasks.get(dep_id)
            if not dep or dep.status != TaskStatus.COMPLETED:
                return False
        return True
