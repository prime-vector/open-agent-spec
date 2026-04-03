"""Orchestration loop -- the glue between manager, board, and workers.

Usage:

    loop = OrchestrationLoop(
        manager_spec=".agents/personas/manager.yaml",
        worker_specs=[
            ".agents/personas/researcher.yaml",
            ".agents/personas/writer.yaml",
            ".agents/personas/reviewer.yaml",
        ],
    )
    result = loop.run("Write a blog post about AI agent frameworks")

The loop:
  1. Asks the manager to decompose the objective into tasks.
  2. Posts each task to the board.
  3. Iterates: finds an available task, matches it to a worker by role,
     has the worker execute it, records the result.
  4. Repeats until the board is complete.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .board import TaskBoard, TaskPriority, TaskStatus
from .registry import AgentRegistry
from .runner import AgentRunner

logger = logging.getLogger(__name__)

# Type for event callbacks the dashboard can subscribe to.
EventCallback = Callable[[str, Dict[str, Any]], None]


class OrchestrationLoop:
    """Drives a multi-agent workflow from objective to completion."""

    def __init__(
        self,
        manager_spec: str,
        worker_specs: List[str],
        max_iterations: int = 50,
    ) -> None:
        self.board = TaskBoard()
        self.registry = AgentRegistry()
        self.runners: Dict[str, AgentRunner] = {}
        self._event_listeners: List[EventCallback] = []
        self.max_iterations = max_iterations
        self.events: List[Dict[str, Any]] = []

        # Load the manager.
        self._manager_runner = AgentRunner(manager_spec)
        self.registry.register(
            agent_id=self._manager_runner.agent_name,
            role=self._manager_runner.agent_role,
            spec_path=manager_spec,
        )
        self.runners[self._manager_runner.agent_name] = self._manager_runner

        # Load workers.
        for spec in worker_specs:
            runner = AgentRunner(spec)
            agent_id = runner.agent_name
            self.registry.register(
                agent_id=agent_id,
                role=runner.agent_role,
                spec_path=spec,
            )
            self.runners[agent_id] = runner

    def on_event(self, callback: EventCallback) -> None:
        """Register an event listener (used by the dashboard)."""
        self._event_listeners.append(callback)

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        entry = {"type": event_type, "timestamp": time.time(), **data}
        self.events.append(entry)
        for cb in self._event_listeners:
            try:
                cb(event_type, entry)
            except Exception:
                pass

    # -- Phase 1: Ask the manager to plan --

    def _plan(self, objective: str) -> List[Dict[str, Any]]:
        """Ask the manager agent to decompose the objective into tasks."""
        available_roles = ", ".join(
            sorted(
                {
                    e.role
                    for e in self.registry.all_agents()
                    if e.role != "planner"
                }
            )
        )
        self._emit("plan_start", {"objective": objective, "available_roles": available_roles})

        result = self._manager_runner.run_task(
            task_name="decompose",
            input_data={
                "objective": objective,
                "available_roles": available_roles,
            },
        )

        if "error" in result:
            self._emit("plan_error", {"error": result["error"]})
            raise RuntimeError(f"Manager failed to plan: {result['error']}")

        output = result.get("output", {})
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                raise RuntimeError(f"Manager returned non-JSON output: {output[:200]}")

        tasks = output.get("tasks", [])
        summary = output.get("summary", "")
        self._emit("plan_complete", {"summary": summary, "task_count": len(tasks)})
        return tasks

    # -- Phase 2: Populate the board --

    def _populate_board(self, planned_tasks: List[Dict[str, Any]]) -> None:
        """Post planned tasks onto the board, resolving dependency titles to IDs."""
        title_to_id: Dict[str, str] = {}

        for t in planned_tasks:
            priority_val = t.get("priority", 2)
            try:
                priority = TaskPriority(priority_val)
            except ValueError:
                priority = TaskPriority.NORMAL

            # Resolve depends_on_titles to IDs.
            dep_titles = t.get("depends_on_titles") or []
            dep_ids = [title_to_id[dt] for dt in dep_titles if dt in title_to_id]

            task = self.board.post_task(
                title=t["title"],
                description=t.get("description", ""),
                required_role=t.get("required_role", "analyst"),
                input_data=t.get("input_data") or {},
                priority=priority,
                depends_on=dep_ids,
                source="manager",
            )
            title_to_id[t["title"]] = task.id

            self._emit("task_posted", {"task_id": task.id, "title": task.title, "role": task.required_role})

    # -- Phase 3: Execute --

    def _find_worker(self, role: str) -> Optional[AgentRunner]:
        """Find a runner whose agent matches the required role."""
        entries = self.registry.find_by_role(role)
        if entries:
            return self.runners.get(entries[0].id)
        return None

    def _pick_task_for_worker(self, runner: AgentRunner) -> Optional[Any]:
        """Find the highest-priority available task matching this worker's role."""
        tasks = self.board.available_tasks(role=runner.agent_role)
        return tasks[0] if tasks else None

    def _execute_one(self) -> bool:
        """Try to execute one task.  Returns True if work was done."""
        # Scan all roles for available work.
        for entry in self.registry.all_agents():
            if entry.role == "planner":
                continue
            runner = self.runners.get(entry.id)
            if not runner:
                continue

            task = self._pick_task_for_worker(runner)
            if not task:
                continue

            # Claim and execute.
            if not self.board.claim_task(task.id, entry.id):
                continue

            self.board.start_task(task.id)
            self._emit("task_started", {
                "task_id": task.id,
                "title": task.title,
                "agent": entry.id,
            })

            # Build input: task's own input_data + description as context.
            input_data = dict(task.input_data)
            input_data.setdefault("topic", task.description)
            input_data.setdefault("brief", task.description)
            input_data.setdefault("content", task.description)
            input_data.setdefault("context", "")
            input_data.setdefault("criteria", "quality and completeness")
            input_data.setdefault("source_material", "")
            input_data.setdefault("format", "structured analysis")

            # Pick the first task the runner supports.
            task_name = runner.task_names[0] if runner.task_names else None
            if not task_name:
                self.board.fail_task(task.id, "Worker has no tasks defined")
                self._emit("task_failed", {"task_id": task.id, "error": "no tasks"})
                return True

            result = runner.run_task(task_name=task_name, input_data=input_data)

            if "error" in result:
                self.board.fail_task(task.id, result["error"])
                reg_entry = self.registry.get(entry.id)
                if reg_entry:
                    reg_entry.tasks_failed += 1
                self._emit("task_failed", {
                    "task_id": task.id,
                    "title": task.title,
                    "agent": entry.id,
                    "error": result["error"],
                })
            else:
                output = result.get("output", {})
                self.board.complete_task(task.id, output if isinstance(output, dict) else {"result": output})
                reg_entry = self.registry.get(entry.id)
                if reg_entry:
                    reg_entry.tasks_completed += 1
                self._emit("task_completed", {
                    "task_id": task.id,
                    "title": task.title,
                    "agent": entry.id,
                    "output_preview": str(output)[:200],
                })
            return True

        return False

    # -- Public API --

    def run(self, objective: str) -> Dict[str, Any]:
        """Run the full orchestration loop for an objective.

        Returns a summary with the board state, agent stats, and event log.
        """
        self._emit("orchestration_start", {"objective": objective})

        # Plan.
        planned = self._plan(objective)
        self._populate_board(planned)

        # Execute until the board is done or we hit the iteration limit.
        iterations = 0
        while not self.board.is_complete() and iterations < self.max_iterations:
            did_work = self._execute_one()
            if not did_work:
                # No work available but board isn't complete — might have
                # unsatisfiable dependencies.  Break to avoid infinite loop.
                self._emit("orchestration_stalled", {
                    "reason": "No available tasks but board is not complete",
                    "board_summary": self.board.summary(),
                })
                break
            iterations += 1

        self._emit("orchestration_complete", {
            "iterations": iterations,
            "board_summary": self.board.summary(),
        })

        return {
            "objective": objective,
            "board": self.board.summary(),
            "tasks": [t.to_dict() for t in self.board.all_tasks()],
            "agents": [e.to_dict() for e in self.registry.all_agents()],
            "events": self.events,
            "iterations": iterations,
        }

    def status(self) -> Dict[str, Any]:
        """Snapshot of current state (for the dashboard)."""
        return {
            "board": self.board.summary(),
            "tasks": [t.to_dict() for t in self.board.all_tasks()],
            "agents": [e.to_dict() for e in self.registry.all_agents()],
            "events": self.events[-50:],
        }
