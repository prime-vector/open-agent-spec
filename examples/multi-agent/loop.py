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
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional


def _easter_sunday(year: int) -> date:
    """Compute Easter Sunday for a given year (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _date_context() -> str:
    """Build a date context string with today and upcoming Easter dates."""
    today = date.today()
    easter = _easter_sunday(today.year)
    good_friday = easter - timedelta(days=2)
    easter_saturday = easter - timedelta(days=1)
    easter_monday = easter + timedelta(days=1)
    return (
        f"Today is {today.strftime('%A %d %B %Y')}. "
        f"Easter {today.year}: "
        f"Good Friday {good_friday.strftime('%d %B')}, "
        f"Saturday {easter_saturday.strftime('%d %B')}, "
        f"Easter Sunday {easter.strftime('%d %B')}, "
        f"Easter Monday {easter_monday.strftime('%d %B')}."
    )

from board import TaskBoard, TaskPriority, TaskStatus
from registry import AgentRegistry
from runner import AgentRunner

logger = logging.getLogger(__name__)

# Type for event callbacks the dashboard can subscribe to.
EventCallback = Callable[[str, Dict[str, Any]], None]


class OrchestrationLoop:
    """Drives a multi-agent workflow from objective to completion."""

    def __init__(
        self,
        manager_spec: str,
        worker_specs: List[str],
        concierge_spec: Optional[str] = None,
        max_iterations: int = 50,
    ) -> None:
        self.board = TaskBoard()
        self.registry = AgentRegistry()
        self.runners: Dict[str, AgentRunner] = {}
        self._event_listeners: List[EventCallback] = []
        self.max_iterations = max_iterations
        self.events: List[Dict[str, Any]] = []

        # Load the concierge (optional — user-facing agent).
        self._concierge_runner: Optional[AgentRunner] = None
        if concierge_spec:
            self._concierge_runner = AgentRunner(concierge_spec)
            self.registry.register(
                agent_id=self._concierge_runner.agent_name,
                role=self._concierge_runner.agent_role,
                spec_path=concierge_spec,
            )
            self.runners[self._concierge_runner.agent_name] = self._concierge_runner

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

    # -- Phase 0: Concierge clarifies the request --

    def _clarify(self, user_request: str) -> str:
        """Use the concierge to refine a raw user request into a clear objective."""
        if not self._concierge_runner:
            return user_request

        self._emit("clarify_start", {"user_request": user_request})

        result = self._concierge_runner.run_task(
            task_name="clarify",
            input_data={
                "user_request": user_request,
                "conversation_history": "",
                "current_date": _date_context(),
            },
        )

        if "error" in result:
            self._emit("clarify_error", {"error": result["error"]})
            return user_request  # Fall back to raw request.

        output = result.get("output", {})
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return user_request

        objective = output.get("objective") or user_request
        # Track concierge activity in agent stats.
        concierge_entry = self.registry.get("concierge-agent")
        if concierge_entry:
            concierge_entry.tasks_completed += 1
        self._emit("clarify_complete", {
            "objective": objective,
            "clarification_needed": output.get("clarification_needed", False),
            "scope_notes": output.get("scope_notes", ""),
        })
        return objective

    def _summarise(self, objective: str, tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Use the concierge to produce a user-friendly summary of results."""
        if not self._concierge_runner:
            return None

        self._emit("summarise_start", {"objective": objective})

        result = self._concierge_runner.run_task(
            task_name="summarise",
            input_data={
                "objective": objective,
                "task_results": json.dumps(
                    [{"title": t.get("title"), "result": t.get("result")} for t in tasks],
                    default=str,
                )[:3000],  # Truncate to stay within token limits.
            },
        )

        if "error" in result:
            self._emit("summarise_error", {"error": result["error"]})
            return None

        output = result.get("output", {})
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return None

        # Track concierge summarise activity.
        concierge_entry = self.registry.get("concierge-agent")
        if concierge_entry:
            concierge_entry.tasks_completed += 1
        self._emit("summarise_complete", {"summary": str(output.get("summary", ""))[:200]})
        return output

    # -- Phase 1: Ask the manager to plan --

    def _plan(self, objective: str) -> List[Dict[str, Any]]:
        """Ask the manager agent to decompose the objective into tasks."""
        available_roles = ", ".join(
            sorted(
                {
                    e.role
                    for e in self.registry.all_agents()
                    if e.role not in ("planner", "chat")
                }
            )
        )
        self._emit("plan_start", {"objective": objective, "available_roles": available_roles})

        result = self._manager_runner.run_task(
            task_name="decompose",
            input_data={
                "objective": objective,
                "available_roles": available_roles,
                "current_date": _date_context(),
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

        # Guard: clamp any role the manager hallucinated to the nearest valid one.
        valid_roles = set(available_roles.split(", "))
        for t in tasks:
            if t.get("required_role") not in valid_roles and valid_roles:
                original = t.get("required_role")
                t["required_role"] = sorted(valid_roles)[0]
                logger.warning(
                    "Manager assigned unknown role %r; clamping to %r",
                    original,
                    t["required_role"],
                )

        # Track manager activity in agent stats.
        manager_entry = self.registry.get("manager-agent")
        if manager_entry:
            manager_entry.tasks_completed += 1
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
            if entry.role in ("planner", "chat"):
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
            input_data.setdefault("current_date", date.today().strftime("%A %d %B %Y"))
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
        # Only prepend date context when the objective is time-sensitive.
        time_words = {"today", "tomorrow", "yesterday", "weekend", "week",
                      "month", "easter", "christmas", "holiday", "season",
                      "spring", "summer", "autumn", "winter", "date",
                      "april", "march", "january", "february", "may",
                      "june", "july", "august", "september", "october",
                      "november", "december", "schedule", "upcoming",
                      "current", "recent", "latest", "now", "2024", "2025", "2026"}
        obj_lower = objective.lower()
        if any(w in obj_lower for w in time_words):
            date_ctx = _date_context()
            objective_with_date = f"[{date_ctx}] {objective}"
        else:
            objective_with_date = objective

        self._emit("orchestration_start", {"objective": objective})

        # Phase 0 — Concierge clarifies the raw request into a scoped objective.
        refined_objective = self._clarify(objective_with_date)

        # Phase 1 — Manager plans.
        planned = self._plan(refined_objective)
        self._populate_board(planned)

        # Phase 2 — Workers execute until the board is done.
        iterations = 0
        while not self.board.is_complete() and iterations < self.max_iterations:
            did_work = self._execute_one()
            if not did_work:
                self._emit("orchestration_stalled", {
                    "reason": "No available tasks but board is not complete",
                    "board_summary": self.board.summary(),
                })
                break
            iterations += 1

        # Phase 3 — Concierge summarises the results (using the original
        # objective without the date prefix to avoid leaking context noise).
        task_dicts = [t.to_dict() for t in self.board.all_tasks()]
        summary = self._summarise(objective, task_dicts)

        self._emit("orchestration_complete", {
            "iterations": iterations,
            "board_summary": self.board.summary(),
        })

        result: Dict[str, Any] = {
            "objective": objective,
            "refined_objective": refined_objective,
            "board": self.board.summary(),
            "tasks": task_dicts,
            "agents": [e.to_dict() for e in self.registry.all_agents()],
            "events": self.events,
            "iterations": iterations,
        }
        if summary:
            result["summary"] = summary
        return result

    def status(self) -> Dict[str, Any]:
        """Snapshot of current state (for the dashboard)."""
        return {
            "board": self.board.summary(),
            "tasks": [t.to_dict() for t in self.board.all_tasks()],
            "agents": [e.to_dict() for e in self.registry.all_agents()],
            "events": self.events[-50:],
        }
