"""Tests for the orchestration module."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from oas_cli.orchestration.board import TaskBoard, TaskPriority, TaskStatus
from oas_cli.orchestration.registry import AgentRegistry
from oas_cli.orchestration.runner import AgentRunner
from oas_cli.orchestration.loop import OrchestrationLoop


# ---------------------------------------------------------------------------
# TaskBoard
# ---------------------------------------------------------------------------


class TestTaskBoard:
    def test_post_and_retrieve(self) -> None:
        board = TaskBoard()
        task = board.post_task("Do thing", "Details", "analyst", {"key": "val"})
        assert task.id
        assert task.status == TaskStatus.PENDING
        assert board.get_task(task.id) is task

    def test_available_filters_by_role(self) -> None:
        board = TaskBoard()
        board.post_task("A", "a", "analyst", {})
        board.post_task("B", "b", "writer", {})
        assert len(board.available_tasks("analyst")) == 1
        assert len(board.available_tasks("writer")) == 1
        assert len(board.available_tasks()) == 2

    def test_priority_ordering(self) -> None:
        board = TaskBoard()
        board.post_task("Low", "l", "analyst", {}, priority=TaskPriority.LOW)
        board.post_task("Urgent", "u", "analyst", {}, priority=TaskPriority.URGENT)
        board.post_task("Normal", "n", "analyst", {}, priority=TaskPriority.NORMAL)
        tasks = board.available_tasks("analyst")
        assert tasks[0].title == "Urgent"
        assert tasks[-1].title == "Low"

    def test_dependency_gating(self) -> None:
        board = TaskBoard()
        t1 = board.post_task("First", "f", "analyst", {})
        t2 = board.post_task("Second", "s", "writer", {}, depends_on=[t1.id])
        # t2 not available until t1 completes.
        assert board.available_tasks("writer") == []
        board.claim_task(t1.id, "agent-a")
        board.complete_task(t1.id, {"done": True})
        assert len(board.available_tasks("writer")) == 1

    def test_claim_and_complete_lifecycle(self) -> None:
        board = TaskBoard()
        task = board.post_task("Work", "w", "analyst", {})
        assert board.claim_task(task.id, "agent-a")
        assert task.status == TaskStatus.ASSIGNED
        # Can't claim again.
        assert not board.claim_task(task.id, "agent-b")
        board.start_task(task.id)
        assert task.status == TaskStatus.IN_PROGRESS
        board.complete_task(task.id, {"result": 42})
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"result": 42}

    def test_fail_task(self) -> None:
        board = TaskBoard()
        task = board.post_task("Fail", "f", "analyst", {})
        board.claim_task(task.id, "agent-a")
        board.fail_task(task.id, "oops")
        assert task.status == TaskStatus.FAILED
        assert task.error == "oops"

    def test_is_complete(self) -> None:
        board = TaskBoard()
        assert not board.is_complete()  # empty
        t1 = board.post_task("A", "a", "analyst", {})
        t2 = board.post_task("B", "b", "writer", {})
        assert not board.is_complete()
        board.claim_task(t1.id, "a1")
        board.complete_task(t1.id, {})
        assert not board.is_complete()
        board.claim_task(t2.id, "a2")
        board.fail_task(t2.id, "err")
        assert board.is_complete()  # all done (completed or failed)


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_register_and_find(self) -> None:
        reg = AgentRegistry()
        reg.register("agent-1", "analyst", "spec.yaml")
        reg.register("agent-2", "writer", "spec2.yaml")
        assert len(reg.find_by_role("analyst")) == 1
        assert reg.get("agent-1") is not None
        assert reg.get("nonexistent") is None

    def test_unregister(self) -> None:
        reg = AgentRegistry()
        reg.register("a", "analyst", "s.yaml")
        assert reg.unregister("a")
        assert not reg.unregister("a")
        assert reg.all_agents() == []


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------


class TestAgentRunner:
    def test_loads_spec(self) -> None:
        runner = AgentRunner(".agents/personas/manager.yaml")
        assert runner.agent_name == "manager-agent"
        assert runner.agent_role == "planner"
        assert "decompose" in runner.task_names

    def test_loads_worker_spec(self) -> None:
        runner = AgentRunner(".agents/personas/researcher.yaml")
        assert runner.agent_role == "analyst"
        assert "analyse" in runner.task_names

    def test_loads_concierge_spec(self) -> None:
        runner = AgentRunner(".agents/personas/concierge.yaml")
        assert runner.agent_name == "concierge-agent"
        assert runner.agent_role == "chat"
        assert "clarify" in runner.task_names
        assert "summarise" in runner.task_names
        assert "greet" in runner.task_names


# ---------------------------------------------------------------------------
# OrchestrationLoop (unit-level, mocking LLM calls)
# ---------------------------------------------------------------------------


class TestOrchestrationLoop:
    def _make_loop(self) -> OrchestrationLoop:
        return OrchestrationLoop(
            manager_spec=".agents/personas/manager.yaml",
            worker_specs=[
                ".agents/personas/researcher.yaml",
                ".agents/personas/writer.yaml",
                ".agents/personas/reviewer.yaml",
            ],
        )

    def test_initialises_registry(self) -> None:
        loop = self._make_loop()
        summary = loop.registry.summary()
        assert summary["total"] == 4
        assert "planner" in summary["by_role"]
        assert "analyst" in summary["by_role"]
        assert "writer" in summary["by_role"]
        assert "reviewer" in summary["by_role"]

    def test_populate_board_from_planned_tasks(self) -> None:
        loop = self._make_loop()
        planned = [
            {"title": "Research AI", "description": "Find info", "required_role": "analyst", "priority": 3},
            {"title": "Write post", "description": "Draft article", "required_role": "writer", "priority": 2,
             "depends_on_titles": ["Research AI"]},
        ]
        loop._populate_board(planned)
        assert loop.board.summary()["total"] == 2
        # Writer task blocked by analyst task.
        assert len(loop.board.available_tasks("writer")) == 0
        assert len(loop.board.available_tasks("analyst")) == 1

    @patch("oas_cli.orchestration.runner.AgentRunner.run_task")
    def test_full_run_mocked(self, mock_run_task) -> None:
        """Test the full loop with mocked LLM calls."""
        loop = self._make_loop()

        # Mock responses: manager decomposes, then workers execute.
        manager_response = {
            "output": {
                "tasks": [
                    {"title": "Analyse topic", "description": "Research AI agents",
                     "required_role": "analyst", "priority": 3},
                    {"title": "Write article", "description": "Draft blog post",
                     "required_role": "writer", "priority": 2,
                     "depends_on_titles": ["Analyse topic"]},
                ],
                "summary": "Two-step plan: research then write.",
            }
        }
        analyst_response = {
            "output": {"findings": "AI agents are useful", "key_points": ["point 1"]}
        }
        writer_response = {
            "output": {"content": "Here is the blog post...", "word_count": 500}
        }

        mock_run_task.side_effect = [
            manager_response,
            analyst_response,
            writer_response,
        ]

        result = loop.run("Write a blog post about AI agents")

        assert result["board"]["total"] == 2
        assert result["board"]["by_status"]["completed"] == 2
        assert result["iterations"] == 2
        assert len(result["events"]) > 0

    def _make_loop_with_concierge(self) -> OrchestrationLoop:
        return OrchestrationLoop(
            manager_spec=".agents/personas/manager.yaml",
            worker_specs=[
                ".agents/personas/researcher.yaml",
                ".agents/personas/writer.yaml",
                ".agents/personas/reviewer.yaml",
            ],
            concierge_spec=".agents/personas/concierge.yaml",
        )

    def test_concierge_registers(self) -> None:
        loop = self._make_loop_with_concierge()
        summary = loop.registry.summary()
        assert summary["total"] == 5
        assert "chat" in summary["by_role"]
        assert loop._concierge_runner is not None

    @patch("oas_cli.orchestration.runner.AgentRunner.run_task")
    def test_full_run_with_concierge(self, mock_run_task) -> None:
        """Test concierge clarify → plan → execute → summarise flow."""
        loop = self._make_loop_with_concierge()

        clarify_response = {
            "output": {
                "objective": "Write a 500-word blog post about AI agent frameworks",
                "clarification_needed": False,
                "scope_notes": "Focusing on open-source frameworks",
            }
        }
        manager_response = {
            "output": {
                "tasks": [
                    {"title": "Research frameworks", "description": "Find top AI agent frameworks",
                     "required_role": "analyst", "priority": 3},
                ],
                "summary": "Single research task.",
            }
        }
        analyst_response = {
            "output": {"findings": "LangChain, CrewAI, OA Spec", "key_points": ["three frameworks"]}
        }
        summarise_response = {
            "output": {
                "summary": "The team researched AI agent frameworks and identified three key players.",
                "highlights": ["LangChain", "CrewAI", "OA Spec"],
                "next_steps": ["Draft the blog post"],
            }
        }

        mock_run_task.side_effect = [
            clarify_response,
            manager_response,
            analyst_response,
            summarise_response,
        ]

        result = loop.run("write something about AI agents")

        assert result["objective"] == "write something about AI agents"
        assert result["refined_objective"] == "Write a 500-word blog post about AI agent frameworks"
        assert result["board"]["by_status"]["completed"] == 1
        assert "summary" in result
        assert "highlights" in result["summary"]

    def test_no_concierge_passes_through(self) -> None:
        """Without a concierge, the raw objective passes straight to the manager."""
        loop = self._make_loop()
        assert loop._concierge_runner is None
        # _clarify should be a no-op.
        assert loop._clarify("raw request") == "raw request"

    def test_events_emitted(self) -> None:
        loop = self._make_loop()
        events_received: list = []
        loop.on_event(lambda t, d: events_received.append(t))
        loop._emit("test_event", {"hello": "world"})
        assert "test_event" in events_received
