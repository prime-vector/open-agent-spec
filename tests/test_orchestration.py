"""Tests for the multi-agent orchestration example."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

# Add the example directory to the path so we can import directly.
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "multi-agent")
sys.path.insert(0, os.path.abspath(EXAMPLE_DIR))

from board import TaskBoard, TaskPriority, TaskStatus  # noqa: E402
from loop import OrchestrationLoop  # noqa: E402
from registry import AgentRegistry  # noqa: E402
from runner import AgentRunner  # noqa: E402

PERSONAS_DIR = os.path.join(EXAMPLE_DIR, "personas")


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
        board.post_task("Second", "s", "writer", {}, depends_on=[t1.id])
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
        runner = AgentRunner(os.path.join(PERSONAS_DIR, "manager.yaml"))
        assert runner.agent_name == "manager-agent"
        assert runner.agent_role == "planner"
        assert "decompose" in runner.task_names

    def test_loads_worker_spec(self) -> None:
        runner = AgentRunner(os.path.join(PERSONAS_DIR, "researcher.yaml"))
        assert runner.agent_role == "analyst"
        assert "analyse" in runner.task_names

    def test_loads_concierge_spec(self) -> None:
        runner = AgentRunner(os.path.join(PERSONAS_DIR, "concierge.yaml"))
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
            manager_spec=os.path.join(PERSONAS_DIR, "manager.yaml"),
            worker_specs=[
                os.path.join(PERSONAS_DIR, "researcher.yaml"),
                os.path.join(PERSONAS_DIR, "writer.yaml"),
                os.path.join(PERSONAS_DIR, "reviewer.yaml"),
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
            {
                "title": "Research AI",
                "description": "Find info",
                "required_role": "analyst",
                "priority": 3,
            },
            {
                "title": "Write post",
                "description": "Draft article",
                "required_role": "writer",
                "priority": 2,
                "depends_on_titles": ["Research AI"],
            },
        ]
        loop._populate_board(planned)
        assert loop.board.summary()["total"] == 2
        # Writer task blocked by analyst task.
        assert len(loop.board.available_tasks("writer")) == 0
        assert len(loop.board.available_tasks("analyst")) == 1

    @patch("runner.AgentRunner.run_task")
    def test_full_run_mocked(self, mock_run_task) -> None:
        """Test the full loop with mocked LLM calls."""
        loop = self._make_loop()

        # Mock responses: manager decomposes, then workers execute.
        manager_response = {
            "output": {
                "tasks": [
                    {
                        "title": "Analyse topic",
                        "description": "Research AI agents",
                        "required_role": "analyst",
                        "priority": 3,
                    },
                    {
                        "title": "Write article",
                        "description": "Draft blog post",
                        "required_role": "writer",
                        "priority": 2,
                        "depends_on_titles": ["Analyse topic"],
                    },
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
            manager_spec=os.path.join(PERSONAS_DIR, "manager.yaml"),
            worker_specs=[
                os.path.join(PERSONAS_DIR, "researcher.yaml"),
                os.path.join(PERSONAS_DIR, "writer.yaml"),
                os.path.join(PERSONAS_DIR, "reviewer.yaml"),
            ],
            concierge_spec=os.path.join(PERSONAS_DIR, "concierge.yaml"),
        )

    def test_concierge_registers(self) -> None:
        loop = self._make_loop_with_concierge()
        summary = loop.registry.summary()
        assert summary["total"] == 5
        assert "chat" in summary["by_role"]
        assert loop._concierge_runner is not None

    @patch("runner.AgentRunner.run_task")
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
                    {
                        "title": "Research frameworks",
                        "description": "Find top AI agent frameworks",
                        "required_role": "analyst",
                        "priority": 3,
                    },
                ],
                "summary": "Single research task.",
            }
        }
        analyst_response = {
            "output": {
                "findings": "LangChain, CrewAI, OA Spec",
                "key_points": ["three frameworks"],
            }
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
        assert (
            result["refined_objective"]
            == "Write a 500-word blog post about AI agent frameworks"
        )
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

    def test_status_snapshot(self) -> None:
        loop = self._make_loop()
        loop._populate_board(
            [
                {
                    "title": "A",
                    "description": "a",
                    "required_role": "analyst",
                    "priority": 2,
                },
            ]
        )
        status = loop.status()
        assert "board" in status
        assert "tasks" in status
        assert "agents" in status
        assert "events" in status
        assert len(status["tasks"]) == 1

    @patch("runner.AgentRunner.run_task")
    def test_max_iterations_limit(self, mock_run_task) -> None:
        """Loop should stop after max_iterations even if board isn't complete."""
        loop = OrchestrationLoop(
            manager_spec=os.path.join(PERSONAS_DIR, "manager.yaml"),
            worker_specs=[os.path.join(PERSONAS_DIR, "researcher.yaml")],
            max_iterations=1,
        )
        # Manager plans two tasks — but we cap at 1 iteration.
        manager_response = {
            "output": {
                "tasks": [
                    {
                        "title": "Task A",
                        "description": "a",
                        "required_role": "analyst",
                        "priority": 2,
                    },
                    {
                        "title": "Task B",
                        "description": "b",
                        "required_role": "analyst",
                        "priority": 2,
                    },
                ],
                "summary": "Two tasks.",
            }
        }
        analyst_response = {"output": {"findings": "done", "key_points": ["ok"]}}
        mock_run_task.side_effect = [manager_response, analyst_response]

        result = loop.run("Do two things")
        assert result["iterations"] <= 1

    @patch("runner.AgentRunner.run_task")
    def test_worker_failure_recorded(self, mock_run_task) -> None:
        """A worker returning an error should mark the task as failed."""
        loop = self._make_loop()

        manager_response = {
            "output": {
                "tasks": [
                    {
                        "title": "Bad task",
                        "description": "will fail",
                        "required_role": "analyst",
                        "priority": 2,
                    },
                ],
                "summary": "One task.",
            }
        }
        mock_run_task.side_effect = [
            manager_response,
            {"error": "LLM timeout", "code": "RUN_ERROR", "stage": "run"},
        ]

        result = loop.run("Do something")
        assert result["board"]["by_status"].get("failed") == 1
        failed_task = [t for t in result["tasks"] if t["status"] == "failed"]
        assert len(failed_task) == 1
        assert "LLM timeout" in failed_task[0]["error"]

    @patch("runner.AgentRunner.run_task")
    def test_unmatched_role_clamped(self, mock_run_task) -> None:
        """Tasks with an unknown role should be clamped to a valid worker role."""
        loop = self._make_loop()

        manager_response = {
            "output": {
                "tasks": [
                    {
                        "title": "Deploy app",
                        "description": "push to prod",
                        "required_role": "devops",
                        "priority": 4,
                    },
                ],
                "summary": "One task.",
            }
        }
        worker_response = {"output": {"findings": "deployed"}}
        mock_run_task.side_effect = [manager_response, worker_response]

        result = loop.run("Deploy the app")
        # The unknown 'devops' role gets clamped to the nearest valid role,
        # so the task completes rather than stalling.
        assert result["board"]["by_status"].get("completed") == 1
        assert result["board"]["by_status"].get("pending", 0) == 0

    @patch("runner.AgentRunner.run_task")
    def test_clarify_error_falls_back(self, mock_run_task) -> None:
        """If the concierge fails to clarify, the raw objective passes through."""
        loop = self._make_loop_with_concierge()

        clarify_error = {"error": "API key missing"}
        manager_response = {
            "output": {
                "tasks": [
                    {
                        "title": "Do it",
                        "description": "just do it",
                        "required_role": "analyst",
                        "priority": 2,
                    },
                ],
                "summary": "One task.",
            }
        }
        analyst_response = {"output": {"findings": "done", "key_points": ["ok"]}}
        # summarise also errors — should still complete
        summarise_error = {"error": "API key missing"}

        mock_run_task.side_effect = [
            clarify_error,
            manager_response,
            analyst_response,
            summarise_error,
        ]

        result = loop.run("raw objective")
        # Should fall back to the raw objective.
        assert result["refined_objective"] == "raw objective"
        assert result["board"]["by_status"]["completed"] == 1
        # No summary key when summarise fails.
        assert "summary" not in result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_task_to_dict(self) -> None:
        board = TaskBoard()
        task = board.post_task(
            "Title", "Desc", "analyst", {"key": "val"}, priority=TaskPriority.HIGH
        )
        d = task.to_dict()
        assert d["id"] == task.id
        assert d["title"] == "Title"
        assert d["description"] == "Desc"
        assert d["required_role"] == "analyst"
        assert d["input_data"] == {"key": "val"}
        assert d["priority"] == 3
        assert d["status"] == "pending"
        assert d["assigned_to"] is None
        assert d["depends_on"] == []
        assert d["source"] == "manager"
        assert isinstance(d["created_at"], float)

    def test_agent_entry_to_dict(self) -> None:
        from registry import AgentEntry

        entry = AgentEntry(id="a1", role="analyst", spec_path="s.yaml")
        d = entry.to_dict()
        assert d["id"] == "a1"
        assert d["role"] == "analyst"
        assert d["spec_path"] == "s.yaml"
        assert d["tasks_completed"] == 0
        assert d["tasks_failed"] == 0
        assert isinstance(d["registered_at"], float)

    def test_board_summary_structure(self) -> None:
        board = TaskBoard()
        board.post_task("A", "a", "analyst", {})
        t2 = board.post_task("B", "b", "writer", {})
        board.claim_task(t2.id, "w1")
        board.complete_task(t2.id, {})
        s = board.summary()
        assert s["total"] == 2
        assert s["by_status"]["pending"] == 1
        assert s["by_status"]["completed"] == 1

    def test_registry_summary_structure(self) -> None:
        reg = AgentRegistry()
        reg.register("a", "analyst", "s.yaml")
        reg.register("b", "analyst", "s2.yaml")
        reg.register("c", "writer", "s3.yaml")
        s = reg.summary()
        assert s["total"] == 3
        assert s["by_role"]["analyst"] == 2
        assert s["by_role"]["writer"] == 1


# ---------------------------------------------------------------------------
# AgentRunner edge cases
# ---------------------------------------------------------------------------


class TestAgentRunnerEdgeCases:
    def test_spec_data_property(self) -> None:
        runner = AgentRunner(os.path.join(PERSONAS_DIR, "manager.yaml"))
        assert isinstance(runner.spec_data, dict)
        assert "agent" in runner.spec_data
        assert "tasks" in runner.spec_data

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            AgentRunner("/nonexistent/path/to/spec.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: a: valid: {yaml: [")
        with pytest.raises(Exception):
            AgentRunner(str(bad))

    def test_load_non_dict_raises(self, tmp_path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="did not parse to a dict"):
            AgentRunner(str(bad))
