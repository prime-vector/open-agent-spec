"""Tests for Immutable Inference Sandboxing (IIS).

Covers:
  _resolve_sandbox  — root / task-level merge semantics
  _check_sandbox    — SANDBOX_TOOL_VIOLATION, SANDBOX_DOMAIN_VIOLATION,
                      SANDBOX_PATH_VIOLATION
  _invoke_with_tools integration — violation raises before dispatch
  chain-wide immutability — upstream mutations never leak downstream
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from oas_cli.runner import (
    OARunError,
    _check_sandbox,
    _resolve_sandbox,
    run_task_from_spec,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _minimal_spec(*, tasks: dict | None = None, sandbox: dict | None = None) -> dict:
    spec: dict = {
        "open_agent_spec": "1.4.0",
        "agent": {"name": "test-agent", "description": "test", "role": "analyst"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
        "tasks": tasks
        or {
            "run": {
                "description": "test task",
                "output": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
                "prompts": {"system": "You are helpful.", "user": "Do it."},
            }
        },
    }
    if sandbox is not None:
        spec["sandbox"] = sandbox
    return spec


# ── _resolve_sandbox ──────────────────────────────────────────────────────────


class TestResolveSandbox:
    def test_returns_empty_dict_when_no_sandbox(self):
        spec = _minimal_spec()
        assert _resolve_sandbox(spec, "run") == {}

    def test_returns_root_sandbox_when_no_task_override(self):
        root = {"tools": {"allow": ["file.read"]}}
        spec = _minimal_spec(sandbox=root)
        assert _resolve_sandbox(spec, "run") == root

    def test_task_sandbox_overrides_root(self):
        root = {"tools": {"allow": ["file.read", "http.get"]}}
        task_sandbox = {"tools": {"allow": ["http.get"]}}
        spec = _minimal_spec(sandbox=root)
        spec["tasks"]["run"]["sandbox"] = task_sandbox
        result = _resolve_sandbox(spec, "run")
        assert result == {"tools": {"allow": ["http.get"]}}

    def test_task_sandbox_merges_missing_root_keys(self):
        root = {"http": {"allow_domains": ["api.example.com"]}}
        task_sandbox = {"tools": {"allow": ["http.get"]}}
        spec = _minimal_spec(sandbox=root)
        spec["tasks"]["run"]["sandbox"] = task_sandbox
        result = _resolve_sandbox(spec, "run")
        # task-level wins on overlap; root keys not overridden are inherited
        assert result["tools"] == {"allow": ["http.get"]}
        assert result["http"] == {"allow_domains": ["api.example.com"]}

    def test_returns_copy_not_reference(self):
        root = {"tools": {"allow": ["file.read"]}}
        spec = _minimal_spec(sandbox=root)
        result = _resolve_sandbox(spec, "run")
        result["tools"]["allow"].append("http.get")
        # Mutating the result must not affect the spec
        assert spec["sandbox"]["tools"]["allow"] == ["file.read"]


# ── _check_sandbox — tool enforcement ────────────────────────────────────────


class TestCheckSandboxToolViolation:
    def test_allow_list_permits_listed_tool(self):
        sandbox = {"tools": {"allow": ["file.read", "http.get"]}}
        _check_sandbox("file.read", {}, sandbox, "task")  # no exception

    def test_allow_list_blocks_unlisted_tool(self):
        sandbox = {"tools": {"allow": ["file.read"]}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox("http.get", {}, sandbox, "my_task")
        err = exc_info.value
        assert err.code == "SANDBOX_TOOL_VIOLATION"
        assert err.task == "my_task"
        assert "http.get" in str(err)
        assert "allow" in str(err).lower()

    def test_deny_list_blocks_listed_tool(self):
        sandbox = {"tools": {"deny": ["file.write", "env.read"]}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox("file.write", {}, sandbox, "task")
        err = exc_info.value
        assert err.code == "SANDBOX_TOOL_VIOLATION"
        assert "deny" in str(err).lower() or "file.write" in str(err)

    def test_deny_list_permits_unlisted_tool(self):
        sandbox = {"tools": {"deny": ["file.write"]}}
        _check_sandbox("file.read", {}, sandbox, "task")  # no exception

    def test_empty_sandbox_permits_any_tool(self):
        _check_sandbox("file.write", {}, {}, "task")  # no exception

    def test_tool_not_in_allow_reports_full_allow_list(self):
        allowed = ["file.read", "env.read"]
        sandbox = {"tools": {"allow": allowed}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox("http.post", {}, sandbox, "task")
        assert "file.read" in str(exc_info.value) or str(allowed) in str(exc_info.value)


# ── _check_sandbox — domain enforcement ──────────────────────────────────────


class TestCheckSandboxDomainViolation:
    def test_allows_exact_domain_match(self):
        sandbox = {"http": {"allow_domains": ["api.example.com"]}}
        _check_sandbox(
            "http.get", {"url": "https://api.example.com/v1"}, sandbox, "task"
        )

    def test_allows_subdomain(self):
        sandbox = {"http": {"allow_domains": ["example.com"]}}
        _check_sandbox(
            "http.get", {"url": "https://api.example.com/v1"}, sandbox, "task"
        )

    def test_blocks_unlisted_domain(self):
        sandbox = {"http": {"allow_domains": ["api.example.com"]}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox(
                "http.post", {"url": "https://evil.io/exfil"}, sandbox, "task"
            )
        err = exc_info.value
        assert err.code == "SANDBOX_DOMAIN_VIOLATION"
        assert "evil.io" in str(err)

    def test_blocks_domain_with_port(self):
        sandbox = {"http": {"allow_domains": ["api.example.com"]}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox(
                "http.get", {"url": "https://attacker.com:443/steal"}, sandbox, "task"
            )
        assert exc_info.value.code == "SANDBOX_DOMAIN_VIOLATION"

    def test_domain_check_skipped_for_non_http_tool(self):
        sandbox = {"http": {"allow_domains": ["api.example.com"]}}
        # file.read with a "url"-like argument should not trigger domain check
        _check_sandbox("file.read", {"url": "https://evil.io"}, sandbox, "task")

    def test_no_allow_domains_permits_any_url(self):
        sandbox = {"http": {}}
        _check_sandbox("http.get", {"url": "https://any-domain.io"}, sandbox, "task")


# ── _check_sandbox — path enforcement ────────────────────────────────────────


class TestCheckSandboxPathViolation:
    def test_allows_path_inside_allow_paths(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sandbox = {"file": {"allow_paths": [str(data_dir)]}}
        _check_sandbox(
            "file.read", {"path": str(data_dir / "report.txt")}, sandbox, "task"
        )

    def test_blocks_path_outside_allow_paths(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sandbox = {"file": {"allow_paths": [str(data_dir)]}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox(
                "file.read", {"path": str(tmp_path / "secret.txt")}, sandbox, "task"
            )
        err = exc_info.value
        assert err.code == "SANDBOX_PATH_VIOLATION"
        assert "secret.txt" in str(err) or "allow_paths" in str(err).lower()

    def test_blocks_path_traversal(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sandbox = {"file": {"allow_paths": [str(data_dir)]}}
        traversal = str(data_dir / ".." / "secret.txt")
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox("file.read", {"path": traversal}, sandbox, "task")
        assert exc_info.value.code == "SANDBOX_PATH_VIOLATION"

    def test_path_check_skipped_for_non_file_tool(self, tmp_path):
        sandbox = {"file": {"allow_paths": [str(tmp_path / "data")]}}
        # http.get with a "path"-like argument must not trigger path check
        _check_sandbox("http.get", {"path": "/etc/passwd"}, sandbox, "task")

    def test_no_allow_paths_permits_any_path(self):
        sandbox = {"file": {}}
        _check_sandbox("file.write", {"path": "/etc/shadow"}, sandbox, "task")

    def test_write_tool_also_checked(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sandbox = {"file": {"allow_paths": [str(data_dir)]}}
        with pytest.raises(OARunError) as exc_info:
            _check_sandbox(
                "file.write", {"path": str(tmp_path / "exfil.txt")}, sandbox, "task"
            )
        assert exc_info.value.code == "SANDBOX_PATH_VIOLATION"


# ── Integration: sandbox fires before dispatch in _invoke_with_tools ──────────


class TestSandboxIntegration:
    """Use run_task_from_spec with a mocked provider + tool to verify the
    sandbox check fires *before* dispatch_tool_call is reached."""

    def _make_spec_with_tool_and_sandbox(self, sandbox: dict) -> dict:
        spec = {
            "open_agent_spec": "1.4.0",
            "agent": {"name": "test", "description": "test", "role": "analyst"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
            "sandbox": sandbox,
            "tools": {
                "reader": {
                    "type": "native",
                    "native": "file.read",
                    "description": "Read a file",
                }
            },
            "tasks": {
                "read": {
                    "description": "Read task",
                    "tools": ["reader"],
                    "output": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                    "prompts": {
                        "system": "You are helpful.",
                        "user": "Read data/report.txt",
                    },
                }
            },
        }
        return spec

    def test_tool_violation_raised_before_dispatch(self):
        """SANDBOX_TOOL_VIOLATION is raised; dispatch_tool_call is never called."""
        from oas_cli.tool_providers.base import InvokeResult, ToolCall

        # Provider returns a tool call for 'file.read' but sandbox blocks it.
        fake_tc = ToolCall(
            id="tc1", name="file.read", arguments={"path": "data/report.txt"}
        )
        fake_result = InvokeResult(is_final=False, text="", tool_calls=[fake_tc])

        spec = self._make_spec_with_tool_and_sandbox(
            {"tools": {"allow": []}}  # empty allow list — nothing permitted
        )

        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True
        mock_provider.invoke_with_tools.return_value = fake_result

        with (
            patch("oas_cli.runner.get_provider", return_value=mock_provider),
            patch("oas_cli.runner.dispatch_tool_call") as mock_dispatch,
        ):
            with pytest.raises(OARunError) as exc_info:
                run_task_from_spec(spec, task_name="read", input_data={})

        assert exc_info.value.code == "SANDBOX_TOOL_VIOLATION"
        mock_dispatch.assert_not_called()

    def test_permitted_tool_is_dispatched(self):
        """When the tool is in the allow list, dispatch proceeds normally."""
        from oas_cli.tool_providers.base import InvokeResult, ToolCall

        fake_tc = ToolCall(
            id="tc1", name="file.read", arguments={"path": "data/report.txt"}
        )
        intermediate = InvokeResult(is_final=False, text="", tool_calls=[fake_tc])
        final = InvokeResult(is_final=True, text='{"content": "hello"}', tool_calls=[])

        spec = self._make_spec_with_tool_and_sandbox(
            {"tools": {"allow": ["file.read"]}}
        )

        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True
        mock_provider.invoke_with_tools.side_effect = [intermediate, final]

        with (
            patch("oas_cli.runner.get_provider", return_value=mock_provider),
            patch(
                "oas_cli.runner.dispatch_tool_call", return_value="file contents"
            ) as mock_dispatch,
        ):
            result = run_task_from_spec(spec, task_name="read", input_data={})

        mock_dispatch.assert_called_once()
        assert result["output"]["content"] == "hello"


# ── Chain-wide immutability ───────────────────────────────────────────────────


class TestInputImmutability:
    """Verify that no task in the chain mutates its caller's input dict."""

    def _chain_spec(self) -> dict:
        """Two-task chain: upstream returns extra_field, downstream checks it doesn't
        contaminate base_input back in run_task_from_spec."""
        return {
            "open_agent_spec": "1.4.0",
            "agent": {"name": "chain", "description": "chain test", "role": "analyst"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
            "tasks": {
                "upstream": {
                    "description": "upstream task",
                    "output": {
                        "type": "object",
                        "properties": {"extra_field": {"type": "string"}},
                    },
                    "prompts": {"system": "sys", "user": "user"},
                },
                "downstream": {
                    "description": "downstream task",
                    "depends_on": ["upstream"],
                    "output": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    },
                    "prompts": {"system": "sys", "user": "user"},
                },
            },
        }

    def test_original_input_dict_unchanged_after_chain(self):
        """The caller's input_data dict must be identical before and after the run."""
        original_input = {"base_key": "base_value"}
        snapshot = copy.deepcopy(original_input)

        spec = self._chain_spec()

        def fake_invoke(system, user, config, history=None):
            return '{"extra_field": "injected", "result": "done"}'

        with patch("oas_cli.runner.invoke_intelligence", side_effect=fake_invoke):
            run_task_from_spec(spec, task_name="downstream", input_data=original_input)

        assert original_input == snapshot, (
            f"Input dict was mutated: before={snapshot}, after={original_input}"
        )

    def test_upstream_output_does_not_contaminate_second_run(self):
        """Running the same spec twice with the same input produces the same output.

        If chain outputs leaked back into the shared input dict, the second run
        would see extra_field in base_input and produce a different result.
        """
        original_input = {"query": "hello"}
        spec = self._chain_spec()

        def fake_invoke(system, user, config, history=None):
            return '{"extra_field": "value", "result": "ok"}'

        with patch("oas_cli.runner.invoke_intelligence", side_effect=fake_invoke):
            result1 = run_task_from_spec(
                spec, task_name="downstream", input_data=copy.deepcopy(original_input)
            )
            result2 = run_task_from_spec(
                spec, task_name="downstream", input_data=copy.deepcopy(original_input)
            )

        # Both runs should see identical upstream chain outputs.
        assert (
            result1["chain"]["upstream"]["input"]
            == result2["chain"]["upstream"]["input"]
        ), (
            "Upstream task received different inputs across runs — input leaked between runs"
        )
