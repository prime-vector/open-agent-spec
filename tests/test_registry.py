"""Tests for spec registry URL resolution and remote spec fetching.

Covers:
  - oa:// shorthand expansion (with and without @version)
  - https:// pass-through
  - Local path detection
  - Remote spec fetching (mocked HTTP)
  - Remote spec delegation in the runner (mocked)
  - Cycle detection for remote specs
  - Error cases: bad URL format, HTTP errors, invalid YAML
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from oas_cli.runner import (
    _REGISTRY_BASE,
    OARunError,
    _fetch_remote_spec,
    _is_remote_ref,
    _resolve_spec_url,
    run_task_from_file,
)

# ── URL resolution ────────────────────────────────────────────────────────────


class TestResolveSpecUrl:
    def test_oa_shorthand_no_version(self):
        url = _resolve_spec_url("oa://prime-vector/summariser")
        assert url == f"{_REGISTRY_BASE}/prime-vector/summariser/latest/spec.yaml"

    def test_oa_shorthand_with_version(self):
        url = _resolve_spec_url("oa://prime-vector/summariser@1.0.0")
        assert url == f"{_REGISTRY_BASE}/prime-vector/summariser/1.0.0/spec.yaml"

    def test_https_url_returned_as_is(self):
        url = "https://example.com/my/spec.yaml"
        assert _resolve_spec_url(url) == url

    def test_http_url_returned_as_is(self):
        url = "http://localhost:8080/spec.yaml"
        assert _resolve_spec_url(url) == url

    def test_oa_shorthand_wrong_segments_raises(self):
        with pytest.raises(OARunError) as exc_info:
            _resolve_spec_url("oa://too/many/parts/here")
        assert exc_info.value.code == "SPEC_LOAD_ERROR"

    def test_oa_shorthand_single_segment_raises(self):
        with pytest.raises(OARunError) as exc_info:
            _resolve_spec_url("oa://just-name")
        assert exc_info.value.code == "SPEC_LOAD_ERROR"


class TestIsRemoteRef:
    def test_oa_scheme(self):
        assert _is_remote_ref("oa://prime-vector/summariser") is True

    def test_https_scheme(self):
        assert _is_remote_ref("https://example.com/spec.yaml") is True

    def test_http_scheme(self):
        assert _is_remote_ref("http://localhost/spec.yaml") is True

    def test_local_relative_path(self):
        assert _is_remote_ref("./shared/spec.yaml") is False

    def test_local_absolute_path(self):
        assert _is_remote_ref("/Users/me/spec.yaml") is False


# ── Remote spec fetching ──────────────────────────────────────────────────────


def _make_mock_response(body: str, status: int = 200):
    mock_resp = MagicMock()
    mock_resp.read.return_value = body.encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_MINIMAL_SPEC_YAML = yaml.dump(
    {
        "open_agent_spec": "1.5.0",
        "agent": {"name": "test", "description": "test", "role": "analyst"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
        "tasks": {
            "run": {
                "description": "test task",
                "input": {"type": "object", "properties": {}, "required": []},
                "output": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
                "prompts": {"system": "s", "user": "u"},
            }
        },
    }
)


class TestFetchRemoteSpec:
    def test_successful_fetch(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_make_mock_response(_MINIMAL_SPEC_YAML),
        ):
            spec = _fetch_remote_spec("https://example.com/spec.yaml")
        assert spec["open_agent_spec"] == "1.5.0"
        assert "tasks" in spec

    def test_http_error_raises(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://example.com/spec.yaml",
                code=404,
                msg="Not Found",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            ),
        ):
            with pytest.raises(OARunError) as exc_info:
                _fetch_remote_spec("https://example.com/spec.yaml")
        assert exc_info.value.code == "SPEC_LOAD_ERROR"
        assert "404" in str(exc_info.value)

    def test_url_error_raises(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(OARunError) as exc_info:
                _fetch_remote_spec("https://example.com/spec.yaml")
        assert exc_info.value.code == "SPEC_LOAD_ERROR"

    def test_invalid_yaml_raises(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_make_mock_response(":: not valid yaml ::"),
        ):
            with pytest.raises(OARunError) as exc_info:
                _fetch_remote_spec("https://example.com/spec.yaml")
        assert exc_info.value.code == "SPEC_LOAD_ERROR"

    def test_non_mapping_yaml_raises(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_make_mock_response("- just\n- a\n- list\n"),
        ):
            with pytest.raises(OARunError) as exc_info:
                _fetch_remote_spec("https://example.com/spec.yaml")
        assert exc_info.value.code == "SPEC_LOAD_ERROR"


# ── Remote delegation in the runner ──────────────────────────────────────────


def _make_coordinator_spec(spec_ref: str, task_ref: str = "run") -> dict:
    return {
        "open_agent_spec": "1.5.0",
        "agent": {"name": "coordinator", "description": "test", "role": "analyst"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
        "prompts": {"system": "s", "user": "u"},
        "tasks": {
            "delegate": {
                "description": "Delegated task",
                "spec": spec_ref,
                "task": task_ref,
            }
        },
    }


class TestRemoteDelegation:
    def _write_coordinator(self, tmp_path: Path, spec_ref: str) -> Path:
        p = tmp_path / "coordinator.yaml"
        p.write_text(yaml.dump(_make_coordinator_spec(spec_ref)))
        return p

    def test_oa_url_delegation_executes_remote_spec(self, tmp_path):
        coordinator = self._write_coordinator(tmp_path, "oa://prime-vector/summariser")

        with patch(
            "oas_cli.runner._fetch_remote_spec",
            return_value=yaml.safe_load(_MINIMAL_SPEC_YAML),
        ):
            with patch(
                "oas_cli.runner.invoke_intelligence", return_value='{"result": "ok"}'
            ):
                result = run_task_from_file(coordinator, task_name="delegate")

        assert result["task"] == "delegate"
        assert "delegated_to" in result
        assert "prime-vector/summariser" in result["delegated_to"]

    def test_https_url_delegation_executes_remote_spec(self, tmp_path):
        coordinator = self._write_coordinator(
            tmp_path, "https://example.com/specs/myspec.yaml"
        )

        with patch(
            "oas_cli.runner._fetch_remote_spec",
            return_value=yaml.safe_load(_MINIMAL_SPEC_YAML),
        ):
            with patch(
                "oas_cli.runner.invoke_intelligence", return_value='{"result": "ok"}'
            ):
                result = run_task_from_file(coordinator, task_name="delegate")

        assert result["task"] == "delegate"
        assert "https://example.com" in result["delegated_to"]

    def test_remote_cycle_detection(self, tmp_path):
        """A remote spec that delegates back to the same URL raises DELEGATION_CYCLE_ERROR."""
        remote_url = "https://example.com/cycle.yaml"

        # Coordinator delegates to remote_url
        coordinator = self._write_coordinator(tmp_path, remote_url)

        # The remote spec also delegates to itself
        self_delegating_spec = {
            "open_agent_spec": "1.5.0",
            "agent": {"name": "cyclic", "description": "cycles", "role": "analyst"},
            "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
            "prompts": {"system": "s", "user": "u"},
            "tasks": {
                "run": {
                    "description": "Self-delegating",
                    "spec": remote_url,
                    "task": "run",
                }
            },
        }

        with patch(
            "oas_cli.runner._fetch_remote_spec", return_value=self_delegating_spec
        ):
            with pytest.raises(OARunError) as exc_info:
                run_task_from_file(coordinator, task_name="delegate")

        assert exc_info.value.code == "DELEGATION_CYCLE_ERROR"

    def test_remote_task_not_found_raises(self, tmp_path):
        p = tmp_path / "coordinator.yaml"
        p.write_text(
            yaml.dump(
                _make_coordinator_spec("oa://prime-vector/summariser", "ghost_task")
            )
        )
        coordinator = p

        with patch(
            "oas_cli.runner._fetch_remote_spec",
            return_value=yaml.safe_load(_MINIMAL_SPEC_YAML),
        ):
            with pytest.raises(OARunError) as exc_info:
                run_task_from_file(coordinator, task_name="delegate")

        assert exc_info.value.code == "TASK_NOT_FOUND"

    def test_registry_index_is_valid_json(self):
        """Smoke-test the bundled registry index."""
        index_path = (
            Path(__file__).parent.parent
            / "Website"
            / "public"
            / "registry"
            / "index.json"
        )
        with index_path.open() as f:
            data = json.load(f)

        assert "specs" in data
        assert len(data["specs"]) >= 5
        for spec in data["specs"]:
            assert "id" in spec
            assert "url" in spec
            assert spec["url"].startswith("https://openagentspec.dev/registry/")
