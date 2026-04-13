"""Tests for the OAS tool provider layer.

Covers:
- NativeToolProvider: all five built-in tools
- CustomToolProvider: class loading, describe(), call()
- MCPToolProvider: tools/list discovery, tools/call, result normalisation,
  env-var header expansion, error handling
- Registry helpers: get_tool_provider, resolve_task_tools, dispatch_tool_call
- _invoke_with_tools loop in runner (multi-turn with mocks)
- JSON schema: tools: block and per-task tools: array
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from oas_cli.tool_providers.base import (
    InvokeResult,
    ToolCall,
    ToolError,
    ToolNotFoundError,
    ToolTypeNotSupportedError,
)
from oas_cli.tool_providers.custom import CustomToolProvider
from oas_cli.tool_providers.native import NativeToolProvider, available_native_tools
from oas_cli.tool_providers.registry import (
    dispatch_tool_call,
    get_tool_provider,
    resolve_task_tools,
)

# ── NativeToolProvider ────────────────────────────────────────────────────────


class TestNativeToolProvider:
    def test_available_native_tools_contains_expected(self):
        ids = available_native_tools()
        assert "file.read" in ids
        assert "file.write" in ids
        assert "http.get" in ids
        assert "http.post" in ids
        assert "env.read" in ids

    def test_unknown_tool_raises(self):
        with pytest.raises(ToolError, match="Unknown native tool"):
            NativeToolProvider(enabled_ids=["no.such.tool"])

    def test_describe_returns_definitions(self):
        p = NativeToolProvider(enabled_ids=["file.read", "env.read"])
        defs = p.describe()
        assert len(defs) == 2
        names = {d.name for d in defs}
        assert "file_read" in names
        assert "env_read" in names

    # ── file.read ──────────────────────────────────────────────────────────

    def test_file_read_success(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("world", encoding="utf-8")
        p = NativeToolProvider(enabled_ids=["file.read"])
        result = p.call("file_read", {"path": str(f)})
        assert result == "world"

    def test_file_read_not_found(self):
        p = NativeToolProvider(enabled_ids=["file.read"])
        with pytest.raises(ToolError, match="file not found"):
            p.call("file_read", {"path": "/no/such/file.txt"})

    # ── file.write ─────────────────────────────────────────────────────────

    def test_file_write_creates_file(self, tmp_path):
        dest = tmp_path / "sub" / "out.txt"
        p = NativeToolProvider(enabled_ids=["file.write"])
        result = p.call("file_write", {"path": str(dest), "content": "hello"})
        assert dest.read_text() == "hello"
        assert "Written" in result

    # ── env.read ───────────────────────────────────────────────────────────

    def test_env_read_existing_var(self, monkeypatch):
        monkeypatch.setenv("_OA_TEST_VAR", "abc123")
        p = NativeToolProvider(enabled_ids=["env.read"])
        result = p.call("env_read", {"name": "_OA_TEST_VAR"})
        assert result == "abc123"

    def test_env_read_missing_var(self, monkeypatch):
        monkeypatch.delenv("_OA_MISSING", raising=False)
        p = NativeToolProvider(enabled_ids=["env.read"])
        result = p.call("env_read", {"name": "_OA_MISSING"})
        assert result == ""

    # ── http.get (mocked) ──────────────────────────────────────────────────

    def test_http_get_success(self):
        p = NativeToolProvider(enabled_ids=["http.get"])
        response_mock = MagicMock()
        response_mock.read.return_value = b"OK response"
        response_mock.__enter__ = lambda s: response_mock
        response_mock.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=response_mock):
            result = p.call("http_get", {"url": "https://example.com"})
        assert result == "OK response"

    def test_http_get_http_error(self):
        import urllib.error

        p = NativeToolProvider(enabled_ids=["http.get"])
        err = urllib.error.HTTPError("https://example.com", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(ToolError, match="HTTP 404"):
                p.call("http_get", {"url": "https://example.com"})

    # ── http.post (mocked) ─────────────────────────────────────────────────

    def test_http_post_success(self):
        p = NativeToolProvider(enabled_ids=["http.post"])
        response_mock = MagicMock()
        response_mock.read.return_value = b'{"status":"ok"}'
        response_mock.__enter__ = lambda s: response_mock
        response_mock.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=response_mock):
            result = p.call(
                "http_post", {"url": "https://api.example.com", "body": {"k": "v"}}
            )
        assert '{"status":"ok"}' in result

    # ── call routing ───────────────────────────────────────────────────────

    def test_call_unknown_tool_raises_not_found(self):
        p = NativeToolProvider(enabled_ids=["env.read"])
        with pytest.raises(ToolNotFoundError):
            p.call("file_read", {"path": "/tmp/x"})

    # ── openai schema ──────────────────────────────────────────────────────

    def test_to_openai_schema(self):
        p = NativeToolProvider(enabled_ids=["file.read"])
        defs = p.describe()
        schema = defs[0].to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "file_read"
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_to_anthropic_schema(self):
        p = NativeToolProvider(enabled_ids=["file.read"])
        defs = p.describe()
        schema = defs[0].to_anthropic_schema()
        assert schema["name"] == "file_read"
        assert "input_schema" in schema


# ── CustomToolProvider ────────────────────────────────────────────────────────


class _FakeTool:
    """Minimal in-test custom tool class."""

    def describe(self):
        return [
            {
                "name": "fake_tool",
                "description": "A fake tool for testing.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        ]

    def call(self, tool_name: str, arguments: dict) -> str:
        return f"FAKE:{arguments.get('text', '')}"


class TestCustomToolProvider:
    def test_loads_class_and_describes(self, monkeypatch):
        # Inject fake class into sys.modules path
        import sys
        import types

        mod = types.ModuleType("_test_fake_tools")
        mod.FakeTool = _FakeTool
        monkeypatch.setitem(sys.modules, "_test_fake_tools", mod)

        p = CustomToolProvider(
            tool_name="my_fake",
            tool_config={"type": "custom", "module": "_test_fake_tools.FakeTool"},
        )
        defs = p.describe()
        assert len(defs) == 1
        assert defs[0].name == "fake_tool"

    def test_calls_tool(self, monkeypatch):
        import sys
        import types

        mod = types.ModuleType("_test_fake_tools2")
        mod.FakeTool = _FakeTool
        monkeypatch.setitem(sys.modules, "_test_fake_tools2", mod)

        p = CustomToolProvider(
            tool_name="my_fake",
            tool_config={"type": "custom", "module": "_test_fake_tools2.FakeTool"},
        )
        result = p.call("fake_tool", {"text": "hello"})
        assert result == "FAKE:hello"

    def test_missing_module_raises(self):
        with pytest.raises(ToolError, match="no_such_module"):
            CustomToolProvider(
                tool_name="x",
                tool_config={"type": "custom", "module": "no_such_module.Class"},
            )

    def test_no_module_field_raises(self):
        with pytest.raises(ToolError, match="must specify a 'module'"):
            CustomToolProvider(tool_name="x", tool_config={"type": "custom"})

    def test_synthesises_definition_without_describe(self, monkeypatch):
        import sys
        import types

        class _MinimalTool:
            def call(self, name, args):
                return "ok"

        mod = types.ModuleType("_test_minimal_tools")
        mod.MinimalTool = _MinimalTool
        monkeypatch.setitem(sys.modules, "_test_minimal_tools", mod)

        p = CustomToolProvider(
            tool_name="my_minimal",
            tool_config={
                "type": "custom",
                "module": "_test_minimal_tools.MinimalTool",
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}},
            },
        )
        defs = p.describe()
        assert len(defs) == 1
        assert defs[0].description == "Does something"


# ── Registry helpers ──────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_provider_native(self):
        provider = get_tool_provider(
            "read_file", {"type": "native", "native": "file.read"}
        )
        assert isinstance(provider, NativeToolProvider)

    def test_get_provider_unknown_type_raises(self):
        with pytest.raises(ToolTypeNotSupportedError):
            get_tool_provider("x", {"type": "totally_unsupported"})

    def test_get_provider_native_missing_native_field_raises(self):
        with pytest.raises(ToolError, match="missing the 'native' field"):
            get_tool_provider("x", {"type": "native"})

    def test_resolve_task_tools_empty_when_no_tools(self):
        spec = {
            "tasks": {"my_task": {"description": "d", "output": {}}},
        }
        result = resolve_task_tools(spec, "my_task")
        assert result == []

    def test_resolve_task_tools_returns_pairs(self):
        spec = {
            "tools": {
                "read_file": {"type": "native", "native": "file.read"},
                "get_env": {"type": "native", "native": "env.read"},
            },
            "tasks": {
                "my_task": {
                    "description": "d",
                    "output": {},
                    "tools": ["read_file", "get_env"],
                }
            },
        }
        pairs = resolve_task_tools(spec, "my_task")
        assert len(pairs) == 2
        _, defn = pairs[0]
        assert defn.name == "file_read"

    def test_resolve_task_tools_missing_top_level_raises(self):
        spec = {
            "tools": {},
            "tasks": {
                "my_task": {"description": "d", "output": {}, "tools": ["missing_tool"]}
            },
        }
        with pytest.raises(ToolError, match="not declared"):
            resolve_task_tools(spec, "my_task")

    def test_dispatch_tool_call_success(self):
        p = NativeToolProvider(enabled_ids=["env.read"])
        defn = p.describe()[0]
        pairs = [(p, defn)]
        result = dispatch_tool_call("env_read", {"name": "PATH"}, pairs)
        assert isinstance(result, str)

    def test_dispatch_tool_call_unknown_raises(self):
        p = NativeToolProvider(enabled_ids=["env.read"])
        defn = p.describe()[0]
        pairs = [(p, defn)]
        with pytest.raises(ToolNotFoundError):
            dispatch_tool_call("file_read", {"path": "/tmp/x"}, pairs)


# ── _invoke_with_tools loop (runner integration) ──────────────────────────────


class TestInvokeWithToolsLoop:
    """Unit-tests for the multi-turn tool-call loop in runner.py."""

    def _make_spec(self) -> dict:
        return {
            "open_agent_spec": "1.5.0",
            "agent": {"name": "test", "description": "test"},
            "intelligence": {
                "type": "llm",
                "engine": "openai",
                "model": "gpt-4o",
            },
            "tasks": {
                "ask": {
                    "description": "Ask something",
                    "output": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "tools": ["get_env"],
                }
            },
            "tools": {
                "get_env": {"type": "native", "native": "env.read"},
            },
            "prompts": {
                "system": "You are a helpful assistant.",
                "user": "What is PATH?",
            },
        }

    def test_single_turn_final_answer(self):
        """Provider returns is_final immediately — no tool execution."""
        spec = self._make_spec()
        final_result = InvokeResult(is_final=True, text='{"answer": "direct"}')

        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True
        mock_provider.invoke_with_tools.return_value = final_result

        with patch("oas_cli.runner.get_provider", return_value=mock_provider):
            from oas_cli.runner import _invoke_with_tools
            from oas_cli.tool_providers.registry import resolve_task_tools

            tools = resolve_task_tools(spec, "ask")
            result = _invoke_with_tools(
                "system", "user", tools, {"engine": "openai", "model": "gpt-4o"}, "ask"
            )

        assert result == '{"answer": "direct"}'
        mock_provider.invoke_with_tools.assert_called_once()

    def test_one_tool_call_then_final(self, monkeypatch):
        """Provider requests one tool call, gets result, then returns final answer."""
        monkeypatch.setenv("MY_TEST_ENV", "42")
        spec = self._make_spec()

        tool_call = ToolCall(
            id="call_1", name="env_read", arguments={"name": "MY_TEST_ENV"}
        )
        turn1 = InvokeResult(is_final=False, tool_calls=[tool_call])
        turn2 = InvokeResult(is_final=True, text='{"answer": "42"}')

        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True
        mock_provider.invoke_with_tools.side_effect = [turn1, turn2]

        with patch("oas_cli.runner.get_provider", return_value=mock_provider):
            from oas_cli.runner import _invoke_with_tools
            from oas_cli.tool_providers.registry import resolve_task_tools

            tools = resolve_task_tools(spec, "ask")
            result = _invoke_with_tools(
                "system", "user", tools, {"engine": "openai", "model": "gpt-4o"}, "ask"
            )

        assert result == '{"answer": "42"}'
        assert mock_provider.invoke_with_tools.call_count == 2
        # Verify the tool result was injected into the second call's messages
        second_call_messages = mock_provider.invoke_with_tools.call_args_list[1][1][
            "messages"
        ]
        tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
        assert tool_msg["content"] == "42"

    def test_fallback_for_non_tool_providers(self):
        """Providers that don't support tools get an augmented system prompt instead."""
        spec = self._make_spec()

        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = False

        with (
            patch("oas_cli.runner.get_provider", return_value=mock_provider),
            patch(
                "oas_cli.runner.invoke_intelligence", return_value='{"answer":"ok"}'
            ) as mock_invoke,
        ):
            from oas_cli.runner import _invoke_with_tools
            from oas_cli.tool_providers.registry import resolve_task_tools

            tools = resolve_task_tools(spec, "ask")
            _invoke_with_tools(
                "You are helpful.",
                "user",
                tools,
                {"engine": "codex", "model": "x"},
                "ask",
            )

        mock_invoke.assert_called_once()
        call_system = mock_invoke.call_args[0][0]
        assert "env_read" in call_system

    def test_loop_cap_raises_run_error(self):
        """Loop exits with OARunError after _MAX_TOOL_ITERATIONS."""
        from oas_cli.runner import _MAX_TOOL_ITERATIONS, OARunError

        spec = self._make_spec()
        tool_call = ToolCall(id="call_1", name="env_read", arguments={"name": "X"})
        never_final = InvokeResult(is_final=False, tool_calls=[tool_call])

        mock_provider = MagicMock()
        mock_provider.supports_tools.return_value = True
        mock_provider.invoke_with_tools.return_value = never_final

        with patch("oas_cli.runner.get_provider", return_value=mock_provider):
            from oas_cli.runner import _invoke_with_tools
            from oas_cli.tool_providers.registry import resolve_task_tools

            tools = resolve_task_tools(spec, "ask")
            with pytest.raises(OARunError, match="exceeded"):
                _invoke_with_tools(
                    "system",
                    "user",
                    tools,
                    {"engine": "openai", "model": "gpt-4o"},
                    "ask",
                )

        assert mock_provider.invoke_with_tools.call_count == _MAX_TOOL_ITERATIONS


# ── MCPToolProvider ───────────────────────────────────────────────────────────


def _mock_urlopen(responses: list[dict]):
    """Return a context manager mock that yields JSON responses in order."""

    call_count = {"n": 0}

    class _Resp:
        def __init__(self, data):
            self._data = json.dumps(data).encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def _urlopen(req, timeout=30):
        idx = call_count["n"]
        call_count["n"] += 1
        return _Resp(responses[idx])

    return _urlopen


_LIST_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "tools": [
            {
                "name": "search_web",
                "description": "Search the web for information.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        ]
    },
}

_CALL_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {"content": [{"type": "text", "text": "Paris is sunny and 22°C."}]},
}


class TestMCPToolProvider:
    def _provider(self, **extra):
        from oas_cli.tool_providers.mcp import MCPToolProvider

        config = {"type": "mcp", "endpoint": "http://localhost:3000", **extra}
        return MCPToolProvider(tool_name="my_mcp", tool_config=config)

    # ── construction ───────────────────────────────────────────────────────

    def test_missing_endpoint_raises(self):
        from oas_cli.tool_providers.mcp import MCPToolProvider

        with pytest.raises(ToolError, match="endpoint"):
            MCPToolProvider(tool_name="x", tool_config={"type": "mcp"})

    # ── describe / tools/list ──────────────────────────────────────────────

    def test_describe_calls_tools_list(self):
        p = self._provider()
        with patch(
            "urllib.request.urlopen", side_effect=_mock_urlopen([_LIST_RESPONSE])
        ):
            defs = p.describe()

        assert len(defs) == 2
        names = {d.name for d in defs}
        assert "search_web" in names
        assert "get_weather" in names

    def test_describe_is_cached(self):
        p = self._provider()
        urlopen_mock = MagicMock(side_effect=_mock_urlopen([_LIST_RESPONSE]))
        with patch("urllib.request.urlopen", urlopen_mock):
            p.describe()
            p.describe()

        # Only one HTTP call despite two describe() calls
        assert urlopen_mock.call_count == 1

    def test_describe_parses_input_schema(self):
        p = self._provider()
        with patch(
            "urllib.request.urlopen", side_effect=_mock_urlopen([_LIST_RESPONSE])
        ):
            defs = p.describe()

        search = next(d for d in defs if d.name == "search_web")
        assert "query" in search.parameters["properties"]

    def test_empty_tools_list_raises(self):
        p = self._provider()
        empty_response = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        with patch(
            "urllib.request.urlopen", side_effect=_mock_urlopen([empty_response])
        ):
            with pytest.raises(ToolError, match="no tools"):
                p.describe()

    # ── call / tools/call ──────────────────────────────────────────────────

    def test_call_success(self):
        p = self._provider()
        with patch(
            "urllib.request.urlopen",
            side_effect=_mock_urlopen([_LIST_RESPONSE, _CALL_RESPONSE]),
        ):
            result = p.call("get_weather", {"location": "Paris"})

        assert result == "Paris is sunny and 22°C."

    def test_call_unknown_tool_raises(self):
        p = self._provider()
        with patch(
            "urllib.request.urlopen", side_effect=_mock_urlopen([_LIST_RESPONSE])
        ):
            with pytest.raises(ToolNotFoundError, match="magic_wand"):
                p.call("magic_wand", {})

    def test_call_sends_correct_jsonrpc_payload(self):
        p = self._provider()
        captured = {}

        def _capture(req, timeout=30):
            captured["body"] = json.loads(req.data.decode())

            class _R:
                def read(self):
                    return json.dumps(_CALL_RESPONSE).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    pass

            return _R()

        # First call discovers tools, second executes
        with patch(
            "urllib.request.urlopen",
            side_effect=[_mock_urlopen([_LIST_RESPONSE])("_"), _capture],
        ):
            # Seed the cache so only one HTTP round-trip happens
            with patch(
                "urllib.request.urlopen", side_effect=_mock_urlopen([_LIST_RESPONSE])
            ):
                p.describe()
            with patch("urllib.request.urlopen", _capture):
                p.call("get_weather", {"location": "Paris"})

        assert captured["body"]["method"] == "tools/call"
        assert captured["body"]["params"]["name"] == "get_weather"
        assert captured["body"]["params"]["arguments"] == {"location": "Paris"}

    # ── result normalisation ───────────────────────────────────────────────

    def test_extract_bare_string(self):
        from oas_cli.tool_providers.mcp import _extract_call_result

        assert _extract_call_result("hello", "t", "http://x") == "hello"

    def test_extract_content_block_list(self):
        from oas_cli.tool_providers.mcp import _extract_call_result

        result = {
            "content": [
                {"type": "text", "text": "foo"},
                {"type": "text", "text": "bar"},
            ]
        }
        assert _extract_call_result(result, "t", "http://x") == "foo\nbar"

    def test_extract_content_string(self):
        from oas_cli.tool_providers.mcp import _extract_call_result

        assert _extract_call_result({"content": "direct"}, "t", "http://x") == "direct"

    def test_extract_result_key(self):
        from oas_cli.tool_providers.mcp import _extract_call_result

        assert _extract_call_result({"result": 42}, "t", "http://x") == "42"

    def test_extract_none_returns_empty(self):
        from oas_cli.tool_providers.mcp import _extract_call_result

        assert _extract_call_result(None, "t", "http://x") == ""

    # ── env-var header expansion ───────────────────────────────────────────

    def test_env_var_expanded_in_headers(self, monkeypatch):
        monkeypatch.setenv("MY_MCP_KEY", "secret-token")
        p = self._provider(headers={"X-API-Key": "${MY_MCP_KEY}"})

        captured_headers = {}

        def _capture(req, timeout=30):
            captured_headers.update(dict(req.headers))

            class _R:
                def read(self):
                    return json.dumps(_LIST_RESPONSE).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    pass

            return _R()

        with patch("urllib.request.urlopen", _capture):
            p.describe()

        # urllib title-cases header names
        assert captured_headers.get("X-api-key") == "secret-token"

    def test_missing_env_var_warns(self, monkeypatch):
        import warnings

        monkeypatch.delenv("MISSING_VAR", raising=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # env-var expansion happens at construction time inside _resolve_headers
            self._provider(headers={"X-Token": "${MISSING_VAR}"})

        assert any("MISSING_VAR" in str(warning.message) for warning in w)

    # ── HTTP error handling ────────────────────────────────────────────────

    def test_http_error_raises_tool_error(self):
        import urllib.error

        p = self._provider()
        err = urllib.error.HTTPError(
            "http://localhost:3000", 500, "Internal Server Error", {}, None
        )
        # HTTPError.read() is not available without a real response; mock it
        err.read = lambda: b"server exploded"
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(ToolError, match="MCP HTTP 500"):
                p.describe()

    def test_json_rpc_error_raises_tool_error(self):
        p = self._provider()
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        with patch(
            "urllib.request.urlopen", side_effect=_mock_urlopen([error_response])
        ):
            with pytest.raises(ToolError, match="Method not found"):
                p.describe()

    # ── registry integration ───────────────────────────────────────────────

    def test_registry_routes_mcp_type(self):
        from oas_cli.tool_providers.mcp import MCPToolProvider

        p = get_tool_provider(
            "my_server",
            {"type": "mcp", "endpoint": "http://localhost:3000"},
        )
        assert isinstance(p, MCPToolProvider)

    def test_to_openai_schema_shape(self):
        p = self._provider()
        with patch(
            "urllib.request.urlopen", side_effect=_mock_urlopen([_LIST_RESPONSE])
        ):
            defs = p.describe()

        schema = defs[0].to_openai_schema()
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]

    # ── validator ─────────────────────────────────────────────────────────

    def test_validator_accepts_mcp_tool(self):
        from oas_cli.validators import validate_spec

        spec = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test", "role": "test"},
            "tools": {
                "my_mcp": {
                    "type": "mcp",
                    "endpoint": "http://localhost:3000",
                    "description": "A remote MCP server",
                }
            },
            "tasks": {
                "task1": {
                    "tools": ["my_mcp"],
                    "input": {"query": {"type": "string"}},
                    "output": {"result": {"type": "string"}},
                }
            },
            "prompts": {"system": "test", "user": "test"},
        }
        validate_spec(spec)  # must not raise

    def test_validator_rejects_mcp_without_endpoint(self):
        from oas_cli.validators import validate_spec

        spec = {
            "open_agent_spec": "1.0",
            "agent": {"name": "test", "role": "test"},
            "tools": {"my_mcp": {"type": "mcp"}},
            "tasks": {
                "task1": {
                    "input": {"q": {"type": "string"}},
                    "output": {"r": {"type": "string"}},
                }
            },
            "prompts": {"system": "test", "user": "test"},
        }
        with pytest.raises(ValueError, match="endpoint"):
            validate_spec(spec)
