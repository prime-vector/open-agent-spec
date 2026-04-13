"""Tests for history threading — OAS injects prior turns between system and user."""

from __future__ import annotations

from pathlib import Path

import pytest

from oas_cli.providers.openai_http import (
    OpenAIProvider,
    _build_chat_completions_payload,
    _build_responses_payload,
)
from oas_cli.providers.registry import invoke_intelligence

# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


class TestChatCompletionsPayloadHistory:
    def test_no_history_produces_two_messages(self):
        payload = _build_chat_completions_payload(
            "sys", "user msg", "gpt-4o-mini", 0.7, 1000
        )
        msgs = payload["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "user msg"}

    def test_history_injected_between_system_and_user(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        payload = _build_chat_completions_payload(
            "sys", "follow-up", "gpt-4o-mini", 0.7, 1000, history
        )
        msgs = payload["messages"]
        assert len(msgs) == 4
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "hi"}
        assert msgs[2] == {"role": "assistant", "content": "hello"}
        assert msgs[3] == {"role": "user", "content": "follow-up"}

    def test_empty_history_list_treated_as_no_history(self):
        payload = _build_chat_completions_payload(
            "sys", "msg", "gpt-4o-mini", 0.7, 1000, []
        )
        assert len(payload["messages"]) == 2

    def test_none_history_treated_as_no_history(self):
        payload = _build_chat_completions_payload(
            "sys", "msg", "gpt-4o-mini", 0.7, 1000, None
        )
        assert len(payload["messages"]) == 2


class TestResponsesPayloadHistory:
    def test_history_injected_into_responses_payload(self):
        history = [{"role": "user", "content": "prev"}]
        payload = _build_responses_payload("sys", "cur", "gpt-4o-mini", 0.7, history)
        turns = payload["input"]
        assert turns[0]["role"] == "system"
        assert turns[1]["role"] == "user"
        assert turns[1]["content"] == "prev"
        assert turns[2]["content"] == "cur"


# ---------------------------------------------------------------------------
# Provider-level: OpenAIProvider.invoke passes history
# ---------------------------------------------------------------------------


class TestOpenAIProviderHistory:
    def test_history_forwarded_to_payload(self, monkeypatch):
        captured: dict = {}

        def fake_http_post(url, payload, headers, timeout):
            captured["payload"] = payload
            return '{"choices":[{"message":{"content":"{\\"reply\\":\\"ok\\"}"}}]}'

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        import oas_cli.providers.openai_http as m

        monkeypatch.setattr(m, "_http_post", fake_http_post)

        OpenAIProvider().invoke(
            system="sys",
            user="hello again",
            config={"engine": "openai", "model": "gpt-4o-mini"},
            history=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ],
        )

        msgs = captured["payload"]["messages"]
        assert msgs[1]["content"] == "first"
        assert msgs[2]["content"] == "reply"
        assert msgs[3]["content"] == "hello again"


# ---------------------------------------------------------------------------
# Provider registry: invoke_intelligence threads history
# ---------------------------------------------------------------------------


class TestInvokeIntelligenceHistory:
    def test_history_passed_through_to_provider(self, monkeypatch):
        received: dict = {}

        class MockProvider:
            def invoke(self, *, system, user, config, history=None):
                received["history"] = history
                return '{"reply": "ok"}'

        monkeypatch.setattr(
            "oas_cli.providers.registry.get_provider",
            lambda _config: MockProvider(),
        )

        history = [{"role": "user", "content": "turn 1"}]
        invoke_intelligence("sys", "user", {"engine": "openai"}, history)

        assert received["history"] == history

    def test_no_history_passes_none(self, monkeypatch):
        received: dict = {}

        class MockProvider:
            def invoke(self, *, system, user, config, history=None):
                received["history"] = history
                return '{"reply": "ok"}'

        monkeypatch.setattr(
            "oas_cli.providers.registry.get_provider",
            lambda _config: MockProvider(),
        )

        invoke_intelligence("sys", "user", {"engine": "openai"})
        assert received["history"] is None


# ---------------------------------------------------------------------------
# Runner integration: history extracted from input_data
# ---------------------------------------------------------------------------


class TestRunnerHistoryExtraction:
    """Verify the runner picks up `history` from input_data and forwards it."""

    @pytest.fixture()
    def chat_spec(self, tmp_path: Path) -> Path:
        spec = tmp_path / "chat.yaml"
        spec.write_text(
            """
open_agent_spec: "1.4.0"
agent:
  name: chat-agent
  role: assistant
intelligence:
  type: llm
  engine: openai
  model: gpt-4o-mini
tasks:
  chat:
    input:
      type: object
      properties:
        message: {type: string}
        history:
          type: array
          items:
            type: object
            properties:
              role: {type: string}
              content: {type: string}
      required: [message]
    output:
      type: object
      properties:
        reply: {type: string}
      required: [reply]
    prompts:
      system: You are helpful.
      user: "{message}"
"""
        )
        return spec

    def test_history_forwarded_from_input(self, chat_spec: Path, monkeypatch):
        received: dict = {}

        def fake_invoke(system, user, config, history=None):
            received["history"] = history
            return '{"reply": "got it"}'

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        import oas_cli.runner as runner_mod

        monkeypatch.setattr(runner_mod, "invoke_intelligence", fake_invoke)

        from oas_cli.runner import run_task_from_file

        history = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "first reply"},
        ]
        result = run_task_from_file(
            chat_spec, "chat", {"message": "second", "history": history}
        )

        assert result["output"]["reply"] == "got it"
        assert received["history"] == history

    def test_no_history_in_input_passes_none(self, chat_spec: Path, monkeypatch):
        received: dict = {}

        def fake_invoke(system, user, config, history=None):
            received["history"] = history
            return '{"reply": "ok"}'

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        import oas_cli.runner as runner_mod

        monkeypatch.setattr(runner_mod, "invoke_intelligence", fake_invoke)

        from oas_cli.runner import run_task_from_file

        run_task_from_file(chat_spec, "chat", {"message": "hello"})
        assert received["history"] is None

    def test_empty_history_list_passes_none(self, chat_spec: Path, monkeypatch):
        received: dict = {}

        def fake_invoke(system, user, config, history=None):
            received["history"] = history
            return '{"reply": "ok"}'

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        import oas_cli.runner as runner_mod

        monkeypatch.setattr(runner_mod, "invoke_intelligence", fake_invoke)

        from oas_cli.runner import run_task_from_file

        run_task_from_file(chat_spec, "chat", {"message": "hello", "history": []})
        # empty list → treated as None (no history)
        assert not received["history"]
