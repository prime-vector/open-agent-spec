"""Tests for reasoning-effort tiers and their per-engine mappings."""

from __future__ import annotations

import pytest

from oas_cli.adapters.codex_adapter import _build_codex_command
from oas_cli.providers.anthropic_http import (
    _apply_reasoning,
    _extract_text_blocks,
    _supports_temperature,
)
from oas_cli.providers.openai_http import (
    _build_chat_completions_payload,
    _build_responses_payload,
)
from oas_cli.reasoning import (
    codex_reasoning_flags,
    normalise_effort,
    openai_reasoning_params,
)

# ---------------------------------------------------------------------------
# normalise_effort
# ---------------------------------------------------------------------------


class TestNormaliseEffort:
    @pytest.mark.parametrize("value", ["low", "MEDIUM", " High "])
    def test_valid_values_lowercased(self, value):
        assert normalise_effort(value) in ("low", "medium", "high")

    def test_none_and_empty_are_none(self):
        assert normalise_effort(None) is None
        assert normalise_effort("") is None
        assert normalise_effort("  ") is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            normalise_effort("turbo")


# ---------------------------------------------------------------------------
# OpenAI mapping
# ---------------------------------------------------------------------------


class TestOpenAIReasoning:
    def test_chat_completions_flat_field(self):
        assert openai_reasoning_params("high", responses_api=False) == {
            "reasoning_effort": "high"
        }

    def test_responses_api_nested(self):
        assert openai_reasoning_params("low", responses_api=True) == {
            "reasoning": {"effort": "low"}
        }

    def test_unset_returns_empty(self):
        assert openai_reasoning_params(None, responses_api=False) == {}

    def test_chat_payload_reasoning_model_shape(self):
        # Reasoning models: max_completion_tokens (NOT max_tokens) and no
        # temperature — both are rejected by o-series / GPT-5 reasoning models.
        payload = _build_chat_completions_payload(
            "sys", "usr", "o4-mini", 0.7, 1000, None, "medium"
        )
        assert payload["reasoning_effort"] == "medium"
        assert payload["max_completion_tokens"] == 1000
        assert "max_tokens" not in payload
        assert "temperature" not in payload

    def test_chat_payload_standard_model_shape(self):
        # Standard models / OpenAI-compatible servers keep max_tokens + temperature.
        payload = _build_chat_completions_payload("sys", "usr", "gpt-4o", 0.7, 1000)
        assert "reasoning_effort" not in payload
        assert payload["max_tokens"] == 1000
        assert "max_completion_tokens" not in payload
        assert payload["temperature"] == 0.7

    def test_responses_payload_includes_effort_drops_temperature(self):
        payload = _build_responses_payload("sys", "usr", "o4-mini", 0.7, None, "high")
        assert payload["reasoning"] == {"effort": "high"}
        assert "temperature" not in payload


# ---------------------------------------------------------------------------
# Anthropic mapping
# ---------------------------------------------------------------------------


class TestAnthropicReasoning:
    def test_apply_reasoning_sets_effort_and_adaptive_thinking(self):
        payload = {"model": "claude-opus-4-8", "max_tokens": 1000, "temperature": 0.7}
        _apply_reasoning(payload, "high")
        assert payload["output_config"] == {"effort": "high"}
        assert payload["thinking"] == {"type": "adaptive"}
        # temperature is rejected alongside adaptive thinking on current models.
        assert "temperature" not in payload

    def test_apply_reasoning_preserves_existing_output_config(self):
        payload = {"model": "claude-opus-4-8", "output_config": {"format": "x"}}
        _apply_reasoning(payload, "low")
        assert payload["output_config"] == {"format": "x", "effort": "low"}

    def test_apply_reasoning_noop_when_unset(self):
        payload = {"model": "claude-opus-4-8", "max_tokens": 1000, "temperature": 0.7}
        _apply_reasoning(payload, None)
        assert "output_config" not in payload
        assert "thinking" not in payload
        assert payload["temperature"] == 0.7

    def test_extract_text_skips_thinking_block(self):
        data = {
            "content": [
                {"type": "thinking", "thinking": "let me reason..."},
                {"type": "text", "text": "the answer"},
            ]
        }
        assert _extract_text_blocks(data) == "the answer"

    def test_extract_text_handles_untyped_blocks(self):
        # Test doubles / older payloads may omit the type field.
        data = {"content": [{"text": "hello"}]}
        assert _extract_text_blocks(data) == "hello"


# ---------------------------------------------------------------------------
# Anthropic temperature gating (Opus 4.7/4.8 + Fable 5 reject `temperature`)
# ---------------------------------------------------------------------------


class TestAnthropicTemperatureGate:
    def test_supports_temperature_by_model(self):
        assert _supports_temperature("claude-sonnet-4-5-20250929") is True
        assert _supports_temperature("claude-haiku-4-5-20251001") is True
        assert _supports_temperature("claude-opus-4-6") is True
        assert _supports_temperature("claude-opus-4-7") is False
        assert _supports_temperature("claude-opus-4-8") is False
        assert _supports_temperature("claude-fable-5") is False

    def _capture_payload(self, monkeypatch, model: str) -> dict:
        import json

        import oas_cli.providers.anthropic_http as m

        captured: dict = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {
                        "content": [{"type": "text", "text": "{}"}],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _Resp()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
        m.AnthropicProvider().invoke_verbose(
            system="s", user="u", config={"model": model}
        )
        return captured["body"]

    def test_temperature_omitted_for_opus_4_8(self, monkeypatch):
        body = self._capture_payload(monkeypatch, "claude-opus-4-8")
        assert "temperature" not in body

    def test_temperature_sent_for_sonnet(self, monkeypatch):
        body = self._capture_payload(monkeypatch, "claude-sonnet-4-5-20250929")
        assert body["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Codex mapping
# ---------------------------------------------------------------------------


class TestCodexReasoning:
    def test_flags_for_tier(self):
        assert codex_reasoning_flags("high") == ["-c", "model_reasoning_effort=high"]

    def test_no_flags_when_unset(self):
        assert codex_reasoning_flags(None) == []

    def test_command_includes_reasoning_flag(self):
        cmd, _cwd = _build_codex_command("do a thing", {"reasoning_effort": "low"})
        assert "-c" in cmd
        assert "model_reasoning_effort=low" in cmd

    def test_command_omits_flag_when_unset(self):
        cmd, _cwd = _build_codex_command("do a thing", {})
        assert "model_reasoning_effort" not in " ".join(cmd)
