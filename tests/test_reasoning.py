"""Tests for reasoning-effort tiers and their per-engine mappings."""

from __future__ import annotations

import pytest

from oas_cli.adapters.codex_adapter import _build_codex_command
from oas_cli.providers.anthropic_http import _apply_thinking, _extract_text_blocks
from oas_cli.providers.openai_http import (
    _build_chat_completions_payload,
    _build_responses_payload,
)
from oas_cli.reasoning import (
    anthropic_thinking_budget,
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

    def test_chat_payload_includes_effort(self):
        payload = _build_chat_completions_payload(
            "sys", "usr", "gpt-5", 0.7, 1000, None, "medium"
        )
        assert payload["reasoning_effort"] == "medium"

    def test_chat_payload_omits_when_unset(self):
        payload = _build_chat_completions_payload("sys", "usr", "gpt-4o", 0.7, 1000)
        assert "reasoning_effort" not in payload

    def test_responses_payload_includes_effort(self):
        payload = _build_responses_payload("sys", "usr", "o4-mini", 0.7, None, "high")
        assert payload["reasoning"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# Anthropic mapping
# ---------------------------------------------------------------------------


class TestAnthropicReasoning:
    def test_budget_increases_with_tier(self):
        low = anthropic_thinking_budget("low")
        med = anthropic_thinking_budget("medium")
        high = anthropic_thinking_budget("high")
        assert low < med < high

    def test_budget_none_when_unset(self):
        assert anthropic_thinking_budget(None) is None

    def test_apply_thinking_sets_block_and_constraints(self):
        payload = {"model": "claude-opus-4", "max_tokens": 1000, "temperature": 0.7}
        _apply_thinking(payload, "high")
        budget = anthropic_thinking_budget("high")
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": budget}
        # Extended thinking requires temperature 1 and max_tokens above the budget.
        assert payload["temperature"] == 1.0
        assert payload["max_tokens"] > budget

    def test_apply_thinking_noop_when_unset(self):
        payload = {"model": "claude-opus-4", "max_tokens": 1000, "temperature": 0.7}
        _apply_thinking(payload, None)
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
