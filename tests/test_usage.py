"""Tests for token-usage normalisation, cost estimation, and envelope wiring."""

from __future__ import annotations

from oas_cli.providers import InvokeOutcome
from oas_cli.providers.registry import invoke_intelligence, pop_last_usage
from oas_cli.runner import _invoke_with_tools, run_task_from_spec
from oas_cli.tool_providers.base import InvokeResult, ToolCall, ToolDefinition
from oas_cli.usage import estimate_cost_usd, from_anthropic, from_openai

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_openai_chat_completions_shape(self):
        usage = from_openai(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_openai_responses_shape_derives_total(self):
        usage = from_openai({"input_tokens": 8, "output_tokens": 4})
        assert usage == {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}

    def test_anthropic_shape_derives_total(self):
        usage = from_anthropic({"input_tokens": 100, "output_tokens": 20})
        assert usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

    def test_missing_or_garbage_returns_none(self):
        assert from_openai(None) is None
        assert from_openai({}) is None
        assert from_anthropic("nonsense") is None


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def test_known_model_costs_input_and_output_separately(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        # gpt-4o is (2.50, 10.00) per 1M → 12.50 total.
        assert estimate_cost_usd("gpt-4o", usage) == 12.5

    def test_longest_prefix_wins(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        # gpt-4o-mini (0.15) must win over gpt-4o (2.50) for a -mini model id.
        assert estimate_cost_usd("gpt-4o-mini-2024-07-18", usage) == 0.15

    def test_unknown_model_returns_none(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 100}
        assert estimate_cost_usd("some-unlisted-model", usage) is None

    def test_no_usage_returns_none(self):
        assert estimate_cost_usd("gpt-4o", None) is None


# ---------------------------------------------------------------------------
# invoke_intelligence captures usage for the ContextVar
# ---------------------------------------------------------------------------


class TestInvokeIntelligenceUsage:
    def test_usage_and_cost_recorded(self, monkeypatch):
        class VerboseProvider:
            def invoke_verbose(self, *, system, user, config, history=None):
                return InvokeOutcome(
                    text='{"ok": true}',
                    usage={
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                )

        monkeypatch.setattr(
            "oas_cli.providers.registry.get_provider",
            lambda _c: VerboseProvider(),
        )
        invoke_intelligence("sys", "usr", {"engine": "openai", "model": "gpt-4o"})
        usage = pop_last_usage()
        assert usage["total_tokens"] == 1_000_000
        assert usage["estimated_cost_usd"] == 2.5  # gpt-4o input rate
        # pop clears it.
        assert pop_last_usage() is None

    def test_provider_without_invoke_verbose_falls_back(self, monkeypatch):
        class LegacyProvider:
            def invoke(self, *, system, user, config, history=None):
                return '{"ok": true}'

        monkeypatch.setattr(
            "oas_cli.providers.registry.get_provider",
            lambda _c: LegacyProvider(),
        )
        text = invoke_intelligence("sys", "usr", {"engine": "openai"})
        assert text == '{"ok": true}'
        assert pop_last_usage() is None


# ---------------------------------------------------------------------------
# Envelope wiring
# ---------------------------------------------------------------------------


def _spec() -> dict:
    return {
        "open_agent_spec": "1.5.0",
        "agent": {"name": "a", "description": "d"},
        "intelligence": {"type": "llm", "engine": "openai", "model": "gpt-4o"},
        "tasks": {
            "t": {
                "description": "d",
                "output": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                },
                "prompts": {"system": "s", "user": "{{ q }}"},
            }
        },
    }


class TestEnvelopeUsage:
    def test_envelope_includes_usage_when_reported(self, monkeypatch):
        def fake_invoke(system, user, config, history=None):
            # Simulate the registry recording usage for this call.
            from oas_cli.providers import registry

            registry._LAST_USAGE.set(
                {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            )
            return '{"ok": true}'

        monkeypatch.setattr("oas_cli.runner.invoke_intelligence", fake_invoke)
        result = run_task_from_spec(_spec(), task_name="t", input_data={"q": "hi"})
        assert result["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    def test_envelope_usage_none_when_not_reported(self, monkeypatch):
        # A fake that does not touch the ContextVar → usage stays None.
        pop_last_usage()  # clear any residue from a prior test on this context
        monkeypatch.setattr(
            "oas_cli.runner.invoke_intelligence",
            lambda s, u, c, h=None: '{"ok": true}',
        )
        result = run_task_from_spec(_spec(), task_name="t", input_data={"q": "hi"})
        assert result["usage"] is None

    def test_envelope_usage_captured_on_tool_fallback(self, monkeypatch):
        # A tool-declared task on a text-only provider falls back through
        # invoke_intelligence (which records usage). The runner must capture it
        # from the tools branch, not just the no-tools branch.
        pop_last_usage()  # clear residue

        def fake_invoke_with_tools(*args, **kwargs):
            from oas_cli.providers import registry

            registry._LAST_USAGE.set(
                {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            )
            return '{"ok": true}'

        # Force the tools branch without standing up real tool providers.
        monkeypatch.setattr(
            "oas_cli.runner.resolve_task_tools", lambda spec, name: [("t", object())]
        )
        monkeypatch.setattr("oas_cli.runner._invoke_with_tools", fake_invoke_with_tools)

        result = run_task_from_spec(_spec(), task_name="t", input_data={"q": "hi"})
        assert result["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }


class TestToolLoopUsage:
    def test_native_loop_sums_usage_across_turns(self, monkeypatch):
        pop_last_usage()  # clear residue

        class FakeToolProvider:
            def __init__(self):
                self.calls = 0

            def supports_tools(self):
                return True

            def invoke_with_tools(self, *, system, messages, tools, config):
                self.calls += 1
                if self.calls == 1:
                    return InvokeResult(
                        is_final=False,
                        tool_calls=[ToolCall(id="1", name="t", arguments={})],
                        usage={
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    )
                return InvokeResult(
                    is_final=True,
                    text='{"ok": true}',
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 4,
                        "total_tokens": 24,
                    },
                )

        monkeypatch.setattr(
            "oas_cli.runner.get_provider", lambda _c: FakeToolProvider()
        )
        monkeypatch.setattr(
            "oas_cli.runner.dispatch_tool_call", lambda name, args, tools: "tool out"
        )

        tools = [("t", ToolDefinition(name="t", description="d"))]
        text = _invoke_with_tools(
            "sys", "usr", tools, {"model": "gpt-4o"}, "task", None, sandbox=None
        )
        assert text == '{"ok": true}'

        usage = pop_last_usage()
        # Summed across both turns: prompt 10+20, completion 5+4, total 15+24.
        assert usage["prompt_tokens"] == 30
        assert usage["completion_tokens"] == 9
        assert usage["total_tokens"] == 39
        # Cost enrichment applied for a known model.
        assert usage["estimated_cost_usd"] is not None

    def test_native_loop_usage_none_when_provider_omits_it(self, monkeypatch):
        pop_last_usage()  # clear residue

        class NoUsageProvider:
            def supports_tools(self):
                return True

            def invoke_with_tools(self, *, system, messages, tools, config):
                return InvokeResult(is_final=True, text='{"ok": true}')

        monkeypatch.setattr("oas_cli.runner.get_provider", lambda _c: NoUsageProvider())
        tools = [("t", ToolDefinition(name="t", description="d"))]
        _invoke_with_tools(
            "sys", "usr", tools, {"model": "gpt-4o"}, "task", None, sandbox=None
        )
        assert pop_last_usage() is None
