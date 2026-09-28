"""OpenTelemetry execution evidence stays standard, optional, and content-safe."""

from __future__ import annotations

from typing import Any

import pytest

from oas_cli import runner
from oas_cli.providers import record_usage
from oas_cli.runner import OARunError, run_task_from_spec
from oas_cli.telemetry import (
    NOOP_TELEMETRY,
    OpenTelemetryAdapter,
    spec_identity,
)
from oas_cli.tool_providers.base import (
    InvokeResult,
    ToolCall,
    ToolDefinition,
    ToolProvider,
)

otel_sdk = pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


def _spec(*, tools: bool = False, sandbox: dict[str, Any] | None = None) -> dict:
    task: dict[str, Any] = {
        "prompts": {
            "system": "SECRET SYSTEM PROMPT",
            "user": "Review {text}",
        },
        "input": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "output": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    }
    spec: dict[str, Any] = {
        "open_agent_spec": "1.6.1",
        "agent": {"name": "reviewer"},
        "intelligence": {
            "type": "llm",
            "engine": "openai",
            "model": "test-model",
        },
        "tasks": {"review": task},
    }
    if tools:
        spec["tools"] = {
            "http.get": {
                "type": "native",
                "native": "http.get",
            }
        }
        task["tools"] = ["http.get"]
    if sandbox is not None:
        task["sandbox"] = sandbox
    return spec


@pytest.fixture
def captured_telemetry():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = OpenTelemetryAdapter(
        tracer_provider=provider,
        configure_otlp=False,
    )
    yield adapter, exporter
    provider.shutdown()


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {span.name: span for span in exporter.get_finished_spans()}


def test_spec_identity_is_canonical_and_sensitive_to_declaration():
    left = {"agent": {"name": "a"}, "open_agent_spec": "1.6.1"}
    reordered = {"open_agent_spec": "1.6.1", "agent": {"name": "a"}}
    changed = {"open_agent_spec": "1.6.1", "agent": {"name": "b"}}

    assert spec_identity(left) == spec_identity(reordered)
    assert spec_identity(left).startswith("sha256:")
    assert spec_identity(left) != spec_identity(changed)


def test_model_and_validation_spans_use_metadata_only(monkeypatch, captured_telemetry):
    adapter, exporter = captured_telemetry
    spec = _spec()
    statuses_seen_during_execution: list[str | None] = []

    def fake_invoke(*args, **kwargs):
        statuses_seen_during_execution.append(
            adapter._agent_spans.get()[-1].attributes.get("oa.conformance.status")
        )
        record_usage(
            {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            "test-model",
        )
        return '{"summary":"SECRET MODEL RESPONSE"}'

    monkeypatch.setattr(runner, "invoke_intelligence", fake_invoke)
    result = run_task_from_spec(
        spec,
        task_name="review",
        input_data={"text": "SECRET USER INPUT"},
        telemetry=adapter,
    )

    assert result["output"] == {"summary": "SECRET MODEL RESPONSE"}
    spans = _spans_by_name(exporter)
    agent = spans["invoke_agent reviewer"]
    model = spans["chat test-model"]
    validation = spans["oa.validation"]

    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent.attributes["oa.spec.hash"] == spec_identity(spec)
    assert agent.attributes["oa.conformance.status"] == "conformant"
    assert statuses_seen_during_execution == ["indeterminate"]
    assert agent.attributes["oa.telemetry.content_capture"] is False
    assert model.attributes["gen_ai.provider.name"] == "openai"
    assert model.attributes["gen_ai.request.model"] == "test-model"
    assert model.attributes["gen_ai.usage.input_tokens"] == 11
    assert model.attributes["gen_ai.usage.output_tokens"] == 4
    assert validation.attributes["oa.validation.result"] == "passed"

    serialized_attributes = repr(
        [dict(span.attributes) for span in exporter.get_finished_spans()]
    )
    assert "SECRET SYSTEM PROMPT" not in serialized_attributes
    assert "SECRET USER INPUT" not in serialized_attributes
    assert "SECRET MODEL RESPONSE" not in serialized_attributes


class _ToolProvider(ToolProvider):
    def describe(self):
        return [ToolDefinition(name="http.get", description="Fetch a URL")]

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        return "SECRET TOOL RESULT"


class _ToolCallingModel:
    def __init__(self, url: str):
        self.url = url
        self.calls = 0

    def supports_tools(self) -> bool:
        return True

    def invoke_with_tools(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return InvokeResult(
                is_final=False,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="http.get",
                        arguments={"url": self.url},
                    )
                ],
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            )
        return InvokeResult(
            is_final=True,
            text='{"summary":"done"}',
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
            },
        )


def test_declared_tool_and_allowed_sandbox_are_conformant(
    monkeypatch, captured_telemetry
):
    adapter, exporter = captured_telemetry
    model = _ToolCallingModel("https://allowed.example/data")
    monkeypatch.setattr(runner, "get_provider", lambda config: model)
    monkeypatch.setattr(
        runner,
        "resolve_task_tools",
        lambda spec, task: [(_ToolProvider(), _ToolProvider().describe()[0])],
    )

    run_task_from_spec(
        _spec(
            tools=True,
            sandbox={
                "tools": {"allow": ["http.get"]},
                "http": {"allow_domains": ["allowed.example"]},
            },
        ),
        "review",
        {"text": "private"},
        telemetry=adapter,
    )

    spans = _spans_by_name(exporter)
    assert spans["execute_tool http.get"].attributes["oa.tool.result"] == "succeeded"
    assert spans["oa.sandbox.decision"].attributes["oa.sandbox.result"] == "allowed"
    assert (
        spans["invoke_agent reviewer"].attributes["oa.conformance.status"]
        == "conformant"
    )


def test_blocked_sandbox_action_is_violation_prevented(monkeypatch, captured_telemetry):
    adapter, exporter = captured_telemetry
    model = _ToolCallingModel("https://blocked.example/data")
    monkeypatch.setattr(runner, "get_provider", lambda config: model)
    monkeypatch.setattr(
        runner,
        "resolve_task_tools",
        lambda spec, task: [(_ToolProvider(), _ToolProvider().describe()[0])],
    )

    with pytest.raises(OARunError) as caught:
        run_task_from_spec(
            _spec(
                tools=True,
                sandbox={
                    "tools": {"allow": ["http.get"]},
                    "http": {"allow_domains": ["allowed.example"]},
                },
            ),
            "review",
            {"text": "private"},
            telemetry=adapter,
        )

    assert caught.value.code == "SANDBOX_DOMAIN_VIOLATION"
    spans = _spans_by_name(exporter)
    sandbox = spans["oa.sandbox.decision"]
    agent = spans["invoke_agent reviewer"]
    assert sandbox.attributes["oa.sandbox.result"] == "blocked"
    assert agent.attributes["oa.conformance.status"] == "violation_prevented"
    assert agent.attributes["oa.conformance.reason"] == "sandbox_domain_violation"


def test_schema_failure_is_nonconformant_not_provider_failure(
    monkeypatch, captured_telemetry
):
    adapter, exporter = captured_telemetry
    monkeypatch.setattr(runner, "invoke_intelligence", lambda *a, **k: "{}")

    with pytest.raises(OARunError) as caught:
        run_task_from_spec(
            _spec(),
            "review",
            {"text": "private"},
            telemetry=adapter,
        )

    assert caught.value.code == "OUTPUT_SCHEMA_ERROR"
    spans = _spans_by_name(exporter)
    agent = spans["invoke_agent reviewer"]
    assert agent.attributes["oa.conformance.status"] == "nonconformant"
    assert agent.attributes["oa.conformance.reason"] == "output_schema_error"


def test_provider_failure_remains_indeterminate(monkeypatch, captured_telemetry):
    adapter, exporter = captured_telemetry

    def fail_invoke(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(runner, "invoke_intelligence", fail_invoke)

    with pytest.raises(OARunError) as caught:
        run_task_from_spec(
            _spec(),
            "review",
            {"text": "private"},
            telemetry=adapter,
        )

    assert caught.value.code == "RUN_ERROR"
    agent = _spans_by_name(exporter)["invoke_agent reviewer"]
    assert agent.attributes["oa.conformance.status"] == "indeterminate"


def test_contract_result_is_recorded(monkeypatch, captured_telemetry):
    adapter, exporter = captured_telemetry
    monkeypatch.setattr(
        runner, "invoke_intelligence", lambda *a, **k: '{"summary":"ok"}'
    )
    monkeypatch.setattr(runner, "CONTRACTS_ENABLED", True)
    monkeypatch.setattr(runner, "validate_task_output", lambda output, contract: None)

    spec = _spec()
    spec["behavioural_contract"] = {
        "version": "1",
        "description": "No secrets",
        "response_contract": {"required_fields": ["summary"]},
    }
    run_task_from_spec(spec, "review", {"text": "private"}, telemetry=adapter)

    contract = _spans_by_name(exporter)["oa.contract.evaluate"]
    assert contract.attributes["oa.contract.enabled"] is True
    assert contract.attributes["oa.contract.result"] == "passed"


def test_delegated_execution_has_its_own_lifecycle_and_identity(
    monkeypatch, captured_telemetry, tmp_path
):
    adapter, exporter = captured_telemetry
    delegated_path = tmp_path / "delegated.yaml"
    delegated_spec = _spec()
    delegated_spec["agent"]["name"] = "delegated-reviewer"
    delegated_path.write_text("agent: {}\n")

    # The runner receives parsed delegated data from _load_spec; the file only
    # supplies the path used for cycle/relative-reference handling.
    monkeypatch.setattr(runner, "_load_spec", lambda path: delegated_spec)
    statuses_seen_during_execution: list[tuple[str | None, ...]] = []

    def fake_invoke(*args, **kwargs):
        statuses_seen_during_execution.append(
            tuple(
                span.attributes.get("oa.conformance.status")
                for span in adapter._agent_spans.get()
            )
        )
        return '{"summary":"delegated"}'

    monkeypatch.setattr(runner, "invoke_intelligence", fake_invoke)
    root_spec = {
        "open_agent_spec": "1.6.1",
        "agent": {"name": "coordinator"},
        "tasks": {"review": {"spec": delegated_path.name}},
    }

    run_task_from_spec(
        root_spec,
        "review",
        {"text": "private"},
        spec_path=tmp_path / "root.yaml",
        telemetry=adapter,
    )

    spans = _spans_by_name(exporter)
    assert statuses_seen_during_execution == [("indeterminate", "indeterminate")]
    assert spans["invoke_agent coordinator"].attributes["oa.conformance.status"] == (
        "conformant"
    )
    assert (
        spans["invoke_agent delegated-reviewer"].attributes["oa.conformance.status"]
        == "conformant"
    )
    assert spans["invoke_agent coordinator"].attributes[
        "oa.spec.hash"
    ] == spec_identity(root_spec)
    assert spans["invoke_agent delegated-reviewer"].attributes[
        "oa.spec.hash"
    ] == spec_identity(delegated_spec)


def test_noop_telemetry_is_available_without_sdk_calls():
    with NOOP_TELEMETRY.agent_run(_spec(), "review") as observation:
        observation.succeed()
    assert NOOP_TELEMETRY.close() is None
