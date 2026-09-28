"""Optional OpenTelemetry execution evidence for the OA runtime.

The runner speaks in OA concepts through :class:`TelemetryAdapter`; only this
module knows about the OpenTelemetry SDK.  The default adapter is a no-op, so
normal execution neither imports OTEL nor changes behaviour.

Telemetry is metadata-only.  Prompts, inputs, outputs, tool arguments/results,
environment values, and secrets are deliberately not accepted by this API.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

CONFORMANT = "conformant"
INDETERMINATE = "indeterminate"
NONCONFORMANT = "nonconformant"
VIOLATION_PREVENTED = "violation_prevented"


class TelemetryUnavailableError(RuntimeError):
    """Raised when telemetry is requested without the optional OTEL packages."""


class Observation(Protocol):
    """A running telemetry observation exposed to the OA runtime."""

    def succeed(self, **metadata: Any) -> None: ...

    def fail(self, error: BaseException, **metadata: Any) -> None: ...


class TelemetryAdapter(Protocol):
    """OA-facing telemetry interface; implementations translate to a backend."""

    def agent_run(
        self, spec: dict[str, Any], task_name: str
    ) -> Any: ...  # context manager[Observation]

    def task_run(
        self, task_name: str, declared_tools: list[str]
    ) -> Any: ...  # context manager[Observation]

    def model_call(self, provider: str, model: str) -> Any: ...

    def tool_call(self, name: str, call_id: str | None = None) -> Any: ...

    def sandbox_decision(
        self, *, tool_name: str, result: str, reason: str | None = None
    ) -> None: ...

    def validation_result(
        self, *, kind: str, result: str, reason: str | None = None
    ) -> None: ...

    def contract_result(
        self, *, enabled: bool, result: str, reason: str | None = None
    ) -> None: ...

    def conformance(self, status: str, reason: str | None = None) -> None: ...

    def close(self) -> None: ...


class _NoOpObservation:
    def succeed(self, **metadata: Any) -> None:
        return None

    def fail(self, error: BaseException, **metadata: Any) -> None:
        return None


class NoOpTelemetry:
    """Effectively free adapter used unless telemetry is explicitly enabled."""

    _observation = _NoOpObservation()

    @contextmanager
    def agent_run(
        self, spec: dict[str, Any], task_name: str
    ) -> Iterator[_NoOpObservation]:
        yield self._observation

    @contextmanager
    def task_run(
        self, task_name: str, declared_tools: list[str]
    ) -> Iterator[_NoOpObservation]:
        yield self._observation

    @contextmanager
    def model_call(self, provider: str, model: str) -> Iterator[_NoOpObservation]:
        yield self._observation

    @contextmanager
    def tool_call(
        self, name: str, call_id: str | None = None
    ) -> Iterator[_NoOpObservation]:
        yield self._observation

    def sandbox_decision(
        self, *, tool_name: str, result: str, reason: str | None = None
    ) -> None:
        return None

    def validation_result(
        self, *, kind: str, result: str, reason: str | None = None
    ) -> None:
        return None

    def contract_result(
        self, *, enabled: bool, result: str, reason: str | None = None
    ) -> None:
        return None

    def conformance(self, status: str, reason: str | None = None) -> None:
        return None

    def close(self) -> None:
        return None


NOOP_TELEMETRY = NoOpTelemetry()


def spec_identity(spec: dict[str, Any]) -> str:
    """Return the SHA-256 identity of the canonical parsed OA specification.

    Canonical JSON makes key ordering and YAML presentation irrelevant while
    preserving the supplied declaration. Runtime input, prompt overrides,
    provider responses, and environment configuration are not part of *spec*
    and therefore cannot enter the identity. The identity does not claim to
    include every invocation-specific or runtime-resolved value.
    """

    canonical = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _clean_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop unset values; OTEL attributes cannot contain ``None``."""

    return {key: value for key, value in attributes.items() if value is not None}


_SPECIFIC_CONFORMANCE = {NONCONFORMANT, VIOLATION_PREVENTED}


def _should_update_conformance(current: Any, new: str) -> bool:
    """Keep specific execution outcomes from being replaced by a later default."""

    if current in _SPECIFIC_CONFORMANCE:
        return new in _SPECIFIC_CONFORMANCE and new == current
    if new in _SPECIFIC_CONFORMANCE:
        return True
    return True


@dataclass
class _OtelObservation:
    span: Any
    status_type: Any
    status_code: Any
    finished: bool = False

    def succeed(self, **metadata: Any) -> None:
        for key, value in _clean_attributes(metadata).items():
            self.span.set_attribute(key, value)
        self.span.set_status(self.status_type(self.status_code.OK))
        self.finished = True

    def fail(self, error: BaseException, **metadata: Any) -> None:
        for key, value in _clean_attributes(metadata).items():
            self.span.set_attribute(key, value)
        self.span.record_exception(error)
        self.span.set_status(self.status_type(self.status_code.ERROR, str(error)))
        self.finished = True


class OpenTelemetryAdapter:
    """Translate OA execution concepts into standard OpenTelemetry spans."""

    def __init__(
        self,
        *,
        tracer_provider: Any | None = None,
        configure_otlp: bool = True,
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise TelemetryUnavailableError(
                "OpenTelemetry support is not installed. Install with: "
                "pip install 'open-agent-spec[otel]'"
            ) from exc

        self._trace = trace
        self._span_kind = SpanKind
        self._status = Status
        self._status_code = StatusCode
        self._owned_provider: Any | None = None

        if tracer_provider is None:
            resource = Resource.create(
                {
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "open-agent-spec"),
                }
            )
            tracer_provider = TracerProvider(resource=resource)
            self._owned_provider = tracer_provider

            if configure_otlp:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )
                except ImportError as exc:
                    raise TelemetryUnavailableError(
                        "The OTLP exporter is not installed. Install with: "
                        "pip install 'open-agent-spec[otel]'"
                    ) from exc
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter())
                )

        self._tracer_provider = tracer_provider
        self._tracer = tracer_provider.get_tracer("open-agent-spec", "1.6.1")
        self._agent_spans: ContextVar[tuple[Any, ...]] = ContextVar(
            "oa_telemetry_agent_spans", default=()
        )

    @contextmanager
    def _span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: Any = None,
    ) -> Iterator[_OtelObservation]:
        with self._tracer.start_as_current_span(
            name,
            kind=kind or self._span_kind.INTERNAL,
            attributes=_clean_attributes(attributes or {}),
        ) as span:
            observation = _OtelObservation(
                span=span,
                status_type=self._status,
                status_code=self._status_code,
            )
            try:
                yield observation
            except Exception as exc:
                if not observation.finished:
                    observation.fail(exc)
                raise

    @contextmanager
    def agent_run(
        self, spec: dict[str, Any], task_name: str
    ) -> Iterator[_OtelObservation]:
        agent = spec.get("agent") or {}
        agent_name = str(agent.get("name") or "unnamed-agent")
        task = (spec.get("tasks") or {}).get(task_name) or {}
        declared_tools = [str(name) for name in (task.get("tools") or [])]
        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": agent_name,
            "oa.spec.name": agent_name,
            "oa.spec.version": str(spec.get("open_agent_spec") or "unknown"),
            "oa.spec.hash": spec_identity(spec),
            "oa.task.name": task_name,
            "oa.task.declared_tools": declared_tools,
            "oa.sandbox.enabled": bool(spec.get("sandbox") or task.get("sandbox")),
            "oa.contract.enabled": bool(
                spec.get("behavioural_contract") or task.get("behavioural_contract")
            ),
            "oa.conformance.status": INDETERMINATE,
            "oa.telemetry.content_capture": False,
        }
        with self._span(
            f"invoke_agent {agent_name}", attributes=attributes
        ) as observation:
            stack = self._agent_spans.get()
            token = self._agent_spans.set((*stack, observation.span))
            try:
                yield observation
            finally:
                self._agent_spans.reset(token)

    def task_run(self, task_name: str, declared_tools: list[str]) -> Any:
        return self._span(
            f"execute_task {task_name}",
            attributes={
                "oa.task.name": task_name,
                "oa.task.declared_tools": declared_tools,
            },
        )

    def model_call(self, provider: str, model: str) -> Any:
        return self._span(
            f"chat {model}",
            kind=self._span_kind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model,
            },
        )

    def tool_call(self, name: str, call_id: str | None = None) -> Any:
        return self._span(
            f"execute_tool {name}",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": call_id,
            },
        )

    def sandbox_decision(
        self, *, tool_name: str, result: str, reason: str | None = None
    ) -> None:
        with self._span(
            "oa.sandbox.decision",
            attributes={
                "gen_ai.tool.name": tool_name,
                "oa.sandbox.enabled": True,
                "oa.sandbox.result": result,
                "oa.conformance.reason": reason,
            },
        ) as observation:
            observation.succeed()

    def validation_result(
        self, *, kind: str, result: str, reason: str | None = None
    ) -> None:
        with self._span(
            "oa.validation",
            attributes={
                "oa.validation.kind": kind,
                "oa.validation.result": result,
                "oa.conformance.reason": reason,
            },
        ) as observation:
            observation.succeed()

    def contract_result(
        self, *, enabled: bool, result: str, reason: str | None = None
    ) -> None:
        with self._span(
            "oa.contract.evaluate",
            attributes={
                "oa.contract.enabled": enabled,
                "oa.contract.result": result,
                "oa.conformance.reason": reason,
            },
        ) as observation:
            observation.succeed()

    def conformance(self, status: str, reason: str | None = None) -> None:
        attributes = _clean_attributes(
            {
                "oa.conformance.status": status,
                "oa.conformance.reason": reason,
            }
        )
        current = self._trace.get_current_span()
        spans = (current, *self._agent_spans.get())
        for span in spans:
            current_status = span.attributes.get("oa.conformance.status")
            if not _should_update_conformance(current_status, status):
                continue
            for key, value in attributes.items():
                span.set_attribute(key, value)

    def close(self) -> None:
        if self._owned_provider is not None:
            self._owned_provider.force_flush()
            self._owned_provider.shutdown()


def create_telemetry(enabled: bool) -> TelemetryAdapter:
    """Create the explicitly requested adapter, or return the shared no-op."""

    if not enabled:
        return NOOP_TELEMETRY
    return OpenTelemetryAdapter()
