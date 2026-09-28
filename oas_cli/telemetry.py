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
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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

    def prompt_override(self, kind: str) -> None: ...

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

    def prompt_override(self, kind: str) -> None:
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


_CONFORMANCE_RANK = {
    INDETERMINATE: 0,
    CONFORMANT: 1,
    VIOLATION_PREVENTED: 2,
    NONCONFORMANT: 3,
}


def _should_update_conformance(
    current: str | None, current_reason: str | None, new: str
) -> bool:
    """Prefer the most specific outcome recorded for an agent execution."""

    if current is None:
        return True
    if current == INDETERMINATE and current_reason and new == CONFORMANT:
        return False
    if current == CONFORMANT and new == INDETERMINATE and current_reason:
        return True
    return _CONFORMANCE_RANK.get(new, 0) >= _CONFORMANCE_RANK.get(current, 0)


@dataclass
class _AgentState:
    span: Any
    status: str = INDETERMINATE
    reason: str | None = None


@dataclass
class _OtelObservation:
    span: Any
    status_type: Any
    status_code: Any
    finished: bool = False

    def succeed(self, **metadata: Any) -> None:
        try:
            for key, value in _clean_attributes(metadata).items():
                self.span.set_attribute(key, value)
            self.span.set_status(self.status_type(self.status_code.OK))
        except Exception:
            # Telemetry must never alter OA execution semantics.
            pass
        self.finished = True

    def fail(self, error: BaseException, **metadata: Any) -> None:
        safe_metadata = {"error.type": type(error).__name__, **metadata}
        try:
            for key, value in _clean_attributes(safe_metadata).items():
                self.span.set_attribute(key, value)
            self.span.record_exception(error)
            self.span.set_status(self.status_type(self.status_code.ERROR, str(error)))
        except Exception:
            # Telemetry must never alter OA execution semantics.
            pass
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
        try:
            self._instrumentation_version = version("open-agent-spec")
        except PackageNotFoundError:  # pragma: no cover - source checkout
            self._instrumentation_version = "unknown"

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
        self._tracer = tracer_provider.get_tracer(
            "open-agent-spec", self._instrumentation_version
        )
        self._agent_spans: ContextVar[tuple[Any, ...]] = ContextVar(
            "oa_telemetry_agent_spans", default=()
        )
        self._agent_states: ContextVar[tuple[_AgentState, ...]] = ContextVar(
            "oa_telemetry_agent_states", default=()
        )

    @contextmanager
    def _span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: Any = None,
    ) -> Iterator[_OtelObservation | _NoOpObservation]:
        try:
            span_context = self._tracer.start_as_current_span(
                name,
                kind=kind if kind is not None else self._span_kind.INTERNAL,
                attributes=_clean_attributes(attributes or {}),
            )
            span = span_context.__enter__()
        except Exception:
            # A broken or unavailable tracer is equivalent to disabled telemetry.
            yield _NoOpObservation()
            return

        observation = _OtelObservation(
            span=span,
            status_type=self._status,
            status_code=self._status_code,
        )
        error_info: tuple[Any, Any, Any] = (None, None, None)
        try:
            yield observation
        except Exception as exc:
            error_info = sys.exc_info()
            if not observation.finished:
                observation.fail(exc)
            raise
        finally:
            try:
                span_context.__exit__(*error_info)
            except Exception:
                # Export/finalisation failures must not change OA semantics.
                pass

    @contextmanager
    def agent_run(self, spec: dict[str, Any], task_name: str) -> Iterator[Observation]:
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
            "oa.prompt.override": "none",
            "oa.telemetry.content_capture": False,
        }
        with self._span(
            f"invoke_agent {agent_name}", attributes=attributes
        ) as observation:
            observation_span = getattr(observation, "span", None)
            stack = self._agent_spans.get()
            token = self._agent_spans.set((*stack, observation_span))
            state_stack = self._agent_states.get()
            state_token = self._agent_states.set(
                (*state_stack, _AgentState(observation_span))
            )
            try:
                yield observation
            finally:
                self._agent_spans.reset(token)
                self._agent_states.reset(state_token)

    def task_run(self, task_name: str, declared_tools: list[str]) -> Any:
        return self._span(
            f"execute_task {task_name}",
            attributes={
                "oa.task.name": task_name,
                "oa.task.declared_tools": declared_tools,
            },
        )

    def model_call(self, provider: str, model: str) -> Any:
        provider_alias = str(provider).strip().lower()
        standard_provider = {
            "anthropic": "anthropic",
            "azure": "azure.ai.openai",
            "azure_openai": "azure.ai.openai",
            "gemini": "google",
            "google": "google",
            "grok": "x_ai",
            "openai": "openai",
            "xai": "x_ai",
        }.get(provider_alias, provider_alias)
        return self._span(
            f"chat {model}",
            kind=self._span_kind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": standard_provider,
                "gen_ai.request.model": model,
                "oa.engine.name": provider,
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
        try:
            states = list(self._agent_states.get())
            current = self._trace.get_current_span()
            attributes = _clean_attributes(
                {
                    "oa.conformance.status": status,
                    "oa.conformance.reason": reason,
                }
            )
            updated = False
            if status in {NONCONFORMANT, VIOLATION_PREVENTED} or (
                status == INDETERMINATE and reason
            ):
                targets = states
            else:
                targets = states[-1:] if states else []
            for state in targets:
                if not _should_update_conformance(state.status, state.reason, status):
                    continue
                state.status = status
                state.reason = reason
                updated = True
                for key, value in attributes.items():
                    state.span.set_attribute(key, value)
            # Preserve the current child observation's evidence as well, while
            # never reading attributes from a possibly non-recording span.
            if updated or not states:
                for key, value in attributes.items():
                    current.set_attribute(key, value)
        except Exception:
            # A sampled-out or otherwise unavailable span must not break a run.
            return None

    def prompt_override(self, kind: str) -> None:
        try:
            for state in self._agent_states.get():
                state.span.set_attribute("oa.prompt.override", kind)
            if not self._agent_states.get():
                self._trace.get_current_span().set_attribute("oa.prompt.override", kind)
        except Exception:
            return None

    def close(self) -> None:
        if self._owned_provider is not None:
            self._owned_provider.force_flush()
            self._owned_provider.shutdown()


def create_telemetry(enabled: bool) -> TelemetryAdapter:
    """Create the explicitly requested adapter, or return the shared no-op."""

    if not enabled:
        return NOOP_TELEMETRY
    return OpenTelemetryAdapter()
