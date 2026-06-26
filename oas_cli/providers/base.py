"""Base class and exception types for OA intelligence providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oas_cli.tool_providers.base import InvokeResult


@dataclass
class InvokeOutcome:
    """A provider invocation result: the raw text plus optional token usage.

    ``usage`` is the canonical token-count dict
    ``{"prompt_tokens", "completion_tokens", "total_tokens"}`` when the provider
    can report it, else ``None`` — e.g. local servers that omit usage, the Codex
    CLI, or custom router classes that only return text.
    """

    text: str
    usage: dict[str, int] | None = None


class ProviderError(RuntimeError):
    """Raised when a provider call fails (HTTP error, bad response shape, etc.)."""


class EngineNotSupportedError(ProviderError):
    """Raised when no provider is registered for the requested engine."""


class IntelligenceProvider(ABC):
    """Minimal interface every provider must implement.

    OA handles prompt construction, task orchestration, and output parsing.
    A provider's only job is: take (system, user, config) → return raw string.
    """

    @abstractmethod
    def invoke(
        self,
        *,
        system: str,
        user: str,
        config: dict,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Invoke the model and return the raw text response.

        Args:
            system:  System prompt.
            user:    User message (template already rendered).
            config:  Provider config dict from intelligence_config (model, temperature, …).
            history: Optional prior-turn messages in OpenAI wire format
                     ``[{"role": "user"|"assistant", "content": "…"}, …]``.
                     Injected between the system message and the current user turn.
                     OA never stores history — callers pass it in via ``input.history``.

        Returns:
            Raw string output from the model — no parsing.

        Raises:
            ProviderError: On any HTTP or response-shape failure.
        """

    def invoke_verbose(
        self,
        *,
        system: str,
        user: str,
        config: dict,
        history: list[dict[str, Any]] | None = None,
    ) -> InvokeOutcome:
        """Invoke the model and additionally report token usage when available.

        The default implementation delegates to :meth:`invoke` and reports no
        usage, so providers that only return text (custom routers, the Codex
        CLI, third-party subclasses) keep working unchanged. HTTP providers
        override this to capture the API ``usage`` object.
        """
        text = self.invoke(system=system, user=user, config=config, history=history)
        return InvokeOutcome(text=text, usage=None)

    def supports_tools(self) -> bool:
        """Return True if this provider natively supports multi-turn tool calling."""
        return False

    def invoke_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: dict,
    ) -> InvokeResult:
        """Single-turn invocation supporting tool use (function calling).

        Args:
            system:   System prompt.
            messages: Full conversation history in OpenAI message format.
            tools:    OpenAI-format tool definitions (``[{"type": "function", …}]``).
            config:   Provider config dict.

        Returns:
            ``InvokeResult`` — either ``is_final=True`` with ``text``, or
            ``is_final=False`` with ``tool_calls`` to execute.

        Raises:
            ProviderError: On transport or response-shape failure.
            NotImplementedError: If the provider does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool use. "
            "Override invoke_with_tools() or use a provider that supports it."
        )
