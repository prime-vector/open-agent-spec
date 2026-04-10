"""Base class and exception types for OAS intelligence providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """Raised when a provider call fails (HTTP error, bad response shape, etc.)."""


class EngineNotSupportedError(ProviderError):
    """Raised when no provider is registered for the requested engine."""


class IntelligenceProvider(ABC):
    """Minimal interface every provider must implement.

    OAS handles prompt construction, task orchestration, and output parsing.
    A provider's only job is: take (system, user, config) → return raw string.
    """

    @abstractmethod
    def invoke(self, *, system: str, user: str, config: dict) -> str:
        """Invoke the model and return the raw text response.

        Args:
            system: System prompt.
            user:   User message (template already rendered).
            config: Provider config dict from intelligence_config (model, temperature, …).

        Returns:
            Raw string output from the model — no parsing.

        Raises:
            ProviderError: On any HTTP or response-shape failure.
        """
