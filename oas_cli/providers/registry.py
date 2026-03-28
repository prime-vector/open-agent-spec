"""Provider registry — maps engine names to provider instances."""

from __future__ import annotations

import time
from collections.abc import Callable

from .anthropic_http import AnthropicProvider
from .base import EngineNotSupportedError, IntelligenceProvider, ProviderError
from .codex import CodexProvider
from .openai_http import OpenAIProvider


def get_provider(config: dict) -> IntelligenceProvider:
    """Return the provider for the engine named in *config*.

    Args:
        config: Intelligence config dict (keys: engine, model, endpoint, …).

    Raises:
        EngineNotSupportedError: If the engine is not recognised.
    """
    engine = (config.get("engine") or "openai").lower()

    if engine == "openai":
        return OpenAIProvider()
    if engine == "anthropic":
        return AnthropicProvider()
    if engine == "codex":
        return CodexProvider()

    raise EngineNotSupportedError(
        f"Unknown engine '{engine}'. Supported: openai, anthropic, codex."
    )


def invoke_intelligence(system: str, user: str, config: dict) -> str:
    """Select the right provider and call it with retry.

    This is the single call-site that ``oas_cli.runner`` uses. It keeps the
    runner decoupled from every individual provider.

    Args:
        system: System prompt string.
        user:   Rendered user prompt string.
        config: Intelligence config produced by ``_build_intelligence_config``.

    Returns:
        Raw string response from the model.

    Raises:
        ProviderError: Propagated from the underlying provider.
        EngineNotSupportedError: If the engine is not recognised.
    """
    provider = get_provider(config)
    return _with_retry(lambda: provider.invoke(system=system, user=user, config=config))


def _with_retry(fn: Callable[[], str], retries: int = 2, delay: float = 1.0) -> str:
    """Call *fn* up to ``retries + 1`` times, sleeping *delay* s between attempts.

    Only retries on ``ProviderError`` (transient HTTP failures).
    ``EngineNotSupportedError`` and other exceptions are re-raised immediately.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except EngineNotSupportedError:
            raise
        except ProviderError:
            if attempt == retries:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover
