"""Provider registry — maps engine names to provider instances."""

from __future__ import annotations

import time
from collections.abc import Callable

from .anthropic_http import AnthropicProvider
from .base import EngineNotSupportedError, IntelligenceProvider, ProviderError
from .codex import CodexProvider
from .custom import CustomProvider
from .openai_http import OpenAIProvider

# Per-engine defaults applied before the spec config (spec values always win).
# All engines in this table are routed to OpenAIProvider (OpenAI-compatible HTTP).
_OPENAI_COMPAT_DEFAULTS: dict[str, dict] = {
    # Grok / xAI — OpenAI-compatible endpoint at api.x.ai
    "grok": {
        "endpoint": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "model": "grok-3-latest",
    },
    # xai is an alias for grok (same company / same API)
    "xai": {
        "endpoint": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "model": "grok-3-latest",
    },
    # Local LLMs (Ollama, LM Studio, vLLM, llama.cpp server, …)
    # These expose an OpenAI-compatible API; no key required by default.
    "local": {
        "endpoint": "http://localhost:11434/v1",
        "api_key_env": None,   # no auth for local servers
        "model": "llama3.2",
    },
    # Cortex — user-hosted or enterprise; user must supply endpoint in spec.
    # api_key_env defaults to OPENAI_API_KEY but can be overridden.
    "cortex": {
        "api_key_env": "OPENAI_API_KEY",
    },
    # Custom — falls through to CustomProvider; no defaults needed here.
    "custom": {},
}

# Engines that are fully OpenAI-compatible (no specialised provider class needed)
_OPENAI_COMPAT_ENGINES = frozenset({"grok", "xai", "local", "cortex"})


def get_provider(config: dict) -> IntelligenceProvider:
    """Return the provider for the engine named in *config*.

    Args:
        config: Intelligence config dict (keys: engine, model, endpoint, …).
                Engine-specific defaults have already been merged in by
                ``invoke_intelligence`` before this is called.

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
    if engine in _OPENAI_COMPAT_ENGINES:
        return OpenAIProvider()
    if engine == "custom":
        return CustomProvider()

    raise EngineNotSupportedError(
        f"Unknown engine '{engine}'. "
        "Supported: openai, anthropic, grok, xai, local, cortex, custom, codex."
    )


def invoke_intelligence(system: str, user: str, config: dict) -> str:
    """Select the right provider and call it with retry.

    Engine-specific defaults are applied here so that ``get_provider`` and the
    provider itself always see a fully-resolved config (spec values win over
    defaults — later dict keys win in Python's ``{**defaults, **config}``).

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
    engine = (config.get("engine") or "openai").lower()
    defaults = _OPENAI_COMPAT_DEFAULTS.get(engine, {})
    # Spec values always win: defaults fill gaps, never override explicit config.
    resolved = {**defaults, **config}
    provider = get_provider(resolved)
    return _with_retry(lambda: provider.invoke(system=system, user=user, config=resolved))


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
