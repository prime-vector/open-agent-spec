"""Codex CLI provider — delegates to the existing codex_adapter."""

from __future__ import annotations

from .base import IntelligenceProvider, ProviderError


class CodexProvider(IntelligenceProvider):
    """Routes invocations through the Codex CLI adapter.

    Merges system and user into a single prompt string because the codex adapter
    accepts a flat prompt rather than separate roles.
    """

    def invoke(self, *, system: str, user: str, config: dict) -> str:
        try:
            from oas_cli.adapters import (
                codex_adapter,  # local import avoids circular dep
            )
        except ImportError as exc:
            raise ProviderError("codex_adapter is not available in this environment") from exc

        combined = f"{system}\n\n{user}".strip() if system else user
        result = codex_adapter.invoke(combined, config)
        if isinstance(result, str):
            return result
        # codex_adapter may return a dict — normalise to string for the runner
        import json
        return json.dumps(result)
