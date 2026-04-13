"""Custom provider — supports user-supplied Python router classes.

When ``intelligence.module`` is set in the spec, the named class is
dynamically imported and called.  The expected interface is:

    class MyRouter:
        def __init__(self, endpoint: str, model: str, config: dict): ...
        def run(self, prompt: str, **kwargs) -> str: ...   # returns JSON string

When ``module`` is absent the provider falls back to the ``OpenAIProvider``
(OpenAI-compatible HTTP), so a bare ``engine: custom`` with only an
``endpoint`` works without any Python glue code.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import IntelligenceProvider, ProviderError


class CustomProvider(IntelligenceProvider):
    """Routes to a user-supplied Python class, or falls back to HTTP."""

    def invoke(
        self,
        *,
        system: str,
        user: str,
        config: dict,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        module_path: str | None = config.get("module")

        if not module_path:
            # No module specified — behave as a plain OpenAI-compatible endpoint.
            from .openai_http import OpenAIProvider

            return OpenAIProvider().invoke(
                system=system, user=user, config=config, history=history
            )

        return self._invoke_class(module_path, system, user, config)

    @staticmethod
    def _invoke_class(module_path: str, system: str, user: str, config: dict) -> str:
        """Dynamically load ``module_path`` and call ``run()`` on an instance."""
        if "." not in module_path:
            raise ProviderError(
                f"intelligence.module must be 'module.ClassName', got: '{module_path}'"
            )

        module_name, class_name = module_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            raise ProviderError(
                f"Cannot import module '{module_name}' (from intelligence.module={module_path!r}): {exc}"
            ) from exc

        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ProviderError(
                f"Class '{class_name}' not found in module '{module_name}'"
            )

        endpoint = config.get("endpoint", "")
        model = config.get("model", "")
        try:
            router = cls(endpoint=endpoint, model=model, config=config)
        except Exception as exc:
            raise ProviderError(
                f"Failed to instantiate {module_path}(endpoint, model, config): {exc}"
            ) from exc

        # The legacy interface takes a single merged prompt string.
        prompt = f"{system}\n\n{user}".strip() if system else user
        try:
            result = router.run(prompt)
        except Exception as exc:
            raise ProviderError(f"{module_path}.run() raised: {exc}") from exc

        if not isinstance(result, str):
            import json

            try:
                return json.dumps(result)
            except Exception as exc:
                raise ProviderError(
                    f"{module_path}.run() returned non-string, non-serialisable value: {type(result)}"
                ) from exc
        return result
