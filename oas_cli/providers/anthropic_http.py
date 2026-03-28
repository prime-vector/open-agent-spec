"""Anthropic provider — raw HTTP, no SDK dependency."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import IntelligenceProvider, ProviderError

_DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(IntelligenceProvider):
    """Calls Anthropic Claude via raw HTTP — no anthropic SDK required.

    Reads ANTHROPIC_API_KEY from the environment (or api_key_env in the spec).
    """

    def invoke(self, *, system: str, user: str, config: dict) -> str:
        api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(
                f"Anthropic API key not set — export {api_key_env} or set "
                "intelligence.config.api_key_env in your spec."
            )

        endpoint = config.get("endpoint", _DEFAULT_ENDPOINT)
        if not endpoint.endswith("/messages"):
            endpoint = endpoint.rstrip("/") + "/messages"

        model = config.get("model", _DEFAULT_MODEL)
        temperature = float(config.get("temperature", 0.7))
        max_tokens = int(config.get("max_tokens", 1000))

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ProviderError(f"Anthropic HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected Anthropic response shape: {data}"
            ) from exc
