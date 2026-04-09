"""OpenAI provider — raw HTTP, no SDK dependency."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import IntelligenceProvider, ProviderError

# Default to the Chat Completions API; set intelligence.endpoint in your spec
# to switch to the Responses API (https://api.openai.com/v1/responses) or a
# compatible endpoint (Azure, local proxy, etc.).
_DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(IntelligenceProvider):
    """Calls OpenAI via raw HTTP — no openai SDK required.

    Reads OPENAI_API_KEY from the environment (or api_key_env in the spec).
    """

    def invoke(self, *, system: str, user: str, config: dict) -> str:
        api_key_env: str | None = config.get("api_key_env", "OPENAI_API_KEY")
        api_key: str | None = os.environ.get(api_key_env) if api_key_env else None

        # Require a key only when the config explicitly names an env var and it's absent.
        # Local / anonymous endpoints (api_key_env=None or "") skip this check.
        if api_key_env and not api_key:
            raise ProviderError(
                f"API key not set — export {api_key_env} or set "
                "intelligence.config.api_key_env in your spec. "
                "For local/unauthenticated endpoints set api_key_env to null."
            )

        endpoint = config.get("endpoint", _DEFAULT_ENDPOINT)
        # Accept bare base URL (e.g. https://api.openai.com/v1) — append path.
        if not endpoint.endswith("/chat/completions") and not endpoint.endswith(
            "/responses"
        ):
            endpoint = endpoint.rstrip("/") + "/chat/completions"

        model = config.get("model", _DEFAULT_MODEL)
        temperature = float(config.get("temperature", 0.7))
        max_tokens = int(config.get("max_tokens", 1000))

        if endpoint.endswith("/responses"):
            payload = _build_responses_payload(system, user, model, temperature)
        else:
            payload = _build_chat_completions_payload(
                system, user, model, temperature, max_tokens
            )

        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"

        return _http_post(endpoint, payload, headers=extra_headers)


def _build_chat_completions_payload(
    system: str, user: str, model: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _build_responses_payload(
    system: str, user: str, model: str, temperature: float
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }


def _http_post(url: str, payload: dict, headers: dict) -> str:
    all_headers = {
        "Content-Type": "application/json",
        **headers,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=all_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise ProviderError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise ProviderError(f"OpenAI request failed: {exc}") from exc

    return _extract_text(data, url)


def _extract_text(data: dict, url: str) -> str:
    """Extract the text content from a Chat Completions or Responses API response."""
    # Responses API shape: data["output"][0]["content"][0]["text"]
    if "output" in data:
        try:
            return data["output"][0]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected OpenAI Responses API shape: {data}"
            ) from exc

    # Chat Completions shape: data["choices"][0]["message"]["content"]
    if "choices" in data:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected OpenAI Chat Completions shape: {data}"
            ) from exc

    raise ProviderError(f"Unrecognised OpenAI response shape from {url}: {data}")
