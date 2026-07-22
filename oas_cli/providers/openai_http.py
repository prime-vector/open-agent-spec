"""OpenAI provider — raw HTTP, no SDK dependency."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from oas_cli.reasoning import normalise_effort, openai_reasoning_params
from oas_cli.tool_providers.base import InvokeResult, ToolCall
from oas_cli.usage import from_openai

from .base import IntelligenceProvider, InvokeOutcome, ProviderError, scrub_secrets

# Default to the Chat Completions API; set intelligence.endpoint in your spec
# to switch to the Responses API (https://api.openai.com/v1/responses) or a
# compatible endpoint (Azure, local proxy, etc.).
_DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# TODO: model names go stale — consider requiring specs to always declare
#       intelligence.model explicitly and dropping this fallback entirely.
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_TIMEOUT = 60


class OpenAIProvider(IntelligenceProvider):
    """Calls OpenAI via raw HTTP — no openai SDK required.

    Reads OPENAI_API_KEY from the environment (or api_key_env in the spec).
    """

    def invoke(
        self,
        *,
        system: str,
        user: str,
        config: dict,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        return self.invoke_verbose(
            system=system, user=user, config=config, history=history
        ).text

    def invoke_verbose(
        self,
        *,
        system: str,
        user: str,
        config: dict,
        history: list[dict[str, Any]] | None = None,
    ) -> InvokeOutcome:
        api_key_env: str | None = config.get("api_key_env", "OPENAI_API_KEY")
        raw = os.environ.get(api_key_env) if api_key_env else None
        api_key: str | None = raw.strip() if raw else raw

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
        timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))

        reasoning_effort = config.get("reasoning_effort")
        if endpoint.endswith("/responses"):
            payload = _build_responses_payload(
                system, user, model, temperature, history, reasoning_effort
            )
        else:
            payload = _build_chat_completions_payload(
                system, user, model, temperature, max_tokens, history, reasoning_effort
            )

        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"

        data = _http_post_raw(endpoint, payload, headers=extra_headers, timeout=timeout)
        return InvokeOutcome(
            text=_extract_text(data, endpoint),
            usage=from_openai(data.get("usage")),
        )

    def supports_tools(self) -> bool:
        return True

    def invoke_with_tools(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: dict,
    ) -> InvokeResult:
        api_key_env: str | None = config.get("api_key_env", "OPENAI_API_KEY")
        raw = os.environ.get(api_key_env) if api_key_env else None
        api_key: str | None = raw.strip() if raw else raw
        if api_key_env and not api_key:
            raise ProviderError(
                f"API key not set — export {api_key_env} or set "
                "intelligence.config.api_key_env in your spec."
            )

        # Tool use is only supported on the Chat Completions endpoint.
        endpoint = config.get("endpoint", _DEFAULT_ENDPOINT)
        if not endpoint.endswith("/chat/completions"):
            endpoint = (
                endpoint.rstrip("/").removesuffix("/responses") + "/chat/completions"
            )

        model = config.get("model", _DEFAULT_MODEL)
        temperature = float(config.get("temperature", 0.7))
        max_tokens = int(config.get("max_tokens", 1000))
        timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))

        all_messages = [{"role": "system", "content": system}, *messages]
        payload: dict[str, Any] = {"model": model, "messages": all_messages}
        _apply_sampling_and_reasoning(
            payload,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=config.get("reasoning_effort"),
            responses_api=False,
        )
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        extra_headers: dict[str, str] = {}
        if api_key:
            extra_headers["Authorization"] = f"Bearer {api_key}"

        data = _http_post_raw(endpoint, payload, headers=extra_headers, timeout=timeout)

        usage = from_openai(data.get("usage"))
        message = data.get("choices", [{}])[0].get("message", {})
        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "stop")

        if finish_reason == "tool_calls":
            raw_calls = message.get("tool_calls") or []
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"].get("arguments", "{}")),
                )
                for tc in raw_calls
            ]
            return InvokeResult(is_final=False, tool_calls=tool_calls, usage=usage)

        text = message.get("content") or ""
        return InvokeResult(is_final=True, text=text, usage=usage)


def _apply_sampling_and_reasoning(
    payload: dict[str, Any],
    *,
    max_tokens: int | None,
    temperature: float,
    reasoning_effort: object,
    responses_api: bool,
) -> None:
    """Set token-limit, temperature and reasoning params on *payload*, in place.

    Reasoning models (those given a ``reasoning_effort``) diverge from standard
    chat models on the OpenAI API: Chat Completions requires
    ``max_completion_tokens`` (``max_tokens`` is rejected), and a non-default
    ``temperature`` is rejected — so it is omitted. Standard models and
    OpenAI-compatible servers (grok / local / cortex) keep ``max_tokens`` +
    ``temperature`` for maximum compatibility.
    """
    effort = normalise_effort(reasoning_effort)
    if effort is None:
        if not responses_api and max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature
    elif not responses_api and max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    payload.update(
        openai_reasoning_params(reasoning_effort, responses_api=responses_api)
    )


def _build_chat_completions_payload(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    history: list[dict[str, Any]] | None = None,
    reasoning_effort: object = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    payload: dict[str, Any] = {"model": model, "messages": messages}
    _apply_sampling_and_reasoning(
        payload,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        responses_api=False,
    )
    return payload


def _build_responses_payload(
    system: str,
    user: str,
    model: str,
    temperature: float,
    history: list[dict[str, Any]] | None = None,
    reasoning_effort: object = None,
) -> dict[str, Any]:
    turns: list[dict[str, Any]] = [{"role": "system", "content": system}]
    if history:
        turns.extend(history)
    turns.append({"role": "user", "content": user})
    payload: dict[str, Any] = {"model": model, "input": turns}
    _apply_sampling_and_reasoning(
        payload,
        max_tokens=None,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        responses_api=True,
    )
    return payload


def _http_post_raw(
    url: str, payload: dict, headers: dict, timeout: int = _DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """POST to *url* and return the parsed JSON response as a dict."""
    all_headers = {
        "Content-Type": "application/json",
        **headers,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=all_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise ProviderError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        msg = scrub_secrets(str(exc), all_headers)
        raise ProviderError(f"OpenAI request failed: {msg}") from exc


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
