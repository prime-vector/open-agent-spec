"""Anthropic provider — raw HTTP, no SDK dependency."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from oas_cli.reasoning import normalise_effort
from oas_cli.tool_providers.base import InvokeResult, ToolCall
from oas_cli.usage import from_anthropic

from .base import IntelligenceProvider, InvokeOutcome, ProviderError

_DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
# TODO: model names go stale — consider requiring specs to always declare
#       intelligence.model explicitly and dropping this fallback entirely.
_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_DEFAULT_TIMEOUT = 60
_ANTHROPIC_VERSION = "2023-06-01"


def _apply_reasoning(payload: dict[str, Any], reasoning_effort: object) -> None:
    """Apply a reasoning-effort tier to an Anthropic request, in place.

    Current Claude models (Opus 4.5+, Sonnet 4.6, Fable 5) expose the tier
    directly via ``output_config.effort`` (GA — no beta header), paired with
    adaptive thinking. The legacy ``thinking.budget_tokens`` control is rejected
    with a 400 on these models, and ``temperature`` is rejected alongside
    adaptive thinking on the latest models — so it is dropped when effort is set.
    No-op when no effort is requested.

    The author opts in by setting ``reasoning_effort``; ``effort`` errors on
    models without effort support (e.g. Sonnet 4.5, Haiku 4.5), so it must be
    paired with a capable model.
    """
    effort = normalise_effort(reasoning_effort)
    if effort is None:
        return
    payload.setdefault("output_config", {})["effort"] = effort
    payload["thinking"] = {"type": "adaptive"}
    payload.pop("temperature", None)


def _extract_text_blocks(data: dict[str, Any]) -> str:
    """Join the text block(s) from an Anthropic message response.

    With extended thinking enabled the content array leads with ``thinking``
    blocks, so the answer is not necessarily ``content[0]``. Prefer blocks
    explicitly typed ``text``; fall back to any block carrying ``text`` (older
    payloads / test doubles) while still skipping thinking blocks.
    """
    blocks = data["content"]
    texts = [
        b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ]
    if not texts:
        texts = [
            b["text"]
            for b in blocks
            if isinstance(b, dict) and "text" in b and b.get("type") != "thinking"
        ]
    if not texts:
        raise KeyError("no text block in Anthropic response content")
    return "\n".join(texts)


class AnthropicProvider(IntelligenceProvider):
    """Calls Anthropic Claude via raw HTTP — no anthropic SDK required.

    Reads ANTHROPIC_API_KEY from the environment (or api_key_env in the spec).
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
        timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))

        messages: list[dict[str, Any]] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        _apply_reasoning(payload, config.get("reasoning_effort"))

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
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ProviderError(f"Anthropic HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        try:
            text = _extract_text_blocks(data)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected Anthropic response shape: {data}") from exc

        return InvokeOutcome(text=text, usage=from_anthropic(data.get("usage")))

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
        timeout = int(config.get("timeout", _DEFAULT_TIMEOUT))

        # Convert OpenAI-format tool defs to Anthropic format.
        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if anthropic_tools:
            payload["tools"] = anthropic_tools
        _apply_reasoning(payload, config.get("reasoning_effort"))

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
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ProviderError(f"Anthropic HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        stop_reason = data.get("stop_reason")
        if stop_reason == "tool_use":
            tool_calls = [
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                )
                for block in data.get("content", [])
                if block.get("type") == "tool_use"
            ]
            return InvokeResult(is_final=False, tool_calls=tool_calls)

        # Extract text from the final response.
        text_blocks = [
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        ]
        return InvokeResult(is_final=True, text="\n".join(text_blocks))
