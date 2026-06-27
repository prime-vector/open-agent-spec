"""Reasoning-effort tiers — map a portable ``low|medium|high`` control onto each
provider's native reasoning knob.

A spec declares one portable value (``intelligence.config.reasoning_effort``) and
OA translates it per engine, so cost and quality can be dialled to a task's
difficulty without rewriting the spec per provider:

- OpenAI / Codex expose reasoning effort as ``low``/``medium``/``high`` directly
  (an API parameter, or a CLI ``-c`` override for Codex).
- Current Claude models (Opus 4.5+, Sonnet 4.6, Fable 5) expose the same tier
  via ``output_config.effort`` — a near 1:1 mapping. Applied in the Anthropic
  provider (``anthropic_http._apply_reasoning``), which also pairs it with
  adaptive thinking and drops the now-rejected ``temperature`` field.

Status: **spike.** Mappings target the current OpenAI / Codex / Opus-4.x line and
are exercised by ``scripts/verify_reasoning.py`` against live models. Older Claude
models (Sonnet 4.5, Haiku 4.5) do not support ``effort`` — the spec author opts in
by pairing ``reasoning_effort`` with a capable model/engine.
"""

from __future__ import annotations

from typing import Any

VALID_EFFORTS = ("low", "medium", "high")


def normalise_effort(value: object) -> str | None:
    """Return the lower-cased effort tier, or None when unset.

    Raises ``ValueError`` for a non-empty value outside ``low|medium|high`` so a
    typo fails loudly rather than silently disabling reasoning. (``oa validate``
    also catches this via the schema enum, before any model call.)
    """
    if value is None:
        return None
    effort = str(value).strip().lower()
    if not effort:
        return None
    if effort not in VALID_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {VALID_EFFORTS}, got {value!r}"
        )
    return effort


def openai_reasoning_params(value: object, *, responses_api: bool) -> dict[str, Any]:
    """Reasoning parameters to merge into an OpenAI-style request payload.

    Chat Completions uses a flat ``reasoning_effort`` field; the Responses API
    nests it under ``reasoning.effort``. Returns ``{}`` when no effort is set.

    Only meaningful for reasoning-capable models — the spec author opts in by
    setting ``reasoning_effort`` and is responsible for pairing it with a model
    (and engine) that accepts it.
    """
    effort = normalise_effort(value)
    if effort is None:
        return {}
    if responses_api:
        return {"reasoning": {"effort": effort}}
    return {"reasoning_effort": effort}


def codex_reasoning_flags(value: object) -> list[str]:
    """Codex CLI ``-c`` override flags for an effort tier, or ``[]`` when unset."""
    effort = normalise_effort(value)
    if effort is None:
        return []
    return ["-c", f"model_reasoning_effort={effort}"]
