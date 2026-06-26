"""Reasoning-effort tiers — map a portable ``low|medium|high`` control onto each
provider's native reasoning knob.

A spec declares one portable value (``intelligence.config.reasoning_effort``) and
OA translates it per engine, so cost and quality can be dialled to a task's
difficulty without rewriting the spec per provider:

- OpenAI / Codex expose reasoning effort as ``low``/``medium``/``high`` directly
  (an API parameter, or a CLI ``-c`` override for Codex).
- Anthropic has no tier — it takes an extended-thinking *token budget*, so the
  tier is mapped to a budget here.

Status: **spike.** The per-engine values below — especially the Anthropic token
budgets — are starting points to validate against real models (Opus 4.x, Codex),
not tuned production figures. The point is to prove the mapping works end to end.
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


# Anthropic has no tier control — extended thinking takes a token budget. These
# are spike defaults, tunable once validated against Opus 4.x.
_ANTHROPIC_THINKING_BUDGET = {"low": 1024, "medium": 4096, "high": 16384}


def anthropic_thinking_budget(value: object) -> int | None:
    """Extended-thinking token budget for an effort tier, or None when unset."""
    effort = normalise_effort(value)
    if effort is None:
        return None
    return _ANTHROPIC_THINKING_BUDGET[effort]


def codex_reasoning_flags(value: object) -> list[str]:
    """Codex CLI ``-c`` override flags for an effort tier, or ``[]`` when unset."""
    effort = normalise_effort(value)
    if effort is None:
        return []
    return ["-c", f"model_reasoning_effort={effort}"]
