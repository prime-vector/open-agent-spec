"""Token-usage normalisation and best-effort cost estimation.

Providers report token counts in different shapes (OpenAI uses
``prompt_tokens`` / ``completion_tokens``; Anthropic uses ``input_tokens`` /
``output_tokens``). This module maps them to one canonical dict and, for models
whose pricing we know, attaches an estimated USD cost.

The pricing table is intentionally small and hand-maintained. When a model is
not listed, cost is reported as ``None`` rather than guessed — a wrong number is
worse than no number, especially when the point is to give finance a figure they
can trust.
"""

from __future__ import annotations

CanonicalUsage = dict[str, int]


def from_openai(raw: object) -> CanonicalUsage | None:
    """Normalise an OpenAI Chat Completions / Responses ``usage`` object.

    Chat Completions reports ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens``; the Responses API reports ``input_tokens`` /
    ``output_tokens``. Both are handled.
    """
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens", raw.get("input_tokens"))
    completion = raw.get("completion_tokens", raw.get("output_tokens"))
    total = raw.get("total_tokens")
    return _canonical(prompt, completion, total)


def from_anthropic(raw: object) -> CanonicalUsage | None:
    """Normalise an Anthropic ``usage`` object (``input_tokens`` / ``output_tokens``).

    Anthropic does not report a total, so it is derived as input + output.
    """
    if not isinstance(raw, dict):
        return None
    return _canonical(raw.get("input_tokens"), raw.get("output_tokens"), None)


def _canonical(
    prompt: object, completion: object, total: object
) -> CanonicalUsage | None:
    """Build the canonical token dict, or None when nothing usable was reported."""
    if prompt is None and completion is None and total is None:
        return None
    p = int(prompt or 0)
    c = int(completion or 0)
    t = int(total) if total is not None else p + c
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


# USD per 1,000,000 tokens, as (input_rate, output_rate). Hand-maintained and
# best-effort — list prices drift, so treat these as indicative, not billing.
# Matched by longest model-id prefix; unknown models resolve to None.
_PRICE_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "grok-3": (3.00, 15.00),
}


def estimate_cost_usd(model: str | None, usage: CanonicalUsage | None) -> float | None:
    """Best-effort USD cost for *usage* on *model*, or None when price is unknown.

    Matching is by longest model-id prefix so e.g. ``gpt-4o-mini`` wins over
    ``gpt-4o`` for ``gpt-4o-mini-2024-07-18``.
    """
    if not model or not usage:
        return None
    match: str | None = None
    for prefix in _PRICE_PER_M:
        if model.startswith(prefix) and (match is None or len(prefix) > len(match)):
            match = prefix
    if match is None:
        return None
    in_rate, out_rate = _PRICE_PER_M[match]
    cost = (
        usage["prompt_tokens"] / 1_000_000 * in_rate
        + usage["completion_tokens"] / 1_000_000 * out_rate
    )
    return round(cost, 6)
