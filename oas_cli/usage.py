"""Token-usage normalisation and best-effort cost estimation.

Providers report token counts in different shapes (OpenAI uses
``prompt_tokens`` / ``completion_tokens``; Anthropic uses ``input_tokens`` /
``output_tokens``). This module maps them to one canonical dict and, for models
whose pricing we know, attaches an estimated USD cost.

The pricing table is intentionally small and hand-maintained. When a model is
not listed, cost is reported as ``None`` rather than guessed — a wrong number is
worse than no number, especially when the point is to give finance a figure they
can trust.

Cost is a **pay-as-you-go API list-price estimate** and can be overridden, since
list prices don't reflect subscriptions, committed-use, or negotiated rates.
Rates resolve in layers (first match wins):

1. **Per-spec** — ``intelligence.config.pricing``: either ``{input_per_1m,
   output_per_1m}`` or the string ``"none"`` to disable cost for that spec.
2. **Global** — the ``OA_PRICING`` env var: JSON ``{model_id: {input, output}}``
   that extends/overrides the built-in table, or ``"none"`` to disable globally.
3. **Built-in** — the hand-maintained table below.

Token *counts* are always reported regardless — they are the figure to track
against any plan.
"""

from __future__ import annotations

import json
import os

CanonicalUsage = dict[str, int]

# Sentinels for layered rate resolution.
_DISABLED = object()  # cost estimation explicitly turned off at this layer
_NOT_SET = object()  # no override at this layer — fall through


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
# Matched by longest model-id prefix; unknown models resolve to None (never
# guessed). Entries are deliberately specific: older Opus (4.0/4.1) was priced
# differently, so a broad "claude-opus-4" prefix would misprice them.
_PRICE_PER_M: dict[str, tuple[float, float]] = {
    # OpenAI — GPT-5.x (current generation as of 2026-06)
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.3-codex": (1.75, 14.00),
    # OpenAI — o-series reasoning (legacy; superseded by GPT-5.x)
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    # OpenAI — GPT-4 family (legacy)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    # Anthropic (Claude) — current generation
    "claude-opus-4-5": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
    # xAI
    "grok-3": (3.00, 15.00),
}


def estimate_cost_usd(
    model: str | None,
    usage: CanonicalUsage | None,
    *,
    pricing: object = None,
) -> float | None:
    """Best-effort USD cost for *usage* on *model*, or None when price is unknown.

    *pricing* is the per-spec override (``intelligence.config.pricing``). Rates
    resolve per-spec → ``OA_PRICING`` env → built-in table; any layer may set
    ``"none"`` to disable cost. See the module docstring.
    """
    if not model or not usage:
        return None
    rate = _resolve_rate(model, pricing)
    if rate is None:
        return None
    in_rate, out_rate = rate
    cost = (
        usage["prompt_tokens"] / 1_000_000 * in_rate
        + usage["completion_tokens"] / 1_000_000 * out_rate
    )
    return round(cost, 6)


def _extract_rate(obj: object) -> tuple[float, float] | None:
    """Pull (input, output) rates from a dict, accepting either key style."""
    if not isinstance(obj, dict):
        return None
    inp = obj.get("input_per_1m", obj.get("input"))
    out = obj.get("output_per_1m", obj.get("output"))
    if inp is None or out is None:
        return None
    try:
        return (float(inp), float(out))
    except (TypeError, ValueError):
        return None


def _table_rate(
    model: str, table: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    """Longest model-id prefix match, so ``gpt-4o-mini`` beats ``gpt-4o``."""
    match: str | None = None
    for prefix in table:
        if model.startswith(prefix) and (match is None or len(prefix) > len(match)):
            match = prefix
    return table[match] if match is not None else None


def _spec_rate(pricing: object) -> object:
    """Per-spec layer → (in, out) rate, _DISABLED, or _NOT_SET."""
    if pricing is None:
        return _NOT_SET
    if isinstance(pricing, str):
        return _DISABLED if pricing.strip().lower() == "none" else _NOT_SET
    rate = _extract_rate(pricing)
    return rate if rate is not None else _NOT_SET


def _env_rate(model: str) -> object:
    """OA_PRICING env layer → (in, out) rate, _DISABLED, or _NOT_SET."""
    raw = os.environ.get("OA_PRICING")
    if not raw or not raw.strip():
        return _NOT_SET
    if raw.strip().lower() == "none":
        return _DISABLED
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return _NOT_SET
    if not isinstance(data, dict):
        return _NOT_SET
    table = {k: rate for k, v in data.items() if (rate := _extract_rate(v)) is not None}
    found = _table_rate(model, table) if table else None
    return found if found is not None else _NOT_SET


def _resolve_rate(model: str, pricing: object) -> tuple[float, float] | None:
    """Resolve the effective rate across layers: per-spec → env → built-in."""
    spec = _spec_rate(pricing)
    if spec is _DISABLED:
        return None
    if spec is not _NOT_SET:
        return spec  # type: ignore[return-value]
    env = _env_rate(model)
    if env is _DISABLED:
        return None
    if env is not _NOT_SET:
        return env  # type: ignore[return-value]
    return _table_rate(model, _PRICE_PER_M)
