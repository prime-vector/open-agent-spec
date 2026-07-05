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

A pricing override that is **present but invalid** (a negative rate, a malformed
``OA_PRICING`` value, an unrecognised ``pricing`` value) raises
``InvalidPricingError`` rather than silently falling back to the built-in table.
Cost must fail closed, not quietly misstate money. Absent overrides — and models
simply not listed in ``OA_PRICING`` — fall through normally.
"""

from __future__ import annotations

import json
import os

CanonicalUsage = dict[str, int]


class InvalidPricingError(ValueError):
    """Raised when a pricing override is present but malformed or out of range."""


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


def _as_int(value: object) -> int:
    """Coerce a reported token count to int, treating anything non-numeric as 0."""
    return int(value) if isinstance(value, (int, float)) else 0


def _canonical(
    prompt: object, completion: object, total: object
) -> CanonicalUsage | None:
    """Build the canonical token dict, or None when nothing usable was reported."""
    if prompt is None and completion is None and total is None:
        return None
    p = _as_int(prompt)
    c = _as_int(completion)
    t = _as_int(total) if total is not None else p + c
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


def _rate_from_dict(obj: object, source: str) -> tuple[float, float]:
    """Parse a rate dict into (input, output), raising on anything invalid.

    Accepts either key style (``input``/``output`` or ``input_per_1m``/
    ``output_per_1m``). Both keys are required and both rates must be
    non-negative numbers — *source* names the layer for the error message.
    """
    if not isinstance(obj, dict):
        raise InvalidPricingError(
            f"{source} must be an object with input/output rates, got "
            f"{type(obj).__name__}"
        )
    inp = obj.get("input_per_1m", obj.get("input"))
    out = obj.get("output_per_1m", obj.get("output"))
    if inp is None or out is None:
        raise InvalidPricingError(
            f"{source} must set both input (input_per_1m) and output (output_per_1m)"
        )
    try:
        rate = (float(inp), float(out))
    except (TypeError, ValueError):
        raise InvalidPricingError(f"{source} rates must be numbers") from None
    if rate[0] < 0 or rate[1] < 0:
        raise InvalidPricingError(f"{source} rates must be non-negative")
    return rate


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
    """Per-spec layer → (in, out) rate, _DISABLED, or _NOT_SET.

    Raises InvalidPricingError when ``pricing`` is present but invalid, so a typo
    can't silently fall back to the built-in list price.
    """
    if pricing is None:
        return _NOT_SET
    if isinstance(pricing, str):
        if pricing.strip().lower() == "none":
            return _DISABLED
        raise InvalidPricingError(
            "intelligence.config.pricing string must be 'none' (to disable cost) "
            f"or an object with rates, got {pricing!r}"
        )
    return _rate_from_dict(pricing, "intelligence.config.pricing")


def _env_rate(model: str) -> object:
    """OA_PRICING env layer → (in, out) rate, _DISABLED, or _NOT_SET.

    A model simply absent from a valid OA_PRICING map falls through (the env map
    extends the built-in table). But a *malformed* OA_PRICING value raises rather
    than silently reverting to list prices.
    """
    raw = os.environ.get("OA_PRICING")
    if not raw or not raw.strip():
        return _NOT_SET
    if raw.strip().lower() == "none":
        return _DISABLED
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        raise InvalidPricingError(
            "OA_PRICING must be valid JSON ({model: {input, output}}) or 'none'"
        ) from None
    if not isinstance(data, dict):
        raise InvalidPricingError(
            "OA_PRICING must be a JSON object {model: {input, output}} or 'none'"
        )
    # Validate every entry — a bad entry anywhere is operator error, fail closed.
    table = {k: _rate_from_dict(v, f"OA_PRICING[{k!r}]") for k, v in data.items()}
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
