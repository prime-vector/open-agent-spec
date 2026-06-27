#!/usr/bin/env python3
"""Live smoke-test for reasoning-effort tiers + token observability.

Fires the same prompt at ``low`` and ``high`` reasoning effort against each engine
whose API key is present, and prints the captured token usage (and best-effort
cost) for each — so you can confirm the effort knob actually changes spend, not
just that the request is accepted.

This is the "tier 3" check from the reasoning spike: mocked unit tests prove the
wiring; only live calls prove the mapping moves real tokens.

Usage:
    export OPENAI_API_KEY=...          # and/or ANTHROPIC_API_KEY
    python scripts/verify_reasoning.py
    python scripts/verify_reasoning.py --engine anthropic --model claude-opus-4-8
    python scripts/verify_reasoning.py --effort low --effort high --effort medium

Model defaults can be overridden per engine:
    OA_OPENAI_MODEL=o4-mini  OA_ANTHROPIC_MODEL=claude-opus-4-8  ...

Notes:
- Needs network and real API keys; makes a handful of paid calls (cents).
- OpenAI ``reasoning_effort`` requires a *reasoning* model (o-series / GPT-5);
  ``gpt-4o`` will reject it. Set OA_OPENAI_MODEL accordingly.
- Anthropic ``effort`` requires a capable model (Opus 4.5+, Sonnet 4.6, Fable 5).
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from a source checkout without installing.
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)


def _load_dotenv(path: str) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, existing env wins.

    Lets you keep API keys in the gitignored .env instead of exporting them in
    the shell that runs this script.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(os.path.join(_ROOT, ".env"))

from oas_cli.runner import OARunError, run_task_from_spec  # noqa: E402

# A multi-constraint puzzle — enough reasoning that effort should change spend.
# (Single easy questions are too noisy: thinking varies run to run, so use
# --repeat to average if the per-call deltas look like noise.)
QUESTION = (
    "Five houses in a row, numbered 1-5, each a different colour "
    "(red, green, blue, yellow, white). Clues: the red house is somewhere left "
    "of the blue house; the green house is immediately right of the white house; "
    "the yellow house is at one of the two ends; the blue house is not house 5; "
    "house 3 is white. Work through it step by step, then give the colour order "
    "as the final line."
)

ENGINES = [
    {
        "engine": "openai",
        "model": os.getenv("OA_OPENAI_MODEL", "o4-mini"),
        "key": "OPENAI_API_KEY",
        # endpoint omitted — the OpenAI default applies.
    },
    {
        "engine": "anthropic",
        "model": os.getenv("OA_ANTHROPIC_MODEL", "claude-opus-4-8"),
        "key": "ANTHROPIC_API_KEY",
        # No endpoint — exercises the per-engine default (override via env if needed).
        "endpoint": os.getenv("OA_ANTHROPIC_ENDPOINT"),
    },
]


def _build_spec(engine: str, model: str, effort: str, endpoint: str | None) -> dict:
    intelligence: dict = {
        "type": "llm",
        "engine": engine,
        "model": model,
        "config": {"reasoning_effort": effort, "max_tokens": 2000},
    }
    if endpoint:
        intelligence["endpoint"] = endpoint
    return {
        "open_agent_spec": "1.5.0",
        "agent": {"name": "reasoning-probe", "description": "effort smoke-test"},
        "intelligence": intelligence,
        "tasks": {
            "solve": {
                "description": "Solve a small reasoning problem.",
                "response_format": "text",
                "prompts": {
                    "system": "Reason carefully and show your steps.",
                    "user": "{{ question }}",
                },
            }
        },
    }


def _run_one(engine: str, model: str, effort: str, endpoint: str | None) -> dict | None:
    """Run the probe once; return the usage dict (or None on failure)."""
    spec = _build_spec(engine, model, effort, endpoint)
    try:
        result = run_task_from_spec(
            spec, task_name="solve", input_data={"question": QUESTION}
        )
    except OARunError as exc:
        print(f"  {effort:>6} → ERROR [{exc.code}] {exc}")
        return None
    except Exception as exc:
        print(f"  {effort:>6} → ERROR {type(exc).__name__}: {exc}")
        return None

    usage = result.get("usage")
    if not usage:
        print(f"  {effort:>6} → no usage reported (engine omitted it)")
        return None

    cost = usage.get("estimated_cost_usd")
    cost_str = f"  ~${cost:.6f}" if cost is not None else ""
    print(
        f"  {effort:>6} → {usage['total_tokens']:>6} tok "
        f"(in {usage['prompt_tokens']}, out {usage['completion_tokens']}){cost_str}"
    )
    return usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", action="append", help="restrict to engine(s); repeatable"
    )
    parser.add_argument("--model", help="override the model for the chosen engine")
    parser.add_argument(
        "--effort",
        action="append",
        help="effort tier(s) to fire; repeatable (default: low, high)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="samples per tier; >1 averages out adaptive-thinking noise",
    )
    args = parser.parse_args()

    efforts = args.effort or ["low", "high"]
    repeat = max(1, args.repeat)
    engines = ENGINES
    if args.engine:
        wanted = {e.lower() for e in args.engine}
        engines = [e for e in ENGINES if e["engine"] in wanted]
    if args.model and len(engines) == 1:
        engines[0] = {**engines[0], "model": args.model}

    ran_any = False
    for spec in engines:
        engine, model, key_env = spec["engine"], spec["model"], spec["key"]
        endpoint = spec.get("endpoint")
        if not os.getenv(key_env):
            print(f"\n{engine} ({model}): skipped — {key_env} not set")
            continue

        ran_any = True
        print(f"\n{engine} ({model}):")
        means: dict[str, float] = {}
        for eff in efforts:
            samples = [
                u
                for _ in range(repeat)
                if (u := _run_one(engine, model, eff, endpoint)) is not None
            ]
            if samples:
                tot = [s["total_tokens"] for s in samples]
                means[eff] = sum(tot) / len(tot)
                if repeat > 1:
                    print(f"  {eff:>6} → mean {means[eff]:.0f} tok over {len(tot)}")

        # Headline: did higher effort actually cost more tokens (on the mean)?
        if "low" in means and "high" in means:
            delta = means["high"] - means["low"]
            verdict = "✓ effort moves tokens" if delta > 0 else "⚠ no increase"
            suffix = "" if repeat > 1 else " (n=1 — noisy; use --repeat)"
            print(f"  delta: high - low = {delta:+.0f} tok  [{verdict}]{suffix}")

    if not ran_any:
        print(
            "\nNo API keys set. Export OPENAI_API_KEY and/or ANTHROPIC_API_KEY "
            "and re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
