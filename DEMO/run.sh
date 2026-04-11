#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Researcher Agent — Demo Run Script
#
# Each command below maps to a stage in the video walkthrough.
# Run them in order, or jump straight to the full pipeline (Step 3).
#
# Prerequisites:
#   pip install open-agent-spec
#   export OPENAI_API_KEY=sk-...
# ─────────────────────────────────────────────────────────────────────────────

TOPIC="AI agents in software development"
SPEC="DEMO/researcher.yaml"

# ── Step 1: Run a single task ─────────────────────────────────────────────────
# Show that each task works in isolation.
# `plan` takes a topic and returns structured research questions.

echo "=== STEP 1: Plan research questions ==="
oa run \
  --spec  "$SPEC" \
  --task  plan \
  --input "{\"topic\": \"$TOPIC\"}"

echo ""

# ── Step 2: Run the research stage ───────────────────────────────────────────
# OA automatically runs `plan` first because research depends_on: [plan].
# You don't need to do anything — the chain resolves automatically.

echo "=== STEP 2: Research (auto-runs plan first) ==="
oa run \
  --spec  "$SPEC" \
  --task  research \
  --input "{\"topic\": \"$TOPIC\"}"

echo ""

# ── Step 3: Full pipeline — one command ──────────────────────────────────────
# This triggers the complete plan → research → report chain.
# The final output is a markdown report.

echo "=== STEP 3: Full pipeline — plan → research → report ==="
oa run \
  --spec  "$SPEC" \
  --task  report \
  --input "{\"topic\": \"$TOPIC\"}"

echo ""

# ── Step 4: Clean JSON output (for pipelines / scripting) ────────────────────
# --quiet strips all logging; stdout is pure JSON you can pipe or redirect.

echo "=== STEP 4: Quiet mode — pipe-friendly JSON ==="
oa run \
  --spec   "$SPEC" \
  --task   plan \
  --input  "{\"topic\": \"$TOPIC\"}" \
  --quiet

echo ""

# ── Step 5: Try a different topic — zero code changes ────────────────────────
echo "=== STEP 5: Different topic, same agent ==="
oa run \
  --spec  "$SPEC" \
  --task  report \
  --input '{"topic": "impact of remote work on engineering teams"}'
