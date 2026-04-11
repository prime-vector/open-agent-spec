#!/usr/bin/env bash
# Spec Composition demo — three tasks, two shared specialists, zero duplication.
set -euo pipefail

SPEC="examples/spec-composition/coordinator.yaml"
TEXT="Open Agent Spec lets you write multi-step agents as plain YAML. \
Tasks can delegate to shared specialist specs, so you build modular \
pipelines without a framework — just composable, reusable logic."

echo "═══════════════════════════════════════════════════════════"
echo " OA Spec Composition Demo"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "── Task 1: summarise (delegates to shared/summariser.yaml) ──"
oa run --spec "$SPEC" --task summarise \
       --input "{\"text\": \"$TEXT\"}"

echo ""
echo "── Task 2: analyse_sentiment (delegates to shared/sentiment.yaml) ──"
oa run --spec "$SPEC" --task analyse_sentiment \
       --input "{\"text\": \"$TEXT\"}"

echo ""
echo "── Task 3: sentiment_of_summary (data dep on summarise → sentiment, both delegated) ──"
oa run --spec "$SPEC" --task sentiment_of_summary \
       --input "{\"text\": \"$TEXT\"}"
