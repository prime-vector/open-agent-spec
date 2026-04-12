#!/usr/bin/env bash
# Registry demo — pulls two shared specs from the OA registry and runs them
# as a two-step pipeline (summarise → sentiment_of_summary).
#
# Prerequisites:
#   export OPENAI_API_KEY=sk-...
#   pip install open-agent-spec   (or: pip install -e ../../)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1: summarise (via oa://prime-vector/summariser) ==="
oa run --spec pipeline.yaml --task summarise --input input.json | python3 -m json.tool

echo ""
echo "=== Full pipeline: summarise → sentiment_of_summary ==="
oa run --spec pipeline.yaml --task sentiment_of_summary --input input.json | python3 -m json.tool
