#!/usr/bin/env bash
# Two-terminal demo of OAS + MCP.
#
# Terminal 1 — start the MCP server:
#   python3 examples/mcp-local/server.py
#
# Terminal 2 — run the agent (this script):
#   ./examples/mcp-local/run.sh
#   ./examples/mcp-local/run.sh "The quick brown fox jumps over the lazy dog"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="$SCRIPT_DIR/agent.yaml"
TEXT="${1:-Hello world, this is a test of the MCP tool integration.}"

echo "==> Spec : $SPEC"
echo "==> Text : $TEXT"
echo ""

oa run "$SPEC" --task analyse --input "{\"text\": \"$TEXT\"}"
