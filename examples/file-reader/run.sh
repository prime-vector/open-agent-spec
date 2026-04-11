#!/usr/bin/env bash
# Run the file-reader example.
# Requires OPENAI_API_KEY to be set.
#
# Usage:
#   ./run.sh                         # summarise the bundled notes.txt
#   ./run.sh /path/to/your/file.txt  # summarise any file you like

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="$SCRIPT_DIR/file-reader.yaml"
FILE="${1:-$SCRIPT_DIR/notes.txt}"

echo "==> Spec : $SPEC"
echo "==> File : $FILE"
echo ""

oa run "$SPEC" --task summarise --input "{\"path\": \"$FILE\"}"
