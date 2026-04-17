#!/usr/bin/env bash
# ============================================================
# build_sandbox.sh — Build the Plodder sandbox Docker image
# Run this once before using the sandbox feature.
#
# Usage:
#   bash scripts/build_sandbox.sh
#   bash scripts/build_sandbox.sh --no-cache   (clean rebuild)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="plodder-sandbox:latest"
DOCKERFILE="$REPO_ROOT/Dockerfile.sandbox"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Plodder Sandbox Image Builder          ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "→ Dockerfile : $DOCKERFILE"
echo "→ Image      : $IMAGE_NAME"
echo "→ Context    : $REPO_ROOT"
echo ""

NO_CACHE=""
if [[ "${1:-}" == "--no-cache" ]]; then
  NO_CACHE="--no-cache"
  echo "→ Mode       : Clean rebuild (--no-cache)"
fi

docker build $NO_CACHE \
  -t "$IMAGE_NAME" \
  -f "$DOCKERFILE" \
  "$REPO_ROOT"

echo ""
echo "✅ Build complete: $IMAGE_NAME"
echo ""
echo "To verify:"
echo "  docker run --rm $IMAGE_NAME python3 --version"
echo "  docker run --rm $IMAGE_NAME node --version"
echo ""
