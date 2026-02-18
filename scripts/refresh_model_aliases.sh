#!/usr/bin/env bash
set -euo pipefail

# Create stable alias symlinks that point to the latest timestamped merged model folders.
# Example:
#   transcripts_on_curriculum -> transcripts_on_curriculum_20260217_165256

ROOT_DIR="${1:-$HOME/mindprint-model/output/merged}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root directory not found: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"

ALIASES=(
  "transcripts_on_curriculum"
  "transcripts"
  "curriculum"
)

updated=0
for alias in "${ALIASES[@]}"; do
  latest="$(ls -dt "${alias}"_* 2>/dev/null | head -1 || true)"
  if [[ -z "$latest" ]]; then
    echo "skip: no timestamped directory found for alias '$alias'"
    continue
  fi

  ln -sfn "$latest" "$alias"
  echo "ok: $alias -> $latest"
  updated=$((updated + 1))
done

echo "done: updated $updated alias link(s) in $ROOT_DIR"
