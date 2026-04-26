#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sirencodec-sync [branch]

Fetch origin, optionally switch to a branch, pull fast-forward changes, then run
uv sync. Local tracked and untracked edits are stashed first by default.

Environment:
  SIRENCODEC_WORKDIR   repo directory, default /workspace
  SIRENCODEC_STASH     1 to stash local edits before switching/pulling, default 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_dir="${SIRENCODEC_WORKDIR:-/workspace}"
branch="${1:-}"

cd "$repo_dir"

if [[ ! -d .git ]]; then
  echo "ERROR: $repo_dir is not a git checkout. Rebuild the image with .git in the Docker context or mount a git repo at $repo_dir." >&2
  exit 2
fi

git config --global --add safe.directory "$repo_dir" >/dev/null 2>&1 || true

if [[ "${SIRENCODEC_STASH:-1}" == "1" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    git stash push -u -m "sirencodec-sync $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
fi

git fetch --prune origin

if [[ -n "$branch" ]]; then
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git switch "$branch"
  elif git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git switch --track "origin/$branch"
  else
    git switch "$branch"
  fi
fi

git pull --ff-only
uv sync --frozen --python python3

echo "Synced $(git branch --show-current) at $(git rev-parse --short HEAD)"
