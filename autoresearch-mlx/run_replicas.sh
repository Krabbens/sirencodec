#!/usr/bin/env bash
# 3 seeds × configs from git: baseline 7d81fb5, mid af40b90, best aa2e494
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN_REL="autoresearch-mlx/train.py"
inject_seed() {
  python3 - "$TRAIN_REL" << 'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text()
ins = "    cfg.seed = int(os.environ.get(\"AUTORESEARCH_SEED\", str(cfg.seed)))\n"
if ins in t:
    sys.exit(0)
old = "    cfg = Config()\n"
if old not in t:
    raise SystemExit("expected cfg = Config() line")
t = t.replace(old, old + ins, 1)
p.write_text(t)
PY
}
backup="$(mktemp)"
cp "$TRAIN_REL" "$backup"
cleanup() { cp "$backup" "$TRAIN_REL"; rm -f "$backup"; }
trap cleanup EXIT

for commit in 7d81fb5 af40b90 aa2e494; do
  git show "${commit}:${TRAIN_REL}" > "$TRAIN_REL"
  inject_seed
  for seed in 0 1 2; do
    log="$ROOT/autoresearch-mlx/replica_${commit}_seed${seed}.log"
    echo "=== $commit seed=$seed -> $log" >&2
    ( cd "$ROOT/autoresearch-mlx" && AUTORESEARCH_SEED="$seed" uv run python train.py > "$log" 2>&1 )
  done
done
