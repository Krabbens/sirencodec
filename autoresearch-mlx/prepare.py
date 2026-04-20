#!/usr/bin/env python3
"""
Fixed data + validation metric for autoresearch-mlx (sirencodec).

Do not change during experiment loops — see program.md.

``val_bpb`` here is the **mean combined training objective** from ``tools/train_mlx.make_train_fn``
on the validation split (same scale as the scalar you optimize; lower is better). The name is kept
for protocol / results.tsv compatibility with the original LM-style autoresearch docs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Monorepo root (parent of autoresearch-mlx/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from tools.train_mlx import (
    Config,
    MLXCodec,
    _collect_audio_paths,
    _load_audio_batch,
    make_train_fn,
    mel_filterbank_mx,
    synth_batch,
)

CACHE_DIR = Path.home() / ".cache" / "autoresearch"
MANIFEST = CACHE_DIR / "audio_manifest.txt"

# Evaluation: batch count and train/val split (frozen).
VAL_NUM_BATCHES = 10
TRAIN_VAL_SPLIT = 0.9  # fraction of files for training when using disk audio


def repo_data_dir() -> Path:
    return REPO_ROOT / "data"


def write_manifest(paths: list[Path]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(str(p.resolve()) for p in paths) + ("\n" if paths else ""))


def load_manifest() -> list[Path]:
    if not MANIFEST.is_file():
        return []
    lines = [ln.strip() for ln in MANIFEST.read_text().splitlines() if ln.strip()]
    return [Path(ln) for ln in lines]


def discover_audio_paths() -> list[Path]:
    """Local audio only: repo ``data/`` tree, no network."""
    root = repo_data_dir()
    if not root.is_dir():
        return []
    return _collect_audio_paths(root)


def ensure_manifest() -> list[Path]:
    """
    Refresh ``~/.cache/autoresearch/audio_manifest.txt`` from ``<repo>/data`` when missing
    or empty. If no local audio, manifest is empty → training uses synthetic batches.
    """
    paths = load_manifest()
    if paths:
        return paths
    paths = discover_audio_paths()
    write_manifest(paths)
    return paths


def split_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    if not paths:
        return [], []
    n = len(paths)
    if n == 1:
        # One file: same manifest for train/val; loader offsets still vary crops.
        return paths, paths
    n_train = max(1, int(n * TRAIN_VAL_SPLIT))
    if n_train >= n:
        n_train = n - 1
    return paths[:n_train], paths[n_train:]


def evaluate_bpb(
    model: MLXCodec,
    cfg: Config,
    *,
    step: int,
    mel_fb: mx.array | None,
    train_paths: list[Path],
    val_paths: list[Path],
) -> float:
    """
    Mean validation loss (``make_train_fn`` total) over ``VAL_NUM_BATCHES`` batches.
    """
    paths_for_val = val_paths if val_paths else train_paths
    if paths_for_val:
        losses: list[float] = []
        off = 10_000_000
        for _ in range(VAL_NUM_BATCHES):
            batch = _load_audio_batch(cfg, paths_for_val, off)
            off += cfg.batch
            loss_fn = make_train_fn(model, cfg, batch, step, mel_fb)
            loss = loss_fn(model)
            mx.eval(loss)
            losses.append(float(loss.item()))
        return sum(losses) / max(len(losses), 1)

    # Synthetic: deterministic val keys separate from typical train keys
    losses = []
    base = 9_000_000
    for i in range(VAL_NUM_BATCHES):
        batch = synth_batch(cfg, key=base + i * 9973)
        loss_fn = make_train_fn(model, cfg, batch, step, mel_fb)
        loss = loss_fn(model)
        mx.eval(loss)
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    """Regenerate audio manifest from ``<repo>/data`` (optional maintainer step)."""
    paths = discover_audio_paths()
    write_manifest(paths)
    print(f"Wrote {len(paths)} paths to {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
