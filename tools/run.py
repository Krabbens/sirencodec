#!/usr/bin/env python3
"""Dispatch historical SirenCodec trainers and tools from one entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULES = {
    "train": "sirencodec.core.train",
    "train_pipeline": "sirencodec.core.train_pipeline",
    "train_vocos_vq": "sirencodec.core.train_vocos_vq",
}
TOOLS = {
    "bench_fps": ROOT / "tools" / "bench_fps.py",
    "diffusion_mel_demo": ROOT / "tools" / "diffusion_mel_demo.py",
    "grid_search_mlx": ROOT / "tools" / "grid_search_mlx.py",
    "infer_mlx": ROOT / "tools" / "infer_mlx.py",
    "train_mlx": ROOT / "tools" / "train_mlx.py",
}


def main() -> None:
    if len(sys.argv) < 2:
        commands = " | ".join([*MODULES, *TOOLS])
        print(
            f"Usage: python tools/run.py <command> [args...]\n  {commands}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    command = sys.argv.pop(1)
    if command in MODULES:
        runpy.run_module(MODULES[command], run_name="__main__")
        return
    if command in TOOLS:
        runpy.run_path(str(TOOLS[command]), run_name="__main__")
        return
    print(f"Unknown command: {command}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
