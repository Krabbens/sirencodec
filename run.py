#!/usr/bin/env python3
"""Single entrypoint: `python run.py <command> [args...]`."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MODULES = {
    "train": "sirencodec.core.train",
    "train_pipeline": "sirencodec.core.train_pipeline",
    "train_vocos_vq": "sirencodec.core.train_vocos_vq",
}

TOOLS = {
    "bench_fps": ROOT / "tools" / "bench_fps.py",
    "bench_lowfps": ROOT / "tools" / "bench_lowfps.py",
    "watch": ROOT / "tools" / "watch.py",
    "precompute_mels": ROOT / "tools" / "precompute_mels.py",
    "diffusion_mel_demo": ROOT / "tools" / "diffusion_mel_demo.py",
    "train_mlx": ROOT / "tools" / "train_mlx.py",
    "infer_mlx": ROOT / "tools" / "infer_mlx.py",
}


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python run.py <command> [args...]\n"
            "  train | train_pipeline | train_vocos_vq — core trainers\n"
            "  sidecars mel_refiner|distill ... — refiner / student vocoder\n"
            "  bench_fps | bench_lowfps | watch | precompute_mels | diffusion_mel_demo | train_mlx | infer_mlx — tools\n",
            file=sys.stderr,
        )
        sys.exit(1)
    cmd = sys.argv.pop(1)
    if cmd in MODULES:
        runpy.run_module(MODULES[cmd], run_name="__main__")
        return
    if cmd == "sidecars":
        from sirencodec.sidecars import main as sc_main

        sc_main()
        return
    if cmd in TOOLS:
        runpy.run_path(str(TOOLS[cmd]), run_name="__main__")
        return
    print(f"Unknown command: {cmd}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
