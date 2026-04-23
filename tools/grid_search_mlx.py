#!/usr/bin/env python3
"""
Grid search for train_mlx: short runs; default maximize cos_ema from logs.
  Preset ``codebooks``: sweep RVQ layouts (``--cb-layouts``; default ≥32 total runs with 5 LR points) × lr;
  parses ``u0=…/K(…%)`` from logs for ``util_min`` / ``util_mean`` (anti-collapse).
  Use ``--score util_cos`` to rank by utilization first, then cosine.

Sensible parameter scaling (defaults):
  - lr: log-spaced in [lr_min, lr_max] (aggressive range vs single fixed step size).
  - HF bundle: (stft_hf_emphasis, lambda_stft_cos) ramp together — stronger HF weight
    pairs with stronger spectral-cos term (avoids absurd combos like G=2 with λ=0).
  - Optional diagonal mode: zip equal-length lists instead of full Cartesian product.

Does not touch your main mlx_checkpoints / mlx_spectrograms unless you pass --base-dir elsewhere.

Parallelism: ``-j/--jobs`` (default **2** to limit RAM; use ``-j 1`` sequential, ``-j 0`` = auto cap 2).
Optional --parallel-step-scale divides step count by concurrent jobs. Default is 2000 steps per run.

Speed defaults: --fast on every train_mlx run; default 2000 --steps unless you pass --steps,
or --calibrate (uses --seconds). Optional --min-grid-steps after scaling. Best run uses
(cos_ema_max, cos_max) lexicographic. OMP/BLAS threads per process are capped when jobs>1.

Slightly lower --log-cos-ema-beta than train_mlx (0.99) by default so cos_ema moves in short
logs; override via --log-cos-ema-beta or --fixed.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

COS_EMA_RE = re.compile(r"cos_ema=([\d.]+)%")
# Waveform cos%% on train_mlx lines (avoid matching stft_cos=… which has no %).
COS_WAVE_RE = re.compile(r"\bcos=([\d.]+)%")
# u0=86/128(67.2%) u1=… from train_mlx step lines
VQ_UTIL_RE = re.compile(r"u\d+=(\d+)/(\d+)\([\d.]+%\)")

# Default train_mlx length per grid run (unless --steps / --calibrate).
DEFAULT_GRID_STEPS = 2000

# Optional floor after --parallel-step-scale (0 = off).
MIN_GRID_STEPS = 200


@dataclass(frozen=True)
class TrainArgs:
    """Extra CLI flags passed to train_mlx (only non-None values)."""

    lr: float | None = None
    lr_schedule: str | None = None
    lr_min_ratio: float | None = None
    lr_warmup_steps: int | None = None
    stft_hf_emphasis: float | None = None
    lambda_stft_cos: float | None = None
    lambda_stft: float | None = None
    lambda_cos: float | None = None
    log_cos_ema_beta: float | None = None
    grad_clip: float | None = None
    seed: int | None = None
    n_codebooks: int | None = None
    codebook_size: int | None = None
    codebook_sizes: str | None = None
    vq_beta: float | None = None
    lambda_marginal: float | None = None

    def as_cli(self) -> list[str]:
        out: list[str] = []
        for f in fields(TrainArgs):
            v = getattr(self, f.name)
            if v is None:
                continue
            flag = "--" + f.name.replace("_", "-")
            out.append(flag)
            out.append(str(v))
        return out


def logspace_lr(lo: float, hi: float, n: int) -> list[float]:
    if n < 2:
        return [math.sqrt(lo * hi)] if n == 1 else []
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    return [10 ** (log_lo + (log_hi - log_lo) * i / (n - 1)) for i in range(n)]


def parse_train_metrics(text: str) -> dict[str, float | None]:
    """Parse cos_ema, waveform cos%%, and VQ utilization from train_mlx step lines."""
    ema_vals: list[float] = []
    cos_vals: list[float] = []
    util_snapshots: list[tuple[float, float]] = []
    for line in text.splitlines():
        if "step " not in line or "ms/step" not in line:
            continue
        me = COS_EMA_RE.search(line)
        if me:
            ema_vals.append(float(me.group(1)) / 100.0)
        mc = COS_WAVE_RE.search(line)
        if mc:
            cos_vals.append(float(mc.group(1)) / 100.0)
        matches = VQ_UTIL_RE.findall(line)
        if matches:
            fracs = [int(nu) / float(k) for nu, k in matches]
            util_snapshots.append((min(fracs), sum(fracs) / float(len(fracs))))
    out: dict[str, float | None] = {
        "cos_ema_max": max(ema_vals) if ema_vals else None,
        "cos_ema_last": ema_vals[-1] if ema_vals else None,
        "cos_raw_max": max(cos_vals) if cos_vals else None,
        "cos_raw_last": cos_vals[-1] if cos_vals else None,
        "util_min": util_snapshots[-1][0] if util_snapshots else None,
        "util_mean": util_snapshots[-1][1] if util_snapshots else None,
        "util_min_worst": min(u[0] for u in util_snapshots) if util_snapshots else None,
    }
    return out


def grid_score(row: dict[str, Any], mode: str = "cos") -> tuple[float, ...]:
    """Ranking tuple; larger is better (lexicographic)."""
    ema = row.get("cos_ema_max")
    cr = row.get("cos_raw_max")
    ema_v = ema if ema is not None else -1.0
    cr_v = cr if cr is not None else -1.0
    if mode == "util_cos":
        umin = row.get("util_min")
        umean = row.get("util_mean")
        umin_v = umin if umin is not None else -1.0
        umean_v = umean if umean is not None else -1.0
        return (umin_v, umean_v, ema_v, cr_v)
    return (ema_v, cr_v)


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.2f}%"


def format_row_metrics(row: dict[str, Any]) -> str:
    um = row.get("util_min")
    umean = row.get("util_mean")
    util_s = ""
    if um is not None and umean is not None:
        util_s = f" util_min={100.0 * um:.1f}% util_mean={100.0 * umean:.1f}%"
    return (
        f"cos_ema_max={_fmt_pct(row.get('cos_ema_max'))} "
        f"cos_max={_fmt_pct(row.get('cos_raw_max'))}{util_s}"
    )


def _parse_cb_layouts(s: str) -> list[tuple[int, ...]]:
    """``"32,32,32;64,32,16"`` → list of per-stage K tuples."""
    out: list[tuple[int, ...]] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(x.strip()) for x in chunk.split(",") if x.strip()]
        if not parts:
            raise ValueError(f"empty layout in {chunk!r}")
        for k in parts:
            if k < 2:
                raise ValueError(f"codebook K must be >= 2, got {k}")
        out.append(tuple(parts))
    if not out:
        raise ValueError("no codebook layouts (use ; between layouts)")
    return out


def build_runs_preset(
    preset: str,
    lr_min: float,
    lr_max: float,
    lr_points: int,
    hf_g: Sequence[float],
    hf_lsc: Sequence[float],
    pair_mode: str,
    lr_schedule: str,
) -> list[tuple[str, TrainArgs]]:
    """Return list of (name, TrainArgs)."""
    runs: list[tuple[str, TrainArgs]] = []

    if preset == "lr_only":
        for i, lr in enumerate(logspace_lr(lr_min, lr_max, lr_points)):
            name = f"lr{i:02d}_{lr:.2e}".replace("+", "")
            runs.append(
                (
                    name,
                    TrainArgs(lr=lr, lr_schedule=lr_schedule),
                )
            )
        return runs

    if preset == "lr_hf_diagonal":
        lrs = logspace_lr(lr_min, lr_max, lr_points)
        n = min(len(lrs), len(hf_g), len(hf_lsc))
        for i in range(n):
            name = f"d{i:02d}_lr{lrs[i]:.2e}_g{hf_g[i]}_lsc{hf_lsc[i]}".replace("+", "")
            runs.append(
                (
                    name,
                    TrainArgs(
                        lr=lrs[i],
                        lr_schedule=lr_schedule,
                        stft_hf_emphasis=hf_g[i],
                        lambda_stft_cos=hf_lsc[i],
                    ),
                )
            )
        return runs

    if preset == "lr_hf_product":
        lrs = logspace_lr(lr_min, lr_max, lr_points)
        if pair_mode == "diagonal":
            pairs = list(zip(hf_g, hf_lsc, strict=True))
        else:
            pairs = list(itertools.product(hf_g, hf_lsc))
        for i, lr in enumerate(lrs):
            for j, (g, lsc) in enumerate(pairs):
                name = f"p{i:02d}_{j:02d}_lr{lr:.2e}_g{g}_lsc{lsc}".replace("+", "")
                runs.append(
                    (
                        name,
                        TrainArgs(
                            lr=lr,
                            lr_schedule=lr_schedule,
                            stft_hf_emphasis=g,
                            lambda_stft_cos=lsc,
                        ),
                    )
                )
        return runs

    raise ValueError(f"unknown preset {preset!r}")


def build_runs_codebooks(
    cb_layouts: str,
    lr_min: float,
    lr_max: float,
    lr_points: int,
    lr_schedule: str,
    *,
    vq_beta: float | None,
    lambda_marginal: float | None,
) -> list[tuple[str, TrainArgs]]:
    layouts = _parse_cb_layouts(cb_layouts)
    lrs = logspace_lr(lr_min, lr_max, lr_points)
    runs: list[tuple[str, TrainArgs]] = []
    run_i = 0
    for layout in layouts:
        nq = len(layout)
        sizes_csv = ",".join(str(k) for k in layout)
        ks_tag = "x".join(str(k) for k in layout)
        for lr in lrs:
            name = f"cb{run_i:02d}_nq{nq}_K{ks_tag}_lr{lr:.2e}".replace("+", "")
            runs.append(
                (
                    name,
                    TrainArgs(
                        lr=lr,
                        lr_schedule=lr_schedule,
                        n_codebooks=nq,
                        codebook_sizes=sizes_csv,
                        vq_beta=vq_beta,
                        lambda_marginal=lambda_marginal,
                    ),
                )
            )
            run_i += 1
    return runs


def _subprocess_env(parallel_runs: int) -> dict[str, str]:
    """Avoid BLAS×N jobs blowing up thread count; MLX still uses GPU."""
    env = os.environ.copy()
    if parallel_runs <= 1:
        return env
    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        env[k] = v
    return env


def resolve_jobs(jobs_arg: int, n_runs: int) -> int:
    """``jobs_arg`` > 0: cap at that many workers. ``0``: auto, at most 2 (RAM-friendly on MLX)."""
    if jobs_arg > 0:
        return min(jobs_arg, max(1, n_runs))
    return max(1, min(n_runs, 2))


# Heuristic steps/sec with --fast + typical Libri batch (skip --calibrate to save ~10–30s startup).
def _train_mlx_speed_extras(librispeech: bool, use_fast: bool) -> list[str]:
    xs: list[str] = []
    if librispeech:
        xs.append("--librispeech")
    if use_fast:
        xs.append("--fast")
    return xs


def calibrate_steps(
    python: str,
    train_script: Path,
    wall_seconds: float,
    extra_train_args: list[str],
) -> int:
    """Rough step count for ~wall_seconds (after a short warmup)."""
    warmup = 10
    measure = 50
    calib_dir = REPO_ROOT / "mlx_grid_search" / "_calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(train_script),
        "--steps",
        str(warmup + measure),
        "--spectrogram-every",
        "0",
        "--checkpoint-every",
        "0",
        "--no-save-audio",
        "--log-every",
        "200",
        "--checkpoint-dir",
        str(calib_dir),
        "--spectrogram-dir",
        str(calib_dir / "spectrograms"),
    ]
    cmd.extend(extra_train_args)
    if "--fast" not in extra_train_args:
        cmd.append("--fast")

    t0 = time.perf_counter()
    r = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=_subprocess_env(1),
    )
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(
            "[grid] calibration run failed; fallback to steps=1000\n"
            + (r.stderr or r.stdout or "")[:4000],
            file=sys.stderr,
        )
        return max(200, int(wall_seconds * 8))

    # Skip JIT-ish first part: attribute ~70% of time to measured segment (heuristic).
    t_per_step = elapsed / float(warmup + measure)
    steps = int(wall_seconds / t_per_step * 0.92)
    return max(120, min(steps, 50_000))


def run_one(
    python: str,
    train_script: Path,
    steps: int,
    name: str,
    base_dir: Path,
    train: TrainArgs,
    fixed: list[str],
    parallel_runs: int,
    default_log_cos_ema_beta: float,
    grid_log_every: int,
) -> dict[str, Any]:
    ckpt = base_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(train_script),
        "--steps",
        str(steps),
        "--spectrogram-every",
        "0",
        "--checkpoint-every",
        "0",
        "--no-save-audio",
        "--log-every",
        str(grid_log_every),
        "--checkpoint-dir",
        str(ckpt),
        "--spectrogram-dir",
        str(ckpt / "spectrograms"),
    ]
    train_cli = train.as_cli()
    cmd.extend(train_cli)
    if "--log-cos-ema-beta" not in train_cli and "--log-cos-ema-beta" not in fixed:
        cmd.extend(["--log-cos-ema-beta", str(default_log_cos_ema_beta)])
    cmd.extend(fixed)

    t0 = time.perf_counter()
    r = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=_subprocess_env(parallel_runs),
    )
    wall = time.perf_counter() - t0
    log_blob = (r.stdout or "") + "\n" + (r.stderr or "")
    m = parse_train_metrics(log_blob)

    return {
        "name": name,
        "exit_code": r.returncode,
        "wall_s": round(wall, 2),
        "steps": steps,
        "cos_ema_max": m["cos_ema_max"],
        "cos_ema_last": m["cos_ema_last"],
        "cos_raw_max": m["cos_raw_max"],
        "cos_raw_last": m["cos_raw_last"],
        "train_args": {f.name: getattr(train, f.name) for f in fields(train)},
        "cmd": cmd,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid search train_mlx for max cos_ema (short runs).")
    ap.add_argument(
        "--preset",
        choices=("lr_only", "lr_hf_diagonal", "lr_hf_product", "codebooks"),
        default="lr_hf_diagonal",
        help="lr_only / lr_hf_*: LR(+HF) sweeps. codebooks: --cb-layouts × log-spaced lr (RVQ K arrangements).",
    )
    ap.add_argument(
        "--cb-layouts",
        type=str,
        default=(
            "32,32,32;32,32,64;32,64,32;64,32,32;64,32,16;64,64,32;32,64,64;64,64,64;"
            "48,32,16;48,48,48;96,32,32;96,64,32;128,32,32;128,64,32;32,128,32;"
            "32,128;64,64;64,96;128,64;128,128"
        ),
        metavar="SPEC",
        help='For preset=codebooks: semicolon-separated layouts ("K0,K1,…"); default ~19 layouts × lr (≥32 runs with --lr-points 5).',
    )
    ap.add_argument(
        "--cb-vq-beta",
        type=float,
        default=None,
        metavar="B",
        help="Optional: pass --vq-beta B to every train_mlx run (commitment; try 1.0–1.8 when codebooks collapse).",
    )
    ap.add_argument(
        "--cb-lambda-marginal",
        type=float,
        default=None,
        metavar="L",
        help="Optional: override --lambda-marginal for grid (e.g. 0.5; None = train_mlx default).",
    )
    ap.add_argument(
        "--score",
        choices=("cos", "util_cos"),
        default="cos",
        help="cos: best cos_ema then cos_raw. util_cos: best util_min then util_mean then cos (fight codebook collapse).",
    )
    ap.add_argument("--lr-min", type=float, default=3.5e-4, help="Log grid low (inclusive).")
    ap.add_argument("--lr-max", type=float, default=1.8e-3, help="Log grid high (inclusive).")
    ap.add_argument("--lr-points", type=int, default=5, help="Number of log-spaced lr values.")
    ap.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=("none", "cosine"),
        help="Short runs: 'none' keeps peak lr; cosine decays over --steps.",
    )
    ap.add_argument(
        "--hf-g",
        type=str,
        default="1.0,1.2,1.45,1.7,2.0",
        help="Comma-separated stft_hf_emphasis values (paired with --hf-lambda-stft-cos).",
    )
    ap.add_argument(
        "--hf-lambda-stft-cos",
        type=str,
        default="0.0,0.04,0.08,0.12,0.16",
        help="Comma-separated lambda_stft_cos values; diagonal zip with --hf-g (same length as lr-points for lr_hf_diagonal).",
    )
    ap.add_argument(
        "--pair-mode",
        choices=("full", "diagonal"),
        default="full",
        help="For lr_hf_product only: full Cartesian g×lsc or single diagonal (zip).",
    )
    ap.add_argument("--seconds", type=float, default=60.0, help="Target wall time per run (calibration).")
    ap.add_argument(
        "--log-cos-ema-beta",
        type=float,
        default=0.97,
        metavar="B",
        help="Forwarded to train_mlx when not set in grid TrainArgs/--fixed. Default 0.97: faster EMA than "
        "train_mlx's 0.99 so cos_ema is informative in ~60s runs.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"train_mlx --steps (default {DEFAULT_GRID_STEPS} if not using --calibrate).",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Measure steps/sec once (~few s) to match --seconds (overrides default step count).",
    )
    ap.add_argument(
        "--parallel-step-scale",
        action="store_true",
        help="Divide step count by min(jobs, runs) when jobs>1 (shorter per-job runs; use with --min-grid-steps).",
    )
    ap.add_argument(
        "--no-grid-fast",
        action="store_true",
        help="Do not pass --fast to train_mlx (slower, closer to full STFT stack).",
    )
    ap.add_argument(
        "--grid-log-every",
        type=int,
        default=200,
        metavar="N",
        help="train_mlx --log-every for grid runs; capped vs steps when steps≤800 so short runs log often enough.",
    )
    ap.add_argument(
        "--min-grid-steps",
        type=int,
        default=0,
        metavar="N",
        help="After --parallel-step-scale, bump steps to at least N (e.g. 200) so metrics are not flat. "
        "0 = no floor.",
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        default="mlx_grid_search",
        help="Under repo root; each run uses base-dir/<run_name>/ for checkpoints.",
    )
    ap.add_argument("--librispeech", action="store_true", help="Pass --librispeech to train_mlx.")
    ap.add_argument(
        "--fixed",
        nargs="*",
        default=[],
        metavar="ARG",
        help="Extra train_mlx args, e.g. --fixed --fast --seed 1",
    )
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable used for training subprocesses.")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=2,
        metavar="N",
        help="Parallel train_mlx workers (default 2 — limits RAM; 1=sequential; 0=auto max 2; raise e.g. 4 if you have headroom). "
        "With --parallel-step-scale, steps are divided by concurrent jobs.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--max-runs",
        type=int,
        default=200,
        help="Safety cap (preset codebooks default layouts ×5 lr ≈95; raise if needed).",
    )
    args = ap.parse_args()

    hf_g = [float(x.strip()) for x in args.hf_g.split(",") if x.strip()]
    hf_lsc = [float(x.strip()) for x in args.hf_lambda_stft_cos.split(",") if x.strip()]
    if args.preset == "lr_hf_diagonal" and len(hf_g) != len(hf_lsc):
        print(
            "[grid] lr_hf_diagonal: trimming to min(len(hf_g), len(hf_lsc)); "
            "use equal-length lists for aligned scaling.",
            file=sys.stderr,
        )

    if args.preset == "codebooks":
        try:
            runs = build_runs_codebooks(
                args.cb_layouts,
                args.lr_min,
                args.lr_max,
                args.lr_points,
                args.lr_schedule,
                vq_beta=args.cb_vq_beta,
                lambda_marginal=args.cb_lambda_marginal,
            )
        except ValueError as e:
            print(f"[grid] codebooks: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        runs = build_runs_preset(
            args.preset,
            args.lr_min,
            args.lr_max,
            args.lr_points,
            hf_g,
            hf_lsc,
            args.pair_mode,
            args.lr_schedule,
        )
    if len(runs) > args.max_runs:
        print(f"[grid] {len(runs)} runs > --max-runs {args.max_runs}; abort.", file=sys.stderr)
        sys.exit(1)
    if args.preset == "codebooks" and len(runs) < 32:
        print(
            f"[grid] warning: only {len(runs)} codebook runs (<32). "
            "Add more layouts in --cb-layouts or increase --lr-points.",
            file=sys.stderr,
        )

    train_script = REPO_ROOT / "tools" / "train_mlx.py"
    if not train_script.is_file():
        print(f"[grid] missing {train_script}", file=sys.stderr)
        sys.exit(1)

    jobs = resolve_jobs(args.jobs, len(runs))
    fs = "on" if not args.no_grid_fast else "off"
    print(
        f"[grid] jobs={jobs}"
        + (" (auto, max 2)" if args.jobs <= 0 else "")
        + f"  --fast {fs}  log-cos-ema-beta→{args.log_cos_ema_beta} (skipped if in --fixed)",
        flush=True,
    )

    use_fast = not args.no_grid_fast
    fixed = _train_mlx_speed_extras(args.librispeech, use_fast) + list(args.fixed)

    steps_before_parallel: int | None = None
    steps: int
    if args.steps is not None:
        steps = args.steps
    elif args.calibrate and not args.dry_run:
        steps = calibrate_steps(
            args.python,
            train_script,
            args.seconds,
            _train_mlx_speed_extras(args.librispeech, use_fast),
        )
    else:
        steps = DEFAULT_GRID_STEPS

    if (
        args.steps is None
        and args.parallel_step_scale
        and jobs > 1
    ):
        pw = min(jobs, len(runs))
        if pw > 1:
            steps_before_parallel = steps
            steps = max(80, int(steps / pw))

    steps_before_min_floor: int | None = None
    if args.steps is None and args.min_grid_steps > 0 and steps < args.min_grid_steps:
        steps_before_min_floor = steps
        steps = args.min_grid_steps

    grid_log_every_eff = args.grid_log_every
    if steps <= 800:
        grid_log_every_eff = min(grid_log_every_eff, max(25, max(1, steps // 5)))

    src = (
        "fixed"
        if args.steps is not None
        else ("calibrated" if args.calibrate and not args.dry_run else f"default {DEFAULT_GRID_STEPS}")
    )
    if args.dry_run:
        msg = f"[grid] dry-run: steps={steps}  log-every={grid_log_every_eff}"
        if steps_before_parallel is not None:
            msg += f" (parallel-scaled from {steps_before_parallel})"
        if steps_before_min_floor is not None:
            msg += f" (min-floor from {steps_before_min_floor})"
        print(msg, flush=True)
    else:
        msg = f"[grid] using steps={steps} ({src}"
        if args.calibrate and not args.dry_run:
            msg += f"; ~{args.seconds:.0f}s from calibrate"
        if steps_before_parallel is not None:
            msg += f"; parallel-scaled from {steps_before_parallel} (÷{min(jobs, len(runs))} concurrent)"
        if steps_before_min_floor is not None:
            msg += f"; min-steps floor→{steps} (cold cos/ema need enough steps; --min-grid-steps 0 to skip)"
        msg += f")  log-every={grid_log_every_eff}"
        print(msg, flush=True)

    base_dir = REPO_ROOT / args.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    print_lock = threading.Lock()

    if args.dry_run:
        for name, train in runs:
            print(f"[grid] === {name} ===", flush=True)
            print(" ", train)
    elif jobs == 1:
        for name, train in runs:
            print(f"[grid] === {name} ===", flush=True)
            row = run_one(
                args.python,
                train_script,
                steps,
                name,
                base_dir,
                train,
                fixed,
                parallel_runs=1,
                default_log_cos_ema_beta=args.log_cos_ema_beta,
                grid_log_every=grid_log_every_eff,
            )
            results.append(row)
            print(f"  exit={row['exit_code']} wall={row['wall_s']}s {format_row_metrics(row)}", flush=True)
    else:
        indexed = list(enumerate(runs))

        def _task(item: tuple[int, tuple[str, TrainArgs]]) -> tuple[int, dict[str, Any]]:
            i, (name, train) = item
            row = run_one(
                args.python,
                train_script,
                steps,
                name,
                base_dir,
                train,
                fixed,
                parallel_runs=jobs,
                default_log_cos_ema_beta=args.log_cos_ema_beta,
                grid_log_every=grid_log_every_eff,
            )
            return i, row

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(_task, it): it[0] for it in indexed}
            pending: dict[int, dict[str, Any]] = {}
            for fut in as_completed(futures):
                i, row = fut.result()
                pending[i] = row
                with print_lock:
                    print(
                        f"[grid] done {row['name']} exit={row['exit_code']} wall={row['wall_s']}s "
                        f"{format_row_metrics(row)}",
                        flush=True,
                    )
        results = [pending[i] for i in range(len(runs))]

    if args.dry_run:
        return

    csv_path = base_dir / "grid_results.csv"
    json_path = base_dir / "grid_results.json"
    if results:
        keys = sorted({k for r in results for k in r.keys() if k not in ("cmd", "train_args")})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys + ["train_args_json"])
            w.writeheader()
            for r in results:
                flat = {k: r.get(k) for k in keys}
                flat["train_args_json"] = json.dumps(r.get("train_args"), sort_keys=True)
                w.writerow(flat)
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)

        score_key = lambda r: grid_score(r, args.score)
        best = max(results, key=score_key, default=None)
        print(f"\n[grid] wrote {csv_path} ({len(results)} runs)")
        if best:
            print(
                f"[grid] best score={score_key(best)!r} (--score {args.score})  name={best['name']}  "
                f"{format_row_metrics(best)}  train_args={best['train_args']}"
            )
        else:
            print("[grid] no cos_ema parsed; check train_mlx logs / --log-cos-ema-beta")


if __name__ == "__main__":
    main()
