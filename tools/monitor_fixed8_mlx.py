#!/usr/bin/env python3
"""Monitor MLX checkpoints with fixed 8s inference metrics."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sirencodec.eval_metrics import quality_metrics_16k  # noqa: E402


STEP_RE = re.compile(r"codec_step(\d+)\.npz$")


def _read_active_run() -> Path:
    p = Path("/tmp/sirencodec_active_run.txt")
    if not p.is_file():
        raise SystemExit("missing /tmp/sirencodec_active_run.txt")
    s = p.read_text().strip()
    if not s:
        raise SystemExit("empty /tmp/sirencodec_active_run.txt")
    run = Path(s)
    if not run.is_absolute():
        run = ROOT / run
    return run


def _checkpoint_step(path: Path) -> int | None:
    m = STEP_RE.search(path.name)
    return int(m.group(1)) if m else None


def _load_pair_metrics(folder: Path) -> dict[str, float | None]:
    orig = next(folder.glob("*_orig.wav"))
    recon = next(folder.glob("*_recon.wav"))
    ref, sr = sf.read(orig, dtype="float32")
    est, sr2 = sf.read(recon, dtype="float32")
    if sr != 16000 or sr2 != 16000:
        raise RuntimeError(f"expected 16 kHz wavs in {folder}, got {sr}/{sr2}")
    return quality_metrics_16k(ref, est)


def _fmt(v: float | None) -> str:
    return "na" if v is None else f"{float(v):.6f}"


def _beats(candidate: dict[str, float | None], baseline: dict[str, float | None], control: dict[str, float | None]) -> bool:
    higher = ("si_sdr_db", "pesq_wb", "stoi", "cos")
    lower = ("lsd_db", "l1")
    for key in higher:
        cv = candidate.get(key)
        bv = baseline.get(key)
        kv = control.get(key)
        if cv is None or bv is None or kv is None or cv <= max(bv, kv):
            return False
    for key in lower:
        cv = candidate.get(key)
        bv = baseline.get(key)
        kv = control.get(key)
        if cv is None or bv is None or kv is None or cv >= min(bv, kv):
            return False
    return True


def _run_infer(checkpoint: Path, input_wav: Path, out_dir: Path, extra_args: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "infer_mlx.py"),
        str(checkpoint),
        "--input",
        str(input_wav),
        "--out-dir",
        str(out_dir),
        "--max-seconds",
        "8",
        "--self-attention-depth",
        "1",
        "--self-attention-heads",
        "2",
        "--no-save-codes",
    ]
    cmd.extend(extra_args)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--interval", type=float, default=180.0)
    p.add_argument("--once", action="store_true")
    p.add_argument(
        "--fixed-root",
        type=Path,
        default=ROOT / "runs" / "semantic_ab_fixed8_20260513_150315",
    )
    p.add_argument("--input-wav", type=Path, default=None)
    p.add_argument(
        "--infer-extra-args",
        type=str,
        default="",
        help="Extra tools/infer_mlx.py args, e.g. architecture flags needed by the checkpoint.",
    )
    args = p.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _read_active_run()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    fixed_root = args.fixed_root if args.fixed_root.is_absolute() else ROOT / args.fixed_root
    input_wav = args.input_wav or next((fixed_root / "baseline_199999").glob("*_orig.wav"))
    if not input_wav.is_absolute():
        input_wav = ROOT / input_wav
    infer_extra_args = shlex.split(args.infer_extra_args)

    baseline = _load_pair_metrics(fixed_root / "baseline_199999")
    control = _load_pair_metrics(fixed_root / "control_259999")
    eval_dir = run_dir / "fixed8_eval"
    infer_dir = eval_dir / "infer"
    tsv = eval_dir / "metrics.tsv"
    eval_dir.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if tsv.is_file():
        with tsv.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    done.add(int(row["step"]))
                except Exception:
                    pass
    else:
        with tsv.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                [
                    "step",
                    "checkpoint",
                    "si_sdr_db",
                    "pesq_wb",
                    "stoi",
                    "lsd_db",
                    "l1",
                    "cos",
                    "beats_baseline_and_control",
                ]
            )

    print(f"[monitor] run={run_dir}", flush=True)
    print(
        "[monitor] thresholds "
        f"baseline pesq={_fmt(baseline['pesq_wb'])} stoi={_fmt(baseline['stoi'])} cos={_fmt(baseline['cos'])}; "
        f"control pesq={_fmt(control['pesq_wb'])} stoi={_fmt(control['stoi'])} cos={_fmt(control['cos'])}",
        flush=True,
    )

    while True:
        ckpts = []
        for ck in sorted((run_dir / "checkpoints").glob("codec_step*.npz")):
            step = _checkpoint_step(ck)
            if step is not None and step not in done:
                ckpts.append((step, ck))
        for step, ck in ckpts:
            out_dir = infer_dir / f"step_{step:08d}"
            print(f"[monitor] evaluating {ck}", flush=True)
            try:
                _run_infer(ck, input_wav, out_dir, infer_extra_args)
                metrics = _load_pair_metrics(out_dir)
                ok = _beats(metrics, baseline, control)
                with tsv.open("a", newline="") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(
                        [
                            step,
                            str(ck),
                            _fmt(metrics["si_sdr_db"]),
                            _fmt(metrics["pesq_wb"]),
                            _fmt(metrics["stoi"]),
                            _fmt(metrics["lsd_db"]),
                            _fmt(metrics["l1"]),
                            _fmt(metrics["cos"]),
                            "yes" if ok else "no",
                        ]
                    )
                done.add(step)
                print(
                    f"[monitor] step={step} pesq={_fmt(metrics['pesq_wb'])} "
                    f"stoi={_fmt(metrics['stoi'])} cos={_fmt(metrics['cos'])} beats={ok}",
                    flush=True,
                )
                if ok:
                    (eval_dir / "SUCCESS").write_text(f"{step}\n")
            except Exception as exc:
                print(f"[monitor] failed {ck}: {exc}", flush=True)
        if args.once:
            return 0
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
