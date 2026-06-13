#!/usr/bin/env python3
"""Benchmark PyTorch SirenCodec inference on a deterministic dataset subset.

The benchmark loads and normalizes the selected audio files first, then measures
model forward time only.  This matches the C++ LiteRT dataset benchmark and
keeps disk I/O out of the timing table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback for importability.
    resource = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sirencodec.config import Config, nominal_rvq_kbps  # noqa: E402
from sirencodec.cuda.codec import CUDACodec  # noqa: E402
from sirencodec.eval_metrics import quality_metrics_16k  # noqa: E402


def select_subset(data_root: Path, fraction: float) -> list[Path]:
    files = sorted(data_root.rglob("*.flac"))
    if not files:
        raise RuntimeError(f"no .flac files under {data_root}")
    n = max(1, round(len(files) * fraction))
    ranked = sorted(
        files,
        key=lambda p: hashlib.md5(str(p.relative_to(data_root)).replace("\\", "/").encode("utf-8")).hexdigest(),
    )
    return ranked[:n]


def write_manifest(paths: list[Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[Path]:
    files = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not files:
        raise RuntimeError(f"empty manifest: {path}")
    return files


def load_audio_mono(path: Path, target_sr: int, samples: int) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if int(sr) != int(target_sr):
        n_new = max(1, int(round(wav.size * target_sr / sr)))
        wav = np.interp(np.linspace(0, wav.size - 1, num=n_new), np.arange(wav.size), wav).astype(np.float32)
    peak = max(float(np.max(np.abs(wav))) if wav.size else 0.0, 1e-5)
    wav = (wav / peak).astype(np.float32)
    if wav.shape[0] >= samples:
        return wav[:samples].copy()
    out = np.zeros((samples,), dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def load_checkpoint(path: Path, device: torch.device) -> tuple[CUDACodec, Config]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg_blob = dict(blob["config"])
    if isinstance(cfg_blob.get("data_dir"), str):
        cfg_blob["data_dir"] = Path(cfg_blob["data_dir"])
    cfg = Config(**cfg_blob)
    model = CUDACodec(cfg).to(device)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    return model, cfg


def summarize(values: list[float], audio_seconds: float) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("empty timing list")
    ordered = sorted(values)

    def percentile(p: float) -> float:
        idx = int(round(p * (len(ordered) - 1)))
        return ordered[idx]

    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_seconds": mean,
        "min_seconds": ordered[0],
        "p50_seconds": percentile(0.50),
        "p90_seconds": percentile(0.90),
        "max_seconds": ordered[-1],
        "x_realtime": audio_seconds / mean,
    }


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_call(device: torch.device, fn) -> tuple[Any, float]:
    sync_if_needed(device)
    start = time.perf_counter()
    out = fn()
    sync_if_needed(device)
    return out, time.perf_counter() - start


def encode_quantized(model: CUDACodec, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    z = model.latent_before_rvq(x)
    if model.cfg.ae_only:
        return z, None
    z_q, _vq_loss, _ent_pos, _marg_ent, indices = model.rvq(z)
    return z_q, indices


def decode_quantized(model: CUDACodec, z_q: torch.Tensor, tlen: int) -> torch.Tensor:
    if model.latent_post is not None:
        z_q = model.latent_post(z_q)
    if model.self_attn_post is not None:
        z_q = model.self_attn_post(z_q)
    return model.decoder(z_q, tlen)


def current_rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    return None


def peak_rss_mib() -> float | None:
    if resource is None:
        return current_rss_mib()
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / 1024.0 / 1024.0
    return peak / 1024.0


def memory_snapshot(label: str, device: torch.device) -> dict[str, float | str | None]:
    snap: dict[str, float | str | None] = {
        "label": label,
        "rss_mib": current_rss_mib(),
        "rss_peak_mib": peak_rss_mib(),
    }
    if device.type == "cuda":
        snap.update(
            {
                "cuda_allocated_mib": torch.cuda.memory_allocated(device) / 1024.0 / 1024.0,
                "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 1024.0 / 1024.0,
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0,
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0,
            }
        )
    return snap


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("data/train-clean-100"))
    p.add_argument("--fraction", type=float, default=0.10, help="Fraction of --data-root to benchmark; default is 0.10")
    p.add_argument("--manifest", type=Path, default=None, help="Use existing newline-delimited file list")
    p.add_argument("--manifest-out", type=Path, default=None, help="Write selected file list")
    p.add_argument("--samples", type=int, default=32000, help="Fixed samples per file; default is 32000, i.e. 2 s at 16 kHz")
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--warmup-files", type=int, default=16)
    p.add_argument("--quality-files", type=int, default=32)
    p.add_argument("-o", "--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit("--fraction must be in (0, 1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    device = torch.device(args.device)
    memory_snapshots: list[dict[str, float | str | None]] = [memory_snapshot("start", device)]
    files = read_manifest(args.manifest) if args.manifest is not None else select_subset(args.data_root, args.fraction)
    if args.manifest_out is not None:
        write_manifest(files, args.manifest_out)

    model, cfg = load_checkpoint(args.checkpoint.resolve(), device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    memory_snapshots.append(memory_snapshot("after_model_load", device))
    audio = [load_audio_mono(path, cfg.sample_rate, args.samples) for path in files]
    memory_snapshots.append(memory_snapshot("after_audio_preload_cpu", device))

    with torch.inference_mode():
        for wav in audio[: max(0, min(args.warmup_files, len(audio)))]:
            x = torch.from_numpy(wav).view(1, -1, 1).to(device)
            _ = model.forward_full(x)[0]
            z_q, _indices = encode_quantized(model, x)
            _ = decode_quantized(model, z_q, x.shape[1])
        sync_if_needed(device)
        memory_snapshots.append(memory_snapshot("after_warmup", device))

        full_timings: list[float] = []
        compress_timings: list[float] = []
        decompress_timings: list[float] = []
        codec_timings: list[float] = []
        quality_rows: list[dict[str, float | None]] = []
        for idx, wav in enumerate(audio):
            x = torch.from_numpy(wav).view(1, -1, 1).to(device)
            y_full, full_elapsed = timed_call(device, lambda: model.forward_full(x)[0])
            (z_q, _indices), compress_elapsed = timed_call(device, lambda: encode_quantized(model, x))
            y_codec, decompress_elapsed = timed_call(device, lambda: decode_quantized(model, z_q, x.shape[1]))
            full_timings.append(full_elapsed)
            compress_timings.append(compress_elapsed)
            decompress_timings.append(decompress_elapsed)
            codec_timings.append(compress_elapsed + decompress_elapsed)
            if idx < max(0, args.quality_files):
                recon = y_codec.detach().cpu().numpy().reshape(-1).astype(np.float32)[: args.samples]
                quality_rows.append(quality_metrics_16k(audio[idx], recon))
            del y_full
        sync_if_needed(device)
        memory_snapshots.append(memory_snapshot("after_benchmark", device))

    audio_seconds = args.samples / float(cfg.sample_rate)
    benchmarks = {
        "full": summarize(full_timings, audio_seconds),
        "compress_only": summarize(compress_timings, audio_seconds),
        "decompress_only": summarize(decompress_timings, audio_seconds),
        "codec_full": summarize(codec_timings, audio_seconds),
    }
    report = {
        "backend": f"pytorch-{args.device}",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "fraction": float(args.fraction),
        "selected_files": len(files),
        "total_flac_files": len(sorted(args.data_root.rglob("*.flac"))) if args.manifest is None else None,
        "manifest": str(args.manifest.resolve()) if args.manifest is not None else None,
        "manifest_out": str(args.manifest_out.resolve()) if args.manifest_out is not None else None,
        "samples_per_file": int(args.samples),
        "audio_seconds_per_file": audio_seconds,
        "timing_excludes_io": True,
        "warmup_files": int(args.warmup_files),
        "benchmark": benchmarks["full"],
        "benchmarks": benchmarks,
        "quality_files": len(quality_rows),
        "metrics_mean": {
            key: (
                None
                if not any(row.get(key) is not None for row in quality_rows)
                else float(np.mean([row[key] for row in quality_rows if row.get(key) is not None]))
            )
            for key in ("si_sdr_db", "pesq_wb", "stoi", "visqol_moslqo", "lsd_db", "l1", "cos")
        },
        "nominal_rvq_kbps": float(nominal_rvq_kbps(cfg)),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "memory": {
            "scope": "process RSS and PyTorch CUDA memory; audio is preloaded on CPU, device input tensor is prepared before each timed forward pass",
            "pid": os.getpid(),
            "rss_peak_mib": peak_rss_mib(),
            "rss_current_mib": current_rss_mib(),
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0 if device.type == "cuda" else None
            ),
            "cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024.0 / 1024.0 if device.type == "cuda" else None
            ),
            "cuda_current_allocated_mib": (
                torch.cuda.memory_allocated(device) / 1024.0 / 1024.0 if device.type == "cuda" else None
            ),
            "cuda_current_reserved_mib": (
                torch.cuda.memory_reserved(device) / 1024.0 / 1024.0 if device.type == "cuda" else None
            ),
            "snapshots": memory_snapshots,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    b = benchmarks["full"]
    print(
        f"{report['backend']} full: files={len(files)} mean={b['mean_seconds']:.6f}s "
        f"p50={b['p50_seconds']:.6f}s p90={b['p90_seconds']:.6f}s xrt={b['x_realtime']:.2f}"
    )
    c = benchmarks["codec_full"]
    print(
        f"{report['backend']} codec_full: mean={c['mean_seconds']:.6f}s "
        f"p50={c['p50_seconds']:.6f}s p90={c['p90_seconds']:.6f}s xrt={c['x_realtime']:.2f}"
    )
    ce = benchmarks["compress_only"]
    de = benchmarks["decompress_only"]
    print(f"compress_only: mean={ce['mean_seconds']:.6f}s xrt={ce['x_realtime']:.2f}")
    print(f"decompress_only: mean={de['mean_seconds']:.6f}s xrt={de['x_realtime']:.2f}")
    m = report["memory"]
    rss_peak = m["rss_peak_mib"]
    mem_line = f"memory: rss_peak={rss_peak:.1f} MiB" if rss_peak is not None else "memory: rss_peak=n/a"
    if device.type == "cuda":
        mem_line += (
            f" cuda_peak_alloc={m['cuda_peak_allocated_mib']:.1f} MiB"
            f" cuda_peak_reserved={m['cuda_peak_reserved_mib']:.1f} MiB"
        )
    print(mem_line)
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
