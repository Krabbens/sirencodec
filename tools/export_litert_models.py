#!/usr/bin/env python3
"""Export SirenCodec PyTorch checkpoints to LiteRT/TFLite FlatBuffers.

This exporter is intentionally separate from the C++ inference bundle exporter.
It produces fixed-shape LiteRT models for:

* encoder: waveform -> continuous latent
* decoder: quantized latent -> waveform
* full_recon: waveform -> waveform reconstruction
* compress_packet: waveform -> RVQ indices + cosine residual norms
* decompress_packet: RVQ indices + cosine residual norms -> waveform

The packet models preserve the codec split used by the C++ implementation.  For
cosine RVQ, indices alone are not sufficient to reconstruct the quantized
latent, so the compressor also exports per-stage residual norms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sirencodec.config import Config, encoder_time_stride  # noqa: E402
from sirencodec.cuda.codec import CUDACodec  # noqa: E402


MODEL_CHOICES = ("encoder", "decoder", "full_recon", "compress_packet", "decompress_packet")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, type):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sizeof_fmt(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def _parse_models(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names or names == ["all"]:
        return list(MODEL_CHOICES)
    unknown = sorted(set(names) - set(MODEL_CHOICES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model(s): {', '.join(unknown)}")
    return names


def load_checkpoint(path: Path) -> tuple[CUDACodec, Config]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in blob or "config" not in blob:
        raise RuntimeError(f"checkpoint must contain 'model' and 'config': {path}")
    cfg_blob = dict(blob["config"])
    if isinstance(cfg_blob.get("data_dir"), str):
        cfg_blob["data_dir"] = Path(cfg_blob["data_dir"])
    cfg = Config(**cfg_blob)
    model = CUDACodec(cfg).eval()
    model.load_state_dict(blob["model"], strict=True)
    return model, cfg


def load_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile

    sr, wav = wavfile.read(path)
    if wav.ndim == 2:
        wav = wav[:, 0]
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        scale = float(max(abs(info.min), info.max))
        wav = wav.astype(np.float32) / scale
    else:
        wav = wav.astype(np.float32)
    if int(sr) != int(target_sr):
        old_x = np.arange(wav.shape[0], dtype=np.float64)
        new_len = max(1, int(round(wav.shape[0] * float(target_sr) / float(sr))))
        new_x = np.linspace(0.0, float(wav.shape[0] - 1), num=new_len, dtype=np.float64)
        wav = np.interp(new_x, old_x, wav.astype(np.float64)).astype(np.float32)
    peak = max(float(np.max(np.abs(wav))) if wav.size else 0.0, 1e-5)
    return (wav / peak).astype(np.float32)


def fit_length(wav: np.ndarray, sample_count: int) -> np.ndarray:
    if wav.shape[0] >= sample_count:
        return wav[:sample_count].astype(np.float32, copy=False)
    out = np.zeros((sample_count,), dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def make_sample_wave(args: argparse.Namespace, sample_rate: int) -> tuple[torch.Tensor, dict[str, Any]]:
    input_wav: np.ndarray | None = None
    if args.input is not None:
        input_wav = load_wav_mono(args.input.resolve(), sample_rate)

    if args.samples is not None:
        sample_count = int(args.samples)
    elif args.seconds is not None:
        sample_count = int(round(float(args.seconds) * sample_rate))
    elif input_wav is not None:
        sample_count = int(input_wav.shape[0])
    else:
        sample_count = int(2 * sample_rate)
    if sample_count <= 0:
        raise RuntimeError(f"sample count must be positive, got {sample_count}")

    wav = fit_length(input_wav, sample_count) if input_wav is not None else np.zeros((sample_count,), dtype=np.float32)
    x = torch.from_numpy(wav).reshape(1, sample_count, 1).contiguous()
    meta = {
        "input": str(args.input.resolve()) if args.input is not None else None,
        "sample_count": sample_count,
        "duration_s": sample_count / float(sample_rate),
        "input_was_padded": bool(input_wav is not None and input_wav.shape[0] < sample_count),
        "input_original_samples": int(input_wav.shape[0]) if input_wav is not None else None,
    }
    return x, meta


class EncoderWrapper(nn.Module):
    def __init__(self, model: CUDACodec):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.latent_before_rvq(x)


class DecoderWrapper(nn.Module):
    def __init__(self, model: CUDACodec, target_len: int):
        super().__init__()
        self.model = model
        self.target_len = int(target_len)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.model.latent_post is not None:
            z = self.model.latent_post(z)
        if self.model.self_attn_post is not None:
            z = self.model.self_attn_post(z)
        return self.model.decoder(z, self.target_len)


class FullReconstructionWrapper(nn.Module):
    def __init__(self, model: CUDACodec):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_reconstruction_only(x)


class CompressPacketWrapper(nn.Module):
    def __init__(self, model: CUDACodec):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.model.latent_before_rvq(x)
        quantized = torch.zeros_like(z)
        indices: list[torch.Tensor] = []
        norms: list[torch.Tensor] = []
        for stage in self.model.rvq.stages:
            residual = z - quantized
            r = stage.project_residual(residual)
            cb = stage.embedding.weight
            if self.model.cfg.vq_cosine:
                eb = F.normalize(cb, dim=-1, eps=1e-8)
                zn = F.normalize(r, dim=-1, eps=1e-8)
                dist = 2.0 - 2.0 * torch.matmul(zn, eb.t())
                idx = torch.argmin(dist, dim=-1)
                z_norm = torch.linalg.vector_norm(r, dim=-1, keepdim=True).clamp_min(1e-8)
                z_q_low = F.embedding(idx, eb) * z_norm
                norms.append(z_norm.squeeze(-1))
            else:
                z2 = torch.sum(r * r, dim=-1, keepdim=True)
                e2 = torch.sum(cb * cb, dim=-1)
                dist = z2 + e2 - 2.0 * torch.matmul(r, cb.t())
                idx = torch.argmin(dist, dim=-1)
                z_q_low = stage.embedding(idx)
                norms.append(torch.ones_like(idx, dtype=r.dtype))
            z_i = stage.out_proj(z_q_low) if stage.out_proj is not None else z_q_low
            quantized = quantized + z_i
            indices.append(idx.to(torch.int32))
        return torch.stack(indices, dim=1), torch.stack(norms, dim=1)


class DecompressPacketWrapper(nn.Module):
    def __init__(self, model: CUDACodec, target_len: int):
        super().__init__()
        self.model = model
        self.target_len = int(target_len)

    def forward(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        batch = indices.shape[0]
        frames = indices.shape[2]
        quantized = torch.zeros(
            (batch, frames, self.model.cfg.latent_dim),
            dtype=norms.dtype,
            device=norms.device,
        )
        for qi, stage in enumerate(self.model.rvq.stages):
            idx = indices[:, qi, :].to(torch.long)
            cb = stage.embedding.weight
            if self.model.cfg.vq_cosine:
                eb = F.normalize(cb, dim=-1, eps=1e-8)
                z_q_low = F.embedding(idx, eb) * norms[:, qi, :].unsqueeze(-1).clamp_min(1e-8)
            else:
                z_q_low = stage.embedding(idx)
            z_i = stage.out_proj(z_q_low) if stage.out_proj is not None else z_q_low
            quantized = quantized + z_i

        z = quantized
        if self.model.latent_post is not None:
            z = self.model.latent_post(z)
        if self.model.self_attn_post is not None:
            z = self.model.self_attn_post(z)
        return self.model.decoder(z, self.target_len)


@dataclass
class PreparedModels:
    modules: dict[str, nn.Module]
    sample_args: dict[str, tuple[torch.Tensor, ...]]


def prepare_models(model: CUDACodec, x: torch.Tensor, requested: set[str]) -> PreparedModels:
    modules: dict[str, nn.Module] = {}
    sample_args: dict[str, tuple[torch.Tensor, ...]] = {}
    target_len = int(x.shape[1])

    with torch.inference_mode():
        z = model.latent_before_rvq(x)
        z_q, *_ = model.rvq(z)
        packet = CompressPacketWrapper(model)(x)

    if "encoder" in requested:
        modules["encoder"] = EncoderWrapper(model).eval()
        sample_args["encoder"] = (x,)
    if "decoder" in requested:
        modules["decoder"] = DecoderWrapper(model, target_len).eval()
        sample_args["decoder"] = (z_q,)
    if "full_recon" in requested:
        modules["full_recon"] = FullReconstructionWrapper(model).eval()
        sample_args["full_recon"] = (x,)
    if "compress_packet" in requested:
        modules["compress_packet"] = CompressPacketWrapper(model).eval()
        sample_args["compress_packet"] = (x,)
    if "decompress_packet" in requested:
        modules["decompress_packet"] = DecompressPacketWrapper(model, target_len).eval()
        sample_args["decompress_packet"] = packet

    return PreparedModels(modules=modules, sample_args=sample_args)


def output_shapes(value: torch.Tensor | tuple[torch.Tensor, ...]) -> list[dict[str, Any]]:
    values = value if isinstance(value, tuple) else (value,)
    return [{"shape": list(v.shape), "dtype": str(v.dtype)} for v in values]


def compare_arrays(got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    if np.issubdtype(got.dtype, np.integer):
        diff = got.astype(np.int64) - ref.astype(np.int64)
        return {
            "kind": "integer",
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "max_abs": int(np.max(np.abs(diff))) if diff.size else 0,
        }
    delta = got.astype(np.float64) - ref.astype(np.float64)
    return {
        "kind": "float",
        "shape": list(got.shape),
        "dtype": str(got.dtype),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


def validate_model(
    model_path: Path,
    sample_args: tuple[torch.Tensor, ...],
    torch_output: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    runs: int,
    num_threads: int,
    duration_s: float,
) -> dict[str, Any]:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != len(sample_args):
        raise RuntimeError(
            f"{model_path.name}: expected {len(sample_args)} LiteRT inputs, got {len(input_details)}"
        )

    for detail, arg in zip(input_details, sample_args):
        array = arg.detach().cpu().numpy()
        if detail["dtype"] == np.int32:
            array = array.astype(np.int32)
        elif detail["dtype"] == np.int64:
            array = array.astype(np.int64)
        else:
            array = array.astype(np.float32)
        interpreter.set_tensor(detail["index"], array)

    interpreter.invoke()
    start = time.perf_counter()
    for _ in range(max(1, runs)):
        interpreter.invoke()
    mean_latency_s = (time.perf_counter() - start) / float(max(1, runs))

    refs = torch_output if isinstance(torch_output, tuple) else (torch_output,)
    comparisons: list[dict[str, Any]] = []
    for detail, ref in zip(output_details, refs):
        got = interpreter.get_tensor(detail["index"])
        comparisons.append(compare_arrays(got, ref.detach().cpu().numpy()))

    return {
        "mean_latency_s": mean_latency_s,
        "x_realtime": duration_s / mean_latency_s if mean_latency_s > 0 else None,
        "num_threads": num_threads,
        "runs": max(1, runs),
        "inputs": _jsonable(input_details),
        "outputs": _jsonable(output_details),
        "comparisons": comparisons,
    }


def export_one(
    name: str,
    module: nn.Module,
    sample_args: tuple[torch.Tensor, ...],
    out_dir: Path,
    *,
    strict_export: str | bool,
    lightweight_conversion: bool,
    validate: bool,
    benchmark_runs: int,
    num_threads: int,
    duration_s: float,
) -> dict[str, Any]:
    import litert_torch

    safe_shape = "x".join(str(dim) for dim in sample_args[0].shape)
    out_path = out_dir / f"{name}_{safe_shape}.tflite"
    report: dict[str, Any] = {
        "name": name,
        "path": str(out_path),
        "sample_args": output_shapes(sample_args),
        "status": "pending",
    }

    module.eval()
    with torch.inference_mode():
        torch_output = module(*sample_args)
    report["torch_outputs"] = output_shapes(torch_output)

    print(f"\n{name}: converting -> {out_path}")
    start = time.perf_counter()
    try:
        edge_model = litert_torch.convert(
            module,
            sample_args=sample_args,
            strict_export=strict_export,
            lightweight_conversion=lightweight_conversion,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        edge_model.export(str(out_path))
        elapsed = time.perf_counter() - start
        size = out_path.stat().st_size
        report.update(
            {
                "status": "ok",
                "convert_s": elapsed,
                "size_bytes": size,
            }
        )
        print(f"{name}: OK {_sizeof_fmt(size)} in {elapsed:.3f}s")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        report.update(
            {
                "status": "failed",
                "convert_s": elapsed,
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=12),
            }
        )
        print(f"{name}: FAILED after {elapsed:.3f}s: {exc}")
        return report

    if validate:
        print(f"{name}: validating with LiteRT interpreter")
        try:
            validation = validate_model(
                out_path,
                sample_args,
                torch_output,
                runs=benchmark_runs,
                num_threads=num_threads,
                duration_s=duration_s,
            )
            report["validation"] = validation
            for idx, cmp in enumerate(validation["comparisons"]):
                if cmp["kind"] == "integer":
                    print(f"{name}: output{idx} max_abs={cmp['max_abs']}")
                else:
                    print(
                        f"{name}: output{idx} max_abs={cmp['max_abs']:.3g} "
                        f"mean_abs={cmp['mean_abs']:.3g}"
                    )
            print(
                f"{name}: LiteRT mean={validation['mean_latency_s']:.6f}s "
                f"xrt={validation['x_realtime']:.2f} ({num_threads} threads)"
            )
        except Exception as exc:  # noqa: BLE001
            report["validation_error"] = repr(exc)
            report["validation_traceback"] = traceback.format_exc(limit=12)
            print(f"{name}: validation FAILED: {exc}")

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Converted CUDA/PyTorch .pt checkpoint")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("artifacts/models/litert_export"))
    parser.add_argument("-i", "--input", type=Path, help="Optional WAV used as representative fixed-shape input")
    parser.add_argument("--seconds", type=float, help="Fixed input duration. Defaults to full --input length or 2s.")
    parser.add_argument("--samples", type=int, help="Fixed input sample count. Overrides --seconds.")
    parser.add_argument("--models", type=_parse_models, default=list(MODEL_CHOICES), help="Comma list or 'all'")
    parser.add_argument("--no-validate", action="store_true", help="Skip LiteRT interpreter validation")
    parser.add_argument("--benchmark-runs", type=int, default=5, help="Interpreter runs during validation")
    parser.add_argument("--num-threads", type=int, default=4, help="LiteRT interpreter CPU threads")
    parser.add_argument("--strict-export", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--lightweight-conversion", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    requested = set(args.models)

    torch.set_grad_enabled(False)
    model, cfg = load_checkpoint(checkpoint)
    x, sample_meta = make_sample_wave(args, int(cfg.sample_rate))
    prepared = prepare_models(model, x, requested)

    if args.strict_export == "true":
        strict_export: str | bool = True
    elif args.strict_export == "false":
        strict_export = False
    else:
        strict_export = "auto"

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir),
        "sample": sample_meta,
        "config": {
            "sample_rate": int(cfg.sample_rate),
            "latent_dim": int(cfg.latent_dim),
            "encoder_time_stride": int(encoder_time_stride(cfg)),
            "n_codebooks": int(cfg.n_codebooks),
            "codebook_size": int(cfg.codebook_size),
            "rvq_code_dim": int(getattr(cfg, "rvq_code_dim", 0) or 0),
            "vq_cosine": bool(cfg.vq_cosine),
        },
        "models": [],
    }

    print(f"checkpoint: {checkpoint}")
    print(
        f"shape:      input={tuple(x.shape)} duration={sample_meta['duration_s']:.3f}s "
        f"stride={encoder_time_stride(cfg)}"
    )
    print(f"out_dir:    {out_dir}")
    print(f"models:     {', '.join(name for name in MODEL_CHOICES if name in requested)}")

    for name in MODEL_CHOICES:
        if name not in requested:
            continue
        result = export_one(
            name,
            prepared.modules[name],
            prepared.sample_args[name],
            out_dir,
            strict_export=strict_export,
            lightweight_conversion=bool(args.lightweight_conversion),
            validate=not args.no_validate,
            benchmark_runs=max(1, int(args.benchmark_runs)),
            num_threads=max(1, int(args.num_threads)),
            duration_s=float(sample_meta["duration_s"]),
        )
        report["models"].append(result)

    report_path = out_dir / f"export_report_{sample_meta['sample_count']}samples.json"
    merged_from_existing = False
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            same_sample = existing.get("sample", {}).get("sample_count") == sample_meta["sample_count"]
            if same_sample and isinstance(existing.get("models"), list):
                existing_by_name = {
                    item.get("name"): item
                    for item in existing["models"]
                    if isinstance(item, dict) and item.get("name")
                }
                for item in report["models"]:
                    existing_by_name[item["name"]] = item
                report["models"] = [
                    existing_by_name[name]
                    for name in MODEL_CHOICES
                    if name in existing_by_name
                ]
                merged_from_existing = True
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not merge existing report {report_path}: {exc}")
    report["merged_from_existing"] = merged_from_existing
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")
    requested_results = [item for item in report["models"] if item.get("name") in requested]
    ok_count = sum(1 for item in requested_results if item.get("status") == "ok")
    total_ok_count = sum(1 for item in report["models"] if item.get("status") == "ok")
    print(f"\nwrote: {report_path}")
    print(f"ok:    {ok_count}/{len(requested_results)} requested, {total_ok_count}/{len(report['models'])} in report")
    return 0 if ok_count == len(requested_results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
