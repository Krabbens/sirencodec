#!/usr/bin/env python3
"""Export a converted CUDA/PyTorch SirenCodec checkpoint for the C++20 inferencer.

The C++ runtime intentionally does not depend on PyTorch.  This script is the
one-time bridge from ``codec_step*_cuda.pt`` to a compact little-endian bundle
containing the inference config and raw float32 tensors.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


MAGIC = b"SIRENCB2"
VERSION = 1


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    return value


def _get_int(cfg: dict[str, Any], key: str, default: int = 0) -> int:
    return int(cfg.get(key, default) or 0)


def _get_float(cfg: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(cfg.get(key, default) or 0.0)


def _get_bool(cfg: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cfg.get(key, default))


def _activation_id(cfg: dict[str, Any]) -> int:
    activation = str(cfg.get("activation") or "gelu").strip().lower()
    if activation in {"snake", "snake_beta"}:
        return 1
    if activation == "gelu":
        return 0
    raise ValueError(f"unsupported activation for C++ export: {activation!r}")


def _decoder_upsample_id(cfg: dict[str, Any]) -> int:
    mode = str(cfg.get("decoder_upsample") or "transpose").strip().lower()
    if mode == "repeat_conv" or _get_bool(cfg, "causal"):
        return 1
    if mode == "transpose":
        return 0
    raise ValueError(f"unsupported decoder_upsample for C++ export: {mode!r}")


def _write_u32(f, value: int) -> None:
    f.write(struct.pack("<I", int(value)))


def _write_i32(f, value: int) -> None:
    f.write(struct.pack("<i", int(value)))


def _write_f32(f, value: float) -> None:
    f.write(struct.pack("<f", float(value)))


def export_bundle(checkpoint: Path, output: Path) -> None:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" not in blob or "config" not in blob:
        raise RuntimeError(f"checkpoint must contain 'model' and 'config': {checkpoint}")

    cfg = dict(blob["config"])
    state: dict[str, torch.Tensor] = blob["model"]
    enc_channels = [int(x) for x in cfg.get("enc_channels", ())]
    if not enc_channels:
        raise RuntimeError("config.enc_channels is empty")
    codebook_sizes_raw = cfg.get("codebook_sizes")
    codebook_sizes = (
        [int(x) for x in codebook_sizes_raw]
        if codebook_sizes_raw is not None
        else [int(cfg.get("codebook_size", 0))] * int(cfg.get("n_codebooks", 0))
    )

    config_json = json.dumps(_as_jsonable(cfg), indent=2, sort_keys=True).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(MAGIC)
        _write_u32(f, VERSION)

        _write_i32(f, _get_int(cfg, "sample_rate", 16000))
        _write_u32(f, len(enc_channels))
        for ch in enc_channels:
            _write_i32(f, ch)

        _write_i32(f, _get_int(cfg, "latent_dim", 512))
        _write_i32(f, _get_int(cfg, "stride1_blocks_per_scale", 0))
        _write_i32(f, _get_int(cfg, "pre_vq_layernorm", 0))
        _write_i32(f, _get_int(cfg, "latent_temporal_depth", 0))
        _write_i32(f, _get_int(cfg, "latent_temporal_post_depth", 0))
        _write_i32(f, _get_int(cfg, "self_attention_depth", 0))
        _write_i32(f, _get_int(cfg, "self_attention_post_depth", 0))
        _write_i32(f, _get_int(cfg, "self_attention_heads", 1))
        _write_i32(f, _get_int(cfg, "decoder_refine_depth", 0))
        _write_f32(f, _get_float(cfg, "decoder_refine_gain", 0.0))
        _write_i32(f, _get_int(cfg, "decoder_band_heads", 1))
        _write_i32(f, _get_int(cfg, "decoder_band_depth", 0))
        _write_f32(f, _get_float(cfg, "decoder_band_gain", 0.0))
        _write_i32(f, _get_int(cfg, "post_lavasr_depth", 0))
        _write_i32(f, _get_int(cfg, "post_lavasr_channels", 0))
        _write_i32(f, _get_int(cfg, "post_lavasr_kernel", 0))
        _write_f32(f, _get_float(cfg, "post_lavasr_gain", 0.0))
        _write_i32(f, int(_get_bool(cfg, "post_lavasr_highpass", False)))
        _write_i32(f, int(_get_bool(cfg, "harmonic_source", False)))
        _write_f32(f, _get_float(cfg, "harmonic_amp", 0.0))
        _write_i32(f, _get_int(cfg, "n_codebooks", len(codebook_sizes)))
        _write_i32(f, _get_int(cfg, "codebook_size", codebook_sizes[0] if codebook_sizes else 0))
        _write_i32(f, _get_int(cfg, "rvq_code_dim", 0))
        _write_i32(f, int(_get_bool(cfg, "vq_cosine", True)))
        _write_i32(f, int(_get_bool(cfg, "ae_only", False)))
        _write_i32(f, int(_get_bool(cfg, "causal", False)))
        _write_i32(f, _activation_id(cfg))
        _write_i32(f, _decoder_upsample_id(cfg))

        _write_u32(f, len(codebook_sizes))
        for size in codebook_sizes:
            _write_i32(f, size)

        _write_u32(f, len(config_json))
        f.write(config_json)

        _write_u32(f, len(state))
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
            encoded_name = name.encode("utf-8")
            if len(encoded_name) > 65535:
                raise RuntimeError(f"tensor name too long: {name}")
            f.write(struct.pack("<H", len(encoded_name)))
            f.write(encoded_name)
            if tensor.ndim > 255:
                raise RuntimeError(f"tensor rank too high: {name} rank={tensor.ndim}")
            f.write(struct.pack("<B", tensor.ndim))
            for dim in tensor.shape:
                _write_u32(f, int(dim))
            _write_u32(f, int(tensor.size))
            f.write(tensor.tobytes(order="C"))

    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint.resolve()),
                "output_bundle": str(output.resolve()),
                "format": "SIRENCB2",
                "version": VERSION,
                "tensor_count": len(state),
                "config": _as_jsonable(cfg),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Converted CUDA/PyTorch .pt checkpoint")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .sirenbin bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_bundle(args.checkpoint.resolve(), args.output)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix(args.output.suffix + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
