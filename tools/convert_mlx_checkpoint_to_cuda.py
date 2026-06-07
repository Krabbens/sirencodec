#!/usr/bin/env python3
"""Convert legacy MLX ``codec_step*.npz`` weights to a CUDA/PyTorch checkpoint.

The old MLX checkpoints store 1-D convolution kernels as ``(out, kernel, in)``
and use MLX-oriented module names.  ``CUDACodec`` uses PyTorch ``Conv1d`` layers
with kernels in ``(out, in, kernel)`` and a few wrapper modules.  This script
maps the names and tensor layouts, verifies a strict load into ``CUDACodec``,
and writes a model-only ``.pt`` checkpoint suitable for ``--init-from`` or
``tools/infer_cuda.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sirencodec.config import Config, nominal_rvq_kbps  # noqa: E402
from sirencodec.cuda.codec import CUDACodec  # noqa: E402


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _default_import_config(args: argparse.Namespace) -> Config:
    return Config(
        dataset=args.dataset,
        batch=args.batch,
        segment=args.segment,
        enc_channels=args.enc_channels,
        latent_dim=args.latent_dim,
        latent_temporal_depth=args.latent_temporal_depth,
        latent_temporal_post_depth=args.latent_temporal_post_depth,
        self_attention_depth=args.self_attention_depth,
        self_attention_post_depth=args.self_attention_post_depth,
        self_attention_heads=args.self_attention_heads,
        decoder_refine_depth=args.decoder_refine_depth,
        decoder_refine_gain=args.decoder_refine_gain,
        decoder_band_heads=args.decoder_band_heads,
        decoder_band_depth=args.decoder_band_depth,
        decoder_band_gain=args.decoder_band_gain,
        post_lavasr_depth=args.post_lavasr_depth,
        post_lavasr_channels=args.post_lavasr_channels,
        post_lavasr_kernel=args.post_lavasr_kernel,
        post_lavasr_gain=args.post_lavasr_gain,
        post_lavasr_highpass=bool(args.post_lavasr_highpass),
        activation=args.activation,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        rvq_code_dim=args.rvq_code_dim,
        vq_cosine=bool(args.vq_cosine),
        decoder_upsample=args.decoder_upsample,
        vq_ema_decay=args.vq_ema_decay,
        lambda_semantic=0.0,
    )


def _torch_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr)).detach().clone()


def _conv_kernel(arr: np.ndarray) -> torch.Tensor:
    """MLX Conv1d ``(out, kernel, in)`` -> PyTorch Conv1d ``(out, in, kernel)``."""
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D convolution kernel, got shape={arr.shape}")
    return torch.from_numpy(np.asarray(arr).transpose(0, 2, 1)).contiguous()


def _put_linear_or_param(out: dict[str, torch.Tensor], target: str, arr: np.ndarray) -> None:
    out[target] = _torch_tensor(arr)


def _map_rvq(name: str) -> str:
    # rvq.q0.embedding.weight -> rvq.stages.0.embedding.weight
    parts = name.split(".")
    if len(parts) < 3 or not parts[1].startswith("q"):
        raise ValueError(f"unexpected RVQ key: {name}")
    stage = int(parts[1][1:])
    return ".".join(["rvq", "stages", str(stage), *parts[2:]])


def _map_attention(
    source: dict[str, np.ndarray],
    out: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> set[str]:
    consumed: set[str] = set()
    layers: set[int] = set()
    prefix = "self_attn_pre.layers."
    for key in source:
        if key.startswith(prefix):
            try:
                layers.add(int(key[len(prefix) :].split(".", 1)[0]))
            except ValueError:
                pass

    for layer in sorted(layers):
        base = f"self_attn_pre.layers.{layer}"
        q_key = f"{base}.attention.query_proj.weight"
        k_key = f"{base}.attention.key_proj.weight"
        v_key = f"{base}.attention.value_proj.weight"
        if q_key in source and k_key in source and v_key in source:
            target = f"{base}.attn.in_proj_weight"
            out[target] = torch.cat(
                [_torch_tensor(source[q_key]), _torch_tensor(source[k_key]), _torch_tensor(source[v_key])],
                dim=0,
            )
            consumed.update({q_key, k_key, v_key})
            bias_target = f"{base}.attn.in_proj_bias"
            if bias_target in target_state:
                out[bias_target] = torch.zeros_like(target_state[bias_target])

        pairs = {
            f"{base}.attention.out_proj.weight": f"{base}.attn.out_proj.weight",
            f"{base}.linear1.weight": f"{base}.ff.0.weight",
            f"{base}.linear1.bias": f"{base}.ff.0.bias",
            f"{base}.linear2.weight": f"{base}.ff.2.weight",
            f"{base}.linear2.bias": f"{base}.ff.2.bias",
            f"{base}.ln1.weight": f"{base}.norm1.weight",
            f"{base}.ln1.bias": f"{base}.norm1.bias",
            f"{base}.ln2.weight": f"{base}.norm2.weight",
            f"{base}.ln2.bias": f"{base}.norm2.bias",
        }
        for src, dst in pairs.items():
            if src in source:
                out[dst] = _torch_tensor(source[src])
                consumed.add(src)

        out_bias = f"{base}.attn.out_proj.bias"
        if out_bias in target_state and out_bias not in out:
            out[out_bias] = torch.zeros_like(target_state[out_bias])

    return consumed


def convert_state(source: dict[str, np.ndarray], cfg: Config) -> dict[str, torch.Tensor]:
    model = CUDACodec(cfg)
    target_state = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    consumed = _map_attention(source, out, target_state)

    for name, arr in source.items():
        if name in consumed:
            continue

        target: str | None = None
        tensor: torch.Tensor | None = None

        if name.startswith("rvq."):
            target = _map_rvq(name)
            tensor = _torch_tensor(arr)

        elif name.startswith("encoder.out."):
            suffix = name.removeprefix("encoder.out.")
            target = f"encoder.layers.{2 * len(cfg.enc_channels)}.conv.{suffix}"
            tensor = _conv_kernel(arr) if suffix == "weight" else _torch_tensor(arr)

        elif name.startswith("encoder.layers.") or name.startswith("latent_pre.") or name.startswith("latent_post."):
            if name.endswith(".weight") and arr.ndim == 3:
                target = name.removesuffix(".weight") + ".conv.weight"
                tensor = _conv_kernel(arr)
            elif name.endswith(".bias") and arr.ndim == 1:
                target = name.removesuffix(".bias") + ".conv.bias"
                tensor = _torch_tensor(arr)
            else:
                target = name
                tensor = _torch_tensor(arr)

        elif name.startswith("decoder.layers."):
            if name.endswith(".conv.weight"):
                target = name.removesuffix(".weight") + ".conv.weight"
                tensor = _conv_kernel(arr)
            elif name.endswith(".conv.bias"):
                target = name.removesuffix(".bias") + ".conv.bias"
                tensor = _torch_tensor(arr)
            else:
                target = name
                tensor = _torch_tensor(arr)

        elif name.startswith("decoder.out."):
            suffix = name.removeprefix("decoder.out.")
            target = f"decoder.out.conv.{suffix}"
            tensor = _conv_kernel(arr) if suffix == "weight" else _torch_tensor(arr)

        elif name.startswith("decoder.refine.layers."):
            if name.endswith(".weight") and arr.ndim == 3:
                target = name.removesuffix(".weight") + ".conv.weight"
                tensor = _conv_kernel(arr)
            elif name.endswith(".bias") and arr.ndim == 1:
                target = name.removesuffix(".bias") + ".conv.bias"
                tensor = _torch_tensor(arr)
            else:
                target = name
                tensor = _torch_tensor(arr)

        elif name.startswith("decoder.post_lavasr."):
            suffix = name.removeprefix("decoder.post_lavasr.")
            if suffix.startswith("in_proj."):
                sub = suffix.removeprefix("in_proj.")
                target = f"decoder.post_lavasr.layers.0.conv.{sub}"
                tensor = _conv_kernel(arr) if sub == "weight" else _torch_tensor(arr)
            elif suffix.startswith("out."):
                sub = suffix.removeprefix("out.")
                final_idx = 1 + 2 * int(cfg.post_lavasr_depth)
                target = f"decoder.post_lavasr.layers.{final_idx}.conv.{sub}"
                tensor = _conv_kernel(arr) if sub == "weight" else _torch_tensor(arr)
            elif suffix.startswith("layers."):
                parts = suffix.split(".")
                old_idx = int(parts[1])
                rest = ".".join(parts[2:])
                if rest in {"weight", "bias"}:
                    new_idx = 1 + old_idx
                    target = f"decoder.post_lavasr.layers.{new_idx}.conv.{rest}"
                    tensor = _conv_kernel(arr) if rest == "weight" else _torch_tensor(arr)
                else:
                    new_idx = old_idx + 1
                    target = f"decoder.post_lavasr.layers.{new_idx}.{rest}"
                    tensor = _torch_tensor(arr)

        if target is None or tensor is None:
            raise ValueError(f"no conversion rule for {name} shape={arr.shape}")
        if target not in target_state:
            raise KeyError(f"converted key {name} -> {target}, but target model has no such tensor")
        expected = tuple(target_state[target].shape)
        got = tuple(tensor.shape)
        if expected != got:
            raise ValueError(f"shape mismatch {name} -> {target}: got {got}, expected {expected}")
        out[target] = tensor.to(dtype=target_state[target].dtype)

    merged = dict(target_state)
    merged.update(out)
    missing = sorted(set(target_state) - set(out))
    # Attention biases are absent in the MLX checkpoint and are intentionally kept at PyTorch zero init.
    allowed_missing = {
        k
        for k in target_state
        if k.endswith(".attn.in_proj_bias") or k.endswith(".attn.out_proj.bias")
    }
    unexpected_missing = [k for k in missing if k not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(
            "conversion did not provide all target tensors; first missing: "
            + ", ".join(unexpected_missing[:20])
        )
    return merged


def _jsonable_config(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cfg.__dict__.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Legacy MLX codec_step*.npz")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output CUDA/PyTorch .pt checkpoint")
    p.add_argument("--dataset", default="train-clean-100")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--segment", type=int, default=16000)
    p.add_argument("--enc-channels", type=_parse_int_tuple, default=(24, 32, 48, 64, 96, 128, 192, 256))
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--latent-temporal-depth", type=int, default=2)
    p.add_argument("--latent-temporal-post-depth", type=int, default=2)
    p.add_argument("--self-attention-depth", type=int, default=1)
    p.add_argument("--self-attention-post-depth", type=int, default=0)
    p.add_argument("--self-attention-heads", type=int, default=2)
    p.add_argument("--decoder-refine-depth", type=int, default=1)
    p.add_argument("--decoder-refine-gain", type=float, default=0.018)
    p.add_argument("--decoder-band-heads", type=int, default=1)
    p.add_argument("--decoder-band-depth", type=int, default=1)
    p.add_argument("--decoder-band-gain", type=float, default=0.08)
    p.add_argument("--post-lavasr-depth", type=int, default=2)
    p.add_argument("--post-lavasr-channels", type=int, default=24)
    p.add_argument("--post-lavasr-kernel", type=int, default=15)
    p.add_argument("--post-lavasr-gain", type=float, default=0.02)
    p.add_argument("--post-lavasr-highpass", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--activation", default="snake_beta")
    p.add_argument("--n-codebooks", type=int, default=3)
    p.add_argument("--codebook-size", type=int, default=32)
    p.add_argument("--rvq-code-dim", type=int, default=8)
    p.add_argument("--vq-cosine", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vq-ema-decay", type=float, default=0.0)
    p.add_argument("--decoder-upsample", choices=["transpose", "repeat_conv"], default="repeat_conv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inp = args.input.resolve()
    if inp.suffix != ".npz":
        raise SystemExit("Only legacy MLX .npz model checkpoints are supported as input")
    cfg = _default_import_config(args)
    with np.load(inp) as data:
        source = {k: np.asarray(data[k]) for k in data.files}
    state = convert_state(source, cfg)

    model = CUDACodec(cfg)
    model.load_state_dict(state, strict=True)

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": state,
        "config": _jsonable_config(cfg),
        "source_checkpoint": str(inp),
        "converted_from": "legacy_mlx_npz",
        "nominal_rvq_kbps": float(nominal_rvq_kbps(cfg)),
    }
    torch.save(payload, out)
    sidecar = out.with_suffix(out.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "source_checkpoint": str(inp),
                "output_checkpoint": str(out),
                "converted_from": "legacy_mlx_npz",
                "num_tensors": len(state),
                "nominal_rvq_kbps": float(nominal_rvq_kbps(cfg)),
                "config": _jsonable_config(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"converted: {inp}")
    print(f"wrote:     {out}")
    print(f"sidecar:   {sidecar}")
    print(f"tensors:   {len(state)}")
    print(f"bitrate:   {nominal_rvq_kbps(cfg):.4f} kbps nominal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
