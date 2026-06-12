#!/usr/bin/env python3
"""Train the conditioned 0-3.5 kHz / 3.5-8 kHz MLX codec."""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from mlx.utils import tree_flatten

from ..config import Config, FAST_STFT_SCALES, LIBRI_STFT_SCALE_WEIGHTS
from .data import _collect_audio_paths, _load_audio_batch, synth_batch
from .losses import (
    mel_filterbank_mx,
    mel_log_bin_losses,
    multi_stft_complex_l1,
    multi_stft_loss,
    multi_stft_mag_l1_linear,
    multi_stft_spectral_convergence,
    multi_stft_spectral_terms,
)
from .train import (
    _cast_module_floats,
    _eval_loss_and_grad_tree,
    _format_vq_util,
    _grad_tree_any_nonfinite,
    _kmeans_centroids_numpy,
    _refresh_optimizer_schedules,
    _tree_n_params,
    build_lr_schedule,
    clip_gradients_global_norm,
    effective_lambda_marginal,
    effective_lambda_stft,
    load_full_checkpoint_into,
    save_full_checkpoint,
    save_reconstruction_wavs,
    save_spectrogram_png,
    update_vq_ema_codebooks,
    vq_reset_dead_codes,
)
from .two_band import (
    DEFAULT_HIGH_CHANNELS,
    TwoBandCodec,
    TwoBandCodecConfig,
)


def _batch_cosine(reference: mx.array, estimate: mx.array) -> mx.array:
    dot = mx.sum(reference * estimate, axis=(1, 2))
    ref_norm = mx.sqrt(mx.sum(reference * reference, axis=(1, 2)) + 1e-8)
    est_norm = mx.sqrt(mx.sum(estimate * estimate, axis=(1, 2)) + 1e-8)
    return mx.mean(dot / (ref_norm * est_norm))


def _reconstruction_terms(
    prediction: mx.array,
    target: mx.array,
    cfg: Config,
    step: int,
    mel_fb: mx.array | None,
    *,
    normalize_floor: float | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    if normalize_floor is not None:
        scale = mx.maximum(
            mx.mean(mx.abs(target)),
            mx.array(float(normalize_floor), dtype=mx.float32),
        )
        prediction = prediction / scale
        target = target / scale
    else:
        scale = mx.array(1.0, dtype=mx.float32)

    l_time = mx.mean(mx.abs(prediction - target))
    use_spectral = any(
        weight > 0
        for weight in (
            cfg.lambda_stft,
            cfg.lambda_stft_grad,
            cfg.lambda_stft_cos,
            cfg.lambda_sc,
            cfg.lambda_complex_stft,
            cfg.lambda_mag_l1,
        )
    )
    if use_spectral and (cfg.lambda_stft_grad > 0 or cfg.lambda_stft_cos > 0):
        l_stft, l_stft_grad, l_stft_cos = multi_stft_spectral_terms(
            prediction,
            target,
            cfg.stft_scales,
            with_grad=cfg.lambda_stft_grad > 0,
            with_cos_1m=cfg.lambda_stft_cos > 0,
            grad_freq_weight=cfg.stft_grad_freq_weight,
            grad_time_weight=cfg.stft_grad_time_weight,
            hf_emphasis=cfg.stft_hf_emphasis,
            scale_weights=cfg.stft_scale_weights,
        )
    elif use_spectral:
        l_stft = multi_stft_loss(
            prediction,
            target,
            cfg.stft_scales,
            hf_emphasis=cfg.stft_hf_emphasis,
            scale_weights=cfg.stft_scale_weights,
        )
        l_stft_grad = mx.array(0.0, dtype=mx.float32)
        l_stft_cos = mx.array(0.0, dtype=mx.float32)
    else:
        l_stft = mx.array(0.0, dtype=mx.float32)
        l_stft_grad = mx.array(0.0, dtype=mx.float32)
        l_stft_cos = mx.array(0.0, dtype=mx.float32)

    stft_weight = effective_lambda_stft(step, cfg)
    total = cfg.lambda_time * l_time + stft_weight * l_stft

    l_sc = mx.array(0.0, dtype=mx.float32)
    if cfg.lambda_sc > 0:
        l_sc = multi_stft_spectral_convergence(
            prediction,
            target,
            cfg.stft_scales,
            scale_weights=cfg.stft_scale_weights,
        )
        total = total + stft_weight * cfg.lambda_sc * l_sc

    l_complex = mx.array(0.0, dtype=mx.float32)
    if cfg.lambda_complex_stft > 0:
        l_complex = multi_stft_complex_l1(
            prediction,
            target,
            cfg.stft_scales,
            scale_weights=cfg.stft_scale_weights,
        )
        total = total + stft_weight * cfg.lambda_complex_stft * l_complex

    if cfg.lambda_stft_grad > 0:
        total = total + stft_weight * cfg.lambda_stft_grad * l_stft_grad
    if cfg.lambda_stft_cos > 0:
        total = total + stft_weight * cfg.lambda_stft_cos * l_stft_cos

    l_mag = mx.array(0.0, dtype=mx.float32)
    if cfg.lambda_mag_l1 > 0:
        l_mag = multi_stft_mag_l1_linear(
            prediction,
            target,
            cfg.stft_scales,
            hf_emphasis=cfg.stft_hf_emphasis,
            scale_weights=cfg.stft_scale_weights,
        )
        total = total + stft_weight * cfg.lambda_mag_l1 * l_mag

    l_mel = mx.array(0.0, dtype=mx.float32)
    if mel_fb is not None and (cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0):
        mel_l1, mel_l2 = mel_log_bin_losses(
            prediction,
            target,
            mel_fb,
            cfg.mel_n_fft,
            cfg.mel_hop,
        )
        l_mel = cfg.lambda_mel_l1 * mel_l1 + cfg.lambda_mel_l2 * mel_l2
        total = total + stft_weight * l_mel

    cosine = _batch_cosine(target, prediction)
    if cfg.lambda_cos > 0:
        total = total + cfg.lambda_cos * (1.0 - cosine)

    return total, {
        "time": l_time,
        "stft": l_stft,
        "stft_grad": l_stft_grad,
        "stft_cos": l_stft_cos,
        "sc": l_sc,
        "complex": l_complex,
        "mag": l_mag,
        "mel": l_mel,
        "cosine": cosine,
        "scale": scale,
    }


def make_two_band_train_fn(
    model: TwoBandCodec,
    cfg: Config,
    batch: mx.array,
    step: int,
    mel_fb: mx.array | None,
):
    """Create a differentiable joint loss and expose same-forward metrics."""
    forward_metrics: dict[str, object] = {}

    def loss_fn(m: TwoBandCodec) -> mx.array:
        output = m(batch)
        full_loss, full = _reconstruction_terms(
            output.reconstruction,
            batch,
            cfg,
            step,
            mel_fb,
        )
        low_loss, low = _reconstruction_terms(
            output.low_reconstruction,
            output.low_target,
            cfg,
            step,
            mel_fb,
        )
        high_loss, high = _reconstruction_terms(
            output.high_reconstruction,
            output.high_target,
            cfg,
            step,
            mel_fb,
            normalize_floor=m.two_band_cfg.high_loss_floor,
        )

        vq_loss = (
            m.two_band_cfg.low_vq_weight * output.low_vq_loss
            + m.two_band_cfg.high_vq_weight * output.high_vq_loss
        )
        total = (
            full_loss
            + m.two_band_cfg.low_loss_weight * low_loss
            + m.two_band_cfg.high_loss_weight * high_loss
            + vq_loss
        )

        log_k = mx.array(math.log(32.0), dtype=mx.float32)
        marginal_weight = effective_lambda_marginal(step, cfg)
        low_marginal_gap = mx.maximum(
            mx.array(0.0, dtype=mx.float32),
            log_k - output.low_marginal_entropy,
        )
        high_marginal_gap = mx.maximum(
            mx.array(0.0, dtype=mx.float32),
            log_k - output.high_marginal_entropy,
        )
        if cfg.lambda_marginal > 0:
            total = total + marginal_weight * (
                m.two_band_cfg.low_marginal_weight * low_marginal_gap
                + m.two_band_cfg.high_marginal_weight * high_marginal_gap
            )

        low_entropy_gap = mx.maximum(
            mx.array(0.0, dtype=mx.float32),
            log_k - output.low_entropy,
        )
        high_entropy_gap = mx.maximum(
            mx.array(0.0, dtype=mx.float32),
            log_k - output.high_entropy,
        )
        if cfg.lambda_entropy > 0:
            total = total + cfg.lambda_entropy * (low_entropy_gap + high_entropy_gap)

        forward_metrics.clear()
        forward_metrics.update(
            output=output,
            loss_full=full_loss,
            loss_low=low_loss,
            loss_high=high_loss,
            vq_total=output.low_vq_loss + output.high_vq_loss,
            low_vq=output.low_vq_loss,
            high_vq=output.high_vq_loss,
            low_marginal=output.low_marginal_entropy,
            high_marginal=output.high_marginal_entropy,
            low_entropy=output.low_entropy,
            high_entropy=output.high_entropy,
            marginal_weight=mx.array(marginal_weight, dtype=mx.float32),
            full=full,
            low=low,
            high=high,
        )
        return total

    loss_fn.forward_metrics = forward_metrics  # type: ignore[attr-defined]
    return loss_fn


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    return value


def _write_checkpoint_meta(
    path: Path,
    *,
    model: TwoBandCodec,
    run_dir: Path,
) -> None:
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(
        architecture=model.architecture_name,
        split_hz=model.two_band_cfg.split_hz,
        fir_taps=model.two_band_cfg.fir_taps,
        nominal_bitrate_bps=model.nominal_bitrate_bps,
        run_dir=str(run_dir),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _checkpoint(
    model: TwoBandCodec,
    optimizer: Adam,
    run_dir: Path,
    step: int,
    data_offset: int,
) -> tuple[Path, Path]:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / f"codec_step{step}.npz"
    model.save_weights(str(weights_path))
    full_path = ckpt_dir / f"ckpt_step{step}.safetensors"
    save_full_checkpoint(
        full_path,
        step=step,
        data_off=data_offset,
        model=model,
        opt=optimizer,
    )
    _write_checkpoint_meta(full_path, model=model, run_dir=run_dir)
    return weights_path, full_path


def _zero_vq_optimizer_moments(
    optimizer: Adam,
    branch: str,
    n_codebooks: int,
) -> None:
    """Discard Adam history after externally replacing RVQ embedding rows."""
    branch_state = optimizer.state.get(branch)
    if not isinstance(branch_state, dict):
        return
    rvq_state = branch_state.get("rvq")
    if not isinstance(rvq_state, dict):
        return
    for stage_index in range(n_codebooks):
        stage_state = rvq_state.get(f"q{stage_index}")
        if not isinstance(stage_state, dict):
            continue
        embedding_state = stage_state.get("embedding")
        if not isinstance(embedding_state, dict):
            continue
        weight_state = embedding_state.get("weight")
        if not isinstance(weight_state, dict):
            continue
        for key in ("m", "v"):
            value = weight_state.get(key)
            if isinstance(value, mx.array):
                weight_state[key] = mx.zeros_like(value)


def _nonfinite_gradient_keys(grads) -> list[str]:
    bad: list[str] = []
    for key, grad in tree_flatten(grads):
        if not isinstance(grad, mx.array):
            continue
        mx.eval(grad)
        if not bool(mx.all(mx.isfinite(grad)).item()):
            bad.append(key)
            if len(bad) >= 12:
                break
    return bad


def _bootstrap_rvq_codebooks(
    codec,
    batch: mx.array,
    *,
    seed: int,
) -> int:
    """Deterministically initialize every Euclidean RVQ stage from batch residuals."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    z = codec.latent_before_rvq(batch)
    mx.eval(z)
    quantized = mx.zeros_like(z)
    initialized = 0
    for stage_index in range(codec.cfg.n_codebooks):
        stage = getattr(codec.rvq, f"q{stage_index}")
        residual = z - quantized
        projected = stage.in_proj(residual) if stage.in_proj is not None else residual
        mx.eval(projected)
        values = np.array(projected, dtype=np.float32).reshape(
            -1,
            int(projected.shape[-1]),
        )
        codebook_size = int(stage.num_embeddings)
        centers = _kmeans_centroids_numpy(values, codebook_size, rng)
        if centers.shape[0] < codebook_size:
            extra = rng.integers(0, values.shape[0], size=codebook_size - centers.shape[0])
            centers = np.vstack([centers, values[extra]])
        per_dim_std = np.maximum(np.std(values, axis=0, keepdims=True), 1e-4)
        centers = centers + 0.1 * per_dim_std * rng.standard_normal(centers.shape).astype(
            np.float32
        )
        stage.embedding["weight"] = mx.array(centers.astype(np.float32))
        z_stage, _, _, _ = stage(residual)
        mx.eval(z_stage)
        quantized = quantized + z_stage
        initialized += codebook_size
    return initialized


def _resume(
    path: Path,
    model: TwoBandCodec,
) -> tuple[int, int, dict[str, mx.array] | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise ValueError(f"missing checkpoint metadata: {meta_path}")
        meta = json.loads(meta_path.read_text())
        if meta.get("architecture") != model.architecture_name:
            raise ValueError(
                f"checkpoint architecture is {meta.get('architecture')!r}, "
                f"expected {model.architecture_name!r}"
            )
        if float(meta.get("split_hz", 0.0)) != float(model.two_band_cfg.split_hz):
            raise ValueError("checkpoint split_hz differs from current configuration")
        return int(meta["step"]) + 1, int(meta.get("data_off", 0)), dict(mx.load(str(path)))
    model.load_weights(str(path), strict=True)
    stem = path.stem
    if not stem.startswith("codec_step"):
        raise ValueError("NPZ resume filename must be codec_stepN.npz")
    return int(stem.removeprefix("codec_step")) + 1, 0, None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=330_000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--segment", type=int, default=16_000)
    p.add_argument("--data-dir", type=Path, default=Path("data/train-clean-100"))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lr-min", type=float, default=1.2e-7)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument(
        "--lr-total-steps",
        type=int,
        default=None,
        help="Cosine horizon; defaults to --steps. Use 330000 for short smoke runs.",
    )
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=2_500)
    p.add_argument("--spectrogram-every", type=int, default=2_500)
    p.add_argument("--load-audio-threads", type=int, default=8)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--vq-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vq-reset-every", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.batch < 1 or args.segment < 256:
        raise SystemExit("--steps/--batch must be positive and --segment must be >= 256")
    if args.lr <= 0 or not (0 < args.lr_min <= args.lr):
        raise SystemExit("require 0 < --lr-min <= --lr")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"runs/two_band_937bps_fresh_cosine_{timestamp}")
    ).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    lr_total_steps = args.steps if args.lr_total_steps is None else args.lr_total_steps
    if lr_total_steps < args.steps:
        raise SystemExit("--lr-total-steps must be >= --steps")
    stft_scales = FAST_STFT_SCALES if args.fast else Config().stft_scales
    stft_weights = None if args.fast else LIBRI_STFT_SCALE_WEIGHTS
    cfg = Config(
        sample_rate=16_000,
        segment=args.segment,
        batch=args.batch,
        load_audio_threads=args.load_audio_threads,
        prefetch_audio=False,
        steps=lr_total_steps,
        lr=args.lr,
        lr_schedule="cosine",
        lr_min_ratio=args.lr_min / args.lr,
        lr_warmup_steps=min(args.warmup_steps, lr_total_steps),
        seed=args.seed,
        enc_channels=(24, 32, 48, 64, 96, 128, 192, 256),
        latent_dim=512,
        latent_temporal_depth=2,
        latent_temporal_post_depth=2,
        self_attention_depth=1,
        self_attention_post_depth=0,
        self_attention_heads=8,
        decoder_refine_depth=1,
        decoder_refine_gain=0.1,
        decoder_band_heads=1,
        post_lavasr_depth=2,
        post_lavasr_channels=24,
        post_lavasr_kernel=15,
        post_lavasr_gain=0.03,
        post_lavasr_highpass=True,
        activation="snake_beta",
        decoder_activation="snake_beta",
        decoder_upsample="repeat_conv",
        rvq_code_dim=8,
        vq_cosine=False,
        vq_ema_decay=0.99,
        lambda_time=1.0,
        lambda_stft=0.5,
        lambda_stft_grad=0.0 if args.fast else 0.1,
        lambda_stft_cos=0.0,
        lambda_sc=1.0,
        lambda_complex_stft=0.1,
        lambda_mag_l1=0.0 if args.fast else 0.15,
        lambda_mel_l1=0.0 if args.fast else 0.12,
        lambda_mel_l2=0.0,
        lambda_vq=5.0,
        lambda_marginal=0.35,
        marginal_tau=0.04,
        marginal_boost_steps=24_000,
        marginal_boost_mult=2.5,
        lambda_entropy=0.0,
        lambda_cos=0.15,
        stft_ramp_steps=8_000,
        stft_ramp_start_frac=0.25,
        stft_scales=stft_scales,
        stft_scale_weights=stft_weights,
        stft_hf_emphasis=1.0,
        mel_n_fft=1024,
        mel_hop=256,
        n_mels=80,
        mel_fmin=0.0,
        mel_fmax=8_000.0,
        grad_clip_norm=args.grad_clip,
        grad_accum_steps=1,
        vq_reset_every=args.vq_reset_every,
        data_dir=None if args.synthetic else args.data_dir.resolve(),
        use_bf16=args.bf16,
    )
    tb_cfg = TwoBandCodecConfig(
        split_hz=3_500.0,
        fir_taps=127,
        high_channels=DEFAULT_HIGH_CHANNELS,
        high_latent_dim=256,
        high_input_gain=1.0,
        high_loss_floor=0.1,
        low_loss_weight=0.5,
        high_loss_weight=0.1,
        low_vq_weight=5.0,
        high_vq_weight=0.0,
        low_marginal_weight=1.0,
        high_marginal_weight=0.0,
        low_codebooks=2,
        high_codebooks=1,
        codebook_size=32,
    )

    mx.random.seed(cfg.seed)
    model = TwoBandCodec(cfg, tb_cfg)
    if cfg.use_bf16:
        _cast_module_floats(model, mx.bfloat16)
        mx.eval(model.parameters())

    start_step = 0
    data_offset = 0
    resume_flat = None
    if args.resume is not None:
        start_step, data_offset, resume_flat = _resume(args.resume.resolve(), model)
        if start_step >= args.steps:
            raise SystemExit(f"checkpoint already reached step {start_step - 1}")

    lr_schedule = build_lr_schedule(cfg, start_step=0)
    optimizer = Adam(lr_schedule)
    optimizer.init(model.parameters())
    if start_step > 0:
        optimizer.state["step"] = mx.array(int(start_step), dtype=mx.uint64)
        _refresh_optimizer_schedules(optimizer)
    if resume_flat is not None:
        load_full_checkpoint_into(resume_flat, model=model, opt=optimizer, label="two-band resume")
        mx.eval(model.parameters(), optimizer.state)

    mel_fb = None
    if cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0:
        mel_fb = mel_filterbank_mx(
            cfg.sample_rate,
            cfg.mel_n_fft,
            cfg.n_mels,
            cfg.mel_fmin,
            cfg.mel_fmax or cfg.sample_rate / 2,
        )

    audio_paths: list[Path] | None = None
    if cfg.data_dir is not None:
        audio_paths = _collect_audio_paths(cfg.data_dir)
        if not audio_paths:
            raise SystemExit(f"no audio files under {cfg.data_dir}")

    bootstrap_batch: mx.array | None = None
    if args.vq_bootstrap and start_step == 0:
        if cfg.use_bf16:
            _cast_module_floats(model, mx.float32)
            mx.eval(model.parameters())
        bootstrap_batch = (
            _load_audio_batch(cfg, audio_paths, data_offset)
            if audio_paths is not None
            else synth_batch(cfg, key=cfg.seed * 10_007)
        )
        low_target, high_target = model.filterbank.split(bootstrap_batch)
        low_reset = _bootstrap_rvq_codebooks(
            model.low_codec,
            low_target,
            seed=cfg.seed + 101,
        )
        high_reset = _bootstrap_rvq_codebooks(
            model.high_codec,
            high_target * tb_cfg.high_input_gain,
            seed=cfg.seed + 202,
        )
        if cfg.use_bf16:
            _cast_module_floats(model, mx.bfloat16)
        mx.eval(model.parameters())
        print(f"[vq-bootstrap] low={low_reset} high={high_reset}", flush=True)

    run_config = {
        "architecture": model.architecture_name,
        "base_config": _jsonable(asdict(cfg)),
        "two_band_config": _jsonable(asdict(tb_cfg)),
        "nominal_bitrate_bps": model.nominal_bitrate_bps,
        "parameter_count": _tree_n_params(model.parameters()),
        "data_files": len(audio_paths) if audio_paths is not None else 0,
        "command": " ".join(sys.argv),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (run_dir / "train_command.txt").write_text(" ".join(sys.argv) + "\n")

    log_path = run_dir / "train.tsv"
    if not log_path.is_file():
        log_path.write_text(
            "step\tloss\tloss_full\tloss_low\tloss_high\tfull_l1\tlow_l1\thigh_l1\t"
            "low_vq\thigh_vq\tlow_marginal\thigh_marginal\tfull_cos\tlow_cos\thigh_cos\t"
            "lr\tms_per_step\n"
        )

    params_m = _tree_n_params(model.parameters()) / 1e6
    print(
        f"TwoBandCodec split=3500Hz FIR={tb_cfg.fir_taps} "
        f"bitrate={model.nominal_bitrate_bps:.1f}bps params={params_m:.2f}M "
        f"data={len(audio_paths) if audio_paths is not None else 'synthetic'} "
        f"steps={start_step}..{args.steps - 1} lr_horizon={cfg.steps} run={run_dir}",
        flush=True,
    )

    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("[signal] stop requested; finishing current step", flush=True)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    last_step = start_step - 1
    last_batch: mx.array | None = None
    started = time.time()
    for step in range(start_step, args.steps):
        step_started = time.time()
        if bootstrap_batch is not None and step == start_step:
            batch = bootstrap_batch
        else:
            batch = (
                _load_audio_batch(cfg, audio_paths, data_offset)
                if audio_paths is not None
                else synth_batch(cfg, key=step * 1_000_003 + cfg.seed * 10_007)
            )
        data_offset += cfg.batch
        loss_fn = make_two_band_train_fn(model, cfg, batch, step, mel_fb)
        value_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = value_and_grad(model)
        _eval_loss_and_grad_tree(loss, grads)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value) or _grad_tree_any_nonfinite(grads):
            bad_keys = _nonfinite_gradient_keys(grads)
            raise RuntimeError(
                f"non-finite loss or gradient at step {step}: "
                f"loss={loss_value} tensors={bad_keys}"
            )
        if cfg.grad_clip_norm > 0:
            grads = clip_gradients_global_norm(grads, cfg.grad_clip_norm)
        optimizer.update(model, grads)
        last_step = step
        last_batch = batch

        metrics = loss_fn.forward_metrics
        output = metrics["output"]
        if 0.0 < cfg.vq_ema_decay < 1.0:
            update_vq_ema_codebooks(model.low_codec, output.low_target, model.low_cfg)
            update_vq_ema_codebooks(model.high_codec, output.high_target, model.high_cfg)
        if cfg.vq_reset_every > 0 and step > 0 and step % cfg.vq_reset_every == 0:
            low_reset, _ = vq_reset_dead_codes(model.low_codec, output.low_target, model.low_cfg)
            high_reset, _ = vq_reset_dead_codes(model.high_codec, output.high_target, model.high_cfg)
            if low_reset or high_reset:
                if low_reset:
                    _zero_vq_optimizer_moments(
                        optimizer,
                        "low_codec",
                        model.low_cfg.n_codebooks,
                    )
                if high_reset:
                    _zero_vq_optimizer_moments(
                        optimizer,
                        "high_codec",
                        model.high_cfg.n_codebooks,
                    )
                print(f"[vq-reset] low={low_reset} high={high_reset}", flush=True)

        should_log = step % args.log_every == 0 or step == args.steps - 1
        if should_log:
            scalars = [
                metrics["loss_full"],
                metrics["loss_low"],
                metrics["loss_high"],
                metrics["low_vq"],
                metrics["high_vq"],
                metrics["low_marginal"],
                metrics["high_marginal"],
                metrics["full"]["time"],
                metrics["low"]["time"],
                metrics["high"]["time"],
                metrics["full"]["cosine"],
                metrics["low"]["cosine"],
                metrics["high"]["cosine"],
            ]
            mx.eval(*scalars, *output.low_indices, *output.high_indices)
            lr_value = lr_schedule(optimizer.step) if callable(lr_schedule) else mx.array(lr_schedule)
            mx.eval(lr_value)
            ms = (time.time() - step_started) * 1000.0
            low_util = _format_vq_util(output.low_indices, (32, 32)).strip()
            high_util = _format_vq_util(output.high_indices, (32,)).strip()
            print(
                f"step {step:6d}/{args.steps} loss={loss_value:.5f} "
                f"full={float(metrics['loss_full'].item()):.4f} "
                f"low={float(metrics['loss_low'].item()):.4f} "
                f"high={float(metrics['loss_high'].item()):.4f} "
                f"vq={float(metrics['low_vq'].item()):.4f}/{float(metrics['high_vq'].item()):.4f} "
                f"marg={float(metrics['low_marginal'].item()):.3f}/"
                f"{float(metrics['high_marginal'].item()):.3f} "
                f"cos={100.0 * float(metrics['full']['cosine'].item()):.1f}% "
                f"low[{low_util}] high[{high_util}] lr={float(lr_value.item()):.2e} "
                f"{ms:.0f}ms",
                flush=True,
            )
            with log_path.open("a") as log:
                log.write(
                    f"{step}\t{loss_value:.8f}\t"
                    f"{float(metrics['loss_full'].item()):.8f}\t"
                    f"{float(metrics['loss_low'].item()):.8f}\t"
                    f"{float(metrics['loss_high'].item()):.8f}\t"
                    f"{float(metrics['full']['time'].item()):.8f}\t"
                    f"{float(metrics['low']['time'].item()):.8f}\t"
                    f"{float(metrics['high']['time'].item()):.8f}\t"
                    f"{float(metrics['low_vq'].item()):.8f}\t"
                    f"{float(metrics['high_vq'].item()):.8f}\t"
                    f"{float(metrics['low_marginal'].item()):.8f}\t"
                    f"{float(metrics['high_marginal'].item()):.8f}\t"
                    f"{float(metrics['full']['cosine'].item()):.8f}\t"
                    f"{float(metrics['low']['cosine'].item()):.8f}\t"
                    f"{float(metrics['high']['cosine'].item()):.8f}\t"
                    f"{float(lr_value.item()):.10g}\t{ms:.3f}\n"
                )

        checkpoint_due = (
            args.checkpoint_every > 0
            and step > 0
            and (step % args.checkpoint_every == 0 or step == args.steps - 1)
        )
        if checkpoint_due:
            paths = _checkpoint(model, optimizer, run_dir, step, data_offset)
            print(f"[checkpoint] {paths[0]} {paths[1]}", flush=True)

        spectrogram_due = (
            args.spectrogram_every > 0
            and step > 0
            and (step % args.spectrogram_every == 0 or step == args.steps - 1)
        )
        if spectrogram_due:
            output = model(batch[:1])
            mx.eval(output.reconstruction)
            stem = run_dir / "spectrograms" / f"step_{step:08d}"
            save_spectrogram_png(
                batch[0, :, 0],
                output.reconstruction[0, :, 0],
                cfg.sample_rate,
                stem.with_suffix(".png"),
                step,
            )
            save_reconstruction_wavs(
                batch[0, :, 0],
                output.reconstruction[0, :, 0],
                cfg.sample_rate,
                stem,
            )

        if stop_requested:
            break

    if last_step >= start_step:
        final_paths = _checkpoint(model, optimizer, run_dir, last_step, data_offset)
        print(f"[final] {final_paths[0]} {final_paths[1]}", flush=True)
    elapsed = time.time() - started
    status = {
        "last_step": last_step,
        "requested_steps": args.steps,
        "lr_total_steps": cfg.steps,
        "stopped": bool(stop_requested),
        "elapsed_seconds": elapsed,
        "data_offset": data_offset,
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(
        f"done step={last_step} stopped={stop_requested} elapsed={elapsed / 60.0:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
