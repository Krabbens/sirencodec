"""Tests for per-stage RVQ codebook sizes and MLX bitstream helpers."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_infer_mlx():
    name = "infer_mlx_t"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "infer_mlx.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_train_mlx():
    pytest.importorskip("mlx.core")
    name = "train_mlx_t"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "train_mlx.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_infer_effective_codebook_sizes_cfg():
    im = _load_infer_mlx()

    class Cfg:
        pass

    u = Cfg()
    u.codebook_sizes = None
    u.codebook_size = 64
    u.n_codebooks = 3
    assert im._effective_codebook_sizes_cfg(u) == (64, 64, 64)

    v = Cfg()
    v.codebook_sizes = (256, 128, 64)
    v.codebook_size = 999
    v.n_codebooks = 3
    assert im._effective_codebook_sizes_cfg(v) == (256, 128, 64)


def test_infer_nominal_bitrate_sum_log2():
    im = _load_infer_mlx()

    class Cfg:
        pass

    c = Cfg()
    c.codebook_sizes = (128, 64)
    c.codebook_size = 128
    c.n_codebooks = 2
    c.sample_rate = 16_000
    c.enc_channels = (24, 32, 48, 64, 96, 128, 192, 256)
    bps = im.nominal_bitrate_bps(c)
    stride = 2 ** len(c.enc_channels)
    want = (math.log2(128) + math.log2(64)) * (16_000 / stride)
    assert abs(bps - want) < 1e-5


def test_infer_pack_v1_uniform_and_v2_mixed():
    im = _load_infer_mlx()
    codes_u = [np.arange(5, dtype=np.int16), np.arange(5, dtype=np.int16)]
    blob_u = im.pack_vq_bitstream(
        codes_u,
        codebook_sizes=(64, 64),
        sample_rate=16_000,
        audio_samples=800,
    )
    assert blob_u[0:4] == im._MAGIC
    assert blob_u[4] == 1
    assert im._vq_packed_header_len(blob_u) == im._HEADER_SIZE

    codes_m = [np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.int16)]
    blob_m = im.pack_vq_bitstream(
        codes_m,
        codebook_sizes=(128, 32),
        sample_rate=16_000,
        audio_samples=800,
    )
    assert blob_m[4] == 2
    assert im._vq_packed_header_len(blob_m) == 32 + 4


def test_infer_pack_binary_rvq_is_one_bit_per_stage():
    im = _load_infer_mlx()
    n_frames = 5
    codes = [np.array([q & 1, 1, 0, q & 1, 1], dtype=np.int32) for q in range(12)]
    blob = im.pack_vq_bitstream(
        codes,
        codebook_sizes=(2,) * 12,
        sample_rate=16_000,
        audio_samples=1280,
    )
    assert blob[4] == 1
    assert blob[23] == 1
    assert im._vq_packed_header_len(blob) == im._HEADER_SIZE
    assert len(blob) == im._HEADER_SIZE + math.ceil((12 * n_frames) / 8)


def test_infer_pack_v3_scalar_width_stream():
    im = _load_infer_mlx()
    codes = [np.arange(12, dtype=np.int32).reshape(4, 3) % 4]
    blob = im.pack_vq_bitstream(
        codes,
        codebook_sizes=(4,),
        sample_rate=24_000,
        audio_samples=1024,
    )
    assert blob[4] == 3
    assert im._vq_packed_header_len(blob) == 36
    assert len(blob) == 36 + math.ceil((4 * 3 * 2) / 8)


def test_train_mlx_parse_and_effective_and_nominal():
    tm = _load_train_mlx()
    assert tm.parse_codebook_sizes_arg("256, 128 , 64") == (256, 128, 64)
    cfg = tm.Config(
        n_codebooks=3,
        codebook_size=128,
        codebook_sizes=(256, 128, 64),
        enc_channels=(24, 32, 48, 64, 96, 128, 192, 256),
    )
    assert tm.effective_codebook_sizes(cfg) == (256, 128, 64)
    kbps = tm.nominal_rvq_kbps(cfg)
    st = tm.encoder_time_stride(cfg)
    want = (math.log2(256) + math.log2(128) + math.log2(64)) * (16_000 / st) / 1000.0
    assert abs(kbps - want) < 1e-6

    cfg_u = tm.Config(n_codebooks=2, codebook_size=32)
    assert tm.effective_codebook_sizes(cfg_u) == (32, 32)

    with pytest.raises(ValueError):
        bad = tm.Config(n_codebooks=2, codebook_size=32, codebook_sizes=(32,))
        tm.effective_codebook_sizes(bad)


def test_train_mlx_forward_variable_k():
    tm = _load_train_mlx()
    import mlx.core as mx

    ch = (8, 16, 16, 16, 16, 16, 16, 16)
    cfg = tm.Config(
        n_codebooks=2,
        codebook_sizes=(32, 16),
        batch=2,
        segment=512,
        latent_dim=32,
        enc_channels=ch,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
    )
    model = tm.MLXCodec(cfg)
    x = mx.random.normal(shape=(2, 512, 1))
    y, vq, ent, marg, idx = model.forward_full(x)
    mx.eval(y, vq, ent, marg)
    assert y.shape == x.shape
    assert idx is not None
    assert len(idx) == 2


def test_train_mlx_forward_binary_rvq_returns_int32_indices():
    tm = _load_train_mlx()
    import mlx.core as mx

    ch = (8, 16, 16, 16, 16, 16, 16, 16)
    cfg = tm.Config(
        n_codebooks=12,
        codebook_size=2,
        batch=1,
        segment=512,
        latent_dim=32,
        enc_channels=ch,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
    )
    model = tm.MLXCodec(cfg)
    x = mx.random.normal(shape=(1, 512, 1))
    y, vq, ent, marg, idx = model.forward_full(x)
    mx.eval(y, vq, ent, marg, *idx)
    assert y.shape == x.shape
    assert idx is not None
    assert len(idx) == 12
    assert all(ix.dtype == mx.int32 for ix in idx)


def test_train_mlx_turboquant_forward_returns_coordinate_indices():
    tm = _load_train_mlx()
    import mlx.core as mx

    cfg = tm.Config(
        quantizer="turboquant",
        turboquant_bits=2,
        turboquant_code_dim=4,
        batch=1,
        segment=64,
        sample_rate=16_000,
        latent_dim=8,
        enc_channels=(4, 8),
        latent_temporal_depth=0,
        latent_temporal_post_depth=0,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
    )
    model = tm.MLXCodec(cfg)
    x = mx.random.normal(shape=(1, 64, 1))
    y, vq, ent, marg, idx = model.forward_full(x)
    mx.eval(y, vq, ent, marg, *idx)
    assert y.shape == x.shape
    assert idx is not None
    assert len(idx) == 1
    assert idx[0].dtype == mx.int32
    assert idx[0].shape == (1, 16, 4)
    assert tm.effective_codebook_sizes(cfg) == (4,)
    assert tm.nominal_rvq_kbps(cfg) == pytest.approx((2 * 4 * (16_000 / 4)) / 1000.0)


def test_train_mlx_lux_vocos_decoder_outputs_feature_frames():
    tm = _load_train_mlx()
    import mlx.core as mx

    cfg = tm.Config(
        quantizer="turboquant",
        turboquant_bits=2,
        turboquant_code_dim=4,
        decoder_backend="lux_vocos",
        sample_rate=24_000,
        batch=1,
        segment=64,
        latent_dim=8,
        enc_channels=(4, 8),
        latent_temporal_depth=0,
        latent_temporal_post_depth=0,
        decoder_refine_depth=1,
        lux_vocos_feature_dim=100,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
    )
    model = tm.MLXCodec(cfg)
    x = mx.random.normal(shape=(1, 64, 1))
    y, vq, ent, marg, idx = model.forward_full(x)
    mx.eval(y, vq, ent, marg, *idx)
    assert y.shape == (1, 16, 100)
    assert idx is not None
    assert idx[0].shape == (1, 16, 4)


def test_train_mlx_binary_rvq_tie_breaks_to_lower_int32_index():
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from sirencodec.mlx.codec import VectorQuantizerStage

    cfg = _load_train_mlx().Config(
        latent_dim=8,
        n_codebooks=1,
        codebook_size=2,
        vq_cosine=False,
        rvq_code_dim=0,
    )
    stage = VectorQuantizerStage(cfg, 2)
    stage.embedding["weight"] = mx.zeros_like(stage.embedding["weight"])
    residual = mx.zeros((1, 4, 8), dtype=mx.float32)
    _, _, _, idx = stage(residual)
    mx.eval(idx)
    assert idx.dtype == mx.int32
    assert np.array(idx).tolist() == [[0, 0, 0, 0]]


def test_train_mlx_binary_usage_loss_uses_hard_assignments():
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from sirencodec.mlx.codec import VectorQuantizerStage

    cfg = _load_train_mlx().Config(
        latent_dim=8,
        n_codebooks=1,
        codebook_size=2,
        vq_cosine=False,
        rvq_code_dim=0,
        lambda_binary_usage=1.0,
        binary_usage_tau=0.04,
    )
    stage = VectorQuantizerStage(cfg, 2)
    stage.embedding["weight"] = mx.zeros_like(stage.embedding["weight"])
    residual = mx.zeros((1, 4, 8), dtype=mx.float32)
    _, vq_loss, _, idx = stage(residual)
    mx.eval(vq_loss, idx)
    assert idx.dtype == mx.int32
    assert np.array(idx).tolist() == [[0, 0, 0, 0]]
    assert float(np.array(vq_loss)) > 0.49


def test_mlx_init_from_loads_compatible_weights_for_binary_rvq(tmp_path):
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    from sirencodec.mlx.codec import MLXCodec
    from sirencodec.mlx.train import load_model_checkpoint_weights

    ch = (8, 16, 16, 16, 16, 16, 16, 16)
    base_cfg = _load_train_mlx().Config(
        n_codebooks=3,
        codebook_size=32,
        batch=1,
        segment=512,
        latent_dim=32,
        enc_channels=ch,
    )
    seed = MLXCodec(base_cfg)
    ck = tmp_path / "codec_step329999.npz"
    seed.save_weights(str(ck))

    binary_cfg = _load_train_mlx().Config(
        n_codebooks=12,
        codebook_size=2,
        batch=1,
        segment=512,
        latent_dim=32,
        enc_channels=ch,
    )
    model = MLXCodec(binary_cfg)
    load_model_checkpoint_weights(model, ck, label="init-from")
    mx.eval(model.parameters())

    assert np.allclose(np.array(model.encoder.out.weight), np.array(seed.encoder.out.weight))
    assert np.array(model.rvq.q0.embedding.weight).shape[0] == 2
    assert np.array(model.rvq.q11.embedding.weight).shape[0] == 2


def test_mlx_rvq_distill_only_forward_and_mask():
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from sirencodec.mlx.codec import MLXCodec
    from sirencodec.mlx.train import make_train_fn, mask_rvq_gradients, rvq_bottleneck_latent

    ch = (8, 16, 16, 16, 16, 16, 16, 16)
    base = dict(
        batch=1,
        segment=512,
        latent_dim=32,
        enc_channels=ch,
        lambda_time=0.0,
        lambda_stft=0.0,
        lambda_sc=0.0,
        lambda_complex_stft=0.0,
        lambda_mel_l1=0.0,
        lambda_cos=0.0,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
        lambda_vq=0.1,
    )
    teacher = MLXCodec(_load_train_mlx().Config(n_codebooks=3, codebook_size=32, **base))
    student_cfg = _load_train_mlx().Config(
        n_codebooks=12,
        codebook_size=2,
        lambda_rvq_distill=1.0,
        rvq_distill_only=True,
        **base,
    )
    student = MLXCodec(student_cfg)
    x = mx.random.normal(shape=(1, 512, 1))
    target, _, _, _, _ = rvq_bottleneck_latent(teacher, x)
    mx.eval(target)

    loss_fn = make_train_fn(
        student,
        student_cfg,
        x,
        0,
        rvq_distill_target=target,
        rvq_distill_weight=1.0,
        rvq_distill_only=True,
    )
    loss, grads = nn.value_and_grad(student, loss_fn)(student)
    mx.eval(loss)
    fm = loss_fn.forward_metrics
    assert fm["idx"] is not None
    assert len(fm["idx"]) == 12
    assert isinstance(fm["l_rvq_distill"], mx.array)
    assert bool(fm["rvq_distill_only"]) is True

    masked = mask_rvq_gradients(grads)
    mx.eval(masked)
    flat = dict(tree_flatten(masked))
    assert np.allclose(np.array(flat["encoder.out.weight"]), 0.0)
    assert np.any(np.array(flat["rvq.q0.embedding.weight"]) != 0.0)
