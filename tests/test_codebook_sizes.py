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


def test_mlx_custom_cosine_metric_matches_reference():
    _load_train_mlx()
    import mlx.core as mx
    from sirencodec.mlx.kernels import batch_mean_cosine_metric
    from sirencodec.mlx.train import batch_mean_cosine

    x = mx.random.normal(shape=(3, 256, 1))
    y = mx.random.normal(shape=(3, 256, 1))
    ref = batch_mean_cosine(x, y)
    got = batch_mean_cosine_metric(x, y)
    mx.eval(ref, got)
    assert abs(float(ref.item()) - float(got.item())) < 1e-5


def test_mlx_spectral_subset_and_large_scale_cycling():
    tm = _load_train_mlx()
    import mlx.core as mx

    pred = mx.reshape(mx.arange(8, dtype=mx.float32), (4, 2, 1))
    tgt = -pred
    p, q = tm._spectral_loss_batch(pred, tgt, step=0, max_items=2)
    mx.eval(p, q)
    assert np.array_equal(np.array(p[:, 0, 0]), np.array([0.0, 2.0], dtype=np.float32))
    assert np.array_equal(np.array(q[:, 0, 0]), np.array([0.0, -2.0], dtype=np.float32))

    p2, _ = tm._spectral_loss_batch(pred, tgt, step=1, max_items=2)
    mx.eval(p2)
    assert np.array_equal(np.array(p2[:, 0, 0]), np.array([4.0, 6.0], dtype=np.float32))

    cfg = tm.Config(
        stft_scales=((512, 128), (2048, 512), (4096, 1024)),
        stft_scale_weights=(0.5, 2.0, 4.0),
        stft_large_min_fft=4096,
        stft_large_every=4,
    )
    scales0, weights0 = tm._active_stft_scales(cfg, 0)
    assert scales0 == cfg.stft_scales
    assert weights0 == cfg.stft_scale_weights
    scales1, weights1 = tm._active_stft_scales(cfg, 1)
    assert scales1 == ((512, 128), (2048, 512))
    assert weights1 == (0.5, 2.0)


def test_mlx_waveform_hf_losses_behave():
    tm = _load_train_mlx()
    import mlx.core as mx
    from sirencodec.mlx.losses import high_frequency_stft_terms

    x = mx.random.normal(shape=(2, 256, 1))
    shifted = mx.concatenate([x[:, 3:, :], x[:, :3, :]], axis=1)
    same_sisdr = tm.batch_neg_log_si_sdr(x, x)
    shifted_sisdr = tm.batch_neg_log_si_sdr(x, shifted)
    mx.eval(same_sisdr, shifted_sisdr)
    assert float(same_sisdr.item()) < float(shifted_sisdr.item())

    z = mx.zeros((1, 256, 1))
    alt = mx.reshape((mx.arange(256, dtype=mx.float32) % 2) * 2.0 - 1.0, (1, 256, 1)) * 0.25
    pre_zero = tm.batch_preemph_l1(z, z)
    pre_alt = tm.batch_preemph_l1(z, alt)
    hf_same, _ = high_frequency_stft_terms(
        alt,
        alt,
        ((128, 32),),
        sample_rate=16_000,
        min_hz=2000.0,
    )
    hf_missing, _ = high_frequency_stft_terms(
        z,
        alt,
        ((128, 32),),
        sample_rate=16_000,
        min_hz=2000.0,
    )
    mx.eval(pre_zero, pre_alt, hf_same, hf_missing)
    assert float(pre_zero.item()) == pytest.approx(0.0)
    assert float(pre_alt.item()) > 0.0
    assert float(hf_missing.item()) > float(hf_same.item())


def test_mlx_under1k_hf_config_forward_backward_is_finite():
    tm = _load_train_mlx()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    cfg = tm.Config(
        batch=1,
        segment=512,
        enc_channels=(8, 12, 16, 16, 16, 16, 16, 16),
        latent_dim=32,
        n_codebooks=2,
        codebook_sizes=(256, 128),
        rvq_code_dim=0,
        pre_vq_layernorm=True,
        latent_temporal_depth=1,
        latent_temporal_post_depth=0,
        self_attention_depth=1,
        self_attention_post_depth=1,
        self_attention_heads=4,
        decoder_refine_blocks_per_scale=1,
        stft_scales=((128, 32),),
        stft_scale_weights=(1.0,),
        lambda_stft=0.1,
        lambda_sc=0.1,
        lambda_complex_stft=0.05,
        lambda_hf_under=0.1,
        lambda_hf_sc=0.1,
        hf_min_hz=2000.0,
        lambda_sisdr=0.1,
        lambda_preemph=0.1,
        lambda_ae_anchor_time=0.1,
        lambda_ae_anchor_cos=0.05,
        lambda_mel_l1=0.0,
        lambda_vq=0.1,
        lambda_marginal=0.0,
        lambda_cos=0.1,
    )
    assert tm.nominal_rvq_kbps(cfg) * 1000.0 <= 1000.0
    model = tm.MLXCodec(cfg)
    x = mx.random.normal(shape=(1, 512, 1)) * 0.05
    loss_fn = tm.make_train_fn(model, cfg, x, 0, None)
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    flat_grads = [g for _, g in tree_flatten(grads)]
    mx.eval(loss, *flat_grads)
    assert np.isfinite(float(loss.item()))
    for g in flat_grads:
        bad = mx.any(mx.logical_or(mx.isnan(g), mx.isinf(g)))
        mx.eval(bad)
        assert not bool(bad.item())


def test_mlx_state_space_shape_and_backward_is_finite():
    _load_train_mlx()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from sirencodec.mlx.codec import StateSpace1D

    layer = StateSpace1D(32, state_dim=4, expand=1, bidirectional=True)
    x_short = mx.random.normal(shape=(2, 1, 32))
    x_long = mx.random.normal(shape=(2, 500, 32))
    y_short = layer(x_short)
    y_long = layer(x_long)
    mx.eval(y_short, y_long)
    assert y_short.shape == x_short.shape
    assert y_long.shape == x_long.shape

    def loss_fn(m):
        y = m(x_long)
        return mx.mean(y * y)

    loss, grads = nn.value_and_grad(layer, loss_fn)(layer)
    flat_grads = [g for _, g in tree_flatten(grads)]
    mx.eval(loss, *flat_grads)
    assert np.isfinite(float(loss.item()))
    for g in flat_grads:
        bad = mx.any(mx.logical_or(mx.isnan(g), mx.isinf(g)))
        mx.eval(bad)
        assert not bool(bad.item())


def test_mlx_state_space_rejects_invalid_attention_and_causal_combinations():
    tm = _load_train_mlx()
    with pytest.raises(ValueError, match="mutually exclusive"):
        tm.MLXCodec(tm.Config(self_attention_depth=1, state_space_depth=1))
    with pytest.raises(ValueError, match="causal"):
        tm.MLXCodec(
            tm.Config(
                self_attention_depth=0,
                self_attention_post_depth=0,
                state_space_depth=1,
                state_space_bidirectional=True,
                causal=True,
            )
        )
