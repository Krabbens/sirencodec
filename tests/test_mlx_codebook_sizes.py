"""Tests for per-stage RVQ codebook sizes (train_mlx + infer_mlx bitstream)."""
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
