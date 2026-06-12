"""Tests for the conditioned two-band MLX codec."""
from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from sirencodec.config import Config
from sirencodec.mlx.train_two_band import make_two_band_train_fn
from sirencodec.mlx.two_band import (
    FixedComplementaryFIR,
    TwoBandCodec,
    TwoBandCodecConfig,
    compact_high_channels,
    nominal_two_band_bitrate_bps,
    stack_indices,
)


def _small_config() -> tuple[Config, TwoBandCodecConfig]:
    base = Config(
        sample_rate=16_000,
        segment=512,
        batch=1,
        enc_channels=(4, 8),
        latent_dim=16,
        latent_temporal_depth=0,
        latent_temporal_post_depth=0,
        self_attention_depth=0,
        self_attention_post_depth=0,
        decoder_refine_depth=0,
        decoder_band_heads=1,
        post_lavasr_depth=0,
        speech_control_depth=0,
        harmonic_source=False,
        rvq_code_dim=4,
        lambda_time=1.0,
        lambda_stft=0.0,
        lambda_stft_grad=0.0,
        lambda_stft_cos=0.0,
        lambda_sc=0.0,
        lambda_complex_stft=0.0,
        lambda_mag_l1=0.0,
        lambda_mel_l1=0.0,
        lambda_mel_l2=0.0,
        lambda_cos=0.0,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
        use_bf16=False,
    )
    two_band = TwoBandCodecConfig(
        high_channels=compact_high_channels(base),
        high_latent_dim=8,
    )
    return base, two_band


def test_filterbank_is_exactly_complementary():
    x = mx.random.normal(shape=(2, 1024, 1))
    filterbank = FixedComplementaryFIR(cutoff_hz=3_500.0, num_taps=127)
    low, high = filterbank.split(x)
    error = mx.max(mx.abs(x - low - high))
    mx.eval(error)
    assert float(error.item()) < 1e-5


def test_filterbank_separates_low_and_high_sines():
    sample_rate = 16_000
    length = 4096
    t = mx.arange(length, dtype=mx.float32) / sample_rate
    low_sine = mx.sin(2.0 * math.pi * 1_000.0 * t)[None, :, None]
    high_sine = mx.sin(2.0 * math.pi * 6_000.0 * t)[None, :, None]
    filterbank = FixedComplementaryFIR(
        sample_rate=sample_rate,
        cutoff_hz=3_500.0,
        num_taps=127,
    )
    low_from_low, high_from_low = filterbank.split(low_sine)
    low_from_high, high_from_high = filterbank.split(high_sine)
    energies = [
        mx.mean(low_from_low**2),
        mx.mean(high_from_low**2),
        mx.mean(low_from_high**2),
        mx.mean(high_from_high**2),
    ]
    mx.eval(*energies)
    assert float(energies[0].item()) > 20.0 * float(energies[1].item())
    assert float(energies[3].item()) > 20.0 * float(energies[2].item())


def test_nominal_default_bitrate_is_937_5_bps():
    assert nominal_two_band_bitrate_bps(Config()) == pytest.approx(937.5)


def test_forward_and_index_only_decode_preserve_shapes():
    base, two_band = _small_config()
    model = TwoBandCodec(base, two_band)
    x = mx.random.normal(shape=(1, base.segment, 1))
    output = model(x)
    decoded = model.decode_indices(
        output.low_indices,
        output.high_indices,
        base.segment,
    )
    low_indices = stack_indices(output.low_indices)
    high_indices = stack_indices(output.high_indices)
    mx.eval(output.reconstruction, decoded, low_indices, high_indices)

    assert output.reconstruction.shape == x.shape
    assert decoded.shape == x.shape
    assert low_indices.shape == (1, 128, 2)
    assert high_indices.shape == (1, 128, 1)
    assert float(mx.max(mx.abs(decoded - output.reconstruction)).item()) < 1e-4


def test_joint_loss_has_finite_gradients():
    base, two_band = _small_config()
    model = TwoBandCodec(base, two_band)
    x = mx.random.normal(shape=(1, base.segment, 1))
    loss_fn = make_two_band_train_fn(model, base, x, step=0, mel_fb=None)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = value_and_grad(model)
    flat_grads = []

    def collect(tree):
        if isinstance(tree, dict):
            for value in tree.values():
                collect(value)
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                collect(value)
        else:
            flat_grads.append(tree)

    collect(grads)
    mx.eval(loss, *flat_grads)
    assert math.isfinite(float(loss.item()))
    assert flat_grads
    assert all(bool(mx.all(mx.isfinite(grad)).item()) for grad in flat_grads)
