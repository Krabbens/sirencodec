from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch

import sirencodec
from sirencodec.config import (
    Config,
    argparse_defaults_from_config,
    effective_codebook_sizes,
    encoder_time_stride,
    nominal_rvq_kbps,
    parse_positive_int_list_arg,
    parse_stft_scale_weights_arg,
    parse_stft_scales_arg,
)
from sirencodec.cuda.codec import (
    CUDACodec,
    MultiPeriodWaveDiscriminator,
    MultiScaleWaveDiscriminator,
    ResidualVectorQuantizer,
    build_wave_discriminator,
)
from sirencodec.cuda.data import dataset_dir_candidates
from sirencodec.cuda.train import (
    _codec_eval_score,
    _active_stft_scales,
    _spectral_loss_batch,
    batch_preemph_l1,
    batch_neg_log_si_sdr,
    compute_loss,
    config_from_args,
    curriculum_quantize_blend,
    curriculum_state,
    forward_full_for_curriculum_step,
    main,
    parse_args,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_exports_are_minimal():
    assert sirencodec.__all__ == ["cuda", "mlx"]


def test_config_defaults_are_coherent():
    cfg = Config()
    sizes = effective_codebook_sizes(cfg)
    assert len(sizes) == cfg.n_codebooks
    assert encoder_time_stride(cfg) == 2 ** len(cfg.enc_channels)
    assert nominal_rvq_kbps(cfg) > 0
    assert cfg.load_audio_threads >= 1
    assert cfg.semantic_batch_items >= 1
    assert cfg.semantic_every >= 1
    assert isinstance(cfg.eval_seed, int)
    assert cfg.disc_type in {"msd", "mpd", "msmpd"}
    assert all(p > 0 for p in cfg.disc_periods)


def test_argparse_defaults_cover_current_config():
    defaults = argparse_defaults_from_config(Config())
    assert defaults["batch"] == Config().batch
    assert defaults["checkpoint_dir"] == Config().checkpoint_dir
    assert defaults["dataset"] == Config().dataset
    assert defaults["semantic_model"] == Config().semantic_model
    assert defaults["semantic_every"] == Config().semantic_every
    assert defaults["lambda_preemph"] == Config().lambda_preemph
    assert defaults["lambda_fm"] == Config().lambda_fm
    assert defaults["disc_periods"] == ",".join(str(x) for x in Config().disc_periods)


def test_stft_parsers_accept_valid_values_and_reject_bad_ones():
    assert parse_stft_scales_arg("512,128;1024,256") == ((512, 128), (1024, 256))
    assert parse_stft_scale_weights_arg("1,1.5,2") == (1.0, 1.5, 2.0)
    assert parse_positive_int_list_arg("6,9,12") == (6, 9, 12)
    with pytest.raises(ValueError):
        parse_stft_scales_arg("")
    with pytest.raises(ValueError):
        parse_stft_scale_weights_arg("0,0")
    with pytest.raises(ValueError):
        parse_positive_int_list_arg("0,2")


def test_pyproject_console_entrypoint_points_to_cuda_trainer():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["train"] == "sirencodec.cuda.train:main"


def test_infer_tool_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "infer_mlx.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "checkpoint" in result.stdout


def test_infer_tool_module_loads():
    mod = _load_tool(ROOT / "tools" / "infer_mlx.py", "infer_mlx_smoke")
    assert hasattr(mod, "pack_vq_bitstream")
    assert hasattr(mod, "nominal_bitrate_bps")


def test_curriculum_enters_phase_c_only_when_gan_is_enabled():
    cfg = Config(
        steps=100,
        curriculum=True,
        curriculum_ae_frac=0.10,
        curriculum_vq_ramp_frac=0.20,
        curriculum_gan_frac=0.20,
        lambda_adv=0.5,
    )
    assert curriculum_state(0, cfg).phase == "A/AE"
    assert curriculum_state(0, cfg).reset_enabled is True
    assert curriculum_state(15, cfg).phase == "B/RVQ"
    phase_c = curriculum_state(35, cfg)
    assert phase_c.phase == "C/GAN"
    assert 0.0 < phase_c.gan_mult <= 1.0
    assert curriculum_state(90, cfg).phase == "D/full"

    cfg_no_gan = Config(
        steps=100,
        curriculum=True,
        curriculum_ae_frac=0.10,
        curriculum_vq_ramp_frac=0.20,
        curriculum_gan_frac=0.20,
        lambda_adv=0.0,
    )
    assert curriculum_state(35, cfg_no_gan).phase == "D/full"


def test_curriculum_quantize_blend_ramps_through_phase_b():
    cfg = Config(
        steps=100,
        curriculum=True,
        curriculum_ae_frac=0.10,
        curriculum_vq_ramp_frac=0.40,
        lambda_adv=0.0,
    )
    assert curriculum_quantize_blend(0, cfg) == pytest.approx(0.0)
    assert 0.0 < curriculum_quantize_blend(20, cfg) < 1.0
    assert curriculum_quantize_blend(80, cfg) == pytest.approx(1.0)


def test_curriculum_quantize_blend_can_train_hard_rvq_immediately():
    cfg = Config(
        steps=100,
        curriculum=True,
        curriculum_ae_frac=0.10,
        curriculum_vq_ramp_frac=0.40,
        curriculum_quantize_blend=False,
        lambda_adv=0.0,
    )
    assert curriculum_quantize_blend(0, cfg) == pytest.approx(0.0)
    assert curriculum_quantize_blend(10, cfg) == pytest.approx(1.0)
    assert curriculum_quantize_blend(20, cfg) == pytest.approx(1.0)


def test_curriculum_forward_helper_uses_ae_only_phase_without_sticking():
    cfg = Config(
        batch=1,
        segment=64,
        steps=10,
        enc_channels=(4, 4),
        latent_dim=8,
        latent_temporal_depth=0,
        latent_temporal_post_depth=0,
        n_codebooks=1,
        codebook_size=8,
        rvq_code_dim=4,
        ae_only=False,
        curriculum=True,
        curriculum_ae_frac=0.50,
        curriculum_vq_ramp_frac=0.50,
        lambda_adv=0.0,
    )
    model = CUDACodec(cfg)
    x = torch.randn(1, 64, 1)

    _, _, _, _, idx_ae = forward_full_for_curriculum_step(model, cfg, x, 0)
    _, _, _, _, idx_rvq = forward_full_for_curriculum_step(model, cfg, x, 9)

    assert idx_ae is None
    assert idx_rvq is not None
    assert model.cfg.ae_only is False


def test_forward_full_can_return_continuous_anchor():
    cfg = Config(
        batch=1,
        segment=64,
        enc_channels=(4, 4),
        latent_dim=8,
        latent_temporal_depth=0,
        latent_temporal_post_depth=0,
        n_codebooks=1,
        codebook_size=8,
        rvq_code_dim=0,
    )
    model = CUDACodec(cfg)
    x = torch.randn(1, 64, 1)
    y, _, _, _, idx, y_cont = model.forward_full(x, return_continuous=True)
    assert y.shape == x.shape
    assert y_cont.shape == x.shape
    assert idx is not None


def test_ae_anchor_loss_is_only_active_during_rvq_phase():
    cfg = Config(
        batch=1,
        segment=64,
        steps=10,
        enc_channels=(4, 4),
        latent_dim=8,
        latent_temporal_depth=0,
        latent_temporal_post_depth=0,
        n_codebooks=1,
        codebook_size=8,
        rvq_code_dim=0,
        stft_scales=(),
        lambda_time=0.0,
        lambda_stft=0.0,
        lambda_sc=0.0,
        lambda_complex_stft=0.0,
        lambda_mag_l1=0.0,
        lambda_stft_grad=0.0,
        lambda_stft_cos=0.0,
        lambda_cos=0.0,
        cos_hinge=0.0,
        lambda_vq=0.0,
        lambda_marginal=0.0,
        lambda_ae_anchor_time=0.5,
        lambda_ae_anchor_cos=0.1,
        curriculum=True,
        curriculum_ae_frac=0.50,
        curriculum_vq_ramp_frac=0.50,
    )
    model = CUDACodec(cfg)
    x = torch.randn(1, 64, 1)
    _, m_ae = compute_loss(model, cfg, x, 0, None)
    _, m_rvq = compute_loss(model, cfg, x, 9, None)
    assert float(m_ae["ae_anchor"]) == pytest.approx(0.0)
    assert float(m_rvq["ae_anchor"]) > 0.0


def test_si_sdr_loss_prefers_aligned_waveform():
    x = torch.randn(3, 128, 1)
    noisy = torch.roll(x, shifts=3, dims=1)
    assert batch_neg_log_si_sdr(x, x) < batch_neg_log_si_sdr(x, noisy)
    assert batch_neg_log_si_sdr(x, x) < batch_neg_log_si_sdr(x, -x)


def test_preemphasis_loss_penalizes_high_frequency_error():
    x = torch.zeros(2, 32, 1)
    y = x.clone()
    y[:, ::2, :] = 0.1
    assert batch_preemph_l1(x, x) == pytest.approx(0.0)
    assert float(batch_preemph_l1(x, y, 0.97)) > 0.0


def test_rvq_residual_stages_all_carry_reconstruction_gradient():
    cfg = Config(
        latent_dim=4,
        n_codebooks=3,
        codebook_size=8,
        codebook_sizes=None,
        rvq_code_dim=0,
        lambda_entropy=0.0,
        lambda_marginal=0.0,
    )
    rvq = ResidualVectorQuantizer(cfg)
    z = torch.randn(2, 5, cfg.latent_dim, requires_grad=True)

    z_q, _, _, _, _ = rvq(z)
    z_q.sum().backward()

    assert z.grad is not None
    assert torch.allclose(z.grad, torch.full_like(z.grad, float(cfg.n_codebooks)))


def test_codec_eval_score_ignores_ae_only_rows():
    assert _codec_eval_score({"si_sdr_db": 6.0}, None) is None
    assert _codec_eval_score({"si_sdr_db": 1.25}, 880.0) == pytest.approx(1.25)
    assert _codec_eval_score({"si_sdr_db": None}, 880.0) is None


def test_multiscale_wave_discriminator_returns_one_logits_tensor_per_scale():
    disc = MultiScaleWaveDiscriminator(n_scales=3)
    x = torch.randn(2, 4096, 1)
    outs = disc(x)
    assert len(outs) == 3
    assert all(t.ndim == 2 for t in outs)
    assert all(t.shape[0] == 2 for t in outs)


def test_multiperiod_wave_discriminator_returns_logits_and_features():
    disc = MultiPeriodWaveDiscriminator(periods=(2, 3), base_channels=4)
    x = torch.randn(2, 513, 1)
    outs, feats = disc.forward_features(x)
    assert len(outs) == 2
    assert len(feats) == 2
    assert all(t.ndim == 2 for t in outs)
    assert all(t.shape[0] == 2 for t in outs)
    assert all(group for group in feats)


def test_discriminator_builder_accepts_mpd_config():
    cfg = Config(disc_type="mpd", disc_base_channels=4, disc_periods=(2, 3))
    disc = build_wave_discriminator(cfg)
    outs = disc(torch.randn(1, 257, 1))
    assert len(outs) == 2


def test_dataset_dir_candidates_cover_supported_datasets():
    assert dataset_dir_candidates("cv-corpus")[0].as_posix().endswith("data/cv-corpus")
    libri = dataset_dir_candidates("train-clean-100")
    assert any(path.as_posix().endswith("data/train-clean-100") for path in libri)
    assert any(path.as_posix().endswith("data/LibriSpeech/train-clean-100") for path in libri)
    libri_360 = dataset_dir_candidates("train-clean-360")
    assert any(path.as_posix().endswith("data/train-clean-360") for path in libri_360)
    assert any(path.as_posix().endswith("data/LibriSpeech/train-clean-360") for path in libri_360)


def test_train_parser_accepts_json_config_and_cli_overrides(tmp_path: Path):
    cfg_path = tmp_path / "train.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dataset": "cv-corpus",
                "batch": 7,
                "curriculum": True,
                "lambda_adv": 0.03,
                "lambda_fm": 0.7,
                "disc_type": "mpd",
                "disc_base_channels": 8,
                "disc_periods": "2,5",
                "eval_seed": 123,
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(["--config", str(cfg_path), "--dataset", "train-clean-100", "--batch", "9"])
    cfg = config_from_args(args)
    assert cfg.dataset == "train-clean-100"
    assert cfg.batch == 9
    assert cfg.curriculum is True
    assert cfg.lambda_adv == pytest.approx(0.03)
    assert cfg.lambda_fm == pytest.approx(0.7)
    assert cfg.disc_type == "mpd"
    assert cfg.disc_base_channels == 8
    assert cfg.disc_periods == (2, 5)
    assert cfg.eval_seed == 123


def test_train_parser_rejects_unknown_json_config_keys(tmp_path: Path):
    cfg_path = tmp_path / "bad_train.json"
    cfg_path.write_text(json.dumps({"dataset": "cv-corpus", "not_a_real_flag": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown keys"):
        parse_args(["--config", str(cfg_path)])


def test_sub1k_template_avoids_factorized_rvq_cold_start():
    data = json.loads((ROOT / "configs" / "sub1k_200.json").read_text(encoding="utf-8"))
    assert data["lambda_adv"] == 0.0
    assert data["activation"] == "snake_beta"
    assert data["fast"] is False
    assert data["batch"] == 128
    assert data["grad_accum_steps"] == 2
    assert data["loss_balancer"] == "off"
    assert data["enc_channels"] == "56,80,112,160,224,320,448,640"
    assert data["latent_dim"] == 512
    assert data["stft_scales"] == "512,128;1024,256;2048,512;4096,1024;8192,2048"
    assert data["stft_scale_weights"] == "0.5,1,1.5,2,3"
    assert data["spectral_batch_items"] == 16
    assert data["stft_large_min_fft"] == 4096
    assert data["stft_large_every"] == 4
    assert data.get("curriculum", False) is False
    assert data["curriculum_ae_frac"] <= 0.05
    assert data["curriculum_vq_ramp_frac"] >= 0.65
    assert data["curriculum_vq_start"] >= 0.25
    assert data["curriculum_entropy_start"] >= 0.25
    assert data["lambda_stft"] >= 0.5
    assert data["stft_ramp_steps"] == 0
    assert data["stft_hf_emphasis"] >= 1.0
    assert data["lambda_sc"] >= 0.5
    assert data["lambda_complex_stft"] >= 0.5
    assert data["lambda_mel_l1"] >= 0.03
    assert data["lambda_time"] >= 2.0
    assert data["lambda_cos"] >= 0.3
    assert data["cos_target"] >= 0.85
    assert data["lambda_sisdr"] > 0.0
    assert data["lambda_vq"] >= 0.5
    assert data["lambda_ae_anchor_time"] > 0.0
    assert data["lambda_ae_anchor_cos"] > 0.0
    assert data["vq_ema_decay"] <= 0.95
    assert data["rvq_code_dim"] == 0
    assert data["latent_temporal_post_depth"] == 0
    assert data["pre_vq_layernorm"] is True
    assert data["n_codebooks"] == 2
    assert data["codebook_sizes"] == "256,128"


def test_sub1k_template_resolves_to_strong_spectral_loss():
    cfg_path = ROOT / "configs" / "sub1k_200.json"
    args = parse_args(["--config", str(cfg_path), "--steps", "1", "--batch", "1"])
    cfg = config_from_args(args)
    assert cfg.curriculum is False
    assert cfg.enc_channels == (56, 80, 112, 160, 224, 320, 448, 640)
    assert cfg.latent_dim == 512
    assert cfg.stft_scales == ((512, 128), (1024, 256), (2048, 512), (4096, 1024), (8192, 2048))
    assert cfg.stft_scale_weights == (0.5, 1.0, 1.5, 2.0, 3.0)
    assert cfg.spectral_batch_items == 16
    assert cfg.stft_large_min_fft == 4096
    assert cfg.stft_large_every == 4
    assert cfg.lambda_stft == pytest.approx(0.65)
    assert cfg.stft_ramp_steps == 0
    assert cfg.stft_hf_emphasis == pytest.approx(1.5)
    assert cfg.curriculum_ae_frac == pytest.approx(0.05)
    assert cfg.curriculum_vq_ramp_frac == pytest.approx(0.65)
    assert cfg.curriculum_vq_start == pytest.approx(0.25)
    assert cfg.curriculum_entropy_start == pytest.approx(0.25)
    assert cfg.lambda_sisdr == pytest.approx(0.75)
    assert cfg.lambda_ae_anchor_time == pytest.approx(0.50)
    assert cfg.lambda_ae_anchor_cos == pytest.approx(0.10)
    assert cfg.loss_balancer == "off"
    base_cfg = Config(
        n_codebooks=cfg.n_codebooks,
        codebook_size=cfg.codebook_size,
        codebook_sizes=cfg.codebook_sizes,
        rvq_code_dim=cfg.rvq_code_dim,
        latent_temporal_post_depth=cfg.latent_temporal_post_depth,
        pre_vq_layernorm=cfg.pre_vq_layernorm,
    )
    assert effective_codebook_sizes(cfg) == effective_codebook_sizes(base_cfg)
    base_params = sum(p.numel() for p in CUDACodec(base_cfg).parameters())
    wide_params = sum(p.numel() for p in CUDACodec(cfg).parameters())
    assert wide_params / base_params == pytest.approx(2.93, rel=0.03)


def test_sub1k_semantic_ft_30_template_matches_trunk_plus_semantic():
    cfg_path = ROOT / "configs" / "sub1k_semantic_ft_30.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["epochs"] == 30
    assert data["lr"] == pytest.approx(2e-5)
    assert data["lr_warmup_steps"] == 400
    assert data["lr_plateau_patience"] == 800
    assert data["lr_plateau_cooldown"] == 250
    assert data["lr_min_ratio"] == pytest.approx(0.20)
    assert data["lambda_semantic"] == pytest.approx(0.5)
    assert data["semantic_model"] == "HUBERT_BASE"
    assert data["semantic_layers"] == "9"
    assert data["semantic_batch_items"] == 16
    assert data["semantic_every"] == 4
    assert data["lambda_marginal"] == pytest.approx(0.10)
    assert data["marginal_boost_steps"] == 0
    assert data["curriculum"] is False
    assert data["enc_channels"] == "56,80,112,160,224,320,448,640"
    assert data["codebook_sizes"] == "256,128"

    args = parse_args(["--config", str(cfg_path), "--steps", "1", "--batch", "1"])
    cfg = config_from_args(args)
    assert cfg.lambda_semantic == pytest.approx(0.5)
    assert cfg.semantic_layers == (9,)
    assert cfg.semantic_batch_items == 16
    assert cfg.semantic_every == 4
    assert cfg.lambda_marginal == pytest.approx(0.10)
    assert cfg.marginal_boost_steps == 0


def test_init_from_rejects_continue():
    cfg_path = ROOT / "configs" / "sub1k_semantic_ft_30.json"
    with pytest.raises(SystemExit, match="use either --continue or --init-from"):
        main(["--config", str(cfg_path), "--continue", "dummy.pt", "--init-from", "other.pt"])


def test_init_from_missing_checkpoint_exits():
    cfg_path = ROOT / "configs" / "sub1k_semantic_ft_30.json"
    with pytest.raises(SystemExit, match="--init-from checkpoint not found"):
        main(["--config", str(cfg_path), "--init-from", str(ROOT / "this_checkpoint_does_not_exist_12345.pt")])


def test_init_from_warm_start_runs_one_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import numpy as np
    import soundfile as sf

    monkeypatch.chdir(tmp_path)
    audio_dir = tmp_path / "data" / "train-clean-360"
    audio_dir.mkdir(parents=True)
    wav = (0.1 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 16000, endpoint=False))).astype(np.float32)
    sf.write(str(audio_dir / "one.wav"), wav, 16000)

    cfg_path = ROOT / "configs" / "sub1k_semantic_ft_30.json"
    seed_path = tmp_path / "seed_weights.pt"
    seed_args = parse_args(
        [
            "--config",
            str(cfg_path),
            "--steps",
            "1",
            "--batch",
            "2",
            "--lambda-semantic",
            "0",
            "--no-bf16",
            "--eval-every",
            "0",
        ]
    )
    seed_cfg = config_from_args(seed_args)
    torch.save({"model": CUDACodec(seed_cfg).state_dict()}, seed_path)

    main(
        [
            "--config",
            str(cfg_path),
            "--init-from",
            str(seed_path),
            "--steps",
            "1",
            "--batch",
            "2",
            "--lambda-semantic",
            "0",
            "--no-bf16",
            "--eval-every",
            "0",
        ]
    )
    exp = tmp_path / "experiments"
    assert exp.is_dir()
    runs = list(exp.iterdir())
    assert len(runs) == 1
    meta = json.loads((runs[0] / "train_config.json").read_text(encoding="utf-8"))
    assert meta.get("init_from") == str(seed_path.resolve())
    assert meta.get("resume_from") is None


def test_sub1k_harmonic_template_keeps_bitrate_and_enables_mpd():
    cfg_path = ROOT / "configs" / "sub1k_harmonic_20.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data.get("curriculum", False) is False
    assert data["epochs"] == 20
    assert data["n_codebooks"] == 2
    assert data["codebook_sizes"] == "256,128"
    assert data["disc_type"] == "msmpd"
    assert data["lambda_adv"] > 0.0
    assert data["lambda_fm"] > 0.0
    assert data["lambda_preemph"] > 0.0
    assert data["curriculum_ae_frac"] == pytest.approx(0.05)
    assert data["curriculum_vq_ramp_frac"] <= 0.15
    assert data["curriculum_quantize_blend"] is False
    assert data["lambda_stft_excess"] > 0.0
    assert data["stft_large_every"] <= 2

    args = parse_args(["--config", str(cfg_path), "--steps", "1", "--batch", "1"])
    cfg = config_from_args(args)
    assert cfg.curriculum is False
    ref_args = parse_args(["--config", str(ROOT / "configs" / "sub1k_200.json"), "--steps", "1", "--batch", "1"])
    ref_cfg = config_from_args(ref_args)
    assert effective_codebook_sizes(cfg) == effective_codebook_sizes(ref_cfg)
    assert encoder_time_stride(cfg) == encoder_time_stride(ref_cfg)
    assert nominal_rvq_kbps(cfg) == pytest.approx(nominal_rvq_kbps(ref_cfg))
    assert cfg.disc_periods == (2, 3, 5, 7, 11)
    assert cfg.curriculum_quantize_blend is False
    assert cfg.lambda_stft_excess > 0.0


def test_spectral_loss_batch_limits_large_fft_work_and_wraps():
    pred = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2, 1)
    tgt = -pred
    p, q = _spectral_loss_batch(pred, tgt, step=0, max_items=4)
    assert p.shape[0] == 4
    assert q.shape[0] == 4
    assert torch.equal(p[:, 0, 0], torch.tensor([0.0, 2.0, 4.0, 6.0]))
    p2, _ = _spectral_loss_batch(pred, tgt, step=1, max_items=4)
    assert torch.equal(p2[:, 0, 0], torch.tensor([8.0, 10.0, 0.0, 2.0]))


def test_large_stft_scales_are_cycled_for_speed():
    cfg = Config(
        stft_scales=((512, 128), (2048, 512), (4096, 1024), (8192, 2048)),
        stft_scale_weights=(0.5, 2.0, 4.0, 6.0),
        stft_large_min_fft=4096,
        stft_large_every=4,
    )
    scales0, weights0 = _active_stft_scales(cfg, 0)
    assert scales0 == cfg.stft_scales
    assert weights0 == cfg.stft_scale_weights
    scales1, weights1 = _active_stft_scales(cfg, 1)
    assert scales1 == ((512, 128), (2048, 512))
    assert weights1 == (0.5, 2.0)
