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
from sirencodec.cuda.codec import MultiScaleWaveDiscriminator
from sirencodec.cuda.data import dataset_dir_candidates
from sirencodec.cuda.train import config_from_args, curriculum_state, parse_args

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


def test_argparse_defaults_cover_current_config():
    defaults = argparse_defaults_from_config(Config())
    assert defaults["batch"] == Config().batch
    assert defaults["checkpoint_dir"] == Config().checkpoint_dir
    assert defaults["dataset"] == Config().dataset
    assert defaults["semantic_model"] == Config().semantic_model
    assert defaults["semantic_every"] == Config().semantic_every


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


def test_multiscale_wave_discriminator_returns_one_logits_tensor_per_scale():
    disc = MultiScaleWaveDiscriminator(n_scales=3)
    x = torch.randn(2, 4096, 1)
    outs = disc(x)
    assert len(outs) == 3
    assert all(t.ndim == 2 for t in outs)
    assert all(t.shape[0] == 2 for t in outs)


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


def test_train_parser_rejects_unknown_json_config_keys(tmp_path: Path):
    cfg_path = tmp_path / "bad_train.json"
    cfg_path.write_text(json.dumps({"dataset": "cv-corpus", "not_a_real_flag": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown keys"):
        parse_args(["--config", str(cfg_path)])
