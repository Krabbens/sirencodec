from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import sirencodec
from sirencodec.config import (
    Config,
    argparse_defaults_from_config,
    effective_codebook_sizes,
    encoder_time_stride,
    nominal_rvq_kbps,
    parse_stft_scale_weights_arg,
    parse_stft_scales_arg,
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


def test_argparse_defaults_cover_current_config():
    defaults = argparse_defaults_from_config(Config())
    assert defaults["batch"] == Config().batch
    assert defaults["checkpoint_dir"] == Config().checkpoint_dir
    assert defaults["librispeech"] == Config().use_librispeech


def test_stft_parsers_accept_valid_values_and_reject_bad_ones():
    assert parse_stft_scales_arg("512,128;1024,256") == ((512, 128), (1024, 256))
    assert parse_stft_scale_weights_arg("1,1.5,2") == (1.0, 1.5, 2.0)
    with pytest.raises(ValueError):
        parse_stft_scales_arg("")
    with pytest.raises(ValueError):
        parse_stft_scale_weights_arg("0,0")


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
