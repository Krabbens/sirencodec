"""ABR (per-frame RVQ depth): encode/decode round-trip and depth=full regression."""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sirencodec.core.train import AudioCodec, CodecConfig


def _make_cfg(**kwargs):
    d = dict(
        n_codebooks=4,
        codebook_size=64,
        codebook_dim=32,
        segment_length=3200,
        enc_channels=[16, 32, 32, 32],
        enc_strides=[2, 4, 5, 8],
        entropy_coding_enabled=False,
        use_fsq=False,
    )
    d.update(kwargs)
    return CodecConfig(**d)


def test_encode_decode_roundtrip_heuristic():
    cfg = _make_cfg(abr_enabled=True, abr_mode="heuristic")
    codec = AudioCodec(cfg)
    codec.eval()
    x = torch.randn(2, 1, cfg.segment_length) * 0.1
    with torch.no_grad():
        enc = codec.encode(x)
        assert isinstance(enc, tuple)
        indices, depth = enc
        assert depth is not None and depth.shape[0] == 2
        y = codec.decode(indices, target_length=x.size(-1), depth=depth)
    assert y.shape == x.shape


def test_forward_matches_full_depth():
    """With abr off, forward should match abr full (all layers every frame)."""
    base = dict(n_codebooks=4, codebook_size=32, codebook_dim=24, segment_length=3200)
    cfg_off = _make_cfg(**base, abr_enabled=False)
    cfg_full = _make_cfg(**base, abr_enabled=True, abr_mode="full")
    torch.manual_seed(0)
    x = torch.randn(1, 1, cfg_off.segment_length) * 0.1
    c0 = AudioCodec(cfg_off)
    c1 = AudioCodec(cfg_full)
    with torch.no_grad():
        c1.load_state_dict(c0.state_dict())
    c0.eval()
    c1.eval()
    with torch.no_grad():
        y0, *_r0 = c0(x)
        y1, *_r1 = c1(x)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-4)


def test_learned_policy_forward_shapes():
    cfg = _make_cfg(abr_enabled=True, abr_mode="learned", entropy_coding_enabled=False)
    codec = AudioCodec(cfg)
    x = torch.randn(1, 1, cfg.segment_length) * 0.1
    y, indices, *_rest = codec(x)
    assert y.shape == x.shape
    assert len(indices) == cfg.n_codebooks


def test_decode_without_depth_same_as_all_layers():
    cfg = _make_cfg(abr_enabled=False, n_codebooks=3, codebook_size=32, codebook_dim=16)
    codec = AudioCodec(cfg)
    x = torch.randn(1, 1, cfg.segment_length) * 0.05
    codec.eval()
    with torch.no_grad():
        z = codec.encoder(x)
        zq, indices, *_ = codec.quantizer(z, n_codebooks=cfg.n_codebooks, depth=None)
        y0 = codec.decoder(zq, target_length=x.size(-1))
        enc = codec.encode(x)
        y1 = codec.decode(enc, target_length=x.size(-1), depth=None)
    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-4)
