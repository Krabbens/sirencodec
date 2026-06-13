#!/usr/bin/env python3
"""MLX neural codec inference: load ``codec_step*.npz``, write original + reconstructed WAV.

  Nominal RVQ bitrate: ``sum_i log2(K_i) * (sr / stride)``; uniform K matches ``n_codebooks * log2(K) * …``.
  Compact bitstream: ``*_codes.bin`` — indices packed to ``ceil(log2(K))`` bits each (not uint16).
  Optional ``--save-npz-codes`` for debugging. ``--no-save-codes`` skips both.

  uv run python tools/infer_mlx.py mlx_checkpoints/codec_step50000.npz -i sample.wav -o out_infer/
  uv run python tools/infer_mlx.py ckpt.npz --random-dev --seed 42
  uv run python tools/infer_mlx.py ckpt.npz --random-test-clean   # LibriSpeech test-clean (holdout vs train-clean-100)
  # --random-dev: manifest val split (MLX train_mlx --librispeech still saw these in random batches)
  # --random-test-clean: speakers/utterances outside train-clean-100 (needs downloaded test-clean)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import struct
import sys
from pathlib import Path

_HEADER_SIZE = 32
_MAGIC = b"VQX1"


def encoder_time_stride(cfg) -> int:
    """Match ``train_mlx.encoder_time_stride``."""
    return 2 ** len(cfg.enc_channels)


def encoded_frame_count(num_samples: int, cfg) -> int:
    n = max(1, int(num_samples))
    for _ in cfg.enc_channels:
        n = (n + 1) // 2
    return max(1, n)


def _quantizer_kind_cfg(cfg) -> str:
    return (getattr(cfg, "quantizer", "rvq") or "rvq").strip().lower()


def _decoder_backend_cfg(cfg) -> str:
    return (getattr(cfg, "decoder_backend", "waveform") or "waveform").strip().lower()


def _uses_lux_vocos_cfg(cfg) -> bool:
    return _decoder_backend_cfg(cfg) == "lux_vocos"


def _turboquant_code_dim_cfg(cfg) -> int:
    d = int(getattr(cfg, "turboquant_code_dim", 0) or 0)
    return d if d > 0 else int(cfg.latent_dim)


def _effective_codebook_sizes_cfg(cfg) -> tuple[int, ...]:
    """Match ``train_mlx.effective_codebook_sizes`` without importing MLX train loop."""
    if _quantizer_kind_cfg(cfg) == "turboquant":
        bits = int(getattr(cfg, "turboquant_bits", 2))
        if bits not in (2, 4):
            raise ValueError(f"turboquant_bits must be 2 or 4, got {bits}")
        return (1 << bits,)
    if cfg.codebook_sizes is not None:
        return tuple(int(x) for x in cfg.codebook_sizes)
    return (int(cfg.codebook_size),) * int(cfg.n_codebooks)


def nominal_bitrate_bps(cfg) -> float:
    """Bottleneck bitstream rate at the codec latent frame rate."""
    import math

    enc_stride = encoder_time_stride(cfg)
    if _quantizer_kind_cfg(cfg) == "turboquant":
        return (
            float(int(getattr(cfg, "turboquant_bits", 2)) * _turboquant_code_dim_cfg(cfg))
            * (cfg.sample_rate / enc_stride)
        )
    sizes = _effective_codebook_sizes_cfg(cfg)
    return sum(math.log2(float(k)) for k in sizes) * (cfg.sample_rate / enc_stride)


def _vq_packed_header_len(blob: bytes) -> int:
    """``*.bin`` header length: v1=32, v2=32+2*ncb, v3=32+4*ncb."""
    if len(blob) < 32 or blob[0:4] != _MAGIC:
        return 32
    if blob[4] == 2:
        ncb = struct.unpack_from("<H", blob, 5)[0]
        return 32 + 2 * int(ncb)
    if blob[4] == 3:
        ncb = struct.unpack_from("<H", blob, 5)[0]
        return 32 + 4 * int(ncb)
    return 32


def bits_per_index(codebook_size: int) -> int:
    return max(1, (int(codebook_size) - 1).bit_length())


def pack_vq_bitstream(
    codes: list,
    *,
    codebook_sizes: tuple[int, ...],
    sample_rate: int,
    audio_samples: int,
) -> bytes:
    """Header + tightly packed indices.

    v1 (uniform K): legacy 32-byte header. v2 (mixed K): header length ``32 + 2*ncb`` with uint16 ``K_q`` tail.
    v3 (scalar-width streams): header length ``32 + 4*ncb`` with uint16 ``K_q,width_q`` pairs.
    """
    import numpy as np

    ncb = len(codes)
    if len(codebook_sizes) != ncb:
        raise ValueError(f"codebook_sizes len {len(codebook_sizes)} != ncb {ncb}")
    arrays = [np.asarray(c, dtype=np.int64) for c in codes]
    if not arrays:
        raise ValueError("codes cannot be empty")
    n_frames = int(arrays[0].shape[0])
    widths: list[int] = []
    for q, a in enumerate(arrays):
        if a.shape[0] != n_frames:
            raise ValueError(f"code stream {q} has {a.shape[0]} frames, expected {n_frames}")
        if a.ndim == 1:
            widths.append(1)
        elif a.ndim == 2:
            widths.append(int(a.shape[1]))
        else:
            raise ValueError(f"code stream {q} must be 1-D or 2-D, got shape {a.shape}")
    uniform = len(set(int(k) for k in codebook_sizes)) == 1
    scalar_width = any(w != 1 for w in widths)

    out = bytearray()
    bit_buf = 0
    bit_n = 0

    if uniform and not scalar_width:
        K0 = int(codebook_sizes[0])
        bp = bits_per_index(K0)
        mask = (1 << bp) - 1
        header = bytearray(_HEADER_SIZE)
        header[0:4] = _MAGIC
        header[4] = 1
        struct.pack_into("<H", header, 5, ncb)
        struct.pack_into("<I", header, 7, K0)
        struct.pack_into("<I", header, 11, n_frames)
        struct.pack_into("<I", header, 15, int(sample_rate))
        struct.pack_into("<I", header, 19, int(audio_samples))
        header[23] = bp
        for t in range(n_frames):
            for q in range(ncb):
                v = int(arrays[q][t]) & mask
                for bi in range(bp):
                    bit = (v >> (bp - 1 - bi)) & 1
                    bit_buf = (bit_buf << 1) | bit
                    bit_n += 1
                    if bit_n == 8:
                        out.append(bit_buf & 0xFF)
                        bit_buf = 0
                        bit_n = 0
        if bit_n:
            out.append((bit_buf << (8 - bit_n)) & 0xFF)
        return bytes(header) + bytes(out)

    if scalar_width:
        hdr_len = 32 + 4 * ncb
        header = bytearray(hdr_len)
        header[0:4] = _MAGIC
        header[4] = 3
        struct.pack_into("<H", header, 5, ncb)
        struct.pack_into("<I", header, 7, int(max(codebook_sizes)))
        struct.pack_into("<I", header, 11, n_frames)
        struct.pack_into("<I", header, 15, int(sample_rate))
        struct.pack_into("<I", header, 19, int(audio_samples))
        header[23] = 0
        for q in range(ncb):
            struct.pack_into("<H", header, 32 + 4 * q, int(codebook_sizes[q]))
            struct.pack_into("<H", header, 34 + 4 * q, int(widths[q]))
        for t in range(n_frames):
            for q in range(ncb):
                bp = bits_per_index(codebook_sizes[q])
                mask = (1 << bp) - 1
                row = arrays[q][t] if arrays[q].ndim == 2 else np.asarray([arrays[q][t]])
                for c in range(widths[q]):
                    v = int(row[c]) & mask
                    for bi in range(bp):
                        bit = (v >> (bp - 1 - bi)) & 1
                        bit_buf = (bit_buf << 1) | bit
                        bit_n += 1
                        if bit_n == 8:
                            out.append(bit_buf & 0xFF)
                            bit_buf = 0
                            bit_n = 0
        if bit_n:
            out.append((bit_buf << (8 - bit_n)) & 0xFF)
        return bytes(header) + bytes(out)

    hdr_len = 32 + 2 * ncb
    header = bytearray(hdr_len)
    header[0:4] = _MAGIC
    header[4] = 2
    struct.pack_into("<H", header, 5, ncb)
    struct.pack_into("<I", header, 7, int(max(codebook_sizes)))
    struct.pack_into("<I", header, 11, n_frames)
    struct.pack_into("<I", header, 15, int(sample_rate))
    struct.pack_into("<I", header, 19, int(audio_samples))
    header[23] = 0
    for q in range(ncb):
        struct.pack_into("<H", header, 32 + 2 * q, int(codebook_sizes[q]))
    for t in range(n_frames):
        for q in range(ncb):
            bp = bits_per_index(codebook_sizes[q])
            mask = (1 << bp) - 1
            v = int(arrays[q][t]) & mask
            for bi in range(bp):
                bit = (v >> (bp - 1 - bi)) & 1
                bit_buf = (bit_buf << 1) | bit
                bit_n += 1
                if bit_n == 8:
                    out.append(bit_buf & 0xFF)
                    bit_buf = 0
                    bit_n = 0
    if bit_n:
        out.append((bit_buf << (8 - bit_n)) & 0xFF)
    return bytes(header) + bytes(out)


def _print_low_bitrate_hint() -> None:
    sys.stderr.write(
        "\n"
        "  ~36× niższy bitrate (~1 kbps przy obecnym ~36 kbps) wymaga MNIEJ bitów na sekundę audio,\n"
        "  nie tylko lepszego pakowania pliku. Kierunki (train_mlx / nowy arch):\n"
        "    • większy stride w czasie (głębszy encoder / 16→64→256) → mniej ramek latentnych/s;\n"
        "    • mniej `--n-codebooks` i/lub mniejsze `--codebook-size`;\n"
        "    • osobny model pod niski bitrate (np. 1 codebook 1–2 kbps + silną kwantyzację).\n"
        "  Samo packowanie binarne usuwa marnotrawstwo stałej 16-bitowej szerokości indeksu.\n\n"
    )


def validation_entries(manifest_path: Path) -> list[dict]:
    """Pick validation-style entries from a manifest using explicit dev/test tags or a stable 10% hash split."""
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    all_entries: list[dict] = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line))
    dev_explicit = [
        e
        for e in all_entries
        if "test" in e.get("subset", "") or "val" in e.get("subset", "") or "dev" in e.get("subset", "")
    ]
    train_pool = [
        e
        for e in all_entries
        if "test" not in e.get("subset", "")
        and "val" not in e.get("subset", "")
        and "dev" not in e.get("subset", "")
    ]
    dev_entries: list[dict] = []
    for e in train_pool:
        h = int(hashlib.md5((e["path"] + "_split").encode()).hexdigest(), 16) % 10
        if h >= 9:
            dev_entries.append(e)
    if dev_explicit:
        return dev_explicit
    return dev_entries


def pick_random_validation_audio(manifest_path: Path, seed: int | None) -> Path:
    entries = validation_entries(manifest_path)
    if not entries:
        raise RuntimeError(f"no validation entries in {manifest_path}")
    rng = random.Random(seed)
    e = rng.choice(entries)
    p = Path(e["path"])
    if not p.is_file():
        raise FileNotFoundError(f"manifest path missing on disk: {p}")
    return p.resolve()


def pick_random_test_clean(librispeech_root: Path, seed: int | None) -> Path:
    """Official LibriSpeech ``test-clean`` (disjoint from train-clean-100 speakers)."""
    test_clean = librispeech_root / "LibriSpeech" / "test-clean"
    if not test_clean.is_dir():
        raise FileNotFoundError(
            f"LibriSpeech test-clean not found: {test_clean}\n"
            "  Download test-clean.tar.gz from https://www.openslr.org/12/ and extract under "
            f"{librispeech_root / 'LibriSpeech'}/ so that test-clean/*.flac exists."
        )
    exts = {".wav", ".flac", ".ogg"}
    files = [p for p in test_clean.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"no audio files under {test_clean}")
    rng = random.Random(seed)
    return rng.choice(files).resolve()


def _load_train_mlx():
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("train_mlx", root / "train_mlx.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load train_mlx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_mlx"] = mod  # required so dataclasses can resolve Config
    spec.loader.exec_module(mod)
    return mod


def load_audio_mono(path: Path, target_sr: int, max_seconds: float | None) -> "numpy.ndarray":
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    if sr != target_sr:
        n_new = max(1, int(round(wav.size * target_sr / sr)))
        wav = np.interp(
            np.linspace(0, wav.size - 1, num=n_new),
            np.arange(wav.size),
            wav,
        ).astype(np.float32)
    if max_seconds is not None:
        cap = int(target_sr * max_seconds)
        wav = wav[:cap]
    m = float(wav.max()) if wav.size else 1.0
    m = max(abs(m), float(-wav.min()) if wav.size else 1.0, 1e-5)
    return (wav / m).astype(np.float32)


def spectral_noise_filter(
    wav_1d: "numpy.ndarray",
    *,
    strength: float = 0.35,
    noise_percentile: float = 20.0,
    min_gain: float = 0.20,
    n_fft: int = 1024,
    hop: int = 256,
) -> "numpy.ndarray":
    """Lightweight STFT-domain denoiser for post-decode inference output."""
    import numpy as np

    x = np.asarray(wav_1d, dtype=np.float32).reshape(-1)
    if x.size == 0 or strength <= 0.0:
        return x.copy()
    n_fft = int(n_fft)
    hop = int(hop)
    if n_fft < 16 or hop < 1:
        raise ValueError("noise filter requires n_fft >= 16 and hop >= 1")
    strength = float(np.clip(strength, 0.0, 1.0))
    noise_percentile = float(np.clip(noise_percentile, 0.0, 100.0))
    min_gain = float(np.clip(min_gain, 0.0, 1.0))

    win = np.hanning(n_fft).astype(np.float32)
    starts = list(range(0, max(1, x.size), hop))
    padded_len = starts[-1] + n_fft
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[: x.size] = x

    spec = []
    for start in starts:
        spec.append(np.fft.rfft(padded[start : start + n_fft] * win))
    S = np.stack(spec, axis=0)
    mag = np.abs(S).astype(np.float32)
    noise = np.percentile(mag, noise_percentile, axis=0).astype(np.float32)
    noise_power = (noise[None, :] * (1.0 + 2.0 * strength)) ** 2
    raw_mask = (mag**2) / (mag**2 + noise_power + 1e-12)
    raw_mask = np.maximum(raw_mask, min_gain)
    mask = (1.0 - strength) + strength * raw_mask

    y = np.zeros(padded_len, dtype=np.float32)
    norm = np.zeros(padded_len, dtype=np.float32)
    for frame_i, start in enumerate(starts):
        frame = np.fft.irfft(S[frame_i] * mask[frame_i], n=n_fft).astype(np.float32)
        y[start : start + n_fft] += frame * win
        norm[start : start + n_fft] += win * win
    good = norm > 1e-8
    y[good] /= norm[good]
    return y[: x.size].astype(np.float32)


def numpy_cosine(a: "numpy.ndarray", b: "numpy.ndarray") -> float:
    import numpy as np

    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return 0.0
    av = np.asarray(a[:n], dtype=np.float32).reshape(-1)
    bv = np.asarray(b[:n], dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def resample_linear(wav_1d: "numpy.ndarray", src_sr: int, dst_sr: int) -> "numpy.ndarray":
    import numpy as np

    x = np.asarray(wav_1d, dtype=np.float32).reshape(-1)
    if int(src_sr) == int(dst_sr) or x.size == 0:
        return x.copy()
    n_new = max(1, int(round(x.size * float(dst_sr) / float(src_sr))))
    return np.interp(
        np.linspace(0, x.size - 1, num=n_new),
        np.arange(x.size),
        x,
    ).astype(np.float32)


def resample_audio(wav_1d: "numpy.ndarray", src_sr: int, dst_sr: int) -> "numpy.ndarray":
    import math
    import numpy as np

    x = np.asarray(wav_1d, dtype=np.float32).reshape(-1)
    if int(src_sr) == int(dst_sr) or x.size == 0:
        return x.copy()
    try:
        from scipy.signal import resample_poly

        g = math.gcd(int(src_sr), int(dst_sr))
        return resample_poly(x, int(dst_sr) // g, int(src_sr) // g).astype(np.float32)
    except Exception:
        return resample_linear(x, src_sr, dst_sr)


def _torch_device(device: str):
    import torch

    d = (device or "auto").strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            d = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            d = "mps"
        else:
            d = "cpu"
    return torch.device(d)


def render_lux_vocos_features(
    features: "numpy.ndarray",
    cfg,
    *,
    device: str = "auto",
) -> "numpy.ndarray":
    """Render predicted LuxTTS/Vocos log-mel features ``[frames, mels]`` to waveform."""
    import numpy as np
    import torch

    try:
        from huggingface_hub import snapshot_download
        from linacodec.vocoder.vocos import Vocos
        from torch.nn.utils import parametrize
    except ImportError as e:
        raise RuntimeError(
            "Lux/Vocos rendering needs optional dependencies: install huggingface-hub, linacodec, and torch"
        ) from e

    model_ref = str(getattr(cfg, "lux_vocos_model", "YatharthS/LuxTTS") or "YatharthS/LuxTTS")
    model_path = Path(model_ref).expanduser()
    if model_path.is_dir():
        vocos_root = model_path
    else:
        vocos_root = Path(snapshot_download(model_ref))
    config_path = vocos_root / "vocoder" / "config.yaml"
    weights_path = vocos_root / "vocoder" / "vocos.bin"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Lux/Vocos files not found under {vocos_root}/vocoder")

    dev = _torch_device(device)
    vocos = Vocos.from_hparams(str(config_path)).to(dev).eval()
    for idx in (0, 1):
        try:
            parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[idx], "weight")
        except Exception:
            pass
    state = torch.load(str(weights_path), map_location=dev)
    vocos.load_state_dict(state)
    vocos.freq_range = 12000
    requested_sr = int(getattr(cfg, "lux_vocos_output_sample_rate", 48000) or 48000)
    return_48k = requested_sr >= 48000
    if hasattr(vocos, "return_48k"):
        vocos.return_48k = bool(return_48k)

    feat_np = np.asarray(features, dtype=np.float32)
    if feat_np.ndim != 2:
        raise ValueError(f"Lux/Vocos features must be [frames, mels], got {feat_np.shape}")
    with torch.inference_mode():
        feat = torch.from_numpy(feat_np.T[None, :, :]).to(dev)
        wav = vocos.decode(feat).squeeze().clamp(-1, 1)
    out = wav.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    actual_sr = 48000 if return_48k else 24000
    if actual_sr != requested_sr:
        out = resample_audio(out, actual_sr, requested_sr)
    return out


def infer_waveform(
    tm, model, cfg, wav_1d: "numpy.ndarray"
) -> tuple["numpy.ndarray", float, list["numpy.ndarray"] | None]:
    """Chunked encode-decode.

    Returns waveform samples for ``decoder_backend=waveform`` or Lux/Vocos feature frames
    for ``decoder_backend=lux_vocos``, plus mean waveform cosine when available and code streams.
    """
    import mlx.core as mx
    import numpy as np

    seg = cfg.segment
    T = int(wav_1d.shape[0])
    outs: list[np.ndarray] = []
    cos_vals: list[float] = []
    feature_mode = _uses_lux_vocos_cfg(cfg)
    n_streams = len(_effective_codebook_sizes_cfg(cfg))
    codes_acc: list[list[np.ndarray]] | None = None if cfg.ae_only else [[] for _ in range(n_streams)]

    start = 0
    while start < T:
        end = min(start + seg, T)
        chunk = wav_1d[start:end]
        if chunk.shape[0] < seg:
            chunk = np.pad(chunk, (0, seg - chunk.shape[0]))
        x = mx.array(chunk.reshape(1, seg, 1))
        y, _, _, _, idx_list = model.forward_full(x)
        mx.eval(y)
        if idx_list is not None and codes_acc is not None:
            mx.eval(*idx_list)
            for q in range(len(idx_list)):
                a = np.asarray(idx_list[q][0], dtype=np.int32)
                codes_acc[q].append(a)
        valid = end - start
        if feature_mode:
            valid_frames = encoded_frame_count(valid, cfg)
            y_np = np.array(y[0, :valid_frames, :], dtype=np.float32)
            outs.append(y_np)
        else:
            y_np = np.array(y[0, :valid, 0], dtype=np.float32)
            outs.append(y_np)
            cos_vals.append(numpy_cosine(wav_1d[start:end], y_np))
        start = end

    recon = np.concatenate(outs, axis=0) if outs else np.zeros((0, 0) if feature_mode else 0, dtype=np.float32)
    mean_cos = float(sum(cos_vals) / max(len(cos_vals), 1))
    if codes_acc is not None:
        codes = [np.concatenate(codes_acc[q], axis=0) for q in range(n_streams)]
    else:
        codes = None
    return recon, mean_cos, codes


def main() -> None:
    tm = _load_train_mlx()
    import numpy as np

    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="MLX codec: load .npz checkpoint, reconstruct WAV")
    p.add_argument("checkpoint", type=Path, help="codec_step*.npz from train_mlx")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=Path, help="Input .wav / .flac / .ogg / .mp3 (any file you choose)")
    src.add_argument(
        "--random-dev",
        action="store_true",
        help=(
            "Random file from pipeline val split (manifest 10%% hash). "
            "Note: train_mlx --librispeech still samples from the whole tree, so overlap is likely."
        ),
    )
    src.add_argument(
        "--random-test-clean",
        action="store_true",
        help=(
            "Random utterance from LibriSpeech test-clean under --librispeech-root "
            "(holdout vs train-clean-100; download test-clean from OpenSLR-12 if missing)"
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=repo / "data" / "cv-corpus" / "master_manifest.jsonl",
        help="Used with --random-dev (default: repo data/cv-corpus/master_manifest.jsonl)",
    )
    p.add_argument(
        "--librispeech-root",
        type=Path,
        default=repo / "data" / "librispeech",
        help="Used with --random-test-clean (expects LibriSpeech/test-clean/)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --random-dev / --random-test-clean (default: nondeterministic)",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("infer_mlx_out"),
        help="Directory for *_orig.wav and *_recon.wav",
    )
    p.add_argument("--ae-only", action="store_true", help="Checkpoint trained without VQ")
    dcfg = tm.Config()
    p.add_argument("--sample-rate", type=int, default=dcfg.sample_rate, help="Must match training sample rate")
    p.add_argument("--segment", type=int, default=dcfg.segment, help="Chunk length for inference")
    p.add_argument(
        "--enc-channels",
        type=str,
        default=",".join(str(x) for x in dcfg.enc_channels),
        help="Must match training (comma-separated); default = current train_mlx Config",
    )
    p.add_argument(
        "--stride1-blocks-per-scale",
        type=int,
        default=dcfg.stride1_blocks_per_scale,
        help="Must match training (stride-1 blocks per scale in encoder/decoder)",
    )
    p.add_argument("--latent-dim", type=int, default=dcfg.latent_dim, help="Must match training")
    p.add_argument("--self-attention-depth", type=int, default=dcfg.self_attention_depth, help="Must match training")
    p.add_argument(
        "--self-attention-post-depth",
        type=int,
        default=dcfg.self_attention_post_depth,
        help="Must match training",
    )
    p.add_argument("--self-attention-heads", type=int, default=dcfg.self_attention_heads, help="Must match training")
    p.add_argument("--decoder-refine-depth", type=int, default=dcfg.decoder_refine_depth, help="Must match training")
    p.add_argument("--decoder-refine-gain", type=float, default=dcfg.decoder_refine_gain, help="Must match training")
    p.add_argument("--decoder-band-heads", type=int, default=dcfg.decoder_band_heads, help="Must match training")
    p.add_argument("--decoder-band-depth", type=int, default=dcfg.decoder_band_depth, help="Must match training")
    p.add_argument("--decoder-band-gain", type=float, default=dcfg.decoder_band_gain, help="Must match training")
    p.add_argument("--post-lavasr-depth", type=int, default=dcfg.post_lavasr_depth, help="Must match training")
    p.add_argument("--post-lavasr-channels", type=int, default=dcfg.post_lavasr_channels, help="Must match training")
    p.add_argument("--post-lavasr-kernel", type=int, default=dcfg.post_lavasr_kernel, help="Must match training")
    p.add_argument("--post-lavasr-gain", type=float, default=dcfg.post_lavasr_gain, help="Must match training")
    p.add_argument(
        "--post-lavasr-highpass",
        action=argparse.BooleanOptionalAction,
        default=dcfg.post_lavasr_highpass,
        help="Must match training",
    )
    p.add_argument("--speech-control-depth", type=int, default=dcfg.speech_control_depth, help="Must match training")
    p.add_argument("--speech-control-channels", type=int, default=dcfg.speech_control_channels, help="Must match training")
    p.add_argument("--speech-control-gain", type=float, default=dcfg.speech_control_gain, help="Must match training")
    p.add_argument("--speech-residual-gain", type=float, default=dcfg.speech_residual_gain, help="Must match training")
    p.add_argument("--speech-hf-gate-floor", type=float, default=dcfg.speech_hf_gate_floor, help="Must match training")
    p.add_argument(
        "--harmonic-source",
        action=argparse.BooleanOptionalAction,
        default=dcfg.harmonic_source,
        help="Must match training",
    )
    p.add_argument("--harmonic-harmonics", type=int, default=dcfg.harmonic_harmonics, help="Must match training")
    p.add_argument("--harmonic-amp", type=float, default=dcfg.harmonic_amp, help="Must match training")
    p.add_argument("--harmonic-f0-min", type=float, default=dcfg.harmonic_f0_min, help="Must match training")
    p.add_argument("--harmonic-f0-max", type=float, default=dcfg.harmonic_f0_max, help="Must match training")
    p.add_argument(
        "--pre-vq-layernorm",
        action=argparse.BooleanOptionalAction,
        default=dcfg.pre_vq_layernorm,
        help="Must match training (--pre-vq-layernorm / --no-pre-vq-layernorm)",
    )
    p.add_argument(
        "--quantizer",
        type=str,
        choices=("rvq", "turboquant"),
        default=dcfg.quantizer,
        help="Must match training bottleneck",
    )
    p.add_argument(
        "--turboquant-bits",
        type=int,
        choices=(2, 4),
        default=dcfg.turboquant_bits,
        help="Must match TurboQuant training",
    )
    p.add_argument(
        "--turboquant-code-dim",
        type=int,
        default=dcfg.turboquant_code_dim,
        help="Must match TurboQuant training (0 = full latent_dim)",
    )
    p.add_argument("--n-codebooks", type=int, default=dcfg.n_codebooks, help="Must match training")
    p.add_argument("--codebook-size", type=int, default=dcfg.codebook_size, help="Must match training (if uniform K)")
    p.add_argument(
        "--codebook-sizes",
        type=str,
        default=None,
        metavar="K0,K1,…",
        help="Must match training when per-stage K differ (e.g. 256,128,64)",
    )
    p.add_argument(
        "--decoder-backend",
        type=str,
        choices=("waveform", "lux_vocos"),
        default=dcfg.decoder_backend,
        help="Must match training decoder backend",
    )
    p.add_argument("--lux-vocos-feature-dim", type=int, default=dcfg.lux_vocos_feature_dim)
    p.add_argument("--lux-vocos-n-fft", type=int, default=dcfg.lux_vocos_n_fft)
    p.add_argument("--lux-vocos-hop", type=int, default=dcfg.lux_vocos_hop)
    p.add_argument("--lux-vocos-fmin", type=float, default=dcfg.lux_vocos_fmin)
    p.add_argument("--lux-vocos-fmax", type=float, default=dcfg.lux_vocos_fmax)
    p.add_argument("--lux-vocos-model", type=str, default=dcfg.lux_vocos_model)
    p.add_argument("--lux-vocos-output-sample-rate", type=int, default=dcfg.lux_vocos_output_sample_rate)
    p.add_argument(
        "--lux-vocos-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device for frozen Lux/Vocos rendering",
    )
    p.add_argument(
        "--vq-cosine",
        action=argparse.BooleanOptionalAction,
        default=dcfg.vq_cosine,
        help="Must match training (Euclidean RVQ checkpoints: --no-vq-cosine)",
    )
    p.add_argument("--max-seconds", type=float, default=None, help="Trim input to this many seconds")
    p.add_argument(
        "--no-save-codes",
        action="store_true",
        help="Do not write any code bitstream (*.bin / *.npz)",
    )
    p.add_argument(
        "--save-npz-codes",
        action="store_true",
        help="Also write *_codes.npz (int32 per index; debug)",
    )
    p.add_argument(
        "--low-bitrate-hint",
        action="store_true",
        help="Print stderr note on reaching ~1 kbps (36× lower than ~36 kbps nominal)",
    )
    p.add_argument(
        "--noise-filter",
        action="store_true",
        help="Apply a lightweight spectral noise gate to *_recon.wav; raw output is kept as *_recon_raw.wav",
    )
    p.add_argument("--noise-filter-strength", type=float, default=0.35, help="Denoise blend in [0,1]")
    p.add_argument(
        "--noise-filter-percentile",
        type=float,
        default=20.0,
        help="Percentile over STFT frames used as the noise-floor estimate",
    )
    p.add_argument("--noise-filter-min-gain", type=float, default=0.20, help="Lower mask gain in [0,1]")
    p.add_argument("--noise-filter-n-fft", type=int, default=1024)
    p.add_argument("--noise-filter-hop", type=int, default=256)
    args = p.parse_args()

    if not (0.0 <= args.noise_filter_strength <= 1.0):
        print("--noise-filter-strength must be in [0, 1]", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.noise_filter_percentile <= 100.0):
        print("--noise-filter-percentile must be in [0, 100]", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.noise_filter_min_gain <= 1.0):
        print("--noise-filter-min-gain must be in [0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.noise_filter_n_fft < 16 or args.noise_filter_hop < 1:
        print("--noise-filter-n-fft must be >= 16 and --noise-filter-hop must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.speech_control_depth < 0 or args.speech_control_channels < 1:
        print("--speech-control-depth must be >= 0 and --speech-control-channels must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.speech_control_gain < 0 or args.speech_residual_gain < 0:
        print("--speech-control-gain and --speech-residual-gain must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.speech_hf_gate_floor <= 1.0):
        print("--speech-hf-gate-floor must be in [0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.sample_rate < 1:
        print("--sample-rate must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.segment < 1:
        print("--segment must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.turboquant_code_dim < 0:
        print("--turboquant-code-dim must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.turboquant_code_dim > args.latent_dim:
        print("--turboquant-code-dim must be <= --latent-dim", file=sys.stderr)
        sys.exit(1)
    if args.decoder_backend == "lux_vocos":
        if args.lux_vocos_feature_dim < 1:
            print("--lux-vocos-feature-dim must be >= 1", file=sys.stderr)
            sys.exit(1)
        if args.lux_vocos_n_fft < 2 or args.lux_vocos_n_fft % 2 != 0:
            print("--lux-vocos-n-fft must be a positive even integer", file=sys.stderr)
            sys.exit(1)
        if args.lux_vocos_hop < 1:
            print("--lux-vocos-hop must be >= 1", file=sys.stderr)
            sys.exit(1)
        if args.lux_vocos_fmin < 0:
            print("--lux-vocos-fmin must be >= 0", file=sys.stderr)
            sys.exit(1)
        if args.lux_vocos_fmax is not None and args.lux_vocos_fmax <= args.lux_vocos_fmin:
            print("--lux-vocos-fmax must be > --lux-vocos-fmin", file=sys.stderr)
            sys.exit(1)
        if args.lux_vocos_output_sample_rate < 1:
            print("--lux-vocos-output-sample-rate must be >= 1", file=sys.stderr)
            sys.exit(1)

    ck = args.checkpoint.resolve()
    if not ck.is_file():
        print(f"Missing checkpoint: {ck}", file=sys.stderr)
        sys.exit(1)
    if args.random_dev:
        man = args.manifest.resolve()
        try:
            inp = pick_random_validation_audio(man, args.seed)
        except (FileNotFoundError, RuntimeError) as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(f"random-dev: {inp}")
    elif args.random_test_clean:
        root = args.librispeech_root.resolve()
        try:
            inp = pick_random_test_clean(root, args.seed)
        except (FileNotFoundError, RuntimeError) as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(f"random-test-clean (holdout vs train-clean-100): {inp}")
    else:
        inp = args.input.resolve()
        if not inp.is_file():
            print(f"Missing input: {inp}", file=sys.stderr)
            sys.exit(1)

    try:
        enc_ch = tuple(int(x.strip()) for x in args.enc_channels.split(",") if x.strip())
    except ValueError:
        enc_ch = ()
    if len(enc_ch) < 1:
        print("--enc-channels invalid", file=sys.stderr)
        sys.exit(1)
    cb_sizes_infer: tuple[int, ...] | None = None
    if args.codebook_sizes:
        try:
            cb_sizes_infer = tm.parse_codebook_sizes_arg(args.codebook_sizes)
        except ValueError as e:
            print(f"--codebook-sizes: {e}", file=sys.stderr)
            sys.exit(1)
        if len(cb_sizes_infer) != args.n_codebooks:
            print(
                f"--codebook-sizes: need {args.n_codebooks} values, got {len(cb_sizes_infer)}",
                file=sys.stderr,
            )
            sys.exit(1)
    cfg = tm.Config(
        sample_rate=args.sample_rate,
        segment=args.segment,
        enc_channels=enc_ch,
        stride1_blocks_per_scale=args.stride1_blocks_per_scale,
        latent_dim=args.latent_dim,
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
        speech_control_depth=args.speech_control_depth,
        speech_control_channels=args.speech_control_channels,
        speech_control_gain=args.speech_control_gain,
        speech_residual_gain=args.speech_residual_gain,
        speech_hf_gate_floor=args.speech_hf_gate_floor,
        harmonic_source=bool(args.harmonic_source),
        harmonic_harmonics=args.harmonic_harmonics,
        harmonic_amp=args.harmonic_amp,
        harmonic_f0_min=args.harmonic_f0_min,
        harmonic_f0_max=args.harmonic_f0_max,
        pre_vq_layernorm=args.pre_vq_layernorm,
        quantizer=args.quantizer,
        turboquant_bits=args.turboquant_bits,
        turboquant_code_dim=args.turboquant_code_dim,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        codebook_sizes=cb_sizes_infer,
        decoder_backend=args.decoder_backend,
        lux_vocos_feature_dim=args.lux_vocos_feature_dim,
        lux_vocos_n_fft=args.lux_vocos_n_fft,
        lux_vocos_hop=args.lux_vocos_hop,
        lux_vocos_fmin=args.lux_vocos_fmin,
        lux_vocos_fmax=args.lux_vocos_fmax,
        lux_vocos_model=args.lux_vocos_model,
        lux_vocos_output_sample_rate=args.lux_vocos_output_sample_rate,
        vq_cosine=args.vq_cosine,
        ae_only=bool(args.ae_only),
    )
    model = tm.MLXCodec(cfg)
    try:
        model.load_weights(str(ck), strict=True)
    except Exception as e:
        try:
            model.load_weights(str(ck), strict=False)
            print(f"[load] strict load failed; loaded matching weights with strict=False ({e})", file=sys.stderr)
        except Exception as e2:
            print(f"load_weights failed: {e}", file=sys.stderr)
            print(f"  after strict=False retry: {e2}", file=sys.stderr)
            print("Hint: training Config must match (try without --ae-only or with it).", file=sys.stderr)
            sys.exit(1)

    wav = load_audio_mono(inp, cfg.sample_rate, args.max_seconds)
    model_out, mean_cos, codes = infer_waveform(tm, model, cfg, wav)
    output_sr = int(cfg.sample_rate)
    orig_to_write = wav
    if _uses_lux_vocos_cfg(cfg):
        features = model_out
        recon = render_lux_vocos_features(features, cfg, device=args.lux_vocos_device)
        output_sr = int(cfg.lux_vocos_output_sample_rate)
        orig_to_write = resample_audio(wav, cfg.sample_rate, output_sr)
        mean_cos = numpy_cosine(orig_to_write, recon)
    else:
        recon = model_out
    recon_to_write = recon
    filtered_cos: float | None = None
    raw_recon_path: Path | None = None
    if args.noise_filter:
        recon_to_write = spectral_noise_filter(
            recon,
            strength=args.noise_filter_strength,
            noise_percentile=args.noise_filter_percentile,
            min_gain=args.noise_filter_min_gain,
            n_fft=args.noise_filter_n_fft,
            hop=args.noise_filter_hop,
        )
        filtered_cos = numpy_cosine(orig_to_write, recon_to_write)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    try:
        import soundfile as sf
    except ImportError:
        print("pip install soundfile", file=sys.stderr)
        sys.exit(1)

    o_path = out_dir / f"{stem}_orig.wav"
    r_path = out_dir / f"{stem}_recon.wav"
    sf.write(str(o_path), np.clip(orig_to_write, -1.0, 1.0), output_sr, subtype="PCM_16")
    if args.noise_filter:
        raw_recon_path = out_dir / f"{stem}_recon_raw.wav"
        sf.write(str(raw_recon_path), np.clip(recon, -1.0, 1.0), output_sr, subtype="PCM_16")
    sf.write(str(r_path), np.clip(recon_to_write, -1.0, 1.0), output_sr, subtype="PCM_16")

    enc_stride = encoder_time_stride(cfg)
    codes_bin_path = out_dir / f"{stem}_codes.bin"
    codes_npz_path = out_dir / f"{stem}_codes.npz"
    dur_sec = wav.shape[0] / float(cfg.sample_rate)

    blob: bytes | None = None
    cb_pack = _effective_codebook_sizes_cfg(cfg)
    if codes is not None and not args.no_save_codes:
        code_widths = tuple(int(c.shape[1]) if getattr(c, "ndim", 1) == 2 else 1 for c in codes)
        blob = pack_vq_bitstream(
            codes,
            codebook_sizes=cb_pack,
            sample_rate=cfg.sample_rate,
            audio_samples=int(wav.shape[0]),
        )
        codes_bin_path.write_bytes(blob)
        if args.save_npz_codes:
            np.savez_compressed(
                str(codes_npz_path),
                **{f"indices_q{q}": codes[q] for q in range(len(codes))},
                sample_rate=np.int32(cfg.sample_rate),
                n_codebooks=np.int32(len(codes)),
                codebook_size=np.int32(cfg.codebook_size),
                codebook_sizes=np.array(cb_pack, dtype=np.int32),
                code_widths=np.array(code_widths, dtype=np.int32),
                encoder_stride=np.int32(enc_stride),
                audio_samples=np.int32(wav.shape[0]),
                latent_frames=np.int32(codes[0].shape[0]),
                quantizer=np.array(_quantizer_kind_cfg(cfg)),
                decoder_backend=np.array(_decoder_backend_cfg(cfg)),
            )

    nom_bps = nominal_bitrate_bps(cfg)

    print(f"checkpoint: {ck.name}")
    if args.random_test_clean:
        print("holdout:    LibriSpeech test-clean (not in train-clean-100)")
    print(f"input:      {inp}  ({wav.shape[0] / cfg.sample_rate:.2f}s @ {cfg.sample_rate} Hz)")
    if _uses_lux_vocos_cfg(cfg):
        print(
            f"decoder:    lux_vocos features ({cfg.lux_vocos_feature_dim} mel bins) "
            f"-> frozen Vocos @ {output_sr} Hz"
        )
    if cfg.ae_only:
        print("bitrate:    N/A (ae_only — no VQ codes)")
    else:
        szs = _effective_codebook_sizes_cfg(cfg)
        if _quantizer_kind_cfg(cfg) == "turboquant":
            bits_formula = f"{int(cfg.turboquant_bits)}×{_turboquant_code_dim_cfg(cfg)}"
        else:
            bits_formula = "+".join(f"log2({k})" for k in szs)
        print(
            f"bitrate:    ~{nom_bps / 1000.0:.1f} kbps nominal  "
            f"(({bits_formula})×{cfg.sample_rate}/{enc_stride} Hz frames)"
        )
        if blob is not None and codes is not None:
            hlen = _vq_packed_header_len(blob)
            payload_bits = (len(blob) - hlen) * 8
            eff_kbps = (payload_bits / dur_sec) / 1000.0 if dur_sec > 0 else 0.0
            widths = [int(c.shape[1]) if c.ndim == 2 else 1 for c in codes]
            bppf = sum(bits_per_index(k) * w for k, w in zip(szs, widths))
            theo_bits = int(codes[0].shape[0]) * bppf
            naive_bits = int(codes[0].shape[0]) * sum(widths) * 16
            naive_kbps = naive_bits / dur_sec / 1000.0 if dur_sec > 0 else 0.0
            print(
                f"packed:     ~{eff_kbps:.2f} kbps payload in .bin ({bppf} bits/frame × "
                f"{codes[0].shape[0]} frames; header {hlen} B)"
            )
            print(
                f"            16-bit-per-index storage ~{naive_kbps:.1f} kbps (wasteful vs variable bit-width packing)"
            )
    if args.noise_filter:
        print(
            f"denoise:    spectral gate strength={args.noise_filter_strength:g} "
            f"p{args.noise_filter_percentile:g} min_gain={args.noise_filter_min_gain:g} "
            f"nfft={args.noise_filter_n_fft} hop={args.noise_filter_hop}"
        )
        print(f"mean cos:   raw {100.0 * mean_cos:.1f}%  filtered {100.0 * float(filtered_cos):.1f}%")
    else:
        print(f"mean cos:   {100.0 * mean_cos:.1f}%")
    print(f"wrote:      {o_path}")
    if raw_recon_path is not None:
        print(f"            {raw_recon_path}")
    print(f"            {r_path}")
    if codes is not None and not args.no_save_codes:
        print(f"            {codes_bin_path}  ({len(blob)} bytes)")
        if args.save_npz_codes:
            print(f"            {codes_npz_path}  ({codes_npz_path.stat().st_size} bytes npz)")
    if args.low_bitrate_hint:
        _print_low_bitrate_hint()


if __name__ == "__main__":
    main()
