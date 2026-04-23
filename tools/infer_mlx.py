#!/usr/bin/env python3
"""MLX neural codec inference: load ``codec_step*.npz``, write original + reconstructed WAV.

  Nominal RVQ bitrate: ``sum_i log2(K_i) * (sr / stride)``; uniform K matches ``n_codebooks * log2(K) * …``.
  Compact bitstream: ``*_codes.bin`` — indices packed to ``ceil(log2(K))`` bits each (not uint16).
  Optional ``--save-npz-codes`` for debugging. ``--no-save-codes`` skips both.

  uv run python run.py infer_mlx mlx_checkpoints/codec_step50000.npz -i sample.wav -o out_infer/
  uv run python run.py infer_mlx ckpt.npz --random-dev --seed 42
  uv run python run.py infer_mlx ckpt.npz --random-test-clean   # LibriSpeech test-clean (holdout vs train-clean-100)
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


def _effective_codebook_sizes_cfg(cfg) -> tuple[int, ...]:
    """Match ``train_mlx.effective_codebook_sizes`` without importing MLX train loop."""
    if cfg.codebook_sizes is not None:
        return tuple(int(x) for x in cfg.codebook_sizes)
    return (int(cfg.codebook_size),) * int(cfg.n_codebooks)


def nominal_bitrate_bps(cfg) -> float:
    """RVQ bitstream: ``sum_i log2(K_i) * (sr / encoder_stride)``."""
    import math

    enc_stride = encoder_time_stride(cfg)
    sizes = _effective_codebook_sizes_cfg(cfg)
    return sum(math.log2(float(k)) for k in sizes) * (cfg.sample_rate / enc_stride)


def _vq_packed_header_len(blob: bytes) -> int:
    """``*.bin`` header length: 32 (v1) or 32+2*ncb (v2 variable K)."""
    if len(blob) < 32 or blob[0:4] != _MAGIC:
        return 32
    if blob[4] == 2:
        ncb = struct.unpack_from("<H", blob, 5)[0]
        return 32 + 2 * int(ncb)
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
    """Header + tightly packed indices: per frame, q0..q_{Q-1} with ``bits_per_index(K_q)`` each.

    v1 (uniform K): legacy 32-byte header. v2 (mixed K): header length ``32 + 2*ncb`` with uint16 ``K_q`` tail.
    """
    ncb = len(codes)
    if len(codebook_sizes) != ncb:
        raise ValueError(f"codebook_sizes len {len(codebook_sizes)} != ncb {ncb}")
    n_frames = int(codes[0].shape[0])
    uniform = len(set(int(k) for k in codebook_sizes)) == 1

    out = bytearray()
    bit_buf = 0
    bit_n = 0

    if uniform:
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
                v = int(codes[q][t]) & mask
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
            v = int(codes[q][t]) & mask
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
        "  Samo packowanie binarne usuwa ~44% marnotrawstwa uint16 vs 9 bitów/indeks na dysku.\n\n"
    )


def validation_entries(manifest_path: Path) -> list[dict]:
    """Match ``MultilingualSpeechDataset(..., mode='dev')`` in data_pipeline (hash 10% or explicit dev/test)."""
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


def infer_waveform(
    tm, model, cfg, wav_1d: "numpy.ndarray"
) -> tuple["numpy.ndarray", float, list["numpy.ndarray"] | None]:
    """Chunked encode–decode; returns recon, mean cos, and per-codebook VQ index streams (uint16) or None if ae_only."""
    import mlx.core as mx
    import numpy as np

    seg = cfg.segment
    T = int(wav_1d.shape[0])
    outs: list[np.ndarray] = []
    cos_vals: list[float] = []
    codes_acc: list[list[np.ndarray]] | None = None if cfg.ae_only else [[] for _ in range(cfg.n_codebooks)]

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
                a = np.asarray(idx_list[q][0], dtype=np.uint16)
                codes_acc[q].append(a)
        valid = end - start
        y_np = np.array(y[0, :valid, 0], dtype=np.float32)
        outs.append(y_np)
        cos = tm.batch_mean_cosine(x[:, :valid, :], y[:, :valid, :])
        mx.eval(cos)
        cos_vals.append(float(cos.item()))
        start = end

    recon = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
    mean_cos = float(sum(cos_vals) / max(len(cos_vals), 1))
    if codes_acc is not None:
        codes = [np.concatenate(codes_acc[q]) for q in range(cfg.n_codebooks)]
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
    p.add_argument(
        "--pre-vq-layernorm",
        action=argparse.BooleanOptionalAction,
        default=dcfg.pre_vq_layernorm,
        help="Must match training (--pre-vq-layernorm / --no-pre-vq-layernorm)",
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
        help="Also write *_codes.npz (uint16 per index; debug)",
    )
    p.add_argument(
        "--low-bitrate-hint",
        action="store_true",
        help="Print stderr note on reaching ~1 kbps (36× lower than ~36 kbps nominal)",
    )
    args = p.parse_args()

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
        enc_channels=enc_ch,
        stride1_blocks_per_scale=args.stride1_blocks_per_scale,
        latent_dim=args.latent_dim,
        pre_vq_layernorm=args.pre_vq_layernorm,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        codebook_sizes=cb_sizes_infer,
        vq_cosine=args.vq_cosine,
        ae_only=bool(args.ae_only),
    )
    model = tm.MLXCodec(cfg)
    try:
        model.load_weights(str(ck))
    except Exception as e:
        print(f"load_weights failed: {e}", file=sys.stderr)
        print("Hint: training Config must match (try without --ae-only or with it).", file=sys.stderr)
        sys.exit(1)

    wav = load_audio_mono(inp, cfg.sample_rate, args.max_seconds)
    recon, mean_cos, codes = infer_waveform(tm, model, cfg, wav)

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
    sf.write(str(o_path), np.clip(wav, -1.0, 1.0), cfg.sample_rate, subtype="PCM_16")
    sf.write(str(r_path), np.clip(recon, -1.0, 1.0), cfg.sample_rate, subtype="PCM_16")

    enc_stride = encoder_time_stride(cfg)
    codes_bin_path = out_dir / f"{stem}_codes.bin"
    codes_npz_path = out_dir / f"{stem}_codes.npz"
    dur_sec = wav.shape[0] / float(cfg.sample_rate)

    blob: bytes | None = None
    cb_pack = _effective_codebook_sizes_cfg(cfg)
    if codes is not None and not args.no_save_codes:
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
                n_codebooks=np.int32(cfg.n_codebooks),
                codebook_size=np.int32(cfg.codebook_size),
                codebook_sizes=np.array(cb_pack, dtype=np.int32),
                encoder_stride=np.int32(enc_stride),
                audio_samples=np.int32(wav.shape[0]),
                latent_frames=np.int32(codes[0].shape[0]),
            )

    nom_bps = nominal_bitrate_bps(cfg)

    print(f"checkpoint: {ck.name}")
    if args.random_test_clean:
        print("holdout:    LibriSpeech test-clean (not in train-clean-100)")
    print(f"input:      {inp}  ({wav.shape[0] / cfg.sample_rate:.2f}s @ {cfg.sample_rate} Hz)")
    if cfg.ae_only:
        print("bitrate:    N/A (ae_only — no VQ codes)")
    else:
        szs = _effective_codebook_sizes_cfg(cfg)
        bits_formula = "+".join(f"log2({k})" for k in szs)
        print(
            f"bitrate:    ~{nom_bps / 1000.0:.1f} kbps nominal  "
            f"(({bits_formula})×{cfg.sample_rate}/{enc_stride} Hz frames)"
        )
        if blob is not None and codes is not None:
            hlen = _vq_packed_header_len(blob)
            payload_bits = (len(blob) - hlen) * 8
            eff_kbps = (payload_bits / dur_sec) / 1000.0 if dur_sec > 0 else 0.0
            bppf = sum(bits_per_index(k) for k in szs)
            theo_bits = int(codes[0].shape[0]) * bppf
            bp_avg = bppf / max(len(szs), 1)
            naive_kbps = (theo_bits * (16 / bp_avg)) / dur_sec / 1000.0 if dur_sec > 0 else 0.0
            print(
                f"packed:     ~{eff_kbps:.2f} kbps payload in .bin ({bppf} bits/frame × "
                f"{codes[0].shape[0]} frames; header {hlen} B)"
            )
            print(
                f"            uint16-per-index storage ~{naive_kbps:.1f} kbps (wasteful vs variable bit-width packing)"
            )
    print(f"mean cos:   {100.0 * mean_cos:.1f}%")
    print(f"wrote:      {o_path}")
    print(f"            {r_path}")
    if codes is not None and not args.no_save_codes:
        print(f"            {codes_bin_path}  ({len(blob)} bytes)")
        if args.save_npz_codes:
            print(f"            {codes_npz_path}  ({codes_npz_path.stat().st_size} bytes npz)")
    if args.low_bitrate_hint:
        _print_low_bitrate_hint()


if __name__ == "__main__":
    main()
