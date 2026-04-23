"""
Multilingual speech data pipeline for codec training.

Datasets:
  - LibriSpeech (en): train-clean-100, train-clean-360, train-other-500, dev/test
  - CommonVoice (multilingual): en, de, fr, es, it, nl, pt, ru, zh, ja, ar, hi
  - VCTK (en): 110 speakers, diverse accents
  - DNS Challenge noise: augmentation only (babble, music, noise, reverberation)

Features:
  - Streaming download + extract (tarball/parquet)
  - Resample to 16kHz mono
  - Random segment extraction (configurable length)
  - On-the-fly augmentation: SNR noise mixing, reverberation, speed perturbation
  - Weighted sampling across languages for balanced multilingual training
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as TT
from pathlib import Path
import json, math, random, subprocess, csv, time, hashlib, shutil
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import soundfile as sf


def load_audio(path: str):
    """Load audio file, preferring soundfile for FLAC compatibility."""
    if str(path).lower().endswith('.flac'):
        data, sr = sf.read(path, dtype='float32')
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)
        return waveform, sr
    return torchaudio.load(str(path))


# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
@dataclass
class DataConfig:
    data_dir: str = "data"  # root directory for all datasets
    sample_rate: int = 16000
    segment_length: int = 16000  # 1 second
    batch_size: int = 16
    # >0 enables DataLoader prefetch + persistent workers in create_dataloaders; 0 = sync load in the
    # main process (GPU may idle between batches). Micro-benchmarks often set num_workers=0 on purpose
    # (see tools/bench_fps.py) to isolate model step time from worker jitter.
    num_workers: int = 4

    # Strong preprocessing / robustness
    enable_preprocessing: bool = True
    remove_dc: bool = True
    trim_silence: bool = True
    trim_db_threshold: float = 35.0
    silence_margin_ms: float = 20.0
    pad_short_with_repeat: bool = True
    target_rms: float = 0.05
    max_abs_value: float = 0.98
    preemphasis: bool = True
    preemphasis_coeff: float = 0.97

    # Preprocessed cache (run once, then only load)
    use_preprocessed_cache: bool = False
    preprocessed_dir: str = "preprocessed"
    preprocessed_manifest_name: str = "master_manifest_preprocessed.jsonl"

    # Dataset selection
    use_librispeech: bool = True
    librispeech_subsets: list = ("train-clean-100", "train-clean-360", "train-other-500")

    use_commonvoice: bool = True
    commonvoice_languages: list = ("en", "de", "fr", "es", "it", "nl", "pt", "ru", "zh", "ja", "ar", "hi")

    use_vctk: bool = True

    use_dns_noise: bool = True  # augmentation only

    # Augmentation
    augment_noise_snr_range: Tuple[float, float] = (5.0, 40.0)
    augment_reverb_prob: float = 0.3
    augment_speed_perturb_factors: tuple = (0.9, 1.0, 1.1)
    augment_speed_perturb_prob: float = 0.3
    enable_augmentation: bool = True

    # Language balancing: target fractions per batch (None = equal)
    language_weights: Optional[dict] = None  # {"en": 0.4, "de": 0.1, ...}

    # Download control
    max_datasets: Optional[int] = None  # limit for testing


# ═══════════════════════════════════════════════
# PREPROCESSING HELPERS
# ═══════════════════════════════════════════════
def _trim_silence_waveform(cfg: DataConfig, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Trim leading and trailing silence by relative dB threshold."""
    if waveform.numel() == 0:
        return waveform

    abs_wav = waveform.abs().squeeze(0)
    peak = abs_wav.max()
    if not torch.isfinite(peak) or peak <= 0:
        return waveform

    thresh = peak * 10 ** (-cfg.trim_db_threshold / 20.0)
    keep = abs_wav > thresh
    if not torch.any(keep):
        return waveform

    idx = torch.nonzero(keep).flatten()
    margin = int(sample_rate * (cfg.silence_margin_ms / 1000.0))
    start = int(idx.min()) - margin
    end = int(idx.max()) + margin + 1
    start = max(start, 0)
    end = min(end, waveform.size(1))
    return waveform[:, start:end]


def _pre_emphasize_waveform(cfg: DataConfig, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.size(1) <= 1:
        return waveform
    coeff = float(cfg.preemphasis_coeff)
    if coeff <= 0 or not torch.isfinite(torch.tensor(coeff)):
        return waveform
    return F.pad(waveform[:, 1:] - coeff * waveform[:, :-1], (1, 0))


def _normalize_rms_waveform(waveform: torch.Tensor, target_rms: float) -> torch.Tensor:
    eps = 1e-8
    rms = torch.sqrt((waveform ** 2).mean() + eps)
    if not torch.isfinite(rms) or rms <= eps or target_rms <= 0:
        return waveform
    return waveform * (target_rms / rms)


def apply_preprocessing(cfg: DataConfig, waveform: torch.Tensor, sample_rate: int, finalize: bool = False) -> torch.Tensor:
    """Apply strict preprocessing steps.

    Set finalize=True to run only final hardening (nan cleanup / RMS / clamp).
    """
    if not cfg.enable_preprocessing:
        return waveform

    if finalize:
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = _normalize_rms_waveform(waveform, cfg.target_rms)
        if cfg.max_abs_value > 0:
            waveform = torch.clamp(waveform, -cfg.max_abs_value, cfg.max_abs_value)
        return waveform

    if cfg.remove_dc:
        waveform = waveform - waveform.mean(dim=1, keepdim=True)

    if cfg.trim_silence:
        waveform = _trim_silence_waveform(cfg, waveform, sample_rate)

    if cfg.preemphasis:
        waveform = _pre_emphasize_waveform(cfg, waveform)

    return waveform


# ═══════════════════════════════════════════════
# UTILITY: run command, stream output
# ═══════════════════════════════════════════════
def _run(cmd: str, cwd=None, env=None):
    """Run shell command, print output, raise on failure."""
    print(f"  >> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env,
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout[-500:])  # last 500 chars
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout


def _download_url(url: str, dest: Path):
    """Download using wget or curl (macOS often lacks wget).

    If ``dest`` already exists with non-zero size, resumes partial downloads
    (``wget -c`` / ``curl -C -``) so flaky transfers can be retried.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume = dest.exists() and dest.stat().st_size > 0
    print(f"  >> download -> {dest}" + (" (resume)" if resume else ""))
    if shutil.which("wget"):
        cmd = ["wget", "-q", "--show-progress", "-O", str(dest), url]
        if resume:
            cmd.insert(1, "-c")
        r = subprocess.run(cmd, capture_output=True, text=True)
    elif shutil.which("curl"):
        cmd = [
            "curl", "-fL", "--retry", "5", "--retry-delay", "5",
            "--progress-bar", "-o", str(dest), url,
        ]
        if resume:
            cmd[1:1] = ["-C", "-"]
        r = subprocess.run(cmd, capture_output=True, text=True)
    else:
        raise RuntimeError("Neither wget nor curl found in PATH")
    if r.returncode != 0:
        err = (r.stderr or "")[-500:]
        raise RuntimeError(f"Download failed ({url}): {err}")


# ═══════════════════════════════════════════════
# DATASET DOWNLOADERS
# ═══════════════════════════════════════════════

def download_librispeech(data_dir: str, subsets: list, max_datasets=None):
    """Download LibriSpeech train-clean-100/360, train-other-500.

    Source: openslr.org (mirror) or OpenSLR.
    Format: .flac files organized by speaker/chapter.
    """
    base = Path(data_dir) / "librispeech"
    base.mkdir(parents=True, exist_ok=True)

    urls = {
        "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    }

    for subset in subsets:
        if max_datasets is not None and max_datasets <= 0:
            break
        subset_dir = base / subset
        manifest = subset_dir / "manifest.jsonl"
        if manifest.exists():
            print(f"[LibriSpeech] {subset}: already downloaded ({manifest.stat().st_size} bytes manifest)")
            if max_datasets is not None:
                max_datasets -= 1
            continue

        print(f"[LibriSpeech] Downloading {subset}...")
        tar_path = base / f"{subset}.tar.gz"
        if not tar_path.exists():
            _download_url(urls[subset], tar_path)
        _run(f"tar xzf {tar_path} -C {base}")
        _run(f"rm {tar_path}")

        # Build manifest
        print(f"[LibriSpeech] Building manifest for {subset}...")
        flac_files = list((base / "LibriSpeech" / subset).rglob("*.flac"))
        entries = []
        for f in flac_files:
            rel = f.relative_to(base / "LibriSpeech" / subset)
            entries.append({
                "path": str(f),
                "relative_path": str(rel),
                "language": "en",
                "dataset": "librispeech",
                "subset": subset,
            })
        with open(manifest, "w") as fout:
            for e in entries:
                fout.write(json.dumps(e) + "\n")
        print(f"[LibriSpeech] {subset}: {len(entries)} files")
        if max_datasets is not None:
            max_datasets -= 1


def download_commonvoice(data_dir: str, languages: list, max_datasets=None):
    """Download Common Voice multilingual corpus.

    Source: commonvoice.mozilla.org or OPUS/openslr mirrors.
    Format: .mp3 files + tsv metadata.
    We use the HuggingFace datasets mirror for easier access.
    """
    base = Path(data_dir) / "commonvoice"
    base.mkdir(parents=True, exist_ok=True)

    # Try using huggingface datasets library if available
    try:
        from datasets import load_dataset
        print("[CommonVoice] Using HuggingFace datasets library")
        use_hf = True
    except ImportError:
        print("[CommonVoice] HuggingFace datasets not available, falling back to manual download")
        use_hf = False

    if use_hf:
        for lang in languages:
            if max_datasets is not None and max_datasets <= 0:
                break
            lang_dir = base / lang
            manifest = lang_dir / "manifest.jsonl"
            if manifest.exists():
                print(f"[CommonVoice] {lang}: manifest exists, skipping")
                if max_datasets is not None:
                    max_datasets -= 1
                continue

            print(f"[CommonVoice] Loading {lang} via HuggingFace...")
            try:
                cv_lang_code = lang if lang != "zh" else "zh-CN"
                ds = load_dataset(f"mozilla-foundation/common_voice_11_0", cv_lang_code,
                                  split="train", streaming=True)
                # For streaming, we can't enumerate all files easily.
                # Instead, we note this and use the streaming approach in the dataset class.
                print(f"[CommonVoice] {lang}: streaming mode (no local manifest)")
                # Create a marker file
                (lang_dir / "hf_streaming.txt").write_text(f"common_voice_11_0:{lang}")
                lang_dir.mkdir(parents=True, exist_ok=True)
                if max_datasets is not None:
                    max_datasets -= 1
            except Exception as e:
                print(f"[CommonVoice] Failed to load {lang}: {e}")
    else:
        print("[CommonVoice] Install 'pip install datasets' for automatic multilingual download")
        print("  Or manually download from https://commonvoice.mozilla.org/en/datasets")


def download_vctk(data_dir: str, max_datasets=None):
    """Download VCTK corpus (110 English speakers, diverse accents).

    Source: https://datashare.ed.ac.uk/handle/10283/3443
    Format: .flac files
    """
    base = Path(data_dir) / "vctk"
    base.mkdir(parents=True, exist_ok=True)
    manifest = base / "manifest.jsonl"

    if manifest.exists():
        print("[VCTK] Already downloaded")
        return

    print("[VCTK] Downloading...")
    zip_path = base / "VCTK-Corpus.tar.gz"
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus.tar.gz"
    # Fallback mirror:
    alt_url = "https://www.openslr.org/resources/59/VCTK-Corpus.tar.gz"

    try:
        _download_url(url, zip_path)
    except RuntimeError:
        print("[VCTK] Primary URL failed, trying alternate...")
        _download_url(alt_url, zip_path)

    _run(f"tar xzf {zip_path} -C {base}")
    _run(f"rm {zip_path}")

    # Build manifest
    print("[VCTK] Building manifest...")
    wav_files = list(base.rglob("*.wav")) + list(base.rglob("*.flac"))
    entries = []
    for f in wav_files:
        entries.append({
            "path": str(f),
            "language": "en",
            "dataset": "vctk",
        })
    with open(manifest, "w") as fout:
        for e in entries:
            fout.write(json.dumps(e) + "\n")
    print(f"[VCTK] {len(entries)} files")


def download_dns_noise(data_dir: str):
    """Download DNS Challenge noise dataset for augmentation.

    Source: https://github.com/microsoft/DNS-Challenge
    Contains: babble, music, noise, reverberation (RIRs)
    """
    base = Path(data_dir) / "dns_noise"
    base.mkdir(parents=True, exist_ok=True)

    if (base / "noise").exists() or (base / "download_done.txt").exists():
        print("[DNS Noise] Already downloaded")
        return

    print("[DNS Noise] Downloading...")
    try:
        # Clone the DNS Challenge repo (just the noise files)
        _run(f"git clone --depth 1 https://github.com/microsoft/DNS-Challenge.git {base}/repo")
        _run(f"mv {base}/repo/datasets/noise {base}/noise")
        _run(f"mv {base}/repo/datasets/rir {base}/rir")
        _run(f"rm -rf {base}/repo")
        (base / "download_done.txt").write_text("done")
    except RuntimeError as e:
        print(f"[DNS Noise] Download failed: {e}")
        print("  Noise augmentation will use whatever is available in {base}/noise")
        (base / "download_done.txt").write_text("partial")


# ═══════════════════════════════════════════════
# MANIFEST BUILDING
# ═══════════════════════════════════════════════

def build_manifests(data_dir: str):
    """Build all manifests. Call after downloads complete."""
    base = Path(data_dir)

    # Combine all manifests into a master manifest
    all_entries = []

    for manifest_path in base.rglob("manifest.jsonl"):
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    all_entries.append(json.loads(line))

    master = base / "master_manifest.jsonl"
    with open(master, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[Master Manifest] {len(all_entries)} total entries → {master}")
    return all_entries


def build_preprocessed_dataset(data_config: DataConfig, force: bool = False, max_items: Optional[int] = None):
    """Precompute strict preprocessing for all raw files and build a ready-to-load manifest.

    This is intended for offline preparation. After this, set use_preprocessed_cache=True
    in DataConfig / training args to only load cached files at train time.
    """
    raw_manifest = Path(data_config.data_dir) / "master_manifest.jsonl"
    if not raw_manifest.exists():
        raise FileNotFoundError(
            f"No master manifest found at {raw_manifest}. Run download_and_prepare() first."
        )

    pre_dir = Path(data_config.data_dir) / data_config.preprocessed_dir
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre_manifest = pre_dir / data_config.preprocessed_manifest_name

    # Load entries
    entries = []
    with open(raw_manifest) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if max_items is not None and max_items > 0:
        entries = entries[:max_items]

    processed = []
    augmentor = AudioAugmentor(data_config)
    cached = 0
    written = 0
    skipped = 0

    for entry in entries:
        src_path = entry.get("path")
        if not src_path:
            continue
        out_name = f"{hashlib.md5(src_path.encode()).hexdigest()}.wav"
        out_path = pre_dir / out_name

        if out_path.exists() and not force:
            skipped += 1
        else:
            try:
                waveform, src_sr = load_audio(src_path)
            except Exception as e:
                print(f"[Precompute] Failed load {src_path}: {e}")
                continue

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if src_sr != data_config.sample_rate:
                resampler = augmentor._get_resampler(src_sr, data_config.sample_rate)
                waveform = resampler(waveform)

            waveform = apply_preprocessing(data_config, waveform, data_config.sample_rate, finalize=False)
            waveform = apply_preprocessing(data_config, waveform, data_config.sample_rate, finalize=True)
            if not torch.isfinite(waveform).all():
                print(f"[Precompute] Non-finite values in {src_path}, skipping")
                continue

            try:
                sf.write(
                    str(out_path),
                    waveform.squeeze(0).detach().cpu().float().numpy(),
                    data_config.sample_rate,
                )
            except Exception as e:
                print(f"[Precompute] Failed write {out_path}: {e}")
                continue
            written += 1

        out_entry = dict(entry)
        out_entry["path"] = str(out_path)
        out_entry["source_path"] = src_path
        out_entry["preprocessed"] = True
        processed.append(out_entry)
        cached += 1

    with open(pre_manifest, "w") as f:
        for entry in processed:
            f.write(json.dumps(entry) + "\n")

    print(f"[Precompute] Saved {len(processed)} preprocessed entries → {pre_manifest}")
    print(f"[Precompute] New files: {written}, reused: {skipped}")
    return pre_manifest


# ═══════════════════════════════════════════════
# AUDIO AUGMENTATION
# ═══════════════════════════════════════════════

class AudioAugmentor:
    """On-the-fly audio augmentation.

    1. Speed perturbation (0.9x, 1.0x, 1.1x) — simulates speaker variation
    2. Noise mixing at random SNR — robustness to background noise
    3. Reverberation (convolution with RIR) — room acoustics
    4. Random volume/gain variation
    """

    def __init__(self, cfg: DataConfig, noise_dir: Optional[str] = None, rir_dir: Optional[str] = None):
        self.cfg = cfg
        self.sr = cfg.sample_rate

        # Load noise files for augmentation
        self.noise_files = []
        if noise_dir and Path(noise_dir).exists():
            self.noise_files = list(Path(noise_dir).rglob("*.wav"))
            self.noise_files += list(Path(noise_dir).rglob("*.flac"))
            if self.noise_files:
                print(f"[Augmentor] Loaded {len(self.noise_files)} noise files")

        # Load RIR files for reverberation
        self.rir_files = []
        if rir_dir and Path(rir_dir).exists():
            self.rir_files = list(Path(rir_dir).rglob("*.wav"))
            self.rir_files += list(Path(rir_dir).rglob("*.flac"))
            if self.rir_files:
                print(f"[Augmentor] Loaded {len(self.rir_files)} RIR files")

        # Resampler cache
        self._resampler_cache = {}

    def _get_resampler(self, src_sr, dst_sr):
        key = (src_sr, dst_sr)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = TT.Resample(src_sr, dst_sr)
        return self._resampler_cache[key]

    def augment(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Apply random augmentations to a waveform.

        Args:
            waveform: [1, T] at sample_rate
            sample_rate: original sample rate of the waveform
        Returns:
            [1, T'] possibly at different length due to speed perturbation
        """
        # 1. Speed perturbation
        if random.random() < self.cfg.augment_speed_perturb_prob:
            factor = random.choice(self.cfg.augment_speed_perturb_factors)
            if factor != 1.0:
                waveform = self._speed_perturb(waveform, sample_rate, factor)

        # 2. Noise mixing
        if self.noise_files and random.random() < 0.5:
            snr = random.uniform(*self.cfg.augment_noise_snr_range)
            waveform = self._add_noise(waveform, sample_rate, snr)

        # 3. Reverberation
        if self.rir_files and random.random() < self.cfg.augment_reverb_prob:
            waveform = self._add_reverb(waveform, sample_rate)

        # 4. Random gain
        gain = random.uniform(0.5, 1.5)
        waveform = waveform * gain

        # Clamp
        waveform = torch.clamp(waveform, -1.0, 1.0)
        return waveform

    def _speed_perturb(self, waveform, sr, factor):
        """Speed perturbation via resampling."""
        # torchaudio: change speed by resampling
        resampler = self._get_resampler(sr, int(sr * factor))
        sped = resampler(waveform)
        # Resample back to original length (conceptually same rate)
        resampler_back = self._get_resampler(int(sr * factor), sr)
        return resampler_back(sped)

    def _add_noise(self, waveform, sr, snr_db):
        """Mix a random noise file at the specified SNR."""
        if not self.noise_files:
            return waveform

        noise_path = random.choice(self.noise_files)
        try:
            noise_wave, noise_sr = load_audio(str(noise_path))
        except Exception:
            return waveform

        # Resample noise if needed
        if noise_sr != sr:
            resampler = self._get_resampler(noise_sr, sr)
            noise_wave = resampler(noise_wave)

        # Convert to mono
        if noise_wave.size(0) > 1:
            noise_wave = noise_wave.mean(dim=0, keepdim=True)

        # Match length
        target_len = waveform.size(1)
        if noise_wave.size(1) < target_len:
            noise_wave = noise_wave.repeat(1, (target_len // noise_wave.size(1)) + 1)
        noise_wave = noise_wave[:, :target_len]

        # Compute SNR
        signal_power = (waveform ** 2).mean()
        noise_power = (noise_wave ** 2).mean()
        if noise_power < 1e-10:
            return waveform
        snr_linear = 10 ** (snr_db / 10)
        desired_noise_power = signal_power / snr_linear
        noise_wave = noise_wave * math.sqrt(desired_noise_power / noise_power)

        return waveform + noise_wave

    def _add_reverb(self, waveform, sr):
        """Convolve with a random RIR."""
        if not self.rir_files:
            return waveform

        rir_path = random.choice(self.rir_files)
        try:
            rir_wave, rir_sr = load_audio(str(rir_path))
        except Exception:
            return waveform

        if rir_sr != sr:
            resampler = self._get_resampler(rir_sr, sr)
            rir_wave = resampler(rir_wave)

        if rir_wave.size(0) > 1:
            # Pick one channel or average
            rir_wave = rir_wave.mean(dim=0, keepdim=True)

        # Normalize RIR
        rir_max = rir_wave.abs().max().clamp_min(1e-8)
        rir_wave = rir_wave / rir_max

        # Convolve (simple 1D convolution)
        # waveform: [1, T], rir: [1, T_rir]
        rir = rir_wave.squeeze(0)
        # Use torch.nn.functional.conv1d
        w = waveform.unsqueeze(0)  # [1, 1, T]
        r = rir.view(1, 1, -1).to(waveform.device)  # [1, 1, T_rir]
        # Flip RIR for convolution
        rir_flipped = r.flip(-1)
        convolved = F.conv1d(w, rir_flipped, padding=rir.size(-1) - 1)
        # Extract valid part
        result = convolved.squeeze(0)  # [1, T']
        return result[:, :waveform.size(1)]


# ═══════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════

class MultilingualSpeechDataset(Dataset):
    """Unified dataset combining all speech sources.

    Reads from a master manifest and provides random segments.
    Supports weighted sampling for language balancing.
    """

    def __init__(self, data_config: DataConfig, mode: str = "train"):
        self.cfg = data_config
        self.mode = mode
        self.sr = data_config.sample_rate
        self.segment_len = data_config.segment_length
        self._skip_heavy_preprocessing = data_config.use_preprocessed_cache

        # Load manifest
        master_path = Path(data_config.data_dir) / "master_manifest.jsonl"
        if data_config.use_preprocessed_cache:
            preprocessed_path = Path(data_config.data_dir) / data_config.preprocessed_dir / data_config.preprocessed_manifest_name
            if preprocessed_path.exists():
                master_path = preprocessed_path
                print(f"[Dataset] Loading preprocessed manifest: {master_path}")
            else:
                print(
                    f"[Dataset] WARNING: preprocessed manifest missing ({preprocessed_path}); "
                    "falling back to raw manifest and running online preprocessing"
                )
                self._skip_heavy_preprocessing = False

        if not master_path.exists():
            raise FileNotFoundError(
                f"No master manifest found at {master_path}. "
                "Run download_and_prepare() first."
            )

        # Read all entries from manifest
        all_entries = []
        with open(master_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    all_entries.append(entry)

        # Separate explicit test/dev/val subsets from training subsets
        dev_explicit = [e for e in all_entries if "test" in e.get("subset","") or "val" in e.get("subset","") or "dev" in e.get("subset","")]
        train_pool = [e for e in all_entries if "test" not in e.get("subset","") and "val" not in e.get("subset","") and "dev" not in e.get("subset","")]

        # Deterministic 90/10 split via hash — ensures train and val are always disjoint
        train_entries, dev_entries = [], []
        for e in train_pool:
            h = int(hashlib.md5((e["path"] + "_split").encode()).hexdigest(), 16) % 10
            if h < 9:
                train_entries.append(e)
            else:
                dev_entries.append(e)

        if mode == "train":
            if dev_explicit:
                # Have explicit dev set — use full training pool
                self.entries = train_pool
            else:
                # No explicit dev — use 90% hash split
                self.entries = train_entries
        elif dev_explicit:
            # Have explicit dev set — use it
            self.entries = dev_explicit
        else:
            # No explicit dev — use 10% hash split
            self.entries = dev_entries

        print(f"[Dataset] mode={mode}: {len(self.entries)} entries")

        # Language statistics for weighted sampling
        self._lang_counts = {}
        for e in self.entries:
            lang = e.get("language", "unknown")
            self._lang_counts[lang] = self._lang_counts.get(lang, 0) + 1

        # Build per-language index lists
        self._lang_indices = {}
        for i, e in enumerate(self.entries):
            lang = e.get("language", "unknown")
            if lang not in self._lang_indices:
                self._lang_indices[lang] = []
            self._lang_indices[lang].append(i)

        print(f"[Dataset] Languages: {self._lang_counts}")

        # Augmentor
        noise_dir = Path(data_config.data_dir) / "dns_noise" / "noise"
        rir_dir = Path(data_config.data_dir) / "dns_noise" / "rir"
        self.augmentor = AudioAugmentor(data_config, str(noise_dir), str(rir_dir))

    def __len__(self):
        return max(len(self.entries) * 10, 100_000)  # virtual length for epoch control

    def __getitem__(self, idx):
        # Map virtual index to real entry (with wrapping)
        entry_idx = idx % len(self.entries)
        entry = self.entries[entry_idx]

        # Load audio
        audio_path = entry["path"]
        try:
            waveform, orig_sr = load_audio(audio_path)
        except Exception as e:
            # Return silence on failure (rare)
            print(f"[Dataset] Failed to load {audio_path}: {e}")
            return torch.zeros(1, self.segment_len), "unknown"

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != self.sr:
            resampler = self.augmentor._get_resampler(orig_sr, self.sr)
            waveform = resampler(waveform)

        # Strong preprocessing before augmentation (skip when precomputed cache is used)
        if not self._skip_heavy_preprocessing:
            waveform = self._prepare_waveform(waveform)

        # Extract random segment or pad
        waveform = self._extract_segment(waveform)

        # Augment (train only)
        if self.mode == "train" and self.cfg.enable_augmentation:
            waveform = self.augmentor.augment(waveform, self.sr)

        # Final hardening (precomputed cache already stores hardened waveforms)
        if not self._skip_heavy_preprocessing:
            waveform = self._finalize_waveform(waveform)

        language = entry.get("language", "unknown")
        return waveform, language

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply strict preprocessing before augmentation."""
        return apply_preprocessing(self.cfg, waveform, self.sr, finalize=False)

    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim leading and trailing silence by relative dB threshold."""
        return _trim_silence_waveform(self.cfg, waveform, self.sr)

    def _pre_emphasize(self, waveform: torch.Tensor) -> torch.Tensor:
        return _pre_emphasize_waveform(self.cfg, waveform)

    def _extract_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        total_len = waveform.size(1)
        if total_len == 0:
            return torch.zeros(1, self.segment_len, device=waveform.device)

        if total_len < self.segment_len:
            if self.cfg.pad_short_with_repeat:
                repeat = (self.segment_len // total_len) + 1
                return waveform.repeat(1, repeat)[:, : self.segment_len]
            return F.pad(waveform, (0, self.segment_len - total_len))

        start = random.randint(0, total_len - self.segment_len)
        return waveform[:, start : start + self.segment_len]

    def _normalize_rms(self, waveform: torch.Tensor, target_rms: float) -> torch.Tensor:
        eps = 1e-8
        rms = torch.sqrt((waveform ** 2).mean() + eps)
        if not torch.isfinite(rms) or rms <= eps or target_rms <= 0:
            return waveform
        return waveform * (target_rms / rms)

    def _finalize_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        return apply_preprocessing(self.cfg, waveform, self.sr, finalize=True)

    def get_language_weights(self):
        """Compute sampling weights for balanced language representation."""
        if self.cfg.language_weights is not None:
            weights = self.cfg.language_weights
        else:
            # Equal weight per language
            n_lang = len(self._lang_counts)
            weights = {lang: 1.0 / n_lang for lang in self._lang_counts}

        # Build per-sample weights
        sample_weights = torch.zeros(len(self.entries))
        for i, entry in enumerate(self.entries):
            lang = entry.get("language", "unknown")
            lang_total = self._lang_counts.get(lang, 1)
            target_weight = weights.get(lang, 1.0 / len(self._lang_counts))
            # Oversample underrepresented languages
            sample_weights[i] = target_weight / lang_total

        # Normalize
        sample_weights = sample_weights / sample_weights.sum()
        return sample_weights


# ═══════════════════════════════════════════════
# DATA LOADER FACTORY
# ═══════════════════════════════════════════════

def create_dataloaders(
    data_config: DataConfig,
    train_batch_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with language-balanced sampling."""

    train_bs = train_batch_size or data_config.batch_size
    val_bs = val_batch_size or data_config.batch_size

    # Train dataset with balanced sampling
    train_ds = MultilingualSpeechDataset(data_config, mode="train")
    sample_weights = train_ds.get_language_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Pinned memory only helps CPU tensors → CUDA; MPS ignores it and logs a warning.
    pin = torch.cuda.is_available()
    nw = data_config.num_workers
    _dl_extras = {}
    if nw > 0:
        _dl_extras["persistent_workers"] = True
        _dl_extras["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
        **_dl_extras,
    )

    # Validation dataset (no augmentation)
    val_ds = MultilingualSpeechDataset(data_config, mode="dev")
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        drop_last=False,
        **_dl_extras,
    )

    return train_loader, val_loader


# ═══════════════════════════════════════════════
# HIGH-LEVEL SETUP
# ═══════════════════════════════════════════════

def download_and_prepare(data_config: DataConfig):
    """Download all datasets and build manifests."""
    data_dir = data_config.data_dir

    print("=" * 60)
    print("DOWNLOADING MULTILINGUAL SPEECH DATA")
    print("=" * 60)

    if data_config.use_librispeech:
        download_librispeech(data_dir, data_config.librispeech_subsets,
                             max_datasets=data_config.max_datasets)

    if data_config.use_commonvoice:
        download_commonvoice(data_dir, data_config.commonvoice_languages,
                             max_datasets=data_config.max_datasets)

    if data_config.use_vctk:
        download_vctk(data_dir, max_datasets=data_config.max_datasets)

    if data_config.use_dns_noise:
        download_dns_noise(data_dir)

    print("=" * 60)
    print("BUILDING MANIFESTS")
    print("=" * 60)
    build_manifests(data_dir)


# ═══════════════════════════════════════════════
# QUICK TEST / ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        cfg = DataConfig(
            data_dir=sys.argv[2] if len(sys.argv) > 2 else "data",
        )
        download_and_prepare(cfg)
    elif len(sys.argv) > 1 and sys.argv[1] == "precompute":
        data_dir = "data"
        force = False
        max_items = None
        args = sys.argv[2:]
        if len(args) > 0 and not args[0].startswith("--"):
            data_dir = args[0]
            args = args[1:]
        i = 0
        while i < len(args):
            if args[i] == "--force":
                force = True
            elif args[i].startswith("--max-items="):
                max_items = int(args[i].split("=", 1)[1])
            elif args[i] == "--max-items" and i + 1 < len(args):
                max_items = int(args[i + 1])
                i += 1
            i += 1

        cfg = DataConfig(data_dir=data_dir)
        build_preprocessed_dataset(cfg, force=force, max_items=max_items)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        cfg = DataConfig(data_dir="data", batch_size=4)
        ds = MultilingualSpeechDataset(cfg, mode="train")
        print(f"\nDataset length (virtual): {len(ds)}")

        # Test a few samples
        for i in range(5):
            wave, lang = ds[i]
            print(f"  Sample {i}: shape={wave.shape}, lang={lang}, "
                  f"rms={wave.pow(2).mean().sqrt().item():.4f}")

        # Test DataLoader
        loader = DataLoader(ds, batch_size=4, num_workers=2, shuffle=True)
        batch = next(iter(loader))
        print(f"\nBatch: waveforms {batch[0].shape}, languages: {batch[1]}")
    else:
        print("Usage: python data_pipeline.py download [data_dir]")
        print("       python data_pipeline.py precompute [data_dir] [--force] [--max-items N]")
        print("       python data_pipeline.py test")
