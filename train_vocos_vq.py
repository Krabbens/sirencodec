"""
LuxTTS-style VQ Codec with Pretrained Vocos Vocoder.

Architecture:
  Audio → MelSpectrogram (100-dim, 24kHz) → VQ → Vocos → 24kHz audio

Uses pretrained Vocos vocoder (charactr/vocos-mel-24khz) as decoder.
Trains: mel encoder (simple conv) + VQ codebooks + fine-tunes Vocos backbone.

Inspired by LuxTTS pipeline which uses the same Vocos vocoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import time
import math
import csv
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
import argparse
import json
import numpy as np

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
try:
    from vocos import Vocos
    HAS_VOCOS = True
except ImportError:
    HAS_VOCOS = False
    print("WARNING: vocos not installed. Run: pip install vocos")

try:
    from pesq import pesq as _pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    import pystoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class AudioDataset(torch.utils.data.Dataset):
    """Load audio files from master_manifest.jsonl (data_pipeline format)."""
    def __init__(self, manifest_path, segment_length=24000):
        self.segment_length = segment_length
        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                import json
                entry = json.loads(line)
                # data_pipeline stores: {"wav_path": "...", "text": "...", "lang": "..."}
                wav = entry.get("path") or entry.get("wav_path") or entry.get("audio_path", "")
                if wav and os.path.exists(wav):
                    self.entries.append(wav)
        print(f"[Dataset] Loaded {len(self.entries)} valid entries from {manifest_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wav_path = self.entries[idx]
        try:
            audio, sr = torchaudio.load(wav_path)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            # Resample to 24kHz FIRST
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            # Normalize
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max()
        except Exception:
            audio = torch.zeros(1, self.segment_length)

        # Extract segment
        if audio.shape[1] > self.segment_length:
            start = torch.randint(0, audio.shape[1] - self.segment_length, (1,)).item()
            audio = audio[:, start:start + self.segment_length]
        elif audio.shape[1] < self.segment_length:
            pad = torch.zeros(1, self.segment_length - audio.shape[1])
            audio = torch.cat([audio, pad], dim=1)

        return audio, wav_path


def collate_fn(batch):
    audios = torch.stack([b[0] for b in batch])
    uids = [b[1] for b in batch]
    return audios, uids


# ──────────────────────────────────────────────
# Mel Feature Extraction (LuxTTS style)
# ──────────────────────────────────────────────
class MelExtractor(nn.Module):
    """Extract 100-dim mel spectrogram at 24kHz, hop=256 (LuxTTS config)."""
    def __init__(self, n_mels=100, n_fft=1024, hop_length=256, sample_rate=24000):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1.0,
        )

    def forward(self, audio):
        """audio: [B, 1, T] → mel: [B, n_mels, T_frames]"""
        audio = audio.squeeze(1)  # [B, T]
        mel = self.mel(audio)  # [B, n_mels, T]
        # Log scale (LuxTTS uses log mel)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel


# ──────────────────────────────────────────────
# Vector Quantization
# ──────────────────────────────────────────────
class VectorQuantize(nn.Module):
    """Single codebook VQ with learned codebook (gradient-based, no EMA lag)."""
    def __init__(self, dim, codebook_size, ema_decay=0.99):
        super().__init__()
        self.dim = dim
        self.n_codes = codebook_size

        self.embed = nn.Parameter(torch.randn(codebook_size, dim) * 0.02)
        self.register_buffer("usage_count", torch.zeros(codebook_size))

    def forward(self, x):
        # x: [B, D, T] → [B*T, D]
        B, D, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, D)

        # L2 distance → nearest neighbor
        dist = torch.cdist(x_flat.unsqueeze(0), self.embed.unsqueeze(0)).squeeze(0)
        indices = dist.argmin(dim=-1)
        quantized = F.embedding(indices, self.embed)

        # Commitment: encoder→codebook + codebook→encoder (symmetric)
        commitment_loss = F.mse_loss(x_flat, quantized.detach())
        codebook_loss = F.mse_loss(quantized, x_flat.detach())

        # Straight-through estimator
        quantized_st = x_flat + (quantized - x_flat).detach()
        quantized_st = quantized_st.reshape(B, T, D).permute(0, 2, 1)

        # Dead code revival via periodic reseeding
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.n_codes).float()
                self.usage_count.mul_(0.99).add_(onehot.sum(0), alpha=0.01)
                dead = self.usage_count < 0.01
                if dead.any():
                    n_dead = dead.sum().item()
                    rand_idx = torch.randint(0, x_flat.size(0), (n_dead,), device=x_flat.device)
                    self.embed.data[dead] = x_flat[rand_idx].detach() + torch.randn(n_dead, self.dim, device=x_flat.device) * 0.01
                    self.usage_count[dead] = 0.1

        utilization = len(indices.unique()) / self.n_codes
        return quantized_st, indices, commitment_loss + codebook_loss, utilization


class ResidualVQ(nn.Module):
    """Multiple VQ codebooks in residual fashion."""
    def __init__(self, dim, codebook_size, n_codebooks, ema_decay=0.99):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.layers = nn.ModuleList([
            VectorQuantize(dim, codebook_size, ema_decay)
            for _ in range(n_codebooks)
        ])

    def forward(self, x, n_codebooks=None):
        n_q = n_codebooks or self.n_codebooks
        residual = x
        quantized_total = torch.zeros_like(x)
        all_indices = []
        total_commit = 0.0
        total_util = 0.0

        for i in range(n_q):
            quantized, indices, commit, util = self.layers[i](residual)
            residual = residual - quantized.detach()
            quantized_total = quantized_total + quantized
            all_indices.append(indices)
            total_commit += commit
            total_util += util

        return quantized_total, all_indices, total_commit / n_q, total_util / n_q


class DynamicRVQ(nn.Module):
    """Residual VQ with per-frame dynamic codebook allocation.

    Frames with high energy/complexity use more codebooks (more bits).
    Silent/simple frames use fewer codebooks (fewer bits).
    Achieves variable bitrate within a single utterance.
    """
    def __init__(self, dim, codebook_size, n_codebooks, ema_decay=0.99):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.layers = nn.ModuleList([
            VectorQuantize(dim, codebook_size, ema_decay)
            for _ in range(n_codebooks)
        ])
        # Learned thresholds for codebook allocation (initialized for uniform split)
        if n_codebooks > 1:
            self.thresholds = nn.Parameter(torch.linspace(0.2, 0.8, n_codebooks - 1))
        else:
            self.register_parameter('thresholds', None)

    def _compute_complexity(self, x):
        """Compute per-frame complexity score [B, T] from input features."""
        # Energy + variance across dimensions = complexity proxy
        energy = x.pow(2).mean(dim=1)  # [B, T]
        # Normalize to [0, 1] per batch
        energy_min = energy.min(dim=1, keepdim=True)[0]
        energy_max = energy.max(dim=1, keepdim=True)[0]
        energy_range = (energy_max - energy_min).clamp(min=1e-6)
        complexity = (energy - energy_min) / energy_range  # [B, T]
        return complexity

    def forward(self, x, n_codebooks=None):
        """x: [B, D, T]. Returns per-frame variable-bitrate quantization."""
        B, D, T = x.shape

        # Determine per-frame codebook counts
        if n_codebooks is not None and isinstance(n_codebooks, int):
            # Uniform mode: all frames use same count (backward compat)
            n_q = n_codebooks
            frame_counts = torch.full((B, T), n_q, device=x.device, dtype=torch.long)
        elif self.thresholds is not None:
            # Dynamic mode: per-frame allocation
            complexity = self._compute_complexity(x)  # [B, T]
            # Straight-through thresholds
            thresh_st = self.thresholds + (self.thresholds - self.thresholds.detach())
            # Count how many thresholds each frame exceeds
            frame_counts = (complexity.unsqueeze(-1) > thresh_st).sum(dim=-1) + 1  # [B, T], range [1, N]
            n_q = frame_counts.max().item()  # max codebooks used this batch
        else:
            n_q = self.n_codebooks
            frame_counts = torch.full((B, T), n_q, device=x.device, dtype=torch.long)

        # Per-frame quantization
        quantized_total = torch.zeros_like(x)
        residual = x.clone()
        all_indices = [[] for _ in range(self.n_codebooks)]  # per-codebook indices
        total_commit = 0.0
        total_util = 0.0
        total_bits = 0.0

        for cb_idx in range(self.n_codebooks):
            quantized, indices, commit, util = self.layers[cb_idx](residual)
            # Mask: which frames use this codebook?
            mask = (frame_counts > cb_idx).unsqueeze(1)  # [B, 1, T]
            quantized_total = quantized_total + quantized * mask
            residual = residual - quantized.detach() * mask
            total_commit += commit * mask.float().mean()
            total_util += util
            # Track bits used
            total_bits += mask.float().sum()
            # Store indices for logging (masked ones are garbage but won't be counted)

        # Effective utilization and bitrate
        active_mask = (frame_counts > 0).float()
        avg_util = total_util / self.n_codebooks
        avg_bits_per_frame = total_bits / (B * T)  # effective codebooks per frame

        return quantized_total, all_indices, total_commit, avg_util, frame_counts.float().mean(), avg_bits_per_frame


class LearnedUpsampler(nn.Module):
    """Learned upsampling via ConvTranspose1d instead of bilinear.

    Replaces F.interpolate with trainable transposed conv.
    Preserves sharper transitions between frames.
    """
    def __init__(self, n_mels, upsampling_factor):
        super().__init__()
        self.upsampling_factor = upsampling_factor
        # Groups=n_mels = depthwise conv, each mel bin upsampled independently
        self.conv = nn.ConvTranspose1d(
            n_mels, n_mels,
            kernel_size=upsampling_factor * 2,
            stride=upsampling_factor,
            padding=upsampling_factor // 2,
            groups=n_mels,
            bias=False,
        )
        # Initialize to bilinear interpolation weights
        with torch.no_grad():
            kernel = torch.zeros(n_mels, 1, upsampling_factor * 2)
            for k in range(upsampling_factor * 2):
                kernel[:, 0, k] = max(0, 1 - abs(k / upsampling_factor - 1))
            self.conv.weight.copy_(kernel)

    def forward(self, x):
        """x: [B, n_mels, T_low] → [B, n_mels, T_high]"""
        return self.conv(x)


class DeltaRVQ(nn.Module):
    """Delta-coded RVQ: quantizes frame differences instead of absolute values.
    
    Speech changes slowly → deltas are small → concentrated near zero →
    same codebook covers smaller range → effectively higher resolution.
    Parallelizable (no sequential dependency).
    """
    def __init__(self, dim, codebook_size, n_codebooks, ema_decay=0.99):
        super().__init__()
        self.rvq = ResidualVQ(dim, codebook_size, n_codebooks, ema_decay)
    
    def forward(self, mel, n_codebooks=None):
        """mel: [B, dim, T] → quantized mel"""
        B, D, T = mel.shape
        
        # Compute deltas (prepend first frame)
        first = mel[:, :, 0:1]
        delta = torch.cat([first, mel[:, :, 1:] - mel[:, :, :-1]], dim=2)  # [B, dim, T]
        
        # Quantize deltas (small values → better codebook utilization)
        delta_q, indices, commit, util = self.rvq(delta, n_codebooks)
        
        # Reconstruct via cumsum
        mel_q = delta_q.cumsum(dim=2)  # [B, dim, T]
        
        return mel_q, indices, commit, util


# ──────────────────────────────────────────────
# Multi-Resolution STFT Loss (reconstruction, not adversarial)
# ──────────────────────────────────────────────
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_configs = [
            (1024, 120),
            (2048, 240),
            (512, 50),
        ]

    def forward(self, x, y):
        """x: recon, y: target. Both [B, 1, T]"""
        total_loss = 0.0
        x, y = x.squeeze(1), y.squeeze(1)
        for n_fft, hop in self.stft_configs:
            w = torch.hann_window(n_fft, device=x.device)
            X = torch.stft(x, n_fft, hop, window=w, return_complex=True)
            Y = torch.stft(y, n_fft, hop, window=w, return_complex=True)
            # Trim to same length
            min_t = min(X.shape[2], Y.shape[2])
            X, Y = X[:,:,:min_t], Y[:,:,:min_t]
            mag_loss = F.l1_loss(X.abs(), Y.abs())
            phase_loss = F.l1_loss(X.angle(), Y.angle())
            total_loss += mag_loss + phase_loss
        return total_loss / len(self.stft_configs)
# ──────────────────────────────────────────────
class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_freqs = n_fft // 2 + 1
        self.convs = nn.ModuleList([
            nn.Sequential(nn.utils.weight_norm(nn.Conv2d(1, 32, (3, 9), (1, 1), (1, 4))), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.utils.weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.utils.weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.utils.weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))), nn.LeakyReLU(0.1)),
        ])
        self.out_conv = nn.Sequential(nn.utils.weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1))))

    def forward(self, x):
        x = x.squeeze(1)
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x, self.n_fft, self.hop_length, window=window, return_complex=True)
        mag = stft.abs().unsqueeze(1)
        feats = []
        h = mag
        for conv in self.convs:
            h = conv(h)
            feats.append(h)
        return self.out_conv(h), feats


class MultiResolutionSTFTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([
            STFTDiscriminator(n_fft=1024, hop_length=120),
            STFTDiscriminator(n_fft=2048, hop_length=240),
            STFTDiscriminator(n_fft=512, hop_length=50),
        ])

    def forward(self, x):
        outputs, all_feats = [], []
        for disc in self.discs:
            out, feats = disc(x)
            outputs.append(out)
            all_feats.extend(feats)
        return outputs, all_feats


# ──────────────────────────────────────────────
# SI-SDR
# ──────────────────────────────────────────────
def si_sdr(pred, target, eps=1e-8):
    pred = pred.squeeze()
    target = target.squeeze()
    min_len = min(pred.shape[0], target.shape[0])
    pred, target = pred[:min_len], target[:min_len]
    target_norm = target.dot(target)
    if target_norm < eps:
        return -100.0
    proj = target * (pred.dot(target) / target_norm)
    noise = pred - proj
    sdr = 10 * math.log10(max(proj.dot(proj) / (noise.dot(noise) + eps), eps))
    return sdr


def si_snri(pred, target, ref, eps=1e-8):
    """Scale-Invariant SNR Improvement: how much better is pred vs ref relative to target."""
    def si_snr_val(est, tgt):
        est, tgt = est[:min(len(est), len(tgt))], tgt[:min(len(est), len(tgt))]
        tgt_n = tgt.dot(tgt)
        if tgt_n < eps:
            return -100.0
        proj = tgt * (est.dot(tgt) / tgt_n)
        noise = est - proj
        return 10 * math.log10(max(proj.dot(proj) / (noise.dot(noise) + eps), eps))
    return si_snr_val(pred, target) - si_snr_val(ref, target)


def log_spectral_distance(pred, target, sr=24000, n_fft=1024, hop=256):
    """Log-Spectral Distance (LSD) between two signals."""
    min_len = min(len(pred), len(target))
    pred, target = pred[:min_len], target[:min_len]
    pred_t = pred.detach().clone() if isinstance(pred, torch.Tensor) else torch.tensor(pred)
    target_t = target.detach().clone() if isinstance(target, torch.Tensor) else torch.tensor(target)
    window = torch.hann_window(n_fft, device=pred_t.device)
    pred_spec = torch.stft(pred_t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    target_spec = torch.stft(target_t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    pred_mag = pred_spec.abs().clamp(min=1e-10)
    target_mag = target_spec.abs().clamp(min=1e-10)
    lsd = torch.sqrt(((pred_mag.log() - target_mag.log()) ** 2).mean(dim=0)).mean().item()
    return lsd


def compute_f0_rmse(pred, target, sr=24000, frame_len=1024, hop=256):
    """F0 RMSE using librosa pyin."""
    if not HAS_LIBROSA:
        return None
    min_len = min(len(pred), len(target))
    pred, target = pred[:min_len], target[:min_len]
    try:
        f0_pred, _, _ = librosa.pyin(pred.astype(float), sr=sr, frame_length=frame_len, hop_length=hop, fmin=50, fmax=500)
        f0_target, _, _ = librosa.pyin(target.astype(float), sr=sr, frame_length=frame_len, hop_length=hop, fmin=50, fmax=500)
        # Only compare voiced frames
        mask = ~np.isnan(f0_pred) & ~np.isnan(f0_target)
        if mask.sum() < 5:
            return None
        rmse = np.sqrt(((f0_pred[mask] - f0_target[mask]) ** 2).mean())
        return rmse
    except:
        return None


def compute_stoi(ref, deg, sr=24000):
    """STOI score (range 0-1, higher=better)."""
    if not HAS_STOI:
        return None
    try:
        min_len = min(len(ref), len(deg))
        return pystoi.stoi(ref[:min_len].astype(np.float64), deg[:min_len].astype(np.float64), fs_sig=sr)
    except:
        return None


def compute_snr(pred, target, eps=1e-8):
    """Simple SNR in dB."""
    min_len = min(len(pred), len(target))
    pred, target = pred[:min_len], target[:min_len]
    signal_power = (target ** 2).mean()
    noise_power = ((pred - target) ** 2).mean()
    if noise_power < eps:
        return 100.0
    return 10 * math.log10(signal_power / noise_power)


# ──────────────────────────────────────────────
# Zipformer Block (LuxTTS-style)
# ──────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class ZipformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=256, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
        )
        self.norm3 = RMSNorm(dim)
        self.ff1 = nn.Linear(dim, ff_dim * 2)
        self.ff2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        """x: [B, T, D]"""
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        conv_out = self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
        x = x + conv_out
        x_norm = self.norm3(x)
        ff_hidden = self.ff1(x_norm)
        x1, x2 = ff_hidden.chunk(2, dim=-1)
        x = x + self.ff2(x1 * F.silu(x2))
        return x


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class VocosVQConfig:
    # Audio
    sample_rate: int = 24000
    segment_length: int = 24000  # 1 sec at 24kHz

    # Mel features
    n_mels: int = 100  # Full mel resolution
    n_fft: int = 1024
    mel_fps: int = 30  # Frame rate for mel extraction
    hop_length: int = 800  # Computed from mel_fps

    # Zipformer bottleneck
    zipformer_dim: int = 64
    zipformer_layers: int = 4
    zipformer_heads: int = 4

    # Vocos always expects 94fps (hop=256) features
    vocos_fps: int = 94

    # Quantization: FSQ (Finite Scalar Quantization)
    # FSQ quantizes each dimension independently to L levels
    # levels=[3,3,3,3,3,3,3,3] = 8 dims × 3 levels = 3^8 = 6561 combos ≈ 12.7 bits/frame
    # levels=[4,4,4,4,4,4,4,4] = 8 dims × 4 levels = 4^8 = 65536 combos ≈ 16 bits/frame
    # levels=[5,5,5,5,5,5,5,5] = 8 dims × 5 levels = 5^8 ≈ 232K combos ≈ 17.9 bits/frame
    fsq_levels: list = None  # e.g., [5]*8 for 8D×5levels
    use_fsq: bool = True

    # Delta coding: quantize frame differences instead of absolute values
    use_delta: bool = False

    # Direct mel VQ: ResidualVQ (fixed bitrate) vs DynamicRVQ (variable). Native 94fps forces ResidualVQ.
    use_dynamic_rvq: bool = True

    # PCA+FSQ: fixed PCA dim-reduction before FSQ quantization (no learned encoder)
    pca_dim: int = 0  # 0=disabled, 4/8/16 = PCA project to k dims then FSQ
    pca_path: str = "pca_components.pt"
    pca_fsq_levels: int = 3  # FSQ levels per PCA dim

    # Nonlinear MLP bottleneck + FSQ (per-frame, tanh-bounded; replaces PCA when nl_dim>0)
    nl_dim: int = 0  # 0=disabled, k = latent dims before FSQ
    nl_fsq_levels: int = 3  # FSQ levels per nonlinear latent dim
    nl_hidden: int = 64  # MLP hidden width
    # 1 = original depth (100→h→k); 2+ adds extra h→h blocks before the latent projection
    nl_layers: int = 1

    # Bottleneck: if > 0, compress mel to bottleneck_dim before VQ
    # If 0 or None, VQ directly on mel bins (higher quality, higher bitrate)
    bottleneck_dim: int = 16

    # RVQ fallback (if use_fsq=False)
    codebook_size: int = 1024
    n_codebooks: int = 4
    ema_decay: float = 0.95

    # Training
    batch_size: int = 32  # Increased from 16 for better GPU utilization
    total_steps: int = 100000
    lr_gen: float = 1e-4
    lr_disc: float = 2.5e-5
    warmup_steps: int = 5000  # delay adversarial / disc training until after this step
    # LR: linear warmup (lr_start_factor → 1.0), then cosine decay to lr_min_ratio
    lr_warmup_steps: int = -1  # -1 = same as warmup_steps
    lr_start_factor: float = 0.0  # LR multiplier at step 0 (0 = start near zero)
    lr_min_ratio: float = 0.05  # cosine floor as fraction of base lr
    # Curriculum: ramp segment length from segment_length_min → segment_length over segment_ramp_steps
    segment_ramp_steps: int = 0  # 0 = fixed cfg.segment_length only
    segment_length_min: int = 6000  # e.g. 0.25s @ 24kHz when ramping up
    use_torch_compile: bool = False  # torch.compile (CUDA); dynamic=True auto when segment ramps
    dataloader_num_workers: int = 8  # per process; use 2 when running DDP to limit CPU oversubscription
    # Resume: new schedule / optimizer while keeping weights from checkpoint
    reset_lr_schedule: bool = False  # LR+adv warmup use (step-start) over (total_steps-start) budget
    reset_optimizer_on_resume: bool = False  # do not load Adam state from ckpt
    reset_segment_curriculum: bool = False  # segment ramp from 0 after resume (vs global step)
    grad_clip: float = 1.0
    betas: tuple = (0.8, 0.99)

    # Loss weights
    lambda_mel: float = 45.0
    # Multi-res STFT (mag+phase L1 on waveform); 0=off. Try ~2–15 vs lambda_mel scale.
    lambda_stft: float = 0.0
    # Direct L1 on quantized mel vs target mel (before vocoder); 0=off. Try ~5–25 for NL+FSQ.
    lambda_mel_q: float = 0.0
    lambda_adv: float = 1.0
    lambda_feat: float = 2.0
    lambda_commit: float = 1.0  # Bottleneck VQ needs strong commit penalty
    commit_warmup_steps: int = 0  # No warmup — commit must constrain from step 0

    # Paths
    data_dir: str = "data"
    log_tsv: str = "log_vocos_vq.tsv"
    results_tsv: str = "results_vocos_vq.tsv"
    vocos_pretrained: str = "charactr/vocos-mel-24khz"
    exp_dir: str = ""  # experiment output dir (auto-generated if empty)


# ──────────────────────────────────────────────
# FSQ (Finite Scalar Quantization) - simple correct implementation
# ──────────────────────────────────────────────
class FSQ(nn.Module):
    """Finite Scalar Quantization.

    Each dimension is independently quantized to L levels in [-1, 1].
    No codebook, no dead codes, 100% utilization.
    levels: list of per-dimension quantization levels
    """
    def __init__(self, levels):
        super().__init__()
        self.levels = levels
        self.n_dims = len(levels)

    def forward(self, x):
        """x: [B, n_dims, T] → quantized [B, n_dims, T]"""
        B, D, T = x.shape

        # Quantize each dimension
        quantized = torch.zeros_like(x)
        all_indices = []

        for d in range(self.n_dims):
            l = self.levels[d]
            # Levels uniformly spaced in [-1, 1]
            centers = torch.linspace(-1 + 1/l, 1 - 1/l, l, device=x.device)
            x_d = x[:, d, :]  # [B, T]
            # Find nearest center
            dist = (x_d.unsqueeze(-1) - centers).abs()  # [B, T, L]
            idx = dist.argmin(dim=-1)  # [B, T]
            quantized[:, d, :] = centers[idx]
            all_indices.append(idx)  # [B, T]

        # Commitment loss
        commitment_loss = F.mse_loss(x, quantized.detach())

        # Straight-through estimator
        quantized_st = x + (quantized - x).detach()

        # 100% utilization
        utilization = 1.0

        return quantized_st, all_indices, commitment_loss, utilization


# ──────────────────────────────────────────────
# Model — Zipformer Bottleneck
# ──────────────────────────────────────────────
class VocosVQCodec(nn.Module):
    """VQ codec with Zipformer bottleneck + pretrained Vocos vocoder.

    Architecture:
      Audio → Mel(16) @ 30fps
      → Zipformer enc (16→64, temporal context)
      → Linear(64→16) bottleneck
      → RVQ on 16-dim bottleneck
      → Linear(16→64) expand
      → Zipformer dec (64→16)
      → Linear(16→100) → Interpolate → Vocos → Audio
    """
    def __init__(self, cfg: VocosVQConfig):
        super().__init__()
        self.cfg = cfg
        self.vocos_hop = cfg.sample_rate // cfg.vocos_fps  # ~255 @ 24kHz / 94fps
        # Native Vocos rate: mel_fps matches vocos_fps → same hop, no temporal upsampling
        self.native_vocos_mel = cfg.mel_fps == cfg.vocos_fps
        self.low_hop = self.vocos_hop if self.native_vocos_mel else (cfg.sample_rate // cfg.mel_fps)

        # Mel extraction at low frame rate (or Vocos-native hop when native_vocos_mel)
        self.mel_extractor = MelExtractor(
            n_mels=cfg.n_mels, n_fft=cfg.n_fft,
            hop_length=self.low_hop, sample_rate=cfg.sample_rate,
        )

        self.use_nl_fsq = False
        self.use_pca_fsq = False

        # ── Nonlinear MLP + FSQ (per-frame): mel(100)→MLP→k→tanh→FSQ→MLP→mel(100) ──
        if cfg.nl_dim > 0:
            self.use_nl_fsq = True
            k = cfg.nl_dim
            h = cfg.nl_hidden
            n_layers = cfg.nl_layers
            lvl = cfg.nl_fsq_levels

            enc_layers = [nn.Linear(cfg.n_mels, h), nn.SiLU()]
            for _ in range(n_layers - 1):
                enc_layers += [nn.Linear(h, h), nn.SiLU()]
            enc_layers.append(nn.Linear(h, k))
            self.nl_enc = nn.Sequential(*enc_layers)

            self.nl_fsq = FSQ([lvl] * k)

            dec_layers = [nn.Linear(k, h), nn.SiLU()]
            for _ in range(n_layers - 1):
                dec_layers += [nn.Linear(h, h), nn.SiLU()]
            dec_layers.append(nn.Linear(h, cfg.n_mels))
            self.nl_dec = nn.Sequential(*dec_layers)

            bits = k * math.log2(lvl)
            br = bits * cfg.mel_fps
            n_params = sum(p.numel() for p in self.nl_enc.parameters()) + sum(
                p.numel() for p in self.nl_dec.parameters()
            )
            enc_desc = "→".join([str(cfg.n_mels)] + [str(h)] * n_layers + [str(k)])
            dec_desc = "→".join([str(k)] + [str(h)] * n_layers + [str(cfg.n_mels)])
            print(f"NL+FSQ: mel({cfg.n_mels}) → MLP({enc_desc}) → tanh → "
                  f"FSQ({k}×{lvl}, {bits:.1f} bit) → MLP({dec_desc}) "
                  f"@ {cfg.mel_fps}fps = {br:.0f} bps  (~{n_params/1e3:.1f}k bottleneck params)")
            self.zip_enc = None
            self.zip_dec = None
            self.fsq = None
            self.vq = None
            self.bottleneck_enc = None
            self.bottleneck_dec = None
            self.use_bottleneck = False
            self.use_dynamic = False

        # ── PCA+FSQ path: fixed dim-reduction, no learned encoder ──
        elif cfg.pca_dim > 0 and os.path.exists(cfg.pca_path):
            self.use_pca_fsq = True
            pca = torch.load(cfg.pca_path, map_location="cpu")
            k = cfg.pca_dim
            self.register_buffer("pca_mean", pca["mean"])                      # [100]
            self.register_buffer("pca_V", pca["components"][:k].T)             # [100, k]
            self.register_buffer("pca_Vinv", pca["components"][:k])            # [k, 100]
            cumvar = pca["cumvar"][k - 1].item() * 100
            lvl = cfg.pca_fsq_levels
            self.pca_fsq = FSQ([lvl] * k)
            # Normalize PCA outputs to [-1, 1] for FSQ
            self.register_buffer("pca_scale", pca["eigenvalues"][:k].sqrt() * 2.5)  # ~±2.5σ range
            bits = k * math.log2(lvl)
            br = bits * cfg.mel_fps
            print(f"PCA+FSQ: mel(100) → PCA({k}d, {cumvar:.1f}% var) → "
                  f"FSQ({k}×{lvl}, {bits:.1f}bit) @ {cfg.mel_fps}fps = {br:.0f} bps")
            # No encoder/decoder modules needed
            self.zip_enc = None
            self.zip_dec = None
            self.fsq = None
            self.vq = None
            self.bottleneck_enc = None
            self.bottleneck_dec = None
            self.use_bottleneck = False
            self.use_dynamic = False

        # ── Quantization: Zipformer+FSQ or plain RVQ ──
        elif cfg.use_fsq and cfg.fsq_levels:
            # Zipformer bottleneck with FSQ
            self.zip_enc = nn.Sequential(
                nn.Linear(cfg.n_mels, cfg.zipformer_dim),
                *[ZipformerBlock(cfg.zipformer_dim, num_heads=cfg.zipformer_heads,
                                ff_dim=cfg.zipformer_dim*4) for _ in range(cfg.zipformer_layers)],
                RMSNorm(cfg.zipformer_dim),
            )
            self.fsq_dim = 8
            self.fsq_levels = [4] * self.fsq_dim
            self.fsq = FSQ(self.fsq_levels)
            self.bottleneck_enc = nn.Linear(cfg.zipformer_dim, self.fsq_dim)
            self.bottleneck_dec = nn.Linear(self.fsq_dim, cfg.zipformer_dim)
            self.zip_dec = nn.Sequential(
                *[ZipformerBlock(cfg.zipformer_dim, num_heads=cfg.zipformer_heads,
                                ff_dim=cfg.zipformer_dim*4) for _ in range(cfg.zipformer_layers)],
                RMSNorm(cfg.zipformer_dim),
                nn.Linear(cfg.zipformer_dim, cfg.n_mels),
            )
            bits_per_frame = self.fsq_dim * math.log2(4)
            bitrate = bits_per_frame * cfg.mel_fps
            print(f"Zipformer+FSQ: {cfg.n_mels} mel → {cfg.zipformer_dim}d → "
                  f"FSQ({self.fsq_dim}×4, {bits_per_frame:.0f}bit) @ {cfg.mel_fps}fps = {bitrate:.0f} bps")
        else:
            if cfg.bottleneck_dim and cfg.bottleneck_dim > 0:
                # Bottleneck path: mel → bottleneck → RVQ → mel
                self.bottleneck_dim = cfg.bottleneck_dim
                self.zip_enc = nn.Sequential(
                    nn.Linear(cfg.n_mels, 128),
                    nn.SiLU(),
                    nn.Linear(128, self.bottleneck_dim),
                )
                QClass = DeltaRVQ if cfg.use_delta else ResidualVQ
                prefix = "Delta" if cfg.use_delta else ""
                self.vq = QClass(
                    dim=self.bottleneck_dim, codebook_size=cfg.codebook_size,
                    n_codebooks=cfg.n_codebooks, ema_decay=cfg.ema_decay,
                )
                self.zip_dec = nn.Sequential(
                    nn.Linear(self.bottleneck_dim, 128),
                    nn.SiLU(),
                    nn.Linear(128, cfg.n_mels),
                )
                self.fsq = None
                self.bottleneck_enc = None
                self.bottleneck_dec = None
                self.use_bottleneck = True
                bits_per_frame = cfg.n_codebooks * math.log2(cfg.codebook_size)
                bitrate = bits_per_frame * cfg.mel_fps
                print(f"Zipformer+{prefix}RVQ: {cfg.n_mels} mel → {self.bottleneck_dim}d bottleneck → "
                      f"{cfg.n_codebooks}×{cfg.codebook_size} @ {cfg.mel_fps}fps = {bitrate:.0f} bps")
            else:
                # Direct VQ on mel bins — ResidualVQ (fixed bitrate) or DynamicRVQ (variable)
                self.bottleneck_dim = cfg.n_mels
                if cfg.use_delta:
                    QClass = DeltaRVQ
                    prefix = "Delta"
                elif self.native_vocos_mel or (not cfg.use_dynamic_rvq):
                    QClass = ResidualVQ
                    prefix = "Residual"
                else:
                    QClass = DynamicRVQ
                    prefix = "Dynamic"
                self.vq = QClass(
                    dim=cfg.n_mels, codebook_size=cfg.codebook_size,
                    n_codebooks=cfg.n_codebooks, ema_decay=cfg.ema_decay,
                )
                self.zip_enc = None
                self.zip_dec = None
                self.fsq = None
                self.bottleneck_enc = None
                self.bottleneck_dec = None
                self.use_bottleneck = False
                self.use_dynamic = (QClass is DynamicRVQ) and not cfg.use_delta
                bpf = cfg.n_codebooks * math.log2(cfg.codebook_size)
                br = bpf * cfg.mel_fps
                if self.use_dynamic:
                    min_bpf = 1 * math.log2(cfg.codebook_size)
                    max_bpf = cfg.n_codebooks * math.log2(cfg.codebook_size)
                    print(f"Direct{prefix}RVQ: VQ on {cfg.n_mels} mel bins → "
                          f"{cfg.n_codebooks}×{cfg.codebook_size} @ {cfg.mel_fps}fps = "
                          f"{min_bpf:.0f}-{max_bpf:.0f} bps (dynamic per-frame)")
                else:
                    print(f"Direct{prefix}RVQ: VQ on {cfg.n_mels} mel bins → "
                          f"{cfg.n_codebooks}×{cfg.codebook_size} @ {cfg.mel_fps}fps = {br:.0f} bps "
                          f"(mel hop={self.low_hop})")

        # Temporal context conv: smooth VQ quantization artifacts
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.n_mels, kernel_size=3, padding=1, groups=cfg.n_mels),
            nn.SiLU(),
        )

        # Learned upsampler: low fps → Vocos rate (identity when already native)
        self.upsampling_factor = max(1, self.low_hop // self.vocos_hop)
        if self.native_vocos_mel or self.upsampling_factor <= 1:
            self.learned_upsampler = nn.Identity()
            self.upsampling_factor = 1
        else:
            self.learned_upsampler = LearnedUpsampler(cfg.n_mels, self.upsampling_factor)

        # 100 mel bins — Vocos expects 100 bins, no expansion needed
        if cfg.n_mels == 100:
            self.mel_expand = nn.Identity()
        else:
            self.mel_expand = nn.Linear(cfg.n_mels, 100)

        # Pretrained Vocos vocoder
        print(f"Loading pretrained Vocos from {cfg.vocos_pretrained}...")
        self.vocos = Vocos.from_pretrained(cfg.vocos_pretrained)
        for p in self.vocos.parameters():
            p.requires_grad = True
        print(f"Vocos params: {sum(p.numel() for p in self.vocos.parameters()) / 1e6:.2f}M")

    def forward(self, audio, n_codebooks=None):
        """audio: [B, 1, T]"""
        mel = self.mel_extractor(audio)  # [B, n_mels, T_low]

        if self.use_nl_fsq:
            # Nonlinear MLP + FSQ: per-frame encode → tanh → FSQ → decode
            mel_in = mel.transpose(1, 2)  # [B, T, n_mels]
            z = self.nl_enc(mel_in)  # [B, T, k]
            z_tanh = torch.tanh(z).transpose(1, 2)  # [B, k, T]
            z_q, indices, commit_loss, util = self.nl_fsq(z_tanh)
            mel_dec = self.nl_dec(z_q.transpose(1, 2))  # [B, T, n_mels]
            mel_q_t = mel_dec.transpose(1, 2)  # [B, n_mels, T]
            k = self.cfg.nl_dim
            avg_frame_counts = 1.0
            avg_bits_per_frame = k * math.log2(self.cfg.nl_fsq_levels) / math.log2(self.cfg.codebook_size)

        elif self.use_pca_fsq:
            # PCA+FSQ path: fixed projection, no learned encoder
            B, D, T = mel.shape
            mel_centered = mel - self.pca_mean.unsqueeze(0).unsqueeze(2)   # [B, 100, T]
            z = torch.einsum("bdt,dk->bkt", mel_centered, self.pca_V)     # [B, k, T]
            z_norm = (z / self.pca_scale.unsqueeze(0).unsqueeze(2)).clamp(-1, 1)
            z_q, indices, commit_loss, util = self.pca_fsq(z_norm)
            z_denorm = z_q * self.pca_scale.unsqueeze(0).unsqueeze(2)
            mel_q_t = torch.einsum("bkt,kd->bdt", z_denorm, self.pca_Vinv) + self.pca_mean.unsqueeze(0).unsqueeze(2)
            k = z.shape[1]
            avg_frame_counts = 1.0
            avg_bits_per_frame = k * math.log2(self.cfg.pca_fsq_levels) / math.log2(self.cfg.codebook_size)

        elif self.fsq is not None:
            # Zipformer+FSQ path
            mel_in = mel.transpose(1, 2)
            h_enc = self.zip_enc(mel_in)
            z = self.bottleneck_enc(h_enc)
            z_tanh = torch.tanh(z)
            z_q, indices, commit_loss, util = self.fsq(z_tanh)
            h_dec = self.bottleneck_dec(z_q)
            mel_q = self.zip_dec(h_dec)
            mel_q_t = mel_q.transpose(1, 2)
            avg_frame_counts = float(self.n_codebooks)
            avg_bits_per_frame = float(self.n_codebooks)
        elif self.use_bottleneck and self.zip_enc is not None and self.vq is not None:
            # Bottleneck path: mel → encoder → VQ → decoder → mel
            mel_in = mel.transpose(1, 2)  # [B, T, n_mels]
            h_enc = self.zip_enc(mel_in)  # [B, T, bottleneck_dim]
            h_enc_t = h_enc.transpose(1, 2)  # [B, bottleneck_dim, T]
            h_q, indices, commit_loss, util = self.vq(h_enc_t, n_codebooks)
            h_q_t = h_q.transpose(1, 2)  # [B, T, bottleneck_dim]
            mel_q = self.zip_dec(h_q_t)  # [B, T, n_mels]
            mel_q_t = mel_q.transpose(1, 2)  # [B, n_mels, T]
            avg_frame_counts = float(n_codebooks or self.vq.n_codebooks)
            avg_bits_per_frame = avg_frame_counts
        else:
            # Direct VQ on mel bins (DynamicRVQ or DeltaRVQ)
            vq_out = self.vq(mel, n_codebooks)
            if self.use_dynamic:
                # DynamicRVQ: returns 6 values
                mel_q, indices, commit_loss, util, avg_frame_counts, avg_bits_per_frame = vq_out
            else:
                # DeltaRVQ/ResidualVQ: returns 4 values
                mel_q, indices, commit_loss, util = vq_out
                avg_frame_counts = float(n_codebooks or self.vq.n_codebooks)
                avg_bits_per_frame = avg_frame_counts
            mel_q_t = mel_q if mel_q.dim() == 3 else mel_q.transpose(1, 2)

        # Temporal smoothing: reduce VQ stepping artifacts
        mel_q_smooth = self.temporal_conv(mel_q_t) + mel_q_t  # residual connection

        # Learned upsampling: low fps → Vocos rate (skip when native 94fps)
        if isinstance(self.learned_upsampler, nn.Identity):
            mel_94 = mel_q_smooth
        else:
            mel_94 = self.learned_upsampler(mel_q_smooth)
            target_frames = mel_q_smooth.shape[2] * self.upsampling_factor
            if mel_94.shape[2] > target_frames:
                mel_94 = mel_94[:, :, :target_frames]
        mel_94 = mel_94.transpose(1, 2)  # [B, T_94, n_mels]

        # Expand to 100 mel bins for Vocos
        mel_100 = self.mel_expand(mel_94)
        mel_100_t = mel_100.transpose(1, 2)  # [B, 100, T_94]

        # Vocos vocoder
        x = self.vocos.backbone(mel_100_t)
        audio_recon = self.vocos.head(x)

        return audio_recon, mel, mel_q_t, indices, commit_loss, util, avg_frame_counts, avg_bits_per_frame

    def encode_to_coarse_mel(self, audio, n_codebooks=None):
        """Mel + VQ + temporal smoothing, no vocoder. For mel refiner / distillation."""
        mel = self.mel_extractor(audio)
        if self.use_nl_fsq:
            mel_in = mel.transpose(1, 2)
            z = self.nl_enc(mel_in)
            z_tanh = torch.tanh(z).transpose(1, 2)
            z_q, indices, commit_loss, util = self.nl_fsq(z_tanh)
            mel_dec = self.nl_dec(z_q.transpose(1, 2))
            mel_q_t = mel_dec.transpose(1, 2)
        elif self.use_pca_fsq:
            B, D, T = mel.shape
            mel_centered = mel - self.pca_mean.unsqueeze(0).unsqueeze(2)
            z = torch.einsum("bdt,dk->bkt", mel_centered, self.pca_V)
            z_norm = (z / self.pca_scale.unsqueeze(0).unsqueeze(2)).clamp(-1, 1)
            z_q, indices, commit_loss, util = self.pca_fsq(z_norm)
            z_denorm = z_q * self.pca_scale.unsqueeze(0).unsqueeze(2)
            mel_q_t = torch.einsum("bkt,kd->bdt", z_denorm, self.pca_Vinv) + self.pca_mean.unsqueeze(0).unsqueeze(2)
        elif self.fsq is not None:
            mel_in = mel.transpose(1, 2)
            h_enc = self.zip_enc(mel_in)
            z = self.bottleneck_enc(h_enc)
            z_tanh = torch.tanh(z)
            z_q, indices, commit_loss, util = self.fsq(z_tanh)
            h_dec = self.bottleneck_dec(z_q)
            mel_q = self.zip_dec(h_dec)
            mel_q_t = mel_q.transpose(1, 2)
        elif self.use_bottleneck and self.zip_enc is not None and self.vq is not None:
            mel_in = mel.transpose(1, 2)
            h_enc = self.zip_enc(mel_in)
            h_enc_t = h_enc.transpose(1, 2)
            h_q, indices, commit_loss, util = self.vq(h_enc_t, n_codebooks)
            h_q_t = h_q.transpose(1, 2)
            mel_q = self.zip_dec(h_q_t)
            mel_q_t = mel_q.transpose(1, 2)
        else:
            vq_out = self.vq(mel, n_codebooks)
            if self.use_dynamic:
                mel_q, indices, commit_loss, util, _, _ = vq_out
            else:
                mel_q, indices, commit_loss, util = vq_out
            mel_q_t = mel_q if mel_q.dim() == 3 else mel_q.transpose(1, 2)
        mel_q_smooth = self.temporal_conv(mel_q_t) + mel_q_t
        return mel, mel_q_smooth, indices, commit_loss, util

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def adversarial_g_loss(scores):
    return sum(F.mse_loss(s, torch.ones_like(s)) for s in scores) / max(len(scores), 1)


def adversarial_d_loss(scores_real, scores_fake):
    return (sum(F.mse_loss(s, torch.ones_like(s)) for s in scores_real) +
            sum(F.mse_loss(s, torch.zeros_like(s)) for s in scores_fake)) / max(len(scores_real) + len(scores_fake), 1)


def feature_matching_loss(feats_real, feats_fake):
    loss = 0.0
    for fr, ff in zip(feats_real, feats_fake):
        # Handle shape mismatches via min dimensions
        min_shape = [min(a, b) for a, b in zip(fr.shape, ff.shape)]
        fr_sliced = fr[(slice(None),) * (fr.dim() - 2) + (slice(min_shape[-2]), slice(min_shape[-1]))]
        ff_sliced = ff[(slice(None),) * (ff.dim() - 2) + (slice(min_shape[-2]), slice(min_shape[-1]))]
        loss += F.l1_loss(ff_sliced, fr_sliced.detach()) / (fr_sliced.abs().mean() + 1e-5)
    return loss


def _auto_exp_name(cfg):
    """Generate experiment folder name from config."""
    if cfg.nl_dim > 0:
        depth = f"_L{cfg.nl_layers}" if cfg.nl_layers != 1 else ""
        tag = f"nl{cfg.nl_dim}_fsq{cfg.nl_fsq_levels}{depth}_{cfg.mel_fps}fps"
    elif cfg.pca_dim > 0:
        tag = f"pca{cfg.pca_dim}_fsq{cfg.pca_fsq_levels}_{cfg.mel_fps}fps"
    elif cfg.use_fsq and cfg.fsq_levels:
        tag = f"zipfsq_{cfg.mel_fps}fps"
    elif cfg.bottleneck_dim and cfg.bottleneck_dim > 0:
        tag = f"bn{cfg.bottleneck_dim}_{cfg.n_codebooks}x{cfg.codebook_size}_{cfg.mel_fps}fps"
    else:
        tag = f"direct_{cfg.n_codebooks}x{cfg.codebook_size}_{cfg.mel_fps}fps"
    return tag


def _save_spectrogram(mel_orig, mel_recon, audio_orig, audio_recon, step, out_dir, sr=24000):
    """Save spectrogram comparison PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    mo = mel_orig[0].detach().cpu().numpy()
    mr = mel_recon[0].detach().cpu().numpy()
    axes[0, 0].imshow(mo, aspect="auto", origin="lower")
    axes[0, 0].set_title("Original mel")
    axes[0, 1].imshow(mr, aspect="auto", origin="lower")
    axes[0, 1].set_title("Reconstructed mel")
    diff = mo - mr[:, :mo.shape[1]] if mr.shape[1] >= mo.shape[1] else mo[:, :mr.shape[1]] - mr
    axes[1, 0].imshow(diff, aspect="auto", origin="lower", cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 0].set_title("Difference (orig - recon)")
    ao = audio_orig[0].squeeze().detach().cpu().numpy()
    ar = audio_recon[0].squeeze().detach().cpu().numpy()
    t_len = min(len(ao), len(ar), sr)  # 1 sec max
    axes[1, 1].plot(ao[:t_len], alpha=0.6, label="orig", linewidth=0.5)
    axes[1, 1].plot(ar[:t_len], alpha=0.6, label="recon", linewidth=0.5)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Waveform (1s)")
    plt.suptitle(f"Step {step}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"step_{step}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _lr_multiplier(step: int, cfg: VocosVQConfig, *, total_steps: int | None = None) -> float:
    """Linear warmup then cosine decay to lr_min_ratio (multiplier on base lr)."""
    T = cfg.total_steps if total_steps is None else total_steps
    w = cfg.lr_warmup_steps if cfg.lr_warmup_steps >= 0 else cfg.warmup_steps
    w = max(1, min(w, max(1, T - 1)))
    if step < w:
        return ((step + 1) / w) * (1.0 - cfg.lr_start_factor) + cfg.lr_start_factor
    denom = max(1, T - w)
    t = step - w
    progress = min(1.0, t / float(denom))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min_ratio + (1.0 - cfg.lr_min_ratio) * cos


def _segment_len_for_step(step: int, cfg: VocosVQConfig) -> int:
    lo = min(cfg.segment_length_min, cfg.segment_length)
    hi = cfg.segment_length
    if cfg.segment_ramp_steps <= 0 or lo >= hi:
        return hi
    frac = min(1.0, (step + 1) / float(cfg.segment_ramp_steps))
    return int(lo + frac * (hi - lo))


def _write_resume_state(exp: Path, ckpt_path: Path, step: int, total_steps: int) -> None:
    """Persist last checkpoint path + global step for shell / human resume."""
    payload = {
        "step": step,
        "total_steps": total_steps,
        "checkpoint": str(ckpt_path.resolve()),
        "exp_dir": str(exp.resolve()),
    }
    out = exp / "resume_state.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


def _unwrap_compiled(module: nn.Module) -> nn.Module:
    """torch.compile wraps the module; state_dict keys get _orig_mod. prefix."""
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    if isinstance(module, DDP):
        return module.module
    return module


def _raw_module_state_dict(module: nn.Module) -> dict:
    """Checkpoint-friendly state_dict (no DDP / compile prefixes)."""
    return _unwrap_compiled(_unwrap_ddp(module)).state_dict()


def _normalize_ckpt_state_dict(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from checkpoints saved while model was compiled."""
    if not state_dict:
        return state_dict
    sample = next(iter(state_dict.keys()))
    if not sample.startswith("_orig_mod."):
        return state_dict
    prefix = "_orig_mod."
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _audio_b1t(wav: torch.Tensor) -> torch.Tensor:
    """Vocos head often returns [B, T]; training batch audio is [B, 1, T]."""
    if wav.dim() == 2:
        return wav.unsqueeze(1)
    return wav


def _crop_audio_time(x: torch.Tensor, cur_len: int, training: bool) -> torch.Tensor:
    """x: [B, 1, T] → [B, 1, cur_len]"""
    if x.shape[2] < cur_len:
        return F.pad(x, (0, cur_len - x.shape[2]))
    extra = x.shape[2] - cur_len
    if extra == 0:
        return x
    if training:
        off = torch.randint(0, extra + 1, (1,)).item()
    else:
        off = extra // 2
    return x[:, :, off : off + cur_len]


def run_training(
    cfg,
    device,
    resume_ckpt="",
    *,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    distributed: bool = False,
):
    """Main training loop. DistributedDataParallel when `distributed` (torchrun)."""
    rank0 = rank == 0
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Experiment directory
    if not cfg.exp_dir:
        cfg.exp_dir = os.path.join("experiments", _auto_exp_name(cfg))
    exp = Path(cfg.exp_dir)
    (exp / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp / "spectrograms").mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    cfg.log_tsv = str(exp / "log.tsv")
    if rank0:
        print(f"Experiment dir: {exp}")
        if distributed:
            gb = cfg.batch_size * world_size
            print(f"DDP: world_size={world_size} rank={rank} local_batch={cfg.batch_size} global_batch={gb}")
        _lw = cfg.lr_warmup_steps if cfg.lr_warmup_steps >= 0 else cfg.warmup_steps
        print(
            f"LR schedule: linear warmup {_lw} steps (start_factor={cfg.lr_start_factor}) "
            f"→ cosine decay to {cfg.lr_min_ratio:.2%} of base lr"
        )
        if cfg.segment_ramp_steps > 0:
            print(
                f"Segment curriculum: {cfg.segment_length_min} → {cfg.segment_length} samples "
                f"over {cfg.segment_ramp_steps} steps"
            )
        if cfg.lambda_stft > 0 or cfg.lambda_mel_q > 0:
            print(
                f"Extra losses: lambda_stft={cfg.lambda_stft}  lambda_mel_q={cfg.lambda_mel_q} "
                "(0=disabled for either)"
            )

    model = VocosVQCodec(cfg).to(device)
    disc = MultiResolutionSTFTDiscriminator().to(device)
    mrstft_loss_fn = MultiResolutionSTFTLoss().to(device)

    if rank0:
        print(f"Codec params: {model.count_params():.2f}M")
        print(f"Disc params: {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M")

    start_step = 0
    resume_data = None
    if resume_ckpt and os.path.exists(resume_ckpt):
        resume_data = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(_normalize_ckpt_state_dict(resume_data["model"]))
        disc.load_state_dict(_normalize_ckpt_state_dict(resume_data["disc"]))
        start_step = resume_data.get("step", 0)
        if rank0:
            print(f"Resumed weights from {resume_ckpt} at step {start_step}")

    # Inductor + mel/Vocos STFT (complex) → broken compiled backward. Use aot_eager (no Inductor).
    # Skip compile under DDP (compile+DDP ordering is stack-sensitive; enable single-GPU first).
    if cfg.use_torch_compile and device.type == "cuda" and not distributed:
        try:
            lo = min(cfg.segment_length_min, cfg.segment_length)
            hi = cfg.segment_length
            compile_dynamic = cfg.segment_ramp_steps > 0 and lo < hi
            model = torch.compile(model, backend="aot_eager", dynamic=compile_dynamic)
            if rank0:
                print(
                    "torch.compile: codec only, backend=aot_eager"
                    + (f", dynamic={compile_dynamic}" if compile_dynamic else ", dynamic=False")
                    + " (Inductor skipped — complex STFT bw)"
                )
        except Exception as e:
            if rank0:
                print(f"torch.compile skipped: {e}")
    elif cfg.use_torch_compile and distributed and rank0:
        print("torch.compile: disabled under DDP (use single-GPU --compile if needed)")

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        disc = DDP(
            disc,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    gen_params = list(model.parameters())
    disc_params = list(disc.parameters())
    opt_gen = optim.AdamW(gen_params, lr=cfg.lr_gen, betas=cfg.betas)
    opt_disc = optim.AdamW(disc_params, lr=cfg.lr_disc, betas=cfg.betas)

    if resume_data and "opt_gen" in resume_data and not cfg.reset_optimizer_on_resume:
        try:
            opt_gen.load_state_dict(resume_data["opt_gen"])
            opt_disc.load_state_dict(resume_data["opt_disc"])
        except Exception as e:
            if rank0:
                print(f"Optimizer state not restored (fresh AdamW): {e}")
    elif start_step > 0 and cfg.reset_optimizer_on_resume and rank0:
        print("Optimizer: fresh AdamW (--reset-optimizer-on-resume)")

    if rank0 and resume_ckpt and start_step > 0 and os.path.exists(resume_ckpt):
        _write_resume_state(exp, Path(resume_ckpt).resolve(), start_step, cfg.total_steps)
        print(f"resume_state.json → step {start_step}")

    if rank0 and start_step > 0 and (
        cfg.reset_lr_schedule or cfg.reset_optimizer_on_resume or cfg.reset_segment_curriculum
    ):
        print(
            f"Resume overrides: reset_lr_schedule={cfg.reset_lr_schedule} "
            f"reset_optimizer={cfg.reset_optimizer_on_resume} "
            f"reset_segment_curriculum={cfg.reset_segment_curriculum}"
        )

    nw = int(cfg.dataloader_num_workers)
    pin = device.type == "cuda"
    dl_kw: dict = {"num_workers": nw, "collate_fn": collate_fn, "pin_memory": pin}
    if nw > 0:
        dl_kw["prefetch_factor"] = 2

    # Data: precomputed or raw
    precomputed_mels = os.path.join(cfg.data_dir, "precomputed_24fps/mels.npy")
    precomputed_audio = os.path.join(cfg.data_dir, "precomputed_24fps/audio.npy")

    train_sampler = None
    idx_gen = torch.Generator(device=torch.device("cpu"))
    idx_gen.manual_seed(42 + rank * 1_000_003)

    if os.path.exists(precomputed_mels) and os.path.exists(precomputed_audio):
        # Load precomputed numpy arrays (CPU); transfer only minibatches — safe for multi-GPU RAM
        mels = torch.from_numpy(np.load(precomputed_mels)).float()  # [N, n_mels, T_mel]
        audios = torch.from_numpy(np.load(precomputed_audio)).float()  # [N, T_audio]
        n_total = len(mels)
        n_train = int(n_total * 0.9)
        n_dev = n_total - n_train

        train_mels = mels[:n_train]
        train_audios = audios[:n_train]
        dev_mels = mels[n_train : n_train + n_dev : 50]
        dev_audios = audios[n_train : n_train + n_dev : 50]
        if rank0:
            print(f"Loaded {n_train} train + {len(dev_mels)} dev precomputed segments (CPU tensors)")
        train_loader = None
        dev_loader = None
        use_precomputed = True
    else:
        use_precomputed = False
        train_mels = train_audios = dev_mels = dev_audios = None
        master_manifest = os.path.join(cfg.data_dir, "master_manifest.jsonl")
        if os.path.exists(master_manifest):
            full_ds = AudioDataset(master_manifest, cfg.segment_length)
            n = len(full_ds)
            n_train = int(n * 0.9)
            n_dev = n - n_train
            split_g = torch.Generator().manual_seed(42)
            train_ds, dev_ds = torch.utils.data.random_split(full_ds, [n_train, n_dev], generator=split_g)
            if distributed:
                train_sampler = DistributedSampler(
                    train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    sampler=train_sampler,
                    **dl_kw,
                )
            else:
                train_loader = DataLoader(
                    train_ds, batch_size=cfg.batch_size, shuffle=True, **dl_kw
                )
            dev_loader = DataLoader(
                dev_ds,
                batch_size=1,
                shuffle=False,
                num_workers=min(2, nw),
                collate_fn=collate_fn,
                pin_memory=pin,
            )
            if rank0:
                print(f"Train: {n_train} | Dev: {n_dev}")
        else:
            train_loader = None
            dev_loader = None
            if rank0:
                print("WARNING: No precomputed data or manifest found.")

    # Logging (extended TSV when fresh; resume keeps legacy row shape if old header)
    log_file = None
    log_extended = True
    if rank0:
        if start_step > 0 and os.path.exists(cfg.log_tsv):
            with open(cfg.log_tsv) as _lf:
                _hdr = _lf.readline()
            log_extended = "stft_loss" in _hdr
        if start_step > 0:
            log_file = open(cfg.log_tsv, "a")
        else:
            log_file = open(cfg.log_tsv, "w")
            log_file.write(
                "step\tmel_loss\tstft_loss\tmel_q_loss\tadv_loss\tcommit_loss\t"
                "vq_util\tn_active\tbitrate\tgrad_norm\tlr\n"
            )

    data_iter = iter(train_loader) if train_loader else None
    step_times = []

    grad_norm = torch.tensor(0.0)  # init for logging

    for step in range(start_step, cfg.total_steps):
        t0 = time.time()

        use_phase = cfg.reset_lr_schedule and start_step > 0
        phase_step = step - start_step if use_phase else step
        phase_total = cfg.total_steps - start_step if use_phase else cfg.total_steps
        lr_mult = _lr_multiplier(
            phase_step, cfg, total_steps=phase_total if use_phase else None
        )
        for pg in opt_gen.param_groups:
            pg["lr"] = cfg.lr_gen * lr_mult
        for pg in opt_disc.param_groups:
            pg["lr"] = cfg.lr_disc * lr_mult

        seg_step = (step - start_step) if (cfg.reset_segment_curriculum and start_step > 0) else step
        cur_seg = _segment_len_for_step(seg_step, cfg)

        # Get batch
        if use_precomputed:
            idx = torch.randint(0, len(train_mels), (cfg.batch_size,), generator=idx_gen)
            mel_target = train_mels[idx].to(device)  # [B, n_mels, T_mel]
            audio = train_audios[idx].unsqueeze(1).to(device)  # [B, 1, T_audio]
            audio = _crop_audio_time(audio, cur_seg, training=True)
        elif train_loader:
            try:
                audio, _ = next(data_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(step)
                data_iter = iter(train_loader)
                audio, _ = next(data_iter)
            audio = audio.to(device)
            audio = _crop_audio_time(audio, cur_seg, training=True)
            mel_target = None
        else:
            audio = torch.randn(cfg.batch_size, 1, cur_seg, device=device) * 0.1
            mel_target = None

        # Forward (DynamicRVQ handles per-frame bitrate internally)
        audio_recon, mel_orig, mel_q, indices, commit_loss, util, avg_frame_counts, avg_bits_per_frame = model(audio)

        # Compute effective bitrate
        if isinstance(avg_bits_per_frame, torch.Tensor):
            avg_bits_per_frame = avg_bits_per_frame.item()
        if isinstance(avg_frame_counts, torch.Tensor):
            avg_frame_counts = avg_frame_counts.item()
        bits_per_token = avg_bits_per_frame * math.log2(cfg.codebook_size)
        eff_bitrate = bits_per_token * cfg.mel_fps
        n_active = int(round(avg_frame_counts))

        # Mel loss: use mel from reconstructed audio (gradient flows through vocoder)
        audio_recon_b1t = _audio_b1t(audio_recon)
        mel_recon = model.mel_extractor(audio_recon_b1t)
        min_frames = min(mel_recon.shape[2], mel_orig.shape[2])
        loss_mel = F.l1_loss(mel_recon[:, :, :min_frames], mel_orig[:, :, :min_frames].detach())

        loss_stft = torch.zeros((), device=device, dtype=mel_recon.dtype)
        if cfg.lambda_stft > 0:
            min_t = min(audio_recon_b1t.shape[2], audio.shape[2])
            loss_stft = mrstft_loss_fn(
                audio_recon_b1t[:, :, :min_t], audio[:, :, :min_t].detach()
            )

        loss_mel_q = torch.zeros((), device=device, dtype=mel_recon.dtype)
        if cfg.lambda_mel_q > 0:
            min_mq = min(mel_q.shape[2], mel_orig.shape[2])
            loss_mel_q = F.l1_loss(
                mel_q[:, :, :min_mq], mel_orig[:, :, :min_mq].detach()
            )

        # Adversarial (same step basis as LR when reset_lr_schedule)
        adv_ref = phase_step if use_phase else step
        use_adv = adv_ref > cfg.warmup_steps
        loss_adv = 0.0
        loss_feat = 0.0
        if use_adv:
            real_out, real_feats = disc(audio)
            fake_out, fake_feats = disc(audio_recon_b1t)
            loss_adv = adversarial_g_loss(fake_out)
            loss_feat = feature_matching_loss(real_feats, fake_feats)

        # Commit loss warmup: ramp from 0 to full over first K steps
        commit_scale = (
            min(1.0, adv_ref / cfg.commit_warmup_steps) if cfg.commit_warmup_steps > 0 else 1.0
        )

        # Total loss
        loss = (cfg.lambda_mel * loss_mel +
                cfg.lambda_stft * loss_stft +
                cfg.lambda_mel_q * loss_mel_q +
                cfg.lambda_adv * loss_adv +
                cfg.lambda_feat * loss_feat +
                cfg.lambda_commit * commit_scale * commit_loss)

        # Generator backward
        opt_gen.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(gen_params, cfg.grad_clip)
        opt_gen.step()

        # Discriminator backward
        if use_adv:
            real_out, _ = disc(audio)
            fake_out, _ = disc(audio_recon_b1t.detach())
            disc_loss = adversarial_d_loss(real_out, fake_out)
            if disc_loss > 0 and torch.isfinite(disc_loss):
                opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5.0)
                opt_disc.step()

        # Clear cache periodically to prevent VRAM fragmentation slowdown
        if step % 1000 == 0 and step > 0:
            torch.cuda.empty_cache()

        # Logging
        if rank0 and step % 200 == 0:
            lr = opt_gen.param_groups[0]["lr"]
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            util_val = util.item() if isinstance(util, torch.Tensor) else float(util)
            commit_val = commit_loss.item() if isinstance(commit_loss, torch.Tensor) else 0.0
            adv_val = loss_adv.item() if isinstance(loss_adv, torch.Tensor) else 0.0
            mel_val = loss_mel.item() if isinstance(loss_mel, torch.Tensor) else 0.0
            stft_val = loss_stft.item() if isinstance(loss_stft, torch.Tensor) else 0.0
            melq_val = loss_mel_q.item() if isinstance(loss_mel_q, torch.Tensor) else 0.0

            elapsed = time.time() - t0
            step_times.append(elapsed)
            if len(step_times) > 100:
                step_times.pop(0)
            avg_time = sum(step_times) / len(step_times)
            eta = avg_time * (cfg.total_steps - step) / 3600

            if cfg.nl_dim > 0:
                bits_per_frame = cfg.nl_dim * math.log2(cfg.nl_fsq_levels)
            elif cfg.pca_dim > 0:
                bits_per_frame = cfg.pca_dim * math.log2(cfg.pca_fsq_levels)
            else:
                bits_per_frame = n_active * math.log2(cfg.codebook_size)
            eff_bitrate = bits_per_frame * cfg.mel_fps

            msg = (f"[{step/cfg.total_steps*100:5.1f}%] Step {step:6d}/{cfg.total_steps}  "
                   f"mel={mel_val:.3f}  stft={stft_val:.3f}  melq={melq_val:.3f}  "
                   f"adv={adv_val:.3f}  commit={commit_val:.4f}  "
                   f"vq={util_val:.1%}  cb={n_active}  {eff_bitrate:.0f}bps  "
                   f"grad={grad_val:.1f}  lr={lr:.6f}  "
                   f"{avg_time*1000:.0f}ms/step  ETA:{eta:.1f}h")
            print(msg)
            sys.stdout.flush()

            if log_file is not None:
                if log_extended:
                    log_file.write(
                        f"{step}\t{mel_val:.4f}\t{stft_val:.4f}\t{melq_val:.4f}\t{adv_val:.4f}\t"
                        f"{commit_val:.4f}\t{util_val:.4f}\t{n_active}\t{eff_bitrate:.1f}\t"
                        f"{grad_val:.4f}\t{lr:.6f}\n"
                    )
                else:
                    log_file.write(
                        f"{step}\t{mel_val:.4f}\t{adv_val:.4f}\t{commit_val:.4f}\t"
                        f"{util_val:.4f}\t{n_active}\t{eff_bitrate:.1f}\t{grad_val:.4f}\t{lr:.6f}\n"
                    )
                log_file.flush()

        # Evaluation + checkpoint (rank 0 only); all ranks stay in sync via barrier
        if step > 0 and step % 5000 == 0:
            model.eval()
            disc.eval()
            dev_audio_t = dev_recon = None
            if rank0:
                with torch.no_grad():
                    if use_precomputed:
                        n_dev_eval = min(5, len(dev_mels))
                        dev_idx = torch.arange(n_dev_eval)
                        dev_audio_t = dev_audios[dev_idx].unsqueeze(1).to(device)
                        dev_recon, _, _, _, _, _, _, _ = model(dev_audio_t)
                    elif dev_loader:
                        dev_audio_list = []
                        dev_recon_list = []
                        n_eval = 0
                        for eval_audio, _ in dev_loader:
                            eval_audio = eval_audio.to(device)
                            eval_recon, _, _, _, _, _, _, _ = model(eval_audio)
                            dev_audio_list.append(eval_audio)
                            dev_recon_list.append(eval_recon)
                            n_eval += 1
                            if n_eval >= 5:
                                break
                        dev_audio_t = torch.cat(dev_audio_list, dim=0)
                        dev_recon = torch.cat(dev_recon_list, dim=0)
                    else:
                        dev_audio_t = None
                        dev_recon = None

                    if dev_audio_t is not None and dev_recon is not None:
                        n_samples = min(5, dev_audio_t.shape[0])
                        all_sdr, all_pesq, all_stoi, all_snr, all_lsd, all_si_snri = [], [], [], [], [], []
                        all_f0_rmse = []

                        for i in range(n_samples):
                            ref = dev_audio_t[i].squeeze().cpu().numpy()
                            deg = dev_recon[i].squeeze().detach().cpu().numpy()

                            all_sdr.append(si_sdr(torch.tensor(deg), torch.tensor(ref)))

                            if HAS_PESQ:
                                try:
                                    ref_16 = torchaudio.functional.resample(
                                        torch.tensor(ref), orig_freq=cfg.sample_rate, new_freq=16000
                                    )
                                    deg_16 = torchaudio.functional.resample(
                                        torch.tensor(deg), orig_freq=cfg.sample_rate, new_freq=16000
                                    )
                                    ml = min(len(ref_16), len(deg_16))
                                    p = _pesq(16000, ref_16.numpy()[:ml], deg_16.numpy()[:ml], "wb")
                                    all_pesq.append(p)
                                except Exception:
                                    pass

                            stoi_val = compute_stoi(ref, deg, sr=cfg.sample_rate)
                            if stoi_val is not None:
                                all_stoi.append(stoi_val)

                            all_snr.append(compute_snr(deg, ref))

                            all_lsd.append(
                                log_spectral_distance(torch.tensor(deg), torch.tensor(ref), sr=cfg.sample_rate)
                            )

                            all_si_snri.append(
                                si_snri(torch.tensor(deg), torch.tensor(ref), torch.zeros_like(torch.tensor(ref)))
                            )

                            f0_rmse = compute_f0_rmse(deg, ref, sr=cfg.sample_rate)
                            if f0_rmse is not None:
                                all_f0_rmse.append(f0_rmse)

                        avg_sdr = sum(all_sdr) / len(all_sdr) if all_sdr else 0.0
                        avg_pesq = sum(all_pesq) / len(all_pesq) if all_pesq else -1.0
                        avg_stoi = sum(all_stoi) / len(all_stoi) if all_stoi else -1.0
                        avg_snr = sum(all_snr) / len(all_snr) if all_snr else 0.0
                        avg_lsd = sum(all_lsd) / len(all_lsd) if all_lsd else -1.0
                        avg_si_snri = sum(all_si_snri) / len(all_si_snri) if all_si_snri else 0.0
                        avg_f0_rmse = sum(all_f0_rmse) / len(all_f0_rmse) if all_f0_rmse else -1.0

                        metrics_str = f"SI-SDR={avg_sdr:.2f}dB | PESQ={avg_pesq:.3f}"
                        if avg_stoi >= 0:
                            metrics_str += f" | STOI={avg_stoi:.3f}"
                        metrics_str += f" | SNR={avg_snr:.2f}dB | LSD={avg_lsd:.3f}"
                        if avg_f0_rmse >= 0:
                            metrics_str += f" | F0_RMSE={avg_f0_rmse:.1f}Hz"
                        metrics_str += f" | SI-SNRi={avg_si_snri:.2f}dB"
                        print(f"  [EVAL] {metrics_str}")
                    else:
                        print("  [EVAL] No dev data available")

                if dev_audio_t is not None and dev_recon is not None:
                    with torch.no_grad():
                        mel_r = model.mel_extractor(dev_recon[:1])
                        mel_o = model.mel_extractor(dev_audio_t[:1])
                    _save_spectrogram(
                        mel_o,
                        mel_r,
                        dev_audio_t[:1],
                        dev_recon[:1],
                        step,
                        str(exp / "spectrograms"),
                        cfg.sample_rate,
                    )

                ckpt_dir = exp / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                ckpt_path = ckpt_dir / f"codec_step{step}.pt"
                torch.save(
                    {
                        "model": _raw_module_state_dict(model),
                        "disc": _raw_module_state_dict(disc),
                        "opt_gen": opt_gen.state_dict(),
                        "opt_disc": opt_disc.state_dict(),
                        "step": step,
                        "cfg": asdict(cfg),
                    },
                    ckpt_path,
                )
                _write_resume_state(exp, ckpt_path, step, cfg.total_steps)
                print(f"  [CHECKPOINT] Saved {ckpt_path.name}")
                sys.stdout.flush()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if distributed:
                dist.barrier()
            model.train()
            disc.train()

    # Final
    if log_file is not None:
        log_file.close()
    if rank0:
        print(f"\nTraining complete: {cfg.total_steps} steps")

        ckpt_dir = exp / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        final_pt = ckpt_dir / "codec_final.pt"
        torch.save(
            {
                "model": _raw_module_state_dict(model),
                "disc": _raw_module_state_dict(disc),
                "step": cfg.total_steps,
                "cfg": {k: v for k, v in asdict(cfg).items()},
            },
            final_pt,
        )
        _write_resume_state(exp, final_pt, cfg.total_steps, cfg.total_steps)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vocos VQ Codec Training with FSQ/RVQ")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--vocos-pretrained", type=str, default="charactr/vocos-mel-24khz")

    # RVQ options
    parser.add_argument("--rvq", action="store_true", help="Use RVQ instead of FSQ")
    parser.add_argument("--delta", action="store_true", help="Use delta-coded RVQ (quantize frame differences)")
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--n-codebooks", type=int, default=4)
    parser.add_argument("--bottleneck-dim", type=int, default=16, help="Bottleneck dim (0 = direct VQ on mel bins)")

    # FSQ options (bottleneck design)
    parser.add_argument("--fsq-dims", type=int, default=12, help="FSQ bottleneck dimensions")
    parser.add_argument("--fsq-levels", type=int, default=4, help="FSQ levels per dimension")
    parser.add_argument("--commit-weight", type=float, default=1.0, help="Commitment loss weight (lambda_commit)")
    parser.add_argument(
        "--lambda-stft",
        type=float,
        default=0.0,
        help="Multi-res STFT loss weight on waveform (0=off; try ~5–15 with --lambda-mel 45)",
    )
    parser.add_argument(
        "--lambda-mel-q",
        type=float,
        default=0.0,
        help="L1 weight: quantized mel vs target mel before vocoder (0=off; try ~10–25 for NL+FSQ)",
    )
    parser.add_argument("--ema-decay", type=float, default=0.95, help="EMA decay for VQ codebook")

    # Frame rate
    parser.add_argument("--mel-fps", type=int, default=30, help="Mel frame rate (94=native Vocos rate, 30=hop800, ...)")
    parser.add_argument(
        "--residual-rvq",
        action="store_true",
        help="Use ResidualVQ (fixed bitrate) instead of DynamicRVQ for direct mel VQ (auto on when --mel-fps matches vocos 94)",
    )
    parser.add_argument("--log-tsv", type=str, default=None, help="Step log path (auto: experiments/<name>/log.tsv)")
    parser.add_argument("--exp-dir", type=str, default="", help="Experiment output dir (auto-generated if empty)")

    # PCA+FSQ (no learned encoder)
    parser.add_argument("--pca-dim", type=int, default=0, help="PCA dims before FSQ (0=disabled, 4/8 recommended)")
    parser.add_argument("--pca-path", type=str, default="pca_components.pt")
    parser.add_argument("--pca-fsq-levels", type=int, default=3, help="FSQ levels per PCA dimension")

    # Nonlinear MLP + FSQ (per-frame bottleneck; takes precedence over PCA when --nl-dim > 0)
    parser.add_argument("--nl-dim", type=int, default=0, help="Nonlinear latent dims before FSQ (0=disabled, e.g. 4)")
    parser.add_argument("--nl-fsq-levels", type=int, default=3, help="FSQ levels per nonlinear latent dim")
    parser.add_argument("--nl-hidden", type=int, default=64, help="MLP hidden width for NL encoder/decoder")
    parser.add_argument(
        "--nl-layers",
        type=int,
        default=1,
        help="Hidden depth: 1 = original (100→h→k), 2+ adds extra h→h layers before latent",
    )

    parser.add_argument("--warmup-steps", type=int, default=5000, help="Adversarial warmup before enabling discriminator")
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=-1,
        help="LR linear warmup length (-1 = same as --warmup-steps); then cosine decay to --lr-min-ratio",
    )
    parser.add_argument("--lr-min-ratio", type=float, default=0.05, help="Cosine LR floor as fraction of base lr")
    parser.add_argument(
        "--lr-start-factor",
        type=float,
        default=0.0,
        help="LR multiplier at step 0 during LR warmup (0 = start near zero)",
    )
    parser.add_argument(
        "--segment-ramp-steps",
        type=int,
        default=0,
        help="Linearly ramp segment length from --segment-length-min to full over this many steps (0=off)",
    )
    parser.add_argument(
        "--segment-length-min",
        type=int,
        default=6000,
        help="Starting segment length (samples) when --segment-ramp-steps > 0",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile codec (aot_eager, no Inductor); disc eager",
    )
    parser.add_argument("--resume-ckpt", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--reset-lr-schedule",
        action="store_true",
        help="After resume: restart LR cosine + adv warmup from 0 for remaining steps (weights kept)",
    )
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help="After resume: do not load Adam state from checkpoint (fresh momentum)",
    )
    parser.add_argument(
        "--reset-segment-curriculum",
        action="store_true",
        help="After resume: restart segment-length ramp from 0 (if segment_ramp_steps > 0)",
    )
    parser.add_argument(
        "--lr-gen",
        type=float,
        default=None,
        help="Generator AdamW base LR (default: 1e-4; DDP 8×GPU + global batch 80 often uses 1e-3 vs batch-8 baseline)",
    )
    parser.add_argument(
        "--lr-disc",
        type=float,
        default=None,
        help="Discriminator AdamW base LR (default: 2.5e-5; scaled with --lr-gen for GAN balance)",
    )
    parser.add_argument(
        "--data-num-workers",
        type=int,
        default=None,
        help="DataLoader workers per process (default: 8 single-GPU, 2 under DDP to limit CPU contention)",
    )

    args = parser.parse_args()

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA (set CUDA_VISIBLE_DEVICES and use torchrun).")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0 and device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.1f}GB")

    dw_default = 2 if distributed else 8
    dataloader_num_workers = args.data_num_workers if args.data_num_workers is not None else dw_default

    fsq_levels = [args.fsq_levels] * args.fsq_dims if not args.rvq else None

    use_dynamic_rvq = not args.residual_rvq
    cfg_kw = dict(
        total_steps=args.steps,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        mel_fps=args.mel_fps,
        vocos_pretrained=args.vocos_pretrained,
        use_fsq=not args.rvq,
        use_delta=args.delta,
        use_dynamic_rvq=use_dynamic_rvq,
        bottleneck_dim=args.bottleneck_dim,
        fsq_levels=fsq_levels,
        codebook_size=args.codebook_size,
        n_codebooks=args.n_codebooks,
        lambda_commit=args.commit_weight,
        lambda_stft=args.lambda_stft,
        lambda_mel_q=args.lambda_mel_q,
        ema_decay=args.ema_decay,
        log_tsv=args.log_tsv or "",
        exp_dir=args.exp_dir,
        pca_dim=args.pca_dim,
        pca_path=args.pca_path,
        pca_fsq_levels=args.pca_fsq_levels,
        nl_dim=args.nl_dim,
        nl_fsq_levels=args.nl_fsq_levels,
        nl_hidden=args.nl_hidden,
        nl_layers=args.nl_layers,
        warmup_steps=args.warmup_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
        lr_start_factor=args.lr_start_factor,
        segment_ramp_steps=args.segment_ramp_steps,
        segment_length_min=args.segment_length_min,
        use_torch_compile=args.compile,
        reset_lr_schedule=args.reset_lr_schedule,
        reset_optimizer_on_resume=args.reset_optimizer_on_resume,
        reset_segment_curriculum=args.reset_segment_curriculum,
        dataloader_num_workers=dataloader_num_workers,
    )
    if args.lr_gen is not None:
        cfg_kw["lr_gen"] = args.lr_gen
    if args.lr_disc is not None:
        cfg_kw["lr_disc"] = args.lr_disc
    cfg = VocosVQConfig(**cfg_kw)

    try:
        run_training(
            cfg,
            device,
            resume_ckpt=args.resume_ckpt,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            distributed=distributed,
        )
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
