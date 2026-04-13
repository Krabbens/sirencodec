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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import time
import math
import csv
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
import argparse

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
    """Single codebook VQ with EMA updates."""
    def __init__(self, dim, codebook_size, ema_decay=0.99):
        super().__init__()
        self.dim = dim
        self.n_codes = codebook_size
        self.decay = ema_decay

        embed = torch.randn(codebook_size, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("ema_count", torch.ones(codebook_size))
        self.register_buffer("ema_weight", embed.clone())

    def forward(self, x):
        # x: [B, D, T] → [B*T, D]
        B, D, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, D)

        # Nearest neighbor
        dist = torch.cdist(x_flat.unsqueeze(0), self.embed.unsqueeze(0)).squeeze(0)
        indices = dist.argmin(dim=-1)
        quantized = F.embedding(indices, self.embed)

        # EMA codebook update
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.n_codes).float()
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_weight.mul_(self.decay).add_(onehot.T @ x_flat, alpha=1 - self.decay)
                n = self.ema_count.clamp(min=1e-5)
                self.embed.copy_(self.ema_weight / n.unsqueeze(1))
                # Dead code revival
                dead = self.ema_count < 1.0
                if dead.any():
                    n_dead = dead.sum().item()
                    rand_idx = torch.randint(0, x_flat.size(0), (n_dead,), device=x_flat.device)
                    self.embed[dead] = x_flat[rand_idx].detach()
                    self.ema_count[dead] = 1.0

        # Straight-through estimator
        commitment_loss = F.mse_loss(x_flat, quantized.detach())
        quantized_st = x_flat + (quantized - x_flat).detach()
        quantized_st = quantized_st.reshape(B, T, D).permute(0, 2, 1)

        utilization = len(indices.unique()) / self.n_codes
        return quantized_st, indices, commitment_loss, utilization


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


# ──────────────────────────────────────────────
# MRSTFT Discriminator (from train.py)
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


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class VocosVQConfig:
    # Audio
    sample_rate: int = 24000
    segment_length: int = 24000  # 1 sec at 24kHz

    # Mel features
    n_mels: int = 100
    n_fft: int = 1024
    mel_fps: int = 94  # Frame rate for mel extraction (94=hop256, 30=hop800, 20=hop1200, 10=hop2400)
    hop_length: int = 256  # Computed from mel_fps if not set

    # Vocos always expects 94fps (hop=256) features
    vocos_fps: int = 94

    # Quantization: FSQ (Finite Scalar Quantization)
    # FSQ quantizes each dimension independently to L levels
    # levels=[3,3,3,3,3,3,3,3] = 8 dims × 3 levels = 3^8 = 6561 combos ≈ 12.7 bits/frame
    # levels=[4,4,4,4,4,4,4,4] = 8 dims × 4 levels = 4^8 = 65536 combos ≈ 16 bits/frame
    # levels=[5,5,5,5,5,5,5,5] = 8 dims × 5 levels = 5^8 ≈ 232K combos ≈ 17.9 bits/frame
    fsq_levels: list = None  # e.g., [5]*8 for 8D×5levels
    use_fsq: bool = True

    # RVQ fallback (if use_fsq=False)
    codebook_size: int = 1024
    n_codebooks: int = 4
    ema_decay: float = 0.99

    # Training
    batch_size: int = 16
    total_steps: int = 100000
    lr_gen: float = 1e-4
    lr_disc: float = 2.5e-5
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    betas: tuple = (0.8, 0.99)

    # Loss weights
    lambda_mel: float = 45.0
    lambda_adv: float = 1.0
    lambda_feat: float = 2.0
    lambda_commit: float = 0.1  # Lower for FSQ (projection-based)

    # Paths
    data_dir: str = "data"
    log_tsv: str = "log_vocos_vq.tsv"
    results_tsv: str = "results_vocos_vq.tsv"
    vocos_pretrained: str = "charactr/vocos-mel-24khz"


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
# Model
# ──────────────────────────────────────────────
class VocosVQCodec(nn.Module):
    """VQ codec with pretrained Vocos vocoder.

    Audio → Mel → FSQ/VQ → Vocos → Audio
    """
    def __init__(self, cfg: VocosVQConfig):
        super().__init__()
        self.cfg = cfg

        # Compute hop_length from mel_fps
        self.low_hop = cfg.sample_rate // cfg.mel_fps
        self.vocos_hop = cfg.sample_rate // cfg.vocos_fps  # 256

        # Mel feature extraction at LOW frame rate
        self.mel_extractor = MelExtractor(
            n_mels=cfg.n_mels, n_fft=cfg.n_fft,
            hop_length=self.low_hop, sample_rate=cfg.sample_rate,
        )

        # Quantization: FSQ or RVQ
        if cfg.use_fsq and cfg.fsq_levels:
            # FSQ: project mel to fsq_dims, quantize, project back
            self.fsq_input_dim = len(cfg.fsq_levels)
            self.fsq = FSQ(cfg.fsq_levels)
            # Projection layers to/from FSQ space
            self.proj_in = nn.Conv1d(cfg.n_mels, self.fsq_input_dim, 1)
            self.proj_out = nn.Conv1d(self.fsq_input_dim, cfg.n_mels, 1)
            self.vq = None
            self.is_fsq = True
            # Calculate bitrate
            n_combos = 1
            for l in cfg.fsq_levels:
                n_combos *= l
            bits_per_frame = math.log2(n_combos)
            frame_rate = cfg.sample_rate / cfg.hop_length
            bitrate = bits_per_frame * frame_rate
            print(f"FSQ: {self.fsq_input_dim} dims × {cfg.fsq_levels[0]} levels = "
                  f"{bits_per_frame:.1f} bits/frame @ {frame_rate:.0f}fps = {bitrate:.0f} bps")
        else:
            self.vq = ResidualVQ(
                dim=cfg.n_mels, codebook_size=cfg.codebook_size,
                n_codebooks=cfg.n_codebooks, ema_decay=cfg.ema_decay,
            )
            self.fsq = None
            self.is_fsq = False
            bitrate = cfg.n_codebooks * math.log2(cfg.codebook_size) * cfg.mel_fps
            print(f"RVQ: {cfg.n_codebooks} codebooks × {cfg.codebook_size} @ {cfg.mel_fps}fps = "
                  f"{bitrate:.0f} bps (mel hop={self.low_hop})")

        # Pretrained Vocos vocoder
        print(f"Loading pretrained Vocos from {cfg.vocos_pretrained}...")
        self.vocos = Vocos.from_pretrained(cfg.vocos_pretrained)
        # Enable gradients for fine-tuning
        for p in self.vocos.parameters():
            p.requires_grad = True
        print(f"Vocos params: {sum(p.numel() for p in self.vocos.parameters()) / 1e6:.2f}M")

    def forward(self, audio, n_codebooks=None):
        """audio: [B, 1, T]"""
        # Extract mel features at LOW frame rate
        mel = self.mel_extractor(audio)  # [B, n_mels, T_low]

        # Quantization
        if self.is_fsq:
            mel_proj = self.proj_in(mel)
            mel_q_fsq, indices, commit_loss, util = self.fsq(mel_proj)
            mel_q = self.proj_out(mel_q_fsq)
        else:
            mel_q, indices, commit_loss, util = self.vq(mel, n_codebooks)

        # Upsample VQ'd mel from low fps to 94fps for Vocos
        if mel_q.shape[2] != mel.shape[2] * (self.low_hop // self.vocos_hop):
            target_frames = mel.shape[2] * (self.low_hop // self.vocos_hop)
            mel_q_up = F.interpolate(mel_q, size=target_frames, mode='linear', align_corners=False)
        else:
            mel_q_up = mel_q

        # Vocos vocoder (use backbone+head directly to preserve gradients)
        x = self.vocos.backbone(mel_q_up)  # [B, dim, T_vocos]
        audio_recon = self.vocos.head(x)  # [B, 1, T_audio]

        return audio_recon, mel, mel_q, indices, commit_loss, util

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


def run_training(cfg, device):
    """Main training loop."""
    # Create model
    model = VocosVQCodec(cfg).to(device)
    disc = MultiResolutionSTFTDiscriminator().to(device)

    print(f"Codec params: {model.count_params():.2f}M")
    print(f"Disc params: {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M")

    # Optimizers
    gen_params = list(model.parameters())
    disc_params = list(disc.parameters())

    opt_gen = optim.AdamW(gen_params, lr=cfg.lr_gen, betas=cfg.betas)
    opt_disc = optim.AdamW(disc_params, lr=cfg.lr_disc, betas=cfg.betas)

    # LR scheduler
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return 0.5 + 0.5 * ((step + 1) / cfg.warmup_steps)
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)))

    sched_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
    sched_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda)

    # Data
    master_manifest = os.path.join(cfg.data_dir, "master_manifest.jsonl")

    if os.path.exists(master_manifest):
        train_ds = AudioDataset(master_manifest, cfg.segment_length)
        # Split 90/10 for train/dev
        n = len(train_ds)
        n_train = int(n * 0.9)
        n_dev = n - n_train
        train_ds, dev_ds = torch.utils.data.random_split(train_ds, [n_train, n_dev])
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False)
        print(f"Train: {n_train} | Dev: {n_dev}")
    else:
        train_loader = None
        dev_loader = None
        print("WARNING: No master_manifest.jsonl found. Using synthetic data.")

    # Logging
    log_file = open(cfg.log_tsv, 'w')
    log_file.write("step\tmel_loss\tadv_loss\tcommit_loss\tvq_util\tgrad_norm\tlr\n")

    data_iter = iter(train_loader) if train_loader else None
    step_times = []

    for step in range(cfg.total_steps):
        t0 = time.time()

        # Get batch
        if train_loader:
            try:
                audio, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                audio, _ = next(data_iter)
        else:
            audio = torch.randn(cfg.batch_size, 1, cfg.segment_length) * 0.1

        audio = audio.to(device)

        # Forward
        audio_recon, mel_orig, mel_q, indices, commit_loss, util = model(audio)

        # Mel reconstruction loss (on audio level via mel of reconstructed)
        mel_recon = model.mel_extractor(audio_recon)
        # Handle frame count mismatches (boundary effects)
        min_frames = min(mel_recon.shape[2], mel_orig.shape[2])
        loss_mel = F.l1_loss(mel_recon[:, :, :min_frames], mel_orig[:, :, :min_frames].detach())

        # Adversarial
        use_adv = step > cfg.warmup_steps
        loss_adv = 0.0
        loss_feat = 0.0
        if use_adv:
            real_out, real_feats = disc(audio)
            fake_out, fake_feats = disc(audio_recon)
            loss_adv = adversarial_g_loss(fake_out)
            loss_feat = feature_matching_loss(real_feats, fake_feats)

        # Total loss
        loss = (cfg.lambda_mel * loss_mel +
                cfg.lambda_adv * loss_adv +
                cfg.lambda_feat * loss_feat +
                cfg.lambda_commit * commit_loss)

        # Generator backward
        if torch.isfinite(loss):
            opt_gen.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(gen_params, cfg.grad_clip)
            if torch.isfinite(grad_norm):
                opt_gen.step()
                sched_gen.step()

        # Discriminator backward
        if use_adv and step > cfg.warmup_steps:
            real_out, _ = disc(audio)
            fake_out, _ = disc(audio_recon.detach())
            disc_loss = adversarial_d_loss(real_out, fake_out)
            if disc_loss > 0 and torch.isfinite(disc_loss):
                opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5.0)
                opt_disc.step()
                sched_disc.step()

        # Logging
        if step % 200 == 0:
            lr = sched_gen.get_last_lr()[0]
            grad_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            util_val = util.item() if isinstance(util, torch.Tensor) else float(util)
            commit_val = commit_loss.item() if isinstance(commit_loss, torch.Tensor) else 0.0
            adv_val = loss_adv.item() if isinstance(loss_adv, torch.Tensor) else 0.0
            mel_val = loss_mel.item() if isinstance(loss_mel, torch.Tensor) else 0.0

            elapsed = time.time() - t0
            step_times.append(elapsed)
            if len(step_times) > 100:
                step_times.pop(0)
            avg_time = sum(step_times) / len(step_times)
            eta = avg_time * (cfg.total_steps - step) / 3600

            msg = (f"[{step/cfg.total_steps*100:5.1f}%] Step {step:6d}/{cfg.total_steps}  "
                   f"mel={mel_val:.3f}  adv={adv_val:.3f}  commit={commit_val:.4f}  "
                   f"vq={util_val:.1%}  grad={grad_val:.1f}  lr={lr:.6f}  "
                   f"{avg_time*1000:.0f}ms/step  ETA:{eta:.1f}h")
            print(msg)
            sys.stdout.flush()

            log_file.write(f"{step}\t{mel_val:.4f}\t{adv_val:.4f}\t{commit_val:.4f}\t"
                          f"{util_val:.4f}\t{grad_val:.4f}\t{lr:.6f}\n")
            log_file.flush()

        # Evaluation
        if step > 0 and step % 5000 == 0 and dev_loader:
            model.eval()
            with torch.no_grad():
                all_sdr = []
                all_pesq = []
                n_eval = 0
                for eval_audio, _ in dev_loader:
                    eval_audio = eval_audio.to(device)
                    eval_recon, _, _, _, _, _ = model(eval_audio)
                    sdr = si_sdr(eval_recon, eval_audio)
                    all_sdr.append(sdr)
                    n_eval += 1
                    if n_eval >= 5:
                        break

                avg_sdr = sum(all_sdr) / len(all_sdr) if all_sdr else 0.0

                # PESQ
                if HAS_PESQ:
                    for eval_audio, _ in dev_loader:
                        eval_audio = eval_audio.to(device)
                        eval_recon, _, _, _, _, _ = model(eval_audio)
                        ref = eval_audio.squeeze().cpu().numpy()
                        deg = eval_recon.squeeze().detach().cpu().numpy()
                        min_len = min(len(ref), len(deg))
                        try:
                            p = _pesq(16000, ref[:min_len], deg[:min_len], "wb")
                            all_pesq.append(p)
                        except:
                            pass
                        if len(all_pesq) >= 5:
                            break

                avg_pesq = sum(all_pesq) / len(all_pesq) if all_pesq else -1.0

            print(f"  [EVAL] SI-SDR={avg_sdr:.2f}dB | PESQ={avg_pesq:.3f}")
            sys.stdout.flush()

            # Save checkpoint
            ckpt_dir = Path("checkpoints_vocos_vq")
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"codec_step{step}.pt"
            torch.save({
                'model': model.state_dict(),
                'disc': disc.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
                'step': step,
            }, ckpt_path)
            print(f"  [CHECKPOINT] Saved {ckpt_path.name}")
            sys.stdout.flush()

            model.train()

    # Final
    log_file.close()
    print(f"\nTraining complete: {cfg.total_steps} steps")

    # Save final
    ckpt_dir = Path("checkpoints_vocos_vq")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'disc': disc.state_dict(),
        'step': cfg.total_steps,
    }, ckpt_dir / "codec_final.pt")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vocos VQ Codec Training with FSQ/RVQ")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--vocos-pretrained", type=str, default="charactr/vocos-mel-24khz")

    # RVQ options
    parser.add_argument("--rvq", action="store_true", help="Use RVQ instead of FSQ")
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--n-codebooks", type=int, default=4)

    # FSQ options
    parser.add_argument("--fsq-dims", type=int, default=32, help="FSQ dimensions")
    parser.add_argument("--fsq-levels", type=int, default=5, help="FSQ levels per dimension")
    parser.add_argument("--commit-weight", type=float, default=0.1, help="Commitment loss weight")

    # Frame rate
    parser.add_argument("--mel-fps", type=int, default=94, help="Mel frame rate (94=hop256, 30=hop800, 20=hop1200, 10=hop2400)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    fsq_levels = [args.fsq_levels] * args.fsq_dims if not args.rvq else None

    cfg = VocosVQConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        mel_fps=args.mel_fps,
        vocos_pretrained=args.vocos_pretrained,
        use_fsq=not args.rvq,
        fsq_levels=fsq_levels,
        codebook_size=args.codebook_size,
        n_codebooks=args.n_codebooks,
    )

    run_training(cfg, device)


if __name__ == "__main__":
    main()
