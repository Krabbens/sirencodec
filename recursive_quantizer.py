"""Recursive residual quantizer for speech mel spectrograms.

Instead of quantizing each frame independently, we quantize the RESIDUAL
from a first-order predictor. Speech changes slowly, so residuals are
concentrated near zero → much easier to quantize → better quality at same bitrate.

Architecture:
  mel[t] → predict from mel[t-1] → residual = mel[t] - pred → quantize residual
  At decoder: mel_q[t] = pred + residual_q[t]

Bitrate: 4 codebooks × 32 codes × 24fps = 480bps max
But since residuals are small, effective information density is higher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualRVQ(nn.Module):
    """Recursive residual RVQ: quantizes prediction residuals instead of raw frames.
    
    At each timestep:
    1. Predict mel[t] from mel_q[t-1] (linear predictor)
    2. Compute residual = mel[t] - prediction
    3. Quantize residual with RVQ
    4. Reconstruct: mel_q[t] = prediction + residual_q[t]
    """
    def __init__(self, dim, codebook_size, n_codebooks, ema_decay=0.99):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        
        # Linear predictor: mel_q[t-1] → mel_pred[t]
        self.predictor = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        
        # RVQ for residuals
        from train_vocos_vq import VectorQuantize, ResidualVQ
        self.rvq = ResidualVQ(dim, codebook_size, n_codebooks, ema_decay)
    
    def forward(self, mel, n_codebooks=None):
        """mel: [B, dim, T] → quantized mel, indices, commit_loss, util"""
        B, D, T = mel.shape
        device = mel.device
        
        # Initialize with first frame quantized
        first_frame = mel[:, :, 0:1]  # [B, dim, 1]
        first_q, first_idx, first_commit, first_util = self.rvq(first_frame, n_codebooks)
        
        mel_q_list = [first_q]
        all_indices = [first_idx]
        total_commit = first_commit
        total_util = first_util.item() if isinstance(first_util, torch.Tensor) else float(first_util)
        
        mel_prev_q = first_q
        
        for t in range(1, T):
            # Predict from previous quantized frame
            # Use linear predictor on running context
            pred = self.predictor(mel_prev_q)[:, :, -1:]  # [B, dim, 1]
            
            # Residual
            residual = mel[:, :, t:t+1] - pred  # [B, dim, 1]
            
            # Quantize residual
            res_q, res_idx, res_commit, res_util = self.rvq(residual, n_codebooks)
            
            # Reconstruct
            mel_t_q = pred + res_q  # [B, dim, 1]
            mel_q_list.append(mel_t_q)
            all_indices.append(res_idx)
            total_commit = total_commit + res_commit
            total_util = total_util + (res_util.item() if isinstance(res_util, torch.Tensor) else float(res_util))
            mel_prev_q = torch.cat([mel_prev_q, mel_t_q], dim=2)
        
        mel_q = torch.cat(mel_q_list, dim=2)  # [B, dim, T]
        avg_commit = total_commit / T
        avg_util = total_util / T
        
        return mel_q, all_indices, avg_commit, avg_util


class SlidingRecursiveQuantizer(nn.Module):
    """Sliding window recursive quantizer.
    
    Splits mel into low-frequency (slow-changing) and high-frequency (fast-changing)
    components using learned projection. Quantizes each at different rates.
    
    - Low: 10 mel bins @ 24fps → 4 cb × 32
    - High: 6 mel bins @ 12fps → 2 cb × 32
    - Total: 480 + 120 = 600bps → with codebook dropout ~360bps avg
    """
    def __init__(self, n_mels=16, codebook_size=32, n_cb_low=4, n_cb_high=2, fps_low=24, fps_high=12):
        super().__init__()
        self.n_mels = n_mels
        self.fps_low = fps_low
        self.fps_high = fps_high
        
        # Learned split: mel → low + high components
        self.split_proj = nn.Conv1d(n_mels, n_mels, 1)
        
        # Low-frequency: first 10 bins (energy, formants)
        self.n_low = 10
        # High-frequency: last 6 bins (fricatives, noise)
        self.n_high = n_mels - self.n_low
        
        # RVQ for low at full rate
        from train_vocos_vq import ResidualVQ
        self.rvq_low = ResidualVQ(self.n_low, codebook_size, n_cb_low)
        # RVQ for high at half rate (downsample → quantize → upsample)
        self.rvq_high = ResidualVQ(self.n_high, codebook_size, n_cb_high)
        
        # Merge back
        self.merge_proj = nn.Conv1d(n_mels, n_mels, 1)
    
    def forward(self, mel, n_codebooks=None):
        """mel: [B, n_mels, T]"""
        B, D, T = mel.shape
        
        # Split
        mel_split = self.split_proj(mel)  # [B, n_mels, T]
        mel_low = mel_split[:, :self.n_low, :]   # [B, 10, T]
        mel_high = mel_split[:, self.n_low:, :]  # [B, 6, T]
        
        # Quantize low at full rate
        low_q, low_idx, low_commit, low_util = self.rvq_low(mel_low, n_codebooks)
        
        # Quantize high at half rate: downsample → RVQ → upsample
        if T > 1:
            high_ds = F.interpolate(mel_high, size=(T + 1) // 2, mode='linear', align_corners=False)
            high_q_ds, high_idx, high_commit, high_util = self.rvq_high(high_ds, n_codebooks)
            high_q = F.interpolate(high_q_ds, size=T, mode='linear', align_corners=False)
        else:
            high_q, high_idx, high_commit, high_util = self.rvq_high(mel_high, n_codebooks)
        
        # Merge
        merged = torch.cat([low_q, high_q], dim=1)  # [B, n_mels, T]
        mel_q = self.merge_proj(merged)
        
        # Average metrics
        total_commit = (low_commit + high_commit) / 2
        avg_util = (low_util + high_util) / 2
        
        return mel_q, low_idx, total_commit, avg_util
