"""Generate spectrogram comparison: original vs reconstructed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sirencodec.core.train_vocos_vq import VocosVQCodec, VocosVQConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr = 24000

# Load checkpoint
ckpt_path = "checkpoints_vocos_vq/codec_step10000.pt"
print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = VocosVQConfig(n_codebooks=4, codebook_size=256, mel_fps=24, use_fsq=False)
model = VocosVQCodec(cfg)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()

# Load a test file
import json
with open("data/master_manifest.jsonl") as f:
    entries = [json.loads(line) for line in f]

import random
random.seed(42)
entry = random.choice(entries)
path = entry['path']
print(f"File: {path}")

waveform, sample_rate = torchaudio.load(path, backend="soundfile")
if sample_rate != sr:
    waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Take 3-second segment
seg_len = sr * 3
if waveform.shape[1] > seg_len:
    start = (waveform.shape[1] - seg_len) // 2
    waveform = waveform[:, start:start+seg_len]

# Run inference
audio_in = waveform.unsqueeze(0).to(device)
with torch.no_grad():
    recon, _, _, _, _, util = model(audio_in)

# Get audio tensors
orig = audio_in.squeeze(0).cpu()
deg = recon.squeeze(0).cpu()

# Compute mel spectrograms (100-dim, 24fps)
n_mels = 100
n_fft = 1024
hop_low = sr // 24  # 1000

mel_orig = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_low, n_mels=n_mels)(orig)
mel_deg = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_low, n_mels=n_mels)(deg)

mel_orig_log = np.log(mel_orig.squeeze().numpy().clip(min=1e-5))
mel_deg_log = np.log(mel_deg.squeeze().numpy().clip(min=1e-5))

# Waveform
orig_w = orig.squeeze().numpy()
deg_w = deg.squeeze().numpy()
# Trim to same length
ml = min(len(orig_w), len(deg_w))
orig_w = orig_w[:ml]
deg_w = deg_w[:ml]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# Waveforms
axes[0, 0].plot(orig_w, color='steelblue', linewidth=0.5)
axes[0, 0].set_title(f"Original Audio ({os.path.basename(path)})")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlabel("Samples")

axes[0, 1].plot(deg_w, color='coral', linewidth=0.5)
axes[0, 1].set_title(f"Reconstructed (step 10000)")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_xlabel("Samples")

# Spectrograms
im0 = axes[1, 0].imshow(mel_orig_log, aspect='auto', origin='lower', cmap='magma')
axes[1, 0].set_title("Original Mel Spectrogram")
axes[1, 0].set_ylabel("Mel bins")
axes[1, 0].set_xlabel("Time frames")
plt.colorbar(im0, ax=axes[1, 0], label="log energy")

im1 = axes[1, 1].imshow(mel_deg_log, aspect='auto', origin='lower', cmap='magma')
axes[1, 1].set_title("Reconstructed Mel Spectrogram")
axes[1, 1].set_ylabel("Mel bins")
axes[1, 1].set_xlabel("Time frames")
plt.colorbar(im1, ax=axes[1, 1], label="log energy")

plt.suptitle(f"SirenCodec — Step 10000 | 4×256 @ 24fps = 768bps | VQ util={util:.1%}", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("spectrogram_step10000.png", dpi=150, bbox_inches='tight')
print("Saved: spectrogram_step10000.png")
