"""Test inference on a saved checkpoint."""
import torch
import torchaudio
import os
import sys

from train_vocos_vq import VocosVQCodec, VocosVQConfig

try:
    from pesq import pesq as _pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
ckpt_path = "checkpoints_vocos_vq/codec_step10000.pt"
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Build model from config if stored, else rebuild
cfg = VocosVQConfig(n_codebooks=4, codebook_size=256, mel_fps=24, use_fsq=False)
model = VocosVQCodec(cfg)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()
print(f"Model loaded. Step: {ckpt.get('step', 'unknown')}")

# Load a few test files from dev set
manifest_path = "data/cv-corpus/master_manifest.jsonl"
import json
with open(manifest_path) as f:
    entries = [json.loads(line) for line in f]

# Pick 5 random files
import random
random.seed(42)
test_files = random.sample(entries, 5)

sr = 24000
seg_len = 24000  # 1 second

all_pesq = []
all_sdr = []

for i, entry in enumerate(test_files):
    path = entry['path']
    waveform, sample_rate = torchaudio.load(path, backend="soundfile")
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Take a segment
    total_len = waveform.shape[1]
    if total_len > seg_len:
        start = (total_len - seg_len) // 2
        waveform = waveform[:, start:start+seg_len]

    # Pad if too short
    if waveform.shape[1] < seg_len:
        pad = seg_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]

    # Inference
    with torch.no_grad():
        recon, mel_orig, mel_q, indices, commit_loss, util = model(waveform)

    # Metrics
    ref = waveform.squeeze().cpu().numpy()
    deg = recon.squeeze().detach().cpu().numpy()

    # SI-SDR
    def si_sdr(est, target):
        # Trim to same length
        ml = min(len(est), len(target))
        est, target = est[:ml], target[:ml]
        target = target / target.norm()
        est = est / est.norm()
        sdr = 10 * torch.log10((target ** 2).sum() / ((target - est) ** 2).sum() + 1e-10)
        return sdr.item()

    ref_t = torch.tensor(ref)
    deg_t = torch.tensor(deg)
    sdr = si_sdr(deg_t, ref_t)
    all_sdr.append(sdr)

    # PESQ at 16kHz
    if HAS_PESQ:
        ref_16 = torchaudio.functional.resample(torch.tensor(ref), orig_freq=sr, new_freq=16000)
        deg_16 = torchaudio.functional.resample(torch.tensor(deg), orig_freq=sr, new_freq=16000)
        ml = min(len(ref_16), len(deg_16))
        try:
            p = _pesq(16000, ref_16.numpy()[:ml], deg_16.numpy()[:ml], "wb")
            all_pesq.append(p)
        except Exception as e:
            print(f"  PESQ error: {e}")

    pesq_str = f"{all_pesq[-1]:.3f}" if all_pesq else "N/A"
    print(f"  [{i+1}] {os.path.basename(path)} | VQ util={util:.1%} | SDR={sdr:.2f}dB | PESQ={pesq_str}")

print(f"\n{'='*60}")
print(f"Mean SI-SDR: {sum(all_sdr)/len(all_sdr):.2f} dB")
if all_pesq:
    print(f"Mean PESQ:   {sum(all_pesq)/len(all_pesq):.3f}")
else:
    print(f"Mean PESQ:   N/A")
print(f"Config: 4×256 @ 24fps = 960 bps max")

# Save a sample audio
with torch.no_grad():
    test_audio = waveform
    recon_audio, _, _, _, _, _ = model(test_audio)

torchaudio.save("test_original.wav", test_audio.squeeze(0).cpu(), sr)
torchaudio.save("test_reconstructed.wav", recon_audio.squeeze(0).cpu(), sr)
print(f"\nSaved test_original.wav and test_reconstructed.wav")
