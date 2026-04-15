"""Precompute mel spectrograms and audio segments for fast training."""
import torch, torchaudio, json, os, numpy as np
from pathlib import Path
from tqdm import tqdm

N_MELS = 16
N_FFT = 1024
HOP_LENGTH = 1000  # 24fps: 24000/1000=24
SAMPLE_RATE = 24000
SEGMENT_SAMPLES = 24000  # 1 second

def extract_mel(audio_24k):
    """Extract 16-dim mel spectrogram at 24fps."""
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, center=True, power=1.0,
    )
    mel = mel_fn(audio_24k.squeeze(0))  # [n_mels, T]
    mel = torch.log(mel.clamp(min=1e-5))
    return mel

def main():
    manifest_path = "data/master_manifest.jsonl"
    output_dir = Path("data/precomputed_24fps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    entries = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            wav = entry.get("path", "")
            if wav and os.path.exists(wav):
                entries.append(wav)
    
    print(f"Processing {len(entries)} audio files...")
    
    all_mels = []
    all_audio = []
    total_segments = 0
    
    for wav_path in tqdm(entries):
        try:
            audio, sr = torchaudio.load(wav_path)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            if audio.abs().max() > 0:
                audio = audio / audio.abs().max()
        except Exception:
            continue
        
        # Extract 1-second segments
        n_segs = audio.shape[1] // SEGMENT_SAMPLES
        for i in range(n_segs):
            start = i * SEGMENT_SAMPLES
            seg = audio[:, start:start + SEGMENT_SAMPLES]
            if seg.shape[1] < SEGMENT_SAMPLES:
                continue
            mel = extract_mel(seg)  # [16, T_mel]
            all_mels.append(mel.numpy())
            all_audio.append(seg.squeeze(0).numpy())
            total_segments += 1
    
    print(f"Total segments: {total_segments}")
    
    # Save as numpy arrays
    mel_arr = np.array(all_mels)  # [N, 16, T_mel]
    audio_arr = np.array(all_audio)  # [N, 24000]
    
    np.save(output_dir / "mels.npy", mel_arr)
    np.save(output_dir / "audio.npy", audio_arr)
    print(f"Saved: mels {mel_arr.shape}, audio {audio_arr.shape}")
    print(f"Total size: {mel_arr.nbytes / 1e6:.1f}MB + {audio_arr.nbytes / 1e6:.1f}MB")

if __name__ == "__main__":
    main()
