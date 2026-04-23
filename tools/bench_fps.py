"""Quick benchmark: compare FPS/codebook configs at fixed bitrate."""
import torch, math, time, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sirencodec.core.train_vocos_vq import VocosVQCodec, VocosVQConfig, AudioDataset, collate_fn

DEVICE = torch.device("cuda")

# Test configs at ~480bps
CONFIGS = [
    {"name": "30fps_4x16", "mel_fps": 30, "n_codebooks": 4, "codebook_size": 16},
    {"name": "60fps_8x4",  "mel_fps": 60, "n_codebooks": 8, "codebook_size": 4},
    {"name": "94fps_2x16", "mel_fps": 94, "n_codebooks": 2, "codebook_size": 16},
]

N_STEPS = 500
BATCH_SIZE = 16

def benchmark(cfg_dict):
    cfg = VocosVQConfig(
        mel_fps=cfg_dict["mel_fps"],
        n_codebooks=cfg_dict["n_codebooks"],
        codebook_size=cfg_dict["codebook_size"],
        total_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        use_fsq=False,
        warmup_steps=100,
    )
    model = VocosVQCodec(cfg).to(DEVICE)
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    def lr_fn(s):
        if s < 100: return 0.5 + 0.5*(s+1)/100
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    # Data
    manifest = "data/cv-corpus/master_manifest.jsonl"
    ds = AudioDataset(manifest, 24000)
    n = len(ds)
    train_ds, dev_ds = torch.utils.data.random_split(ds, [int(n*0.9), n-int(n*0.9)])
    # num_workers=0: synchronous loading so measured step time is model+opt only (not overlapped with workers).
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=0, collate_fn=collate_fn)
    dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=1, shuffle=False)
    
    bitrate = cfg.n_codebooks * math.log2(cfg.codebook_size) * cfg.mel_fps
    print(f"\n{'='*60}")
    print(f"  {cfg_dict['name']}: {cfg.n_codebooks}×{cfg.codebook_size} @ {cfg.mel_fps}fps = {bitrate:.0f} bps")
    print(f"{'='*60}")
    
    data_iter = iter(train_dl)
    results = []
    
    for step in range(N_STEPS):
        try:
            audio, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            audio, _ = next(data_iter)
        audio = audio.to(DEVICE)
        
        audio_recon, mel_orig, mel_q, indices, commit_loss, util = model(audio)
        
        mel_recon = model.mel_extractor(audio_recon)
        min_frames = min(mel_recon.shape[2], mel_orig.shape[2])
        loss_mel = torch.nn.functional.l1_loss(
            mel_recon[:, :, :min_frames], mel_orig[:, :, :min_frames].detach())
        
        loss = 45.0 * loss_mel + commit_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if step % 100 == 0:
            u = util.item() if isinstance(util, torch.Tensor) else float(util)
            c = commit_loss.item() if isinstance(commit_loss, torch.Tensor) else 0.0
            m = loss_mel.item()
            print(f"  Step {step:4d}: mel={m:.3f}  commit={c:.4f}  vq={u:.1%}")
    
    # Eval
    model.eval()
    with torch.no_grad():
        sdrs = []
        for i, (dev_audio, _) in enumerate(dev_dl):
            dev_audio = dev_audio.to(DEVICE)
            recon, _, _, _, _, _ = model(dev_audio)
            # SI-SDR
            p = recon.squeeze(); t = dev_audio.squeeze()
            ml = min(len(p), len(t))
            p, t = p[:ml], t[:ml]
            tn = t.dot(t)
            if tn < 1e-8:
                sdrs.append(-100)
                continue
            proj = t * (p.dot(t) / tn)
            noise = p - proj
            sdr = 10 * math.log10(max(proj.dot(proj) / (noise.dot(noise) + 1e-8), 1e-8))
            sdrs.append(sdr)
            if i >= 4: break
        avg_sdr = sum(sdrs) / len(sdrs) if sdrs else -100
    
    print(f"  [EVAL] SI-SDR={avg_sdr:.2f}dB | mel@{N_STEPS}={loss_mel.item():.3f}")
    results.append({"name": cfg_dict["name"], "bitrate": bitrate,
                    "sdr": avg_sdr, "mel": loss_mel.item()})
    return results[0]


if __name__ == "__main__":
    print("FPS/CODEBOOK SWEEP at ~480bps")
    all_results = []
    for c in CONFIGS:
        r = benchmark(c)
        all_results.append(r)
    
    print(f"\n{'='*60}")
    print(f"{'Config':<16} {'Bitrate':>8} {'mel':>8} {'SI-SDR':>10}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r['name']:<16} {r['bitrate']:8.0f}bps {r['mel']:8.3f} {r['sdr']:10.2f}dB")
    print(f"{'='*60}")
