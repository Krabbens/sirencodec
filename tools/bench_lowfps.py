"""Sweep: lower fps + bigger codebooks at fixed bitrate."""
import torch, math, time, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sirencodec.core.train_vocos_vq import VocosVQCodec, VocosVQConfig, AudioDataset, collate_fn

DEVICE = torch.device("cuda")

# All configs at ~480bps
CONFIGS = [
    {"name": "30fps_4x16",  "fps": 30, "n_cb": 4, "cb_size": 16},    # 30*4*4=480
    {"name": "25fps_4x32",  "fps": 25, "n_cb": 4, "cb_size": 32},    # 25*4*5=500
    {"name": "24fps_4x32",  "fps": 24, "n_cb": 4, "cb_size": 32},    # 24*4*5=480
]

N_STEPS = 500
BATCH = 16

def run(cfg_dict):
    cfg = VocosVQConfig(
        mel_fps=cfg_dict["fps"], n_codebooks=cfg_dict["n_cb"],
        codebook_size=cfg_dict["cb_size"], total_steps=N_STEPS,
        batch_size=BATCH, use_fsq=False, warmup_steps=100,
    )
    model = VocosVQCodec(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5+0.5*(s+1)/100 if s<100 else 1.0)

    ds = AudioDataset("data/master_manifest.jsonl", 24000)
    n = len(ds)
    train_ds, dev_ds = torch.utils.data.random_split(ds, [int(n*0.9), n-int(n*0.9)])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                           num_workers=0, collate_fn=collate_fn)
    dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=1)

    bitrate = cfg.n_codebooks * math.log2(cfg.codebook_size) * cfg.mel_fps
    print(f"\n{'='*55}")
    print(f"  {cfg_dict['name']}: {cfg.n_codebooks}×{cfg.codebook_size} @ {cfg.mel_fps}fps = {bitrate:.0f}bps")
    print(f"{'='*55}")

    di = iter(train_dl)
    for step in range(N_STEPS):
        try: audio, _ = next(di)
        except StopIteration:
            di = iter(train_dl); audio, _ = next(di)
        audio = audio.to(DEVICE)

        audio_recon, mel_orig, mel_q, indices, commit_loss, util = model(audio)
        mel_recon = model.mel_extractor(audio_recon)
        mf = min(mel_recon.shape[2], mel_orig.shape[2])
        loss_mel = torch.nn.functional.l1_loss(mel_recon[:,:,:mf], mel_orig[:,:,:mf].detach())

        loss = 45.0 * loss_mel + commit_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            u = util.item() if isinstance(util, torch.Tensor) else float(util)
            print(f"  Step {step:4d}: mel={loss_mel.item():.3f}  commit={commit_loss.item():.3f}  vq={u:.0%}")

    # Eval
    model.eval()
    with torch.no_grad():
        for dev_audio, _ in dev_dl:
            dev_audio = dev_audio.to(DEVICE)
            recon, _, _, _, _, _ = model(dev_audio)
            p = recon.squeeze(); t = dev_audio.squeeze()
            ml = min(len(p), len(t)); p, t = p[:ml].cpu(), t[:ml].cpu()
            tn = t.dot(t)
            if tn < 1e-8: sdr = -100
            else:
                proj = t * (p.dot(t) / tn); noise = p - proj
                sdr = 10 * math.log10(max(proj.dot(proj) / (noise.dot(noise) + 1e-8), 1e-8))
            print(f"  [EVAL] SI-SDR={sdr:.2f}dB | mel={loss_mel.item():.3f}")
            return {"name": cfg_dict["name"], "bitrate": bitrate, "sdr": sdr, "mel": loss_mel.item()}

if __name__ == "__main__":
    print("LOWER FPS + BIGGER CODEBOOKS SWEEP @ ~480bps")
    results = []
    for c in CONFIGS:
        r = run(c)
        results.append(r)
        torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"{'Config':<15} {'bps':>6} {'mel':>7} {'SI-SDR':>10}")
    print(f"{'='*55}")
    for r in results:
        print(f"{r['name']:<15} {r['bitrate']:6.0f} {r['mel']:7.3f} {r['sdr']:10.2f}dB")
    print(f"{'='*55}")
