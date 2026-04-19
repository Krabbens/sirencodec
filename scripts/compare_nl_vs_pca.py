#!/usr/bin/env python3
"""Compare NL vs PCA experiments: load final checkpoints, run PESQ/STOI/SI-SDR on dev set."""
import sys, os, json, math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import torch
import torchaudio
from dataclasses import fields

from sirencodec.core.train_vocos_vq import (
    VocosVQCodec,
    VocosVQConfig,
    AudioDataset,
    collate_fn,
    _normalize_ckpt_state_dict,
)

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


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = (est * ref).sum()
    s_target = dot * ref / (ref.pow(2).sum() + 1e-8)
    noise = est - s_target
    return 10 * torch.log10(s_target.pow(2).sum() / (noise.pow(2).sum() + 1e-8) + 1e-8)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    valid_fields = {f.name for f in fields(VocosVQConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    cfg = VocosVQConfig(**filtered)
    model = VocosVQCodec(cfg).to(device)
    model.load_state_dict(_normalize_ckpt_state_dict(ckpt["model"]))
    model.eval()
    step = ckpt.get("step", -1)
    return model, cfg, step


def eval_model(model, cfg, device, data_dir="data", n_samples=50):
    manifest = os.path.join(data_dir, "master_manifest.jsonl")
    ds = AudioDataset(manifest, cfg.segment_length)
    n = len(ds)
    n_train = int(n * 0.9)
    _, dev_ds = torch.utils.data.random_split(
        ds, [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    loader = torch.utils.data.DataLoader(dev_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_pesq, all_stoi, all_sdr = [], [], []
    with torch.no_grad():
        for i, (audio, _) in enumerate(loader):
            if i >= n_samples:
                break
            audio = audio.to(device)
            recon, *_ = model(audio)
            ref = audio.squeeze().cpu()
            est = recon.squeeze().cpu()
            ml = min(len(ref), len(est))
            ref, est = ref[:ml], est[:ml]

            all_sdr.append(si_sdr(est, ref).item())

            if HAS_PESQ:
                try:
                    r16 = torchaudio.functional.resample(ref, cfg.sample_rate, 16000).numpy()
                    e16 = torchaudio.functional.resample(est, cfg.sample_rate, 16000).numpy()
                    ml16 = min(len(r16), len(e16))
                    all_pesq.append(_pesq(16000, r16[:ml16], e16[:ml16], "wb"))
                except Exception:
                    pass

            if HAS_STOI:
                try:
                    s = pystoi.stoi(ref.numpy(), est.numpy(), cfg.sample_rate, extended=False)
                    all_stoi.append(s)
                except Exception:
                    pass

    return {
        "pesq": sum(all_pesq) / len(all_pesq) if all_pesq else -1,
        "stoi": sum(all_stoi) / len(all_stoi) if all_stoi else -1,
        "si_sdr": sum(all_sdr) / len(all_sdr) if all_sdr else -999,
        "n": len(all_pesq),
    }


def find_best_ckpt(exp_dir):
    """Prefer codec_final.pt; else latest codec_step*.pt."""
    ckpt_dir = Path(exp_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, None
    final = ckpt_dir / "codec_final.pt"
    if final.exists():
        return str(final), final.name
    ckpts = sorted(ckpt_dir.glob("codec_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))
    if ckpts:
        p = ckpts[-1]
        return str(p), p.name
    return None, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = {
        "PCA(4)+FSQ3 @ 94fps (596 bps)": "experiments/pca4_fsq3_94fps",
        "NL(4)+FSQ3 @ 94fps (596 bps)": "experiments/nl4_fsq3_94fps_warm15k_50k",
        "NL(4)+FSQ4 @ 94fps (752 bps)": "experiments/nl4_fsq4_94fps_warm15k_50k",
    }

    print("=" * 90)
    print(f"{'Experiment':<40} {'Step':>6} {'PESQ':>7} {'STOI':>7} {'SI-SDR':>8} {'bps':>6}  ckpt")
    print("=" * 90)

    for name, exp_dir in experiments.items():
        ckpt, ckpt_label = find_best_ckpt(exp_dir)
        if ckpt is None:
            print(f"{name:<40} {'N/A':>6} {'--':>7} {'--':>7} {'--':>8}  (no checkpoints)")
            continue
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckpt_step = int(meta.get("step", -1))
        total_steps = int(meta.get("cfg", {}).get("total_steps", 0) or 0)
        partial = ""
        if ckpt_label != "codec_final.pt" and total_steps > 0 and ckpt_step < total_steps - 500:
            partial = f" [partial {ckpt_step}/{total_steps}]"
        model, cfg, step = load_model(ckpt, device)
        if cfg.nl_dim > 0:
            bps = cfg.nl_dim * math.log2(cfg.nl_fsq_levels) * cfg.mel_fps
        elif cfg.pca_dim > 0:
            bps = cfg.pca_dim * math.log2(cfg.pca_fsq_levels) * cfg.mel_fps
        else:
            bps = cfg.n_codebooks * math.log2(cfg.codebook_size) * cfg.mel_fps
        metrics = eval_model(model, cfg, device, n_samples=50)
        print(f"{name:<40} {step:>6} {metrics['pesq']:>7.3f} {metrics['stoi']:>7.3f} "
              f"{metrics['si_sdr']:>8.2f} {bps:>6.0f}  {ckpt_label}{partial}")
        del model
        torch.cuda.empty_cache()

    print("=" * 90)


if __name__ == "__main__":
    main()
