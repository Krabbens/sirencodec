#!/usr/bin/env python3
"""
Training pipeline for CODEC-RESEARCHER.

Usage:
    python train_pipeline.py                    # Default: ARCH-A-v2b, 100k steps
    python train_pipeline.py --arch arch-a-v2b  # Specific architecture
    python train_pipeline.py --steps 200000     # More steps
    python train_pipeline.py --resume           # Resume from latest checkpoint
    python train_pipeline.py --no-eval          # Skip PESQ eval (faster)
    python train_pipeline.py --eval-every 5000  # Eval frequency
"""
import os, sys, json, math, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

# Import from project modules
from .train import (
    CodecConfig, AudioCodec, AudioCodecASPK, AudioCodecB,
    AudioCodecC, AudioCodecD,
    MultiPeriodDiscriminator, MultiScaleDiscriminator,
    MultiResolutionSTFTDiscriminator,
    MultiResolutionSTFTLoss,
    PsychoacousticMaskedMelLoss,
    feature_matching_loss,
    adversarial_g_loss, adversarial_d_loss,
    get_curriculum_codebooks,
)

# Optional data pipeline
try:
    from ..data_pipeline import DataConfig, MultilingualSpeechDataset, create_dataloaders
    HAS_DATA_PIPELINE = True
except ImportError:
    HAS_DATA_PIPELINE = False

# Debug logging for runtime evidence (session cc2f00); repo-local, portable.
from .. import REPO_ROOT

DEBUG_LOG_PATH = REPO_ROOT / ".cursor" / "debug-cc2f00.log"
DEBUG_SESSION_ID = "cc2f00"

def _debug_log(location, message, data, hypothesis_id, run_id):
    # region agent log
    if not hasattr(_debug_log, "_call_count"):
        _debug_log._call_count = 0
    _debug_log._call_count += 1
    if _debug_log._call_count <= 25:
        print(
            f"[agent-debug] {run_id}#{_debug_log._call_count} "
            f"{hypothesis_id} {location}:{message}"
        )
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{int(time.time()*1000)}_{run_id}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")
    # endregion


# ──────────────────────────────────────────────
# Pipeline Config
# ──────────────────────────────────────────────
def make_config(arch, steps, batch_size, segment_length, eval_every,
                save_every, log_every, use_real_data, data_dir, num_workers,
                codebook_size, psych_masking, data_num_workers, decoder_type,
                use_mrstft=True,
                use_preprocessed_cache=False, preprocessed_dir="preprocessed",
                preprocessed_manifest_name="master_manifest_preprocessed.jsonl",
                enable_augmentation=True,
                stage1_steps=50000, skip_stage1=False, codebook_dim=128,
                warmup_steps=5000, grad_clip=1.0, lambda_mel=None,
                frame_skip_target=0.35):
    # HiFi-GAN needs lower LR and slower warmup (it's more sensitive than Vocos)
    lr_gen = 5e-5 if decoder_type == "hifigan" else 1e-4  # Further reduced for stability
    lr_disc = 2.5e-5 if decoder_type == "hifigan" else 1.5e-5  # Reduced for stability
    warmup = warmup_steps  # tuneable warmup from CLI
    grad_clip = grad_clip  # configurable gradient clipping
    if lambda_mel is None:
        lambda_mel = 45.0

    cfg = CodecConfig(
        architecture=arch,
        use_real_data=use_real_data,
        data_dir=data_dir,
        total_steps=steps,
        warmup_steps=warmup,
        batch_size=batch_size,
        segment_length=segment_length,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        curriculum_enabled=False,
        data_num_workers=data_num_workers,
        use_mrstft=use_mrstft,
        use_psych_masking=psych_masking,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        decoder_type=decoder_type,
        lr_gen=lr_gen,
        lr_disc=lr_disc,
        grad_clip=grad_clip,
        lambda_mel=lambda_mel,
        log_tsv='log.tsv',
        results_tsv='results.tsv',
        stage1_steps=stage1_steps,
        skip_stage1=skip_stage1,
        frame_skip_target=frame_skip_target,
    )
    cfg.use_preprocessed_cache = use_preprocessed_cache
    cfg.preprocessed_dir = preprocessed_dir
    cfg.preprocessed_manifest_name = preprocessed_manifest_name
    cfg.enable_augmentation = enable_augmentation
    return cfg


# ──────────────────────────────────────────────
# Checkpoint management
# ──────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
LATEST_LINK = CHECKPOINT_DIR / "latest.pt"

def find_latest_checkpoint():
    """Find the latest checkpoint."""
    if LATEST_LINK.exists():
        return LATEST_LINK
    checkpoints = sorted(CHECKPOINT_DIR.glob("codec_step*.pt"))
    return checkpoints[-1] if checkpoints else None

def load_checkpoint(path, codec, mrstft_disc, opt_gen, opt_disc, device):
    """Load checkpoint and return starting step and losses."""
    # region agent log
    _debug_log(
        "train_pipeline.py:110",
        "load_checkpoint_enter",
        {"checkpoint": str(path)},
        hypothesis_id="A",
        run_id="pre-fix",
    )
    # endregion
    print(f"  Loading checkpoint: {path}")
    ckpt = torch.load(path, weights_only=False, map_location=device)
    codec.load_state_dict(ckpt['codec'])
    start_step = ckpt.get('step', 0) + 1
    # region agent log
    _debug_log(
        "train_pipeline.py:115",
        "load_checkpoint_state",
        {
            "start_step": start_step,
            "has_mrstft_disc": "mrstft_disc" in ckpt,
            "has_opt_gen": "opt_gen" in ckpt,
            "has_opt_disc": "opt_disc" in ckpt,
        },
        hypothesis_id="B",
        run_id="pre-fix",
    )
    # endregion

    if mrstft_disc and 'mrstft_disc' in ckpt:
        try:
            mrstft_disc.load_state_dict(ckpt['mrstft_disc'])
        except:
            print("  WARNING: Could not load MRSTFT discriminator state, starting fresh")

    if opt_gen and 'opt_gen' in ckpt:
        try:
            before_keys = len(opt_gen.state_dict()['state']) if opt_gen.state_dict().get('state') else 0
            opt_gen.load_state_dict(ckpt['opt_gen'])
            after_keys = len(opt_gen.state_dict()['state']) if opt_gen.state_dict().get('state') else 0
            # region agent log
            _debug_log(
                "train_pipeline.py:124",
                "load_checkpoint_opt_gen",
                {"loaded_ok": True, "before_state_size": before_keys, "after_state_size": after_keys},
                hypothesis_id="C",
                run_id="pre-fix",
            )
            # endregion
        except:
            print("  WARNING: Could not load optimizer state")
            # region agent log
            _debug_log(
                "train_pipeline.py:127",
                "load_checkpoint_opt_gen_fail",
                {"loaded_ok": False},
                hypothesis_id="C",
                run_id="pre-fix",
            )
            # endregion

    loss_mel = ckpt.get('loss_mel', 5.0)
    loss_commit = ckpt.get('loss_commit', 1.0)
    print(f"  Resuming from step {start_step} (mel={loss_mel:.3f}, commit={loss_commit:.3f})")
    return start_step, loss_mel, loss_commit

def save_checkpoint(step, codec, mrstft_disc, opt_gen, opt_disc,
                    loss_mel, loss_commit, cfg, suffix=None):
    """Save checkpoint."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    ckpt = {
        'step': step,
        'codec': codec.state_dict(),
        'opt_gen': opt_gen.state_dict() if opt_gen else None,
        'config': cfg,
        'loss_mel': loss_mel,
        'loss_commit': loss_commit,
    }
    if mrstft_disc:
        ckpt['mrstft_disc'] = mrstft_disc.state_dict()
    if opt_disc:
        ckpt['opt_disc'] = opt_disc.state_dict()

    # Save numbered checkpoint
    if suffix:
        path = CHECKPOINT_DIR / f"codec_step{step}_{suffix}.pt"
    else:
        path = CHECKPOINT_DIR / f"codec_step{step}.pt"
    torch.save(ckpt, path)

    # Update latest link
    torch.save(ckpt, LATEST_LINK)

    # Clean old checkpoints (keep last 5)
    checkpoints = sorted(CHECKPOINT_DIR.glob("codec_step*.pt"))
    if len(checkpoints) > 6:  # 5 + latest.pt
        for old in checkpoints[:-5]:
            old.unlink()
            print(f"  Cleaned old checkpoint: {old.name}")

    return path


# ──────────────────────────────────────────────
# Status bar
# ──────────────────────────────────────────────
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ──────────────────────────────────────────────
# Stage runner (Stage 1: no VQ, Stage 2: with VQ)
# ──────────────────────────────────────────────
def run_stage(codec, cfg, device, start_step, end_step,
              mrstft_disc, opt_gen, opt_disc,
              sched_gen, sched_disc,
              train_loader, val_loader, mel_loss_fn, mrstft_loss_fn,
              use_vq=True, skip_eval=False, freeze_encoder=False):
    """Run one stage of training."""
    codec.train()
    if mrstft_disc: mrstft_disc.train()

    # Get discriminator params for gradient clipping
    disc_params = list(mrstft_disc.parameters()) if mrstft_disc else []
    def _grad_l2_norm(params):
        total = torch.tensor(0.0, device=device)
        for p in params:
            if p.grad is not None:
                g = p.grad.detach().float()
                total += (g * g).sum()
        return torch.sqrt(total)

    if not use_vq:
        print("  VQ bypassed: training encoder → decoder directly")
        codec.quantizer.eval()
        for p in codec.quantizer.parameters():
            p.requires_grad = False
    else:
        print("  VQ enabled: training encoder → VQ → decoder")
        if freeze_encoder:
            # Freeze encoder, only train VQ + decoder
            print("  Encoder frozen: training VQ + decoder only")
            for p in codec.encoder.parameters():
                p.requires_grad = False
        # Rebuild optimizer for trainable params only
        gen_params = [p for p in codec.parameters() if p.requires_grad]
        stage_lr = cfg.lr_gen
        if not math.isfinite(stage_lr) or stage_lr <= 0:
            stage_lr = abs(stage_lr) if stage_lr < 0 else 1e-4
            if stage_lr == 0.0:
                stage_lr = 1e-4
            stage_lr = float(stage_lr)
        opt_gen = optim.AdamW(gen_params, lr=stage_lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        for pg in opt_gen.param_groups:
            if pg["lr"] <= 0 or not math.isfinite(pg["lr"]):
                pg["lr"] = abs(stage_lr) if stage_lr > 0 else 1e-4
        # Fix: Use relative step for scheduler so LR doesn't drop to 0 at stage boundary
        stage_start_step = start_step
        def lr_lambda(step):
            relative_step = max(0, step - stage_start_step)
            stage_warmup = min(cfg.warmup_steps, cfg.total_steps // 10)  # 10% warmup for stage 2
            if relative_step < stage_warmup:
                return relative_step / max(1, stage_warmup)
            else:
                remaining = cfg.total_steps - stage_start_step
                return 0.5 * (1 + math.cos(math.pi * (relative_step - stage_warmup) / max(1, remaining - stage_warmup)))
        sched_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
        # region agent log
        _debug_log(
            "train_pipeline.py:223",
            "stage2_scheduler_rebuilt",
            {
                "stage_start_step": start_step,
                "cfg_lr_gen": cfg.lr_gen,
                "stage_lr": stage_lr,
                "cfg_warmup": cfg.warmup_steps,
                "relative_stage_warmup": min(cfg.warmup_steps, cfg.total_steps // 10),
                "sched_last_epoch": sched_gen.last_epoch,
                "sched_get_last_lr": float(sched_gen.get_last_lr()[0]),
                "gen_lr_base": opt_gen.param_groups[0]["lr"] if opt_gen.param_groups else 0.0,
                "freeze_encoder": freeze_encoder,
            },
            hypothesis_id="A",
            run_id="pre-fix",
        )
        # endregion

    data_iter = None
    step_times = []
    last_latency = {
        "encode_latency_ms": 0.0,
        "decode_latency_ms": 0.0,
        "total_latency_ms": 0.0,
    }
    grad_clip_value = float(getattr(cfg, "grad_clip", 1.0))
    grad_clip_auto = bool(getattr(cfg, "grad_clip_auto", False))
    grad_clip_auto_window = max(1, int(getattr(cfg, "grad_clip_auto_window", 30)))
    grad_clip_auto_threshold = max(0.0, min(1.0, float(getattr(cfg, "grad_clip_auto_threshold", 0.25))))
    grad_clip_auto_floor = max(1e-4, float(getattr(cfg, "grad_clip_auto_floor", 0.1)))
    grad_clip_auto_decay = max(0.1, min(0.99, float(getattr(cfg, "grad_clip_auto_decay", 0.7))))
    grad_clip_ratio_history = []

    # Hoist mel transform when psych loss off — old path built new MelSpectrogram every step.
    mel_fb = None
    if not mel_loss_fn:
        mel_fb = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=1024, hop_length=cfg.hop_length,
        ).to(device)

    for step in range(start_step, end_step):
        t0 = time.time()

        # Get batch
        if train_loader:
            if data_iter is None:
                data_iter = iter(train_loader)
            try:
                wave, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                wave, _ = next(data_iter)
        else:
            wave = torch.randn(cfg.batch_size, 1, cfg.segment_length, device=device)

        wave = wave.to(device)

        # Normalize input audio to prevent numerical issues
        wave = wave / (wave.abs().max(dim=-1, keepdim=True).values + 1e-5)

        # Discriminator warmup (Stage 2 only)
        if use_vq:
            disc_delay = cfg.warmup_steps
            disc_progress = min(1.0, max(0.0, (step - disc_delay) / (cfg.total_steps * 0.25)))
        else:
            disc_progress = 0

        # Forward pass (AMP disabled - causes NaN gradients with this architecture)
        kf_frac = 0.0  # default for non-ARCH-C architectures
        if use_vq:
            n_cb = get_curriculum_codebooks(step, cfg.curriculum_schedule) if cfg.curriculum_enabled else cfg.n_codebooks
            if cfg.architecture == "arch-c-v1":
                x_hat, indices, commit_loss, cb_loss, vq_util, entropy_loss, kf_frac = codec(wave, step=step)
            else:
                x_hat, indices, commit_loss, cb_loss, vq_util, entropy_loss = codec(wave, n_codebooks=n_cb)
        else:
            # Stage 1: bypass VQ
            z = codec.encoder(wave)
            x_hat = codec.decoder(z)
            x_hat = x_hat[..., :wave.size(-1)]
            commit_loss = torch.tensor(0.0, device=device)
            cb_loss = torch.tensor(0.0, device=device)
            vq_util = 0.0
            entropy_loss = torch.tensor(0.0, device=device)
            n_cb = 0

        # Mel loss
        if mel_loss_fn:
            loss_mel = mel_loss_fn(x_hat.squeeze(1), wave.squeeze(1))
        else:
            assert mel_fb is not None
            mel_pred = mel_fb(x_hat.squeeze(1))
            mel_tgt = mel_fb(wave.squeeze(1))
            loss_mel = F.l1_loss(mel_pred.log().clamp(min=-10), mel_tgt.log().clamp(min=-10))

        # MRSTFT recon loss — only Stage 2 uses it in loss_gen; Stage 1 skipped (was dead compute).
        mr_pred_mags = mr_tgt_mags = None
        if mrstft_loss_fn and use_vq:
            mr_pred_mags, mr_tgt_mags = mrstft_loss_fn.compute_mags_pair(x_hat, wave)
            loss_mrstft = mrstft_loss_fn.loss_from_mags(mr_pred_mags, mr_tgt_mags)
        else:
            loss_mrstft = 0.0

        # Adversarial + feature matching (Cycle 25: MRSTFT discriminator)
        # Cycle 32: Enable adversarial in Stage 1 for Mamba/Zipformer decoders
        # (they need adversarial signal to learn realistic audio generation)
        loss_adv = 0.0
        loss_feat = 0.0
        disc_loss = torch.tensor(0.0, device=device)
        disc_grad_before = torch.tensor(0.0, device=device)
        disc_grad_after = torch.tensor(0.0, device=device)
        disc_grad_ratio = torch.tensor(0.0, device=device)
        adv_start = int(getattr(cfg, "adv_start_step", cfg.warmup_steps))
        adv_every = max(1, int(getattr(cfg, "adv_every", 1)))
        adv_ok = step > adv_start and (step % adv_every == 0)
        # Stage 1 (no VQ): optional mel-only AE — skip MRSTFT disc (huge speedup on MPS/CUDA).
        stage_allows_adv = use_vq or not getattr(cfg, "no_adv_stage1", False)
        use_adv = cfg.use_mrstft and mrstft_disc and adv_ok and stage_allows_adv

        if use_adv:
            if mr_pred_mags is not None and mr_tgt_mags is not None:
                pred_mags_b1ft = [m.unsqueeze(1) for m in mr_pred_mags]
                tgt_mags_b1ft = [m.unsqueeze(1) for m in mr_tgt_mags]
                real_out, real_feats = mrstft_disc.forward_from_mags(tgt_mags_b1ft)
                fake_out, fake_feats = mrstft_disc.forward_from_mags(pred_mags_b1ft)
            else:
                real_out, real_feats = mrstft_disc(wave)
                fake_out, fake_feats = mrstft_disc(x_hat)
            loss_adv = adversarial_g_loss(fake_out)
            loss_feat = feature_matching_loss(real_feats, fake_feats)

        # Generator loss
        if use_vq:
            loss_gen = (cfg.lambda_mel * loss_mel
                        + cfg.lambda_adv * loss_adv
                        + cfg.lambda_feat * loss_feat
                        + cfg.lambda_commit * commit_loss
                        + cfg.lambda_codebook * cb_loss
                        + 0.1 * entropy_loss)
            if mrstft_loss_fn:
                loss_gen += 1.0 * loss_mrstft  # Cycle 26: was lambda_mel * 0.5 → dominated
        else:
            # Stage 1 (autoencoder): include adversarial loss for Mamba/Zipformer decoders
            loss_gen = cfg.lambda_mel * loss_mel + cfg.lambda_adv * loss_adv + cfg.lambda_feat * loss_feat

        # Backward pass (AMP disabled - causes NaN gradients)
        opt_gen.zero_grad()
        
        # NaN/inf check on loss
        loss_val = loss_gen.item() if isinstance(loss_gen, torch.Tensor) else loss_gen
        if not math.isfinite(loss_val):
            print(f"\n  WARNING: NaN/Inf loss at step {step}, skipping step")
            sched_gen.step()
            continue

        # Backward (no loss clamping - it kills gradients!)
        loss_gen.backward()
        trainable_params = [p for p in codec.parameters() if p.requires_grad]
        
        # Check for NaN/Inf in gradients
        grad_has_nan_inf = False
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_has_nan_inf = True
                break
        
        if grad_has_nan_inf:
            grad_norm = torch.tensor(0.0, device=device)
            grad_norm_before = torch.tensor(0.0, device=device)
            grad_norm_after = torch.tensor(0.0, device=device)
            grad_norm_ratio = torch.tensor(0.0, device=device)
            grad_clip_ratio = torch.tensor(0.0, device=device)
            grad_clip_hit = torch.tensor(0.0, device=device)
            if step % cfg.log_every == 0:
                print(f"\n  WARNING: NaN/Inf gradients at step {step}, skipping optimizer step")
        else:
            grad_norm_before = _grad_l2_norm(trainable_params)
            grad_clip_active = grad_clip_value
            if not torch.isfinite(grad_norm_before):
                grad_clip_ratio = torch.tensor(0.0, device=device)
                grad_clip_hit = torch.tensor(0.0, device=device)
            else:
                grad_norm_before_f = float(grad_norm_before.item())
                grad_clip_ratio = torch.tensor(1.0, device=device)
                grad_clip_hit = torch.tensor(0.0, device=device)
                if grad_clip_active > 0:
                    clip_ratio_f = min(1.0, grad_clip_active / max(1e-12, grad_norm_before_f))
                    grad_clip_ratio = torch.tensor(clip_ratio_f, device=device)
                    grad_clip_hit = torch.tensor(1.0 if grad_norm_before_f > grad_clip_active else 0.0, device=device)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_active)
            if torch.isfinite(grad_norm):
                opt_gen.step()
                sched_gen.step()
            grad_norm_after = _grad_l2_norm(trainable_params)
            if torch.isfinite(grad_norm_before) and grad_norm_before.item() > 0:
                grad_norm_ratio = grad_norm_after / (grad_norm_before + 1e-12)
            else:
                grad_norm_ratio = torch.tensor(0.0, device=device)

            if grad_clip_auto:
                grad_clip_ratio_history.append(float(grad_clip_ratio.item()))
                if len(grad_clip_ratio_history) > grad_clip_auto_window:
                    grad_clip_ratio_history.pop(0)
                if len(grad_clip_ratio_history) == grad_clip_auto_window and grad_clip_value > grad_clip_auto_floor:
                    severe_clip_frac = sum(
                        1.0 for ratio in grad_clip_ratio_history if ratio < grad_clip_auto_threshold
                    ) / len(grad_clip_ratio_history)
                    if severe_clip_frac >= 0.7:
                        old_clip = grad_clip_value
                        grad_clip_value = max(grad_clip_auto_floor, grad_clip_value * grad_clip_auto_decay)
                        if grad_clip_value < old_clip:
                            grad_clip_ratio_history = []
                            _debug_log(
                                "train_pipeline.py:467",
                                "auto_clip_damped",
                                {
                                    "step": step,
                                    "old_clip": old_clip,
                                    "new_clip": grad_clip_value,
                                    "severe_clip_frac": severe_clip_frac,
                                    "window": grad_clip_auto_window,
                                    "threshold": grad_clip_auto_threshold,
                                },
                                hypothesis_id="B",
                                run_id="pre-fix",
                            )
                            print(
                                f"\n  [clip-auto] step {step}: clip_norm {old_clip:.4f} -> {grad_clip_value:.4f}"
                                f" (severe clip ratio {severe_clip_frac:.2f})"
                            )

        # Discriminator update (Cycle 25: MRSTFT only)
        # Cycle 32: Train discriminator in Stage 1 for Mamba/Zipformer decoders
        if opt_disc and use_adv and mrstft_disc:
            disc_grad_before = _grad_l2_norm(disc_params)
            # Reuse real logits from the generator pass (feature matching already detached reals).
            real_for_d = [r.detach() for r in real_out]
            if mr_pred_mags is not None:
                fake_out, _ = mrstft_disc.forward_from_mags(
                    [m.unsqueeze(1).detach() for m in mr_pred_mags]
                )
            else:
                fake_out, _ = mrstft_disc(x_hat.detach())
            disc_loss = adversarial_d_loss(real_for_d, fake_out)
            if disc_loss > 0 and torch.isfinite(disc_loss):
                opt_disc.zero_grad()
                disc_loss.backward()
                if disc_params:  # Only clip if discriminator has params
                    disc_grad = torch.nn.utils.clip_grad_norm_(disc_params, 5.0)
                opt_disc.step()
                if sched_disc:
                    sched_disc.step()
            disc_grad_after = _grad_l2_norm(disc_params)
            if torch.isfinite(disc_grad_before) and disc_grad_before.item() > 0:
                disc_grad_ratio = disc_grad_after / (disc_grad_before + 1e-12)
            else:
                disc_grad_ratio = torch.tensor(0.0, device=device)
        elif not use_adv:
            disc_grad_ratio = torch.tensor(0.0, device=device)

        # Timing
        elapsed = time.time() - t0
        step_times.append(elapsed)
        if len(step_times) > 100:
            step_times.pop(0)
        avg_step = sum(step_times) / len(step_times)

        # Logging
        if step % cfg.log_every == 0 or step == start_step:
            raw_bps = cfg.bitrate if use_vq else 0
            eff_bps = raw_bps * float(vq_util) if use_vq else 0.0
            metrics = {
                'mel': loss_mel.item() if isinstance(loss_mel, torch.Tensor) else loss_mel,
                'adv': loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv,
                'commit': commit_loss.item() if isinstance(commit_loss, torch.Tensor) else commit_loss,
                'feat': loss_feat.item() if isinstance(loss_feat, torch.Tensor) else loss_feat,
                'cb': cb_loss.item() if isinstance(cb_loss, torch.Tensor) else cb_loss,
                'entropy': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
                'mrstft': loss_mrstft.item() if isinstance(loss_mrstft, torch.Tensor) else loss_mrstft,
                'disc': disc_loss.item() if isinstance(disc_loss, torch.Tensor) else disc_loss,
                'gen': loss_gen.item() if isinstance(loss_gen, torch.Tensor) else loss_gen,
                'grad_before': grad_norm_before.item() if isinstance(grad_norm_before, torch.Tensor) else grad_norm_before,
                'grad_after': grad_norm_after.item() if isinstance(grad_norm_after, torch.Tensor) else grad_norm_after,
                'grad_ratio': grad_norm_ratio.item() if isinstance(grad_norm_ratio, torch.Tensor) else grad_norm_ratio,
                'grad_clip_ratio': grad_clip_ratio.item() if isinstance(grad_clip_ratio, torch.Tensor) else grad_clip_ratio,
                'grad_clip_hit': grad_clip_hit.item() if isinstance(grad_clip_hit, torch.Tensor) else grad_clip_hit,
                'grad_clip_cap': grad_clip_value,
                'disc_grad_before': disc_grad_before.item() if isinstance(disc_grad_before, torch.Tensor) else disc_grad_before,
                'disc_grad_after': disc_grad_after.item() if isinstance(disc_grad_after, torch.Tensor) else disc_grad_after,
                'disc_grad_ratio': disc_grad_ratio.item() if isinstance(disc_grad_ratio, torch.Tensor) else disc_grad_ratio,
                'vq_util': vq_util,
                'kf_frac': kf_frac,
                'grad': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'lr': sched_gen.get_last_lr()[0],
                'adv_active': 1.0 if use_adv else 0.0,
                'enc_lat': float(last_latency.get("encode_latency_ms", 0.0)),
                'dec_lat': float(last_latency.get("decode_latency_ms", 0.0)),
                'tot_lat': float(last_latency.get("total_latency_ms", 0.0)),
                'eff_bps': eff_bps,
            }
            # region agent log
            if step == start_step:
                if use_vq:
                    stage_warmup = min(cfg.warmup_steps, cfg.total_steps // 10)
                    rel = step - start_step
                    if rel < stage_warmup:
                        stage_factor = rel / stage_warmup
                    else:
                        remaining = cfg.total_steps - start_step
                        stage_factor = 0.5 * (1 + math.cos(math.pi * (rel - stage_warmup) / max(1, remaining - stage_warmup)))
                else:
                    stage_factor = 1.0
                _debug_log(
                    "train_pipeline.py:404",
                    "train_step_lr_probe",
                    {
                        "step": step,
                        "sched_last_epoch": sched_gen.last_epoch,
                        "lr_logged": float(sched_gen.get_last_lr()[0]),
                        "opt_lr": float(opt_gen.param_groups[0]["lr"]) if opt_gen.param_groups else 0.0,
                        "stage_factor_predict": stage_factor,
                        "start_step": start_step,
                        "use_vq": bool(use_vq),
                        "use_adv": bool(use_adv),
                    },
                    hypothesis_id="A",
                    run_id="pre-fix",
                )
            # endregion
            stage_label = "S1" if not use_vq else "S2"
            print_status(step, end_step, metrics, avg_step, stage_label)

            with open(cfg.log_tsv, "a") as f:
                f.write(f"{step}\t{raw_bps:.0f}\t{metrics['eff_bps']:.2f}\t{n_cb}\t"
                        f"{metrics['gen']:.4f}\t{metrics['mel']:.4f}\t"
                        f"{metrics['adv']:.4f}\t{metrics['feat']:.4f}\t"
                        f"{metrics['commit']:.4f}\t{metrics['cb']:.4f}\t"
                        f"{metrics['entropy']:.4f}\t{metrics['mrstft']:.4f}\t"
                        f"{metrics['disc']:.4f}\t{metrics['grad_before']:.4f}\t"
                        f"{metrics['grad_after']:.4f}\t{metrics['grad_ratio']:.4f}\t"
                        f"{metrics['grad_clip_ratio']:.4f}\t{metrics['grad_clip_hit']:.0f}\t"
                        f"{metrics['grad_clip_cap']:.4f}\t"
                        f"{metrics['disc_grad_before']:.4f}\t{metrics['disc_grad_after']:.4f}\t"
                        f"{metrics['disc_grad_ratio']:.4f}\t{vq_util:.4f}\t{metrics['grad']:.4f}\t"
                        f"{metrics['lr']:.6f}\t{metrics['adv_active']:.1f}\t"
                        f"{metrics['enc_lat']:.2f}\t{metrics['dec_lat']:.2f}\t{metrics['tot_lat']:.2f}\t"
                        f"{stage_label}\n")

        # Evaluation
        if step % cfg.eval_every == 0 and step > start_step and not skip_eval:
            sys.stdout.write("\n")
            from sirencodec.core.train import evaluate_codec
            codec.eval()
            eval_metrics = evaluate_codec(codec, cfg, device, num_samples=5, val_loader=val_loader)
            codec.train()
            last_latency["encode_latency_ms"] = eval_metrics["encode_latency_ms"]
            last_latency["decode_latency_ms"] = eval_metrics["decode_latency_ms"]
            last_latency["total_latency_ms"] = eval_metrics["total_latency_ms"]
            pesq_str = f"{eval_metrics['pesq']:.3f}" if eval_metrics['pesq'] > 0 else "N/A"
            sys.stdout.write(f"  [EVAL] SI-SDR={eval_metrics['si_sdr']:.2f}dB | "
                           f"PESQ={pesq_str} | VQ={eval_metrics['vq_utilization']:.2%} | "
                           f"Latency={eval_metrics['total_latency_ms']:.1f}ms\n")
            results_file = Path(cfg.results_tsv)
            if not results_file.exists():
                with open(results_file, "w") as f:
                    f.write(
                        "step\tsi_sdr\tpesq\tencode_latency_ms\tdecode_latency_ms\t"
                        "total_latency_ms\tvq_utilization\tn_evaluated\n"
                    )
            with open(results_file, "a") as f:
                f.write(
                    f"{step}\t{eval_metrics['si_sdr']:.4f}\t{eval_metrics['pesq']:.4f}\t"
                    f"{eval_metrics['encode_latency_ms']:.2f}\t{eval_metrics['decode_latency_ms']:.2f}\t"
                    f"{eval_metrics['total_latency_ms']:.2f}\t{eval_metrics['vq_utilization']:.6f}\t"
                    f"{eval_metrics['n_evaluated']}\n"
                )
            sys.stdout.flush()

        # Checkpoint
        if step % cfg.save_every == 0 and step > start_step:
            sys.stdout.write("\n")
            suffix = "autoencoder" if not use_vq else None
            save_path = save_checkpoint(step, codec, mrstft_disc, opt_gen, opt_disc,
                                        metrics['mel'], metrics['commit'], cfg, suffix=suffix)
            sys.stdout.write(f"  [CHECKPOINT] Saved {save_path.name}\n")
            sys.stdout.flush()

    sys.stdout.write("\n")
    return end_step


def print_status(step, total_steps, metrics, step_time, stage_label=""):
    """Print a single-line status update."""
    pct = step / total_steps * 100
    eta = (total_steps - step) * step_time
    stage = f"[{stage_label}] " if stage_label else ""
    parts = [
        f"\r{stage}[{pct:5.1f}%] Step {step:>6d}/{total_steps}",
        f"mel={metrics.get('mel', 0):.3f}",
        f"adv={metrics.get('adv', 0):.3f}",
        f"feat={metrics.get('feat', 0):.3f}",
        f"cb={metrics.get('cb', 0):.3f}",
        f"ent={metrics.get('entropy', 0):.3f}",
        f"commit={metrics.get('commit', 0):.3f}",
        f"disc={metrics.get('disc', 0):.3f}",
        f"mrstft={metrics.get('mrstft', 0):.3f}",
        f"g-ratio={metrics.get('grad_ratio', 0):.2f}",
        f"clip={metrics.get('grad_clip_hit', 0):.0f}@{metrics.get('grad_clip_ratio', 0):.2f}",
        f"dg-ratio={metrics.get('disc_grad_ratio', 0):.2f}",
        f"vq={metrics.get('vq_util', 0):.1%}",
        f"kf={metrics.get('kf_frac', 0):.1%}",
        f"adv-on={metrics.get('adv_active', 0):.0f}",
        f"grad={metrics.get('grad', 0):.1f}",
        f"lr={metrics.get('lr', 0):.6f}",
        f"{step_time*1000:.0f}ms/step",
        f"ETA:{format_time(eta)}",
    ]
    sys.stdout.write("  ".join(parts))
    sys.stdout.flush()


# ──────────────────────────────────────────────
# Device (CUDA / Apple MPS / CPU) — full arch-a-v2b + Vocos + losses
# ──────────────────────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────
def run_pipeline(args):
    device = pick_device()
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    elif device.type == "mps":
        print("Device: Apple MPS (full arch-a-v2b — Vocos iSTFT, mel, MRSTFT, EMA VQ)")
    else:
        print("WARNING: Running on CPU - this will be very slow!")

    # Build config
    cfg = make_config(
        arch=args.arch,
        steps=args.steps,
        batch_size=args.batch_size,
        segment_length=args.segment,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        use_real_data=args.real_data,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        codebook_size=args.codebook_size,
        psych_masking=args.psych_masking,
        data_num_workers=args.data_num_workers,
        decoder_type=args.decoder,
        use_preprocessed_cache=args.use_preprocessed,
        preprocessed_dir=args.preprocessed_dir,
        preprocessed_manifest_name=args.preprocessed_manifest_name,
        enable_augmentation=not args.no_augment and not args.use_preprocessed,
        use_mrstft=not args.no_mrstft,
        stage1_steps=args.stage1_steps,
        skip_stage1=args.skip_stage1,
        codebook_dim=args.codebook_dim,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        lambda_mel=args.lambda_mel,
        frame_skip_target=args.frame_skip_target,
    )

    cfg.adv_start_step = (
        int(args.adv_start_step) if getattr(args, "adv_start_step", None) is not None else cfg.warmup_steps
    )
    cfg.adv_every = max(1, int(getattr(args, "adv_every", 1)))
    cfg.no_adv_stage1 = bool(getattr(args, "no_adv_stage1", False))
    print(
        f"MRSTFT adversarial: enable after step {cfg.adv_start_step}, "
        f"every {cfg.adv_every} step(s) (discriminator is the main cost)"
    )
    if cfg.no_adv_stage1:
        print("  Stage 1: adversarial OFF (--no-adv-stage1) — mel-only autoencoder; Stage 2 still uses disc.")

    # Inductor+complex-decoder path is unstable in this stack (complex gradient lowering issue).
    # Force a stable compile backend unless user explicitly asks for inductor with hifigan.
    if args.torch_compile and args.torch_compile_backend == "inductor":
        if getattr(cfg, "decoder_type", args.decoder) in {"vocos", "zipformer", "zipformer2", "mamba"}:
            _debug_log(
                "train_pipeline.py:748",
                "compile_backend_forced_fallback",
                {
                    "requested_backend": args.torch_compile_backend,
                    "fallback_backend": "aot_eager",
                    "decoder_type": args.decoder,
                },
                hypothesis_id="E",
                run_id="pre-fix",
            )
            print(
                "Torch compile backend fallback: inductor -> aot_eager"
                " (complex-operator path for selected decoder is unstable under inductor)"
            )
            args.torch_compile_backend = "aot_eager"
    cfg.grad_clip_auto = bool(args.grad_clip_auto)
    cfg.grad_clip_auto_window = int(max(1, args.grad_clip_auto_window))
    cfg.grad_clip_auto_threshold = float(args.grad_clip_auto_threshold)
    cfg.grad_clip_auto_floor = float(args.grad_clip_auto_floor)
    cfg.grad_clip_auto_decay = float(args.grad_clip_auto_decay)
    # region agent log
    _debug_log(
        "train_pipeline.py:534",
        "cfg_built",
        {
            "arch": cfg.architecture,
            "total_steps": cfg.total_steps,
            "stage1_steps": cfg.stage1_steps,
            "skip_stage1": bool(cfg.skip_stage1),
            "warmup_steps": cfg.warmup_steps,
            "use_mrstft": bool(cfg.use_mrstft),
            "lr_gen": cfg.lr_gen,
            "lambda_mel": cfg.lambda_mel,
            "lr_disc": cfg.lr_disc,
            "grad_clip_auto": bool(cfg.grad_clip_auto),
            "grad_clip_auto_window": cfg.grad_clip_auto_window,
            "grad_clip_auto_threshold": cfg.grad_clip_auto_threshold,
            "grad_clip_auto_floor": cfg.grad_clip_auto_floor,
            "grad_clip_auto_decay": cfg.grad_clip_auto_decay,
            "resume": bool(args.resume),
            "load_checkpoint": bool(args.load_checkpoint),
            "adv_start_step": int(getattr(cfg, "adv_start_step", cfg.warmup_steps)),
            "adv_every": int(getattr(cfg, "adv_every", 1)),
            "no_adv_stage1": bool(getattr(cfg, "no_adv_stage1", False)),
        },
        hypothesis_id="D",
        run_id="pre-fix",
    )
    # endregion

    # Create models
    if cfg.architecture == "arch-a-spk":
        codec = AudioCodecASPK(cfg).to(device)
    elif cfg.architecture == "arch-b-v1":
        codec = AudioCodecB(cfg).to(device)
    elif cfg.architecture == "arch-c-v1":
        codec = AudioCodecC(cfg).to(device)
    elif cfg.architecture == "arch-d-v1":
        codec = AudioCodecD(cfg).to(device)
    else:
        codec = AudioCodec(cfg).to(device)

    # Cycle 25: MRSTFT-only adversarial
    mrstft_disc = MultiResolutionSTFTDiscriminator().to(device) if cfg.use_mrstft else None

    if args.torch_compile:
        # Wrap model(s) with torch.compile for potential speedup.
        # Keep fallback if unsupported kernels appear.
        try:
            codec = torch.compile(
                codec,
                backend=args.torch_compile_backend,
                mode=args.torch_compile_mode,
                fullgraph=args.torch_compile_fullgraph,
            )
            if mrstft_disc is not None:
                mrstft_disc = torch.compile(
                    mrstft_disc,
                    backend=args.torch_compile_backend,
                    mode=args.torch_compile_mode,
                    fullgraph=args.torch_compile_fullgraph,
                )
            _debug_log(
                "train_pipeline.py:734",
                "torch_compile_enabled",
                {
                    "compile_backend": args.torch_compile_backend,
                    "compile_mode": args.torch_compile_mode,
                    "fullgraph": bool(args.torch_compile_fullgraph),
                },
                hypothesis_id="E",
                run_id="pre-fix",
            )
            print(
                f"Torch compile: enabled (backend={args.torch_compile_backend}, mode={args.torch_compile_mode}, "
                f"fullgraph={args.torch_compile_fullgraph})"
            )
        except Exception as e:
            _debug_log(
                "train_pipeline.py:747",
                "torch_compile_failed",
                {"error": str(e)},
                hypothesis_id="E",
                run_id="pre-fix",
            )
            print(f"[warn] torch.compile failed ({e}); continuing without compilation")

    print(f"Params: {codec.count_params():.2f}M")
    print(f"Target bitrate: {cfg.bitrate} bps")

    # Optimizers
    gen_params = list(codec.parameters())
    disc_params = list(mrstft_disc.parameters()) if mrstft_disc else []

    opt_gen = optim.AdamW(gen_params, lr=cfg.lr_gen, betas=cfg.betas, weight_decay=cfg.weight_decay)
    opt_disc = optim.AdamW(disc_params, lr=cfg.lr_disc, betas=cfg.betas, weight_decay=cfg.weight_decay) if disc_params else None

    # Stage tracking (defined before scheduler so it can use current_stage)
    current_stage = 1
    start_step = 0

    # Learning rate scheduler
    # Fix: Use proper warmup that ensures non-zero LR from step 1
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            # Linear warmup: start at 0.5, ramp to 1.0
            return 0.5 + 0.5 * ((step + 1) / cfg.warmup_steps)
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)))

    sched_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda)
    sched_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda) if opt_disc else None
    # region agent log
    _debug_log(
        "train_pipeline.py:601",
        "global_scheduler_created",
        {
            "scheduler_last_epoch": sched_gen.last_epoch,
            "opt_gen_lr": float(opt_gen.param_groups[0]["lr"]) if opt_gen.param_groups else 0.0,
            "opt_disc_exists": bool(opt_disc),
            "lr_warmup_steps": cfg.warmup_steps,
        },
        hypothesis_id="B",
        run_id="pre-fix",
    )
    # endregion

    # Loss functions
    if cfg.use_psych_masking:
        mel_loss_fn = PsychoacousticMaskedMelLoss(cfg).to(device)
    else:
        mel_loss_fn = None  # use default mel loss

    mrstft_loss_fn = MultiResolutionSTFTLoss().to(device) if cfg.use_mrstft else None

    # Data loading
    train_loader = val_loader = None
    use_real_data = False
    if cfg.use_real_data and HAS_DATA_PIPELINE:
        print("Loading real speech data...")
        data_cfg = DataConfig(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.data_num_workers,
            segment_length=cfg.segment_length,
            enable_augmentation=getattr(cfg, "enable_augmentation", True),
            use_preprocessed_cache=getattr(cfg, "use_preprocessed_cache", False),
            preprocessed_dir=getattr(cfg, "preprocessed_dir", "preprocessed"),
            preprocessed_manifest_name=getattr(cfg, "preprocessed_manifest_name", "master_manifest_preprocessed.jsonl"),
        )
        train_loader, val_loader = create_dataloaders(data_cfg)
        use_real_data = True
        print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    else:
        print("Using synthetic data (sine sweeps)")

    # Log file
    log_file = Path(cfg.log_tsv)
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write(
                "step\traw_bitrate_bps\teff_bitrate_bps\tn_codebooks\tloss_total\t"
                "loss_mel\tloss_adv\tloss_feat\tloss_commit\tloss_cb\tloss_entropy\t"
                "loss_mrstft\tloss_disc\tgrad_before\tgrad_after\tgrad_ratio\t"
                "disc_grad_before\tdisc_grad_after\tdisc_grad_ratio\tvq_utilization\t"
                "grad_norm\tlr\tadv_active\tencode_latency_ms\tdecode_latency_ms\t"
                "total_latency_ms\tstage\n"
            )

    # Resume from checkpoint
    if args.resume:
        latest = find_latest_checkpoint()
        if latest:
            start_step, _, _ = load_checkpoint(
                latest, codec, mrstft_disc, opt_gen, opt_disc, device
            )
            # Detect stage from step number
            current_stage = 2 if start_step >= cfg.stage1_steps else 1
            print(f"  Resumed: stage {current_stage}, step {start_step}")
            sched_gen.last_epoch = max(start_step - 1, -1)
            if sched_disc is not None:
                sched_disc.last_epoch = max(start_step - 1, -1)
            # region agent log
            _debug_log(
                "train_pipeline.py:655",
                "resume_resume_state",
                {
                    "latest_checkpoint": str(latest),
                    "start_step": start_step,
                    "current_stage": current_stage,
                    "sched_gen_last_epoch": sched_gen.last_epoch,
                    "sched_disc_last_epoch": sched_disc.last_epoch if sched_disc else None,
                },
                hypothesis_id="B",
                run_id="pre-fix",
            )
            # endregion
        else:
            print("No checkpoint found, starting from scratch")
    
    # Load checkpoint for fine-tuning (Cycle 29: transfer learning)
    if args.load_checkpoint:
        print(f"\nLoading checkpoint for fine-tuning: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, weights_only=False, map_location=device)
        # Load only matching keys (handles architecture changes)
        model_dict = codec.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['codec'].items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        codec.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} params")
        print(f"  Skipping Stage 1 (using pretrained encoder/decoder)")
        cfg.skip_stage1 = True
        current_stage = 2
        start_step = 0  # Reset step counter for fine-tuning

    # ═══════════════════════════════════════
    # VQ training mode: Mamba uses standard two-stage training
    # Stage 1: autoencoder (no VQ), Stage 2: VQ fine-tuning
    # ═══════════════════════════════════════
    use_vq_from_start = False  # Disabled — Mamba uses standard two-stage now

    t_start = time.time()
    stage1_end = min(cfg.stage1_steps, cfg.total_steps) if not cfg.skip_stage1 else 0

    if current_stage == 1 and stage1_end > start_step and not cfg.skip_stage1:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Autoencoder training (encoder → decoder, no VQ)")
        print(f"Steps: {start_step} → {stage1_end}")
        print(f"{'='*60}")
        # region agent log
        _debug_log(
            "train_pipeline.py:685",
            "enter_stage1",
            {
                "current_stage": current_stage,
                "start_step": start_step,
                "stage1_end": stage1_end,
                "skip_stage1": bool(cfg.skip_stage1),
            },
            hypothesis_id="D",
            run_id="pre-fix",
        )
        # endregion

        start_step = run_stage(
            codec, cfg, device, start_step, stage1_end,
            mrstft_disc, opt_gen, opt_disc,
            sched_gen, sched_disc,
            train_loader, val_loader, mel_loss_fn, mrstft_loss_fn,
            use_vq=False, skip_eval=args.no_eval,
        )

        # Save stage 1 checkpoint
        print(f"\n{'='*60}")
        print(f"Stage 1 complete! Saving autoencoder checkpoint...")
        save_path = save_checkpoint(start_step, codec, mrstft_disc, opt_gen, opt_disc,
                                   0, 0, cfg, suffix="autoencoder")
        print(f"Saved: {save_path}")
        print(f"{'='*60}\n")

    # ═══════════════════════════════════════
    # STAGE 2: VQ fine-tuning
    # Insert VQ between frozen encoder and decoder
    # ═══════════════════════════════════════
    if start_step < cfg.total_steps:
        print(f"\n{'='*60}")
        print(f"STAGE 2: VQ fine-tuning (encoder → VQ → decoder)")
        print(f"Steps: {start_step} → {cfg.total_steps}")
        print(f"{'='*60}")
        # region agent log
        _debug_log(
            "train_pipeline.py:712",
            "enter_stage2",
            {
                "current_stage": current_stage,
                "start_step": start_step,
                "total_steps": cfg.total_steps,
                "skip_stage1": bool(cfg.skip_stage1),
            },
            hypothesis_id="A",
            run_id="pre-fix",
        )
        # endregion

        start_step = run_stage(
            codec, cfg, device, start_step, cfg.total_steps,
            mrstft_disc, opt_gen, opt_disc,
            sched_gen, sched_disc,
            train_loader, val_loader, mel_loss_fn, mrstft_loss_fn,
            use_vq=True, skip_eval=args.no_eval,
        freeze_encoder=False,  # Default to not freezing encoder in Stage 2
        )

    # Final checkpoint
    final_path = save_checkpoint(cfg.total_steps - 1, codec, mrstft_disc, opt_gen, opt_disc,
                                 0, 0, cfg)

    total_time = time.time() - t_start
    print(f"\nTraining complete: {cfg.total_steps} steps in {format_time(total_time)}")
    print(f"Final checkpoint: {final_path}")
    print(f"Average speed: {total_time/cfg.total_steps*1000:.0f}ms/step")

    # Final eval
    if not args.no_eval:
        from sirencodec.core.train import evaluate_codec
        codec.eval()
        eval_metrics = evaluate_codec(codec, cfg, device, num_samples=10, val_loader=val_loader)
        pesq_str = f"{eval_metrics['pesq']:.3f}" if eval_metrics['pesq'] > 0 else "N/A"
        print(f"\nFinal Evaluation:")
        print(f"  SI-SDR: {eval_metrics['si_sdr']:.2f} dB")
        print(f"  PESQ:   {pesq_str}")
        print(f"  VQ util: {eval_metrics['vq_utilization']:.2%}")
        print(f"  Latency: {eval_metrics['total_latency_ms']:.1f} ms")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CODEC Training Pipeline")
    parser.add_argument("--arch", type=str, default="arch-a-v2b",
                       choices=["arch-a-v2b", "arch-a-spk", "arch-b-v1", "arch-c-v1", "arch-d-v1"])
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment", type=int, default=16000, help="Audio segment length in samples")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--no-eval", action="store_true", help="Skip PESQ evaluation")
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--real-data", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, default="data/cv-corpus")
    parser.add_argument("--data-num-workers", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)  # unused, kept for compat
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--no-mrstft", action="store_true", help="Disable MRSTFT adversarial (MSE only)")
    parser.add_argument("--use-preprocessed", action="store_true", help="Use precomputed preprocessing cache")
    parser.add_argument("--preprocessed-dir", type=str, default="preprocessed", help="Directory under data_dir with precomputed audio")
    parser.add_argument("--preprocessed-manifest-name", type=str, default="master_manifest_preprocessed.jsonl", help="Filename of preprocessed manifest")
    parser.add_argument("--no-augment", action="store_true", help="Disable audio augmentation during training")
    parser.add_argument("--psych-mask", action="store_true", default=True, dest="psych_masking")
    parser.add_argument("--decoder", type=str, default="vocos", choices=["vocos", "hifigan", "zipformer", "zipformer2", "mamba"])
    parser.add_argument("--stage1-steps", type=int, default=50000, help="Stage 1 (autoencoder) steps")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip stage 1")
    parser.add_argument("--codebook-dim", type=int, default=128, help="Latent dimension (128=Mimi-style, 64=small)")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Load weights from specific checkpoint file (not resume)")
    parser.add_argument("--frame-skip-target", type=float, default=0.35, help="Target fraction of keyframes (ARCH-C: 0.35=default, 0.25=aggressive)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (before optimizer step)")
    parser.add_argument("--grad-clip-auto", action="store_true", help="Auto-reduce grad clip when heavy clipping persists")
    parser.add_argument("--grad-clip-auto-window", type=int, default=30, help="Window for auto-clip detection")
    parser.add_argument("--grad-clip-auto-threshold", type=float, default=0.25, help="Heavy-clipping threshold on clip ratio")
    parser.add_argument("--grad-clip-auto-floor", type=float, default=0.1, help="Minimum clip norm when auto-damping")
    parser.add_argument("--grad-clip-auto-decay", type=float, default=0.7, help="Multiplicative decay when auto-damp triggers")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="LR warmup steps")
    parser.add_argument(
        "--adv-start-step",
        type=int,
        default=None,
        help="First step after which MRSTFT adversarial + disc run (default: same as --warmup-steps)",
    )
    parser.add_argument(
        "--adv-every",
        type=int,
        default=1,
        help="Run adversarial + discriminator only every N steps (1=all; 4 cuts adv-phase cost ~3-4×)",
    )
    parser.add_argument(
        "--no-adv-stage1",
        action="store_true",
        help="Stage 1 (autoencoder, no VQ): train mel-only; no MRSTFT discriminator (much faster). Stage 2 unchanged.",
    )
    parser.add_argument("--lambda-mel", type=float, default=None, help="Override mel-loss weight")
    parser.add_argument("--torch-compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--torch-compile-backend", type=str, default="aot_eager", choices=["inductor", "aot_eager", "eager"],
                        help="Backend for torch.compile (aot_eager is default for stability)")
    parser.add_argument("--torch-compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--torch-compile-fullgraph", action="store_true", help="Use fullgraph=True for torch.compile")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (2x speedup)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")

    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
