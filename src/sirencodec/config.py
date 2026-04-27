"""Training configuration and STFT/codebook CLI parsers (no MLX)."""
from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Smoke / ``--fast`` preset (three scales).
DEFAULT_STFT_SCALES: tuple[tuple[int, int], ...] = ((512, 128), (1024, 256), (2048, 512))
FAST_STFT_SCALES: tuple[tuple[int, int], ...] = ((512, 128), (1024, 256))
# LibriSpeech default: wide multi-scale STFT (Parallel WaveGAN-style).
LIBRI_STFT_SCALES: tuple[tuple[int, int], ...] = ((1024, 256), (2048, 512), (4096, 1024))
LIBRI_STFT_SCALE_WEIGHTS: tuple[float, ...] = (1.0, 1.75, 2.5)


def _detect_physical_cpu_cores() -> int:
    env = (sys.platform or "").lower()
    try:
        if env.startswith("win"):
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            out = (proc.stdout or "").strip()
            if out:
                val = int(out)
                if val > 0:
                    return val
        elif env == "darwin":
            proc = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                check=True,
            )
            out = (proc.stdout or "").strip()
            if out:
                val = int(out)
                if val > 0:
                    return val
        elif env.startswith("linux"):
            physical: set[tuple[str, str]] = set()
            current_phys = "0"
            current_core = ""
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_core:
                            physical.add((current_phys, current_core))
                        current_phys = "0"
                        current_core = ""
                        continue
                    if ":" not in line:
                        continue
                    key, value = [x.strip() for x in line.split(":", 1)]
                    if key == "physical id":
                        current_phys = value
                    elif key == "core id":
                        current_core = value
            if current_core:
                physical.add((current_phys, current_core))
            if physical:
                return len(physical)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 1))


@dataclass
class Config:
    """Defaults: LibriSpeech, GAN-free spectral recipe (SC + Complex STFT + multi-scale log-mag)."""

    sample_rate: int = 16000
    # Longer segment = more latent frames per chunk (helps RVQ stats at high time downsampling).
    segment: int = 16384  # 1.0 s @ 16 kHz
    # FastGAN preset: keep architecture, cut loss compute.
    batch: int = 8
    # Disk batch: concurrent ``sf.read`` per row when ``>1`` (Libri I/O often leaves GPU idle if sequential).
    load_audio_threads: int = field(default_factory=_detect_physical_cpu_cores)  # 0 or 1 = legacy sequential loop; else ``min(batch, threads, 32)`` workers.
    # Prefetch the next batch on a worker thread while the current step runs (only with ``--data-dir`` / Libri).
    prefetch_audio: bool = True
    # Number of ready/queued CPU batches when prefetching. >1 hides disk jitter better on Windows.
    prefetch_audio_batches: int = 2
    steps: int = 250_000
    lr: float = 8e-5
    # Adam LR: cosine decay to lr×lr_min_ratio by last step, or none = constant --lr.
    lr_schedule: str = "cosine"  # "none" | "cosine"
    lr_min_ratio: float = 0.25  # cosine floor = lr * ratio (0 = hard zero at end)
    lr_warmup_steps: int = 0  # 0 = off; else linear 0→lr for this many optimizer steps first
    # Global L2 grad clip before Adam (0 = off). Helps deeper stride1 stacks + sharp marginal softmax.
    grad_clip_norm: float = 5.0
    # Micro-batch ``batch`` × ``grad_accum_steps`` optimizer updates = average gradient (effective larger batch).
    grad_accum_steps: int = 6
    seed: int = 0
    # Encoder / decoder channels (NLC layout: batch, time, channels). len = number of stride-2 stages.
    # 8 stages → 2^8=256× time downsampling; default RVQ 3×K=32 → 3·log2(32)·(sr/256) ≈ 0.94 kb/s nominal index rate @ 16 kHz.
    enc_channels: tuple[int, ...] = (24, 32, 48, 64, 96, 128, 192, 256)
    # Per resolution: stride-1 Conv+GELU (same width) before each stride-2 down / after each stride-2 up (0 = legacy).
    # Default 0: cold start with 1 can yield all-NaN gradients (deep MLX graph + default init); use 1 after stable warmup or resume.
    stride1_blocks_per_scale: int = 0
    # Wider D at same n_q×K×f_latent → same nominal index b/s, richer codewords in R^D.
    latent_dim: int = 512
    # LayerNorm on encoder output before RVQ (off by default; can interact badly with cosine RVQ early).
    pre_vq_layernorm: bool = False
    # Residual dilated Conv1d on latent sequence (NLC): smooths temporal VQ trajectory.
    latent_temporal_depth: int = 2  # 0 = off; applied after optional pre_vq_ln, before RVQ
    latent_temporal_post_depth: int = 2  # 0 = off; applied on quantized z_q before decoder
    # Residual VQ (SoundStream / EnCodec style): each stage quantizes the remaining residual
    n_codebooks: int = 3
    codebook_size: int = 32
    # Per-stage codebook sizes (coarse→fine often uses decreasing K, e.g. 256,128,64). None = repeat ``codebook_size``.
    codebook_sizes: tuple[int, ...] | None = None
    # Cosine / spherical assignment: argmin ||r̂−ê|| with r̂,ê unit; z_q = ê·‖r‖ — same K,n_q → same nominal bitrate, often better u
    vq_cosine: bool = True
    vq_commitment: float = 1.35  # β on commitment (higher → less codebook collapse at low bitrate)
    lambda_vq: float = 5.0  # scale sum of VQ losses vs waveform/STFT
    ae_only: bool = False  # if True, skip VQ (pure autoencoder)
    # Loss (STFT often dominates RVQ at 256× downsample — ramp + lower default helps)
    lambda_time: float = 1.0
    lambda_stft: float = 0.5  # Main log-mag L1 (multi-scale); higher without GAN.
    # Linear ramp: weight = lambda_stft * (stft_ramp_start_frac .. 1.0) over steps [0, stft_ramp_steps]
    stft_ramp_steps: int = 8_000  # 0 = no ramp (full λ_stft from step 0)
    stft_ramp_start_frac: float = 0.25  # higher (no GAN → spectral carries reconstruction quality)
    # Multi-scale STFT: (n_fft, hop) pairs — LibriSpeech preset by default.
    stft_scales: tuple[tuple[int, int], ...] = LIBRI_STFT_SCALES
    # Per-scale weights (same count as stft_scales); emphasize larger FFTs for fine harmonics.
    stft_scale_weights: tuple[float, ...] | None = LIBRI_STFT_SCALE_WEIGHTS
    # Limit expensive spectral losses to N batch items per micro-batch (0 = all items).
    # Waveform, RVQ, and adversarial/semantic terms still see the full micro-batch.
    spectral_batch_items: int = 0
    # Run scales with n_fft >= this threshold only every N optimizer steps (1 = every step).
    # This keeps 4096/8192 FFT pressure without paying their full cost on every update.
    stft_large_min_fft: int = 0
    stft_large_every: int = 1
    # Extra emphasis on high-frequency bins in log-mag L1: weight ∝ 1 + γ·(f/F_max)² (0 = uniform mean)
    stft_hf_emphasis: float = 1.0
    # Differentiable "sharpness": L1 on finite differences of log-mag STFT (freq + time). Uses same λ ramp as STFT.
    lambda_stft_grad: float = 0.1
    # Weight freq-axis vs time-axis gradient L1 (harmonic stripes = structure along f; default favors ∂/∂f).
    stft_grad_freq_weight: float = 3.0
    stft_grad_time_weight: float = 0.5
    # Mean (1 − cosine) between flattened log-mag STFT(pred) and STFT(tgt) per scale (same ramp as STFT). Independent of --lambda-cos (waveform).
    lambda_stft_cos: float = 0.05
    # Penalize log-STFT energy that rises above the target by more than this margin.
    # This specifically fights broadband "spectral floor" smear between harmonic peaks.
    lambda_stft_excess: float = 0.0
    stft_excess_margin: float = 0.20
    # Spectral Convergence (Yamamoto et al. 2020): ``‖|S(ŷ)|−|S(y)|‖_F / ‖|S(y)|‖_F`` per scale.
    # Scale-invariant; emphasizes harmonic peaks — classic GAN-free high-freq pressure. Same ramp as STFT.
    lambda_sc: float = 1.0
    # Complex STFT L1 on ``Re``/``Im`` (cheap vs mag-only): captures phase, keys "non-metallic" timbre without GAN.
    lambda_complex_stft: float = 0.1
    # Log-mel spectrogram (single STFT scale): mean |Δlog mel| and mean (Δlog mel)². Weighted by same λ_stft ramp as STFT.
    mel_n_fft: int = 1024
    mel_hop: int = 256
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = None  # None → Nyquist (sample_rate / 2)
    lambda_mel_l1: float = 0.06
    lambda_mel_l2: float = 0.0
    # Per-position softmax entropy (can be ~log(K) while hard indices collapse); 0 = off
    lambda_entropy: float = 0.0
    # Batch-marginal softmax entropy term: loss += λ·(log K − H(marginal)) (same grad as −λ·H)
    lambda_marginal: float = 0.35  # higher → stronger push of batch-marginal H toward log K (more codes used)
    # Softmax(-dist/τ) for batch-marginal H(p̄). Too-large τ → H≈log K even when hard indices collapsed (u≪K).
    marginal_tau: float = 0.04
    # Early collapse window: u/K often drops right after step 0 (random distances look diverse).
    # Linear decay: λ_marg_eff = λ_marg * (mult + (1−mult)·t), t∈[0,1] over [0, boost_steps]; then λ_marg.
    marginal_boost_steps: int = 24_000  # 0 = off; longer helps 100k+ runs before λ_marg drops to base
    marginal_boost_mult: float = 2.5  # step-0 multiplier (≥1); decays to 1.0 at end of boost
    # Replace codebook rows unused in the current batch when unique/K is below threshold (anti-collapse)
    vq_reset_every: int = 1000  # 0 = off; each reset syncs encoder+VQ to CPU (keep ≥500 for speed)
    # Reset dead rows when unique/K < this. 0.08 only fired for <~10/128 codes; 0.42 catches ~30–50% usage.
    vq_reset_collapse_frac: float = 0.42
    vq_reset_noise: float = 0.12  # Gaussian noise on new rows (× batch residual std); breaks identical copies
    vq_reset_shuffle: bool = True  # if severe collapse, permute all rows after fill (see severe_shuffle_frac in vq_reset_dead_codes)
    # Fill dead rows with K-means centroids of batch residuals (vs random picks); spreads codebook on data manifold.
    vq_reset_kmeans: bool = True
    # If unique ≤ this, replace **all** K rows (dead-only refill leaves the winning row → u stays 1).
    vq_reset_full_refresh_max_unique: int = 4
    vq_reset_log_every: int = 5000  # 0 = print every reset; else only when step % this == 0
    # Waveform cosine similarity (per-sample mean over batch); orig/recon [B,T,1]
    lambda_cos: float = 0.15  # adds λ·(1 − cos); pushes toward cos → 1
    cos_hinge: float = 0.0  # adds w·max(0, cos_target − cos); use to chase e.g. 90%+
    cos_target: float = 0.9
    # Differentiable negative log SI-SDR ratio on waveform. Cheap phase/waveform anchor for mag-heavy STFT presets.
    lambda_sisdr: float = 0.0
    # High-pass waveform anchor: L1 after pre-emphasis, useful when high harmonics smear under mag-only losses.
    lambda_preemph: float = 0.0
    preemph_coef: float = 0.97
    # During RVQ phases, keep a continuous z->decoder reconstruction anchor alive.
    # This prevents the encoder/decoder path learned in A from drifting while hard RVQ is optimized.
    lambda_ae_anchor_time: float = 0.0
    lambda_ae_anchor_cos: float = 0.0
    # Reconstruction loss balancer: off = raw lambdas, grad = EnCodec-style gradient-ratio balancing wrt y_hat.
    loss_balancer: str = "off"  # "off" | "grad"
    loss_balancer_eps: float = 1e-8
    loss_balancer_max_scale: float = 10.0
    # Frozen SSL teacher used only during training (not part of the codec model/checkpoints at inference time).
    lambda_semantic: float = 0.0
    semantic_model: str = "HUBERT_BASE"
    semantic_layers: tuple[int, ...] = (12,)
    semantic_batch_items: int = 8
    semantic_every: int = 16
    # Optional waveform GAN (hinge). Generator adversarial term participates in the reconstruction loss balancer.
    lambda_adv: float = 0.0
    # Discriminator feature matching. Strongly stabilizes MPD/MSD GAN starts.
    lambda_fm: float = 0.0
    disc_lr: float = 2e-4
    disc_type: str = "msd"  # "msd" | "mpd" | "msmpd"
    disc_base_channels: int = 32
    disc_scales: int = 3
    disc_periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    # Curriculum: A continuous AE, B RVQ/loss ramp, optional C GAN ramp, D full fine-tune.
    curriculum: bool = False
    curriculum_ae_frac: float = 0.15
    curriculum_vq_ramp_frac: float = 0.35
    curriculum_gan_frac: float = 0.25
    curriculum_vq_start: float = 0.10
    curriculum_entropy_start: float = 0.0
    curriculum_adv_start: float = 0.10
    # If false, phase B trains reconstruction with hard RVQ immediately while only VQ/marginal weights ramp.
    curriculum_quantize_blend: bool = True
    # Logging
    log_every: int = 50
    # EMA of waveform cos % in the log line: ema ← β·ema + (1−β)·cos; 0 = print raw cos only
    log_cos_ema_beta: float = 0.99
    # Spectrogram PNGs (matplotlib Agg)
    spectrogram_every: int = 2500  # 0 = off
    spectrogram_dir: str = "mlx_spectrograms_best_long"
    save_audio: bool = True  # with PNG: also write *_orig.wav / *_recon.wav (soundfile)
    # Longer PNG/WAV than training segment: seconds of audio (0 = use training batch for viz)
    spectrogram_seconds: float = 8.0  # 0 = use training batch length for PNG/WAV
    checkpoint_every: int = 0  # 0 = trainer default (10 epochs on fresh runs); explicit CLI still means updates
    checkpoint_dir: str = "mlx_checkpoints"
    # Dataset name chosen at the CLI. Fresh training runs require ``--dataset``.
    dataset: str | None = None
    # Deprecated legacy field kept for old checkpoints / resumes.
    use_librispeech: bool = False
    # Optional data root (recursive *.wav / *.flac / *.ogg / *.mp3). Filled from ``dataset`` in the trainer.
    data_dir: Path | None = None
    # Extra mean L1 on **linear** STFT magnitudes (same scales as log-STFT; weighted by λ_stft ramp).
    # Without GAN this carries more of the absolute harmonic-amplitude shaping.
    lambda_mag_l1: float = 0.15
    # Encoder/decoder activation: ``gelu`` | ``snake`` | ``snake_beta`` (DAC-style sin²).
    activation: str = "snake_beta"
    # Factorized RVQ: project latent D→d before codebook (0 = use full ``latent_dim``).
    rvq_code_dim: int = 8
    # EMA codebook update decay in ``(0,1)`` (0 = disabled). Updated on CPU after each optimizer step.
    vq_ema_decay: float = 0.99
    # Decoder upsample: ``transpose`` (ConvTranspose1d) or ``repeat_conv`` (repeat×2 + Conv1d).
    decoder_upsample: str = "repeat_conv"
    # Causal conv encoder/decoder (left padding only); streaming-friendly.
    causal: bool = False
    # bf16 weights/activations where safe (STFT kept fp32).
    use_bf16: bool = True
    # ``mx.compile`` on spectral loss block (fixed B,T from first batch).
    use_compile: bool = True
    # Save ``ckpt_stepN.safetensors`` with weights + Adam state + data offset (+ EMA if any).
    full_checkpoint: bool = True
    # Validation / logging: 0 = off. Writes ``log_mlx_tsv`` and optional ``results_tsv_path`` row.
    eval_every: int = 5000
    eval_clips: int = 4
    eval_seconds: float = 3.0
    # Fixed validation clip seed. Keeping this independent of step makes eval rows comparable.
    eval_seed: int = 0
    log_mlx_tsv: str = "log_mlx.tsv"
    results_tsv_path: str = "results.tsv"
    # Adaptive / BWE stubs (see ``sirencodec.adaptive``): ``none`` | ``bwe_stub`` | ``fps_stub``.
    adaptive_mode: str = "none"


def encoder_time_stride(cfg: Config) -> int:
    """Temporal downsampling factor: ``2 ** len(enc_channels)`` (each encoder stage is stride 2)."""
    return 2 ** len(cfg.enc_channels)


def effective_codebook_sizes(cfg: Config) -> tuple[int, ...]:
    """Resolved ``(K0, K1, …)`` length ``n_codebooks``."""
    if cfg.codebook_sizes is not None:
        if len(cfg.codebook_sizes) != cfg.n_codebooks:
            raise ValueError(
                f"codebook_sizes has length {len(cfg.codebook_sizes)}, expected n_codebooks={cfg.n_codebooks}"
            )
        for k in cfg.codebook_sizes:
            if int(k) < 2:
                raise ValueError(f"codebook size must be >= 2, got {k}")
        return tuple(int(x) for x in cfg.codebook_sizes)
    if cfg.codebook_size < 2:
        raise ValueError("codebook_size must be >= 2")
    return (int(cfg.codebook_size),) * cfg.n_codebooks


def mean_log_codebook_for_entropy(cfg: Config) -> float:
    """Mean ``log(K_i)`` — target for averaged marginal / entropy gaps when stages differ in K."""
    logs = [math.log(float(k)) for k in effective_codebook_sizes(cfg)]
    return sum(logs) / float(len(logs))


def nominal_rvq_kbps(cfg: Config) -> float:
    """Nominal index bitrate (kb/s): ``sum_i log2(K_i) × (sr / encoder_stride)``."""
    st = encoder_time_stride(cfg)
    fps = cfg.sample_rate / st
    bps = sum(math.log2(float(k)) for k in effective_codebook_sizes(cfg)) * fps
    return bps / 1000.0


def parse_codebook_sizes_arg(s: str) -> tuple[int, ...]:
    """``"256,128,64"`` → ``(256, 128, 64)``."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --codebook-sizes")
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v < 2:
            raise ValueError(f"codebook size must be >= 2, got {v}")
        out.append(v)
    return tuple(out)


def parse_stft_scales_arg(s: str) -> tuple[tuple[int, int], ...]:
    """``"512,128;1024,256"`` → ``((512,128), (1024,256))``."""
    out: list[tuple[int, int]] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError(f"expected n_fft,hop pair, got {chunk!r}")
        out.append((int(parts[0].strip()), int(parts[1].strip())))
    if not out:
        raise ValueError("empty --stft-scales")
    return tuple(out)


def parse_stft_scale_weights_arg(s: str) -> tuple[float, ...]:
    """``"1,1.5,2"`` → ``(1.0, 1.5, 2.0)`` — one weight per ``--stft-scales`` pair, order preserved."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --stft-scale-weights")
    out: list[float] = []
    for p in parts:
        v = float(p)
        if v < 0:
            raise ValueError(f"STFT scale weight must be >= 0, got {v}")
        out.append(v)
    if sum(out) <= 0:
        raise ValueError("sum of --stft-scale-weights must be > 0")
    return tuple(out)


def parse_positive_int_list_arg(s: str) -> tuple[int, ...]:
    """``"6,9,12"`` -> ``(6, 9, 12)``."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty integer list")
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v < 1:
            raise ValueError(f"expected positive integer, got {v}")
        out.append(v)
    return tuple(out)


def argparse_defaults_from_config(dc: Config | None = None) -> dict[str, object]:
    """Map ``Config`` → ``ArgumentParser.set_defaults`` (CLI stays in sync with dataclass)."""
    c = dc if dc is not None else Config()
    stft_s = ";".join(f"{n},{h}" for n, h in c.stft_scales)
    stft_w = None
    if c.stft_scale_weights is not None:
        stft_w = ",".join(str(float(x)) for x in c.stft_scale_weights)
    d: dict[str, object] = {
        "enc_channels": ",".join(str(x) for x in c.enc_channels),
        "steps": c.steps,
        "batch": c.batch,
        "grad_accum_steps": c.grad_accum_steps,
        "load_audio_threads": c.load_audio_threads,
        "prefetch_audio": c.prefetch_audio,
        "prefetch_audio_batches": c.prefetch_audio_batches,
        "segment": c.segment,
        "lr": c.lr,
        "lr_schedule": c.lr_schedule,
        "lr_min_ratio": c.lr_min_ratio,
        "lr_warmup_steps": c.lr_warmup_steps,
        "grad_clip": c.grad_clip_norm,
        "seed": c.seed,
        "stride1_blocks_per_scale": c.stride1_blocks_per_scale,
        "latent_dim": c.latent_dim,
        "pre_vq_layernorm": c.pre_vq_layernorm,
        "latent_temporal_depth": c.latent_temporal_depth,
        "latent_temporal_post_depth": c.latent_temporal_post_depth,
        "n_codebooks": c.n_codebooks,
        "codebook_size": c.codebook_size,
        "lambda_time": c.lambda_time,
        "lambda_stft": c.lambda_stft,
        "lambda_stft_grad": c.lambda_stft_grad,
        "lambda_stft_cos": c.lambda_stft_cos,
        "lambda_stft_excess": c.lambda_stft_excess,
        "stft_excess_margin": c.stft_excess_margin,
        "lambda_sc": c.lambda_sc,
        "lambda_complex_stft": c.lambda_complex_stft,
        "stft_grad_freq_weight": c.stft_grad_freq_weight,
        "stft_grad_time_weight": c.stft_grad_time_weight,
        "stft_ramp_steps": c.stft_ramp_steps,
        "stft_ramp_start": c.stft_ramp_start_frac,
        "stft_scales": stft_s,
        "stft_scale_weights": stft_w,
        "spectral_batch_items": c.spectral_batch_items,
        "stft_large_min_fft": c.stft_large_min_fft,
        "stft_large_every": c.stft_large_every,
        "stft_hf_emphasis": c.stft_hf_emphasis,
        "mel_n_fft": c.mel_n_fft,
        "mel_hop": c.mel_hop,
        "n_mels": c.n_mels,
        "mel_fmin": c.mel_fmin,
        "mel_fmax": c.mel_fmax,
        "lambda_mel_l1": c.lambda_mel_l1,
        "lambda_mel_l2": c.lambda_mel_l2,
        "lambda_vq": c.lambda_vq,
        "lambda_entropy": c.lambda_entropy,
        "lambda_marginal": c.lambda_marginal,
        "marginal_tau": c.marginal_tau,
        "marginal_boost_steps": c.marginal_boost_steps,
        "marginal_boost_mult": c.marginal_boost_mult,
        "vq_reset_every": c.vq_reset_every,
        "vq_reset_collapse_frac": c.vq_reset_collapse_frac,
        "vq_reset_noise": c.vq_reset_noise,
        "vq_reset_shuffle": c.vq_reset_shuffle,
        "vq_reset_kmeans": c.vq_reset_kmeans,
        "vq_reset_full_refresh_max_unique": c.vq_reset_full_refresh_max_unique,
        "vq_reset_log_every": c.vq_reset_log_every,
        "vq_cosine": c.vq_cosine,
        "vq_beta": c.vq_commitment,
        "lambda_cos": c.lambda_cos,
        "cos_hinge": c.cos_hinge,
        "cos_target": c.cos_target,
        "lambda_sisdr": c.lambda_sisdr,
        "lambda_preemph": c.lambda_preemph,
        "preemph_coef": c.preemph_coef,
        "lambda_ae_anchor_time": c.lambda_ae_anchor_time,
        "lambda_ae_anchor_cos": c.lambda_ae_anchor_cos,
        "loss_balancer": c.loss_balancer,
        "loss_balancer_eps": c.loss_balancer_eps,
        "loss_balancer_max_scale": c.loss_balancer_max_scale,
        "lambda_semantic": c.lambda_semantic,
        "semantic_model": c.semantic_model,
        "semantic_layers": ",".join(str(x) for x in c.semantic_layers),
        "semantic_batch_items": c.semantic_batch_items,
        "semantic_every": c.semantic_every,
        "lambda_adv": c.lambda_adv,
        "lambda_fm": c.lambda_fm,
        "disc_lr": c.disc_lr,
        "disc_type": c.disc_type,
        "disc_base_channels": c.disc_base_channels,
        "disc_scales": c.disc_scales,
        "disc_periods": ",".join(str(x) for x in c.disc_periods),
        "curriculum": c.curriculum,
        "curriculum_ae_frac": c.curriculum_ae_frac,
        "curriculum_vq_ramp_frac": c.curriculum_vq_ramp_frac,
        "curriculum_gan_frac": c.curriculum_gan_frac,
        "curriculum_vq_start": c.curriculum_vq_start,
        "curriculum_entropy_start": c.curriculum_entropy_start,
        "curriculum_adv_start": c.curriculum_adv_start,
        "curriculum_quantize_blend": c.curriculum_quantize_blend,
        "lambda_mag_l1": c.lambda_mag_l1,
        "activation": c.activation,
        "rvq_code_dim": c.rvq_code_dim,
        "vq_ema_decay": c.vq_ema_decay,
        "decoder_upsample": c.decoder_upsample,
        "causal": c.causal,
        "bf16": c.use_bf16,
        "compile_loss": c.use_compile,
        "full_checkpoint": c.full_checkpoint,
        "eval_every": c.eval_every,
        "eval_clips": c.eval_clips,
        "eval_seconds": c.eval_seconds,
        "eval_seed": c.eval_seed,
        "log_mlx_tsv": c.log_mlx_tsv,
        "results_tsv": c.results_tsv_path,
        "adaptive_mode": c.adaptive_mode,
        "dataset": c.dataset,
        "log_every": c.log_every,
        "log_cos_ema_beta": c.log_cos_ema_beta,
        "spectrogram_every": c.spectrogram_every,
        "spectrogram_dir": c.spectrogram_dir,
        "save_audio": c.save_audio,
        "spectrogram_seconds": c.spectrogram_seconds,
        "checkpoint_every": c.checkpoint_every,
        "checkpoint_dir": c.checkpoint_dir,
    }
    return d
