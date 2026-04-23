from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sirencodec.eval_metrics import pesq_wb_16k, stoi_16k, visqol_speech_16k

_SF_READ_LOCK = threading.Lock()
_MP3_UNSAFE_SUFFIXES = {".mp3"}


@dataclass(frozen=True)
class ClipRow:
    client_id: str
    rel_path: str
    age: str
    gender: str
    accents: str
    locale: str
    segment: str


def _read_validated_tsv(tsv_path: Path) -> list[ClipRow]:
    rows: list[ClipRow] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for d in r:
            rows.append(
                ClipRow(
                    client_id=(d.get("client_id") or "").strip(),
                    rel_path=(d.get("path") or "").strip(),
                    age=(d.get("age") or "").strip(),
                    gender=(d.get("gender") or "").strip(),
                    accents=(d.get("accents") or "").strip(),
                    locale=(d.get("locale") or "").strip(),
                    segment=(d.get("segment") or "").strip(),
                )
            )
    return rows


def _read_clip_durations_ms(tsv_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for d in r:
            clip = (d.get("clip") or "").strip()
            ms = (d.get("duration[ms]") or "").strip()
            if not clip or not ms:
                continue
            try:
                out[clip] = int(ms)
            except Exception:
                continue
    return out


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int] | None:
    import soundfile as sf

    try:
        if path.suffix.lower() in _MP3_UNSAFE_SUFFIXES:
            with _SF_READ_LOCK:
                wav, sr = sf.read(str(path), always_2d=True)
        else:
            wav, sr = sf.read(str(path), always_2d=True)
        x = wav[:, 0].astype(np.float32, copy=False).reshape(-1)
        return x, int(sr)
    except Exception:
        return None


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32, copy=False)
    if x.size < 2 or sr <= 0:
        return np.zeros((0,), dtype=np.float32)
    n_new = int(round(x.size * (16000.0 / float(sr))))
    n_new = max(1, n_new)
    t_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False, dtype=np.float64)
    t_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float64)
    y = np.interp(t_new, t_old, x.astype(np.float64)).astype(np.float32)
    return y


def _peak_normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x if peak <= 1e-8 else (x / peak).astype(np.float32, copy=False)


def _basic_quality_metrics_16k(x16: np.ndarray) -> dict[str, float]:
    """Non-intrusive per-clip heuristics for filtering dataset 'chłam'."""
    x = np.nan_to_num(np.asarray(x16, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(x.size)
    if n <= 0:
        return {
            "rms_dbfs": -200.0,
            "peak_dbfs": -200.0,
            "clip_frac": 0.0,
            "dc_offset": 0.0,
            "silence_frac": 1.0,
            "centroid_hz": 0.0,
            "bandwidth_hz": 0.0,
            "hf_ratio": 0.0,
            "snr_est_db": -200.0,
            "noise_floor_dbfs": -200.0,
            "spectral_flatness": 1.0,
            "low_mid_ratio": 0.0,
            "high_band_ratio": 0.0,
            "muffled_index": 0.0,
        }
    absx = np.abs(x)
    peak = float(absx.max())
    rms = float(np.sqrt(np.mean(x * x)))
    rms_dbfs = 20.0 * math.log10(max(1e-12, rms))
    peak_dbfs = 20.0 * math.log10(max(1e-12, peak))
    clip_frac = float(np.mean(absx >= 0.999))  # expects float audio in [-1,1]
    dc_offset = float(np.mean(x))
    silence_frac = float(np.mean(absx < 1e-4))

    # Single-shot spectrum features (fast, crude, robust enough for filtering).
    win = np.hanning(n).astype(np.float32, copy=False)
    spec = np.fft.rfft((x * win).astype(np.float64), n=n)
    mag = np.abs(spec).astype(np.float64)
    mag_sum = float(np.sum(mag)) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / 16000.0).astype(np.float64)
    centroid = float(np.sum(freqs * mag) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))
    # HF energy ratio above 4 kHz
    e = (mag * mag)
    e_sum = float(np.sum(e)) + 1e-12
    hf_ratio = float(np.sum(e[freqs >= 4000.0]) / e_sum)

    # Band ratios for \"muffling\" (telephone / heavy low-pass).
    low_band = float(np.sum(e[(freqs >= 80.0) & (freqs < 1000.0)]))
    mid_band = float(np.sum(e[(freqs >= 1000.0) & (freqs < 4000.0)]))
    high_band = float(np.sum(e[(freqs >= 4000.0) & (freqs < 7600.0)]))
    low_mid_ratio = float((low_band + mid_band) / e_sum)
    high_band_ratio = float(high_band / e_sum)
    muffled_index = float(math.log10((low_band + 1e-12) / (high_band + 1e-12)))

    # Spectral flatness (0..1): high ~ noise-like.
    m_use = mag[(freqs >= 80.0) & (freqs < 7600.0)]
    if m_use.size < 8:
        spectral_flatness = 1.0
    else:
        m_pos = np.maximum(m_use, 1e-12)
        gm = float(np.exp(np.mean(np.log(m_pos))))
        am = float(np.mean(m_pos))
        spectral_flatness = float(gm / (am + 1e-12))

    # SNR-ish estimate from short-time RMS percentiles (no external VAD).
    frame = int(min(2048, max(512, n // 32)))
    hop = max(1, frame // 4)
    if n < frame + hop:
        frame_lin = np.array([float(rms)], dtype=np.float64)
    else:
        n_frames = 1 + (n - frame) // hop
        idx = (np.arange(n_frames, dtype=np.int64)[:, None] * hop) + np.arange(frame, dtype=np.int64)
        frames = x[idx]
        frame_lin = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float64)

    # Ignore mostly-silent frames; otherwise noise_floor dives to -inf and SNR explodes.
    mask = (frame_lin * frame_lin) > 1e-8
    if int(np.count_nonzero(mask)) < max(3, frame_lin.size // 8):
        mask = np.ones_like(frame_lin, dtype=bool)

    frame_use = frame_lin[mask]

    # Robust SNR in *linear* energy domain (log-percentiles exaggerate outliers).
    e_lin = (frame_use * frame_use).astype(np.float64)
    e_sorted = np.sort(e_lin)
    q = lambda p: float(np.quantile(e_sorted, p))

    noise_e = float(max(q(0.20), 1e-16))
    speech_e = float(max(q(0.90), noise_e * 4.0))
    snr_est_db = float(10.0 * math.log10((speech_e + 1e-16) / (noise_e + 1e-16)))
    snr_est_db = float(np.clip(snr_est_db, 0.0, 45.0))

    noise_floor_dbfs = float(10.0 * math.log10(max(noise_e, 1e-16)))
    speech_level_dbfs = float(10.0 * math.log10(max(speech_e, 1e-16)))
    noise_floor_dbfs = float(np.clip(noise_floor_dbfs, -120.0, -10.0))
    speech_level_dbfs = float(np.clip(speech_level_dbfs, -120.0, 0.0))

    return {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "clip_frac": clip_frac,
        "dc_offset": dc_offset,
        "silence_frac": silence_frac,
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "hf_ratio": hf_ratio,
        "snr_est_db": snr_est_db,
        "noise_floor_dbfs": noise_floor_dbfs,
        "spectral_flatness": spectral_flatness,
        "low_mid_ratio": low_mid_ratio,
        "high_band_ratio": high_band_ratio,
        "muffled_index": muffled_index,
    }


def _badness_score(
    qm: dict[str, float],
    *,
    thr_silence: float,
    thr_clip: float,
    thr_rms: float,
    thr_snr: float,
    thr_flat: float,
    thr_muffled: float,
    w_silence: float,
    w_clip: float,
    w_rms: float,
    w_snr: float,
    w_flat: float,
    w_muffled: float,
) -> tuple[float, list[str]]:
    flags: list[str] = []
    score = 0.0

    if qm["silence_frac"] > thr_silence:
        score += w_silence * (qm["silence_frac"] - thr_silence) / max(1e-6, 1.0 - thr_silence)
        flags.append("too_much_silence")
    if qm["clip_frac"] > thr_clip:
        score += w_clip * (qm["clip_frac"] / max(1e-6, thr_clip))
        flags.append("clipping")
    if qm["rms_dbfs"] < thr_rms:
        score += w_rms * ((thr_rms - qm["rms_dbfs"]) / max(1e-6, abs(thr_rms)))
        flags.append("too_quiet")
    if qm["snr_est_db"] < thr_snr:
        score += w_snr * ((thr_snr - qm["snr_est_db"]) / max(1e-6, thr_snr))
        flags.append("low_snr")
    if qm["spectral_flatness"] > thr_flat:
        score += w_flat * ((qm["spectral_flatness"] - thr_flat) / max(1e-6, 1.0 - thr_flat))
        flags.append("noise_like_spectrum")
    if qm["muffled_index"] > thr_muffled:
        score += w_muffled * ((qm["muffled_index"] - thr_muffled) / max(1e-6, thr_muffled))
        flags.append("muffled")

    return float(score), flags


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/cv-corpus", help="Root containing locale subdirs (e.g. pl/)")
    p.add_argument("--locale", type=str, default="pl", help="Locale subdir to process (e.g. pl)")
    p.add_argument("--split", type=str, default="validated.tsv", help="TSV filename inside locale dir")
    p.add_argument("--out-dir", type=str, default="preprocessing", help="Where to write outputs")
    p.add_argument("--limit", type=int, default=0, help="Process at most N clips (0 = all)")
    p.add_argument("--max-seconds", type=float, default=0.0, help="If >0, crop each clip to first N seconds before metrics")
    p.add_argument("--bad-min-score", type=float, default=2.0, help="Write rows with badness_score >= this to *_bad.tsv")
    p.add_argument("--thr-silence", type=float, default=0.75)
    p.add_argument("--thr-clip", type=float, default=1e-3)
    p.add_argument("--thr-rms-dbfs", type=float, default=-42.0)
    p.add_argument("--thr-snr-db", type=float, default=18.0)
    p.add_argument("--thr-flatness", type=float, default=0.35)
    p.add_argument("--thr-muffled", type=float, default=2.25)
    p.add_argument("--w-silence", type=float, default=3.0)
    p.add_argument("--w-clip", type=float, default=8.0)
    p.add_argument("--w-rms", type=float, default=2.0)
    p.add_argument("--w-snr", type=float, default=3.0)
    p.add_argument("--w-flat", type=float, default=4.0)
    p.add_argument("--w-muffled", type=float, default=3.0)
    p.add_argument("--log-every", type=int, default=200, help="Print progress every N processed clips")
    p.add_argument("--resume", action="store_true", help="If output TSV exists, skip already-processed clips")
    args = p.parse_args(argv)

    data_root = Path(args.data_dir).expanduser().resolve()
    loc_dir = (data_root / args.locale).resolve()
    clips_dir = loc_dir / "clips"
    split_tsv = loc_dir / args.split
    durations_tsv = loc_dir / "clip_durations.tsv"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_tsv.is_file():
        raise SystemExit(f"missing split TSV: {split_tsv}")
    if not clips_dir.is_dir():
        raise SystemExit(f"missing clips dir: {clips_dir}")

    rows = _read_validated_tsv(split_tsv)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    durations_ms = _read_clip_durations_ms(durations_tsv) if durations_tsv.is_file() else {}

    t0 = time.time()
    out_tsv = out_dir / f"cv_{args.locale}_metrics.tsv"
    bad_tsv = out_dir / f"cv_{args.locale}_bad.tsv"
    out_json = out_dir / f"cv_{args.locale}_summary.json"
    metrics_hdr = (
        "i\tclip\tabs_path\tduration_s\tsr_in\tcrop_s\t"
        "rms_dbfs\tpeak_dbfs\tclip_frac\tdc_offset\tsilence_frac\tcentroid_hz\tbandwidth_hz\thf_ratio\t"
        "snr_est_db\tnoise_floor_dbfs\tspectral_flatness\tlow_mid_ratio\thigh_band_ratio\tmuffled_index\t"
        "badness_score\tbad_flags\t"
        "stoi\tpesq_wb\tvisqol_moslqo\n"
    )
    bad_hdr = (
        "i\tclip\tabs_path\tbadness_score\tbad_flags\t"
        "rms_dbfs\tsnr_est_db\tspectral_flatness\tmuffled_index\tsilence_frac\tclip_frac\n"
    )
    print(f"[cv:{args.locale}] start rows={len(rows)}  mode=dataset_quality", flush=True)

    done: set[str] = set()
    if args.resume and out_tsv.is_file():
        try:
            with out_tsv.open("r", encoding="utf-8") as rf:
                hdr = rf.readline()
                if hdr and hdr.strip() != metrics_hdr.strip():
                    raise SystemExit(
                        f"Refusing --resume: header mismatch in {out_tsv.name}. "
                        f"Delete it or rerun without --resume."
                    )
                if hdr:
                    for line in rf:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 2:
                            done.add(parts[1])
        except Exception:
            done = set()

    # Corpus demographics counts (from TSV, independent of missing audio).
    gender_counts: dict[str, int] = {}
    age_counts: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    missing_audio = 0
    ok_audio = 0
    processed = 0

    def bump(d: dict[str, int], k: str) -> None:
        kk = k if k else "unspecified"
        d[kk] = d.get(kk, 0) + 1

    mode = "a" if (args.resume and out_tsv.is_file()) else "w"
    bad_mode = "a" if (args.resume and bad_tsv.is_file()) else "w"
    n_bad_written = 0
    with out_tsv.open(mode, encoding="utf-8", newline="") as f, bad_tsv.open(bad_mode, encoding="utf-8", newline="") as bf:
        if mode == "w":
            f.write(metrics_hdr)
        if bad_mode == "w":
            bf.write(bad_hdr)
        for i, r in enumerate(rows):
            bump(gender_counts, r.gender)
            bump(age_counts, r.age)
            bump(speaker_counts, r.client_id)

            clip = r.rel_path
            if clip in done:
                continue
            audio_path = clips_dir / clip
            dur_s = float(durations_ms[clip]) / 1000.0 if clip in durations_ms else None
            crop_s = float(args.max_seconds) if float(args.max_seconds) > 0 else (dur_s or 0.0)

            loaded = _load_audio_mono(audio_path)
            if loaded is None:
                missing_audio += 1
                na = "\t".join(["na"] * 22)
                f.write(f"{i}\t{clip}\t{audio_path}\t{dur_s if dur_s is not None else 'na'}\tna\t{crop_s:.3f}\t{na}\n")
                processed += 1
                continue

            x, sr = loaded
            if float(args.max_seconds) > 0 and sr > 0:
                n = int(max(1.0, float(args.max_seconds)) * float(sr))
                x = x[:n]
            x16 = _resample_to_16k(x, sr)
            qm = _basic_quality_metrics_16k(x16)
            x16n = _peak_normalize(x16)
            # self-reference: reference == estimate
            stoi = stoi_16k(x16n, x16n)
            pesq = pesq_wb_16k(x16n, x16n)
            visq = visqol_speech_16k(x16n, x16n)
            bad, flags = _badness_score(
                qm,
                thr_silence=float(args.thr_silence),
                thr_clip=float(args.thr_clip),
                thr_rms=float(args.thr_rms_dbfs),
                thr_snr=float(args.thr_snr_db),
                thr_flat=float(args.thr_flatness),
                thr_muffled=float(args.thr_muffled),
                w_silence=float(args.w_silence),
                w_clip=float(args.w_clip),
                w_rms=float(args.w_rms),
                w_snr=float(args.w_snr),
                w_flat=float(args.w_flat),
                w_muffled=float(args.w_muffled),
            )
            flag_s = ",".join(flags) if flags else ""
            ok_audio += 1
            f.write(
                f"{i}\t{r.rel_path}\t{audio_path}\t{dur_s if dur_s is not None else 'na'}\t{sr}\t{crop_s:.3f}\t"
                f"{qm['rms_dbfs']:.6f}\t{qm['peak_dbfs']:.6f}\t{qm['clip_frac']:.6f}\t{qm['dc_offset']:.6f}\t{qm['silence_frac']:.6f}\t"
                f"{qm['centroid_hz']:.3f}\t{qm['bandwidth_hz']:.3f}\t{qm['hf_ratio']:.6f}\t"
                f"{qm['snr_est_db']:.6f}\t{qm['noise_floor_dbfs']:.6f}\t{qm['spectral_flatness']:.6f}\t"
                f"{qm['low_mid_ratio']:.6f}\t{qm['high_band_ratio']:.6f}\t{qm['muffled_index']:.6f}\t"
                f"{bad:.6f}\t{flag_s}\t"
                f"{'na' if stoi is None else f'{stoi:.6f}'}\t"
                f"{'na' if pesq is None else f'{pesq:.6f}'}\t"
                f"{'na' if visq is None else f'{visq:.6f}'}\n"
            )
            if bad >= float(args.bad_min_score):
                bf.write(
                    f"{i}\t{r.rel_path}\t{audio_path}\t{bad:.6f}\t{flag_s}\t"
                    f"{qm['rms_dbfs']:.6f}\t{qm['snr_est_db']:.6f}\t{qm['spectral_flatness']:.6f}\t"
                    f"{qm['muffled_index']:.6f}\t{qm['silence_frac']:.6f}\t{qm['clip_frac']:.6f}\n"
                )
                n_bad_written += 1
            processed += 1
            if int(args.log_every) > 0 and processed % int(args.log_every) == 0:
                dt = max(1e-6, time.time() - t0)
                rate = processed / dt
                remain = max(0, len(rows) - processed)
                eta_s = remain / max(1e-9, rate)
                print(f"[cv:{args.locale}] processed={processed}/{len(rows)}  rate={rate:.2f} clip/s  eta={eta_s/60.0:.1f} min", flush=True)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                    bf.flush()
                    os.fsync(bf.fileno())
                except Exception:
                    pass

    # Summary stats.
    dur_total_s = sum(float(ms) for ms in durations_ms.values()) / 1000.0 if durations_ms else None
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_root),
        "locale": args.locale,
        "split_tsv": str(split_tsv),
        "clips_dir": str(clips_dir),
        "mode": "dataset_quality",
        "thresholds": {
            "bad_min_score": float(args.bad_min_score),
            "silence": float(args.thr_silence),
            "clip": float(args.thr_clip),
            "rms_dbfs": float(args.thr_rms_dbfs),
            "snr_db": float(args.thr_snr_db),
            "flatness": float(args.thr_flatness),
            "muffled_index": float(args.thr_muffled),
        },
        "weights": {
            "silence": float(args.w_silence),
            "clip": float(args.w_clip),
            "rms": float(args.w_rms),
            "snr": float(args.w_snr),
            "flat": float(args.w_flat),
            "muffled": float(args.w_muffled),
        },
        "outputs": {"metrics_tsv": str(out_tsv), "bad_tsv": str(bad_tsv)},
        "n_bad_written": int(n_bad_written),
        "n_rows": int(len(rows)),
        "n_speakers": int(len(speaker_counts)),
        "missing_audio": int(missing_audio),
        "ok_audio": int(ok_audio),
        "duration_total_hours_from_clip_durations": None if dur_total_s is None else dur_total_s / 3600.0,
        "gender_counts": dict(sorted(gender_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "age_counts": dict(sorted(age_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "elapsed_s": float(time.time() - t0),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

