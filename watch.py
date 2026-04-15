#!/usr/bin/env python3
"""
Real-time training monitor for CODEC-RESEARCHER.

Usage:
    python watch.py              # Watch current training
    python watch.py --log log.tsv  # Specific log file
    python watch.py --summary      # One-shot summary
    python watch.py --plot         # ASCII plot of metrics
    python watch.py --last 20      # Show last N entries
"""
import os, sys, csv, time, argparse
from pathlib import Path
from collections import defaultdict


def read_log(path):
    """Read log.tsv and return list of dicts."""
    if not Path(path).exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def read_results(path):
    """Read results.tsv and return list of dicts."""
    if not Path(path).exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)


def fmt_num(v, decimals=3):
    try:
        return f"{float(v):.{decimals}f}"
    except:
        return str(v)


def pct(v):
    try:
        return f"{float(v)*100:.1f}%"
    except:
        return str(v)


def ascii_plot(values, labels, width=60, height=15):
    """Simple ASCII plot."""
    if not values:
        return "  (no data)"

    nums = [float(v) for v in values if v not in ('', None)]
    if not nums:
        return "  (no numeric data)"

    mn, mx = min(nums), max(nums)
    if mx == mn:
        mx = mn + 1

    lines = []
    # Header
    lines.append(f"  {fmt_num(mx, 2):>8s} ┤")

    for row in range(height, 0, -1):
        threshold = mn + (mx - mn) * row / height
        line = f"  {fmt_num(threshold, 2):>8s} ┤"

        for v in values:
            try:
                val = float(v)
                if val >= threshold:
                    line += "█"
                elif val >= mn + (mx - mn) * (row-1) / height:
                    line += "░"
                else:
                    line += " "
            except:
                line += " "
            if len(line) > width + 12:
                line = line[:width+12]
        lines.append(line)

    lines.append(f"  {fmt_num(mn, 2):>8s} ┤" + " " * width)
    # X-axis labels
    if labels:
        x_axis = f"           "
        for i in [0, len(labels)//2, len(labels)-1]:
            pos = int(i * width / max(len(labels)-1, 1))
            x_axis = x_axis[:pos] + labels[i][:8] + x_axis[pos+8:]
        # Simpler approach: just show count
        lines.append(f"           0{'':>{width-12}} {len(values)-1}")

    return "\n".join(lines)


def live_watch(log_file, interval=2):
    """Continuously monitor training progress."""
    print("=" * 80)
    print("CODEC Training Monitor (live)")
    print("=" * 80)

    last_count = 0
    while True:
        rows = read_log(log_file)
        if not rows:
            print(f"\n  Waiting for {log_file}...", end="")
            time.sleep(interval)
            continue

        if len(rows) == last_count:
            time.sleep(interval)
            continue
        last_count = len(rows)

        # Clear and print status
        print("\n" * 50)
        print_status(rows)
        time.sleep(interval)


def print_status(rows, last_n=None):
    """Print current training status."""
    if last_n:
        rows = rows[-last_n:]

    first = rows[0]
    last = rows[-1]
    total_steps = int(last.get('step', 0))

    # Check if we know total
    total_expected = None
    try:
        # Try to infer from progress
        step = int(last.get('step', 0))
        if step < 50000:
            total_expected = 100000
        elif step < 100000:
            total_expected = 100000
        elif step < 200000:
            total_expected = 200000
        else:
            total_expected = step + 50000
    except:
        pass

    print("\n" + "=" * 80)
    print("CODEC Training Status")
    print("=" * 80)

    # Progress
    if total_expected:
        pct_done = total_steps / total_expected * 100
        bar_len = 40
        filled = int(bar_len * total_steps / total_expected)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n  Progress: [{bar}] {pct_done:.1f}% ({total_steps:,}/{total_expected:,})")
    else:
        print(f"\n  Step: {total_steps:,}")

    # Current metrics
    print(f"\n  Current (step {total_steps:,}):")
    print(f"    Mel loss:    {fmt_num(last.get('loss_mel', 'N/A'))}")
    print(f"    Adv loss:    {fmt_num(last.get('loss_adv', 'N/A'))}")
    print(f"    Commit loss: {fmt_num(last.get('loss_commit', 'N/A'))}")
    print(f"    VQ util:     {pct(last.get('vq_utilization', 'N/A'))}")
    print(f"    Grad norm:   {fmt_num(last.get('grad_norm', 'N/A'))}")
    print(f"    LR:          {fmt_num(last.get('lr', 'N/A'), 6)}")
    print(f"    Total loss:  {fmt_num(last.get('loss_total', 'N/A'))}")

    # Trends
    if len(rows) >= 5:
        recent_mel = [float(r.get('loss_mel', 0)) for r in rows[-5:] if r.get('loss_mel')]
        recent_vq = [float(r.get('vq_utilization', 0)) for r in rows[-5:] if r.get('vq_utilization')]

        if recent_mel and len(recent_mel) >= 2:
            mel_trend = "↓" if recent_mel[-1] < recent_mel[0] else "↑" if recent_mel[-1] > recent_mel[0] else "→"
            print(f"\n  Trends (last {len(rows[-5:])} logs):")
            print(f"    Mel:  {fmt_num(recent_mel[0])} → {fmt_num(recent_mel[-1])} {mel_trend}")
        if recent_vq and len(recent_vq) >= 2:
            vq_trend = "↑" if recent_vq[-1] > recent_vq[0] else "↓" if recent_vq[-1] < recent_vq[0] else "→"
            print(f"    VQ:   {pct(recent_vq[0])} → {pct(recent_vq[-1])} {vq_trend}")

    # Range
    if len(rows) >= 2:
        first_step = int(first.get('step', 0))
        last_step = int(last.get('step', 0))
        span = last_step - first_step
        print(f"\n  Range: step {first_step:,} → {last_step:,} ({len(rows)} log entries)")


def summary(log_file, results_file):
    """One-shot summary."""
    rows = read_log(log_file)
    results = read_results(results_file)

    if rows:
        print_status(rows)

    if results:
        print(f"\n\n{'=' * 80}")
        print(f"Cycle History ({len(results)} cycles)")
        print(f"{'=' * 80}")
        print(f"  {'Cycle':<6} {'Phase':<12} {'Arch':<16} {'Bitrate':<8} {'PESQ':<6} {'Verdict':<12}")
        print(f"  {'─' * 60}")
        for r in results[-15:]:  # Last 15 cycles
            print(f"  {r.get('cycle', '?'):<6} {r.get('phase', '?'):<12} "
                  f"{r.get('arch_id', '?'):<16} {r.get('bitrate_bps', '?'):<8} "
                  f"{r.get('pesq_est', '?'):<6} {r.get('verdict', '?'):<12}")

    # Checkpoints
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("codec_step*.pt"))
        print(f"\n\n{'=' * 80}")
        print(f"Checkpoints ({len(ckpts)} files)")
        print(f"{'=' * 80}")
        total_size = sum(c.stat().st_size for c in ckpts)
        print(f"  Total size: {total_size / 1e9:.2f} GB")
        for c in ckpts[-5:]:
            size_mb = c.stat().st_size / 1e6
            print(f"  {c.name:<25} {size_mb:.0f} MB")


def plot_metrics(log_file, metric='loss_mel', width=70):
    """ASCII plot of a specific metric."""
    rows = read_log(log_file)
    if not rows:
        print(f"No data in {log_file}")
        return

    steps = [r.get('step', '') for r in rows]
    values = [r.get(metric, '') for r in rows]

    print(f"\n{'=' * 80}")
    print(f"  {metric} over {len(values)} log entries")
    print(f"{'=' * 80}")
    print(ascii_plot(values, steps, width=width))
    print(f"  {'':>8} └{'─' * width}> step")


def main():
    parser = argparse.ArgumentParser(description="CODEC Training Monitor")
    parser.add_argument("--log", type=str, default="log.tsv")
    parser.add_argument("--results", type=str, default="results.tsv")
    parser.add_argument("--live", action="store_true", help="Watch mode (auto-refresh)")
    parser.add_argument("--summary", action="store_true", help="One-shot summary")
    parser.add_argument("--plot", type=str, nargs='*', help="Plot metrics (default: loss_mel)")
    parser.add_argument("--last", type=int, default=None, help="Show last N entries")
    parser.add_argument("--interval", type=int, default=2, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if args.live:
        live_watch(args.log, args.interval)
    elif args.summary:
        summary(args.log, args.results)
    elif args.plot is not None:
        metrics = args.plot if args.plot else ['loss_mel']
        for m in metrics:
            plot_metrics(args.log, m)
    else:
        # Default: show status
        rows = read_log(args.log)
        if not rows:
            print(f"Waiting for training to start ({args.log} not found)...")
            import time
            for i in range(30):
                time.sleep(1)
                rows = read_log(args.log)
                if rows:
                    break
                print(f"  Still waiting... ({i+1}s)", end="\r")
            if not rows:
                print("\nNo training data found after 30s.")
                print("Start training with: python train_pipeline.py")
                return
        print_status(rows, args.last)


if __name__ == "__main__":
    main()
