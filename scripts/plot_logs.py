from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
import seaborn as sns

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


RUN_ID_WIDTH = 15

PRESENTATION_RUN_ORDER = [
    "20260426_094053",
    "20260426_160128",
    "20260426_211231",
    "20260425_004452",
    "20260426_002013",
    "20260424_023406",
    "20260426_223149",
]

RUN_LABEL_OVERRIDES = {
    "20260426_094053": (
        "Final: full model, balancer OFF",
        "AE -> RVQ -> full model; fast setting, balancer off, stft=1.25, vq=0.25",
    ),
    "20260426_160128": (
        "RVQ ablation: balancer OFF",
        "AE -> RVQ; stronger STFT, small VQ, no gradient balancer",
    ),
    "20260426_211231": (
        "RVQ quality run",
        "AE -> RVQ; high cosine / low L1 short run, balancer off",
    ),
    "20260425_004452": (
        "Full + GAN baseline",
        "AE -> RVQ -> GAN -> full model; gradient balancer and adversarial loss",
    ),
    "20260426_002013": (
        "Full model, light VQ",
        "AE -> RVQ -> full model; gradient balancer, lighter VQ and marginal regularization",
    ),
    "20260424_023406": (
        "RVQ baseline: strong VQ",
        "AE -> RVQ; older baseline with strong VQ pressure and gradient balancer",
    ),
    "20260426_223149": (
        "GAN smoke test",
        "AE -> RVQ -> GAN; short adversarial test before full training",
    ),
    "20260426_090416": (
        "AE only: fast reconstruction",
        "Autoencoder-only reconstruction run, balancer off",
    ),
    "20260426_200126": (
        "AE only: long check",
        "Autoencoder-only check with high throughput and no VQ",
    ),
}

PRESENTATION_METRICS = [
    ("l1", "Waveform L1", "lower is better"),
    ("stft", "STFT loss", "lower is better"),
    ("cos_pct", "Cosine similarity [%]", "higher is better"),
    ("loss_total", "Training objective", "lower is better"),
]

FINAL_METRICS = [
    ("l1", "Waveform L1", "lower is better", True),
    ("stft", "STFT loss", "lower is better", True),
    ("cos_pct", "Cosine similarity [%]", "higher is better", False),
    ("samples_per_s", "Throughput [samples/s]", "higher is better", False),
]


def _read_table(path: Path, *, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, na_values=["-", "na", "NA", "nan", ""])
    string_cols = {
        "phase",
        "balancer",
        "hypothesis",
        "arch_id",
        "verdict",
        "key_finding",
        "next_action",
        "u0",
        "u1",
        "u2",
        "u3",
    }
    for col in df.columns:
        if col in string_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["run"] = path.parent.name[:RUN_ID_WIDTH]
    df["source_file"] = str(path)
    return df


def _load_logs(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/logs.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            df = _read_table(path, sep=",")
        except Exception as exc:
            print(f"[warn] skipped {path}: {exc}")
            continue
        if not df.empty and "step" in df.columns:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_eval_logs(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/log_mlx.tsv")):
        if path.stat().st_size == 0:
            continue
        try:
            df = _read_table(path, sep="\t")
        except Exception as exc:
            print(f"[warn] skipped {path}: {exc}")
            continue
        if not df.empty and "step" in df.columns:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _prepare_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    numeric_cols = [
        "step",
        "progress_pct",
        "loss_total",
        "l1",
        "sisdr_loss",
        "stft",
        "sc",
        "cx",
        "sgrad",
        "cos_pct",
        "ema_cos_pct",
        "grad_norm",
        "vq_loss",
        "marg_ent",
        "ms_per_update",
        "samples_per_s",
        "updates_per_s",
        "epochs_per_hour",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "phase" not in df.columns:
        df["phase"] = "unknown"
    df["run_rows"] = df.groupby("run")["step"].transform("size")
    df["run_last_step"] = df.groupby("run")["step"].transform("max")
    return df


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def _phase_path(phases: pd.Series) -> str:
    ordered: list[str] = []
    for phase in phases.dropna().astype(str):
        if phase and phase not in ordered:
            ordered.append(phase)
    return " -> ".join(ordered)


def _short_phase_name(phase_path: str) -> str:
    if "D/full" in phase_path:
        return "full model"
    if "C/GAN" in phase_path:
        return "GAN stage"
    if "B/RVQ" in phase_path:
        return "RVQ"
    if "A/AE" in phase_path or "ae" in phase_path:
        return "AE only"
    return phase_path or "unknown"


def _auto_run_label(row: pd.Series) -> tuple[str, str]:
    run = str(row["run"])
    phase_path = str(row.get("phase_path", ""))
    model = _short_phase_name(phase_path)
    balancer = str(row.get("balancer", "")).strip()
    balancer_label = "balancer OFF" if balancer == "off" else "grad balancer" if balancer == "grad" else balancer
    stft = _fmt(row.get("stft_weight"), digits=2)
    vq = _fmt(row.get("vq_weight"), digits=2)
    adv = float(row.get("adv_weight") or 0)
    label = f"{model}: {balancer_label}"
    if adv > 0:
        label = f"{model} + GAN: {balancer_label}"
    detail = f"{phase_path}; stft={stft}, vq={vq}, run {run[-6:]}"
    return label, detail


def _build_run_metadata(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = df.sort_values("step").groupby("run", as_index=False).tail(1).copy()
    phase_paths = df.groupby("run")["phase"].apply(_phase_path).rename("phase_path")
    final_rows = final_rows.merge(phase_paths, on="run", how="left")

    labels: list[str] = []
    details: list[str] = []
    for _, row in final_rows.iterrows():
        override = RUN_LABEL_OVERRIDES.get(str(row["run"]))
        if override is None:
            label, detail = _auto_run_label(row)
        else:
            label, detail = override
        labels.append(label)
        details.append(detail)

    final_rows["run_label"] = labels
    final_rows["run_detail"] = details
    final_rows["run_label_wrapped"] = final_rows["run_label"].map(lambda value: "\n".join(textwrap.wrap(value, 28)))
    columns = [
        "run",
        "run_label",
        "run_label_wrapped",
        "run_detail",
        "phase_path",
        "run_rows",
        "run_last_step",
        "step",
        "stft_weight",
        "vq_weight",
        "adv_weight",
        "marginal_weight",
        "balancer",
        "loss_total",
        "l1",
        "stft",
        "cos_pct",
        "vq_loss",
        "samples_per_s",
        "ms_per_update",
    ]
    for col in columns:
        if col not in final_rows.columns:
            final_rows[col] = pd.NA
    return final_rows[columns]


def _attach_run_metadata(df: pd.DataFrame, run_meta: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = ["run", "run_label", "run_label_wrapped", "run_detail", "phase_path"]
    result = df.merge(run_meta[cols], on="run", how="left")
    result["run_label"] = result["run_label"].fillna(result["run"])
    result["run_label_wrapped"] = result["run_label_wrapped"].fillna(result["run_label"])
    return result


def _presentation_runs(df: pd.DataFrame, run_meta: pd.DataFrame, limit: int) -> list[str]:
    available = set(df["run"].dropna().astype(str))
    selected = [run for run in PRESENTATION_RUN_ORDER if run in available]
    candidates = (
        run_meta[run_meta["run"].isin(available)]
        .sort_values(["run_last_step", "run_rows", "run"], ascending=[False, False, False])
        ["run"]
        .tolist()
    )
    for run in candidates:
        if run not in selected and len(selected) < limit:
            selected.append(run)
    return selected[:limit]


def _setup_presentation_theme() -> None:
    sns.set_theme(context="talk", style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3542",
            "axes.labelcolor": "#1f2933",
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "grid.color": "#d7dce2",
            "grid.linewidth": 0.9,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _run_palette(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    colors = sns.color_palette("colorblind", n_colors=max(len(labels), 3))
    return dict(zip(labels, colors))


def _annotate_bars(ax: plt.Axes, values: pd.Series, *, digits: int = 3) -> None:
    if values.empty:
        return
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin if xmax != xmin else 1
    offset = span * 0.015
    for patch, value in zip(ax.patches, values):
        x = patch.get_width()
        ha = "left" if x >= 0 else "right"
        ax.text(
            x + (offset if x >= 0 else -offset),
            patch.get_y() + patch.get_height() / 2,
            _fmt(value, digits=digits),
            va="center",
            ha=ha,
            fontsize=9,
            color="#1f2933",
        )


def _plot_presentation_training(df: pd.DataFrame, out_dir: Path, runs: list[str]) -> None:
    available_metrics = [(col, label, note) for col, label, note in PRESENTATION_METRICS if col in df.columns]
    if not available_metrics or not runs:
        return

    data = df[df["run"].isin(runs)].copy()
    data = data.dropna(subset=["progress_pct"])
    if data.empty:
        return

    label_lookup = data.drop_duplicates("run").set_index("run")["run_label_wrapped"].to_dict()
    labels = [label_lookup[run] for run in runs if run in label_lookup]
    palette = _run_palette(labels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    axes_flat = axes.ravel()

    for ax, (metric, title, note) in zip(axes_flat, available_metrics):
        metric_data = data[["run", "run_label_wrapped", "progress_pct", metric]].dropna()
        if metric_data.empty:
            ax.axis("off")
            continue
        metric_data = metric_data.sort_values(["run", "progress_pct"])
        metric_data["smoothed"] = metric_data.groupby("run")[metric].transform(
            lambda values: values.rolling(window=5, min_periods=1, center=True).median()
        )
        sns.lineplot(
            data=metric_data,
            x="progress_pct",
            y="smoothed",
            hue="run_label_wrapped",
            palette=palette,
            linewidth=2.6,
            alpha=0.95,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{title}\n{note}", fontsize=14)
        ax.set_xlabel("Training progress")
        ax.set_ylabel(title)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(True, axis="y", alpha=0.55)

    for ax in axes_flat[len(available_metrics) :]:
        ax.axis("off")

    handles = [Line2D([0], [0], color=palette[label], lw=3) for label in labels]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Training curves for the key SirenCodec runs", fontsize=21, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.935,
        "Smoothed curves, readable run names, and progress-normalized x-axis for slide comparisons.",
        ha="center",
        fontsize=11,
        color="#52616f",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(out_dir / "presentation_training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_presentation_final_results(run_meta: pd.DataFrame, out_dir: Path, runs: list[str]) -> None:
    data = run_meta[run_meta["run"].isin(runs)].copy()
    available_metrics = [(col, label, note, lower) for col, label, note, lower in FINAL_METRICS if col in data.columns]
    if data.empty or not available_metrics:
        return

    labels = data["run_label_wrapped"].tolist()
    palette = _run_palette(labels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes_flat = axes.ravel()

    for ax, (metric, title, note, lower_is_better) in zip(axes_flat, available_metrics):
        metric_data = data[["run_label_wrapped", metric]].dropna().sort_values(metric, ascending=lower_is_better)
        if metric_data.empty:
            ax.axis("off")
            continue
        colors = [palette[label] for label in metric_data["run_label_wrapped"]]
        ax.barh(metric_data["run_label_wrapped"], metric_data[metric], color=colors, height=0.62)
        ax.invert_yaxis()
        ax.set_title(f"{title}\n{note}", fontsize=14)
        ax.set_xlabel("Final value")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="x", alpha=0.55)
        _annotate_bars(ax, metric_data[metric])

    for ax in axes_flat[len(available_metrics) :]:
        ax.axis("off")

    fig.suptitle("Final training results by experiment", fontsize=21, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.935,
        "Each bar uses a descriptive run name so the slide explains what changed between experiments.",
        ha="center",
        fontsize=11,
        color="#52616f",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    fig.savefig(out_dir / "presentation_final_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_presentation_run_legend(run_meta: pd.DataFrame, out_dir: Path, runs: list[str]) -> None:
    data = run_meta[run_meta["run"].isin(runs)].copy()
    if data.empty:
        return
    order = {run: index for index, run in enumerate(runs)}
    data["order"] = data["run"].map(order)
    data = data.sort_values("order")

    rows = []
    for _, row in data.iterrows():
        rows.append(
            [
                "\n".join(textwrap.wrap(str(row["run_label"]), 24)),
                "\n".join(textwrap.wrap(str(row["run_detail"]), 45)),
                "\n".join(textwrap.wrap(str(row["phase_path"]), 24)),
                _fmt(row["run_last_step"], digits=0),
                _fmt(row["l1"]),
                _fmt(row["cos_pct"], digits=1),
                _fmt(row["stft"]),
            ]
        )

    fig_height = max(5, 1.0 + 0.82 * len(rows))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    ax.set_title("Experiment legend for presentation slides", fontsize=21, fontweight="bold", loc="left", pad=16)
    columns = ["Run name", "What changed", "Phase path", "Final step", "L1", "Cos [%]", "STFT"]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.19, 0.34, 0.17, 0.09, 0.07, 0.07, 0.07],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.2)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dce2")
        if row_idx == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(weight="bold", color="#1f2933")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_dir / "presentation_run_legend.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _select_runs(df: pd.DataFrame, *, limit: int = 10, min_rows: int = 3) -> list[str]:
    summary = (
        df.groupby("run", as_index=False)
        .agg(rows=("step", "size"), last_step=("step", "max"))
        .query("rows >= @min_rows")
        .sort_values(["last_step", "run"], ascending=[False, False])
    )
    return summary.head(limit)["run"].tolist()


def _save_lineplot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    title: str,
    path: Path,
    height: float = 5.2,
    aspect: float = 1.8,
) -> None:
    if x not in df.columns or y not in df.columns:
        return
    data = df[[x, y, hue]].dropna()
    if data.empty:
        return
    g = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        kind="line",
        height=height,
        aspect=aspect,
        linewidth=1.8,
        alpha=0.9,
    )
    g.set_axis_labels(x, y)
    g.fig.suptitle(title, y=1.02)
    g.tight_layout()
    g.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(g.fig)


def _plot_latest_dashboard(df: pd.DataFrame, out_dir: Path, run: str) -> None:
    data = df[df["run"] == run].sort_values("step")
    if data.empty:
        return
    run_title = str(data["run_label"].iloc[0]) if "run_label" in data.columns else run
    metrics = [
        ("loss_total", "Loss total"),
        ("cos_pct", "Cosine [%]"),
        ("l1", "Waveform L1"),
        ("stft", "STFT loss"),
        ("vq_loss", "VQ loss"),
        ("samples_per_s", "Samples/s"),
    ]
    available = [(col, label) for col, label in metrics if col in data.columns and data[col].notna().any()]
    if not available:
        return

    n = len(available)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.5 * nrows), squeeze=False)
    for ax, (col, label) in zip(axes.ravel(), available):
        sns.lineplot(data=data, x="step", y=col, hue="phase", marker="o", ax=ax)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel(col)
        ax.legend(title="phase", loc="best", fontsize=8)
    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")
    fig.suptitle(f"{run_title} - training dashboard", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"dashboard_{run}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_codebook_usage(df: pd.DataFrame, out_dir: Path, runs: list[str]) -> None:
    cols = [c for c in ["u0", "u1", "u2", "u3"] if c in df.columns]
    if not cols:
        return
    label_col = "run_label_wrapped" if "run_label_wrapped" in df.columns else "run"
    x_col = "progress_pct" if "progress_pct" in df.columns else "step"
    id_cols = list(dict.fromkeys(["run", label_col, "step", x_col]))
    data = df[df["run"].isin(runs)][[*id_cols, *cols]].copy()
    for col in cols:
        parsed = data[col].astype(str).str.extract(r"(?P<used>\d+)\s*/\s*(?P<total>\d+)")
        data[f"{col}_frac"] = pd.to_numeric(parsed["used"], errors="coerce") / pd.to_numeric(parsed["total"], errors="coerce")
    frac_cols = [f"{col}_frac" for col in cols]
    long = data.melt(
        id_vars=id_cols,
        value_vars=frac_cols,
        var_name="codebook",
        value_name="used_frac",
    )
    long["codebook"] = long["codebook"].str.replace("_frac", "", regex=False)
    long = long.dropna(subset=["used_frac"])
    if long.empty:
        return
    g = sns.relplot(
        data=long,
        x=x_col,
        y="used_frac",
        hue="codebook",
        col=label_col,
        col_wrap=2,
        kind="line",
        height=3.2,
        aspect=1.55,
        marker="o",
        facet_kws={"sharex": False, "sharey": True},
    )
    g.set_axis_labels("Training progress" if x_col == "progress_pct" else "step", "used codes / K")
    g.set_titles("{col_name}", size=12, weight="bold")
    g.set(ylim=(0, 1.05))
    if x_col == "progress_pct":
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    g.fig.suptitle("RVQ codebook usage in selected runs", y=1.02)
    g.tight_layout()
    g.savefig(out_dir / "codebook_usage_latest_runs.png", dpi=170, bbox_inches="tight")
    g.savefig(out_dir / "presentation_codebook_usage.png", dpi=220, bbox_inches="tight")
    plt.close(g.fig)


def _plot_final_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [m for m in ["loss_total", "cos_pct", "l1", "samples_per_s", "ms_per_update"] if m in df.columns]
    if not metrics:
        return
    final_rows = df.sort_values("step").groupby("run", as_index=False).tail(1)
    final_rows = final_rows[final_rows["run_rows"] >= 2].copy()
    if final_rows.empty:
        return
    final_rows = final_rows.sort_values("run_last_step", ascending=False).head(16)
    label_col = "run_label_wrapped" if "run_label_wrapped" in final_rows.columns else "run"
    long = final_rows.melt(id_vars=[label_col], value_vars=metrics, var_name="metric", value_name="value").dropna()
    if long.empty:
        return
    g = sns.catplot(
        data=long,
        x="value",
        y=label_col,
        col="metric",
        kind="bar",
        col_wrap=2,
        sharex=False,
        height=3.2,
        aspect=1.45,
    )
    g.set_axis_labels("Final value", "Run")
    g.fig.suptitle("Final metrics with readable run names", y=1.02)
    g.tight_layout()
    g.savefig(out_dir / "final_metrics_by_run.png", dpi=170, bbox_inches="tight")
    plt.close(g.fig)


def _plot_presentation_eval_results(eval_df: pd.DataFrame, out_dir: Path, runs: list[str]) -> None:
    if eval_df.empty:
        return
    for col in ["step", "sisdr_db", "lsd_db", "l1", "cos", "idx_bps"]:
        if col in eval_df.columns:
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
    metrics = [
        ("sisdr_db", "SI-SDR [dB]", "higher is better", False),
        ("lsd_db", "LSD [dB]", "lower is better", True),
        ("l1", "Eval L1", "lower is better", True),
        ("cos", "Eval cosine", "higher is better", False),
    ]
    metrics = [(col, title, note, lower) for col, title, note, lower in metrics if col in eval_df.columns]
    if not metrics:
        return

    final_rows = eval_df.sort_values("step").groupby("run", as_index=False).tail(1)
    final_rows = final_rows[final_rows["run"].isin(runs)].copy()
    if final_rows.empty:
        return
    label_col = "run_label_wrapped" if "run_label_wrapped" in final_rows.columns else "run"
    labels = final_rows[label_col].tolist()
    palette = _run_palette(labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes_flat = axes.ravel()
    for ax, (metric, title, note, lower_is_better) in zip(axes_flat, metrics):
        metric_data = final_rows[[label_col, metric]].dropna().sort_values(metric, ascending=lower_is_better)
        if metric_data.empty:
            ax.axis("off")
            continue
        colors = [palette[label] for label in metric_data[label_col]]
        ax.barh(metric_data[label_col], metric_data[metric], color=colors, height=0.62)
        ax.invert_yaxis()
        ax.set_title(f"{title}\n{note}", fontsize=14)
        ax.set_xlabel("Final eval value")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="x", alpha=0.55)
        _annotate_bars(ax, metric_data[metric])

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Evaluation results by experiment", fontsize=21, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.935,
        "Final values from log_mlx.tsv, mapped to the same descriptive run names as the training plots.",
        ha="center",
        fontsize=11,
        color="#52616f",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    fig.savefig(out_dir / "presentation_eval_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_eval_metrics(eval_df: pd.DataFrame, out_dir: Path) -> None:
    if eval_df.empty:
        return
    for col in ["step", "sisdr_db", "pesq_wb", "stoi", "lsd_db", "l1", "cos", "idx_bps"]:
        if col in eval_df.columns:
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
    metrics = [m for m in ["sisdr_db", "lsd_db", "l1", "cos", "idx_bps", "pesq_wb", "stoi"] if m in eval_df.columns]
    label_col = "run_label_wrapped" if "run_label_wrapped" in eval_df.columns else "run"
    long = eval_df.melt(id_vars=[label_col, "step"], value_vars=metrics, var_name="metric", value_name="value").dropna()
    if long.empty:
        return
    g = sns.relplot(
        data=long,
        x="step",
        y="value",
        hue=label_col,
        col="metric",
        kind="line",
        marker="o",
        col_wrap=2,
        facet_kws={"sharex": False, "sharey": False},
        height=3.2,
        aspect=1.45,
    )
    g.set_axis_labels("step", "value")
    g.fig.suptitle("Evaluation metrics from log_mlx.tsv", y=1.02)
    g.tight_layout()
    g.savefig(out_dir / "eval_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(g.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seaborn plots from SirenCodec experiment logs.")
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, default=Path("wykresy"))
    parser.add_argument("--latest-runs", type=int, default=10)
    args = parser.parse_args()

    _setup_presentation_theme()
    args.out.mkdir(parents=True, exist_ok=True)

    logs = _prepare_logs(_load_logs(args.experiments))
    if logs.empty:
        raise SystemExit("No logs.csv files found.")

    run_meta = _build_run_metadata(logs)
    logs = _attach_run_metadata(logs, run_meta)

    logs.to_csv(args.out / "combined_logs.csv", index=False)
    final_rows = logs.sort_values("step").groupby("run", as_index=False).tail(1)
    final_rows.to_csv(args.out / "final_training_metrics.csv", index=False)
    run_meta.to_csv(args.out / "presentation_run_summary.csv", index=False)

    latest_runs = _select_runs(logs, limit=args.latest_runs, min_rows=3)
    latest = logs[logs["run"].isin(latest_runs)].copy()
    selected_presentation_runs = _presentation_runs(logs, run_meta, limit=min(args.latest_runs, 7))

    _save_lineplot(
        latest,
        x="step",
        y="loss_total",
        hue="run_label_wrapped",
        title="Loss total - latest meaningful runs",
        path=args.out / "loss_total_latest_runs.png",
    )
    for metric, title in [
        ("cos_pct", "Cosine similarity [%] - latest meaningful runs"),
        ("l1", "Waveform L1 - latest meaningful runs"),
        ("stft", "STFT loss - latest meaningful runs"),
        ("samples_per_s", "Training throughput - latest meaningful runs"),
        ("ms_per_update", "Update time - latest meaningful runs"),
    ]:
        _save_lineplot(
            latest,
            x="step",
            y=metric,
            hue="run_label_wrapped",
            title=title,
            path=args.out / f"{metric}_latest_runs.png",
        )

    if latest_runs:
        _plot_latest_dashboard(logs, args.out, latest_runs[0])
    _plot_codebook_usage(logs, args.out, latest_runs[:6])
    _plot_final_metrics(logs, args.out)
    _plot_presentation_training(logs, args.out, selected_presentation_runs)
    _plot_presentation_final_results(run_meta, args.out, selected_presentation_runs)
    _plot_presentation_run_legend(run_meta, args.out, selected_presentation_runs)

    eval_logs = _load_eval_logs(args.experiments)
    if not eval_logs.empty:
        eval_logs = _attach_run_metadata(eval_logs, run_meta)
    _plot_eval_metrics(eval_logs, args.out)
    _plot_presentation_eval_results(eval_logs, args.out, selected_presentation_runs)

    print(f"Wrote plots and CSV summaries to {args.out.resolve()}")


if __name__ == "__main__":
    main()
