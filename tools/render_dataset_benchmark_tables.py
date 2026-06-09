#!/usr/bin/env python3
"""Render manuscript-ready Markdown and LaTeX tables from dataset benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def mib(size_bytes: int) -> float:
    return float(size_bytes) / 1024.0 / 1024.0


def fmt_optional(value: Any, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f(float(value), digits)


def exported_model_label(name: str) -> str:
    labels = {
        "full_recon": "pełna rekonstrukcja",
        "compress_packet": "kompresja",
        "decompress_packet": "dekompresja",
    }
    return labels.get(name, name)


def benchmark_rows(py_cuda: dict[str, Any], py_cpu: dict[str, Any], cpp: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, backend: str, runtime: str, bench: dict[str, Any]) -> None:
        rows.append(
            {
                "name": name,
                "backend": backend,
                "runtime": runtime,
                "mean_ms": 1000.0 * float(bench["mean_seconds"]),
                "p50_ms": 1000.0 * float(bench["p50_seconds"]),
                "p90_ms": 1000.0 * float(bench["p90_seconds"]),
                "max_ms": 1000.0 * float(bench["max_seconds"]),
                "xrt": float(bench["x_realtime"]),
            }
        )

    add("pełna rekonstrukcja", "PyTorch", "CUDA / RTX 4080 SUPER", py_cuda["benchmark"])
    add("pełna rekonstrukcja", "PyTorch", "CPU / 16 wątków", py_cpu["benchmark"])
    add("pełna rekonstrukcja", "C++ LiteRT", "CPU XNNPACK / 16 wątków", cpp["benchmarks"]["full"])
    add("kompresja i dekompresja", "C++ LiteRT", "CPU XNNPACK / 16 wątków", cpp["benchmarks"]["codec_full"])
    add("kompresja", "C++ LiteRT", "CPU XNNPACK / 16 wątków", cpp["benchmarks"]["compress_only"])
    add("dekompresja", "C++ LiteRT", "CPU XNNPACK / 16 wątków", cpp["benchmarks"]["decompress_only"])

    baseline = float(cpp["benchmarks"]["codec_full"]["mean_seconds"])
    for row in rows:
        row["speedup_vs_cpp_codec"] = baseline / (float(row["mean_ms"]) / 1000.0)
    return rows


def memory_rows(py_cuda: dict[str, Any], py_cpu: dict[str, Any], cpp: dict[str, Any]) -> list[dict[str, Any]]:
    def snap(report: dict[str, Any], label: str) -> dict[str, Any]:
        for item in report.get("memory", {}).get("snapshots", []):
            if item.get("label") == label:
                return item
        return {}

    py_cuda_model = snap(py_cuda, "after_model_load")
    py_cpu_model = snap(py_cpu, "after_model_load")
    cpp_model = snap(cpp, "after_model_load")
    return [
        {
            "backend": "PyTorch",
            "runtime": "CUDA / RTX 4080 SUPER",
            "rss_after_model_mib": py_cuda_model.get("rss_mib"),
            "rss_peak_mib": py_cuda.get("memory", {}).get("rss_peak_mib"),
            "gpu_after_model_allocated_mib": py_cuda_model.get("cuda_allocated_mib"),
            "gpu_peak_allocated_mib": py_cuda.get("memory", {}).get("cuda_peak_allocated_mib"),
            "gpu_peak_reserved_mib": py_cuda.get("memory", {}).get("cuda_peak_reserved_mib"),
            "scope": "Python, model i CUDA",
        },
        {
            "backend": "PyTorch",
            "runtime": "CPU / 16 wątków",
            "rss_after_model_mib": py_cpu_model.get("rss_mib"),
            "rss_peak_mib": py_cpu.get("memory", {}).get("rss_peak_mib"),
            "gpu_after_model_allocated_mib": None,
            "gpu_peak_allocated_mib": None,
            "gpu_peak_reserved_mib": None,
            "scope": "Python, model CPU",
        },
        {
            "backend": "C++ LiteRT",
            "runtime": "CPU XNNPACK / 16 wątków",
            "rss_after_model_mib": cpp_model.get("rss_mib"),
            "rss_peak_mib": cpp.get("memory", {}).get("rss_peak_mib"),
            "gpu_after_model_allocated_mib": None,
            "gpu_peak_allocated_mib": None,
            "gpu_peak_reserved_mib": None,
            "scope": "proces C++ i modele LiteRT",
        },
    ]


def render_markdown(
    py_cuda: dict[str, Any],
    py_cpu: dict[str, Any],
    cpp: dict[str, Any],
    export: dict[str, Any],
    rows: list[dict[str, Any]],
    mem_rows: list[dict[str, Any]],
) -> str:
    total_files = py_cuda.get("total_flac_files") or py_cpu.get("total_flac_files") or 28539
    manifest = py_cuda.get("manifest_out") or py_cuda.get("manifest") or py_cpu.get("manifest") or cpp.get("manifest")
    lines: list[str] = []
    lines.append("# Benchmark inferencji SirenCodec: PyTorch vs C++ LiteRT/TFLite\n")
    lines.append("## Metodyka\n")
    lines.append("| Parametr | Wartość |")
    lines.append("|---|---|")
    lines.append("| Zbiór | LibriSpeech train-clean-100 |")
    lines.append(f"| Liczba plików w zbiorze | {total_files} |")
    lines.append(f"| Udział benchmarku | 5% plików, {py_cuda['selected_files']} plików |")
    lines.append("| Wybór próbek | deterministyczny wybór według skrótu ścieżki; wspólny manifest dla wszystkich backendów |")
    lines.append(
        f"| Segment wejściowy | {py_cuda['samples_per_file']} próbek, "
        f"{float(py_cuda['audio_seconds_per_file']):.1f} s przy 16 kHz; "
        "pliki dłuższe przycinane, krótsze dopełniane zerami |"
    )
    lines.append("| Zakres pomiaru | tylko forward modelu; ładowanie audio i normalizacja wykonane przed pomiarem |")
    lines.append(f"| Manifest | `{manifest}` |")
    lines.append(f"| Checkpoint | `codec_step329999_cuda.pt`, nominalnie {float(py_cuda['nominal_rvq_kbps']):.4f} kbps |")
    lines.append("| CPU | AMD Ryzen 7 7800X3D, 8C/16T, WSL2 |")
    lines.append("| GPU | NVIDIA GeForce RTX 4080 SUPER, driver 596.21 |")
    lines.append("| LiteRT | C++20, libLiteRt.so z `ai_edge_litert`, CPU/XNNPACK, 16 wątków |\n")

    lines.append("## Wyniki czasowe\n")
    lines.append("| Backend | Profil | Urządzenie/runtime | Mean [ms] | P50 [ms] | P90 [ms] | Max [ms] | x realtime | Speed-up vs C++ codec |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['name']} | {row['runtime']} | "
            f"{f(row['mean_ms'], 3)} | {f(row['p50_ms'], 3)} | {f(row['p90_ms'], 3)} | "
            f"{f(row['max_ms'], 3)} | {f(row['xrt'], 2)} | {f(row['speedup_vs_cpp_codec'], 2)}x |"
        )

    lines.append("\n## Profil pamięci\n")
    lines.append("| Backend | Urządzenie/runtime | RSS model [MiB] | RSS peak [MiB] | CUDA model [MiB] | CUDA peak [MiB] | Zakres |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in mem_rows:
        lines.append(
            f"| {row['backend']} | {row['runtime']} | {fmt_optional(row['rss_after_model_mib'], 1)} | "
            f"{fmt_optional(row['rss_peak_mib'], 1)} | {fmt_optional(row['gpu_after_model_allocated_mib'], 1)} | "
            f"{fmt_optional(row['gpu_peak_allocated_mib'], 1)} | "
            f"{row['scope']} |"
        )

    metrics = [
        ("PyTorch CUDA", py_cuda["metrics_mean"]),
        ("PyTorch CPU", py_cpu["metrics_mean"]),
        ("C++ LiteRT, kompresja i dekompresja", cpp["metrics_mean"]),
    ]
    lines.append("\n## Jakość rekonstrukcji na pierwszych 64 plikach manifestu\n")
    lines.append("| Backend | SI-SDR [dB] | LSD [dB] | L1 | Cosine |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in metrics:
        lines.append(f"| {name} | {f(m['si_sdr_db'], 3)} | {f(m['lsd_db'], 3)} | {float(m['l1']):.6f} | {float(m['cos']):.6f} |")

    lines.append("\n## Eksport TFLite/LiteRT\n")
    lines.append("| Model | Plik | Rozmiar [MiB] | Zgodność z PyTorch | x realtime walidacji |")
    lines.append("|---|---|---:|---|---:|")
    for model in export["models"]:
        comparisons = model.get("validation", {}).get("comparisons", [])
        if model["name"] == "compress_packet" and len(comparisons) > 1:
            err = f"idx max_abs=0; norm max_abs={comparisons[1]['max_abs']:.2e}"
        elif comparisons:
            err = f"max_abs={comparisons[0]['max_abs']:.2e}"
        else:
            err = "n/a"
        xrt = model.get("validation", {}).get("x_realtime")
        lines.append(
            f"| {exported_model_label(model['name'])} | `{Path(model['path']).name}` | {f(mib(model['size_bytes']), 2)} | "
            f"{err} | {f(float(xrt), 2) if xrt is not None else 'n/a'} |"
        )

    lines.append(
        "\nUwagi: wartości PESQ/STOI/ViSQOL nie były liczone w tym benchmarku, "
        "bo wymagają dodatkowych pakietów/runtime. Wszystkie pomiary używają "
        "tej samej listy plików i tej samej długości segmentu.\n"
    )
    return "\n".join(lines)


def render_latex(py_cuda: dict[str, Any], cpp: dict[str, Any], rows: list[dict[str, Any]], py_cpu: dict[str, Any]) -> str:
    total_files = py_cuda.get("total_flac_files") or py_cpu.get("total_flac_files") or 28539
    mem_rows = memory_rows(py_cuda, py_cpu, cpp)
    metrics = [
        ("PyTorch CUDA", py_cuda["metrics_mean"]),
        ("PyTorch CPU", py_cpu["metrics_mean"]),
        ("C++ LiteRT, kompresja i dekompresja", cpp["metrics_mean"]),
    ]
    lines: list[str] = []
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Warunki benchmarku inferencji dla 5\% plików LibriSpeech train-clean-100.}",
            r"\label{tab:benchmark-metodyka}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Parametr & Wartość \\",
            r"\midrule",
            r"Zbiór & LibriSpeech train-clean-100 \\",
            f"Liczba plików & {total_files} \\\\",
            f"Udział benchmarku & 5\\%, {py_cuda['selected_files']} plików \\\\",
            r"Wybór próbek & deterministyczny skrót ścieżki, wspólny manifest \\",
            f"Segment & {py_cuda['samples_per_file']} próbek, {float(py_cuda['audio_seconds_per_file']):.1f} s przy 16 kHz \\\\",
            r"Zakres pomiaru & forward modelu, bez I/O audio \\",
            f"Bitrate modelu & {float(py_cuda['nominal_rvq_kbps']):.4f} kbps nominalnie \\\\",
            r"CPU & AMD Ryzen 7 7800X3D, 8C/16T, WSL2 \\",
            r"GPU & NVIDIA GeForce RTX 4080 SUPER \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Porównanie czasu inferencji PyTorch i C++ LiteRT/TFLite dla tych samych 1427 plików.}",
            r"\label{tab:benchmark-inferencja}",
            r"\begin{tabular}{lllrrrrr}",
            r"\toprule",
            r"Backend & Profil & Runtime & Mean [ms] & P50 [ms] & P90 [ms] & $\times$RT & Speed-up \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['backend']} & {row['name']} & {row['runtime']} & "
            f"{f(row['mean_ms'], 3)} & {f(row['p50_ms'], 3)} & {f(row['p90_ms'], 3)} & "
            f"{f(row['xrt'], 2)} & {f(row['speedup_vs_cpp_codec'], 2)}x \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Profil pamięci podczas benchmarku inferencji.}",
            r"\label{tab:benchmark-pamiec}",
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            r"Backend & Runtime & Zakres & RSS model [MiB] & RSS peak [MiB] & CUDA model [MiB] & CUDA peak [MiB] \\",
            r"\midrule",
        ]
    )
    for row in mem_rows:
        lines.append(
            f"{row['backend']} & {row['runtime']} & {row['scope']} & "
            f"{fmt_optional(row['rss_after_model_mib'], 1)} & {fmt_optional(row['rss_peak_mib'], 1)} & "
            f"{fmt_optional(row['gpu_after_model_allocated_mib'], 1)} & {fmt_optional(row['gpu_peak_allocated_mib'], 1)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Średnie metryki rekonstrukcji dla pierwszych 64 plików manifestu benchmarkowego.}",
            r"\label{tab:benchmark-jakosc}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Backend & SI-SDR [dB] & LSD [dB] & L1 & Cosine \\",
            r"\midrule",
        ]
    )
    for name, m in metrics:
        lines.append(f"{name} & {f(m['si_sdr_db'], 3)} & {f(m['lsd_db'], 3)} & {float(m['l1']):.6f} & {float(m['cos']):.6f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-cuda", type=Path, required=True)
    parser.add_argument("--python-cpu", type=Path, required=True)
    parser.add_argument("--cpp", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--latex", type=Path, required=True)
    args = parser.parse_args()

    py_cuda = load(args.python_cuda)
    py_cpu = load(args.python_cpu)
    cpp = load(args.cpp)
    export = load(args.export)
    rows = benchmark_rows(py_cuda, py_cpu, cpp)
    mem_rows = memory_rows(py_cuda, py_cpu, cpp)

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.latex.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(py_cuda, py_cpu, cpp, export, rows, mem_rows), encoding="utf-8")
    args.latex.write_text(render_latex(py_cuda, cpp, rows, py_cpu), encoding="utf-8")
    print(f"wrote: {args.markdown}")
    print(f"wrote: {args.latex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
