"""Generate publication figures from resultados_benchmark_d40.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import matplotlib
except ImportError as exc:
    raise SystemExit(
        "Matplotlib is required; run this script with a Python environment "
        "that has matplotlib installed."
    ) from exc

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, ScalarFormatter


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "resultados_benchmark_d40.json"


def _read_results() -> list[dict[str, Any]]:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    instances = payload["instances"]
    if not instances:
        raise ValueError(f"No benchmark instances found in {RESULTS}.")
    return sorted(instances, key=lambda row: row["dimension"])


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _make_time_figure(instances: list[dict[str, Any]]) -> None:
    dimensions = [row["dimension"] for row in instances]
    tensor_times = [row["tensor_crt_wall_time_seconds"] for row in instances]
    measured = [
        (
            row["dimension"],
            row["polyhedral_baseline"]["wall_time_seconds"],
        )
        for row in instances
        if row["polyhedral_baseline"]["status"] == "completado"
    ]
    timed_out = [
        row["dimension"]
        for row in instances
        if row["polyhedral_baseline"]["status"].startswith("TIMEOUT")
    ]
    cutoff = max(
        row["polyhedral_baseline"]["timeout_seconds"] for row in instances
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(
        dimensions,
        tensor_times,
        color="#1769aa",
        linewidth=2.1,
        marker="o",
        markersize=5,
        label="Tensor-CRT (exact)",
        zorder=3,
    )
    if measured:
        ax.plot(
            [point[0] for point in measured],
            [point[1] for point in measured],
            color="#d55e00",
            linewidth=1.8,
            marker="s",
            markersize=5,
            label="Canonical polyhedral enumeration",
            zorder=3,
        )
    ax.axhline(
        cutoff,
        color="#d55e00",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label=f"Polyhedral timeout cutoff ({cutoff:g} s)",
    )
    if timed_out:
        ax.scatter(
            timed_out,
            [cutoff] * len(timed_out),
            marker="^",
            s=62,
            facecolors="white",
            edgecolors="#d55e00",
            linewidths=1.6,
            label="Censored: time $>15$ s",
            zorder=4,
        )
        ax.annotate(
            "$>15$ s",
            xy=(timed_out[0], cutoff),
            xytext=(timed_out[0] + 2.2, cutoff * 1.45),
            arrowprops={"arrowstyle": "->", "color": "#d55e00", "lw": 0.9},
            color="#9b3b00",
            fontsize=8,
        )

    ax.set_yscale("log")
    ax.set_xlim(min(dimensions) - 1, max(dimensions) + 1)
    ax.set_ylim(0.008, max(max(tensor_times) * 1.35, cutoff * 2.8))
    ax.set_xticks(dimensions)
    ax.set_xlabel(r"Dimension $d=kn$")
    ax.set_ylabel("Wall-clock time (seconds, logarithmic scale)")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(which="major", axis="y", alpha=0.35)
    ax.grid(which="minor", axis="y", alpha=0.12)
    ax.text(
        0.02,
        0.97,
        r"Contraction bound: $O(k^2 n^3 (kn+1)^n \log M)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.96)

    for extension in ("pdf", "png"):
        output = HERE / f"fig_tiempos_d40.{extension}"
        options = {"bbox_inches": "tight"}
        if extension == "png":
            options["dpi"] = 300
        fig.savefig(output, **options)
    plt.close(fig)


def _make_memory_figure(instances: list[dict[str, Any]]) -> None:
    dimensions = [row["dimension"] for row in instances]
    simplex_counts = [
        int(row["normalized_volume_H_1"]) for row in instances
    ]
    memory_mb = [
        row["tensor_crt_tracemalloc_peak_mb"] for row in instances
    ]
    prime_counts = [row["crt_prime_count"] for row in instances]

    fig = plt.figure(figsize=(8.8, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(5.0, 1.0))
    ax_count = fig.add_subplot(grid[0])
    ax_memory = ax_count.twinx()
    ax_primes = fig.add_subplot(grid[1], sharex=ax_count)

    count_line = ax_count.plot(
        dimensions,
        simplex_counts,
        color="#6a3d9a",
        marker="o",
        linewidth=2.1,
        markersize=5,
        label=r"Unimodular simplices: $H_{k,n}(1)$",
        zorder=3,
    )[0]
    ax_count.set_yscale("log")
    ax_count.set_ylabel(r"Simplex count $H_{k,n}(1)$ (log scale)", color="#6a3d9a")
    ax_count.tick_params(axis="y", labelcolor="#6a3d9a")
    ax_count.yaxis.set_major_locator(LogLocator(base=10))
    ax_count.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))

    memory_line = ax_memory.plot(
        dimensions,
        memory_mb,
        color="#008b72",
        marker="D",
        markersize=4.5,
        linewidth=1.8,
        linestyle="--",
        label="Tensor-CRT peak tracemalloc",
        zorder=4,
    )[0]
    ax_memory.set_ylabel("Tensor-CRT peak memory (MB)", color="#008b72")
    ax_memory.tick_params(axis="y", labelcolor="#008b72")
    ax_memory.set_ylim(0, max(memory_mb) * 1.28)
    ax_count.grid(which="major", axis="y", alpha=0.35)
    ax_count.grid(which="minor", axis="y", alpha=0.12)
    ax_count.tick_params(axis="x", labelbottom=False)
    ax_count.legend(
        (count_line, memory_line),
        (count_line.get_label(), memory_line.get_label()),
        loc="upper left",
        frameon=True,
        framealpha=0.96,
    )
    ax_count.text(
        0.99,
        0.04,
        "The simplex count grows exponentially; tensor storage does not store simplices.",
        transform=ax_count.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    ax_primes.step(
        dimensions,
        prime_counts,
        where="mid",
        color="#d55e00",
        linewidth=1.8,
    )
    ax_primes.plot(
        dimensions,
        prime_counts,
        linestyle="none",
        marker="o",
        color="#d55e00",
        markersize=4,
    )
    ax_primes.set_yticks((1, 2, 3))
    ax_primes.set_ylim(0.5, 3.5)
    ax_primes.set_ylabel("CRT primes")
    ax_primes.set_xlabel(r"Dimension $d=kn$")
    ax_primes.set_xticks(dimensions)
    ax_primes.grid(axis="y", alpha=0.25)
    ax_primes.grid(axis="x", alpha=0.12)

    for extension in ("pdf", "png"):
        output = HERE / f"fig_memoria_simplices_d40.{extension}"
        options = {"bbox_inches": "tight"}
        if extension == "png":
            options["dpi"] = 300
        fig.savefig(output, **options)
    plt.close(fig)


def main() -> None:
    _style()
    instances = _read_results()
    _make_time_figure(instances)
    _make_memory_figure(instances)
    for name in (
        "fig_tiempos_d40.pdf",
        "fig_tiempos_d40.png",
        "fig_memoria_simplices_d40.pdf",
        "fig_memoria_simplices_d40.png",
    ):
        print(HERE / name)


if __name__ == "__main__":
    main()
