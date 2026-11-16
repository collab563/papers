"""Generate publication figures from the Treewidth-Tensor-CRT result JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados_treewidth_crt.json"
COLORS = {
    "cylinder": "#345995",
    "layered_dag": "#e07a2d",
    "series_parallel": "#2a7f62",
}
LABELS = {
    "cylinder": "Cyclic crowns",
    "layered_dag": "Layered DAGs",
    "series_parallel": "Series-parallel",
}
MARKERS = {2: "o", 4: "s"}


def _save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_scaling_figure(rows: list[dict[str, object]]) -> None:
    figure, (time_axis, memory_axis) = plt.subplots(
        1, 2, figsize=(12.2, 4.7), constrained_layout=True
    )
    for family in ("cylinder", "layered_dag", "series_parallel"):
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["dimension"],
        )
        dimensions = [int(row["dimension"]) for row in family_rows]
        color = COLORS[family]
        time_axis.plot(
            dimensions,
            [float(row["wall_time_seconds"]) for row in family_rows],
            color=color,
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        memory_axis.plot(
            dimensions,
            [float(row["tracemalloc_peak_mb"]) for row in family_rows],
            color=color,
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        for row in family_rows:
            marker = MARKERS.get(int(row["treewidth"]), "D")
            for axis, key in (
                (time_axis, "wall_time_seconds"),
                (memory_axis, "tracemalloc_peak_mb"),
            ):
                axis.scatter(
                    [int(row["dimension"])],
                    [float(row[key])],
                    marker=marker,
                    s=62,
                    facecolor=color if int(row["treewidth"]) == 2 else "white",
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=3,
                )

    time_axis.axhline(
        15,
        color="#9b2226",
        linestyle="--",
        linewidth=1.4,
        label="Historical 15 s timeout threshold",
    )
    time_axis.axvspan(20, 40.5, color="#9b2226", alpha=0.045, zorder=0)
    time_axis.annotate(
        "Historical C$_{k,5}$ triangulation records\n(timeout threshold; not these DAG/SP instances)",
        xy=(30, 15),
        xytext=(21, 12.2),
        fontsize=8,
        color="#7f1d1d",
        arrowprops={"arrowstyle": "->", "color": "#9b2226", "lw": 0.8},
    )
    time_axis.set(
        xlabel="Poset dimension $d$",
        ylabel="Wall time (s)",
        title="Exact CRT computation",
        xlim=(10.5, 41),
        ylim=(0, 16.8),
    )
    memory_axis.set(
        xlabel="Poset dimension $d$",
        ylabel="Peak Python-traced memory (MB)",
        title="Tensor-state memory",
        xlim=(10.5, 41),
    )
    memory_axis.set_yscale("log")
    for axis in (time_axis, memory_axis):
        axis.set_xticks([12, 16, 20, 24, 30, 36, 40])
        axis.grid(True, which="major", linewidth=0.55, alpha=0.35)
        axis.tick_params(axis="x", labelrotation=35)

    family_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[family],
            marker="o",
            linewidth=1.5,
            label=LABELS[family],
        )
        for family in ("cylinder", "layered_dag", "series_parallel")
    ]
    width_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            markerfacecolor="#333333",
            linestyle="None",
            label="treewidth 2",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="s",
            markerfacecolor="white",
            linestyle="None",
            label="treewidth 4",
        ),
    ]
    time_axis.legend(
        handles=family_handles + width_handles,
        loc="upper left",
        frameon=True,
        fontsize=8,
        ncol=2,
    )
    figure.suptitle(
        "Bounded-treewidth order-polytope counting across three families",
        fontsize=13,
    )
    _save(figure, "fig_treewidth_scaling")


def generate_dag_volume_figure(rows: list[dict[str, object]]) -> None:
    dag_rows = sorted(
        (row for row in rows if row["family"] == "layered_dag"),
        key=lambda row: row["dimension"],
    )
    dimensions = [int(row["dimension"]) for row in dag_rows]
    normalized_volumes = [float(int(row["normalized_volume_H_1"])) for row in dag_rows]
    memory = [float(row["tracemalloc_peak_mb"]) for row in dag_rows]

    figure, volume_axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    memory_axis = volume_axis.twinx()
    volume_axis.semilogy(
        dimensions,
        normalized_volumes,
        color=COLORS["layered_dag"],
        marker="o",
        linewidth=2,
        markersize=6,
        label=r"Exact simplex count $H_P(1)$",
    )
    memory_axis.plot(
        dimensions,
        memory,
        color=COLORS["cylinder"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=5,
        label="Tensor-DP peak memory",
    )
    volume_axis.annotate(
        r"$1.42\times 10^{21}$",
        xy=(dimensions[-1], normalized_volumes[-1]),
        xytext=(-120, -5),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        fontsize=9,
    )
    volume_axis.set_ylim(top=3.0e21)
    volume_axis.set(
        xlabel="Poset dimension $d$",
        ylabel=r"Normalized volume / unimodular simplex count $H_P(1)$",
        title="Layered-DAG instances: exact volume grows while bag memory stays small",
    )
    memory_axis.set_ylabel("Peak Python-traced tensor memory (MB)")
    volume_axis.set_xticks(dimensions)
    volume_axis.grid(True, which="both", linewidth=0.55, alpha=0.35)
    handles_a, labels_a = volume_axis.get_legend_handles_labels()
    handles_b, labels_b = memory_axis.get_legend_handles_labels()
    volume_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")
    volume_axis.text(
        0.99,
        0.02,
        "All six layered DAGs have measured treewidth 2.",
        transform=volume_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    _save(figure, "fig_dag_volumes_d40")


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload["instances"]
    if len(rows) != 12:
        raise ValueError(f"Expected 12 benchmark rows; found {len(rows)}.")
    if not any(
        row["family"] == "layered_dag"
        and int(row["dimension"]) == 40
        and int(row["normalized_volume_H_1"]) == 1_423_698_065_794_236_220_560
        for row in rows
    ):
        raise ValueError("The d=40 layered-DAG volume does not match the recorded run.")
    generate_scaling_figure(rows)
    generate_dag_volume_figure(rows)
    print("Generated vector PDF and 300-dpi PNG figures:")
    for stem in ("fig_treewidth_scaling", "fig_dag_volumes_d40"):
        for extension in ("pdf", "png"):
            path = ROOT / f"{stem}.{extension}"
            print(f"  {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
