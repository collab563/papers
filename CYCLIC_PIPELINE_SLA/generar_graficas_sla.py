"""Generate publication figures and the complete SLA comparison table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "resultados_pipeline_sla.json"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _read_results() -> dict[str, Any]:
    with RESULTS_PATH.open(encoding="utf-8") as source:
        results = json.load(source)
    if {row["n"] for row in results["scaling"]} != {3, 4, 5}:
        raise ValueError("El JSON debe incluir escalamiento para n=3, 4 y 5.")
    if len(results["sla_instances"]) != 23:
        raise ValueError("Se esperaban exactamente 23 instancias certificadas.")
    if sum(len(item["thresholds"]) for item in results["sla_instances"]) != 92:
        raise ValueError("Se esperaban exactamente 92 comparaciones SLA.")
    for test in results["sampling_validation"]["linear_extension_descent_chi_square_tests"]:
        if "empirical_descent_counts" not in test:
            raise ValueError("Faltan los histogramas empíricos de descensos.")
    return results


def _save_figure(figure: plt.Figure, stem: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(
            HERE / f"{stem}.{extension}",
            bbox_inches="tight",
            dpi=300 if extension == "png" else None,
        )
    plt.close(figure)


def _figure_scaling(results: dict[str, Any]) -> None:
    rows = results["scaling"]
    figure, (time_axis, memory_axis) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {3: "#0072B2", 4: "#D55E00", 5: "#009E73"}
    for n in (3, 4, 5):
        subset = sorted((row for row in rows if row["n"] == n), key=lambda row: row["k"])
        time_axis.plot(
            [row["k"] for row in subset],
            [row["wall_time_seconds"] for row in subset],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=colors[n],
            label=fr"$n={n}$",
        )
        memory_axis.plot(
            [row["k"] for row in subset],
            [row["tracemalloc_peak_mb"] for row in subset],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            color=colors[n],
            label=fr"$n={n}$ ({3**n} states)",
        )

    time_axis.set_xscale("log")
    time_axis.set_yscale("log")
    time_axis.set_xlabel("Pipeline length $k$")
    time_axis.set_ylabel("Wall-clock time (seconds)")
    time_axis.set_title("(a) Exact schedule counting")
    time_axis.grid(True, which="both", alpha=0.2)
    time_axis.legend(frameon=False)

    memory_axis.set_xscale("log")
    memory_axis.set_xlabel("Pipeline length $k$")
    memory_axis.set_ylabel("Peak traced memory (MiB)")
    memory_axis.set_title("(b) Memory depends on state-space size")
    memory_axis.grid(True, which="major", alpha=0.2)
    memory_axis.legend(frameon=False)
    figure.suptitle("Linear-time transfer computation with fixed perimeter", y=1.02)
    figure.tight_layout()
    _save_figure(figure, "fig_linear_scaling_sla")


def _figure_bounds(results: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    colors = plt.get_cmap("viridis")
    style_data = (
        ("probability_exact", "-", "Exact"),
        ("pitman_chernoff_bound", "--", "Pitman–Chernoff"),
        ("hoeffding_bound", ":", "Hoeffding"),
    )
    for n, axis in zip(range(3, 8), axes_flat):
        instances = [item for item in results["sla_instances"] if item["n"] == n]
        for color_index, instance in enumerate(instances):
            color = colors(color_index / max(1, len(instances) - 1))
            thresholds = sorted(instance["thresholds"], key=lambda item: item["delta"])
            deltas = [item["delta"] for item in thresholds]
            for key, line_style, _ in style_data:
                values = [item[key] for item in thresholds]
                positive = [(delta, value) for delta, value in zip(deltas, values) if value > 0]
                if positive:
                    axis.plot(
                        [point[0] for point in positive],
                        [point[1] for point in positive],
                        linestyle=line_style,
                        color=color,
                        linewidth=0.9,
                        alpha=0.78,
                    )
        axis.set_yscale("log")
        axis.set_title(fr"$n={n}$ ({len(instances)} instances)")
        axis.grid(True, which="both", alpha=0.18)
        axis.set_xticks((0.1, 0.2, 0.3, 0.4))
        axis.set_xlabel("Congestion margin $\\delta$")
        if n in (3, 6):
            axis.set_ylabel("Violation probability / upper bound")
    axes_flat[-1].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color="#444444", linestyle=line_style, linewidth=1.5, label=label)
        for _, line_style, label in style_data
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle("Exact SLA tails and analytical upper bounds", y=1.01)
    figure.tight_layout(rect=(0, 0.05, 1, 0.98))
    _save_figure(figure, "fig_pitman_chernoff_bounds")


def _figure_sampling(results: dict[str, Any]) -> None:
    tests = results["sampling_validation"]["linear_extension_descent_chi_square_tests"]
    figure, axes = plt.subplots(1, len(tests), figsize=(9.0, 3.8), squeeze=False)
    for axis, test in zip(axes[0], tests):
        coefficients = [int(value) for value in test["exact_h_star_coefficients"]]
        total = sum(coefficients)
        theoretical = [value / total for value in coefficients]
        empirical_counts = test["empirical_descent_counts"]
        empirical = [value / test["samples"] for value in empirical_counts]
        descents = list(range(len(coefficients)))
        axis.bar(
            descents,
            empirical,
            width=0.68,
            color="#80A9C6",
            edgecolor="#285D8E",
            linewidth=0.5,
            label="Empirical",
        )
        axis.plot(
            descents,
            theoretical,
            color="#B33C2E",
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=r"Exact $h_s^*/H(1)$",
        )
        axis.set_xlabel(r"Number of descents $s$")
        axis.set_ylabel("Probability")
        axis.set_title(
            fr"$C_{{{test['k']},{test['n']}}}$: $p_{{\chi^2}}={test['p_value']:.3f}$"
        )
        axis.set_xticks(descents)
        axis.grid(axis="y", alpha=0.2)
    figure.legend(
        handles=[
            Patch(facecolor="#80A9C6", edgecolor="#285D8E", label="Empirical"),
            Line2D([0], [0], color="#B33C2E", marker="o", label=r"Exact $h_s^*/H(1)$"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle("Uniform linear-extension sampling: descent distribution", y=1.02)
    figure.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save_figure(figure, "fig_uniform_sampling_chi2")


def _write_sla_table(results: dict[str, Any]) -> None:
    output = HERE / "sla_comparison_table.tex"
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{rrrrrrrrr}",
        r"\caption{All 92 exact and empirical SLA comparisons across the 23 certified instances.}\label{tab:sla-all}\\",
        r"\toprule",
        r"$k$ & $n$ & $D$ & $\delta$ & Threshold & $P_{\rm exact}$ & $P_{\rm PC}$ & $P_{\rm H}$ & Empirical \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{9}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"$k$ & $n$ & $D$ & $\delta$ & Threshold & $P_{\rm exact}$ & $P_{\rm PC}$ & $P_{\rm H}$ & Empirical \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for instance in sorted(results["sla_instances"], key=lambda item: (item["n"], item["k"])):
        for row in sorted(instance["thresholds"], key=lambda item: item["delta"]):
            empirical = row["empirical_frequency"]
            empirical_text = "--" if empirical is None else f"{empirical:.3e}"
            values = (
                instance["k"],
                instance["n"],
                instance["degree"],
                f'{row["delta"]:.2f}',
                row["integer_threshold"],
                f'{row["probability_exact"]:.3e}',
                f'{row["pitman_chernoff_bound"]:.3e}',
                f'{row["hoeffding_bound"]:.3e}',
                empirical_text,
            )
            lines.append(" & ".join(map(str, values)) + r" \\")
    lines.extend([r"\end{longtable}", r"\end{landscape}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scaling_table(results: dict[str, Any]) -> None:
    output = HERE / "scaling_results_table.tex"
    lines = [
        r"\begin{landscape}",
        r"\begin{longtable}{rrrrrrrrr}",
        r"\caption{Complete exact-count scaling benchmark for $m=2$.}\label{tab:scaling}\\",
        r"\toprule",
        r"$k$ & $n$ & Tasks & States & Seconds & Peak MiB & Bits & $\log_{10}L$ & Exact $L$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{9}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"$k$ & $n$ & Tasks & States & Seconds & Peak MiB & Bits & $\log_{10}L$ & Exact $L$ \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{9}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in sorted(results["scaling"], key=lambda item: (item["n"], item["k"])):
        values = (
            row["k"],
            row["n"],
            row["task_count"],
            row["state_count"],
            f'{row["wall_time_seconds"]:.4f}',
            f'{row["tracemalloc_peak_mb"]:.3f}',
            row["count_bit_length"],
            f'{row["log10_count"]:.6f}',
            row["exact_count"],
        )
        lines.append(" & ".join(map(str, values)) + r" \\")
    lines.extend([r"\end{longtable}", r"\end{landscape}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = _read_results()
    _figure_scaling(results)
    _figure_bounds(results)
    _figure_sampling(results)
    _write_scaling_table(results)
    _write_sla_table(results)
    print("Figuras guardadas: fig_linear_scaling_sla, fig_pitman_chernoff_bounds, fig_uniform_sampling_chi2 (PDF y PNG).")
    print(f"Tablas LaTeX guardadas: {HERE / 'scaling_results_table.tex'} y {HERE / 'sla_comparison_table.tex'}")


if __name__ == "__main__":
    main()
