"""Generate the three publication figures from the exact Lee--Yang results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "resultados_lee_yang.json"
mp.mp.dps = 60
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _verify_exact_self_dual_peaks(instances: list[dict]) -> None:
    """Prove chi(1)>=chi(t) for each supplied polynomial by exact algebra."""
    t = sp.symbols("t")
    divisor = sp.Poly((t - 1) ** 2, t)
    for instance in instances:
        H = sp.Poly.from_list(
            [int(value) for value in instance["polynomial_coefficients_descending"]],
            gens=t,
        )
        H1 = H.eval(1)
        H_prime = H.diff()
        H_second = H_prime.diff()
        numerator = sp.Poly(
            t * H_prime.as_expr() * H.as_expr()
            + t**2 * H_second.as_expr() * H.as_expr()
            - t**2 * H_prime.as_expr() ** 2,
            t,
        )
        difference = sp.Poly(
            numerator.eval(1) * H.as_expr() ** 2
            - numerator.as_expr() * H1**2,
            t,
        )
        quotient, remainder = sp.div(difference, divisor)
        if remainder.as_expr() != 0 or not any(quotient.all_coeffs()):
            raise ArithmeticError(
                f"No se verificó el factor autodual para H_{{{instance['k']},{instance['n']}}}."
            )
        if any(coefficient < 0 for coefficient in quotient.all_coeffs()):
            raise ArithmeticError(
                "La positividad exacta de chi(1)-chi(t) falló para "
                f"H_{{{instance['k']},{instance['n']}}}."
            )


def _figure_gap(data: dict) -> None:
    instances = data["instances"]
    grouped: dict[int, list[dict]] = {n: [] for n in range(3, 8)}
    for item in instances:
        grouped[item["n"]].append(item)

    fig, ax = plt.subplots(figsize=(8.1, 4.7), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=2, vmax=8)
    means_n: list[int] = []
    means_gap: list[float] = []
    for n, items in grouped.items():
        if not items:
            continue
        offsets = np.linspace(-0.095, 0.095, len(items))
        gaps = []
        for offset, item in zip(offsets, sorted(items, key=lambda row: row["k"])):
            gap = float(item["spectral"]["d0_distance_to_zero"])
            gaps.append(gap)
            ax.scatter(
                n + offset,
                gap,
                c=[cmap(norm(item["k"]))],
                s=37,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        means_n.append(n)
        means_gap.append(float(np.mean(gaps)))

    log_means = np.log(np.asarray(means_gap))
    slope, intercept = np.polyfit(np.asarray(means_n, dtype=float), log_means, 1)
    fit_y = np.exp(intercept + slope * np.asarray(means_n, dtype=float))
    total = float(np.sum((log_means - log_means.mean()) ** 2))
    r_squared = 1 - float(np.sum((log_means - (intercept + slope * np.asarray(means_n))) ** 2)) / total
    fit_grid = np.linspace(3, 7, 200)
    ax.plot(
        fit_grid,
        np.exp(intercept + slope * fit_grid),
        color="#34495e",
        linestyle="--",
        linewidth=1.4,
        label=(
            "Finite-range exponential fit to perimeter means\n"
            rf"$\bar{{d}}_0={np.exp(intercept):.3g}e^{{{slope:.3f}n}}$, $R^2={r_squared:.3f}$"
        ),
        zorder=2,
    )
    ax.plot(means_n, fit_y, "D", color="#34495e", markersize=4, zorder=4)

    edge3 = 7 - 4 * np.sqrt(3)
    ax.hlines(
        edge3,
        2.78,
        3.22,
        color="#c0392b",
        linestyle=":",
        linewidth=1.6,
        label=rf"$7-4\sqrt{{3}}={edge3:.5f}$ ($n=3$ edge)",
    )
    z0 = -38 - 24 * np.sqrt(3)
    beta_plus = (z0 + np.sqrt(z0**2 - 4)) / 2
    ax.hlines(
        abs(beta_plus),
        3.78,
        4.22,
        color="#d68910",
        linestyle=":",
        linewidth=1.6,
        label=rf"$|\beta_+^{{(4)}}|={abs(beta_plus):.5f}$ ($n=4$ edge)",
    )
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02
    )
    colorbar.set_label("Layer length $k$")
    ax.set_yscale("log")
    ax.set_xlim(2.65, 7.35)
    ax.set_xticks(range(3, 8))
    ax.set_xlabel("Cylinder perimeter $n$")
    ax.set_ylabel(r"Extremal Lee--Yang distance $|r_{\max}(k,n)|$")
    ax.set_title("Finite-perimeter Lee--Yang gap and its trend toward $t=0$")
    ax.grid(True, which="major", color="#d9e0e6", linewidth=0.65)
    ax.grid(True, which="minor", color="#edf0f3", linewidth=0.45)
    ax.legend(loc="upper right", frameon=True, framealpha=0.96)
    fig.text(
        0.01,
        -0.025,
        "The dashed fit describes the five observed perimeter means; it is not an asymptotic theorem.",
        fontsize=7.5,
        color="#4a5560",
    )
    _save_figure(fig, "fig_lee_yang_gap")


def _physical_curves(item: dict, t_values: list[mp.mpf]) -> tuple[list[float], list[float]]:
    roots = [mp.mpf(value) for value in item["spectral"]["roots_sorted_ascending"]]
    normalization = mp.mpf(item["k"] * item["n"])
    chi: list[float] = []
    density: list[float] = []
    for value in t_values:
        populations = [value / (value - root) for root in roots]
        density.append(float(mp.fsum(populations) / normalization))
        chi.append(
            float(
                mp.fsum(
                    -value * root / (value - root) ** 2 for root in roots
                )
                / normalization
            )
        )
    return chi, density


def _figure_susceptibility(data: dict) -> None:
    _verify_exact_self_dual_peaks(data["instances"])
    selected = {
        n: max((item for item in data["instances"] if item["n"] == n), key=lambda row: row["k"])
        for n in range(3, 8)
    }
    t_mp = [mp.e ** (mp.log(mp.mpf("0.05")) + i * (mp.log(4) - mp.log(mp.mpf("0.05"))) / 1000) for i in range(1001)]
    t_float = np.asarray([float(value) for value in t_mp])
    index_one = int(np.argmin(np.abs(t_float - 1.0)))
    colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.86, 5))
    fig, (ax_chi, ax_u) = plt.subplots(1, 2, figsize=(10.8, 4.45), sharex=True, constrained_layout=True)
    for color, n in zip(colors, range(3, 8)):
        item = selected[n]
        chi, density = _physical_curves(item, t_mp)
        label = rf"$n={n},\ k={item['k']}$"
        ax_chi.plot(t_float, chi, color=color, linewidth=1.65, label=label)
        ax_u.plot(t_float, density, color=color, linewidth=1.65, label=label)
        at_one = chi[index_one]
        ax_chi.scatter([1], [at_one], color=[color], edgecolor="white", linewidth=0.6, s=30, zorder=4)
        ax_u.scatter([1], [density[index_one]], color=[color], edgecolor="white", linewidth=0.6, s=30, zorder=4)
        for dual_t in (0.25, 4.0):
            index = min(range(len(t_float)), key=lambda i: abs(t_float[i] - dual_t))
            ax_chi.scatter([t_float[index]], [chi[index]], marker="o", facecolors="none", edgecolors=[color], s=24, linewidth=0.8, zorder=4)

    for ax in (ax_chi, ax_u):
        ax.set_xscale("log")
        ax.set_xlim(0.05, 4)
        ax.axvline(1, color="#555555", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.grid(True, which="major", color="#d9e0e6", linewidth=0.6)
        ax.grid(True, which="minor", color="#edf0f3", linewidth=0.4)
        ax.set_xlabel("Physical fugacity $t$")
    ax_chi.set_ylabel(r"Susceptibility per site $\chi_{k,n}(t)$")
    ax_chi.set_title(r"Self-dual susceptibility: $\chi(1/t)=\chi(t)$")
    ax_chi.text(
        0.04,
        0.95,
        r"Exact finite-instance maximum at $t^*=1$",
        transform=ax_chi.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#ccd4db", "alpha": 0.92},
    )
    ax_u.set_ylabel(r"Mean excitation density $u_{k,n}(t)$")
    ax_u.set_title(r"Pitman quasiparticle density")
    ax_u.text(
        0.04,
        0.95,
        r"$u(1/t)=(n-1)/n-u(t)$",
        transform=ax_u.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#ccd4db", "alpha": 0.92},
    )
    ax_chi.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.96)
    ax_u.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.96)
    fig.suptitle("Thermodynamic response across cylinder perimeters", fontsize=11)
    _save_figure(fig, "fig_susceptibility_duality")


def _figure_bkw(data: dict) -> None:
    comparisons = data["lee_yang_bkw_density_comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, n in zip(axes, (3, 4)):
        comparison = comparisons[str(n)]
        t_grid = np.asarray([float(value) for value in comparison["t_grid"]])
        empirical = np.asarray(
            [float(value) for value in comparison["empirical_cumulative_zero_density"]]
        )
        bkw = np.asarray([float(value) for value in comparison["bkw_cumulative_density"]])
        ax.plot(t_grid, bkw, color="#2471a3", linewidth=1.8, label=r"BKW $\int \rho_n(t)\,dt$")

        instance = next(
            item for item in data["instances"] if item["k"] == 8 and item["n"] == n
        )
        roots = np.asarray(
            [float(value) for value in instance["spectral"]["roots_sorted_ascending"]]
        )
        left, right = map(float, comparison["interval"])
        in_interval = roots[(roots >= left) & (roots <= right)]
        x_step = np.concatenate(([left], in_interval, [right]))
        y_step = np.concatenate(
            ([0.0], np.arange(1, len(in_interval) + 1) / (8 * n), [len(in_interval) / (8 * n)])
        )
        ax.step(
            x_step,
            y_step,
            where="post",
            color="#c0392b",
            linewidth=1.4,
            label=rf"Empirical zeros of $H_{{8,{n}}}$",
        )
        ax.set_xlabel("Negative fugacity $t$")
        ax.set_ylabel(r"Cumulative zero density $N(t)/(kn)$")
        ax.set_title(rf"$n={n}$, interval $[{left:.4g},{right:.4g}]$")
        ax.grid(True, color="#e2e7eb", linewidth=0.6)
        ax.legend(loc="best", frameon=True, framealpha=0.96)
        ax.text(
            0.03,
            0.96,
            f"empirical mass = {comparison['empirical_total_mass'][:8]}\n"
            f"BKW mass = {comparison['bkw_total_mass_grid_approx'][:8]}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#ccd4db", "alpha": 0.92},
        )
    fig.suptitle("Finite Lee--Yang zero distributions and BKW spectral limits", fontsize=11)
    _save_figure(fig, "fig_bkw_density")


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    _figure_gap(data)
    _figure_susceptibility(data)
    _figure_bkw(data)
    print("Exact autodual peak certificate: passed for all 23 instances.")
    for stem in (
        "fig_lee_yang_gap",
        "fig_susceptibility_duality",
        "fig_bkw_density",
    ):
        print(f"Wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main()
