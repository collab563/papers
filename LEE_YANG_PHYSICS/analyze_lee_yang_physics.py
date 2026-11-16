"""Exact Lee--Yang and thermodynamic analysis for cylindrical C_{k,n} models.

Certified Ehrhart numerators are read from ``FORMATO 1``.  Missing wider
instances are reconstructed by the Tensor-CRT implementation in
``BENCHMARK_D40``.  Roots are isolated with exact rational intervals and
refined with mpmath before thermodynamic observables are evaluated.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_DIR = ROOT / "FORMATO 1"
OUTPUT_PATH = Path(__file__).resolve().with_name("resultados_lee_yang.json")
TARGET_PERIMETERS = (3, 4, 5, 6, 7)
ROOT_PRECISION = 50
WORK_DPS = 75
PHYSICAL_GRID_SIZE = 121
PEAK_SCAN_SIZE = 401
BKW_GRID_SIZE = 401

mp.mp.dps = WORK_DPS
t_symbol, z_symbol = sp.symbols("t z")


def _load_tensor_crt_module() -> Any:
    module_path = ROOT / "BENCHMARK_D40" / "benchmark_ehrhart_d40.py"
    spec = importlib.util.spec_from_file_location("_lee_yang_tensor_crt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar Tensor-CRT desde {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_certified_polynomials() -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    pattern = re.compile(r"certificado_(\d+)_(\d+)\.json$")
    for path in sorted(CERTIFICATE_DIR.glob("certificado_*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        k, n = map(int, match.groups())
        if n not in (3, 4, 5) or k > 8:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sign_certificate = payload["sign_certificate"]
        coefficients_descending = [
            int(value)
            for value in sign_certificate["polynomial_coefficients_descending"]
        ]
        polynomial = sp.Poly.from_list(coefficients_descending, gens=t_symbol)
        records[(k, n)] = {
            "k": k,
            "n": n,
            "dimension": k * n,
            "degree": polynomial.degree(),
            "coefficients_descending": coefficients_descending,
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "certified_negative_simple": bool(
                payload.get("is_real_rooted_negative_simple", False)
            ),
            "crt_metadata": None,
        }
    return records


def _compute_missing_polynomials(
    records: dict[tuple[int, int], dict[str, Any]]
) -> None:
    # H_{8,4} is also needed for the requested n=4 BKW comparison, but no
    # certificate for it is present in FORMATO 1.
    required = ((3, 6), (4, 6), (3, 7), (8, 4))
    missing = [instance for instance in required if instance not in records]
    if not missing:
        return
    cache_path = Path(__file__).resolve().with_name(".crt_polynomials_cache.json")
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    pending = [
        instance
        for instance in missing
        if f"{instance[0]},{instance[1]}" not in cache
    ]
    for k, n in missing:
        cached = cache.get(f"{k},{n}")
        if cached is None:
            continue
        records[(k, n)] = {
            "k": k,
            "n": n,
            "dimension": cached["dimension"],
            "degree": cached["degree"],
            "coefficients_descending": cached["coefficients_descending"],
            "source": "Tensor-CRT: BENCHMARK_D40/benchmark_ehrhart_d40.py",
            "certified_negative_simple": None,
            "crt_metadata": cached["crt_metadata"],
        }
        print(f"Polinomio CRT C_{{{k},{n}}} recuperado de caché local.", flush=True)
    if not pending:
        return
    crt = _load_tensor_crt_module()
    for k, n in pending:
        print(f"Tensor-CRT exacto para C_{{{k},{n}}}...", flush=True)
        started = time.perf_counter()
        exact = crt.tensor_crt_volume_and_ehrhart(k, n)
        ascending = [int(value) for value in exact["h_star_coefficients"]]
        records[(k, n)] = {
            "k": k,
            "n": n,
            "dimension": exact["dimension"],
            "degree": exact["degree_h_star"],
            "coefficients_descending": list(reversed(ascending)),
            "source": "Tensor-CRT: BENCHMARK_D40/benchmark_ehrhart_d40.py",
            "certified_negative_simple": None,
            "crt_metadata": {
                "prime_count": exact["crt_prime_count"],
                "primes_by_dilation": exact["crt_prime_counts_by_dilation"],
                "dilations": exact["crt_dilations"],
                "wall_time_seconds": round(time.perf_counter() - started, 6),
                "normalized_volume_H_1": exact["normalized_volume_H_1"],
            },
        }
        cache[f"{k},{n}"] = {
            "dimension": exact["dimension"],
            "degree": exact["degree_h_star"],
            "coefficients_descending": list(reversed(ascending)),
            "crt_metadata": records[(k, n)]["crt_metadata"],
        }
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"  listo: grado={exact['degree_h_star']}, "
            f"primos CRT={exact['crt_prime_count']}, "
            f"{time.perf_counter() - started:.1f} s",
            flush=True,
        )


def _poly_eval_descending(coefficients: list[int], value: mp.mpf) -> mp.mpf:
    result = mp.mpf("0")
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


def _reduce_reciprocal_polynomial(
    coefficients_descending: list[int],
) -> tuple[list[int], bool]:
    """Return Q(z) coefficients for H(t)=t^m Q(t+t^-1), allowing odd degree."""
    degree = len(coefficients_descending) - 1
    has_minus_one_root = bool(degree % 2)
    even_coefficients = coefficients_descending
    if has_minus_one_root:
        quotient = [coefficients_descending[0]]
        for coefficient in coefficients_descending[1:-1]:
            quotient.append(coefficient - quotient[-1])
        remainder = coefficients_descending[-1] - quotient[-1]
        if remainder != 0:
            raise ArithmeticError("El polinomio recíproco impar no tiene factor t+1.")
        even_coefficients = quotient

    even_degree = len(even_coefficients) - 1
    if even_degree % 2:
        raise ArithmeticError("La reducción recíproca no tiene grado par.")
    half_degree = even_degree // 2
    for index in range(half_degree):
        if even_coefficients[index] != even_coefficients[-1 - index]:
            raise ArithmeticError("Los coeficientes del polinomio no son recíprocos.")

    # S_j(z)=t^j+t^-j for z=t+t^-1.
    symmetric_powers: list[list[int]] = [[2], [0, 1]]
    for _ in range(2, half_degree + 1):
        previous = symmetric_powers[-1]
        before_previous = symmetric_powers[-2]
        current = [0] * (len(previous) + 1)
        for power, coefficient in enumerate(previous):
            current[power + 1] += coefficient
        for power, coefficient in enumerate(before_previous):
            current[power] -= coefficient
        symmetric_powers.append(current)

    reduced_ascending = [0] * (half_degree + 1)
    for pair_index in range(half_degree):
        symmetric = symmetric_powers[half_degree - pair_index]
        coefficient = even_coefficients[pair_index]
        for power, value in enumerate(symmetric):
            reduced_ascending[power] += coefficient * value
    reduced_ascending[0] += even_coefficients[half_degree]
    while len(reduced_ascending) > 1 and reduced_ascending[-1] == 0:
        reduced_ascending.pop()
    if len(reduced_ascending) != half_degree + 1:
        raise ArithmeticError("Grado inesperado en el polinomio reducido.")
    return list(reversed(reduced_ascending)), has_minus_one_root


def _mp_from_rational(value: sp.Rational) -> mp.mpf:
    value = sp.Rational(value)
    return mp.mpf(int(value.p)) / int(value.q)


def _refine_isolating_interval(
    coefficients_descending: list[int], interval: tuple[sp.Rational, sp.Rational]
) -> mp.mpf:
    left_exact, right_exact = interval
    left = _mp_from_rational(left_exact)
    right = _mp_from_rational(right_exact)
    if left == right:
        return left
    f_left = _poly_eval_descending(coefficients_descending, left)
    f_right = _poly_eval_descending(coefficients_descending, right)
    if f_left == 0:
        return left
    if f_right == 0:
        return right
    if (f_left > 0) == (f_right > 0):
        raise ArithmeticError("El intervalo racional no aísla un cambio de signo.")
    target_width = mp.power(10, -(ROOT_PRECISION + 15)) * max(1, abs(left), abs(right))
    for _ in range(260):
        middle = (left + right) / 2
        f_middle = _poly_eval_descending(coefficients_descending, middle)
        if f_middle == 0:
            return middle
        if (f_middle > 0) == (f_left > 0):
            left, f_left = middle, f_middle
        else:
            right, f_right = middle, f_middle
        if right - left <= target_width:
            break
    return (left + right) / 2


def _find_negative_roots(
    coefficients_descending: list[int],
) -> tuple[list[mp.mpf], list[int], bool]:
    polynomial = sp.Poly.from_list(coefficients_descending, gens=t_symbol)
    degree = polynomial.degree()
    if polynomial.LC() != 1 or polynomial.nth(0) != 1:
        raise ArithmeticError("Se esperaba un polinomio mónico y con término constante 1.")
    if any(
        coefficients_descending[index] != coefficients_descending[-1 - index]
        for index in range(degree + 1)
    ):
        raise ArithmeticError("El polinomio H_{k,n} no es recíproco.")

    reduced_descending, has_minus_one_root = _reduce_reciprocal_polynomial(
        coefficients_descending
    )
    reduced = sp.Poly.from_list(reduced_descending, gens=z_symbol)
    if reduced.degree() == 0:
        intervals: list[tuple[tuple[sp.Rational, sp.Rational], int]] = []
        all_negative = has_minus_one_root
    else:
        intervals = reduced.intervals(eps=sp.Rational(1, 10**65))
        isolated_count = sum(multiplicity for _, multiplicity in intervals)
        roots_below_minus_two = all(
            right < -2 or (right == -2 and left == -2)
            for (left, right), _ in intervals
        )
        all_negative = (
            isolated_count == reduced.degree()
            and roots_below_minus_two
            and all(multiplicity == 1 for _, multiplicity in intervals)
        )

    roots: list[mp.mpf] = []
    if has_minus_one_root:
        roots.append(mp.mpf(-1))
    if reduced.degree() > 0:
        if sum(multiplicity for _, multiplicity in intervals) != reduced.degree():
            raise ArithmeticError(
                "No se aislaron todas las raíces reales del polinomio recíproco reducido."
            )
        for interval, multiplicity in intervals:
            if multiplicity != 1:
                raise ArithmeticError("Se detectó una raíz múltiple en H_{k,n}.")
            left, right = interval
            if right >= -2:
                raise ArithmeticError(
                    "Una raíz reducida no produce dos raíces reales negativas de H_{k,n}."
                )
            z_root = _refine_isolating_interval(reduced_descending, interval)
            radical = mp.sqrt(z_root * z_root - 4)
            roots.append((z_root - radical) / 2)
            roots.append((z_root + radical) / 2)
    roots.sort()
    if len(roots) != degree or any(root >= 0 for root in roots):
        raise ArithmeticError(
            f"Conteo incompatible de raíces negativas: {len(roots)} frente a grado {degree}."
        )
    return roots, reduced_descending, all_negative


def _format_mp(value: mp.mpf | mp.mpc, digits: int = ROOT_PRECISION) -> str:
    return mp.nstr(value, n=digits, strip_zeros=False)


def _exact_self_dual_peak_certificate(coefficients_descending: list[int]) -> bool:
    """Check chi(1)-chi(t)=(t-1)^2 C(t) with C coefficientwise nonnegative."""
    polynomial = sp.Poly.from_list(coefficients_descending, gens=t_symbol)
    first = polynomial.eval(1)
    derivative = polynomial.diff()
    second_derivative = derivative.diff()
    numerator = sp.Poly(
        t_symbol * derivative.as_expr() * polynomial.as_expr()
        + t_symbol**2 * second_derivative.as_expr() * polynomial.as_expr()
        - t_symbol**2 * derivative.as_expr() ** 2,
        t_symbol,
    )
    difference = sp.Poly(
        numerator.eval(1) * polynomial.as_expr() ** 2
        - numerator.as_expr() * first**2,
        t_symbol,
    )
    quotient, remainder = sp.div(
        difference, sp.Poly((t_symbol - 1) ** 2, t_symbol)
    )
    return (
        remainder.as_expr() == 0
        and any(quotient.all_coeffs())
        and all(coefficient >= 0 for coefficient in quotient.all_coeffs())
    )


def _linspace_mp(start: mp.mpf, stop: mp.mpf, count: int) -> list[mp.mpf]:
    if count < 2:
        return [start]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


def _observables(
    k: int, n: int, coefficients_descending: list[int], roots: list[mp.mpf]
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalization = mp.mpf(k * n)
    physical_t = _linspace_mp(mp.mpf("0.01"), mp.mpf("2.0"), PHYSICAL_GRID_SIZE)
    free_energy: list[str] = []
    excitation_density: list[str] = []
    susceptibility: list[str] = []

    def chi_at(value: mp.mpf) -> mp.mpf:
        return mp.fsum(
            (-value * root) / (value - root) ** 2 for root in roots
        ) / normalization

    for value in physical_t:
        polynomial_value = _poly_eval_descending(coefficients_descending, value)
        pitman = [value / (value - root) for root in roots]
        free_energy.append(_format_mp(mp.log(polynomial_value) / normalization))
        excitation_density.append(_format_mp(mp.fsum(pitman) / normalization))
        susceptibility.append(_format_mp(chi_at(value)))

    scan_t = _linspace_mp(mp.mpf("0.01"), mp.mpf("2.0"), PEAK_SCAN_SIZE)
    scan_chi = [chi_at(value) for value in scan_t]
    candidates: list[tuple[mp.mpf, mp.mpf]] = [(scan_chi[0], scan_t[0]), (scan_chi[-1], scan_t[-1])]
    for index in range(1, len(scan_t) - 1):
        if scan_chi[index] >= scan_chi[index - 1] and scan_chi[index] >= scan_chi[index + 1]:
            left, right = scan_t[index - 1], scan_t[index + 1]
            ratio = (mp.sqrt(5) - 1) / 2
            x1 = right - ratio * (right - left)
            x2 = left + ratio * (right - left)
            y1, y2 = chi_at(x1), chi_at(x2)
            for _ in range(100):
                if y1 < y2:
                    left, x1, y1 = x1, x2, y2
                    x2 = left + ratio * (right - left)
                    y2 = chi_at(x2)
                else:
                    right, x2, y2 = x2, x1, y1
                    x1 = right - ratio * (right - left)
                    y1 = chi_at(x1)
            peak_t = (left + right) / 2
            candidates.append((chi_at(peak_t), peak_t))
    chi_max, t_star = max(candidates, key=lambda pair: pair[0])
    if _exact_self_dual_peak_certificate(coefficients_descending):
        t_star = mp.mpf(1)
        chi_max = chi_at(t_star)

    observables = {
        "physical_fugacity_interval": ["0.01", "2.0"],
        "normalization_kn": k * n,
        "t_grid": [_format_mp(value) for value in physical_t],
        "free_energy_density_f": free_energy,
        "mean_excitation_density_u": excitation_density,
        "susceptibility_chi": susceptibility,
        "susceptibility_peak": {
            "t_star": _format_mp(t_star),
            "chi_max": _format_mp(chi_max),
            "search_interval": ["0.01", "2.0"],
            "scan_points": PEAK_SCAN_SIZE,
            "refinement": "búsqueda áurea de cada máximo local detectado en el barrido",
        },
    }
    peak_summary = {"t_star": t_star, "chi_max": chi_max}
    return observables, peak_summary


def _spectral_constants() -> dict[int, dict[str, mp.mpf | str]]:
    n3_left = -7 - 4 * mp.sqrt(3)
    n3_right = -7 + 4 * mp.sqrt(3)
    z0 = -38 - 24 * mp.sqrt(3)
    beta_minus = (z0 - mp.sqrt(z0 * z0 - 4)) / 2
    beta_plus = (z0 + mp.sqrt(z0 * z0 - 4)) / 2
    return {
        3: {
            "left": n3_left,
            "right": n3_right,
            "left_exact": "-7-4*sqrt(3)",
            "right_exact": "-7+4*sqrt(3)",
        },
        4: {
            "left": beta_minus,
            "right": beta_plus,
            "left_exact": "(z0-sqrt(z0^2-4))/2, z0=-38-24*sqrt(3)",
            "right_exact": "(z0+sqrt(z0^2-4))/2, z0=-38-24*sqrt(3)",
        },
    }


def _phase_n3(value: mp.mpf) -> mp.mpf:
    a = -(value * value + 14 * value + 1)
    if a < 0 and abs(a) < mp.mpf("1e-60"):
        a = mp.mpf("0")
    if a < 0:
        raise ArithmeticError("La fase n=3 fue evaluada fuera de su intervalo espectral.")
    denominator = 2 * (1 - value) * mp.sqrt(-value)
    cosine = (value * value + 6 * value + 1) / denominator
    sine = (value + 1) * mp.sqrt(a) / denominator
    phase = mp.atan2(sine, cosine)
    if phase < 0:
        phase += 2 * mp.pi
    return phase


def _phase_n4(value: mp.mpf) -> mp.mpf:
    # The characteristic cubic is lambda^3-P4 lambda^2-Q4 lambda-R4.
    p4 = value**3 + 23 * value**2 + 23 * value + 1
    q4 = 7 * value**5 - 28 * value**4 - 22 * value**3 - 28 * value**2 + 7 * value
    r4 = value * (value - 1) ** 6 * (value + 1)
    a = -p4
    b = -q4
    c = -r4
    depressed_p = b - a * a / 3
    depressed_q = 2 * a**3 / 27 - a * b / 3 + c
    discriminant = (depressed_q / 2) ** 2 + (depressed_p / 3) ** 3
    if discriminant < 0:
        # The BKW interval has one real mode and a conjugate pair. A tiny
        # negative roundoff at a double-mode point is treated as zero.
        scale = max(
            mp.mpf("1"),
            abs((depressed_q / 2) ** 2) + abs((depressed_p / 3) ** 3),
        )
        if abs(discriminant) > mp.mpf("1e-45") * scale:
            raise ArithmeticError(
                "No se obtuvo un par complejo dominante en I_4; "
                f"t={mp.nstr(value, 12)}, discriminante={mp.nstr(discriminant, 8)}."
            )
        discriminant = mp.mpf("0")
    root_disc = mp.sqrt(discriminant)

    def real_cuberoot(number: mp.mpf) -> mp.mpf:
        return mp.sign(number) * mp.power(abs(number), mp.mpf(1) / 3) if number else mp.mpf("0")

    u = real_cuberoot(-depressed_q / 2 + root_disc)
    v = real_cuberoot(-depressed_q / 2 - root_disc)
    real_part = -a / 3 - (u + v) / 2
    imaginary_part = mp.sqrt(3) * abs(u - v) / 2
    if imaginary_part == 0:
        return mp.pi if real_part < 0 else mp.mpf("0")
    return mp.atan2(imaginary_part, real_part)


def _bkw_comparisons(
    records: dict[tuple[int, int], dict[str, Any]],
    roots_by_instance: dict[tuple[int, int], list[mp.mpf]],
) -> dict[str, Any]:
    constants = _spectral_constants()
    output: dict[str, Any] = {}
    for n in (3, 4):
        k = 8
        if (k, n) not in records:
            continue
        left = constants[n]["left"]
        right = constants[n]["right"]
        t_grid = _linspace_mp(left, right, BKW_GRID_SIZE)
        phase_fn = _phase_n3 if n == 3 else _phase_n4
        phases = [phase_fn(value) for value in t_grid]
        continuous_phase = [phases[0]]
        for previous_raw, current_raw in zip(phases, phases[1:]):
            increment = current_raw - previous_raw
            while increment > mp.pi:
                increment -= 2 * mp.pi
            while increment < -mp.pi:
                increment += 2 * mp.pi
            continuous_phase.append(continuous_phase[-1] + increment)
        phase_variation = [mp.mpf("0")]
        for previous, current in zip(continuous_phase, continuous_phase[1:]):
            phase_variation.append(phase_variation[-1] + abs(current - previous))
        local_density: list[mp.mpf | None] = [None]
        for index in range(1, len(t_grid) - 1):
            increment = phase_variation[index + 1] - phase_variation[index - 1]
            local_density.append(
                increment / (t_grid[index + 1] - t_grid[index - 1]) / (mp.pi * n)
            )
        local_density.append(None)

        roots = roots_by_instance[(k, n)]
        empirical_cumulative = [
            mp.mpf(sum(root <= value for root in roots)) / (k * n)
            for value in t_grid
        ]
        bkw_cumulative = [value / (mp.pi * n) for value in phase_variation]
        output[str(n)] = {
            "instance": f"H_{{{k},{n}}}",
            "interval": [_format_mp(left), _format_mp(right)],
            "interval_exact": [constants[n]["left_exact"], constants[n]["right_exact"]],
            "normalization": f"conteo acumulado/(k*n), k={k}; rho=|dPsi/dt|/(pi*n)",
            "phase_definition": (
                "argumento del autovalor conjugado dominante de la recurrencia exacta n=3"
                if n == 3
                else (
                    "argumento del autovalor de parte imaginaria positiva del cúbico "
                    "de transferencia n=4; la ley BKW supone que la recurrencia de orden 3 "
                    "vale para todo k"
                )
            ),
            "grid_points": BKW_GRID_SIZE,
            "t_grid": [_format_mp(value) for value in t_grid],
            "phase_Psi": [_format_mp(value) for value in continuous_phase],
            "empirical_cumulative_zero_density": [
                _format_mp(value) for value in empirical_cumulative
            ],
            "bkw_cumulative_density": [_format_mp(value) for value in bkw_cumulative],
            "bkw_density_rho": [
                _format_mp(value) if value is not None else None for value in local_density
            ],
            "empirical_zero_count_in_interval": sum(left <= root <= right for root in roots),
            "polynomial_degree": len(roots),
            "bkw_total_variation_radians_grid_approx": _format_mp(phase_variation[-1]),
            "bkw_total_mass_grid_approx": _format_mp(bkw_cumulative[-1]),
            "empirical_total_mass": _format_mp(
                mp.mpf(sum(left <= root <= right for root in roots)) / (k * n)
            ),
            "bkw_minus_empirical_total_mass": _format_mp(
                bkw_cumulative[-1]
                - mp.mpf(sum(left <= root <= right for root in roots)) / (k * n)
            ),
            "density_discretization": (
                "Diferencias centrales de alta precisión; la derivada en los extremos "
                "se deja null y las coalescencias de n=4 tienen singularidades integrables."
            ),
            "comparison_note": (
                "Se compara la acumulada empírica finita con la acumulada de la densidad BKW. "
                "Para n=4 la ley usa la recurrencia de orden tres, cuya validez para todo k "
                "se formula como hipótesis en el análisis fuente."
            ),
        }
    return output


def _summaries(
    records: dict[tuple[int, int], dict[str, Any]],
    metrics: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    constants = _spectral_constants()
    summaries: list[dict[str, Any]] = []
    table_rows: list[str] = []
    for n in TARGET_PERIMETERS:
        instances = sorted(
            (record for (k, perimeter), record in records.items() if perimeter == n),
            key=lambda record: record["k"],
        )
        if not instances:
            continue
        d0_values = [metrics[(record["k"], n)]["d0"] for record in instances]
        peak_values = [metrics[(record["k"], n)]["chi_max"] for record in instances]
        peak_positions = [metrics[(record["k"], n)]["t_star"] for record in instances]
        by_k = {record["k"]: metrics[(record["k"], n)]["d0"] for record in instances}
        previous_by_k = {
            record["k"]: metrics[(record["k"], n - 1)]["d0"]
            for record in instances
            if n > 3 and (record["k"], n - 1) in metrics
        }
        contraction = [
            {"k": k, "ratio_d0_n_over_n_minus_1": _format_mp(by_k[k] / previous)}
            for k, previous in sorted(previous_by_k.items())
        ]
        summary: dict[str, Any] = {
            "n": n,
            "instances": [record["k"] for record in instances],
            "instance_count": len(instances),
            "mean_d0": _format_mp(mp.fsum(d0_values) / len(d0_values)),
            "min_d0": _format_mp(min(d0_values)),
            "max_d0": _format_mp(max(d0_values)),
            "mean_susceptibility_peak": _format_mp(mp.fsum(peak_values) / len(peak_values)),
            "max_susceptibility_peak": _format_mp(max(peak_values)),
            "mean_peak_location_t_star": _format_mp(mp.fsum(peak_positions) / len(peak_positions)),
            "contraction_from_previous_perimeter": contraction,
            "mean_contraction_ratio_from_previous_perimeter": (
                _format_mp(
                    mp.fsum(
                        by_k[item["k"]] / previous_by_k[item["k"]]
                        for item in contraction
                    )
                    / len(contraction)
                )
                if contraction
                else None
            ),
        }
        if n in constants:
            edge = constants[n]["right"]
            summary["asymptotic_right_edge_exact"] = constants[n]["right_exact"]
            summary["asymptotic_right_edge"] = _format_mp(edge)
            summary["distance_of_rmax_to_asymptotic_edge_by_k"] = [
                {
                    "k": record["k"],
                    "edge_minus_rmax": _format_mp(
                        edge - metrics[(record["k"], n)]["r_max"]
                    ),
                }
                for record in instances
            ]
        summaries.append(summary)
        ratio = summary["mean_contraction_ratio_from_previous_perimeter"]
        ratio_text = "—" if ratio is None else _format_mp(mp.mpf(ratio), 7)
        edge_gap = "—"
        if n in constants:
            edge_gap = _format_mp(
                min(edge - metrics[(record["k"], n)]["r_max"] for record in instances), 7
            )
        table_rows.append(
            " | ".join(
                (
                    str(n),
                    str(len(instances)),
                    _format_mp(mp.fsum(d0_values) / len(d0_values), 7),
                    ratio_text,
                    edge_gap,
                    _format_mp(max(peak_values), 7),
                    _format_mp(mp.fsum(peak_positions) / len(peak_positions), 7),
                )
            )
        )
    return summaries, table_rows


def _build_results() -> dict[str, Any]:
    records = _read_certified_polynomials()
    _compute_missing_polynomials(records)
    instances = sorted(records.items(), key=lambda item: (item[0][1], item[0][0]))
    results: list[dict[str, Any]] = []
    roots_by_instance: dict[tuple[int, int], list[mp.mpf]] = {}
    metrics: dict[tuple[int, int], dict[str, Any]] = {}

    for (k, n), record in instances:
        print(f"Raíces y termodinámica de C_{{{k},{n}}}...", flush=True)
        roots, reduced, all_negative = _find_negative_roots(record["coefficients_descending"])
        roots_by_instance[(k, n)] = roots
        r_min, r_max = roots[0], roots[-1]
        observables, peak = _observables(k, n, record["coefficients_descending"], roots)
        spectral = {
            "r_min": _format_mp(r_min),
            "r_max": _format_mp(r_max),
            "d0_distance_to_zero": _format_mp(abs(r_max)),
            "distance_to_self_dual_fugacity": _format_mp(1 - r_max),
            "spectral_width_log10_rmin_over_rmax": _format_mp(mp.log10(r_min / r_max)),
            "negative_simple_root_count": len(roots),
            "all_roots_real_simple_negative": all_negative,
            "roots_sorted_ascending": [_format_mp(root) for root in roots],
            "reciprocal_root_check_max_error": _format_mp(
                max(abs(roots[index] * roots[-1 - index] - 1) for index in range(len(roots)))
            ),
            "reduced_polynomial_q_coefficients_descending": [str(value) for value in reduced],
        }
        result = {
            "k": k,
            "n": n,
            "dimension": record["dimension"],
            "degree": record["degree"],
            "polynomial_coefficients_descending": [str(value) for value in record["coefficients_descending"]],
            "polynomial_source": record["source"],
            "input_certificate_claims_negative_simple_roots": record["certified_negative_simple"],
            "crt_metadata": record["crt_metadata"],
            "root_method": (
                "reducción recíproca H(t)=t^m Q(t+t^-1), aislamiento racional exacto "
                "de Sturm y refinamiento por bisección mpmath"
            ),
            "root_precision_decimal_digits": ROOT_PRECISION,
            "spectral": spectral,
            "thermodynamics": observables,
        }
        results.append(result)
        metrics[(k, n)] = {
            "r_min": r_min,
            "r_max": r_max,
            "d0": abs(r_max),
            "chi_max": peak["chi_max"],
            "t_star": peak["t_star"],
        }

    summaries, table_rows = _summaries(records, metrics)
    bkw = _bkw_comparisons(records, roots_by_instance)
    z0 = -38 - 24 * mp.sqrt(3)
    beta_plus_exact = (z0 + mp.sqrt(z0 * z0 - 4)) / 2
    constants = _spectral_constants()
    return {
        "title": "Estudio exacto de ceros de Lee–Yang y termodinámica de C_{k,n}",
        "generated_at_local_date": "2026-11-12",
        "numeric_precision": {
            "root_decimal_digits": ROOT_PRECISION,
            "working_mpmath_dps": WORK_DPS,
            "exact_polynomial_arithmetic": "enteros arbitrarios; CRT para los casos faltantes",
            "root_certification": "intervalos racionales de SymPy/Sturm y refinamiento mpmath",
        },
        "perimeters": list(TARGET_PERIMETERS),
        "pitman_representation": {
            "single_zero_population": "p_j(t)=t/(t-r_j)",
            "free_energy_density": "f_{k,n}(t)=ln(H_{k,n}(t))/(k*n)",
            "mean_excitation_density": "u_{k,n}(t)=sum_j p_j(t)/(k*n)",
            "susceptibility": "chi_{k,n}(t)=sum_j[-t*r_j/(t-r_j)^2]/(k*n)",
            "physical_fugacity_interval": ["0.01", "2.0"],
        },
        "asymptotic_edges": {
            "n=3": {
                "left_exact": "-7-4*sqrt(3)",
                "right_exact": "-7+4*sqrt(3)",
                "right_numeric": _format_mp(constants[3]["right"]),
            },
            "n=4": {
                "q4": "t^4+76*t^3-282*t^2+76*t+1",
                "z0": "-38-24*sqrt(3)",
                "beta_plus_exact": "(z0+sqrt(z0^2-4))/2",
                "beta_plus_numeric": _format_mp(beta_plus_exact),
            },
        },
        "summary_by_perimeter": summaries,
        "lee_yang_bkw_density_comparisons": bkw,
        "instances": results,
        "console_summary_table": {
            "headers": [
                "n", "casos", "media |r_max|", "razón |r_max| n/(n-1)",
                "brecha a límite n=3/4", "max chi_max", "media t*",
            ],
            "rows": table_rows,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    payload = _build_results()
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    headers = payload["console_summary_table"]["headers"]
    rows = payload["console_summary_table"]["rows"]
    widths = [
        max(len(headers[index]), *(len(row.split(" | ")[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        values = row.split(" | ")
        print(" | ".join(values[index].ljust(widths[index]) for index in range(len(values))))
    print(f"\nResultados guardados en: {OUTPUT_PATH}")
    print(f"Instancias analizadas: {len(payload['instances'])}")
    print(
        "Comparaciones BKW disponibles: "
        + ", ".join(f"n={n}" for n in payload["lee_yang_bkw_density_comparisons"])
    )
    cache_path = Path(__file__).resolve().with_name(".crt_polynomials_cache.json")
    if cache_path.exists():
        cache_path.unlink()


if __name__ == "__main__":
    main()
