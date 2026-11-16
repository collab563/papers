"""Exact cyclic-pipeline schedule counting, sampling, and SLA analysis."""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
import tracemalloc
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_DIR = ROOT / "FORMATO 1"
LEE_YANG_RESULTS = ROOT / "LEE_YANG_PHYSICS" / "resultados_lee_yang.json"
OUTPUT_PATH = Path(__file__).resolve().with_name("resultados_pipeline_sla.json")

PERIMETERS = (3, 4, 5, 6, 7)
WINDOW_COUNTS = (3, 4, 6)
SCALE_K_VALUES = (10, 100, 1_000, 10_000, 50_000, 100_000)
SCHEDULE_TEST_SAMPLES = 20_000
EXTENSION_TEST_SAMPLES = 20_000
SLA_SAMPLE_COUNT = 2_000
SLA_DELTAS = (0.10, 0.20, 0.30, 0.40)
SAMPLING_SEED = 20261112

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def _validate_positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}.")


def _transfer_mapping(n: int, window_count: int) -> tuple[tuple[int, ...], np.ndarray]:
    shape = (window_count,) * n
    coordinates = np.indices(shape, dtype=np.intp).reshape(n, -1)
    thresholds = np.maximum(coordinates, np.roll(coordinates, 1, axis=0))
    mapping = np.ravel_multi_index(thresholds, shape)
    return shape, mapping


def _suffix_sums_in_place(tensor: np.ndarray) -> None:
    for axis in range(tensor.ndim):
        reversed_view = np.flip(tensor, axis=axis)
        np.add.accumulate(reversed_view, axis=axis, out=reversed_view)


def _transfer(values: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    working = values.copy()
    _suffix_sums_in_place(working)
    transferred = np.empty(working.size, dtype=object)
    np.take(working.reshape(-1), mapping, out=transferred)
    return transferred.reshape(working.shape)


def count_cyclic_schedules_linear(k: int, n: int, m: int) -> int:
    """Count order-preserving schedules on C_{k,n} with labels 0..m.

    For fixed n and m, each layer applies n suffix-sum transforms on the
    (m+1)^n state tensor, followed by the cyclic max-threshold map.
    """
    _validate_positive_integer("k", k)
    _validate_positive_integer("n", n, 3)
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError("m debe ser un entero >= 0.")

    window_count = m + 1
    shape, mapping = _transfer_mapping(n, window_count)
    tensor = np.ones(shape, dtype=object)
    next_tensor = np.empty(tensor.size, dtype=object)

    for _ in range(k - 1):
        _suffix_sums_in_place(tensor)
        np.take(tensor.reshape(-1), mapping, out=next_tensor)
        tensor, next_tensor = next_tensor.reshape(shape), tensor.reshape(-1)

    return sum(tensor.flat)


@lru_cache(maxsize=8)
def _schedule_sampling_data(
    k: int, n: int, m: int
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[np.ndarray, ...],
]:
    window_count = m + 1
    shape, mapping = _transfer_mapping(n, window_count)
    states = tuple(np.ndindex(shape))
    mapped_states = tuple(states[int(index)] for index in mapping)
    successors = tuple(
        tuple(
            target_index
            for target_index, target in enumerate(states)
            if all(target[axis] >= threshold[axis] for axis in range(n))
        )
        for threshold in mapped_states
    )

    ways: list[np.ndarray] = [np.ones(shape, dtype=object)]
    for _ in range(k - 1):
        ways.append(_transfer(ways[-1], mapping))
    ways.reverse()
    return states, successors, tuple(ways)


def _weighted_choice(
    indices: tuple[int, ...] | range,
    weights: Any,
    rng: random.Random,
) -> int:
    total = sum(int(weights[index]) for index in indices)
    if total <= 0:
        raise ArithmeticError("No hay continuaciones válidas para el estado actual.")
    ticket = rng.randrange(total)
    for index in indices:
        weight = int(weights[index])
        if ticket < weight:
            return index
        ticket -= weight
    raise ArithmeticError("Falló la selección exacta ponderada.")


def _draw_schedule(
    states: tuple[tuple[int, ...], ...],
    successors: tuple[tuple[int, ...], ...],
    ways: tuple[np.ndarray, ...],
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    state_index = _weighted_choice(range(len(states)), ways[0].reshape(-1), rng)
    schedule = [states[state_index]]
    for layer in range(len(ways) - 1):
        state_index = _weighted_choice(
            successors[state_index], ways[layer + 1].reshape(-1), rng
        )
        schedule.append(states[state_index])
    return tuple(schedule)


def _is_valid_schedule(
    schedule: tuple[tuple[int, ...], ...], n: int, m: int
) -> bool:
    if any(len(layer) != n for layer in schedule):
        return False
    if any(value < 0 or value > m for layer in schedule for value in layer):
        return False
    return all(
        lower[position] <= upper[position]
        and lower[position] <= upper[(position + 1) % n]
        for lower, upper in zip(schedule, schedule[1:])
        for position in range(n)
    )


def sample_uniform_schedule(
    k: int,
    n: int,
    m: int,
    rng: random.Random | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Draw one valid schedule uniformly using exact backward-DP weights."""
    _validate_positive_integer("k", k)
    _validate_positive_integer("n", n, 3)
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError("m debe ser un entero >= 0.")
    generator = rng if rng is not None else random.SystemRandom()
    states, successors, ways = _schedule_sampling_data(k, n, m)
    schedule = _draw_schedule(states, successors, ways, generator)
    if not _is_valid_schedule(schedule, n, m):
        raise ArithmeticError("El muestreador produjo una planificación inválida.")
    return schedule


def _predecessor_masks(k: int, n: int) -> tuple[int, ...]:
    predecessors = [0] * (k * n)
    for layer in range(k - 1):
        for position in range(n):
            lower = layer * n + position
            predecessors[(layer + 1) * n + position] |= 1 << lower
            predecessors[(layer + 1) * n + (position + 1) % n] |= 1 << lower
    return tuple(predecessors)


@lru_cache(maxsize=16)
def _linear_extension_completion_counter(k: int, n: int) -> Any:
    predecessors = _predecessor_masks(k, n)
    full_mask = (1 << (k * n)) - 1

    @lru_cache(maxsize=None)
    def completions(placed: int) -> int:
        if placed == full_mask:
            return 1
        available = 0
        for vertex, required in enumerate(predecessors):
            bit = 1 << vertex
            if not placed & bit and required & placed == required:
                available |= bit
        total = 0
        while available:
            bit = available & -available
            available ^= bit
            total += completions(placed | bit)
        return total

    return completions


def sample_uniform_linear_extension(
    k: int,
    n: int,
    rng: random.Random | None = None,
) -> tuple[int, ...]:
    """Sample a linear extension exactly by DP over order ideals (kn <= 18)."""
    _validate_positive_integer("k", k)
    _validate_positive_integer("n", n, 3)
    task_count = k * n
    if task_count > 18:
        raise ValueError("El muestreo por ideales está limitado a k*n <= 18.")

    generator = rng if rng is not None else random.SystemRandom()
    predecessors = _predecessor_masks(k, n)
    completions = _linear_extension_completion_counter(k, n)
    placed = 0
    extension: list[int] = []
    while len(extension) < task_count:
        available_vertices = tuple(
            vertex
            for vertex, required in enumerate(predecessors)
            if not placed & (1 << vertex) and required & placed == required
        )
        weights = {
            vertex: completions(placed | (1 << vertex))
            for vertex in available_vertices
        }
        selected = _weighted_choice(
            available_vertices,
            weights,
            generator,
        )
        extension.append(selected + 1)
        placed |= 1 << selected
    return tuple(extension)


def _descent_count(extension: tuple[int, ...]) -> int:
    return sum(left > right for left, right in zip(extension, extension[1:]))


def _regularized_gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma, for chi-square p-values."""
    if a <= 0 or x < 0:
        raise ValueError("Los argumentos de Gamma Q deben ser válidos.")
    if x == 0:
        return 1.0
    log_scale = -x + a * math.log(x) - math.lgamma(a)
    epsilon = 1e-14
    tiny = 1e-300

    if x < a + 1.0:
        term = 1.0 / a
        total = term
        current_a = a
        for _ in range(10_000):
            current_a += 1.0
            term *= x / current_a
            total += term
            if abs(term) <= abs(total) * epsilon:
                break
        return min(1.0, max(0.0, 1.0 - total * math.exp(log_scale)))

    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for iteration in range(1, 10_001):
        coefficient = -iteration * (iteration - a)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    return min(1.0, max(0.0, math.exp(log_scale) * fraction))


def _chi_square_summary(
    observed: Counter[int],
    expected_probabilities: tuple[float, ...],
    sample_count: int,
) -> dict[str, float | int]:
    expected = [sample_count * probability for probability in expected_probabilities]
    statistic = sum(
        (observed[index] - expectation) ** 2 / expectation
        for index, expectation in enumerate(expected)
        if expectation > 0
    )
    degrees_of_freedom = sum(expectation > 0 for expectation in expected) - 1
    p_value = _regularized_gamma_q(degrees_of_freedom / 2, statistic / 2)
    return {
        "chi_square_statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
    }


def _valid_schedule_catalog(k: int, n: int, m: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    valid: list[tuple[tuple[int, ...], ...]] = []
    for flat_values in product(range(m + 1), repeat=k * n):
        schedule = tuple(
            tuple(flat_values[layer * n : (layer + 1) * n])
            for layer in range(k)
        )
        if _is_valid_schedule(schedule, n, m):
            valid.append(schedule)
    return tuple(valid)


def _validate_schedule_sampler(
    k: int,
    n: int,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    m = 1
    catalog = _valid_schedule_catalog(k, n, m)
    states, successors, ways = _schedule_sampling_data(k, n, m)
    if count_cyclic_schedules_linear(k, n, m) != len(catalog):
        raise ArithmeticError("El conteo de transferencia no coincide con enumeración.")

    rng = random.Random(seed)
    observed: Counter[int] = Counter()
    catalog_index = {schedule: index for index, schedule in enumerate(catalog)}
    for _ in range(sample_count):
        sample = _draw_schedule(states, successors, ways, rng)
        if not _is_valid_schedule(sample, n, m):
            raise ArithmeticError("La prueba de muestreo encontró una planificación inválida.")
        observed[catalog_index[sample]] += 1

    summary = _chi_square_summary(
        Counter({index: observed[index] for index in range(len(catalog))}),
        tuple(1.0 / len(catalog) for _ in catalog),
        sample_count,
    )
    return {
        "k": k,
        "n": n,
        "m": m,
        "valid_schedules": len(catalog),
        "samples": sample_count,
        **summary,
        "indistinguishable_at_5_percent": summary["p_value"] >= 0.05,
    }


def _validate_extension_sampler(
    record: dict[str, Any],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    observed: Counter[int] = Counter()
    rng = random.Random(seed)
    for _ in range(sample_count):
        extension = sample_uniform_linear_extension(
            record["k"], record["n"], rng
        )
        observed[_descent_count(extension)] += 1

    coefficients = record["h_star_ascending"]
    total = sum(coefficients)
    summary = _chi_square_summary(
        observed,
        tuple(coefficient / total for coefficient in coefficients),
        sample_count,
    )
    return {
        "k": record["k"],
        "n": record["n"],
        "tasks": record["task_count"],
        "samples": sample_count,
        "exact_linear_extensions": str(total),
        "exact_h_star_coefficients": [str(value) for value in coefficients],
        "empirical_descent_counts": [
            observed[descent] for descent in range(len(coefficients))
        ],
        **summary,
        "indistinguishable_at_5_percent": summary["p_value"] >= 0.05,
    }


def _load_certified_instances() -> list[dict[str, Any]]:
    certificate_pattern = re.compile(r"certificado_(\d+)_(\d+)\.json$")
    certificates: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(CERTIFICATE_DIR.glob("certificado_*.json")):
        match = certificate_pattern.fullmatch(path.name)
        if match is None:
            continue
        k, n = map(int, match.groups())
        if n not in PERIMETERS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        coefficients = [
            int(value)
            for value in payload["sign_certificate"][
                "polynomial_coefficients_descending"
            ]
        ]
        certificates[(k, n)] = {
            "coefficients_descending": coefficients,
            "negative_roots_certified": bool(
                payload.get("is_real_rooted_negative_simple", False)
            ),
            "source": path.relative_to(ROOT).as_posix(),
        }

    payload = json.loads(LEE_YANG_RESULTS.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for source_record in payload["instances"]:
        k = int(source_record["k"])
        n = int(source_record["n"])
        degree = int(source_record["degree"])
        if n not in PERIMETERS or degree > 40:
            continue
        key = (k, n)
        if key in seen:
            raise ArithmeticError(f"Instancia duplicada en resultados: {key}.")
        seen.add(key)

        coefficients_descending = [
            int(value)
            for value in source_record["polynomial_coefficients_descending"]
        ]
        certificate = certificates.get(key)
        if certificate is not None:
            if coefficients_descending != certificate["coefficients_descending"]:
                raise ArithmeticError(
                    f"El polinomio de {key} no coincide con su certificado."
                )
            polynomial_source = certificate["source"]
        else:
            polynomial_source = source_record["polynomial_source"]

        roots = [
            str(value)
            for value in source_record["spectral"]["roots_sorted_ascending"]
        ]
        roots_are_certified = bool(
            source_record["spectral"]["all_roots_real_simple_negative"]
        )
        if (
            len(coefficients_descending) != degree + 1
            or len(roots) != degree
            or any(float(root) >= 0 for root in roots)
            or not roots_are_certified
            or any(
                coefficients_descending[index]
                != coefficients_descending[-1 - index]
                for index in range(degree + 1)
            )
        ):
            raise ArithmeticError(f"Coeficientes o raíces inválidos para {key}.")
        if certificate and not certificate["negative_roots_certified"]:
            raise ArithmeticError(f"El certificado de {key} no confirma raíces negativas.")

        h_star = list(reversed(coefficients_descending))
        pitman_parameters = [
            1.0 / (1.0 - float(root))
            for root in roots
        ]
        if not math.isclose(sum(pitman_parameters), degree / 2, abs_tol=1e-9):
            raise ArithmeticError(f"La media Pitman no coincide con D/2 para {key}.")

        records.append(
            {
                "k": k,
                "n": n,
                "task_count": k * n,
                "dimension": k * n,
                "degree": degree,
                "h_star_ascending": h_star,
                "roots": roots,
                "pitman_parameters": pitman_parameters,
                "polynomial_source": polynomial_source,
                "roots_source": "LEE_YANG_PHYSICS/resultados_lee_yang.json",
                "negative_roots_certified": (
                    True
                    if certificate
                    else roots_are_certified
                ),
            }
        )
    records.sort(key=lambda record: (record["n"], record["k"]))
    if not records:
        raise FileNotFoundError("No hay polinomios certificados elegibles.")
    return records


def _log10_integer(value: int) -> float:
    if value <= 0:
        raise ValueError("El logaritmo requiere un entero positivo.")
    bits = value.bit_length()
    shift = max(0, bits - 53)
    leading = value >> shift
    return math.log10(leading) + shift * math.log10(2)


def _benchmark_scaling() -> list[dict[str, Any]]:
    m = 2
    rows: list[dict[str, Any]] = []
    for perimeter in (3, 4, 5):
        for k in SCALE_K_VALUES:
            tracemalloc.start()
            started = time.perf_counter()
            exact_count = count_cyclic_schedules_linear(k, perimeter, m)
            elapsed = time.perf_counter() - started
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                {
                    "k": k,
                    "n": perimeter,
                    "m": m,
                    "window_count": m + 1,
                    "task_count": k * perimeter,
                    "state_count": (m + 1) ** perimeter,
                    "wall_time_seconds": round(elapsed, 6),
                    "tracemalloc_peak_mb": round(peak_bytes / (1024 * 1024), 6),
                    "count_bit_length": exact_count.bit_length(),
                    "log10_count": round(_log10_integer(exact_count), 12),
                    "exact_count": str(exact_count),
                }
            )
            del exact_count
    return rows


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def _chernoff_bound(
    parameters: list[float], target: float
) -> tuple[float, float]:
    def derivative(log_t: float) -> float:
        total = 0.0
        for probability in parameters:
            if probability <= 0.0:
                continue
            if probability >= 1.0:
                total += 1.0
                continue
            log_odds = math.log(probability) - math.log1p(-probability)
            shifted = log_t + log_odds
            if shifted >= 0:
                total += 1.0 / (1.0 + math.exp(-shifted))
            else:
                exponential = math.exp(shifted)
                total += exponential / (1.0 + exponential)
        return total - target

    def log_mgf(log_t: float) -> float:
        total = 0.0
        for probability in parameters:
            if probability <= 0.0:
                continue
            if probability >= 1.0:
                total += log_t
                continue
            total += _logaddexp(
                math.log1p(-probability),
                math.log(probability) + log_t,
            )
        return total

    low = 0.0
    high = 1.0
    while derivative(high) < 0:
        high *= 2.0
        if high > 128:
            raise ArithmeticError("No se pudo acotar el óptimo Chernoff.")
    for _ in range(100):
        middle = (low + high) / 2.0
        if derivative(middle) < 0:
            low = middle
        else:
            high = middle
    log_t = (low + high) / 2.0
    log_bound = log_mgf(log_t) - target * log_t
    return math.exp(min(0.0, log_bound)), math.exp(log_t)


def _analyze_sla(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    empirical_by_instance: dict[tuple[int, int], Counter[int]] = {}
    sample_counts: dict[tuple[int, int], int] = {}
    eligible = [record for record in records if record["task_count"] <= 18]
    rng = random.Random(SAMPLING_SEED + 100)
    for record in eligible:
        counts: Counter[int] = Counter()
        for _ in range(SLA_SAMPLE_COUNT):
            extension = sample_uniform_linear_extension(
                record["k"], record["n"], rng
            )
            counts[_descent_count(extension)] += 1
        key = (record["k"], record["n"])
        empirical_by_instance[key] = counts
        sample_counts[key] = SLA_SAMPLE_COUNT

    results: list[dict[str, Any]] = []
    for record in records:
        coefficients = record["h_star_ascending"]
        total_extensions = sum(coefficients)
        degree = record["degree"]
        instance_result: dict[str, Any] = {
            "k": record["k"],
            "n": record["n"],
            "task_count": record["task_count"],
            "dimension": record["dimension"],
            "degree": degree,
            "polynomial_source": record["polynomial_source"],
            "roots_source": record["roots_source"],
            "negative_roots_certified": record["negative_roots_certified"],
            "h_star_coefficients_ascending": [
                str(value) for value in coefficients
            ],
            "roots_sorted_ascending": record["roots"],
            "pitman_bernoulli_parameters": [
                format(value, ".17g")
                for value in record["pitman_parameters"]
            ],
            "H_at_1": str(total_extensions),
            "thresholds": [],
        }
        empirical_counts = empirical_by_instance.get((record["k"], record["n"]))
        for delta in SLA_DELTAS:
            delta_fraction = {
                0.10: (1, 10),
                0.20: (1, 5),
                0.30: (3, 10),
                0.40: (2, 5),
            }[delta]
            target = Fraction(
                (delta_fraction[1] + delta_fraction[0]) * degree,
                2 * delta_fraction[1],
            )
            fraction = float(target)
            threshold = (
                target.numerator + target.denominator - 1
            ) // target.denominator
            tail_count = sum(coefficients[threshold:])
            exact_probability = tail_count / total_extensions
            chernoff_probability, optimal_t = _chernoff_bound(
                record["pitman_parameters"], fraction
            )
            hoeffding_probability = math.exp(-(delta * delta) * degree / 2)
            empirical_frequency: float | None = None
            if empirical_counts is not None:
                empirical_frequency = sum(
                    count
                    for descent, count in empirical_counts.items()
                    if descent >= threshold
                ) / sample_counts[(record["k"], record["n"])]

            instance_result["thresholds"].append(
                {
                    "delta": delta,
                    "target_mean_multiple": format(
                        1 + delta_fraction[0] / delta_fraction[1], ".2f"
                    ),
                    "target_value": format(fraction, ".12g"),
                    "integer_threshold": threshold,
                    "exact_tail_numerator": str(tail_count),
                    "exact_tail_denominator": str(total_extensions),
                    "probability_exact": exact_probability,
                    "pitman_chernoff_bound": chernoff_probability,
                    "pitman_chernoff_optimal_t": optimal_t,
                    "hoeffding_bound": hoeffding_probability,
                    "empirical_frequency": empirical_frequency,
                    "empirical_sample_count": (
                        sample_counts.get((record["k"], record["n"]))
                    ),
                }
            )
        results.append(instance_result)
    return results


def _verify_small_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for n in (3, 4, 5):
        for window_count in WINDOW_COUNTS:
            q = window_count
            direct = sum(
                math.prod(
                    min(upper[axis], upper[(axis + 1) % n]) + 1
                    for axis in range(n)
                )
                for upper in product(range(q), repeat=n)
            )
            transferred = count_cyclic_schedules_linear(2, n, q - 1)
            if transferred != direct:
                raise ArithmeticError(
                    f"Fallo de conteo C_{{2,{n}}}, {q} ventanas: "
                    f"{transferred} != {direct}."
                )
            checks.append(
                {
                    "k": 2,
                    "n": n,
                    "window_count": q,
                    "exact_count": str(transferred),
                    "matches_direct_two_layer_count": True,
                }
            )

    by_instance = {(record["k"], record["n"]): record for record in records}
    for key in ((2, 3), (3, 3)):
        record = by_instance[key]
        m = 1
        dimension = record["dimension"]
        expected = sum(
            record["h_star_ascending"][index]
            * math.comb(m + dimension - index, dimension)
            for index in range(min(m, record["degree"]) + 1)
        )
        actual = count_cyclic_schedules_linear(key[0], key[1], m)
        if actual != expected:
            raise ArithmeticError(f"El conteo Ehrhart no coincide para {key}.")
    return checks


def _format_scaling_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'k':>8} {'n':>3} {'vent.':>6} {'tareas':>9} {'estados':>8} "
        f"{'segundos':>10} {'pico MB':>10} {'bits':>8} {'log10 L':>14}"
    )
    separator = "-" * len(header)
    lines = [header, separator]
    for row in rows:
        lines.append(
            f"{row['k']:8d} {row['n']:3d} {row['window_count']:6d} "
            f"{row['task_count']:9d} {row['state_count']:8d} "
            f"{row['wall_time_seconds']:10.4f} "
            f"{row['tracemalloc_peak_mb']:10.3f} "
            f"{row['count_bit_length']:8d} {row['log10_count']:14.6f}"
        )
    return "\n".join(lines)


def _format_sla_table(instances: list[dict[str, Any]]) -> str:
    header = (
        f"{'k,n':>7} {'D':>3} {'delta':>5} {'umbral':>7} "
        f"{'P exacta':>12} {'Pitman-Chernoff':>16} "
        f"{'Hoeffding':>12} {'empirical':>12}"
    )
    lines = [header, "-" * len(header)]
    for instance in instances:
        for result in instance["thresholds"]:
            empirical = result["empirical_frequency"]
            lines.append(
                f"{instance['k']:2d},{instance['n']:<4d} "
                f"{instance['degree']:3d} {result['delta']:5.2f} "
                f"{result['integer_threshold']:7d} "
                f"{result['probability_exact']:12.4e} "
                f"{result['pitman_chernoff_bound']:16.4e} "
                f"{result['hoeffding_bound']:12.4e} "
                f"{'N/A' if empirical is None else f'{empirical:.4e}':>12}"
            )
    return "\n".join(lines)


def main() -> None:
    records = _load_certified_instances()
    small_count_checks = _verify_small_counts(records)

    schedule_tests = [
        _validate_schedule_sampler(2, 3, SCHEDULE_TEST_SAMPLES, SAMPLING_SEED + 1),
        _validate_schedule_sampler(3, 3, SCHEDULE_TEST_SAMPLES, SAMPLING_SEED + 2),
    ]
    by_instance = {(record["k"], record["n"]): record for record in records}
    extension_tests = [
        _validate_extension_sampler(
            by_instance[(2, 3)], EXTENSION_TEST_SAMPLES, SAMPLING_SEED + 2
        ),
        _validate_extension_sampler(
            by_instance[(3, 3)], EXTENSION_TEST_SAMPLES, SAMPLING_SEED + 4
        ),
    ]

    scaling = _benchmark_scaling()
    sla_instances = _analyze_sla(records)
    result = {
        "title": "Planificador exacto y garantías SLA para pipelines cíclicos",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "schedule_count": (
                "DP exacta por capas con transformadas de suma sufija eje por eje "
                "y mapeo de umbral cíclico max(x_i,x_{i-1})."
            ),
            "schedule_sampler": "DP hacia atrás con selección entera ponderada exacta.",
            "linear_extension_sampler": (
                "DP memoizada de completaciones sobre ideales de orden; kn <= 18."
            ),
            "pitman_representation": (
                "H(t)/H(1)=product_j(1-p_j+p_j*t), p_j=1/(1-r_j)."
            ),
        },
        "count_validation": small_count_checks,
        "sampling_validation": {
            "schedule_chi_square_tests": schedule_tests,
            "linear_extension_descent_chi_square_tests": extension_tests,
            "chi_square_significance_level": 0.05,
        },
        "scaling_parameters": {
            "k_values": list(SCALE_K_VALUES),
            "n_values": [3, 4, 5],
            "m": 2,
            "window_count": 3,
            "state_counts_by_n": {str(n): 3**n for n in (3, 4, 5)},
            "max_coupled_tasks": SCALE_K_VALUES[-1] * 5,
            "note": (
                "La barrida cubre n=3,4,5 con tres ventanas, alcanza 500000 tareas "
                "para n=5 y valida además 3,4,6 ventanas para cada n."
            ),
        },
        "scaling": scaling,
        "sla_instances": sla_instances,
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Escalamiento exacto O(k), n=3,4,5, m=2 (3 ventanas)")
    print(_format_scaling_table(scaling))
    print()
    print("SLA: cola exacta de des(natural), cota optima de Pitman-Chernoff y Hoeffding")
    print(_format_sla_table(sla_instances))
    print()
    print("Validacion de muestreo uniforme (20 000 muestras por instancia)")
    for test in schedule_tests:
        print(
            f"Schedules C_{{{test['k']},{test['n']}}}, m={test['m']}: "
            f"chi2={test['chi_square_statistic']:.3f}, "
            f"gl={test['degrees_of_freedom']}, p={test['p_value']:.4f}"
        )
    for test in extension_tests:
        print(
            f"Extensiones C_{{{test['k']},{test['n']}}}: "
            f"chi2={test['chi_square_statistic']:.3f}, "
            f"gl={test['degrees_of_freedom']}, p={test['p_value']:.4f}"
        )
    print(f"\nResultados estructurados: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
