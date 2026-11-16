"""Benchmark independiente de conteo Ehrhart exacto para C_{k,n}."""

from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np


RESULTS_PATH = Path(__file__).resolve().with_name("resultados_benchmark_d40.json")
INSTANCES = (
    (2, 3),
    (3, 3),
    (3, 4),
    (4, 4),
    (4, 5),
    (5, 5),
    (6, 5),
    (7, 5),
    (8, 5),
)
BASELINE_TIMEOUT_SECONDS = 15
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _is_prime_64(number: int) -> bool:
    if number < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if number % prime == 0:
            return number == prime
    odd_part = number - 1
    powers_of_two = 0
    while odd_part % 2 == 0:
        powers_of_two += 1
        odd_part //= 2
    for base in _MILLER_RABIN_BASES:
        if base % number == 0:
            continue
        witness = pow(base, odd_part, number)
        if witness in (1, number - 1):
            continue
        for _ in range(powers_of_two - 1):
            witness = witness * witness % number
            if witness == number - 1:
                break
        else:
            return False
    return True


def _crt_primes_for_bound(bound: int) -> list[int]:
    """Devuelve el mínimo prefijo de primos de 61 bits cuyo producto supera bound."""
    primes: list[int] = []
    product = 1
    candidate = (1 << 61) - 1
    while product <= bound:
        while not _is_prime_64(candidate):
            candidate -= 2
        primes.append(candidate)
        product *= candidate
        candidate -= 2
    return primes


def _contract_modular(k: int, n: int, dilation: int, prime: int) -> int:
    """Contracción por capas con sumas sufijas modulares eje por eje."""
    dimension = dilation + 1
    shape = (dimension,) * n
    tensor = np.ones(shape, dtype=np.int64)
    for _ in range(k - 1):
        for axis in range(n):
            current_slice = [slice(None)] * n
            following_slice = [slice(None)] * n
            for coordinate in range(dimension - 2, -1, -1):
                current_slice[axis] = coordinate
                following_slice[axis] = coordinate + 1
                current = tensor[tuple(current_slice)]
                following = tensor[tuple(following_slice)]
                np.add(current, following, out=current)
                np.remainder(current, prime, out=current)

        source = tensor.reshape(-1)
        contracted = np.empty_like(source)
        strides = [dimension ** (n - axis - 1) for axis in range(n)]
        chunk_size = 1 << 19
        for start in range(0, source.size, chunk_size):
            stop = min(start + chunk_size, source.size)
            flat_indices = np.arange(start, stop, dtype=np.int64)
            mapped_indices = np.zeros(stop - start, dtype=np.int64)
            for axis in range(n):
                coordinate = flat_indices // strides[axis] % dimension
                previous = flat_indices // strides[(axis - 1) % n] % dimension
                mapped_indices += np.maximum(coordinate, previous) * strides[axis]
            contracted[start:stop] = source[mapped_indices]
        tensor = contracted.reshape(shape)
    residues = tensor.reshape(-1)
    while residues.size > 1:
        starts = np.arange(0, residues.size, 3, dtype=np.intp)
        residues = np.add.reduceat(residues, starts, dtype=np.int64)
        np.remainder(residues, prime, out=residues)
    return int(residues[0]) % prime


def _count_by_crt(k: int, n: int, dilation: int) -> tuple[int, int]:
    bound = (dilation + 1) ** (k * n)
    primes = _crt_primes_for_bound(bound)
    value = 0
    modulus = 1
    for prime in primes:
        residue = _contract_modular(k, n, dilation, prime)
        correction = (residue - value) * pow(modulus, -1, prime) % prime
        value += modulus * correction
        modulus *= prime
    if value > bound or modulus <= bound:
        raise ArithmeticError("La reconstrucción CRT no respeta la cota exacta.")
    return value, len(primes)


def tensor_crt_volume_and_ehrhart(k: int, n: int) -> dict[str, Any]:
    """Obtiene el h*, H(1), volumen y L(0)..L(d) con aritmética exacta."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k debe ser un entero >= 2.")
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("n debe ser un entero >= 3.")

    dimension = k * n
    degree = dimension - k
    half_degree = degree // 2
    # La última muestra verifica, además, la simetría Gorenstein del h*.
    crt_dilations = range(half_degree + 2)
    counts: dict[int, int] = {}
    primes_by_dilation: dict[int, int] = {}
    for dilation in crt_dilations:
        counts[dilation], primes_by_dilation[dilation] = _count_by_crt(
            k, n, dilation
        )

    first_half: list[int] = []
    for dilation in range(half_degree + 1):
        coefficient = counts[dilation] - sum(
            first_half[index]
            * math.comb(dilation + dimension - index, dimension)
            for index in range(dilation)
        )
        first_half.append(coefficient)

    check_dilation = half_degree + 1
    check_coefficient = counts[check_dilation] - sum(
        first_half[index]
        * math.comb(check_dilation + dimension - index, dimension)
        for index in range(check_dilation)
    )
    if check_coefficient != first_half[degree - check_dilation]:
        raise ArithmeticError(
            f"La simetría de h* falló para C_{{{k},{n}}}: "
            f"{check_coefficient} != {first_half[degree - check_dilation]}."
        )
    h_star = [
        first_half[index if index <= half_degree else degree - index]
        for index in range(degree + 1)
    ]

    lattice_counts = []
    for dilation in range(dimension + 1):
        exact_count = sum(
            h_star[index]
            * math.comb(dilation + dimension - index, dimension)
            for index in range(min(dilation, degree) + 1)
        )
        if dilation in counts and exact_count != counts[dilation]:
            raise ArithmeticError(
                f"El conteo CRT y la reconstrucción Ehrhart difieren en m={dilation}."
            )
        lattice_counts.append(str(exact_count))

    normalized_volume = sum(h_star)
    return {
        "dimension": dimension,
        "degree_h_star": degree,
        "h_star_coefficients": [str(coefficient) for coefficient in h_star],
        "normalized_volume_H_1": str(normalized_volume),
        "euclidean_volume_exact": _reduced_fraction(
            normalized_volume, math.factorial(dimension)
        ),
        "euclidean_volume_unreduced": (
            f"{normalized_volume}/{math.factorial(dimension)}"
        ),
        "lattice_point_counts_L_0_to_L_d": lattice_counts,
        "crt_dilations": list(counts),
        "crt_prime_counts_by_dilation": {
            str(dilation): prime_count
            for dilation, prime_count in primes_by_dilation.items()
        },
        "crt_prime_count": max(primes_by_dilation.values()),
        "count_reconstruction_note": (
            "Las dilataciones CRT independientes hasta floor((d-k)/2)+1 "
            "reconstruyen h*; la simetría h*_i=h*_(d-k-i) se comprueba "
            "exactamente y L(0)..L(d) se evalúa desde ese h* exacto."
        ),
    }


def _reduced_fraction(numerator: int, denominator: int) -> str:
    divisor = math.gcd(numerator, denominator)
    return f"{numerator // divisor}/{denominator // divisor}"


def _predecessor_masks(k: int, n: int) -> tuple[int, ...]:
    predecessors = [0] * (k * n)
    for layer in range(k - 1):
        for position in range(n):
            lower = layer * n + position
            upper_same = (layer + 1) * n + position
            upper_next = (layer + 1) * n + (position + 1) % n
            predecessors[upper_same] |= 1 << lower
            predecessors[upper_next] |= 1 << lower
    return tuple(predecessors)


def _enumerate_linear_extensions_worker(
    k: int, n: int, result_queue: multiprocessing.queues.Queue
) -> None:
    """Enumera exactamente los símplices de la triangulación canónica."""
    predecessors = _predecessor_masks(k, n)
    full_mask = (1 << (k * n)) - 1
    count = 0

    def visit(placed: int) -> None:
        nonlocal count
        if placed == full_mask:
            count += 1
            return
        available = 0
        for vertex, required in enumerate(predecessors):
            bit = 1 << vertex
            if not placed & bit and required & placed == required:
                available |= bit
        while available:
            bit = available & -available
            available ^= bit
            visit(placed | bit)

    visit(0)
    result_queue.put(count)


def polyhedral_linear_extension_baseline(
    k: int, n: int, exact_simplex_count: int
) -> dict[str, Any]:
    """Cuenta extensiones lineales, deteniéndose al alcanzar 15 segundos."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_enumerate_linear_extensions_worker,
        args=(k, n, result_queue),
    )
    started = time.perf_counter()
    process.start()
    process.join(BASELINE_TIMEOUT_SECONDS)
    elapsed = time.perf_counter() - started
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join()
    enumerated_count: int | None = None
    enumeration_error: str | None = None
    if not timed_out:
        try:
            enumerated_count = int(result_queue.get(timeout=1))
        except queue.Empty as exc:
            enumeration_error = str(exc) or "El proceso no devolvió un conteo."
    result_queue.close()

    if enumerated_count is not None and enumerated_count != exact_simplex_count:
        raise ArithmeticError(
            "La enumeración poliédrica no coincide con H(1): "
            f"{enumerated_count} != {exact_simplex_count}."
        )
    simplex_bytes_packed = (
        exact_simplex_count * (k * n + 1) * ((k * n + 7) // 8)
    )
    simplex_bytes_dense = exact_simplex_count * (k * n + 1) * (k * n) * 8
    return {
        "method": "enumeración de extensiones lineales / símplices unimodulares",
        "timeout_seconds": BASELINE_TIMEOUT_SECONDS,
        "wall_time_seconds": round(elapsed, 6),
        "status": (
            "TIMEOUT > 15s"
            if timed_out
            else "completado"
            if enumerated_count is not None
            else "error"
        ),
        "enumerated_simplex_count": (
            str(enumerated_count) if enumerated_count is not None else None
        ),
        "exact_simplex_count_H_1": str(exact_simplex_count),
        "exact_count_source_on_timeout": (
            None if enumerated_count is not None else "Tensor-CRT: H_{k,n}(1)"
        ),
        "enumeration_error": enumeration_error,
        "explicit_triangulation_memory_estimate_bytes_bitpacked": str(
            simplex_bytes_packed
        ),
        "explicit_triangulation_memory_estimate_bytes_dense_int64": str(
            simplex_bytes_dense
        ),
        "memory_estimate_note": (
            "Carga útil de (d+1) vértices 0/1 por símplice; se informa "
            "almacenamiento bit-packed y almacenamiento denso int64, sin "
            "contar el overhead de contenedores."
        ),
    }


def _polytope_inequalities(k: int, n: int) -> list[list[int]]:
    dimension = k * n
    rows: list[list[int]] = []
    for vertex in range(dimension):
        lower_bound = [0] * (dimension + 1)
        lower_bound[vertex + 1] = 1
        rows.append(lower_bound)
        upper_bound = [0] * (dimension + 1)
        upper_bound[0] = 1
        upper_bound[vertex + 1] = -1
        rows.append(upper_bound)
    for layer in range(k - 1):
        for position in range(n):
            lower = layer * n + position
            for upper in (
                (layer + 1) * n + position,
                (layer + 1) * n + (position + 1) % n,
            ):
                row = [0] * (dimension + 1)
                row[upper + 1] = 1
                row[lower + 1] = -1
                rows.append(row)
    return rows


def _write_latte_input(path: Path, k: int, n: int) -> None:
    rows = _polytope_inequalities(k, n)
    content = ["H-representation", "begin", f"{len(rows)} {k * n + 1} integer"]
    content.extend(" ".join(str(value) for value in row) for row in rows)
    content.extend(("end", ""))
    path.write_text("\n".join(content), encoding="ascii")


def _write_normaliz_input(path: Path, k: int, n: int) -> None:
    rows = _polytope_inequalities(k, n)
    content = [f"amb_space {k * n}", "inhom_inequalities"]
    content.extend(
        " ".join(str(value) for value in row[1:] + row[:1]) for row in rows
    )
    content.extend(("end", ""))
    path.write_text("\n".join(content), encoding="ascii")


def external_tool_baselines(k: int, n: int) -> list[dict[str, Any]]:
    """Detecta y ejecuta Normaliz, LattE y PyNormaliz cuando estén presentes."""
    results: list[dict[str, Any]] = []
    normaliz = shutil.which("normaliz")
    latte = shutil.which("count")
    py_normaliz_module = next(
        (
            name
            for name in ("PyNormaliz", "pynormaliz")
            if importlib.util.find_spec(name) is not None
        ),
        None,
    )
    with tempfile.TemporaryDirectory(prefix="ehrhart_d40_") as temporary:
        folder = Path(temporary)
        if normaliz:
            input_path = folder / "polytope.in"
            _write_normaliz_input(input_path, k, n)
            results.append(
                _run_external(
                    "Normaliz",
                    [normaliz, str(input_path)],
                    BASELINE_TIMEOUT_SECONDS,
                    folder,
                )
            )
        if latte:
            input_path = folder / "polytope.ine"
            _write_latte_input(input_path, k, n)
            results.append(
                _run_external(
                    "LattE count",
                    [latte, str(input_path)],
                    BASELINE_TIMEOUT_SECONDS,
                    folder,
                )
            )
        if py_normaliz_module:
            input_path = folder / "inequalities.json"
            input_path.write_text(
                json.dumps(
                    [
                        row[1:] + row[:1]
                        for row in _polytope_inequalities(k, n)
                    ]
                ),
                encoding="ascii",
            )
            program = (
                f"import json,sys; from {py_normaliz_module} import NmzCone; "
                "rows=json.load(open(sys.argv[1])); "
                "cone=NmzCone(inhom_inequalities=rows); "
                "print(cone.EhrhartSeries())"
            )
            results.append(
                _run_external(
                    "PyNormaliz",
                    [sys.executable, "-c", program, str(input_path)],
                    BASELINE_TIMEOUT_SECONDS,
                    folder,
                )
            )
    return results


def _run_external(
    name: str, command: list[str], timeout: int, working_directory: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "tool": name,
            "status": f"TIMEOUT > {timeout}s",
            "wall_time_seconds": round(time.perf_counter() - started, 6),
            "error": str(exc),
        }
    return {
        "tool": name,
        "status": "completado" if completed.returncode == 0 else "error",
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "return_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _run_instance(k: int, n: int) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    exact = tensor_crt_volume_and_ehrhart(k, n)
    wall_time = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    baseline = polyhedral_linear_extension_baseline(
        k, n, int(exact["normalized_volume_H_1"])
    )
    external = external_tool_baselines(k, n)
    return {
        "k": k,
        "n": n,
        **exact,
        "tensor_crt_wall_time_seconds": round(wall_time, 6),
        "tensor_crt_tracemalloc_peak_mb": round(peak_bytes / (1024 * 1024), 3),
        "polyhedral_baseline": baseline,
        "external_tool_baselines": external,
    }


def _format_table(results: list[dict[str, Any]]) -> str:
    headers = (
        "k,n",
        "d",
        "primos",
        "Tensor-CRT s",
        "pico MB",
        "baseline s",
        "H(1)",
        "Volumen euclideo exacto",
    )
    rows = []
    for result in results:
        baseline = result["polyhedral_baseline"]
        baseline_time = (
            baseline["status"]
            if baseline["status"] != "completado"
            else f"{baseline['wall_time_seconds']:.3f}s"
        )
        rows.append(
            (
                f"{result['k']},{result['n']}",
                str(result["dimension"]),
                str(result["crt_prime_count"]),
                f"{result['tensor_crt_wall_time_seconds']:.3f}",
                f"{result['tensor_crt_tracemalloc_peak_mb']:.2f}",
                baseline_time,
                result["normalized_volume_H_1"],
                result["euclidean_volume_exact"],
            )
        )
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    header_line = " | ".join(
        headers[column].ljust(widths[column]) for column in range(len(headers))
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(
            row[column].ljust(widths[column]) for column in range(len(headers))
        )
        for row in rows
    ]
    return "\n".join((header_line, separator, *body))


def main() -> None:
    results: list[dict[str, Any]] = []
    for k, n in INSTANCES:
        result = _run_instance(k, n)
        results.append(result)
        print(
            f"Completado C_{{{k},{n}}} (d={k * n}): "
            f"H(1)={result['normalized_volume_H_1']}",
            flush=True,
        )
    payload = {
        "benchmark": "Tensor-CRT Ehrhart C_{k,n}, dimensiones hasta 40",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": sys.platform,
        "external_tools_detected": {
            "normaliz": shutil.which("normaliz"),
            "count": shutil.which("count"),
            "PyNormaliz": bool(
                importlib.util.find_spec("PyNormaliz")
                or importlib.util.find_spec("pynormaliz")
            ),
        },
        "instances": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(_format_table(results))
    print(f"\nResultados completos: {RESULTS_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
