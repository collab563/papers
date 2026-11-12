"""Reconstrucción exacta del numerador Ehrhart a partir de sus conteos."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import sympy as sp


_PROCESS_CONTRACTOR: Callable[[int], object] | None = None


def _initialize_process_worker(contractor_func: Callable[[int], object]) -> None:
    global _PROCESS_CONTRACTOR
    _PROCESS_CONTRACTOR = contractor_func


def _evaluate_process_worker(m: int) -> object:
    if _PROCESS_CONTRACTOR is None:
        raise RuntimeError("El contractor del proceso no fue inicializado.")
    return _PROCESS_CONTRACTOR(m)


def _as_integer_count(value: object, m: int) -> sp.Integer:
    """Valida un conteo y lo convierte a entero de SymPy sin perder exactitud."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"El conteo para m={m} no puede ser booleano.")
    if isinstance(value, (int, sp.Integer)):
        return sp.Integer(value)
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 2**53:
            raise ValueError(
                f"El conteo numérico para m={m} no es seguro para reconstrucción exacta."
            )
        nearest = round(value)
        if not math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                f"El conteo numérico para m={m} no está suficientemente cerca de un entero."
            )
        return sp.Integer(nearest)
    if isinstance(value, complex):
        if abs(value.imag) > 1e-7:
            raise ValueError(f"El conteo para m={m} tiene parte imaginaria no nula.")
        return _as_integer_count(float(value.real), m)
    if isinstance(value, sp.Rational) and value.q == 1:
        return sp.Integer(value)
    raise TypeError(f"El contractor retornó un conteo no entero para m={m}: {value!r}.")


def reconstruct_hstar_polynomial(
    k: int,
    n: int,
    contractor_func: Callable[[int], object],
    workers: int = 1,
) -> sp.Poly:
    """Reconstruye exactamente ``H_{k,n}(t)`` mediante la base binomial.

    Se evalúa ``L_{k,n}(m)`` hasta ``floor(d/2)+1``, donde ``d=k*(n-1)``.
    Los primeros coeficientes se recuperan por sustitución hacia delante y
    los restantes se completan mediante la simetría palíndroma, que se valida
    con el punto adicional.

    ``L(m) = sum(h[j] * binomial(m + d - j, d), j=0..m)``.

    Los valores de entrada deben ser enteros exactos o valores flotantes que
    representen con seguridad un entero. El polinomio resultante tiene dominio
    entero y se retorna como ``sympy.Poly``. Con ``workers > 1``, las
    dilataciones se evalúan en procesos independientes.

    Args:
        workers: Número de procesos para evaluar los conteos de las dilataciones.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k debe ser un entero mayor o igual que 2.")
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("n debe ser un entero mayor o igual que 3.")
    if not callable(contractor_func):
        raise TypeError("contractor_func debe ser invocable con un argumento m.")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers debe ser un entero positivo.")

    degree = k * (n - 1)
    half_degree = degree // 2
    dilations = range(half_degree + 2)
    if workers == 1:
        evaluated_counts = map(contractor_func, dilations)
        counts = [
            _as_integer_count(value, m)
            for m, value in zip(dilations, evaluated_counts)
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(dilations)),
            initializer=_initialize_process_worker,
            initargs=(contractor_func,),
        ) as executor:
            evaluated_counts = executor.map(_evaluate_process_worker, dilations)
            counts = [
                _as_integer_count(value, m)
                for m, value in zip(dilations, evaluated_counts)
            ]
    half_coefficients: list[sp.Integer] = []
    for m, count in enumerate(counts):
        coefficient = sp.Integer(count) - sum(
            half_coefficients[index]
            * sp.binomial(m + k * n - index, k * n)
            for index in range(m)
        )
        half_coefficients.append(sp.Integer(coefficient))

    mirror_index = (degree - 1) // 2
    if half_coefficients[-1] != half_coefficients[mirror_index]:
        raise ValueError(
            "El conteo extra no confirma la simetría palíndroma del h*-polinomio."
        )
    coefficients = [
        half_coefficients[index if index <= half_degree else degree - index]
        for index in range(degree + 1)
    ]
    variable = sp.Symbol("t")
    expression = sum(
        coefficient * variable**index
        for index, coefficient in enumerate(coefficients)
    )
    return sp.Poly(expression, variable, domain=sp.ZZ)