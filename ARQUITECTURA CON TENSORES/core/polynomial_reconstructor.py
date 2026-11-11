"""Reconstrucción exacta del numerador Ehrhart a partir de sus conteos."""

from __future__ import annotations

import math
from typing import Callable

import sympy as sp


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
    k: int, n: int, contractor_func: Callable[[int], object]
) -> sp.Poly:
    """Reconstruye exactamente ``H_{k,n}(t)`` mediante la base binomial.

    Se evalúa ``L_{k,n}(m)`` para cada ``m = 0, ..., k*n`` y se resuelve en
    SymPy el sistema triangular

    ``L(m) = sum(h[j] * binomial(m + k*n - j, k*n), j=0..k*n)``.

    Los valores de entrada deben ser enteros exactos o valores flotantes que
    representen con seguridad un entero. El polinomio resultante tiene dominio
    entero y se retorna como ``sympy.Poly``.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k debe ser un entero mayor o igual que 2.")
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError("n debe ser un entero mayor o igual que 3.")
    if not callable(contractor_func):
        raise TypeError("contractor_func debe ser invocable con un argumento m.")

    dimension = k * n
    counts = [
        _as_integer_count(contractor_func(m), m) for m in range(dimension + 1)
    ]
    basis = sp.Matrix(
        [
            [
                sp.binomial(m + dimension - j, dimension)
                for j in range(dimension + 1)
            ]
            for m in range(dimension + 1)
        ]
    )
    coefficients = basis.LUsolve(sp.Matrix(counts))
    variable = sp.Symbol("t")
    expression = sum(
        coefficient * variable**index
        for index, coefficient in enumerate(coefficients)
    )
    return sp.Poly(expression, variable, domain=sp.ZZ)