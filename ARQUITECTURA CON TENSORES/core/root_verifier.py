"""Verificación exacta de raíces reales negativas y reciprocidad."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp


def _variation_count(signs: list[int]) -> int:
    """Cuenta cambios de signo omitiendo los ceros, como en la regla de Sturm."""
    nonzero = [sign for sign in signs if sign]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def verify_real_rootedness(
    H_poly: Any,
    k: int | None = None,
    n: int | None = None,
    export_json: str | Path | None = None,
) -> dict[str, Any]:
    """Verifica el polinomio con Sturm y prepara un certificado JSON.

    La cuenta de Sturm se calcula exactamente sobre los racionales en
    ``(-infinito, 0)``. La simplicidad se comprueba mediante el máximo común
    divisor con la derivada. Si se proporcionan ``k`` y ``n``, también se
    valida que el grado sea ``k*n-k`` antes de comprobar la simetría
    recíproca. Sin ellos, la simetría se prueba respecto al grado efectivo.

    Args:
        H_poly: Expresión SymPy o ``sympy.Poly`` de coeficientes racionales.
        k: Número opcional de capas.
        n: Tamaño opcional de la circunferencia.
        export_json: Ruta opcional a la que escribir el certificado.

    Returns:
        Diccionario con el reporte, resultados booleanos y certificado de
        signos de la secuencia de Sturm.
    """
    if (k is None) != (n is None):
        raise ValueError("k y n deben proporcionarse juntos.")
    if k is not None and (
        isinstance(k, bool)
        or isinstance(n, bool)
        or not isinstance(k, int)
        or not isinstance(n, int)
        or k < 2
        or n < 3
    ):
        raise ValueError("Se requieren k >= 2 y n >= 3.")

    polynomial = H_poly if isinstance(H_poly, sp.Poly) else sp.Poly(H_poly)
    if polynomial.is_zero:
        raise ValueError("El polinomio cero no tiene un conjunto finito de raíces.")
    variable = polynomial.gen
    degree = polynomial.degree()
    expected_degree = k * n - k if k is not None and n is not None else degree
    degree_matches = degree == expected_degree
    coefficients = polynomial.all_coeffs()
    reciprocal = degree_matches and all(
        coefficients[index] == coefficients[-1 - index]
        for index in range(len(coefficients))
    )

    sturm_sequence = polynomial.sturm()
    signs_at_negative_infinity: list[int] = []
    signs_at_zero: list[int] = []
    for member in sturm_sequence:
        leading_sign = int(sp.sign(member.LC()))
        sign_at_infinity = leading_sign * (-1 if member.degree() % 2 else 1)
        signs_at_negative_infinity.append(sign_at_infinity)
        signs_at_zero.append(int(sp.sign(member.eval(0))))

    negative_root_count = int(polynomial.count_roots(-sp.oo, 0))
    simple = degree == 0 or polynomial.gcd(polynomial.diff()).degree() == 0
    all_roots_negative_simple = negative_root_count == degree and simple
    certificate = {
        "method": "Sturm exacto sobre Q en (-infinito, 0)",
        "polynomial_coefficients_descending": [str(c) for c in coefficients],
        "sturm_sequence_coefficients_descending": [
            [str(coefficient) for coefficient in member.all_coeffs()]
            for member in sturm_sequence
        ],
        "signs_at_negative_infinity": signs_at_negative_infinity,
        "signs_at_zero": signs_at_zero,
        "variations_at_negative_infinity": _variation_count(
            signs_at_negative_infinity
        ),
        "variations_at_zero": _variation_count(signs_at_zero),
        "negative_root_count": negative_root_count,
        "degree": degree,
        "roots_simple": simple,
        "roots_all_real_simple_strictly_negative": all_roots_negative_simple,
        "expected_degree": expected_degree,
        "degree_matches_expected": degree_matches,
        "reciprocal_symmetry": reciprocal,
    }
    report_lines = [
        f"Polinomio: {polynomial.as_expr()}",
        f"Grado: {degree} (esperado: {expected_degree}; coincide: {'sí' if degree_matches else 'no'})",
        f"Raíces negativas contadas por Sturm: {negative_root_count}/{degree}",
        f"Raíces simples: {'sí' if simple else 'no'}",
        "Todas las raíces son reales, simples y estrictamente negativas: "
        f"{'sí' if all_roots_negative_simple else 'no'}",
        f"Simetría recíproca: {'sí' if reciprocal else 'no'}",
    ]
    result: dict[str, Any] = {
        "polynomial": str(polynomial.as_expr()),
        "is_real_rooted_negative_simple": all_roots_negative_simple,
        "is_reciprocal": reciprocal,
        "report": "\n".join(report_lines),
        "sign_certificate": certificate,
    }
    if export_json is not None:
        path = Path(export_json)
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result["exported_json"] = str(path)
    return result