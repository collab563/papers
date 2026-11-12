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


def _integer_sturm_sequence(
    polynomial: sp.Poly,
) -> tuple[list[sp.Poly], list[sp.Rational]]:
    """Calcula una cadena de Sturm primitiva, conservando sus escalas exactas."""
    denominator, integer_polynomial = polynomial.monic().clear_denoms(convert=True)
    content, first = integer_polynomial.primitive()
    sequence = [first]
    scales = [sp.Rational(content, denominator)]
    if first.degree() == 0:
        return sequence, scales

    derivative_content, second = first.diff().primitive()
    sequence.append(second)
    scales.append(scales[0] * derivative_content)

    while sequence[-1].degree() > 0:
        previous, current = sequence[-2:]
        exponent = previous.degree() - current.degree() + 1
        pseudo_remainder = previous.prem(current)
        if pseudo_remainder.is_zero:
            break
        if current.LC() < 0 and exponent % 2:
            pseudo_remainder = -pseudo_remainder

        content, following = (-pseudo_remainder).primitive()
        if following.is_zero:
            break
        sequence.append(following)
        scales.append(
            scales[-2]
            * sp.Rational(content, abs(current.LC()) ** exponent)
        )
    return sequence, scales


def _sturm_sign_data(
    sequence: list[sp.Poly],
) -> tuple[list[int], list[int], int, int]:
    signs_at_negative_infinity: list[int] = []
    signs_at_zero: list[int] = []
    for member in sequence:
        leading_sign = int(sp.sign(member.LC()))
        sign_at_infinity = leading_sign * (-1 if member.degree() % 2 else 1)
        signs_at_negative_infinity.append(sign_at_infinity)
        signs_at_zero.append(int(sp.sign(member.eval(0))))
    variations_at_negative_infinity = _variation_count(signs_at_negative_infinity)
    variations_at_zero = _variation_count(signs_at_zero)
    return (
        signs_at_negative_infinity,
        signs_at_zero,
        variations_at_negative_infinity,
        variations_at_zero,
    )


def _gamma_polynomial(polynomial: sp.Poly) -> sp.Poly:
    """Obtiene Gamma de H(t)=(1+t)^d Gamma(t/(1+t)^2)."""
    degree = polynomial.degree()
    coefficients_ascending = list(reversed(polynomial.all_coeffs()))
    gamma_coefficients: list[sp.Expr] = []
    for index in range(degree // 2 + 1):
        known_terms = sum(
            gamma_coefficients[previous]
            * sp.binomial(degree - 2 * previous, index - previous)
            for previous in range(index)
        )
        gamma_coefficients.append(coefficients_ascending[index] - known_terms)

    variable = sp.Symbol("x")
    expression = sum(
        coefficient * variable**index
        for index, coefficient in enumerate(gamma_coefficients)
    )
    return sp.Poly(expression, variable, domain=polynomial.domain)


def verify_real_rootedness(
    H_poly: Any,
    k: int | None = None,
    n: int | None = None,
    export_json: str | Path | None = None,
    use_gamma: bool = False,
) -> dict[str, Any]:
    """Verifica el polinomio con Sturm y prepara un certificado JSON.

    La secuencia de Sturm se calcula con pseudo-residuos enteros primitivos.
    Si ``use_gamma`` es verdadero, un polinomio par palindrómico se certifica
    mediante su polinomio reducido Gamma.

    Args:
        H_poly: Expresión SymPy o ``sympy.Poly`` de coeficientes racionales.
        k: Número opcional de capas.
        n: Tamaño opcional de la circunferencia.
        export_json: Ruta opcional a la que escribir el certificado.
        use_gamma: Certifica mediante Gamma si el grado es par y H es palíndromo.

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
    degree = polynomial.degree()
    expected_degree = k * n - k if k is not None and n is not None else degree
    degree_matches = degree == expected_degree
    coefficients = polynomial.all_coeffs()
    reciprocal = degree_matches and all(
        coefficients[index] == coefficients[-1 - index]
        for index in range(len(coefficients))
    )

    sturm_polynomial = polynomial
    gamma: sp.Poly | None = None
    if use_gamma:
        if degree % 2 or not reciprocal:
            raise ValueError(
                "La certificación Gamma requiere grado par y simetría recíproca."
            )
        gamma = _gamma_polynomial(polynomial)
        sturm_polynomial = gamma

    sturm_sequence, sturm_scales = _integer_sturm_sequence(sturm_polynomial)
    (
        signs_at_negative_infinity,
        signs_at_zero,
        variations_at_negative_infinity,
        variations_at_zero,
    ) = _sturm_sign_data(sturm_sequence)
    sturm_root_count = variations_at_negative_infinity - variations_at_zero
    sturm_simple = (
        sturm_polynomial.degree() == 0
        or sturm_polynomial.gcd(sturm_polynomial.diff()).degree() == 0
    )
    if gamma is None:
        negative_root_count = sturm_root_count
        simple = sturm_simple
    else:
        gamma_degree = gamma.degree()
        negative_root_count = 2 * sturm_root_count + int(
            gamma_degree < degree // 2
        )
        simple = (
            gamma_degree == degree // 2
            and sturm_simple
            and sturm_root_count == gamma_degree
        )
    all_roots_negative_simple = negative_root_count == degree and simple
    sturm_sequence_coefficients = [
        [
            str(scale * coefficient)
            for coefficient in member.all_coeffs()
        ]
        for member, scale in zip(sturm_sequence, sturm_scales)
    ]
    certificate = {
        "method": (
            "Sturm exacto sobre Gamma en (-infinito, 0)"
            if gamma is not None
            else "Sturm exacto sobre Q en (-infinito, 0)"
        ),
        "polynomial_coefficients_descending": [str(c) for c in coefficients],
        "sturm_sequence_coefficients_descending": sturm_sequence_coefficients,
        "signs_at_negative_infinity": signs_at_negative_infinity,
        "signs_at_zero": signs_at_zero,
        "variations_at_negative_infinity": variations_at_negative_infinity,
        "variations_at_zero": variations_at_zero,
        "negative_root_count": negative_root_count,
        "degree": degree,
        "roots_simple": simple,
        "roots_all_real_simple_strictly_negative": all_roots_negative_simple,
        "expected_degree": expected_degree,
        "degree_matches_expected": degree_matches,
        "reciprocal_symmetry": reciprocal,
    }
    if gamma is not None:
        certificate["reduced_polynomial_coefficients_descending"] = [
            str(coefficient) for coefficient in gamma.all_coeffs()
        ]
    report_lines = [
        f"Polinomio: {polynomial.as_expr()}",
        f"Grado: {degree} (esperado: {expected_degree}; coincide: {'sí' if degree_matches else 'no'})",
        (
            f"Raíces negativas contadas por Sturm"
            f"{' sobre Gamma' if gamma is not None else ''}: "
            f"{negative_root_count}/{degree}"
        ),
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