"""Pruebas de conteos Ehrhart y numeradores h* conocidos."""

from pathlib import Path
import sys

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.contraction_engine import BoundaryMPSContractor
from core.interlacing_analyzer import (
    check_strict_interlacing,
    find_polynomial_recurrence,
    load_certified_family,
)
from core.polynomial_reconstructor import reconstruct_hstar_polynomial


def test_hstar_k2_n3() -> None:
    contractor = BoundaryMPSContractor(2, 3)
    polynomial = reconstruct_hstar_polynomial(2, 3, contractor)
    expected = sp.Poly(
        sp.Symbol("t") ** 4
        + 11 * sp.Symbol("t") ** 3
        + 24 * sp.Symbol("t") ** 2
        + 11 * sp.Symbol("t")
        + 1,
        sp.Symbol("t"),
    )
    assert polynomial == expected


def test_hstar_k3_n3() -> None:
    contractor = BoundaryMPSContractor(3, 3)
    polynomial = reconstruct_hstar_polynomial(3, 3, contractor)
    t = sp.Symbol("t")
    expected = sp.Poly(
        t**6 + 18 * t**5 + 93 * t**4 + 160 * t**3 + 93 * t**2 + 18 * t + 1,
        t,
    )
    assert polynomial == expected


def test_exact_ehrhart_counts_k3_n3() -> None:
    contractor = BoundaryMPSContractor(3, 3)
    assert contractor(1) == 28
    assert contractor(2) == 328


def test_certified_interlacing_checks_degree_gap() -> None:
    certificates = Path(__file__).resolve().parents[1] / "certificados_json"
    family = load_certified_family(certificates, ((3, 3), (4, 3)))
    interlaces, order, reason = check_strict_interlacing(
        family[(3, 3)], family[(4, 3)]
    )
    assert not interlaces
    assert len(order) == 6 + 8
    assert "grados difieren en más de uno" in reason


def test_exact_order_two_recurrence_for_n3() -> None:
    certificates = Path(__file__).resolve().parents[1] / "certificados_json"
    levels = tuple((k, 3) for k in range(2, 9))
    family = load_certified_family(certificates, levels)
    result = find_polynomial_recurrence(family, n=3, order=2)
    t = sp.Symbol("t")

    assert result.coefficient_polynomials == (
        t**2 + 6 * t + 1,
        sp.expand(t * (t - 1) ** 2),
    )
    assert result.degree_bound == 3
    assert result.unique


def test_order_three_recurrence_is_not_unique_on_n3_samples() -> None:
    certificates = Path(__file__).resolve().parents[1] / "certificados_json"
    levels = tuple((k, 3) for k in range(2, 9))
    family = load_certified_family(certificates, levels)
    result = find_polynomial_recurrence(family, n=3, order=3)

    assert result.coefficient_polynomials is not None
    assert not result.unique
    assert result.degree_bound == 3


def test_recurrence_search_checks_n4_orders_two_and_three_bounds() -> None:
    certificates = Path(__file__).resolve().parents[1] / "certificados_json"
    levels = tuple((k, 4) for k in range(3, 8))
    family = load_certified_family(certificates, levels)
    order_two = find_polynomial_recurrence(
        family,
        n=4,
        order=2,
        coefficient_degree_bounds=(4, 8),
        reciprocal_spans=(3, 6),
    )
    order_three = find_polynomial_recurrence(
        family,
        n=4,
        order=3,
        coefficient_degree_bounds=(4, 7, 10),
        reciprocal_spans=(3, 6, 9),
    )

    assert order_two.coefficient_polynomials is None
    assert "cotas y simetrías especificadas" in order_two.message
    assert order_two.degree_bounds == (4, 8)

    assert order_three.degree_bounds == (4, 7, 10)
    assert order_three.verified_levels == (6, 7)
    assert order_three.coefficient_polynomials is not None
    assert order_three.unique
    assert order_three.integer_coefficients
    assert order_three.reciprocal_shifts == (0, 0, 0)
    t = sp.Symbol("t")
    p, q, r = order_three.coefficient_polynomials
    for k in order_three.verified_levels:
        expected = family[(k, 4)].polynomial.as_expr()
        recurrence = (
            p * family[(k - 1, 4)].polynomial.as_expr()
            + q * family[(k - 2, 4)].polynomial.as_expr()
            + r * family[(k - 3, 4)].polynomial.as_expr()
        )
        assert sp.expand(expected - recurrence) == 0
    for coefficient, reciprocal_span, degree_bound in zip(
        (p, q, r), (3, 6, 9), (4, 7, 10)
    ):
        assert sp.degree(coefficient, t) <= degree_bound
        assert all(
            coefficient.is_integer
            for coefficient in sp.Poly(coefficient, t).all_coeffs()
        )
        assert sp.expand(
            coefficient - t**reciprocal_span * coefficient.subs(t, 1 / t)
        ) == 0