"""Pruebas de conteos Ehrhart y numeradores h* conocidos."""

from pathlib import Path
import sys

import sympy as sp
import opt_einsum as oe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.contraction_engine import BoundaryMPSContractor
from core.interlacing_analyzer import (
    CertifiedPolynomial,
    _find_unrestricted_recurrence,
    _check_integer_recurrence_at_one,
    _integer_point_on_affine_line,
    _modular_consistency,
    _n5_levels_in_directory,
    _scalar_recurrence_at_one,
    check_strict_interlacing,
    find_polynomial_recurrence,
    load_certified_family,
)
from core.polynomial_reconstructor import reconstruct_hstar_polynomial
from core.root_verifier import verify_real_rootedness


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


def test_parallel_reconstruction_matches_sequential() -> None:
    contractor = BoundaryMPSContractor(2, 3)
    sequential = reconstruct_hstar_polynomial(2, 3, contractor)
    parallel = reconstruct_hstar_polynomial(2, 3, contractor, workers=2)

    assert parallel == sequential


def test_reconstruction_uses_only_half_dilations_and_checks_symmetry() -> None:
    contractor = BoundaryMPSContractor(2, 3)
    polynomial = reconstruct_hstar_polynomial(2, 3, contractor)

    assert polynomial.degree() == 4
    assert sorted(contractor._count_cache) == list(range(4))
    assert polynomial.all_coeffs() == list(reversed(polynomial.all_coeffs()))


def test_crt_contraction_matches_integer_object_contraction() -> None:
    contractor = BoundaryMPSContractor(3, 3)

    for dilation in range(3):
        operands = contractor._build_operands(dilation)
        expected = int(oe.contract(*operands, optimize="auto-hq", backend="numpy"))
        assert contractor.contract_crt(dilation) == expected


def test_integer_sturm_preserves_the_exact_certificate_sequence() -> None:
    t = sp.Symbol("t")
    polynomial = sp.Poly(t**4 + 11 * t**3 + 24 * t**2 + 11 * t + 1, t)
    result = verify_real_rootedness(polynomial)

    assert result["sign_certificate"]["sturm_sequence_coefficients_descending"] == [
        [str(coefficient) for coefficient in member.all_coeffs()]
        for member in polynomial.sturm()
    ]


def test_gamma_certifies_a_palindromic_polynomial() -> None:
    t = sp.Symbol("t")
    polynomial = sp.Poly(t**4 + 11 * t**3 + 24 * t**2 + 11 * t + 1, t)
    result = verify_real_rootedness(polynomial, use_gamma=True)
    certificate = result["sign_certificate"]

    assert result["is_real_rooted_negative_simple"]
    assert certificate["method"] == "Sturm exacto sobre Gamma en (-infinito, 0)"
    assert certificate["reduced_polynomial_coefficients_descending"] == [
        "4",
        "7",
        "1",
    ]

    negative_roots = sp.Poly(t**2 + 6 * t + 1, t)
    negative_result = verify_real_rootedness(negative_roots, use_gamma=True)
    assert negative_result["is_real_rooted_negative_simple"]
    assert negative_result["sign_certificate"]["negative_root_count"] == 2

    repeated_roots = sp.Poly((1 + t) ** 4, t)
    repeated_result = verify_real_rootedness(repeated_roots, use_gamma=True)
    assert not repeated_result["is_real_rooted_negative_simple"]
    assert repeated_result["sign_certificate"]["negative_root_count"] == 1


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


def test_n5_order_four_search_reports_free_parameters() -> None:
    t = sp.Symbol("t")
    family = {
        (k, 5): CertifiedPolynomial(k, 5, sp.Poly(1, t), Path(f"n5-{k}.json"))
        for k in range(2, 7)
    }

    result = find_polynomial_recurrence(family, n=5, order=4)

    assert result.coefficient_polynomials is not None
    assert result.equations > 0
    assert not result.unique
    assert result.free_parameters > 0
    assert result.coefficient_symmetries is not None
    assert result.coefficient_symmetries[0] == "recíproca"


def test_n5_order_four_without_transitions_reports_model_dimension() -> None:
    family = load_certified_family(
        Path(__file__).resolve().parents[1] / "certificados_json",
        ((3, 5), (4, 5)),
    )

    result = find_polynomial_recurrence(family, n=5, order=4)

    assert result.coefficient_polynomials is None
    assert result.equations == 0
    assert result.free_parameters == 24
    assert "datos insuficientes" in result.message


def test_n5_order_four_recovers_an_exact_expanded_recurrence() -> None:
    t = sp.Symbol("t")
    expected = (
        t**4 + 1,
        t**9 + t,
        t**13 + t,
        t**17 + t,
    )
    polynomials = {
        2: sp.Poly(1 + t, t),
        3: sp.Poly(2 + t**2, t),
        4: sp.Poly(3 + t**3, t),
        5: sp.Poly(4 + t**4, t),
    }
    for k in (6, 7):
        polynomials[k] = sp.Poly(
            sp.expand(
                sum(
                    expected[index - 1]
                    * polynomials[k - index].as_expr()
                    for index in range(1, 5)
                )
            ),
            t,
        )
    family = {
        (k, 5): CertifiedPolynomial(k, 5, polynomial, Path(f"n5-{k}.json"))
        for k, polynomial in polynomials.items()
    }

    result = find_polynomial_recurrence(family, n=5, order=4)

    assert result.coefficient_polynomials == expected
    assert result.unique
    assert result.integer_coefficients
    assert result.free_parameters == 0
    assert result.verified_levels == (6, 7)


def test_n5_unrestricted_recurrence_reports_exact_rank_and_solution() -> None:
    t = sp.Symbol("t")
    operators = (t**4, t**8, t**12, t**16)
    polynomials = {
        2: sp.Poly(1 + t, t),
        3: sp.Poly(2 + t**2, t),
        4: sp.Poly(3 + t**3, t),
        5: sp.Poly(4 + t**4, t),
    }
    for k in (6, 7):
        polynomials[k] = sp.Poly(
            sp.expand(
                sum(
                    operators[index - 1]
                    * polynomials[k - index].as_expr()
                    for index in range(1, 5)
                )
            ),
            t,
        )
    family = {
        (k, 5): CertifiedPolynomial(k, 5, polynomial, Path(f"n5-{k}.json"))
        for k, polynomial in polynomials.items()
    }

    result = _find_unrestricted_recurrence(family, 5, (4, 8, 12, 16))

    assert result.coefficient_polynomials == operators
    assert not result.unique
    assert result.integer_coefficients is None
    assert result.rank is not None
    assert result.rank < result.unknowns
    assert result.augmented_rank == result.rank
    assert result.verified_levels == (6, 7)

    expanded = _find_unrestricted_recurrence(
        family, 5, (6, 14, 20, 26)
    )
    assert expanded.unknowns == 70
    assert expanded.rank is not None
    assert expanded.augmented_rank == expanded.rank
    assert expanded.coefficient_polynomials is not None
    for k in expanded.verified_levels:
        reconstructed = sum(
            expanded.coefficient_polynomials[index - 1]
            * family[(k - index, 5)].polynomial.as_expr()
            for index in range(1, 5)
        )
        assert sp.expand(
            family[(k, 5)].polynomial.as_expr() - reconstructed
        ) == 0


def test_n5_order_five_reports_rank_without_transitions() -> None:
    family = load_certified_family(
        Path(__file__).resolve().parents[1] / "certificados_json",
        ((3, 5), (4, 5)),
    )

    result = find_polynomial_recurrence(family, n=5, order=5)

    assert result.coefficient_polynomials is None
    assert result.order == 5
    assert result.unknowns == 35
    assert result.rank == 0
    assert result.augmented_rank == 0
    assert result.free_parameters == 35


def test_n5_symmetric_order_five_uses_k7_k8_transitions() -> None:
    t = sp.Symbol("t")
    operators = (
        t**4 + 1,
        t + t**9,
        t + t**13,
        t + t**17,
        t + t**21,
    )
    polynomials = {
        k: sp.Poly((1 + t) ** (4 * k), t) for k in range(2, 7)
    }
    for k in (7, 8):
        polynomials[k] = sp.Poly(
            sp.expand(
                sum(
                    operators[index - 1]
                    * polynomials[k - index].as_expr()
                    for index in range(1, 6)
                )
            ),
            t,
        )
    family = {
        (k, 5): CertifiedPolynomial(k, 5, polynomial, Path(f"n5-{k}.json"))
        for k, polynomial in polynomials.items()
    }

    result = find_polynomial_recurrence(family, n=5, order=5)

    assert result.equations > 0
    assert result.verified_levels == (7, 8)
    assert result.rank is not None
    assert result.augmented_rank == result.rank
    assert result.unknowns == result.rank + result.free_parameters
    assert result.coefficient_polynomials is not None
    assert len(result.coefficient_polynomials) == 5


def test_integer_point_on_rational_affine_line_uses_bezout_congruences() -> None:
    point = _integer_point_on_affine_line(
        (sp.Rational(1, 3), sp.Rational(2, 3)),
        (sp.Rational(2, 1), sp.Rational(-2, 1)),
    )

    assert point is not None
    (
        integer_solution,
        primitive_direction,
        numerator,
        denominator,
        direction_shift,
    ) = point
    assert integer_solution == (0, 1)
    assert primitive_direction == (1, -1)
    assert numerator == -1
    assert denominator == 3
    assert direction_shift == 0


def test_integer_point_on_affine_line_is_reduced_to_minimum_norm() -> None:
    point = _integer_point_on_affine_line(
        (sp.Rational(10, 3), sp.Rational(-7, 3)),
        (sp.Rational(1), sp.Rational(-1)),
    )

    assert point is not None
    integer_solution, primitive_direction, numerator, denominator, shift = point
    assert integer_solution == (0, 1)
    assert primitive_direction == (1, -1)
    assert numerator == -1
    assert denominator == 3
    assert shift == -3
    norm_squared = sum(value * value for value in integer_solution)
    assert all(
        sum(
            (value + delta * coefficient) ** 2
            for value, coefficient in zip(integer_solution, primitive_direction)
        )
        >= norm_squared
        for delta in (-1, 1)
    )


def test_integer_point_on_affine_line_can_have_no_solution() -> None:
    point = _integer_point_on_affine_line(
        (sp.Rational(1, 2), sp.Rational(0)),
        (sp.Rational(2), sp.Rational(1)),
    )

    assert point is None


def test_integer_order_five_operators_are_checked_at_one() -> None:
    t = sp.Symbol("t")
    operators = (t + 1, t + 2, t + 3, t**2 - 1, t**2 - 1)

    s_at_one, t_at_one, matches = _check_integer_recurrence_at_one(
        operators,
        3,
        (sp.Rational(2), sp.Rational(3), sp.Rational(4)),
        0,
    )

    assert s_at_one == 0
    assert t_at_one == 0
    assert matches


def test_n5_scalar_recurrence_at_one_is_exact_but_unvalidated() -> None:
    family = load_certified_family(
        Path(__file__).resolve().parents[1] / "certificados_json",
        ((3, 5), (4, 5)),
    )

    order, coefficients, equations, free_parameters, levels = (
        _scalar_recurrence_at_one(family, 5)
    )

    assert order == 1
    assert coefficients == (sp.Rational(484096, 1365),)
    assert equations == 1
    assert free_parameters == 0
    assert levels == (4,)


def test_n5_directory_discovery_includes_levels_above_seven(
    tmp_path: Path,
) -> None:
    (tmp_path / "certificado_3_5.json").touch()
    (tmp_path / "certificado_8_5.json").touch()
    (tmp_path / "certificado_bad_5.json").touch()

    assert _n5_levels_in_directory(tmp_path) == ((3, 5), (8, 5))


def test_modular_inconsistency_is_not_misreported_as_q_inconsistency() -> None:
    prime = 2_305_843_009_213_693_951
    matrix = sp.Matrix([[prime]])
    vector = sp.Matrix([1])

    consistent, rank, augmented_rank = _modular_consistency(
        matrix, vector, 1, prime
    )

    assert not consistent
    assert rank == 0
    assert augmented_rank == 1