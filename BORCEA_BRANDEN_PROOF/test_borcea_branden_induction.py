"""Exact finite checks for refined order-ideal polynomials of cyclic crowns."""

from __future__ import annotations

import itertools
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(__file__).resolve().with_name("resultados_borcea_branden.json")
CASES = ((3, 3), (4, 3), (3, 4), (4, 4), (3, 5))
ROOT_ISOLATION_EPSILON = sp.Rational(1, 10**12)

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

T, U, V = sp.symbols("t u v")
Ideal = frozenset[int]


def _all_ideals(k: int, n: int) -> list[tuple[tuple[int, ...], Ideal]]:
    result: list[tuple[tuple[int, ...], Ideal]] = []
    for heights in itertools.product(range(k + 1), repeat=n):
        if not all(
            heights[i] <= heights[(i - 1) % n] + 1
            for i in range(n)
        ):
            continue
        ideal = frozenset(
            layer * n + column
            for column, height in enumerate(heights)
            for layer in range(height)
        )
        result.append((heights, ideal))
    return result


def _successors(k: int, n: int) -> dict[int, tuple[int, ...]]:
    return {
        vertex: tuple(
            (vertex // n + 1) * n + next_column
            for next_column in (vertex % n, (vertex % n + 1) % n)
            if vertex // n + 1 < k
        )
        for vertex in range(k * n)
    }


def _maximal_elements(
    ideal: Ideal,
    successors: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            vertex
            for vertex in ideal
            if not any(successor in ideal for successor in successors[vertex])
        )
    )


def _canonical_origin(heights: tuple[int, ...]) -> int:
    """Put a unique proper maximum-height cyclic arc at coordinate zero."""
    maximum = max(heights, default=0)
    starts = [
        i
        for i, height in enumerate(heights)
        if height == maximum and heights[(i - 1) % len(heights)] < maximum
    ]
    if len(starts) == 1 and any(height < maximum for height in heights):
        return starts[0]
    return 0


def _label(vertex: int, n: int, origin: int) -> int:
    layer, column = divmod(vertex, n)
    return layer * n + (column - origin) % n + 1


def _ordered_maxima(
    maxima: tuple[int, ...],
    n: int,
    origin: int,
) -> tuple[int, ...]:
    return tuple(sorted(maxima, key=lambda vertex: _label(vertex, n, origin)))


def _polynomial_vector(
    n: int,
    successors: dict[int, tuple[int, ...]],
) -> Any:
    @lru_cache(maxsize=None)
    def refined(ideal: Ideal, origin: int) -> tuple[tuple[int, sp.Expr], ...]:
        maxima = _maximal_elements(ideal, successors)
        if not ideal:
            return ()
        if len(ideal) == 1:
            return ((maxima[0], sp.Integer(1)),)

        values: dict[int, sp.Expr] = {}
        for maximal in maxima:
            predecessor = ideal - {maximal}
            predecessor_values = dict(refined(predecessor, origin))
            label_m = _label(maximal, n, origin)
            value = sp.Integer(0)
            for previous_maximal, previous_polynomial in predecessor_values.items():
                if _label(previous_maximal, n, origin) < label_m:
                    value += previous_polynomial
                else:
                    value += T * previous_polynomial
            values[maximal] = sp.Poly(sp.expand(value), T, domain=sp.ZZ).as_expr()
        return tuple(sorted(values.items()))

    return refined


def _linear_extension_count(
    successors: dict[int, tuple[int, ...]],
) -> Any:
    @lru_cache(maxsize=None)
    def count(ideal: Ideal) -> int:
        if not ideal:
            return 1
        return sum(
            count(ideal - {maximal})
            for maximal in _maximal_elements(ideal, successors)
        )

    return count


def _root_certificate(polynomial: sp.Poly) -> dict[str, Any]:
    if polynomial.is_zero:
        return {
            "degree": -1,
            "real_root_count_with_multiplicity": 0,
            "positive_root_count": 0,
            "only_real_nonpositive_roots": False,
            "isolating_intervals": [],
        }

    intervals = polynomial.intervals(eps=ROOT_ISOLATION_EPSILON)
    real_root_count = sum(multiplicity for _, multiplicity in intervals)
    reversed_coefficients = list(reversed(polynomial.all_coeffs()))
    zero_multiplicity = next(
        (
            index
            for index, coefficient in enumerate(reversed_coefficients)
            if coefficient != 0
        ),
        polynomial.degree() + 1,
    )
    positive_root_count = int(
        polynomial.count_roots(0, sp.oo) - zero_multiplicity
    )
    only_nonpositive = (
        real_root_count == polynomial.degree()
        and positive_root_count == 0
    )
    return {
        "degree": polynomial.degree(),
        "real_root_count_with_multiplicity": real_root_count,
        "positive_root_count": positive_root_count,
        "only_real_nonpositive_roots": only_nonpositive,
        "isolating_intervals": [
            {
                "lower": sp.sstr(lower),
                "upper": sp.sstr(upper),
                "multiplicity": multiplicity,
            }
            for (lower, upper), multiplicity in intervals
        ],
    }


def _bezoutian(
    first: sp.Poly,
    second: sp.Poly,
) -> sp.Matrix:
    degree = max(first.degree(), second.degree())
    if degree <= 0:
        return sp.zeros(0, 0)
    numerator = (
        first.as_expr().subs(T, U) * second.as_expr().subs(T, V)
        - second.as_expr().subs(T, U) * first.as_expr().subs(T, V)
    )
    kernel = sp.cancel(numerator / (U - V))
    bivariate = sp.Poly(kernel, U, V)
    return sp.Matrix(
        [
            [
                bivariate.coeff_monomial(U**i * V**j)
                for j in range(degree)
            ]
            for i in range(degree)
        ]
    )


def _matrix_as_strings(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [sp.sstr(matrix[row, column]) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _vertex_data(vertex: int, n: int, origin: int) -> dict[str, int]:
    layer, column = divmod(vertex, n)
    return {
        "layer": layer + 1,
        "cyclic_column": column,
        "normalized_label": _label(vertex, n, origin),
    }


def _monomial_exponent(value: int | None) -> int | None:
    return value


def _minor_nonnegative_on_nonnegative_axis(
    positive_exponent: int | None,
    negative_exponent: int | None,
) -> tuple[bool, str]:
    if negative_exponent is None:
        if positive_exponent is None:
            return True, "0"
        return True, _monomial(positive_exponent)
    if positive_exponent is None:
        return False, f"-{_monomial(negative_exponent)}"
    if positive_exponent == negative_exponent:
        return True, "0"
    positive = _monomial(positive_exponent)
    negative = _monomial(negative_exponent)
    return False, f"{positive} - {negative}"


def _monomial(exponent: int) -> str:
    if exponent == 0:
        return "1"
    if exponent == 1:
        return "t"
    return f"t^{exponent}"


def _operator_case(
    k: int,
    n: int,
    ideals: list[tuple[tuple[int, ...], Ideal]],
    successors: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    by_size: dict[int, list[tuple[tuple[int, ...], Ideal]]] = {
        size: []
        for size in range(k * n + 1)
    }
    for heights, ideal in ideals:
        by_size[len(ideal)].append((heights, ideal))
    for entries in by_size.values():
        entries.sort(key=lambda item: item[0])

    levels: list[dict[str, int]] = []
    for size in range(1, k * n + 1):
        rows = [
            (heights, ideal, maximal)
            for heights, ideal in by_size[size]
            for maximal in _maximal_elements(ideal, successors)
        ]
        if size == 1:
            column_count = 1
            nonzero_count = len(rows)
        else:
            column_count = sum(
                len(_maximal_elements(ideal, successors))
                for _, ideal in by_size[size - 1]
            )
            nonzero_count = sum(
                len(_maximal_elements(ideal - {maximal}, successors))
                for _, ideal, maximal in rows
            )
        levels.append(
            {
                "level": size,
                "row_states": len(rows),
                "column_states": column_count,
                "nonzero_entries": nonzero_count,
            }
        )

    tested_minors = 0
    nonnegative_minors = 0
    first_counterexample: dict[str, Any] | None = None
    tested_origins: list[int] = []
    # N=2 is the first nontrivial cut-sum transition. A single negative
    # 2x2 minor disproves total nonnegativity of the global operator.
    for origin in range(n):
        tested_origins.append(origin)
        rows = [
            (heights, ideal, maximal)
            for heights, ideal in by_size[2]
            for maximal in _ordered_maxima(
                _maximal_elements(ideal, successors),
                n,
                origin,
            )
        ]
        columns = [
            (heights, ideal, maximal)
            for heights, ideal in by_size[1]
            for maximal in _ordered_maxima(
                _maximal_elements(ideal, successors),
                n,
                origin,
            )
        ]
        row_entries: list[list[int | None]] = []
        for _, ideal, maximal in rows:
            predecessor = ideal - {maximal}
            entries: list[int | None] = []
            for _, column_ideal, previous_maximal in columns:
                if column_ideal != predecessor or previous_maximal not in _maximal_elements(
                    column_ideal, successors
                ):
                    entries.append(None)
                elif _label(previous_maximal, n, origin) < _label(maximal, n, origin):
                    entries.append(0)
                else:
                    entries.append(1)
            row_entries.append(entries)

        for row_a, row_b in itertools.combinations(range(len(rows)), 2):
            for column_a, column_b in itertools.combinations(
                range(len(columns)), 2
            ):
                positive_left = row_entries[row_a][column_a]
                positive_right = row_entries[row_b][column_b]
                negative_left = row_entries[row_a][column_b]
                negative_right = row_entries[row_b][column_a]
                positive_exponent = (
                    None
                    if positive_left is None or positive_right is None
                    else positive_left + positive_right
                )
                negative_exponent = (
                    None
                    if negative_left is None or negative_right is None
                    else negative_left + negative_right
                )
                passed, determinant = _minor_nonnegative_on_nonnegative_axis(
                    _monomial_exponent(positive_exponent),
                    _monomial_exponent(negative_exponent),
                )
                tested_minors += 1
                nonnegative_minors += int(passed)
                if not passed and first_counterexample is None:
                    first_counterexample = {
                        "k": k,
                        "n": n,
                        "origin": origin,
                        "level": 2,
                        "determinant": determinant,
                        "row_states": [
                            {
                                "heights": list(rows[index][0]),
                                "maximal_vertex": _vertex_data(
                                    rows[index][2], n, origin
                                ),
                            }
                            for index in (row_a, row_b)
                        ],
                        "column_states": [
                            {
                                "heights": list(columns[index][0]),
                                "last_vertex": _vertex_data(
                                    columns[index][2], n, origin
                                ),
                            }
                            for index in (column_a, column_b)
                        ],
                    }

    return {
        "exact_operator_entry": (
            "M_N^(o)[(I,m),(J,w)] = 0 unless J=I\\\\{m} and "
            "w in max(J); otherwise 1 if L_o(w)<L_o(m), t if "
            "L_o(w)>L_o(m). For N=1 the empty-ideal predecessor "
            "has one base state with entry 1."
        ),
        "state_indexing": (
            "Rows are (I,m,o) with |I|=N and m maximal in I; columns "
            "are (J,w,o) with |J|=N-1 and w maximal in J. The full "
            "operator is the direct sum over cyclic origins o."
        ),
        "levels": levels,
        "minor_test": {
            "scope": "all 2x2 minors of every origin block at N=2",
            "origins_tested": tested_origins,
            "tested_count": tested_minors,
            "nonnegative_on_t_ge_0_count": nonnegative_minors,
            "all_tested_minors_nonnegative": (
                tested_minors == nonnegative_minors
            ),
            "first_counterexample": first_counterexample,
            "interpretation": (
                "A negative 2x2 minor disproves total nonnegativity of this "
                "global matrix; this does not contradict the finite "
                "interlacing checks of the specific ideal-polynomial vectors."
            ),
        },
    }


def _analyze_case(k: int, n: int) -> dict[str, Any]:
    ideals = _all_ideals(k, n)
    successors = _successors(k, n)
    refined = _polynomial_vector(n, successors)
    extension_count = _linear_extension_count(successors)
    ideal_records: list[dict[str, Any]] = []
    roots_tested = 0
    roots_passed = 0
    ordered_pairs_tested = 0
    ordered_pairs_passed = 0
    bezoutians_tested = 0
    bezoutians_passed = 0
    endpoint_tests = 0
    endpoint_tests_passed = 0
    cut_sum_count_checks = 0

    for heights, ideal in ideals:
        origin = _canonical_origin(heights)
        maxima = _maximal_elements(ideal, successors)
        ordered = _ordered_maxima(maxima, n, origin)
        vector = dict(refined(ideal, origin))
        polynomial_by_maximal: dict[int, sp.Poly] = {
            maximal: sp.Poly(vector[maximal], T, domain=sp.ZZ)
            for maximal in ordered
        }
        root_pass_by_maximal: dict[int, bool] = {}
        polynomial_data: list[dict[str, Any]] = []
        for maximal in ordered:
            polynomial = polynomial_by_maximal[maximal]
            root_data = _root_certificate(polynomial)
            root_pass_by_maximal[maximal] = root_data[
                "only_real_nonpositive_roots"
            ]
            roots_tested += 1
            roots_passed += int(root_pass_by_maximal[maximal])
            polynomial_data.append(
                {
                    "maximal_element": _vertex_data(maximal, n, origin),
                    "polynomial": {
                        "degree": polynomial.degree(),
                        "coefficients_ascending": [
                            str(coefficient)
                            for coefficient in reversed(polynomial.all_coeffs())
                        ],
                    },
                    "root_certificate": root_data,
                }
            )

        pairs: list[dict[str, Any]] = []
        pair_flags: list[bool] = []
        for first_maximal, second_maximal in itertools.combinations(ordered, 2):
            first = polynomial_by_maximal[first_maximal]
            second = polynomial_by_maximal[second_maximal]
            matrix = _bezoutian(first, second)
            psd = matrix.is_positive_semidefinite is True
            degree_gap_ok = abs(first.degree() - second.degree()) <= 1
            both_rooted = (
                root_pass_by_maximal[first_maximal]
                and root_pass_by_maximal[second_maximal]
            )
            interlaces = degree_gap_ok and psd and both_rooted
            ordered_pairs_tested += 1
            ordered_pairs_passed += int(interlaces)
            bezoutians_tested += 1
            bezoutians_passed += int(psd)

            diagonal_form = sp.expand(
                sum(
                    matrix[i, j] * T ** (i + j)
                    for i in range(matrix.rows)
                    for j in range(matrix.cols)
                )
            )
            pairs.append(
                {
                    "first_maximal_element": _vertex_data(
                        first_maximal, n, origin
                    ),
                    "second_maximal_element": _vertex_data(
                        second_maximal, n, origin
                    ),
                    "degree_difference_at_most_one": degree_gap_ok,
                    "both_have_only_real_nonpositive_roots": both_rooted,
                    "interlaces_in_declared_order": interlaces,
                    "bezoutian_positive_semidefinite": psd,
                    "bezoutian_basis": "ascending powers 1,t,...,t^(d-1)",
                    "bezoutian_matrix": _matrix_as_strings(matrix),
                    "diagonal_quadratic_form_coefficients_ascending": [
                        str(coefficient)
                        for coefficient in reversed(
                            sp.Poly(diagonal_form, T).all_coeffs()
                        )
                    ]
                    if diagonal_form != 0
                    else ["0"],
                }
            )
            pair_flags.append(interlaces)

        if len(ordered) >= 2:
            endpoint_tests += 1
            endpoint_passed = pair_flags[len(ordered) - 2]
            endpoint_tests_passed += int(endpoint_passed)
        else:
            endpoint_passed = True

        polynomial_sum_at_one = sum(
            int(polynomial.eval(1))
            for polynomial in polynomial_by_maximal.values()
        ) if ideal else 1
        exact_extensions = extension_count(ideal)
        if polynomial_sum_at_one != exact_extensions:
            raise ArithmeticError(
                f"El corte no conserva el conteo de extensiones para "
                f"(k,n)=({k},{n}), alturas={heights}."
            )
        cut_sum_count_checks += 1

        normalized_heights = [
            heights[(origin + offset) % n]
            for offset in range(n)
        ]
        f_coefficients = [
            {
                "y_power": power,
                "maximal_element": _vertex_data(maximal, n, origin),
                "t_coefficients_ascending": polynomial_data[power]["polynomial"][
                    "coefficients_ascending"
                ],
            }
            for power, maximal in enumerate(ordered)
        ]
        ideal_records.append(
            {
                "height_vector": list(heights),
                "ideal_size": len(ideal),
                "canonical_origin": origin,
                "normalized_height_vector": normalized_heights,
                "origin_normalization_applied": origin != 0,
                "maxima_in_canonical_order": [
                    _vertex_data(maximal, n, origin)
                    for maximal in ordered
                ],
                "refined_polynomials": polynomial_data,
                "F_I_t_y_coefficients": f_coefficients,
                "ordered_interlacing_pairs": pairs,
                "branden_sequence_pass": (
                    all(root_pass_by_maximal.values()) and all(pair_flags)
                ),
                "endpoint_polynomial_test_pass": endpoint_passed,
                "linear_extension_count_from_h_at_1": str(
                    polynomial_sum_at_one
                ),
                "linear_extension_count_independent_recurrence": str(
                    exact_extensions
                ),
            }
        )

    operator = _operator_case(k, n, ideals, successors)
    return {
        "k": k,
        "n": n,
        "task_count": k * n,
        "ideal_count": len(ideals),
        "refined_polynomial_count": roots_tested,
        "real_rootedness": {
            "tested": roots_tested,
            "passed": roots_passed,
            "rate": roots_passed / roots_tested if roots_tested else 1.0,
        },
        "interlacing": {
            "ideals_tested": len(ideals),
            "ideals_passed": sum(
                record["branden_sequence_pass"] for record in ideal_records
            ),
            "ordered_pairs_tested": ordered_pairs_tested,
            "ordered_pairs_passed": ordered_pairs_passed,
            "rate": (
                ordered_pairs_passed / ordered_pairs_tested
                if ordered_pairs_tested
                else 1.0
            ),
            "canonical_order": (
                "Increasing normalized natural label L_o(m); o is the start "
                "of the unique proper cyclic arc of maximum column height, "
                "or 0 when there is no unique such arc."
            ),
        },
        "bezoutian_and_stability": {
            "bezoutians_tested": bezoutians_tested,
            "bezoutians_positive_semidefinite": bezoutians_passed,
            "rate": (
                bezoutians_passed / bezoutians_tested
                if bezoutians_tested
                else 1.0
            ),
            "stable_bivariate_F_ideals": sum(
                record["branden_sequence_pass"] for record in ideal_records
            ),
            "stable_bivariate_F_rate": (
                sum(record["branden_sequence_pass"] for record in ideal_records)
                / len(ideal_records)
                if ideal_records
                else 1.0
            ),
            "matrix_parameter_note": (
                "The coefficient Bezoutian is a constant symmetric matrix, "
                "not a matrix-valued function of t. Its positive-semidefinite "
                "certificate implies its diagonal quadratic form is "
                "nonnegative for every real t."
            ),
        },
        "extremal_polynomial_tests": {
            "endpoint_ideals_tested": endpoint_tests,
            "endpoint_pairs_passed": endpoint_tests_passed,
            "rate": (
                endpoint_tests_passed / endpoint_tests if endpoint_tests else 1.0
            ),
            "scope": (
                "For every ideal with at least two maxima, test the first/last "
                "polynomial in canonical order; single-maximum ideals are "
                "vacuous."
            ),
        },
        "cut_sum_count_checks": {
            "tested": cut_sum_count_checks,
            "passed": cut_sum_count_checks,
        },
        "transition_operator": operator,
        "ideals": ideal_records,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    root_tested = sum(item["real_rootedness"]["tested"] for item in results)
    root_passed = sum(item["real_rootedness"]["passed"] for item in results)
    ideal_tested = sum(item["ideal_count"] for item in results)
    ideal_sequence_passed = sum(
        item["interlacing"]["ideals_passed"] for item in results
    )
    pair_tested = sum(
        item["interlacing"]["ordered_pairs_tested"] for item in results
    )
    pair_passed = sum(
        item["interlacing"]["ordered_pairs_passed"] for item in results
    )
    bezoutian_tested = sum(
        item["bezoutian_and_stability"]["bezoutians_tested"] for item in results
    )
    bezoutian_passed = sum(
        item["bezoutian_and_stability"]["bezoutians_positive_semidefinite"]
        for item in results
    )
    endpoint_tested = sum(
        item["extremal_polynomial_tests"]["endpoint_ideals_tested"]
        for item in results
    )
    endpoint_passed = sum(
        item["extremal_polynomial_tests"]["endpoint_pairs_passed"]
        for item in results
    )
    minor_tested = sum(
        item["transition_operator"]["minor_test"]["tested_count"]
        for item in results
    )
    minor_passed = sum(
        item["transition_operator"]["minor_test"][
            "nonnegative_on_t_ge_0_count"
        ]
        for item in results
    )
    first_minor = next(
        (
            item["transition_operator"]["minor_test"]["first_counterexample"]
            for item in results
            if item["transition_operator"]["minor_test"]["first_counterexample"]
        ),
        None,
    )
    return {
        "cases": len(results),
        "ideals_tested": ideal_tested,
        "refined_polynomials_tested": root_tested,
        "real_rootedness": {
            "passed": root_passed,
            "tested": root_tested,
            "rate": root_passed / root_tested if root_tested else 1.0,
        },
        "branden_interlacing": {
            "ideal_sequences_passed": ideal_sequence_passed,
            "ideal_sequences_tested": ideal_tested,
            "sequence_rate": (
                ideal_sequence_passed / ideal_tested if ideal_tested else 1.0
            ),
            "ordered_pairs_passed": pair_passed,
            "ordered_pairs_tested": pair_tested,
            "pair_rate": pair_passed / pair_tested if pair_tested else 1.0,
        },
        "bezoutian_psd": {
            "passed": bezoutian_passed,
            "tested": bezoutian_tested,
            "rate": (
                bezoutian_passed / bezoutian_tested
                if bezoutian_tested
                else 1.0
            ),
        },
        "extremal_polynomial_tests": {
            "passed": endpoint_passed,
            "tested": endpoint_tested,
            "rate": endpoint_passed / endpoint_tested if endpoint_tested else 1.0,
        },
        "transition_minor_test": {
            "nonnegative_2x2_minors": minor_passed,
            "tested_2x2_minors": minor_tested,
            "rate": minor_passed / minor_tested if minor_tested else 1.0,
            "first_counterexample": first_minor,
        },
        "general_n_induction": {
            "status": "not established by finite computation",
            "reason": (
                "The exact computations cover only the five requested "
                "(k,n) instances. In addition, the global transition matrix "
                "has a negative 2x2 minor, so total nonnegativity of that "
                "operator cannot serve as a general preservation proof."
            ),
        },
    }


def main() -> None:
    started = time.perf_counter()
    case_results = [_analyze_case(k, n) for k, n in CASES]
    summary = _summarize(case_results)
    output = {
        "title": (
            "Exact finite verification of Brändén cut-sum polynomials "
            "for cyclic-crown order ideals"
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sympy_version": sp.__version__,
        "cases": [
            {"k": k, "n": n}
            for k, n in CASES
        ],
        "definitions": {
            "order_ideal_heights": (
                "0 <= h_i <= k and h_i <= h_(i-1 mod n) + 1; "
                "column i contains layers 1 through h_i."
            ),
            "cut_sum": (
                "H_I^(m)(t) = sum_{L_o(w)<L_o(m)} H_(I\\\\{m})^(w)(t) "
                "+ t sum_{L_o(w)>L_o(m)} H_(I\\\\{m})^(w)(t); "
                "H_empty=1."
            ),
            "canonical_origin": (
                "Rotate to the unique start of a proper cyclic arc on which "
                "the height is maximal; use origin 0 when the maximum set "
                "is the full circle or has multiple arcs."
            ),
            "branden_relation_convention": (
                "For ordered pair f,g, f preceq g is certified when "
                "abs(deg(f)-deg(g))<=1 and the coefficient Bezoutian "
                "(f(u)g(v)-g(u)f(v))/(u-v) is positive semidefinite."
            ),
            "root_proof_method": (
                "SymPy exact real-root isolation (Sturm intervals) counts "
                "all roots with multiplicity; exact positive-root count "
                "subtracts the zero-root multiplicity."
            ),
            "bezoutian_scope": (
                "The matrix is constant in the coefficient basis. "
                "Positive semidefiniteness is an exact pairwise "
                "interlacing certificate; it is not a pointwise matrix "
                "evaluation in t."
            ),
            "operator_minor_scope": (
                "All 2x2 minors are checked at N=2 for every cyclic-origin "
                "block. This necessary TP2 test is not a test of every "
                "higher-order minor or a proof for arbitrary n."
            ),
        },
        "summary": summary,
        "case_results": case_results,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Verificación exacta de ideales de orden cíclicos (SymPy)")
    print(
        "Raíces reales no positivas H_I^(m): "
        f"{summary['real_rootedness']['passed']}/"
        f"{summary['real_rootedness']['tested']} "
        f"({100 * summary['real_rootedness']['rate']:.2f}%)."
    )
    print(
        "Secuencias de interlacing de Brändén: "
        f"{summary['branden_interlacing']['ideal_sequences_passed']}/"
        f"{summary['branden_interlacing']['ideal_sequences_tested']} ideales; "
        f"{summary['branden_interlacing']['ordered_pairs_passed']}/"
        f"{summary['branden_interlacing']['ordered_pairs_tested']} pares."
    )
    print(
        "Bezoutianos PSD: "
        f"{summary['bezoutian_psd']['passed']}/"
        f"{summary['bezoutian_psd']['tested']} pares "
        f"({100 * summary['bezoutian_psd']['rate']:.2f}%)."
    )
    print(
        "Pruebas polinomiales extremales (primero/último máximo): "
        f"{summary['extremal_polynomial_tests']['passed']}/"
        f"{summary['extremal_polynomial_tests']['tested']}."
    )
    minor_test = summary["transition_minor_test"]
    print(
        "Menores 2x2 de M_N en N=2: "
        f"{minor_test['nonnegative_2x2_minors']}/"
        f"{minor_test['tested_2x2_minors']} no negativos para t>=0."
    )
    if minor_test["first_counterexample"] is not None:
        example = minor_test["first_counterexample"]
        print(
            "Contraejemplo exacto a la no negatividad total: "
            f"determinante {example['determinant']} "
            f"en (k,n)=({example['k']},{example['n']}), "
            f"origen={example['origin']}, N={example['level']}."
        )
    print(
        "Cierre para n arbitrario: no demostrado por esta verificación finita; "
        "la prueba TP2 del operador global falla."
    )
    print(
        "Operador: estados (I,m,o) por nivel; entrada 0, 1 o t según la "
        "relación de etiquetas L_o(w)<L_o(m)."
    )
    print(f"Resultados simbólicos: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
