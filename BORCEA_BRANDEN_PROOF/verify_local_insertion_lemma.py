"""Exact finite checks for valley-normalized local insertion on cyclic crowns."""

from __future__ import annotations

import itertools
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy as sp

from test_borcea_branden_induction import (
    _bezoutian,
    _maximal_elements,
    _root_certificate,
    _successors,
)


OUTPUT_PATH = Path(__file__).resolve().with_name("resultados_lema_local.json")
CASES = ((3, 3), (4, 3), (3, 4), (4, 4), (3, 5), (3, 6))
T = sp.Symbol("t")
Ideal = frozenset[int]

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _all_ideals(k: int, n: int) -> list[tuple[tuple[int, ...], Ideal]]:
    ideals: list[tuple[tuple[int, ...], Ideal]] = []
    for heights in itertools.product(range(k + 1), repeat=n):
        if not all(heights[column] <= heights[(column - 1) % n] + 1 for column in range(n)):
            continue
        ideal = frozenset(
            layer * n + column
            for column, height in enumerate(heights)
            for layer in range(height)
        )
        ideals.append((heights, ideal))
    return ideals


def _valley_origin(heights: tuple[int, ...]) -> int:
    """Choose the least-index column of minimum height as the cyclic seam."""
    minimum = min(heights)
    return min(column for column, height in enumerate(heights) if height == minimum)


def _label(vertex: int, n: int, origin: int) -> int:
    layer, column = divmod(vertex, n)
    return layer * n + (column - origin) % n


def _refined_polynomials(
    n: int, successors: dict[int, tuple[int, ...]]
) -> Any:
    @lru_cache(maxsize=None)
    def refined(ideal: Ideal, origin: int) -> tuple[tuple[int, sp.Expr], ...]:
        if not ideal:
            return ()
        maxima = _maximal_elements(ideal, successors)
        if len(ideal) == 1:
            return ((maxima[0], sp.Integer(1)),)

        values: dict[int, sp.Expr] = {}
        for maximal in maxima:
            predecessor = ideal - {maximal}
            previous_values = dict(refined(predecessor, origin))
            label_m = _label(maximal, n, origin)
            value = sp.Integer(0)
            for previous_maximal, previous_polynomial in previous_values.items():
                if _label(previous_maximal, n, origin) < label_m:
                    value += previous_polynomial
                else:
                    value += T * previous_polynomial
            values[maximal] = sp.Poly(sp.expand(value), T, domain=sp.ZZ).as_expr()
        return tuple(sorted(values.items()))

    return refined


def _newly_maximal_parents(
    ideal: Ideal, maximal: int, n: int
) -> set[int]:
    """Parents of m that become maximal after m is removed."""
    layer, column = divmod(maximal, n)
    if layer == 0:
        return set()

    candidates = (
        (
            (layer - 1) * n + (column - 1) % n,
            layer * n + (column - 1) % n,
        ),
        (
            (layer - 1) * n + column,
            layer * n + (column + 1) % n,
        ),
    )
    parents = set()
    for parent, other_successor in candidates:
        if parent not in ideal:
            raise AssertionError("An ideal must contain every predecessor of m.")
        if other_successor not in ideal:
            parents.add(parent)
    return parents


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _sign_changes(values: list[int]) -> int:
    signs = [_sign(value) for value in values if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def _positive_semidefinite_exact(matrix: sp.Matrix) -> bool:
    """Exact symmetric Schur-complement test with nonzero diagonal pivoting."""
    if matrix.rows != matrix.cols or matrix != matrix.T:
        return False
    active = matrix
    while active.rows:
        diagonal = [active[index, index] for index in range(active.rows)]
        if any(value < 0 for value in diagonal):
            return False
        pivot = next(
            (index for index, value in enumerate(diagonal) if value > 0),
            None,
        )
        if pivot is None:
            return all(
                active[row, column] == 0
                for row in range(active.rows)
                for column in range(row + 1, active.cols)
            )
        remaining = [index for index in range(active.rows) if index != pivot]
        if not remaining:
            return True
        pivot_value = active[pivot, pivot]
        column = active.extract(remaining, [pivot])
        row = active.extract([pivot], remaining)
        active = active.extract(remaining, remaining) - column * row / pivot_value
    return True


def _coefficient_tuple(polynomial: sp.Poly) -> tuple[int, ...]:
    return tuple(int(coefficient) for coefficient in reversed(polynomial.all_coeffs()))


@lru_cache(maxsize=20_000)
def _cached_ordered_bezoutian_psd(
    first_coefficients: tuple[int, ...],
    second_coefficients: tuple[int, ...],
) -> bool:
    first = sp.Poly.from_list(list(reversed(first_coefficients)), gens=T)
    second = sp.Poly.from_list(list(reversed(second_coefficients)), gens=T)
    return _positive_semidefinite_exact(_bezoutian(second, first))


def _weight(maximal: int, parent: int, origin: int, n: int) -> sp.Expr:
    return sp.Integer(1) if _label(parent, n, origin) < _label(maximal, n, origin) else T


def _vertex(vertex: int, n: int, origin: int) -> dict[str, int]:
    layer, column = divmod(vertex, n)
    return {
        "layer_zero_based": layer,
        "column_zero_based": column,
        "label": _label(vertex, n, origin),
    }


def _local_transition(
    ideal: Ideal,
    heights: tuple[int, ...],
    first_maximal: int,
    second_maximal: int,
    origin: int,
    n: int,
    maxima: tuple[int, ...],
    successors: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    first_ideal = ideal - {first_maximal}
    second_ideal = ideal - {second_maximal}
    first_parents = set(_maximal_elements(first_ideal, successors))
    second_parents = set(_maximal_elements(second_ideal, successors))
    first_new = _newly_maximal_parents(ideal, first_maximal, n)
    second_new = _newly_maximal_parents(ideal, second_maximal, n)

    expected_first = (set(maxima) - {first_maximal}) | first_new
    expected_second = (set(maxima) - {second_maximal}) | second_new
    first_minus_second = {second_maximal} | (first_new - second_new)
    second_minus_first = {first_maximal} | (second_new - first_new)
    parent_identity = (
        first_parents == expected_first
        and second_parents == expected_second
        and first_parents - second_parents == first_minus_second
        and second_parents - first_parents == second_minus_first
    )

    parent_union = first_parents | second_parents
    ordered_parents = sorted(
        parent_union,
        key=lambda vertex: _label(vertex, n, origin),
        reverse=True,
    )
    weight_rows = []
    constant_differences: list[int] = []
    linear_differences: list[int] = []
    at_one_differences: list[int] = []
    for parent in ordered_parents:
        weight_first = (
            _weight(first_maximal, parent, origin, n)
            if parent in first_parents
            else sp.Integer(0)
        )
        weight_second = (
            _weight(second_maximal, parent, origin, n)
            if parent in second_parents
            else sp.Integer(0)
        )
        difference = sp.Poly(
            sp.expand(weight_first - weight_second), T, domain=sp.ZZ
        )
        constant = int(difference.nth(0))
        linear = int(difference.nth(1))
        constant_differences.append(constant)
        linear_differences.append(linear)
        at_one_differences.append(constant + linear)
        weight_rows.append(
            {
                "parent": _vertex(parent, n, origin),
                "weight_first": sp.sstr(weight_first),
                "weight_second": sp.sstr(weight_second),
                "difference": sp.sstr(difference.as_expr()),
            }
        )

    constant_variations = _sign_changes(constant_differences)
    linear_variations = _sign_changes(linear_differences)
    at_one_variations = _sign_changes(at_one_differences)
    sign_change_pass = constant_variations <= 1 and linear_variations <= 1
    first_layer, first_column = divmod(first_maximal, n)
    second_layer, second_column = divmod(second_maximal, n)
    return {
        "height_vector": list(heights),
        "canonical_origin": origin,
        "first_maximal": _vertex(first_maximal, n, origin),
        "second_maximal": _vertex(second_maximal, n, origin),
        "strictly_decreasing_label_order": (
            _label(first_maximal, n, origin) > _label(second_maximal, n, origin)
        ),
        "local_parent_column_neighborhoods": {
            "first": sorted({(first_column - 1) % n, first_column}),
            "second": sorted({(second_column - 1) % n, second_column}),
        },
        "parent_sets": {
            "max_I_without_first": [
                _vertex(vertex, n, origin) for vertex in sorted(first_parents)
            ],
            "max_I_without_second": [
                _vertex(vertex, n, origin) for vertex in sorted(second_parents)
            ],
            "new_parents_from_local_cover_neighborhoods": {
                "first": [_vertex(vertex, n, origin) for vertex in sorted(first_new)],
                "second": [_vertex(vertex, n, origin) for vertex in sorted(second_new)],
            },
            "first_difference_formula": (
                "{second maximal} union "
                "(new_parents(first) minus new_parents(second))"
            ),
            "second_difference_formula": (
                "{first maximal} union "
                "(new_parents(second) minus new_parents(first))"
            ),
            "first_minus_second": [
                _vertex(vertex, n, origin)
                for vertex in sorted(first_parents - second_parents)
            ],
            "second_minus_first": [
                _vertex(vertex, n, origin)
                for vertex in sorted(second_parents - first_parents)
            ],
            "identity_verified": parent_identity,
            "maximal_layer_indices": [first_layer, second_layer],
        },
        "cut_weight_vectors_in_decreasing_parent_label_order": weight_rows,
        "branden_sign_change_check": {
            "criterion": (
                "For w(first)-w(second), each of the constant and t "
                "coefficient sign sequences, ordered by decreasing parent "
                "label, has at most one sign change."
            ),
            "constant_coefficient_sign_changes": constant_variations,
            "t_coefficient_sign_changes": linear_variations,
            "evaluated_at_t_equals_1_sign_changes": at_one_variations,
            "passed": sign_change_pass,
        },
    }


_successors_cache: dict[tuple[int, int], dict[int, tuple[int, ...]]] = {}


def _analyze_order(
    k: int,
    n: int,
    ideals: list[tuple[tuple[int, ...], Ideal]],
    rotate_to_valley: bool,
) -> dict[str, Any]:
    successors = _successors_cache.setdefault((k, n), _successors(k, n))
    refined = _refined_polynomials(n, successors)
    mode = "valley_rotation" if rotate_to_valley else "no_rotation"
    rooted_tested = 0
    rooted_passed = 0
    sequence_tested = len(ideals)
    sequence_passed = 0
    pair_tested = 0
    pair_passed = 0
    bezout_tested = 0
    bezout_passed = 0
    transition_records: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    ideal_passes: list[dict[str, Any]] = []

    for heights, ideal in ideals:
        origin = _valley_origin(heights) if rotate_to_valley else 0
        maxima = _maximal_elements(ideal, successors)
        ordered = tuple(
            sorted(maxima, key=lambda vertex: _label(vertex, n, origin), reverse=True)
        )
        polynomial_values = dict(refined(ideal, origin))
        polynomials = {
            maximal: sp.Poly(polynomial_values[maximal], T, domain=sp.ZZ)
            for maximal in ordered
        }

        rooted_by_maximal: dict[int, bool] = {}
        for maximal in ordered:
            certificate = _root_certificate(polynomials[maximal])
            passed = bool(certificate["only_real_nonpositive_roots"])
            rooted_by_maximal[maximal] = passed
            rooted_tested += 1
            rooted_passed += int(passed)
            if not passed and len(failure_examples) < 20:
                failure_examples.append(
                    {
                        "kind": "rootedness",
                        "height_vector": list(heights),
                        "maximal": _vertex(maximal, n, origin),
                        "certificate": certificate,
                    }
                )

        all_pairs_pass = True
        for first, second in itertools.combinations(ordered, 2):
            first_poly = polynomials[first]
            second_poly = polynomials[second]
            # The requested sequence is ordered by decreasing labels, so use
            # the standard Bezoutian orientation of the reverse ordered pair.
            psd = _cached_ordered_bezoutian_psd(
                _coefficient_tuple(first_poly),
                _coefficient_tuple(second_poly),
            )
            bezout_tested += 1
            bezout_passed += int(psd)
            degree_gap = abs(first_poly.degree() - second_poly.degree()) <= 1
            pair_ok = psd and degree_gap and rooted_by_maximal[first] and rooted_by_maximal[second]
            pair_tested += 1
            pair_passed += int(pair_ok)
            all_pairs_pass &= pair_ok
            if not pair_ok and len(failure_examples) < 20:
                bezoutian = _bezoutian(second_poly, first_poly)
                failure_examples.append(
                    {
                        "kind": "ordered_pair",
                        "height_vector": list(heights),
                        "first_maximal": _vertex(first, n, origin),
                        "second_maximal": _vertex(second, n, origin),
                        "bezoutian_orientation": "B(second, first)",
                        "degree_gap_at_most_one": degree_gap,
                        "bezoutian_positive_semidefinite": psd,
                        "bezoutian": [
                            [sp.sstr(bezoutian[i, j]) for j in range(bezoutian.cols)]
                            for i in range(bezoutian.rows)
                        ],
                    }
                )

        ideal_pass = all(rooted_by_maximal.values()) and all_pairs_pass
        sequence_passed += int(ideal_pass)
        ideal_passes.append(
            {
                "height_vector": list(heights),
                "ideal_size": len(ideal),
                "canonical_origin": origin,
                "ordered_maxima": [_vertex(vertex, n, origin) for vertex in ordered],
                "sequence_passed": ideal_pass,
            }
        )

        for first, second in zip(ordered, ordered[1:]):
            record = _local_transition(
                ideal, heights, first, second, origin, n, maxima, successors
            )
            transition_records.append(record)

    transitions_tested = len(transition_records)
    parent_identities_passed = sum(
        item["parent_sets"]["identity_verified"] for item in transition_records
    )
    sign_changes_passed = sum(
        item["branden_sign_change_check"]["passed"] for item in transition_records
    )
    return {
        "ordering_mode": mode,
        "maximum_order": "strictly decreasing L_c0(m)",
        "canonical_origin": (
            "c0(h) is the least original column attaining min(h); if all "
            "heights are equal, this deterministically gives 0. The seam "
            "touches a minimum column, so a proper maximum-height block "
            "cannot cross normalized columns n-1 to 0."
            if rotate_to_valley
            else "c0=0; no cyclic rotation."
        ),
        "real_rootedness": {
            "tested": rooted_tested,
            "passed": rooted_passed,
            "rate": rooted_passed / rooted_tested if rooted_tested else 1.0,
        },
        "branden_sequences": {
            "ideals_tested": sequence_tested,
            "ideals_passed": sequence_passed,
            "rate": sequence_passed / sequence_tested if sequence_tested else 1.0,
        },
        "ordered_interlacing_pairs": {
            "tested": pair_tested,
            "passed": pair_passed,
            "rate": pair_passed / pair_tested if pair_tested else 1.0,
        },
        "bezoutian_psd": {
            "tested": bezout_tested,
            "passed": bezout_passed,
            "rate": bezout_passed / bezout_tested if bezout_tested else 1.0,
        },
        "local_insertion_transitions": {
            "tested": transitions_tested,
            "parent_set_identities_passed": parent_identities_passed,
            "parent_set_identity_rate": (
                parent_identities_passed / transitions_tested
                if transitions_tested
                else 1.0
            ),
            "sign_change_condition_passed": sign_changes_passed,
            "sign_change_rate": (
                sign_changes_passed / transitions_tested
                if transitions_tested
                else 1.0
            ),
            "criterion_scope": (
                "Coefficientwise sign variation: for each consecutive cut "
                "pair, the constant- and t-coefficient sequences of the "
                "difference vector have at most one sign change."
            ),
            "transitions": transition_records,
        },
        "ideal_sequence_results": ideal_passes,
        "failure_examples": failure_examples,
    }


def _critical_c35_vectors() -> list[tuple[int, ...]]:
    return [
        (3, first, second, 2, 3)
        for first, second in itertools.product((1, 2), repeat=2)
    ]


def _analyze_case(k: int, n: int) -> dict[str, Any]:
    ideals = _all_ideals(k, n)
    no_rotation = _analyze_order(k, n, ideals, rotate_to_valley=False)
    valley_rotation = _analyze_order(k, n, ideals, rotate_to_valley=True)
    critical_vectors = _critical_c35_vectors() if (k, n) == (3, 5) else []
    height_set = {heights for heights, _ in ideals}
    if any(vector not in height_set for vector in critical_vectors):
        raise AssertionError("A designated C_{3,5} boundary witness is not an ideal.")

    return {
        "k": k,
        "n": n,
        "dimension": k * n,
        "ideal_count": len(ideals),
        "unrotated_boundary_crossing_maximum_blocks": sum(
            1
            for heights, _ in ideals
            if len({i for i, h in enumerate(heights) if h == max(heights)}) < n
            and 0 in {i for i, h in enumerate(heights) if h == max(heights)}
            and n - 1 in {i for i, h in enumerate(heights) if h == max(heights)}
        ),
        "four_designated_C35_boundary_witnesses": (
            [
                {
                    "height_vector": list(vector),
                    "valley_origin": _valley_origin(vector),
                    "sequence_passed_no_rotation": next(
                        item["sequence_passed"]
                        for item in no_rotation["ideal_sequence_results"]
                        if item["height_vector"] == list(vector)
                    ),
                    "sequence_passed_valley_rotation": next(
                        item["sequence_passed"]
                        for item in valley_rotation["ideal_sequence_results"]
                        if item["height_vector"] == list(vector)
                    ),
                }
                for vector in critical_vectors
            ]
        ),
        "no_rotation": no_rotation,
        "valley_rotation": valley_rotation,
    }


def _rate(value: dict[str, Any], category: str) -> str:
    return f"{100 * value[category]['rate']:.1f}%"


def _print_table(cases: list[dict[str, Any]]) -> None:
    headers = (
        f"{'(k,n)':8} {'ideals':>6} {'roots no/valley':>17} "
        f"{'sequences no/valley':>20} {'PSD pairs no/valley':>20} "
        f"{'local signs no/valley':>22}"
    )
    print("\nLema de Inserción Local: comparación por convención de orden")
    print(headers)
    print("-" * len(headers))
    for case in cases:
        plain = case["no_rotation"]
        valley = case["valley_rotation"]
        rooted_plain = _rate(plain, "real_rootedness")
        rooted_valley = _rate(valley, "real_rootedness")
        seq_plain = _rate(plain, "branden_sequences")
        seq_valley = _rate(valley, "branden_sequences")
        psd_plain = _rate(plain, "bezoutian_psd")
        psd_valley = _rate(valley, "bezoutian_psd")
        sign_plain = plain["local_insertion_transitions"]["sign_change_rate"]
        sign_valley = valley["local_insertion_transitions"]["sign_change_rate"]
        print(
            f"({case['k']},{case['n']})".ljust(8)
            + f"{case['ideal_count']:>6} "
            + f"{rooted_plain:>7}/{rooted_valley:<7} "
            + f"{seq_plain:>7}/{seq_valley:<7} "
            + f"{psd_plain:>7}/{psd_valley:<7} "
            + f"{100 * sign_plain:>6.1f}%/{100 * sign_valley:<6.1f}%"
        )

    print("\nTransiciones locales: comprobación exacta de conjuntos de padres")
    for case in cases:
        local = case["valley_rotation"]["local_insertion_transitions"]
        print(
            f"  C_{{{case['k']},{case['n']}}}: "
            f"{local['parent_set_identities_passed']}/"
            f"{local['tested']} identidades."
        )
    c35 = next(case for case in cases if (case["k"], case["n"]) == (3, 5))
    print(
        "  C_{3,5}: "
        f"{len(c35['four_designated_C35_boundary_witnesses'])} testigos "
        "críticos de costura verificados; "
        f"{c35['unrotated_boundary_crossing_maximum_blocks']} ideales tienen "
        "un bloque máximo que cruza el borde original."
    )
    c36 = next(case for case in cases if (case["k"], case["n"]) == (3, 6))
    print(
        f"  C_{{3,6}}: examinados todos sus {c36['ideal_count']} ideales y "
        f"{c36['valley_rotation']['local_insertion_transitions']['tested']} "
        "transiciones consecutivas."
    )
    failed_c36 = [
        item["height_vector"]
        for item in c36["valley_rotation"]["ideal_sequence_results"]
        if not item["sequence_passed"]
    ]
    if failed_c36:
        print(
            "  C_{3,6}: contraejemplos de secuencia bajo el criterio por pares: "
            + "; ".join(str(vector) for vector in failed_c36)
        )


def main() -> None:
    started = time.perf_counter()
    cases = []
    for k, n in CASES:
        print(f"Analizando C_{{{k},{n}}}...", flush=True)
        cases.append(_analyze_case(k, n))
    summary = {
        "cases": len(cases),
        "ideals_tested": sum(case["ideal_count"] for case in cases),
        "valley_rotated_sequences_passed": sum(
            case["valley_rotation"]["branden_sequences"]["ideals_passed"]
            for case in cases
        ),
        "valley_rotated_sequences_tested": sum(
            case["valley_rotation"]["branden_sequences"]["ideals_tested"]
            for case in cases
        ),
        "valley_rotated_bezoutians_passed": sum(
            case["valley_rotation"]["bezoutian_psd"]["passed"] for case in cases
        ),
        "valley_rotated_bezoutians_tested": sum(
            case["valley_rotation"]["bezoutian_psd"]["tested"] for case in cases
        ),
        "local_parent_identities_passed": sum(
            case["valley_rotation"]["local_insertion_transitions"][
                "parent_set_identities_passed"
            ]
            for case in cases
        ),
        "local_transitions_tested": sum(
            case["valley_rotation"]["local_insertion_transitions"]["tested"]
            for case in cases
        ),
        "local_sign_change_checks_passed": sum(
            case["valley_rotation"]["local_insertion_transitions"][
                "sign_change_condition_passed"
            ]
            for case in cases
        ),
    }
    output = {
        "title": "Exact finite verification of the valley-normalized local insertion lemma",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sympy_version": sp.__version__,
        "cases": [{"k": k, "n": n} for k, n in CASES],
        "definitions": {
            "canonical_seam": (
                "c0(h) is the smallest original column attaining min(h). "
                "L_c0(r,c)=r*n+((c-c0) mod n), with zero-based layers."
            ),
            "maximum_order": "Strictly decreasing L_c0(m).",
            "cut_sum": (
                "H_I^(m)(t)=sum_{L(w)<L(m)}H_(I\\\\{m})^(w)(t)"
                "+t*sum_{L(w)>L(m)}H_(I\\\\{m})^(w)(t)."
            ),
            "local_parent_formula": (
                "For m=(r,c), r>0, newly maximal parents after deleting m "
                "are (r-1,c-1) when (r,c-1) is absent and (r-1,c) when "
                "(r,c+1) is absent; the parent columns lie in {c-1,c}."
            ),
            "consecutive_parent_difference": (
                "max(I\\\\{ma})\\\\max(I\\\\{mb}) = "
                "{mb} union (new_parents(ma)\\\\new_parents(mb)); reverse "
                "the roles for the opposite difference."
            ),
            "sign_change_operationalization": (
                "For each local pair, align the cut vectors by decreasing "
                "parent label. Independently test the sign variation of the "
                "constant and t coefficients of w(ma)-w(mb); each must have "
                "at most one sign change. This is the finite criterion "
                "reported here, not a proof of an unrestricted theorem."
            ),
            "branden_pair_test": (
                "For every ordered pair f,g in the proposed sequence, "
                "require real nonpositive roots, degree difference at most "
                "one, and an exact positive-semidefinite standard "
                "coefficient Bezoutian B(g,f) when labels decrease. This "
                "fixes the sign convention under reversal of the label order."
            ),
            "C35_critical_witness_definition": (
                "The four valid height vectors (3,a,b,2,3), a,b in {1,2}, "
                "whose maximum-height block crosses the original seam."
            ),
        },
        "summary": summary,
        "case_results": cases,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_table(cases)
    print(f"\nResultados exactos: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
