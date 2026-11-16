"""Exact algebraic D4-equivariant realization and finite-range n=4 checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "resultados_n4_transfer.json"
REFERENCE = ROOT / "LEE_YANG_PHYSICS" / "resultados_lee_yang.json"
T, LAMBDA = sp.symbols("t lambda")
N = 4
ROOT_EPSILON = sp.Rational(1, 10**18)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArithmeticError(message)


def _linear_extension_hstar(k: int, n: int = N) -> list[int]:
    """Count naturally labeled linear extensions by descents via ideal paths."""
    current: dict[tuple[tuple[int, ...], int], list[int]] = {
        ((0,) * n, -1): [1]
    }
    for size in range(k * n):
        following: dict[tuple[tuple[int, ...], int], list[int]] = {}
        for (heights, last_label), coefficients in current.items():
            for column in range(n):
                row = heights[column]
                if row == k or row > heights[(column - 1) % n]:
                    continue
                updated = list(heights)
                updated[column] += 1
                label = row * n + column
                descent = int(last_label >= 0 and label < last_label)
                key = (tuple(updated), label)
                target = following.get(key)
                if target is None:
                    target = [0] * (size + 2)
                    following[key] = target
                for degree, count in enumerate(coefficients):
                    target[degree + descent] += count
        current = following

    result = [0] * (k * n + 1)
    full_height = (k,) * n
    for (heights, _last_label), coefficients in current.items():
        if heights == full_height:
            for degree, count in enumerate(coefficients):
                result[degree] += count
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _poly(coefficients_ascending: list[int]) -> sp.Poly:
    return sp.Poly.from_list(list(reversed(coefficients_ascending)), gens=T)


def _mask_image(mask: int, permutation: tuple[int, ...]) -> int:
    image = 0
    for column, target in enumerate(permutation):
        image |= ((mask >> column) & 1) << target
    return image


def _d4_actions() -> list[tuple[int, ...]]:
    return [
        tuple((column + rotation) % N for column in range(N))
        for rotation in range(N)
    ] + [
        tuple((reflection - column) % N for column in range(N))
        for reflection in range(N)
    ]


def _d4_mask_orbits() -> list[list[int]]:
    actions = _d4_actions()
    unseen = set(range(1 << N))
    orbits: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        orbit = sorted({_mask_image(seed, action) for action in actions})
        orbits.append(orbit)
        unseen.difference_update(orbit)
    return orbits


def _build_d4_equivariant_embedding(
    companion: sp.Matrix,
    u_active: sp.Matrix,
    v_active: sp.Matrix,
    orbits: list[list[int]],
) -> dict[str, Any]:
    """Embed a 3-state recurrence in the 16 mask coordinates, without local semantics."""
    dimension = 1 << N
    columns: list[sp.Matrix] = []
    for orbit in orbits:
        vector = sp.zeros(dimension, 1)
        for mask in orbit:
            vector[mask] = 1
        columns.append(vector)
    for orbit in orbits:
        representative = orbit[0]
        for mask in orbit[1:]:
            vector = sp.zeros(dimension, 1)
            vector[mask] = 1
            vector[representative] = -1
            columns.append(vector)

    basis = sp.Matrix.hstack(*columns)
    _require(basis.det() != 0, "The D4 mask-orbit basis is singular.")
    basis_inverse = basis.inv()
    basis_operator = sp.zeros(dimension)
    basis_operator[:3, :3] = companion
    operator = basis * basis_operator * basis_inverse

    u_basis = sp.zeros(1, dimension)
    v_basis = sp.zeros(dimension, 1)
    for index in range(3):
        u_basis[index] = u_active[index]
        v_basis[index] = v_active[index]
    u_masks = u_basis * basis_inverse
    v_masks = basis * v_basis

    permutation_matrices: list[sp.Matrix] = []
    for action in _d4_actions():
        permutation_matrix = sp.zeros(dimension)
        for mask in range(dimension):
            permutation_matrix[_mask_image(mask, action), mask] = 1
        permutation_matrices.append(permutation_matrix)
    commutes_with_d4 = all(
        operator * group == group * operator for group in permutation_matrices
    )
    _require(commutes_with_d4, "The padded operator is not D4-equivariant.")

    return {
        "basis_description": (
            "D4 orbit-indicator vectors followed by within-orbit difference vectors"
        ),
        "active_subspace": "span of the first three D4 orbit-indicator vectors",
        "active_dimension": 3,
        "commutes_with_all_eight_D4_actions": commutes_with_d4,
        "mask_order": [format(mask, "04b") for mask in range(dimension)],
        "matrix_in_mask_basis": [
            [sp.sstr(sp.factor(operator[row, column])) for column in range(dimension)]
            for row in range(dimension)
        ],
        "u_row_in_mask_basis": [
            sp.sstr(sp.factor(u_masks[0, column])) for column in range(dimension)
        ],
        "v_column_in_mask_basis": [
            sp.sstr(sp.factor(v_masks[row, 0])) for row in range(dimension)
        ],
        "interpretation": (
            "This is a noncanonical D4-equivariant algebraic embedding of the "
            "three-term recurrence. The 16 masks are not asserted to form a "
            "local Markov transfer."
        ),
    }


def _root_isolation_record(
    k: int,
    polynomial: sp.Poly,
    left_beta: tuple[sp.Rational, sp.Rational],
    right_beta: tuple[sp.Rational, sp.Rational],
) -> dict[str, Any]:
    isolated = polynomial.intervals(eps=ROOT_EPSILON)
    all_real_simple = (
        len(isolated) == polynomial.degree()
        and all(multiplicity == 1 for _interval, multiplicity in isolated)
    )
    strictly_inside = all_real_simple and all(
        interval[0] > left_beta[1] and interval[1] < right_beta[0]
        for interval, _multiplicity in isolated
    )
    return {
        "k": k,
        "degree": polynomial.degree(),
        "all_roots_real_and_simple": all_real_simple,
        "all_roots_strictly_inside_I4": strictly_inside,
        "root_isolating_intervals": [
            {
                "lower": str(interval[0]),
                "upper": str(interval[1]),
                "multiplicity": multiplicity,
            }
            for interval, multiplicity in isolated
        ],
    }


def _reference_checks(
    polynomials: dict[int, list[int]],
) -> dict[str, Any]:
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = {
        int(record["k"]): [
            int(value) for value in reversed(record["polynomial_coefficients_descending"])
        ]
        for record in payload["instances"]
        if int(record["n"]) == N
    }
    compared = sorted(k for k in expected if k in polynomials)
    mismatches = [
        k for k in compared if expected[k] != polynomials[k]
    ]
    _require(not mismatches, f"Reference polynomial mismatch for n=4, k={mismatches}.")
    return {
        "reference_file": "LEE_YANG_PHYSICS/resultados_lee_yang.json",
        "compared_k_values": compared,
        "all_coefficients_match": True,
    }


def analyze() -> dict[str, Any]:
    p4 = T**3 + 23 * T**2 + 23 * T + 1
    q4_transfer = 7 * T**5 - 28 * T**4 - 22 * T**3 - 28 * T**2 + 7 * T
    r4 = T * (T - 1) ** 6 * (T + 1)
    q4_extremal = T**4 + 76 * T**3 - 282 * T**2 + 76 * T + 1
    s6 = T**6 + 7 * T**5 + 27 * T**4 + 58 * T**3 + 27 * T**2 + 7 * T + 1
    companion = sp.Matrix(
        [
            [p4, q4_transfer, r4],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    characteristic_polynomial = sp.expand(
        LAMBDA**3 - p4 * LAMBDA**2 - q4_transfer * LAMBDA - r4
    )
    _require(
        sp.expand(companion.charpoly(LAMBDA).as_expr() - characteristic_polynomial) == 0,
        "The companion matrix characteristic polynomial is inconsistent.",
    )

    discriminant = sp.factor(
        sp.discriminant(characteristic_polynomial, LAMBDA)
    )
    expected_discriminant = sp.expand(-4 * T * q4_extremal * s6**2)
    _require(
        sp.expand(discriminant - expected_discriminant) == 0,
        "The exact cubic discriminant factorization did not verify.",
    )

    z0 = -38 - 24 * sp.sqrt(3)
    beta_minus = sp.simplify((z0 - sp.sqrt(z0**2 - 4)) / 2)
    beta_plus = sp.simplify((z0 + sp.sqrt(z0**2 - 4)) / 2)
    q4_intervals = sp.Poly(q4_extremal, T).intervals(eps=ROOT_EPSILON)
    negative_q4_intervals = [
        interval
        for interval, _multiplicity in q4_intervals
        if interval[1] < 0
    ]
    _require(
        len(negative_q4_intervals) == 2,
        "Expected two negative real roots of the extremal polynomial q4.",
    )
    left_beta, right_beta = negative_q4_intervals

    s6_intervals = sp.Poly(s6, T).intervals(eps=ROOT_EPSILON)
    coalescence_intervals = [
        interval
        for interval, _multiplicity in s6_intervals
        if interval[0] > left_beta[1] and interval[1] < right_beta[0]
    ]

    polynomial_coefficients = {
        k: _linear_extension_hstar(k) for k in range(13)
    }
    polynomials = {
        k: _poly(coefficients)
        for k, coefficients in polynomial_coefficients.items()
    }
    reference_check = _reference_checks(polynomial_coefficients)

    recurrence_residuals = {
        k: sp.expand(
            polynomials[k].as_expr()
            - p4 * polynomials[k - 1].as_expr()
            - q4_transfer * polynomials[k - 2].as_expr()
            - r4 * polynomials[k - 3].as_expr()
        )
        for k in range(3, 13)
    }
    recurrence_k_values = [
        k for k, residual in recurrence_residuals.items() if residual == 0
    ]

    v_active = sp.Matrix(
        [
            polynomials[4].as_expr(),
            polynomials[3].as_expr(),
            polynomials[2].as_expr(),
        ]
    )
    u_active = sp.Matrix([[1, 0, 0]]) * companion**-3
    scalar_identity_k_values: list[int] = []
    for k in range(2, 13):
        represented = sp.cancel(
            (u_active * companion ** (k - 1) * v_active)[0]
        )
        if sp.expand(represented - polynomials[k].as_expr()) == 0:
            scalar_identity_k_values.append(k)

    d4_orbits = _d4_mask_orbits()
    embedding = _build_d4_equivariant_embedding(
        companion, u_active, v_active, d4_orbits
    )
    embedding["characteristic_polynomial"] = (
        f"lambda^13*({sp.sstr(characteristic_polynomial)})"
    )
    embedding["identity_for_recurrence_defined_sequence_all_k_ge_2"] = True
    embedding["identity_formula"] = (
        "u=e1^T*A^(-3), v=(H4,H3,H2)^T, "
        "H_k=u^T*A^(k-1)*v; recurrence used for k>=5"
    )
    embedding["local_transfer_claim_certified"] = False

    root_records = [
        _root_isolation_record(k, polynomials[k], left_beta, right_beta)
        for k in range(2, 13)
    ]
    _require(
        all(record["all_roots_real_and_simple"] for record in root_records),
        "At least one tested n=4 polynomial failed exact real/simple root isolation.",
    )
    _require(
        all(record["all_roots_strictly_inside_I4"] for record in root_records),
        "At least one tested n=4 polynomial has a root outside I4.",
    )
    _require(
        recurrence_k_values == list(range(3, 13)),
        "The proposed cubic recurrence failed a finite exact check.",
    )
    _require(
        scalar_identity_k_values == list(range(2, 13)),
        "The companion representation failed a finite scalar identity check.",
    )

    return {
        "metadata": {
            "n": N,
            "finite_layers_tested": list(range(2, 13)),
            "method": (
                "Exact maximal-chain dynamic programming on cyclic ideal height "
                "vectors, independently checked against stored n=4 polynomials."
            ),
            "arbitrary_k_claim": (
                "the supplied exact cubic recurrence defines the scalar sequence "
                "for all k>=2; the combinatorial values are independently checked "
                "through k=12"
            ),
        },
        "reference_validation": reference_check,
        "recurrence": {
            "P4": sp.sstr(sp.expand(p4)),
            "Q4": sp.sstr(sp.expand(q4_transfer)),
            "R4": sp.sstr(sp.expand(r4)),
            "companion_matrix_3x3": [
                [sp.sstr(entry) for entry in row]
                for row in companion.tolist()
            ],
            "characteristic_polynomial": sp.sstr(characteristic_polynomial),
            "exact_residuals_zero_for_k": recurrence_k_values,
            "scalar_representation_tested_k": scalar_identity_k_values,
            "u_row_for_uTA_power_k_minus_1_v": [
                sp.sstr(sp.factor(value)) for value in u_active.tolist()[0]
            ],
            "v_column": [sp.sstr(value) for value in v_active],
            "initial_polynomials": {
                str(k): [str(value) for value in polynomial_coefficients[k]]
                for k in range(2, 5)
            },
            "status": (
                "recurrence for k>=5 with exact H2,H3,H4 boundary polynomials; "
                "the companion realization gives the scalar identity for every "
                "k>=2, and independent combinatorial checks pass through k=12"
            ),
        },
        "d4_mask_audit": {
            "state_count": 16,
            "states": [format(mask, "04b") for mask in range(16)],
            "d4_orbits": [
                [format(mask, "04b") for mask in orbit] for orbit in d4_orbits
            ],
            "dimension_of_d4_invariant_function_space": len(d4_orbits),
            "observation": (
                "The natural D4-invariant function space on 16 masks has dimension "
                "6. A 3-dimensional invariant subspace is possible but is not "
                "selected by orbit symmetry alone."
            ),
        },
        "sixteen_state_operator": embedding,
        "extremal_polynomial_and_spectrum": {
            "q4": sp.sstr(sp.expand(q4_extremal)),
            "cubic_discriminant": sp.sstr(discriminant),
            "factorization_verified": (
                "-4*t*q4(t)*(t^6+7*t^5+27*t^4+58*t^3+27*t^2+7*t+1)^2"
            ),
            "q4_is_a_discriminant_factor": True,
            "generic_spectral_decomposition": (
                "For distinct roots lambda_i of the cubic, "
                "H_k=sum_i c_i*lambda_i^(k-2), where for {i,j,l}={1,2,3}, "
                "c_i=(H4-(lambda_j+lambda_l)*H3+lambda_j*lambda_l*H2)/"
                "((lambda_i-lambda_j)*(lambda_i-lambda_l)); "
                "coalescing values are interpreted by limits."
            ),
            "interval_exact": {
                "z0": "-38-24*sqrt(3)",
                "beta_minus": "(z0-sqrt(z0^2-4))/2",
                "beta_plus": "(z0+sqrt(z0^2-4))/2",
            },
            "interval_decimal": [
                str(sp.N(beta_minus, 24)),
                str(sp.N(beta_plus, 24)),
            ],
            "negative_q4_root_isolations": [
                [str(left_beta[0]), str(left_beta[1])],
                [str(right_beta[0]), str(right_beta[1])],
            ],
            "discriminant_sign_on_I4": (
                "negative except at the two real roots of the sextic factor, "
                "where a pair of cubic modes coalesces; hence the generic cubic "
                "spectrum consists of one real root and one conjugate pair"
            ),
            "coalescence_isolations_inside_I4": [
                [str(interval[0]), str(interval[1])]
                for interval in coalescence_intervals
            ],
            "spectral_branch_condition_for_all_k": (
                "the exact cubic spectral decomposition is available; a global "
                "all-k real-root interlacing proof is not established by this script"
            ),
        },
        "finite_root_verification": {
            "root_isolation_epsilon": str(ROOT_EPSILON),
            "all_11_polynomials_real_simple_and_inside_I4": True,
            "instances": root_records,
        },
        "polynomial_data": {
            str(k): {
                "degree": polynomials[k].degree(),
                "h_star_coefficients_ascending": [
                    str(value) for value in polynomial_coefficients[k]
                ],
                "H_at_1": str(sum(polynomial_coefficients[k])),
            }
            for k in range(2, 13)
        },
        "limitations": [
            (
                "The 16x16 operator is the requested noncanonical D4-equivariant "
                "algebraic embedding of the companion recurrence, not a local "
                "Markov transfer or a combinatorial bijection on masks."
            ),
            (
                "The arbitrary-k root-interlacing claim is not inferred from the "
                "finite root-isolation checks through k=12."
            ),
        ],
    }


def main() -> None:
    result = analyze()
    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("N=4 exact transfer audit")
    print("=" * 102)
    print(" k | degree | H_{k,4}(1)                 | real/simple roots | inside I4 | recurrence")
    print("-" * 102)
    for k in range(2, 13):
        record = result["finite_root_verification"]["instances"][k - 2]
        h_at_one = result["polynomial_data"][str(k)]["H_at_1"]
        if k == 2:
            recurrence_status = "base"
        elif k in result["recurrence"]["exact_residuals_zero_for_k"]:
            recurrence_status = "pass"
        else:
            recurrence_status = "fail"
        print(
            f"{k:2d} | {record['degree']:6d} | {h_at_one:26s} | "
            f"{str(record['all_roots_real_and_simple']):17s} | "
            f"{str(record['all_roots_strictly_inside_I4']):9s} | "
            f"{recurrence_status}"
        )
    print("-" * 102)
    print(f"q4(t) factor in cubic discriminant: {result['extremal_polynomial_and_spectrum']['q4']}")
    print(
        "I4 endpoints:",
        result["extremal_polynomial_and_spectrum"]["interval_exact"],
    )
    print(
        "16-state operator: exact D4-equivariant algebraic embedding of the "
        "3x3 companion; no local Markov transfer is claimed."
    )
    print(
        "Scalar representation follows the supplied recurrence for all k>=2; "
        "exact combinatorial/root checks cover k<=12."
    )
    print("All-k real-root interlacing theorem: not established.")
    print(f"JSON: {OUTPUT}")


if __name__ == "__main__":
    main()
