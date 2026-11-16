"""Exact spectral and Sturm audit of the n=4 interval (-alpha, 0)."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp
from mpmath import iv


ROOT = Path(__file__).resolve().parents[1]
PROOF_DIR = Path(__file__).resolve().parent
CERTIFICATE = PROOF_DIR / "certificado_teorema_n4.json"
TRANSFER_DATA = PROOF_DIR / "resultados_n4_transfer.json"
T, LAMBDA, Z = sp.symbols("t lambda z")
P4 = T**3 + 23 * T**2 + 23 * T + 1
Q4 = 7 * T**5 - 28 * T**4 - 22 * T**3 - 28 * T**2 + 7 * T
R4 = T * (T - 1) ** 6 * (T + 1)
Q4_DISC = T**4 + 76 * T**3 - 282 * T**2 + 76 * T + 1
Q6 = T**6 + 7 * T**5 + 27 * T**4 + 58 * T**3 + 27 * T**2 + 7 * T + 1
CHI = LAMBDA**3 - P4 * LAMBDA**2 - Q4 * LAMBDA - R4


def _iv_rational(value: sp.Rational | int) -> object:
    value = sp.Rational(value)
    return iv.mpf(value.p) / value.q


def _iv_segment(lower: sp.Rational, upper: sp.Rational) -> object:
    return _iv_rational(lower) + iv.mpf([0, 1]) * _iv_rational(upper - lower)


def _iv_evaluate(poly: sp.Poly, value: object) -> object:
    result = iv.mpf(0)
    for coefficient in poly.all_coeffs():
        result = result * value + _iv_rational(sp.Rational(coefficient))
    return result


def _polynomial_from_transfer_data(k: int) -> sp.Poly:
    data = json.loads(TRANSFER_DATA.read_text(encoding="utf-8"))
    ascending = [
        int(value)
        for value in data["polynomial_data"][str(k)]["h_star_coefficients_ascending"]
    ]
    return sp.Poly.from_list(list(reversed(ascending)), gens=T, domain=sp.ZZ)


def _polynomial_from_certificate(k: int) -> sp.Poly:
    path = ROOT / "FORMATO 1" / f"certificado_{k}_4.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sp.Poly(
        sp.sympify(payload["polynomial"], locals={"t": T}),
        T,
        domain=sp.ZZ,
    )


def _spectral_sample() -> dict[str, object]:
    iv.dps = 80
    sample_t = sp.Rational(-1, 1000)
    characteristic = sp.Poly(CHI.subs(T, sample_t), LAMBDA, domain=sp.QQ)
    isolators = characteristic.intervals(eps=sp.Rational(1, 10**40))

    h0 = sp.Integer(1)
    h1 = T**3 + 11 * T**2 + 11 * T + 1
    h2 = _polynomial_from_transfer_data(2).as_expr()
    numerator = sp.Poly(
        LAMBDA**2 * h0
        + LAMBDA * (h1 - P4 * h0)
        + (h2 - P4 * h1 - Q4 * h0),
        LAMBDA,
        domain=sp.QQ.frac_field(T),
    )
    numerator_at_sample = sp.Poly(
        numerator.as_expr().subs(T, sample_t), LAMBDA, domain=sp.QQ
    )
    derivative = characteristic.diff()

    records: list[dict[str, object]] = []
    for interval, multiplicity in isolators:
        lower, upper = interval
        root_iv = _iv_segment(lower, upper)
        derivative_iv = _iv_evaluate(derivative, root_iv)
        if derivative_iv.a <= 0 <= derivative_iv.b:
            raise ArithmeticError("The sample root derivative enclosure contains zero.")
        coefficient_iv = (
            _iv_evaluate(numerator_at_sample, root_iv) / derivative_iv
        )
        absolute_root_iv = iv.sqrt(root_iv**2)
        records.append(
            {
                "isolator": [str(lower), str(upper)],
                "multiplicity": multiplicity,
                "root_iv80": str(root_iv),
                "absolute_value_iv80": str(absolute_root_iv),
                "spectral_coefficient_iv80": str(coefficient_iv),
                "coefficient_positive": coefficient_iv.a > 0,
                "coefficient_negative": coefficient_iv.b < 0,
            }
        )

    dominant_indices = [
        i
        for i, record in enumerate(records)
        if all(
            iv.mpf(record["absolute_value_iv80"]).a
            > iv.mpf(other["absolute_value_iv80"]).b
            for j, other in enumerate(records)
            if j != i
        )
    ]
    if len(dominant_indices) != 1:
        raise ArithmeticError("The 80-digit sample does not isolate a unique dominant root.")
    dominant = records[dominant_indices[0]]
    if not (
        iv.mpf(dominant["root_iv80"]).a > 0
        and dominant["coefficient_positive"]
    ):
        raise ArithmeticError("The dominant sample mode is not positive.")

    return {
        "t": str(sample_t),
        "precision_decimal_digits": 80,
        "all_eigenvalues_simple_real": len(records) == 3
        and all(record["multiplicity"] == 1 for record in records),
        "eigenvalues_in_increasing_order": records,
        "unique_dominant_branch": "positive real eigenvalue",
        "dominant_coefficient_positive": True,
    }


def audit() -> dict[str, object]:
    discriminant = sp.factor(sp.discriminant(CHI, LAMBDA))
    expected_discriminant = -4 * T * Q4_DISC * Q6**2
    if sp.expand(discriminant - expected_discriminant) != 0:
        raise ArithmeticError("The exact characteristic discriminant changed.")

    beta_z = -38 - 24 * sp.sqrt(3)
    beta_minus = sp.simplify(
        (beta_z - sp.sqrt(beta_z**2 - 4)) / 2
    )
    beta_plus = sp.simplify(
        (beta_z + sp.sqrt(beta_z**2 - 4)) / 2
    )
    if sp.simplify(beta_minus * beta_plus - 1) != 0:
        raise ArithmeticError("The endpoints of I4 are not reciprocal.")
    alpha = -beta_plus

    q4_poly = sp.Poly(Q4_DISC, T, domain=sp.ZZ)
    q6_poly = sp.Poly(Q6, T, domain=sp.ZZ)
    if q4_poly.count_roots(-sp.oo, 0) != 2:
        raise ArithmeticError("The quartic factor does not have two negative roots.")
    q4_roots_minus_one_zero = q4_poly.count_roots(-1, 0)
    q6_roots_minus_one_zero = q6_poly.count_roots(-1, 0)
    if q4_roots_minus_one_zero != 1 or q6_roots_minus_one_zero != 1:
        raise ArithmeticError("Unexpected discriminant root count in (-1,0).")
    if q6_poly.count_roots(-sp.Rational(1, 3), -sp.Rational(1, 4)) != 1:
        raise ArithmeticError("The q6 collision in (-1,0) was not isolated.")
    if q6_poly.count_roots(-sp.Rational(1, 79), 0) != 0:
        raise ArithmeticError("A q6 collision lies in the right exterior window.")
    q6_intervals = [
        interval
        for interval, multiplicity in q6_poly.intervals(
            eps=sp.Rational(1, 10**24)
        )
        if interval[0] > -1 and interval[1] < 0
    ]
    if len(q6_intervals) != 1:
        raise ArithmeticError("Could not isolate the q6 root in (-1,0).")

    if q4_poly.eval(-79) >= 0 or q4_poly.eval(-sp.Rational(1, 79)) >= 0:
        raise ArithmeticError("The rational interval [-79,-1/79] is not inside I4.")

    opposite_tie = sp.factor(P4 * Q4 + R4)
    expected_tie = (
        8
        * T
        * (T + 1)
        * (T**2 - 5 * T + 1)
        * (T**4 + 20 * T**3 + 22 * T**2 + 20 * T + 1)
    )
    if sp.expand(opposite_tie - expected_tie) != 0:
        raise ArithmeticError("The exact equal-modulus condition did not factor as expected.")
    z_quartic = -10 - 4 * sp.sqrt(5)
    if not (
        sp.sign(z_quartic - beta_z) == 1
        and sp.sign(-2 - z_quartic) == 1
    ):
        raise ArithmeticError("The negative roots of the tie quartic are not inside I4.")
    if sp.Poly(T**2 - 5 * T + 1, T).count_roots(-1, 0) != 0:
        raise ArithmeticError("The quadratic tie factor unexpectedly has a negative root.")

    h0 = sp.Integer(1)
    h1 = sp.Poly(T**3 + 11 * T**2 + 11 * T + 1, T, domain=sp.ZZ)
    h2 = _polynomial_from_transfer_data(2)
    numerator = sp.Poly(
        LAMBDA**2 * h0
        + LAMBDA * (h1.as_expr() - P4 * h0)
        + (
            h2.as_expr()
            - P4 * h1.as_expr()
            - Q4 * h0
        ),
        LAMBDA,
        domain=sp.QQ.frac_field(T),
    )
    coefficient_resultant = sp.factor(sp.resultant(CHI, numerator.as_expr(), LAMBDA))
    expected_resultant = 16 * T**2 * (T - 1) ** 2 * Q6**2
    if sp.expand(coefficient_resultant - expected_resultant) != 0:
        raise ArithmeticError("The spectral-coefficient resultant is not exact.")

    h = {0: sp.Poly(1, T), 1: h1, 2: h2}
    matched_certificates = []
    matched_transfer_data = []
    for k in range(3, 13):
        h[k] = sp.Poly(
            sp.expand(
                P4 * h[k - 1].as_expr()
                + Q4 * h[k - 2].as_expr()
                + R4 * h[k - 3].as_expr()
            ),
            T,
            domain=sp.ZZ,
        )
        if k <= 7:
            if h[k] != _polynomial_from_certificate(k):
                raise ArithmeticError(
                    f"Recurrence-generated H_{{{k},4}} differs from its certificate."
                )
            matched_certificates.append(k)
            if h[k] != _polynomial_from_transfer_data(k):
                raise ArithmeticError(
                    f"Recurrence-generated H_{{{k},4}} differs from exact DP data."
                )
            matched_transfer_data.append(k)

    finite_counts = []
    for k in range(3, 13):
        polynomial = h[k]
        degree = 3 * k
        if polynomial.degree() != degree:
            raise ArithmeticError(f"Unexpected degree at k={k}.")
        if sp.degree(sp.gcd(polynomial, polynomial.diff()).as_expr(), T) != 0:
            raise ArithmeticError(f"H_{{{k},4}} is not square-free.")
        inside = polynomial.count_roots(-79, -sp.Rational(1, 79))
        total_negative = polynomial.count_roots(-sp.oo, 0)
        if inside != degree or total_negative != degree:
            raise ArithmeticError(f"Exact Sturm count failed at k={k}.")
        finite_counts.append(
            {
                "k": k,
                "degree": degree,
                "I4_roots": degree,
                "right_window_roots": 0,
                "left_window_roots": 0,
                "roots_simple_and_negative": True,
                "sturm_count_on_rational_subset": int(inside),
            }
        )

    sample = _spectral_sample()
    result: dict[str, object] = {
        "status": "exact_right_window_spectrum_and_finite_sturm_checks_verified",
        "verification_script": "PROOF_N4_TRANSFER/audit_n4_right_window.py",
        "interval": {
            "I4_endpoints": [
                sp.sstr(beta_minus),
                sp.sstr(beta_plus),
            ],
            "beta_minus_decimal": str(sp.N(beta_minus, 40)),
            "beta_plus_decimal": str(sp.N(beta_plus, 40)),
            "alpha_exact": sp.sstr(alpha),
            "alpha_decimal": str(sp.N(alpha, 40)),
            "rational_subset_inside_I4": "[-79,-1/79]",
        },
        "discriminant": {
            "exact_factorization": sp.sstr(discriminant),
            "q4_roots_in_minus_one_zero": int(q4_roots_minus_one_zero),
            "q6_roots_in_minus_one_zero": int(q6_roots_minus_one_zero),
            "q6_roots_in_right_window": 0,
            "right_window_regime": (
                "For beta_plus<t<0, t<0 and q4(t)>0, so Delta_X(t)>0: "
                "the three characteristic roots are real and simple."
            ),
            "critical_points": {
                "beta_plus": "the simple q4 root -alpha",
                "q6_root_in_minus_one_zero": {
                    "isolator": [str(v) for v in q6_intervals[0]],
                    "rational_interval": "(-1/3,-1/4)",
                    "location": "inside I4, strictly left of -alpha",
                    "multiplicity_in_discriminant": 2,
                },
                "t_zero": "endpoint; chi_4(lambda;0)=lambda**2*(lambda-1)",
            },
        },
        "real_eigenvalue_dominance": {
            "equal_absolute_value_condition": sp.sstr(opposite_tie),
            "no_tie_in_right_window": True,
            "proof_note": (
                "For distinct real roots, an equal-modulus pair must be opposite. "
                "The tie polynomial factors as above; its negative roots are "
                "t=-1 and the reciprocal pair corresponding to z=-10-4*sqrt(5), "
                "all outside (-alpha,0)."
            ),
            "80_digit_sample": sample,
            "dominant_mode_on_right_window": (
                "The positive real eigenvalue is uniquely dominant throughout "
                "(-alpha,0); its coefficient is positive and nonzero."
            ),
            "no_second_complex_or_alternating_dominance_window": True,
        },
        "spectral_coefficients": {
            "initial_values": "H0=1, H1=t^3+11*t^2+11*t+1, H2 from exact DP data",
            "coefficient_formula": (
                "c_j=N(lambda_j)/chi'(lambda_j), "
                "N(lambda)=lambda^2*H0+lambda*(H1-P4*H0)"
                "+(H2-P4*H1-Q4*H0)"
            ),
            "exact_resultant": sp.sstr(coefficient_resultant),
            "all_coefficients_nonzero_on_right_window": True,
            "endpoint_caveat": (
                "The simple-root coefficient formula is undefined at beta_plus "
                "(the two largest positive roots coalesce) and at t=0 "
                "(lambda_2=lambda_3=0); a uniform coefficient bound on the "
                "closed window needs confluent endpoint limits."
            ),
        },
        "finite_root_distribution": {
            "counts": finite_counts,
            "distribution": "(3k,0,0), not (k,k,k), for k=3,...,12",
            "recurrence_generated_k": list(range(3, 13)),
            "matches_exact_certificates_k": matched_certificates,
            "matches_independent_transfer_data_k": matched_transfer_data,
            "exact_count_argument": (
                "Sturm counts all degree 3k roots in [-79,-1/79], which lies "
                "strictly inside I4; degree saturation leaves no roots in either "
                "exterior window."
            ),
        },
        "all_k_status": (
            "This right-window spectral audit is finite and does not by itself "
            "establish an all-k root count; combine it with "
            "audit_n4_q6_branch.py."
        ),
    }

    report = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    report["right_window_spectral_audit"] = result
    has_q6_audit = (
        "analytic_q6_branch_audit" in report["global_interval_audit"]
    )
    report["right_window_spectral_audit"]["all_k_status"] = (
        "The three-window phase theorem places all 3k roots in I4 for "
        "every k>=3; degree saturation excludes roots in both exterior "
        "windows."
        if has_q6_audit
        else "This right-window spectral audit is finite and does not by "
        "itself establish an all-k root count; run audit_n4_q6_branch.py."
    )
    if not has_q6_audit:
        report["global_claims"]["all_k_real_rootedness"] = (
            "Not assessed by this right-window audit; run "
            "audit_n4_q6_branch.py for the global root-count theorem."
        )
    report["global_claims"]["right_window_root_count_finite"] = (
        "Exact Sturm counts give 0 roots in (-alpha,0) for k=3,...,12."
    )
    CERTIFICATE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    audit_result = audit()
    print("Exact right-window spectral audit passed.")
    print(
        "alpha =",
        audit_result["interval"]["alpha_decimal"],
    )
    print(
        "Discriminant roots in (-1,0): q4=",
        audit_result["discriminant"]["q4_roots_in_minus_one_zero"],
        "q6=",
        audit_result["discriminant"]["q6_roots_in_minus_one_zero"],
    )
    print(
        "Finite Sturm distribution:",
        audit_result["finite_root_distribution"]["distribution"],
    )
    print(f"Updated certificate: {CERTIFICATE}")
