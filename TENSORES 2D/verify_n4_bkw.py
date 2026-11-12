"""Exact Beraha--Kahane--Weiss checks for the n=4 recurrence."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


t, lam, z = sp.symbols("t lambda z")
P4 = t**3 + 23 * t**2 + 23 * t + 1
Q4 = 7 * t**5 - 28 * t**4 - 22 * t**3 - 28 * t**2 + 7 * t
R4 = t * (t - 1) ** 6 * (t + 1)
CHI4 = lam**3 - P4 * lam**2 - Q4 * lam - R4
Q4_DISCRIMINANT = t**4 + 76 * t**3 - 282 * t**2 + 76 * t + 1
SIXTH_FACTOR = t**6 + 7 * t**5 + 27 * t**4 + 58 * t**3 + 27 * t**2 + 7 * t + 1


def _read_certificate(level: int) -> sp.Poly:
    certificate_path = (
        Path(__file__).resolve().parents[1]
        / "FORMATO 1"
        / f"certificado_{level}_4.json"
    )
    if not certificate_path.is_file():
        raise FileNotFoundError(f"Missing n=4 certificate: {certificate_path}")
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    polynomial = sp.Poly(
        sp.sympify(payload["polynomial"], locals={"t": t}),
        t,
        domain=sp.ZZ,
    )
    if polynomial.degree() != 3 * level:
        raise AssertionError(f"Certificate k={level} has an unexpected degree.")
    return polynomial


def _beta_endpoints() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    z_boundary = -38 - 24 * sp.sqrt(3)
    radical = sp.sqrt(z_boundary**2 - 4)
    beta_minus = sp.simplify((z_boundary - radical) / 2)
    beta_plus = sp.simplify((z_boundary + radical) / 2)
    if sp.simplify(beta_minus * beta_plus - 1) != 0:
        raise AssertionError("The two negative quartic roots are not reciprocal.")
    if sp.Poly(Q4_DISCRIMINANT, t).count_roots(-sp.oo, 0) != 2:
        raise AssertionError("The quartic does not have exactly two negative roots.")
    return beta_minus, beta_plus, z_boundary


def _verify_discriminant() -> None:
    expected = -4 * t * Q4_DISCRIMINANT * SIXTH_FACTOR**2
    actual = sp.factor(sp.discriminant(CHI4, lam))
    if sp.expand(actual - expected) != 0:
        raise AssertionError("The cubic discriminant factorization is incorrect.")
    if sp.expand(R4 - t * (t - 1) ** 6 * (t + 1)) != 0:
        raise AssertionError("R_4 does not match its stated exact factorization.")


def _verify_collision_and_dominance_conditions(
    z_boundary: sp.Expr,
) -> None:
    pair_tie = sp.factor(P4**3 * R4 + Q4**3)
    pair_tie_quotient = sp.Poly(sp.cancel(pair_tie / t), t, domain=sp.ZZ)
    if pair_tie_quotient.count_roots(-sp.oo, 0) != 0:
        raise AssertionError(
            "The condition for a tie between the conjugate pair and the "
            "real eigenvalue vanishes on the negative axis."
        )

    # This requested condition is false: the minus sign has two roots in I_4.
    opposite_sign_condition = sp.Poly(
        sp.cancel((P4**3 * R4 - Q4**3) / t), t, domain=sp.ZZ
    )
    left_interval = (sp.Integer(-50), sp.Integer(-49))
    right_interval = (sp.Rational(-21, 1000), sp.Rational(-1, 50))
    if opposite_sign_condition.count_roots(*left_interval) != 1:
        raise AssertionError("The first root of P_4^3 R_4-Q_4^3 was not isolated.")
    if opposite_sign_condition.count_roots(*right_interval) != 1:
        raise AssertionError("The second root of P_4^3 R_4-Q_4^3 was not isolated.")
    # q4 has exactly two negative roots and is negative between them, so
    # these four rational endpoints lie in I_4 without numerical comparison.
    if any(
        Q4_DISCRIMINANT.subs(t, endpoint) >= 0
        for endpoint in (*left_interval, *right_interval)
    ):
        raise AssertionError("The isolated roots are not inside I_4.")

    # At t=-1 the characteristic polynomial has a dominant imaginary pair
    # and a zero third root; this also handles P_4(-1)=R_4(-1)=0.
    if sp.factor(CHI4.subs(t, -1) - lam * (lam**2 + 48)) != 0:
        raise AssertionError("The exact characteristic polynomial at -1 changed.")

    # A tie in modulus between a nonzero real root c and the conjugate pair
    # forces c^3=R_4 and P_4*c+Q_4=0, hence P_4^3*R_4+Q_4^3=0.
    if sp.factor(pair_tie / t).is_zero:
        raise AssertionError("The pair-dominance obstruction is identically zero.")

    # Outside I_4, all roots are real and distinct. Equal absolute values
    # would require two roots to be opposites, which forces P_4*Q_4+R_4=0.
    opposite_real_roots = sp.factor(P4 * Q4 + R4)
    expected_opposite_real_roots = (
        8
        * t
        * (t + 1)
        * (t**2 - 5 * t + 1)
        * (t**4 + 20 * t**3 + 22 * t**2 + 20 * t + 1)
    )
    if sp.expand(opposite_real_roots - expected_opposite_real_roots) != 0:
        raise AssertionError("Unexpected opposite-real-root condition.")
    z_other = -10 - 4 * sp.sqrt(5)
    if sp.sign(z_other - z_boundary) != 1 or sp.sign(-2 - z_other) != 1:
        raise AssertionError("An exterior modulus crossing was not in I_4.")

    # The sixth-degree discriminant factor has two real negative roots in I_4.
    z_cubic = z**3 + 7 * z**2 + 24 * z + 44
    z_cubic_discriminant = sp.discriminant(sp.diff(z_cubic, z), z)
    if z_cubic_discriminant != -92 or sp.sign(z_cubic.subs(z, z_boundary)) != -1:
        raise AssertionError("Could not locate the repeated-eigenvalue points.")
    if z_cubic.subs(z, -2) != 16:
        raise AssertionError("Unexpected endpoint value for the z-cubic.")
    if sp.sign(sp.diff(z_cubic, z).subs(z, -sp.Rational(7, 3))) == -1:
        raise AssertionError("The z-cubic is not strictly increasing.")
    if sp.diff(z_cubic, z).subs(z, -sp.Rational(7, 3)) <= 0:
        raise AssertionError("The z-cubic derivative must be positive.")
    if Q4_DISCRIMINANT.subs(t, -100) <= 0 or Q4_DISCRIMINANT.subs(
        t, sp.Rational(-1, 1000)
    ) <= 0:
        raise AssertionError("The spectral-order witnesses are not outside I_4.")
    if Q4_DISCRIMINANT.subs(t, -79) >= 0 or Q4_DISCRIMINANT.subs(
        t, sp.Rational(-1, 79)
    ) >= 0:
        raise AssertionError("The finite root-check interval is not inside I_4.")


def _verify_initial_coefficients(
    h3: sp.Poly, h4: sp.Poly, h5: sp.Poly
) -> tuple[sp.Expr, sp.Expr]:
    a0 = h3.as_expr()
    a1 = h4.as_expr() - P4 * a0
    a2 = h5.as_expr() - P4 * h4.as_expr() - Q4 * a0
    numerator = sp.Poly(lam**2 * a0 + lam * a1 + a2, lam, domain=sp.QQ.frac_field(t))
    resultant = sp.factor(sp.resultant(CHI4, numerator.as_expr(), lam))
    expected = 16 * t**5 * (t - 1) ** 20 * (t + 1) ** 3 * SIXTH_FACTOR**2
    if sp.expand(resultant - expected) != 0:
        raise AssertionError("Unexpected BKW coefficient resultant.")

    # At t=-1 the vanishing mode is lambda=0, not either dominant mode.
    if sp.factor(numerator.as_expr().subs(t, -1) - 1536 * lam) != 0:
        raise AssertionError("Unexpected BKW numerator at t=-1.")
    return numerator.as_expr(), resultant


def _verify_sample_spectral_order(value: sp.Rational) -> None:
    specialized = sp.Poly(CHI4.subs(t, value), lam, domain=sp.QQ)
    intervals = sp.polys.polytools.intervals(
        specialized, eps=sp.Rational(1, 10**6)
    )
    if len(intervals) != 3 or any(multiplicity != 1 for _, multiplicity in intervals):
        raise AssertionError(f"Expected three simple real eigenvalues at t={value}.")

    def absolute_bounds(interval: tuple[sp.Rational, sp.Rational]) -> tuple[sp.Expr, sp.Expr]:
        lower, upper = interval
        if upper < 0:
            return -upper, -lower
        if lower > 0:
            return lower, upper
        return sp.Integer(0), max(-lower, upper)

    ordered = sorted(
        (absolute_bounds(interval) for interval, _ in intervals),
        key=lambda bounds: (bounds[0] + bounds[1]) / 2,
        reverse=True,
    )
    if not (
        ordered[0][0] > ordered[1][1]
        and ordered[1][0] > ordered[2][1]
    ):
        raise AssertionError(f"Strict modulus ordering failed at t={value}.")


def _verify_certificates_and_roots() -> None:
    certificates = {level: _read_certificate(level) for level in range(3, 8)}
    for level in range(6, 8):
        predicted = sp.Poly(
            sp.expand(
                P4 * certificates[level - 1].as_expr()
                + Q4 * certificates[level - 2].as_expr()
                + R4 * certificates[level - 3].as_expr()
            ),
            t,
            domain=sp.ZZ,
        )
        if predicted != certificates[level]:
            raise AssertionError(f"The recurrence does not match certificate k={level}.")

    for level, polynomial in certificates.items():
        if sp.degree(sp.gcd(polynomial, polynomial.diff()).as_expr(), t) != 0:
            raise AssertionError(f"The certified polynomial k={level} is not square-free.")
        if polynomial.count_roots(-79, sp.Rational(-1, 79)) != 3 * level:
            raise AssertionError(
                f"Not all roots for k={level} lie in the rational subinterval of I_4."
            )

    # Extend the exact recurrence as a finite check only; this does not prove
    # real-rootedness for all sufficiently large k.
    for level in range(8, 13):
        certificates[level] = sp.Poly(
            sp.expand(
                P4 * certificates[level - 1].as_expr()
                + Q4 * certificates[level - 2].as_expr()
                + R4 * certificates[level - 3].as_expr()
            ),
            t,
            domain=sp.ZZ,
        )
        polynomial = certificates[level]
        if sp.degree(sp.gcd(polynomial, polynomial.diff()).as_expr(), t) != 0:
            raise AssertionError(f"The generated polynomial k={level} is not square-free.")
        if polynomial.count_roots(-79, sp.Rational(-1, 79)) != 3 * level:
            raise AssertionError(
                f"Not all roots for generated k={level} lie in the rational subinterval."
            )


def main() -> None:
    _verify_discriminant()
    beta_minus, beta_plus, z_boundary = _beta_endpoints()
    _verify_collision_and_dominance_conditions(z_boundary)

    h3, h4, h5 = (_read_certificate(level) for level in (3, 4, 5))
    numerator, resultant = _verify_initial_coefficients(h3, h4, h5)
    _verify_sample_spectral_order(sp.Integer(-100))
    _verify_sample_spectral_order(sp.Rational(-1, 1000))
    _verify_certificates_and_roots()

    print(f"Discriminant verified: -4*t*q4(t)*q6(t)^2")
    print(
        "beta_- =", sp.N(beta_minus, 16),
        "; beta_+ =", sp.N(beta_plus, 16),
        "; beta_-*beta_+ = 1",
    )
    print(
        "On I_4 the dominant modes are a conjugate pair (coalescing at q6=0); "
        "outside I_4 the three real eigenvalues have strictly ordered moduli."
    )
    print(
        "P_4^3*R_4 + Q_4^3 has no negative zero (the pair-dominance condition)."
    )
    print(
        "Counterexample to the requested +/- nonvanishing: "
        "P_4^3*R_4 - Q_4^3 has roots in (-50,-49) and "
        "(-21/1000,-1/50), both inside I_4."
    )
    print("chi_4(lambda;-1) =", sp.factor(CHI4.subs(t, -1)))
    print(
        "BKW coefficient resultant:",
        sp.factor(resultant),
    )
    print(
        "The dominant-mode coefficients are nonzero at t=-1; at q6=0 "
        "eigenvalues coincide and the simple-mode BKW coefficients are replaced "
        "by their confluent form."
    )
    print("Exact certificate agreement: k=3,...,7.")
    print(
        "Exact Sturm check: all 3k roots are simple and in "
        "(-79,-1/79) for k=3,...,12."
    )
    print(
        "BKW establishes the negative-real accumulation set [beta_-,beta_+]; "
        "it alone does not establish that every zero is real and simple "
        "for all sufficiently large k."
    )


if __name__ == "__main__":
    main()
