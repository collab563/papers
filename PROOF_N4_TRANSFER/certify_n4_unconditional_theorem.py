"""Certify the interval root-counting argument for the n=4 cubic recurrence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp
from mpmath import iv


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "resultados_n4_transfer.json"
JSON_PATH = HERE / "certificado_teorema_n4.json"
TEX_PATH = HERE / "theorem_n4_proof.tex"
T = sp.symbols("t")
ISOLATION_EPSILON = sp.Rational(1, 10**50)
BOUNDARY_WIDTH = sp.Rational(1, 10**12)
MAX_INTERVAL_SUBDIVISION = 48


def _variations_at(poly: sp.Poly, point: sp.Expr) -> int:
    signs: list[int] = []
    for expression in sp.sturm(poly.as_expr(), T):
        member = sp.Poly(expression, T)
        if point == -sp.oo:
            sign = int(sp.sign(member.LC()))
            if member.degree() % 2:
                sign = -sign
        elif point == sp.oo:
            sign = int(sp.sign(member.LC()))
        else:
            sign = int(sp.sign(member.eval(point)))
        if sign:
            signs.append(sign)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _iv_coefficient(value: sp.Expr) -> Any:
    value = sp.Rational(value)
    return iv.mpf(int(value.p)) / int(value.q)


def _iv_poly(poly: sp.Poly, x: Any) -> Any:
    result = iv.mpf("0")
    for coefficient in poly.all_coeffs():
        result = result * x + _iv_coefficient(coefficient)
    return result


def _iv_rational(value: sp.Rational) -> Any:
    value = sp.Rational(value)
    return iv.mpf(int(value.p)) / int(value.q)


def _iv_segment(left: sp.Rational, right: sp.Rational) -> Any:
    left_iv = _iv_rational(left)
    width_iv = _iv_rational(right - left)
    return left_iv + iv.mpf([0, 1]) * width_iv


def _iv_cbrt(value: Any) -> Any:
    if value.a >= 0:
        return iv.exp(iv.log(value) / 3)
    if value.b <= 0:
        return -iv.exp(iv.log(-value) / 3)
    magnitude = iv.fabs(value)
    radius = iv.exp(iv.log(magnitude) / 3)
    return iv.mpf([-1, 1]) * radius


def _iv_nonnegative(value: Any) -> Any:
    if value.a >= 0:
        return value
    return value - value.a


def _iv_intersection(left: Any, right: Any) -> Any:
    lower = left.a if left.a >= right.a else right.a
    upper = left.b if left.b <= right.b else right.b
    if lower > upper:
        raise ArithmeticError("Interval Newton produced an empty intersection.")
    return iv.mpf([lower, upper])


def _evaluate_cell(x: Any, polys: dict[str, Any], use_boundary: bool = False) -> dict[str, Any]:
    a = _iv_poly(polys["A"], x)
    b = _iv_poly(polys["B"], x)
    c = _iv_poly(polys["C"], x)
    q4 = _iv_poly(polys["q4"], x)
    s6 = _iv_poly(polys["s6"], x)
    d6 = _iv_poly(polys["d6"], x)
    h1 = _iv_poly(polys["h1"], x)
    h2 = _iv_poly(polys["h2"], x)
    p = -b - a * a / 3
    q = -2 * a**3 / 27 - a * b / 3 - c
    tq4 = x
    for root in polys["q4_roots"]:
        tq4 *= x - root
    tq4 = _iv_nonnegative(tq4)
    rad = s6 * iv.sqrt(tq4 / 27)
    u = _iv_cbrt(-q / 2 + rad)
    v = _iv_cbrt(-q / 2 - rad)
    s = u + v
    for _ in range(8):
        derivative = 3 * s**2 + p
        if derivative.a <= 0:
            break
        midpoint = iv.mpf((s.a + s.b) / 2)
        residual = midpoint**3 + p * midpoint + q
        s = _iv_intersection(s, midpoint - residual / derivative)
    lam1 = s + a / 3
    re2 = -s / 2 + a / 3
    im2_sq = _iv_nonnegative(3 * s**2 / 4 + p)
    im2 = iv.sqrt(im2_sq)
    uuvv = (u + v / 2) ** 2 + iv.mpf(3) / 4 * v**2
    r2 = re2**2 + im2_sq
    denominator = 3 * s**2 + p
    if denominator.a <= 0:
        raise ArithmeticError("The coefficient denominator interval contains zero.")
    c1 = (h2 - 2 * re2 * h1 + r2) / denominator
    re_c2 = (1 - c1) / 2
    num_im = re2 * re_c2 - (h1 - c1 * lam1) / 2
    near_collision = not use_boundary and d6.a > iv.mpf("1.9") and (
        (s6.a <= 0 <= s6.b)
        or (im2.a <= 0 <= im2.b)
        or (
            s6.a >= -iv.mpf("0.05")
            and s6.b <= iv.mpf("0.05")
        )
    )
    if near_collision:
        if d6.a <= iv.mpf("1.9"):
            raise ArithmeticError("D6 lower bound failed in an S6 cancellation cell.")
        p8 = _iv_poly(polys["p8"], x)
        p12 = _iv_poly(polys["p12"], x)
        s0 = 2 * (x + 1) * p8 / (3 * d6)
        delta_s_s6 = (
            -8 * x * (x + 1) * q4 * s6 * p8
            / (
                d6**3
                * (
                    3 * s0**2
                    + p
                    + 3 * s0 * (s - s0)
                    + (s - s0) ** 2
                )
            )
        )
        c2p = -(x + 1) * (x**2 + 4 * x + 1) / 2
        c1p = (x + 1) ** 2 * (
            x**4 + 23 * x**3 + 72 * x**2 + 23 * x + 1
        ) / 6
        rem_s_s6 = (
            -2 * x * (x + 1) * p12 / d6**2
            + delta_s_s6 * (c2p * (s + s0) + c1p)
        )
        if tq4.a <= 0:
            raise ArithmeticError(
                f"The nonnegative t*q4 enclosure is not strictly positive "
                f"for x={x}, S6={s6}, t*q4={tq4}."
            )
        im_c2 = (
            rem_s_s6
            / denominator
            * 3
            * uuvv
            / iv.sqrt(tq4)
        )
    elif im2.a > 0 or im2.b < 0:
        im_c2 = num_im / im2
    elif use_boundary:
        im_c2 = iv.mpf([0, 0])
    else:
        raise ArithmeticError(
            "An S6 zero was not handled by the cancellation formula: "
            f"x={x}, S6={s6}, D6={d6}, im2={im2}"
        )

    r2_abs = iv.sqrt(r2)
    if r2.a <= 0:
        raise ArithmeticError("The squared modulus R^2 interval contains zero.")
    lambda_ratio = iv.sqrt(lam1**2) / r2_abs
    if use_boundary:
        w_re = im2 * re_c2
        w_abs = iv.sqrt(w_re**2 + num_im**2)
        if w_abs.a <= 0:
            raise ArithmeticError("The boundary W interval contains zero.")
        envelope = (
            iv.sqrt(c1**2)
            * lam1**2
            * iv.sqrt(im2**2)
            / (2 * w_abs * r2)
        )
    else:
        if re_c2.a <= 0:
            raise ArithmeticError("The real part of c2 is not certified positive.")
        envelope = iv.sqrt(c1**2) * lam1**2 / (2 * re_c2 * r2)
    return {
        "lam1": lam1,
        "re2": re2,
        "im2": im2,
        "re_c2": re_c2,
        "im_c2": im_c2,
        "num_im": num_im,
        "envelope": envelope,
        "lambda_ratio": lambda_ratio,
        "near_collision": near_collision,
    }


def _isolations(poly: sp.Poly) -> list[dict[str, Any]]:
    return [
        {
            "lower": str(interval[0]),
            "upper": str(interval[1]),
            "multiplicity": multiplicity,
        }
        for interval, multiplicity in poly.intervals(eps=ISOLATION_EPSILON)
    ]


def _interval_audit(
    q4: sp.Poly,
    s6: sp.Poly,
    d6: sp.Poly,
    h1: sp.Poly,
    h2: sp.Poly,
) -> dict[str, Any]:
    iv.dps = 80
    p8 = sp.Poly(
        T**8 + 113 * T**7 + 2074 * T**6 + 11119 * T**5
        + 21770 * T**4 + 11119 * T**3 + 2074 * T**2 + 113 * T + 1,
        T,
    )
    p12 = sp.Poly(
        3 * T**12 + 316 * T**11 + 5863 * T**10 + 55260 * T**9
        + 270233 * T**8 + 686888 * T**7 + 911994 * T**6
        + 686888 * T**5 + 270233 * T**4 + 55260 * T**3
        + 5863 * T**2 + 316 * T + 3,
        T,
    )
    polys = {
        "A": sp.Poly(T**3 + 23 * T**2 + 23 * T + 1, T),
        "B": sp.Poly(7 * T**5 - 28 * T**4 - 22 * T**3 - 28 * T**2 + 7 * T, T),
        "C": sp.Poly(T * (T - 1) ** 6 * (T + 1), T),
        "q4": q4,
        "s6": s6,
        "d6": d6,
        "h1": h1,
        "h2": h2,
        "p8": p8,
        "p12": p12,
        "q4_roots": [
            _iv_segment(interval[0], interval[1])
            for interval, _multiplicity in q4.intervals(eps=ISOLATION_EPSILON)
        ],
    }
    negative = [
        interval
        for interval, _multiplicity in q4.intervals(eps=ISOLATION_EPSILON)
        if interval[1] < 0
    ]
    left_isolator, right_isolator = negative
    if (
        left_isolator[1] - left_isolator[0] >= BOUNDARY_WIDTH
        or right_isolator[1] - right_isolator[0] >= BOUNDARY_WIDTH
    ):
        raise ArithmeticError("A q4 isolator is wider than its 1e-12 boundary cell.")
    edge = BOUNDARY_WIDTH
    inner_left = left_isolator[0] + edge
    inner_right = right_isolator[1] - edge
    step = (inner_right - inner_left) / 1998
    cells: list[tuple[sp.Rational, sp.Rational, bool]] = [
        (left_isolator[0], inner_left, True)
    ]
    cells.extend(
        (
            inner_left + index * step,
            inner_left + (index + 1) * step,
            False,
        )
        for index in range(1998)
    )
    cells.append((inner_right, right_isolator[1], True))

    max_envelope = iv.mpf("0")
    max_lambda_ratio = iv.mpf("0")
    cell_records: list[dict[str, Any]] = []
    phase_base_cells = [(left, right) for left, right, boundary in cells if not boundary]
    envelope_visits = 0

    def certify_envelope_cell(
        left: sp.Rational,
        right: sp.Rational,
        boundary: bool,
        base_index: int,
        depth: int = 0,
    ) -> None:
        nonlocal max_envelope, max_lambda_ratio, envelope_visits
        envelope_visits += 1
        if envelope_visits % 10000 == 0:
            print(
                f"Envelope refinement nodes: {envelope_visits}; "
                f"base cell {base_index}, depth {depth}.",
                flush=True,
            )
        try:
            values = _evaluate_cell(
                _iv_segment(left, right), polys, use_boundary=boundary
            )
            envelope = values["envelope"]
            lambda_ratio = values["lambda_ratio"]
            if (
                envelope.a >= 0
                and envelope.b < iv.mpf("0.02")
                and lambda_ratio.a >= 0
                and lambda_ratio.b < 1
            ):
                max_envelope = (
                    envelope if envelope.b > max_envelope.b else max_envelope
                )
                max_lambda_ratio = (
                    lambda_ratio
                    if lambda_ratio.b > max_lambda_ratio.b
                    else max_lambda_ratio
                )
                cell_records.append(
                    {
                        "base_index": base_index,
                        "boundary": boundary,
                        "subdivision_depth": depth,
                        "left": str(left),
                        "right": str(right),
                        "envelope_iv80": str(envelope),
                        "lambda1_over_lambda2_abs_iv80": str(lambda_ratio),
                        "near_S6": values["near_collision"],
                    }
                )
                return
        except ArithmeticError:
            if depth >= MAX_INTERVAL_SUBDIVISION:
                raise
        else:
            if depth >= MAX_INTERVAL_SUBDIVISION:
                raise ArithmeticError(
                    f"Envelope bound is unresolved in base cell {base_index}: "
                    f"E2={values['envelope']}, "
                    f"lambda1/abs(lambda2)={values['lambda_ratio']}"
                )
        middle = (left + right) / 2
        certify_envelope_cell(left, middle, boundary, base_index, depth + 1)
        certify_envelope_cell(middle, right, boundary, base_index, depth + 1)

    for index, (left, right, boundary) in enumerate(cells):
        certify_envelope_cell(left, right, boundary, index)
        if (index + 1) % 50 == 0:
            print(
                f"Certified envelope cells: {index + 1}/{len(cells)}.",
                flush=True,
            )

    def endpoint_complex(t_value: sp.Rational, which: str) -> Any:
        value = _evaluate_cell(_iv_segment(t_value, t_value), polys)
        if which == "lambda":
            return iv.mpc(value["re2"], value["im2"])
        return iv.mpc(value["re_c2"], value["im_c2"])

    def interval_argument(value: Any) -> Any:
        return iv.atan2(value.imag, value.real)

    phase_sums = {
        "lambda": iv.mpf("0"),
        "c2": iv.mpf("0"),
    }
    phase_cell_count = 0

    def accumulate_phase_cell(
        left: sp.Rational,
        right: sp.Rational,
        depth: int = 0,
    ) -> None:
        nonlocal phase_cell_count
        try:
            box = _evaluate_cell(_iv_segment(left, right), polys)
            if (box["re2"] ** 2 + box["im2"] ** 2).a <= 0:
                raise ArithmeticError("lambda2 interval contains zero.")
            if (box["re_c2"] ** 2 + box["im_c2"] ** 2).a <= 0:
                raise ArithmeticError("c2 interval contains zero.")
            midpoint = (left + right) / 2
            midpoint_values = _evaluate_cell(
                _iv_segment(midpoint, midpoint), polys
            )
            increments = {}
            for name, re_key, im_key in (
                ("lambda", "re2", "im2"),
                ("c2", "re_c2", "im_c2"),
            ):
                path_box = iv.mpc(box[re_key], box[im_key])
                midpoint_value = iv.mpc(
                    midpoint_values[re_key], midpoint_values[im_key]
                )
                rotated_path = path_box / midpoint_value
                if rotated_path.real.a <= 0:
                    raise ArithmeticError(
                        f"{name} path is not contained in a certified half-plane."
                    )
                path_argument = interval_argument(rotated_path)
                if path_argument.b - path_argument.a >= iv.pi:
                    raise ArithmeticError(f"{name} path argument is ambiguous.")
                z_left = endpoint_complex(left, name)
                z_right = endpoint_complex(right, name)
                increments[name] = interval_argument(z_right / z_left)
                if increments[name].b - increments[name].a >= iv.pi:
                    raise ArithmeticError(f"{name} argument step is ambiguous.")
            for name in phase_sums:
                phase_sums[name] += increments[name]
            phase_cell_count += 1
            return
        except ArithmeticError:
            if depth >= MAX_INTERVAL_SUBDIVISION:
                raise
        middle = (left + right) / 2
        accumulate_phase_cell(left, middle, depth + 1)
        accumulate_phase_cell(middle, right, depth + 1)

    for left, right in phase_base_cells:
        accumulate_phase_cell(left, right)
        if (phase_cell_count % 50) == 0:
            print(
                f"Certified phase subcells: {phase_cell_count}.",
                flush=True,
            )

    phase_over_pi = {
        "lambda": phase_sums["lambda"] / iv.pi,
        "c2": phase_sums["c2"] / iv.pi,
    }
    target_checks = {
        "lambda_near_minus_one": (
            phase_over_pi["lambda"].a >= iv.mpf("-1.00001")
            and phase_over_pi["lambda"].b <= iv.mpf("-0.99999")
        ),
        "c2_near_minus_one": (
            phase_over_pi["c2"].a >= iv.mpf("-1.00001")
            and phase_over_pi["c2"].b <= iv.mpf("-0.99999")
        ),
    }
    if not all(target_checks.values()):
        raise ArithmeticError(
            "The accumulated phase enclosures do not certify the requested winding: "
            f"lambda={phase_over_pi['lambda']}, c2={phase_over_pi['c2']}"
        )

    left_boundary = _evaluate_cell(
        _iv_segment(left_isolator[0], inner_left), polys, use_boundary=True
    )
    right_boundary = _evaluate_cell(
        _iv_segment(inner_right, right_isolator[1]), polys, use_boundary=True
    )
    left_inner_lambda = endpoint_complex(inner_left, "lambda")
    right_inner_lambda = endpoint_complex(inner_right, "lambda")
    left_inner_c2 = endpoint_complex(inner_left, "c2")
    right_inner_c2 = endpoint_complex(inner_right, "c2")
    endpoint_sign_checks = {
        "lambda2_negative_at_left_endpoint": left_boundary["re2"].b < 0,
        "lambda2_positive_at_right_endpoint": right_boundary["re2"].a > 0,
        "c2_imaginary_positive_at_left_endpoint": left_boundary["num_im"].a > 0,
        "c2_imaginary_negative_at_right_endpoint": right_boundary["num_im"].b < 0,
        "c2_real_part_positive_at_both_endpoints": (
            left_boundary["re_c2"].a > 0 and right_boundary["re_c2"].a > 0
        ),
        "lambda2_imaginary_positive_at_inner_edges": (
            left_inner_lambda.imag.a > 0 and right_inner_lambda.imag.a > 0
        ),
        "c2_imaginary_positive_left_negative_right_at_inner_edges": (
            left_inner_c2.imag.a > 0 and right_inner_c2.imag.b < 0
        ),
    }
    if not all(endpoint_sign_checks.values()):
        raise ArithmeticError(
            "The endpoint quadrant limits needed to identify the exact phase lift "
            f"were not certified: {endpoint_sign_checks}"
        )

    left_lambda_angle = interval_argument(left_inner_lambda)
    right_lambda_angle = interval_argument(right_inner_lambda)
    left_c2_angle = interval_argument(left_inner_c2)
    right_c2_angle = interval_argument(right_inner_c2)
    full_phase_over_pi = {
        "lambda": (
            phase_sums["lambda"]
            + iv.pi
            - left_lambda_angle
            + right_lambda_angle
        )
        / iv.pi,
        "c2": (
            phase_sums["c2"]
            + iv.pi
            - left_c2_angle
            + right_c2_angle
        )
        / iv.pi,
    }
    exact_lift_checks = {
        "lambda_interval_isolates_minus_one": (
            full_phase_over_pi["lambda"].a > -1.00001
            and full_phase_over_pi["lambda"].b < -0.99999
        ),
        "c2_interval_isolates_minus_one": (
            full_phase_over_pi["c2"].a > -1.00001
            and full_phase_over_pi["c2"].b < -0.99999
        ),
    }
    if not all(exact_lift_checks.values()):
        raise ArithmeticError(
            "The endpoint-extended phase enclosures do not isolate the requested "
            f"integer lifts: {full_phase_over_pi}"
        )
    return {
        "base_cell_count": len(cells),
        "boundary_width": str(edge),
        "boundary_root_isolator_widths": [
            str(left_isolator[1] - left_isolator[0]),
            str(right_isolator[1] - right_isolator[0]),
        ],
        "refined_envelope_cell_count": len(cell_records),
        "boundary_cell_count": 2,
        "interior_cell_count": 1998,
        "maximum_subdivision_depth": max(
            record["subdivision_depth"] for record in cell_records
        ),
        "phase_refined_cell_count": phase_cell_count,
        "maximum_envelope_iv80": str(max_envelope),
        "maximum_lambda1_over_lambda2_abs_iv80": str(max_lambda_ratio),
        "envelope_bound_below_0_02": True,
        "lambda1_mode_dominated_by_complex_pair": True,
        "phase_lambda_delta_over_pi_iv80": str(phase_over_pi["lambda"]),
        "phase_c2_delta_over_pi_iv80": str(phase_over_pi["c2"]),
        "phase_target_checks": target_checks,
        "endpoint_sign_checks": endpoint_sign_checks,
        "endpoint_phase_limits": {
            "lambda_left": "pi",
            "lambda_right": "0",
            "c2_left": "pi/2",
            "c2_right": "-pi/2",
        },
        "full_phase_lambda_delta_over_pi_iv80": str(full_phase_over_pi["lambda"]),
        "full_phase_c2_delta_over_pi_iv80": str(full_phase_over_pi["c2"]),
        "exact_phase_lift_checks": exact_lift_checks,
        "exact_phase_lift": {
            "lambda": "-pi",
            "c2": "-pi",
            "Phi_k": "-(k+1)*pi",
            "deduced_from_interval_enclosures_and_endpoint_quadrants": True,
        },
        "requested_minus_three_pi_lambda_lift_certified": False,
        "three_k_sign_change_conclusion": (
            "not determined by the reflected upper branch alone; see "
            "audit_n4_q6_branch.py"
        ),
        "cells": cell_records,
    }


def _latex_report(data: dict[str, Any]) -> str:
    interval_result = data["global_interval_audit"]
    if interval_result["certified"]:
        interval_status = (
            "The 2,000-cell, 80-digit interval audit certifies the strict bound "
            r"$\mathcal{E}_2<0.02$, the dominance "
            r"$|\lambda_1|/|\lambda_2|<1$, and the reflected upper-branch phase "
            r"$\Delta\arg\widetilde{\lambda}_2=-\pi$, "
            r"$\Delta\arg\widetilde{c}_2=-\pi$."
        )
    else:
        interval_status = (
            "The requested global interval/phase audit is incomplete: "
            + interval_result["failure"]
        )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\title{{Audit of the $W$-Normalized $n=4$ Root-Counting Argument}}
\author{{Wendel Le\'on Alvarado}}
\date{{November 12, 2026}}
\begin{{document}}
\maketitle

\section*{{Exact identities}}
For $H_{{0,4}}=1$, exact symbolic calculation verifies
\[
H_{{3,4}}-A H_{{2,4}}-B H_{{1,4}}=C.
\]
The Sturm counts for $q_4$ are
${data["q4"]["negative_roots"]}$ negative roots, with one on each side of
$-1$. Since $q_4(-1)={data["q4"]["at_minus_one"]}<0$, and $t<0$ throughout
$I_4$, it follows that $tq_4(t)>0$ on its interior.

The exact polynomial gcd and Jordan checks give
\[
\gcd(D_6,S_6)={data["D6_S6_gcd"]},\qquad
\operatorname{{rem}}(d_3,S_6)={data["jordan_remainder"]}.
\]
The repeated-root value of $\operatorname{{num}}_{{\rm im}}$ is divisible
by $S_6$; its quotient is recorded in the JSON certificate.

\section*{{Sturm and interval certification}}
For $H_{{1,4}}=t^3+11t^2+11t+1$, the exact Sturm variation counts are
${data["h1"]["sturm_variations"]["negative_infinity"]}$ at $-\infty$ and
${data["h1"]["sturm_variations"]["positive_infinity"]}$ at $+\infty$; the polynomial has
{data["h1"]["distinct_real_roots"]} distinct real roots.

{interval_status}

The all-$k$ cubic recurrence is proved independently by the exact 20-state
frontier transfer. This file audits the reflected upper-half-plane branch
only. The signed q6 continuation and three-window sign argument are checked
by \texttt{{audit\_n4\_q6\_branch.py}}; they are not inferred from the
single-branch phase reported above.

\section*{{Conclusion}}
The exact identities and interval checks are recorded in the
machine-readable certificate. Consult the q6 branch audit and the
structural-transfer certificate for the unconditional root theorem.
\end{{document}}
"""


def certify() -> dict[str, Any]:
    iv.dps = 80
    A = (T + 1) * (T**2 + 22 * T + 1)
    B = T * (7 * T**4 - 28 * T**3 - 22 * T**2 - 28 * T + 7)
    C = T * (T - 1) ** 6 * (T + 1)
    q4 = sp.Poly(T**4 + 76 * T**3 - 282 * T**2 + 76 * T + 1, T)
    s6 = sp.Poly(T**6 + 7 * T**5 + 27 * T**4 + 58 * T**3 + 27 * T**2 + 7 * T + 1, T)
    d6 = sp.Poly(sp.expand(A**2 + 3 * B), T)
    p = sp.expand(-B - A**2 / 3)
    q = sp.expand(-2 * A**3 / 27 - A * B / 3 - C)
    p8 = (
        T**8 + 113 * T**7 + 2074 * T**6 + 11119 * T**5
        + 21770 * T**4 + 11119 * T**3 + 2074 * T**2 + 113 * T + 1
    )
    s0 = sp.cancel(2 * (T + 1) * p8 / (3 * d6.as_expr()))
    f = lambda value: sp.cancel(value**3 + p * value + q)
    s0_cardano_residual = sp.factor(
        f(s0) - 8 * T * (T + 1) * q4.as_expr() * s6.as_expr() ** 2 * p8
        / d6.as_expr() ** 3
    )
    if s0_cardano_residual != 0:
        raise ArithmeticError("The supplied s0 Cardano residual identity failed.")

    polynomial_data = json.loads(
        RESULTS_PATH.read_text(encoding="utf-8")
    )["polynomial_data"]
    h0 = sp.Poly(1, T)
    h1 = sp.Poly(T**3 + 11 * T**2 + 11 * T + 1, T)
    h1_variations = {
        "negative_infinity": _variations_at(h1, -sp.oo),
        "positive_infinity": _variations_at(h1, sp.oo),
    }
    h1_real_roots = (
        h1_variations["negative_infinity"] - h1_variations["positive_infinity"]
    )
    h1_isolations = _isolations(h1)
    if h1_real_roots != 3 or len(h1_isolations) != 3:
        raise ArithmeticError("Sturm counts do not certify three distinct H1 roots.")
    h2 = sp.Poly.from_list(
        list(reversed([int(value) for value in polynomial_data["2"]["h_star_coefficients_ascending"]])),
        gens=T,
    )
    h3 = sp.Poly.from_list(
        list(reversed([int(value) for value in polynomial_data["3"]["h_star_coefficients_ascending"]])),
        gens=T,
    )
    recurrence_residual = sp.expand(
        h3.as_expr() - A * h2.as_expr() - B * h1.as_expr() - C * h0.as_expr()
    )
    if recurrence_residual != 0:
        raise ArithmeticError("The H0=1 recurrence identity failed.")

    s = sp.symbols("s")
    f_s = s**3 + p * s + q
    lambda_single_s = s + A / 3
    re_pair_s = -s / 2 + A / 3
    im_pair_sq_s = 3 * s**2 / 4 + p
    pair_modulus_sq_s = sp.expand(re_pair_s**2 + im_pair_sq_s)
    coefficient_denominator_s = 3 * s**2 + p
    coefficient_single_s = sp.cancel(
        (
            h2.as_expr()
            - 2 * re_pair_s * h1.as_expr()
            + pair_modulus_sq_s
        )
        / coefficient_denominator_s
    )
    numerator_im_s = sp.cancel(
        re_pair_s * (1 - coefficient_single_s) / 2
        - (h1.as_expr() - coefficient_single_s * lambda_single_s) / 2
    )
    rem_sym = sp.cancel(coefficient_denominator_s * numerator_im_s)
    rem_numerator, rem_denominator = sp.together(rem_sym).as_numer_denom()
    rem_sym_reduced = sp.rem(
        sp.Poly(rem_numerator, s),
        sp.Poly(f_s, s),
        domain=sp.QQ.frac_field(T),
    )
    rem_sym_reduced_expr = sp.cancel(
        rem_sym_reduced.as_expr() / rem_denominator
    )
    p12 = (
        3 * T**12 + 316 * T**11 + 5863 * T**10 + 55260 * T**9
        + 270233 * T**8 + 686888 * T**7 + 911994 * T**6
        + 686888 * T**5 + 270233 * T**4 + 55260 * T**3
        + 5863 * T**2 + 316 * T + 3
    )
    s0_rem_identity = sp.factor(
        rem_sym_reduced_expr.subs(s, s0)
        + 2 * T * (T + 1) * s6.as_expr() * p12 / d6.as_expr() ** 2
    )
    c2_rem_coefficient = -(T + 1) * (T**2 + 4 * T + 1) / 2
    c1_rem_coefficient = (T + 1) ** 2 * (
        T**4 + 23 * T**3 + 72 * T**2 + 23 * T + 1
    ) / 6
    rem_c2_identity = sp.factor(
        sp.diff(rem_sym_reduced_expr, s, 2) / 2 - c2_rem_coefficient
    )
    rem_c1_identity = sp.factor(
        sp.diff(rem_sym_reduced_expr, s)
        - 2 * c2_rem_coefficient * s
        - c1_rem_coefficient
    )
    if s0_rem_identity != 0 or rem_c2_identity != 0 or rem_c1_identity != 0:
        raise ArithmeticError("An exact canceled Im(c2) remainder identity failed.")

    q4_variations = {
        "negative_infinity": _variations_at(q4, -sp.oo),
        "minus_one": _variations_at(q4, sp.Integer(-1)),
        "zero": _variations_at(q4, sp.Integer(0)),
        "positive_infinity": _variations_at(q4, sp.oo),
    }
    negative_roots = (
        q4_variations["negative_infinity"] - q4_variations["zero"]
    )
    roots_left = (
        q4_variations["negative_infinity"] - q4_variations["minus_one"]
    )
    roots_right = q4_variations["minus_one"] - q4_variations["zero"]
    if (negative_roots, roots_left, roots_right) != (2, 1, 1):
        raise ArithmeticError("Sturm counts do not certify the two negative q4 roots.")
    if q4.eval(-1) != -432:
        raise ArithmeticError("Unexpected q4(-1).")

    gcd_d6_s6 = sp.gcd(d6, s6)
    if gcd_d6_s6.degree() != 0:
        raise ArithmeticError("D6 and S6 are not coprime.")

    double_num = sp.expand(-(A * B + 9 * C) / 2)
    single_num = sp.expand(A * d6.as_expr() - 2 * double_num)
    jordan = sp.expand(
        d6.as_expr() ** 2 * h3.as_expr()
        - d6.as_expr() * (single_num + double_num) * h2.as_expr()
        + single_num * double_num * h1.as_expr()
    )
    jordan_remainder = sp.rem(jordan, s6.as_expr(), T)
    if jordan_remainder != 0:
        raise ArithmeticError("The Jordan remainder modulo S6 is nonzero.")

    lambda_double = sp.cancel(double_num / d6.as_expr())
    lambda_single = sp.cancel(single_num / d6.as_expr())
    c1_collision = sp.cancel(
        (h2.as_expr() - 2 * lambda_double * h1.as_expr() + lambda_double**2)
        / (lambda_single - lambda_double) ** 2
    )
    re_c2_collision = sp.cancel((1 - c1_collision) / 2)
    num_im_collision = sp.factor(
        lambda_double * re_c2_collision
        - (h1.as_expr() - c1_collision * lambda_single) / 2
    )
    num_im_num, num_im_den = sp.fraction(num_im_collision)
    num_im_quotient, num_im_remainder = sp.div(
        sp.Poly(num_im_num, T), s6, domain=sp.QQ
    )
    if num_im_remainder.as_expr() != 0 or sp.gcd(sp.Poly(num_im_den, T), s6).degree() != 0:
        raise ArithmeticError("The collision numerator is not exactly divisible by S6.")

    q4_intervals = q4.intervals(eps=ISOLATION_EPSILON)
    negative_q4_isolations = [
        entry
        for entry in _isolations(q4)
        if sp.Rational(entry["upper"]) < 0
    ]
    s6_intervals = s6.intervals(eps=ISOLATION_EPSILON)
    iv_root_checks = []
    for (lo, hi), multiplicity in s6_intervals:
        if hi >= 0:
            continue
        x = iv.mpf([str(lo), str(hi)])
        q_value = _iv_poly(q4, x)
        product = x * q_value
        if product.a <= 0:
            raise ArithmeticError("A negative S6 root isolator failed t*q4>0.")
        iv_root_checks.append(
            {
                "isolator": [str(lo), str(hi)],
                "multiplicity": multiplicity,
                "t_q4_iv80": str(product),
                "positive": True,
            }
        )
    try:
        interval_result = {
            "certified": True,
            **_interval_audit(q4, s6, d6, h1, h2),
        }
    except ArithmeticError as error:
        interval_result = {
            "certified": False,
            "failure": str(error),
        }

    certificate = {
        "status": (
            "upper_half_plane_interval_subaudit_verified"
            if interval_result["certified"]
            else "exact_subclaims_verified_global_interval_proof_incomplete"
        ),
        "theorem_status": (
            "This interval subaudit certifies the reflected upper branch only; "
            "the signed q6 branch and root count are handled by "
            "audit_n4_q6_branch.py."
            if interval_result["certified"]
            else "not proved or disproved"
        ),
        "precision": {"mpmath_iv_decimal_digits": 80},
        "normalized_recurrence": {
            "H0": "1",
            "exact_residual": str(recurrence_residual),
            "verified": True,
            "scope": (
                "This interval subaudit checks H0 through H3; the all-k "
                "recurrence is verified by verify_n4_frontier_transfer.py."
            ),
        },
        "h1": {
            "polynomial": str(h1.as_expr()),
            "sturm_variations": h1_variations,
            "distinct_real_roots": h1_real_roots,
            "isolations": h1_isolations,
            "verified": True,
        },
        "q4": {
            "sturm_variations": q4_variations,
            "negative_roots": negative_roots,
            "roots_left_of_minus_one": roots_left,
            "roots_right_of_minus_one": roots_right,
            "at_minus_one": str(q4.eval(-1)),
            "negative_root_isolations": negative_q4_isolations,
            "t_q4_positive_on_open_I4": True,
        },
        "D6_S6_gcd": sp.sstr(gcd_d6_s6.as_expr()),
        "s0_cardano_residual_identity": {
            "exact_residual": sp.sstr(s0_cardano_residual),
            "verified": True,
            "s0": sp.sstr(s0),
        },
        "cancelled_im_c2_identities": {
            "P12_remainder_at_s0_residual": sp.sstr(s0_rem_identity),
            "c2_remainder_coefficient_residual": sp.sstr(rem_c2_identity),
            "c1_remainder_coefficient_residual": sp.sstr(rem_c1_identity),
            "verified": True,
        },
        "jordan_remainder": sp.sstr(jordan_remainder),
        "collision_numerator": {
            "num_im_remainder_after_division_by_S6": str(num_im_remainder.as_expr()),
            "exactly_divisible": True,
        },
        "iv80_s6_root_checks": iv_root_checks,
        "global_interval_audit": interval_result,
        "global_claims": {
            "2000_cell_envelope_bound": (
                "certified" if interval_result["certified"] else "not certified"
            ),
            "continuous_phase_lift": (
                "certified with endpoint branch identified"
                if interval_result["certified"]
                else "not certified"
            ),
            "all_k_real_rootedness": (
                "not assessed by this upper-branch interval subaudit; see "
                "audit_n4_q6_branch.py and verify_n4_frontier_transfer.py"
                if interval_result["certified"]
                else "not proved or disproved"
            ),
        },
    }
    JSON_PATH.write_text(
        json.dumps(certificate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return certificate


def main() -> None:
    result = certify()
    print("N=4 W-normalized proof audit")
    print("=" * 72)
    print("H0=1 recurrence identity: exact.")
    print(
        f"q4 Sturm count: {result['q4']['negative_roots']} negative roots; "
        "one on each side of -1."
    )
    print(f"gcd(D6,S6): {result['D6_S6_gcd']}.")
    print(f"Jordan remainder modulo S6: {result['jordan_remainder']}.")
    print(
        "Collision numerator divisibility by S6: "
        f"{result['collision_numerator']['exactly_divisible']}."
    )
    interval_result = result["global_interval_audit"]
    if interval_result["certified"]:
        print("The 2,000-cell envelope and phase enclosures passed.")
        print(
            "Certified phase lift: "
            f"{interval_result['exact_phase_lift']['Phi_k']}."
        )
        print(
            "This is the reflected upper branch only; run "
            "audit_n4_q6_branch.py for the signed branch and root theorem."
        )
    else:
        print("The 2,000-cell envelope and phase lift are NOT certified.")
        print(f"First interval obstruction: {interval_result['failure']}")
        print("The all-k theorem is not established.")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
