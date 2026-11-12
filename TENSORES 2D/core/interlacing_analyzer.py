"""Root interlacing and exact polynomial-recurrence experiments.

The analyzer consumes the JSON certificates produced by ``root_verifier``.
Root comparisons are exact: SymPy isolates each root in rational intervals,
then those intervals are refined until the roots from adjacent polynomials
have a certified total order. Recurrence searches solve coefficient systems
over ``QQ`` with a configurable upper bound on the degrees of the polynomial
coefficients.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import textwrap
from dataclasses import dataclass
from math import gcd, lcm
from pathlib import Path
from typing import Sequence

import sympy as sp


N3_LEVELS = tuple((k, 3) for k in range(2, 9))
N4_LEVELS = tuple((k, 4) for k in range(3, 8))
N5_LEVELS = tuple((k, 5) for k in range(2, 8))
DEFAULT_RECURRENCE_DEGREE = 6
RECURRENCE_FILTER_PRIME = 2_305_843_009_213_693_951


@dataclass(frozen=True)
class CertifiedPolynomial:
    """A polynomial paired with its validated JSON certificate."""

    k: int
    n: int
    polynomial: sp.Poly
    path: Path


@dataclass(frozen=True)
class RecurrenceSearchResult:
    """Outcome of a bounded exact recurrence search."""

    n: int
    order: int
    degree_bound: int
    coefficient_polynomials: tuple[sp.Expr, ...] | None
    unique: bool
    equations: int
    message: str
    degree_bounds: tuple[int, ...] | None = None
    reciprocal_shifts: tuple[int, ...] | None = None
    verified_levels: tuple[int, ...] = ()
    integer_coefficients: bool | None = None
    candidate_models: int = 0
    free_parameters: int = 0
    coefficient_symmetries: tuple[str, ...] | None = None
    rank: int | None = None
    augmented_rank: int | None = None
    unknowns: int | None = None
    integer_family_solution: bool = False
    integer_parameter: int | None = None
    integer_parameter_denominator: int | None = None
    integer_direction_shift: int | None = None


def _read_certified_polynomial(path: Path, k: int, n: int) -> CertifiedPolynomial:
    """Load a certificate and check the exact facts used by this analysis."""
    data = json.loads(path.read_text(encoding="utf-8"))
    certificate = data.get("sign_certificate")
    if not isinstance(certificate, dict):
        raise ValueError(f"{path} no contiene un certificado de Sturm.")

    coefficients = certificate.get("polynomial_coefficients_descending")
    if not isinstance(coefficients, list) or not coefficients:
        raise ValueError(f"{path} no contiene coeficientes del polinomio.")
    polynomial = sp.Poly.from_list(
        [sp.Rational(coefficient) for coefficient in coefficients],
        gens=sp.Symbol("t"),
        domain=sp.QQ,
    )
    expected_degree = k * n - k
    if polynomial.degree() != expected_degree:
        raise ValueError(
            f"{path}: grado {polynomial.degree()}, se esperaba {expected_degree}."
        )
    if certificate.get("degree") != expected_degree:
        raise ValueError(f"{path} declara un grado de certificado inconsistente.")
    if certificate.get("negative_root_count") != expected_degree:
        raise ValueError(f"{path} no certifica todas las raíces negativas.")
    if not certificate.get("roots_simple"):
        raise ValueError(f"{path} no certifica raíces simples.")
    if not certificate.get("roots_all_real_simple_strictly_negative"):
        raise ValueError(f"{path} no certifica raíces reales negativas simples.")
    if data.get("is_real_rooted_negative_simple") is not True:
        raise ValueError(f"{path} no marca el polinomio como certificado.")
    if not certificate.get("reciprocal_symmetry"):
        raise ValueError(f"{path} no certifica la simetría recíproca.")
    if polynomial.count_roots(-sp.oo, 0) != expected_degree:
        raise ValueError(
            f"El polinomio de {path} no reproduce su cuenta Sturm declarada."
        )
    if polynomial.gcd(polynomial.diff()).degree() != 0:
        raise ValueError(f"El polinomio de {path} no es square-free.")
    return CertifiedPolynomial(k, n, polynomial, path)


def load_certified_family(
    certificate_dir: str | Path,
    levels: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], CertifiedPolynomial]:
    """Load and validate each requested ``certificado_k_n.json`` file."""
    directory = Path(certificate_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio de certificados: {directory}")

    family: dict[tuple[int, int], CertifiedPolynomial] = {}
    for k, n in levels:
        path = directory / f"certificado_{k}_{n}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Falta el certificado requerido: {path}")
        family[(k, n)] = _read_certified_polynomial(path, k, n)
    return family


def _certified_root_order(
    left: CertifiedPolynomial,
    right: CertifiedPolynomial,
) -> tuple[list[str], list[tuple[str, sp.Rational, sp.Rational]]]:
    """Order both root sets by disjoint rational intervals, refining as needed."""
    common = left.polynomial.gcd(right.polynomial)
    if common.degree() > 0 and common.count_roots(-sp.oo, 0) > 0:
        raise ValueError(
            f"H_{{{left.k},{left.n}}} y H_{{{right.k},{right.n}}} "
            "comparten al menos una raíz negativa; el entrelazamiento estricto falla."
        )

    for precision in (20, 40, 80, 160):
        tagged: list[tuple[str, sp.Rational, sp.Rational]] = []
        for label, item in (("r", left), ("s", right)):
            intervals = item.polynomial.intervals(
                eps=sp.Rational(1, 10**precision)
            )
            for (lower, upper), multiplicity in intervals:
                lower, upper = sp.Rational(lower), sp.Rational(upper)
                if upper < 0 and multiplicity == 1:
                    tagged.append((label, lower, upper))
        tagged.sort(key=lambda root: root[1])
        if len(tagged) != left.polynomial.degree() + right.polynomial.degree():
            continue
        if all(first[2] < second[1] for first, second in zip(tagged, tagged[1:])):
            return [root[0] for root in tagged], tagged
    raise ArithmeticError("No se pudo certificar un orden disjunto de raíces.")


def check_strict_interlacing(
    left: CertifiedPolynomial,
    right: CertifiedPolynomial,
) -> tuple[bool, list[str], str]:
    """Check strict alternating order ``r1 < s1 < r2 < s2 < ...`` exactly.

    The stated alternating pattern can hold only when the root counts differ
    by at most one. Root locations and ordering are nevertheless computed
    from exact rational isolating intervals, not floating-point approximations.
    """
    order, _ = _certified_root_order(left, right)
    left_degree = left.polynomial.degree()
    right_degree = right.polynomial.degree()
    if abs(left_degree - right_degree) > 1:
        return (
            False,
            order,
            "los grados difieren en más de uno, por lo que la alternancia "
            "estricta solicitada es imposible",
        )

    expected: list[str] = []
    for _ in range(min(left_degree, right_degree)):
        expected.extend(("r", "s"))
    if left_degree > right_degree:
        expected.append("r")
    elif right_degree > left_degree:
        expected.append("s")
    if order != expected:
        return False, order, "el orden exacto de las raíces no alterna"
    return True, order, "orden alternante certificado con intervalos racionales"


def _recurrence_system(
    family: dict[tuple[int, int], CertifiedPolynomial],
    n: int,
    coefficient_bases: Sequence[Sequence[sp.Expr]],
) -> tuple[sp.Matrix, sp.Matrix, tuple[sp.Symbol, ...], int, tuple[int, ...]]:
    """Build coefficient equations from the requested polynomial bases."""
    levels = sorted(k for k, family_n in family if family_n == n)
    if not levels:
        raise ValueError(f"No hay polinomios cargados para n={n}.")
    variable = sp.Symbol("t")
    order = len(coefficient_bases)
    unknown_count = sum(len(basis) for basis in coefficient_bases)
    unknowns = sp.symbols(f"c0:{unknown_count}")
    rows: list[list[sp.Expr]] = []
    values: list[sp.Expr] = []
    equation_count = 0
    recurrence_levels: list[int] = []

    for k in levels:
        if all((k - offset, n) in family for offset in range(1, order + 1)):
            target = family[(k, n)].polynomial
            shifted_terms = []
            for offset, basis in enumerate(coefficient_bases, start=1):
                previous = family[(k - offset, n)].polynomial
                shifted_terms.extend(
                    sp.Poly(term * previous.as_expr(), variable)
                    for term in basis
                )
            max_degree = max(
                [target.degree(), *(term.degree() for term in shifted_terms)]
            )
            for power in range(max_degree + 1):
                rows.append([term.nth(power) for term in shifted_terms])
                values.append(target.nth(power))
                equation_count += 1
            recurrence_levels.append(k)

    return (
        sp.Matrix(rows),
        sp.Matrix(values),
        unknowns,
        equation_count,
        tuple(recurrence_levels),
    )


def _reciprocal_basis(
    span: int, shift: int, variable: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Basis for ``t**shift`` times polynomials reciprocal of span ``span``."""
    basis = [
        variable**(shift + power) + variable ** (shift + span - power)
        for power in range((span + 1) // 2)
    ]
    if span % 2 == 0:
        basis.append(variable ** (shift + span // 2))
    return tuple(basis)


def _antireciprocal_basis(
    span: int, shift: int, variable: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Basis for shifted polynomials anti-reciprocal of the given span."""
    return tuple(
        variable ** (shift + power) - variable ** (shift + span - power)
        for power in range((span + 1) // 2)
    )


def _reciprocal_mod_t_basis(
    span: int, antireciprocal: bool, variable: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Reciprocal/anti-reciprocal span with both endpoint coefficients zero."""
    basis = [
        variable**power
        + (-1 if antireciprocal else 1) * variable ** (span - power)
        for power in range(1, (span + 1) // 2)
    ]
    if span % 2 == 0 and not antireciprocal:
        basis.append(variable ** (span // 2))
    return tuple(basis)


def _integer_point_on_affine_line(
    origin: Sequence[sp.Expr],
    direction: Sequence[sp.Expr],
) -> tuple[tuple[int, ...], tuple[int, ...], int, int, int] | None:
    """Find the normalized integer point on a rational affine line, if any."""
    if len(origin) != len(direction):
        raise ValueError("El origen y la dirección deben tener la misma dimensión.")

    rational_origin = tuple(sp.Rational(value) for value in origin)
    rational_direction = tuple(sp.Rational(value) for value in direction)
    denominator = 1
    for value in rational_origin:
        denominator = lcm(denominator, int(value.q))
    integer_origin = tuple(
        int(value * denominator) for value in rational_origin
    )

    direction_denominator = 1
    for value in rational_direction:
        direction_denominator = lcm(direction_denominator, int(value.q))
    integer_direction = tuple(
        int(value * direction_denominator) for value in rational_direction
    )
    direction_gcd = 0
    for value in integer_direction:
        direction_gcd = gcd(direction_gcd, abs(value))
    if direction_gcd == 0:
        raise ValueError("La dirección de la familia afín no puede ser cero.")
    primitive_direction = tuple(
        value // direction_gcd for value in integer_direction
    )
    first_nonzero = next(value for value in primitive_direction if value)
    if first_nonzero < 0:
        primitive_direction = tuple(-value for value in primitive_direction)
    if gcd(*primitive_direction) != 1:
        raise ArithmeticError("La dirección entera no es primitiva.")

    residue = 0
    modulus = 1
    for numerator, coefficient in zip(integer_origin, primitive_direction):
        inverse, _, bezout_gcd = sp.gcdex(coefficient, denominator)
        bezout_gcd = int(bezout_gcd)
        inverse = int(inverse)
        if bezout_gcd < 0:
            bezout_gcd = -bezout_gcd
            inverse = -inverse
        right_hand_side = -numerator
        if right_hand_side % bezout_gcd:
            return None

        next_modulus = denominator // bezout_gcd
        if next_modulus == 1:
            next_residue = 0
        else:
            next_residue = (
                inverse * (right_hand_side // bezout_gcd)
            ) % next_modulus

        inverse_modulus, _, common_gcd = sp.gcdex(
            modulus, next_modulus
        )
        common_gcd = int(common_gcd)
        inverse_modulus = int(inverse_modulus)
        difference = next_residue - residue
        if difference % common_gcd:
            return None
        combined_modulus = modulus // common_gcd * next_modulus
        if combined_modulus == 1:
            residue = 0
        else:
            multiplier = (
                (difference // common_gcd) * inverse_modulus
            ) % (next_modulus // common_gcd)
            residue = (residue + modulus * multiplier) % combined_modulus
        modulus = combined_modulus

    if modulus != denominator:
        raise ArithmeticError(
            "La congruencia conjunta no alcanzó el denominador común."
        )
    if 2 * residue > denominator:
        residue -= denominator
    if not (-denominator < 2 * residue <= denominator):
        raise ArithmeticError("El parámetro no está en el intervalo normalizado.")

    integer_numerators = tuple(
        numerator + residue * coefficient
        for numerator, coefficient in zip(integer_origin, primitive_direction)
    )
    if any(value % denominator for value in integer_numerators):
        raise ArithmeticError("El parámetro CRT no produjo un vector entero.")
    integer_solution = tuple(
        value // denominator for value in integer_numerators
    )
    norm_direction_squared = sum(
        coefficient * coefficient for coefficient in primitive_direction
    )
    dot_solution_direction = sum(
        value * coefficient
        for value, coefficient in zip(integer_solution, primitive_direction)
    )
    direction_shift = -(
        2 * dot_solution_direction + norm_direction_squared
    ) // (2 * norm_direction_squared)
    integer_solution = tuple(
        value + direction_shift * coefficient
        for value, coefficient in zip(integer_solution, primitive_direction)
    )
    return (
        integer_solution,
        primitive_direction,
        residue,
        denominator,
        direction_shift,
    )


def _polynomials_from_coordinates(
    coordinates: Sequence[sp.Expr],
    bases: Sequence[Sequence[sp.Expr]],
    variable: sp.Symbol,
) -> tuple[sp.Expr, ...]:
    polynomials: list[sp.Expr] = []
    offset = 0
    for basis in bases:
        polynomials.append(
            sp.Poly(
                sum(
                    coordinates[offset + index] * term
                    for index, term in enumerate(basis)
                ),
                variable,
                domain=sp.QQ,
            ).as_expr()
        )
        offset += len(basis)
    return tuple(polynomials)


def _check_integer_recurrence_at_one(
    coefficient_polynomials: Sequence[sp.Expr],
    scalar_order: int | None,
    scalar_coefficients: Sequence[sp.Rational] | None,
    scalar_free_parameters: int,
) -> tuple[sp.Rational, sp.Rational, bool]:
    """Check the order-five tail and scalar specialization at ``t=1``."""
    if len(coefficient_polynomials) != 5:
        raise ValueError("Se requieren cinco operadores para comprobar orden 5.")
    variable = sp.Symbol("t")
    values = tuple(
        sp.Rational(polynomial.subs(variable, 1))
        for polynomial in coefficient_polynomials
    )
    scalar_values = tuple(scalar_coefficients or ())
    scalar_values += (sp.S.Zero,) * (5 - len(scalar_values))
    matches = (
        scalar_order is not None
        and scalar_free_parameters == 0
        and values == scalar_values
    )
    return values[3], values[4], matches


def _find_n5_symmetric_recurrence(
    family: dict[tuple[int, int], CertifiedPolynomial],
    order: int,
) -> RecurrenceSearchResult:
    """Search reciprocal/anti-reciprocal structural models for n=5."""
    n = 5
    if order not in (4, 5):
        raise ValueError("La búsqueda simétrica n=5 solo admite orden 4 o 5.")
    variable = sp.Symbol("t")
    spans = (4, 8, 12, 16, 20)[:order]
    levels = sorted(k for k, family_n in family if family_n == n)
    equation_levels = [
        k
        for k in levels
        if all((k - offset, n) in family for offset in range(1, order + 1))
    ]
    if not equation_levels:
        expanded_bases = [
            _reciprocal_basis(4, 0, variable),
            *(
                _reciprocal_basis(span, 1, variable)
                for span in spans[1:]
            ),
        ]
        free_parameters = sum(len(basis) for basis in expanded_bases)
        degree_bounds = (5, *(span + 1 for span in spans[1:]))
        return RecurrenceSearchResult(
            n,
            order,
            max(degree_bounds),
            None,
            False,
            0,
            "datos insuficientes: no hay identidades de recurrencia "
            f"de orden {order} para n=5; {free_parameters} parámetros libres "
            "en el modelo ampliado recíproco",
            degree_bounds,
            None,
            (),
            None,
            0,
            free_parameters,
            ("recíproca",) * order,
            0,
            0,
            free_parameters,
        )

    candidates: list[RecurrenceSearchResult] = []
    last_equation_count = 0
    last_degree_bounds = (4, *spans[1:])
    last_symmetries: tuple[str, ...] = ()
    model_options = (
        (False, False, (4, *spans[1:])),
        (True, False, (4, *(span + 1 for span in spans[1:]))),
        (False, True, (5, *spans[1:])),
        (True, True, (5, *(span + 1 for span in spans[1:]))),
    )
    for expanded, expanded_p, degree_bounds in model_options:
        for antisymmetry in itertools.product((False, True), repeat=order - 1):
            p_span = 5 if expanded_p else 4
            bases: list[tuple[sp.Expr, ...]] = [
                _reciprocal_basis(p_span, 0, variable)
            ]
            symmetries = ["recíproca"]
            for span, antireciprocal in zip(spans[1:], antisymmetry):
                if expanded:
                    base = (
                        _antireciprocal_basis(span, 1, variable)
                        if antireciprocal
                        else _reciprocal_basis(span, 1, variable)
                    )
                else:
                    base = _reciprocal_mod_t_basis(
                        span, antireciprocal, variable
                    )
                bases.append(base)
                symmetries.append(
                    "antirrecíproca" if antireciprocal else "recíproca"
                )

            matrix, vector, _, equation_count, verified_levels = (
                _recurrence_system(family, n, bases)
            )
            last_equation_count = equation_count
            last_degree_bounds = degree_bounds
            last_symmetries = tuple(symmetries)
            try:
                solution_matrix, _ = matrix.gauss_jordan_solve(vector)
            except ValueError:
                continue
            solution = tuple(solution_matrix)
            free_symbols = set().union(*(entry.free_symbols for entry in solution))
            free_parameters = len(free_symbols)
            unknown_count = sum(len(basis) for basis in bases)
            rank = unknown_count - free_parameters
            representative: tuple[sp.Expr, ...] = tuple(
                sp.expand(entry.subs({symbol: 0 for symbol in free_symbols}))
                for entry in solution
            )
            integer_family_solution = False
            integer_parameter: int | None = None
            integer_parameter_denominator: int | None = None
            integer_direction_shift: int | None = None
            if free_parameters == 1:
                parameter = next(iter(free_symbols))
                direction = tuple(
                    sp.diff(entry, parameter) for entry in solution
                )
                integer_point = _integer_point_on_affine_line(
                    representative, direction
                )
                if integer_point is not None:
                    (
                        integer_coordinates,
                        _,
                        integer_parameter,
                        integer_parameter_denominator,
                        integer_direction_shift,
                    ) = integer_point
                    representative = tuple(
                        sp.Integer(value) for value in integer_coordinates
                    )
                    integer_family_solution = True
            coefficient_polynomials = _polynomials_from_coordinates(
                representative, bases, variable
            )

            for k in verified_levels:
                target = family[(k, n)].polynomial.as_expr()
                reconstructed = sum(
                    coefficient_polynomials[index - 1]
                    * family[(k - index, n)].polynomial.as_expr()
                    for index in range(1, order + 1)
                )
                if sp.Poly(target - reconstructed, variable, domain=sp.QQ) != 0:
                    raise ArithmeticError(
                        f"La recurrencia no reproduce exactamente el nivel "
                        f"k={k}, n={n}."
                    )

            integer_coefficients = integer_family_solution or (
                not free_parameters and all(
                    coefficient.is_Integer
                    for polynomial in coefficient_polynomials
                    for coefficient in sp.Poly(
                        polynomial, variable
                    ).all_coeffs()
                )
            )
            unique_solution = not free_parameters or integer_family_solution
            candidates.append(
                RecurrenceSearchResult(
                    n,
                    order,
                    max(degree_bounds),
                    coefficient_polynomials,
                    unique_solution,
                    equation_count,
                    "Solución entera única en la familia 1-paramétrica"
                    if integer_family_solution
                    else "familia exacta no única"
                    if free_parameters
                    else "solución exacta única sobre Q",
                    degree_bounds,
                    None,
                    verified_levels,
                    (
                        integer_coefficients
                        if not free_parameters or integer_family_solution
                        else None
                    ),
                    1,
                    free_parameters,
                    tuple(symmetries),
                    rank,
                    rank,
                    unknown_count,
                    integer_family_solution,
                    integer_parameter,
                    integer_parameter_denominator,
                    integer_direction_shift,
                )
            )
        if candidates:
            break

    if not candidates:
        rank = matrix.rank()
        augmented_rank = matrix.row_join(vector).rank()
        unknown_count = sum(len(basis) for basis in bases)
        return RecurrenceSearchResult(
            n,
            order,
            max(last_degree_bounds),
            None,
            False,
            last_equation_count,
            "sin solución exacta con las cotas y simetrías especificadas",
            last_degree_bounds,
            None,
            tuple(equation_levels),
            None,
            0,
            0,
            last_symmetries,
            rank,
            augmented_rank,
            unknown_count,
        )

    candidates.sort(
        key=lambda candidate: (
            not candidate.integer_family_solution,
            candidate.free_parameters != 0,
            candidate.free_parameters,
            candidate.integer_coefficients is not True,
        )
    )
    selected = candidates[0]
    unique_model = selected.unique and len(candidates) == 1
    if selected.integer_family_solution:
        message = "Solución entera única en la familia 1-paramétrica"
    elif unique_model:
        message = (
            "solución exacta única e integral"
            if selected.integer_coefficients
            else "solución exacta única sobre Q, pero no integral"
        )
    elif selected.free_parameters:
        message = f"solución no única: {selected.free_parameters} parámetros libres"
    else:
        message = (
            f"{len(candidates)} modelos de simetría compatibles; "
            "los datos no determinan uno único"
        )
    return RecurrenceSearchResult(
        n,
        order,
        selected.degree_bound,
        selected.coefficient_polynomials,
        unique_model,
        selected.equations,
        message,
        selected.degree_bounds,
        selected.reciprocal_shifts,
        selected.verified_levels,
        selected.integer_coefficients if unique_model else None,
        len(candidates),
        selected.free_parameters,
        selected.coefficient_symmetries,
        selected.rank,
        selected.augmented_rank,
        selected.unknowns,
        selected.integer_family_solution,
        selected.integer_parameter,
        selected.integer_parameter_denominator,
        selected.integer_direction_shift,
    )


def _find_unrestricted_recurrence(
    family: dict[tuple[int, int], CertifiedPolynomial],
    n: int,
    degree_bounds: Sequence[int],
) -> RecurrenceSearchResult:
    """Solve an unrestricted polynomial-operator recurrence over ``QQ``."""
    order = len(degree_bounds)
    if any(bound < 0 for bound in degree_bounds):
        raise ValueError("Las cotas de grado deben ser no negativas.")
    variable = sp.Symbol("t")
    bases = [
        tuple(variable**power for power in range(bound + 1))
        for bound in degree_bounds
    ]
    matrix, vector, _, equations, verified_levels = _recurrence_system(
        family, n, bases
    )
    unknown_count = sum(len(basis) for basis in bases)
    if not verified_levels:
        return RecurrenceSearchResult(
            n,
            order,
            max(degree_bounds),
            None,
            False,
            equations,
            f"datos insuficientes: no hay transiciones para orden {order}",
            tuple(degree_bounds),
            None,
            (),
            None,
            0,
            unknown_count,
            None,
            0,
            0,
            unknown_count,
        )
    modular_consistent, modular_rank, modular_augmented_rank = (
        _modular_consistency(matrix, vector, unknown_count)
    )
    if not modular_consistent:
        return RecurrenceSearchResult(
            n,
            order,
            max(degree_bounds),
            None,
            False,
            equations,
            "obstrucción: sistema inconsistente módulo "
            f"{RECURRENCE_FILTER_PRIME}; esta prueba modular no excluye "
            "una solución sobre Q con denominadores divisibles por el primo",
            tuple(degree_bounds),
            None,
            verified_levels,
            None,
            0,
            0,
            None,
            modular_rank,
            modular_augmented_rank,
            unknown_count,
        )

    try:
        solution_matrix, _ = matrix.gauss_jordan_solve(vector)
    except ValueError:
        rank = matrix.rank()
        augmented_rank = matrix.row_join(vector).rank()
        return RecurrenceSearchResult(
            n,
            order,
            max(degree_bounds),
            None,
            False,
            equations,
            "sistema exactamente inconsistente sobre Q",
            tuple(degree_bounds),
            None,
            verified_levels,
            None,
            0,
            0,
            None,
            rank,
            augmented_rank,
            unknown_count,
        )
    solution = tuple(solution_matrix)
    free_symbols = set().union(*(entry.free_symbols for entry in solution))
    free_parameters = len(free_symbols)
    representative = tuple(
        sp.expand(entry.subs({symbol: 0 for symbol in free_symbols}))
        for entry in solution
    )
    coefficients: list[sp.Expr] = []
    offset = 0
    for basis in bases:
        coefficients.append(
            sp.Poly(
                sum(
                    representative[offset + index] * term
                    for index, term in enumerate(basis)
                ),
                variable,
                domain=sp.QQ,
            ).as_expr()
        )
        offset += len(basis)

    for k in verified_levels:
        target = family[(k, n)].polynomial.as_expr()
        reconstructed = sum(
            coefficients[index - 1]
            * family[(k - index, n)].polynomial.as_expr()
            for index in range(1, order + 1)
        )
        if sp.Poly(target - reconstructed, variable, domain=sp.QQ) != 0:
            raise ArithmeticError(
                f"La recurrencia no reproduce exactamente el nivel k={k}, n={n}."
            )

    unique = free_parameters == 0
    exact_rank = unknown_count - free_parameters
    integer_coefficients = unique and all(
        coefficient.is_Integer
        for polynomial in coefficients
        for coefficient in sp.Poly(polynomial, variable).all_coeffs()
    )
    message = (
        "solución única en Z[t]"
        if integer_coefficients
        else "solución única sobre Q[t]"
        if unique
        else f"familia consistente con {free_parameters} parámetros libres"
    )
    return RecurrenceSearchResult(
        n,
        order,
        max(degree_bounds),
        tuple(coefficients),
        unique,
        equations,
        message,
        tuple(degree_bounds),
        None,
        verified_levels,
        integer_coefficients if unique else None,
        1,
        free_parameters,
        None,
        exact_rank,
        exact_rank,
        unknown_count,
    )


def _modular_consistency(
    matrix: sp.Matrix,
    vector: sp.Matrix,
    unknown_count: int,
    prime: int = RECURRENCE_FILTER_PRIME,
) -> tuple[bool, int, int]:
    """Test consistency over a prime field using integer-only elimination."""
    pivots: dict[int, list[int]] = {}
    rank = 0
    has_inconsistent_row = False
    for row_index in range(matrix.rows):
        row = [
            int(matrix[row_index, column]) % prime
            for column in range(unknown_count)
        ]
        rhs = int(vector[row_index]) % prime
        for pivot_column in sorted(pivots):
            factor = row[pivot_column]
            if factor:
                pivot_row = pivots[pivot_column]
                for column in range(pivot_column, unknown_count):
                    row[column] = (
                        row[column] - factor * pivot_row[column]
                    ) % prime
                rhs = (rhs - factor * pivot_row[unknown_count]) % prime

        pivot_column = next(
            (column for column, value in enumerate(row) if value), None
        )
        if pivot_column is None:
            if rhs:
                has_inconsistent_row = True
            continue

        inverse = pow(row[pivot_column], prime - 2, prime)
        row = [(value * inverse) % prime for value in row]
        normalized_rhs = (rhs * inverse) % prime
        row.append(normalized_rhs)
        pivots[pivot_column] = row
        rank += 1
    augmented_rank = rank + int(has_inconsistent_row)
    return augmented_rank == rank, rank, augmented_rank


def _scalar_recurrence_at_one(
    family: dict[tuple[int, int], CertifiedPolynomial],
    n: int,
) -> tuple[int | None, tuple[sp.Rational, ...] | None, int, int, tuple[int, ...]]:
    """Find the smallest exact scalar recurrence visible at ``t=1``."""
    levels = sorted(k for k, family_n in family if family_n == n)
    values = {
        k: sp.Rational(family[(k, n)].polynomial.eval(1)) for k in levels
    }
    for order in range(1, len(levels)):
        rows: list[list[sp.Rational]] = []
        targets: list[sp.Rational] = []
        transition_levels: list[int] = []
        for k in levels:
            if all((k - offset, n) in family for offset in range(1, order + 1)):
                rows.append([values[k - offset] for offset in range(1, order + 1)])
                targets.append(values[k])
                transition_levels.append(k)
        if not rows:
            continue
        matrix = sp.Matrix(rows)
        vector = sp.Matrix(targets)
        if matrix.rank() != matrix.row_join(vector).rank():
            continue
        solution, _ = matrix.gauss_jordan_solve(vector)
        return (
            order,
            tuple(sp.Rational(value) for value in solution),
            len(rows),
            len(set().union(*(entry.free_symbols for entry in solution))),
            tuple(transition_levels),
        )
    return None, None, 0, 0, ()


def _n5_levels_in_directory(certificate_dir: Path) -> tuple[tuple[int, int], ...]:
    """Return all available n=5 levels, including levels above seven."""
    available = set(N5_LEVELS)
    for path in certificate_dir.glob("certificado_*_5.json"):
        parts = path.stem.split("_")
        if len(parts) == 3 and parts[0] == "certificado" and parts[2] == "5":
            try:
                level = int(parts[1])
            except ValueError:
                continue
            if level >= 2:
                available.add((level, 5))
    return tuple(
        level
        for level in sorted(available)
        if (certificate_dir / f"certificado_{level[0]}_5.json").is_file()
    )


def find_polynomial_recurrence(
    family: dict[tuple[int, int], CertifiedPolynomial],
    n: int,
    order: int,
    max_degree: int = DEFAULT_RECURRENCE_DEGREE,
    coefficient_degree_bounds: Sequence[int] | None = None,
    reciprocal_spans: Sequence[int] | None = None,
) -> RecurrenceSearchResult:
    """Search for a recurrence with polynomial coefficients over ``QQ``.

    The searched identity is
    ``H[k,n] = sum(A_j(t) * H[k-j,n], j=1..order)``. Order four for
    ``n=5`` uses the fixed reciprocal/anti-reciprocal structural models.
    Otherwise, by default, each
    coefficient is an unrestricted polynomial with degree at most
    ``max_degree``. Optional asymmetric degree bounds and reciprocal spans
    constrain each coefficient to ``t**shift`` times a reciprocal polynomial.
    The function distinguishes unique solutions from finite-data
    underdetermined families.
    """
    if n == 5 and order in (4, 5):
        if coefficient_degree_bounds is not None or reciprocal_spans is not None:
            raise ValueError(
                "Las búsquedas n=5 de orden 4 y 5 usan cotas estructurales fijas."
            )
        return _find_n5_symmetric_recurrence(family, order)
    if order not in (2, 3):
        raise ValueError(
            "Solo se admiten órdenes 2 y 3, salvo órdenes 4 y 5 simétricos "
            "para n=5."
        )
    if max_degree < 0:
        raise ValueError("max_degree debe ser no negativo.")
    if (coefficient_degree_bounds is None) != (reciprocal_spans is None):
        raise ValueError(
            "coefficient_degree_bounds y reciprocal_spans deben darse juntos."
        )
    if coefficient_degree_bounds is not None and (
        len(coefficient_degree_bounds) != order
        or len(reciprocal_spans or ()) != order
        or any(bound < 0 for bound in coefficient_degree_bounds)
        or any(span < 0 for span in (reciprocal_spans or ()))
    ):
        raise ValueError("Las cotas y spans deben ser no negativos y tener longitud order.")
    if coefficient_degree_bounds is not None and any(
        span > bound
        for span, bound in zip(reciprocal_spans or (), coefficient_degree_bounds)
    ):
        raise ValueError("Cada reciprocal span debe ser <= su cota de grado.")

    levels = sorted(k for k, family_n in family if family_n == n)
    equation_levels = [
        k
        for k in levels
        if all((k - offset, n) in family for offset in range(1, order + 1))
    ]
    if not equation_levels:
        return RecurrenceSearchResult(
            n,
            order,
            max_degree,
            None,
            False,
            0,
            "datos insuficientes: no hay identidades de recurrencia "
            f"de orden {order} para n={n}",
            tuple(coefficient_degree_bounds) if coefficient_degree_bounds else None,
            None,
            (),
        )

    variable = sp.Symbol("t")
    if coefficient_degree_bounds is None:
        degree_bound_options = (
            (degree_bound,) * order for degree_bound in range(max_degree + 1)
        )
    else:
        degree_bound_options = (tuple(coefficient_degree_bounds),)

    last_equation_count = 0
    last_degree_bounds = (max_degree,) * order
    last_shifts: tuple[int, ...] | None = None
    candidates: list[RecurrenceSearchResult] = []
    for degree_bounds in degree_bound_options:
        last_degree_bounds = tuple(degree_bounds)
        if reciprocal_spans is None:
            shift_options: Sequence[tuple[int, ...]] = ((),)
        else:
            shift_ranges = [
                range(bound - span + 1)
                for bound, span in zip(degree_bounds, reciprocal_spans)
            ]
            shift_options = tuple(
                tuple(shifts)
                for shifts in itertools.product(*shift_ranges)
            )

        for shifts in shift_options:
            last_shifts = shifts if reciprocal_spans is not None else None
            coefficient_bases = []
            for index, bound in enumerate(degree_bounds):
                if reciprocal_spans is None:
                    coefficient_bases.append(
                        tuple(variable**power for power in range(bound + 1))
                    )
                else:
                    coefficient_bases.append(
                        _reciprocal_basis(
                            reciprocal_spans[index], shifts[index], variable
                        )
                    )

            matrix, vector, unknowns, equation_count, verified_levels = (
                _recurrence_system(family, n, coefficient_bases)
            )
            last_equation_count = equation_count
            if matrix.rank() != matrix.row_join(vector).rank():
                continue

            solution_set = sp.linsolve((matrix, vector), unknowns)
            solution = next(iter(solution_set))
            free_symbols = set().union(*(entry.free_symbols for entry in solution))
            unique = not free_symbols
            representative = tuple(
                sp.expand(entry.subs({symbol: 0 for symbol in free_symbols}))
                for entry in solution
            )
            coefficient_polynomials = []
            offset = 0
            for basis in coefficient_bases:
                coefficient_polynomials.append(
                    sp.Poly(
                        sum(
                            representative[offset + index] * term
                            for index, term in enumerate(basis)
                        ),
                        variable,
                        domain=sp.QQ,
                    ).as_expr()
                )
                offset += len(basis)
            for k in verified_levels:
                target = family[(k, n)].polynomial.as_expr()
                reconstructed = sum(
                    coefficient_polynomials[offset - 1]
                    * family[(k - offset, n)].polynomial.as_expr()
                    for offset in range(1, order + 1)
                )
                if sp.Poly(target - reconstructed, variable, domain=sp.QQ) != 0:
                    raise ArithmeticError(
                        f"La recurrencia no reproduce exactamente el nivel "
                        f"k={k}, n={n}."
                    )
            integer_coefficients = unique and all(
                coefficient.is_Integer
                for polynomial in coefficient_polynomials
                for coefficient in sp.Poly(polynomial, variable).all_coeffs()
            )
            candidates.append(RecurrenceSearchResult(
                n,
                order,
                max(degree_bounds),
                tuple(coefficient_polynomials),
                unique,
                equation_count,
                "solución exacta única" if unique else "familia exacta no única",
                tuple(degree_bounds),
                last_shifts,
                verified_levels,
                integer_coefficients,
            ))
        if candidates:
            break

    if candidates:
        candidates.sort(
            key=lambda candidate: (
                not candidate.unique,
                candidate.integer_coefficients is not True,
            )
        )
        selected = candidates[0]
        globally_unique = selected.unique and len(candidates) == 1
        integer_unique = globally_unique and selected.integer_coefficients is True
        if globally_unique:
            message = (
                "solución exacta única e integral"
                if integer_unique
                else "solución exacta única sobre Q, pero no integral"
            )
        elif not selected.unique:
            message = "familia exacta no única"
        else:
            message = (
                f"{len(candidates)} modelos de simetría compatibles; "
                "los datos no determinan uno único"
            )
        return RecurrenceSearchResult(
            n,
            order,
            selected.degree_bound,
            selected.coefficient_polynomials,
            globally_unique,
            selected.equations,
            message,
            selected.degree_bounds,
            selected.reciprocal_shifts,
            selected.verified_levels,
            integer_unique if globally_unique else None,
            len(candidates),
        )

    return RecurrenceSearchResult(
        n,
        order,
        max(last_degree_bounds),
        None,
        False,
        last_equation_count,
        "sin solución exacta con las cotas y simetrías especificadas; "
        "esto no descarta otros grados, spans o modelos de simetría",
        tuple(last_degree_bounds),
        last_shifts,
        tuple(equation_levels),
    )


def run_experimental_analysis(
    certificate_dir: str | Path,
    max_recurrence_degree: int = DEFAULT_RECURRENCE_DEGREE,
) -> None:
    """Print the interlacing and recurrence reports for the available families."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    certificate_path = Path(certificate_dir)
    family = load_certified_family(certificate_path, (*N3_LEVELS, *N4_LEVELS))
    n5_levels = _n5_levels_in_directory(certificate_path)
    if n5_levels:
        family.update(load_certified_family(certificate_path, n5_levels))
    print("=" * 72)
    print("ANÁLISIS EXPERIMENTAL: ENTRELAZAMIENTO E INTERLACING")
    print("=" * 72)
    families = (N3_LEVELS, N4_LEVELS, n5_levels)
    for levels in families:
        if not levels:
            continue
        print(f"\nFamilia n={levels[0][1]}:")
        available_levels = sorted(
            level for level in levels if level in family
        )
        for (k, n), next_level in zip(
            available_levels, available_levels[1:]
        ):
            if next_level[0] != k + 1:
                continue
            current = family[(k, n)]
            following = family[next_level]
            interlaces, _, reason = check_strict_interlacing(current, following)
            encoding = sys.stdout.encoding or "ascii"
            try:
                marker = "✓" if interlaces else "✗"
                marker.encode(encoding)
            except UnicodeEncodeError:
                marker = "[OK]" if interlaces else "[NO]"
            status = "detectado" if interlaces else "no se cumple"
            print(
                f"{marker} Entrelazamiento {status} entre "
                f"H_{{{k},{n}}} y H_{{{k + 1},{n}}}: {reason}."
            )

    print("\n" + "=" * 72)
    print("BÚSQUEDA DE RECURRENCIAS POLINÓMICAS SOBRE Q")
    print("=" * 72)
    print(
        "Para n=4 se buscan coeficientes recíprocos mod t con spans "
        "estructurales 3, 6 y 9. Cotas: orden 2 (P<=4, Q<=8), "
        "orden 3 (P<=4, Q<=7, R<=10). La familia disponible cubre "
        "k=3,...,7."
    )
    for n in (3, 4):
        available = sorted(k for k, family_n in family if family_n == n)
        print(f"\nFamilia n={n}, niveles k={available}:")
        for order in (2, 3):
            if n == 4:
                degree_bounds = (4, 8) if order == 2 else (4, 7, 10)
                reciprocal_spans = (3, 6) if order == 2 else (3, 6, 9)
                result = find_polynomial_recurrence(
                    family,
                    n,
                    order,
                    coefficient_degree_bounds=degree_bounds,
                    reciprocal_spans=reciprocal_spans,
                )
            else:
                result = find_polynomial_recurrence(
                    family, n, order, max_recurrence_degree
                )
            if result.coefficient_polynomials is None:
                try:
                    "✗".encode(sys.stdout.encoding or "ascii")
                    marker = "✗"
                except UnicodeEncodeError:
                    marker = "[NO]"
                print(f"{marker} Orden {order}: {result.message}.")
                continue
            names = ("P", "Q", "R")[:order]
            uniqueness = "única" if result.unique else "no única"
            bounds_text = ", ".join(
                f"{name}<={bound}"
                for name, bound in zip(
                    names,
                    result.degree_bounds or (result.degree_bound,) * order,
                )
            )
            symmetry_text = ""
            if result.reciprocal_shifts is not None:
                symmetry_text = (
                    f", desplazamientos recíprocos={result.reciprocal_shifts}"
                )
            try:
                "✓".encode(sys.stdout.encoding or "ascii")
                marker = "✓"
            except UnicodeEncodeError:
                marker = "[OK]"
            print(
                f"{marker} Orden {order} ({result.message}; cotas {bounds_text}"
                f"{symmetry_text}; {result.equations} ecuaciones; "
                f"verificada en transiciones k={list(result.verified_levels)}; "
                f"modelos compatibles={result.candidate_models})."
            )
            recurrence_terms = " + ".join(
                f"{name}(t) H_{{k-{index},{n}}}(t)"
                for index, name in enumerate(names, start=1)
            )
            print(f"  H_{{k,{n}}}(t) = {recurrence_terms}")
            for name, polynomial in zip(names, result.coefficient_polynomials):
                formatted = textwrap.fill(
                    sp.sstr(polynomial),
                    width=84,
                    subsequent_indent="    ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                print(f"  {name}(t) = {formatted}")
            if result.integer_coefficients is True:
                integrality = "coeficientes enteros"
            elif result.integer_coefficients is False:
                integrality = "coeficientes racionales no enteros"
            else:
                integrality = "integralidad no determinada por la solución no única"
            print(f"  Solución {uniqueness} en los datos; {integrality}.")
            if not result.unique:
                print(
                    "  Advertencia: los datos finitos no determinan una fórmula "
                    "única ni prueban una recurrencia para todo k."
                )

    if n5_levels:
        available = sorted(k for k, family_n in family if family_n == 5)
        print(f"\nFamilia n=5, niveles k={available}:")
        print(
            "  Orden 4: spans (4, 8, 12, 16); P palíndromo; "
            "Q/R/S divisibles por t con simetría recíproca o antirrecíproca; "
            "cotas ampliadas (4, 9, 13, 17)."
        )
        missing_levels = sorted(
            set(range(2, 8)) - set(available)
        )
        if missing_levels:
            print(f"  Certificados ausentes para k={missing_levels}.")
        result = find_polynomial_recurrence(family, n=5, order=4)
        if result.coefficient_polynomials is None:
            print(
                f"  Orden 4 simétrico: {result.message}; "
                f"{result.equations} ecuaciones; "
                f"parámetros libres={result.free_parameters}."
            )
        else:
            names = ("P_5", "Q_5", "R_5", "S_5")
            for name, polynomial in zip(
                names, result.coefficient_polynomials
            ):
                formatted = textwrap.fill(
                    sp.sstr(polynomial),
                    width=84,
                    subsequent_indent="    ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                print(f"  {name}(t) = {formatted}")
            print(
                f"  Orden 4: {result.equations} ecuaciones; "
                f"{'única' if result.unique else 'no única'}; "
                f"{result.message}; "
                f"parámetros libres={result.free_parameters}; "
                f"integral={result.integer_coefficients}; "
                f"simetrías={result.coefficient_symmetries}; "
                f"niveles verificados={list(result.verified_levels)}."
            )
            recurrence_terms = " + ".join(
                f"{name}(t) H_{{k-{index},5}}(t)"
                for index, name in enumerate(names, start=1)
            )
            print(f"  H_{{k,5}}(t) = {recurrence_terms}")

        print("  Diagnóstico de orden 4 sin simetría ni divisibilidad:")
        unrestricted_bounds = (4, 12, 16, 20)
        diagnostic = _find_unrestricted_recurrence(
            family, 5, unrestricted_bounds
        )
        for bounds, candidate in (
            (unrestricted_bounds, diagnostic),
            ((6, 14, 20, 26), _find_unrestricted_recurrence(
                family, 5, (6, 14, 20, 26)
            )),
        ):
            print(
                f"    cotas={bounds}: {candidate.message}; "
                f"ecuaciones={candidate.equations}; "
                f"rango={candidate.rank}/{candidate.augmented_rank}; "
                f"incógnitas={candidate.unknowns}; "
                f"parámetros libres={candidate.free_parameters}; "
                f"transiciones={list(candidate.verified_levels)}."
            )
            if candidate.coefficient_polynomials is not None:
                for name, polynomial in zip(
                    ("P_5", "Q_5", "R_5", "S_5"),
                    candidate.coefficient_polynomials,
                ):
                    print(f"      {name}(t) = {sp.sstr(polynomial)}")

        order5 = _find_unrestricted_recurrence(
            family, 5, (4, 12, 16, 20, 24)
        )
        symmetric_order5 = find_polynomial_recurrence(
            family, n=5, order=5
        )
        print(
            "  Orden 5 simétrico: spans=(4, 8, 12, 16, 20); "
            "P<=4 (ampliable a 5), Q/R/S/T divisibles por t "
            "y recíprocos/antirrecíprocos mod t."
        )
        if symmetric_order5.coefficient_polynomials is None:
            print(
                f"    {symmetric_order5.message}; "
                f"ecuaciones={symmetric_order5.equations}; "
                f"rango={symmetric_order5.rank}/"
                f"{symmetric_order5.augmented_rank}; "
                f"incógnitas={symmetric_order5.unknowns}; "
                f"parámetros libres={symmetric_order5.free_parameters}; "
                f"transiciones={list(symmetric_order5.verified_levels)}."
            )
        else:
            print(
                f"    {symmetric_order5.message}; "
                f"ecuaciones={symmetric_order5.equations}; "
                f"rango={symmetric_order5.rank}/"
                f"{symmetric_order5.augmented_rank}; "
                f"incógnitas={symmetric_order5.unknowns}; "
                f"parámetros libres={symmetric_order5.free_parameters}; "
                f"integral={symmetric_order5.integer_coefficients}; "
                f"simetrías={symmetric_order5.coefficient_symmetries}; "
                f"transiciones={list(symmetric_order5.verified_levels)}; "
                f"alpha={symmetric_order5.integer_parameter}/"
                f"{symmetric_order5.integer_parameter_denominator}; "
                f"reducción beta={symmetric_order5.integer_direction_shift}."
            )
            for name, polynomial in zip(
                ("P_5", "Q_5", "R_5", "S_5", "T_5"),
                symmetric_order5.coefficient_polynomials,
            ):
                print(f"    {name}(t) = {sp.sstr(polynomial)}")
        print(
            "  Orden 5 sin restricciones, cotas=(4, 12, 16, 20, 24): "
            f"{order5.message}; ecuaciones={order5.equations}; "
            f"rango={order5.rank}/{order5.augmented_rank}; "
            f"incógnitas={order5.unknowns}; "
            f"parámetros libres={order5.free_parameters}; "
            f"transiciones={list(order5.verified_levels)}."
        )
        scalar_order, scalar_coefficients, scalar_equations, scalar_free, scalar_levels = (
            _scalar_recurrence_at_one(family, 5)
        )
        scalar_values = [
            (k, sp.Rational(family[(k, 5)].polynomial.eval(1)))
            for k in available
        ]
        print(f"  Valores exactos H_{{k,5}}(1): {scalar_values}.")
        if scalar_order is None:
            print(
                "  Recurrencia escalar en t=1: no determinada con las "
                "transiciones disponibles."
            )
        else:
            print(
                f"  Orden escalar mínimo compatible en t=1: {scalar_order}; "
                f"coeficientes={scalar_coefficients}; "
                f"ecuaciones={scalar_equations}; parámetros libres="
                f"{scalar_free}; transiciones={list(scalar_levels)}."
            )
            if scalar_equations <= scalar_order:
                print(
                    "  El ajuste escalar no tiene transiciones sobrantes "
                    "para validación independiente."
                )
        if symmetric_order5.integer_family_solution:
            s_at_one, t_at_one, scalar_matches = (
                _check_integer_recurrence_at_one(
                    symmetric_order5.coefficient_polynomials or (),
                    scalar_order,
                    scalar_coefficients,
                    scalar_free,
                )
            )
            expected_scalar = (384, -10240, -32768)
            scalar_matches_expected = (
                scalar_order == 3
                and scalar_free == 0
                and tuple(scalar_coefficients or ()) == expected_scalar
            )
            print(
                "  Verificación de solución entera: "
                f"S_5(1)=0: {s_at_one == 0}; "
                f"T_5(1)=0: {t_at_one == 0}; "
                "coincide con la recurrencia escalar "
                f"{expected_scalar} en t=1: "
                f"{scalar_matches and scalar_matches_expected}."
            )

        for order, bounds in (
            (4, (4, 12, 16, 20)),
            (5, (4, 12, 16, 20, 24)),
        ):
            unknown_count = sum(bound + 1 for bound in bounds)
            next_transition = max(2 + order, (max(available) + 1))
            needed = set(range(next_transition - order, next_transition + 1))
            missing_for_first = sorted(needed - set(available))
            print(
                f"  Para empezar a fijar orden {order} "
                f"({unknown_count} incógnitas libres), faltan niveles "
                f"{missing_for_first} para obtener la siguiente transición "
                f"k={next_transition}; después se requieren los niveles "
                f"contiguos k={next_transition + 1}, {next_transition + 2}, "
                f"… hasta rango completo {unknown_count}/{unknown_count}. "
                "El nivel exacto de unicidad depende de los coeficientes "
                "de esos certificados."
            )
        if scalar_order is not None and scalar_equations <= scalar_order:
            print(
                f"  Para validar la recurrencia escalar observada de orden "
                f"{scalar_order} hace falta el siguiente nivel contiguo "
                f"k={max(available) + 1}."
            )
    else:
        print("\nFamilia n=5: no hay certificados disponibles en certificate_dir.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Comprueba entrelazamiento de raíces y busca recurrencias exactas "
            "a partir de certificados JSON de Sturm."
        )
    )
    parser.add_argument(
        "--certificate-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "certificados_json",
        help="Directorio con archivos certificado_k_n.json.",
    )
    parser.add_argument(
        "--max-recurrence-degree",
        type=int,
        default=DEFAULT_RECURRENCE_DEGREE,
        help=(
            "Grado máximo por coeficiente polinómico en el ajuste sobre Q "
            "(por defecto: 3)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point for standalone use."""
    arguments = _parse_args()
    run_experimental_analysis(
        arguments.certificate_dir, arguments.max_recurrence_degree
    )


if __name__ == "__main__":
    main()
