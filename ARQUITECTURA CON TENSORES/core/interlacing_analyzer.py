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
from pathlib import Path
from typing import Sequence

import sympy as sp


N3_LEVELS = tuple((k, 3) for k in range(2, 9))
N4_LEVELS = tuple((k, 4) for k in range(3, 8))
DEFAULT_RECURRENCE_DEGREE = 6


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
    ``H[k,n] = sum(A_j(t) * H[k-j,n], j=1..order)``. By default, each
    coefficient is an unrestricted polynomial with degree at most
    ``max_degree``. Optional asymmetric degree bounds and reciprocal spans
    constrain each coefficient to ``t**shift`` times a reciprocal polynomial.
    The function distinguishes unique solutions from finite-data
    underdetermined families.
    """
    if order not in (2, 3):
        raise ValueError("Solo se admiten recurrencias de orden 2 o 3.")
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
    family = load_certified_family(
        certificate_dir, (*N3_LEVELS, *N4_LEVELS)
    )
    print("=" * 72)
    print("ANÁLISIS EXPERIMENTAL: ENTRELAZAMIENTO E INTERLACING")
    print("=" * 72)
    for levels in (N3_LEVELS, N4_LEVELS):
        print(f"\nFamilia n={levels[0][1]}:")
        for (k, n), next_level in zip(levels, levels[1:]):
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
