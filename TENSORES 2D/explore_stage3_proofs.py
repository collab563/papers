"""Experimentos exactos para explorar dos estrategias de la Etapa 3.

Los resultados de las pruebas de raíces, entrelazamiento y positividad son
finitos: sirven como diagnóstico y no establecen afirmaciones uniformes.
"""

from __future__ import annotations

from functools import cache
from itertools import combinations, product
from math import comb

import sympy as sp

from verify_n3_symbolic import _derive_transfer


t, s, lam = sp.symbols("t s lambda")


def _predecessor_masks(k: int, n: int, partial_layer: int) -> tuple[int, ...]:
    """Predecessor masks for k complete layers and a partial next layer."""
    count = k * n + partial_layer
    predecessors = [0] * count
    for layer in range(k + (1 if partial_layer else 0)):
        vertices = n if layer < k else partial_layer
        for vertex in range(vertices):
            element = layer * n + vertex
            if layer:
                predecessors[element] = (
                    1 << ((layer - 1) * n + vertex)
                ) | (1 << ((layer - 1) * n + (vertex - 1) % n))
    return tuple(predecessors)


def _add_coefficients(target: list[int], source: list[int], shift: int) -> None:
    needed = len(source) + shift
    if len(target) < needed:
        target.extend([0] * (needed - len(target)))
    for index, coefficient in enumerate(source):
        target[index + shift] += coefficient


def _descent_polynomial(predecessors: tuple[int, ...]) -> tuple[sp.Poly, list[int]]:
    """Count linear extensions by descents using a DP on order ideals."""
    element_count = len(predecessors)
    full_mask = (1 << element_count) - 1
    layer: dict[int, dict[int, list[int]]] = {0: {-1: [1]}}

    for _size in range(element_count):
        next_layer: dict[int, dict[int, list[int]]] = {}
        for mask, endings in layer.items():
            available = full_mask ^ mask
            while available:
                bit = available & -available
                available ^= bit
                element = bit.bit_length() - 1
                if predecessors[element] & mask != predecessors[element]:
                    continue
                next_mask = mask | bit
                next_endings = next_layer.setdefault(next_mask, {})
                for previous, coefficients in endings.items():
                    shift = int(previous > element)
                    target = next_endings.setdefault(
                        element, [0] * (len(coefficients) + shift)
                    )
                    _add_coefficients(target, coefficients, shift)
        layer = next_layer

    descent_counts = [0] * element_count
    for coefficients in layer[full_mask].values():
        _add_coefficients(descent_counts, coefficients, 0)
    variable = sp.Symbol("t")
    polynomial = sp.Poly(
        sum(count * variable**index for index, count in enumerate(descent_counts)),
        variable,
        domain=sp.ZZ,
    )
    return polynomial, descent_counts


def _descent_components(
    elements: tuple[int, ...], predecessor_closure: tuple[int, ...]
) -> dict[int, sp.Poly]:
    """Descent polynomials of extensions ending at each element."""
    local_index = {element: index for index, element in enumerate(elements)}
    local_predecessors = tuple(
        sum(
            1 << local_index[predecessor]
            for predecessor in elements
            if predecessor_closure[element] & (1 << predecessor)
        )
        for element in elements
    )
    element_count = len(elements)
    full_mask = (1 << element_count) - 1
    layer: dict[int, dict[int, list[int]]] = {0: {-1: [1]}}

    for _size in range(element_count):
        next_layer: dict[int, dict[int, list[int]]] = {}
        for mask, endings in layer.items():
            available = full_mask ^ mask
            while available:
                bit = available & -available
                available ^= bit
                element = bit.bit_length() - 1
                if local_predecessors[element] & mask != local_predecessors[element]:
                    continue
                next_mask = mask | bit
                next_endings = next_layer.setdefault(next_mask, {})
                for previous, coefficients in endings.items():
                    shift = int(
                        previous >= 0
                        and elements[previous] > elements[element]
                    )
                    target = next_endings.setdefault(
                        element, [0] * (len(coefficients) + shift)
                    )
                    _add_coefficients(target, coefficients, shift)
        layer = next_layer

    variable = sp.Symbol("t")
    result: dict[int, sp.Poly] = {}
    for local_last, coefficients in layer.get(full_mask, {}).items():
        expression = sum(
            coefficient * variable**degree
            for degree, coefficient in enumerate(coefficients)
        )
        result[elements[local_last]] = sp.Poly(
            expression, variable, domain=sp.ZZ
        )
    return result


@cache
def _cached_descent_components(
    elements: tuple[int, ...], predecessor_closure: tuple[int, ...]
) -> tuple[tuple[int, sp.Poly], ...]:
    return tuple(_descent_components(elements, predecessor_closure).items())


def _p_partition_counts(
    descent_counts: list[int], element_count: int, degree: int
) -> list[int]:
    """Count P-partitions via the exact linear-extension decomposition."""
    return [
        sum(
            descent_counts[j] * comb(m + element_count - j, element_count)
            for j in range(min(m, len(descent_counts) - 1) + 1)
        )
        for m in range(degree + 1)
    ]


def _recover_from_p_partition_counts(
    counts: list[int], element_count: int, degree: int
) -> sp.Poly:
    coefficients: list[int] = []
    for m, count in enumerate(counts):
        coefficient = count - sum(
            coefficients[j] * comb(m + element_count - j, element_count)
            for j in range(m)
        )
        coefficients.append(coefficient)
    variable = sp.Symbol("t")
    return sp.Poly(
        sum(value * variable**index for index, value in enumerate(coefficients)),
        variable,
        domain=sp.ZZ,
    )


def _negative_simple_roots(polynomial: sp.Poly) -> tuple[bool, list[tuple[sp.Rational, sp.Rational]]]:
    degree = polynomial.degree()
    if degree <= 0 or sp.degree(sp.gcd(polynomial, polynomial.diff()).as_expr()) != 0:
        return False, []
    if polynomial.count_roots(-sp.oo, 0) != degree:
        return False, []
    intervals = sp.polys.polytools.intervals(
        polynomial, eps=sp.Rational(1, 10**20)
    )
    roots: list[tuple[sp.Rational, sp.Rational]] = []
    for (left, right), multiplicity in intervals:
        if multiplicity != 1 or right >= 0:
            return False, []
        roots.append((left, right))
    return len(roots) == degree, roots


def _strictly_interlace(
    first: sp.Poly,
    second: sp.Poly,
    first_roots: list[tuple[sp.Rational, sp.Rational]],
    second_roots: list[tuple[sp.Rational, sp.Rational]],
) -> bool:
    if abs(first.degree() - second.degree()) > 1:
        return False
    if sp.gcd(first, second).degree() > 0:
        return False

    precision = 20
    while precision <= 320:
        epsilon = sp.Rational(1, 10**precision)
        first_roots = [
            interval
            for interval, _ in sp.polys.polytools.intervals(first, eps=epsilon)
        ]
        second_roots = [
            interval
            for interval, _ in sp.polys.polytools.intervals(second, eps=epsilon)
        ]
        ordered = []
        for left, right in first_roots:
            ordered.append((left, right, 0))
        for left, right in second_roots:
            ordered.append((left, right, 1))
        ordered.sort(key=lambda item: item[0])
        if all(ordered[i][1] < ordered[i + 1][0] for i in range(len(ordered) - 1)):
            pattern = [item[2] for item in ordered]
            if first.degree() == second.degree():
                return pattern in (
                    [0, 1] * first.degree(),
                    [1, 0] * first.degree(),
                )
            if first.degree() > second.degree():
                return pattern == [0, 1] * second.degree() + [0]
            return pattern == [1, 0] * first.degree() + [1]
        precision *= 2
    return False


def _run_interpolation_experiment() -> None:
    print("EXPERIMENTO 1: interpolación vértice a vértice")
    for n in (3, 4):
        for k in (2, 3, 4):
            previous: sp.Poly | None = None
            previous_roots: list[tuple[sp.Rational, sp.Rational]] = []
            print(f"  n={n}, k={k}:")
            for r in range(n + 1):
                predecessors = _predecessor_masks(k, n, r)
                direct_polynomial, descent_counts = _descent_polynomial(predecessors)
                degree = direct_polynomial.degree()
                expected_degree = k * (n - 1) + max(0, r - 1)
                if degree != expected_degree:
                    raise AssertionError(
                        f"deg H(P_{r})={degree}; se esperaba {expected_degree} "
                        f"para n={n}, k={k}."
                    )

                counts = _p_partition_counts(
                    descent_counts, len(predecessors), degree
                )
                recovered = _recover_from_p_partition_counts(
                    counts, len(predecessors), degree
                )
                if recovered != direct_polynomial:
                    raise AssertionError(
                        f"La conversión exacta de L(m) falló en n={n}, k={k}, r={r}."
                    )

                polynomial = recovered
                root_ok, roots = _negative_simple_roots(polynomial)
                print(
                    f"    P_{r}: |P|={len(predecessors)}, grado={degree}, "
                    f"L(0..d) exacto={len(counts)} valores, "
                    f"raíces simples negativas={root_ok}"
                )
                if previous is not None:
                    interlaces = (
                        root_ok
                        and _negative_simple_roots(previous)[0]
                        and _strictly_interlace(
                            previous, polynomial, previous_roots, roots
                        )
                    )
                    print(f"      P_{r-1} y P_{r}: entrelazamiento estricto={interlaces}")
                previous = polynomial
                previous_roots = roots

            print(
                f"    Simetría de grado verificada: "
                f"deg(P_0)={k * (n - 1)}, deg(P_1)={k * (n - 1)}, "
                f"deg(P_r)=k(n-1)+r-1 para r>=2."
            )


def _companion_matrix(order: int, coefficients: tuple[sp.Expr, ...]) -> sp.Matrix:
    matrix = sp.zeros(order)
    for column, coefficient in enumerate(coefficients):
        matrix[0, column] = coefficient
    for row in range(1, order):
        matrix[row, row - 1] = 1
    return matrix


def _sign_variations(vector: sp.Matrix) -> int:
    signs = []
    for value in vector:
        sign = sp.sign(value)
        if sign:
            signs.append(sign)
    return sum(signs[i] != signs[i - 1] for i in range(1, len(signs)))


def _all_minors(matrix: sp.Matrix, order: int) -> list[int]:
    signs: list[int] = []
    for rows in combinations(range(matrix.rows), order):
        for columns in combinations(range(matrix.cols), order):
            determinant = matrix.extract(rows, columns).det(method="domain-ge")
            signs.append(int(sp.sign(determinant)))
    return signs


def _all_minor_signs(matrix: sp.Matrix) -> list[int]:
    return [
        sign
        for order in range(1, matrix.rows + 1)
        for sign in _all_minors(matrix, order)
    ]


def _principal_minor_signs(matrix: sp.Matrix) -> list[int]:
    signs = []
    for order in range(1, matrix.rows + 1):
        for indices in combinations(range(matrix.rows), order):
            determinant = matrix.extract(indices, indices).det(method="domain-ge")
            signs.append(int(sp.sign(determinant)))
    return signs


def _sign_summary(signs: list[int]) -> str:
    positives = sum(sign > 0 for sign in signs)
    zeros = sum(sign == 0 for sign in signs)
    negatives = sum(sign < 0 for sign in signs)
    return f"+{positives}/0:{zeros}/-{negatives}"


def _find_diagonal_sign_conjugations(matrix: sp.Matrix) -> list[tuple[int, ...]]:
    solutions = []
    for tail in product((-1, 1), repeat=matrix.rows - 1):
        diagonal = (1,) + tail
        if all(
            diagonal[row] * matrix[row, column] * diagonal[column] >= 0
            for row in range(matrix.rows)
            for column in range(matrix.cols)
        ):
            solutions.append(diagonal)
    return solutions


def _run_transfer_experiment() -> None:
    print("\nEXPERIMENTO 2: positividad total y oscilación bajo t=-s")
    transfer3, initial3, terminal3 = _derive_transfer()
    companion3 = _companion_matrix(
        2, (t**2 + 6 * t + 1, t * (t - 1) ** 2)
    )
    companion4 = _companion_matrix(
        3,
        (
            t**3 + 23 * t**2 + 23 * t + 1,
            7 * t**5 - 28 * t**4 - 22 * t**3 - 28 * t**2 + 7 * t,
            t * (t - 1) ** 6 * (t + 1),
        ),
    )
    print(
        "  Compañera n=3 en t=-s:",
        companion3.subs(t, -s),
    )
    print(
        "  Compañera n=4 en t=-s:",
        companion4.subs(t, -s),
    )
    print(
        "  Polinomios característicos en t=-s:",
        sp.factor(companion3.subs(t, -s).charpoly(lam).as_expr()),
        "y",
        sp.factor(companion4.subs(t, -s).charpoly(lam).as_expr()),
    )
    first3 = sp.eye(2)[:, 0]
    first4 = sp.eye(3)[:, 0]
    companion_resolvent3 = sp.cancel(
        (first3.T * (lam * sp.eye(2) - companion3).inv() * first3)[0]
    ).subs(t, -s)
    companion_resolvent4 = sp.cancel(
        (first4.T * (lam * sp.eye(3) - companion4).inv() * first4)[0]
    ).subs(t, -s)
    print(
        "  Resolventes escalares e_1^T(lambda I-C)^(-1)e_1 en t=-s:",
        sp.factor(companion_resolvent3),
        "y",
        sp.factor(companion_resolvent4),
    )
    resolvent3 = sp.cancel(
        (terminal3 * (lam * sp.eye(6) - transfer3).inv() * initial3)[0]
    )
    resolvent_denominator = sp.denom(resolvent3)
    chi3 = lam**2 - (t**2 + 6 * t + 1) * lam - t * (t - 1) ** 2
    if sp.expand(resolvent_denominator + chi3) == 0:
        resolvent_denominator = -resolvent_denominator
    print(
        "  Transferencia exacta n=3: dimensión 6; denominador escalar del "
        "resolvente =",
        sp.factor(resolvent_denominator),
    )

    sample_points = (
        sp.Rational(1, 20),
        sp.Rational(1, 10),
        sp.Rational(1, 2),
        sp.Integer(1),
        sp.Integer(3),
        sp.Integer(7),
        sp.Integer(12),
        sp.Integer(14),
        sp.Integer(20),
    )
    interval_low = 7 - 4 * sp.sqrt(3)
    interval_high = 7 + 4 * sp.sqrt(3)
    conjugation_counts: dict[int, int] = {power: 0 for power in range(1, 9)}
    tn_counts: dict[int, int] = {power: 0 for power in range(1, 4)}
    tp_counts: dict[int, int] = {power: 0 for power in range(1, 4)}
    for value in sample_points:
        specialized = transfer3.subs(t, -value)
        signs = _find_diagonal_sign_conjugations(specialized)
        location = "dentro" if value > interval_low and value < interval_high else "fuera"
        print(
            f"  s={value} ({location} del intervalo): "
            f"conjugación diagonal no negativa={'sí' if signs else 'no'}"
        )
        original_power = sp.eye(6)
        power_signs: dict[int, list[tuple[int, ...]]] = {}
        for power in range(1, 9):
            original_power *= specialized
            power_signs = _find_diagonal_sign_conjugations(original_power)
            if power_signs:
                conjugation_counts[power] += 1
            print(f"    M_3(-s)^{power}: conjugación diagonal no negativa={'sí' if power_signs else 'no'}")
            if power <= 3:
                matrix_power = original_power
                if signs:
                    diagonal = sp.diag(*signs[0])
                    matrix_power = diagonal * original_power * diagonal
                signs_1 = _all_minors(matrix_power, 1)
                signs_2 = _all_minors(matrix_power, 2)
                signs_3 = _all_minors(matrix_power, 3)
                principal = _principal_minor_signs(matrix_power)
                all_minors = _all_minor_signs(matrix_power)
                totally_nonnegative = all(sign >= 0 for sign in all_minors)
                strictly_totally_positive = all(sign > 0 for sign in all_minors)
                tn_counts[power] += int(totally_nonnegative)
                tp_counts[power] += int(strictly_totally_positive)
                print(
                    f"      menores 1/2/3={_sign_summary(signs_1)}/"
                    f"{_sign_summary(signs_2)}/{_sign_summary(signs_3)}, "
                    f"todos los menores principales={_sign_summary(principal)}, "
                    f"todos los menores (TN)={totally_nonnegative}, "
                    f"TP={strictly_totally_positive}"
                )

        raw_reachable = initial3.subs(t, -value)
        raw_variations_ok = True
        raw_variation_trace = [_sign_variations(raw_reachable)]
        for _ in range(8):
            next_reachable = specialized * raw_reachable
            if _sign_variations(next_reachable) > _sign_variations(raw_reachable):
                raw_variations_ok = False
            raw_reachable = next_reachable
            raw_variation_trace.append(_sign_variations(raw_reachable))
        print(
            "    M sin conjugación: V no creciente desde v_1(-s)=",
            raw_variations_ok,
            "; V=",
            raw_variation_trace,
        )

    print(
        "  Conjugaciones no negativas halladas para potencias M^p en la malla:",
        ", ".join(
            f"p={power}: {count} puntos" for power, count in conjugation_counts.items()
        ),
    )
    print(
        "  Potencias TN/TP en la malla (usando D para la matriz si existe; "
        "si no, base original):",
        "; ".join(
            f"p={power}: TN={tn_counts[power]}, TP={tp_counts[power]}"
            for power in range(1, 4)
        ),
    )
    print(
        "  TN exige todos los menores no negativos; oscilatoriedad exige además "
        "una potencia totalmente positiva. Los resultados siguen siendo finitos "
        "en s y p."
    )


def _transitive_predecessor_masks(
    cover_predecessors: tuple[int, ...],
) -> tuple[int, ...]:
    closure: list[int] = []
    for element, direct in enumerate(cover_predecessors):
        predecessors = direct
        pending = direct
        while pending:
            bit = pending & -pending
            pending ^= bit
            predecessors |= closure[bit.bit_length() - 1]
        closure.append(predecessors)
    return tuple(closure)


def _maximal_elements(
    elements: tuple[int, ...], predecessor_closure: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        element
        for element in elements
        if not any(
            predecessor_closure[upper] & (1 << element)
            for upper in elements
            if upper != element
        )
    )


def _induced_descent_polynomial(
    elements: tuple[int, ...], predecessor_closure: tuple[int, ...]
) -> sp.Poly:
    """Descent polynomial of a naturally labelled induced subposet."""
    local_index = {element: index for index, element in enumerate(elements)}
    local_predecessors = tuple(
        sum(
            1 << local_index[predecessor]
            for predecessor in elements
            if predecessor_closure[element] & (1 << predecessor)
        )
        for element in elements
    )
    polynomial, _ = _descent_polynomial(local_predecessors)
    return polynomial


def _upper_filters(
    elements: tuple[int, ...],
    predecessor_closure: tuple[int, ...],
    excluded_ideal: int,
) -> list[tuple[int, ...]]:
    eligible = tuple(
        element for element in elements if not excluded_ideal & (1 << element)
    )
    filters = []
    for choice in range(1, 1 << len(eligible)):
        selected = tuple(
            eligible[index]
            for index in range(len(eligible))
            if choice & (1 << index)
        )
        selected_mask = sum(1 << element for element in selected)
        if all(
            not (
                predecessor_closure[upper] & (1 << lower)
                and not selected_mask & (1 << upper)
            )
            for lower in selected
            for upper in elements
        ):
            filters.append(selected)
    return filters


def _real_root_intervals(
    polynomial: sp.Poly, precision: int
) -> list[tuple[sp.Rational, sp.Rational]]:
    if polynomial.is_zero or polynomial.degree() < 0:
        raise ValueError("El polinomio cero no tiene un multiconjunto finito de raíces.")
    if polynomial.degree() == 0:
        return []
    if polynomial.count_roots(-sp.oo, sp.oo) != polynomial.degree():
        raise ValueError(f"{polynomial.as_expr()} no es real-raíz.")
    intervals = sp.polys.polytools.intervals(
        polynomial, eps=sp.Rational(1, 10**precision)
    )
    return [
        interval
        for interval, multiplicity in intervals
        for _ in range(multiplicity)
    ]


def _has_common_interlacer(polynomials: list[sp.Poly]) -> bool:
    """Check the equal-degree weak common-interlacing criterion exactly."""
    nonzero = [polynomial for polynomial in polynomials if not polynomial.is_zero]
    if not nonzero:
        return False
    if any(polynomial.degree() != nonzero[0].degree() for polynomial in nonzero):
        return False
    degree = nonzero[0].degree()
    if degree <= 1:
        return True
    if any(
        polynomial.count_roots(-sp.oo, sp.oo) != degree
        for polynomial in nonzero
    ):
        return False

    for precision in (20, 40, 80, 160, 320, 640):
        root_sets = [
            _real_root_intervals(polynomial, precision)
            for polynomial in nonzero
        ]
        feasible = True
        for index in range(degree - 1):
            greatest_left_upper = max(roots[index][1] for roots in root_sets)
            least_right_lower = min(roots[index + 1][0] for roots in root_sets)
            if greatest_left_upper > least_right_lower:
                feasible = False
                break
        if feasible:
            return True
    return False


def _real_nonpositive_roots(polynomial: sp.Poly) -> bool:
        if polynomial.is_zero:
            return False
        degree = polynomial.degree()
        if degree <= 0:
            return True
        nonzero_root_factor = polynomial
        while nonzero_root_factor.eval(0) == 0:
            nonzero_root_factor = sp.div(
                nonzero_root_factor, sp.Poly(t, t, domain=sp.ZZ)
            )[0]
        if nonzero_root_factor.degree() == 0:
            return True
        if nonzero_root_factor.count_roots(-sp.oo, sp.oo) != nonzero_root_factor.degree():
            return False
        intervals = sp.polys.polytools.intervals(
            nonzero_root_factor, eps=sp.Rational(1, 10**30)
        )
        return all(interval[1] <= 0 for (interval, _) in intervals)


def _multiplicity_in_isolating_interval(
        polynomial: sp.Poly, left: sp.Rational, right: sp.Rational
) -> int:
        if left == right:
            factor = sp.Poly(t - left, t, domain=sp.QQ)
            multiplicity = 0
            remainder = polynomial
            while remainder.eval(left) == 0:
                remainder = sp.div(remainder, factor)[0]
                multiplicity += 1
            return multiplicity
        current = polynomial
        multiplicity = 0
        while current.degree() > 0:
            if current.count_roots(left, right) == 0:
                break
            multiplicity += 1
            current = current.gcd(current.diff())
        return multiplicity


def _directed_weak_interlacing(
        first: sp.Poly, second: sp.Poly
) -> bool:
        """Test f <= g: roots alternate, with the larger-degree roots surrounding."""
        if first.is_zero or second.is_zero:
            return False
        degree_first = first.degree()
        degree_second = second.degree()
        if degree_first > degree_second or degree_second - degree_first > 1:
            return False
        if not _real_nonpositive_roots(first) or not _real_nonpositive_roots(second):
            return False

        product = first * second
        distinct_roots = product.sqf_part()
        if distinct_roots.degree() == 0:
            return True
        intervals = sp.polys.polytools.intervals(
            distinct_roots, eps=sp.Rational(1, 10**40)
        )
        prefix_first = 0
        prefix_second = 0
        for (left, right), _multiplicity in intervals:
            prefix_first += _multiplicity_in_isolating_interval(first, left, right)
            prefix_second += _multiplicity_in_isolating_interval(second, left, right)
            difference = prefix_first - prefix_second
            if degree_first == degree_second:
                if abs(difference) > 1:
                    return False
            elif difference not in (0, -1):
                return False
        return prefix_first == degree_first and prefix_second == degree_second


def _solve_component_matrix(
        input_components: list[sp.Poly], output_components: list[sp.Poly]
) -> sp.Matrix | None:
        """Find a rational constant matrix representing the observed vectors."""
        if not input_components:
            return None
        variable_count = len(input_components)
        max_degree = max(
            [poly.degree() for poly in input_components + output_components],
            default=0,
        )
        matrix_rows = [
            [input_poly.nth(degree) for input_poly in input_components]
            for degree in range(max_degree + 1)
        ]
        input_matrix = sp.Matrix(matrix_rows)
        output_rows = []
        for output_poly in output_components:
            target = sp.Matrix(
                [output_poly.nth(degree) for degree in range(max_degree + 1)]
            )
            try:
                solution, parameters = input_matrix.gauss_jordan_solve(target)
            except ValueError:
                return None
            if parameters.rows:
                parameter_substitution = {parameter: 0 for parameter in parameters}
                solution = solution.subs(parameter_substitution)
            output_rows.append([sp.Rational(value) for value in solution])
        return sp.Matrix(output_rows)


def _solve_polynomial_component_matrix(
        input_components: list[sp.Poly], output_components: list[sp.Poly]
) -> list[list[sp.Poly]] | None:
        """Find a minimum-degree exact matrix over Q[t] for the observed vectors."""
        variable = input_components[0].gen
        max_output_degree = max(
            (poly.degree() for poly in output_components), default=0
        )
        max_input_degree = max(
            (poly.degree() for poly in input_components), default=0
        )
        max_bound = max_output_degree
        for bound in range(max_bound + 1):
            equation_degree = max(max_output_degree, max_input_degree + bound)
            coefficient_matrix = sp.Matrix(
                [
                    [
                        input_poly.nth(degree - shift)
                        if degree >= shift
                        else 0
                        for input_poly in input_components
                        for shift in range(bound + 1)
                    ]
                    for degree in range(equation_degree + 1)
                ]
            )
            rows: list[list[sp.Poly]] = []
            all_rows_solved = True
            for output_poly in output_components:
                target = sp.Matrix(
                    [
                        output_poly.nth(degree)
                        for degree in range(equation_degree + 1)
                    ]
                )
                try:
                    solution, parameters = coefficient_matrix.gauss_jordan_solve(
                        target
                    )
                except ValueError:
                    all_rows_solved = False
                    break
                if parameters.rows:
                    solution = solution.subs(
                        {parameter: 0 for parameter in parameters}
                    )
                row = []
                for input_index in range(len(input_components)):
                    expression = sum(
                        solution[input_index * (bound + 1) + shift]
                        * variable**shift
                        for shift in range(bound + 1)
                    )
                    row.append(sp.Poly(expression, variable, domain=sp.QQ))
                rows.append(row)
            if all_rows_solved:
                return rows
        return None


def _polynomial_matrix_product(
        matrix: list[list[sp.Poly]], vector: list[sp.Poly]
) -> list[sp.Poly]:
        variable = vector[0].gen
        return [
            sp.Poly(
                sp.expand(
                    sum(
                        matrix[row][column].as_expr()
                        * vector[column].as_expr()
                        for column in range(len(vector))
                    )
                ),
                variable,
                domain=sp.QQ,
            )
            for row in range(len(matrix))
        ]


def _matrix_minor_signs(matrix: sp.Matrix) -> list[int]:
        signs = []
        for order in range(1, min(matrix.rows, matrix.cols) + 1):
            for rows in combinations(range(matrix.rows), order):
                for columns in combinations(range(matrix.cols), order):
                    determinant = matrix.extract(rows, columns).det()
                    signs.append(int(sp.sign(determinant)))
        return signs


def _run_maximal_element_experiment() -> None:
        print("\nEXPERIMENTO 4: vector de componentes por último elemento maximal")
        cases = ((3, 2), (3, 3), (4, 2), (4, 3), (5, 2))
        sequence_checks = 0
        sequence_successes = 0
        vertex_identity_checks = 0
        vertex_identity_successes = 0
        component_identity_checks = 0
        component_identity_successes = 0
        operator_checks = 0
        operator_tn = 0
        for n, k in cases:
            print(f"  n={n}, k={k}:")
            previous_components: list[sp.Poly] | None = None
            for r in range(n + 1):
                elements = tuple(range(k * n + r))
                closure = _transitive_predecessor_masks(
                    _predecessor_masks(k, n, r)
                )
                maxima = tuple(
                    sorted(_maximal_elements(elements, closure), reverse=True)
                )
                all_components = _descent_components(elements, closure)
                components = [all_components[maximum] for maximum in maxima]

                root_checks = [
                    _real_nonpositive_roots(component)
                    for component in components
                ]
                pair_results = [
                    _directed_weak_interlacing(components[left], components[right])
                    for left in range(len(components))
                    for right in range(left + 1, len(components))
                ]
                sequence_holds = all(root_checks) and all(pair_results)
                sequence_checks += 1
                sequence_successes += int(sequence_holds)
                print(
                    f"    r={r}: máximos(desc)={maxima}, grados="
                    f"{[component.degree() for component in components]}, "
                    f"raíces no positivas={all(root_checks)}, "
                    f"pares dirigidos={sum(pair_results)}/{len(pair_results)}, "
                    f"secuencia completa={sequence_holds}"
                )
                if not all(root_checks):
                    failed_labels = [
                        maxima[index]
                        for index, valid in enumerate(root_checks)
                        if not valid
                    ]
                    print(f"      componentes sin raíces todas no positivas: {failed_labels}")

                if previous_components is not None:
                    new_vertex = k * n + (r - 1)
                    new_vertex_component = all_components[new_vertex]
                    previous_total = sum(
                        (component.as_expr() for component in previous_components),
                        sp.Integer(0),
                    )
                    vertex_identity = sp.expand(
                        new_vertex_component.as_expr() - previous_total
                    ) == 0
                    vertex_identity_checks += 1
                    vertex_identity_successes += int(vertex_identity)
                    if not vertex_identity:
                        raise AssertionError(
                            f"H^({new_vertex})_P{r} != H_P{r-1} para n={n}, k={k}."
                        )

                    component_identities = []
                    component_boundary_terms: dict[
                        int, tuple[tuple[int, ...], sp.Expr]
                    ] = {}
                    for maximum in maxima:
                        if maximum == new_vertex:
                            continue
                        without_maximum = tuple(
                            element for element in elements if element != maximum
                        )
                        induced_components = _descent_components(
                            without_maximum, closure
                        )
                        induced_maxima = _maximal_elements(
                            without_maximum, closure
                        )
                        boundary_expression = sum(
                            (
                                (t if last > maximum else 1)
                                * induced_components[last].as_expr()
                                for last in induced_maxima
                            ),
                            sp.Integer(0),
                        )
                        component_identity = sp.expand(
                            all_components[maximum].as_expr()
                            - boundary_expression
                        ) == 0
                        component_identities.append(component_identity)
                        component_boundary_terms[maximum] = (
                            induced_maxima,
                            sp.expand(boundary_expression),
                        )
                    component_identity_checks += len(component_identities)
                    component_identity_successes += sum(component_identities)
                    if not all(component_identities):
                        raise AssertionError(
                            f"Identidad por subposet borrado falló para n={n}, k={k}, r={r}."
                        )

                    input_components = previous_components
                    output_components = components
                    transformation = _solve_component_matrix(
                        input_components, output_components
                    )
                    polynomial_transformation = _solve_polynomial_component_matrix(
                        input_components, output_components
                    )
                    operator_checks += 1
                    if transformation is None:
                        print(
                            f"      identidades: fila v_r={vertex_identity}, "
                            f"componentes vía P\\{{m}}={sum(component_identities)}/"
                            f"{len(component_identities)}."
                        )
                    else:
                        predicted = transformation * sp.Matrix(
                            [component.as_expr() for component in input_components]
                        )
                        maps_exactly = all(
                            sp.expand(
                                predicted[index]
                                - output_components[index].as_expr()
                            )
                            == 0
                            for index in range(len(output_components))
                        )
                        signs = _matrix_minor_signs(transformation)
                        is_totally_nonnegative = all(sign >= 0 for sign in signs)
                        operator_tn += int(is_totally_nonnegative)
                        print(
                            f"      matriz constante "
                            f"{transformation.rows}x{transformation.cols} exacta="
                            f"{maps_exactly}, TN={is_totally_nonnegative}, "
                            f"signos de menores={_sign_summary(signs)}"
                        )
                        print(f"      matriz racional T_r={transformation.tolist()}")

                    for maximum, (induced_maxima, expression) in (
                        component_boundary_terms.items()
                    ):
                        print(
                            f"      H_P{r}^({maximum}) = "
                            f"sum_(p in max((P{r-1}\\{{{maximum}}}) + {{v_r}})) "
                            f"factor frontera por p="
                            f"{[(p, 't' if p > maximum else '1') for p in induced_maxima]}; "
                            f"polinomio exacto={expression}; "
                            f"identidad exacta={sp.expand(expression - all_components[maximum].as_expr()) == 0}"
                        )

                    if polynomial_transformation is None:
                        print(
                            "      no se encontró matriz polinómica exacta "
                            "sobre Q[t] dentro de la cota buscada."
                        )
                    else:
                        polynomial_image = _polynomial_matrix_product(
                            polynomial_transformation, input_components
                        )
                        polynomial_matrix_exact = all(
                            sp.expand(
                                polynomial_image[index].as_expr()
                                - output_components[index].as_expr()
                            )
                            == 0
                            for index in range(len(output_components))
                        )
                        polynomial_matrix_degree = max(
                            (
                                entry.degree()
                                for row in polynomial_transformation
                                for entry in row
                            ),
                            default=0,
                        )
                        print(
                            "      matriz polinómica T_r(t) exacta en este vector="
                            f"{polynomial_matrix_exact}; grado máximo de entradas="
                            f"{polynomial_matrix_degree}"
                        )

                previous_components = components

        print(
            "  Resumen: secuencia completa "
            f"{sequence_successes}/{sequence_checks}; identidad de v_r "
            f"{vertex_identity_successes}/{vertex_identity_checks}; identidades "
            f"por subposet P\\{{m}}={component_identity_successes}/"
            f"{component_identity_checks}; matrices constantes inspeccionadas "
            f"{operator_checks}, TN={operator_tn}."
        )
        print(
            "  Sólo una matriz constante TN pasa el certificado matricial "
            "suficiente usado aquí; una matriz polinómica que reproduce este vector "
            "particular no certifica que el operador preserve toda secuencia."
        )
def _run_filter_decomposition_experiment() -> None:
    print("\nEXPERIMENTO 3: descomposición recursiva por filtros superiores")
    cases = ((3, 2), (3, 3), (4, 2))
    identity_checks = 0
    identity_successes = 0
    common_interlacer_checks = 0
    common_interlacer_successes = 0
    for n, k in cases:
        print(f"  n={n}, k={k}:")
        for r in range(1, n + 1):
            q_cover_predecessors = _predecessor_masks(k, n, r - 1)
            q_predecessors = _transitive_predecessor_masks(
                q_cover_predecessors
            )
            elements = tuple(range(k * n + r - 1))
            new_vertex = k * n + (r - 1)
            q_plus_predecessors = _transitive_predecessor_masks(
                _predecessor_masks(k, n, r)
            )
            ideal_below_new_vertex = q_plus_predecessors[new_vertex]
            filters = _upper_filters(
                elements, q_predecessors, ideal_below_new_vertex
            )
            polynomial_q = _induced_descent_polynomial(
                elements, q_predecessors
            )
            polynomial_q_plus_v = _induced_descent_polynomial(
                tuple(range(new_vertex + 1)), q_plus_predecessors
            )
            terms = [polynomial_q]
            right_hand_side = polynomial_q.as_expr()
            for filter_elements in filters:
                filter_mask = sum(1 << element for element in filter_elements)
                complement = tuple(
                    element
                    for element in elements
                    if not filter_mask & (1 << element)
                )
                filter_polynomial = _induced_descent_polynomial(
                    filter_elements, q_predecessors
                )
                complement_polynomial = _induced_descent_polynomial(
                    complement, q_predecessors
                )
                term = sp.Poly(
                    sp.expand(
                        t
                        * filter_polynomial.as_expr()
                        * complement_polynomial.as_expr()
                    ),
                    t,
                    domain=sp.ZZ,
                )
                terms.append(term)
                right_hand_side += term.as_expr()

            right_hand_side_poly = sp.Poly(
                sp.expand(right_hand_side), t, domain=sp.ZZ
            )
            identity_holds = right_hand_side_poly == polynomial_q_plus_v
            identity_checks += 1
            identity_successes += int(identity_holds)

            degrees = sorted({term.degree() for term in terms})
            all_terms_real_rooted = all(
                term.count_roots(-sp.oo, sp.oo) == term.degree()
                for term in terms
            )
            non_real_terms = [
                term.as_expr()
                for term in terms
                if term.count_roots(-sp.oo, sp.oo) != term.degree()
            ]
            common_interlacer = _has_common_interlacer(terms)
            common_applicable = len(degrees) == 1 and all_terms_real_rooted
            if common_applicable:
                common_interlacer_checks += 1
                common_interlacer_successes += int(common_interlacer)
            if not all_terms_real_rooted:
                common_status = "no: hay sumandos no real-raíz"
            elif not common_applicable:
                common_status = "no evaluable por criterio clásico: grados distintos"
            else:
                common_status = "sí" if common_interlacer else "no"
            sum_is_real_rooted = (
                polynomial_q_plus_v.count_roots(-sp.oo, sp.oo)
                == polynomial_q_plus_v.degree()
            )
            q_real_rooted = (
                polynomial_q.count_roots(-sp.oo, sp.oo)
                == polynomial_q.degree()
            )
            print(
                f"    r={r}: filtros no vacíos={len(filters)}, "
                f"identidad exacta={identity_holds}, "
                f"grados de sumandos={degrees}, "
                f"sumandos real-raíz={all_terms_real_rooted}, "
                f"entrelazador común={common_status}, "
                f"H_Q real-raíz={q_real_rooted}, "
                f"H_(Q+v) real-raíz={sum_is_real_rooted}"
            )
            if non_real_terms:
                print(f"      testigo de sumando no real-raíz: {non_real_terms[0]}")
            if not identity_holds:
                print(
                    "      diferencia exacta RHS-LHS=",
                    sp.factor(right_hand_side_poly.as_expr() - polynomial_q_plus_v.as_expr()),
                )
    print(
        "  Resumen: identidad exacta "
        f"{identity_successes}/{identity_checks}; criterio clásico de "
        "entrelazamiento común aplicable en "
        f"{common_interlacer_checks}/{identity_checks} pasos y satisfecho en "
        f"{common_interlacer_successes}/{common_interlacer_checks} de ellos."
    )
    print(
        "  El criterio de Obreschkoff/Brändén citado requiere una familia "
        "real-raíz de grado común; no se infiere su conclusión en los pasos "
        "con grados mixtos."
    )


def _profile_elements(profile: tuple[int, ...], n: int) -> tuple[int, ...]:
    return tuple(
        layer * n + vertex
        for vertex, height in enumerate(profile)
        for layer in range(height)
    )


def _heights_from_elements(
    elements: tuple[int, ...], k: int, n: int
) -> tuple[int, ...]:
    element_set = set(elements)
    return tuple(
        sum(layer * n + vertex in element_set for layer in range(k))
        for vertex in range(n)
    )


def _run_ideal_deletion_experiment() -> None:
    print("\nEXPERIMENTO 5: clausura inductiva por borrado de máximos")
    cases = ((3, 3), (4, 3), (3, 4), (3, 5))
    for k, n in cases:
        absolute_difference_profiles = tuple(
            profile
            for profile in product(range(k + 1), repeat=n)
            if any(profile)
            and all(
                abs(profile[index] - profile[(index + 1) % n]) <= 1
                for index in range(n)
            )
        )
        profiles = tuple(
            profile
            for profile in product(range(k + 1), repeat=n)
            if any(profile)
            and all(
                profile[index] <= profile[(index - 1) % n] + 1
                for index in range(n)
            )
        )
        profile_set = set(profiles)
        absolute_difference_ideals = tuple(
            profile
            for profile in absolute_difference_profiles
            if profile in profile_set
        )
        prefix_profiles = tuple(
            profile
            for profile in profiles
            if all(profile[index] >= profile[index + 1] for index in range(n - 1))
        )
        closure = _transitive_predecessor_masks(
            _predecessor_masks(k, n, 0)
        )
        rooted_count = 0
        complete_sequence_count = 0
        prefix_count = 0
        prefix_sequence_count = 0
        deletion_checks = 0
        deletion_successes = 0
        deletion_closure_checks = 0
        deletion_closure_successes = 0
        base_case_checks = 0
        base_case_successes = 0

        for profile in profiles:
            elements = _profile_elements(profile, n)
            components = dict(
                _cached_descent_components(elements, closure)
            )
            total_polynomial = sp.Poly(
                sp.expand(
                    sum(
                        (polynomial.as_expr() for polynomial in components.values()),
                        sp.Integer(0),
                    )
                ),
                t,
                domain=sp.ZZ,
            )
            degree = total_polynomial.degree()
            simple_negative_roots = (
                degree == 0
                or (
                    total_polynomial.gcd(total_polynomial.diff()).degree() == 0
                    and total_polynomial.count_roots(-sp.oo, 0) == degree
                )
            )
            rooted_count += int(simple_negative_roots)

            maxima = tuple(
                sorted(_maximal_elements(elements, closure), reverse=True)
            )
            sequence_holds = all(
                _real_nonpositive_roots(components[maximum])
                for maximum in maxima
            ) and all(
                _directed_weak_interlacing(
                    components[maxima[left]], components[maxima[right]]
                )
                for left in range(len(maxima))
                for right in range(left + 1, len(maxima))
            )
            complete_sequence_count += int(sequence_holds)
            if profile in prefix_profiles:
                prefix_count += 1
                prefix_sequence_count += int(sequence_holds)

            if len(elements) == 1:
                base_case_checks += 1
                base_case_successes += int(
                    len(components) == 1
                    and next(iter(components.values())) == sp.Poly(1, t)
                )
                continue

            for maximum in maxima:
                remaining = tuple(
                    element for element in elements if element != maximum
                )
                remaining_profile = _heights_from_elements(remaining, k, n)
                if remaining_profile not in profile_set:
                    raise AssertionError(
                        f"Quitar máximo {maximum} de {profile} no produjo "
                        "un ideal de alturas admisible."
                    )

                remaining_components = dict(
                    _cached_descent_components(remaining, closure)
                )
                remaining_maxima = _maximal_elements(remaining, closure)
                right_hand_side = sum(
                    (
                        (t if last > maximum else 1)
                        * remaining_components[last].as_expr()
                        for last in remaining_maxima
                    ),
                    sp.Integer(0),
                )
                exact_identity = sp.expand(
                    components[maximum].as_expr() - right_hand_side
                ) == 0
                deletion_checks += 1
                deletion_successes += int(exact_identity)
                child_is_ideal = remaining_profile in profile_set
                deletion_closure_checks += 1
                deletion_closure_successes += int(child_is_ideal)
                if not exact_identity or not child_is_ideal:
                    raise AssertionError(
                        f"Identidad o clausura por borrado falló para perfil "
                        f"{profile}, máximo {maximum}."
                    )

        not_actual_examples = tuple(
            profile
            for profile in absolute_difference_profiles
            if profile not in profile_set
        )
        omitted_examples = tuple(
            profile
            for profile in profiles
            if profile not in set(absolute_difference_profiles)
        )
        nonclosed_absolute_example = None
        for profile in absolute_difference_profiles:
            elements = _profile_elements(profile, n)
            maxima = _maximal_elements(elements, closure)
            for maximum in maxima:
                reduced_profile = _heights_from_elements(
                    tuple(element for element in elements if element != maximum),
                    k,
                    n,
                )
                if reduced_profile not in set(absolute_difference_profiles):
                    nonclosed_absolute_example = (profile, maximum, reduced_profile)
                    break
            if nonclosed_absolute_example is not None:
                break

        print(
            f"  C_{{{k},{n}}}: ideales no vacíos (criterio exacto de cierres)="
            f"{len(profiles)}; perfiles con |h_i-h_(i+1)|<=1="
            f"{len(absolute_difference_profiles)} "
            f"(ideales válidos={len(absolute_difference_ideals)}; "
            f"perfiles de ideal omitidos={len(omitted_examples)}; "
            f"perfiles de diferencia absoluta que no son ideales="
            f"{len(not_actual_examples)}); "
            f"H_I real, simple y estrictamente negativo="
            f"{rooted_count}/{len(profiles)} "
            f"({100 * rooted_count / len(profiles):.1f}%)."
        )
        print(
            f"    Secuencia completa dirigida={complete_sequence_count}/"
            f"{len(profiles)} ({100 * complete_sequence_count / len(profiles):.1f}%); "
            f"perfiles ideales con frente prefijo sin salto cíclico="
            f"{prefix_sequence_count}/{prefix_count} "
            f"({100 * prefix_sequence_count / prefix_count if prefix_count else 0:.1f}%)."
        )
        print(
            f"    Identidades de borrado de máximo={deletion_successes}/"
            f"{deletion_checks}; casos base singleton={base_case_successes}/"
            f"{base_case_checks} (H_singleton=1); clausura del conjunto de ideales "
            f"tras borrar máximo={deletion_closure_successes}/"
            f"{deletion_closure_checks}."
        )
        if not_actual_examples:
            print(
                "    Ejemplo de perfil de diferencias absolutas que NO es ideal "
                f"para las coberturas del código: {not_actual_examples[0]}."
            )
        if omitted_examples:
            print(
                "    Ejemplo de ideal válido no capturado por la condición de "
                f"diferencias absolutas: {omitted_examples[0]}."
            )
        if nonclosed_absolute_example is not None:
            profile, maximum, reduced_profile = nonclosed_absolute_example
            print(
                "    La familia de diferencias absolutas no es cerrada al borrar "
                f"máximos: perfil {profile}, m={maximum} -> {reduced_profile}."
            )

    print(
        "  En las coberturas naturales implementadas, un perfil es ideal si y "
        "sólo si h_i <= h_(i-1)+1 cíclicamente. La condición de diferencias "
        "absolutas indicada en la solicitud no caracteriza todos esos ideales "
        "ni es estable bajo borrado maximal."
    )
    print(
        "  El paso inductivo se verifica en cada par (I,m); los singletons "
        "se tratan como caso base H_{ {m} }^(m)=1, pues la suma literal "
        "sobre el conjunto vacio es vacia."
    )


def main() -> None:
    _run_interpolation_experiment()
    _run_transfer_experiment()
    _run_filter_decomposition_experiment()
    _run_maximal_element_experiment()
    _run_ideal_deletion_experiment()


if __name__ == "__main__":
    main()
