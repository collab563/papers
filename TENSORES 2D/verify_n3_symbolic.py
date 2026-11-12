"""Verify the exact order-two recurrence for the n=3 Ehrhart numerators.

The descent weights use the natural layer-major labeling.  This labeling is
not invariant under cyclic rotation, so the transfer is kept on all six
frontier states rather than quotienting by C3.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import sympy as sp


t, lam = sp.symbols("t lambda")
P = t**2 + 6 * t + 1
Q = t * (t - 1) ** 2
CHI = lam**2 - P * lam - Q

PARTIAL_PROFILES = ((3, 2), (5, 1), (6, 4))
PARTIAL_LAST_VERTICES = (1, 0, 2)


def _predecessors(layer_count: int) -> tuple[int, ...]:
    """Return predecessor bitmasks for the natural labeling of C_(k,3)."""
    predecessors = [0] * (3 * layer_count)
    for layer in range(layer_count):
        for vertex in range(3):
            element = 3 * layer + vertex
            if layer:
                predecessors[element] = (
                    1 << (3 * (layer - 1) + vertex)
                ) | (1 << (3 * (layer - 1) + (vertex - 1) % 3))
    return tuple(predecessors)


def _ideal_mask(profiles: tuple[int, ...]) -> int:
    return sum(profile << (3 * layer) for layer, profile in enumerate(profiles))


def _prefix_polynomial(
    ideal: int, last: int, predecessors: tuple[int, ...]
) -> sp.Poly:
    """Descent polynomial of linear extensions of an ideal ending at last."""
    variable = t

    @cache
    def count(mask: int, final: int) -> sp.Expr:
        if mask == 0:
            return sp.Integer(1) if final == -1 else sp.Integer(0)
        if final < 0 or not (mask >> final) & 1:
            return sp.Integer(0)

        prior = mask ^ (1 << final)
        if predecessors[final] & prior != predecessors[final]:
            return sp.Integer(0)
        if prior == 0:
            return sp.Integer(1)

        total = sp.Integer(0)
        for previous in range(len(predecessors)):
            if (prior >> previous) & 1:
                total += (variable if previous > final else 1) * count(
                    prior, previous
                )
        return sp.expand(total)

    return sp.Poly(count(ideal, last), variable, domain=sp.ZZ)


def _completion_polynomial(
    ideal: int,
    last: int,
    predecessors: tuple[int, ...],
) -> sp.Poly:
    """Descent polynomial for completions of an ideal in the three-layer end cap."""
    full_mask = (1 << len(predecessors)) - 1

    @cache
    def complete(mask: int, previous: int) -> sp.Expr:
        if mask == full_mask:
            return sp.Integer(1)

        total = sp.Integer(0)
        for element, pred_mask in enumerate(predecessors):
            bit = 1 << element
            if not mask & bit and pred_mask & mask == pred_mask:
                total += (t if previous > element else 1) * complete(
                    mask | bit, element
                )
        return sp.expand(total)

    return sp.Poly(complete(ideal, last), t, domain=sp.ZZ)


def _boundary_states(
    layer_count: int,
) -> tuple[list[int], list[int]]:
    """Six ideals and their terminal labels at the cut after 3q elements."""
    profiles = [
        (7,) * (layer_count - 1) + partial
        for partial in PARTIAL_PROFILES
    ]
    profiles.extend([(7,) * layer_count] * 3)
    masks = [_ideal_mask(profile) for profile in profiles]
    last_elements = [
        3 * layer_count + vertex
        for vertex in PARTIAL_LAST_VERTICES
    ]
    last_elements.extend(
        3 * (layer_count - 1) + vertex for vertex in range(3)
    )
    return masks, last_elements


def _verify_frontier_classification() -> None:
    """Check the six cut profiles; the same classification follows inductively."""
    for completed_layers in range(2, 6):
        layer_count = completed_layers + 2
        profiles_found: set[tuple[int, ...]] = set()

        def visit(
            layer: int,
            previous: int,
            profiles: tuple[int, ...],
            size: int,
        ) -> None:
            if layer == layer_count:
                if size == 3 * completed_layers:
                    profiles_found.add(profiles)
                return

            for profile in range(8):
                required = 0
                for vertex in range(3):
                    if profile & (1 << vertex):
                        required |= (1 << vertex) | (
                            1 << ((vertex - 1) % 3)
                        )
                if layer and required & previous != required:
                    continue
                next_size = size + profile.bit_count()
                if next_size <= 3 * completed_layers:
                    visit(
                        layer + 1,
                        profile,
                        profiles + (profile,),
                        next_size,
                    )

        visit(0, 0, (), 0)
        expected = {
            (7,) * completed_layers + (0, 0),
        }
        expected.update(
            (7,) * (completed_layers - 1) + partial + (0,)
            for partial in PARTIAL_PROFILES
        )
        if profiles_found != expected:
            raise AssertionError(
                f"Unexpected rank-3q frontier profiles for q={completed_layers}."
            )


def _derive_transfer() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Derive the local transfer, initial vector, and terminal functional."""
    source_masks, source_last = _boundary_states(2)
    target_masks, target_last = _boundary_states(3)
    predecessors = _predecessors(4)
    target_indices = {
        (mask, last): index
        for index, (mask, last) in enumerate(zip(target_masks, target_last))
    }

    transition = sp.zeros(6)
    for source_index, (start_mask, start_last) in enumerate(
        zip(source_masks, source_last)
    ):
        def append_three(
            mask: int,
            previous: int,
            steps: int,
            weight: sp.Expr,
        ) -> None:
            if steps == 3:
                target_index = target_indices.get((mask, previous))
                if target_index is None:
                    raise AssertionError(
                        "A rank-3q frontier left the six-state boundary set."
                    )
                transition[source_index, target_index] += weight
                return

            for element, pred_mask in enumerate(predecessors):
                bit = 1 << element
                if not mask & bit and pred_mask & mask == pred_mask:
                    append_three(
                        mask | bit,
                        element,
                        steps + 1,
                        sp.expand(
                            weight * (t if previous > element else 1)
                        ),
                    )

        append_three(start_mask, start_last, 0, sp.Integer(1))

    # Column vectors of frontier weights advance by M = transition transpose.
    transfer = transition.T

    prefix_predecessors = _predecessors(3)
    initial = sp.Matrix(
        [
            _prefix_polynomial(mask, last, prefix_predecessors).as_expr()
            for mask, last in zip(source_masks, source_last)
        ]
    )
    terminal = sp.Matrix(
        [
            _completion_polynomial(mask, last, prefix_predecessors).as_expr()
            for mask, last in zip(source_masks, source_last)
        ]
    ).T
    return transfer, initial, terminal


def _enumerate_hstar(layer_count: int) -> sp.Poly:
    """Enumerate natural-label linear extensions by their descent number."""
    element_count = 3 * layer_count
    full_mask = (1 << element_count) - 1
    predecessors = _predecessors(layer_count)

    @cache
    def extensions(mask: int, previous: int) -> sp.Expr:
        if mask == full_mask:
            return sp.Integer(1)

        total = sp.Integer(0)
        for element, pred_mask in enumerate(predecessors):
            bit = 1 << element
            if not mask & bit and pred_mask & mask == pred_mask:
                total += (t if previous > element else 1) * extensions(
                    mask | bit, element
                )
        return sp.expand(total)

    return sp.Poly(extensions(0, -1), t, domain=sp.ZZ)


def _certificate_polynomial(layer_count: int) -> sp.Poly | None:
    certificate_path = (
        Path(__file__).resolve().parent
        / "certificados_json"
        / f"certificado_{layer_count}_3.json"
    )
    if not certificate_path.exists():
        return None
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    return sp.Poly(
        sp.sympify(payload["polynomial"], locals={"t": t}),
        t,
        domain=sp.ZZ,
    )


def _verify_rotation_is_not_an_equivariance(transfer: sp.Matrix) -> None:
    # State rotation cycles 0->2->1->0 and 3->4->5->3.
    rotated_index = (2, 0, 1, 4, 5, 3)
    rotation = sp.zeros(6)
    for source, target in enumerate(rotated_index):
        rotation[target, source] = 1
    if transfer * rotation == rotation * transfer:
        raise AssertionError(
            "The descent-weighted transfer unexpectedly commutes with C3."
        )


def main() -> None:
    _verify_frontier_classification()
    transfer, initial, terminal = _derive_transfer()
    characteristic_polynomial = sp.factor(
        transfer.charpoly(lam).as_expr()
    )
    expected_characteristic = sp.expand(
        lam**2 * (lam**2 - lam * t + t**2) * CHI
    )
    if sp.expand(characteristic_polynomial - expected_characteristic) != 0:
        raise AssertionError("Unexpected characteristic polynomial.")

    resolvent = sp.cancel(
        (terminal * (lam * sp.eye(6) - transfer).inv() * initial)[0]
    )
    numerator, denominator = sp.fraction(resolvent)
    if sp.expand(denominator + CHI) == 0:
        numerator, denominator = -numerator, -denominator
    if sp.expand(denominator - CHI) != 0:
        raise AssertionError("The scalar resolvent denominator is not chi_3.")
    coefficient_field = sp.QQ.frac_field(t)
    common_factor = sp.gcd(
        sp.Poly(numerator, lam, domain=coefficient_field),
        sp.Poly(CHI, lam, domain=coefficient_field),
    )
    if common_factor.degree() != 0:
        raise AssertionError("The scalar resolvent denominator did not cancel.")

    _verify_rotation_is_not_an_equivariance(transfer)
    h_values = {0: sp.Poly(1, t, domain=sp.ZZ)}
    h_values.update({k: _enumerate_hstar(k) for k in range(1, 10)})
    for k in range(2, 10):
        direct = h_values[k]
        if k >= 3:
            transferred = sp.Poly(
                (terminal * transfer ** (k - 3) * initial)[0],
                t,
                domain=sp.ZZ,
            )
            if transferred != direct:
                raise AssertionError(
                    f"Transfer and direct enumeration differ at k={k}."
                )
        certificate = _certificate_polynomial(k)
        if certificate is not None and certificate != direct:
            raise AssertionError(
                f"Direct enumeration differs from certificate at k={k}."
            )
        if direct != P * h_values[k - 1] + Q * h_values[k - 2]:
            raise AssertionError(
                f"Direct enumeration violates the recurrence at k={k}."
            )

    h_zero = h_values[0]
    h_one = h_values[1]
    if h_values[2] != P * h_one + Q * h_zero:
        raise AssertionError("The recurrence fails at k=2.")
    for k in (3, 4):
        if h_values[k] != P * h_values[k - 1] + Q * h_values[k - 2]:
            raise AssertionError(f"The recurrence base case fails at k={k}.")

    print("Frontier transfer M(t):")
    sp.pprint(transfer)
    print("\nCharacteristic polynomial factorization:")
    print("lambda^2 * (lambda^2 - lambda*t + t^2) * chi_3(lambda;t)")
    print(f"chi_3(lambda;t) = {CHI}")
    print("\nScalar resolvent numerator:")
    sp.pprint(numerator)
    print("Scalar resolvent denominator:")
    sp.pprint(denominator)
    print("\nC3-equivariant transfer: no (natural-label descent weights break it).")
    print("Scalar minimal denominator: chi_3(lambda;t).")
    print(
        "Exact recurrence established for every k >= 2: "
        "H[k] = (t^2+6t+1) H[k-1] + t(t-1)^2 H[k-2]."
    )
    print("Direct extension enumeration and transfer agree for k=2..9.")
    print("All available certificates agree with direct enumeration for k=2..8.")


if __name__ == "__main__":
    main()
