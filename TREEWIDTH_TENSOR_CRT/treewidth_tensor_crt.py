"""Exact order-polytope counting with treewidth tensors and CRT reconstruction."""

from __future__ import annotations

import itertools
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(__file__).resolve().with_name("resultados_treewidth_crt.json")
REFERENCE_PATH = ROOT / "BENCHMARK_D40" / "resultados_benchmark_d40.json"
PRIME_BITS = 61
PRIME_LIMIT = (1 << PRIME_BITS) - 1


@dataclass(frozen=True)
class Poset:
    """Finite poset represented by its Hasse cover relations ``lower < upper``."""

    name: str
    elements: tuple[int, ...]
    covers: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("Poset elements must be unique.")
        element_set = set(self.elements)
        if any(u not in element_set or v not in element_set or u == v for u, v in self.covers):
            raise ValueError("Every cover must join two distinct poset elements.")
        if len(set(self.covers)) != len(self.covers):
            raise ValueError("Poset covers must be unique.")
        _topological_order(self.elements, self.covers)

    @property
    def dimension(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class NiceNode:
    """A nice-tree-decomposition node; bags are listed in sorted element order."""

    kind: str
    bag: tuple[int, ...]
    children: tuple["NiceNode", ...] = ()
    vertex: int | None = None


@dataclass(frozen=True)
class TreeDecomposition:
    root: NiceNode
    elimination_order: tuple[int, ...]
    elimination_bags: tuple[tuple[int, ...], ...]
    width: int


def _topological_order(
    elements: Iterable[int], covers: Iterable[tuple[int, int]]
) -> tuple[int, ...]:
    vertices = tuple(elements)
    outgoing = {v: [] for v in vertices}
    indegree = {v: 0 for v in vertices}
    for lower, upper in covers:
        outgoing[lower].append(upper)
        indegree[upper] += 1
    ready = sorted(v for v in vertices if indegree[v] == 0)
    order: list[int] = []
    while ready:
        vertex = ready.pop(0)
        order.append(vertex)
        for successor in outgoing[vertex]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(vertices):
        raise ValueError("Poset relations must form a directed acyclic graph.")
    return tuple(order)


def _transitive_reduction(
    elements: tuple[int, ...], relations: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Return the Hasse covers of a DAG, retaining its original element labels."""
    edges = set(relations)
    order = _topological_order(elements, edges)
    positions = {v: i for i, v in enumerate(order)}
    reduced: list[tuple[int, int]] = []
    adjacency = {v: set() for v in elements}
    for lower, upper in sorted(edges, key=lambda edge: (positions[edge[0]], positions[edge[1]])):
        adjacency[lower].add(upper)

    for lower, upper in sorted(edges, key=lambda edge: (positions[edge[0]], positions[edge[1]])):
        stack = [successor for successor in adjacency[lower] if successor != upper]
        seen: set[int] = set()
        redundant = False
        while stack:
            current = stack.pop()
            if current == upper:
                redundant = True
                break
            if current not in seen:
                seen.add(current)
                stack.extend(adjacency[current])
        if not redundant:
            reduced.append((lower, upper))
    return tuple(sorted(reduced))


def poset_from_relations(
    name: str, dimension: int, relations: Iterable[tuple[int, int]]
) -> Poset:
    """Construct a poset from DAG relations and reduce them to Hasse covers."""
    if dimension < 0:
        raise ValueError("The dimension cannot be negative.")
    elements = tuple(range(dimension))
    covers = _transitive_reduction(elements, relations)
    return Poset(name=name, elements=elements, covers=covers)


def cyclic_crown(k: int, n: int) -> Poset:
    """Return the layered cyclic crown with covers to the same/next column."""
    if k < 1 or n < 2:
        raise ValueError("A crown requires k >= 1 and n >= 2.")
    elements = tuple(range(k * n))
    covers = []
    for level in range(k - 1):
        for column in range(n):
            lower = level * n + column
            covers.append((lower, (level + 1) * n + column))
            covers.append((lower, (level + 1) * n + (column + 1) % n))
    return Poset(f"C_{{{k},{n}}}", elements, tuple(sorted(set(covers))))


def _undirected_adjacency(poset: Poset) -> dict[int, set[int]]:
    adjacency = {v: set() for v in poset.elements}
    for lower, upper in poset.covers:
        adjacency[lower].add(upper)
        adjacency[upper].add(lower)
    return adjacency


def _min_fill_decomposition(poset: Poset) -> TreeDecomposition:
    """Build elimination bags with deterministic minimum-fill tie breaking."""
    adjacency = _undirected_adjacency(poset)
    active = set(poset.elements)
    order: list[int] = []
    bags_by_vertex: dict[int, tuple[int, ...]] = {}
    later_neighbors: dict[int, tuple[int, ...]] = {}

    while active:
        def score(vertex: int) -> tuple[int, int, int]:
            neighbors = sorted(adjacency[vertex] & active)
            fill = sum(
                1
                for i, u in enumerate(neighbors)
                for v in neighbors[i + 1 :]
                if v not in adjacency[u]
            )
            return fill, len(neighbors), vertex

        vertex = min(active, key=score)
        neighbors = tuple(sorted(adjacency[vertex] & active))
        bags_by_vertex[vertex] = tuple(sorted((vertex, *neighbors)))
        later_neighbors[vertex] = neighbors
        order.append(vertex)
        for i, u in enumerate(neighbors):
            for v in neighbors[i + 1 :]:
                adjacency[u].add(v)
                adjacency[v].add(u)
        active.remove(vertex)

    index = {vertex: i for i, vertex in enumerate(order)}
    children: dict[int | None, list[int]] = {None: []}
    for vertex in order:
        later = later_neighbors[vertex]
        parent = min(later, key=index.get) if later else None
        children.setdefault(parent, []).append(vertex)
        children.setdefault(vertex, [])

    max_bag = max((len(bag) for bag in bags_by_vertex.values()), default=0)

    def make_transition(tree: NiceNode, target: tuple[int, ...]) -> NiceNode:
        current = set(tree.bag)
        wanted = set(target)
        for vertex in sorted(current - wanted):
            current.remove(vertex)
            tree = NiceNode("FORGET", tuple(sorted(current)), (tree,), vertex)
        for vertex in sorted(wanted - current):
            current.add(vertex)
            tree = NiceNode("INTRODUCE", tuple(sorted(current)), (tree,), vertex)
        return tree

    def build_vertex(vertex: int) -> NiceNode:
        bag = bags_by_vertex[vertex]
        child_vertices = sorted(children[vertex], key=index.get)
        if not child_vertices:
            tree = NiceNode("LEAF", ())
            tree = make_transition(tree, bag)
        else:
            tree = make_transition(build_vertex(child_vertices[0]), bag)
            for child_vertex in child_vertices[1:]:
                other = make_transition(build_vertex(child_vertex), bag)
                tree = NiceNode("JOIN", bag, (tree, other))
        return tree

    roots = sorted(children[None], key=index.get)
    if not roots:
        root = NiceNode("LEAF", ())
    else:
        root = make_transition(build_vertex(roots[0]), ())
        for vertex in roots[1:]:
            other = make_transition(build_vertex(vertex), ())
            root = NiceNode("JOIN", (), (root, other))

    return TreeDecomposition(
        root=root,
        elimination_order=tuple(order),
        elimination_bags=tuple(bags_by_vertex[v] for v in order),
        width=max_bag - 1 if order else 0,
    )


def _modular_mul_near_2_61(
    left: np.ndarray, right: np.ndarray, prime: int
) -> np.ndarray:
    """Vectorized exact multiplication modulo a 61-bit prime near 2**61."""
    if left.shape != right.shape:
        raise ValueError("Hadamard operands must have identical shapes.")
    gap = PRIME_LIMIT + 1 - prime
    if not 0 < gap < (1 << 20):
        raise ValueError("CRT primes must be within 2**20 of 2**61.")

    a = np.asarray(left, dtype=np.uint64)
    b = np.asarray(right, dtype=np.uint64)
    mask32 = np.uint64((1 << 32) - 1)
    mask61 = np.uint64(PRIME_LIMIT)
    a0, a1 = a & mask32, a >> np.uint64(32)
    b0, b1 = b & mask32, b >> np.uint64(32)
    low_product = a0 * b0
    cross = a0 * b1 + a1 * b0
    high_product = a1 * b1

    with np.errstate(over="ignore"):
        low = low_product + ((cross & mask32) << np.uint64(32))
    carry = (low < low_product).astype(np.uint64)
    high = high_product + (cross >> np.uint64(32)) + carry
    folded_low = low & mask61
    folded_high = ((low >> np.uint64(61)) + (high << np.uint64(3))) % np.uint64(prime)
    folded_low %= np.uint64(prime)

    scaled = np.zeros_like(folded_high)
    addend = folded_high.copy()
    multiplier = gap
    while multiplier:
        if multiplier & 1:
            scaled = (scaled + addend) % prime
        multiplier >>= 1
        if multiplier:
            addend = (addend + addend) % prime

    result = (folded_low + scaled) % prime
    return result.astype(np.int64, copy=False)


def _sum_axis_mod(tensor: np.ndarray, axis: int, prime: int) -> np.ndarray:
    """Cumulative sum on one axis with reduction after every safe int64 addition."""
    moved = np.moveaxis(tensor, axis, -1)
    prefix = np.empty_like(moved)
    running = moved[..., 0].copy()
    prefix[..., 0] = running
    for index in range(1, moved.shape[-1]):
        running = (running + moved[..., index]) % prime
        prefix[..., index] = running
    return prefix


def _forget_vertex(
    tensor: np.ndarray,
    child_bag: tuple[int, ...],
    parent_bag: tuple[int, ...],
    vertex: int,
    poset: Poset,
    alphabet_size: int,
    prime: int,
) -> np.ndarray:
    """Contract a forgotten variable over all comparison-compatible values."""
    axis = child_bag.index(vertex)
    remaining_shape = tuple(alphabet_size for _ in parent_bag)
    lower = np.zeros(remaining_shape, dtype=np.int64)
    upper = np.full(remaining_shape, alphabet_size - 1, dtype=np.int64)
    axes = {v: i for i, v in enumerate(parent_bag)}

    for lower_neighbor, upper_neighbor in poset.covers:
        if upper_neighbor == vertex and lower_neighbor in axes:
            shape = [1] * len(parent_bag)
            shape[axes[lower_neighbor]] = alphabet_size
            coordinate = np.arange(alphabet_size, dtype=np.int64).reshape(shape)
            lower = np.maximum(lower, coordinate)
        elif lower_neighbor == vertex and upper_neighbor in axes:
            shape = [1] * len(parent_bag)
            shape[axes[upper_neighbor]] = alphabet_size
            coordinate = np.arange(alphabet_size, dtype=np.int64).reshape(shape)
            upper = np.minimum(upper, coordinate)

    prefix = _sum_axis_mod(tensor, axis, prime)
    upper_index = upper[..., np.newaxis]
    lower_index = np.maximum(lower - 1, 0)[..., np.newaxis]
    through_upper = np.take_along_axis(prefix, upper_index, axis=-1)[..., 0]
    before_lower = np.take_along_axis(prefix, lower_index, axis=-1)[..., 0]
    before_lower = np.where(lower > 0, before_lower, 0)
    result = (through_upper - before_lower) % prime
    result = np.where(lower <= upper, result, 0).astype(np.int64, copy=False)
    return result


def count_modular(
    poset: Poset,
    m: int,
    prime: int,
    decomposition: TreeDecomposition | None = None,
) -> int:
    """Count order-preserving maps P -> {0,...,m} modulo ``prime``."""
    if m < 0:
        raise ValueError("The dilation parameter m must be nonnegative.")
    if prime <= 1:
        raise ValueError("The modulus must be greater than one.")
    td = decomposition or _min_fill_decomposition(poset)
    q = m + 1

    def evaluate(node: NiceNode) -> np.ndarray:
        if node.kind == "LEAF":
            return np.array(1, dtype=np.int64)
        if node.kind == "INTRODUCE":
            child = evaluate(node.children[0])
            axis = node.bag.index(node.vertex)
            expanded = np.expand_dims(child, axis=axis)
            shape = list(expanded.shape)
            shape[axis] = q
            return np.broadcast_to(expanded, shape).copy()
        if node.kind == "FORGET":
            child_node = node.children[0]
            child = evaluate(child_node)
            return _forget_vertex(
                child,
                child_node.bag,
                node.bag,
                node.vertex,
                poset,
                q,
                prime,
            )
        if node.kind == "JOIN":
            left = evaluate(node.children[0])
            right = evaluate(node.children[1])
            return _modular_mul_near_2_61(left, right, prime)
        raise ValueError(f"Unknown nice-decomposition node kind: {node.kind}")

    return int(evaluate(td.root).item())


def _primes_61_bit(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 1 << PRIME_BITS
    for _ in range(count):
        candidate = int(sympy.prevprime(candidate))
        if candidate.bit_length() != PRIME_BITS:
            raise RuntimeError("Could not generate enough distinct 61-bit primes.")
        primes.append(candidate)
    return primes


def minimum_crt_prime_count(dimension: int) -> int:
    """Minimum number of 61-bit primes whose product exceeds (d+1)**d."""
    if dimension < 0:
        raise ValueError("The dimension cannot be negative.")
    bound = (dimension + 1) ** dimension
    product = 1
    count = 0
    while product <= bound:
        prime = int(sympy.prevprime((1 << PRIME_BITS) if count == 0 else last_prime))
        if prime.bit_length() != PRIME_BITS:
            raise RuntimeError("The exact CRT bound requires more 61-bit primes.")
        product *= prime
        count += 1
        last_prime = prime
    return count


def _crt_reconstruct(residues: list[int], primes: list[int]) -> int:
    value = 0
    modulus = 1
    for residue, prime in zip(residues, primes):
        adjustment = ((residue - value) % prime) * pow(modulus % prime, -1, prime) % prime
        value += modulus * adjustment
        modulus *= prime
    return value


def _ehrhart_h_star(point_counts: list[int], dimension: int) -> list[int]:
    coefficients = [
        sum(
            (-1) ** (degree - dilation)
            * math.comb(dimension + 1, degree - dilation)
            * point_counts[dilation]
            for dilation in range(degree + 1)
        )
        for degree in range(dimension + 1)
    ]
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return coefficients


def exact_ehrhart_data(
    poset: Poset, decomposition: TreeDecomposition | None = None
) -> dict[str, object]:
    """Recover L(0)..L(d), h*, and exact normalized/Euclidean volumes by CRT."""
    d = poset.dimension
    bound = (d + 1) ** d
    prime_count = minimum_crt_prime_count(d)
    primes = _primes_61_bit(prime_count)
    residues_by_dilation: list[list[int]] = [[] for _ in range(d + 1)]
    td = decomposition or _min_fill_decomposition(poset)

    for prime in primes:
        for dilation in range(d + 1):
            residues_by_dilation[dilation].append(
                count_modular(poset, dilation, prime, td)
            )

    point_counts = [
        _crt_reconstruct(residues, primes) for residues in residues_by_dilation
    ]
    if any(value < 0 or value > bound for value in point_counts):
        raise ArithmeticError("CRT reconstruction escaped the exact point-count bound.")
    h_star = _ehrhart_h_star(point_counts, d)
    if any(coefficient < 0 for coefficient in h_star):
        raise ArithmeticError("The reconstructed h*-vector has a negative coefficient.")
    normalized_volume = sum(h_star)
    euclidean_volume = Fraction(normalized_volume, math.factorial(d))
    return {
        "dimension": d,
        "treewidth": td.width,
        "maximum_bag_size": td.width + 1 if d else 0,
        "crt_prime_count": prime_count,
        "crt_primes": [str(prime) for prime in primes],
        "crt_modulus_product": str(math.prod(primes)),
        "point_count_upper_bound": str(bound),
        "lattice_point_counts_L_0_to_L_d": [str(value) for value in point_counts],
        "h_star_coefficients": [str(value) for value in h_star],
        "normalized_volume_H_1": str(normalized_volume),
        "euclidean_volume_exact": str(euclidean_volume),
        "maximum_tensor_entries": max(
            ((dilation + 1) ** (td.width + 1) for dilation in range(d + 1)),
            default=1,
        )
        if d
        else 1,
    }


def _network_poset(width: int, depth: int) -> Poset:
    """Irregular layered DAG with local links and nonredundant residual skips."""
    dimension = width * depth
    relations: list[tuple[int, int]] = []

    def vertex(layer: int, column: int) -> int:
        return layer * width + column

    for layer in range(depth - 1):
        for column in range(width):
            relations.append((vertex(layer, column), vertex(layer + 1, column)))
        if layer % 3 == 0:
            relations.append((vertex(layer, 0), vertex(layer + 1, 1)))
    for layer in range(depth - 2):
        if layer % 4 == 0:
            relations.append((vertex(layer, 1), vertex(layer + 2, 2 % width)))
    return poset_from_relations(f"LayeredDAG_w{width}_depth{depth}", dimension, relations)


def _series_parallel_poset(dimension: int) -> Poset:
    """A series composition of diamonds, padded with a chain to exact dimension."""
    relations: list[tuple[int, int]] = []
    used = 1
    previous = 0
    while used + 3 <= dimension:
        left, right, merge = used, used + 1, used + 2
        relations.extend(
            (
                (previous, left),
                (previous, right),
                (left, merge),
                (right, merge),
            )
        )
        previous = merge
        used += 3
    for vertex in range(used, dimension):
        relations.append((previous, vertex))
        previous = vertex
    return poset_from_relations(f"SeriesParallel_d{dimension}", dimension, relations)


def _brute_count(poset: Poset, m: int) -> int:
    return sum(
        all(assignment[lower] <= assignment[upper] for lower, upper in poset.covers)
        for assignment in itertools.product(range(m + 1), repeat=poset.dimension)
    )


def _run_small_exact_checks() -> dict[str, object]:
    checks = [
        Poset("small_chain_4", tuple(range(4)), ((0, 1), (1, 2), (2, 3))),
        Poset("small_V", tuple(range(4)), ((0, 2), (1, 2), (2, 3))),
        Poset("small_antichain_5", tuple(range(5)), ()),
    ]
    prime = _primes_61_bit(1)[0]
    comparisons = 0
    for poset in checks:
        td = _min_fill_decomposition(poset)
        for dilation in range(3):
            exact = _brute_count(poset, dilation)
            modular = count_modular(poset, dilation, prime, td)
            if modular != exact % prime:
                raise AssertionError(
                    f"Small exact check failed for {poset.name}, m={dilation}."
                )
            comparisons += 1

    for prime_value in _primes_61_bit(3):
        left = np.array([0, 1, prime_value - 2, prime_value - 1], dtype=np.int64)
        right = left[::-1].copy()
        observed = _modular_mul_near_2_61(left, right, prime_value)
        expected = np.array(
            [(int(a) * int(b)) % prime_value for a, b in zip(left, right)],
            dtype=np.int64,
        )
        if not np.array_equal(observed, expected):
            raise AssertionError("The vectorized 61-bit modular product is incorrect.")
    return {
        "brute_force_comparisons": comparisons,
        "modular_product_prime_checks": 3,
        "status": "passed",
    }


def _load_cylinder_references() -> dict[tuple[int, int], dict[str, object]]:
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    references = {}
    for row in payload["instances"]:
        if (row["k"], row["n"]) in {(3, 4), (4, 4)}:
            references[(row["k"], row["n"])] = row
    if set(references) != {(3, 4), (4, 4)}:
        raise FileNotFoundError("Required C_{3,4}/C_{4,4} benchmark references are missing.")
    return references


def _benchmark_instance(
    poset: Poset,
    reference: dict[str, object] | None = None,
    polyhedral_reference: dict[str, object] | None = None,
) -> dict[str, object]:
    tracemalloc.start()
    started = time.perf_counter()
    td = _min_fill_decomposition(poset)
    result = exact_ehrhart_data(poset, td)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.update(
        {
            "name": poset.name,
            "cover_count": len(poset.covers),
            "elimination_order": list(td.elimination_order),
            "elimination_bags": [list(bag) for bag in td.elimination_bags],
            "wall_time_seconds": round(elapsed, 6),
            "tracemalloc_peak_mb": round(peak / (1024 * 1024), 6),
        }
    )
    if reference is not None:
        matches = (
            result["h_star_coefficients"] == reference["h_star_coefficients"]
            and result["normalized_volume_H_1"] == reference["normalized_volume_H_1"]
            and result["euclidean_volume_exact"] == reference["euclidean_volume_exact"]
        )
        result["cylinder_reference_validation"] = {
            "status": "passed" if matches else "failed",
            "h_star_matches": result["h_star_coefficients"] == reference["h_star_coefficients"],
            "normalized_volume_matches": result["normalized_volume_H_1"]
            == reference["normalized_volume_H_1"],
            "euclidean_volume_matches": result["euclidean_volume_exact"]
            == reference["euclidean_volume_exact"],
        }
        if not matches:
            raise AssertionError(f"Reference validation failed for {poset.name}.")

    if polyhedral_reference is None:
        result["polyhedral_baseline_reference"] = {
            "status": "no_dimension_matched_reference",
            "note": "No baseline record at this dimension; no timeout is inferred.",
        }
    else:
        baseline = polyhedral_reference["polyhedral_baseline"]
        result["polyhedral_baseline_reference"] = {
            "status": baseline["status"],
            "reference_instance": f"C_{{{polyhedral_reference['k']},{polyhedral_reference['n']}}}",
            "reference_dimension": polyhedral_reference["dimension"],
            "wall_time_seconds": baseline["wall_time_seconds"],
            "timeout_seconds": baseline["timeout_seconds"],
            "explicit_triangulation_memory_estimate_bytes_dense_int64": baseline[
                "explicit_triangulation_memory_estimate_bytes_dense_int64"
            ],
            "memory_metric": "estimated payload, not measured peak RSS",
            "comparison_scope": "same dimension, different poset; contextual comparison only",
        }
    return result


def _print_tables(results: list[dict[str, object]]) -> None:
    print("\nTreewidth-Tensor-CRT: exact Ehrhart recovery")
    print(
        f"{'Instance':34} {'d':>3} {'tw':>3} {'bag':>4} "
        f"{'CRT':>3} {'seconds':>10} {'peak MB':>10} {'H_P(1)':>18}"
    )
    print("-" * 95)
    for row in results:
        print(
            f"{str(row['name']):34.34} {row['dimension']:>3} {row['treewidth']:>3} "
            f"{row['maximum_bag_size']:>4} {row['crt_prime_count']:>3} "
            f"{row['wall_time_seconds']:>10.3f} {row['tracemalloc_peak_mb']:>10.2f} "
            f"{str(row['normalized_volume_H_1']):>18}"
        )
    print("\nCylinder reference checks")
    for row in results:
        validation = row.get("cylinder_reference_validation")
        if validation:
            print(
                f"  {row['name']}: {validation['status']} "
                f"(h*: {validation['h_star_matches']}, "
                f"H(1): {validation['normalized_volume_matches']}, "
                f"Vol: {validation['euclidean_volume_matches']})"
            )
    print("\nCurrent-run polyhedral baseline: not run (no configured triangulation executable).")
    print("Dimension-matched historical references (different poset; not apples-to-apples):")
    for row in results:
        reference = row.get("polyhedral_baseline_reference", {})
        if "reference_instance" in reference:
            scope = (
                "same poset"
                if row["name"] == reference["reference_instance"]
                else "same dimension, different poset"
            )
            print(
                f"  {row['name']} vs {reference['reference_instance']} ({scope}): "
                f"{reference['status']} in {reference['wall_time_seconds']} s; "
                "dense payload estimate "
                f"{int(reference['explicit_triangulation_memory_estimate_bytes_dense_int64']):.3e} bytes"
            )
        elif row["dimension"] >= 20:
            print(f"  {row['name']}: no same-dimension historical baseline.")
    print("Historical timeout records are measured; memory figures are payload estimates, not peaks.")


def run_benchmarks() -> dict[str, object]:
    small_checks = _run_small_exact_checks()
    references = _load_cylinder_references()
    benchmark_payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    polyhedral_references = {
        row["dimension"]: row for row in benchmark_payload["instances"]
    }
    instances: list[tuple[Poset, dict[str, object] | None, str]] = []
    for k, n in ((3, 4), (4, 4)):
        instances.append((cyclic_crown(k, n), references[(k, n)], "cylinder"))

    for width, depth in ((3, 5), (4, 5), (3, 8), (3, 10), (4, 9), (4, 10)):
        instances.append((_network_poset(width, depth), None, "layered_dag"))

    for dimension in (18, 25, 32, 40):
        instances.append((_series_parallel_poset(dimension), None, "series_parallel"))

    results: list[dict[str, object]] = []
    for poset, reference, family in instances:
        print(f"Running {poset.name} (d={poset.dimension})...", flush=True)
        result = _benchmark_instance(
            poset, reference, polyhedral_references.get(poset.dimension)
        )
        result["family"] = family
        results.append(result)

    payload = {
        "benchmark": "Treewidth-Tensor-CRT exact order-polytope counting",
        "python_version": __import__("sys").version.split()[0],
        "numpy_version": np.__version__,
        "crt_prime_bits": PRIME_BITS,
        "crt_bound": "L_P(m) <= (m+1)^d <= (d+1)^d",
        "small_exact_checks": small_checks,
        "polyhedral_baseline_policy": {
            "timeout_seconds": 15,
            "status": "dimension_matched_reference_only",
            "reason": (
                "Prior C_{k,5} baseline rows are included only as contextual, same-dimension "
                "references; they are different posets and are not represented as current-run "
                "measurements."
            ),
        },
        "instances": results,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_tables(results)
    print(f"\nStructured results: {OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    run_benchmarks()
