"""Contracción exacta o acelerada de la red tensorial cilíndrica."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np
import opt_einsum as oe
import sympy as sp

from .tensor_builder import build_order_polytope_tensor


_FIRST_CRT_PRIME = 2_147_483_647
_INT64_MAX = int(np.iinfo(np.int64).max)


class BoundaryMPSContractor:
    """Cuenta mapas de orden mediante una red tensorial con frontera periódica.

    Cada tensor local representa una restricción de cubierta entre dos capas.
    Los índices que representan cada coordenada se comparten entre todos los
    factores incidentes y un vector de unos suma cada coordenada una sola vez.
    ``opt_einsum`` selecciona la ruta de contracción (por defecto ``auto-hq``).

    Args:
        k: Número de capas del poset (``k >= 2``).
        n: Número de vértices por circunferencia (``n >= 3``).
        mode: ``"exact"`` para aritmética entera arbitraria o
            ``"accelerated"`` para PyTorch.
        device: Dispositivo PyTorch en modo acelerado.
        dtype: ``"float64"`` o ``"complex128"`` para tensores PyTorch.
        optimize: Estrategia de búsqueda de ruta aceptada por opt_einsum.

    En modo exacto, ``contract(m)`` retorna un ``int`` arbitrario. En modo
    acelerado retorna un ``float`` (o un complejo numérico) y está destinado
    a inspecciones donde la precisión de coma flotante sea aceptable.
    """

    def __init__(
        self,
        k: int,
        n: int,
        mode: str = "exact",
        device: str = "cpu",
        dtype: str = "float64",
        optimize: str = "auto-hq",
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 2:
            raise ValueError("k debe ser un entero mayor o igual que 2.")
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 3:
            raise ValueError("n debe ser un entero mayor o igual que 3.")
        if mode not in {"exact", "accelerated"}:
            raise ValueError("mode debe ser 'exact' o 'accelerated'.")
        if mode == "exact" and device != "cpu":
            raise ValueError("El modo exacto usa NumPy en CPU; use mode='accelerated' para CUDA.")

        self.k = int(k)
        self.n = int(n)
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.optimize = optimize
        self._path_cache: dict[int, Any] = {}
        self._count_cache: dict[int, int | float | complex] = {}

        if mode == "accelerated":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "El modo acelerado requiere instalar la dependencia 'torch'."
                ) from exc
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
            if device != "cpu" and not device.startswith("cuda"):
                raise ValueError("device debe ser 'cpu' o un dispositivo CUDA.")
            if dtype not in {"float64", "complex128"}:
                raise ValueError("dtype acelerado debe ser 'float64' o 'complex128'.")
            self._torch_dtype = (
                torch.float64 if dtype == "float64" else torch.complex128
            )

    def __call__(self, m: int) -> int | float | complex:
        """Alias invocable de ``contract`` para uso como función de conteo."""
        return self.contract(m)

    def contract(self, m: int) -> int | float | complex:
        """Evalúa exactamente o aproximadamente ``L_{k,n}(m)``."""
        if isinstance(m, bool) or not isinstance(m, Integral):
            raise TypeError("m debe ser un entero.")
        if m < 0:
            raise ValueError("m debe ser no negativo.")
        m = int(m)
        if m in self._count_cache:
            return self._count_cache[m]

        if self.mode == "exact":
            return self.contract_crt(m)

        operands = self._build_operands(m)
        cache_key = m + 1
        if cache_key not in self._path_cache:
            self._path_cache[cache_key] = oe.contract_path(
                *operands, optimize=self.optimize
            )[0]
        path = self._path_cache[cache_key]

        result = oe.contract(*operands, optimize=path, backend="torch")
        scalar = result.item()
        count = complex(scalar) if isinstance(scalar, complex) else float(scalar)
        if isinstance(count, complex) and count.imag == 0:
            count = count.real

        self._count_cache[m] = count
        return count

    def contract_crt(self, m: int) -> int:
        """Cuenta exactamente mediante contracciones modulares y CRT."""
        if isinstance(m, bool) or not isinstance(m, Integral):
            raise TypeError("m debe ser un entero.")
        if m < 0:
            raise ValueError("m debe ser no negativo.")
        m = int(m)
        if m in self._count_cache:
            cached = self._count_cache[m]
            if not isinstance(cached, int):
                raise TypeError("El conteo cacheado no es entero exacto.")
            return cached

        upper_bound = (m + 1) ** (self.k * self.n)
        primes: list[int] = []
        modulus_product = 1
        prime = _FIRST_CRT_PRIME
        while modulus_product <= upper_bound:
            primes.append(prime)
            modulus_product *= prime
            prime = int(sp.prevprime(prime))

        value = 0
        accumulated_modulus = 1
        for prime in primes:
            residue = self._contract_modular(m, prime)
            correction = (
                (residue - value) * pow(accumulated_modulus, -1, prime)
            ) % prime
            value += accumulated_modulus * correction
            accumulated_modulus *= prime

        if value > upper_bound:
            raise ArithmeticError(
                "La reconstrucción CRT excedió la cota exacta del conteo."
            )
        self._count_cache[m] = value
        return value

    def _contract_modular(self, m: int, prime: int) -> int:
        """Evalúa la red en Z/prime Z sin desbordar los acumuladores int64."""
        local_tensor = build_order_polytope_tensor(
            m, backend="numpy", dtype=np.int64
        )
        ones = np.ones(m + 1, dtype=np.int64)
        operands: list[Any] = []
        for layer in range(self.k - 1):
            for i in range(self.n):
                current = layer * self.n + i
                next_i = layer * self.n + (i + 1) % self.n
                upper = (layer + 1) * self.n + i
                operands.extend((local_tensor, [current, next_i, upper]))
        for layer in range(self.k):
            for i in range(self.n):
                operands.extend((ones, [layer * self.n + i]))
        operands.append([])

        path_key = m + 1
        if path_key not in self._path_cache:
            self._path_cache[path_key] = oe.contract_path(
                *operands, optimize=self.optimize
            )[0]
        path = self._path_cache[path_key]
        live = [
            (operands[index], tuple(operands[index + 1]))
            for index in range(0, len(operands) - 2, 2)
        ]
        dimensions = {label: m + 1 for _, labels in live for label in labels}
        for contraction in path:
            if len(contraction) != 2:
                raise ValueError(
                    "La contracción modular requiere una ruta de pasos binarios."
                )
            left_index, right_index = contraction
            left_array, left_labels = live[left_index]
            right_array, right_labels = live[right_index]
            remaining = [
                (array, labels)
                for index, (array, labels) in enumerate(live)
                if index not in (left_index, right_index)
            ]
            remaining_labels = {
                label for _, labels in remaining for label in labels
            }
            right_label_set = set(right_labels)
            shared = [label for label in left_labels if label in right_label_set]
            batch_labels = [
                label for label in shared if label in remaining_labels
            ]
            contracted_labels = [
                label for label in shared if label not in remaining_labels
            ]
            left_only = [label for label in left_labels if label not in shared]
            right_only = [label for label in right_labels if label not in shared]

            left_order = batch_labels + left_only + contracted_labels
            right_order = batch_labels + contracted_labels + right_only
            left_array = np.transpose(
                left_array, [left_labels.index(label) for label in left_order]
            )
            right_array = np.transpose(
                right_array, [right_labels.index(label) for label in right_order]
            )
            batch_size = math.prod(dimensions[label] for label in batch_labels)
            row_size = math.prod(dimensions[label] for label in left_only)
            contraction_size = math.prod(
                dimensions[label] for label in contracted_labels
            )
            column_size = math.prod(dimensions[label] for label in right_only)
            left_matrix = left_array.reshape(
                batch_size, row_size, contraction_size
            )
            right_matrix = right_array.reshape(
                batch_size, contraction_size, column_size
            )
            result = np.zeros(
                (batch_size, row_size, column_size), dtype=np.int64
            )
            # Bound each dot-product chunk so its int64 sum cannot overflow.
            block_size = max(
                1, _INT64_MAX // (prime - 1) ** 2
            )
            offset = 0
            while offset < contraction_size:
                partial = np.matmul(
                    left_matrix[:, :, offset : offset + block_size],
                    right_matrix[:, offset : offset + block_size, :],
                )
                result = (result + partial) % prime
                offset += block_size

            result_labels = batch_labels + left_only + right_only
            result_shape = [dimensions[label] for label in result_labels]
            result = result.reshape(result_shape)
            next_live = [
                (array, labels)
                for index, (array, labels) in enumerate(live)
                if index not in (left_index, right_index)
            ]
            next_live.append((result, tuple(result_labels)))
            live = next_live

        if len(live) != 1 or live[0][1]:
            raise ArithmeticError("La contracción modular no produjo un escalar.")
        return int(live[0][0].item()) % prime

    def _build_operands(self, m: int) -> list[Any]:
        """Crea los factores y sus etiquetas para la contracción completa."""
        dimension = m + 1
        backend = "numpy" if self.mode == "exact" else "torch"
        dtype = object if self.mode == "exact" else None
        local_tensor = build_order_polytope_tensor(
            m,
            backend=backend,
            device=self.device,
            dtype=self._torch_dtype if self.mode == "accelerated" else dtype,
        )
        if self.mode == "exact":
            local_tensor = np.asarray(local_tensor, dtype=object)
            ones = np.ones(dimension, dtype=object)
        else:
            import torch

            ones = torch.ones(
                dimension, dtype=self._torch_dtype, device=self.device
            )

        operands: list[Any] = []
        for layer in range(self.k - 1):
            for i in range(self.n):
                current = layer * self.n + i
                next_i = layer * self.n + (i + 1) % self.n
                upper = (layer + 1) * self.n + i
                operands.extend((local_tensor, [current, next_i, upper]))

        for layer in range(self.k):
            for i in range(self.n):
                operands.extend((ones, [layer * self.n + i]))

        operands.append([])
        return operands