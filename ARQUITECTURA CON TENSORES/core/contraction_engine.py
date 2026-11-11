"""Contracción exacta o acelerada de la red tensorial cilíndrica."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import opt_einsum as oe

from .tensor_builder import build_order_polytope_tensor


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

        operands = self._build_operands(m)
        cache_key = m + 1
        if cache_key not in self._path_cache:
            self._path_cache[cache_key] = oe.contract_path(
                *operands, optimize=self.optimize
            )[0]
        path = self._path_cache[cache_key]

        if self.mode == "exact":
            result = oe.contract(*operands, optimize=path, backend="numpy")
            count = int(np.asarray(result).item())
        else:
            result = oe.contract(*operands, optimize=path, backend="torch")
            scalar = result.item()
            count = complex(scalar) if isinstance(scalar, complex) else float(scalar)
            if isinstance(count, complex) and count.imag == 0:
                count = count.real

        self._count_cache[m] = count
        return count

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