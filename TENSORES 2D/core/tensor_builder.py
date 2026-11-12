"""Construcción de factores locales para el poset de corona multicapa."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def build_order_polytope_tensor(
    m: int,
    backend: str = "numpy",
    device: str = "cpu",
    dtype: Any = None,
) -> Any:
    """Construye el tensor local de la relación de orden entre dos capas.

    El resultado tiene forma ``(m + 1, m + 1, m + 1)`` y los ejes
    corresponden, respectivamente, a ``x[j, i]``, ``x[j, i + 1]`` y
    ``x[j + 1, i]``. Una entrada vale uno exactamente cuando
    ``x[j + 1, i] >= max(x[j, i], x[j, i + 1])``.

    Args:
        m: Dilatación no negativa; cada coordenada toma valores de 0 a m.
        backend: ``"numpy"`` o ``"torch"``.
        device: Dispositivo PyTorch (por ejemplo, ``"cpu"`` o ``"cuda"``).
        dtype: Tipo opcional del backend. NumPy usa ``int64`` por defecto;
            PyTorch usa ``float64`` por defecto para ser compatible con
            contracciones aceleradas.

    Returns:
        Tensor local de rango tres.

    Raises:
        TypeError: Si ``m`` no es un entero.
        ValueError: Si ``m`` es negativo, el backend no es válido o la
            combinación de backend y dispositivo no está disponible.
    """
    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("m debe ser un entero.")
    if m < 0:
        raise ValueError("m debe ser no negativo.")

    dimension = int(m) + 1
    if backend == "numpy":
        values = np.arange(dimension)
        valid = values[None, None, :] >= np.maximum(
            values[:, None, None], values[None, :, None]
        )
        return valid.astype(np.int64 if dtype is None else dtype, copy=False)

    if backend != "torch":
        raise ValueError("backend debe ser 'numpy' o 'torch'.")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "El backend PyTorch requiere instalar la dependencia 'torch'."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    try:
        values = torch.arange(dimension, device=device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Dispositivo PyTorch no válido o no disponible: {device}") from exc
    local_dtype = torch.float64 if dtype is None else dtype
    return (
        values[None, None, :]
        >= torch.maximum(values[:, None, None], values[None, :, None])
    ).to(dtype=local_dtype)