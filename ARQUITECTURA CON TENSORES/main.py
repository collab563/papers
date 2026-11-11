"""CLI para reconstruir y verificar el h*-polinomio de un poset C_{k,n}."""

from __future__ import annotations

import argparse

from core.contraction_engine import BoundaryMPSContractor
from core.interlacing_analyzer import run_experimental_analysis
from core.polynomial_reconstructor import reconstruct_hstar_polynomial
from core.root_verifier import verify_real_rootedness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye exactamente el h*-polinomio Ehrhart de la red "
            "cilíndrica C_{k,n} y comprueba su real-rootedness."
        )
    )
    parser.add_argument("--k", type=int, help="Número de capas (>= 2).")
    parser.add_argument("--n", type=int, help="Vértices por capa (>= 3).")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU exacta por defecto; CUDA ejecuta la contracción acelerada.",
    )
    parser.add_argument(
        "--mode",
        choices=("exact", "accelerated"),
        default=None,
        help="Modo de contracción; por defecto se infiere de --device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float64", "complex128"),
        default="float64",
        help="Precisión PyTorch para el modo acelerado.",
    )
    parser.add_argument(
        "--export-json",
        nargs="?",
        const="root_certificate.json",
        metavar="RUTA",
        help="Exporta el reporte y el certificado Sturm a JSON.",
    )
    parser.add_argument(
        "--analyze-interlacing",
        action="store_true",
        help="Analiza entrelazamiento y recurrencias desde los certificados existentes.",
    )
    parser.add_argument(
        "--certificate-dir",
        help=(
            "Directorio de certificados para --analyze-interlacing "
            "(por defecto: certificados_json junto a main.py)."
        ),
    )
    parser.add_argument(
        "--max-recurrence-degree",
        type=int,
        default=6,
        help="Grado máximo de coeficientes en el análisis de recurrencias.",
    )
    return parser.parse_args()


def main() -> None:
    """Ejecuta el pipeline completo de conteo, reconstrucción y verificación."""
    args = _parse_args()
    if args.analyze_interlacing:
        if args.k is not None or args.n is not None:
            raise SystemExit("--analyze-interlacing se ejecuta sin --k ni --n.")
        if args.max_recurrence_degree < 0:
            raise SystemExit("--max-recurrence-degree debe ser no negativo.")
        from pathlib import Path

        certificate_dir = (
            Path(args.certificate_dir)
            if args.certificate_dir
            else Path(__file__).resolve().parent / "certificados_json"
        )
        run_experimental_analysis(certificate_dir, args.max_recurrence_degree)
        return
    if args.k is None or args.n is None:
        raise SystemExit("--k y --n son obligatorios fuera del modo de análisis.")
    mode = args.mode or ("accelerated" if args.device == "cuda" else "exact")
    contractor = BoundaryMPSContractor(
        args.k, args.n, mode=mode, device=args.device, dtype=args.dtype
    )
    polynomial = reconstruct_hstar_polynomial(args.k, args.n, contractor)
    verification = verify_real_rootedness(
        polynomial, args.k, args.n, export_json=args.export_json
    )

    print(f"H_{{{args.k},{args.n}}}(t) = {polynomial.as_expr()}")
    print(verification["report"])
    if args.export_json:
        print(f"Certificado JSON: {verification['exported_json']}")


if __name__ == "__main__":
    main()