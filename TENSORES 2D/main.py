"""CLI para reconstruir y verificar el h*-polinomio de un poset C_{k,n}."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
        "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help="Procesos para evaluar dilataciones en modo exacto.",
    )
    verification_group = parser.add_mutually_exclusive_group()
    verification_group.add_argument(
        "--skip-sturm",
        action="store_true",
        help="Omite Sturm; --export-json exporta solo el polinomio no certificado.",
    )
    verification_group.add_argument(
        "--gamma-sturm",
        action="store_true",
        help="Certifica mediante Gamma si H tiene grado par y es palíndromo.",
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
        certificate_dir = (
            Path(args.certificate_dir)
            if args.certificate_dir
            else Path(__file__).resolve().parent / "certificados_json"
        )
        run_experimental_analysis(certificate_dir, args.max_recurrence_degree)
        return
    if args.k is None or args.n is None:
        raise SystemExit("--k y --n son obligatorios fuera del modo de análisis.")
    if args.workers < 1:
        raise SystemExit("--workers debe ser un entero positivo.")
    mode = args.mode or ("accelerated" if args.device == "cuda" else "exact")
    if mode == "accelerated" and not args.skip_sturm:
        raise SystemExit(
            "Sturm exacto requiere --mode exact; use --skip-sturm para omitirlo."
        )
    if mode == "accelerated" and args.export_json:
        raise SystemExit("El JSON exacto requiere --mode exact.")
    contractor = BoundaryMPSContractor(
        args.k, args.n, mode=mode, device=args.device, dtype=args.dtype
    )
    workers = args.workers if mode == "exact" else 1
    polynomial = reconstruct_hstar_polynomial(
        args.k, args.n, contractor, workers=workers
    )
    verification = None
    if not args.skip_sturm:
        verification = verify_real_rootedness(
            polynomial,
            args.k,
            args.n,
            export_json=args.export_json,
            use_gamma=args.gamma_sturm,
        )
    elif args.export_json:
        payload = {
            "polynomial": str(polynomial.as_expr()),
            "polynomial_coefficients_descending": [
                str(coefficient) for coefficient in polynomial.all_coeffs()
            ],
            "degree": polynomial.degree(),
            "verification_skipped": True,
            "sign_certificate": None,
        }
        Path(args.export_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"H_{{{args.k},{args.n}}}(t) = {polynomial.as_expr()}")
    if verification is None:
        print("Verificación Sturm omitida.")
    else:
        print(verification["report"])
    if args.export_json:
        if verification is None:
            print(f"Polinomio JSON (sin certificado): {args.export_json}")
        else:
            print(f"Certificado JSON: {verification['exported_json']}")


if __name__ == "__main__":
    main()