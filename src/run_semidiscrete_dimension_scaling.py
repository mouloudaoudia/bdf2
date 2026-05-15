#!/usr/bin/env python3
"""Dimension-scaling check for the benign semidiscrete parabolic benchmark.

The script repeats the benign semidiscrete benchmark at fixed temporal
resolution while increasing the number of semidiscrete unknowns.  It writes a
compact CSV table and a plain-text table for use in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence

import bdf2_temporal_interface as base
import run_additional_semidiscrete_benchmark as semibenign

OUTDIR = base.OUTDIR
OUTDIR.mkdir(exist_ok=True)


def _format_sci(x: float) -> str:
    return f"{x:.6e}"


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _timed_error(problem: base.LinearSwitchProblem, N: int, method: str, repeats: int) -> Dict[str, float]:
    h = problem.T / N
    times: List[float] = []
    einf = 0.0
    e2 = 0.0

    # Warm-up run to initialize cached factorizations and imports.
    base.integrate_problem(problem, h, method)

    for _ in range(repeats):
        t0 = time.perf_counter()
        tgrid, Y = base.integrate_problem(problem, h, method)
        times.append(time.perf_counter() - t0)

        pointwise = []
        for i, t in enumerate(tgrid):
            err = base.as_1d(Y[i, :]) - base.as_1d(problem.exact(float(t)))
            pointwise.append(base.norm_l2(err))
        einf = max(pointwise)
        e2 = (h * sum(e * e for e in pointwise)) ** 0.5

    return {
        "Einf": float(einf),
        "E2": float(e2),
        "median_seconds": float(statistics.median(times)),
        "mean_seconds": float(statistics.mean(times)),
        "min_seconds": float(min(times)),
        "max_seconds": float(max(times)),
    }


def run_dimension_scaling(J_values: Sequence[int], N: int, repeats: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for J in J_values:
        problem = semibenign.build_semidiscrete_benign_problem(J=J)
        standard = _timed_error(problem, N=N, method="standard", repeats=repeats)
        corrected = _timed_error(problem, N=N, method="corrected", repeats=repeats)

        rows.append({
            "J": J,
            "dimension": problem.dim,
            "N": N,
            "standard_Einf": standard["Einf"],
            "corrected_Einf": corrected["Einf"],
            "Einf_gain_standard_over_corrected": standard["Einf"] / corrected["Einf"],
            "standard_E2": standard["E2"],
            "corrected_E2": corrected["E2"],
            "E2_gain_standard_over_corrected": standard["E2"] / corrected["E2"],
            "standard_median_seconds": standard["median_seconds"],
            "corrected_median_seconds": corrected["median_seconds"],
            "runtime_ratio_corrected_over_standard": corrected["median_seconds"] / standard["median_seconds"],
        })

    return rows


def write_text_table(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = []
    lines.append("Dimension-scaling check for the manufactured benign semidiscrete benchmark at N=128")
    lines.append("")
    lines.append("Dimension | Standard Einf | Corrected Einf | Einf gain | Standard E2 | Corrected E2 | E2 gain | Runtime ratio")
    lines.append("---:|---:|---:|---:|---:|---:|---:|---:")
    for row in rows:
        lines.append(
            f"{int(row['dimension'])} | "
            f"{_format_sci(float(row['standard_Einf']))} | "
            f"{_format_sci(float(row['corrected_Einf']))} | "
            f"{float(row['Einf_gain_standard_over_corrected']):.2f} | "
            f"{_format_sci(float(row['standard_E2']))} | "
            f"{_format_sci(float(row['corrected_E2']))} | "
            f"{float(row['E2_gain_standard_over_corrected']):.2f} | "
            f"{float(row['runtime_ratio_corrected_over_standard']):.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the semidiscrete dimension-scaling check.")
    parser.add_argument("--J-values", default="160,640,2560", help="Comma-separated spatial interval counts. Default: 160,640,2560.")
    parser.add_argument("--N", type=int, default=128, help="Temporal grid size used in the scaling check. Default: 128.")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats. Default: 5.")
    args = parser.parse_args()

    J_values = [int(item.strip()) for item in args.J_values.split(",") if item.strip()]
    rows = run_dimension_scaling(J_values=J_values, N=args.N, repeats=args.repeats)

    csv_path = OUTDIR / "semidiscrete_dimension_scaling.csv"
    txt_path = OUTDIR / "semidiscrete_dimension_scaling_table.txt"
    _write_csv(csv_path, rows)
    write_text_table(txt_path, rows)

    base.print_table("Dimension-scaling check for the benign semidiscrete benchmark", rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
