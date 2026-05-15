#!/usr/bin/env python3
"""
Fixed-interface-fraction benign scalar convergence check.

This script repeats the benign scalar convergence experiment while holding the
interface fraction theta fixed throughout refinement. The calculation complements
the fixed-event-time scalar tests in the main numerical suite.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import bdf2_temporal_interface as bdf2
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Could not import bdf2_temporal_interface.py: {exc}")

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

T = 1.0
TAU = 0.37
TARGET_THETA = 0.37
BETA = 1.0
MU = 10.0

# These values satisfy N = 1 modulo 100. Since tau = 37/100,
# the fractional part of tau*N is 37/100 for every listed N.
N_VALUES = [101, 201, 401, 801, 1601, 3201]

OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
CSV_PATH = OUTDIR / "fixed_theta_benign_scalar.csv"
TXT_PATH = OUTDIR / "fixed_theta_benign_scalar_table.txt"
FIG_PATH = OUTDIR / "fixed_theta_benign_scalar.png"


def eoc(prev_error: float, curr_error: float, prev_h: float, curr_h: float) -> Optional[float]:
    if prev_error <= 0.0 or curr_error <= 0.0 or prev_h <= curr_h:
        return None
    return math.log(prev_error / curr_error) / math.log(prev_h / curr_h)


def compute_errors(problem: bdf2.LinearSwitchProblem, n_steps: int, method: str) -> Dict[str, float]:
    h = T / n_steps
    kind, k, theta = bdf2.crossing_info(TAU, h, T)
    if kind != "crossing":
        raise RuntimeError(f"N={n_steps}: expected a crossing step, got {kind}.")
    if abs(theta - TARGET_THETA) > 5e-13:
        raise RuntimeError(f"N={n_steps}: theta={theta}, expected {TARGET_THETA}.")

    tgrid, y_values = bdf2.integrate_problem(problem, h, method)
    pointwise_errors = []
    for i, t in enumerate(tgrid):
        err = bdf2.as_1d(y_values[i, :]) - bdf2.as_1d(problem.exact(float(t)))
        pointwise_errors.append(bdf2.norm_l2(err))

    max_error = max(pointwise_errors)
    time_l2_error = math.sqrt(h * sum(value * value for value in pointwise_errors))
    return {"N": n_steps, "h": h, "k": k, "theta": theta, "Einf": max_error, "E2": time_l2_error}


def sci(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}e}"


def eoc_text(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    print("=" * 80)
    print("Fixed-interface-fraction benign scalar convergence check")
    print("=" * 80)
    print(f"tau = {TAU}, theta = {TARGET_THETA}")
    print(f"N values = {N_VALUES}")
    print()

    problem = bdf2.build_benign_scalar_problem(T=T, tau=TAU, beta=BETA, mu=MU)

    rows: List[Dict[str, object]] = []
    previous: Dict[str, Dict[str, float]] = {}

    for n_steps in N_VALUES:
        standard = compute_errors(problem, n_steps, "standard")
        corrected = compute_errors(problem, n_steps, "corrected")
        h = standard["h"]

        row: Dict[str, object] = {
            "N": n_steps,
            "h": h,
            "crossing_index_k": standard["k"],
            "theta": standard["theta"],
            "standard_Einf": standard["Einf"],
            "standard_Einf_EOC": None,
            "corrected_Einf": corrected["Einf"],
            "corrected_Einf_EOC": None,
            "standard_E2": standard["E2"],
            "standard_E2_EOC": None,
            "corrected_E2": corrected["E2"],
            "corrected_E2_EOC": None,
            "Einf_gain_standard_over_corrected": standard["Einf"] / corrected["Einf"],
            "E2_gain_standard_over_corrected": standard["E2"] / corrected["E2"],
        }

        if previous:
            row["standard_Einf_EOC"] = eoc(previous["standard"]["Einf"], standard["Einf"], previous["standard"]["h"], h)
            row["corrected_Einf_EOC"] = eoc(previous["corrected"]["Einf"], corrected["Einf"], previous["corrected"]["h"], h)
            row["standard_E2_EOC"] = eoc(previous["standard"]["E2"], standard["E2"], previous["standard"]["h"], h)
            row["corrected_E2_EOC"] = eoc(previous["corrected"]["E2"], corrected["E2"], previous["corrected"]["h"], h)

        rows.append(row)
        previous = {"standard": standard, "corrected": corrected}

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with TXT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("Fixed-interface-fraction benign scalar convergence check (tau = 0.37, theta = 0.37).\n\n")
        handle.write("| N | h | theta | Standard E_inf | Standard EOC | Corrected E_inf | Corrected EOC | Standard E2 | Standard EOC | Corrected E2 | Corrected EOC | E_inf gain | E2 gain |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['N']} | {float(row['h']):.8e} | {float(row['theta']):.2f} | "
                f"{sci(float(row['standard_Einf']))} | {eoc_text(row['standard_Einf_EOC'])} | "
                f"{sci(float(row['corrected_Einf']))} | {eoc_text(row['corrected_Einf_EOC'])} | "
                f"{sci(float(row['standard_E2']))} | {eoc_text(row['standard_E2_EOC'])} | "
                f"{sci(float(row['corrected_E2']))} | {eoc_text(row['corrected_E2_EOC'])} | "
                f"{float(row['Einf_gain_standard_over_corrected']):.2f} | {float(row['E2_gain_standard_over_corrected']):.2f} |\n"
            )

    if HAVE_MPL:
        hs = [float(row["h"]) for row in rows]
        standard_errors = [float(row["standard_Einf"]) for row in rows]
        corrected_errors = [float(row["corrected_Einf"]) for row in rows]
        plt.figure(figsize=(7.2, 5.2))
        plt.loglog(hs, standard_errors, marker="o", label="standard")
        plt.loglog(hs, corrected_errors, marker="o", label="corrected")
        plt.gca().invert_xaxis()
        plt.xlabel("h")
        plt.ylabel(r"$E_\infty$")
        plt.title(r"Benign scalar convergence at fixed $\theta=0.37$")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_PATH, dpi=600, bbox_inches="tight")
        plt.close()

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {TXT_PATH}")
    if HAVE_MPL:
        print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
