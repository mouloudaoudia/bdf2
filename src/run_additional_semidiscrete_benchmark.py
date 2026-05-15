"""Supplemental semidiscrete benchmark and cost diagnostics for the BDF2 temporal-interface study.

The script imports ``bdf2_temporal_interface.py`` and adds:

1. a benign semidiscrete parabolic benchmark;
2. standard-versus-corrected timing diagnostics;
3. an algorithmic cost table;
4. a combined finest-grid summary.

Typical use:
    python bdf2_temporal_interface.py
    python run_additional_semidiscrete_benchmark.py --task all

Timing values are machine-dependent.  The error tables and figures are deterministic for
fixed parameters.
"""
from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import bdf2_temporal_interface as base

try:
    import scipy.sparse as sp
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    sp = None
    HAVE_SCIPY = False

OUTDIR = base.OUTDIR
OUTDIR.mkdir(exist_ok=True)
EPS = getattr(base, "EPS", 1.0e-14)


def _save_csv(filename: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path = OUTDIR / filename
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_if_exists(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_rows_without_duplicates(rows: List[Dict[str, object]], new_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for row in rows + new_rows:
        key = (str(row.get("study", "")), str(row.get("method", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_semidiscrete_benign_problem(
    T: float = 1.0,
    tau: float = 0.37,
    beta: float = 1.0,
    lam: float = 1.0,
    kappa_minus: float = 1.0,
    kappa_plus: float = 3.0,
    J: int = 160,
) -> base.LinearSwitchProblem:
    """Vector-valued semidiscrete parabolic benchmark in the benign regime.

    The manufactured solution is

        U(t,x_j) = sin(pi x_j) * ( exp(-t) + beta * max(t - tau, 0)^2 ).

    Hence [U'] = 0 and [U''] = 2 beta sin(pi x_j), exactly matching the benign
    crossing-defect setting.  The spatial operator is the same finite-difference
    Dirichlet heat operator used in the original severe semidiscrete test.
    """
    if J < 3:
        raise ValueError("J must be at least 3.")

    dx = 1.0 / J
    x = np.linspace(dx, 1.0 - dx, J - 1)
    phi = np.sin(np.pi * x)
    n = len(phi)

    if HAVE_SCIPY:
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        L = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="csc") / (dx * dx)
        I = sp.eye(n, format="csc")
    else:
        L = np.zeros((n, n), dtype=float)
        for i in range(n):
            L[i, i] = -2.0
            if i > 0:
                L[i, i - 1] = 1.0
            if i < n - 1:
                L[i, i + 1] = 1.0
        L /= dx * dx
        I = np.eye(n)

    A_minus = kappa_minus * L - lam * I
    A_plus = kappa_plus * L - lam * I

    def a(t: float) -> float:
        z = max(t - tau, 0.0)
        return math.exp(-t) + beta * z * z

    def a1(t: float) -> float:
        z = max(t - tau, 0.0)
        return -math.exp(-t) + 2.0 * beta * z

    def exact(t: float) -> np.ndarray:
        return phi * a(t)

    def exact_d1(t: float) -> np.ndarray:
        return phi * a1(t)

    def g_minus(t: float) -> np.ndarray:
        return exact_d1(t) - A_minus @ exact(t)

    def g_plus(t: float) -> np.ndarray:
        return exact_d1(t) - A_plus @ exact(t)

    return base.LinearSwitchProblem(
        T=T,
        tau=tau,
        dim=n,
        A_minus=A_minus,
        A_plus=A_plus,
        g_minus=g_minus,
        g_plus=g_plus,
        exact=exact,
        exact_d1=exact_d1,
        jump1=lambda: np.zeros_like(phi),
        jump2=lambda: 2.0 * beta * phi,
    )


def run_semidiscrete_benign(J: int, m_min: int, m_max: int) -> Dict[str, List[Dict[str, object]]]:
    problem = build_semidiscrete_benign_problem(J=J)
    studies = base.compute_error_rows(problem, m_values=range(m_min, m_max + 1), methods=("standard", "corrected"))

    base.print_table("Semidiscrete benign benchmark: standard", studies["standard"])
    base.print_table("Semidiscrete benign benchmark: corrected", studies["corrected"])

    _save_csv("pde_benign_standard.csv", studies["standard"])
    _save_csv("pde_benign_corrected.csv", studies["corrected"])
    base.plot_convergence(studies, "Semidiscrete benign parabolic benchmark", "figure_semidiscrete_pde_benign.png")

    new_summary = []
    for method, rows in studies.items():
        last = rows[-1]
        new_summary.append({
            "study": "pde_benign",
            "method": method,
            "finest_h": last["h"],
            "finest_Einf": last["Einf"],
            "finest_Einf_EOC": last["Einf_EOC"],
            "finest_E2": last["E2"],
            "finest_E2_EOC": last["E2_EOC"],
        })

    original_summary = _read_csv_if_exists(OUTDIR / "summary_finest_results.csv")
    combined_summary = _append_rows_without_duplicates(original_summary, new_summary)
    _save_csv("summary_finest_results_full.csv", combined_summary)

    return studies


def timing_cost_diagnostics(J: int, Ns: Sequence[int], repeats: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for N in Ns:
        problem = build_semidiscrete_benign_problem(J=J)
        h = problem.T / N
        medians: Dict[str, float] = {}
        for method in ("standard", "corrected"):
            # Warm-up once.
            base.integrate_problem(problem, h, method)
            times = []
            last_Einf = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                tgrid, Y = base.integrate_problem(problem, h, method)
                times.append(time.perf_counter() - t0)
                pointwise = []
                for i, t in enumerate(tgrid):
                    err = base.as_1d(Y[i, :]) - base.as_1d(problem.exact(float(t)))
                    pointwise.append(base.norm_l2(err))
                last_Einf = max(pointwise)
            med = statistics.median(times)
            medians[method] = med
            rows.append({
                "study": "pde_benign_timing",
                "method": method,
                "dimension": problem.dim,
                "N": N,
                "h": h,
                "repeats": repeats,
                "median_seconds": med,
                "mean_seconds": statistics.mean(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "Einf_last_run": last_Einf,
                "relative_to_standard_median": None,
            })

        standard_med = medians.get("standard")
        if standard_med and standard_med > 0:
            for row in rows:
                if row["N"] == N:
                    row["relative_to_standard_median"] = float(row["median_seconds"]) / standard_med

    _save_csv("timing_cost_diagnostics.csv", rows)
    base.print_table("Timing/cost diagnostics for semidiscrete benign test", rows)
    return rows


def write_algorithmic_cost_table() -> None:
    rows = [
        {
            "method": "standard BDF2",
            "event_information_needed": "none",
            "extra_work_at_crossing": "none",
            "dominant_cost": "usual implicit BDF2 solve",
            "effect_on_first_crossing_residual": "none",
            "interpretation": "baseline",
        },
        {
            "method": "restart/history reinitialization",
            "event_information_needed": "event time and post-event model",
            "extra_work_at_crossing": "start-up/reconstruction steps after event",
            "dominant_cost": "implicit solves used for restart/history rebuilding",
            "effect_on_first_crossing_residual": "does not cancel the mixed crossing residual if standard crossing value has already been generated",
            "interpretation": "robust history repair, not direct defect cancellation",
        },
        {
            "method": "defect-targeted correction",
            "event_information_needed": "event fraction theta and derivative-jump data J1/J2",
            "extra_work_at_crossing": "one vector correction added to the BDF2 right-hand side",
            "dominant_cost": "same implicit BDF2 solve; correction is lower-order algebraic work",
            "effect_on_first_crossing_residual": "cancels the targeted leading crossing term when jump data are accurate",
            "interpretation": "local crossing-defect cancellation when jump data are available",
        },
    ]
    _save_csv("algorithmic_cost_table.csv", rows)


def write_environment_and_manifest() -> None:
    env_lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"scipy_available={HAVE_SCIPY}",
        f"matplotlib_available={base.HAVE_MPL}",
    ]
    if HAVE_SCIPY:
        try:
            import scipy
            env_lines.append(f"scipy={scipy.__version__}")
        except Exception:
            pass
    if base.HAVE_MPL:
        try:
            import matplotlib
            env_lines.append(f"matplotlib={matplotlib.__version__}")
        except Exception:
            pass
    (OUTDIR / "run_environment.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    files = sorted(p.name for p in OUTDIR.glob("*"))
    (OUTDIR / "output_manifest.txt").write_text("\n".join(files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semidiscrete benign benchmark and cost diagnostics for the BDF2 temporal-interface study.")
    parser.add_argument("--task", choices=("all", "benign", "timing", "tables"), default="all")
    parser.add_argument("--J", type=int, default=160, help="Number of spatial intervals; dimension is J-1. Default: 160.")
    parser.add_argument("--m-min", type=int, default=5, help="Minimum power m for N=2^m in convergence test. Default: 5.")
    parser.add_argument("--m-max", type=int, default=9, help="Maximum power m for N=2^m in convergence test. Default: 9.")
    parser.add_argument("--timing-N", default="64,128,256", help="Comma-separated N values for timing diagnostics. Default: 64,128,256.")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats. Default: 5.")
    args = parser.parse_args()

    print(f"Writing supplemental outputs to: {OUTDIR.resolve()}")

    if args.task in ("all", "benign"):
        run_semidiscrete_benign(J=args.J, m_min=args.m_min, m_max=args.m_max)

    if args.task in ("all", "timing"):
        Ns = tuple(int(s.strip()) for s in args.timing_N.split(",") if s.strip())
        timing_cost_diagnostics(J=args.J, Ns=Ns, repeats=args.repeats)

    if args.task in ("all", "tables"):
        write_algorithmic_cost_table()
        write_environment_and_manifest()

    print("\nDone. Semidiscrete benign outputs, timing/cost diagnostics, and summary tables were written.")


if __name__ == "__main__":
    main()
