#!/usr/bin/env python3
"""
Additional diagnostic script for the BDF2 temporal-interface supplementary package.

Run from the package root:

    python src/run_unaligned_restart_and_jump_sensitivity.py

It adds two additional diagnostics:

1. A genuine unaligned-crossing restart diagnostic.
   It compares standard BDF2, post-crossing restart, and direct defect correction
   when tau = 0.37 is strictly inside the first crossing step.  The table reports
   the first crossing residual and global errors.  Standard BDF2 and post-crossing
   restart have the same first-crossing residual by construction; direct correction
   changes that residual.

2. Jump-data sensitivity diagnostics.
   They perturb the supplied jump data by J_hat = (1 + eps) J and report how the
   corrected residual and global errors change.  This directly documents the
   exact-jump-data limitation without overclaiming.

The script writes CSV tables, plain-text manuscript-ready tables, figures, and a
small manifest to the existing outputs/ folder.
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bdf2_temporal_interface as base  # noqa: E402

try:
    import run_additional_semidiscrete_benchmark as semi_extra  # noqa: E402
    HAVE_SEMI_EXTRA = True
except Exception:  # pragma: no cover
    semi_extra = None
    HAVE_SEMI_EXTRA = False

EPS = getattr(base, "EPS", 1.0e-14)


def save_csv(filename: str, rows: List[Dict[str, object]]) -> Path:
    path = OUTDIR / filename
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def eoc(errors: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    for k in range(1, len(errors)):
        if errors[k - 1] > 0 and errors[k] > 0:
            out.append(math.log(errors[k - 1] / errors[k], 2.0))
        else:
            out.append(None)
    return out


def fmt_float(x: object, sig: int = 6) -> str:
    if x is None:
        return "---"
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x == 0:
            return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e4:
            return f"{x:.{sig}e}"
        return f"{x:.{sig}f}"
    return str(x)


def write_plain_table(filename: str, title: str, rows: List[Dict[str, object]]) -> Path:
    path = OUTDIR / filename
    if not rows:
        path.write_text(title + "\n(empty)\n", encoding="utf-8")
        return path
    headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    rendered = []
    for row in rows:
        rendered_row = {h: fmt_float(row[h]) for h in headers}
        rendered.append(rendered_row)
        for h in headers:
            widths[h] = max(widths[h], len(rendered_row[h]))
    lines = [title, "=" * len(title)]
    lines.append(" | ".join(h.ljust(widths[h]) for h in headers))
    lines.append("-+-".join("-" * widths[h] for h in headers))
    for row in rendered:
        lines.append(" | ".join(row[h].ljust(widths[h]) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def crossing_residual_exact(problem: base.LinearSwitchProblem, h: float) -> Tuple[float, float, float, float]:
    """Return theta, standard exact crossing residual norm/scalar, predicted leading norm/scalar, corrected residual."""
    kind, n, theta = base.crossing_info(problem.tau, h, problem.T)
    if kind != "crossing" or n < 1:
        raise ValueError("Expected a genuine unaligned crossing with n >= 1.")
    t_nm1 = (n - 1) * h
    t_n = n * h
    t_np1 = (n + 1) * h
    y_nm1 = base.as_1d(problem.exact(t_nm1))
    y_n = base.as_1d(problem.exact(t_n))
    y_np1 = base.as_1d(problem.exact(t_np1))
    dy_np1 = base.as_1d(problem.exact_d1(t_np1))
    residual_vec = (3.0 * y_np1 - 4.0 * y_n + y_nm1) / (2.0 * h) - dy_np1
    predicted_vec = base.c1(theta) * base.as_1d(problem.jump1()) + h * base.c2(theta) * base.as_1d(problem.jump2())
    corrected_vec = residual_vec - predicted_vec
    if problem.dim == 1:
        return theta, float(residual_vec[0]), float(predicted_vec[0]), float(corrected_vec[0])
    return theta, base.norm_l2(residual_vec), base.norm_l2(predicted_vec), base.norm_l2(corrected_vec)


def global_errors(problem: base.LinearSwitchProblem, tgrid: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    h = float(tgrid[1] - tgrid[0])
    pointwise = []
    for i, t in enumerate(tgrid):
        err = base.as_1d(Y[i, :]) - base.as_1d(problem.exact(float(t)))
        pointwise.append(base.norm_l2(err))
    return max(pointwise), math.sqrt(h * sum(e * e for e in pointwise))


def integrate_with_scaled_correction(problem: base.LinearSwitchProblem, h: float, jump_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """BDF2 with one crossing-step correction using scaled jumps J_hat = jump_scale * J."""
    N = base.ensure_integer_steps(problem.T, h)
    tgrid = np.linspace(0.0, problem.T, N + 1)
    Y = np.zeros((N + 1, problem.dim), dtype=float)
    Y[0, :] = base.as_1d(problem.exact(0.0))
    cache: Dict[Tuple[str, float], object] = {}
    Y[1, :] = base.sdirk2_step_linear(Y[0, :], tgrid[0], h, problem.A_minus, problem.g_minus, cache, "minus")

    J1_hat = jump_scale * base.as_1d(problem.jump1())
    J2_hat = jump_scale * base.as_1d(problem.jump2())

    for n in range(1, N):
        tn = float(tgrid[n])
        tnp1 = float(tgrid[n + 1])

        if tnp1 < problem.tau - EPS:
            Y[n + 1, :] = base.bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_minus, problem.g_minus(tnp1), cache, "minus")
            continue

        if tn < problem.tau - EPS and tnp1 > problem.tau + EPS:
            theta = (problem.tau - tn) / h
            Y[n + 1, :] = base.corrected_bdf2_crossing_step(
                Y[n - 1, :], Y[n, :], h, theta, problem.A_plus, problem.g_plus(tnp1), J1_hat, J2_hat, cache
            )
            continue

        Y[n + 1, :] = base.bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_plus, problem.g_plus(tnp1), cache, "plus")

    return tgrid, Y


def integrate_postcrossing_restart(problem: base.LinearSwitchProblem, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Post-crossing restart under the paper's convention.

    The crossing value is first produced by the same standard BDF2 mixed stencil.
    Then the next right-side value is produced by the one-step SDIRK2 starter, and
    ordinary BDF2 is resumed with right-side history.
    """
    N = base.ensure_integer_steps(problem.T, h)
    tgrid = np.linspace(0.0, problem.T, N + 1)
    Y = np.zeros((N + 1, problem.dim), dtype=float)
    Y[0, :] = base.as_1d(problem.exact(0.0))
    cache: Dict[Tuple[str, float], object] = {}
    Y[1, :] = base.sdirk2_step_linear(Y[0, :], tgrid[0], h, problem.A_minus, problem.g_minus, cache, "minus")
    restart_next_step = False

    for n in range(1, N):
        tn = float(tgrid[n])
        tnp1 = float(tgrid[n + 1])

        if restart_next_step:
            # Reinitialize the post-interface history after the crossing value already exists.
            Y[n + 1, :] = base.sdirk2_step_linear(Y[n, :], tn, h, problem.A_plus, problem.g_plus, cache, "plus")
            restart_next_step = False
            continue

        if tnp1 < problem.tau - EPS:
            Y[n + 1, :] = base.bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_minus, problem.g_minus(tnp1), cache, "minus")
            continue

        if tn < problem.tau - EPS and tnp1 > problem.tau + EPS:
            # The crucial convention: the crossing value is still standard BDF2.
            Y[n + 1, :] = base.bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_plus, problem.g_plus(tnp1), cache, "plus")
            restart_next_step = True
            continue

        Y[n + 1, :] = base.bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_plus, problem.g_plus(tnp1), cache, "plus")

    return tgrid, Y


def run_genuine_unaligned_restart_diagnostic() -> List[Dict[str, object]]:
    problem = base.build_benign_scalar_problem(T=1.0, tau=0.37, beta=1.0, mu=10.0)
    Ns = [2**m for m in range(5, 12)]

    per_method_errors: Dict[str, List[Tuple[float, float]]] = {
        "standard": [],
        "post_crossing_restart": [],
        "direct_correction": [],
    }
    cache_rows: Dict[int, Dict[str, object]] = {}

    for N in Ns:
        h = problem.T / N
        theta, R_std, predicted, R_corr = crossing_residual_exact(problem, h)
        row: Dict[str, object] = {
            "N": N,
            "h": h,
            "theta": theta,
            "standard_crossing_residual": R_std,
            "restart_crossing_residual": R_std,
            "corrected_crossing_residual": R_corr,
            "local_residual_gain_abs": abs(R_std) / max(abs(R_corr), 1e-300),
        }

        tgrid, Y = base.integrate_problem(problem, h, "standard")
        per_method_errors["standard"].append(global_errors(problem, tgrid, Y))
        row["standard_Einf"], row["standard_E2"] = per_method_errors["standard"][-1]

        tgrid, Y = integrate_postcrossing_restart(problem, h)
        per_method_errors["post_crossing_restart"].append(global_errors(problem, tgrid, Y))
        row["restart_Einf"], row["restart_E2"] = per_method_errors["post_crossing_restart"][-1]

        tgrid, Y = base.integrate_problem(problem, h, "corrected")
        per_method_errors["direct_correction"].append(global_errors(problem, tgrid, Y))
        row["corrected_Einf"], row["corrected_E2"] = per_method_errors["direct_correction"][-1]

        row["restart_minus_standard_crossing_residual"] = row["restart_crossing_residual"] - row["standard_crossing_residual"]
        row["corrected_vs_standard_Einf_gain"] = row["standard_Einf"] / row["corrected_Einf"]
        row["corrected_vs_restart_Einf_gain"] = row["restart_Einf"] / row["corrected_Einf"]
        cache_rows[N] = row

    # Add EOCs.
    for method, prefix in [
        ("standard", "standard"),
        ("post_crossing_restart", "restart"),
        ("direct_correction", "corrected"),
    ]:
        Einfs = [err[0] for err in per_method_errors[method]]
        E2s = [err[1] for err in per_method_errors[method]]
        eoc_inf = eoc(Einfs)
        eoc_2 = eoc(E2s)
        for i, N in enumerate(Ns):
            cache_rows[N][f"{prefix}_Einf_EOC"] = eoc_inf[i]
            cache_rows[N][f"{prefix}_E2_EOC"] = eoc_2[i]

    rows = [cache_rows[N] for N in Ns]
    save_csv("genuine_unaligned_restart_diagnostic.csv", rows)

    compact = []
    for row in rows:
        compact.append({
            "N": row["N"],
            "h": row["h"],
            "theta": row["theta"],
            "R_standard": row["standard_crossing_residual"],
            "R_restart": row["restart_crossing_residual"],
            "R_corrected": row["corrected_crossing_residual"],
            "std_Einf": row["standard_Einf"],
            "restart_Einf": row["restart_Einf"],
            "corr_Einf": row["corrected_Einf"],
            "corr/std_gain": row["corrected_vs_standard_Einf_gain"],
        })
    write_plain_table("genuine_unaligned_restart_diagnostic_table.txt", "Genuine unaligned-crossing restart diagnostic", compact)

    if HAVE_MPL:
        hs = [float(r["h"]) for r in rows]
        plt.figure(figsize=(7.0, 5.0))
        plt.loglog(hs, [float(r["standard_Einf"]) for r in rows], marker="o", label="standard")
        plt.loglog(hs, [float(r["restart_Einf"]) for r in rows], marker="o", label="post-crossing restart")
        plt.loglog(hs, [float(r["corrected_Einf"]) for r in rows], marker="o", label="direct correction")
        plt.gca().invert_xaxis()
        plt.xlabel("h")
        plt.ylabel(r"$E_\infty$")
        plt.title("Genuine unaligned crossing: standard, restart, correction")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "figure_genuine_unaligned_restart_diagnostic.png", dpi=600, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7.0, 5.0))
        plt.semilogy([int(r["N"]) for r in rows], [abs(float(r["standard_crossing_residual"])) for r in rows], marker="o", label="standard/restart residual")
        plt.semilogy([int(r["N"]) for r in rows], [abs(float(r["corrected_crossing_residual"])) for r in rows], marker="o", label="corrected residual")
        plt.xlabel("N")
        plt.ylabel("absolute first-crossing residual")
        plt.title("First-crossing residual: restart does not change it")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "figure_genuine_unaligned_crossing_residuals.png", dpi=600, bbox_inches="tight")
        plt.close()

    return rows


def run_jump_sensitivity_scalar() -> List[Dict[str, object]]:
    problem = base.build_benign_scalar_problem(T=1.0, tau=0.37, beta=1.0, mu=10.0)
    N = 2048
    h = problem.T / N
    theta, R_std, predicted, R_corr_exact = crossing_residual_exact(problem, h)
    tgrid, Y = base.integrate_problem(problem, h, "standard")
    standard_Einf, standard_E2 = global_errors(problem, tgrid, Y)

    eps_values = [-2e-1, -1e-1, -5e-2, -1e-2, 0.0, 1e-2, 5e-2, 1e-1, 2e-1]
    rows: List[Dict[str, object]] = []
    for eps in eps_values:
        scale = 1.0 + eps
        tgrid, Y = integrate_with_scaled_correction(problem, h, scale)
        Einf, E2 = global_errors(problem, tgrid, Y)
        R_after = R_std - scale * predicted
        rows.append({
            "epsilon": eps,
            "jump_scale": scale,
            "N": N,
            "h": h,
            "theta": theta,
            "standard_crossing_residual": R_std,
            "corrected_crossing_residual": R_after,
            "standard_Einf": standard_Einf,
            "corrected_Einf": Einf,
            "Einf_gain_standard_over_corrected": standard_Einf / Einf,
            "standard_E2": standard_E2,
            "corrected_E2": E2,
            "E2_gain_standard_over_corrected": standard_E2 / E2,
        })

    save_csv("jump_data_sensitivity_benign_scalar.csv", rows)
    compact = [{
        "epsilon": r["epsilon"],
        "R_after": r["corrected_crossing_residual"],
        "corr_Einf": r["corrected_Einf"],
        "Einf_gain": r["Einf_gain_standard_over_corrected"],
        "corr_E2": r["corrected_E2"],
        "E2_gain": r["E2_gain_standard_over_corrected"],
    } for r in rows]
    write_plain_table("jump_data_sensitivity_benign_scalar_table.txt", "Jump-data sensitivity: benign scalar benchmark", compact)

    if HAVE_MPL:
        eps_plot = [float(r["epsilon"]) for r in rows]
        gain_plot = [float(r["Einf_gain_standard_over_corrected"]) for r in rows]
        plt.figure(figsize=(7.0, 5.0))
        plt.semilogx([max(e, 1e-6) for e in eps_plot], gain_plot, marker="o")
        plt.xlabel(r"jump-data perturbation $\epsilon$ (0 plotted at $10^{-6}$)")
        plt.ylabel(r"$E_\infty$ gain: standard / corrected")
        plt.title("Sensitivity to perturbed jump data: benign scalar")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(OUTDIR / "figure_jump_data_sensitivity_benign_scalar.png", dpi=600, bbox_inches="tight")
        plt.close()

    return rows


def run_jump_sensitivity_semidiscrete() -> List[Dict[str, object]]:
    if not HAVE_SEMI_EXTRA:
        return []
    problem = semi_extra.build_semidiscrete_benign_problem(T=1.0, tau=0.37, beta=1.0, J=160)
    N = 128
    h = problem.T / N
    theta, R_std, predicted, R_corr_exact = crossing_residual_exact(problem, h)
    tgrid, Y = base.integrate_problem(problem, h, "standard")
    standard_Einf, standard_E2 = global_errors(problem, tgrid, Y)

    eps_values = [-2e-1, -1e-1, -5e-2, -1e-2, 0.0, 1e-2, 5e-2, 1e-1, 2e-1]
    rows: List[Dict[str, object]] = []
    for eps in eps_values:
        scale = 1.0 + eps
        tgrid, Y = integrate_with_scaled_correction(problem, h, scale)
        Einf, E2 = global_errors(problem, tgrid, Y)
        # For vector case, this is the norm of residual_after = R_exact - scale * predicted_vector.
        # We recompute from the exact residual vector for accuracy.
        kind, n, theta = base.crossing_info(problem.tau, h, problem.T)
        t_nm1 = (n - 1) * h
        t_n = n * h
        t_np1 = (n + 1) * h
        y_nm1 = base.as_1d(problem.exact(t_nm1))
        y_n = base.as_1d(problem.exact(t_n))
        y_np1 = base.as_1d(problem.exact(t_np1))
        dy_np1 = base.as_1d(problem.exact_d1(t_np1))
        residual_vec = (3.0 * y_np1 - 4.0 * y_n + y_nm1) / (2.0 * h) - dy_np1
        predicted_vec = base.c1(theta) * base.as_1d(problem.jump1()) + h * base.c2(theta) * base.as_1d(problem.jump2())
        R_after_norm = base.norm_l2(residual_vec - scale * predicted_vec)
        rows.append({
            "epsilon": eps,
            "jump_scale": scale,
            "N": N,
            "dimension": problem.dim,
            "h": h,
            "theta": theta,
            "standard_crossing_residual_norm": R_std,
            "corrected_crossing_residual_norm": R_after_norm,
            "standard_Einf": standard_Einf,
            "corrected_Einf": Einf,
            "Einf_gain_standard_over_corrected": standard_Einf / Einf,
            "standard_E2": standard_E2,
            "corrected_E2": E2,
            "E2_gain_standard_over_corrected": standard_E2 / E2,
        })

    save_csv("jump_data_sensitivity_benign_semidiscrete.csv", rows)
    compact = [{
        "epsilon": r["epsilon"],
        "dimension": r["dimension"],
        "R_after_norm": r["corrected_crossing_residual_norm"],
        "corr_Einf": r["corrected_Einf"],
        "Einf_gain": r["Einf_gain_standard_over_corrected"],
        "corr_E2": r["corrected_E2"],
        "E2_gain": r["E2_gain_standard_over_corrected"],
    } for r in rows]
    write_plain_table("jump_data_sensitivity_benign_semidiscrete_table.txt", "Jump-data sensitivity: benign semidiscrete benchmark", compact)

    if HAVE_MPL:
        plt.figure(figsize=(7.0, 5.0))
        plt.semilogx([max(float(r["epsilon"]), 1e-6) for r in rows], [float(r["Einf_gain_standard_over_corrected"]) for r in rows], marker="o")
        plt.xlabel(r"jump-data perturbation $\epsilon$ (0 plotted at $10^{-6}$)")
        plt.ylabel(r"$E_\infty$ gain: standard / corrected")
        plt.title("Sensitivity to perturbed jump data: benign semidiscrete")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(OUTDIR / "figure_jump_data_sensitivity_benign_semidiscrete.png", dpi=600, bbox_inches="tight")
        plt.close()

    return rows


def write_manifest(genuine_rows: List[Dict[str, object]], scalar_rows: List[Dict[str, object]], semi_rows: List[Dict[str, object]]) -> Path:
    lines = []
    lines.append("Additional diagnostic outputs")
    lines.append("================================")
    lines.append("")
    lines.append("Generated by: python src/run_unaligned_restart_and_jump_sensitivity.py")
    lines.append("")
    lines.append("Main new outputs:")
    for name in [
        "genuine_unaligned_restart_diagnostic.csv",
        "genuine_unaligned_restart_diagnostic_table.txt",
        "figure_genuine_unaligned_restart_diagnostic.png",
        "figure_genuine_unaligned_crossing_residuals.png",
        "jump_data_sensitivity_benign_scalar.csv",
        "jump_data_sensitivity_benign_scalar_table.txt",
        "figure_jump_data_sensitivity_benign_scalar.png",
        "jump_data_sensitivity_benign_semidiscrete.csv",
        "jump_data_sensitivity_benign_semidiscrete_table.txt",
        "figure_jump_data_sensitivity_benign_semidiscrete.png",
    ]:
        if (OUTDIR / name).exists():
            lines.append(f"- outputs/{name}")
    lines.append("")
    if genuine_rows:
        last = genuine_rows[-1]
        lines.append("Finest-grid genuine unaligned restart diagnostic:")
        lines.append(f"- N = {last['N']}, h = {float(last['h']):.8e}, theta = {float(last['theta']):.6f}")
        lines.append(f"- Standard crossing residual = {float(last['standard_crossing_residual']):.8e}")
        lines.append(f"- Restart crossing residual  = {float(last['restart_crossing_residual']):.8e}")
        lines.append(f"- Corrected crossing residual = {float(last['corrected_crossing_residual']):.8e}")
        lines.append(f"- Corrected/standard max-error gain = {float(last['corrected_vs_standard_Einf_gain']):.4f}")
        lines.append("")
    if scalar_rows:
        zero = next((r for r in scalar_rows if abs(float(r["epsilon"])) < 1e-15), scalar_rows[0])
        minus = scalar_rows[0]
        plus = scalar_rows[-1]
        lines.append("Jump-data sensitivity, benign scalar:")
        lines.append(f"- At epsilon = 0, max-error gain = {float(zero['Einf_gain_standard_over_corrected']):.4f}")
        lines.append(f"- At epsilon = {float(minus['epsilon']):.3g}, max-error gain = {float(minus['Einf_gain_standard_over_corrected']):.4f}")
        lines.append(f"- At epsilon = {float(plus['epsilon']):.3g}, max-error gain = {float(plus['Einf_gain_standard_over_corrected']):.4f}")
        lines.append("")
    if semi_rows:
        zero = next((r for r in semi_rows if abs(float(r["epsilon"])) < 1e-15), semi_rows[0])
        minus = semi_rows[0]
        plus = semi_rows[-1]
        lines.append("Jump-data sensitivity, benign semidiscrete:")
        lines.append(f"- Dimension = {zero['dimension']}, N = {zero['N']}")
        lines.append(f"- At epsilon = 0, max-error gain = {float(zero['Einf_gain_standard_over_corrected']):.4f}")
        lines.append(f"- At epsilon = {float(minus['epsilon']):.3g}, max-error gain = {float(minus['Einf_gain_standard_over_corrected']):.4f}")
        lines.append(f"- At epsilon = {float(plus['epsilon']):.3g}, max-error gain = {float(plus['Einf_gain_standard_over_corrected']):.4f}")
        lines.append("")
    path = OUTDIR / "additional_diagnostics_manifest.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_checksums() -> None:
    files = []
    for folder in [OUTDIR, ROOT / "figures", ROOT / "src"]:
        if folder.exists():
            files.extend(p for p in folder.rglob("*") if p.is_file())
    checksum_path = ROOT / "checksums_sha256.txt"
    rows = []
    for p in sorted(files):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(f"{h}  {p.relative_to(ROOT).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    print("BDF2 additional numerical diagnostics")
    print(f"Package root: {ROOT}")
    print(f"Writing to:    {OUTDIR}")

    genuine_rows = run_genuine_unaligned_restart_diagnostic()
    scalar_rows = run_jump_sensitivity_scalar()
    semi_rows = run_jump_sensitivity_semidiscrete()
    manifest = write_manifest(genuine_rows, scalar_rows, semi_rows)
    update_checksums()

    print("\nDONE. New files are in outputs/.")
    print(f"Read this summary: {manifest.relative_to(ROOT)}")
    print("For submission, zip the whole bdf2_temporal_interface_supplementary folder after this run.")


if __name__ == "__main__":
    main()
