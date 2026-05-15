#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ==========================================================
# Output paths
# ==========================================================
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
EPS = 1.0e-14


# ==========================================================
# Utilities
# ==========================================================

def ensure_integer_steps(T: float, h: float, tol: float = 1e-13) -> int:
    q = T / h
    N = int(round(q))
    if abs(q - N) > tol:
        raise ValueError(f"T/h must be an integer, got T={T}, h={h}, T/h={q}.")
    return N


def save_csv(filename: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path = OUTDIR / filename
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows: List[Dict[str, object]], digits: int = 8) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not rows:
        print("(empty)")
        return

    def fmt(x: object) -> str:
        if x is None:
            return "---"
        if isinstance(x, float):
            if x == 0.0:
                return f"{x:.{digits}f}"
            if abs(x) < 1e-3 or abs(x) >= 1e4:
                return f"{x:.{digits}e}"
            return f"{x:.{digits}f}"
        return str(x)

    headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(row[h])))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(fmt(row[h]).ljust(widths[h]) for h in headers))


def eoc_from_errors(errors: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    for k in range(1, len(errors)):
        e1 = float(errors[k - 1])
        e2 = float(errors[k])
        if e1 <= 0.0 or e2 <= 0.0:
            out.append(None)
        else:
            out.append(math.log(e1 / e2, 2.0))
    return out


def norm_inf(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 0:
        return float(abs(arr))
    return float(np.max(np.abs(arr)))


def norm_l2(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 0:
        return float(abs(arr))
    return float(np.linalg.norm(arr.reshape(-1), 2))


def as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def crossing_info(tau: float, h: float, T: float) -> Tuple[str, int, float]:
    """
    Returns (kind, n, theta), where
      kind = 'aligned' if tau = t_n,
      kind = 'crossing' if t_n < tau < t_{n+1},
      theta = (tau - t_n)/h for the crossing case.
    """
    N = ensure_integer_steps(T, h)
    q = tau / h
    nq = int(round(q))
    if 0 <= nq <= N and abs(q - nq) < 1e-14:
        return ("aligned", nq, 0.0)
    n = int(math.floor(q))
    if not (0 <= n < N):
        raise ValueError("tau must lie inside [0,T].")
    theta = (tau - n * h) / h
    return ("crossing", n, theta)


# ==========================================================
# Crossing coefficients from the paper
# Standard residual = c1(theta) J1 + h c2(theta) J2 + ...
# ==========================================================

def c1(theta: float) -> float:
    return 0.5 * (1.0 - 3.0 * theta)


def c2(theta: float) -> float:
    return -0.25 * (1.0 + 2.0 * theta - 3.0 * theta * theta)


def c3(theta: float) -> float:
    return (1.0 + 3.0 * theta + 3.0 * theta * theta - 3.0 * theta**3) / 12.0


# ==========================================================
# Manufactured exact solutions
# ==========================================================

def exact_benign(t: float, tau: float, beta: float) -> float:
    z = max(t - tau, 0.0)
    return math.exp(t) + beta * z * z


def exact_benign_d1(t: float, tau: float, beta: float) -> float:
    z = max(t - tau, 0.0)
    return math.exp(t) + 2.0 * beta * z


def exact_benign_d2(t: float, tau: float, beta: float) -> float:
    if t < tau:
        return math.exp(t)
    return math.exp(t) + 2.0 * beta


def exact_benign_d3(t: float, tau: float, beta: float) -> float:
    return math.exp(t)


def exact_severe(t: float, tau: float, beta: float) -> float:
    z = max(t - tau, 0.0)
    return math.exp(t) + beta * z


def exact_severe_d1(t: float, tau: float, beta: float) -> float:
    return math.exp(t) + (beta if t >= tau - EPS else 0.0)


def exact_severe_d2(t: float, tau: float, beta: float) -> float:
    return math.exp(t)


def exact_severe_d3(t: float, tau: float, beta: float) -> float:
    return math.exp(t)


# ==========================================================
# Direct residual validation on exact data
# ==========================================================

def direct_crossing_residual(
    exact: Callable[[float], float],
    exact_d1: Callable[[float], float],
    tau: float,
    T: float,
    h: float,
) -> Tuple[float, float]:
    kind, n, theta = crossing_info(tau, h, T)
    if kind != "crossing":
        raise ValueError("direct_crossing_residual requires a genuine crossing step.")
    if n < 1:
        raise ValueError("Need n >= 1 for BDF2 crossing step.")
    tn1 = (n - 1) * h
    tn = n * h
    tnp1 = (n + 1) * h
    residual = (3.0 * exact(tnp1) - 4.0 * exact(tn) + exact(tn1)) / (2.0 * h) - exact_d1(tnp1)
    return residual, theta


def defect_validation_rows(
    case: str,
    T: float,
    tau: float,
    beta: float,
    m_values: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    if case == "benign":
        exact = lambda t: exact_benign(t, tau, beta)
        exact_d1 = lambda t: exact_benign_d1(t, tau, beta)
        J1 = 0.0
        J2 = 2.0 * beta
    elif case == "severe":
        exact = lambda t: exact_severe(t, tau, beta)
        exact_d1 = lambda t: exact_severe_d1(t, tau, beta)
        J1 = beta
        J2 = 0.0
    else:
        raise ValueError("case must be 'benign' or 'severe'.")

    for m in m_values:
        N = 2**m
        h = T / N
        kind, n, theta = crossing_info(tau, h, T)
        if kind != "crossing" or n < 1:
            continue
        residual, theta = direct_crossing_residual(exact, exact_d1, tau, T, h)
        predicted = c1(theta) * J1 + h * c2(theta) * J2
        rows.append({
            "case": case,
            "N": N,
            "h": h,
            "theta": theta,
            "residual": residual,
            "predicted_leading": predicted,
            "difference": residual - predicted,
            "scaled_difference_over_h2": (residual - predicted) / (h * h),
        })
    return rows


# ==========================================================
# Linear switched affine ODE problem: y' = A y + g(t)
# ==========================================================

@dataclass
class LinearSwitchProblem:
    T: float
    tau: float
    dim: int
    A_minus: object
    A_plus: object
    g_minus: Callable[[float], np.ndarray]
    g_plus: Callable[[float], np.ndarray]
    exact: Callable[[float], np.ndarray]
    exact_d1: Callable[[float], np.ndarray]
    jump1: Callable[[], np.ndarray]
    jump2: Callable[[], np.ndarray]


def make_solver(B):
    if HAVE_SCIPY and sp.issparse(B):
        lu = spla.splu(B.tocsc())
        return lu.solve
    B = np.asarray(B, dtype=float)
    def solve(rhs):
        rhs = np.asarray(rhs, dtype=float)
        return np.linalg.solve(B, rhs)
    return solve


def sdirk2_step_linear(y_n, t_n, h, A, gfun, cache: Dict[Tuple[str, float], Callable], keyname: str):
    gamma = 1.0 - 1.0 / math.sqrt(2.0)
    n = len(y_n)
    key = (f"sdirk2_{keyname}", h)
    if key not in cache:
        if HAVE_SCIPY and sp.issparse(A):
            B = sp.eye(n, format="csc") - gamma * h * A
        else:
            B = np.eye(n) - gamma * h * np.asarray(A, dtype=float)
        cache[key] = make_solver(B)
    solve = cache[key]
    g1 = as_1d(gfun(t_n + gamma * h))
    k1 = solve(A @ y_n + g1)
    g2 = as_1d(gfun(t_n + h))
    k2 = solve(A @ (y_n + (1.0 - gamma) * h * k1) + g2)
    return as_1d(y_n + h * ((1.0 - gamma) * k1 + gamma * k2))


def bdf2_step_linear(y_nm1, y_n, h, A, g_np1, cache: Dict[Tuple[str, float], Callable], keyname: str):
    n = len(y_n)
    key = (f"bdf2_{keyname}", h)
    if key not in cache:
        if HAVE_SCIPY and sp.issparse(A):
            B = (3.0 / (2.0 * h)) * sp.eye(n, format="csc") - A
        else:
            B = (3.0 / (2.0 * h)) * np.eye(n) - np.asarray(A, dtype=float)
        cache[key] = make_solver(B)
    solve = cache[key]
    rhs = (4.0 * y_n - y_nm1) / (2.0 * h) + as_1d(g_np1)
    return as_1d(solve(rhs))


def corrected_bdf2_crossing_step(y_nm1, y_n, h, theta, A_plus, g_plus_np1, J1, J2, cache):
    n = len(y_n)
    key = ("bdf2_plus", h)
    if key not in cache:
        if HAVE_SCIPY and sp.issparse(A_plus):
            B = (3.0 / (2.0 * h)) * sp.eye(n, format="csc") - A_plus
        else:
            B = (3.0 / (2.0 * h)) * np.eye(n) - np.asarray(A_plus, dtype=float)
        cache[key] = make_solver(B)
    solve = cache[key]
    corr = c1(theta) * as_1d(J1) + h * c2(theta) * as_1d(J2)
    rhs = (4.0 * y_n - y_nm1) / (2.0 * h) + as_1d(g_plus_np1) + corr
    return as_1d(solve(rhs))


def integrate_problem(problem: LinearSwitchProblem, h: float, method: str) -> Tuple[np.ndarray, np.ndarray]:
    if method not in {"standard", "corrected", "restart"}:
        raise ValueError("method must be 'standard', 'corrected', or 'restart'.")
    N = ensure_integer_steps(problem.T, h)
    if problem.tau <= h + 1e-15:
        raise ValueError("Choose tau > h so that BDF2 has valid prehistory before the interface.")

    tgrid = np.linspace(0.0, problem.T, N + 1)
    Y = np.zeros((N + 1, problem.dim), dtype=float)
    Y[0, :] = as_1d(problem.exact(0.0))

    cache: Dict[Tuple[str, float], Callable] = {}
    Y[1, :] = sdirk2_step_linear(as_1d(Y[0, :]), tgrid[0], h, problem.A_minus, problem.g_minus, cache, "minus")

    J1 = as_1d(problem.jump1())
    J2 = as_1d(problem.jump2())

    for n in range(1, N):
        tn = tgrid[n]
        tnp1 = tgrid[n + 1]

        if tnp1 < problem.tau - EPS:
            Y[n + 1, :] = bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_minus, problem.g_minus(tnp1), cache, "minus")
            continue

        if tn < problem.tau - EPS and tnp1 > problem.tau + EPS:
            theta = (problem.tau - tn) / h
            if method == "corrected":
                Y[n + 1, :] = corrected_bdf2_crossing_step(Y[n - 1, :], Y[n, :], h, theta, problem.A_plus, problem.g_plus(tnp1), J1, J2, cache)
            else:
                Y[n + 1, :] = bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_plus, problem.g_plus(tnp1), cache, "plus")
            continue

        if abs(tn - problem.tau) <= EPS:
            if method == "restart":
                Y[n + 1, :] = sdirk2_step_linear(Y[n, :], tn, h, problem.A_plus, problem.g_plus, cache, "plus")
            else:
                Y[n + 1, :] = bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_plus, problem.g_plus(tnp1), cache, "plus")
            continue

        Y[n + 1, :] = bdf2_step_linear(Y[n - 1, :], Y[n, :], h, problem.A_plus, problem.g_plus(tnp1), cache, "plus")

    return tgrid, Y


def compute_error_rows(problem: LinearSwitchProblem, m_values: Sequence[int], methods: Sequence[str]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for method in methods:
        hs: List[float] = []
        errs_inf: List[float] = []
        errs_2: List[float] = []
        Ns: List[int] = []
        for m in m_values:
            N = 2**m
            h = problem.T / N
            tgrid, Y = integrate_problem(problem, h, method)
            pointwise = []
            for i, t in enumerate(tgrid):
                err = as_1d(Y[i, :]) - as_1d(problem.exact(float(t)))
                pointwise.append(norm_l2(err))
            hs.append(h)
            Ns.append(N)
            errs_inf.append(max(pointwise))
            errs_2.append(math.sqrt(h * sum(e * e for e in pointwise)))

        eoc_inf = eoc_from_errors(errs_inf)
        eoc_2 = eoc_from_errors(errs_2)
        rows = []
        for i in range(len(hs)):
            rows.append({
                "method": method,
                "N": Ns[i],
                "h": hs[i],
                "Einf": errs_inf[i],
                "Einf_EOC": eoc_inf[i],
                "E2": errs_2[i],
                "E2_EOC": eoc_2[i],
            })
        out[method] = rows
    return out


# ==========================================================
# Problem builders
# ==========================================================

def build_benign_scalar_problem(T=1.0, tau=0.37, beta=1.0, mu=10.0) -> LinearSwitchProblem:
    A_minus = np.array([[-mu]], dtype=float)
    A_plus = np.array([[-mu]], dtype=float)

    def exact(t: float) -> np.ndarray:
        return np.array([exact_benign(t, tau, beta)], dtype=float)

    def exact_d1(t: float) -> np.ndarray:
        return np.array([exact_benign_d1(t, tau, beta)], dtype=float)

    def g_minus(t: float) -> np.ndarray:
        return exact_d1(t) - A_minus @ exact(t)

    def g_plus(t: float) -> np.ndarray:
        return exact_d1(t) - A_plus @ exact(t)

    return LinearSwitchProblem(
        T=T,
        tau=tau,
        dim=1,
        A_minus=A_minus,
        A_plus=A_plus,
        g_minus=g_minus,
        g_plus=g_plus,
        exact=exact,
        exact_d1=exact_d1,
        jump1=lambda: np.array([0.0]),
        jump2=lambda: np.array([2.0 * beta]),
    )


def build_severe_scalar_problem(T=1.0, tau=0.37, beta=1.0, mu=10.0) -> LinearSwitchProblem:
    A_minus = np.array([[-mu]], dtype=float)
    A_plus = np.array([[-mu]], dtype=float)

    def exact(t: float) -> np.ndarray:
        return np.array([exact_severe(t, tau, beta)], dtype=float)

    def exact_d1(t: float) -> np.ndarray:
        return np.array([exact_severe_d1(t, tau, beta)], dtype=float)

    def g_minus(t: float) -> np.ndarray:
        return exact_d1(t) - A_minus @ exact(t)

    def g_plus(t: float) -> np.ndarray:
        return exact_d1(t) - A_plus @ exact(t)

    return LinearSwitchProblem(
        T=T,
        tau=tau,
        dim=1,
        A_minus=A_minus,
        A_plus=A_plus,
        g_minus=g_minus,
        g_plus=g_plus,
        exact=exact,
        exact_d1=exact_d1,
        jump1=lambda: np.array([beta]),
        jump2=lambda: np.array([0.0]),
    )


def build_semidiscrete_severe_problem(
    T=1.0,
    tau=0.37,
    beta=1.0,
    lam=1.0,
    kappa_minus=1.0,
    kappa_plus=3.0,
    J=120,
) -> LinearSwitchProblem:
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

    def base(t: float) -> float:
        return math.exp(-t) + beta * max(t - tau, 0.0)

    def base_d1(t: float) -> float:
        return -math.exp(-t) + (beta if t >= tau - EPS else 0.0)

    def exact(t: float) -> np.ndarray:
        return phi * base(t)

    def exact_d1(t: float) -> np.ndarray:
        return phi * base_d1(t)

    def g_minus(t: float) -> np.ndarray:
        return exact_d1(t) - A_minus @ exact(t)

    def g_plus(t: float) -> np.ndarray:
        return exact_d1(t) - A_plus @ exact(t)

    return LinearSwitchProblem(
        T=T,
        tau=tau,
        dim=n,
        A_minus=A_minus,
        A_plus=A_plus,
        g_minus=g_minus,
        g_plus=g_plus,
        exact=exact,
        exact_d1=exact_d1,
        jump1=lambda: beta * phi,
        jump2=lambda: np.zeros_like(phi),
    )


# ==========================================================
# Plotting
# ==========================================================

def plot_convergence(studies: Dict[str, List[Dict[str, object]]], title: str, filename: str) -> None:
    if not HAVE_MPL:
        return
    plt.figure(figsize=(7.2, 5.2))
    for method, rows in studies.items():
        hs = [float(row["h"]) for row in rows]
        errs = [float(row["Einf"]) for row in rows]
        plt.loglog(hs, errs, marker="o", label=method)
    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel(r"$E_\infty$")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=600, bbox_inches="tight")
    plt.close()


def plot_theta_scan(T: float, beta: float, h: float, filename: str) -> None:
    if not HAVE_MPL:
        return
    thetas = np.linspace(0.05, 0.95, 181)
    severe_standard = np.abs([c1(th) * beta for th in thetas])
    benign_standard = np.abs([h * c2(th) * (2.0 * beta) for th in thetas])
    plt.figure(figsize=(7.2, 5.2))
    plt.plot(thetas, severe_standard, label=r"$|c_1(\theta)J_1|$")
    plt.plot(thetas, benign_standard, label=r"$|h c_2(\theta)J_2|$")
    plt.axvline(1.0 / 3.0, linestyle="--")
    plt.xlabel(r"$\theta$")
    plt.ylabel("predicted leading magnitude")
    plt.title("Sensitivity of leading crossing terms to interface location")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=600, bbox_inches="tight")
    plt.close()


# ==========================================================
# Driver
# ==========================================================

def main() -> None:
    print(f"Writing outputs to: {OUTDIR.resolve()}")

    # 1) Direct defect validation
    rows_benign = defect_validation_rows("benign", T=1.0, tau=0.37, beta=1.0, m_values=range(5, 12))
    rows_severe = defect_validation_rows("severe", T=1.0, tau=0.37, beta=1.0, m_values=range(5, 12))
    print_table("Direct defect validation: benign case", rows_benign)
    print_table("Direct defect validation: severe case", rows_severe)
    save_csv("direct_defect_benign.csv", rows_benign)
    save_csv("direct_defect_severe.csv", rows_severe)

    # 2) Benign scalar convergence
    benign_problem = build_benign_scalar_problem(T=1.0, tau=0.37, beta=1.0, mu=10.0)
    benign_studies = compute_error_rows(benign_problem, m_values=range(5, 12), methods=("standard", "corrected"))
    print_table("Benign scalar benchmark: standard", benign_studies["standard"])
    print_table("Benign scalar benchmark: corrected", benign_studies["corrected"])
    save_csv("benign_standard.csv", benign_studies["standard"])
    save_csv("benign_corrected.csv", benign_studies["corrected"])
    plot_convergence(benign_studies, "Benign scalar manufactured benchmark", "figure_benign_scalar.png")

    # 3) Severe scalar convergence
    severe_problem = build_severe_scalar_problem(T=1.0, tau=0.37, beta=1.0, mu=10.0)
    severe_studies = compute_error_rows(severe_problem, m_values=range(5, 12), methods=("standard", "corrected"))
    print_table("Severe scalar benchmark: standard", severe_studies["standard"])
    print_table("Severe scalar benchmark: corrected", severe_studies["corrected"])
    save_csv("severe_standard.csv", severe_studies["standard"])
    save_csv("severe_corrected.csv", severe_studies["corrected"])
    plot_convergence(severe_studies, "Severe scalar manufactured benchmark", "figure_severe_scalar.png")

    # 4) Aligned restart experiment
    aligned_problem = build_severe_scalar_problem(T=1.0, tau=0.5, beta=1.0, mu=10.0)
    aligned_studies = compute_error_rows(aligned_problem, m_values=range(5, 12), methods=("standard", "restart"))
    print_table("Aligned-event experiment: standard", aligned_studies["standard"])
    print_table("Aligned-event experiment: restart", aligned_studies["restart"])
    save_csv("aligned_standard.csv", aligned_studies["standard"])
    save_csv("aligned_restart.csv", aligned_studies["restart"])
    plot_convergence(aligned_studies, "Aligned-event test: standard versus restart", "figure_aligned_restart.png")

    # 5) Semidiscrete severe benchmark
    pde_problem = build_semidiscrete_severe_problem(T=1.0, tau=0.37, beta=1.0, lam=1.0, kappa_minus=1.0, kappa_plus=3.0, J=120)
    pde_studies = compute_error_rows(pde_problem, m_values=range(5, 10), methods=("standard", "corrected"))
    print_table("Semidiscrete severe benchmark: standard", pde_studies["standard"])
    print_table("Semidiscrete severe benchmark: corrected", pde_studies["corrected"])
    save_csv("pde_standard.csv", pde_studies["standard"])
    save_csv("pde_corrected.csv", pde_studies["corrected"])
    plot_convergence(pde_studies, "Semidiscrete parabolic benchmark", "figure_semidiscrete_pde.png")

    # 6) Sensitivity scan in theta
    plot_theta_scan(T=1.0, beta=1.0, h=2.0**-9, filename="figure_theta_sensitivity.png")

    # 7) Compact summary file
    summary_rows = []
    for label, studies in [
        ("benign", benign_studies),
        ("severe", severe_studies),
        ("aligned", aligned_studies),
        ("pde", pde_studies),
    ]:
        for method, rows in studies.items():
            summary_rows.append({
                "study": label,
                "method": method,
                "finest_h": rows[-1]["h"],
                "finest_Einf": rows[-1]["Einf"],
                "finest_Einf_EOC": rows[-1]["Einf_EOC"],
                "finest_E2": rows[-1]["E2"],
                "finest_E2_EOC": rows[-1]["E2_EOC"],
            })
    save_csv("summary_finest_results.csv", summary_rows)
    print_table("Summary on finest grids", summary_rows)

    print("\nDone.")
    print("Generated CSV tables and figures inside the output folder.")


if __name__ == "__main__":
    main()
