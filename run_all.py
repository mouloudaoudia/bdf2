#!/usr/bin/env python3
"""Run the complete numerical reproduction workflow for the supplementary package."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

COMMANDS = [
    [sys.executable, str(SRC / "bdf2_temporal_interface.py")],
    [sys.executable, str(SRC / "run_fixed_theta_benign_scalar.py")],
    [
        sys.executable,
        str(SRC / "run_additional_semidiscrete_benchmark.py"),
        "--task", "all",
        "--J", "160",
        "--m-min", "5",
        "--m-max", "9",
        "--timing-N", "64,128,256",
        "--repeats", "5",
    ],
    [
        sys.executable,
        str(SRC / "run_semidiscrete_dimension_scaling.py"),
        "--J-values", "160,640,2560",
        "--N", "128",
        "--repeats", "5",
    ],
    [sys.executable, str(SRC / "run_unaligned_restart_and_jump_sensitivity.py")],
    [sys.executable, str(SRC / "figure_1_concept.py")],
    [sys.executable, str(SRC / "figure_2_mechanisms.py")],
]


def main() -> None:
    for command in COMMANDS:
        print("\n" + "=" * 80)
        print("Running:", " ".join(command))
        print("=" * 80)
        subprocess.run(command, cwd=str(ROOT), check=True)

    print("\nAll computations completed.")
    print("Numerical outputs are in the outputs folder.")


if __name__ == "__main__":
    main()
