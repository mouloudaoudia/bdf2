# BDF2 temporal-interface supplementary package

Author and maintainer: Mouloud Aoudia

This package contains the Python scripts, numerical output tables, and figure files for the article
**Defect-Targeted Correction of BDF2 at Isolated Temporal Interfaces: A Local Analysis**.

## Contents

- `src/bdf2_temporal_interface.py`: scalar and semidiscrete validation suite.
- `src/run_fixed_theta_benign_scalar.py`: fixed-interface-fraction benign scalar convergence check.
- `src/run_additional_semidiscrete_benchmark.py`: benign semidiscrete benchmark and timing diagnostics.
- `src/run_semidiscrete_dimension_scaling.py`: dimension-scaling check for the benign semidiscrete benchmark.
- `src/run_unaligned_restart_and_jump_sensitivity.py`: genuine unaligned-crossing restart diagnostic and jump-data sensitivity diagnostics.
- `src/figure_1_concept.py`: temporal-interface geometry figure.
- `src/figure_2_mechanisms.py`: standard, restart, and correction comparison figure.
- `outputs/`: reference CSV outputs, plain-text tables, and numerical figures from completed runs.
- `figures/`: high-resolution conceptual figures.

## Requirements

Python 3.9 or newer is required. Install the required packages from the package root with:

```bash
python -m pip install -r requirements.txt
```

The scripts use NumPy, SciPy, and Matplotlib.

## Complete workflow

From the package root, run:

```bash
python run_all.py
```

On Windows, the same workflow can be started by double-clicking `run_all.bat`.
On Linux or macOS, `run_all.sh` provides the same workflow.

The complete workflow writes newly computed files to:

```text
outputs
```

The stored reference files are already available in the same folder.

## Individual commands

The scalar and semidiscrete validation suite can be run with:

```bash
python src/bdf2_temporal_interface.py
```

The fixed-interface-fraction benign scalar check can be run with:

```bash
python src/run_fixed_theta_benign_scalar.py
```

The benign semidiscrete benchmark and timing diagnostics can be run with:

```bash
python src/run_additional_semidiscrete_benchmark.py --task all --J 160 --m-min 5 --m-max 9 --timing-N 64,128,256 --repeats 5
```

The dimension-scaling check can be run with:

```bash
python src/run_semidiscrete_dimension_scaling.py --J-values 160,640,2560 --N 128 --repeats 5
```

The genuine unaligned-crossing restart diagnostic and jump-data sensitivity diagnostics can be run with:

```bash
python src/run_unaligned_restart_and_jump_sensitivity.py
```

The conceptual figures can be regenerated with:

```bash
python src/figure_1_concept.py
python src/figure_2_mechanisms.py
```

## Output files from the additional diagnostics

The additional diagnostics write:

```text
outputs/genuine_unaligned_restart_diagnostic.csv
outputs/genuine_unaligned_restart_diagnostic_table.txt
outputs/figure_genuine_unaligned_restart_diagnostic.png
outputs/figure_genuine_unaligned_crossing_residuals.png
outputs/jump_data_sensitivity_benign_scalar.csv
outputs/jump_data_sensitivity_benign_scalar_table.txt
outputs/figure_jump_data_sensitivity_benign_scalar.png
outputs/jump_data_sensitivity_benign_semidiscrete.csv
outputs/jump_data_sensitivity_benign_semidiscrete_table.txt
outputs/figure_jump_data_sensitivity_benign_semidiscrete.png
outputs/additional_diagnostics_manifest.txt
```

These diagnostics supplement the original convergence and cost tests. The unaligned-crossing restart diagnostic verifies that post-crossing restart leaves the first crossing residual unchanged under the convention studied in the article, while direct correction acts on that crossing residual. The jump-data sensitivity diagnostics measure the effect of perturbing the supplied jump values.

## Timing diagnostics

Timing values depend on the processor, operating system, Python version, and installed BLAS/LAPACK libraries. Error tables and convergence figures are deterministic for fixed input parameters; runtime ratios should be treated as machine-dependent diagnostics.

## Figure quality

The conceptual figures are stored at 600 dpi. The numerical figures in `outputs/` are regenerated at high pixel resolution suitable for manuscript submission and reproduction.
