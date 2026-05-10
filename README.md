# BDF2 temporal-interface reproducibility package

This repository reproduces the numerical tables and figures for the study
**Enhancing BDF2 at Isolated Temporal Interfaces via Defect-Targeted Correction**.

## Contents

- `src/bdf2_temporal_interface.py`: main numerical validation suite.
- `src/run_additional_semidiscrete_benchmark.py`: benign semidiscrete parabolic benchmark and timing diagnostics.
- `src/figure_1_concept.py`: script for the opening conceptual figure.
- `src/figure_2_mechanisms.py`: script for the mechanism-comparison figure.
- `outputs/`: reference CSV outputs and numerical figures from a completed run.
- `figures/`: high-resolution conceptual figures.

## Python requirements

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The scripts require Python 3.9 or newer and use NumPy, SciPy, and Matplotlib.

## Reproducing the numerical outputs

From the repository root, run:

```bash
cd src
python bdf2_temporal_interface.py
python run_additional_semidiscrete_benchmark.py --task all --J 160 --m-min 5 --m-max 9 --timing-N 64,128,256 --repeats 5
```

The scripts write CSV tables and PNG figures to `src/bdf2_temporal_interface_outputs/`.
To compare with the stored reference outputs, compare that folder with `outputs/`.

## Reproducing the conceptual figures

From the repository root, run:

```bash
python src/figure_1_concept.py
python src/figure_2_mechanisms.py
```

These scripts write high-resolution PNG and PDF figures.  The stored copies are in `figures/`.

## Notes on timing diagnostics

Timing values depend on the processor, operating system, Python version, and installed BLAS/LAPACK libraries.  The error tables and convergence figures are deterministic for fixed input parameters; timing ratios should be interpreted as machine-dependent diagnostics.
