# Host–microbe evolutionary rescue: figure-generation code

This repository contains the Python implementation of the host–microbe evolutionary rescue model and the scripts used to generate **Figures 1–4 and Supplementary Figures S1–S3** of the accompanying manuscript.

The calculations examine how microbial effects on host fitness, non-additive host–microbe interactions, and microbial inheritance loss influence host adaptation, demographic rescue, and host–microbe association. The figure scripts implement both the full dynamical model and its first-order quasi-microbial-equilibrium (QME) approximation.

## Repository structure

```text
.
├── README.md
├── scripts/
│   ├── qme_rescue_model.py
│   └── make_pnas_figures.py
└── figures/                 # Created automatically when figures are generated
```

- **`scripts/qme_rescue_model.py`** defines the model parameters, full frequency and demographic equations, first-order QME expressions, numerical ODE solvers, and utilities for checking physical states and comparing model predictions.
- **`scripts/make_pnas_figures.py`** sets the figure-specific parameter values, runs trajectories and parameter sweeps, performs robustness comparisons, and creates the publication figures.

Both Python files must be in the same `scripts/` directory. The plotting script determines the output directory relative to its own location, not the shell's current working directory.

## Requirements

- Python **3.10 or later**
- NumPy
- SciPy
- Matplotlib

To install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
```

## Reproduce the figures

From the repository root, run:

```bash
python scripts/make_pnas_figures.py
```

This generates PDF and PNG versions of all seven figures in `figures/`. The PDF outputs are suitable for vector-based manuscript figures; the PNG outputs are saved at 400 dpi.

| Manuscript figure | Output filename (PDF and PNG) | Content |
| --- | --- | --- |
| Figure 1 | `fig1_framework_qme` | Four host–microbe classes, fast-variable relaxation, and the QME manifold |
| Figure 2 | `fig2_rescue_mechanisms` | Microbial buffering, non-additive acceleration, and rescue requiring both contributions |
| Figure 3 | `fig3_rescue_landscape` | Rescue bottleneck depth and time below a quasi-extinction threshold across microbial-effect parameters |
| Figure 4 | `fig4_selection_amplified_dilution` | How selection-amplified microbial loss changes prevalence, association, and demographic rescue |
| Figure S1 | `figS1_rescue_validation` | Full-model versus QME trajectories for the rescue scenarios |
| Figure S2 | `figS2_sad_validation` | Full-model versus QME trajectories under different microbial-loss strengths |
| Figure S3 | `figS3_qme_qualitative_robustness` | Sobol-sampled and parameter-grid comparisons of qualitative QME predictions with the full system |

For example, the first figure is saved as `figures/fig1_framework_qme.pdf` and `figures/fig1_framework_qme.png`.

**Runtime:** Figures 3 and S3 involve numerical parameter sweeps and can take appreciably longer than the individual trajectory plots. The full command regenerates every figure; no precomputed simulation data are required.

## Model and numerical details

The model uses three frequency variables: microbial prevalence (`p_m`), rescue-allele frequency (`p_g`), and host–microbe association (`D = p_gm − p_g p_m`), together with host abundance (`N`). Fitness components include host selection (`s_g`), microbial selection (`s_m`), and a non-additive interaction (`e`). Microbial acquisition, loss, and inheritance-related loss are parameterized by `beta`, `ell0`, and `phi`; demographic dynamics include environmental decline (`r`) and density regulation (`c`).

The shared model implementation is in `qme_rescue_model.py`. The plotting script calls that implementation rather than independently defining the model equations. ODE trajectories use SciPy's `solve_ivp` with the LSODA method. Figure 3 uses a vectorized fourth-order Runge–Kutta grid calculation. Figure S3 uses a scrambled Sobol sequence with a fixed seed (`20260728`); its default sample size is 1,024 parameter combinations.

Figure-specific parameter choices and numerical resolutions are documented in `make_pnas_figures.py`. The S1–S3 figures provide comparisons between the full dynamics and the QME approximation, including qualitative robustness outside the weakest-selection regime. Small differences in appearance or numerical output may arise across library versions and installed fonts.

## Reproducibility and scope

The scripts generate theoretical and numerical results; they do not require external datasets or network access. The figures depend on the model definitions, parameter values, solver settings, and plotting code in these two files. For the precise interpretation of each figure and the assumptions underlying the QME approximation, consult the accompanying manuscript and Supplementary Information.

## AI-assisted code development

Anthropic's Claude Opus was used to assist in developing the figure-generation code. Responsibility for the model, computational results, code review, and final figures rests with the authors.
