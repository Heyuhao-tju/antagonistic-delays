# Antagonistic delays in large random ecosystems

Python code accompanying the manuscript:

**Antagonistic delays in large random ecosystems: self-delay destroys, interaction delay repairs**

This repository contains the numerical calculations and plotting scripts for the main-text and supplementary figures. The scripts examine how self-delay and interaction delay affect the local stability of large random ecological communities, using exponential, Gamma, and discrete delay kernels. Additional experiments consider correlated random interaction matrices with elliptic spectral support.

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── code/
    ├── Fig1_*.py
    ├── Fig2.framework.py
    ├── Fig3_*.py
    ├── Fig4_*.py
    ├── FigS1_*.py
    ├── FigS2.pigani.py
    ├── FigS3.*.py
    ├── FigS4.ratio_k.py
    ├── FigS5.Antagonistic_delay_elliptic.py
    ├── FigS6.Predator–prey-like.py
    ├── FigS7.Mixed_communities.py
    └── TableS1.rightmost_roots.py
```

## Software requirements

The reference environment is Python 3.10.11 with the following packages:

- NumPy 2.2.6
- SciPy 1.15.3
- Matplotlib 3.10.5
- threadpoolctl 3.6.0

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Most plotting scripts use Matplotlib's `TkAgg` backend and require Tk support. For execution without a graphical interface, change the backend to `Agg` before importing `matplotlib.pyplot` and omit `plt.show()`.

The figures use Arial where available.

## Running the scripts

Run the scripts from the `code` directory:

```bash
cd code
```

For example, reproduce Fig. 1(c) with:

```bash
python "Fig1_c.R(α,k).py"
```

Reproduce Fig. 4(c) with:

```bash
python "Fig4_c.lambda_cloud.py"
```

Print the numerical values reported in Table S1 with:

```bash
python "TableS1.rightmost_roots.py"
```

Keep quotation marks around filenames containing parentheses, Greek letters, or other special characters.

## Scripts and corresponding figures

| Figure or table | Script(s) | Calculation |
|---|---|---|
| Fig. 1(a) | `Fig1_a.R_self(μ).py` | Pure self-delay with an exponential kernel |
| Fig. 1(b) | `Fig1_b.R_inter(τ).py` | Interaction-delay repair at fixed self-delay |
| Fig. 1(c) | `Fig1_c.R(α,k).py` | Stability thresholds as total delay varies |
| Fig. 1(d) | `Fig1_d.k(α,0.7).py` | Stability regimes at a fixed self-delay fraction |
| Fig. 1(e) | `Fig1_e.phase_diagram.py` | Phase diagram for exponential kernels |
| Fig. 1(f–h) | `Fig1_f-h.lambda_cloud.py` | Characteristic-root scatter plots |
| Fig. 2 | `Fig2.framework.py` | Unified delay landscape and connections to earlier models |
| Fig. 3(a) | `Fig3_a.R_self(μ).py` | Comparison of self-delay effects across kernel shapes |
| Fig. 3(b) | `Fig3_b.R_inter(τ)_gamma_2.py` | Distributed interaction delay with discrete self-delay |
| Fig. 3(c) | `Fig3_c.R_inter(τ)_gamma_1.py` | Interaction-delay effects with matched kernel shapes |
| Fig. 3(d) | `Fig3_d.R(α,0.5)_gamma.py` | Combined delays at \(k=0.5\) |
| Fig. 4(a,b) | `Fig4_a-b.b&tilde_b_plane.py` | Critical curves and spectral disks for discrete delays |
| Fig. 4(c) | `Fig4_c.lambda_cloud.py` | Characteristic roots and theoretical contours for discrete delays |
| Fig. S1 | `FigS1_*.py` | Gamma-distributed kernels with shape parameter \(n=2\) |
| Fig. S2 | `FigS2.pigani.py` | Geometry of the discrete-delay stability boundary |
| Fig. S3 | `FigS3.ab.py`, `FigS3.cd.py` | Geometric illustration of the isometric mapping |
| Fig. S4 | `FigS4.ratio_k.py` | Discrete-delay spectra for different self-delay fractions |
| Fig. S5 | `FigS5.Antagonistic_delay_elliptic.py` | Delay antagonism in correlated random communities |
| Fig. S6 | `FigS6.Predator–prey-like.py` | Interaction-delay effects in a negatively correlated community |
| Fig. S7 | `FigS7.Mixed_communities.py` | Discrete interaction delays in a positively correlated community |
| Table S1 | `TableS1.rightmost_roots.py` | Maximum real parts of characteristic roots sampled on theoretical ellipse boundaries |

## Parameters and random matrices

Model parameters, delay values, and sampling settings are specified in each script. Use the default settings to reproduce the corresponding manuscript results. Parameter choices differ between figures.

Random interaction matrices are generated within the scripts using specified random seeds. Within each experiment, the same matrix is retained when comparing different delay configurations.

No external dataset is required.

## Output files

Output filenames are specified in the plotting scripts. Running the scripts from the `code` directory keeps outputs together: scripts using relative paths save to the working directory, while others save beside the script.

Scripts associated with individual panels generate the corresponding plots. Some manuscript figures require these outputs to be assembled into the final panel layout.

Phase-diagram scripts perform nested parameter scans and can take substantially longer than scripts generating individual curves.

## Additional options

The Fig. S5 script includes a numerical self-test:

```bash
python "FigS5.Antagonistic_delay_elliptic.py" --self-test
```

Its output directory can also be specified:

```bash
python "FigS5.Antagonistic_delay_elliptic.py" --output-dir "../figures/FigS5"
```

The scripts for Figs. S5–S7 save figures without opening an interactive window by default. Add `--show` to display them:

```bash
python "FigS6.Predator–prey-like.py" --show
```

These scripts reproduce the parameter configurations described in the manuscript. In particular, the discrete-delay solver in the Fig. S7 script is configured for interaction delays \(0\), \(\mu/2\), and \(\mu\).

## Citation

Please cite the associated manuscript when using these scripts or reproducing the reported results.
