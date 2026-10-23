# Time Series Forecasting — Group 4
**Course:** Time Series Forecasting (2026/27 — 6th Edition)  
**Program:** Pós-graduação em Applied AI & Machine Learning  
**Institution:** ISEG Executive Education  
**Instructors:** Prof. Jorge Caiado, Rubens Dias  
**Group:** 4  
**Members:** José Galão, Raquel Rocha, Vitor Antunes

---

## What this project does

The group forecasts **Linux Kernel commit velocity** from 1.39 million commits
(April 2005 – September 2026) and compares two forecasting families on the same chronological
split:

* **Classical econometrics** — additive and multiplicative Holt-Winters, ARIMA, SARIMAX, and an
  ARIMAX with exogenous covariates (authors, code churn) plus the release-cycle phase.
* **Deep learning** — 1D-CNN, SimpleRNN, LSTM and GRU, each in three feature regimes
  (univariate, multivariate, multivariate + release phase).

The deliverable is a 15–20 page report plus a defence. Everything under `output/` is the
evidence for that report and is tracked in Git.

---

## Quickstart

You can run this on **Linux, macOS or Windows**. The recommended runner is
[`uv`](https://github.com/astral-sh/uv): it reads the dependencies declared inline in each script
(PEP 723) and needs no manual virtual environment.

> **The processed data is already in the repository.**
> `dataset/kernel_daily_ts.csv` and `dataset/kernel_weekly_ts.csv` are tracked in Git. You do
> **not** need the multi-gigabyte Linux clone or the raw commit extract to run any model.

### 1. (Optional) Rebuild the dataset

Only needed if you change the extractor — if you just want to run the models, skip to the next
step. The rebuild takes about 3 seconds when the local chunk cache is present, because no Git walk
is required:

```bash
uv run dataset/Extractor/extractor.py
```

Re-walking the Linux history requires a local clone (the path is printed in the banner) and is
much slower:

```bash
uv run dataset/Extractor/extractor.py --force-tags     # re-walk refs/tags
uv run dataset/Extractor/extractor.py --force-extract  # re-walk 1.39M commits
```

Both commands need `uv`, which the next step installs.

### 2. Install `uv`

* **Linux / macOS**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows (PowerShell)**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  # or via winget:
  winget install --id=astral-sh.uv
  ```

### 3. Run the pipeline

**Run from the repository root and in order** — script 04 only reads tables written by 02 and 03,
so it fails on a partial run. The scripts rely on flat imports (`from utils import ...`), which
resolve only because the script's own directory goes on `sys.path`; running them as
`python -m scripts.01_...` or from another directory breaks them.

* **Linux / macOS**
  ```bash
  uv run scripts/01_eda_decomposition.py      # seconds
  uv run scripts/02_classical_econometrics.py # a couple of minutes
  uv run scripts/03_deep_learning.py          # slow step: 12 nets trained on CPU
  uv run scripts/04_benchmark_consolidado.py  # instant
  ```
* **Windows (PowerShell / Command Prompt)**
  ```powershell
  uv run scripts\01_eda_decomposition.py
  uv run scripts\02_classical_econometrics.py
  uv run scripts\03_deep_learning.py
  uv run scripts\04_benchmark_consolidado.py
  ```

Script 03 is deliberately pinned to CPU (`device = torch.device("cpu")`): the local GPU
(NVIDIA MX150, compute capability 6.1) has no kernels in current PyTorch builds. It also fixes
`torch.manual_seed(42)` per feature regime, so the run is reproducible.

### Alternative to steps 2 and 3: without `uv` (`venv` + `pip`)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy statsmodels scipy matplotlib seaborn scikit-learn torch
python scripts/01_eda_decomposition.py
# ... and so on, in order
```

Python 3.10 or newer is required.

---

## Pipeline and artifacts

| Script | Purpose |
|---|---|
| `01_eda_decomposition.py` | Descriptive statistics, outliers, additive/multiplicative decomposition, ACF/PACF |
| `02_classical_econometrics.py` | ADF tests, correlograms, Holt-Winters, ARIMA/SARIMAX/ARIMAX, residual diagnostics |
| `03_deep_learning.py` | Sliding windows (W = 28 days): 4 architectures across 3 feature regimes |
| `04_benchmark_consolidado.py` | Joins both families into one ranking and one comparison chart |

Artifacts each script writes:

| Script | Figures | Tables | Summary |
|---|---|---|---|
| `01` | `fig_00_*`, `fig_01`, `fig_02` | `tab_00_*`–`tab_02` | `EDA_SUMMARY.md` |
| `02` | `fig_03`, `fig_04`, `fig_05`, `fig_05b` | `tab_03`–`tab_05`, `tab_05b` | `ECONOMETRICS_SUMMARY.md` |
| `03` | `fig_06`, `fig_07` | `tab_06`, `tab_07` | `DEEP_LEARNING_SUMMARY.md` |
| `04` | `fig_08` | `tab_08` | — |

Support tables without a number (`deep_learning.csv`, `curvas_aprendizagem.csv`,
`alisamento_exponencial.csv`, `arima_arimax.csv`, `diagnostico_residuos.csv`) hold the detail
behind those figures.

---

## Repository structure

```
.
├── dataset/
│   ├── dataset_fields_description.md   # Plain-language guide to all 15 columns
│   ├── kernel_daily_ts.csv             # Daily series (D, s = 7 micro-seasonality)
│   ├── kernel_weekly_ts.csv            # Weekly series (W-MON, s = 52 macro-seasonality)
│   └── Extractor/
│       ├── README.md                   # Pipeline documentation and formulation
│       ├── extractor.py                # Parallel git extraction and resampling
│       └── kernel_raw_commits.csv      # Raw commit history (~183 MB, git-ignored)
├── scripts/
│   ├── 01_eda_decomposition.py         # EDA, outliers, decomposition, correlograms
│   ├── 02_classical_econometrics.py    # Holt-Winters, ARIMA/SARIMAX/ARIMAX
│   ├── 03_deep_learning.py             # 1D-CNN, SimpleRNN, LSTM, GRU
│   ├── 04_benchmark_consolidado.py     # Master ranking across both families
│   ├── utils.py                        # Shared paths, loaders, metrics, plotting
│   ├── arquitecturas_neuronais/        # The four torch model classes
│   ├── templates/                      # Markdown templates for the generated summaries
│   └── analise_covariaveis_release/    # Reproducible experiment behind the release-phase variables
├── output/
│   ├── images/png/                     # Figures at 300 DPI
│   ├── images/svg/                     # Vector versions for the report
│   ├── tables/                         # Every CSV quoted in the report
│   ├── EDA_SUMMARY.md                  # Generated — edit scripts/templates/, not this file
│   ├── ECONOMETRICS_SUMMARY.md         # Generated
│   └── DEEP_LEARNING_SUMMARY.md        # Generated
├── docs/                               # Course syllabus and evaluation requirements (PDF)
├── class-code/                         # Instructors' reference implementation (not part of the pipeline)
├── AGENTS.md                           # Working notes for contributors and AI agents
├── GUIA_RELATORIO_E_DIVISAO_TRABALHO.md  # Report structure, artifact mapping, work split
├── CONCEITOS_E_GLOSSARIO.md            # Method glossary (decomposition, ARIMA, DL architectures)
└── README.md
```

---

## Key documents

* **`dataset/dataset_fields_description.md`** — non-technical explanation of all 15 columns,
  including the release-calendar semantics (why the clock is measured at the *start* of each
  period, so the week a release lands still reports the age of the previous cycle).
* **`dataset/Extractor/README.md`** — extraction architecture, parallel benchmarks, and syllabus
  mapping.
* **`scripts/analise_covariaveis_release/README.md`** — why the release-cycle phase is in the
  model: the like-for-like comparison, the encodings tested and rejected, and the limitations
  (the phase is the only predetermined covariate; the global gain is a nowcast).
* **`output/*_SUMMARY.md`** — generated interpretations for each report chapter. These are
  **generated**: edit `scripts/templates/*.md` and re-run the corresponding script.
