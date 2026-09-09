# Time Series Forecasting — Group 4
**Course:** Time Series Forecasting (2026/27 — 6th Edition)  
**Program:** Pós-graduação em Applied AI & Machine Learning  
**Institution:** ISEG Executive Education  
**Instructors:** Prof. Jorge Caiado, Rubens Dias  
**Group:** 4
**Members:** José Galão, Raquel Rocha, Vitor Antunes

---

## Quickstart: How to Run the Project

You can run this project on **both Linux / macOS and Windows**.  
The recommended runner is [**`uv`**](https://github.com/astral-sh/uv)—it automatically manages Python versions and dependencies inline (PEP 723) without needing manual virtual environments.

> **Dataset already included:** The processed time series files (`Dataset/kernel_daily_ts.csv` and `Dataset/kernel_weekly_ts.csv`) are already tracked in Git. You **do not** need to clone the ~50 GB Linux repository or extract raw commits to run models.

---

### Option A: Using `uv` (Recommended — Zero-Config)

#### 1. Install `uv`
* **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  # Or via winget:
  winget install --id=astral-sh.uv
  ```

#### 2. Run Modeling & Analysis Scripts Directly
`uv` automatically resolves all dependencies declared at the top of each script on the fly:
* **Linux / macOS:**
  ```bash
  uv run scripts/01_eda_decomposition.py
  ```
* **Windows (PowerShell / Command Prompt):**
  ```powershell
  uv run scripts\01_eda_decomposition.py
  ```

#### 3. (Optional) Run Fast Dataset Resampling
If you want to re-generate the daily and weekly CSVs from the local commit cache (~15s):
* **Linux / macOS:**
  ```bash
  uv run Dataset/Extractor/extractor.py
  ```
* **Windows:**
  ```powershell
  uv run Dataset\Extractor\extractor.py
  ```

---

### Option B: Using Standard Python `venv` + `pip`

If you prefer standard Python virtual environments without installing `uv`:

#### Linux / macOS
```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install pandas numpy statsmodels scipy matplotlib seaborn scikit-learn pmdarima

# 3. Run analysis scripts
python scripts/01_eda_decomposition.py
```

#### Windows (PowerShell / CMD)
```powershell
# 1. Create and activate a virtual environment (Python 3.10+)
python -m venv .venv

# PowerShell:
.venv\Scripts\Activate.ps1
# (If execution policy blocks scripts: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# Command Prompt (cmd.exe):
.venv\Scripts\activate.bat

# 2. Install dependencies
python -m pip install --upgrade pip
pip install pandas numpy statsmodels scipy matplotlib seaborn scikit-learn pmdarima

# 3. Run analysis scripts
python scripts\01_eda_decomposition.py
```

---

## Project Overview

This repository contains the group project codebase and documentation for the ISEG Time Series Forecasting course. The project explores end-to-end time series analysis, econometric modeling, and deep learning forecasting applied to continuous development velocity in large-scale open-source systems.

### Primary Dataset: Linux Kernel Commit Velocity
Rather than utilizing standard financial or macroeconomic toy datasets, this project benchmarks forecasting methods on the 20-year commit history (1.39M+ commits, April 2005 – September 2026) of the **Linux Kernel mainline repository**.

* **Dataset Field Descriptions & Guide:** See [**`Dataset/dataset_fields_description.md`**](Dataset/dataset_fields_description.md) for a plain-English, non-technical explanation of all 10 columns (commits, merges, unique authors, churn, net code growth) designed for collaborators and product managers.
* **Full Pipeline & Technical Architecture:** See [**`Dataset/Extractor/README.md`**](Dataset/Extractor/README.md) for architectural details, parallel extraction benchmarks, and academic syllabus mapping.
* **Extraction Script:** [`Dataset/Extractor/extractor.py`](Dataset/Extractor/extractor.py) — Multi-core parallel Map-Reduce script that extracts raw Git commits and resamples them into uniform Daily (`D`) and Weekly (`W-MON`) time series tables.

---

## Repository Structure

```
.
├── Dataset/                              # Processed time series ready for econometric modeling
│   ├── dataset_fields_description.md     # Non-technical guide to all columns and metrics
│   ├── kernel_daily_ts.csv               # Daily time series (D, s=7 micro-seasonality)
│   ├── kernel_weekly_ts.csv              # Weekly time series (W-MON, Mondays, s=52 macro-seasonality)
│   └── Extractor/                        # Parallel extraction pipeline & event data
│       ├── README.md                     # Pipeline documentation & mathematical formulation
│       ├── extractor.py                  # Parallel multi-core Git extraction script
│       └── kernel_raw_commits.csv        # Raw commit-level history (~1.39M rows, git-ignored)
├── scripts/                              # Modeling and empirical analysis scripts
│   └── 01_eda_decomposition.py           # Exploratory data analysis, additive/multiplicative decomposition
├── output/                               # Generated artifacts and reports
│   ├── figures/                          # Chronograms, decomposition plots, ACF/PACF graphs
│   ├── tables/                           # Descriptive statistics, outlier tables
│   └── EDA_SUMMARY.md                    # Detailed empirical summary of findings
├── Docs/                                 # Course syllabus & evaluation requirements
│   ├── PUC_TimeSeriesForecasting_2026_6Edio.pdf
│   └── Time Series Forecasting - PT.pdf
├── notes                                 # Exploration links & references
└── README.md                             # Project overview & quickstart
```
