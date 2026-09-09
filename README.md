# Time Series Forecasting — Group 4
**Course:** Time Series Forecasting (2026/27 — 6th Edition)  
**Program:** Pós-graduação em Applied AI & Machine Learning  
**Institution:** ISEG Executive Education  
**Instructors:** Prof. Jorge Caiado, Rubens Dias  

---

## Project Overview

This repository contains the group project codebase and documentation for the ISEG Time Series Forecasting course. The project explores end-to-end time series analysis, econometric modeling, and deep learning forecasting applied to continuous development velocity in large-scale open-source systems.

### Primary Dataset: Linux Kernel Commit Velocity
Rather than utilizing standard financial or macroeconomic toy datasets, this project benchmarks forecasting methods on the 20-year commit history (1.39M+ commits) of the **Linux Kernel mainline repository**.

* **Full Pipeline & Dataset Documentation:** See [**`Dataset/Extractor/README.md`**](Dataset/Extractor/README.md) for architectural details, mathematical motivation, column dictionaries, and syllabus alignment.
* **Extraction Script:** [`Dataset/Extractor/extractor.py`](Dataset/Extractor/extractor.py) — Multi-core parallel Map-Reduce script that extracts raw Git commits and resamples them into uniform Daily (`D`) and Weekly (`W-MON`) time series tables.

---

## Quickstart

### 1. Requirements
* Python 3.10+
* [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip` with `pandas`.

### 2. Generate Dataset
```bash
# Run the parallel extractor from root (automatically uses available CPU cores)
uv run Dataset/Extractor/extractor.py

# Force re-extraction from Git from scratch
uv run Dataset/Extractor/extractor.py --force-extract
```

---

## Repository Structure

```
.
├── Dataset/                   # Resampled time series ready for econometric modeling
│   ├── kernel_daily_ts.csv    # Resampled Daily time series (s=7 seasonality)
│   ├── kernel_weekly_ts.csv   # Resampled Weekly time series (Mondays)
│   └── Extractor/             # Pipeline tooling & event-level data
│       ├── README.md          # Full dataset pipeline & mathematical documentation
│       ├── extractor.py       # High-performance parallel Git extraction script
│       └── kernel_raw_commits.csv # Granular commit-level history (~1.4M rows)
├── Docs/                      # Course syllabus & evaluation requirements
│   ├── PUC_TimeSeriesForecasting_2026_6Edio.pdf
│   └── Time Series Forecasting - PT.pdf
├── notes                      # Exploration links & references
└── README.md                  # Project overview
```
