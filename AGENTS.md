# AGENTS.md — Repository Context & Instructions

## 1. Project Context & Deliverables
* **Course:** Time Series Forecasting (2026/27, 6th Edition) — ISEG Executive Education (PG in Applied AI & ML).
* **Core Problem:** Econometric and deep learning time series modeling of continuous software development velocity across the 20-year commit history (1.39M+ commits, April 2005 – September 2026) of the **Linux Kernel mainline repository**.
* **Deliverable Requirements:** 15–20 page technical report comparing $\ge 2$ forecasting families (e.g., Classical Econometrics vs Deep Learning Sequence Models) + oral defense.

---

## 2. Critical Operational Gotchas

### ⚠️ GitHub 100 MB Limit (`kernel_raw_commits.csv`)
* `Dataset/Extractor/kernel_raw_commits.csv` is **~175 MB**. It is git-ignored and **must never be committed to Git**.
* The resampled series `Dataset/kernel_daily_ts.csv` (~366 KB) and `Dataset/kernel_weekly_ts.csv` (~63 KB) are tracked in Git.
* If accidentally staged, unstage immediately:
  ```bash
  git restore --staged Dataset/Extractor/kernel_raw_commits.csv
  ```

### ⚠️ Boundary Anomalies (Tail & Head Cleaning)
* **Tail cutoff (drop last row):** Extraction ended mid-week on `2026-09-08` / `2026-09-14` with only 2 commits. You **must drop the incomplete final row** (`df.iloc[:-1]`) before fitting models or splitting data to prevent artificial forecast collapse.
* **Head outlier (churn spike):** The first observation (`2005-04-18` weekly / `2005-04-16` daily) is Linus Torvalds' initial Git import, adding 6.75M lines in a single commit. While commit counts are valid, code churn (`insertions`, `total_churn`, `net_code_growth`) has extreme leverage in this initial bucket.

### ⚠️ Explicit Frequency for `statsmodels`
* `pd.read_csv` does not infer or persist datetime frequencies.
* Always explicitly assign `df.index.freq = 'W-MON'` (weekly) or `df.index.freq = 'D'` (daily) right after indexing. Without this, `SARIMAX` and `ExponentialSmoothing` raise warnings or fail to calculate out-of-sample forecast steps.

```python
# Standard loading template
import pandas as pd

# Weekly
df_w = pd.read_csv("Dataset/kernel_weekly_ts.csv", parse_dates=["committer_date"], index_col="committer_date").iloc[:-1]
df_w.index.freq = "W-MON"

# Daily
df_d = pd.read_csv("Dataset/kernel_daily_ts.csv", parse_dates=["committer_date"], index_col="committer_date").iloc[:-1]
df_d.index.freq = "D"
```

---

## 3. Tooling & Execution Commands

Run all Python workflows with [`uv`](https://github.com/astral-sh/uv). No persistent virtual environment is required.

### Modeling & Analysis Scripts
* **PEP 723 inline dependencies (preferred):** Add metadata headers to new scripts:
  ```python
  # /// script
  # dependencies = ["pandas", "statsmodels", "matplotlib", "scikit-learn"]
  # ///
  ```
  Execute directly with:
  ```bash
  uv run <script_path.py>
  ```
* **Ad-hoc one-liners or interactive scripts:**
  ```bash
  uv run --with "pandas,numpy,statsmodels,scipy,matplotlib,seaborn,pmdarima,scikit-learn" python <script_path.py>
  ```
* **Deep Learning scripts:**
  ```bash
  uv run --with "torch,pandas,numpy,matplotlib,scikit-learn" python <script_path.py>
  ```

### Extractor Pipeline (`Dataset/Extractor/extractor.py`)
* The pre-extracted dataset is already available; **do not clone the ~50 GB Linux Git repo** for modeling tasks.
* Fast resample regeneration from existing `kernel_raw_commits.csv` (~15s):
  ```bash
  uv run Dataset/Extractor/extractor.py
  ```
* Full Git extraction (only needed when regenerating from a local bare clone):
  ```bash
  uv run Dataset/Extractor/extractor.py /path/to/linux-bare.git --force-extract --workers 8
  ```

---

## 4. Dataset Schemas & Modeling Conventions

### Files & Granularity
* **`Dataset/kernel_weekly_ts.csv`** (`W-MON`, 1,117 clean observations): Mondays. Primary series for classical econometrics:
  * Decomposition (Additive & Multiplicative, trend vs holiday/summer seasonality)
  * Holt-Winters Exponential Smoothing ($s=52$)
  * ARIMA / SARIMA
  * ARIMAX / SARIMAX with exogenous covariates
* **`Dataset/kernel_daily_ts.csv`** (`D`, 7,815 clean observations): Daily series with day-of-week micro-seasonality ($s=7$). Primary series for Deep Learning sliding-window sequence models (1D-CNN, RNN, LSTM, GRU).
* **`Dataset/Extractor/kernel_raw_commits.csv`**: Raw event log (~1.39M rows, git-ignored).

### Variables & Roles
* **Targets ($Y_t$):** `total_commits` (gross activity) or `non_merge_commits` (direct engineering work, excluding merge commits).
* **Exogenous Features ($X_t$):**
  * `unique_authors` ($X_{1,t}$): Distinct active contributors (team capacity proxy).
  * `total_churn` ($X_{2,t}$): `insertions + deletions` (code volatility proxy).
  * `net_code_growth` ($X_{3,t}$): `insertions - deletions` (net codebase expansion).
  * `files_changed` ($X_{4,t}$): Breadth of code modifications.

---

## 5. Methodological & Evaluation Standards (ISEG Curriculum)

1. **Chronological Splitting:**
   * Always split chronologically (e.g., 80/20 train/test). **Never shuffle or randomly partition time series data**.
2. **Stationarity & Differencing:**
   * Test levels first with Augmented Dickey-Fuller (`adfuller`).
   * For seasonal data, apply seasonal differencing $\nabla_s Y_t = (1 - B^s)Y_t$ before simple differencing $\nabla Y_t = (1 - B)Y_t$.
3. **Residual Diagnostics (Mandatory):**
   * Residuals must be white noise: verify absence of autocorrelation using Ljung-Box test (`acorr_ljungbox`, require $p\text{-value} > 0.05$).
   * Evaluate normality with Jarque-Bera and histogram / Q-Q plots.
4. **Model Selection & Evaluation Metrics:**
   * In-sample selection: Minimum AIC / BIC.
   * Out-of-sample evaluation: Compare static (one-step ahead) and dynamic (multi-step ahead recursive) forecasts on test data using:
     * **RMSE / REQM:** $\sqrt{\frac{1}{m}\sum (Y_t - \hat{Y}_t)^2}$
     * **MAE / EAM:** $\frac{1}{m}\sum |Y_t - \hat{Y}_t|$
     * **MAPE / EPAM:** $\frac{1}{m}\sum \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right| \times 100\%$
     * **sMAPE / EPAMs:** $\frac{1}{m}\sum \frac{|Y_t - \hat{Y}_t|}{|Y_t| + |\hat{Y}_t|} \times 200\%$
