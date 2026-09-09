# Linux Kernel Time Series Dataset Pipeline

## 1. Executive Summary & Motivation

This dataset pipeline transforms the entire revision history of the **Linux Kernel** (1.39+ million commits from April 2005 to present) into regular, equidistant time series for time series analysis and forecasting.

### Why the Linux Kernel?
* **High Real-World Relevance:** Rather than relying on generic textbook datasets (e.g., airline passenger counts, stock prices), this dataset captures the real-world engineering velocity of one of humanity's largest collaborative software projects.
* **Deterministic & Continuous:** Kernel development does not follow standard pull request / issue-tracker workflows; changes are integrated via email patchsets and Git trees by subsystem maintainers and Linus Torvalds. The commit graph represents true integration events over a 20+ year continuous horizon.
* **Rich Time Series Dynamics:**
  * **Dual Seasonality:** Micro-seasonality across the 7-day week (weekend troughs) and annual macro-seasonality (December/January holiday lulls and August summer dips).
  * **Structured Cycles:** The kernel adheres to a disciplined **~9–10 week release cycle** starting with a 2-week "merge window" (heavy commit volume) followed by weekly stabilization release candidates (rc1 through rc8).
  * **Univariate & Multivariate Readiness:** Supports univariate forecasting on commit counts ($Y_t$) and multivariate forecasting with exogenous covariates ($X_t$) such as developer count, churn volume, and release cycle indicators.

---

## 2. Theoretical Framing: From Event Log to Discrete Time Series

### The Point Process vs. Equidistant Time Series
A raw Git commit log is an **asynchronous event stream (temporal point process)**:
$$\{(t_1, e_1), (t_2, e_2), \dots, (t_n, e_n)\}$$
where events occur at arbitrary, irregular timestamps $t_i$.

Classical econometric models (such as **ARIMA, SARIMA, Holt-Winters Exponential Smoothing**) and recurrent neural architectures (**LSTM, GRU**) assume a uniform, discrete temporal grid:
$$Y_t, \quad t \in \{1, 2, \dots, T\}, \quad \Delta t = t_i - t_{i-1} = \text{constant}$$

The mathematical concepts taught in the course depend strictly on this uniform spacing:
1. **The Lag Operator ($B Y_t = Y_{t-1}$):** Lagging by 1 period is only meaningful when $\Delta t$ is fixed (e.g., exactly 1 day or 1 week).
2. **Autocorrelation (ACF) & Partial Autocorrelation (PACF):** Calculating covariance $\text{Cov}(Y_t, Y_{t-k})$ across lag $k$ requires that lag $k$ consistently represents the same physical elapsed duration.

The script `extractor.py` serves as the **uniform resampling and aggregation bridge** converting irregular Git events into discrete-time matrices.

---

## 3. High-Performance Parallel Architecture

Extracting churn and diff metrics across 1.39M+ commits sequentially on a single thread would take 40+ minutes due to the CPU cost of Git's internal tree diffing engine. `extractor.py` employs a **Dynamic Chunked Map-Reduce** architecture using Python's `multiprocessing` library:

```
[1. Discovery (~1.5s)]   git rev-list --all  ──> 1.39M commit hashes
                                                      │
[2. Dynamic Sharding]    Split into 10,000-commit micro-batches
                                                      │
[3. Multi-Core Map]      multiprocessing.Pool (N CPU cores)
                           Worker 1 ──> git log --no-walk=unsorted --stdin ──> chunk_00001.csv
                           Worker 2 ──> git log --no-walk=unsorted --stdin ──> chunk_00002.csv
                           Worker k ──> [Dynamic queue prevents stragglers] ──> chunk_00042.csv
                                                      │
[4. Reduce Phase]        Stream-merge chunk CSVs ──> Dataset/Extractor/kernel_raw_commits.csv
                                                      │
[5. Vectorized TS]       Single-pass Pandas resample ──> Dataset/kernel_daily_ts.csv
                                                     ──> Dataset/kernel_weekly_ts.csv
```

### Architectural Highlights
* **Instant Discovery via DAG Traversal:** `git rev-list --all` scans commit objects without unpacking trees or blobs, collecting 1.39M hashes in under 2 seconds.
* **Dynamic Load Balancing:** Commit diff complexity follows a heavy-tailed power law (a 2-line fix vs. a 50,000-line driver import). Fixed partitioning causes stragglers; micro-chunks (10,000 commits) paired with `pool.imap_unordered` ensure fast workers immediately pull new work, saturating all CPU cores.
* **Streaming `--no-walk=unsorted`:** Bypasses Git's internal in-memory sorting and buffers, streaming diff stats straight into per-worker chunk files without inter-process communication (IPC) overhead or Python GIL contention.
* **Directory Layout Decoupling:** Raw commit telemetry stays isolated in `Dataset/Extractor/`, while the resampled time series ready for modeling are written directly to the parent `Dataset/` directory.
* **Smart Cache:** If `kernel_raw_commits.csv` is already present on disk, the script skips Git extraction entirely and immediately generates the resampled tables in ~2 seconds.

---

## 4. Setup & Execution Guide

### Step 1: Clone the Kernel (Bare Repository)
The working tree of the Linux kernel is over 50 GB. To analyze history and churn, you only need Git database objects. A **bare clone** downloads only ~4.5 GB and saves ~45 GB of disk space:

```bash
git clone --bare https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git /home/cybervitor/.local/gits/linux
```

*(Note: Regular clones with checked-out trees are also fully supported).* 

### Step 2: Run the Extractor
The script uses standard PEP 723 metadata, so `uv` automatically handles dependencies:

```bash
# Run from workspace root:
uv run Dataset/Extractor/extractor.py

# Or run from the Extractor directory:
cd Dataset/Extractor
uv run extractor.py

# Force full re-extraction from Git from scratch:
uv run extractor.py --force-extract

# Explicitly configure worker process count or custom repo path:
uv run extractor.py /path/to/another/repo.git --workers 8 --force-extract
```

---

## 5. Output Data Schema & Directory Layout

```
Dataset/
├── kernel_daily_ts.csv        # Resampled Daily time series (s=7 seasonality)
├── kernel_weekly_ts.csv       # Resampled Weekly time series (Mondays)
└── Extractor/
    ├── README.md              # Pipeline documentation
    ├── extractor.py           # Parallel multi-core extraction script
    └── kernel_raw_commits.csv # Granular commit-level history (~1.4M rows)
```

### File 1: `Dataset/Extractor/kernel_raw_commits.csv`
Granular, event-level table containing all ~1.4M commits.

| Column | Type | Description |
| :--- | :--- | :--- |
| `commit_hash` | string | Full 40-character Git SHA-1 commit hash. |
| `author_date` | ISO-8601 | Original timestamp when the patch was authored. |
| `committer_date` | ISO-8601 | Timestamp when merged/applied into the upstream tree (canonical timeline). |
| `author_email` | string | Contributor email address (used for distinct entity tracking). |
| `is_merge` | integer | `1` if the commit has multiple parents (merge commit), `0` otherwise. |
| `files_changed` | integer | Number of files modified by this commit. |
| `insertions` | integer | Number of lines added (`+`). |
| `deletions` | integer | Number of lines removed (`-`). |

---

### Files 2 & 3: `Dataset/kernel_daily_ts.csv` & `Dataset/kernel_weekly_ts.csv`
Equidistant time series tables resampled to **Daily (`D`)** and **Weekly on Mondays (`W-MON`)** located in the `Dataset/` directory.

| Column | Role | Formula / Aggregation | Description |
| :--- | :--- | :--- | :--- |
| **`committer_date`** | Index | Equidistant timestamp (UTC) | Date of observation (`YYYY-MM-DD`). |
| **`total_commits`** | **Target ($Y_t$)** | $\sum 1$ (Count) | Total commits integrated in the window. |
| **`non_merge_commits`**| Target / Feature | $\sum (\text{is\_merge} == 0)$ | Direct patch commits (excluding merges). |
| **`merge_commits`** | Target / Feature | $\sum (\text{is\_merge} == 1)$ | Subsystem integration merge commits. |
| **`unique_authors`** | **Exogenous ($X_{1,t}$)**| Count distinct email | Number of active contributors in the window. |
| **`files_changed`** | Exogenous ($X_{2,t}$)| $\sum \text{files\_changed}$ | Total files touched across all commits. |
| **`insertions`** | Exogenous ($X_{3,t}$)| $\sum \text{insertions}$ | Total lines added. |
| **`deletions`** | Exogenous ($X_{4,t}$)| $\sum \text{deletions}$ | Total lines deleted. |
| **`total_churn`** | Exogenous ($X_{5,t}$)| $\text{insertions} + \text{deletions}$ | Gross code churn (total line volatility). |
| **`net_code_growth`**| Exogenous ($X_{6,t}$)| $\text{insertions} - \text{deletions}$ | Net expansion of the kernel codebase. |

---

## 6. Academic & Syllabus Mapping (ISEG Course Structure)

This dataset directly supports all phases of Professor Jorge Caiado's curriculum:

| Syllabus Block | Topic | Recommended Series & Application |
| :--- | :--- | :--- |
| **Aula #0** (Slides 1–10) | **Time Series Decomposition** | Apply **Additive** ($Y_t = T_t + S_t + R_t$) and **Multiplicative** ($Y_t = T_t \times S_t \times R_t$) decomposition to `kernel_weekly_ts.csv` to isolate long-term growth trend from holiday seasonality. |
| **Aula #1** (Slides 11–33) | **Exponential Smoothing** | Fit **Simple Exponential Smoothing**, **Holt's Linear Trend**, and **Holt-Winters** with additive/multiplicative seasonality. Evaluate out-of-sample forecast errors (RMSE, MAE, MAPE). |
| **Aula #2 & #3** (Slides 34–66)| **ARIMA & SARIMA Modeling** | Conduct Augmented Dickey-Fuller (ADF) unit root testing, first-differencing $\nabla Y_t = (1-B)Y_t$, ACF/PACF identification, Box-Jenkins estimation, and Ljung-Box residual diagnostics. |
| **Aula #4** (Slides 67–78) | **ARIMAX / SARIMAX with Exogenous Variables** | Model commit velocity $Y_t$ conditioned on exogenous signals $X_t$ (active developer count `unique_authors` and code churn `total_churn`). Compare AIC/BIC and MSE against pure univariate SARIMA. |
| **Aula #5** (Slides 79–111)| **Deep Learning (1D-CNN, RNN, LSTM, GRU)** | Utilize `kernel_daily_ts.csv` (~7,000 observations) to train sequence-to-one sliding window neural networks, testing both univariate architectures and multivariate feature inputs. |
