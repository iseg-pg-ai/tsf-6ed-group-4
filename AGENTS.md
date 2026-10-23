# AGENTS.md — Repository Context & Instructions

## 1. What this repo is

* ISEG Time Series Forecasting (2026/27, 6th Ed.), group 4: forecast **Linux Kernel commit velocity** over 1.39M commits (2005-04 to 2026-09) and compare at least two forecasting families (classical econometrics vs deep learning).
* Deliverable is a 15-20 page report plus defense. Everything under `output/` (figures, tables, `*_SUMMARY.md`) **is** the report's evidence, and is tracked in Git.

## 2. Running things

* All code uses `uv` with PEP 723 inline dependencies. No `pyproject.toml`, no lockfile, no test suite.
* Run scripts **from the repo root** as `uv run scripts/0X_....py`. Imports are flat (`from utils import ...`, `from arquitecturas_neuronais import ...`), so they only resolve because the script's own directory goes on `sys.path`. Invoking them as `python -m scripts.01_...`, importing them from a notebook, or running from another cwd breaks them.
* Order matters: `01` -> `02` -> `03` -> `04`. Script 04 does not compute anything; it only reads `output/tables/tab_05_comparacao_econometrica.csv` (from 02) and `tab_07_dl_benchmark_metricas.csv` (from 03), so it fails on a partial run.
* Script 03 is the slow step (12 nets, 15 epochs, roughly 10-15 min on CPU). It is deliberately pinned to CPU (`device = torch.device("cpu")`) because the local GPU (NVIDIA MX150, compute capability 6.1) has no kernels in current PyTorch builds. Do not "optimise" this back to CUDA. `torch.manual_seed(SEED)` is reset at the start of every feature regime so the same architecture starts from identical weights in all three regimes; do not remove it, the run stops being reproducible.
* Regenerating the dataset (about 3s, deterministic, no Git needed because both caches exist):
  ```bash
  uv run dataset/Extractor/extractor.py            # rebuild daily + weekly CSVs
  uv run dataset/Extractor/extractor.py --force-tags    # re-walk refs/tags (~1s)
  uv run dataset/Extractor/extractor.py --force-extract # re-walk 1.39M commits (slow)
  ```
  The two `--force-*` flags require a local clone at the path printed in the banner (currently `~/.local/gits/linux`).

## 3. Data invariants (the mistakes an agent actually makes)

* `dataset/Extractor/kernel_raw_commits.csv` is ~175 MB and git-ignored. It must never be staged.
* The **last row of both CSVs is an incomplete bin** (extraction stopped mid-week on 2026-09-08). `utils.carregar_serie_temporal` already applies `.iloc[:-1]`; calling `.iloc[:-1]` again silently drops a real week.
* `pd.read_csv` does not restore the datetime frequency. Always set `df.index.freq = "W-MON"` (weekly) or `"D"` (daily); `statsmodels` needs it to resolve out-of-sample steps.
* A `W-MON` label marks the **end** of the bin, not the start: the row dated `2026-09-14` covers `2026-09-08` to `2026-09-14` (hence its 2 commits). Any per-bin join against a calendar must evaluate at `label - 6 days`, or the release phase shifts by one week.
* Columns 11-15 (`last_release_tag`, `last_release_date`, `weeks_since_release`, `is_merge_window`, `rc_week`) are NULL for the 10 weekly / 62 daily rows that precede the first available tag. NULL means *unknown*, never 0 and never "merge window". They are built with `merge_asof(direction="backward")` over **final** releases only; adding `-rc` tags or the next release date would introduce look-ahead bias.
* The release cycle, not the calendar, produces the extreme weeks, and its sign flips in 2015. Of the 40 weeks flagged by the `|Z| > 2` additive-residual test, all fall either in the merge window or in the final rc weeks and none in rc1-rc5. Split by era: 2005-2014 gives 19 positive peaks in the merge window against 4 late-cycle, while 2015-2026 gives 0 positive and all 6 negative drops there, with 11 peaks moving to the late cycle. Raw means agree (~1.0k commits in the modern merge window against ~1.8k elsewhere). Cause: a commit is dated when it was committed on the maintainer branch, so feature work never lands inside the window. Never quote a pooled merge-window statistic without the era split. Script 01 labels outliers from `weeks_since_release` via `classificar_fase_release` and the `FASE_*` constants; the month-based heuristic was removed and must not come back.
* Head outlier: the first observation contains the 6.75M-line initial import, so churn columns there have extreme leverage. Tail trough: the last clean weeks are `1042 / 638 / 415` commits and dominate any weekly REQM — check them before blaming a model.
* The release phase now **is** used: script 02 encodes it as two dummies (`fase_rc_inicial` = weeks 2-5, `fase_rc_final` = weeks 6+, merge window as reference) and that specification wins on REQM (254.00 vs 261.59). All Box-Jenkins specs are estimated on the common 883-row sample (the 10 weeks that precede the first tag have no known phase), so the winner comparison is like-for-like. The phase improves Ljung-Box but does not fix it (p = 0.0070 at lag 15). `last_release_tag` and `last_release_date` remain unused on purpose (identifier and date, not measures). Script 03 reuses the same two dummies on the **daily** series (W=28) as a third feature regime (`Multivariado + Release`, F=8); the 62 pre-tag days are dropped from the training sample of *all* regimes so the three regimes share the same windows. Full experiment log, including the LSTM seed sweep that chose the encoding: `scripts/analise_covariaveis_release/`.

## 4. Code layout

* `scripts/utils.py` — the only place paths and shared behaviour live: `DATASET_DIR`, `PNG_DIR`, `SVG_DIR`, `TABLES_DIR`, `RC_PARAMS`, `carregar_serie_temporal`, `dividir_treino_teste`, `calcular_metricas`, `guardar_figura`, `guardar_tabela`, `guardar_markdown`, `renderizar_template`. New code must use these instead of rebuilding paths; `DATASET_DIR` also has a `Dataset` capitalisation fallback for case-insensitive filesystems.
* `scripts/arquitecturas_neuronais/` — the four torch models (1D-CNN, simple RNN, LSTM, GRU) as classes; script 03 holds the training loop and the sliding-window preparation.
* `scripts/templates/*.md` — templates whose placeholders are filled by `renderizar_template`.
* `scripts/analise_covariaveis_release/` — the reproducible experiment behind the release-phase variables (candidate encodings, like-for-like control, rejected alternatives) plus the written justification. Not part of the numbered pipeline; it writes only inside its own folder.
* `output/*_SUMMARY.md` are **generated**. Edit `scripts/templates/*.md` and re-run, never the output file.
* Artifact numbering is contractual for the report: `fig_00`-`fig_08` (plus `fig_00_matriz_correlacao_pearson`/`spearman` and `fig_05b`), `tab_00`-`tab_08` (plus `tab_00_matriz_correlacao_pearson`/`spearman` and `tab_05b`). Append new artifacts (`fig_09`, `tab_09`) rather than renumbering, or every reference in `GUIA_RELATORIO_E_DIVISAO_TRABALHO.md` goes stale.
* `class-code/` is the instructors' reference implementation, not part of the pipeline. `docs/` holds the course PDFs.

## 5. Conventions

* Prose, printed output, plot labels and the report summaries are in **European Portuguese**; dataset column names, identifiers and the extractor docstrings are in **English**. Keep that split.
* No emojis anywhere, including in these files.
* Target line length is **120**, set only in `.vscode/settings.json` — no `ruff.toml` or `pyproject.toml` is committed. A bare `ruff format` reformats the whole repo at 88; use `ruff format --line-length 120` if you must, or leave formatting alone.
* There are no tests. Verification means re-running the affected script and inspecting the regenerated `output/` artefacts, which are deterministic given the CSVs (script 03 included, since the seed sweep fixed the random initialisation).
* Changing the dataset schema requires updating `dataset/dataset_fields_description.md`, which documents all 15 columns for a non-technical audience. Some literal values quoted there predate the current CSVs; the data wins.
* Methodological red lines (graded): chronological splits only, never shuffle; scalers fitted on the training split only; ADF before differencing; Ljung-Box on residuals. Note that `output/ECONOMETRICS_SUMMARY.md` quotes only lags 1 and 10, while the ARIMAX residuals reject at lags 15 and 20 — do not claim white noise without re-running the test.

## 6. Where the context lives

* `README.md` — quickstart and directory tree (its tree is stale, this file is authoritative).
* `GUIA_RELATORIO_E_DIVISAO_TRABALHO.md` — report structure, chapter-to-artifact mapping, work split.
* `CONCEITOS_E_GLOSSARIO.md` — method glossary (decomposition, ARIMA, DL architectures).
* `dataset/dataset_fields_description.md` — field guide, including the release-calendar semantics.
