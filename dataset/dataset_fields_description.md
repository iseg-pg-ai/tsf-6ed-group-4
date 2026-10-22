# Linux Kernel Time Series dataset: Field Descriptions & Field Guide
*A Non-Technical Guide for Software Collaborators, Product Managers, and Business Analysts*

---

## 1. Introduction & Context

Welcome! If you work alongside software engineers—as a **Product Manager, Project Manager, Scrum Master, Business Analyst, QA Specialist, Technical Writer, or Operations Lead**—you frequently hear terms like *commits*, *pull requests*, *merges*, and *code churn*. 

This document explains the data fields contained in two foundational files:
* **`dataset/kernel_daily_ts.csv`** (daily aggregated development activity)
* **`dataset/kernel_weekly_ts.csv`** (weekly aggregated development activity, sampled every Monday)

The last five columns of both files are derived from a third, much smaller file — **`dataset/kernel_releases.csv`**, the Linux Kernel release calendar — which the extractor reads straight from the project's own Git tags.

Both datasets track the evolution of the **Linux Kernel**—one of the largest, most critical open-source engineering projects in human history—spanning over 20 years (from April 2005 to September 2026) and more than 1.39 million individual code contributions.

Whether you are monitoring team velocity, forecasting software delivery dates, evaluating workload risk, or analyzing development trends, this guide will help you understand precisely what every number in these tables represents in plain, intuitive business language.

---

## 2. A 60-Second Git Primer for Collaborators

Before diving into the columns, here are four simple concepts that explain how software teams work with Git:

1. **A Commit is a "Saved Version":**  
   Think of a commit like taking a permanent snapshot of a document with a note attached (e.g., *"Fixed button alignment on login page"*). A commit records exactly what was changed, by whom, and when.
2. **Branches are "Drafting Rooms":**  
   Developers rarely write code directly on the live, official project. Instead, they make a "branch" (their own isolated copy or drafting room) where they can experiment without breaking things for others.
3. **A Merge is "Combining Drafts":**  
   When a feature is finished and tested, the maintainer integrates that branch back into the main project. That act of joining two work streams creates a special checkpoint called a **Merge Commit**.
4. **Author vs. Committer:**  
   In Git, the **author** is the person who originally wrote the code. The **committer** is the senior maintainer who reviewed, approved, and officially placed it into the main project timeline.

---

## 3. Daily vs. Weekly datasets: Why Are There Two Files?

Both files have the **exact same 15 columns** (10 activity metrics plus 5 release-cycle fields), but they look at the project through different zoom lenses:

| Aspect | `kernel_daily_ts.csv` (Daily) | `kernel_weekly_ts.csv` (Weekly) |
| :--- | :--- | :--- |
| **Observation Window** | Every single calendar day (`2005-04-16`, `2005-04-17`, ...) | Every 7-day week, labeled by Monday (`2005-04-18`, `2005-04-25`, ...) |
| **Micro-Rhythm** | Captures **day-of-week patterns** (e.g., busy Wednesdays vs. quiet weekends) | Smooths out weekend dips to show **sprint-to-sprint trends** |
| **Best Used For** | Daily sprint tracking, machine learning, deep learning (neural nets) | Forecasting quarterly velocity, classical statistics, seasonal business cycles |

---

## 4. Detailed Column-by-Column Reference

Here is what each of the **15 columns** means. The first ten are measured directly from the commit history and are illustrated with concrete examples drawn from the first 10 rows of both datasets. The last five are derived from the project's release calendar and are illustrated with the pattern they produce.

---

### Column 1: `committer_date`
* **Plain-English Name:** Date of Official Integration
* **Data Type:** ISO-8601 Date (`YYYY-MM-DD`)

#### What It Means in Practice
This is the date when the code was officially accepted and stamped into the project's permanent timeline. 

*Why isn't it called `author_date`?*  
Imagine a developer writes a code patch on a Tuesday (`author_date`), but it gets sent for peer review, tested, and finally approved by Linus Torvalds or a team lead on Friday. The **`committer_date`** is Friday—the day the work actually became part of the official software release.

#### Examples from the First 10 Rows
* **In `kernel_daily_ts.csv`:**  
  Row 1 is `2005-04-16` (a Saturday). This is the exact historic day Linus Torvalds introduced Git to the Linux project. Row 2 is `2005-04-17`, Row 3 is `2005-04-18`, and so on.
* **In `kernel_weekly_ts.csv`:**  
  Row 1 is `2005-04-18` (Monday). Every subsequent row advances by exactly 7 days: `2005-04-25`, `2005-05-02`, `2005-05-09`, etc. Each row aggregates the seven days **ending on** that Monday, so the label marks the end of the window rather than its start.

#### Why Collaborators Care
It is your timeline anchor. It allows you to align development velocity with real-world business events (product launches, holidays, conferences, or release deadlines).

---

### Column 2: `total_commits`
* **Plain-English Name:** Total Saved Changes / Total Updates Integrated
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
The total number of approved change-packages integrated into the software during that day or week. Every time someone merged a branch or applied an approved patch, that count went up by one.

#### The Relationship
$$\text{total\_commits} = \text{non\_merge\_commits} + \text{merge\_commits}$$

#### Examples from the First 10 Rows
* **Weekly (`2005-04-18`):** `total_commits = 259` (255 direct changes + 4 merges).
* **Weekly (`2005-05-23`):** `total_commits = 406` (a high-tempo week with 386 direct changes + 20 merges).
* **Daily (`2005-04-23`):** `total_commits = 1` (a quiet Saturday where only a single fix was integrated).

#### Why Collaborators Care
This is the general "heartbeat" or pulse of project activity. A sustained spike usually means teams are rushing toward a release; a sustained drop often reflects holidays, code freezes, or planning periods.

---

### Column 3: `non_merge_commits`
* **Plain-English Name:** Direct Feature & Bug-Fix Commits (Hands-On Engineering)
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
These are commits where engineers did **direct, hands-on work**: writing a new feature, fixing a bug, updating a driver, or editing documentation. 

Unlike merge commits (which are administrative checkpoints), `non_merge_commits` represent individual units of engineering progress.

#### Examples from the First 10 Rows
* **Daily (`2005-04-16`):** Out of 199 total commits, all `199` were non-merge commits (direct files added).
* **Weekly (`2005-04-25`):** `228` non-merge commits out of 238 total commits (95.8% direct engineering work).
* **Weekly (`2005-06-06`):** `166` non-merge commits out of 213 total commits (77.9%).

#### Why Collaborators Care
If you want to measure real engineering productivity, look at `non_merge_commits`. It strips away administrative merging and isolates the actual coding tasks accomplished.

---

### Column 4: `merge_commits`
* **Plain-English Name:** Integration Checkpoints / Combining Branches
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
A merge commit occurs when a maintainer takes an entire branch of work (which might contain 10, 50, or 200 individual commits built by a sub-team) and combines it into the main project.

Think of it like an editor signing off on an entire chapter submitted by a guest author: the merge commit is the stamp on the folder that binds the new chapter into the master manuscript.

#### Examples from the First 10 Rows
* **Daily (`2005-04-16`):** `0` merge commits (the repository was just initialized, so there were no branches to merge yet).
* **Weekly (`2005-04-18`):** Only `4` merge commits out of 259 total commits.
* **Weekly (`2005-05-30`):** Jumps to `45` merge commits (a major integration window where subsystem leaders brought finished batches of work to the project lead).
* **Weekly (`2005-06-06`):** Climbs further to `47` merge commits.

#### Why Collaborators Care
A surge in `merge_commits` indicates an **integration phase**. In product management terms, it tells you that sub-teams are turning in their homework and synchronizing their work with the rest of the company.

---

### Column 5: `unique_authors`
* **Plain-English Name:** Active Contributors / Team Headcount
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
How many distinct human beings actually contributed code that was accepted during that time period. 

*Why is `unique_authors` smaller than `total_commits`?*  
Because one productive developer often makes multiple commits in a day or week. If Alice makes 5 commits and Bob makes 3 commits, that is 8 total commits, but only **2 unique authors**.

#### Examples from the First 10 Rows
* **Daily (`2005-04-16`):** `64` unique authors contributed 199 commits (~3.1 commits per author).
* **Daily (`2005-04-23`):** Exactly `1` author working on Saturday.
* **Weekly (`2005-05-02`):** `118` different contributors participated in that week's 323 commits.
* **Weekly (`2005-05-16`):** Only `53` authors contributed (a quieter week, commit count also dipped to 147).

#### Why Collaborators Care
`unique_authors` is your **workforce capacity indicator**:
* If `total_commits` doubles but `unique_authors` stays flat, your existing team is working overtime (watch out for developer burnout!).
* If `unique_authors` grows steadily, your project is successfully attracting new talent or scaling across departments.

---

### Column 6: `files_changed`
* **Plain-English Name:** Breadth of Code Touched / Blast Radius
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
The total count of distinct files that were modified, added, or deleted across all commits during that day or week. 

If Commit A edits `login.py` and `database.py` (2 files), and Commit B edits `database.py` and `readme.md` (2 files), the total files changed is $2 + 2 = 4$ file touch-events.

#### Examples from the First 10 Rows
* **Daily (`2005-04-16`):** `17,712` files changed! (This was the initial import of the entire operating system into Git).
* **Daily (`2005-04-20`):** Only `16` files changed across 11 commits (typical targeted, small fixes).
* **Weekly (`2005-04-25`):** `463` files changed across 238 commits (~1.9 files touched per commit).
* **Weekly (`2005-05-02`):** `903` files changed across 323 commits.

#### Why Collaborators Care
This indicates the **breadth vs. depth** of the work:
* Touching **few files with many lines** = deep, localized change (e.g., rewriting an internal algorithm; low risk to other teams).
* Touching **hundreds of files with few lines** = widespread architectural change (e.g., updating copyright headers or renaming a common function; high risk of merge conflicts and testing headaches).

---

### Column 7: `insertions`
* **Plain-English Name:** Lines of Code Added (`+`)
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
The number of new lines added to the project. In tools like GitHub or GitLab, these appear highlighted in **green** with a plus sign (`+`).

*Important Note:* In version control, if a developer simply edits an existing sentence or line of code, Git counts that as **deleting the old line** and **inserting the new line**.

#### Examples from the First 10 Rows
* **Daily (`2005-04-16`):** `6,723,547` insertions (the massive original codebase being recorded into Git for the first time).
* **Daily (`2005-04-21`):** `213` insertions across 23 commits (modest daily additions).
* **Weekly (`2005-04-25`):** `9,383` lines inserted.
* **Weekly (`2005-05-16`):** `50,497` lines inserted (a substantial new feature or hardware driver landed that week).

#### Why Collaborators Care
Measures creative/generative construction. Spikes in insertions represent new features being built, new vendor libraries imported, or documentation added.

---

### Column 8: `deletions`
* **Plain-English Name:** Lines of Code Removed (`-`)
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
The number of lines removed from the project. In version control tools, these appear in **red** with a minus sign (`-`).

Deletions happen when engineers remove obsolete features, clean up unused code ("dead code"), or refactor complex scripts into shorter, cleaner versions.

#### Examples from the First 10 Rows
* **Daily (`2005-04-17`):** `693` deletions (higher than the 413 insertions that day).
* **Daily (`2005-04-23`):** Exactly `1` line deleted.
* **Weekly (`2005-04-25`):** `2,545` deletions.
* **Weekly (`2005-05-23`):** `26,453` deletions (a major spring cleaning / refactoring effort).

#### Why Collaborators Care
**Deleting code is a sign of healthy software engineering!** Good teams actively delete unnecessary code to keep the system fast, secure, and easy to maintain. A project with zero deletions is usually accumulating dangerous technical debt.

---

### Column 9: `total_churn`
* **Plain-English Name:** Total Code Volatility / Gross Kinetic Activity
* **Data Type:** Whole Number (Integer $\ge 0$)

#### What It Means in Practice
**Code Churn** measures total movement, regardless of whether lines were added or removed.

#### The Metaphor
Think of an automobile's odometer: whether you drive 10 miles forward or put the car in reverse and back up for 10 miles, the odometer records **20 miles of travel**.  
Similarly, if a developer replaces 100 lines of bad code with 100 lines of clean code, `net_code_growth` is $0$, but `total_churn` is:
$$100 \text{ (added)} + 100 \text{ (removed)} = 200 \text{ lines of churn}$$

#### The Formula
$$\text{total\_churn} = \text{insertions} + \text{deletions}$$

#### Examples from the First 10 Rows
* **Daily (`2005-04-17`):** $413 \text{ insertions} + 693 \text{ deletions} = \mathbf{1,106} \text{ churn}$.
* **Daily (`2005-04-22`):** $1,604 \text{ insertions} + 551 \text{ deletions} = \mathbf{2,155} \text{ churn}$.
* **Weekly (`2005-05-23`):** $26,017 \text{ insertions} + 26,453 \text{ deletions} = \mathbf{52,470} \text{ churn}$.

#### Why Collaborators Care
**Total churn is the premier proxy for code instability and testing workload.** When churn is high, large portions of the system are in flux. For QA and product managers, high churn signals that regression testing must be intensified.

---

### Column 10: `net_code_growth`
* **Plain-English Name:** Net Expansion or Shrinkage of the Codebase
* **Data Type:** Signed Number (Can be positive, zero, or **negative**!)

#### What It Means in Practice
This shows whether the codebase grew larger or became smaller at the end of the period.

#### The Formula
$$\text{net\_code\_growth} = \text{insertions} - \text{deletions}$$

#### Why Can It Be Negative?
If engineers delete more lines than they add (e.g., pruning old features or rewriting a 500-line module into an elegant 100-line script), `net_code_growth` will be a negative number! 

**A negative number is NOT a bug or error in the dataset.** It simply means the software got lighter and leaner.

#### Examples from the First 10 Rows
* **Daily (`2005-04-17`):** $413 - 693 = \mathbf{-280}$ (The codebase shrank by 280 lines).
* **Daily (`2005-04-18`):** $25,533 - 7,527 = \mathbf{+18,006}$ (The codebase expanded by 18,006 lines).
* **Weekly (`2005-04-25`):** $9,383 - 2,545 = \mathbf{+6,838}$ (Net growth of 6,838 lines).
* **Weekly (`2005-05-23`):** $26,017 - 26,453 = \mathbf{-436}$ (The codebase shrank by 436 lines across the week).

#### Why Collaborators Care
Tells you the trajectory of the product. Sustained positive growth means the system is adding capabilities (or bloat). Negative growth periods demonstrate active refactoring, modernization, and debt reduction.

---

### Column 11: `last_release_tag`
* **Plain-English Name:** The Release This Period Belongs To
* **Data Type:** Text (for example `v6.12`); may be empty

#### What It Means in Practice
The Linux Kernel is not shipped as one continuous stream. Every few months the accumulated work is frozen, tested, and published as a numbered **release**. This column records which release was the most recent one still in force when the period began.

Only **final** releases appear here. The intermediate release candidates (`-rc1`, `-rc2`, ...) are tracked separately and never fill this column, because they are working milestones rather than shipped products.

#### Example Pattern
* A period whose `last_release_tag` is `v6.12` belongs to the cycle that will conclude with the following release.
* The same type of week recurs in every cycle with a different tag, which is exactly what makes the field useful for grouping.

#### Why Collaborators Care
It lets you roll period-level numbers up to release-level numbers without guessing where the boundaries are. If someone asks *"how much work went into the 6.13 release?"*, you can answer by grouping on this column instead of reconstructing dates by hand.

---

### Column 12: `last_release_date`
* **Plain-English Name:** When That Release Shipped
* **Data Type:** ISO-8601 Date (`YYYY-MM-DD`); may be empty

#### What It Means in Practice
The publication date of the release named in `last_release_tag`. It is the anchor from which the release-cycle clock in `weeks_since_release` counts.

#### Example Pattern
`last_release_tag` and `last_release_date` describe the same event from two angles: which version shipped, and when. Reading them together prevents ambiguity when two cycles share a similar volume of activity.

#### Why Collaborators Care
It gives you a hard calendar reference for any downstream chart or commitment, and it makes the release-cycle columns auditable: every derived value can be traced back to a real, dated publication.

---

### Column 13: `weeks_since_release`
* **Plain-English Name:** The Release-Cycle Clock
* **Data Type:** Whole Number (Integer $\ge 0$); may be empty

#### What It Means in Practice
How many whole weeks have passed since the last release. The clock restarts at `0` right after a release and climbs one step per week until the next release resets it.

Two details matter:

* **It is measured at the *start* of the period.** The week in which a release actually lands still reports the age of the *previous* cycle, because the bulk of that week's work was completed before the release. The counter restarts in the week that follows.
* **It only ever looks backwards.** The field is built exclusively from releases that had already happened. It never uses the date of the next release, so it cannot leak knowledge of the future into a model.

#### Historical Note
Before December 2014 the kernel shipped on an irregular schedule, and long gaps push this value well above `10`. From December 2014 onwards the project settled into a regular rhythm of roughly nine to ten weeks per cycle, so the field typically ranges from `0` to `9`.

#### Why Collaborators Care
This is the most useful column for explaining why some weeks are busy and others are quiet. It converts *"we had a good week"* into *"we were in week 1 of the cycle"*, which is a statement that can actually be planned around.

---

### Column 14: `is_merge_window`
* **Plain-English Name:** Planned Integration Surge Flag
* **Data Type:** `1` or `0`; may be empty

#### What It Means in Practice
`1` marks the **merge window**: the roughly two-week period immediately after a release, when subsystem maintainers submit the work they have been accumulating while the code was frozen. In this dataset the flag is simply `weeks_since_release` equal to `0` or `1`.

#### Why Collaborators Care
This is the column that separates a **planned burst** from organic growth. A spike in `total_commits` while `is_merge_window = 1` is business as usual and should not be read as a sudden productivity gain; the same spike while `is_merge_window = 0` deserves a much closer look.

---

### Column 15: `rc_week`
* **Plain-English Name:** Stabilisation Week Counter
* **Data Type:** Whole Number (Integer $\ge 0$); may be empty

#### What It Means in Practice
Once the merge window closes, the project enters the **release-candidate (rc) phase**: new features are refused and only corrections are accepted. This column counts those weeks — `1` for the first rc week, `2` for the second, and so on until the next release.

A value of `0` means the period belongs to the merge window rather than the stabilisation phase. It does **not** mean "no activity".

#### Why Collaborators Care
The two phases differ in the *character* of the work, not only in its volume: the merge window brings large structural changes, while the rc phase brings small, targeted fixes. If your QA effort is keyed to risk, this column tells you which kind of work to expect.

---

### About the Release Calendar

All five are derived from **`dataset/kernel_releases.csv`**, a compact table of the kernel's own release tags with the columns `tag`, `release_date`, `is_rc`, `rc_number`, `version_major`, `version_minor` and `version_patch`. It is regenerated directly from Git, independently of the 1.39M-commit history, and can be refreshed on its own.

**Empty is not zero.** Where these five columns are empty, it means no release had been recorded before that period — not that the period had no activity. This can only happen at the very beginning of the series, where the Git history starts before the first available release tag.

---

## 5. Summary Cheat Sheet: How All Fields Connect

The following balance-sheet equations link the columns together:

```
[ Commits Balance ]
total_commits = non_merge_commits + merge_commits

[ Churn Formula ]
total_churn = insertions + deletions

[ Growth Formula ]
net_code_growth = insertions - deletions

[ Team Productivity Ratio ]
Average Commits per Author = total_commits / unique_authors
```

The five release-cycle fields are not balance identities; they follow three definitions:

```
[ Release-Cycle Clock ]
weeks_since_release = whole weeks between the start of the period and the last final release

[ Surge Flag ]
is_merge_window = 1 when weeks_since_release is 0 or 1, otherwise 0

[ Stabilisation Counter ]
rc_week = max(0, weeks_since_release - 1)
```

### Quick Reference Matrix

| Column | Plain-English Concept | Can Be Negative? | What High Values Mean |
| :--- | :--- | :---: | :--- |
| **`committer_date`** | Timeline date | No | Calendar milestone |
| **`total_commits`** | All saved change packages | No | Intense integration activity |
| **`non_merge_commits`** | Direct programming tasks | No | High volume of hands-on feature/bug work |
| **`merge_commits`** | Branch integration checkpoints | No | Sub-teams synchronizing their work |
| **`unique_authors`** | Distinct contributors | No | Large active workforce; healthy contributor community |
| **`files_changed`** | Total files modified | No | Broad, cross-cutting modification footprint |
| **`insertions`** | Lines added (`+`) | No | Rapid expansion, new feature development |
| **`deletions`** | Lines removed (`-`) | No | Active code cleanup, legacy feature removal |
| **`total_churn`** | Total lines modified (`+` and `-`) | No | High volatility; heavy QA and testing required |
| **`net_code_growth`** | Net line difference (`+` minus `-`) | **YES** | Positive = project growing; Negative = codebase slimming down |
| **`last_release_tag`** | Which release was in force | No | Not applicable (text); newer tags = more recently shipped version |
| **`last_release_date`** | When that release shipped | No | Not applicable (date); later dates = closer to the present |
| **`weeks_since_release`** | Release-cycle clock | No | Later in the cycle; stabilisation work, lower volume |
| **`is_merge_window`** | Planned integration surge (`1`/`0`) | No | The two-week post-release push |
| **`rc_week`** | Stabilisation week counter | No | Later rc stage; closer to the next release |

---

## 6. Real-World Case Studies from the First 10 Rows

Let's walk through three contrasting rows from the actual dataset to see how these numbers tell a story:

### Case Study 1: The "Big Bang" Initial Import
* **Source:** `kernel_daily_ts.csv` — Date: `2005-04-16` (Row 1)
* **The Numbers:**
  * `total_commits`: 199
  * `unique_authors`: 64
  * `files_changed`: 17,712
  * `insertions`: 6,723,547
  * `deletions`: 2,757
  * `net_code_growth`: +6,720,790
* **The Story:**  
  Linus Torvalds did not write 6.7 million lines of code in a single Saturday! The Linux operating system had been developed since 1991 in older tools. On April 16, 2005, Linus imported the entire existing operating system into Git. That single milestone created over 17,000 files and 6.7 million lines of code in one initial baseline snapshot.

---

### Case Study 2: The Quiet Weekend
* **Source:** `kernel_daily_ts.csv` — Date: `2005-04-23` (Row 8, Saturday)
* **The Numbers:**
  * `total_commits`: 1
  * `non_merge_commits`: 1
  * `merge_commits`: 0
  * `unique_authors`: 1
  * `files_changed`: 1
  * `insertions`: 1
  * `deletions`: 1
  * `total_churn`: 2
  * `net_code_growth`: 0
* **The Story:**  
  A textbook quiet Saturday. A single contributor showed up, opened exactly one file, fixed one line (deleting 1 line and inserting 1 replacement line), and saved it. Total churn was 2, net growth was exactly 0. This illustrates how the daily dataset captures natural human work-rest cycles.

---

### Case Study 3: The Spring Cleaning Refactor
* **Source:** `kernel_weekly_ts.csv` — Week of: `2005-05-23` (Row 6)
* **The Numbers:**
  * `total_commits`: 406
  * `unique_authors`: 111
  * `files_changed`: 767
  * `insertions`: 26,017
  * `deletions`: 26,453
  * `total_churn`: 52,470
  * `net_code_growth`: **-436**
* **The Story:**  
  A powerhouse week of work: 111 developers produced 406 commits, modifying over 52,000 lines of code across 767 files. But because they deleted 26,453 lines while adding 26,017 lines, the codebase ended the week **436 lines smaller than it began**. This was a dedicated week of performance tuning and debt cleanup.

---

## 7. How to Use This Data in Project Discussions

When collaborating with developers and engineering leads, you can use these metrics to ask insightful, data-driven questions:

1. **"Our `total_churn` is very high this sprint, but `total_commits` is low. Are we rewriting a major module?"**  
   *(Helps QA prepare for heavy regression testing instead of testing new features).*
2. **"Our `unique_authors` dropped 30% over the last two weeks. Was that planned vacation, or are developers blocked on requirements?"**  
   *(Identifies staffing bottlenecks early).*
3. **"`merge_commits` spiked yesterday. Are the feature branches finally merged and ready for user acceptance testing (UAT)?"**  
   *(Helps align product delivery timelines).*
4. **"Our `net_code_growth` has been negative for three weeks straight. Great job cleaning up technical debt!"**  
   *(Celebrates quality and simplification, not just raw volume).*
5. **"`total_commits` doubled but `is_merge_window` is 1 — is that the planned integration surge, or are we simply delivering late?"**  
   *(Separates the predictable post-release push from genuine schedule slippage, which is the difference between a healthy cadence and a team in trouble).*
6. **"`weeks_since_release` is sitting at 9 and the next release has not shipped. Was the cycle extended on purpose?"**  
   *(A stretched cycle usually signals either a large batch of late-arriving work or a quality problem that delayed the stabilisation phase).*

---

*Document compiled for the Time Series Forecasting Course (2026/27 — 6th Edition) at ISEG Executive Education.*
