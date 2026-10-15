# AI Data Cleaning Assistant

Upload a messy CSV. The app detects data quality issues (missing values,
duplicates, wrong types, outliers, inconsistent categories, formatting
errors, irrelevant columns), explains them, and applies fixes you approve.

## Design principle: detection & cleaning are deterministic, the LLM is optional garnish

Every issue is found and every fix is applied using plain pandas/statistics
code in `detectors.py` and `cleaners.py`. No randomness, no model calls.
This means:
- Same input always gives same output (reproducible, unit-testable).
- The LLM (Groq) is only used in `llm_explainer.py` to turn already-computed
  facts into a friendlier sentence. It never decides what's wrong or how to
  fix it, and if the API key is missing or the call fails, the app falls
  back to the plain rule-based description automatically — nothing breaks.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"   # optional, only needed for LLM explanations
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com — no cost for the model
used here (`llama-3.1-8b-instant`).

## Suggested workflow

Fix issues roughly in this order for best results:
1. **Data types first** (e.g. text-that-should-be-numbers) — fixing this
   unlocks correct downstream numeric checks (median imputation, outliers).
2. **Duplicates**
3. **Missing values**
4. **Outliers**
5. **Inconsistent categories / formatting**
6. **Irrelevant columns**

The app doesn't force this order, but re-running "Analyze data" after each
fix will re-classify remaining issues with fresh information (e.g. a column
fixed to numeric will now get median imputation instead of mode).

## File structure

```
app.py             Streamlit UI, session state, wiring
detectors.py        All issue-detection logic (pure functions, no side effects)
cleaners.py          All fix-application logic (pure functions, no side effects)
llm_explainer.py    Optional Groq call, explains issues in plain English
sample_data/         A deliberately messy CSV to try the app on
requirements.txt
```

## Detection methods used

- **Missing values**: `.isna()` count/percentage per column.
- **Duplicates**: exact row match via `df.duplicated()`.
- **Wrong dtypes**: regex/parse-based heuristic — if >90% of a text
  column's values parse as numbers or dates, flag it.
- **Outliers**: IQR method (values outside Q1 - 1.5×IQR to Q3 + 1.5×IQR)
  or z-score (|z| > 3), selectable in the sidebar.
- **Inconsistent categories**: normalize casing/punctuation, then fuzzy
  match remaining values with `difflib.SequenceMatcher` (similarity ≥ 0.82)
  to group likely-same categories written differently.
- **Formatting errors**: leading/trailing whitespace detection.
- **Irrelevant columns**: constant columns (1 unique value) or fully-unique
  ID-like columns.

## Known limitations (worth mentioning in an interview)

- Fuzzy category matching is intentionally skipped for columns that look
  numeric or date-like, since shared digits between different values create
  false-positive similarity matches.
- Outlier detection assumes roughly unimodal numeric distributions; it will
  over-flag legitimately multi-modal data (e.g. a column mixing two very
  different populations).
- Near-duplicate row detection (rows that are *almost* identical, e.g. one
  typo apart) is not implemented — only exact duplicates are caught. This
  is a natural "v2" extension using row-level fuzzy hashing.
