"""
detectors.py
Deterministic issue-detection logic. No randomness, no LLM calls here.
Every function returns a list of "Issue" dicts with a consistent shape so
the Streamlit UI can render them uniformly and the LLM layer (optional)
can explain them without touching the underlying math.
"""

from __future__ import annotations
import re
import difflib
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def make_issue(column: str, issue_type: str, severity: str, detail: str,
               stat: Dict[str, Any], suggested_action: str, action_key: str) -> Dict[str, Any]:
    """Uniform issue record. action_key is used later to trigger the matching cleaner."""
    return {
        "column": column,
        "issue_type": issue_type,     # e.g. "missing_values"
        "severity": severity,         # "low" | "medium" | "high"
        "detail": detail,             # human-readable summary (pre-LLM)
        "stat": stat,                 # raw numbers backing the claim
        "suggested_action": suggested_action,
        "action_key": action_key,     # maps to a function in cleaners.py
    }


# ---------------------------------------------------------------------------
# 1. MISSING VALUES
# ---------------------------------------------------------------------------
def detect_missing_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    issues = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        pct = n_missing / len(df) * 100
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if pct > 90:
            action, action_key = "drop column (too sparse to be useful)", "drop_column"
            severity = "high"
        elif is_numeric:
            action, action_key = "median imputation (robust to outliers)", "impute_median"
            severity = "medium" if pct > 20 else "low"
        else:
            action, action_key = "mode imputation (fill with most frequent value)", "impute_mode"
            severity = "medium" if pct > 20 else "low"

        issues.append(make_issue(
            column=col,
            issue_type="missing_values",
            severity=severity,
            detail=f"Column '{col}' has {n_missing} missing values ({pct:.1f}%).",
            stat={"missing_count": int(n_missing), "missing_pct": round(pct, 2)},
            suggested_action=action,
            action_key=action_key,
        ))
    return issues


# ---------------------------------------------------------------------------
# 2. DUPLICATE ROWS
# ---------------------------------------------------------------------------
def detect_duplicates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    dup_mask = df.duplicated(keep="first")
    n_dupes = dup_mask.sum()
    if n_dupes == 0:
        return []
    pct = n_dupes / len(df) * 100
    return [make_issue(
        column="(all columns)",
        issue_type="duplicate_rows",
        severity="medium" if pct > 5 else "low",
        detail=f"Found {n_dupes} exact duplicate rows ({pct:.1f}% of dataset).",
        stat={"duplicate_count": int(n_dupes), "duplicate_pct": round(pct, 2)},
        suggested_action="drop duplicate rows, keep first occurrence",
        action_key="drop_duplicates",
    )]


# ---------------------------------------------------------------------------
# 3. INCORRECT DATA TYPES
# ---------------------------------------------------------------------------
def _looks_numeric(series: pd.Series) -> bool:
    """True if a text column is actually storing numbers as strings."""
    sample = series.dropna().astype(str).str.strip()
    if len(sample) == 0:
        return False
    # allow commas, decimal points, leading +/-
    cleaned = sample.str.replace(",", "", regex=False)
    numeric_like = cleaned.str.match(r"^[+-]?\d+(\.\d+)?$")
    return numeric_like.mean() > 0.9  # 90%+ of values look numeric


def _looks_like_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(200)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.85


def detect_dtype_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    issues = []
    for col in df.columns:
        # covers classic object dtype AND pandas' newer StringDtype
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue  # already numeric/bool/datetime, skip

        if _looks_numeric(df[col]):
            issues.append(make_issue(
                column=col,
                issue_type="incorrect_dtype",
                severity="medium",
                detail=f"Column '{col}' is stored as text but values look numeric.",
                stat={"current_dtype": "object", "suspected_dtype": "float64"},
                suggested_action="convert to numeric (pd.to_numeric)",
                action_key="cast_numeric",
            ))
        elif _looks_like_date(df[col]):
            issues.append(make_issue(
                column=col,
                issue_type="incorrect_dtype",
                severity="medium",
                detail=f"Column '{col}' is stored as text but values look like dates.",
                stat={"current_dtype": "object", "suspected_dtype": "datetime64"},
                suggested_action="convert to datetime (pd.to_datetime)",
                action_key="cast_datetime",
            ))
    return issues


# ---------------------------------------------------------------------------
# 4. OUTLIERS (IQR method)
# ---------------------------------------------------------------------------
def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> List[Dict[str, Any]]:
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
        else:  # z-score
            mean, std = series.mean(), series.std()
            if std == 0:
                continue
            z = (series - mean).abs() / std
            outlier_mask = z > 3

        n_outliers = outlier_mask.sum()
        if n_outliers == 0:
            continue
        pct = n_outliers / len(series) * 100

        issues.append(make_issue(
            column=col,
            issue_type="outliers",
            severity="medium" if pct > 5 else "low",
            detail=f"Column '{col}' has {n_outliers} outlier value(s) ({pct:.1f}%) via {method.upper()} method.",
            stat={"outlier_count": int(n_outliers), "outlier_pct": round(pct, 2), "method": method},
            suggested_action="review manually, or cap at bounds (winsorize)",
            action_key="cap_outliers",
        ))
    return issues


# ---------------------------------------------------------------------------
# 5. INCONSISTENT CATEGORIES (fuzzy string matching)
# ---------------------------------------------------------------------------
def detect_inconsistent_categories(df: pd.DataFrame, max_unique: int = 50,
                                    similarity_threshold: float = 0.82) -> List[Dict[str, Any]]:
    issues = []
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        values = df[col].dropna().astype(str).str.strip()
        uniques = values.unique()
        if len(uniques) < 2 or len(uniques) > max_unique:
            continue  # too many uniques = probably free text or ID, skip
        if _looks_like_date(df[col]) or _looks_numeric(df[col]):
            continue  # dates/numbers share digits -> fuzzy match gives false positives here

        # normalize for comparison: lowercase, strip punctuation
        normalized = {v: re.sub(r"[^a-z0-9]", "", v.lower()) for v in uniques}
        groups = []
        seen = set()
        for v in uniques:
            if v in seen:
                continue
            cluster = [v]
            seen.add(v)
            for w in uniques:
                if w in seen:
                    continue
                ratio = difflib.SequenceMatcher(None, normalized[v], normalized[w]).ratio()
                if ratio >= similarity_threshold:
                    cluster.append(w)
                    seen.add(w)
            if len(cluster) > 1:
                groups.append(cluster)

        if groups:
            examples = "; ".join([" / ".join(g) for g in groups[:3]])
            issues.append(make_issue(
                column=col,
                issue_type="inconsistent_categories",
                severity="low",
                detail=f"Column '{col}' has similar-looking values that may be the same category: {examples}.",
                stat={"groups": groups},
                suggested_action="normalize casing/spacing and merge into one canonical label",
                action_key="normalize_categories",
            ))
    return issues


# ---------------------------------------------------------------------------
# 6. FORMATTING ERRORS
# ---------------------------------------------------------------------------
def detect_formatting_errors(df: pd.DataFrame) -> List[Dict[str, Any]]:
    issues = []
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        values = df[col].dropna().astype(str)
        if len(values) == 0:
            continue

        n_whitespace = (values != values.str.strip()).sum()

        if n_whitespace > 0:
            pct = n_whitespace / len(values) * 100
            issues.append(make_issue(
                column=col,
                issue_type="formatting_errors",
                severity="low",
                detail=f"Column '{col}' has {n_whitespace} value(s) with leading/trailing whitespace ({pct:.1f}%).",
                stat={"whitespace_count": int(n_whitespace)},
                suggested_action="strip leading/trailing whitespace",
                action_key="strip_whitespace",
            ))
    return issues


# ---------------------------------------------------------------------------
# 7. IRRELEVANT COLUMNS
# ---------------------------------------------------------------------------
def detect_irrelevant_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    issues = []
    n_rows = len(df)
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)

        # constant column: same value everywhere (zero information)
        if n_unique <= 1:
            issues.append(make_issue(
                column=col,
                issue_type="irrelevant_column",
                severity="medium",
                detail=f"Column '{col}' has only {n_unique} unique value — provides no information.",
                stat={"unique_count": int(n_unique)},
                suggested_action="drop column (constant value)",
                action_key="drop_column",
            ))
            continue

        # ID-like column: every value unique, likely a row identifier
        if n_unique == n_rows and n_rows > 20:
            issues.append(make_issue(
                column=col,
                issue_type="irrelevant_column",
                severity="low",
                detail=f"Column '{col}' has a unique value for every row — looks like an ID column.",
                stat={"unique_count": int(n_unique), "row_count": n_rows},
                suggested_action="drop column if not needed for joining/tracking",
                action_key="drop_column",
            ))
    return issues


# ---------------------------------------------------------------------------
# Orchestrator: run everything, return flat list
# ---------------------------------------------------------------------------
def run_all_detectors(df: pd.DataFrame, outlier_method: str = "iqr") -> List[Dict[str, Any]]:
    issues = []
    issues += detect_missing_values(df)
    issues += detect_duplicates(df)
    issues += detect_dtype_issues(df)
    issues += detect_outliers(df, method=outlier_method)
    issues += detect_inconsistent_categories(df)
    issues += detect_formatting_errors(df)
    issues += detect_irrelevant_columns(df)
    return issues
