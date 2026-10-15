"""
cleaners.py
Applies the actual data transformations. Every function takes a DataFrame
and an issue dict, returns (new_df, log_entry). Nothing here calls an LLM —
this is the part of the app that must be 100% reproducible and testable.
"""

from __future__ import annotations
import re
import difflib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def impute_median(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    col = issue["column"]
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    return df, f"Filled {issue['stat']['missing_count']} missing values in '{col}' with median ({median_val:.2f})."


def impute_mode(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    col = issue["column"]
    mode_val = df[col].mode(dropna=True)
    fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
    df[col] = df[col].fillna(fill_val)
    return df, f"Filled {issue['stat']['missing_count']} missing values in '{col}' with mode ('{fill_val}')."


def drop_column(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    col = issue["column"]
    df = df.drop(columns=[col])
    return df, f"Dropped column '{col}'."


def drop_duplicates(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df)
    return df, f"Removed {removed} duplicate rows."


def cast_numeric(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    col = issue["column"]
    cleaned = df[col].astype(str).str.replace(",", "", regex=False)
    df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df, f"Converted '{col}' to numeric type."


def cast_datetime(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    col = issue["column"]
    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df, f"Converted '{col}' to datetime type."


def cap_outliers(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """Winsorize: clip values to the IQR bounds instead of deleting rows."""
    col = issue["column"]
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_affected = int(((df[col] < lower) | (df[col] > upper)).sum())
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df, f"Capped {n_affected} outlier value(s) in '{col}' to range [{lower:.2f}, {upper:.2f}]."


def normalize_categories(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """Merge similar-looking category values into one canonical label
    (the most frequent spelling in each group wins)."""
    col = issue["column"]
    groups = issue["stat"]["groups"]
    replaced = 0
    for group in groups:
        counts = df[col].value_counts()
        # pick the spelling that appears most often as the canonical form
        canonical = max(group, key=lambda v: counts.get(v, 0))
        for variant in group:
            if variant != canonical:
                mask = df[col] == variant
                replaced += int(mask.sum())
                df.loc[mask, col] = canonical
    return df, f"Merged {replaced} inconsistent category value(s) in '{col}' into canonical labels."


def strip_whitespace(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    col = issue["column"]
    df[col] = df[col].astype(str).str.strip()
    return df, f"Stripped leading/trailing whitespace in '{col}'."


# Registry mapping action_key -> function. The UI looks up this dict.
CLEANER_REGISTRY = {
    "impute_median": impute_median,
    "impute_mode": impute_mode,
    "drop_column": drop_column,
    "drop_duplicates": drop_duplicates,
    "cast_numeric": cast_numeric,
    "cast_datetime": cast_datetime,
    "cap_outliers": cap_outliers,
    "normalize_categories": normalize_categories,
    "strip_whitespace": strip_whitespace,
}


def apply_issue(df: pd.DataFrame, issue: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """Look up the right cleaner for this issue and run it."""
    fn = CLEANER_REGISTRY.get(issue["action_key"])
    if fn is None:
        return df, f"No cleaner registered for action '{issue['action_key']}'."
    return fn(df, issue)
