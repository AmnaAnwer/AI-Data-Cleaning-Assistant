"""
app.py
Streamlit front-end. Upload CSV -> detect issues -> optionally explain with
Groq -> user picks which fixes to apply -> preview -> download cleaned CSV.
"""

import io
import pandas as pd
import streamlit as st

from detectors import run_all_detectors
from cleaners import apply_issue
from llm_explainer import explain_all_issues, get_client

st.set_page_config(page_title="AI Data Cleaning Assistant", layout="wide")

st.title("AI Data Cleaning Assistant")
st.caption(
    "Upload a messy CSV. The LLM, if enabled, only explains it never touches your data."
)

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "issues" not in st.session_state:
    st.session_state.issues = []
if "applied_log" not in st.session_state:
    st.session_state.applied_log = []

# ---------------------------------------------------------------------------
# Sidebar: upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    outlier_method = st.selectbox("Outlier detection method", ["iqr", "zscore"], index=0)

    use_llm = st.checkbox("Use AI to explain issues with the given dataset", value=False)

    if uploaded_file is not None and st.session_state.original_df is None:
        st.session_state.original_df = pd.read_csv(uploaded_file)
        st.session_state.df = st.session_state.original_df.copy()

    if st.button("Reset session"):
        st.session_state.df = None
        st.session_state.original_df = None
        st.session_state.issues = []
        st.session_state.applied_log = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if st.session_state.df is None:
    st.info("Upload a CSV from the sidebar to begin.")
    st.stop()

df = st.session_state.df

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Issues found", len(st.session_state.issues))

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
if st.button("🔍 Analyze data"):
    issues = run_all_detectors(df, outlier_method=outlier_method)
    if use_llm:
        with st.spinner("Asking LLM to explain issues in plain English..."):
            issues = explain_all_issues(issues)
    st.session_state.issues = issues
    st.rerun()

# ---------------------------------------------------------------------------
# Show issues + let user apply fixes
# ---------------------------------------------------------------------------
if st.session_state.issues:
    st.subheader(f"Detected issues ({len(st.session_state.issues)})")

    severity_color = {"high": "🔴", "medium": "🟠", "low": "🟡"}

    remaining_issues = []
    for i, issue in enumerate(st.session_state.issues):
        badge = severity_color.get(issue["severity"], "⚪")
        with st.expander(f"{badge} [{issue['issue_type']}] {issue['column']}"):
            text = issue.get("llm_explanation", issue["detail"])
            st.write(text)
            st.caption(f"Suggested action: **{issue['suggested_action']}**")
            st.json(issue["stat"], expanded=False)

            apply_col, skip_col = st.columns([1, 1])
            if apply_col.button("✅ Apply fix", key=f"apply_{i}"):
                new_df, log_entry = apply_issue(df.copy(), issue)
                st.session_state.df = new_df
                st.session_state.applied_log.append(log_entry)
                # remove this issue from the list, keep the rest to re-check later
                st.session_state.issues = [
                    iss for j, iss in enumerate(st.session_state.issues) if j != i
                ]
                st.rerun()
            skip_col.button("Skip", key=f"skip_{i}")

# ---------------------------------------------------------------------------
# Applied changes log + download
# ---------------------------------------------------------------------------
if st.session_state.applied_log:
    st.subheader("Applied changes")
    for entry in st.session_state.applied_log:
        st.write(f"- {entry}")

    st.subheader("Cleaned data preview")
    st.dataframe(st.session_state.df.head(20), use_container_width=True)

    csv_buffer = io.StringIO()
    st.session_state.df.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=csv_buffer.getvalue(),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
