"""
llm_explainer.py
Optional layer. Takes already-computed issue dicts (facts, numbers) and asks
Groq's LLM API to turn them into a friendly, plain-English explanation.

IMPORTANT DESIGN RULE:
This file NEVER decides what is wrong with the data and NEVER performs any
cleaning. It only receives facts that detectors.py already computed, and
rephrases them. This keeps the app's actual behavior deterministic and
testable, while still getting nice natural-language output for the user.

Groq is used because it's fast and has a generous free tier; swapping to
OpenAI/Anthropic later just means changing the client + model name below.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any

try:
    from groq import Groq
except ImportError:
    Groq = None  # library not installed; app should degrade gracefully


def get_client() -> "Groq | None":
    """Build a Groq client from the GROQ_API_KEY env var. Returns None if
    the key or library is missing so the caller can skip LLM explanation."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)


def explain_issue(issue: Dict[str, Any], client=None) -> str:
    """Turn a single issue dict into a one/two-sentence plain-English note.
    Falls back to the deterministic 'detail' text if no client is available."""
    if client is None:
        client = get_client()
    if client is None:
        return issue["detail"]  # graceful fallback, app still works with zero API keys

    prompt = f"""You are a data-cleaning assistant. Explain the following detected
data quality issue in one or two short, plain-English sentences for a non-technical
user. Do NOT invent numbers or facts beyond what is given. Do NOT suggest an action
other than the one given.

Column: {issue['column']}
Issue type: {issue['issue_type']}
Detected facts: {issue['stat']}
Deterministic detail: {issue['detail']}
Suggested action (already decided by rules, just explain it): {issue['suggested_action']}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=2048,
             top_p=1,
             reasoning_effort="medium",
             stop=None
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Any API failure (rate limit, no internet, bad key) must not break the app.
        return issue["detail"] + f" (LLM explanation unavailable: {e})"


def explain_all_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach an 'llm_explanation' field to each issue. Reuses one client
    for the whole batch instead of reconnecting per issue."""
    client = get_client()
    enriched = []
    for issue in issues:
        issue = dict(issue)  # don't mutate original
        issue["llm_explanation"] = explain_issue(issue, client=client)
        enriched.append(issue)
    return enriched
