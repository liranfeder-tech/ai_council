"""
search_engine.py — Pre-Flight Live Search (AI-Planned, Generic)
===============================================================
Fetches live web-search results via Serper.dev *before* the Council debate
begins, so every model is grounded in current facts for ANY question.

There is a single query source: a fast model (Gemini Flash) reads the user's
question and decides what to look up.

  get_live_market_data(question) -> (context_block, citations)
      1. _plan_queries_with_ai(question) turns the question into 2-6 English
         search queries (or decides no live search is needed).
      2. Those queries are run against Serper.dev in parallel, results are
         deduplicated by URL, and the top 10 are formatted into a Verified
         Research Context Block (with STAGE0_RESEARCH_BLOCK_HEADER/FOOTER)
         plus structured citation dicts for the UI.

The name "market data" is historical — the search is fully general now and
relevant to any domain (science, geopolitics, finance, technology, …).

Configuration
-------------
Set SERPER_API_KEY in your .env file.  Every function degrades gracefully to a
no-op when the key is absent.

Dependencies
------------
  requests  (pure HTTP, already in requirements.txt)
"""

from __future__ import annotations

import concurrent.futures
import os
from datetime import datetime
from typing import List, Tuple

import requests

from glossary import (
    STAGE0_QUERY_PLAN_PROMPT,
    STAGE0_RESEARCH_BLOCK_FOOTER,
    STAGE0_RESEARCH_BLOCK_HEADER,
)

_SERPER_URL = "https://google.serper.dev/search"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serper_search(query: str, num: int = 5) -> List[dict]:
    """POST a query to Serper.dev and return the organic results list."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.post(
            _SERPER_URL,
            json={"q": query, "num": num},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("organic", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_is_available() -> bool:
    """Return True when SERPER_API_KEY is present in the environment."""
    return bool(os.environ.get("SERPER_API_KEY"))


def _plan_queries_with_ai(question: str) -> Tuple[List[str], bool]:
    """
    Use Gemini Flash to plan targeted English search queries for ANY question.

    This is the ONLY query source for the pre-flight search: instead of
    hardcoded keyword lists, a fast model decides what facts need looking up.

    Returns
    -------
    (queries, planner_succeeded) : tuple[list[str], bool]
        planner_succeeded is True when the planner ran cleanly — including the
        case where it deliberately returned NO_SEARCH_NEEDED (queries == []
        means "no live search needed", which is correct for purely theoretical
        questions).  planner_succeeded is False ONLY when the planner call
        errored or raised; the caller then falls back to the raw question so
        live search still happens.

    Runs in ~1-2 s and never raises.
    """
    try:
        # local import avoids a circular dep at module load
        from ai_factory import call_model, _is_error_text
        current_date = datetime.now().strftime("%B %Y")
        prompt = STAGE0_QUERY_PLAN_PROMPT.format(
            current_date=current_date,
            question=question[:500],
        )
        raw = call_model("gemini", prompt, [], [])
        # A failed planner call returns an "ERROR: …" string (or, pre-Batch-2,
        # None). That is NOT a deliberate "no search" decision — signal failure
        # so the caller falls back to searching the raw question.
        if _is_error_text(raw):
            return [], False
        if "NO_SEARCH_NEEDED" in raw.upper():
            return [], True
        queries = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 8
        ]
        return queries, True
    except Exception:
        return [], False


def get_live_market_data(question: str) -> Tuple[str, List[dict]]:
    """
    Run a pre-flight live search for the question and return a multi-line
    context block plus structured citation dicts for the UI.

    Query planning is fully AI-driven (_plan_queries_with_ai):
      • planner returned queries        → search them (capped at 6),
      • planner said NO_SEARCH_NEEDED   → return ("", []) (no search — correct
                                          for purely theoretical questions),
      • planner FAILED (error/exception) → fall back to the raw question so
                                          live grounding still happens.

    All Serper.dev searches run in parallel via ThreadPoolExecutor.
    """
    if not search_is_available():
        return "", []

    planned, planner_ok = _plan_queries_with_ai(question)
    if planner_ok:
        if not planned:
            # Model deliberately decided no live search is needed.
            return "", []
        queries = planned[:6]
    else:
        # Planner errored — degrade gracefully so live search still happens.
        queries = [question[:200]]

    # ── Run all Serper searches in parallel ────────────────────────────────
    seen_urls: set = set()
    all_results: List[dict] = []

    def _fetch(q: str) -> List[dict]:
        return _serper_search(q, num=4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 6)) as ex:
        for hits in ex.map(_fetch, queries):
            for hit in hits:
                url = hit.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(hit)

    if not all_results:
        return "", []

    top = all_results[:10]

    citations: List[dict] = [
        {"title": r.get("title", r.get("link", "")), "url": r.get("link", "")}
        for r in top
        if r.get("link")
    ]

    lines: List[str] = []
    for r in top:
        title   = r.get("title", "")
        snippet = r.get("snippet", "")
        url     = r.get("link", "")
        if snippet:
            entry = f"  * {title}: {snippet}"
            if url:
                entry += f"\n    Source: {url}"
            lines.append(entry)

    if not lines:
        return "", []

    context_block = (
        STAGE0_RESEARCH_BLOCK_HEADER
        + "\n\n".join(lines)
        + STAGE0_RESEARCH_BLOCK_FOOTER
    )

    return context_block, citations
