"""
academic_search.py — Academic Literature Search
================================================
Searches Semantic Scholar and PubMed for peer-reviewed papers relevant to the
user's research question.  Results are formatted into a context block that is
injected into every model's prompt so they answer as academic researchers
grounded in actual literature.

No API keys are required — both APIs are freely accessible.

Public API
----------
search_academic_papers(query, year_from, year_to, max_results) → list[dict]
build_literature_context_block(papers) → str
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime
from typing import List, Optional

import requests


# ---------------------------------------------------------------------------
# Semantic Scholar (free, no key for <100 req/min)
# ---------------------------------------------------------------------------

_SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS     = "title,abstract,year,authors,citationCount,url,venue,externalIds"


def _search_semantic_scholar(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 8,
) -> List[dict]:
    """Search Semantic Scholar and return raw API results."""
    params: dict = {
        "query":  query,
        "fields": _SS_FIELDS,
        "limit":  limit,
    }
    if year_from or year_to:
        yf = year_from or 2000
        yt = year_to   or datetime.now().year
        params["year"] = f"{yf}-{yt}"

    try:
        resp = requests.get(
            _SS_SEARCH_URL,
            params=params,
            timeout=12,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# PubMed / NCBI Entrez (free, no key required)
# ---------------------------------------------------------------------------

_PM_SEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PM_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _search_pubmed(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 6,
) -> List[dict]:
    """Search PubMed and return normalised paper dicts."""
    search_params: dict = {
        "db":     "pubmed",
        "term":   query,
        "retmax": limit,
        "retmode": "json",
        "sort":   "relevance",
    }
    if year_from or year_to:
        yf = year_from or 2000
        yt = year_to   or datetime.now().year
        search_params.update({
            "datetype": "pdat",
            "mindate":  str(yf),
            "maxdate":  str(yt),
        })

    try:
        resp = requests.get(_PM_SEARCH_URL, params=search_params, timeout=12)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        sum_resp = requests.get(
            _PM_SUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=12,
        )
        sum_resp.raise_for_status()
        result = sum_resp.json().get("result", {})

        papers: List[dict] = []
        for pmid in ids:
            item = result.get(pmid, {})
            if not item:
                continue
            authors_raw = item.get("authors", [])
            papers.append({
                "title":    item.get("title", ""),
                "abstract": "",
                "year":     str(item.get("pubdate", ""))[:4],
                "authors":  [{"name": a.get("name", "")} for a in authors_raw[:4]],
                "url":      f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "citationCount": None,
                "venue":    item.get("source", ""),
                "source":   "PubMed",
            })
        return papers
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Combined search — parallel, deduplicated
# ---------------------------------------------------------------------------

def search_academic_papers(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    max_results: int = 10,
) -> List[dict]:
    """
    Search Semantic Scholar and PubMed in parallel and return combined results.

    Parameters
    ----------
    query : str
        Free-text research query.
    year_from : int, optional
        Earliest publication year to include.
    year_to : int, optional
        Latest publication year to include.
    max_results : int
        Maximum papers to return (capped at 15).

    Returns
    -------
    list[dict]
        Each dict has: title, abstract, year, authors (str), url,
        citations (int|None), journal (str), source (str).
    """
    max_results = min(max_results, 15)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        ss_fut = ex.submit(_search_semantic_scholar, query, year_from, year_to, max_results)
        pm_fut = ex.submit(_search_pubmed, query, year_from, year_to, max_results // 2)
        ss_raw = ss_fut.result()
        pm_raw = pm_fut.result()

    normalized: List[dict] = []
    seen: set = set()

    for p in ss_raw:
        title_key = (p.get("title") or "").lower().strip()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        authors_list = p.get("authors") or []
        normalized.append({
            "title":     p.get("title", ""),
            "abstract":  p.get("abstract") or "",
            "year":      str(p.get("year", "")),
            "authors":   ", ".join(a.get("name", "") for a in authors_list[:3]),
            "url":       p.get("url") or "",
            "citations": p.get("citationCount"),
            "journal":   p.get("venue") or "",
            "source":    "Semantic Scholar",
        })

    for p in pm_raw:
        title_key = (p.get("title") or "").lower().strip()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        authors_list = p.get("authors") or []
        normalized.append({
            "title":     p.get("title", ""),
            "abstract":  p.get("abstract") or "",
            "year":      str(p.get("year", "")),
            "authors":   ", ".join(a.get("name", "") for a in authors_list[:3]),
            "url":       p.get("url", ""),
            "citations": None,
            "journal":   p.get("venue", ""),
            "source":    "PubMed",
        })

    return normalized[:max_results]


def build_literature_context_block(papers: List[dict]) -> str:
    """
    Format a list of paper dicts into a context block for prompt injection.

    Returns an empty string when the list is empty.
    """
    if not papers:
        return ""

    lines = [
        "=== ACADEMIC LITERATURE CONTEXT ===",
        "The following peer-reviewed papers were retrieved via live academic search.",
        "They are VERIFIED sources. Cite them using (Authors, Year) format.\n",
    ]

    for i, p in enumerate(papers, 1):
        title    = p.get("title")    or "Unknown Title"
        year     = p.get("year")     or "n.d."
        authors  = p.get("authors")  or "Unknown Authors"
        journal  = p.get("journal")  or ""
        url      = p.get("url")      or ""
        abstract = p.get("abstract") or ""
        cit      = p.get("citations")

        entry = f"[{i}] {authors} ({year}). {title}."
        if journal:
            entry += f" {journal}."
        if cit is not None:
            entry += f" [Cited by {cit}]"
        if url:
            entry += f"\n    URL: {url}"
        if abstract:
            snippet = abstract[:350] + ("…" if len(abstract) > 350 else "")
            entry += f"\n    Abstract: {snippet}"
        lines.append(entry)

    lines.append("\n=== END OF LITERATURE CONTEXT ===")
    return "\n\n".join(lines)
