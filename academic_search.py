"""
academic_search.py — Multi-Source, Multilingual Academic Literature Search
==========================================================================
Searches FOUR scholarly databases for papers relevant to the user's research
question and formats the results into a context block injected into every
model's prompt, so the council answers as researchers grounded in real
literature.

Sources
-------
  • OpenAlex          (free, every discipline, every language, has abstracts)
  • Semantic Scholar  (free, English-centric, has abstracts)
  • PubMed / Entrez   (free, biomedical, English; abstracts via efetch)
  • Google Scholar    (via the paid Serper /scholar endpoint; any language,
                       snippet-only "abstracts")

Bilingual planning
------------------
A fast model turns the question into English queries (EN:) plus, when the
question is not in English, queries in its ORIGINAL language (QL:).  Queries are
routed per source (see search_academic_papers) so each engine gets the query
languages it handles best.

Public API
----------
search_academic_papers(query, year_from, year_to, max_results)
    → (list[dict], list[str])   # (papers, degradation / per-source notes)
build_literature_context_block(papers) → str

Paper dict schema (additive — Batch 5 added "language" and "doi"):
    title, abstract, year, authors(str), url, citations(int|None),
    journal(str), source(str), language(ISO code or ""), doi(str or "")
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

import requests

from glossary import ACADEMIC_QUERY_PLAN_PROMPT


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# ISO 639-1 codes → display names for the common languages we surface.
_LANG_NAMES = {
    "en": "English",  "he": "Hebrew",  "ar": "Arabic",  "fr": "French",
    "es": "Spanish",  "pt": "Portuguese", "de": "German", "it": "Italian",
    "ru": "Russian",  "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "nl": "Dutch",    "tr": "Turkish", "fa": "Persian", "hi": "Hindi",
    "pl": "Polish",   "sv": "Swedish",
}

_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/", "doi:",
)


def _has_non_ascii(text: str) -> bool:
    """True when the text contains any non-ASCII character (non-Latin script)."""
    return any(ord(c) > 127 for c in (text or ""))


def _norm_doi(doi: str) -> str:
    """Lowercase a DOI and strip any resolver prefix → bare '10.xxxx/...' form."""
    if not doi:
        return ""
    d = doi.strip().lower()
    for pre in _DOI_PREFIXES:
        if d.startswith(pre):
            d = d[len(pre):]
            break
    return d.strip()


def _norm_title(title: str) -> str:
    """Title dedup key: lowercase, alphanumerics only (drops spacing/punct)."""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """
    Rebuild plain abstract text from OpenAlex's abstract_inverted_index, which
    maps each word → the list of positions where it occurs.  Words are placed at
    their positions and joined in position order.  Truncated to 1500 chars.
    Returns "" for a null/empty index.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda t: t[0])
    return " ".join(w for _, w in positions)[:1500]


# ---------------------------------------------------------------------------
# Bilingual AI query planning
# ---------------------------------------------------------------------------

def _academic_fallback(question: str) -> Tuple[List[str], List[str]]:
    """
    Fallback (en_queries, ql_queries) used when the planner errors or yields no
    usable queries: search the raw question in its own language — a non-ASCII
    question goes to the QL bucket, an English/ASCII one to the EN bucket.
    """
    raw_q = question[:300]
    if _has_non_ascii(question):
        return [], [raw_q]
    return [raw_q], []


_TAG_RX = re.compile(r"^\s*(EN|QL)\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE)


def _parse_planned(raw: str, question: str) -> Tuple[List[str], List[str]]:
    """
    Tolerantly parse the planner output into (en_queries, ql_queries).

    Each well-formed line is "EN: <query>" or "QL: <query>".  Tags are stripped;
    malformed/garbage lines (empty, label rows, list markers, too short) are
    ignored; an untagged but plausible query line is routed by script
    (non-ASCII → QL, else EN).  Duplicates within a bucket are dropped.
    EN capped at 3, QL at 2.
    """
    en: List[str] = []
    ql: List[str] = []
    en_seen: set = set()
    ql_seen: set = set()

    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        m = _TAG_RX.match(s)
        if m:
            tag = m.group(1).upper()
            q = m.group(2).strip().strip('"').strip()
        else:
            # Untagged: drop obvious non-queries, else route by script.
            q = re.sub(r'^[\-\*•\d.)\s]+', "", s).strip().strip('"').strip()
            if q.endswith(":"):
                continue
            tag = "QL" if _has_non_ascii(q) else "EN"
        if len(q) < 3:
            continue
        key = q.lower()
        if tag == "EN":
            if key not in en_seen:
                en_seen.add(key)
                en.append(q)
        else:
            if key not in ql_seen:
                ql_seen.add(key)
                ql.append(q)

    return en[:3], ql[:2]


def _plan_academic_queries(question: str) -> Tuple[List[str], List[str]]:
    """
    Use a fast model to plan bilingual scholarly queries for the question.

    Returns (en_queries, ql_queries).  On a planner error, or when parsing
    yields no usable queries, falls back to searching the raw question in its
    own language (see _academic_fallback).  Never raises.
    """
    try:
        # local import avoids a circular dep at module load
        from ai_factory import call_model, _is_error_text
        current_date = datetime.now().strftime("%B %Y")
        prompt = ACADEMIC_QUERY_PLAN_PROMPT.format(
            current_date=current_date,
            question=question[:500],
        )
        raw = call_model("gemini", prompt, [], [])
        if _is_error_text(raw):
            return _academic_fallback(question)
        en, ql = _parse_planned(raw, question)
        if not en and not ql:
            return _academic_fallback(question)
        return en, ql
    except Exception:
        return _academic_fallback(question)


# ---------------------------------------------------------------------------
# Source 1 — OpenAlex (free; every discipline & language; reconstructed abstracts)
# ---------------------------------------------------------------------------

_OA_URL = "https://api.openalex.org/works"


def _search_openalex(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 8,
) -> Tuple[List[dict], bool]:
    """
    Search OpenAlex /works and return (unified_papers, errored).

    Year range uses the documented from_publication_date / to_publication_date
    filter (ISO dates).  A CONTACT_EMAIL env var, when set, is sent as `mailto`
    (the polite pool); omitted otherwise.  Abstracts are reconstructed from
    abstract_inverted_index.  ``errored`` is True only on HTTP failure.
    """
    params: dict = {"search": query, "per-page": limit}
    filters: List[str] = []
    if year_from:
        filters.append(f"from_publication_date:{int(year_from)}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{int(year_to)}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
    email = os.environ.get("CONTACT_EMAIL", "")
    if email:
        params["mailto"] = email

    try:
        resp = requests.get(
            _OA_URL, params=params, timeout=12,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        works = resp.json().get("results", [])

        papers: List[dict] = []
        for w in works:
            if not isinstance(w, dict):
                continue
            authorships = w.get("authorships") or []
            if not isinstance(authorships, list):
                authorships = []
            authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in authorships[:3] if isinstance(a, dict)
            ]
            source_obj = (w.get("primary_location") or {}).get("source") or {}
            doi_raw = w.get("doi") or ""
            papers.append({
                "title":     w.get("display_name") or "",
                "abstract":  _reconstruct_abstract(w.get("abstract_inverted_index")),
                "year":      str(w.get("publication_year") or ""),
                "authors":   ", ".join(a for a in authors if a),
                "url":       doi_raw or (w.get("primary_location") or {}).get("landing_page_url") or w.get("id") or "",
                "citations": w.get("cited_by_count"),
                "journal":   source_obj.get("display_name") or "",
                "source":    "OpenAlex",
                "language":  (w.get("language") or "") or "",
                "doi":       _norm_doi(doi_raw),
            })
        return papers, False
    except Exception:
        return [], True


# ---------------------------------------------------------------------------
# Source 2 — Semantic Scholar (free, English-centric)
# ---------------------------------------------------------------------------

_SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS     = "title,abstract,year,authors,citationCount,url,venue"


def _search_semantic_scholar(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 8,
) -> Tuple[List[dict], bool]:
    """
    Search Semantic Scholar and return (unified_papers, errored).

    ``errored`` is True only when the HTTP call raised / returned an error
    status — an empty list with errored=False means the API ran fine but found
    nothing (must NOT be reported as "unavailable").

    NOTE: doi="" — Semantic Scholar is currently rate-limiting (HTTP 429) and
    its externalIds.DOI shape could not be verified live this batch, so per the
    no-unverified-parsing rule we do not extract a DOI here.  Cross-source dedup
    falls back to normalized title for S2 papers.
    """
    params: dict = {"query": query, "fields": _SS_FIELDS, "limit": limit}
    if year_from or year_to:
        yf = year_from or 2000
        yt = year_to   or datetime.now().year
        params["year"] = f"{yf}-{yt}"

    try:
        resp = requests.get(
            _SS_SEARCH_URL, params=params, timeout=12,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        papers: List[dict] = []
        for p in data:
            if not isinstance(p, dict):
                continue
            authors_list = p.get("authors") or []
            if not isinstance(authors_list, list):
                authors_list = []
            papers.append({
                "title":     p.get("title", "") or "",
                "abstract":  p.get("abstract") or "",
                "year":      str(p.get("year", "") or ""),
                "authors":   ", ".join(a.get("name", "") for a in authors_list[:3] if isinstance(a, dict)),
                "url":       p.get("url") or "",
                "citations": p.get("citationCount"),
                "journal":   p.get("venue") or "",
                "source":    "Semantic Scholar",
                "language":  "",
                "doi":       "",
            })
        return papers, False
    except Exception:
        return [], True


# ---------------------------------------------------------------------------
# Source 3 — PubMed / NCBI Entrez (free, biomedical, English)
# ---------------------------------------------------------------------------

_PM_SEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PM_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_PM_FETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _fetch_pubmed_abstracts(ids: List[str]) -> dict:
    """
    Retrieve abstracts for a list of PMIDs via ONE efetch call → {pmid: text}.

    The efetch XML wraps each record in <PubmedArticle>; the abstract lives in
    one or more <AbstractText> elements under <Abstract>.  Structured abstracts
    carry a ``Label`` attribute per section (e.g. OBJECTIVE, RESULTS) — those
    labels are preserved and the sections concatenated, truncated to 1500 chars.
    Returns {} on any failure (graceful → callers keep title-only entries).
    """
    if not ids:
        return {}
    try:
        resp = requests.get(
            _PM_FETCH_URL,
            params={
                "db": "pubmed", "id": ",".join(ids),
                "retmode": "xml", "rettype": "abstract",
            },
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        out: dict = {}
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID")
            if not pmid:
                continue
            sections: List[str] = []
            for node in art.findall(".//Abstract/AbstractText"):
                txt = "".join(node.itertext()).strip()
                if not txt:
                    continue
                label = node.attrib.get("Label")
                sections.append(f"{label}: {txt}" if label else txt)
            if sections:
                out[pmid] = " ".join(sections)[:1500]
        return out
    except Exception:
        return {}


def _search_pubmed(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 6,
) -> Tuple[List[dict], bool]:
    """
    Search PubMed and return (unified_papers, errored).

    Pipeline: esearch (PMIDs) → esummary (metadata) → efetch (abstracts).
    ``errored`` is True only when esearch/esummary raised or returned an error
    status.  An efetch failure is non-fatal — papers are returned with empty
    abstracts (title-only entries).
    """
    search_params: dict = {
        "db": "pubmed", "term": query, "retmax": limit,
        "retmode": "json", "sort": "relevance",
    }
    if year_from or year_to:
        yf = year_from or 2000
        yt = year_to   or datetime.now().year
        search_params.update({
            "datetype": "pdat", "mindate": str(yf), "maxdate": str(yt),
        })

    try:
        resp = requests.get(_PM_SEARCH_URL, params=search_params, timeout=12)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return [], True

    if not ids:
        return [], False

    try:
        sum_resp = requests.get(
            _PM_SUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=12,
        )
        sum_resp.raise_for_status()
        result = sum_resp.json().get("result", {})
    except Exception:
        return [], True

    abstracts = _fetch_pubmed_abstracts(ids)   # graceful: {} on failure

    papers: List[dict] = []
    for pmid in ids:
        item = result.get(pmid, {})
        if not item:
            continue
        authors_raw = item.get("authors", [])
        papers.append({
            "title":     item.get("title", "") or "",
            "abstract":  abstracts.get(pmid, ""),
            "year":      str(item.get("pubdate", ""))[:4],
            "authors":   ", ".join(a.get("name", "") for a in authors_raw[:3]),
            "url":       f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "citations": None,
            "journal":   item.get("source", "") or "",
            "source":    "PubMed",
            "language":  "",
            "doi":       "",
        })
    return papers, False


# ---------------------------------------------------------------------------
# Source 4 — Google Scholar via Serper /scholar (paid key; any language)
# ---------------------------------------------------------------------------

_SCHOLAR_URL = "https://google.serper.dev/scholar"


def _split_pub_info(pub_info: str) -> Tuple[str, str]:
    """
    Best-effort split of Serper Scholar's publicationInfo string, e.g.
    "AR Barnosky, KK Hoddy… - Translational Research, 2014 - Elsevier"
    → ("AR Barnosky, KK Hoddy…", "Translational Research").
    """
    if not pub_info:
        return "", ""
    parts = [p.strip() for p in pub_info.split(" - ")]
    authors = parts[0] if parts else ""
    journal = parts[1] if len(parts) > 1 else ""
    journal = re.sub(r",\s*\d{4}\s*$", "", journal).strip()   # drop trailing ", YYYY"
    return authors, journal


def _search_scholar(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 8,
) -> Tuple[List[dict], bool]:
    """
    Search Google Scholar via Serper's /scholar endpoint and return
    (unified_papers, errored).  Handles ANY language natively.

    Year filtering uses Scholar's native as_ylo / as_yhi params (verified live
    to be honored by the endpoint).  The Serper snippet is stored in the
    abstract field (it is a snippet, not a full abstract).  ``errored`` is True
    when the key is missing or the HTTP call fails (caller emits a note and
    proceeds with the other sources — never fabricates scholarly results).
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return [], True

    body: dict = {"q": query, "num": limit}
    if year_from:
        body["as_ylo"] = int(year_from)
    if year_to:
        body["as_yhi"] = int(year_to)

    try:
        resp = requests.post(
            _SCHOLAR_URL, json=body,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=12,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])

        papers: List[dict] = []
        for r in organic:
            if not isinstance(r, dict):
                continue
            authors, journal = _split_pub_info(r.get("publicationInfo") or "")
            yr = r.get("year")
            cit = r.get("citedBy")
            papers.append({
                "title":     r.get("title", "") or "",
                "abstract":  (r.get("snippet") or "")[:1500],
                "year":      str(yr) if yr else "",
                "authors":   authors,
                "url":       r.get("link", "") or "",
                "citations": cit if isinstance(cit, int) else None,
                "journal":   journal,
                "source":    "Google Scholar",
                "language":  "",   # Serper Scholar does not report a language code
                "doi":       "",
            })
        return papers, False
    except Exception:
        return [], True


# ---------------------------------------------------------------------------
# Orchestration — route, search in parallel, merge, dedup, rank
# ---------------------------------------------------------------------------

# Display order for the per-source breakdown note.
_SOURCE_ORDER = ("OpenAlex", "Google Scholar", "Semantic Scholar", "PubMed")


def search_academic_papers(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    max_results: int = 12,
) -> Tuple[List[dict], List[str]]:
    """
    Multi-source, multilingual academic search.

    Plans bilingual queries, routes them across OpenAlex / Semantic Scholar /
    PubMed / Google Scholar, runs everything in one thread pool (≤ 12 engine
    requests), merges with DOI/title dedup (keeping the longer abstract), ranks
    (abstracts first, then citations, then recency) and trims to max_results.

    Returns (papers, notes).  ``notes`` carries a per-source breakdown line plus
    any degradation messages (source unavailable, planner fallbacks).  Paper
    dicts gain additive "language" (ISO or "") and "doi" ("" when absent) fields.
    """
    max_results = min(max_results, 12)
    notes: List[str] = []

    # ── Plan bilingual queries ──────────────────────────────────────────────
    en_q, ql_q = _plan_academic_queries(query)
    if (en_q, ql_q) == _academic_fallback(query):
        notes.append("academic query planner failed — searched the raw question")
    else:
        if not en_q:
            notes.append("planner produced no English queries")
        if _has_non_ascii(query) and not ql_q:
            notes.append("planner produced no original-language queries")

    # ── Routing matrix → job list (cap total engine requests at 12) ─────────
    #   Semantic Scholar : EN queries
    #   PubMed           : first EN query only (biomedical)
    #   OpenAlex         : EN + QL queries
    #   Google Scholar   : QL queries + first EN query
    jobs: List[Tuple[str, str]] = []
    for q in en_q:
        jobs.append(("Semantic Scholar", q))
    if en_q:
        jobs.append(("PubMed", en_q[0]))
    for q in en_q:
        jobs.append(("OpenAlex", q))
    for q in ql_q:
        jobs.append(("OpenAlex", q))
    for q in ql_q:
        jobs.append(("Google Scholar", q))
    if en_q:
        jobs.append(("Google Scholar", en_q[0]))
    jobs = jobs[:12]

    def _run(job: Tuple[str, str]):
        source, q = job
        if source == "Semantic Scholar":
            papers, err = _search_semantic_scholar(q, year_from, year_to, limit=6)
        elif source == "PubMed":
            papers, err = _search_pubmed(q, year_from, year_to, limit=4)
        elif source == "OpenAlex":
            papers, err = _search_openalex(q, year_from, year_to, limit=6)
        else:
            papers, err = _search_scholar(q, year_from, year_to, limit=6)
        return source, papers, err

    raw: List[dict] = []
    src_total: dict = defaultdict(int)
    src_err:   dict = defaultdict(int)
    src_count: dict = defaultdict(int)
    if jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs), 12)) as ex:
            for source, papers, err in ex.map(_run, jobs):
                src_total[source] += 1
                if err:
                    src_err[source] += 1
                src_count[source] += len(papers)
                raw.extend(papers)

    # ── Per-source failure notes (every job for a source errored) ───────────
    for source in _SOURCE_ORDER:
        if src_total[source] and src_err[source] == src_total[source]:
            notes.append(f"{source} unavailable")
    pm_papers = [p for p in raw if p.get("source") == "PubMed"]
    if pm_papers and not any((p.get("abstract") or "").strip() for p in pm_papers):
        notes.append("PubMed returned no abstracts")

    # ── Dedup across ALL sources ────────────────────────────────────────────
    # Primary key: normalized DOI when present on BOTH copies; else normalized
    # title.  Keep the copy with the longer abstract.
    deduped: List[dict] = []
    by_doi: dict = {}
    by_title: dict = {}
    for p in raw:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        doi = (p.get("doi") or "").strip().lower()
        tkey = _norm_title(title)

        target = None
        if doi and doi in by_doi:
            target = by_doi[doi]
        elif tkey and tkey in by_title:
            cand = by_title[tkey]
            cand_title_key = _norm_title(deduped[cand].get("title") or "")
            cand_doi = (deduped[cand].get("doi") or "").strip().lower()
            # Re-verify the candidate slot STILL carries this title: a prior
            # DOI-keyed replacement may have swapped in a different-titled paper,
            # leaving a stale by_title entry.  Only merge on a real title match,
            # and only when the pair does NOT both carry a (distinct) DOI.
            if cand_title_key == tkey and not (doi and cand_doi):
                target = cand

        if target is not None:
            if len(p.get("abstract") or "") > len(deduped[target].get("abstract") or ""):
                deduped[target] = p
            if doi:
                by_doi[doi] = target
            if tkey:
                by_title.setdefault(tkey, target)
        else:
            idx = len(deduped)
            deduped.append(p)
            if doi:
                by_doi[doi] = idx
            if tkey:
                by_title.setdefault(tkey, idx)

    merged_count = len(deduped)

    # ── Rank: abstracts first, then citations (None last), then recency ─────
    def _rank_key(p: dict):
        has_abstract = 0 if (p.get("abstract") or "").strip() else 1
        cit = p.get("citations")
        cit_sort = (0, -cit) if isinstance(cit, int) else (1, 0)
        try:
            yr = int(str(p.get("year") or "")[:4])
        except ValueError:
            yr = 0
        return (has_abstract, cit_sort, -yr)

    deduped.sort(key=_rank_key)
    kept = deduped[:max_results]

    # ── Per-source breakdown note (also surfaced in the Stage-0 UI line) ─────
    summary = ", ".join(
        f"{s}: {src_count[s]}" for s in _SOURCE_ORDER if src_total[s]
    )
    if summary:
        notes.insert(0, f"{summary} — merged {merged_count}, kept {len(kept)}")

    return kept, notes


# ---------------------------------------------------------------------------
# Literature context block
# ---------------------------------------------------------------------------

def _language_label(code: str) -> str:
    """ISO code → display label for non-English tagging ('he' → 'Hebrew')."""
    code = (code or "").strip().lower()
    return _LANG_NAMES.get(code, code.upper())


def build_literature_context_block(papers: List[dict]) -> str:
    """
    Format a list of paper dicts into a context block for prompt injection.
    Returns "" when the list is empty.

    Non-English papers are tagged with their language and every entry shows its
    source database; the model is told it may cite non-English works by their
    original-language titles.
    """
    if not papers:
        return ""

    lines = [
        "=== ACADEMIC LITERATURE CONTEXT ===",
        "The following papers were retrieved via live academic search across "
        "multiple databases (OpenAlex, Semantic Scholar, PubMed, Google Scholar).",
        "They are VERIFIED sources. Cite ALL sources as (Authors, Year). "
        "Non-English sources may be cited with their original-language titles.\n",
    ]

    for i, p in enumerate(papers, 1):
        title    = p.get("title")    or "Unknown Title"
        year     = p.get("year")     or "n.d."
        authors  = p.get("authors")  or "Unknown Authors"
        journal  = p.get("journal")  or ""
        url      = p.get("url")      or ""
        abstract = p.get("abstract") or ""
        cit      = p.get("citations")
        source   = p.get("source")   or ""
        lang     = (p.get("language") or "").strip().lower()

        entry = f"[{i}] {authors} ({year}). {title}."
        if journal:
            entry += f" {journal}."
        if cit is not None:
            entry += f" [Cited by {cit}]"
        if lang and lang != "en":
            entry += f" [{_language_label(lang)}]"
        if source:
            entry += f" [{source}]"
        if url:
            entry += f"\n    URL: {url}"
        if abstract:
            snippet = abstract[:350] + ("…" if len(abstract) > 350 else "")
            entry += f"\n    Abstract: {snippet}"
        lines.append(entry)

    lines.append("\n=== END OF LITERATURE CONTEXT ===")
    return "\n\n".join(lines)
