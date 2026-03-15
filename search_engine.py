"""
search_engine.py — Pre-Flight Live Market Data Fetcher
=======================================================
Fetches live search results via Serper.dev *before* the Council debate begins.

Two complementary public functions are provided:

  get_live_context(question)
      Targeted extraction: runs two precision searches (silver spot price,
      USD/ILS exchange rate), pulls numbers from snippets via regex, and
      returns a single authoritative line such as:
          "VERIFIED MARKET DATA (March 2026): Silver: $32.15/oz, USD/ILS: 3.68"
      This line is prepended to all model prompts as a mandatory instruction.

  get_live_market_data(question)
      Broad extraction: searches the question itself plus topic-specific
      sub-queries (gold, bitcoin, oil, etc.) and returns a multi-line
      Verified Context Block plus structured citation dicts for the UI.
      Silver/ILS are intentionally excluded — get_live_context covers those.

Together, logic_engine.py calls both so models receive:
  1. A concise, unambiguous data line they cannot ignore.
  2. Rich supporting snippets for additional context.

Configuration
-------------
Set SERPER_API_KEY in your .env file.
Every function degrades gracefully to a no-op when the key is absent.

Dependencies
------------
  requests  (pure HTTP, already in requirements.txt)
"""

from __future__ import annotations

import concurrent.futures
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import requests

from glossary import (
    STAGE0_QUERY_PLAN_PROMPT,
    STAGE0_RESEARCH_BLOCK_FOOTER,
    STAGE0_RESEARCH_BLOCK_HEADER,
)

_SERPER_URL = "https://google.serper.dev/search"

# Topic keywords that trigger a pre-flight search.
# Includes Hebrew terms so image-based coin/silver queries also fire.
_TRIGGER_KEYWORDS = {
    "price", "cost", "rate", "exchange", "market", "silver", "gold",
    "bitcoin", "btc", "ethereum", "eth", "crypto", "stock", "usd",
    "ils", "shekel", "dollar", "euro", "eur", "gbp", "pound", "coin",
    "worth", "value", "commodity", "interest", "inflation", "forex",
    "nasdaq", "dow", "s&p", "index", "bond", "yield", "oil", "crude",
    "gas", "platinum", "copper", "gdp", "salary", "wage", "tariff",
    # Numismatic / collector
    "numismatic", "collector", "mint", "graded", "pcgs", "ngc", "proof",
    "bullion", "premium", "mintage", "uncirculated", "ms70", "ms69",
    # Geopolitical / news
    "war", "conflict", "iran", "attack", "ceasefire", "hezbollah",
    "hamas", "gaza", "geopolit", "military", "defense", "sanction",
    "election", "news", "latest", "current", "today", "2026",
    # Investment / finance
    "invest", "investment", "profit", "return", "portfolio", "triple",
    # Hebrew — financial
    "מטבע", "שווי", "כסף", "זהב", "אספן", "אספנות", "מחיר", "ערך",
    "פרמיה", "נדיר", "מנטה", "מוטבע",
    # Hebrew — geopolitical / current events
    "מלחמה", "איראן", "חיזבאללה", "חמאס", "עזה", "גיאופוליטי",
    "בורסה", "מניה", "מניות", "השקעה", "להשקיע", "להרוויח",
    "תשואה", "רווח", "שוק", "כלכלה", "עדכני", "היום", "עכשיו",
}

# ---------------------------------------------------------------------------
# Regex patterns for extracting specific values from Serper snippets
# ---------------------------------------------------------------------------

# Silver spot price — matches e.g. "$32.15/oz", "$31.50 per troy ounce"
_SILVER_RX: List[re.Pattern] = [
    re.compile(r'\$\s*(\d{2,3}(?:\.\d+)?)\s*/?\s*(?:troy\s*)?(?:oz|ounce)', re.I),
    re.compile(r'silver\b[^$\n]{0,40}\$\s*(\d{2,3}(?:\.\d+)?)', re.I),
    re.compile(r'spot\b[^$\n]{0,20}\$\s*(\d{2,3}(?:\.\d+)?)', re.I),
    re.compile(r'(\d{2,3}\.\d+)\s*(?:per|/)\s*(?:troy\s*)?(?:oz|ounce)', re.I),
]

# USD/ILS rate — matches e.g. "1 USD = 3.68 ILS", "3.68 NIS"
_ILS_RX: List[re.Pattern] = [
    re.compile(r'1\s+(?:US\s+)?dollar[^=\n]{0,10}=\s*(\d+\.?\d*)\s*'
               r'(?:Israeli|New|ILS|NIS|shekel)', re.I),
    re.compile(r'USD\s*/?\s*ILS\s*[=:]\s*(\d+\.?\d*)', re.I),
    re.compile(r'1\s+USD\s*=\s*(\d+\.?\d*)\s*(?:ILS|NIS)', re.I),
    re.compile(r'(\d+\.\d+)\s*(?:ILS|NIS)', re.I),
]


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


def _extract_silver(snippets: List[str]) -> Optional[str]:
    """Scan snippets for a plausible silver spot price in USD/oz."""
    for text in snippets:
        for rx in _SILVER_RX:
            m = rx.search(text)
            if m:
                try:
                    val = float(m.group(1))
                    if 15.0 <= val <= 200.0:    # sanity range: $15–$200/oz
                        return f"${val:.2f}/oz"
                except ValueError:
                    pass
    return None


def _extract_ils(snippets: List[str]) -> Optional[str]:
    """Scan snippets for a plausible USD/ILS exchange rate."""
    for text in snippets:
        for rx in _ILS_RX:
            m = rx.search(text)
            if m:
                try:
                    val = float(m.group(1))
                    if 2.5 <= val <= 6.0:       # sanity range: 2.5–6.0 ILS/USD
                        return f"{val:.2f}"
                except ValueError:
                    pass
    return None


def _build_broad_queries(question: str) -> List[str]:
    """
    Return topic-specific supplementary queries for the broad search.
    Covers any question domain: geopolitics, finance, markets, tech, news.
    Silver/ILS are intentionally excluded — get_live_context handles those.
    """
    q = question.lower()
    queries = [question[:200]]      # always search the original question

    # ── Financial / commodity prices ──────────────────────────────────────
    if "gold" in q or "זהב" in q:
        queries.append("gold spot price USD today")
    if any(kw in q for kw in ("bitcoin", "btc", "ביטקוין")):
        queries.append("bitcoin price USD today")
    if any(kw in q for kw in ("ethereum", "eth")):
        queries.append("ethereum price USD today")
    if any(kw in q for kw in ("oil", "crude", "נפט")):
        queries.append("WTI crude oil price per barrel today")
    if any(kw in q for kw in ("euro", "eur")):
        queries.append("EUR USD exchange rate today")
    if "inflation" in q or "אינפלציה" in q:
        queries.append("US Israel inflation rate latest CPI data 2026")

    # ── Geopolitical / current events / news ──────────────────────────────
    _GEO_KW = (
        "war", "conflict", "מלחמה", "iran", "איראן",
        "middle east", "מזרח תיכון", "hezbollah", "חיזבאללה",
        "geopolit", "גיאופוליטי", "attack", "תקיפה", "ceasefire",
        "הסכם", "שביתת נשק", "negotiations", "משא ומתן",
        "hamas", "חמאס", "gaza", "עזה", "west bank", "גדה",
    )
    if any(kw in q for kw in _GEO_KW):
        queries.append("Middle East geopolitical situation latest news 2026")
        if "iran" in q or "איראן" in q:
            queries.append("Iran Israel military conflict latest news March 2026")
            queries.append("Iran war status 2026")
        if "hezbollah" in q or "חיזבאללה" in q:
            queries.append("Hezbollah Lebanon status 2026")
        if "gaza" in q or "עזה" in q or "hamas" in q or "חמאס" in q:
            queries.append("Gaza ceasefire deal latest news 2026")

    # ── Israeli / TASE stock market ────────────────────────────────────────
    _MARKET_IL_KW = (
        "tase", "בורסה", "ta-125", 'ת"א', "tel aviv stock",
        "elbit", "אלביט", "rafael", "רפאל", "iai", "תעשייה אווירית",
        "nice systems", "check point", "שוק ישראל", "מניה", "מניות",
    )
    if any(kw in q for kw in _MARKET_IL_KW):
        queries.append("Israel TASE Tel Aviv stock exchange performance 2026")
        queries.append("TA-125 index today 2026")
        if "elbit" in q or "אלביט" in q:
            queries.append("Elbit Systems ESLT stock price 2026")

    # ── Defense / military industry ────────────────────────────────────────
    _DEFENSE_KW = (
        "defense", "ביטחון", "military", "צבאי", "weapon", "נשק",
        "missile", "טיל", "drone", "כטב\"מ", "iron dome", "כיפת ברזל",
    )
    if any(kw in q for kw in _DEFENSE_KW):
        queries.append("Israel defense industry stocks performance 2026")
        queries.append("global defense sector market 2026")

    # ── Investment / personal finance ──────────────────────────────────────
    _INVEST_KW = (
        "invest", "investment", "להשקיע", "השקעה", "return", "תשואה",
        "portfolio", "תיק", "profit", "רווח", "make money", "להרוויח",
        "triple", "להכפיל", "grow money", "כסף מהיר",
    )
    if any(kw in q for kw in _INVEST_KW):
        queries.append("Israel investment opportunities 2026 market conditions")
        queries.append("best investments during geopolitical uncertainty 2026")

    # ── Crypto / digital assets ────────────────────────────────────────────
    _CRYPTO_KW = ("crypto", "קריפטו", "blockchain", "altcoin", "defi", "nft")
    if any(kw in q for kw in _CRYPTO_KW):
        queries.append("crypto market conditions 2026 regulatory status Israel")

    # ── Numismatic / coin collector searches ──────────────────────────────
    _COIN_KW = ("coin", "מטבע", "numismatic", "collector", "mint", "bullion",
                "proof", "uncirculated", "graded", "pcgs", "ngc",
                "אספן", "אספנות")
    if any(kw in q for kw in _COIN_KW):
        _KNOWN_COINS = [
            ("american eagle",   "American Eagle 1oz silver coin price value collector"),
            ("eagle",            "American Eagle 1oz silver coin price value collector"),
            ("maple leaf",       "Canadian Maple Leaf 1oz silver coin price value"),
            ("maple",            "Canadian Maple Leaf 1oz silver coin price value"),
            ("krugerrand",       "Krugerrand silver coin price value collector"),
            ("britannia",        "British Britannia 1oz silver coin price value"),
            ("libertad",         "Mexican Libertad silver coin price value"),
            ("philharmonic",     "Austrian Philharmonic silver coin price value"),
            ("panda",            "Chinese Silver Panda coin price collector value"),
            ("פנדה",             "Chinese Silver Panda coin price collector value"),
            ("כסף",              "1oz silver coin collector numismatic premium value today"),
        ]
        matched = False
        for keyword, search_query in _KNOWN_COINS:
            if keyword in q:
                queries.append(search_query)
                matched = True
                break
        if not matched:
            queries.append("1oz silver coin collector numismatic premium value today")

        if any(kw in q for kw in ("silver", "כסף", "1oz", "one ounce", "troy")):
            queries.append("silver coin collector premium vs spot price today")
            queries.append("silver bullion coin numismatic value grading premium 2026")

    # ── Technology / AI / software (defined after _COIN_KW to avoid NameError)
    _TECH_KW = (
        "ai", "artificial intelligence", "בינה מלאכותית", "gpt", "claude",
        "software", "tech stock", "semiconductor", "chip", "quantum",
    )
    if any(kw in q for kw in _TECH_KW) and not any(kw in q for kw in _COIN_KW):
        queries.append("AI technology sector market 2026 latest")

    return queries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_is_available() -> bool:
    """Return True when SERPER_API_KEY is present in the environment."""
    return bool(os.environ.get("SERPER_API_KEY"))


_SILVER_KEYWORDS = {
    "silver", "כסף", "bullion", "spot price", "troy", "oz", "ounce",
    "numismatic", "coin", "מטבע", "אספן", "אספנות",
}

_ILS_KEYWORDS = {
    "ils", "shekel", "שקל", "usd", "dollar", "דולר", "exchange rate",
    "forex", "currency", "מטבע חוץ", "שער", "שוק מט\"ח",
}


def get_live_context(question: str) -> str:
    """
    Run targeted searches (silver spot price and/or USD/ILS rate) only when
    the question is actually about those topics.  Returns a single authoritative
    data line, or an empty string when neither topic is relevant.
    """
    q_lower = question.lower()

    if not any(kw in q_lower for kw in _TRIGGER_KEYWORDS):
        return ""
    if not search_is_available():
        return ""

    wants_silver = any(kw in q_lower for kw in _SILVER_KEYWORDS)
    wants_ils    = any(kw in q_lower for kw in _ILS_KEYWORDS)

    # If neither commodity is relevant to this question, skip both searches.
    if not wants_silver and not wants_ils:
        return ""

    data_parts: List[str] = []

    # ── Silver spot price (only when question is about silver/metals/coins) ──
    if wants_silver:
        silver_hits  = _serper_search("silver spot price USD today", num=5)
        silver_texts = [r.get("snippet", "") + " " + r.get("title", "")
                        for r in silver_hits]
        silver_price = _extract_silver(silver_texts)
        if silver_price:
            data_parts.append(f"Silver: {silver_price}")

    # ── USD → ILS exchange rate (only when question involves ILS/currency) ──
    if wants_ils:
        ils_hits  = _serper_search("USD to ILS exchange rate today", num=5)
        ils_texts = [r.get("snippet", "") + " " + r.get("title", "")
                     for r in ils_hits]
        ils_rate  = _extract_ils(ils_texts)
        if ils_rate:
            data_parts.append(f"USD/ILS: {ils_rate}")

    if not data_parts:
        return ""

    date_str = datetime.now().strftime("%B %Y")
    return f"VERIFIED MARKET DATA ({date_str}): {', '.join(data_parts)}"


def _plan_queries_with_ai(question: str) -> List[str]:
    """
    Use Gemini Flash to generate targeted English search queries for any
    question.  This is the "query planning" step: instead of relying solely
    on hardcoded keyword matching, we let a fast model determine what facts
    need to be looked up before the debate begins.

    Returns up to 4 English queries, or [] if the model call fails or decides
    that no live search is needed (purely theoretical/definitional questions).
    Runs in ~1-2 s and degrades gracefully on any error.
    """
    if not search_is_available():
        return []
    try:
        from ai_factory import call_model        # local import avoids circular dep at module load
        prompt = STAGE0_QUERY_PLAN_PROMPT.format(question=question[:500])
        raw = call_model("gemini", prompt, [], [])
        if "NO_SEARCH_NEEDED" in raw.upper():
            return []
        queries = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 8
        ]
        return queries[:4]
    except Exception:
        return []


def get_live_market_data(question: str) -> Tuple[str, List[dict]]:
    """
    Run a broad pre-flight search for the question and return a multi-line
    context block plus structured citation dicts for the UI.

    Two complementary query sources are combined:
      1. AI-planned queries (_plan_queries_with_ai) — Gemini Flash decides
         whether live search is needed and generates targeted English queries
         for ANY question (returns [] for purely theoretical questions).
      2. Keyword-based sub-queries (_build_broad_queries) — deterministic,
         covers known domains (geopolitics, commodities, Israeli market …).

    All Serper.dev searches run in parallel via ThreadPoolExecutor.
    Silver/ILS targeted extraction is handled separately by get_live_context.
    """
    if not search_is_available():
        return "", []

    q_lower = question.lower()

    # ── Build query list ───────────────────────────────────────────────────
    # AI planner and keyword planner run in parallel.
    # AI planner is the primary gate: if it returns [] (NO_SEARCH_NEEDED),
    # we still add keyword queries for known domains as a safety net.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        ai_future = ex.submit(_plan_queries_with_ai, question)
        kw_future = ex.submit(_build_broad_queries, question)
        ai_queries = ai_future.result()
        kw_queries = kw_future.result()

    # If AI says NO_SEARCH_NEEDED and no keyword matches → skip search.
    # "No keyword matches" means _build_broad_queries returned only the raw
    # question itself (which is always appended as queries[0]).
    if not ai_queries and len(kw_queries) <= 1:
        return "", []

    # Merge: AI queries first (most targeted), then keyword queries.
    seen_q: set = {q.lower() for q in ai_queries}
    merged = list(ai_queries)
    for q in kw_queries:
        if q.lower() not in seen_q:
            merged.append(q)
            seen_q.add(q.lower())
    queries = merged

    # ── Short image-based coin query expansion ─────────────────────────────
    _IMG_COIN_KW = ("coin", "מטבע", "שווי", "worth", "value", "ערך", "כסף",
                    "אספן", "אספנות")
    if len(question.strip()) <= 80 and any(kw in q_lower for kw in _IMG_COIN_KW):
        queries += [
            "silver coin price value identification collector premium today",
            "how to identify silver coin value numismatic worth",
            "silver coin spot vs collector premium grading value 2026",
        ]

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
