"""
logic_engine.py — The 3-Stage Debate Orchestrator
===================================================
This module coordinates the full Council debate workflow:

  Pre-flight  Live Search         — fetch verified market data via Serper.dev.
  Stage 1     Parallel Inference  — all selected models answer independently.
  Stage 2     Cross-Critique      — every model reviews every other model's answer.
  Stage 3     Synthesis           — the Master Model writes the final answer.

Design notes
------------
- Pre-flight search runs *before* Stage 1 and injects a Verified Context Block
  into every prompt so all models work from the same live ground truth.
- Live citations now come exclusively from search_engine.py (pre-flight), not
  from Gemini's internal GoogleSearchRetrieval tool (removed from ai_factory).
- Uses concurrent.futures.ThreadPoolExecutor for Stage 1 and Stage 2 so that
  API calls run in parallel and the user waits as little as possible.
- Progress is reported via optional callbacks so the UI can update progress
  bars in real time.
- When image_bytes is provided all stages operate in Vision Expert Mode.
"""

import concurrent.futures
import contextvars
import json
import random
import re
from datetime import datetime
from typing import Callable, List, Optional, TypedDict

# NOTE: ThreadPoolExecutor does NOT propagate contextvars to its reused worker
# threads on ANY CPython version — regular 3.14's thread-inherit behaviour is
# opt-in only (and applies at thread creation, not per task), so every submit
# below is wrapped with contextvars.copy_context().run explicitly. This is what
# carries _ACADEMIC_MODE and the BYOK _SESSION_KEYS into the worker threads.

from glossary import (
    CONTEXTUAL_HALLUCINATION_TRIGGERS,
    HARD_HALLUCINATION_TRIGGERS,
    COUNCIL_CONSENSUS_PROMPT,
    DEVILS_ADVOCATE_CLAUSE,
    FALLBACK_MODEL_KEY,
    MASTER_MODEL_KEY,
    MODELS,
    PROMPT_CRITIQUE,
    PROMPT_DIALECTIC,
    PROMPT_INITIAL,
    PROMPT_EXTRACT_DISAGREEMENTS,
    PROMPT_DEBATE_OPENING,
    PROMPT_DEBATE_RESPONSE,
    PROMPT_REVERIFY_QUERIES,
    FOLLOWUP_CONTEXT_BLOCK,
    TEMPORAL_AUTHORITY_CLAUSE_GENERAL,
    UI_STAGE0_COMPLETE_GENERAL,
    UI_STAGE0_COMPLETE_NONE,
    UI_STAGE0_SEARCHING,
    UI_STAGE3B_NO_DEBATES,
    UI_SPINNER_STAGE3B,
    VISION_MODE_PROMPT,
    PROMPT_ACADEMIC_INITIAL,
    PROMPT_ACADEMIC_CRITIQUE,
    PROMPT_ACADEMIC_DIALECTIC,
    PROMPT_ACADEMIC_CONSENSUS,
    UI_ACADEMIC_SEARCHING,
    UI_ACADEMIC_FOUND,
    UI_ACADEMIC_NOT_FOUND,
    UI_ACADEMIC_STAGE0_LBL,
)
from ai_factory import call_model, call_model_with_citations, _is_error_text
from search_engine import get_live_market_data, search_is_available, _serper_search

# ---------------------------------------------------------------------------
# Academic Mode context variable — thread-safe, inherited by child threads
# ---------------------------------------------------------------------------
# Set to True in run_council_debate() when academic_mode=True.
# All stage functions check this to select the right prompt template.

import contextvars as _cv

_ACADEMIC_MODE: _cv.ContextVar[bool] = _cv.ContextVar("academic_mode", default=False)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[float, str], None]   # (fraction 0-1, status text)


class DisagreementPoint(TypedDict):
    point_id:   int
    summary:    str
    model_keys: List[str]
    excerpt:    str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vision_prefix(images: Optional[List[bytes]]) -> str:
    """Return VISION_MODE_PROMPT when one or more images are present, else empty string."""
    return VISION_MODE_PROMPT if images else ""


# ---------------------------------------------------------------------------
# Stage 1 — Parallel Inference
# ---------------------------------------------------------------------------

def _fetch_one_answer(
    model_key: str,
    question: str,
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
) -> tuple[str, str, List[dict]]:
    """Call a single model and return (model_key, answer_text, citations)."""
    template = PROMPT_ACADEMIC_INITIAL if _ACADEMIC_MODE.get() else PROMPT_INITIAL
    prompt = template.format(
        vision_prefix=_vision_prefix(images),
        verified_context=verified_context,
        question=question,
    )
    answer, citations = call_model_with_citations(
        model_key, prompt, images, images_mime,
        max_tokens=4096,
    )
    return model_key, answer, citations


def run_stage1_parallel_inference(
    active_keys: List[str],
    question: str,
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[dict[str, str], List[dict]]:
    """
    Query all active models simultaneously.

    Parameters
    ----------
    active_keys : list[str]
        Model keys to query (from glossary.MODELS).
    question : str
        The raw user question.
    verified_context : str
        Pre-flight search context block; injected into every prompt.
    image_bytes : bytes, optional
        Raw image bytes forwarded to every model call.
    image_mime : str
        MIME type of the image.
    progress_cb : callable, optional
        Called after each model completes with (fraction_done, status_text).

    Returns
    -------
    tuple[dict[str, str], list[dict]]
        (answers, citations) — citations are no longer populated from this
        stage (Gemini search tool removed); always [] here.
    """
    answers:   dict[str, str] = {}
    citations: List[dict]     = []
    total = len(active_keys)

    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
        futures = {}
        for key in active_keys:
            _ctx = contextvars.copy_context()
            fut = executor.submit(
                _ctx.run, _fetch_one_answer, key, question, verified_context, images, images_mime
            )
            futures[fut] = key

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            model_key, answer, model_citations = future.result()
            answers[model_key] = answer
            citations.extend(model_citations)   # will be [] per model now

            if progress_cb:
                label = MODELS[model_key]["label"]
                if _is_error_text(answer):
                    progress_cb(i / total, f"❌ {label} failed ({i}/{total})")
                else:
                    progress_cb(i / total, f"✅ {label} answered ({i}/{total})")

    return answers, citations


# ---------------------------------------------------------------------------
# Stage 2 — Cross-Critique
# ---------------------------------------------------------------------------

def _build_other_answers_block(
    answers: dict[str, str],
    exclude_key: str,
    labels: dict[str, str],
) -> str:
    """Format all answers except the one being reviewed into a readable block.

    Iterated in anonymised-label order (Expert A first) so the reviewer cannot
    infer identity from dict insertion order or from position in the block.
    """
    lines: List[str] = []
    for key in sorted((k for k in answers if k != exclude_key), key=lambda k: labels[k]):
        lines.append(f"### {labels[key]}\n{answers[key]}")
    return "\n\n---\n\n".join(lines) if lines else "_No other answers available._"


def _critique_one(
    reviewer_key: str,
    target_key: str,
    question: str,
    answers: dict[str, str],
    labels: dict[str, str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    devils_advocate: bool = False,
) -> tuple[str, str, str]:
    """
    One model reviews one other model's answer.

    All identities in the prompt are anonymised via *labels* (Expert A/B/C…).
    When *devils_advocate* is True the DEVIL'S-ADVOCATE clause is prepended so
    this reviewer argues the strongest opposing case (see run_stage2_cross_critique).

    Returns (reviewer_key, target_key, critique_text).
    """
    template = PROMPT_ACADEMIC_CRITIQUE if _ACADEMIC_MODE.get() else PROMPT_CRITIQUE
    prompt = template.format(
        vision_prefix=_vision_prefix(images),
        verified_context=verified_context,
        question=question,
        author_label=labels[target_key],
        author_answer=answers[target_key],
        other_answers=_build_other_answers_block(answers, target_key, labels),
    )
    if devils_advocate:
        prompt = DEVILS_ADVOCATE_CLAUSE + prompt
    critique = call_model(reviewer_key, prompt, images, images_mime)
    return reviewer_key, target_key, critique


def run_stage2_cross_critique(
    active_keys: List[str],
    question: str,
    answers: dict[str, str],
    labels: dict[str, str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[tuple[str, str], str]:
    """
    Each model reviews every other model's answer in parallel.

    Identities in every critique prompt are anonymised via *labels* (Expert A/B/C…).
    For each target exactly ONE reviewer — the next key after it in the
    anon-shuffled order, wrapping around — is assigned the rotating devil's-advocate
    role, so every answer faces at least one deliberately adversarial review.

    Returns
    -------
    dict[tuple[str, str], str]
        Mapping of (reviewer_key, target_key) -> critique_text.
    """
    pairs = [
        (reviewer, target)
        for reviewer in active_keys
        for target in active_keys
        if reviewer != target
    ]

    # Rotating devil's advocate: walk the anon-shuffled order; the reviewer that
    # immediately follows each target (wrapping around) argues against it.
    _ordered = sorted(active_keys, key=lambda k: labels[k])
    _n = len(_ordered)
    devils_advocate_for = {
        _ordered[i]: _ordered[(i + 1) % _n] for i in range(_n)
    } if _n >= 2 else {}

    critiques: dict[tuple[str, str], str] = {}
    total = len(pairs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(total, 8)) as executor:
        futures = {}
        for reviewer, target in pairs:
            _da = reviewer == devils_advocate_for.get(target)
            _ctx = contextvars.copy_context()
            fut = executor.submit(
                _ctx.run, _critique_one, reviewer, target, question, answers, labels,
                verified_context, images, images_mime, _da
            )
            futures[fut] = (reviewer, target)

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            reviewer_key, target_key, critique = future.result()
            critiques[(reviewer_key, target_key)] = critique

            if progress_cb:
                reviewer_label = MODELS[reviewer_key]["label"]
                target_label   = MODELS[target_key]["label"]
                progress_cb(
                    i / total,
                    f"✅ {reviewer_label} reviewed {target_label} ({i}/{total})",
                )

    return critiques


# ---------------------------------------------------------------------------
# Stage 2.4 — Critique verdict parsing  (convergence early-stop)
# Stage 2.5 — Mid-debate re-verification search (contested-claim grounding)
# ---------------------------------------------------------------------------
# These two helpers read the Stage-2 critiques *after* error-filtering and drive
# Batch-6's two new behaviours:
#   • _parse_verdict — extract the machine-readable "VERDICT:" line each reviewer
#     now appends, so the orchestrator can early-stop when all reviewers AGREE.
#   • _collect_reverify_claims + run_reverification — gather the "RE-VERIFY:" lines
#     reviewers flag on contested claims, search them live, and format a block
#     that is appended to the verified context for Stages 3, 3b and 4 ONLY.
#
# run_reverification lives here (not in search_engine) because it orchestrates
# call_model + a glossary prompt + search_engine together, mirroring the other
# multi-component stage functions in this module; it reuses
# search_engine._serper_search for the raw HTTP and search_is_available() for the
# key gate so the Serper plumbing stays in one place.

# Matches the mandatory final "VERDICT: …" line.  Tolerant: case-insensitive,
# allows leading markdown/bullet/emphasis chars and optional bold around the
# value, accepts "_" or " " inside the issue labels.  Searched (not anchored) so
# an inline verdict still parses; the caller takes the LAST match.
_VERDICT_RE = re.compile(
    r"(?i)VERDICT\s*:\s*\**\s*(AGREE|MINOR[ _]?ISSUES|MAJOR[ _]?ISSUES)"
)

# Matches a reviewer's "RE-VERIFY: <claim>" line.  Requires the literal
# "RE-VERIFY:" token, so the older audit checklist's "RE-VERIFICATION REQUIRED"
# text never matches.  Multiline + case-insensitive; tolerates leading
# whitespace / bullet / blockquote markers.
_REVERIFY_LINE_RE = re.compile(r"(?im)^[ \t>*_\-]*RE-VERIFY\s*:\s*(.+?)\s*$")


def _parse_verdict(critique_text: str) -> str:
    """Parse the machine-readable verdict from one critique.

    Returns one of "AGREE" / "MINOR_ISSUES" / "MAJOR_ISSUES".  The LAST occurrence
    wins (the prompt puts it on the final line).  A missing, garbled, or
    error-text critique is treated as MINOR_ISSUES — never AGREE — so a reviewer
    that forgot the line can never silently converge the debate.
    """
    if _is_error_text(critique_text):
        return "MINOR_ISSUES"
    matches = _VERDICT_RE.findall(critique_text)
    if not matches:
        return "MINOR_ISSUES"
    raw = matches[-1].upper().replace(" ", "_")
    if raw == "AGREE":
        return "AGREE"
    if raw == "MAJOR_ISSUES":
        return "MAJOR_ISSUES"
    return "MINOR_ISSUES"


def _collect_reverify_claims(critiques: dict[tuple[str, str], str]) -> List[str]:
    """Collect, strip and case-insensitively dedupe RE-VERIFY claims (cap 8).

    Scans every (non-error) critique for lines of the exact form
    ``RE-VERIFY: <claim>``.  Returns at most 8 unique claims in first-seen order.
    """
    claims: List[str] = []
    seen: set = set()
    for text in critiques.values():
        if _is_error_text(text):
            continue
        for m in _REVERIFY_LINE_RE.finditer(text):
            claim = m.group(1).strip()
            if not claim:
                continue
            key = claim.lower()
            if key in seen:
                continue
            seen.add(key)
            claims.append(claim)
            if len(claims) >= 8:
                return claims
    return claims


def run_reverification(
    question: str,
    critiques: dict[tuple[str, str], str],
) -> tuple[str, List[dict]]:
    """Re-verify Stage-2 contested claims with a live search.

    Steps
    -----
    a. Collect "RE-VERIFY:" claims from the critiques (dedup, cap 8).  None → ("", []).
    b. Search unavailable → ("", []).
    c. Ask Gemini Flash (PROMPT_REVERIFY_QUERIES) to turn the claims into 1-5
       English queries; on planner failure fall back to the claims themselves
       (first 5) as queries.
    d. Run the queries through search_engine._serper_search in parallel (num=3
       each), dedup by URL, keep the top 8 results, and format a STAGE 2.5 block
       plus Stage-0-style citation dicts.
    e. Return (block, citations).  Returns ("", []) when nothing is found.
    """
    claims = _collect_reverify_claims(critiques)
    if not claims:
        return "", []
    if not search_is_available():
        return "", []

    # ── Plan queries from the contested claims (fast model) ──────────────────
    claims_block = "\n".join(f"- {c}" for c in claims)
    current_date = datetime.now().strftime("%B %Y")
    prompt = PROMPT_REVERIFY_QUERIES.format(
        current_date=current_date,
        question=question[:500],
        claims_block=claims_block,
    )
    raw = call_model("gemini", prompt, [], [])
    if _is_error_text(raw):
        # Planner failed — verify the claims verbatim instead.
        queries = claims[:5]
    else:
        queries = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 8
        ][:5]
        if not queries:
            queries = claims[:5]

    # ── Run the queries against Serper.dev in parallel ───────────────────────
    seen_urls: set = set()
    all_results: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 5)) as ex:
        for hits in ex.map(lambda q: _serper_search(q, num=3), queries):
            for hit in hits:
                url = hit.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(hit)

    if not all_results:
        return "", []

    top = all_results[:8]

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
            entry = f"* {title}: {snippet}"
            if url:
                entry += f" / Source: {url}"
            lines.append(entry)

    if not lines:
        return "", []

    block = (
        "=== STAGE 2.5 — RE-VERIFICATION RESULTS (live search on contested claims) ===\n"
        + "\n".join(lines)
        + "\n\nWeigh the contested claims above against these live results: where a "
        "result confirms or refutes a contested claim, defer to the live result over "
        "training memory, and treat any claim these results contradict as unverified.\n"
    )
    return block, citations


# ---------------------------------------------------------------------------
# Shared formatters
# ---------------------------------------------------------------------------

def _format_all_answers(answers: dict[str, str], labels: dict[str, str]) -> str:
    """Render all initial answers into a single text block (anon-label order)."""
    blocks = []
    for key in sorted(answers, key=lambda k: labels[k]):
        blocks.append(f"### {labels[key]}\n{answers[key]}")
    return "\n\n---\n\n".join(blocks)


def _format_all_critiques(
    critiques: dict[tuple[str, str], str],
    labels: dict[str, str],
) -> str:
    """Render all critiques into a single text block (anon-label order)."""
    blocks = []
    for pair in sorted(critiques, key=lambda p: (labels[p[0]], labels[p[1]])):
        reviewer_key, target_key = pair
        blocks.append(
            f"### {labels[reviewer_key]} → {labels[target_key]}\n{critiques[pair]}"
        )
    return "\n\n---\n\n".join(blocks)


def _format_all_dialectic(dialectic: dict[str, str], labels: dict[str, str]) -> str:
    """Render all dialectic responses into a single text block (anon-label order)."""
    blocks = []
    for key in sorted(dialectic, key=lambda k: labels[k]):
        blocks.append(f"### {labels[key]} — Dialectic Response\n{dialectic[key]}")
    return "\n\n---\n\n".join(blocks) if blocks else "_No dialectic responses recorded._"


# ---------------------------------------------------------------------------
# Stage 3 — Dialectic Response (DSAD)
# ---------------------------------------------------------------------------

def _dialectic_one(
    model_key: str,
    question: str,
    initial_answer: str,
    critiques_against: List[str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
) -> tuple[str, str]:
    """
    One model reads all critiques directed AT it and responds:
    defending correct points or acknowledging and refining flawed ones.

    Returns (model_key, dialectic_response_text).
    """
    critique_block = "\n\n---\n\n".join(critiques_against)
    template = PROMPT_ACADEMIC_DIALECTIC if _ACADEMIC_MODE.get() else PROMPT_DIALECTIC
    prompt = template.format(
        vision_prefix=_vision_prefix(images),
        verified_context=verified_context,
        question=question,
        your_initial_answer=initial_answer,
        critique_of_your_answer=critique_block,
    )
    response = call_model(model_key, prompt, images, images_mime)
    return model_key, response


def run_stage3_dialectic(
    active_keys: List[str],
    question: str,
    answers: dict[str, str],
    critiques: dict[tuple[str, str], str],
    labels: dict[str, str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[str, str]:
    """
    Each model responds to the critiques directed at it in parallel.

    Critique authors in the prompt are anonymised via *labels* (Expert A/B/C…).

    Returns
    -------
    dict[str, str]
        Mapping of model_key → dialectic_response_text.
    """
    # Collect critiques received by each model (reviewer shown anonymised)
    critiques_by_target: dict[str, List[str]] = {k: [] for k in active_keys}
    for (reviewer_key, target_key), critique_text in critiques.items():
        if target_key in critiques_by_target:
            critiques_by_target[target_key].append(
                f"**Critique from {labels[reviewer_key]}:**\n{critique_text}"
            )

    # Only models that actually received critique need to respond
    responding = [k for k in active_keys if critiques_by_target.get(k)]
    if not responding:
        return {}

    dialectic: dict[str, str] = {}
    total = len(responding)

    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
        futures = {}
        for key in responding:
            _ctx = contextvars.copy_context()
            fut = executor.submit(
                _ctx.run, _dialectic_one, key, question, answers.get(key, ""),
                critiques_by_target[key], verified_context, images, images_mime,
            )
            futures[fut] = key
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            model_key, response = future.result()
            dialectic[model_key] = response

            if progress_cb:
                label = MODELS[model_key]["label"]
                progress_cb(i / total, f"💬 {label} responded to critique ({i}/{total})")

    return dialectic


# ---------------------------------------------------------------------------
# Stage 3b — Focused Debate
# ---------------------------------------------------------------------------

def _build_critiques_summary(
    critiques: dict[tuple[str, str], str],
    labels: dict[str, str],
) -> str:
    """Flatten each critique (anon-labelled, up to 2000 chars) for the extraction prompt.

    Raised from 150 → 2000 chars so the disagreement extractor sees the actual
    argument, not a truncated opening sentence.  Newlines are flattened to keep
    one entry per line.
    """
    lines: List[str] = []
    for pair in sorted(critiques, key=lambda p: (labels[p[0]], labels[p[1]])):
        reviewer_key, target_key = pair
        snippet = critiques[pair][:2000].replace("\n", " ")
        lines.append(f"{labels[reviewer_key]} → {labels[target_key]}: {snippet}")
    return "\n".join(lines) if lines else "_No critiques._"


def _build_dialectic_summary(dialectic: dict[str, str], labels: dict[str, str]) -> str:
    """Flatten each FULL dialectic response (anon-labelled, no truncation).

    Passes the complete dialectic text (newlines flattened) so the extractor can
    see whether a position was actually defended or conceded.
    """
    lines: List[str] = []
    for key in sorted(dialectic, key=lambda k: labels[k]):
        snippet = dialectic[key].replace("\n", " ")
        lines.append(f"{labels[key]}: {snippet}")
    return "\n".join(lines) if lines else "_No dialectic responses._"


def _extract_disagreement_points(
    question: str,
    active_keys: List[str],
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
    labels: dict[str, str],
) -> list[DisagreementPoint]:
    """
    Ask the master model to identify 2-3 genuine disagreement points.

    The model sees only anonymised expert labels (Expert A/B/C…) and returns
    those labels in an "experts" field; they are mapped back to real model keys
    here so each stored DisagreementPoint always carries REAL keys in
    ``model_keys`` (unknown labels are dropped, the >=2 rule is unchanged).
    Returns [] on any error or when models have converged.
    """
    # Anon labels the model may choose from, plus the reverse map back to keys.
    available_labels = sorted(labels[k] for k in active_keys)
    label_to_key     = {labels[k]: k for k in active_keys}

    prompt = PROMPT_EXTRACT_DISAGREEMENTS.format(
        question=question,
        critiques_summary=_build_critiques_summary(critiques, labels),
        dialectic_summary=_build_dialectic_summary(dialectic, labels),
        available_labels=str(available_labels),
    )
    raw = call_model(MASTER_MODEL_KEY, prompt)
    if raw.startswith("ERROR:"):
        return []
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        points = json.loads(cleaned)
        if not isinstance(points, list):
            return []
        valid: list[DisagreementPoint] = []
        for p in points:
            # Map anon labels back to real model keys; drop any unknown label.
            p["model_keys"] = [
                label_to_key[lbl] for lbl in p.get("experts", []) if lbl in label_to_key
            ]
            p.pop("experts", None)
            if len(p["model_keys"]) >= 2:
                valid.append(p)
        return valid[:3]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def _run_one_debate_round(
    point: DisagreementPoint,
    round_num: int,
    model_key: str,
    question: str,
    labels: dict[str, str],
    verified_context: str = "",
    opponent_openings: str = "",
) -> tuple[int, int, str, str]:
    """Execute one debate turn. Returns (point_id, round_num, model_key, text).

    Round 1 = opening position; round 2 = direct response to the opening
    positions of EVERY other expert on this point (block built by the caller).
    Both rounds carry the verified-context block and use anonymised labels.
    """
    your_label = labels[model_key]
    if round_num == 1:
        prompt = PROMPT_DEBATE_OPENING.format(
            verified_context=verified_context,
            your_label=your_label,
            question=question,
            point_summary=point["summary"],
            excerpt=point["excerpt"],
        )
    else:
        prompt = PROMPT_DEBATE_RESPONSE.format(
            verified_context=verified_context,
            your_label=your_label,
            question=question,
            point_summary=point["summary"],
            opponent_openings=opponent_openings,
        )
    text = call_model(model_key, prompt)
    return point["point_id"], round_num, model_key, text


def run_stage3b_focused_debate(
    active_keys: List[str],
    question: str,
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
    labels: dict[str, str],
    verified_context: str = "",
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """
    Identify genuine disagreements after Stage 3 and run focused 2-round exchanges.

    Identities in every debate prompt are anonymised via *labels* (Expert A/B/C…).
    *verified_context* (the live ground-truth block) is threaded into every round
    so debaters never lose the verified figures, and in Round 2 each debater
    answers the opening positions of ALL other experts on the point (not just one).

    Returns
    -------
    dict with keys:
        disagreement_points : list[DisagreementPoint]   — REAL model keys
        exchanges           : dict[(point_id, round, model_key), text]  — REAL keys
        skipped             : bool
    """
    empty = {"disagreement_points": [], "exchanges": {}, "skipped": True}

    if len(active_keys) < 2:
        return empty

    if progress_cb:
        progress_cb(0.0, "🔍 Extracting disagreement points …")

    points = _extract_disagreement_points(question, active_keys, critiques, dialectic, labels)

    if not points:
        if progress_cb:
            progress_cb(1.0, UI_STAGE3B_NO_DEBATES)
        return empty

    exchanges: dict[tuple[int, int, str], str] = {}
    total_rounds = sum(len(p["model_keys"]) * 2 for p in points)
    completed    = 0

    for point in points:
        m_keys = point["model_keys"]
        pid    = point["point_id"]

        # Round 1 — opening positions (parallel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(m_keys)) as ex:
            r1_futures = {}
            for mk in m_keys:
                _ctx = contextvars.copy_context()
                fut = ex.submit(
                    _ctx.run, _run_one_debate_round, point, 1, mk, question,
                    labels, verified_context,
                )
                r1_futures[fut] = mk
            for fut in concurrent.futures.as_completed(r1_futures):
                p_id, rnd, mk, text = fut.result()
                exchanges[(p_id, rnd, mk)] = text
                completed += 1
                if progress_cb:
                    label = MODELS[mk]["label"]
                    progress_cb(completed / total_rounds,
                                f"⚔️ {label} — Point {pid} Round 1 done")

        # Round 2 — direct responses (needs Round 1, parallel across models).
        # Each debater answers the opening positions of EVERY other expert on
        # this point — not just one — so 3+-way disagreements are fully engaged.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(m_keys)) as ex:
            r2_futures = {}
            for mk in m_keys:
                _others = sorted(
                    (k for k in m_keys if k != mk), key=lambda k: labels[k]
                )
                _blocks = [
                    f"### {labels[k]}\n{exchanges[(pid, 1, k)]}"
                    for k in _others if exchanges.get((pid, 1, k))
                ]
                opponent_openings = (
                    "\n\n".join(_blocks) if _blocks
                    else "_No opposing opening positions were recorded._"
                )
                _ctx = contextvars.copy_context()
                fut = ex.submit(
                    _ctx.run, _run_one_debate_round, point, 2, mk, question,
                    labels, verified_context, opponent_openings,
                )
                r2_futures[fut] = mk
            for fut in concurrent.futures.as_completed(r2_futures):
                p_id, rnd, mk, text = fut.result()
                exchanges[(p_id, rnd, mk)] = text
                completed += 1
                if progress_cb:
                    label = MODELS[mk]["label"]
                    progress_cb(completed / total_rounds,
                                f"⚔️ {label} — Point {pid} Round 2 done")

    if progress_cb:
        progress_cb(1.0, f"✅ Focused debate complete — {len(points)} point(s) debated")

    return {
        "disagreement_points": points,
        "exchanges":           exchanges,
        "skipped":             False,
    }


def _format_focused_debate(
    focused_debate: Optional[dict],
    labels: dict[str, str],
) -> str:
    """Render focused debate exchanges into a readable block for the consensus prompt.

    Speakers are shown by anonymised label (Expert A/B/C…) in anon-label order.
    """
    if not focused_debate or focused_debate.get("skipped", True):
        return "_No focused debate was conducted — models converged after Stage 3._"
    points    = focused_debate.get("disagreement_points", [])
    exchanges = focused_debate.get("exchanges", {})
    if not points:
        return "_No significant disagreements detected._"
    blocks: List[str] = []
    for point in points:
        pid     = point["point_id"]
        summary = point["summary"]
        m_keys  = sorted(point["model_keys"], key=lambda k: labels[k])
        lines   = [f"### Point {pid}: {summary}"]
        for rnd, rnd_label in [(1, "Opening"), (2, "Response")]:
            for mk in m_keys:
                text = exchanges.get((pid, rnd, mk), "")
                if text:
                    lines.append(f"**{labels[mk]} — {rnd_label}:**\n{text}")
        blocks.append("\n\n".join(lines))
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Heartbeat wrapper for long single blocking calls
# ---------------------------------------------------------------------------

def _run_with_heartbeat(thunk, progress_cb=None, label="", start_frac=0.1):
    """Run a blocking ``thunk`` in a worker thread, emitting a progress heartbeat
    every ~3 s so Streamlit Cloud's WebSocket never goes silent long enough to
    drop and rerun the script.  Used for the synthesis call (a 16K-token academic
    report can take 1-2 minutes as one silent block).  ``copy_context().run``
    carries _ACADEMIC_MODE and the BYOK session keys into the worker thread.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(contextvars.copy_context().run, thunk)
        beats = 0
        while not fut.done():
            try:
                concurrent.futures.wait([fut], timeout=3.0)
            except Exception:
                pass
            if not fut.done() and progress_cb:
                beats += 1
                progress_cb(min(start_frac + 0.05 * beats, 0.9), f"{label} ({beats * 3}s)")
        return fut.result()


# ---------------------------------------------------------------------------
# Stage 4 — Consensus Synthesis (DSAD)
# ---------------------------------------------------------------------------

def run_stage4_consensus(
    question: str,
    answers: dict[str, str],
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
    labels: dict[str, str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    focused_debate: Optional[dict] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[str, bool]:
    """
    The Mediator (Master Model) reads the full DSAD transcript and writes
    the Final Certified Answer.  Falls back to FALLBACK_MODEL_KEY on error.

    Any live Stage 0 data (search snippets, citations) is already embedded in
    ``verified_context`` — the consensus prompt is fully domain-neutral.

    Returns
    -------
    tuple[str, bool]
        (final_answer_text, fallback_was_used)
    """
    master_label   = MODELS[MASTER_MODEL_KEY]["label"]
    fallback_label = MODELS[FALLBACK_MODEL_KEY]["label"]

    if progress_cb:
        progress_cb(0.1, f"🏛️ {master_label} (Mediator) is synthesising …")

    if _ACADEMIC_MODE.get():
        prompt = PROMPT_ACADEMIC_CONSENSUS.format(
            vision_prefix=_vision_prefix(images),
            verified_context=verified_context,
            question=question,
            all_answers_block=_format_all_answers(answers, labels),
            all_critiques_block=_format_all_critiques(critiques, labels),
            all_dialectic_block=_format_all_dialectic(dialectic, labels),
            focused_debate_block=_format_focused_debate(focused_debate, labels),
        )
    else:
        prompt = COUNCIL_CONSENSUS_PROMPT.format(
            vision_prefix=_vision_prefix(images),
            verified_context=verified_context,
            question=question,
            all_answers_block=_format_all_answers(answers, labels),
            all_critiques_block=_format_all_critiques(critiques, labels),
            all_dialectic_block=_format_all_dialectic(dialectic, labels),
            focused_debate_block=_format_focused_debate(focused_debate, labels),
        )

    # Claude (MASTER) is always the synthesizer.
    # For academic mode we allow 16K tokens (full 3-part report); regular mode 8K.
    # timeout is sized generously: 16K tokens at ~30-40 tok/s ≈ 400-530s → 600s cap.
    _is_academic       = _ACADEMIC_MODE.get()
    _claude_tokens     = 16000 if _is_academic else 8192
    _gemini_tokens     = 16000
    _synthesis_timeout = 600.0 if _is_academic else 420.0

    final_answer, _cit = _run_with_heartbeat(
        lambda: call_model_with_citations(
            MASTER_MODEL_KEY, prompt, images, images_mime,
            max_tokens=_claude_tokens, request_timeout=_synthesis_timeout,
        ),
        progress_cb=progress_cb,
        label=f"🏛️ {master_label} (Mediator) is synthesising",
        start_frac=0.15,
    )
    fallback_used = False

    if not final_answer or final_answer.startswith("ERROR:"):
        if progress_cb:
            progress_cb(
                0.5,
                f"⚠️ {master_label} failed — switching to {fallback_label} …",
            )
        fb_answer, _fb_cit = _run_with_heartbeat(
            lambda: call_model_with_citations(
                FALLBACK_MODEL_KEY, prompt, images, images_mime,
                max_tokens=_gemini_tokens, request_timeout=_synthesis_timeout,
            ),
            progress_cb=progress_cb,
            label=f"🏛️ {fallback_label} (fallback) is synthesising",
            start_frac=0.55,
        )
        if fb_answer and not fb_answer.startswith("ERROR:"):
            final_answer  = fb_answer
            fallback_used = True

    if progress_cb:
        progress_cb(1.0, "✅ Consensus certified!")

    return final_answer, fallback_used


# ---------------------------------------------------------------------------
# High-level orchestrator (used by app.py)
# ---------------------------------------------------------------------------

def run_council_debate(
    active_keys: List[str],
    question: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    previous_context: Optional[dict] = None,
    code_context: str = "",
    academic_mode: bool = False,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    stage0_cb: Optional[ProgressCallback] = None,
    stage1_cb: Optional[ProgressCallback] = None,
    stage2_cb: Optional[ProgressCallback] = None,
    stage3_cb: Optional[ProgressCallback] = None,
    stage3b_cb: Optional[ProgressCallback] = None,
    stage4_cb: Optional[ProgressCallback] = None,
) -> dict:
    """
    Run the complete 4-stage DSAD debate and return all artefacts.

    Parameters
    ----------
    active_keys : list[str]
        Model keys selected by the user (Master Model added automatically).
    question : str
        The user's question.
    image_bytes : bytes, optional
        Raw image bytes.  When provided all stages use Vision Expert Mode.
    image_mime : str
        MIME type of the uploaded image (default "image/jpeg").
    code_context : str, optional
        Pre-formatted code context block from project_reader.scan_project().
        When provided it is injected as the very first item in verified_context
        so every model reads the source files before answering.
    stage1_cb … stage4_cb : callable, optional
        Per-stage progress callbacks: (fraction: float, status: str) -> None.

    Returns
    -------
    dict with keys:
        "question"            str
        "answers"             dict[str, str]
        "critiques"           dict[tuple[str,str], str]
        "dialectic"           dict[str, str]
        "reliability_scores"  dict[str, int]   — 0-100, derived from dialectic
        "final_answer"        str
        "fallback_used"       bool
        "citations"           list[dict]        — from pre-flight Serper.dev search
        "has_image"           bool
    """
    all_keys    = list(dict.fromkeys([MASTER_MODEL_KEY] + active_keys))
    images      = images      or []
    images_mime = images_mime or []

    # ── Set academic mode context variable (inherited by all child threads) ──
    _ACADEMIC_MODE.set(academic_mode)

    # ── Stage 0: Pre-flight search (live market data + optional academic lit) ─
    _s0_label = UI_ACADEMIC_SEARCHING if academic_mode else UI_STAGE0_SEARCHING
    if stage0_cb:
        stage0_cb(0.0, _s0_label)

    # The pre-flight searches are slow (the academic multi-source search can take
    # 15-60s) and otherwise run as ONE silent synchronous block — long enough for
    # Streamlit Cloud's WebSocket to go idle, drop, and rerun the whole script
    # ("started analysing, then jumped back to the start").  Fix: run the academic
    # and broad searches concurrently in worker threads and emit a heartbeat to the
    # UI every few seconds so the connection never goes silent.  copy_context()
    # carries _ACADEMIC_MODE and the BYOK session keys into the worker threads.
    academic_papers: list = []
    academic_notes: List[str] = []   # honest degradation messages → stage_errors

    def _do_academic_search():
        if not academic_mode:
            return [], []
        try:
            from academic_search import search_academic_papers
            return search_academic_papers(question, year_from, year_to, max_results=12)
        except Exception:
            return [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _s0_ex:
        _fut_acad  = _s0_ex.submit(contextvars.copy_context().run, _do_academic_search)
        _fut_broad = _s0_ex.submit(contextvars.copy_context().run,
                                   get_live_market_data, question)
        _beats = 0
        while not (_fut_acad.done() and _fut_broad.done()):
            try:
                concurrent.futures.wait(
                    [_fut_acad, _fut_broad], timeout=3.0,
                    return_when=concurrent.futures.ALL_COMPLETED,
                )
            except Exception:
                pass
            if not (_fut_acad.done() and _fut_broad.done()) and stage0_cb:
                _beats += 1
                # Fraction creeps toward (not to) 1.0 so the bar keeps moving.
                stage0_cb(min(0.1 + 0.08 * _beats, 0.9), f"{_s0_label} ({_beats * 3}s)")
        academic_papers, academic_notes = _fut_acad.result()
        broad_block, preflight_citations = _fut_broad.result()

    # Stage 0 complete — log the live status to the UI
    if stage0_cb:
        if academic_mode:
            if academic_papers:
                # The per-source breakdown is the note containing "merged"/"kept"
                # (see search_academic_papers); surface it on the Stage-0 line.
                _src_breakdown = next(
                    (nt for nt in academic_notes if "merged" in nt and "kept" in nt),
                    "מקורות מרובים",
                )
                _stage0_msg = UI_ACADEMIC_FOUND.format(
                    n=len(academic_papers), sources=_src_breakdown
                )
            else:
                _stage0_msg = UI_ACADEMIC_NOT_FOUND
        else:
            if broad_block:
                _n_results = broad_block.count("\n•") or broad_block.count("\n-") or "some"
                _stage0_msg = UI_STAGE0_COMPLETE_GENERAL.format(n=_n_results)
            else:
                _stage0_msg = UI_STAGE0_COMPLETE_NONE
        stage0_cb(1.0, _stage0_msg)

    current_date  = datetime.now().strftime("%A, %B %d, %Y")
    context_parts: List[str] = []

    # Temporal Authority Clause: placed FIRST so it is the first thing every
    # model reads.
    #   - Live search results present → general clause anchors the date and
    #     directs every model to use the search snippets below as primary data.
    #   - No search data → just anchor the date.
    if broad_block:
        context_parts.append(
            TEMPORAL_AUTHORITY_CLAUSE_GENERAL.format(current_date=current_date)
        )
        context_parts.append(broad_block)
    else:
        context_parts.append(f"TODAY'S DATE: {current_date}")

    # ── Academic literature block (injected when academic_mode is True) ───────
    if academic_papers:
        from academic_search import build_literature_context_block
        lit_block = build_literature_context_block(academic_papers)
        if lit_block:
            context_parts.append(lit_block)

    # ── Code Review: inject project files as the primary verified source
    if code_context:
        context_parts.insert(0, code_context)

    # ── Follow-up: prepend prior-debate context so every model reads it first
    if previous_context:
        prev_q      = previous_context.get("question", "")
        prev_answer = previous_context.get("final_answer", "")
        # Use str.replace instead of .format() — the previous answer often
        # contains curly braces (JSON, formulas, tables) that would cause
        # a KeyError if passed through Python's str.format().
        fup_block = (
            FOLLOWUP_CONTEXT_BLOCK
            .replace("{prev_question}", prev_q)
            .replace("{prev_answer}",  prev_answer[:8000])
        )
        context_parts.insert(0, fup_block)

    verified_context = "\n\n".join(context_parts)

    # ── Stage 1: Parallel Thesis ───────────────────────────────────────────
    answers, _s1_cit = run_stage1_parallel_inference(
        all_keys, question, verified_context,
        images, images_mime, progress_cb=stage1_cb,
    )

    # Split successful answers from failures. Failed models leave the debate:
    # they are never critiqued, never critique others, never debate, and never
    # appear in the synthesis transcript — but their error text is kept in
    # failed_models so the UI can show which models dropped out and why.
    ok_answers    = {k: v for k, v in answers.items() if not _is_error_text(v)}
    failed_models = {k: v for k, v in answers.items() if _is_error_text(v)}
    ok_keys       = list(ok_answers.keys())
    stage_errors: List[str] = []

    # Surface any Stage 0 academic-search degradations honestly in the UI.
    if academic_notes:
        stage_errors.extend(academic_notes)

    # ── Per-debate anonymisation map ────────────────────────────────────────
    # Shuffle the surviving models and assign neutral "Expert A/B/C…" labels.
    # These labels are used ONLY inside prompt text sent to the models, to strip
    # brand identity AND fixed dict-order position bias out of the debate.  The
    # results dict below keeps REAL model keys everywhere — the UI and saved
    # history depend on them — so anonymisation never leaks past the prompts.
    _shuffled = ok_keys.copy()
    random.shuffle(_shuffled)
    anon_labels = {key: f"Expert {chr(65 + i)}" for i, key in enumerate(_shuffled)}

    # Every model failed in Stage 1 → no usable answer exists to debate.
    # Skip Stages 2-4 entirely and return immediately with a hard-failure marker.
    if not ok_answers:
        if stage4_cb:
            stage4_cb(1.0, "❌ All models failed in Stage 1 — no answer produced")
        return {
            "question":           question,
            "answers":            ok_answers,
            "critiques":          {},
            "dialectic":          {},
            "focused_debate":     {"disagreement_points": [], "exchanges": {}, "skipped": True},
            "reliability_scores": {},
            "anon_labels":        anon_labels,
            "final_answer":       "",
            "fallback_used":      False,
            "citations":          [],
            "has_images":         len(images) > 0,
            "academic_mode":      academic_mode,
            "academic_papers":    academic_papers,
            "failed_models":      failed_models,
            "stage_errors":       stage_errors,
            "debate_failed":      True,
            "critique_verdicts":      {},
            "converged_early":        False,
            "reverification_block":   "",
            "reverification_claims":  [],
        }

    # ── Stage 2: Adversarial Audit ─────────────────────────────────────────
    # Only the models that produced a valid answer participate from here on.
    critiques: dict[tuple[str, str], str] = {}
    if len(ok_keys) >= 2:
        critiques = run_stage2_cross_critique(
            ok_keys, question, ok_answers, anon_labels, verified_context,
            images, images_mime, progress_cb=stage2_cb,
        )
        # Drop any critique that itself came back as an error, so a failed
        # critique never poisons Stage 3 or the synthesis transcript.
        for _rk, _tk in [k for k, v in critiques.items() if _is_error_text(v)]:
            stage_errors.append(
                f"Stage 2: {MODELS[_rk]['label']} → {MODELS[_tk]['label']} critique failed"
            )
            del critiques[(_rk, _tk)]

    # ── Stage 2.4: Machine-readable critique verdicts (Batch 6) ────────────
    # Each surviving critique ends with "VERDICT: AGREE|MINOR_ISSUES|MAJOR_ISSUES".
    # Parse them (last occurrence wins; missing/garbled → MINOR_ISSUES, still
    # counted) to drive the convergence early-stop and the per-model micro-skip.
    # Stored under string keys ("reviewer→target") so the dict is already
    # JSON-safe and round-trips through history_manager._serialize_results /
    # _deserialize_results untouched — no tuple-key special-casing needed.
    _verdict_by_pair: dict[tuple[str, str], str] = {
        pair: _parse_verdict(text) for pair, text in critiques.items()
    }
    critique_verdicts: dict[str, str] = {
        f"{rk}→{tk}": v for (rk, tk), v in _verdict_by_pair.items()
    }
    # Converge only when there is a real vote (>=2 ok models AND at least one
    # parsed verdict) and EVERY parsed verdict is AGREE.  The bool() guard stops
    # an empty verdict set (e.g. all critiques errored out) from vacuously
    # converging the debate.
    converged_early = (
        bool(_verdict_by_pair)
        and len(ok_keys) >= 2
        and all(v == "AGREE" for v in _verdict_by_pair.values())
    )
    # Per-model micro-skip: a model whose every RECEIVED critique is AGREE has
    # nothing to defend, so it is dropped from the Stage-3 dialectic only (it
    # still participates in Stage 3b and Stage 4 as usual).
    _received_verdicts: dict[str, List[str]] = {k: [] for k in ok_keys}
    for (_rk, _tk), _v in _verdict_by_pair.items():
        if _tk in _received_verdicts:
            _received_verdicts[_tk].append(_v)
    _fully_agreed_targets = {
        k for k, vs in _received_verdicts.items() if vs and all(v == "AGREE" for v in vs)
    }

    # Additive Batch-6 result keys — defaults cover every skip path below.
    reverification_block:     str         = ""
    reverification_claims:    List[str]   = []
    reverification_citations: List[dict]  = []

    if converged_early:
        # ── Convergence early-stop ─────────────────────────────────────────
        # All reviewers AGREED → skip Stage 3 (dialectic), the Stage 2.5
        # re-verification search, and Stage 3b (focused debate) entirely.
        # reliability_scores stays {} (the natural no-dialectic result — the UI
        # already skips the Reliability Dashboard when dialectic is empty), and
        # Stage 4 synthesises directly from the theses + agreeing critiques.
        dialectic = {}
        reliability_scores = {}
        focused_debate = {"disagreement_points": [], "exchanges": {}, "skipped": True}
        verified_context_post2 = verified_context
        if stage3_cb:
            stage3_cb(1.0, "🤝 Council converged — no disagreements")
        if stage3b_cb:
            stage3b_cb(1.0, "🤝 Council converged — no disagreements")
    else:
        # ── Stage 2.5: Mid-debate re-verification of contested claims ──────
        # Reviewers may have flagged factual/numeric/time-sensitive claims with
        # "RE-VERIFY:" lines.  Search them live and append the results to the
        # context fed to Stages 3, 3b and 4 ONLY (Stage 1 already ran).  There is
        # no dedicated Stage-2.5 callback, so reuse stage3_cb at fraction 0.0 —
        # but only when a search will actually run (claims flagged + key present).
        _flagged_claims = _collect_reverify_claims(critiques)
        _will_search    = bool(_flagged_claims) and search_is_available()
        if _will_search and stage3_cb:
            stage3_cb(0.0, "🔎 Re-verifying contested claims …")
        reverification_block, reverification_citations = run_reverification(
            question, critiques
        )
        if reverification_block:
            reverification_claims  = _flagged_claims
            verified_context_post2 = verified_context + "\n\n" + reverification_block
        else:
            verified_context_post2 = verified_context

        # ── Stage 3: Dialectic Response ────────────────────────────────────
        dialectic = {}
        if critiques:
            # Micro-skip: exclude models whose every received critique is AGREE.
            _dialectic_keys = [k for k in ok_keys if k not in _fully_agreed_targets]
            dialectic = run_stage3_dialectic(
                _dialectic_keys, question, ok_answers, critiques, anon_labels,
                verified_context_post2, images, images_mime, progress_cb=stage3_cb,
            )
            # Skip any dialectic response that came back as an error.
            for _mk in [k for k, v in dialectic.items() if _is_error_text(v)]:
                stage_errors.append(
                    f"Stage 3: {MODELS[_mk]['label']} dialectic response failed"
                )
                del dialectic[_mk]

        # Reliability scores: deduct ONLY for rejecting verified Stage 0 data.
        #
        # Honest concession to a valid peer critique is NEVER penalised — it is the
        # exact intellectually-honest behaviour the dialectic stage mandates, so the
        # old "retraction" deduction was removed (it punished honesty and rewarded
        # stubbornness).  The two remaining categories both require live Stage 0
        # search data to be present:
        #   1. Contextual Hallucination deductions (-40 each): model reframed the
        #      actual Stage 0 live search data as hypothetical/unconfirmed.
        #   2. Hard Hallucination deductions (-50 each): explicit "if Stage 0 is
        #      correct"-style reframing of live data.
        # IMPORTANT: these deductions apply ONLY when live search context is present
        # (broad_block non-empty — real results were retrieved).  When no live data
        # exists, saying "I cannot verify" is correct behaviour and must NOT be
        # penalised.
        reliability_scores = {}
        for key, resp in dialectic.items():
            # Defensive: a failed dialectic response must never receive a score
            # (also guards resp.lower() against a None/blank slipping through).
            if _is_error_text(resp):
                continue
            low = resp.lower()

            # Only penalise for rejecting Stage 0 data when live data actually exists.
            if broad_block:
                hallucination_hits = sum(
                    1 for t in CONTEXTUAL_HALLUCINATION_TRIGGERS if t in low
                )
                hard_hits = sum(1 for t in HARD_HALLUCINATION_TRIGGERS if t in low)
            else:
                hallucination_hits = 0
                hard_hits          = 0

            reliability_scores[key] = max(
                0,
                100 - hallucination_hits * 40 - hard_hits * 50,
            )

        # ── Stage 3b: Focused Debate ───────────────────────────────────────
        # Only the non-master models that survived Stage 1 debate — master mediates.
        debate_keys = [k for k in ok_keys if k != MASTER_MODEL_KEY]
        focused_debate = run_stage3b_focused_debate(
            active_keys=debate_keys,
            question=question,
            critiques=critiques,
            dialectic=dialectic,
            labels=anon_labels,
            verified_context=verified_context_post2,
            progress_cb=stage3b_cb,
        )

    # ── Stage 4: Consensus Synthesis ──────────────────────────────────────
    # Uses the re-verified context (== verified_context when nothing was
    # re-verified or when the council converged early).
    final_answer, fallback_used = run_stage4_consensus(
        question, ok_answers, critiques, dialectic, anon_labels, verified_context_post2,
        images, images_mime,
        focused_debate=focused_debate,
        progress_cb=stage4_cb,
    )

    # Stage 4 produced no usable answer (master synthesis AND fallback both
    # errored) → mark the whole debate as failed so the UI shows an honest
    # failure instead of dressing up an "ERROR:" string as a certified answer.
    debate_failed = _is_error_text(final_answer)
    if debate_failed:
        stage_errors.append("Stage 4: synthesis failed (master and fallback both errored)")

    # Deduplicate citations from pre-flight search + Stage 2.5 re-verification.
    seen_urls: set = set()
    citations: List[dict] = []
    for c in list(preflight_citations) + reverification_citations:
        if c["url"] not in seen_urls:
            seen_urls.add(c["url"])
            citations.append(c)

    return {
        "question":           question,
        "answers":            ok_answers,
        "critiques":          critiques,
        "dialectic":          dialectic,
        "focused_debate":     focused_debate,
        "reliability_scores": reliability_scores,
        "anon_labels":        anon_labels,
        "final_answer":       final_answer,
        "fallback_used":      fallback_used,
        "citations":          citations,
        "has_images":         len(images) > 0,
        "academic_mode":      academic_mode,
        "academic_papers":    academic_papers,
        "failed_models":      failed_models,
        "stage_errors":       stage_errors,
        "debate_failed":      debate_failed,
        "critique_verdicts":      critique_verdicts,
        "converged_early":        converged_early,
        "reverification_block":   reverification_block,
        "reverification_claims":  reverification_claims,
    }
