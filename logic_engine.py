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

from __future__ import annotations

import concurrent.futures
import re
from datetime import datetime
from typing import Callable, List, Optional

from glossary import (
    CONTEXTUAL_HALLUCINATION_TRIGGERS,
    COUNCIL_CONSENSUS_PROMPT,
    EDUCATIONAL_FRAMING_NOTE,
    FALLBACK_MODEL_KEY,
    MASTER_MODEL_KEY,
    MODELS,
    PROMPT_CRITIQUE,
    PROMPT_DIALECTIC,
    PROMPT_INITIAL,
    STAGE0_LIVE_LABEL,
    STAGE0_MARKET_INSTRUCTION,
    TEMPORAL_AUTHORITY_CLAUSE,
    UI_STAGE0_COMPLETE_TPL,
    UI_STAGE0_SEARCHING,
    VISION_MODE_PROMPT,
)
from ai_factory import call_model, call_model_with_citations
from search_engine import get_live_context, get_live_market_data, search_is_available


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ProgressCallback = Callable[[float, str], None]   # (fraction 0-1, status text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vision_prefix(image_bytes: Optional[bytes]) -> str:
    """Return VISION_MODE_PROMPT when an image is present, else empty string."""
    return VISION_MODE_PROMPT if image_bytes else ""


# ---------------------------------------------------------------------------
# Stage 1 — Parallel Inference
# ---------------------------------------------------------------------------

def _fetch_one_answer(
    model_key: str,
    question: str,
    verified_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> tuple[str, str, List[dict]]:
    """Call a single model and return (model_key, answer_text, citations)."""
    prompt = PROMPT_INITIAL.format(
        vision_prefix=_vision_prefix(image_bytes),
        verified_context=verified_context,
        question=question,
    )
    answer, citations = call_model_with_citations(
        model_key, prompt, image_bytes, image_mime
    )
    return model_key, answer, citations


def run_stage1_parallel_inference(
    active_keys: List[str],
    question: str,
    verified_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
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
        futures = {
            executor.submit(
                _fetch_one_answer, key, question, verified_context,
                image_bytes, image_mime
            ): key
            for key in active_keys
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            model_key, answer, model_citations = future.result()
            answers[model_key] = answer
            citations.extend(model_citations)   # will be [] per model now

            if progress_cb:
                label = MODELS[model_key]["label"]
                progress_cb(i / total, f"✅ {label} answered ({i}/{total})")

    return answers, citations


# ---------------------------------------------------------------------------
# Stage 2 — Cross-Critique
# ---------------------------------------------------------------------------

def _build_other_answers_block(
    answers: dict[str, str],
    exclude_key: str,
) -> str:
    """Format all answers except the one being reviewed into a readable block."""
    lines: List[str] = []
    for key, text in answers.items():
        if key == exclude_key:
            continue
        label = MODELS[key]["label"]
        lines.append(f"### {label}\n{text}")
    return "\n\n---\n\n".join(lines) if lines else "_No other answers available._"


def _critique_one(
    reviewer_key: str,
    target_key: str,
    question: str,
    answers: dict[str, str],
    verified_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> tuple[str, str, str]:
    """
    One model reviews one other model's answer.

    Returns (reviewer_key, target_key, critique_text).
    """
    prompt = PROMPT_CRITIQUE.format(
        vision_prefix=_vision_prefix(image_bytes),
        verified_context=verified_context,
        question=question,
        author_label=MODELS[target_key]["label"],
        author_answer=answers[target_key],
        other_answers=_build_other_answers_block(answers, exclude_key=target_key),
    )
    critique = call_model(reviewer_key, prompt, image_bytes, image_mime)
    return reviewer_key, target_key, critique


def run_stage2_cross_critique(
    active_keys: List[str],
    question: str,
    answers: dict[str, str],
    verified_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[tuple[str, str], str]:
    """
    Each model reviews every other model's answer in parallel.

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

    critiques: dict[tuple[str, str], str] = {}
    total = len(pairs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(total, 8)) as executor:
        futures = {
            executor.submit(
                _critique_one, reviewer, target, question, answers,
                verified_context, image_bytes, image_mime
            ): (reviewer, target)
            for reviewer, target in pairs
        }

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
# Shared formatters
# ---------------------------------------------------------------------------

def _format_all_answers(answers: dict[str, str]) -> str:
    """Render all initial answers into a single text block."""
    blocks = []
    for key, text in answers.items():
        label = MODELS[key]["label"]
        blocks.append(f"### {label}\n{text}")
    return "\n\n---\n\n".join(blocks)


def _format_all_critiques(critiques: dict[tuple[str, str], str]) -> str:
    """Render all critiques into a single text block."""
    blocks = []
    for (reviewer_key, target_key), text in critiques.items():
        reviewer_label = MODELS[reviewer_key]["label"]
        target_label   = MODELS[target_key]["label"]
        blocks.append(f"### {reviewer_label} → {target_label}\n{text}")
    return "\n\n---\n\n".join(blocks)


def _format_all_dialectic(dialectic: dict[str, str]) -> str:
    """Render all dialectic responses into a single text block."""
    blocks = []
    for key, text in dialectic.items():
        label = MODELS[key]["label"]
        blocks.append(f"### {label} — Dialectic Response\n{text}")
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
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> tuple[str, str]:
    """
    One model reads all critiques directed AT it and responds:
    defending correct points or acknowledging and refining flawed ones.

    Returns (model_key, dialectic_response_text).
    """
    critique_block = "\n\n---\n\n".join(critiques_against)
    prompt = PROMPT_DIALECTIC.format(
        vision_prefix=_vision_prefix(image_bytes),
        verified_context=verified_context,
        question=question,
        your_initial_answer=initial_answer,
        critique_of_your_answer=critique_block,
    )
    response = call_model(model_key, prompt, image_bytes, image_mime)
    return model_key, response


def run_stage3_dialectic(
    active_keys: List[str],
    question: str,
    answers: dict[str, str],
    critiques: dict[tuple[str, str], str],
    verified_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[str, str]:
    """
    Each model responds to the critiques directed at it in parallel.

    Returns
    -------
    dict[str, str]
        Mapping of model_key → dialectic_response_text.
    """
    # Collect critiques received by each model
    critiques_by_target: dict[str, List[str]] = {k: [] for k in active_keys}
    for (reviewer_key, target_key), critique_text in critiques.items():
        if target_key in critiques_by_target:
            reviewer_label = MODELS[reviewer_key]["label"]
            critiques_by_target[target_key].append(
                f"**Critique from {reviewer_label}:**\n{critique_text}"
            )

    # Only models that actually received critique need to respond
    responding = [k for k in active_keys if critiques_by_target.get(k)]
    if not responding:
        return {}

    dialectic: dict[str, str] = {}
    total = len(responding)

    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
        futures = {
            executor.submit(
                _dialectic_one,
                key, question, answers.get(key, ""),
                critiques_by_target[key],
                verified_context, image_bytes, image_mime,
            ): key
            for key in responding
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            model_key, response = future.result()
            dialectic[model_key] = response

            if progress_cb:
                label = MODELS[model_key]["label"]
                progress_cb(i / total, f"💬 {label} responded to critique ({i}/{total})")

    return dialectic


# ---------------------------------------------------------------------------
# Stage 4 — Consensus Synthesis (DSAD)
# ---------------------------------------------------------------------------

def run_stage4_consensus(
    question: str,
    answers: dict[str, str],
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
    verified_context: str = "",
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    silver_price: str = "N/A",
    exchange_rate: str = "N/A",
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[str, bool]:
    """
    The Mediator (Master Model) reads the full DSAD transcript and writes
    the Final Certified Answer.  Falls back to FALLBACK_MODEL_KEY on error.

    Parameters
    ----------
    silver_price : str
        Silver spot price extracted from Stage 0 live data (e.g. "$89.26/oz").
        Injected directly into the consensus prompt so the Mediator never
        needs to recall it from training memory.
    exchange_rate : str
        USD/ILS exchange rate from Stage 0 (e.g. "3.07").

    Returns
    -------
    tuple[str, bool]
        (final_answer_text, fallback_was_used)
    """
    master_label   = MODELS[MASTER_MODEL_KEY]["label"]
    fallback_label = MODELS[FALLBACK_MODEL_KEY]["label"]

    if progress_cb:
        progress_cb(0.1, f"🏛️ {master_label} (Mediator) is synthesising …")

    prompt = COUNCIL_CONSENSUS_PROMPT.format(
        vision_prefix=_vision_prefix(image_bytes),
        verified_context=verified_context,
        question=question,
        all_answers_block=_format_all_answers(answers),
        all_critiques_block=_format_all_critiques(critiques),
        all_dialectic_block=_format_all_dialectic(dialectic),
        silver_price=silver_price,
        exchange_rate=exchange_rate,
    )

    final_answer, _cit = call_model_with_citations(
        MASTER_MODEL_KEY, prompt, image_bytes, image_mime
    )
    fallback_used = False

    if final_answer.startswith("ERROR:"):
        if progress_cb:
            progress_cb(
                0.5,
                f"⚠️ {master_label} failed — switching to {fallback_label} …",
            )
        fb_answer, _fb_cit = call_model_with_citations(
            FALLBACK_MODEL_KEY, prompt, image_bytes, image_mime
        )
        if not fb_answer.startswith("ERROR:"):
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
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    stage0_cb: Optional[ProgressCallback] = None,
    stage1_cb: Optional[ProgressCallback] = None,
    stage2_cb: Optional[ProgressCallback] = None,
    stage3_cb: Optional[ProgressCallback] = None,
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
    all_keys = list(dict.fromkeys([MASTER_MODEL_KEY] + active_keys))

    # ── Stage 0: Pre-flight live search ───────────────────────────────────
    if stage0_cb:
        stage0_cb(0.0, UI_STAGE0_SEARCHING)

    clean_data              = get_live_context(question)
    broad_block, preflight_citations = get_live_market_data(question)

    # Extract individual Stage 0 values for:
    #   a) the Temporal Authority Clause (injected into every prompt)
    #   b) the Stage 4 consensus formula (passed explicitly to run_stage4_consensus)
    silver_price  = "N/A"
    exchange_rate = "N/A"
    if clean_data:
        _s = re.search(r'Silver:\s*(\$[\d.]+/oz)', clean_data)
        _e = re.search(r'USD/ILS:\s*([\d.]+)', clean_data)
        if _s:
            silver_price  = _s.group(1)
        if _e:
            exchange_rate = _e.group(1)

    # Stage 0 complete — log the live values to the UI
    if stage0_cb:
        stage0_cb(
            1.0,
            UI_STAGE0_COMPLETE_TPL.format(
                silver=silver_price if silver_price != "N/A" else "N/A (search unavailable)",
                rate=exchange_rate  if exchange_rate != "N/A" else "N/A",
            ),
        )

    current_date  = datetime.now().strftime("%A, %B %d, %Y")
    context_parts: List[str] = []

    # Temporal Authority Clause: placed FIRST so it is the first thing every
    # model reads.  Prevents treating live figures as hypothetical/unverified.
    if clean_data:
        context_parts.append(
            TEMPORAL_AUTHORITY_CLAUSE.format(
                current_date=current_date,
                silver_price=silver_price,
                exchange_rate=exchange_rate,
            )
        )

    context_parts.append(f"TODAY'S DATE: {current_date}")

    if clean_data:
        context_parts.append(
            f"{STAGE0_LIVE_LABEL}\n{clean_data}\n\n"
            + STAGE0_MARKET_INSTRUCTION
            + "\n\n"
            + EDUCATIONAL_FRAMING_NOTE
        )
    if broad_block:
        context_parts.append(broad_block)

    verified_context = "\n\n".join(context_parts)

    # ── Stage 1: Parallel Thesis ───────────────────────────────────────────
    answers, _s1_cit = run_stage1_parallel_inference(
        all_keys, question, verified_context,
        image_bytes, image_mime, progress_cb=stage1_cb,
    )

    # ── Stage 2: Adversarial Audit ─────────────────────────────────────────
    critiques: dict[tuple[str, str], str] = {}
    if len(all_keys) >= 2:
        critiques = run_stage2_cross_critique(
            all_keys, question, answers, verified_context,
            image_bytes, image_mime, progress_cb=stage2_cb,
        )

    # ── Stage 3: Dialectic Response ────────────────────────────────────────
    dialectic: dict[str, str] = {}
    if critiques:
        dialectic = run_stage3_dialectic(
            all_keys, question, answers, critiques, verified_context,
            image_bytes, image_mime, progress_cb=stage3_cb,
        )

    # Reliability scores: two independent deduction categories.
    #
    # 1. Retraction deductions (-20 each): model changed a correct position
    #    under peer pressure (normal peer-review behaviour, mild penalty).
    # 2. Contextual Hallucination deductions (-40 each): model dismissed
    #    Stage 0 live data as hypothetical / unconfirmed — this is a factual
    #    error and receives a steeper penalty.  A model can only score 100
    #    if it accepted live data as authoritative AND held its position.
    _RETRACTION_TRIGGERS = (
        "you are correct", "you're right", "i was wrong", "i acknowledge",
        "i concede", "i was mistaken", "valid point", "i agree with",
        "i must retract", "i now agree", "i stand corrected", "i missed",
        "i overlooked", "correct to point out",
        "אתה צודק", "טעיתי", "אני מסכים", "הערה נכונה",
    )
    reliability_scores: dict[str, int] = {}
    for key, resp in dialectic.items():
        low             = resp.lower()
        retraction_hits = sum(1 for t in _RETRACTION_TRIGGERS           if t in low)
        hallucination_hits = sum(1 for t in CONTEXTUAL_HALLUCINATION_TRIGGERS if t in low)
        reliability_scores[key] = max(
            0,
            100 - retraction_hits * 20 - hallucination_hits * 40,
        )

    # ── Stage 4: Consensus Synthesis ──────────────────────────────────────
    final_answer, fallback_used = run_stage4_consensus(
        question, answers, critiques, dialectic, verified_context,
        image_bytes, image_mime,
        silver_price=silver_price,
        exchange_rate=exchange_rate,
        progress_cb=stage4_cb,
    )

    # Deduplicate citations from pre-flight search
    seen_urls: set = set()
    citations: List[dict] = []
    for c in preflight_citations:
        if c["url"] not in seen_urls:
            seen_urls.add(c["url"])
            citations.append(c)

    return {
        "question":           question,
        "answers":            answers,
        "critiques":          critiques,
        "dialectic":          dialectic,
        "reliability_scores": reliability_scores,
        "final_answer":       final_answer,
        "fallback_used":      fallback_used,
        "citations":          citations,
        "has_image":          image_bytes is not None,
    }
