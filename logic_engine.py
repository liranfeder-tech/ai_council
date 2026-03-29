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
import re
import sys
from datetime import datetime
from typing import Callable, List, Optional, TypedDict

# Python 3.14+ ThreadPoolExecutor propagates context automatically per task.
# Earlier versions need explicit copy_context() wrapping.
_PY314_PLUS = sys.version_info >= (3, 14)

from glossary import (
    CONTEXTUAL_HALLUCINATION_TRIGGERS,
    HARD_HALLUCINATION_TRIGGERS,
    COUNCIL_CONSENSUS_PROMPT,
    EDUCATIONAL_FRAMING_NOTE,
    FALLBACK_MODEL_KEY,
    MASTER_MODEL_KEY,
    MODELS,
    PROMPT_CRITIQUE,
    PROMPT_DIALECTIC,
    PROMPT_INITIAL,
    PROMPT_EXTRACT_DISAGREEMENTS,
    PROMPT_DEBATE_OPENING,
    PROMPT_DEBATE_RESPONSE,
    FOLLOWUP_CONTEXT_BLOCK,
    STAGE0_LIVE_LABEL,
    STAGE0_MARKET_INSTRUCTION,
    TEMPORAL_AUTHORITY_CLAUSE,
    TEMPORAL_AUTHORITY_CLAUSE_GENERAL,
    UI_STAGE0_COMPLETE_COMMODITY,
    UI_STAGE0_COMPLETE_GENERAL,
    UI_STAGE0_COMPLETE_NONE,
    UI_STAGE0_SEARCHING,
    UI_STAGE3B_NO_DEBATES,
    UI_SPINNER_STAGE3B,
    VISION_MODE_PROMPT,
)
from ai_factory import call_model, call_model_with_citations
from search_engine import get_live_context, get_live_market_data, search_is_available


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
    prompt = PROMPT_INITIAL.format(
        vision_prefix=_vision_prefix(images),
        verified_context=verified_context,
        question=question,
    )
    answer, citations = call_model_with_citations(
        model_key, prompt, images, images_mime
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
            if _PY314_PLUS:
                fut = executor.submit(
                    _fetch_one_answer, key, question, verified_context, images, images_mime
                )
            else:
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
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
) -> tuple[str, str, str]:
    """
    One model reviews one other model's answer.

    Returns (reviewer_key, target_key, critique_text).
    """
    prompt = PROMPT_CRITIQUE.format(
        vision_prefix=_vision_prefix(images),
        verified_context=verified_context,
        question=question,
        author_label=MODELS[target_key]["label"],
        author_answer=answers[target_key],
        other_answers=_build_other_answers_block(answers, exclude_key=target_key),
    )
    critique = call_model(reviewer_key, prompt, images, images_mime)
    return reviewer_key, target_key, critique


def run_stage2_cross_critique(
    active_keys: List[str],
    question: str,
    answers: dict[str, str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
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
        futures = {}
        for reviewer, target in pairs:
            if _PY314_PLUS:
                fut = executor.submit(
                    _critique_one, reviewer, target, question, answers,
                    verified_context, images, images_mime
                )
            else:
                _ctx = contextvars.copy_context()
                fut = executor.submit(
                    _ctx.run, _critique_one, reviewer, target, question, answers,
                    verified_context, images, images_mime
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
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
) -> tuple[str, str]:
    """
    One model reads all critiques directed AT it and responds:
    defending correct points or acknowledging and refining flawed ones.

    Returns (model_key, dialectic_response_text).
    """
    critique_block = "\n\n---\n\n".join(critiques_against)
    prompt = PROMPT_DIALECTIC.format(
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
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
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
        futures = {}
        for key in responding:
            if _PY314_PLUS:
                fut = executor.submit(
                    _dialectic_one, key, question, answers.get(key, ""),
                    critiques_by_target[key], verified_context, images, images_mime,
                )
            else:
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

def _build_critiques_summary(critiques: dict[tuple[str, str], str]) -> str:
    """Truncate each critique to first 150 chars for the extraction prompt."""
    lines: List[str] = []
    for (reviewer_key, target_key), text in critiques.items():
        reviewer_label = MODELS[reviewer_key]["label"]
        target_label   = MODELS[target_key]["label"]
        snippet = text[:150].replace("\n", " ")
        lines.append(f"{reviewer_label} → {target_label}: {snippet}…")
    return "\n".join(lines) if lines else "_No critiques._"


def _build_dialectic_summary(dialectic: dict[str, str]) -> str:
    """Truncate each dialectic response to first 150 chars."""
    lines: List[str] = []
    for key, text in dialectic.items():
        label   = MODELS[key]["label"]
        snippet = text[:150].replace("\n", " ")
        lines.append(f"{label}: {snippet}…")
    return "\n".join(lines) if lines else "_No dialectic responses._"


def _extract_disagreement_points(
    question: str,
    active_keys: List[str],
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
) -> list[DisagreementPoint]:
    """
    Ask the master model to identify 2-3 genuine disagreement points.
    Returns [] on any error or when models have converged.
    """
    prompt = PROMPT_EXTRACT_DISAGREEMENTS.format(
        question=question,
        critiques_summary=_build_critiques_summary(critiques),
        dialectic_summary=_build_dialectic_summary(dialectic),
        available_keys=str(active_keys),
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
            p["model_keys"] = [k for k in p.get("model_keys", []) if k in active_keys]
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
    opponent_opening: str = "",
    opponent_key: str = "",
) -> tuple[int, int, str, str]:
    """Execute one debate turn. Returns (point_id, round_num, model_key, text)."""
    your_label = MODELS[model_key]["label"]
    if round_num == 1:
        prompt = PROMPT_DEBATE_OPENING.format(
            your_label=your_label,
            question=question,
            point_summary=point["summary"],
            excerpt=point["excerpt"],
        )
    else:
        opponent_label = MODELS[opponent_key]["label"] if opponent_key else "opponent"
        prompt = PROMPT_DEBATE_RESPONSE.format(
            your_label=your_label,
            point_summary=point["summary"],
            opponent_label=opponent_label,
            opponent_opening=opponent_opening,
        )
    text = call_model(model_key, prompt)
    return point["point_id"], round_num, model_key, text


def run_stage3b_focused_debate(
    active_keys: List[str],
    question: str,
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """
    Identify genuine disagreements after Stage 3 and run focused 2-round exchanges.

    Returns
    -------
    dict with keys:
        disagreement_points : list[DisagreementPoint]
        exchanges           : dict[(point_id, round, model_key), text]
        skipped             : bool
    """
    empty = {"disagreement_points": [], "exchanges": {}, "skipped": True}

    if len(active_keys) < 2:
        return empty

    if progress_cb:
        progress_cb(0.0, "🔍 Extracting disagreement points …")

    points = _extract_disagreement_points(question, active_keys, critiques, dialectic)

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
                if _PY314_PLUS:
                    fut = ex.submit(_run_one_debate_round, point, 1, mk, question)
                else:
                    _ctx = contextvars.copy_context()
                    fut = ex.submit(_ctx.run, _run_one_debate_round, point, 1, mk, question)
                r1_futures[fut] = mk
            for fut in concurrent.futures.as_completed(r1_futures):
                p_id, rnd, mk, text = fut.result()
                exchanges[(p_id, rnd, mk)] = text
                completed += 1
                if progress_cb:
                    label = MODELS[mk]["label"]
                    progress_cb(completed / total_rounds,
                                f"⚔️ {label} — Point {pid} Round 1 done")

        # Round 2 — direct responses (needs Round 1, parallel across models)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(m_keys)) as ex:
            r2_futures = {}
            for mk in m_keys:
                opponent_key     = next((k for k in m_keys if k != mk), "")
                opponent_opening = exchanges.get((pid, 1, opponent_key), "")
                if _PY314_PLUS:
                    fut = ex.submit(
                        _run_one_debate_round, point, 2, mk, question,
                        opponent_opening, opponent_key,
                    )
                else:
                    _ctx = contextvars.copy_context()
                    fut = ex.submit(
                        _ctx.run, _run_one_debate_round, point, 2, mk, question,
                        opponent_opening, opponent_key,
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


def _format_focused_debate(focused_debate: Optional[dict]) -> str:
    """Render focused debate exchanges into a readable block for the consensus prompt."""
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
        m_keys  = point["model_keys"]
        lines   = [f"### Point {pid}: {summary}"]
        for rnd, rnd_label in [(1, "Opening"), (2, "Response")]:
            for mk in m_keys:
                text  = exchanges.get((pid, rnd, mk), "")
                label = MODELS[mk]["label"]
                if text:
                    lines.append(f"**{label} — {rnd_label}:**\n{text}")
        blocks.append("\n\n".join(lines))
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Stage 4 — Consensus Synthesis (DSAD)
# ---------------------------------------------------------------------------

def run_stage4_consensus(
    question: str,
    answers: dict[str, str],
    critiques: dict[tuple[str, str], str],
    dialectic: dict[str, str],
    verified_context: str = "",
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    silver_price: str = "N/A",
    exchange_rate: str = "N/A",
    focused_debate: Optional[dict] = None,
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
        vision_prefix=_vision_prefix(images),
        verified_context=verified_context,
        question=question,
        all_answers_block=_format_all_answers(answers),
        all_critiques_block=_format_all_critiques(critiques),
        all_dialectic_block=_format_all_dialectic(dialectic),
        focused_debate_block=_format_focused_debate(focused_debate),
        silver_price=silver_price,
        exchange_rate=exchange_rate,
    )

    final_answer, _cit = call_model_with_citations(
        MASTER_MODEL_KEY, prompt, images, images_mime
    )
    fallback_used = False

    if final_answer.startswith("ERROR:"):
        if progress_cb:
            progress_cb(
                0.5,
                f"⚠️ {master_label} failed — switching to {fallback_label} …",
            )
        fb_answer, _fb_cit = call_model_with_citations(
            FALLBACK_MODEL_KEY, prompt, images, images_mime
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
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    previous_context: Optional[dict] = None,
    code_context: str = "",
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
        has_commodity_data_early = silver_price != "N/A" or exchange_rate != "N/A"
        if has_commodity_data_early:
            _stage0_msg = UI_STAGE0_COMPLETE_COMMODITY.format(
                silver=silver_price,
                rate=exchange_rate,
            )
        elif broad_block:
            _n_results = broad_block.count("\n•") or broad_block.count("\n-") or "some"
            _stage0_msg = UI_STAGE0_COMPLETE_GENERAL.format(n=_n_results)
        else:
            _stage0_msg = UI_STAGE0_COMPLETE_NONE
        stage0_cb(1.0, _stage0_msg)

    current_date  = datetime.now().strftime("%A, %B %d, %Y")
    context_parts: List[str] = []

    # Temporal Authority Clause: placed FIRST so it is the first thing every
    # model reads.  Two variants:
    #   - Commodity clause (silver+ILS present): enforces live price figures.
    #   - General clause (search results but no commodity prices): anchors
    #     the date and directs models to use the search snippets below.
    has_commodity_data = silver_price != "N/A" or exchange_rate != "N/A"

    if has_commodity_data:
        context_parts.append(
            TEMPORAL_AUTHORITY_CLAUSE.format(
                current_date=current_date,
                silver_price=silver_price,
                exchange_rate=exchange_rate,
            )
        )
    elif broad_block:
        # We have live search results for the question but no commodity prices.
        context_parts.append(
            TEMPORAL_AUTHORITY_CLAUSE_GENERAL.format(current_date=current_date)
        )
    else:
        # No search data at all — just anchor the date.
        context_parts.append(f"TODAY'S DATE: {current_date}")

    if has_commodity_data:
        context_parts.append(
            f"{STAGE0_LIVE_LABEL}\n{clean_data}\n\n"
            + STAGE0_MARKET_INSTRUCTION
            + "\n\n"
            + EDUCATIONAL_FRAMING_NOTE
        )
    if broad_block:
        context_parts.append(broad_block)

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
            .replace("{prev_answer}",  prev_answer[:3000])
        )
        context_parts.insert(0, fup_block)

    verified_context = "\n\n".join(context_parts)

    # ── Stage 1: Parallel Thesis ───────────────────────────────────────────
    answers, _s1_cit = run_stage1_parallel_inference(
        all_keys, question, verified_context,
        images, images_mime, progress_cb=stage1_cb,
    )

    # ── Stage 2: Adversarial Audit ─────────────────────────────────────────
    critiques: dict[tuple[str, str], str] = {}
    if len(all_keys) >= 2:
        critiques = run_stage2_cross_critique(
            all_keys, question, answers, verified_context,
            images, images_mime, progress_cb=stage2_cb,
        )

    # ── Stage 3: Dialectic Response ────────────────────────────────────────
    dialectic: dict[str, str] = {}
    if critiques:
        dialectic = run_stage3_dialectic(
            all_keys, question, answers, critiques, verified_context,
            images, images_mime, progress_cb=stage3_cb,
        )

    # Reliability scores: two independent deduction categories.
    #
    # 1. Retraction deductions (-20 each): model changed a correct position
    #    under peer pressure (normal peer-review behaviour, mild penalty).
    # 2. Contextual Hallucination deductions (-40 each): model rejected
    #    actual Stage 0 commodity data as hypothetical/unconfirmed.
    #    IMPORTANT: these deductions only apply when clean_data is present
    #    (i.e., real commodity figures were retrieved).  When no commodity
    #    data exists, saying "I cannot verify" is correct behaviour and must
    #    NOT be penalised.
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
        retraction_hits = sum(1 for t in _RETRACTION_TRIGGERS if t in low)

        # Only penalise for rejecting Stage 0 data when that data actually exists.
        if clean_data:
            hallucination_hits = sum(
                1 for t in CONTEXTUAL_HALLUCINATION_TRIGGERS if t in low
            )
            hard_hits = sum(1 for t in HARD_HALLUCINATION_TRIGGERS if t in low)
        else:
            hallucination_hits = 0
            hard_hits          = 0

        reliability_scores[key] = max(
            0,
            100 - retraction_hits * 20 - hallucination_hits * 40 - hard_hits * 50,
        )

    # ── Stage 3b: Focused Debate ───────────────────────────────────────────
    # Only the non-master models debate — master stays as mediator
    debate_keys = [k for k in all_keys if k != MASTER_MODEL_KEY]
    focused_debate = run_stage3b_focused_debate(
        active_keys=debate_keys,
        question=question,
        critiques=critiques,
        dialectic=dialectic,
        progress_cb=stage3b_cb,
    )

    # ── Stage 4: Consensus Synthesis ──────────────────────────────────────
    final_answer, fallback_used = run_stage4_consensus(
        question, answers, critiques, dialectic, verified_context,
        images, images_mime,
        silver_price=silver_price,
        exchange_rate=exchange_rate,
        focused_debate=focused_debate,
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
        "focused_debate":     focused_debate,
        "reliability_scores": reliability_scores,
        "final_answer":       final_answer,
        "fallback_used":      fallback_used,
        "citations":          citations,
        "has_images":         len(images) > 0,
    }
