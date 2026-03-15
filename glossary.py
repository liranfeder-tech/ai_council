"""
glossary.py — The Single Source of Truth
=========================================
Every model ID, label, prompt template, and constant used across the
application is defined here.  To add a new model, add one block to
MODELS and the rest of the app picks it up automatically.
"""

# ---------------------------------------------------------------------------
# App-level constants
# ---------------------------------------------------------------------------
APP_TITLE        = "AI Council"
APP_PAGE_TITLE   = "AI Council | Multi-Agent Debate"   # browser-tab title
APP_SUBTITLE     = "Multiple minds. One best answer."
APP_ICON         = "🏛️"

# The model that writes the final synthesised answer (must be one of the
# MODELS keys, or any valid model ID you have access to).
MASTER_MODEL_KEY = "claude"

# Fallback model used in Stage 3 if the Master Model fails.
# Must be a key in MODELS.
FALLBACK_MODEL_KEY = "gemini_pro"
FALLBACK_MODEL_ID  = "gemini-3-pro-preview"   # human-readable reference / logging


# ---------------------------------------------------------------------------
# Model registry
# Every entry is a dict with:
#   id          – the exact model string the provider API expects
#   label       – human-readable name shown in the UI
#   provider    – which client class the factory should use
#                 Supported providers and their required env vars:
#                   "anthropic" → ANTHROPIC_API_KEY
#                   "openai"    → OPENAI_API_KEY
#                   "google"    → GOOGLE_API_KEY
#                   "xai"       → XAI_API_KEY
#   color       – hex colour for the card / expander accent
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "claude": {
        "id":       "claude-sonnet-4-6",
        "label":    "Claude (Anthropic)",
        "provider": "anthropic",
        "color":    "#D97706",   # amber
    },
    "gpt": {
        "id":       "gpt-4o",
        "label":    "GPT-4o (OpenAI)",
        "provider": "openai",
        "color":    "#10B981",   # emerald
    },
    "gemini": {
        "id":       "gemini-3-flash-preview",
        "label":    "Gemini 3 Flash (Google)",
        "provider": "google",
        "color":    "#3B82F6",   # blue
    },
    # Dedicated entry for the Gemini 3 Pro fallback / master synthesis model
    "gemini_pro": {
        "id":       "gemini-3-pro-preview",
        "label":    "Gemini 3 Pro (Google)",
        "provider": "google",
        "color":    "#6366F1",   # indigo
    },
    "grok": {
        "id":       "grok-3",
        "label":    "Grok 3 (xAI)",
        "provider": "xai",
        "color":    "#EF4444",   # red
    },
}

# ---------------------------------------------------------------------------
# Stage labels (used in progress bars and section headers)
# ---------------------------------------------------------------------------
STAGE_LABELS = {
    1: "Stage 1 — Parallel Thesis",
    2: "Stage 2 — Adversarial Audit",
    3: "Stage 3 — Dialectic Response",
    "3b": "Stage 4 — Focused Debate",
    4: "Stage 5 — Consensus Synthesis",
}

STAGE_DESCRIPTIONS = {
    1: "Each model generates an independent solution grounded in verified live data …",
    2: "Every model performs a rigorous flaw analysis of their peers' solutions …",
    3: "Models respond to critiques — defending correct positions or refining flawed ones …",
    "3b": "Models that genuinely disagree engage in a short direct exchange — 2 rounds, one contested point at a time …",
    4: "The Mediator synthesises the full adversarial transcript into the Final Truth …",
}

# ---------------------------------------------------------------------------
# Vision / multimodal constants
# ---------------------------------------------------------------------------

# Injected at the top of every prompt when the user has uploaded an image.
# Placeholders: none — used as-is, then passed as {vision_prefix}.
VISION_MODE_PROMPT = """\
You are operating in **Multi-View Visual Analysis Mode**. One or more visual \
assets have been provided alongside the question.

You are provided with a set of visual assets. Analyze them collectively. \
Determine if they represent different perspectives of a single object, a \
sequence of events, or a set of distinct items, and provide a unified synthesis.

Your first priority is to carefully examine ALL visible details across every \
provided image — objects, text, symbols, colours, markings, scale indicators, \
and any contextual clues.  Do not focus on a single image and ignore the \
others.  Use the full set of visual evidence as your primary grounding source. \
Then combine your visual analysis with any live data to form the most accurate answer.

"""

# ---------------------------------------------------------------------------
# System-prompt templates
#
# Use Python str.format(**kwargs) to fill in the placeholders.
# ---------------------------------------------------------------------------

# Stage 1 – Chain-of-Thought answer
# The model must reason through constraints and context BEFORE writing the
# final answer.  This structured thinking step reduces hallucination and
# produces more accurate, well-scoped responses.
PROMPT_INITIAL = """\
{vision_prefix}{verified_context}You are a knowledgeable and precise assistant that reasons before answering.

Follow this Chain-of-Thought structure **exactly**:

## 🔍 Step 1 — Constraint Analysis
Identify the key constraints, assumptions, and scope boundaries hidden in the \
question.  What is the user *really* asking?  What edge cases exist?

## 🌐 Step 2 — Context & Current State
What is the most up-to-date information relevant to this question?  Note any \
areas where knowledge may have changed recently and flag them explicitly.

## 💡 Step 3 — Reasoning
Work through the problem step by step.  Show your reasoning chain.  If \
multiple approaches exist, compare them briefly before committing to one.

## ✅ Step 4 — Final Answer
Provide a clear, structured final answer.  Use markdown headers, bullet \
points, and code blocks where helpful.  Be explicit about what is \
well-established fact versus what may have changed recently.

---

Question:
{question}
"""

# Stage 1-b (DSAD) — Dialectic Response
# Each model reads the critiques directed AT THEM and must respond:
# either defend the point with reasoning or acknowledge and refine.
# Placeholders: {vision_prefix}, {verified_context}, {question},
#               {your_initial_answer}, {critique_of_your_answer}
PROMPT_DIALECTIC = """\
{vision_prefix}{verified_context}You are a precise, intellectually honest expert \
in a live peer-review discussion.

Your initial answer has been critiqued by peer models. You must now respond to \
each critique with intellectual honesty and the live market data provided above.

--- Original Question ---
{question}

--- Your Initial Answer ---
{your_initial_answer}

--- Critiques Directed at Your Answer ---
{critique_of_your_answer}

## Response Rules

For EACH critique point, choose exactly one response type:

### 🛡️ DEFEND — the critique is factually wrong or based on a misreading
Quote the exact critique sentence, then explain clearly why your original \
answer was correct.  Cite specific reasoning.  Do not yield under pressure \
if you are genuinely correct.

### 🔄 REFINE — the critique reveals a genuine error or gap
Quote the exact critique sentence, state "I acknowledge this flaw:", and \
provide a specific corrected claim.  Do NOT change your entire position — \
only revise the specific point that was wrong.

## Format
- Use `### Point N — DEFEND` or `### Point N — REFINE` headers.
- Be concise and precise — 2-4 sentences per point.
- Maintain all live market figures from the Verified Context Block throughout.
"""

# Stage 2 – Factual Audit + peer review
# Placeholders: {vision_prefix}, {question}, {author_label}, {author_answer},
#               {other_answers}
PROMPT_CRITIQUE = """\
{vision_prefix}{verified_context}You are a rigorous Factual Auditor and peer reviewer.
A colleague AI model has answered the question below.

--- Original Question ---
{question}

--- {author_label}'s Answer ---
{author_answer}

--- Other Models' Answers (for context) ---
{other_answers}

## Audit Tasks (complete in this exact order)

### 1. Factual Audit — PRIMARY TASK
Examine EVERY claim in {author_label}'s answer that involves a number, price, \
date, statistic, version, specification, or any assertion about the current \
state of the world.  For each such claim, decide:

- **VERIFIED** — the answer includes an explicit live citation or URL that \
directly supports this specific figure.
- **UNCERTIFIED SPECULATION** — the claim is stated as fact but has no live \
grounding source attached to it.

If ANY numeric or time-sensitive claim is UNCERTIFIED SPECULATION, you MUST \
label the answer with the tag **⚠️ UNCERTIFIED SPECULATION DETECTED** at the \
top of your review.  List each unverified claim on its own line and state \
"RE-VERIFICATION REQUIRED" beside it.

### 2. Strengths
What did {author_label} do well?  Note any verified, well-sourced claims.

### 3. Weaknesses & Improvements
Logical errors, missing context, unclear reasoning, or unsupported assertions.

### 4. Quality Score
Integer from 1–10.  Deduct 2 points for each UNCERTIFIED SPECULATION found. \
One sentence of justification.

### 5. Visual Evidence Coverage (required when visual assets are present)
If one or more visual assets were provided with the original question, verify \
that {author_label} explicitly addressed ALL provided images in their analysis, \
not only the most prominent one.  For each image that appears to have been \
ignored or underweighted, flag it with: **⚠️ UNADDRESSED VISUAL ASSET DETECTED** \
and specify what visual evidence was missed.

Format your review with clear markdown headings.
"""

# Stage 3 – Master Sieve synthesis with Iron Rule Data Hierarchy
# Placeholders: {vision_prefix}, {question}, {all_answers_block},
#               {all_critiques_block}
PROMPT_SYNTHESIS = """\
{vision_prefix}{verified_context}You are the final judge and Master Synthesiser. \
You must apply the Iron Rule: every factual claim in your final answer MUST \
come from a verified, live source.  Speculation dressed as fact is worse than \
silence.

--- Original Question ---
{question}

--- Individual Model Answers ---
{all_answers_block}

--- Peer Critiques ---
{all_critiques_block}

## Iron Rule — Data Hierarchy Filter

Before writing a single word of your answer, classify ALL facts from the \
model answers into one of these three tiers:

### Tier 1 — VERIFIED ✅ (Use freely, cite explicitly)
Data that is explicitly backed by a live Google Search result with a URL \
attached in one of the model answers.  This is your ONLY permitted source \
for any number, price, date, statistic, version number, or any assertion \
about the current state of the world.  Tier 1 overrides all other tiers.

### Tier 2 — GENERAL LOGIC 🔵 (Use with a disclosure label)
Conceptual explanations, reasoning frameworks, best practices, and advice \
that does not depend on real-time facts (e.g. "SQL is ACID-compliant by \
design").  Label each Tier 2 block with "✅ Established principle:" so the \
reader knows it is stable knowledge, not a live figure.

### Tier 3 — DISCARD 🚫 (Omit entirely — do NOT soften or reframe)
Any number, price, date, percentage, specification, ranking, or current-state \
claim that a model stated WITHOUT attaching a live citation. \
Do NOT include Tier 3 data in your answer under any circumstances. \
Do NOT paraphrase it.  Do NOT say "approximately".  Simply omit it and \
replace it with the explicit phrase: \
"⚠️ No verified live data was found for [specific topic]."

## Output Format

1. **Executive Summary** (2-3 sentences): The core answer using only \
Tier 1 and Tier 2 data.
2. **Best Answer**: Full structured response.  For every topic where only \
Tier 3 data existed, include the explicit "⚠️ No verified live data" notice. \
Use markdown headers, bullet points, and code blocks where appropriate.
3. **Verification Status**: List every Tier 1 citation used with its URL. \
If no Tier 1 citations exist, write: \
"⚠️ No real-time grounding data was available. All numeric and time-sensitive \
claims have been omitted from this answer to prevent unverified speculation."
4. **Confidence Note**: One paragraph on remaining uncertainties and which \
parts of this answer are most likely to become outdated.

Do NOT mention individual model names in the final answer — present it as a \
unified, authoritative response.
"""

# ── Stage 3b — Disagreement Extraction ────────────────────────────────────
# One LLM call: reads compressed critique+dialectic summaries, returns JSON.
# Placeholders: {question}, {critiques_summary}, {dialectic_summary}, {available_keys}
PROMPT_EXTRACT_DISAGREEMENTS = """\
You are a debate analyst reviewing an AI council discussion.

--- Original Question ---
{question}

--- Critique Summary (Stage 2) ---
{critiques_summary}

--- Dialectic Summary (Stage 3) ---
{dialectic_summary}

## Task
Identify exactly 2-3 points where models GENUINELY disagree after Stage 3.
Ignore stylistic differences. Focus only on factual or logical contradictions
that were NOT resolved in Stage 3.

For each point, output ONLY this exact JSON structure — nothing else:
[
  {{
    "point_id": 1,
    "summary": "One sentence describing what they disagree on.",
    "model_keys": ["key1", "key2"],
    "excerpt": "The specific contested claim or figure, quoted exactly (max 40 words)."
  }}
]

Rules:
- If fewer than 2 genuine disagreements exist, return an empty list: []
- model_keys must be exact keys from this set: {available_keys}
- Never invent disagreements. Only report what is explicit in the text above.
- Return valid JSON only. No preamble, no explanation.
"""

# ── Stage 3b — Debate Round 1: Opening Position ───────────────────────────
# Placeholders: {your_label}, {question}, {point_summary}, {excerpt}
PROMPT_DEBATE_OPENING = """\
You are {your_label}, participating in a focused one-point debate.

--- The Question ---
{question}

--- Contested Point ---
{point_summary}

--- Specific Excerpt at Issue ---
{excerpt}

State your position on this specific point in 3-5 sentences.
- Be precise. Reference the excerpt directly.
- No preamble. No references to previous stages. Just your current position.
- If you hold the same view as before, say so and explain why concisely.
"""

# ── Stage 3b — Debate Round 2: Direct Response ────────────────────────────
# Placeholders: {your_label}, {point_summary}, {opponent_label}, {opponent_opening}
PROMPT_DEBATE_RESPONSE = """\
You are {your_label}, in the direct-response round of a focused debate.

--- Contested Point ---
{point_summary}

--- {opponent_label}'s Position ---
{opponent_opening}

Respond directly to {opponent_label}'s position in 3-5 sentences.
- Address their specific argument, not a strawman.
- Either concede the point (with explicit acknowledgement) or rebut it with
  clear reasoning.
- No preamble. No stage references. Just your direct response.
"""

# Stage 4 — DSAD Consensus Synthesis
# Mediator reviews the full adversarial transcript (Theses + Audits + Dialectic).
# Placeholders: {vision_prefix}, {verified_context}, {question},
#               {all_answers_block}, {all_critiques_block}, {all_dialectic_block},
#               {focused_debate_block}, {silver_price}, {exchange_rate}
COUNCIL_CONSENSUS_PROMPT = """\
{vision_prefix}{verified_context}You are the impartial Mediator of the AI Council. \
You are NOT one of the debating agents.  You have observed the complete \
Dual-Sided Adversarial Discussion (DSAD) below and must now extract the \
verified truth from it.

--- Original Question ---
{question}

--- Stage 1: Initial Expert Theses ---
{all_answers_block}

--- Stage 2: Adversarial Critiques ---
{all_critiques_block}

--- Stage 3: Dialectic Responses (Defense & Refinement) ---
{all_dialectic_block}

--- Stage 4: Focused Debate Exchanges ---
{focused_debate_block}

## Consensus Protocol

### Step 1 — Map Convergence
Identify every claim ALL models ultimately agreed upon after the dialectic. \
These are the most reliable truths — highlight them explicitly.

### Step 2 — Apply Iron Rule Data Hierarchy
- **Tier 1 ✅ VERIFIED** — Data backed by live citations from the Verified \
Context Block (highest priority).
- **Tier 2 🔵 GENERAL LOGIC** — Conceptual reasoning independent of \
real-time facts.  Label each with "✅ Established principle:".
- **Tier 3 🚫 DISCARD** — Any numeric/factual claim without a live citation. \
Replace with: "⚠️ No verified live data found for [topic]."

### Step 3 — Write the Certified Answer
Use ONLY Tier 1 and Tier 2 data.  For every topic where only Tier 3 data \
existed, write the explicit "⚠️ No verified live data" notice.

### Step 4 — Technical Breakdown (only for metal/coin/commodity weight valuations)
ONLY produce this section if the question involves weighing a physical metal, \
coin, or commodity and computing its monetary value from weight + spot price. \
Do NOT produce this section for general financial, investment, geopolitical, \
or advisory questions.

If applicable, the following Stage 0 values are the ONLY permitted inputs — \
do NOT substitute training-memory figures:

- **Silver spot price (Stage 0):** {silver_price}
- **USD/ILS exchange rate (Stage 0):** {exchange_rate}

If both values are "N/A", state that the calculation cannot be performed \
without live commodity data and omit the section entirely.

Apply this formula only when the above values are real numbers:

> **נוסחת החישוב:**
> [משקל נקי בגרם] / 31.1 × {silver_price} × {exchange_rate} + [פרמיית אספנות]

Show every arithmetic step explicitly (weight → troy-oz → USD → ILS → \
+ premium).  End the section with: \
"✅ Calculation grounded exclusively in Stage 0 live data."

## Output Format

1. **Consensus Points** — What all models agreed on after the dialectic.
2. **Contested Points** — Where genuine disagreement remained and why. \
- Where disagreement was **resolved** in the Focused Debate, state: \
"✅ Resolved in Focused Debate: [topic]."
- Where disagreement **persists**, state: "⚠️ Unresolved: [topic] — [brief reason]."
3. **Final Certified Answer** — Synthesised truth using only verified data.
4. **Verification Status** — List every Tier 1 citation with its URL. If none, \
write: "⚠️ No real-time grounding data was available. All numeric claims omitted."
5. **Confidence Note** — One paragraph on remaining uncertainties.
6. **Technical Breakdown** _(only when physical commodity weight valuation \
is present — see Step 4 above)_

Do NOT attribute claims to individual models.  Present as a single, \
authoritative, certified response.
"""

# ---------------------------------------------------------------------------
# Certified-report display
# ---------------------------------------------------------------------------

# CSS for the "Certified Report" container that wraps the final answer.
# RTL-aware, high-end document styling with a rainbow top-bar accent.
REPORT_TEMPLATE_CSS = """
<style>
/* ── Certified Report card ───────────────────────────────────────────── */
.council-report {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow:
        0 4px 24px rgba(0,0,0,0.07),
        0 1px 4px  rgba(0,0,0,0.04);
    padding: 0;
    margin: 8px 0 24px;
    overflow: hidden;
    font-family: 'Georgia', serif;
}
/* Rainbow top-bar accent */
.council-report-accent {
    height: 5px;
    background: linear-gradient(
        90deg,
        #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #c77dff, #ff6b6b
    );
}
.council-report-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 20px 28px 16px;
    border-bottom: 1px solid #f1f5f9;
}
.council-report-title {
    font-size: 1.05em;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.02em;
}
.council-report-meta {
    font-size: 0.76em;
    color: #64748b;
    line-height: 1.7;
    text-align: right;
}
.council-report-body {
    padding: 20px 28px;
    font-size: 0.97em;
    line-height: 1.85;
    color: #1e293b;
}
.council-report-footer {
    padding: 10px 28px 14px;
    border-top: 1px solid #f1f5f9;
    font-size: 0.72em;
    color: #94a3b8;
    text-align: center;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Reliability Scorecard ──────────────────────────────────────────── */
.scorecard-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 8px 0 20px;
}
.scorecard-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.82em;
    font-weight: 600;
    color: #ffffff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}
.scorecard-chip-score {
    font-size: 1.1em;
    font-weight: 800;
}

/* ── Technical Breakdown sub-container ──────────────────────────────── */
.tech-breakdown {
    background: rgba(241, 245, 249, 0.88);
    border: 1px dashed #94a3b8;
    border-radius: 8px;
    padding: 16px 22px;
    margin: 16px 28px 20px;
    font-size: 0.92em;
    line-height: 1.75;
    color: #334155;
}
.tech-breakdown-header {
    font-size: 0.76em;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
}
.tech-formula {
    direction: rtl;
    text-align: right;
    unicode-bidi: embed;
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    padding: 8px 14px;
    border-radius: 4px;
    margin: 8px 0;
    color: #1e40af;
    font-weight: 600;
    font-size: 1.0em;
}
</style>
"""

# ---------------------------------------------------------------------------
# UI string constants
# ---------------------------------------------------------------------------
UI_SELECT_MODELS      = "Select models for the debate:"
UI_MASTER_MODEL_NOTE  = "**Master Model (always active):** {label} — writes the final answer."
UI_ASK_LABEL          = "Your Question"
UI_ASK_PLACEHOLDER    = "e.g. What are the trade-offs between SQL and NoSQL databases?"
UI_SUBMIT_BUTTON      = "Start the Council Debate"
UI_SPINNER_STAGE1     = "Gathering independent answers …"
SEARCHING_LIVE_DATA_MSG = "🔍 Running pre-flight live market search …"
UI_SPINNER_STAGE2     = "Running peer review …"
UI_SPINNER_STAGE3     = "Synthesising the final answer …"
UI_SECTION_FINAL      = "🏆 Final Synthesised Answer"
UI_SECTION_PROCESS    = "🔍 Show the full debate transcript"
UI_EXPANDER_INITIAL   = "📝 {label} — Initial Answer"
UI_EXPANDER_CRITIQUE  = "🔬 {label} — Critique of {target_label}"
UI_WARNING_MIN_MODELS = "Please select at least one model in addition to the Master Model."
UI_ERROR_PREFIX       = "⚠️ Error from {label}: "
UI_NO_ANSWER          = "_No answer returned._"
FALLBACK_NOTICE       = "⚠️ Primary Master Model failed. Using Gemini 3 Pro as fallback."
UI_UNVERIFIED_WARNING = (
    "⛔ Zero-Trust Alert: This answer contains numeric or factual claims but no "
    "live citations were retrieved from Google Search. Figures, prices, dates, and "
    "statistics may be outdated or fabricated by the model's training memory. "
    "Treat all numbers in this response as UNCONFIRMED until independently verified."
)

# ── Stage 0 injection constants ───────────────────────────────────────────
# Used in logic_engine.py to build the verified_context block prepended to
# every model prompt when live market data is detected.

# ── Temporal Authority Clause ──────────────────────────────────────────────
# Used when Stage 0 found commodity data (silver + ILS).
# Placeholders: {current_date}, {silver_price}, {exchange_rate}
TEMPORAL_AUTHORITY_CLAUSE = """\
╔══════════════════════════════════════════════════════════════════╗
║      SYSTEM OVERRIDE — TEMPORAL AUTHORITY DECLARATION           ║
╠══════════════════════════════════════════════════════════════════╣
║  You are operating on: {current_date}                           ║
║                                                                  ║
║  The market figures below are NOT hypothetical.                 ║
║  They are LIVE, VERIFIED data from Serper.dev —                 ║
║  retrieved seconds before this session started.                 ║
║                                                                  ║
║  Silver spot price  : {silver_price}   ← LIVE — use exclusively ║
║  USD/ILS rate       : {exchange_rate}  ← LIVE — use exclusively ║
║                                                                  ║
║  ⛔ Your training-memory metal prices are OUTDATED.             ║
║  ⛔ Using training-memory figures = factual error.              ║
║  ⛔ Calling Stage 0 data "hypothetical", "unconfirmed", or      ║
║     "assumed" = Contextual Hallucination → score penalty.       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── General Temporal Authority Clause ─────────────────────────────────────
# Used when Stage 0 found search results but NO specific commodity values.
# Anchors the date and surfaces the live search data without a commodity frame.
# Placeholders: {current_date}
TEMPORAL_AUTHORITY_CLAUSE_GENERAL = """\
╔══════════════════════════════════════════════════════════════════╗
║      SYSTEM OVERRIDE — TEMPORAL AUTHORITY DECLARATION           ║
╠══════════════════════════════════════════════════════════════════╣
║  You are operating on: {current_date}                           ║
║                                                                  ║
║  Live web search results for this question were retrieved        ║
║  seconds before this session started (see below).               ║
║  Use them as your PRIMARY grounding source for current facts.   ║
║                                                                  ║
║  ⛔ Do NOT ignore the search results below.                     ║
║  ⛔ Do NOT invent facts not present in the search results.      ║
║  ✅ If the search results do not cover a sub-topic, say so      ║
║     explicitly rather than guessing from training memory.       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Label wrapping the raw Serper.dev output block inside verified_context.
STAGE0_LIVE_LABEL = "━━━ [LIVE SEARCH DATA — VERIFIED via Serper.dev] ━━━"

# Header injected at the top of the broad research context block that wraps
# all pre-flight search snippets passed to every model in Stages 1-4.
# Appears when Stage 0 found relevant results for a non-commodity question.
STAGE0_RESEARCH_BLOCK_HEADER = (
    "=== LIVE RESEARCH DATA — Pre-Flight Search (AI-Planned) ===\n"
    "The following results were retrieved live seconds before this debate.\n"
    "They are your PRIMARY grounding source for current facts.\n"
    "Prioritise these results over training-memory for any claim about\n"
    "current events, prices, news, or the state of the world.\n\n"
)

STAGE0_RESEARCH_BLOCK_FOOTER = "\n\n=== END LIVE RESEARCH DATA ===\n\n"

# System prompt used by _plan_queries_with_ai() in search_engine.py.
# Gemini Flash receives this before generating targeted English search queries
# for any user question.  Kept here so it can be tuned from a single place.
STAGE0_QUERY_PLAN_PROMPT = """\
You are a search-query planner for a research assistant.
Decide whether the question below benefits from live web search results.

If the question is purely theoretical, mathematical, definitional, or \
historical (no current facts needed), output exactly:
NO_SEARCH_NEEDED

Otherwise, output 1–4 Google search queries in English (one per line) that \
would retrieve the most current, factual information needed to answer it.

Rules:
- Output ONLY the queries (or NO_SEARCH_NEEDED) — no numbers, no bullets, \
no explanation.
- Write every query in English, even if the question is in Hebrew.
- Prefer queries that return recent news or current data \
(append the year when relevant).
- Cover different angles: e.g. current status, market data, expert analysis, \
latest developments.
- Translate Hebrew questions to English before composing queries.

Question: {question}
"""

# Phrases in Stage 3 dialectic responses that signal the model REJECTED
# Stage 0 live data that was explicitly provided (silver/ILS commodity data).
# These are only applied when clean_data is non-empty (see logic_engine.py).
# Each match deducts 40 points from the reliability score.
# NOTE: generic uncertainty ("cannot verify", "I don't know") is NOT listed
# here — admitting honest ignorance is correct behaviour, not hallucination.
CONTEXTUAL_HALLUCINATION_TRIGGERS: tuple = (
    "if the price is correct",
    "if the stage 0",
    "taking stage 0 at face value",
    "cannot confirm the stage 0",
    "if the provided price",
    "if these values are real",
    "if we assume the price",
    "treating as given",
    "assuming this data is correct",
    "if the data is accurate",
)

# Phrases that indicate the model explicitly reframed LIVE Stage 0 commodity
# data as conditional/hypothetical.  Only applied when clean_data is present.
# Each match deducts 50 points (hard penalty).
HARD_HALLUCINATION_TRIGGERS: tuple = (
    "if the stage 0 price is correct",
    "assuming stage 0 is accurate",
    "if stage 0 data is real",
)

# ── Stage 0 progress UI strings ────────────────────────────────────────────
UI_STAGE0_SEARCHING    = "🔍 **שלב 0:** סריקת נתוני אמת ב-Serper.dev..."
UI_TECH_LOG_TITLE      = "פרוטוקול מחשבה טכני (למפתחים)"

# ── Firebase Authentication UI strings ─────────────────────────────────────
# v2 — cache-bust 2026-03-11
UI_AUTH_TITLE          = "🔐 Account Access"
UI_AUTH_LOGIN_TAB      = "Login"
UI_AUTH_SIGNUP_TAB     = "Sign Up"
UI_AUTH_EMAIL_LABEL    = "Email"
UI_AUTH_PASSWORD_LABEL = "Password"
UI_AUTH_LOGIN_BTN      = "Login"
UI_AUTH_SIGNUP_BTN     = "Create Account"
UI_AUTH_LOGOUT_BTN     = "Logout"
UI_AUTH_LOGGED_IN_AS   = "Logged in as"
UI_AUTH_GATE_MSG       = (
    "### 🔐 Login Required\n\n"
    "Please use the **Login** panel in the sidebar to sign in or create "
    "a free account to access the AI Council."
)
UI_AUTH_SUCCESS_LOGIN  = "✅ Welcome back!"
UI_AUTH_SUCCESS_SIGNUP = "✅ Account created — welcome!"
UI_AUTH_NO_API_KEY_MSG = (
    "⚠️ `FIREBASE_WEB_API_KEY` is not set. "
    "Authentication is unavailable."
)
UI_NEW_QUERY_BUTTON    = "🔄 שאילתה חדשה"
UI_CACHE_HIT_BANNER    = (
    "💾 **תשובה שמורה נמצאה בהיסטוריה** — טוען מהמטמון במקום להפעיל מחדש את ה-API "
    "(חיסכון בעלויות). לחץ **🔄 שאילתה חדשה** ואז **{btn}** כדי להריץ שאילתה רענן."
)
UI_CACHE_RERUN_BTN     = "▶️ הרץ בכל זאת (API)"

# ── Daily usage quota ────────────────────────────────────────────────────────
DAILY_QUERY_LIMIT      = 5
API_TIMEOUT_SECONDS    = 90   # max seconds to wait for any single model API call
UI_DAILY_USAGE_TPL     = "📊 שימוש יומי: {count}/{limit} שאילתות"
UI_QUOTA_REACHED       = "🚀 הגעת למכסה היומית שלך. נתראה מחר!"

# ── Follow-up question feature ───────────────────────────────────────────────
UI_FOLLOWUP_LABEL          = "💬 שאלת המשך"
UI_FOLLOWUP_PLACEHOLDER    = "הקלד שאלת המשך בהתבסס על הדיון למעלה..."
UI_FOLLOWUP_SUBMIT_BTN     = "🔄 הפעל דיון המשך"

# Context block injected into verified_context when a follow-up debate is run.
# {prev_question} and {prev_answer} are substituted at runtime.
FOLLOWUP_CONTEXT_BLOCK = """\
╔══════════════════════════════════════════════════════════════════╗
║               FOLLOW-UP QUESTION — PRIOR DEBATE CONTEXT         ║
╠══════════════════════════════════════════════════════════════════╣
║  This is a follow-up to a previous council debate.              ║
║  Use the prior debate context below as ESTABLISHED FOUNDATION.  ║
║  Do NOT repeat what was already settled — build upon it.        ║
╚══════════════════════════════════════════════════════════════════╝

ORIGINAL QUESTION (previous debate):
{prev_question}

FINAL SYNTHESISED ANSWER FROM PREVIOUS DEBATE:
{prev_answer}

══════════════════════════════════════════════════════════════════
Now answer the NEW follow-up question that follows, treating the
above as verified prior-debate consensus.
══════════════════════════════════════════════════════════════════
"""

# ── Email verification ───────────────────────────────────────────────────────
UI_EMAIL_NOT_VERIFIED  = "📧 נא לאמת את כתובת המייל שלך כדי להמשיך."
UI_RESEND_VERIFICATION = "📨 שלח מייל אימות מחדש"
UI_RESEND_SUCCESS      = "✅ מייל אימות נשלח! בדוק את תיבת הדואר שלך."
UI_REFRESH_VERIFY_BTN  = "🔄 אימתתי — רענן סטטוס"
UI_STILL_NOT_VERIFIED  = "⚠️ המייל עדיין לא אומת. בדוק את תיבת הדואר ולחץ על הקישור."
# Shown when commodity data (silver + ILS) was found
UI_STAGE0_COMPLETE_COMMODITY = "✅ **Stage 0 Complete:** Live Silver: {silver} | USD/ILS: {rate}"
# Shown when general search results were found (non-commodity query)
UI_STAGE0_COMPLETE_GENERAL   = "✅ **Stage 0 Complete:** {n} live results fetched for your query"
# Shown when Serper is unavailable or returned nothing
UI_STAGE0_COMPLETE_NONE      = "⚠️ **Stage 0:** No live data available (Serper unavailable)"

# Mandatory instruction that follows the live data line.
# Written to be maximally unambiguous: discard training memory for prices.
STAGE0_MARKET_INSTRUCTION = (
    "MANDATORY INSTRUCTION — DO NOT SKIP:\n"
    "The market figures shown above were retrieved via live web search moments "
    "before this conversation began.  They reflect the CURRENT real-world values.\n"
    "Your training data contains OUTDATED price information.  You MUST:\n"
    "  1. Use the live figures above for EVERY calculation, estimate, and reference.\n"
    "  2. IGNORE any conflicting prices or exchange rates stored in your training.\n"
    "  3. Never substitute a training-memory figure for any value already listed above.\n"
    "Failure to follow this instruction will produce a factually incorrect answer."
)

# Educational framing prepended when financial/numismatic content is detected.
# Purpose: prevents GPT-4o content-policy refusals that trigger on direct
# financial-advice framing, by reframing the task as academic research.
EDUCATIONAL_FRAMING_NOTE = (
    "PLATFORM CONTEXT (read before answering):\n"
    "This is an academic numismatic research and market-education platform. "
    "All valuations, price analyses, and market data provided here are strictly "
    "for RESEARCH, EDUCATIONAL, and INFORMATIONAL purposes.  Nothing in this "
    "conversation constitutes financial, investment, or legal advice.  Treat this "
    "as a scholarly analysis task, not a financial consultation."
)
# ── Stage 3b UI strings ───────────────────────────────────────────────────
UI_STAGE3B_LABEL          = "Stage 4 — Focused Debate"
UI_STAGE3B_POINT_HEADER   = "### 🔥 Point {n}: {summary}"
UI_STAGE3B_ROUND1_LABEL   = "Round 1 — Opening Positions"
UI_STAGE3B_ROUND2_LABEL   = "Round 2 — Direct Responses"
UI_STAGE3B_EXPANDER_MODEL = "⚔️ {label} — Point {n}, Round {r}"
UI_STAGE3B_NO_DEBATES     = "_No significant disagreements detected — models converged after Stage 3._"
UI_SPINNER_STAGE3B        = "⚔️ Stage 4 — Focused Debate running …"

UI_SECTION_CITATIONS  = "📚 Verified Sources & References"
UI_CITATIONS_NO_DATA  = ("ℹ️ Note: This response was generated based on model knowledge "
                         "and logical synthesis. No real-time grounding data was retrieved.")
UI_CITATION_VISIT     = "Visit Source →"
PDF_REPORT_TITLE      = "AI-Playground Research Report"
PDF_BTN_LABEL         = "📥 הורדת דוח מחקר (PDF)"
PDF_GENERATED_BY      = "Generated by AI-Playground Council"

# PDF rendering colours for the Technical Breakdown box.
# These mirror the .tech-breakdown CSS class so the downloaded PDF visually
# matches the on-screen Certified Report.  Used exclusively by report_generator.py.
PDF_TECH_BREAKDOWN_BG     = (241, 245, 249)   # slate-100  ← .tech-breakdown background
PDF_TECH_BREAKDOWN_BORDER = (148, 163, 184)   # slate-400  ← .tech-breakdown border
PDF_TECH_BREAKDOWN_LABEL  = "Technical Breakdown — Verified Calculation"
UI_HISTORY_TITLE           = "📜 Previous Debates"
UI_HISTORY_EMPTY           = "No history found."
UI_HISTORY_CLEAR           = "🗑️ Clear History"
UI_HISTORY_BANNER          = "🕐 Viewing a saved debate — no API tokens consumed."
UI_HISTORY_GROUP_TODAY     = "Today"
UI_HISTORY_GROUP_YESTERDAY = "Yesterday"
UI_HISTORY_GROUP_EARLIER   = "Earlier This Month"
UI_HISTORY_GROUP_OLDER     = "Older"
UI_HISTORY_VERIFIED_BADGE  = "🛡️"
UI_HISTORY_PDF_BTN         = "📥 PDF"

# ── Help & Methodology sidebar section ────────────────────────────────────
UI_HELP_EXPANDER_TITLE = 'כיצד לקרוא את הדו"ח?'
UI_HELP_COUNCIL        = "**המועצה:** שימוש ב-4 מודלים שונים לביקורת צולבת."
UI_HELP_RELIABILITY    = (
    "**מדד האמינות:** ציון 100% (ירוק) אומר שהמודל נצמד לנתוני "
    "האמת של Stage 0 ולא הסתמך על זיכרון אימון."
)
UI_HELP_TECH_APPENDIX  = (
    "**הנספח הטכני:** כל חישוב מגובה בנוסחה שקופה הניתנת לאימות עצמאי."
)
UI_HELP_DATA_DATE_TPL  = "**תאריך עדכון:** הנתונים מעודכנים ל{day_name}, {day} ב{month} {year}."

# Hebrew localisation tables used to render UI_HELP_DATA_DATE_TPL.
# weekday() → 0=Mon … 6=Sun
HEBREW_WEEKDAYS = [
    "יום שני", "יום שלישי", "יום רביעי", "יום חמישי",
    "יום שישי", "יום שבת", "יום ראשון",
]
HEBREW_MONTHS = [
    "", "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני",
    "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
]
UPLOAD_IMAGE_LABEL    = "📷 Upload visual assets for analysis (optional — up to 4 images)"
IMAGE_PREVIEW_HEADER  = "📷 Visual Assets"
RTL_CSS               = """
<style>
/* ── RTL / BiDi layout fix for AI-Playground ─────────────────────────── */
/* Paragraphs, list items, and headings follow their natural text direction */
.stMarkdown p,
.stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    direction: rtl;
    text-align: right;
    unicode-bidi: embed;
}
/* Code blocks and inline code stay strictly LTR */
.stMarkdown pre,
.stMarkdown code,
.stMarkdown .highlight {
    direction: ltr !important;
    text-align: left !important;
    unicode-bidi: isolate;
}
</style>
"""

# CSS for the Mission Control Dashboard shown during debate processing.
# Injected once at debate start; animation names are referenced from Python-generated HTML.
MISSION_CONTROL_CSS = """
<style>
/* ── Mission Control keyframes ───────────────────────────────────────── */
@keyframes sd-pulse {
    0%,100% { transform: scale(1);    }
    50%      { transform: scale(1.12); }
}
@keyframes sd-fade-in {
    from { opacity: 0; transform: translateY(3px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* ── Circle row ──────────────────────────────────────────────────────── */
.sd-wrap {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 6px;
    padding: 14px 0 6px;
    flex-wrap: wrap;
}
.sd-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    min-width: 66px;
}
.sd-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15em;
    font-weight: 700;
    transition: background 0.25s, border-color 0.25s, box-shadow 0.25s;
    direction: ltr;
}
.sd-connector {
    width: 22px;
    height: 3px;
    background: #e2e8f0;
    margin-top: 23px;
    border-radius: 2px;
    flex-shrink: 0;
    transition: background 0.3s;
}
.sd-label {
    font-size: 0.61em;
    font-weight: 600;
    text-align: center;
    max-width: 66px;
    line-height: 1.3;
    letter-spacing: 0.02em;
    direction: ltr;
}

/* ── Tech Log (collapsible <details>) ────────────────────────────────── */
.sd-details { margin: 6px 0 2px; }
.sd-summary {
    cursor: pointer;
    font-size: 0.78em;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.04em;
    padding: 4px 10px;
    border-radius: 4px;
    background: rgba(241, 245, 249, 0.85);
    list-style: none;
    user-select: none;
    direction: ltr;
}
.sd-summary::-webkit-details-marker { display: none; }
.sd-summary::before { content: "▶ "; font-size: 0.75em; }
details[open] .sd-summary::before { content: "▼ "; }
.sd-log-wrap {
    background: rgba(248, 250, 252, 0.92);
    border: 1px solid #e2e8f0;
    border-radius: 0 0 8px 8px;
    padding: 8px 14px;
    font-size: 0.80em;
    color: #475569;
    font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
    direction: ltr;
    text-align: left;
}
.sd-log-line {
    animation: sd-fade-in 0.2s ease;
    padding: 2px 0;
    border-bottom: 1px solid rgba(226, 232, 240, 0.6);
    line-height: 1.55;
}
.sd-log-line:last-child { border-bottom: none; }
</style>
"""

# ── Reliability Gauge — SVG speedometer template ───────────────────────────
# Placeholders (all filled by render_reliability_gauge() in app.py):
#   {uid}         — unique 8-char hex used for scoped CSS animation names
#   {score}       — integer 0-100 shown inside the gauge
#   {score_x}     — SVG x of the arc endpoint (pre-calculated float string)
#   {score_y}     — SVG y of the arc endpoint (pre-calculated float string)
#   {angle}       — CSS needle rotation in degrees (-90 = left, +90 = right)
#   {color}       — hex colour matching the score tier
#   {label}       — tier label ("High Consistency" / "Partial Revision" / etc.)
#   {model_label} — human-readable model name shown below the gauge
#
# Zone boundaries (hardcoded in the track):
#   Red  0-50  : (20,100) → (100, 20)
#   Amber 50-80: (100, 20) → (165, 53)
#   Green 80-100:(165, 53) → (180,100)
RELIABILITY_GAUGE_HTML = """\
<style>
@keyframes rg-swing-{uid} {{
    from {{ transform: rotate(-90deg); }}
    to   {{ transform: rotate({angle}deg); }}
}}
.rg-needle-{uid} {{
    transform-origin: 100px 100px;
    transform: rotate({angle}deg);
    animation: rg-swing-{uid} 1.2s cubic-bezier(.22,.61,.36,1) 0.3s both;
}}
</style>
<div style="text-align:center;display:inline-block;min-width:148px;padding:6px 8px">
  <svg viewBox="0 0 200 128" width="148" height="95" xmlns="http://www.w3.org/2000/svg">
    <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e2e8f0" stroke-width="14" stroke-linecap="round"/>
    <path d="M 20 100 A 80 80 0 0 1 100 20"  fill="none" stroke="#ef4444" stroke-width="14" stroke-linecap="butt" opacity="0.22"/>
    <path d="M 100 20  A 80 80 0 0 1 165 53" fill="none" stroke="#f59e0b" stroke-width="14" stroke-linecap="butt" opacity="0.22"/>
    <path d="M 165 53  A 80 80 0 0 1 180 100" fill="none" stroke="#10b981" stroke-width="14" stroke-linecap="butt" opacity="0.22"/>
    <path d="M 20 100 A 80 80 0 0 1 {score_x} {score_y}" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"/>
    <line x1="100" y1="100" x2="100" y2="28" stroke="{color}" stroke-width="3.5" stroke-linecap="round" class="rg-needle-{uid}"/>
    <circle cx="100" cy="100" r="7" fill="{color}"/>
    <text x="100" y="120" text-anchor="middle" font-size="20" font-weight="800" fill="{color}" font-family="system-ui,sans-serif">{score}</text>
  </svg>
  <div style="font-size:0.72em;font-weight:700;color:#475569;letter-spacing:0.05em;text-transform:uppercase;margin-top:1px;line-height:1.3">{model_label}</div>
  <div style="font-size:0.68em;font-weight:600;margin-top:2px;color:{color}">{label}</div>
</div>\
"""
