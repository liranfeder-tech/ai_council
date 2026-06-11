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
FALLBACK_MODEL_ID  = "gemini-3.1-pro-preview"   # human-readable reference / logging


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
MODELS = {
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
        "id":       "gemini-3.5-flash",
        "label":    "Gemini 3.5 Flash (Google)",
        "provider": "google",
        "color":    "#3B82F6",   # blue
    },
    # Dedicated entry for the Gemini 3.1 Pro fallback / master synthesis model
    "gemini_pro": {
        "id":       "gemini-3.1-pro-preview",
        "label":    "Gemini 3.1 Pro (Google)",
        "provider": "google",
        "color":    "#6366F1",   # indigo
    },
    "grok": {
        "id":       "grok-4.3",
        "label":    "Grok 4.3 (xAI)",
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

## 📊 Step 5 — Confidence
End your entire response with exactly these two lines and nothing after them:
CONFIDENCE: <a single integer from 0 to 100 — your confidence in the Final Answer>
Most likely to change my mind: <one sentence naming the specific evidence that would most shift your conclusion>

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

End your entire response with exactly these two lines and nothing after them:
CONFIDENCE: <a single integer from 0 to 100 — your confidence in your position after this peer review>
Most likely to change my mind: <one sentence naming the specific evidence that would most shift your conclusion>
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

⚖️ Independence rule: Agreement among the other experts is NOT evidence of \
correctness.  Audit every claim on its merits.  If all answers share an \
assumption, challenge that assumption.  Do not soften your critique because \
others agree.

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

# Devil's-advocate role clause — prepended to ONE deterministically-chosen
# reviewer's critique prompt per target (see logic_engine.run_stage2_cross_critique).
# Forces a genuine opposing case so a shaky consensus is stress-tested rather than
# rubber-stamped.  Used in both regular and academic mode.  No placeholders.
DEVILS_ADVOCATE_CLAUSE = """\
🎭 DEVIL'S ADVOCATE ASSIGNMENT
For this one review you are assigned the devil's advocate role.  However \
convincing the answer looks, build the strongest good-faith counter-case \
against it: find the single weakest load-bearing claim it depends on, attack \
that claim directly with concrete reasoning, and argue for the best \
alternative answer the council may have missed.  Do not hedge and do not \
rubber-stamp — make the most compelling honest opposing case you can, then \
proceed with the structured review below.

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
# Experts are anonymised (Expert A/B/C…); the returned labels are mapped back to
# real model keys in logic_engine._extract_disagreement_points.
# Placeholders: {question}, {critiques_summary}, {dialectic_summary}, {available_labels}
PROMPT_EXTRACT_DISAGREEMENTS = """\
You are a debate analyst reviewing an AI council discussion.

--- Original Question ---
{question}

--- Critique Summary (Stage 2) ---
{critiques_summary}

--- Dialectic Summary (Stage 3) ---
{dialectic_summary}

## Task
Identify exactly 2-3 points where the experts GENUINELY disagree after Stage 3.
Ignore stylistic differences. Focus only on factual or logical contradictions
that were NOT resolved in Stage 3.

For each point, output ONLY this exact JSON structure — nothing else:
[
  {{
    "point_id": 1,
    "summary": "One sentence describing what they disagree on.",
    "experts": ["Expert A", "Expert B"],
    "excerpt": "The specific contested claim or figure, quoted exactly (max 40 words)."
  }}
]

Rules:
- If fewer than 2 genuine disagreements exist, return an empty list: []
- "experts" must contain exact expert labels from this set: {available_labels}
- Never invent disagreements. Only report what is explicit in the text above.
- Return valid JSON only. No preamble, no explanation.
"""

# ── Stage 3b — Debate Round 1: Opening Position ───────────────────────────
# Placeholders: {verified_context}, {your_label}, {question}, {point_summary}, {excerpt}
PROMPT_DEBATE_OPENING = """\
You are {your_label}, participating in a focused one-point debate.

--- Verified Context (live ground truth — defer to these figures over memory) ---
{verified_context}

--- The Question ---
{question}

--- Contested Point ---
{point_summary}

--- Specific Excerpt at Issue ---
{excerpt}

State your position on this specific point in 3-5 sentences.
- Be precise. Reference the excerpt directly.
- Ground every figure in the Verified Context above, not training memory.
- No preamble. No references to previous stages. Just your current position.
- If you hold the same view as before, say so and explain why concisely.
"""

# ── Stage 3b — Debate Round 2: Direct Response ────────────────────────────
# Placeholders: {verified_context}, {your_label}, {question}, {point_summary},
#               {opponent_openings}
PROMPT_DEBATE_RESPONSE = """\
You are {your_label}, in the direct-response round of a focused debate.

--- Verified Context (live ground truth — defer to these figures over memory) ---
{verified_context}

--- The Question ---
{question}

--- Contested Point ---
{point_summary}

--- Opening Positions of the Other Experts ---
{opponent_openings}

Respond directly to the other experts' opening positions in 3-6 sentences.
- Address their actual arguments, not strawmen. Name the expert you are answering.
- Either concede a point (with explicit acknowledgement) or rebut it with clear
  reasoning.
- Ground every figure in the Verified Context above, not training memory.
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

The experts below are anonymised as Expert A, Expert B, Expert C, … .  You do \
NOT know which AI system or vendor any expert is, and you do NOT know which \
expert (if any) authored text you might recognise as your own.  Judge every \
position solely on its evidence and reasoning — never on its presumed source \
or on its position in the list.

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
These are the most reliable truths — highlight them explicitly.  But agreement \
alone is NOT proof: when experts conflict, resolve it by weighing each expert's \
stated CONFIDENCE line and whether the claim SURVIVED the adversarial critique \
and the focused debate — never by counting how many experts took each side.

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
# Academic Mode — prompt variants for peer-reviewed research questions
# ---------------------------------------------------------------------------
# These replace the standard PROMPT_INITIAL / PROMPT_CRITIQUE / etc. when the
# user enables Academic Mode.  The key behavioural differences are:
#   1. Models are instructed to identify the SPECIFIC DELIVERABLE requested
#      (treatment plan, protocol, recommendations…) and actually produce it.
#   2. Peer review explicitly checks whether the deliverable was provided.
#   3. The consensus prompt writes a formal academic paper with a mandatory
#      Primary Deliverable section that matches what was asked.

PROMPT_ACADEMIC_INITIAL = """\
{vision_prefix}{verified_context}\
You are a senior academic researcher conducting a systematic evidence-based analysis.

--- Research Question ---
{question}

## Your Academic Response — follow this structure exactly

### Step 1 — Identify the Specific Deliverable
State in ONE sentence what the researcher is specifically asking you to produce \
as output. Examples:
  "The researcher requests a structured treatment protocol for X."
  "The researcher requests clinical recommendations for Y."
  "The researcher requests a comparative effectiveness analysis of Z."

### Step 2 — Literature Evidence
Summarize the most relevant empirical findings from the academic papers in the \
context above (if provided) and from your training knowledge. For each claim \
indicate the source as (Author, Year) or note "training knowledge".

### Step 3 — PRIMARY DELIVERABLE
**This is the most important section.** Provide exactly what was identified in \
Step 1 — do NOT substitute a general analysis:
- If a treatment plan was requested → write a structured, step-by-step plan \
  with eligibility criteria, phases, dosage ranges, and monitoring parameters
- If clinical recommendations were requested → write numbered, actionable \
  evidence-graded recommendations (e.g. Grade A/B/C or Level I/II/III)
- If a protocol was requested → write inclusion/exclusion criteria, procedure \
  steps, and outcome measures
- If an analysis or comparison was requested → write the analysis with explicit \
  conclusions and effect-size estimates where available

Ground every item in the literature. Use clear markdown subheadings.

### Step 4 — Limitations & Evidence Gaps
What does the current evidence NOT support? What further research is needed?

End your entire response with exactly these two lines and nothing after them:
CONFIDENCE: <a single integer from 0 to 100 — your confidence in the Primary Deliverable>
Most likely to change my mind: <one sentence naming the specific evidence that would most shift your conclusion>
"""

PROMPT_ACADEMIC_CRITIQUE = """\
{vision_prefix}{verified_context}\
You are a peer reviewer for a high-impact academic journal.

--- Research Question ---
{question}

--- {author_label}'s Response ---
{author_answer}

--- Other Researchers' Responses (for context) ---
{other_answers}

⚖️ Independence rule: Agreement among the other experts is NOT evidence of \
correctness.  Audit every claim on its merits.  If all responses share an \
assumption, challenge that assumption.  Do not soften your critique because \
others agree.

## Peer Review Tasks

### 1. Deliverable Check — PRIMARY REVIEW TASK
Did the researcher actually produce what was specifically requested (treatment \
plan, protocol, recommendations, analysis)?

- If they provided a full, actionable deliverable: confirm and note its quality
- If they produced only a general summary or discussion INSTEAD of the deliverable: \
  mark **⚠️ DELIVERABLE GAP DETECTED** — specify exactly what is missing

### 2. Evidence Quality Audit
For each specific claim in the deliverable section, classify as:
- **EVIDENCE-BASED** — supported by cited literature or established consensus
- **EXPERT OPINION** — reasonable but not empirically cited (acceptable if noted)
- **UNSUPPORTED** — speculative with no evidence base

### 3. Methodological Critique
Are the reasoning and analysis methods appropriate? Logical gaps? Confounders \
not addressed? Missing sub-populations?

### 4. Quality Score
Integer 1–10. Mandatory deduction: −3 points if the specific deliverable was \
not provided. One sentence of justification.
"""

PROMPT_ACADEMIC_DIALECTIC = """\
{vision_prefix}{verified_context}\
You are a researcher responding to peer review of your academic work.

--- Research Question ---
{question}

--- Your Initial Response ---
{your_initial_answer}

--- Peer Review Critiques ---
{critique_of_your_answer}

## Your Response

For each critique, respond with one of:

### DEFEND — the critique is factually incorrect or based on misreading
Quote the critique. Explain why your position is supported by evidence.

### REFINE — the critique reveals a genuine gap or error
Quote the critique. State "I acknowledge this gap:". Provide the corrected \
or missing content.

**CRITICAL RULE:** If any reviewer flagged a **DELIVERABLE GAP** (you provided \
a general discussion instead of the specifically requested output), you MUST \
fill that gap NOW. Do not merely acknowledge it — write the missing deliverable \
in full, grounded in evidence.

End your entire response with exactly these two lines and nothing after them:
CONFIDENCE: <a single integer from 0 to 100 — your confidence in your refined position>
Most likely to change my mind: <one sentence naming the specific evidence that would most shift your conclusion>
"""

PROMPT_ACADEMIC_CONSENSUS = """\
{vision_prefix}{verified_context}\
You are the Editor-in-Chief synthesising a multi-expert peer review process. \
Your output will be presented as a formal academic research report.

The experts below are anonymised as Expert A, Expert B, Expert C, … .  You do \
NOT know which AI system or vendor any expert is, and you do NOT know which \
expert (if any) authored text you might recognise as your own.  Weigh every \
position solely on its evidence and reasoning — never on its presumed source \
or list position.  When experts conflict, resolve it by weighing each expert's \
stated CONFIDENCE line and whether the claim survived peer critique and the \
focused debate, not by majority count.

--- Research Question ---
{question}

--- Expert Responses (Stage 1) ---
{all_answers_block}

--- Peer Reviews (Stage 2) ---
{all_critiques_block}

--- Author Responses (Stage 3) ---
{all_dialectic_block}

--- Focused Expert Debate (Stage 3b) ---
{focused_debate_block}

## CRITICAL INSTRUCTION
Your PRIMARY obligation is to deliver what the research question specifically \
requests. If a treatment plan was asked for — the report MUST contain a \
treatment plan. If recommendations were asked for — the report MUST contain \
them. Do NOT replace the requested deliverable with a general synthesis of \
what the models said about it.

## Academic Report Structure

### Abstract
(150–200 words) State the research question, method (multi-model peer review), \
key findings, and the primary deliverable.

### Background
What is already known about this topic from the literature? Cite sources.

### Evidence Synthesis
Identify the core empirical findings that the expert discussion converged on. \
Note any unresolved disagreements with brief explanation.

### [PRIMARY DELIVERABLE — name this section after what was requested]
Examples: "Proposed Treatment Protocol", "Clinical Recommendations", \
"Implementation Framework", "Comparative Analysis", "Diagnostic Protocol"

This section is MANDATORY and must be:
- Comprehensive and actionable (not vague)
- Explicitly evidence-graded where possible
- Organised with clear sub-headings (phases, steps, criteria, etc.)
- The direct answer to the research question

### Discussion
Clinical or practical implications. Address major expert disagreements and \
their resolution. Implications for practice and policy.

### Limitations
What does the evidence NOT yet support? Study design gaps? Generalisability? \
What further research is needed?

### References
List all academic papers cited from the literature context above, formatted:
  [n] Authors (Year). Title. Journal. URL

Present as a unified report. Do not attribute content to individual models.
"""

# ---------------------------------------------------------------------------
# Academic Mode — UI strings (Hebrew)
# ---------------------------------------------------------------------------

UI_ACADEMIC_TOGGLE     = "🎓 מצב מחקר אקדמי"
UI_ACADEMIC_CAPTION    = (
    "מחפש מאמרים עמיתים ב-Semantic Scholar ו-PubMed ומנחה את המודלים "
    "לספק את הפלט הספציפי שנדרש (תוכנית טיפול, המלצות, פרוטוקול וכד׳) "
    "בפורמט של מאמר מחקרי מלא עם הפניות."
)
UI_ACADEMIC_CHECKBOX   = "הפעל מצב מחקר אקדמי"
UI_ACADEMIC_YEAR_FROM  = "משנת (פרסום)"
UI_ACADEMIC_YEAR_TO    = "עד שנת (פרסום)"
UI_ACADEMIC_SEARCHING  = "🔍 מחפש מאמרים עמיתים..."
UI_ACADEMIC_FOUND      = "📚 נמצאו {n} מאמרים עמיתים — ישמשו כהקשר לדיון האקדמי"
UI_ACADEMIC_NOT_FOUND  = "📭 לא נמצאו מאמרים עמיתים — הדיון יתבסס על ידע המודלים"
UI_ACADEMIC_STAGE0_LBL = "Stage 0 — Academic Literature Search"

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
FALLBACK_NOTICE       = "⚠️ Primary Master Model failed. Using Gemini 3.1 Pro as fallback."
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
DAILY_QUERY_LIMIT      = 10
API_TIMEOUT_SECONDS    = 150  # max seconds to wait for any single model API call
UI_DAILY_USAGE_TPL     = "📊 שימוש יומי: {count}/{limit} שאילתות"
UI_QUOTA_REACHED       = "🚀 הגעת למכסה היומית שלך. נתראה מחר!"

# ── Follow-up question feature ───────────────────────────────────────────────
UI_FOLLOWUP_LABEL          = "💬 שאלת המשך"
UI_FOLLOWUP_PLACEHOLDER    = "הקלד שאלת המשך בהתבסס על הדיון למעלה..."
UI_FOLLOWUP_IMG_LABEL      = "🖼️ צרף תמונות לניתוח (אופציונלי — עד 4)"
UI_FOLLOWUP_DATA_LABEL     = "📎 צרף קבצי נתונים לניתוח (CSV, Excel, PDF, JSON, TXT ועוד)"
UI_FOLLOWUP_DATA_LOADED    = "✅ {count} קבצים נטענו (~{chars:,} תווים)"
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
# ---------------------------------------------------------------------------
# Data File Upload — Analysis Mode
# ---------------------------------------------------------------------------

# Streamlit type filter (extensions without the dot)
FILE_UPLOAD_ACCEPTED_TYPES = [
    "csv", "tsv", "txt", "md", "json", "jsonl",
    "pdf", "xlsx", "xls", "docx", "log",
]

FILE_UPLOAD_MAX_SIZE_MB    = 15       # MB — per-file size shown in the UI hint
FILE_UPLOAD_MAX_FILES      = 6        # max files per upload batch
FILE_UPLOAD_MAX_ROWS_CSV   = 300      # rows shown for CSV / Excel preview
FILE_UPLOAD_MAX_TOTAL_CHARS = 150_000 # chars across all files combined

# Injected at the top of the file context block.
# {file_count} and {total_chars} are substituted at runtime.
FILE_UPLOAD_CONTEXT_HEADER = """\
╔══════════════════════════════════════════════════════════════════╗
║               UPLOADED DATA FILES — ANALYSIS CONTEXT            ║
║  Files: {file_count}  |  Size: ~{total_chars:,} chars           ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTRUCTION: The file contents below are VERIFIED INPUT DATA.  ║
║  Base all findings and analysis strictly on this data.          ║
║  Do NOT invent rows, metrics, or values not present here.       ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Default analysis question when files are uploaded without a custom question
FILE_UPLOAD_DEFAULT_QUESTION = (
    "Please perform a thorough analysis of the uploaded data. "
    "Identify: (1) key trends and patterns, (2) anomalies or outliers, "
    "(3) actionable insights and recommendations, "
    "(4) data quality issues if any. "
    "Present your findings clearly with specific references to the data."
)

# UI labels
UI_FILE_UPLOAD_LABEL       = "📎 העלה קבצי נתונים לניתוח (CSV, Excel, PDF, JSON, TXT ועוד)"
UI_FILE_UPLOAD_LOADED      = "✅ {count} קבצים נטענו (~{chars:,} תווים)"
UI_FILE_UPLOAD_WARNINGS    = "⚠️ {count} הערות טעינה"
UI_FILE_UPLOAD_CLEAR_BTN   = "🗑️ נקה קבצים"
UI_FILE_UPLOAD_EMPTY       = "⚠️ לא ניתן לחלץ תוכן מהקבצים שהועלו."
UI_FILE_UPLOAD_FILES_HDR   = "📄 קבצים שנטענו"

# ---------------------------------------------------------------------------
# Code Review / Project Analysis Mode
# ---------------------------------------------------------------------------

# File extensions included in a project scan by default.
CODE_REVIEW_EXTENSIONS: set = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml",
    ".md", ".sql", ".sh", ".bash", ".tf",
}

# Directories always skipped during a project scan.
CODE_REVIEW_SKIP_DIRS: set = {
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", ".idea", ".vscode", "coverage",
    ".pytest_cache", ".mypy_cache", ".tox", "pdf_cache",
}

CODE_REVIEW_MAX_FILE_SIZE   = 50_000    # bytes — files larger than this are skipped
CODE_REVIEW_MAX_TOTAL_CHARS = 120_000   # total characters across all files
CODE_REVIEW_MAX_FILES       = 60        # maximum number of files per scan

# Injected at the top of verified_context when project files are loaded.
# {folder_name}, {file_count}, {total_chars} are substituted at runtime.
CODE_REVIEW_CONTEXT_HEADER = """\
╔══════════════════════════════════════════════════════════════════╗
║               PROJECT CODE CONTEXT — {folder_name}              ║
║  Files included: {file_count}  |  Size: ~{total_chars:,} chars  ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTRUCTION: The source files below are VERIFIED GROUND TRUTH.  ║
║  Base all findings strictly on what is present in these files.   ║
║  Do NOT speculate about code that is not shown.                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Default question auto-filled when the user activates Code Review mode
# without typing a custom question.
CODE_REVIEW_DEFAULT_QUESTION = (
    "Please perform a thorough code review of this project. "
    "Identify: (1) bugs and logic errors, (2) security vulnerabilities, "
    "(3) code quality and maintainability issues, "
    "(4) architectural patterns and improvement opportunities. "
    "For each finding state its severity (Critical / High / Medium / Low) "
    "and suggest a concrete fix."
)

# UI strings for the Code Review panel
UI_CODE_REVIEW_TOGGLE       = "📁 ניתוח קוד מפרויקט מקומי"
UI_CODE_REVIEW_PATH_LABEL   = "נתיב לתיקיית הפרויקט"
UI_CODE_REVIEW_PATH_PLACEHOLDER = "/path/to/your/project"
UI_CODE_REVIEW_SCAN_BTN     = "🔍 סרוק פרויקט"
UI_CODE_REVIEW_SCANNING     = "⏳ סורק קבצי פרויקט…"
UI_CODE_REVIEW_SCANNED      = "✅ {count} קבצים נטענו מ-**{name}** (~{chars:,} תווים)"
UI_CODE_REVIEW_WARNINGS     = "⚠️ {count} קבצים הושמטו (גודל / סוג)"
UI_CODE_REVIEW_EMPTY        = "⚠️ לא נמצאו קבצי קוד בתיקייה שנבחרה."
UI_CODE_REVIEW_NOT_FOUND    = "❌ תיקייה לא נמצאה: {path}"
UI_CODE_REVIEW_FILES_HEADER = "📄 קבצים שנטענו"
UI_CODE_REVIEW_CLEAR_BTN    = "🗑️ נקה"

# ---------------------------------------------------------------------------
# API Key Management Dialog
# ---------------------------------------------------------------------------

# Maps provider key (same as in MODELS) → (env var name, display label, placeholder hint)
PROVIDER_KEY_CONFIG: dict = {
    "anthropic": {
        "env":         "ANTHROPIC_API_KEY",
        "label":       "Claude (Anthropic)",
        "description": "רוצה להשתמש בחשבון Claude האישי שלך? "
                       "הדבק כאן את הקוד שקיבלת מ-Anthropic.",
        "placeholder": "sk-ant-api03-…",
        "url":         "https://console.anthropic.com/settings/keys",
    },
    "openai": {
        "env":         "OPENAI_API_KEY",
        "label":       "GPT-4o (OpenAI)",
        "description": "רוצה להשתמש בחשבון ChatGPT/OpenAI האישי שלך? "
                       "הדבק כאן את המפתח שלך.",
        "placeholder": "sk-proj-…",
        "url":         "https://platform.openai.com/api-keys",
    },
    "google": {
        "env":         "GOOGLE_API_KEY",
        "label":       "Gemini (Google)",
        "description": "רוצה להשתמש בחשבון Google AI האישי שלך? "
                       "הדבק כאן את המפתח מ-Google AI Studio.",
        "placeholder": "AIzaSy…",
        "url":         "https://aistudio.google.com/apikey",
    },
    "xai": {
        "env":         "XAI_API_KEY",
        "label":       "Grok (xAI)",
        "description": "רוצה להשתמש בחשבון xAI/Grok האישי שלך? "
                       "הדבק כאן את המפתח מקונסולת xAI.",
        "placeholder": "xai-…",
        "url":         "https://console.x.ai/",
    },
}

# Status indicators shown next to each provider in the dialog
APIKEY_STATUS_ICON = {
    "valid":   "🟢",
    "quota":   "🟠",
    "invalid": "🔴",
    "none":    "⚪",
}
APIKEY_STATUS_LABEL = {
    "valid":   "מחובר",
    "quota":   "מכסה מלאה",
    "invalid": "מפתח שגוי",
    "none":    "Fallback (מפתח האפליקציה)",
}

# UI strings for the key management dialog / sidebar
UI_APIKEY_BTN_SIDEBAR      = "🔑 מפתחות API"
UI_APIKEY_DIALOG_TITLE     = "הגדרת מפתחות API"
UI_APIKEY_DIALOG_SUBTITLE  = (
    "השתמש במפתחות ה-API האישיים שלך כדי לשלוט בעלויות "
    "ולהסיר את מגבלת השאילתות היומית. "
    "המפתחות נשמרים בדפדפן שלך בלבד — לא בשרת ולא במסד הנתונים."
)
UI_APIKEY_FIELD_LABEL      = "מפתח API"
UI_APIKEY_VALIDATE_SAVE    = "🔍 אמת ושמור"
UI_APIKEY_VALIDATING       = "בודק מפתחות…"
UI_APIKEY_CLEAR_BTN        = "🗑️ נקה הכל"
UI_APIKEY_SAVED_OK         = "✅ מפתחות נשמרו בדפדפן — שאילתות ישתמשו בחשבונך"
UI_APIKEY_PARTIAL_SAVED    = "⚠️ חלק מהמפתחות נשמרו — ראה שגיאות למטה"
UI_APIKEY_CLEARED          = "🗑️ מפתחות נוקו — האפליקציה חוזרת ל-Fallback"
UI_APIKEY_ACTIVE_BADGE     = "BYOK"
UI_APIKEY_GET_KEY_LINK     = "קבל מפתח ←"
UI_APIKEY_UNCHANGED_SKIP   = "_(לא השתנה — נשמר כפי שהוא)_"

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


# ---------------------------------------------------------------------------
# Media Generation — prompt templates
# ---------------------------------------------------------------------------

PROMPT_MEDIA_IMAGE_BRIEF = """\
You are a senior creative director with 15+ years of experience creating visual \
advertising for mobile apps and digital marketing campaigns.

Your task: read the AI council debate answer below and convert its core marketing \
message into a single, highly optimised gpt-image-2 image-generation prompt.

--- AI Council Final Answer ---
{final_answer}

--- Additional Campaign Context ---
{campaign_context}

{ref_section}
## Output Rules — follow EXACTLY

1. Output ONLY the image prompt — no preamble, no explanation, no markdown headers.
2. Write in English (gpt-image-2 performs best in English).
3. Structure the prompt as: \
[Shot type & composition] + [Subject / scene] + [Visual style] + [Lighting] + [Mood / atmosphere]
4. Be cinematic and specific — avoid vague adjectives like "beautiful", "amazing", "stunning".
5. Length: 2-3 sentences, no more than 160 words.
6. Do NOT include text, logos, words, UI elements, or watermarks in the scene description.
7. The image must communicate the product's core value proposition through visual \
storytelling alone — show the outcome, not the app interface.
8. For safety / personal-protection apps: show a real person in a relatable dangerous \
scenario where calm, competence, or rescue is the emotional payoff.

Write the gpt-image-2 prompt now:"""

PROMPT_MEDIA_VIDEO_BRIEF = """\
You are a senior creative director specialising in ultra-short video ads (5-6 seconds) \
for mobile apps on Apple Ads, TikTok, Instagram Reels, and YouTube Shorts.

You have ONE job: describe a single, hyper-focused moment that shows the product being \
used in a specific, realistic scenario — nothing more.

--- Original Product / Campaign Brief ---
{question}

--- AI Council Key Insights ---
{final_answer}

--- Additional Campaign Context ---
{campaign_context}

{ref_section}
## CRITICAL RULES — violating any rule makes the prompt worthless

### FORBIDDEN (never include these):
- Generic lifestyle footage (woman walking down street, entering home, drinking coffee)
- Abstract metaphors or symbolic imagery
- Vague emotional scenes with no clear product connection
- More than ONE scene or location
- Any narrative that requires more than 6 seconds to understand

### REQUIRED (every prompt must have ALL of these):
1. ONE specific character in ONE specific situation that directly demonstrates the product
2. The EXACT product action happening on screen (e.g. whispers a code word, presses button, \
   phone screen shows alert) — be precise, not implied
3. A visible consequence or reaction within the same clip (e.g. contact receives alert, \
   rescue arrives, location ping appears)
4. Camera: specify shot type + movement (close-up, handheld push-in, over-shoulder, etc.)
5. Lighting + colour grade that matches the emotional tone (thriller blue, warm golden hour, etc.)
6. Length: 3-4 sentences, max 180 words

### STRUCTURE — follow exactly:
[shot size] + [character + appearance] + [environment] + [ONE physical action in progress] + [camera] + [lighting]

### HARD LIMITS:
- Max 50 words total
- Write as a slightly-animated photograph, not a film synopsis
- FORBIDDEN: "then", "cut to", "next", "screen shows", "notification", \
  "message appears", multiple timeframes, references to sound or music

Write the prompt now — one short paragraph, English only:"""

PROMPT_MEDIA_VIDEO_STORYBOARD = """\
You are writing shot descriptions for an AI text-to-video model.

HARD TECHNICAL LIMITS — the model generates ONE continuous 5-6 second shot per scene:
- CANNOT do: camera cuts, multiple time steps, phone/screen text, UI elements, \
  sound, inner monologue, "then X happens", "cut to", "next we see"
- CAN do: one person making one simple movement, a face changing expression, \
  an object in gentle motion, slow camera drift
- Each shot = a single physical action completable in 3-5 seconds

--- Product ---
{question}

--- Campaign Context ---
{campaign_context}

{ref_section}
Write 3 shot descriptions that together tell this story arc:
  Scene 1 — PROBLEM: the character in the difficult situation (no product yet)
  Scene 2 — ACTIVATION: the single physical gesture of using the product
  Scene 3 — RESOLUTION: someone reacts to receiving the alert

RULES FOR EACH PROMPT — violating any rule ruins the output:
1. HARD LIMIT: 45 words maximum per prompt. Count them.
2. Format: [shot size] + [character + clothing] + [environment] + [ONE action happening NOW] + [camera] + [lighting]
3. Write as if describing a single photograph that is slightly animated
4. FORBIDDEN words: "then", "cut", "next", "screen shows", "phone displays", \
   "message", "notification appears", "meanwhile", "as he/she", "before/after"
5. The action must already be in progress — not about to happen

Output ONLY a JSON array — no explanation, no markdown, no extra text:
[
  {{"scene_number": 1, "title": "סצינה 1 — <2-3 Hebrew words>", "prompt": "<45 words max>"}},
  {{"scene_number": 2, "title": "סצינה 2 — <2-3 Hebrew words>", "prompt": "<45 words max>"}},
  {{"scene_number": 3, "title": "סצינה 3 — <2-3 Hebrew words>", "prompt": "<45 words max>"}}
]"""

# ---------------------------------------------------------------------------
# Media panel — UI strings (Hebrew, matching the app's language convention)
# ---------------------------------------------------------------------------

UI_MEDIA_PANEL_HEADER      = "🎬 ייצור מדיה שיווקית"
UI_MEDIA_PANEL_CAPTION     = "צור תמונות או קליפים מקצועיים המבוססים ישירות על תוצאת הדיון"
UI_MEDIA_IMAGE_EXPANDER    = "🖼️ צור תמונה שיווקית (gpt-image-2)"
UI_MEDIA_VIDEO_EXPANDER    = "🎥 צור קליפ שיווקי (AI Video)"
UI_MEDIA_CONTEXT_LABEL     = "הקשר נוסף לקמפיין (אופציונלי)"
UI_MEDIA_CONTEXT_HELP      = "לדוגמא: קהל יעד, גיל, ערוץ פרסום, טון רצוי, מדינה..."
UI_MEDIA_SIZE_LABEL        = "פורמט / יחס גובה-רוחב"
UI_MEDIA_SIZE_OPTIONS = {
    "Landscape 16:9 — בנר / YouTube / Apple Search Ads": "landscape",
    "Portrait  9:16 — Stories / Reels / מודעות מובייל":  "portrait",
    "Square    1:1  — אינסטגרם / טוויטר":               "square",
}
UI_MEDIA_VIDEO_MODEL_LABEL = "מודל וידאו"
UI_MEDIA_GENERATE_IMG_BTN  = "✨ צור תמונה"
UI_MEDIA_GENERATE_VID_BTN  = "🎬 צור קליפ"
UI_MEDIA_STEP_PROMPT       = "שלב 1/2 — בונה פרומפט יצירתי עם Claude..."
UI_MEDIA_STEP_IMAGE        = "שלב 2/2 — מייצר תמונה עם gpt-image-2 (כ-15 שניות)..."
UI_MEDIA_STEP_VIDEO        = "שלב 2/2 — מייצר קליפ (עד 3 דקות — נא להמתין)..."
UI_MEDIA_DL_IMG_BTN        = "⬇️ הורד תמונה (PNG)"
UI_MEDIA_DL_VID_BTN        = "⬇️ הורד קליפ (MP4)"
UI_MEDIA_PROMPT_USED_HDR   = "הפרומפט שנוצר על-ידי Claude"
UI_MEDIA_ERROR_PREFIX      = "שגיאה בייצור מדיה"
UI_MEDIA_NO_REPLICATE_KEY  = (
    "**⚠️ REPLICATE_API_TOKEN חסר**  \n"
    "הירשם בחינם בכתובת **replicate.com**, קבל API token "
    "והוסף שורה `REPLICATE_API_TOKEN=your_token` לקובץ `.env`."
)
UI_MEDIA_REGEN_BTN         = "🔄 צור שוב"
UI_MEDIA_VIDEO_MODE_LABEL  = "מצב ייצור"
UI_MEDIA_VIDEO_MODE_SINGLE = "קליפ בודד (5-6 שניות)"
UI_MEDIA_VIDEO_MODE_STORY  = "סטוריבורד — 3 קליפים (15-18 שניות יחד)"
UI_MEDIA_STORYBOARD_HDR    = "🎬 הסרטון השיווקי — 3 קליפים"
UI_MEDIA_STORYBOARD_NOTE   = "הצג את 3 הקליפים ברצף לקהל היעד — יחד הם מספרים סיפור מלא"
UI_MEDIA_SCENE_GENERATING  = "מייצר קליפ {n}/3..."
UI_MEDIA_STEP_STORYBOARD   = "שלב 1/2 — בונה תסריט 3 סצינות עם Claude..."

# Image storyboard
UI_MEDIA_IMAGE_MODE_LABEL   = "מצב ייצור תמונה"
UI_MEDIA_IMAGE_MODE_SINGLE  = "תמונה בודדת"
UI_MEDIA_IMAGE_MODE_STORY   = "סטוריבורד — 3-4 תמונות"
UI_MEDIA_IMG_STORY_HDR      = "📸 סטוריבורד תמונות שיווקיות"
UI_MEDIA_IMG_STORY_NOTE     = "הצג את התמונות ברצף — יחד הן מספרות סיפור שלם"
UI_MEDIA_IMG_STORY_COUNT    = "מספר תמונות"
UI_MEDIA_STEP_IMG_STORY     = "שלב 1/2 — בונה תסריט תמונות עם Claude..."
UI_MEDIA_STEP_IMG_GEN       = "שלב 2/2 — מייצר תמונה {n}/{total} עם gpt-image-2..."
UI_MEDIA_IMG_REGEN_STORY    = "🔄 צור סטוריבורד מחדש"
UI_MEDIA_DL_IMG_N           = "⬇️ הורד תמונה {n}"

# App branding upload (optional, shared between image and video)
UI_MEDIA_BRANDING_LABEL     = "לוגו / צילום מסך של האפליקציה (אופציונלי)"
UI_MEDIA_BRANDING_HELP      = (
    "העלה עד 3 תמונות — לוגו, צילומי מסך, או חומרי מיתוג. "
    "Claude ינתח את הצבעים, הסגנון והטון ויגדיר אותם בכל הפרומפטים."
)
UI_MEDIA_BRANDING_ANALYSING = "מנתח מיתוג האפליקציה עם Claude Vision..."
UI_MEDIA_BRANDING_DONE      = "מיתוג אופיין — סגנון ייושם על כל הפרומפטים"

# Reference images (product shots / app screenshots used as visual input)
UI_MEDIA_REF_IMG_LABEL      = "תמונות רפרנס לייצור (אופציונלי)"
UI_MEDIA_REF_IMG_HELP       = (
    "העלה תמונות של המוצר, האפליקציה, דמויות, או כל חומר ויזואלי רלוונטי. "
    "Claude ינתח אותן ויזריק את התיאור הויזואלי לפרומפטים — מה שיגרום לתוצאה "
    "להיות קרובה יותר לחומרי המוצר שלך."
)
UI_MEDIA_REF_ANALYSING      = "מנתח תמונות רפרנס עם Claude Vision..."
UI_MEDIA_REF_DONE           = "תמונות הרפרנס אופיינו — יישמשו להנחיה ויזואלית"

# Visual style selector
UI_MEDIA_STYLE_LABEL        = "סגנון ויזואלי"
UI_MEDIA_STYLE_REALISTIC    = "📷 ריאליסטי — צילום קולנועי"
UI_MEDIA_STYLE_ANIMATION    = "🎨 אנימציה — מצויר / מאויר"

# ---------------------------------------------------------------------------
# Cultural Campaign Generator — market profiles and UI strings
# ---------------------------------------------------------------------------

# Each profile gives Claude the cultural intelligence it needs to generate
# visually and tonally appropriate media for that specific market.
CULTURAL_PROFILES = {
    "🇩🇪 גרמניה": {
        "key":      "germany",
        "language": "German",
        "flag":     "🇩🇪",
        "brief": (
            "German market: tone is direct, rational, and fact-based — avoid hype or "
            "exaggerated claims. Visuals should feel clean, precise, and trustworthy. "
            "Show real, relatable everyday scenarios with natural lighting. People should "
            "appear confident and self-sufficient, not dramatic. Privacy and data-security "
            "themes resonate strongly. Color palette: cool whites, greys, muted blues. "
            "Avoid flashy or oversaturated imagery. Architecture and environments should "
            "feel Northern European (urban, modern, functional). Women are shown as "
            "independent and professional."
        ),
    },
    "🇧🇷 ברזיל": {
        "key":      "brazil",
        "language": "Portuguese (Brazilian)",
        "flag":     "🇧🇷",
        "brief": (
            "Brazilian market: tone is warm, energetic, and community-oriented. Visuals "
            "should feel vibrant and joyful with rich, saturated colors — yellows, greens, "
            "warm oranges. Show diverse, mixed-ethnicity characters in social or family "
            "settings. Outdoor scenes and natural light work well. Emotional storytelling "
            "and human connection are central — smiles, touch, togetherness. Settings: "
            "modern urban Brazil, beaches, colourful neighbourhoods. Women are shown as "
            "confident, expressive, and social. Avoid anything that feels cold or overly "
            "clinical."
        ),
    },
    "🇸🇦 עולם הערבי": {
        "key":      "arabic",
        "language": "Arabic",
        "flag":     "🇸🇦",
        "brief": (
            "Arab / Middle Eastern market: visuals must be culturally sensitive and modest. "
            "Women should be shown with appropriate clothing — hijab or modest dress is "
            "expected in many markets (Gulf, Egypt, Levant). Family values and privacy are "
            "paramount. Tone is warm and respectful, never confrontational. Settings: "
            "modern Middle Eastern interiors, family homes, tasteful urban environments. "
            "Color palette: warm golds, deep teals, cream whites — rich but elegant. "
            "Avoid any imagery of bare skin, revealing clothing, or alcohol. Emphasise "
            "safety, family protection, and community trust."
        ),
    },
    "🇮🇱 ישראל": {
        "key":      "israel",
        "language": "Hebrew",
        "flag":     "🇮🇱",
        "brief": (
            "Israeli market: tone is direct, informal, and no-nonsense — Israelis distrust "
            "corporate polish. Visuals should feel authentic and relatable, showing real "
            "people in everyday Israeli settings (Tel Aviv streets, outdoor cafes, "
            "Mediterranean light). Diverse representation (Jewish and Arab Israelis). "
            "Women are shown as independent and assertive. Humour and self-awareness work "
            "well. Color palette: Mediterranean — warm whites, terracotta, deep blue. "
            "Tech-savvy audience — can show app UI elements subtly."
        ),
    },
    "🇺🇸 ארצות הברית": {
        "key":      "usa",
        "language": "English (American)",
        "flag":     "🇺🇸",
        "brief": (
            "US market: tone is empowering, aspirational, and inclusive. Show diverse "
            "representation across ethnicities, body types, and backgrounds. Visuals are "
            "polished but authentic — avoid stock-photo feel. Settings: modern American "
            "urban and suburban environments. Women are shown as strong, independent, and "
            "in control. Color palette: clean, bright, high-contrast. Emotional storytelling "
            "with a clear hero moment. Safety and personal empowerment themes resonate "
            "strongly."
        ),
    },
    "🇯🇵 יפן": {
        "key":      "japan",
        "language": "Japanese",
        "flag":     "🇯🇵",
        "brief": (
            "Japanese market: tone is subtle, precise, and aesthetically refined. Visuals "
            "should feel clean and minimalist — lots of negative space, soft natural light, "
            "delicate color palettes (blush pinks, soft greens, off-whites). Avoid direct "
            "confrontation or dramatic emotion; show calm competence and quiet confidence. "
            "Settings: modern Japanese interiors, zen gardens, clean urban spaces. "
            "Attention to small details matters — precision and quality signal trust. "
            "Women are shown as capable and graceful."
        ),
    },
    "🇮🇳 הודו": {
        "key":      "india",
        "language": "Hindi / English",
        "flag":     "🇮🇳",
        "brief": (
            "Indian market: tone is warm, family-oriented, and aspirational. Show diverse "
            "ethnic representation — India is highly regional. Vibrant colors work well "
            "(saffron, deep red, turquoise). Settings: modern Indian urban environments, "
            "family homes, mix of traditional and contemporary elements. Women are shown "
            "as empowered within family and social contexts. Safety and protection themes "
            "resonate strongly. Modest clothing is appropriate for broader market appeal. "
            "Avoid anything that could be seen as culturally insensitive to religious or "
            "regional groups."
        ),
    },
    "🇫🇷 צרפת": {
        "key":      "france",
        "language": "French",
        "flag":     "🇫🇷",
        "brief": (
            "French market: tone is elegant, understated, and intellectually confident — "
            "avoid over-selling or American-style hype. Visuals should feel chic and "
            "effortlessly stylish. Settings: Parisian streets, modern French interiors, "
            "cafes, natural light. Women are shown as sophisticated, independent, and "
            "self-assured. Color palette: muted elegance — navy, cream, burgundy, soft "
            "grey. Privacy and personal autonomy resonate strongly. Quality over quantity "
            "messaging works well."
        ),
    },
}

PROMPT_MEDIA_CULTURAL_BRIEF = """\
You are a senior creative director with deep cross-cultural advertising expertise \
specialising in ultra-short mobile ads (5-6 seconds for video, single frame for image).

Your task: generate a highly optimised {media_type} prompt for the {market} market, \
adapted to local visual norms and grounded in the product's actual use case.

--- Original Product / Campaign Brief ---
{question}

--- AI Council Key Insights ---
{final_answer}

--- Additional Campaign Context ---
{campaign_context}

--- Cultural Profile for {market} ---
{cultural_brief}

## Output Rules — follow EXACTLY

1. Output ONLY the {media_type} prompt — no preamble, no explanation, English only.
2. Apply the cultural profile to: characters' appearance and clothing, setting, \
   colour palette, lighting, and emotional tone.
3. The output must feel produced *for* that culture — not just translated.

FOR IMAGES:
- [Shot type] + [Character in culturally-appropriate setting] + [The product moment] + \
  [Visual style + lighting matching the culture's aesthetic]. 2-3 sentences, max 160 words.
- No text, logos, or UI elements in the scene.

FOR VIDEOS (CRITICAL — 5-6 seconds only):
- FORBIDDEN: generic lifestyle footage, woman walking, entering home, abstract scenes.
- REQUIRED: ONE specific character → ONE exact product action on screen → ONE visible result.
- Structure: [WHO + WHERE culturally specific] → [EXACT product moment] → \
  [IMMEDIATE REACTION] + [camera] + [lighting/grade matching {market} aesthetic].
- 3-4 sentences, max 180 words.

Write the {media_type} prompt now — one paragraph, no headers:"""

PROMPT_MEDIA_BRANDING_ANALYSIS = """\
You are a brand analyst. Examine the uploaded app screenshot(s) and/or logo carefully.

Extract and return a compact brand brief in this exact format (plain text, no markdown):

COLOR PALETTE: list the 3-5 dominant hex colors or descriptive names
VISUAL STYLE: one sentence describing the overall aesthetic (e.g. "clean minimal with deep navy and white")
MOOD: one or two adjectives (e.g. "trustworthy, calm")
KEY VISUAL ELEMENTS: any recurring shapes, icons, or motifs visible in the screenshots
TYPOGRAPHY FEEL: describe font weight and style visible (e.g. "bold sans-serif, modern")

Keep the entire brief under 80 words. This brief will be injected into image and video \
generation prompts to ensure visual consistency with the app's brand."""

PROMPT_MEDIA_REF_ANALYSIS = """\
You are a visual analyst. Examine the uploaded reference image(s) carefully.

Describe what you see in concise, visual language suitable as context for an AI \
image or video generation prompt.

Focus on:
- Subject / character: appearance, clothing, expression, age range
- Product / object: shape, colour, design details worth preserving
- Environment / setting: location type, atmosphere, key props
- Any distinctive visual details that should carry through to generated outputs

Return a single plain-text paragraph, max 80 words, in English. \
No markdown, no headers, no bullet points."""

PROMPT_MEDIA_IMAGE_STORYBOARD = """\
You are a senior creative director creating a {n_images}-image photo storyboard \
for a mobile app marketing campaign.

Each image is a standalone cinematic photograph. Together they tell the full story.

--- Product ---
{question}

--- Campaign Context ---
{campaign_context}

{branding_section}
{ref_section}
Story arc ({n_images} images):
  Image 1 — SITUATION: the character and their problem, before the product (set the scene)
  Image 2 — ACTION: the exact moment of using the product
  Image 3 — RESPONSE: someone receives the alert / the product working
  {image_4_instruction}

RULES FOR EACH IMAGE PROMPT:
1. Photorealistic cinematic photography — no illustrations, no graphics
2. Consistent visual identity: same character appearance, same lighting mood across all images
3. Each prompt: [shot size] + [character] + [environment] + [action/moment] + [lighting] + [mood]
4. Max 100 words per prompt. No text, logos, UI in the scene.
5. The images must make sense as a sequence even without captions

Output ONLY a JSON array — no explanation, no markdown:
[
  {{"image_number": 1, "title": "<2-3 Hebrew words>", "prompt": "<English prompt>"}},
  {{"image_number": 2, "title": "<2-3 Hebrew words>", "prompt": "<English prompt>"}},
  {{"image_number": 3, "title": "<2-3 Hebrew words>", "prompt": "<English prompt>"}}{image_4_json}
]"""

# Cultural panel UI strings
UI_CULTURAL_EXPANDER        = "🌍 קמפיין מותאם תרבותית — מספר שווקים במקביל"
UI_CULTURAL_CAPTION         = "בחר שווקי יעד וצור תמונה או קליפ מותאם תרבותית לכל אחד"
UI_CULTURAL_MARKET_SELECT   = "בחר שווקי יעד (ניתן לבחור מספר)"
UI_CULTURAL_MEDIA_TYPE      = "סוג מדיה"
UI_CULTURAL_MEDIA_IMAGE     = "תמונה (gpt-image-2)"
UI_CULTURAL_MEDIA_VIDEO     = "קליפ (Replicate)"
UI_CULTURAL_SIZE_LABEL      = "פורמט"
UI_CULTURAL_CTX_LABEL       = "הקשר קמפיין (אופציונלי — חל על כל השווקים)"
UI_CULTURAL_CTX_PLACEHOLDER = "לדוגמא: אפליקציית מעקב מחזור ובריאות האישה, גיל 20-40..."
UI_CULTURAL_GENERATE_BTN    = "✨ צור לכל השווקים הנבחרים"
UI_CULTURAL_GENERATING      = "מייצר מדיה מותאמת תרבותית..."
UI_CULTURAL_RESULT_HDR      = "תוצאות לפי שוק"
UI_CULTURAL_DL_IMG          = "⬇️ הורד PNG"
UI_CULTURAL_DL_VID          = "⬇️ הורד MP4"
UI_CULTURAL_PROMPT_HDR      = "פרומפט"
UI_CULTURAL_MIN_MARKETS     = "בחר לפחות שוק אחד"
UI_CULTURAL_NO_VIDEO_KEY    = "ייצור קליפים מושבת — הוסף REPLICATE_API_TOKEN ל-.env"
