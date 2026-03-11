"""
report_generator.py — PDF Export Module
=========================================
Converts the debate results dict into a branded, multi-page PDF report.

Font strategy (in priority order)
----------------------------------
1. fonts/Assistant-Regular.ttf + fonts/Assistant-Bold.ttf   (preferred)
   — the Hebrew-native Google Font used throughout the UI.
2. Auto-download from Google Fonts GitHub if the files are missing.
3. Any other Unicode TTF already present in the fonts/ directory.
4. System font fallbacks (macOS Arial Unicode, Linux DejaVu, …).
5. fpdf2 built-in Helvetica — last resort; Hebrew shows as boxes but
   the rest of the document renders correctly.

Hebrew / RTL
------------
Every text string is passed through bidi.algorithm.get_display() before
writing.  Cells containing Hebrew are also right-aligned automatically.

Markdown rendering
------------------
Headers, bullets, numbered lists, code blocks, and horizontal rules are
all handled.  Inline syntax (bold, italic, code, links) is stripped so
the text reads cleanly in print form.

Public API
----------
    generate_pdf(results_dict: dict) -> bytes
        Returns the finished PDF as a byte string for st.download_button().
"""

from __future__ import annotations

import io
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF
from bidi.algorithm import get_display

from glossary import (
    MODELS,
    PDF_GENERATED_BY,
    PDF_REPORT_TITLE,
    PDF_TECH_BREAKDOWN_BG,
    PDF_TECH_BREAKDOWN_BORDER,
    PDF_TECH_BREAKDOWN_LABEL,
)


# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------

FONTS_DIR = Path(__file__).parent / "fonts"

# Canonical filenames the module expects inside fonts/
_REGULAR_NAME = "Assistant-Regular.ttf"
_BOLD_NAME    = "Assistant-Bold.ttf"

# Download URLs (Google Fonts GitHub — stable raw links)
_FONT_URLS: dict[str, str] = {
    _REGULAR_NAME: (
        "https://github.com/google/fonts/raw/main/"
        "ofl/assistant/static/Assistant-Regular.ttf"
    ),
    _BOLD_NAME: (
        "https://github.com/google/fonts/raw/main/"
        "ofl/assistant/static/Assistant-Bold.ttf"
    ),
}

# Fallback search paths if Assistant is unavailable everywhere
_FALLBACK_PATHS: list[Path] = [
    Path("/Library/Fonts/Arial Unicode MS.ttf"),                      # macOS
    Path("/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf"),  # macOS 13+
    Path("/Library/Fonts/Arial.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),                               # Windows
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),      # Linux
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),          # Linux
]


def _ensure_font(filename: str) -> Optional[Path]:
    """
    Return the path to a font file, downloading it if necessary.
    Returns None if the file cannot be obtained.
    """
    dest = FONTS_DIR / filename
    if dest.exists():
        return dest

    url = _FONT_URLS.get(filename)
    if url is None:
        return None

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception:
        return None


def _load_fonts(pdf: FPDF) -> tuple[str, str]:
    """
    Register Regular and Bold fonts with the FPDF instance.

    Returns
    -------
    tuple[str, str]
        (regular_font_name, bold_font_name) to pass to set_font().
        Both are "Helvetica" if no Unicode font could be loaded.
    """
    # Try Assistant first, then fallbacks for the regular weight
    regular_path = _ensure_font(_REGULAR_NAME)
    bold_path    = _ensure_font(_BOLD_NAME)

    if regular_path is None:
        for fp in _FALLBACK_PATHS:
            if fp.exists():
                regular_path = fp
                break

    if regular_path is None:
        return "Helvetica", "Helvetica"

    # Use Regular as Bold fallback when Bold file is missing
    if bold_path is None or not bold_path.exists():
        bold_path = regular_path

    try:
        pdf.add_font("Assistant",  style="",  fname=str(regular_path))
        pdf.add_font("Assistant",  style="B", fname=str(bold_path))
        pdf.add_font("Assistant",  style="I", fname=str(regular_path))  # italic = regular
        return "Assistant", "Assistant"
    except Exception:
        return "Helvetica", "Helvetica"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

# Splits the Mediator's final answer to isolate the Technical Breakdown section.
# Matches "## Technical Breakdown" and the optional "## 📐 Technical Breakdown" variant.
_TD_SPLIT = re.compile(
    r'(?m)^##\s*(?:📐\s*)?Technical Breakdown\b',
    re.IGNORECASE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def _has_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in text)


def _bidi(text: str) -> str:
    """Apply Unicode Bidi algorithm for correct RTL visual ordering."""
    return get_display(text) if _has_hebrew(text) else text


def _strip_inline_md(text: str) -> str:
    """Strip inline Markdown, keeping only the readable content."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # **bold**
    text = re.sub(r"\*(.+?)\*",     r"\1", text)       # *italic*
    text = re.sub(r"_(.+?)_",       r"\1", text)       # _italic_
    text = re.sub(r"`(.+?)`",       r"\1", text)       # `code`
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)    # [link](url)
    return text


def _clean(text: str) -> str:
    """Full pipeline: emoji → inline MD → bidi."""
    return _bidi(_strip_inline_md(_strip_emoji(text)))


# ---------------------------------------------------------------------------
# PDF class with branded header / footer on every page
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    """FPDF subclass that adds the AI-Playground header and footer."""

    def __init__(self, fn_regular: str, fn_bold: str, timestamp: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._fn  = fn_regular   # regular-weight font name
        self._fnb = fn_bold      # bold-weight font name
        self._ts  = timestamp
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(left=20, top=22, right=20)

    # every page header: logo text + thin separator line
    def header(self) -> None:
        self.set_font(self._fnb, "B", 10)
        self.set_text_color(59, 130, 246)     # brand blue
        self.cell(0, 7, PDF_REPORT_TITLE, align="L",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_font(self._fn, "", 8)
        self.set_text_color(148, 163, 184)    # slate-400
        self.cell(0, 5, self._ts, align="L",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)    # slate-200
        self.line(20, self.get_y() + 1, 190, self.get_y() + 1)
        self.ln(5)
        self.set_text_color(0, 0, 0)

    # every page footer: generated-by + page number
    def footer(self) -> None:
        self.set_y(-14)
        self.set_font(self._fn, "I", 8)
        self.set_text_color(148, 163, 184)
        generated = _bidi(PDF_GENERATED_BY)
        self.cell(0, 8, f"{generated}   |   Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


# ---------------------------------------------------------------------------
# Markdown → PDF renderer
# ---------------------------------------------------------------------------

def _section_title(pdf: _ReportPDF, title: str) -> None:
    """Render a numbered section heading with a coloured background bar."""
    pdf.ln(5)
    pdf.set_fill_color(239, 246, 255)     # very light blue
    pdf.set_font(pdf._fnb, "B", 13)
    pdf.set_text_color(30, 64, 175)       # blue-800
    pdf.cell(0, 9, title, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(pdf._fn, "", 11)
    pdf.set_text_color(0, 0, 0)


def _write_md_line(pdf: _ReportPDF, line: str) -> None:
    """Write one Markdown-formatted line to the PDF."""
    fn  = pdf._fn
    fnb = pdf._fnb
    rtl = _has_hebrew(line)
    align = "R" if rtl else "L"

    # H1
    if line.startswith("# "):
        pdf.set_font(fnb, "B", 16)
        pdf.multi_cell(0, 9, _clean(line[2:]), align=align,
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(fn, "", 11)
        return

    # H2
    if line.startswith("## "):
        pdf.set_font(fnb, "B", 13)
        pdf.multi_cell(0, 7, _clean(line[3:]), align=align,
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(fn, "", 11)
        return

    # H3 / H4+
    if re.match(r"^#{3,} ", line):
        depth = len(re.match(r"^(#+)", line).group(1))
        pdf.set_font(fnb, "B", 11)
        pdf.multi_cell(0, 6, _clean(line[depth + 1:]), align=align,
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(fn, "", 11)
        return

    # Horizontal rule
    if re.fullmatch(r"-{3,}", line.strip()):
        pdf.ln(2)
        pdf.set_draw_color(203, 213, 225)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + pdf.epw, pdf.get_y())
        pdf.ln(3)
        return

    # Unordered bullet
    if re.match(r"^[-*] ", line):
        content = _clean(line[2:])
        bullet  = " \u2022" if rtl else "\u2022 "
        pdf.multi_cell(0, 6, bullet + content, align=align,
                       new_x="LMARGIN", new_y="NEXT")
        return

    # Numbered list
    m = re.match(r"^(\d+)\. (.+)", line)
    if m:
        pdf.multi_cell(0, 6, f"{m.group(1)}. {_clean(m.group(2))}",
                       align=align, new_x="LMARGIN", new_y="NEXT")
        return

    # Empty line → spacing
    if not line.strip():
        pdf.ln(2)
        return

    # Regular paragraph (multi_cell wraps automatically → handles long text)
    pdf.multi_cell(0, 6, _clean(line), align=align,
                   new_x="LMARGIN", new_y="NEXT")


def _render_markdown(pdf: _ReportPDF, text: str) -> None:
    """
    Walk the Markdown text line-by-line and write it into the PDF.
    Long answers automatically overflow to new pages via set_auto_page_break.
    Code blocks are rendered in a shaded monospace-style box.
    """
    lines    = text.split("\n")
    in_code  = False
    code_buf: list[str] = []

    def _flush_code() -> None:
        if not code_buf:
            return
        pdf.set_fill_color(245, 245, 245)
        pdf.set_font(pdf._fn, "", 9)
        for cl in code_buf:
            pdf.multi_cell(0, 5, _strip_emoji(cl), fill=True,
                           new_x="LMARGIN", new_y="NEXT")
        pdf.set_fill_color(255, 255, 255)
        pdf.set_font(pdf._fn, "", 11)
        pdf.ln(2)
        code_buf.clear()

    for line in lines:
        if line.startswith("```"):
            if in_code:
                _flush_code()
            in_code = not in_code
            continue

        if in_code:
            code_buf.append(line)
        else:
            _write_md_line(pdf, line)

    if in_code:
        _flush_code()   # close any unclosed fence


def _render_tech_breakdown_box(pdf: _ReportPDF, text: str) -> None:
    """
    Render the Technical Breakdown section in a visually distinct box that
    mirrors the .tech-breakdown CSS container from the web UI:

      - Light slate-100 background for all content rows
      - Dashed-style top / bottom border lines (slate-400)
      - Blue-tinted highlight for blockquote formula lines (> ...)
      - RTL / Hebrew fully supported throughout

    Parameters
    ----------
    text : str
        The raw markdown text starting with (or just after) the
        "## Technical Breakdown" header line.
    """
    pdf.ln(4)
    r_bg, g_bg, b_bg = PDF_TECH_BREAKDOWN_BG
    r_bd, g_bd, b_bd = PDF_TECH_BREAKDOWN_BORDER

    # ── Header bar (mirrors .tech-breakdown-header) ──────────────────────
    pdf.set_fill_color(r_bg, g_bg, b_bg)
    pdf.set_draw_color(r_bd, g_bd, b_bd)
    pdf.set_font(pdf._fnb, "B", 8)
    pdf.set_text_color(100, 116, 139)           # slate-500
    pdf.cell(
        0, 7,
        _clean(PDF_TECH_BREAKDOWN_LABEL.upper()),
        fill=True, new_x="LMARGIN", new_y="NEXT",
    )
    # Separator line under header
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(2)

    # ── Content lines ────────────────────────────────────────────────────
    pdf.set_fill_color(r_bg, g_bg, b_bg)
    pdf.set_font(pdf._fn, "", 10)
    pdf.set_text_color(51, 65, 85)              # slate-700

    in_code: bool       = False
    code_buf: list      = []

    def _flush_code_box() -> None:
        if not code_buf:
            return
        pdf.set_font(pdf._fn, "", 9)
        pdf.set_fill_color(235, 241, 248)
        for cl in code_buf:
            pdf.multi_cell(0, 5, _strip_emoji(cl), fill=True,
                           new_x="LMARGIN", new_y="NEXT")
        code_buf.clear()
        pdf.set_fill_color(r_bg, g_bg, b_bg)
        pdf.set_font(pdf._fn, "", 10)

    for line in text.split("\n"):
        # Drop the ## Technical Breakdown header itself
        if re.match(r'^##\s*(?:📐\s*)?Technical Breakdown\b', line, re.IGNORECASE):
            continue

        # Code fences
        if line.startswith("```"):
            if in_code:
                _flush_code_box()
            in_code = not in_code
            continue
        if in_code:
            code_buf.append(line)
            continue

        # Empty line
        if not line.strip():
            pdf.ln(2)
            continue

        rtl   = _has_hebrew(line)
        align = "R" if rtl else "L"

        # Blockquote — formula highlight (mirrors .tech-formula blue tint)
        if line.startswith(">"):
            formula = _clean(line.lstrip("> ").strip())
            pdf.set_fill_color(239, 246, 255)       # blue-50
            pdf.set_font(pdf._fnb, "B", 10)
            pdf.set_text_color(30, 64, 175)         # blue-800
            fa = "R" if _has_hebrew(formula) else "L"
            pdf.multi_cell(0, 7, formula, fill=True, align=fa,
                           new_x="LMARGIN", new_y="NEXT")
            # Restore box styling
            pdf.set_fill_color(r_bg, g_bg, b_bg)
            pdf.set_font(pdf._fn, "", 10)
            pdf.set_text_color(51, 65, 85)
            continue

        # H2 / H3 sub-headers inside the breakdown
        if re.match(r'^#{2,} ', line):
            depth = len(re.match(r'^(#+)', line).group(1))
            pdf.set_font(pdf._fnb, "B", 10)
            pdf.multi_cell(0, 6, _clean(line[depth + 1:]), fill=True,
                           align=align, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(pdf._fn, "", 10)
            continue

        # Unordered bullet
        if re.match(r'^[-*] ', line):
            bullet  = " \u2022" if rtl else "\u2022 "
            content = _clean(line[2:])
            pdf.multi_cell(0, 6, bullet + content, fill=True,
                           align=align, new_x="LMARGIN", new_y="NEXT")
            continue

        # Numbered list
        m = re.match(r'^(\d+)\. (.+)', line)
        if m:
            pdf.multi_cell(0, 6, f"{m.group(1)}. {_clean(m.group(2))}", fill=True,
                           align=align, new_x="LMARGIN", new_y="NEXT")
            continue

        # Regular paragraph
        pdf.multi_cell(0, 6, _clean(line), fill=True,
                       align=align, new_x="LMARGIN", new_y="NEXT")

    if in_code:
        _flush_code_box()

    pdf.ln(2)

    # ── Bottom border line ───────────────────────────────────────────────
    pdf.set_draw_color(r_bd, g_bd, b_bd)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + pdf.epw, pdf.get_y())
    pdf.ln(5)

    # ── Reset to normal page styling ─────────────────────────────────────
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(pdf._fn, "", 11)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(results_dict: dict) -> bytes:
    """
    Build a formatted PDF research report and return it as bytes.

    Parameters
    ----------
    results_dict : dict
        The dict returned by logic_engine.run_council_debate().
        Required keys: question, final_answer, citations, fallback_used.

    Returns
    -------
    bytes
        Complete PDF file, ready for st.download_button(data=...).
    """
    question           = results_dict.get("question", "")
    final_answer       = results_dict.get("final_answer", "")
    citations          = results_dict.get("citations", [])
    fallback_used      = results_dict.get("fallback_used", False)
    reliability_scores = results_dict.get("reliability_scores", {})
    # Task 2: include day name so the stamp reads e.g. "Wednesday, 11 March 2026 · 12:50"
    timestamp = datetime.now().strftime("%A, %d %B %Y  \u00b7  %H:%M")

    # ── Initialise PDF ───────────────────────────────────────────────────────
    # Pass placeholder font names; _load_fonts() will update them.
    pdf = _ReportPDF(fn_regular="Helvetica", fn_bold="Helvetica",
                     timestamp=timestamp)
    fn_regular, fn_bold = _load_fonts(pdf)
    pdf._fn  = fn_regular
    pdf._fnb = fn_bold

    pdf.add_page()
    pdf.set_font(fn_regular, "", 11)

    # ── Cover block ──────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font(fn_bold, "B", 22)
    pdf.set_text_color(15, 23, 42)      # slate-950
    pdf.multi_cell(0, 13, PDF_REPORT_TITLE, align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(fn_regular, "", 10)
    pdf.set_text_color(100, 116, 139)   # slate-500
    pdf.multi_cell(0, 6, timestamp, align="C",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Fallback notice (amber banner)
    if fallback_used:
        pdf.set_fill_color(255, 251, 235)   # amber-50
        pdf.set_font(fn_regular, "I", 9)
        pdf.set_text_color(146, 64, 14)     # amber-800
        pdf.multi_cell(
            0, 6,
            "Note: The primary synthesis model failed. "
            "Gemini 3 Pro was used as fallback.",
            fill=True, new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Section 1: Question ──────────────────────────────────────────────────
    _section_title(pdf, "1.  Your Question")
    pdf.set_font(fn_bold, "B", 11)
    for q_line in question.split("\n"):
        if not q_line.strip():
            pdf.ln(2)
            continue
        pdf.multi_cell(0, 7, _clean(q_line),
                       align="R" if _has_hebrew(q_line) else "L",
                       new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(fn_regular, "", 11)

    # ── Section 2: Final Synthesised Answer ─────────────────────────────────
    # Split the Mediator's output into narrative body + Technical Breakdown so
    # the breakdown can be rendered in its own styled box (mirrors the web UI).
    _section_title(pdf, "2.  Final Synthesised Answer")
    pdf.set_font(fn_regular, "", 11)
    _td = _TD_SPLIT.search(final_answer)
    if _td:
        _render_markdown(pdf, final_answer[:_td.start()].strip())
        _render_tech_breakdown_box(pdf, final_answer[_td.start():].strip())
    else:
        _render_markdown(pdf, final_answer)

    # ── Section 3: Reliability Scorecard ─────────────────────────────────────
    if reliability_scores:
        _section_title(pdf, "3.  Reliability Scorecard")
        pdf.set_font(fn_regular, "", 10)
        for model_key, score in reliability_scores.items():
            model_label = MODELS.get(model_key, {}).get("label", model_key)
            if score >= 80:
                tier, rr, gg, bb = "High Consistency", 16, 185, 129    # emerald
            elif score >= 50:
                tier, rr, gg, bb = "Partial Revision",  245, 158, 11   # amber
            else:
                tier, rr, gg, bb = "Major Revision",    239, 68,  68   # red
            math_tag = "  [Math Aligned]" if score == 100 else ""
            row_text = _clean(f"{model_label}:  {score}/100 — {tier}{math_tag}")
            pdf.set_fill_color(rr, gg, bb)
            pdf.set_font(fn_bold, "B", 10)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, row_text, fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # ── Section 4: Sources & Citations ───────────────────────────────────────
    _section_title(pdf, "4.  Sources & Citations")
    pdf.set_font(fn_regular, "", 11)

    if citations:
        for i, c in enumerate(citations, start=1):
            title = _clean(c.get("title", c["url"]))
            url   = c["url"]

            pdf.set_font(fn_bold, "B", 10)
            pdf.multi_cell(0, 6, f"[{i}]  {title}",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(fn_regular, "", 9)
            pdf.set_text_color(37, 99, 235)   # blue-600
            pdf.multi_cell(0, 5, f"    {url}",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
    else:
        pdf.set_font(fn_regular, "I", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(
            0, 6,
            "No real-time sources were retrieved. "
            "This report is based on model knowledge and logical synthesis.",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)

    # ── Serialise ────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
