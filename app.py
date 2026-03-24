"""
app.py — Streamlit UI for the Council of AI Models
====================================================
Run with:
    streamlit run app.py

All UI text, model IDs, and prompts are imported from glossary.py.
All API logic is handled by ai_factory.py via logic_engine.py.
This file is purely about layout, widgets, and wiring.

Session-state architecture
--------------------------
st.session_state.current_results  — the debate dict currently displayed
                                    (None when nothing has been run yet)
st.session_state.from_history     — True when the displayed result was
                                    loaded from history (not a live run)

This separation lets _display_results() work identically for both live
debates and history replays without consuming any API tokens.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Optional

load_dotenv()  # טוען את המפתחות מקובץ ה-.env

import streamlit as st
from PIL import Image

_LOGO_PATH = Path(__file__).parent / "logo.png"
_logo_img  = Image.open(_LOGO_PATH) if _LOGO_PATH.exists() else None

from glossary import (
    APP_ICON,
    APP_PAGE_TITLE,
    MISSION_CONTROL_CSS,
    RELIABILITY_GAUGE_HTML,
    APP_SUBTITLE,
    APP_TITLE,
    FALLBACK_NOTICE,
    IMAGE_PREVIEW_HEADER,
    MASTER_MODEL_KEY,
    MODELS,
    PDF_BTN_LABEL,
    REPORT_TEMPLATE_CSS,
    RTL_CSS,
    STAGE_DESCRIPTIONS,
    STAGE_LABELS,
    UI_ASK_LABEL,
    UI_ASK_PLACEHOLDER,
    UI_AUTH_EMAIL_LABEL,
    UI_AUTH_GATE_MSG,
    UI_AUTH_LOGGED_IN_AS,
    UI_AUTH_LOGIN_BTN,
    UI_AUTH_LOGIN_TAB,
    UI_AUTH_LOGOUT_BTN,
    UI_AUTH_NO_API_KEY_MSG,
    UI_AUTH_PASSWORD_LABEL,
    UI_AUTH_SIGNUP_BTN,
    UI_AUTH_SIGNUP_TAB,
    UI_AUTH_SUCCESS_LOGIN,
    UI_AUTH_SUCCESS_SIGNUP,
    UI_AUTH_TITLE,
    UPLOAD_IMAGE_LABEL,
    UI_CITATION_VISIT,
    UI_CITATIONS_NO_DATA,
    UI_ERROR_PREFIX,
    UI_EXPANDER_CRITIQUE,
    UI_EXPANDER_INITIAL,
    HEBREW_MONTHS,
    HEBREW_WEEKDAYS,
    UI_HELP_DATA_DATE_TPL,
    UI_HELP_COUNCIL,
    UI_HELP_EXPANDER_TITLE,
    UI_HELP_RELIABILITY,
    UI_HELP_TECH_APPENDIX,
    UI_HISTORY_BANNER,
    UI_HISTORY_CLEAR,
    UI_HISTORY_EMPTY,
    UI_HISTORY_GROUP_EARLIER,
    UI_HISTORY_GROUP_OLDER,
    UI_HISTORY_GROUP_TODAY,
    UI_HISTORY_GROUP_YESTERDAY,
    UI_HISTORY_PDF_BTN,
    UI_HISTORY_TITLE,
    UI_HISTORY_VERIFIED_BADGE,
    UI_MASTER_MODEL_NOTE,
    UI_NO_ANSWER,
    UI_SECTION_CITATIONS,
    UI_SECTION_FINAL,
    UI_SECTION_PROCESS,
    UI_SELECT_MODELS,
    UI_SPINNER_STAGE1,
    UI_SPINNER_STAGE2,
    UI_SPINNER_STAGE3,
    UI_CACHE_HIT_BANNER,
    UI_CACHE_RERUN_BTN,
    DAILY_QUERY_LIMIT,
    UI_DAILY_USAGE_TPL,
    UI_EMAIL_NOT_VERIFIED,
    UI_NEW_QUERY_BUTTON,
    UI_QUOTA_REACHED,
    UI_REFRESH_VERIFY_BTN,
    UI_RESEND_SUCCESS,
    UI_RESEND_VERIFICATION,
    UI_STILL_NOT_VERIFIED,
    UI_FOLLOWUP_LABEL,
    UI_FOLLOWUP_PLACEHOLDER,
    UI_FOLLOWUP_SUBMIT_BTN,
    UI_SUBMIT_BUTTON,
    UI_TECH_LOG_TITLE,
    UI_UNVERIFIED_WARNING,
    UI_WARNING_MIN_MODELS,
    UI_STAGE3B_LABEL,
    UI_STAGE3B_POINT_HEADER,
    UI_STAGE3B_ROUND1_LABEL,
    UI_STAGE3B_ROUND2_LABEL,
    UI_STAGE3B_EXPANDER_MODEL,
    UI_STAGE3B_NO_DEBATES,
    CODE_REVIEW_DEFAULT_QUESTION,
    UI_CODE_REVIEW_TOGGLE,
    UI_CODE_REVIEW_PATH_LABEL,
    UI_CODE_REVIEW_PATH_PLACEHOLDER,
    UI_CODE_REVIEW_SCAN_BTN,
    UI_CODE_REVIEW_SCANNING,
    UI_CODE_REVIEW_SCANNED,
    UI_CODE_REVIEW_WARNINGS,
    UI_CODE_REVIEW_EMPTY,
    UI_CODE_REVIEW_NOT_FOUND,
    UI_CODE_REVIEW_FILES_HEADER,
    UI_CODE_REVIEW_CLEAR_BTN,
    FILE_UPLOAD_ACCEPTED_TYPES,
    FILE_UPLOAD_DEFAULT_QUESTION,
    FILE_UPLOAD_MAX_SIZE_MB,
    UI_FILE_UPLOAD_LABEL,
    UI_FILE_UPLOAD_LOADED,
    UI_FILE_UPLOAD_WARNINGS,
    UI_FILE_UPLOAD_CLEAR_BTN,
    UI_FILE_UPLOAD_EMPTY,
    UI_FILE_UPLOAD_FILES_HDR,
)
from logic_engine import run_council_debate
from project_reader import scan_project
from file_reader import build_file_context
from report_generator import generate_pdf
from history_manager import (
    check_usage_limit,
    clear_history,
    firestore_status,
    get_usage_count,
    get_user_history,
    load_history,
    save_to_history,
)
from auth_manager import (
    get_email_verified,
    send_verification_email,
    sign_in,
    sign_up,
)


# ---------------------------------------------------------------------------
# Processing-UI assets  (spinner CSS + Status Dashboard)
# ---------------------------------------------------------------------------

# Rainbow rotating ring — pure CSS, zero Python thread needed.
_SPINNER_CSS = """
<style>
@keyframes council-spin {
    to { transform: rotate(360deg); }
}
@keyframes council-hue {
    0%   { border-top-color:#ff6b6b; border-right-color:#ffd93d;
           border-bottom-color:#6bcb77; border-left-color:#4d96ff; }
    33%  { border-top-color:#c77dff; border-right-color:#ff6b6b;
           border-bottom-color:#ffd93d; border-left-color:#6bcb77; }
    66%  { border-top-color:#4d96ff; border-right-color:#c77dff;
           border-bottom-color:#ff6b6b; border-left-color:#ffd93d; }
    100% { border-top-color:#ff6b6b; border-right-color:#ffd93d;
           border-bottom-color:#6bcb77; border-left-color:#4d96ff; }
}
.council-ring-wrap { display:flex; justify-content:center; padding:18px 0 6px; }
.council-ring {
    width:56px; height:56px; border-radius:50%;
    border:6px solid transparent;
    animation: council-spin 0.85s linear infinite,
               council-hue  3.4s  linear infinite;
}
</style>
"""

_SPINNER_HTML = (
    '<div class="council-ring-wrap">'
    '<div class="council-ring"></div>'
    '</div>'
)

# ── Status Dashboard — stage configs ──────────────────────────────────────
# One entry per DSAD stage (0–4).  Colors map to:
#   0 Search → Deep Blue   1 Thesis → Electric Purple   2 Audit → Sunset Orange
#   3 Dialectic → Emerald  4 Synthesis → Gold
_STAGE_CFG = [
    {"icon": "🔍", "label": "Live\nSearch",  "color": "#1e40af", "bg": "#dbeafe", "glow": "rgba(30,64,175,0.45)"},
    {"icon": "⚙️", "label": "Thesis",        "color": "#7c3aed", "bg": "#ede9fe", "glow": "rgba(124,58,237,0.45)"},
    {"icon": "🔬", "label": "Audit",         "color": "#ea580c", "bg": "#ffedd5", "glow": "rgba(234,88,12,0.45)"},
    {"icon": "💬", "label": "Dialectic",     "color": "#059669", "bg": "#d1fae5", "glow": "rgba(5,150,105,0.45)"},
    {"icon": "⚔️", "label": "Focused\nDebate", "color": "#dc2626", "bg": "#fee2e2", "glow": "rgba(220,38,38,0.45)"},
    {"icon": "🏛️", "label": "Synthesis",     "color": "#b45309", "bg": "#fef3c7", "glow": "rgba(180,83,9,0.45)"},
]


def render_status_dashboard(current_stage: int, completed_stages: set) -> str:
    """
    Return an HTML string representing the 5-stage Status Dashboard.

    Parameters
    ----------
    current_stage : int
        Stage index currently executing (0–4), or -1 for none.
    completed_stages : set[int]
        Set of stage indices that have finished successfully.
    """
    items: list = []
    for i, cfg in enumerate(_STAGE_CFG):
        if i in completed_stages:
            # Solid filled circle with glow — complete
            circle_style = (
                f"background:{cfg['color']};"
                f"border:3px solid {cfg['color']};"
                f"box-shadow:0 0 14px {cfg['glow']};"
                "color:#ffffff;"
            )
            icon     = "✓"
            lbl_style = "color:#475569;"
        elif i == current_stage:
            # Pulsing active circle
            circle_style = (
                f"background:{cfg['bg']};"
                f"border:3px solid {cfg['color']};"
                f"color:{cfg['color']};"
                "animation:sd-pulse 1.4s ease-in-out infinite;"
            )
            icon     = cfg["icon"]
            lbl_style = f"color:{cfg['color']};font-weight:700;"
        else:
            # Hollow pending circle
            circle_style = (
                "background:#f1f5f9;"
                "border:3px solid #cbd5e1;"
                "color:#94a3b8;"
            )
            icon     = cfg["icon"]
            lbl_style = "color:#94a3b8;"

        label_html = cfg["label"].replace("\n", "<br>")
        items.append(
            f'<div class="sd-item">'
            f'<div class="sd-circle" style="{circle_style}">{icon}</div>'
            f'<div class="sd-label" style="{lbl_style}">{label_html}</div>'
            f'</div>'
        )
        # Connector between circles (skip after last)
        if i < len(_STAGE_CFG) - 1:
            if i in completed_stages:
                next_color = _STAGE_CFG[i + 1]["color"]
                conn_style = (
                    f"background:linear-gradient(90deg,{cfg['color']},{next_color});"
                )
            else:
                conn_style = "background:#e2e8f0;"
            items.append(f'<div class="sd-connector" style="{conn_style}"></div>')

    return '<div class="sd-wrap">' + "".join(items) + "</div>"


def render_reliability_gauge(model_label: str, score: int) -> str:
    """Return inline HTML/SVG speedometer for one model's reliability score."""
    import math as _math
    import uuid as _uuid

    uid         = _uuid.uuid4().hex[:8]
    angle       = (score / 100 * 180) - 90
    math_angle  = _math.radians(180 - score * 1.8)
    score_x     = 100 + 80 * _math.cos(math_angle)
    score_y     = 100 - 80 * _math.sin(math_angle)

    if score >= 80:
        color = "#10b981"; label = "High Consistency"
    elif score >= 50:
        color = "#f59e0b"; label = "Partial Revision"
    else:
        color = "#ef4444"; label = "Major Revision"

    return RELIABILITY_GAUGE_HTML.format(
        uid=uid,
        score=score,
        score_x=f"{score_x:.1f}",
        score_y=f"{score_y:.1f}",
        angle=f"{angle:.1f}",
        color=color,
        label=label,
        model_label=model_label,
    )


# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_PAGE_TITLE,
    page_icon=_logo_img if _logo_img else APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help":        None,
        "Report a bug":    None,
        "About":           f"**{APP_TITLE}** — Dual-Sided Adversarial Discussion (DSAD) "
                           "with 4 AI models.  For research and educational use only.",
    },
)
st.markdown(RTL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state — initialise once per browser session
# ---------------------------------------------------------------------------
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "from_history" not in st.session_state:
    st.session_state.from_history = False
if "user" not in st.session_state:
    st.session_state.user = None   # None or {"uid": str, "email": str, "id_token": str}
if "_auth_mode" not in st.session_state:
    st.session_state._auth_mode = UI_AUTH_LOGIN_TAB
if "_query_counter" not in st.session_state:
    st.session_state._query_counter = 0   # incremented on "New Query" to reset widgets
if "_force_rerun" not in st.session_state:
    st.session_state._force_rerun = False  # True = skip cache check this run
if "_cached_usage" not in st.session_state:
    st.session_state._cached_usage = None  # (uid, count) — avoids Firestore call on every keystroke


# ---------------------------------------------------------------------------
# Sidebar — model selection, API info, session history
# ---------------------------------------------------------------------------
with st.sidebar:
    if _logo_img:
        col_logo, col_txt = st.columns([1, 2.5])
        with col_logo:
            st.image(_logo_img, width=64)
        with col_txt:
            st.markdown(f"### {APP_TITLE}")
            st.caption(APP_SUBTITLE)
    else:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.caption(APP_SUBTITLE)
    st.divider()

    # ── Authentication ────────────────────────────────────────────────────────
    _user = st.session_state.user

    if _user:
        # Logged-in state
        st.success(f"✅ {UI_AUTH_LOGGED_IN_AS}: **{_user['email']}**")
        if st.button(UI_AUTH_LOGOUT_BTN, use_container_width=True):
            st.session_state.user = None
            st.session_state.current_results = None
            st.session_state.from_history    = False
            st.rerun()
    else:
        # Login / Sign-up form
        st.subheader(UI_AUTH_TITLE)
        _auth_mode = st.radio(
            "Mode",
            [UI_AUTH_LOGIN_TAB, UI_AUTH_SIGNUP_TAB],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="_auth_mode_radio",
        )
        _email    = st.text_input(UI_AUTH_EMAIL_LABEL,    key="auth_email")
        _password = st.text_input(UI_AUTH_PASSWORD_LABEL, type="password", key="auth_password")
        _btn_label = UI_AUTH_LOGIN_BTN if _auth_mode == UI_AUTH_LOGIN_TAB else UI_AUTH_SIGNUP_BTN

        if st.button(_btn_label, type="primary", use_container_width=True, key="auth_submit"):
            if not _email.strip() or not _password.strip():
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Authenticating …"):
                    result = (
                        sign_in(_email.strip(), _password.strip())
                        if _auth_mode == UI_AUTH_LOGIN_TAB
                        else sign_up(_email.strip(), _password.strip())
                    )
                if isinstance(result, dict):
                    st.session_state.user = result
                    success_msg = (
                        UI_AUTH_SUCCESS_LOGIN
                        if _auth_mode == UI_AUTH_LOGIN_TAB
                        else UI_AUTH_SUCCESS_SIGNUP
                    )
                    st.success(success_msg)
                    st.rerun()
                else:
                    if "FIREBASE_WEB_API_KEY" in result:
                        st.warning(UI_AUTH_NO_API_KEY_MSG)
                    else:
                        st.error(result)

    st.divider()

    # ── Model selection ──────────────────────────────────────────────────────
    master_label = MODELS[MASTER_MODEL_KEY]["label"]
    st.info(UI_MASTER_MODEL_NOTE.format(label=master_label))

    st.subheader(UI_SELECT_MODELS)
    selected_keys: list[str] = []
    for key, cfg in MODELS.items():
        if key == MASTER_MODEL_KEY:
            continue
        if st.checkbox(label=cfg["label"], value=True, key=f"chk_{key}"):
            selected_keys.append(key)

    st.divider()
    st.caption(
        "API keys are read from environment variables:\n"
        "- `ANTHROPIC_API_KEY`\n"
        "- `OPENAI_API_KEY`\n"
        "- `GOOGLE_API_KEY`\n"
        "- `XAI_API_KEY`\n"
        "- `SERPER_API_KEY` _(optional — enables pre-flight live search)_\n"
        "- `FIREBASE_WEB_API_KEY` _(required for user authentication)_"
    )

    # ── Cloud Sync status indicator ───────────────────────────────────────────
    if firestore_status():
        st.success("☁️ Cloud Sync Active", icon="✅")
    else:
        st.caption("☁️ _Cloud Sync: offline — local history only_")

    # ── Daily usage quota ─────────────────────────────────────────────────────
    _sb_uid = (st.session_state.user or {}).get("uid")
    # Cache the usage count in session_state so we only hit Firestore once
    # per session instead of on every keystroke / re-render.
    _cached = st.session_state._cached_usage
    if _sb_uid and (_cached is None or _cached[0] != _sb_uid):
        st.session_state._cached_usage = (_sb_uid, get_usage_count(_sb_uid))
    _usage_today   = st.session_state._cached_usage[1] if _sb_uid else 0
    _quota_reached = _usage_today >= DAILY_QUERY_LIMIT
    if _sb_uid:
        st.divider()
        st.progress(min(_usage_today / DAILY_QUERY_LIMIT, 1.0))
        st.caption(UI_DAILY_USAGE_TPL.format(count=_usage_today, limit=DAILY_QUERY_LIMIT))
        if _quota_reached:
            st.warning(UI_QUOTA_REACHED)

    # ── Help & Methodology ────────────────────────────────────────────────────
    with st.expander(UI_HELP_EXPANDER_TITLE, expanded=False):
        from datetime import date as _hdate
        _hd = _hdate.today()
        _data_date_line = UI_HELP_DATA_DATE_TPL.format(
            day_name = HEBREW_WEEKDAYS[_hd.weekday()],
            day      = _hd.day,
            month    = HEBREW_MONTHS[_hd.month],
            year     = _hd.year,
        )
        st.markdown(UI_HELP_COUNCIL)
        st.markdown(UI_HELP_RELIABILITY)
        st.markdown(UI_HELP_TECH_APPENDIX)
        st.markdown(_data_date_line)

    # ── Session history — only for authenticated users ────────────────────────
    st.divider()
    st.subheader(UI_HISTORY_TITLE)

    _uid = (st.session_state.user or {}).get("uid")

    if not _uid:
        # Not logged in — show nothing; each user sees only their own history
        st.caption(UI_HISTORY_EMPTY)
        history = []
    else:
        history = get_user_history(_uid)

    if not history:
        st.caption(UI_HISTORY_EMPTY)
    else:
        from datetime import datetime as _sdt, date as _sdate, timedelta as _std
        from pathlib import Path as _SPath

        _today     = _sdate.today()
        _yesterday = _today - _std(days=1)

        # Bucket entries by relative date into ordered groups
        _groups: dict[str, list] = {
            "today":     [],
            "yesterday": [],
            "earlier":   [],
            "older":     [],
        }
        for _i, _entry in enumerate(history):
            # Timestamp format: "2026-03-11  14:30" (double space separator)
            try:
                _edate = _sdt.strptime(_entry["timestamp"].split()[0], "%Y-%m-%d").date()
            except (ValueError, IndexError):
                _edate = None

            if _edate == _today:
                _groups["today"].append((_i, _entry, _edate))
            elif _edate == _yesterday:
                _groups["yesterday"].append((_i, _entry, _edate))
            elif (
                _edate
                and _edate.year  == _today.year
                and _edate.month == _today.month
            ):
                _groups["earlier"].append((_i, _entry, _edate))
            else:
                _groups["older"].append((_i, _entry, _edate))

        def _render_history_entry(idx: int, entry: dict, edate) -> None:
            q   = entry["question"]
            rs  = entry["results"]

            # 🛡️ badge when the debate has reliability scores (consensus verified)
            shield    = UI_HISTORY_VERIFIED_BADGE if rs.get("reliability_scores") else ""
            short_q   = (q[:48] + "…") if len(q) > 48 else q
            btn_label = f"{shield} {short_q}".strip()

            # Human-readable date sub-text: "11 Mar 2026"
            date_str = (
                f"{edate.day} {edate.strftime('%b %Y')}"
                if edate else entry["timestamp"]
            )

            # PDF quick-download: only show if the cached file still exists on disk
            _cache    = rs.get("pdf_cache_path", "")
            _has_pdf  = bool(_cache and _SPath(_cache).exists())

            if _has_pdf:
                _c1, _c2 = st.columns([3, 1])
                with _c1:
                    if st.button(btn_label, key=f"hist_{idx}", use_container_width=True):
                        st.session_state.current_results = rs
                        st.session_state.from_history    = True
                        st.rerun()
                with _c2:
                    try:
                        _pdf_data  = _SPath(_cache).read_bytes()
                        _pdf_fname = (
                            f"AI_Playground_{edate.isoformat()}.pdf"
                            if edate else "AI_Playground_report.pdf"
                        )
                        st.download_button(
                            label=UI_HISTORY_PDF_BTN,
                            data=_pdf_data,
                            file_name=_pdf_fname,
                            mime="application/pdf",
                            key=f"hist_pdf_{idx}",
                            use_container_width=True,
                        )
                    except Exception:
                        pass
            else:
                if st.button(btn_label, key=f"hist_{idx}", use_container_width=True):
                    st.session_state.current_results = rs
                    st.session_state.from_history    = True
                    st.rerun()

            st.caption(date_str)

        def _render_group(group_label: str, items: list) -> None:
            if not items:
                return
            st.caption(f"**{group_label}**")
            for _gi, _ge, _gd in items:
                _render_history_entry(_gi, _ge, _gd)

        _render_group(UI_HISTORY_GROUP_TODAY,     _groups["today"])
        _render_group(UI_HISTORY_GROUP_YESTERDAY, _groups["yesterday"])
        _render_group(UI_HISTORY_GROUP_EARLIER,   _groups["earlier"])
        _render_group(UI_HISTORY_GROUP_OLDER,     _groups["older"])

        st.divider()
        if st.button(UI_HISTORY_CLEAR, use_container_width=True):
            clear_history()
            # Only clear current view if it came from history
            if st.session_state.from_history:
                st.session_state.current_results = None
                st.session_state.from_history    = False
            st.rerun()


# ---------------------------------------------------------------------------
# Main area — authentication gate
# ---------------------------------------------------------------------------
if _logo_img:
    _hc1, _hc2, _hc3 = st.columns([1, 8, 1])
    with _hc1:
        st.image(_logo_img, width=72)
    with _hc2:
        st.markdown(f"## {APP_TITLE}")
        st.caption(APP_SUBTITLE)
else:
    st.header(f"{APP_ICON} {APP_TITLE}")
    st.subheader(APP_SUBTITLE)
st.write("")

if not st.session_state.user:
    st.info(UI_AUTH_GATE_MSG)
    st.stop()

# ── Email verification gate ───────────────────────────────────────────────
_current_user = st.session_state.user
if not _current_user.get("email_verified", False):
    st.warning(UI_EMAIL_NOT_VERIFIED)
    _id_tok = _current_user.get("id_token", "")
    _vcol1, _vcol2 = st.columns(2)
    with _vcol1:
        if st.button(UI_RESEND_VERIFICATION, use_container_width=True):
            _vresult = send_verification_email(_id_tok)
            if _vresult is True:
                st.success(UI_RESEND_SUCCESS)
            else:
                st.error(_vresult)
    with _vcol2:
        if st.button(UI_REFRESH_VERIFY_BTN, use_container_width=True):
            _vcheck = get_email_verified(_id_tok)
            if _vcheck is True:
                st.session_state.user["email_verified"] = True
                st.rerun()
            elif _vcheck is False:
                st.warning(UI_STILL_NOT_VERIFIED)
            else:
                st.error(_vcheck)   # error string from API
    st.stop()

# ---------------------------------------------------------------------------
# Main area — question input & submit button (authenticated users only)
# ---------------------------------------------------------------------------

_qkey = f"question_input_{st.session_state._query_counter}"
_ukey = f"image_upload_{st.session_state._query_counter}"

question = st.text_area(
    label=UI_ASK_LABEL,
    placeholder=UI_ASK_PLACEHOLDER,
    height=120,
    key=_qkey,
)

# ── Image upload (optional, multiple) ────────────────────────────────────
uploaded_files = st.file_uploader(
    UPLOAD_IMAGE_LABEL,
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key=_ukey,
)

images_bytes: list = []
images_mime:  list = []

if uploaded_files:
    for f in uploaded_files[:4]:   # cap at 4
        images_bytes.append(f.getvalue())
        images_mime.append(f.type or "image/jpeg")

    with st.expander(IMAGE_PREVIEW_HEADER, expanded=True):
        n_cols = min(len(uploaded_files), 4)
        cols   = st.columns(n_cols)
        for col, f in zip(cols, uploaded_files[:4]):
            with col:
                st.image(f.getvalue(), use_container_width=True)
                st.caption(f"{f.name} · {len(f.getvalue()) // 1024} KB")
        if len(uploaded_files) > 4:
            st.caption(f"_Showing first 4 of {len(uploaded_files)} images._")

# ---------------------------------------------------------------------------
# Data file upload — CSV / Excel / PDF / JSON / TXT / DOCX
# ---------------------------------------------------------------------------

_dkey = f"data_upload_{st.session_state._query_counter}"

data_uploaded_files = st.file_uploader(
    UI_FILE_UPLOAD_LABEL,
    type=FILE_UPLOAD_ACCEPTED_TYPES,
    accept_multiple_files=True,
    key=_dkey,
    help=f"עד {FILE_UPLOAD_MAX_SIZE_MB} MB לקובץ. התוכן יוזרק לכל המודלים כהקשר מאומת.",
)

# Process uploaded data files into a context block
if "data_file_context" not in st.session_state:
    st.session_state.data_file_context    = ""
    st.session_state.data_file_names      = []
    st.session_state.data_file_warnings   = []

if data_uploaded_files:
    _pairs = [(f.name, f.getvalue()) for f in data_uploaded_files]
    _df_block, _df_warns = build_file_context(_pairs)
    st.session_state.data_file_context  = _df_block
    st.session_state.data_file_names    = [f.name for f in data_uploaded_files]
    st.session_state.data_file_warnings = _df_warns

    if _df_block:
        _col_info, _col_clr = st.columns([5, 1])
        with _col_info:
            st.success(UI_FILE_UPLOAD_LOADED.format(
                count=len(data_uploaded_files),
                chars=len(_df_block),
            ))
        with _col_clr:
            if st.button(UI_FILE_UPLOAD_CLEAR_BTN, key="data_file_clear"):
                st.session_state.data_file_context  = ""
                st.session_state.data_file_names    = []
                st.session_state.data_file_warnings = []
                st.rerun()

        if _df_warns:
            st.caption(UI_FILE_UPLOAD_WARNINGS.format(count=len(_df_warns)))
            with st.expander(UI_FILE_UPLOAD_FILES_HDR, expanded=False):
                for w in _df_warns:
                    st.caption(f"⚠️ {w}")
    else:
        st.warning(UI_FILE_UPLOAD_EMPTY)

elif not data_uploaded_files and st.session_state.data_file_context:
    # User removed files from the uploader — clear the context
    st.session_state.data_file_context  = ""
    st.session_state.data_file_names    = []
    st.session_state.data_file_warnings = []

# ---------------------------------------------------------------------------
# Code Review panel — scan a local project folder
# ---------------------------------------------------------------------------

# Initialise session keys on first load
if "code_review_context" not in st.session_state:
    st.session_state.code_review_context  = ""   # formatted context block
if "code_review_files" not in st.session_state:
    st.session_state.code_review_files    = []   # list of relative file paths
if "code_review_warnings" not in st.session_state:
    st.session_state.code_review_warnings = []

with st.expander(UI_CODE_REVIEW_TOGGLE, expanded=bool(st.session_state.code_review_context)):
    _cr_col_path, _cr_col_btn, _cr_col_clear = st.columns([5, 1, 1])

    with _cr_col_path:
        _cr_path = st.text_input(
            UI_CODE_REVIEW_PATH_LABEL,
            placeholder=UI_CODE_REVIEW_PATH_PLACEHOLDER,
            key="code_review_path_input",
            label_visibility="collapsed",
        )

    with _cr_col_btn:
        _cr_scan = st.button(UI_CODE_REVIEW_SCAN_BTN, use_container_width=True)

    with _cr_col_clear:
        if st.button(UI_CODE_REVIEW_CLEAR_BTN, use_container_width=True,
                     disabled=not st.session_state.code_review_context):
            st.session_state.code_review_context  = ""
            st.session_state.code_review_files    = []
            st.session_state.code_review_warnings = []
            st.rerun()

    if _cr_scan and _cr_path:
        with st.spinner(UI_CODE_REVIEW_SCANNING):
            _cr_block, _cr_files, _cr_warns = scan_project(_cr_path.strip())
        if _cr_block:
            st.session_state.code_review_context  = _cr_block
            st.session_state.code_review_files    = _cr_files
            st.session_state.code_review_warnings = _cr_warns
            st.rerun()
        else:
            # scan returned nothing — display the reason
            _cr_err = _cr_warns[0] if _cr_warns else UI_CODE_REVIEW_EMPTY
            st.error(_cr_err)

    if st.session_state.code_review_context:
        _cr_name  = st.session_state.code_review_files[0].split("/")[0] if st.session_state.code_review_files else _cr_path
        _cr_chars = len(st.session_state.code_review_context)
        st.success(UI_CODE_REVIEW_SCANNED.format(
            count=len(st.session_state.code_review_files),
            name=_cr_name,
            chars=_cr_chars,
        ))

        if st.session_state.code_review_warnings:
            st.caption(UI_CODE_REVIEW_WARNINGS.format(
                count=len(st.session_state.code_review_warnings)
            ))

        with st.expander(UI_CODE_REVIEW_FILES_HEADER, expanded=False):
            for _f in st.session_state.code_review_files:
                st.caption(f"• {_f}")

# Merge all extra contexts into one block (data files + code review)
_combined_extra_context = "\n\n".join(filter(None, [
    st.session_state.data_file_context,
    st.session_state.code_review_context,
]))

# Auto-fill a default question when context is loaded but no question typed
_effective_question = question.strip()
if not _effective_question:
    if st.session_state.data_file_context:
        _effective_question = FILE_UPLOAD_DEFAULT_QUESTION
    elif st.session_state.code_review_context:
        _effective_question = CODE_REVIEW_DEFAULT_QUESTION

_can_submit = (
    bool(_effective_question) or bool(images_bytes) or bool(_combined_extra_context)
) and not _quota_reached

if st.session_state.current_results is not None:
    _col_submit, _col_clear = st.columns([3, 1])
    with _col_submit:
        start_button = st.button(
            UI_SUBMIT_BUTTON, type="primary", use_container_width=True,
            disabled=not _can_submit,
        )
    with _col_clear:
        if st.button(UI_NEW_QUERY_BUTTON, use_container_width=True, key="new_query_btn"):
            st.session_state.current_results = None
            st.session_state.from_history    = False
            st.session_state._query_counter += 1
            st.rerun()
else:
    start_button = st.button(
        UI_SUBMIT_BUTTON, type="primary", use_container_width=True,
        disabled=not _can_submit,
    )

st.divider()


# ---------------------------------------------------------------------------
# Reliability Scorecard
# ---------------------------------------------------------------------------

class ReliabilityScorecard:
    """
    Calculates a 0-100 Consistency Score for one model based on how many
    retraction phrases appear in its Stage 3 dialectic response.

    Score  ≥ 80  → High Consistency  (green)
    Score  ≥ 50  → Partial Revision  (amber)
    Score  <  50 → Major Revision    (red)
    """

    # Phrases in English + Hebrew that signal a retraction / position change.
    _TRIGGERS = (
        "you are correct", "you're right", "i was wrong", "i acknowledge",
        "i concede", "i was mistaken", "valid point", "i agree with",
        "i must retract", "i now agree", "i stand corrected",
        "i missed", "i overlooked", "correct to point out",
        "אתה צודק", "טעיתי", "אני מסכים", "הערה נכונה",
    )

    _COLOURS = {
        "high":    "#10B981",   # emerald
        "partial": "#F59E0B",   # amber
        "low":     "#EF4444",   # red
    }

    def __init__(self, model_key: str, dialectic_response: str) -> None:
        self.model_key  = model_key
        self._response  = dialectic_response.lower()
        self.retractions: list = [t for t in self._TRIGGERS if t in self._response]
        self.score: int = max(0, 100 - len(self.retractions) * 20)

    @property
    def _tier(self) -> str:
        if self.score >= 80:
            return "high"
        if self.score >= 50:
            return "partial"
        return "low"

    @property
    def colour(self) -> str:
        return self._COLOURS[self._tier]

    @property
    def label(self) -> str:
        return {"high": "High Consistency", "partial": "Partial Revision",
                "low": "Major Revision"}[self._tier]

    def render_chip(self) -> str:
        """Return an HTML chip for use in st.markdown(unsafe_allow_html=True)."""
        model_label = MODELS[self.model_key]["label"]
        # Math Aligned badge: only when the model held its position perfectly
        # (no retractions), meaning its Stage 0 numbers were never challenged.
        math_badge = (
            '<span style="font-size:0.7em;font-weight:700;'
            'background:rgba(255,255,255,0.22);border-radius:3px;'
            'padding:1px 6px;margin-left:5px">🧮 Math Aligned</span>'
            if self.score == 100 else ""
        )
        return (
            f'<span class="scorecard-chip" style="background:{self.colour}">'
            f'{model_label}'
            f'<span class="scorecard-chip-score">{self.score}</span>'
            f'<span style="font-size:0.78em;font-weight:400">{self.label}</span>'
            f'{math_badge}'
            f'</span>'
        )


# ---------------------------------------------------------------------------
# Helpers — rendering
# ---------------------------------------------------------------------------

def _model_badge(model_key: str) -> str:
    """Return an inline HTML badge coloured with the model's brand colour."""
    cfg   = MODELS[model_key]
    color = cfg["color"]
    label = cfg["label"]
    return (
        f'<span style="background:{color};color:white;'
        f'padding:2px 8px;border-radius:4px;font-size:0.85em">{label}</span>'
    )


def _render_citation_cards(citations: list[dict]) -> None:
    """
    3-column grid of source cards.  Falls back to a 'no data' caption
    when grounding was not triggered.
    """
    st.subheader(UI_SECTION_CITATIONS)

    if not citations:
        st.caption(UI_CITATIONS_NO_DATA)
        return

    from urllib.parse import urlparse

    cols_per_row = 3
    for row_start in range(0, len(citations), cols_per_row):
        row_items = citations[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, citation in zip(cols, row_items):
            url   = citation["url"]
            title = citation["title"]
            try:
                domain = urlparse(url).netloc.lstrip("www.")
            except Exception:
                domain = url

            with col:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e2e8f0;border-radius:8px;
                        padding:14px 16px;background:#f8fafc;
                        height:100%;min-height:90px;">
                      <div style="font-size:.78em;color:#64748b;font-weight:600;
                          letter-spacing:.03em;text-transform:uppercase;
                          margin-bottom:4px">{domain}</div>
                      <div style="font-size:.9em;color:#1e293b;font-weight:500;
                          margin-bottom:10px;line-height:1.4">{title}</div>
                      <a href="{url}" target="_blank" style="font-size:.82em;
                          color:#3b82f6;text-decoration:none;font-weight:600"
                      >{UI_CITATION_VISIT}</a>
                    </div>""",
                    unsafe_allow_html=True,
                )


def _display_results(results: dict) -> None:
    """
    Render the complete results view from a results dict.
    Called both after a live debate and when loading from history —
    the rendering is identical in both cases.
    """
    import re as _re
    from datetime import datetime as _dt

    # History / cache banner
    if st.session_state.from_history:
        st.info(UI_HISTORY_BANNER)
        # "Run anyway" button — lets the user force a fresh API call
        if st.button(UI_CACHE_RERUN_BTN, key="force_rerun_btn"):
            st.session_state._force_rerun = True
            st.session_state.current_results = None
            st.session_state.from_history    = False
            st.rerun()

    final     = results["final_answer"]
    citations = results.get("citations", [])

    # ── Certified Report container ────────────────────────────────────────────
    st.markdown(REPORT_TEMPLATE_CSS, unsafe_allow_html=True)
    st.subheader(UI_SECTION_FINAL)

    if results.get("fallback_used"):
        st.warning(FALLBACK_NOTICE)

    report_date = _dt.now().strftime("%A, %B %d, %Y")
    n_models    = len(results.get("answers", {}))

    report_header = f"""
<div class="council-report">
<div class="council-report-accent"></div>
<div class="council-report-header">
  <div>
    <div class="council-report-title">🎮 AI-Playground — Certified Council Report</div>
    <div style="font-size:0.78em;color:#64748b;margin-top:3px">
      DSAD · {n_models} agents · 4 stages
    </div>
  </div>
  <div class="council-report-meta">
    {report_date}<br>
    <span style="font-size:0.9em">Research &amp; Educational Use Only</span>
  </div>
</div>
<div class="council-report-body">
"""
    report_footer = """
</div>
<div class="council-report-footer">
  GENERATED BY AI-PLAYGROUND COUNCIL &nbsp;·&nbsp; FOR RESEARCH &amp; EDUCATIONAL PURPOSES ONLY
</div>
</div>
"""
    if final.startswith("ERROR:"):
        st.error(final)
    else:
        # Split narrative answer from the Technical Breakdown section so the
        # breakdown can be rendered in its own distinct low-opacity container.
        _td_match = _re.search(
            r'(?m)^##\s*(?:📐\s*)?Technical Breakdown\b',
            final,
            _re.IGNORECASE,
        )
        if _td_match:
            narrative_text = final[:_td_match.start()].strip()
            breakdown_text = final[_td_match.start():].strip()
        else:
            narrative_text = final
            breakdown_text = ""

        st.markdown(report_header, unsafe_allow_html=True)
        st.markdown(narrative_text)

        if breakdown_text:
            st.markdown(
                '<div class="tech-breakdown">'
                '<div class="tech-breakdown-header">'
                '📐 Technical Breakdown — Verified Calculation'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(breakdown_text)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(report_footer, unsafe_allow_html=True)

    # ── Zero-Trust alert ──────────────────────────────────────────────────────
    if not citations and not final.startswith("ERROR:") and _re.search(r"\d", final):
        st.error(UI_UNVERIFIED_WARNING)

    # ── Reliability Dashboard (SVG speedometers) ──────────────────────────────
    dialectic          = results.get("dialectic", {})
    reliability_scores = results.get("reliability_scores", {})

    if dialectic:
        st.markdown("#### 📊 Reliability Dashboard")
        gauges_html = "".join(
            render_reliability_gauge(MODELS[key]["label"], reliability_scores.get(key, 100))
            for key in dialectic
        )
        st.markdown(
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;'
            f'justify-content:center;margin:8px 0 20px">{gauges_html}</div>',
            unsafe_allow_html=True,
        )

    # ── PDF download (serve cached file if available, else regenerate) ───────
    try:
        from datetime import date as _date
        from pathlib import Path as _Path
        _cache_path = results.get("pdf_cache_path", "")
        pdf_bytes: Optional[bytes] = None
        if _cache_path:
            try:
                pdf_bytes = _Path(_cache_path).read_bytes()
            except Exception:
                pdf_bytes = None   # cached file missing — fall back to regen
        if pdf_bytes is None:
            pdf_bytes = generate_pdf(results)
        filename = f"AI_Playground_Report_{_date.today().isoformat()}.pdf"
        st.download_button(
            label=PDF_BTN_LABEL,
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as _err:
        st.caption(f"_PDF export unavailable: {_err}_")

    st.divider()

    # ── Citation cards ───────────────────────────────────────────────────────
    _render_citation_cards(citations)

    st.divider()

    # ── Full debate transcript (collapsed) ───────────────────────────────────
    with st.expander(UI_SECTION_PROCESS, expanded=False):

        # Stage 1 — Theses
        st.markdown("### " + STAGE_LABELS[1])
        for model_key, answer_text in results["answers"].items():
            label = MODELS[model_key]["label"]
            with st.expander(UI_EXPANDER_INITIAL.format(label=label), expanded=False):
                st.markdown(_model_badge(model_key), unsafe_allow_html=True)
                st.write("")
                if answer_text.startswith("ERROR:"):
                    st.error(UI_ERROR_PREFIX.format(label=label) + answer_text[7:])
                else:
                    st.markdown(answer_text if answer_text else UI_NO_ANSWER)

        # Stage 2 — Adversarial Audits
        if results.get("critiques"):
            st.markdown("### " + STAGE_LABELS[2])
            for (reviewer_key, target_key), critique_text in results["critiques"].items():
                reviewer_label = MODELS[reviewer_key]["label"]
                target_label   = MODELS[target_key]["label"]
                header = UI_EXPANDER_CRITIQUE.format(
                    label=reviewer_label, target_label=target_label
                )
                with st.expander(header, expanded=False):
                    st.markdown(
                        _model_badge(reviewer_key) + " → " + _model_badge(target_key),
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    if critique_text.startswith("ERROR:"):
                        st.error(
                            UI_ERROR_PREFIX.format(label=reviewer_label) + critique_text[7:]
                        )
                    else:
                        st.markdown(critique_text if critique_text else UI_NO_ANSWER)

        # Stage 3 — Dialectic Responses
        if dialectic:
            st.markdown("### " + STAGE_LABELS[3])
            for model_key, response_text in dialectic.items():
                label = MODELS[model_key]["label"]
                score = reliability_scores.get(model_key, 100)
                header = f"💬 {label} — Dialectic Response  (Consistency: {score}/100)"
                with st.expander(header, expanded=False):
                    st.markdown(_model_badge(model_key), unsafe_allow_html=True)
                    st.write("")
                    if response_text.startswith("ERROR:"):
                        st.error(UI_ERROR_PREFIX.format(label=label) + response_text[7:])
                    else:
                        st.markdown(response_text if response_text else UI_NO_ANSWER)

        # Stage 3b — Focused Debate
        focused_debate = results.get("focused_debate", {})
        if focused_debate and not focused_debate.get("skipped", True):
            st.markdown("---")
            st.markdown("### " + STAGE_LABELS["3b"])
            points    = focused_debate.get("disagreement_points", [])
            exchanges = focused_debate.get("exchanges", {})
            if not points:
                st.caption(UI_STAGE3B_NO_DEBATES)
            else:
                for point in points:
                    pid     = point["point_id"]
                    summary = point["summary"]
                    m_keys  = point["model_keys"]
                    st.markdown(UI_STAGE3B_POINT_HEADER.format(n=pid, summary=summary))
                    st.caption(f"**Contested excerpt:** _{point['excerpt']}_")
                    for rnd, round_label in [
                        (1, UI_STAGE3B_ROUND1_LABEL),
                        (2, UI_STAGE3B_ROUND2_LABEL),
                    ]:
                        st.markdown(f"**{round_label}**")
                        for mk in m_keys:
                            text  = exchanges.get((pid, rnd, mk), "")
                            label = MODELS[mk]["label"]
                            header = UI_STAGE3B_EXPANDER_MODEL.format(
                                label=label, n=pid, r=rnd
                            )
                            with st.expander(header, expanded=(rnd == 1)):
                                st.markdown(_model_badge(mk), unsafe_allow_html=True)
                                st.write("")
                                st.markdown(text or UI_NO_ANSWER)


# ---------------------------------------------------------------------------
# Debate execution — runs when the submit button is pressed OR when a
# follow-up question was submitted from the follow-up panel.
# ---------------------------------------------------------------------------

# ── Follow-up trigger: consume pending follow-up from session state ────────
_fup_question = st.session_state.pop("_pending_followup", None)
_fup_ctx      = st.session_state.pop("_followup_ctx", None)

if start_button or _fup_question:
    # Resolve the effective question and context for this run
    _active_question   = _fup_question or _effective_question
    _previous_ctx      = _fup_ctx      # None for regular questions
    _active_images     = [] if _fup_question else images_bytes
    _active_images_mime = [] if _fup_question else images_mime
    _active_code_ctx   = "" if _fup_question else _combined_extra_context

    if not _active_question.strip() and not _active_images and not _active_code_ctx:
        st.warning("Please provide a question, upload an image, or scan a project folder.")
        st.stop()

    all_active_keys = list(dict.fromkeys([MASTER_MODEL_KEY] + selected_keys))
    if len(all_active_keys) < 2:
        st.warning(UI_WARNING_MIN_MODELS)
        st.stop()

    # ── Cache-hit check: skip for follow-ups (always unique) ─────────────────
    if _active_question.strip() and not _active_images and not st.session_state._force_rerun \
            and not _fup_question:
        _uid_for_cache = (st.session_state.user or {}).get("uid")
        _hist_for_cache = get_user_history(_uid_for_cache) if _uid_for_cache else load_history()
        _q_norm = _active_question.strip().lower()
        _hit = next(
            (e for e in _hist_for_cache if e.get("question", "").strip().lower() == _q_norm),
            None,
        )
        if _hit:
            st.session_state.current_results = _hit["results"]
            st.session_state.from_history    = True
            st.session_state._force_rerun    = False
            st.rerun()

    # Reset force-rerun flag for next cycle
    st.session_state._force_rerun = False

    # ── Usage limit check (server-side gate via Firestore) ────────────────────
    _uid_limit = (st.session_state.user or {}).get("uid")
    if _uid_limit:
        _allowed, _new_count = check_usage_limit(_uid_limit)
        # Bust the cached count so the sidebar reflects the new value
        st.session_state._cached_usage = (_uid_limit, _new_count)
        if not _allowed:
            st.warning(UI_QUOTA_REACHED)
            st.stop()

    # ── Inject CSS (spinner ring + mission control dashboard) ────────────────
    st.markdown(_SPINNER_CSS + MISSION_CONTROL_CSS, unsafe_allow_html=True)

    _n_imgs = len(_active_images)
    _main_label = (
        f"👁️ Vision Council — analysing {_n_imgs} image{'s' if _n_imgs > 1 else ''} …"
        if _active_images
        else ("🔄 AI Council — Follow-up Debate …" if _fup_question
              else "🤖 AI Council is deliberating …")
    )

    # Mission Control circle row — lives OUTSIDE st.status so it stays visible
    # while the spinner/log panel collapses.
    dashboard_ph = st.empty()

    with st.status(_main_label, expanded=True) as status:

        # Rainbow spinner ring (CSS animated)
        st.markdown(_SPINNER_HTML, unsafe_allow_html=True)

        # ── Log placeholder inside the status block ───────────────────────────
        log_ph = st.empty()   # collapsible tech log — re-rendered each callback

        _state: dict = {"stage": -1, "completed": set(), "log": []}

        def _refresh_ui() -> None:                                   # noqa: E306
            """Re-render the circle dashboard and the collapsible log panel."""
            try:
                dashboard_ph.markdown(
                    render_status_dashboard(_state["stage"], _state["completed"]),
                    unsafe_allow_html=True,
                )
                if _state["log"]:
                    lines_html = "".join(
                        f'<div class="sd-log-line">{ln}</div>'
                        for ln in _state["log"][-12:]
                    )
                    log_ph.markdown(
                        f'<details class="sd-details">'
                        f'<summary class="sd-summary">{UI_TECH_LOG_TITLE}</summary>'
                        f'<div class="sd-log-wrap">{lines_html}</div>'
                        f'</details>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        _refresh_ui()   # initial render — all circles pending

        # ── Stage callbacks — each updates dashboard + appends to log ─────────

        def stage0_cb(fraction: float, text: str) -> None:          # noqa: E306
            try:
                if fraction < 1.0:
                    _state["stage"] = 0
                else:
                    _state["completed"].add(0)
                    _state["stage"] = -1
                _state["log"].append(text)
                _refresh_ui()
                _lbl = "🔍 Stage 0 — Complete" if fraction >= 1.0 else "🔍 Stage 0 — searching …"
                status.update(label=_lbl)
            except Exception:
                pass

        def stage1_cb(fraction: float, text: str) -> None:          # noqa: E306
            try:
                if fraction >= 1.0:
                    _state["completed"].add(1)
                    _state["stage"] = -1
                else:
                    _state["stage"] = 1
                _state["log"].append(f"⚙️ Stage 1 — {text}")
                _refresh_ui()
                status.update(label="⚙️ Stage 1 — running …")
            except Exception:
                pass

        def stage2_cb(fraction: float, text: str) -> None:          # noqa: E306
            try:
                if fraction >= 1.0:
                    _state["completed"].add(2)
                    _state["stage"] = -1
                else:
                    _state["stage"] = 2
                _state["log"].append(f"🔬 Stage 2 — {text}")
                _refresh_ui()
                status.update(label="🔬 Stage 2 — running …")
            except Exception:
                pass

        def stage3_cb(fraction: float, text: str) -> None:          # noqa: E306
            try:
                if fraction >= 1.0:
                    _state["completed"].add(3)
                    _state["stage"] = -1
                else:
                    _state["stage"] = 3
                _state["log"].append(f"💬 Stage 3 — {text}")
                _refresh_ui()
                status.update(label="💬 Stage 3 — running …")
            except Exception:
                pass

        def stage3b_cb(fraction: float, text: str) -> None:         # noqa: E306
            try:
                if fraction >= 1.0:
                    _state["completed"].add(4)
                    _state["stage"] = -1
                else:
                    _state["stage"] = 4
                _state["log"].append(f"⚔️ Stage 3b — {text}")
                _refresh_ui()
                status.update(label="⚔️ Stage 4 — Focused Debate running …")
            except Exception:
                pass

        def stage4_cb(fraction: float, text: str) -> None:          # noqa: E306
            try:
                if fraction >= 1.0:
                    _state["completed"].add(5)
                    _state["stage"] = -1
                else:
                    _state["stage"] = 5
                _state["log"].append(f"🏛️ Stage 5 — {text}")
                _refresh_ui()
                status.update(label="🏛️ Stage 5 — Synthesis running …")
            except Exception:
                pass

        # ── Run the full DSAD debate ──────────────────────────────────────────
        _live_results = run_council_debate(
            active_keys=selected_keys,
            question=_active_question,
            images=_active_images,
            images_mime=_active_images_mime,
            previous_context=_previous_ctx,
            code_context=_active_code_ctx,
            stage0_cb=stage0_cb,
            stage1_cb=stage1_cb,
            stage2_cb=stage2_cb,
            stage3_cb=stage3_cb,
            stage3b_cb=stage3b_cb,
            stage4_cb=stage4_cb,
        )

        # ── Final state: all circles complete ─────────────────────────────────
        _state["completed"] = {0, 1, 2, 3, 4, 5}
        _state["stage"]     = -1
        _refresh_ui()

        status.update(
            label="✅ Council deliberation complete — results ready!",
            state="complete",
            expanded=False,
        )

    # ── Persist and re-render from session state ──────────────────────────────
    # Generate PDF once here so save_to_history can cache it to disk.
    # The cached path is stored in session state so _display_results() serves
    # the file directly without regenerating on the first (live-run) display.
    _pdf_for_cache: Optional[bytes] = None
    try:
        _pdf_for_cache = generate_pdf(_live_results)
    except Exception:
        pass

    _current_uid  = (st.session_state.user or {}).get("uid")
    _cached_path  = save_to_history(_live_results, pdf_bytes=_pdf_for_cache, uid=_current_uid)
    if _cached_path:
        _live_results["pdf_cache_path"] = _cached_path

    st.session_state.current_results = _live_results
    st.session_state.from_history    = False
    st.rerun()


# ---------------------------------------------------------------------------
# Results display — driven entirely by session state so it works
# identically for live debates and history replays
# ---------------------------------------------------------------------------
if st.session_state.current_results:
    _display_results(st.session_state.current_results)

    # ── Follow-up question panel ───────────────────────────────────────────
    st.divider()
    _fup_key = f"followup_{st.session_state._query_counter}"
    _fup_q   = st.text_area(
        label=UI_FOLLOWUP_LABEL,
        placeholder=UI_FOLLOWUP_PLACEHOLDER,
        height=90,
        key=_fup_key,
    )
    if st.button(UI_FOLLOWUP_SUBMIT_BTN, type="primary",
                 disabled=not bool((_fup_q or "").strip())):
        st.session_state["_pending_followup"]  = _fup_q.strip()
        st.session_state["_followup_ctx"]      = st.session_state.current_results
        st.session_state.current_results       = None
        st.session_state.from_history          = False
        st.session_state._query_counter       += 1
        st.rerun()
