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
import html
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as _datetime
from pathlib import Path
from typing import Optional

_CURRENT_YEAR = _datetime.now().year

load_dotenv()  # טוען את המפתחות מקובץ ה-.env

import streamlit as st
from streamlit_javascript import st_javascript
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
    UI_FOLLOWUP_IMG_LABEL,
    UI_FOLLOWUP_DATA_LABEL,
    UI_FOLLOWUP_DATA_LOADED,
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
    UI_MEDIA_PANEL_HEADER,
    UI_MEDIA_PANEL_CAPTION,
    UI_MEDIA_IMAGE_EXPANDER,
    UI_MEDIA_VIDEO_EXPANDER,
    UI_MEDIA_CONTEXT_LABEL,
    UI_MEDIA_CONTEXT_HELP,
    UI_MEDIA_SIZE_LABEL,
    UI_MEDIA_SIZE_OPTIONS,
    UI_MEDIA_VIDEO_MODEL_LABEL,
    UI_MEDIA_GENERATE_IMG_BTN,
    UI_MEDIA_GENERATE_VID_BTN,
    UI_MEDIA_STEP_PROMPT,
    UI_MEDIA_STEP_IMAGE,
    UI_MEDIA_STEP_VIDEO,
    UI_MEDIA_DL_IMG_BTN,
    UI_MEDIA_DL_VID_BTN,
    UI_MEDIA_PROMPT_USED_HDR,
    UI_MEDIA_ERROR_PREFIX,
    UI_MEDIA_NO_REPLICATE_KEY,
    UI_MEDIA_REGEN_BTN,
    UI_MEDIA_VIDEO_MODE_LABEL,
    UI_MEDIA_VIDEO_MODE_SINGLE,
    UI_MEDIA_VIDEO_MODE_STORY,
    UI_MEDIA_STORYBOARD_HDR,
    UI_MEDIA_STORYBOARD_NOTE,
    UI_MEDIA_SCENE_GENERATING,
    UI_MEDIA_STEP_STORYBOARD,
    UI_MEDIA_IMAGE_MODE_LABEL,
    UI_MEDIA_IMAGE_MODE_SINGLE,
    UI_MEDIA_IMAGE_MODE_STORY,
    UI_MEDIA_IMG_STORY_HDR,
    UI_MEDIA_IMG_STORY_NOTE,
    UI_MEDIA_IMG_STORY_COUNT,
    UI_MEDIA_STEP_IMG_STORY,
    UI_MEDIA_STEP_IMG_GEN,
    UI_MEDIA_IMG_REGEN_STORY,
    UI_MEDIA_DL_IMG_N,
    UI_MEDIA_BRANDING_LABEL,
    UI_MEDIA_BRANDING_HELP,
    UI_MEDIA_BRANDING_ANALYSING,
    UI_MEDIA_BRANDING_DONE,
    UI_MEDIA_REF_IMG_LABEL,
    UI_MEDIA_REF_IMG_HELP,
    UI_MEDIA_REF_ANALYSING,
    UI_MEDIA_REF_DONE,
    UI_MEDIA_STYLE_LABEL,
    UI_MEDIA_STYLE_REALISTIC,
    UI_MEDIA_STYLE_ANIMATION,
    CULTURAL_PROFILES,
    UI_CULTURAL_EXPANDER,
    UI_CULTURAL_CAPTION,
    UI_CULTURAL_MARKET_SELECT,
    UI_CULTURAL_MEDIA_TYPE,
    UI_CULTURAL_MEDIA_IMAGE,
    UI_CULTURAL_MEDIA_VIDEO,
    UI_CULTURAL_SIZE_LABEL,
    UI_CULTURAL_CTX_LABEL,
    UI_CULTURAL_CTX_PLACEHOLDER,
    UI_CULTURAL_GENERATE_BTN,
    UI_CULTURAL_GENERATING,
    UI_CULTURAL_RESULT_HDR,
    UI_CULTURAL_DL_IMG,
    UI_CULTURAL_DL_VID,
    UI_CULTURAL_PROMPT_HDR,
    UI_CULTURAL_MIN_MARKETS,
    UI_CULTURAL_NO_VIDEO_KEY,
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
    PROVIDER_KEY_CONFIG,
    APIKEY_STATUS_ICON,
    APIKEY_STATUS_LABEL,
    UI_APIKEY_BTN_SIDEBAR,
    UI_APIKEY_DIALOG_TITLE,
    UI_APIKEY_DIALOG_SUBTITLE,
    UI_APIKEY_FIELD_LABEL,
    UI_APIKEY_VALIDATE_SAVE,
    UI_APIKEY_VALIDATING,
    UI_APIKEY_CLEAR_BTN,
    UI_APIKEY_SAVED_OK,
    UI_APIKEY_PARTIAL_SAVED,
    UI_APIKEY_CLEARED,
    UI_APIKEY_ACTIVE_BADGE,
    UI_APIKEY_GET_KEY_LINK,
    UI_APIKEY_UNCHANGED_SKIP,
    UI_ACADEMIC_TOGGLE,
    UI_ACADEMIC_CAPTION,
    UI_ACADEMIC_CHECKBOX,
    UI_ACADEMIC_YEAR_FROM,
    UI_ACADEMIC_YEAR_TO,
)
from logic_engine import run_council_debate
from project_reader import scan_project
from file_reader import build_file_context
from ai_factory import set_session_api_keys, validate_api_key
from report_generator import generate_pdf
from history_manager import (
    check_usage_limit,
    clear_history,
    firestore_status,
    get_usage_count,
    get_user_history,
    increment_usage,
    is_within_usage_limit,
    load_history,
    safe_pdf_cache_path,
    save_to_history,
)
from auth_manager import (
    get_email_verified,
    refresh_session,
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
if "user_api_keys" not in st.session_state:
    st.session_state.user_api_keys = {}    # provider → api_key string (BYOK)
if "_ls_loaded" not in st.session_state:
    st.session_state._ls_loaded = False    # True once localStorage has been read
if "_user_loaded" not in st.session_state:
    st.session_state._user_loaded = False  # True once auth localStorage has been read

# ---------------------------------------------------------------------------
# localStorage bridge — load saved API keys into session on first render
# ---------------------------------------------------------------------------
# st_javascript() is async: first call returns 0 (int); actual value arrives
# on the second render.  We guard with isinstance(..., str) so we only process
# real data and never accidentally mark the session as "loaded" on the dummy 0.
if not st.session_state._ls_loaded:
    _ls_raw = st_javascript("localStorage.getItem('ai_council_keys') ?? ''")
    if isinstance(_ls_raw, str):          # real response (not the initial 0)
        if len(_ls_raw) > 2:              # non-empty JSON object
            try:
                _ls_keys = json.loads(_ls_raw)
                if isinstance(_ls_keys, dict) and _ls_keys:
                    st.session_state.user_api_keys = _ls_keys
                    set_session_api_keys(_ls_keys)
            except (json.JSONDecodeError, TypeError):
                pass
        st.session_state._ls_loaded = True

# ---------------------------------------------------------------------------
# Auth persistence — restore login session from localStorage on page refresh
# ---------------------------------------------------------------------------
# Only runs when the user is not already logged in and we haven't read yet.
if not st.session_state.user and not st.session_state._user_loaded:
    _auth_raw = st_javascript("localStorage.getItem('ai_council_user') ?? ''")
    if isinstance(_auth_raw, str):          # real response (not initial 0)
        if len(_auth_raw) > 2:
            try:
                _stored = json.loads(_auth_raw)
                _rtok   = _stored.get("refresh_token", "")
                if _rtok:
                    _refreshed = refresh_session(_rtok)
                    if isinstance(_refreshed, dict):
                        st.session_state.user = _refreshed
                        # Update localStorage with the new id_token
                        _new_json = json.dumps(_refreshed)
                        st_javascript(
                            f"localStorage.setItem('ai_council_user', {json.dumps(_new_json)})"
                        )
                    else:
                        # Refresh failed (account disabled / token revoked) — clear storage
                        st_javascript("localStorage.removeItem('ai_council_user')")
            except (json.JSONDecodeError, TypeError):
                pass
        st.session_state._user_loaded = True


# ---------------------------------------------------------------------------
# API Key Management — inline helper (called inside sidebar expander)
# ---------------------------------------------------------------------------

def _render_api_key_expander() -> None:
    """
    Renders the BYOK API key management UI inside a sidebar st.expander.
    Keys are persisted in browser localStorage — never in the server DB.
    """
    _byok_active = bool(st.session_state.user_api_keys)
    _expander_label = f"🔑 API Keys {'✅' if _byok_active else ''}"
    with st.expander(_expander_label, expanded=False):
        st.caption("🔒 Your keys are stored locally in your browser and used only as a proxy. They are never saved to our database.")
        st.caption(UI_APIKEY_DIALOG_SUBTITLE)
        st.divider()

        draft: dict = {}
        for provider, cfg in PROVIDER_KEY_CONFIG.items():
            current = st.session_state.user_api_keys.get(provider, "")
            status  = st.session_state.get(f"_key_status_{provider}", "none" if not current else "valid")

            # ── Header row: status dot + label + link ─────────────────────────
            icon       = APIKEY_STATUS_ICON[status]
            status_lbl = APIKEY_STATUS_LABEL[status]
            col_hdr, col_link = st.columns([5, 1])
            with col_hdr:
                st.markdown(f"{icon} **{cfg['label']}** &nbsp; <small style='color:grey'>{status_lbl}</small>",
                            unsafe_allow_html=True)
            with col_link:
                st.link_button(UI_APIKEY_GET_KEY_LINK, cfg["url"], use_container_width=True)

            # ── Key input ─────────────────────────────────────────────────────
            val = st.text_input(
                UI_APIKEY_FIELD_LABEL,
                value=current,
                placeholder=cfg["placeholder"],
                type="password",
                key=f"_apikey_input_{provider}",
                label_visibility="collapsed",
            )
            draft[provider] = val.strip()
            st.write("")   # breathing room between providers

        st.divider()
        col_save, col_clear = st.columns([3, 1])

        with col_save:
            if st.button(UI_APIKEY_VALIDATE_SAVE, type="primary", use_container_width=True):
                # Separate unchanged keys from new/modified ones
                unchanged = {p: k for p, k in draft.items()
                             if k and k == st.session_state.user_api_keys.get(p)}
                to_validate = {p: k for p, k in draft.items()
                               if k and k != st.session_state.user_api_keys.get(p)}

                errors: list[str] = []
                new_keys: dict    = dict(unchanged)   # start with already-valid keys

                if to_validate:
                    with st.spinner(UI_APIKEY_VALIDATING):
                        with ThreadPoolExecutor(max_workers=len(to_validate)) as ex:
                            futs = {ex.submit(validate_api_key, p, k): p
                                    for p, k in to_validate.items()}
                            for fut in as_completed(futs):
                                p = futs[fut]
                                ok, msg = fut.result()
                                if ok:
                                    new_keys[p] = to_validate[p]
                                    st.session_state[f"_key_status_{p}"] = (
                                        "quota" if msg else "valid"
                                    )
                                    if msg:
                                        errors.append(f"🟠 **{PROVIDER_KEY_CONFIG[p]['label']}:** {msg}")
                                else:
                                    st.session_state[f"_key_status_{p}"] = "invalid"
                                    errors.append(f"🔴 **{PROVIDER_KEY_CONFIG[p]['label']}:** {msg}")

                # Clear status for providers whose field was emptied
                for p in PROVIDER_KEY_CONFIG:
                    if not draft.get(p):
                        st.session_state.pop(f"_key_status_{p}", None)

                # Persist valid keys
                st.session_state.user_api_keys = new_keys
                set_session_api_keys(new_keys)
                _ls_json = json.dumps(new_keys)
                st_javascript(f"localStorage.setItem('ai_council_keys', {json.dumps(_ls_json)})")
                st.session_state._ls_loaded = True

                if errors:
                    for e in errors:
                        st.markdown(e)
                    st.warning(UI_APIKEY_PARTIAL_SAVED if new_keys else "No valid keys were saved.")
                else:
                    st.success(UI_APIKEY_SAVED_OK)

                st.rerun()

        with col_clear:
            if st.button(UI_APIKEY_CLEAR_BTN, use_container_width=True):
                st.session_state.user_api_keys = {}
                set_session_api_keys({})
                st_javascript("localStorage.removeItem('ai_council_keys')")
                for p in PROVIDER_KEY_CONFIG:
                    st.session_state.pop(f"_key_status_{p}", None)
                st.info(UI_APIKEY_CLEARED)
                st.rerun()


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
            st.session_state.user         = None
            st.session_state.current_results = None
            st.session_state.from_history    = False
            st.session_state._user_loaded    = True   # don't re-read stale storage
            st_javascript("localStorage.removeItem('ai_council_user')")
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
                    st.session_state.user         = result
                    st.session_state._user_loaded = True
                    # Persist session in localStorage so refresh doesn't log out
                    _user_json = json.dumps(result)
                    st_javascript(
                        f"localStorage.setItem('ai_council_user', {json.dumps(_user_json)})"
                    )
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

    # ── BYOK — API key management ─────────────────────────────────────────────
    _render_api_key_expander()

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

            # PDF quick-download: pdf_cache_path comes from user-writable history,
            # so validate it resolves to a real file inside the PDF cache dir before
            # trusting it as a path (never read an arbitrary path from history).
            _safe_cache = safe_pdf_cache_path(rs.get("pdf_cache_path", ""))
            _has_pdf    = _safe_cache is not None

            if _has_pdf:
                _c1, _c2 = st.columns([3, 1])
                with _c1:
                    if st.button(btn_label, key=f"hist_{idx}", use_container_width=True):
                        st.session_state.current_results = rs
                        st.session_state.from_history    = True
                        st.rerun()
                with _c2:
                    try:
                        _pdf_data  = _safe_cache.read_bytes()
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
                # Persist the updated status to localStorage so the gate
                # does not re-appear on the next page refresh or login.
                _updated_json = json.dumps(st.session_state.user)
                st_javascript(
                    f"localStorage.setItem('ai_council_user', {json.dumps(_updated_json)})"
                )
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
    st.caption(
        "💡 על Streamlit Cloud, השרת לא רואה את המחשב שלך — "
        "העלה את הפרויקט כקובץ ZIP. בהרצה לוקאלית, אפשר גם להזין נתיב מקומי."
    )

    # Option A — ZIP upload (works on Streamlit Cloud)
    _cr_zip = st.file_uploader(
        "📦 העלה ZIP של הפרויקט",
        type=["zip"],
        accept_multiple_files=False,
        key="code_review_zip_input",
    )

    # Option B — local path. The path input and the local-path scan are gated
    # behind COUNCIL_ALLOW_LOCAL_SCAN=1: without it a deployed instance would let
    # any visitor type a server path and read arbitrary files. The ZIP flow above
    # is unaffected; the Scan / Clear buttons stay (the ZIP flow uses Scan too).
    _cr_allow_local = os.environ.get("COUNCIL_ALLOW_LOCAL_SCAN") == "1"
    _cr_path = ""
    if _cr_allow_local:
        _cr_col_path, _cr_col_btn, _cr_col_clear = st.columns([5, 1, 1])
        with _cr_col_path:
            _cr_path = st.text_input(
                UI_CODE_REVIEW_PATH_LABEL,
                placeholder=UI_CODE_REVIEW_PATH_PLACEHOLDER,
                key="code_review_path_input",
                label_visibility="collapsed",
            )
    else:
        _cr_col_btn, _cr_col_clear = st.columns([6, 1])
    with _cr_col_btn:
        _cr_scan = st.button(UI_CODE_REVIEW_SCAN_BTN, use_container_width=True)
    with _cr_col_clear:
        if st.button(UI_CODE_REVIEW_CLEAR_BTN, use_container_width=True,
                     disabled=not st.session_state.code_review_context):
            st.session_state.code_review_context  = ""
            st.session_state.code_review_files    = []
            st.session_state.code_review_warnings = []
            st.rerun()

    # Trigger scan only when the scan button is pressed (zip wins if both present)
    _cr_should_scan = _cr_scan and (bool(_cr_zip) or bool(_cr_path))

    if _cr_should_scan:
        import tempfile, zipfile, shutil
        _scan_target: str = ""
        _cr_temp_dir: Optional[str] = None
        try:
            if _cr_zip is not None:
                _cr_temp_dir = tempfile.mkdtemp(prefix="aicouncil_zip_")
                try:
                    with zipfile.ZipFile(_cr_zip, "r") as _zf:
                        _zf.extractall(_cr_temp_dir)
                    # If the zip wraps everything in a single top folder, use it
                    _entries = [e for e in os.listdir(_cr_temp_dir) if not e.startswith(".")]
                    if len(_entries) == 1 and os.path.isdir(os.path.join(_cr_temp_dir, _entries[0])):
                        _scan_target = os.path.join(_cr_temp_dir, _entries[0])
                    else:
                        _scan_target = _cr_temp_dir
                except zipfile.BadZipFile:
                    st.error("❌ קובץ ה-ZIP פגום או לא תקין")
                    _scan_target = ""
            elif _cr_allow_local:
                _scan_target = _cr_path.strip()

            if _scan_target:
                with st.spinner(UI_CODE_REVIEW_SCANNING):
                    # ZIP extraction temp dir is a trusted sandboxed source; a
                    # user-typed local path is not (it stays gated by the env var).
                    _cr_block, _cr_files, _cr_warns = scan_project(
                        _scan_target, trusted_source=(_cr_zip is not None)
                    )
                if _cr_block:
                    st.session_state.code_review_context  = _cr_block
                    st.session_state.code_review_files    = _cr_files
                    st.session_state.code_review_warnings = _cr_warns
                    st.rerun()
                else:
                    _cr_err = _cr_warns[0] if _cr_warns else UI_CODE_REVIEW_EMPTY
                    st.error(_cr_err)
        finally:
            if _cr_temp_dir and os.path.isdir(_cr_temp_dir):
                shutil.rmtree(_cr_temp_dir, ignore_errors=True)

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

# ---------------------------------------------------------------------------
# Academic Mode — optional panel for research / literature-grounded analysis
# ---------------------------------------------------------------------------

with st.expander(UI_ACADEMIC_TOGGLE, expanded=False):
    st.caption(UI_ACADEMIC_CAPTION)
    _academic_on = st.checkbox(
        UI_ACADEMIC_CHECKBOX,
        value=False,
        key=f"academic_mode_{st.session_state._query_counter}",
    )
    if _academic_on:
        _col_yf, _col_yt = st.columns(2)
        with _col_yf:
            _academic_year_from = st.number_input(
                UI_ACADEMIC_YEAR_FROM,
                min_value=2000,
                max_value=_CURRENT_YEAR,
                value=2019,
                step=1,
                key=f"academic_year_from_{st.session_state._query_counter}",
            )
        with _col_yt:
            _academic_year_to = st.number_input(
                UI_ACADEMIC_YEAR_TO,
                min_value=2000,
                max_value=_CURRENT_YEAR,
                value=_CURRENT_YEAR,
                step=1,
                key=f"academic_year_to_{st.session_state._query_counter}",
            )
    else:
        _academic_year_from = None
        _academic_year_to   = None

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

            # title/domain/url come from live web-search results and from
            # user-writable history — escape them before HTML interpolation.
            safe_title  = html.escape(title or "")
            safe_domain = html.escape(domain or "")
            # Only render a clickable link for real http(s) URLs; anything else
            # (javascript:, data:, relative, empty) shows as a card with no link.
            _is_web_url = isinstance(url, str) and url.startswith(("http://", "https://"))
            _link_html  = (
                f'<a href="{html.escape(url, quote=True)}" target="_blank" '
                f'style="font-size:.82em;color:#3b82f6;text-decoration:none;'
                f'font-weight:600">{UI_CITATION_VISIT}</a>'
                if _is_web_url else ""
            )

            with col:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e2e8f0;border-radius:8px;
                        padding:14px 16px;background:#f8fafc;
                        height:100%;min-height:90px;">
                      <div style="font-size:.78em;color:#64748b;font-weight:600;
                          letter-spacing:.03em;text-transform:uppercase;
                          margin-bottom:4px">{safe_domain}</div>
                      <div style="font-size:.9em;color:#1e293b;font-weight:500;
                          margin-bottom:10px;line-height:1.4">{safe_title}</div>
                      {_link_html}
                    </div>""",
                    unsafe_allow_html=True,
                )


def _safe_image(data: bytes, **kwargs) -> bool:
    """
    Display an image safely, catching PIL decode errors.
    Returns True if displayed successfully, False otherwise.
    """
    if not data:
        st.error("התמונה שהתקבלה ריקה — נסה לייצר מחדש.")
        return False
    try:
        st.image(data, **kwargs)
        return True
    except Exception as _img_err:
        st.error(f"לא ניתן להציג את התמונה ({type(_img_err).__name__}). נסה לייצר מחדש.")
        return False


def _safe_video(data: bytes, **kwargs) -> bool:
    """
    Display a video safely, catching decode errors.
    Returns True if displayed successfully, False otherwise.
    """
    if not data:
        st.error("הקליפ שהתקבל ריק — נסה לייצר מחדש.")
        return False
    try:
        st.video(data, **kwargs)
        return True
    except Exception as _vid_err:
        st.error(f"לא ניתן להציג את הקליפ ({type(_vid_err).__name__}). נסה לייצר מחדש.")
        return False


def _render_media_panel(results: dict) -> None:
    """
    Render the AI media-generation panel below the debate results.
    Sections: optional branding upload, image (single/storyboard), video.
    """
    from media_generator import (
        analyse_branding,
        generate_image_prompt,
        generate_image_storyboard,
        generate_video_prompt,
        generate_image,
        generate_video,
        VIDEO_MODELS,
    )

    _final = results.get("final_answer", "")
    if not _final or _final.startswith("ERROR:"):
        return

    qc       = st.session_state._query_counter
    _question = results.get("question", "")

    st.subheader(UI_MEDIA_PANEL_HEADER)
    st.caption(UI_MEDIA_PANEL_CAPTION)

    # ── Optional branding upload (shared between image and video) ─────────────
    _branding_key     = f"media_branding_files_{qc}"
    _branding_res_key = f"media_branding_notes_{qc}"

    _branding_files = st.file_uploader(
        UI_MEDIA_BRANDING_LABEL,
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=_branding_key,
        help=UI_MEDIA_BRANDING_HELP,
    )
    _branding_notes = st.session_state.get(_branding_res_key, "")

    if _branding_files:
        _br_bytes = [f.getvalue() for f in _branding_files[:3]]
        _br_mime  = [f.type or "image/png" for f in _branding_files[:3]]
        # Analyse only if not yet done for this set of files
        _br_sig = str(sorted(f.name for f in _branding_files))
        if st.session_state.get(f"media_branding_sig_{qc}") != _br_sig:
            with st.spinner(UI_MEDIA_BRANDING_ANALYSING):
                _branding_notes = analyse_branding(_br_bytes, _br_mime)
            st.session_state[_branding_res_key]           = _branding_notes
            st.session_state[f"media_branding_sig_{qc}"] = _br_sig
        if _branding_notes:
            st.success(UI_MEDIA_BRANDING_DONE)

    # ── Section 1: Image generation ──────────────────────────────────────────
    with st.expander(UI_MEDIA_IMAGE_EXPANDER, expanded=False):

        _img_ctx_key      = f"media_img_ctx_{qc}"
        _img_size_key     = f"media_img_size_{qc}"
        _img_mode_key     = f"media_img_mode_{qc}"
        _img_count_key    = f"media_img_count_{qc}"
        _img_res_key      = f"media_img_result_{qc}"
        _img_n_key        = f"media_img_n_{qc}"
        _img_style_key    = f"media_img_style_{qc}"
        _img_ref_key      = f"media_img_ref_{qc}"
        _img_ref_res_key  = f"media_img_ref_notes_{qc}"

        col_ctx, col_size = st.columns([3, 2])
        with col_ctx:
            _img_ctx = st.text_area(
                UI_MEDIA_CONTEXT_LABEL,
                placeholder="לדוגמא: אפליקציית ביטחון אישי, קהל יעד נשים 20-35, לאינסטגרם...",
                height=80,
                key=_img_ctx_key,
                help=UI_MEDIA_CONTEXT_HELP,
            )
        with col_size:
            _size_label = st.selectbox(
                UI_MEDIA_SIZE_LABEL,
                list(UI_MEDIA_SIZE_OPTIONS.keys()),
                key=_img_size_key,
            )
            _size_val = UI_MEDIA_SIZE_OPTIONS[_size_label]

        col_mode, col_count = st.columns(2)
        with col_mode:
            _img_mode = st.radio(
                UI_MEDIA_IMAGE_MODE_LABEL,
                [UI_MEDIA_IMAGE_MODE_STORY, UI_MEDIA_IMAGE_MODE_SINGLE],
                key=_img_mode_key,
            )
            _img_storyboard = (_img_mode == UI_MEDIA_IMAGE_MODE_STORY)
        with col_count:
            if _img_storyboard:
                _n_images = st.radio(
                    UI_MEDIA_IMG_STORY_COUNT,
                    [3, 4],
                    index=1,
                    horizontal=True,
                    key=_img_count_key,
                )
            else:
                _n_images = 1

        # ── Style selector ───────────────────────────────────────────────────
        _img_style_choice = st.radio(
            UI_MEDIA_STYLE_LABEL,
            [UI_MEDIA_STYLE_REALISTIC, UI_MEDIA_STYLE_ANIMATION],
            horizontal=True,
            key=_img_style_key,
        )
        _img_style_val = "animation" if _img_style_choice == UI_MEDIA_STYLE_ANIMATION else "realistic"

        # ── Reference images (optional per-section upload) ───────────────────
        _img_ref_files = st.file_uploader(
            UI_MEDIA_REF_IMG_LABEL,
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=_img_ref_key,
            help=UI_MEDIA_REF_IMG_HELP,
        )
        _img_ref_notes = st.session_state.get(_img_ref_res_key, "")
        if _img_ref_files:
            _ref_bytes = [f.getvalue() for f in _img_ref_files[:3]]
            _ref_mime  = [f.type or "image/png" for f in _img_ref_files[:3]]
            _ref_sig   = str(sorted(f.name for f in _img_ref_files))
            if st.session_state.get(f"media_img_ref_sig_{qc}") != _ref_sig:
                with st.spinner(UI_MEDIA_REF_ANALYSING):
                    from media_generator import analyse_reference_images
                    _img_ref_notes = analyse_reference_images(_ref_bytes, _ref_mime)
                st.session_state[_img_ref_res_key]          = _img_ref_notes
                st.session_state[f"media_img_ref_sig_{qc}"] = _ref_sig
            if _img_ref_notes:
                st.success(UI_MEDIA_REF_DONE)

        # ── Cached results ───────────────────────────────────────────────────
        _cached_img = st.session_state.get(_img_res_key)
        _cached_n   = st.session_state.get(_img_n_key, 0)

        if _cached_img:
            if isinstance(_cached_img, list):
                # Storyboard
                st.markdown(f"### {UI_MEDIA_IMG_STORY_HDR}")
                st.caption(UI_MEDIA_IMG_STORY_NOTE)
                cols = st.columns(min(len(_cached_img), 2))
                for i, img_item in enumerate(_cached_img):
                    with cols[i % 2]:
                        st.markdown(f"**{img_item['title']}**")
                        if img_item.get("error"):
                            st.error(img_item["error"])
                        elif img_item.get("bytes"):
                            _safe_image(img_item["bytes"], use_container_width=True)
                            st.download_button(
                                UI_MEDIA_DL_IMG_N.format(n=img_item["image_number"]),
                                data=img_item["bytes"],
                                file_name=f"scene_{img_item['image_number']}.png",
                                mime="image/png",
                                use_container_width=True,
                                key=f"dl_img_s_{qc}_{_cached_n}_{img_item['image_number']}",
                            )
                        if img_item.get("prompt_used"):
                            with st.expander(UI_MEDIA_PROMPT_USED_HDR, expanded=False):
                                st.code(img_item["prompt_used"], language=None)
                if st.button(UI_MEDIA_IMG_REGEN_STORY, key=f"regen_img_story_{qc}_{_cached_n}"):
                    del st.session_state[_img_res_key]
                    st.rerun()
            elif isinstance(_cached_img, dict) and _cached_img.get("bytes"):
                # Single image
                _safe_image(_cached_img["bytes"], use_container_width=True)
                with st.expander(UI_MEDIA_PROMPT_USED_HDR, expanded=False):
                    st.code(_cached_img["prompt"], language=None)
                dl_col, regen_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        UI_MEDIA_DL_IMG_BTN,
                        data=_cached_img["bytes"],
                        file_name="ai_council_marketing.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_img_{qc}_{_cached_img['gen_n']}",
                    )
                with regen_col:
                    if st.button(UI_MEDIA_REGEN_BTN, use_container_width=True,
                                 key=f"regen_img_{qc}_{_cached_img['gen_n']}"):
                        del st.session_state[_img_res_key]
                        st.rerun()

        # ── Generate button ───────────────────────────────────────────────────
        _img_btn_label = (
            f"✨ צור סטוריבורד ({_n_images} תמונות)"
            if _img_storyboard
            else UI_MEDIA_GENERATE_IMG_BTN
        )
        if st.button(_img_btn_label, type="primary", use_container_width=True,
                     key=f"gen_img_btn_{qc}"):
            _n = st.session_state.get(_img_n_key, 0) + 1
            st.session_state[_img_n_key] = _n

            if _img_storyboard:
                _status = st.status(UI_MEDIA_STEP_IMG_STORY, expanded=True)
                with _status:
                    scenes = generate_image_storyboard(
                        final_answer=_final,
                        question=_question,
                        campaign_context=_img_ctx or "",
                        n_images=_n_images,
                        size=_size_val,
                        branding_notes=_branding_notes,
                        ref_description=_img_ref_notes,
                        style=_img_style_val,
                    )
                    any_ok = any(not s.get("error") for s in scenes)
                    _status.update(
                        label="הושלם!" if any_ok else "שגיאה בחלק מהתמונות",
                        state="complete" if any_ok else "error",
                        expanded=False,
                    )
                st.session_state[_img_res_key] = scenes
                st.rerun()
            else:
                _status = st.status(UI_MEDIA_STEP_PROMPT, expanded=True)
                with _status:
                    img_prompt = generate_image_prompt(
                        _final, _img_ctx or "", ref_description=_img_ref_notes,
                    )
                    if _branding_notes:
                        img_prompt = f"[Brand guidelines: {_branding_notes}] " + img_prompt
                    st.write(UI_MEDIA_STEP_IMAGE)
                    result = generate_image(img_prompt, size=_size_val, style=_img_style_val)
                    if result["error"]:
                        _status.update(
                            label=f"{UI_MEDIA_ERROR_PREFIX}: {result['error']}",
                            state="error", expanded=True,
                        )
                        st.error(result["error"])
                    else:
                        _status.update(label="הושלם!", state="complete", expanded=False)
                        st.session_state[_img_res_key] = {
                            "bytes":  result["bytes"],
                            "prompt": result["prompt_used"],
                            "gen_n":  _n,
                        }
                        st.rerun()

    # ── Section 2: Video generation ──────────────────────────────────────────
    with st.expander(UI_MEDIA_VIDEO_EXPANDER, expanded=False):

        _vid_ctx_key      = f"media_vid_ctx_{qc}"
        _vid_model_key    = f"media_vid_model_{qc}"
        _vid_ar_key       = f"media_vid_ar_{qc}"
        _vid_mode_key     = f"media_vid_mode_{qc}"
        _vid_res_key      = f"media_vid_result_{qc}"
        _vid_n_key        = f"media_vid_n_{qc}"
        _vid_style_key    = f"media_vid_style_{qc}"
        _vid_ref_key      = f"media_vid_ref_{qc}"
        _vid_ref_res_key  = f"media_vid_ref_notes_{qc}"

        _replicate_ok = bool(os.environ.get("REPLICATE_API_TOKEN"))
        if not _replicate_ok:
            st.warning(UI_MEDIA_NO_REPLICATE_KEY)

        _vid_ctx = st.text_area(
            UI_MEDIA_CONTEXT_LABEL,
            placeholder=(
                "לדוגמא: בחורה צעירה בדייט, מרגישה לא בנוח, אומרת את מילת הקוד שלה "
                "בשקט ואיש קשר בא לעזרתה..."
            ),
            height=90,
            key=_vid_ctx_key,
            help=UI_MEDIA_CONTEXT_HELP,
        )

        col_mode, col_model, col_ar = st.columns(3)
        with col_mode:
            _vid_mode = st.radio(
                UI_MEDIA_VIDEO_MODE_LABEL,
                [UI_MEDIA_VIDEO_MODE_STORY, UI_MEDIA_VIDEO_MODE_SINGLE],
                key=_vid_mode_key,
            )
            _storyboard_mode = (_vid_mode == UI_MEDIA_VIDEO_MODE_STORY)
        with col_model:
            _vid_model_options = {cfg["label"]: key for key, cfg in VIDEO_MODELS.items()}
            _vid_model_label = st.selectbox(
                UI_MEDIA_VIDEO_MODEL_LABEL,
                list(_vid_model_options.keys()),
                key=_vid_model_key,
            )
            _vid_model_val = _vid_model_options[_vid_model_label]
        with col_ar:
            _ar_label = st.selectbox(
                UI_MEDIA_SIZE_LABEL,
                list(UI_MEDIA_SIZE_OPTIONS.keys()),
                key=_vid_ar_key,
            )
            _ar_val = UI_MEDIA_SIZE_OPTIONS[_ar_label]

        # ── Style selector ───────────────────────────────────────────────────
        _vid_style_choice = st.radio(
            UI_MEDIA_STYLE_LABEL,
            [UI_MEDIA_STYLE_REALISTIC, UI_MEDIA_STYLE_ANIMATION],
            horizontal=True,
            key=_vid_style_key,
        )
        _vid_style_val = "animation" if _vid_style_choice == UI_MEDIA_STYLE_ANIMATION else "realistic"

        # ── Reference images (optional per-section upload) ───────────────────
        _vid_ref_files = st.file_uploader(
            UI_MEDIA_REF_IMG_LABEL,
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=_vid_ref_key,
            help=UI_MEDIA_REF_IMG_HELP,
        )
        _vid_ref_notes = st.session_state.get(_vid_ref_res_key, "")
        if _vid_ref_files:
            _ref_bytes = [f.getvalue() for f in _vid_ref_files[:3]]
            _ref_mime  = [f.type or "image/png" for f in _vid_ref_files[:3]]
            _ref_sig   = str(sorted(f.name for f in _vid_ref_files))
            if st.session_state.get(f"media_vid_ref_sig_{qc}") != _ref_sig:
                with st.spinner(UI_MEDIA_REF_ANALYSING):
                    from media_generator import analyse_reference_images
                    _vid_ref_notes = analyse_reference_images(_ref_bytes, _ref_mime)
                st.session_state[_vid_ref_res_key]          = _vid_ref_notes
                st.session_state[f"media_vid_ref_sig_{qc}"] = _ref_sig
            if _vid_ref_notes:
                st.success(UI_MEDIA_REF_DONE)

        st.caption(
            "ℹ️ מודלי ווידאו AI אינם מייצרים סאונד. "
            "כל קליפ הוא סצינה אחת רציפה ללא קאטים."
        )
        if _storyboard_mode:
            st.caption(
                "Claude יכתוב תסריט 3 סצינות עצמאיות (45 מילים לכל אחת). "
                "הקליפים יוצרו ברצף — **משך כולל: ~15-18 שניות.**"
            )

        # ── Display cached results ────────────────────────────────────────────
        _cached = st.session_state.get(_vid_res_key)
        if _cached:
            if isinstance(_cached, list):
                # Storyboard result
                st.markdown(f"### {UI_MEDIA_STORYBOARD_HDR}")
                st.caption(UI_MEDIA_STORYBOARD_NOTE)
                for scene in _cached:
                    _s_title = scene.get("title", f"סצינה {scene.get('scene_number','')}")
                    _s_bytes = scene.get("bytes")
                    _s_err   = scene.get("error")
                    _s_prom  = scene.get("prompt_used", "")
                    _s_n     = scene.get("scene_number", 0)
                    st.markdown(f"**{_s_title}**")
                    if _s_err:
                        st.error(f"{UI_MEDIA_ERROR_PREFIX}: {_s_err}")
                    elif _s_bytes:
                        _safe_video(_s_bytes)
                        dl_col2, _ = st.columns([1, 1])
                        with dl_col2:
                            _gen_n = st.session_state.get(_vid_n_key, 1)
                            st.download_button(
                                UI_MEDIA_DL_VID_BTN,
                                data=_s_bytes,
                                file_name=f"scene_{_s_n}.mp4",
                                mime="video/mp4",
                                use_container_width=True,
                                key=f"dl_scene_{qc}_{_gen_n}_{_s_n}",
                            )
                    if _s_prom:
                        with st.expander(UI_MEDIA_PROMPT_USED_HDR, expanded=False):
                            st.code(_s_prom, language=None)
            elif isinstance(_cached, dict) and _cached.get("bytes"):
                # Single clip result
                _safe_video(_cached["bytes"])
                with st.expander(UI_MEDIA_PROMPT_USED_HDR, expanded=False):
                    st.code(_cached["prompt"], language=None)
                dl_col, regen_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        UI_MEDIA_DL_VID_BTN,
                        data=_cached["bytes"],
                        file_name="ai_council_campaign.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"dl_vid_{qc}_{_cached['gen_n']}",
                    )
                with regen_col:
                    if st.button(
                        UI_MEDIA_REGEN_BTN,
                        use_container_width=True,
                        key=f"regen_vid_{qc}_{_cached['gen_n']}",
                    ):
                        del st.session_state[_vid_res_key]
                        st.rerun()

        # ── Generate button ───────────────────────────────────────────────────
        _btn_label = UI_MEDIA_GENERATE_VID_BTN if not _storyboard_mode else "🎬 צור סטוריבורד (3 קליפים)"
        if st.button(
            _btn_label,
            type="primary",
            use_container_width=True,
            disabled=not _replicate_ok,
            key=f"gen_vid_btn_{qc}",
        ):
            from media_generator import generate_video_storyboard

            _n = st.session_state.get(_vid_n_key, 0) + 1
            st.session_state[_vid_n_key] = _n
            _question = results.get("question", "")

            if _storyboard_mode:
                _status = st.status(UI_MEDIA_STEP_STORYBOARD, expanded=True)
                with _status:
                    scenes = generate_video_storyboard(
                        final_answer=_final,
                        question=_question,
                        campaign_context=_vid_ctx or "",
                        model_key=_vid_model_val,
                        aspect_ratio=_ar_val,
                        ref_description=_vid_ref_notes,
                        style=_vid_style_val,
                        progress_cb=lambda n, total, status: st.write(
                            UI_MEDIA_SCENE_GENERATING.format(n=n)
                        ),
                    )
                    any_ok = any(not s.get("error") for s in scenes)
                    _status.update(
                        label="הושלם!" if any_ok else "שגיאה בחלק מהקליפים",
                        state="complete" if any_ok else "error",
                        expanded=False,
                    )
                st.session_state[_vid_res_key] = scenes
                st.rerun()
            else:
                _status = st.status(UI_MEDIA_STEP_PROMPT, expanded=True)
                with _status:
                    vid_prompt = generate_video_prompt(
                        _final, _vid_ctx or "", question=_question,
                        ref_description=_vid_ref_notes,
                    )
                    st.write(UI_MEDIA_STEP_VIDEO)
                    result = generate_video(
                        vid_prompt, model_key=_vid_model_val, aspect_ratio=_ar_val,
                        style=_vid_style_val,
                    )
                    if result["error"]:
                        _status.update(
                            label=f"{UI_MEDIA_ERROR_PREFIX}: {result['error']}",
                            state="error", expanded=True,
                        )
                        st.error(result["error"])
                    else:
                        _status.update(label="הושלם!", state="complete", expanded=False)
                        st.session_state[_vid_res_key] = {
                            "bytes":  result["bytes"],
                            "url":    result["url"],
                            "prompt": result["prompt_used"],
                            "gen_n":  _n,
                        }
                        st.rerun()


def _render_cultural_panel(results: dict) -> None:
    """
    Cultural Campaign Generator — creates culturally-adapted images or video
    clips for multiple markets in parallel, each grounded in the council's
    final answer and tailored to local visual norms and sensitivities.
    """
    from media_generator import generate_cultural_variants, VIDEO_MODELS

    _final = results.get("final_answer", "")
    if not _final or _final.startswith("ERROR:"):
        return

    qc = st.session_state._query_counter

    with st.expander(UI_CULTURAL_EXPANDER, expanded=False):
        st.caption(UI_CULTURAL_CAPTION)

        # ── Controls row ─────────────────────────────────────────────────────
        col_left, col_right = st.columns([2, 1])
        with col_left:
            _selected_markets = st.multiselect(
                UI_CULTURAL_MARKET_SELECT,
                options=list(CULTURAL_PROFILES.keys()),
                default=[],
                key=f"cultural_markets_{qc}",
            )
        with col_right:
            _media_type_label = st.radio(
                UI_CULTURAL_MEDIA_TYPE,
                [UI_CULTURAL_MEDIA_IMAGE, UI_CULTURAL_MEDIA_VIDEO],
                key=f"cultural_media_type_{qc}",
            )
            _media_type = "image" if _media_type_label == UI_CULTURAL_MEDIA_IMAGE else "video"

        col_size, col_vid_model = st.columns(2)
        with col_size:
            _size_label = st.selectbox(
                UI_CULTURAL_SIZE_LABEL,
                list(UI_MEDIA_SIZE_OPTIONS.keys()),
                key=f"cultural_size_{qc}",
            )
            _size_val = UI_MEDIA_SIZE_OPTIONS[_size_label]
        with col_vid_model:
            if _media_type == "video":
                _vid_model_options = {cfg["label"]: key for key, cfg in VIDEO_MODELS.items()}
                _vid_model_label = st.selectbox(
                    UI_MEDIA_VIDEO_MODEL_LABEL,
                    list(_vid_model_options.keys()),
                    key=f"cultural_vid_model_{qc}",
                )
                _vid_model_val = _vid_model_options[_vid_model_label]
            else:
                _vid_model_val = "minimax"

        _campaign_ctx = st.text_area(
            UI_CULTURAL_CTX_LABEL,
            placeholder=UI_CULTURAL_CTX_PLACEHOLDER,
            height=80,
            key=f"cultural_ctx_{qc}",
        )

        _replicate_ok = bool(os.environ.get("REPLICATE_API_TOKEN"))
        if _media_type == "video" and not _replicate_ok:
            st.warning(UI_CULTURAL_NO_VIDEO_KEY)

        _can_generate = (
            len(_selected_markets) >= 1
            and (_media_type == "image" or _replicate_ok)
        )

        if not _selected_markets:
            st.caption(f"_{UI_CULTURAL_MIN_MARKETS}_")

        if st.button(
            UI_CULTURAL_GENERATE_BTN,
            type="primary",
            use_container_width=True,
            disabled=not _can_generate,
            key=f"cultural_gen_btn_{qc}",
        ):
            _status = st.status(UI_CULTURAL_GENERATING, expanded=True)
            with _status:
                for mkt in _selected_markets:
                    flag = CULTURAL_PROFILES[mkt].get("flag", "")
                    st.write(f"{flag} {mkt}...")

                variants = generate_cultural_variants(
                    final_answer=_final,
                    selected_markets=_selected_markets,
                    media_type=_media_type,
                    size=_size_val,
                    campaign_context=_campaign_ctx or "",
                    video_model_key=_vid_model_val,
                    question=results.get("question", ""),
                )
                _status.update(label="הושלם!", state="complete", expanded=False)

            # Cache results in session state
            _n = st.session_state.get(f"cultural_n_{qc}", 0) + 1
            st.session_state[f"cultural_n_{qc}"]       = _n
            st.session_state[f"cultural_results_{qc}_{_n}"] = variants
            st.rerun()

        # ── Display cached results ────────────────────────────────────────────
        _gen_n = st.session_state.get(f"cultural_n_{qc}", 0)
        _cached_variants = st.session_state.get(f"cultural_results_{qc}_{_gen_n}", [])

        if _cached_variants:
            st.markdown(f"### {UI_CULTURAL_RESULT_HDR}")
            # Show results in a responsive grid (max 2 columns)
            _cols_per_row = min(len(_cached_variants), 2)
            for i in range(0, len(_cached_variants), _cols_per_row):
                row_variants = _cached_variants[i : i + _cols_per_row]
                cols = st.columns(len(row_variants))
                for col, variant in zip(cols, row_variants):
                    with col:
                        _mkt   = variant["market"]
                        _flag  = variant.get("flag", "")
                        _err   = variant.get("error")
                        _bytes = variant.get("bytes")
                        _prom  = variant.get("prompt_used", "")

                        st.markdown(f"**{_flag} {_mkt}**")

                        if _err:
                            st.error(f"{UI_MEDIA_ERROR_PREFIX}: {_err}")
                        elif _bytes:
                            if _media_type == "image":
                                _safe_image(_bytes, use_container_width=True)
                                st.download_button(
                                    UI_CULTURAL_DL_IMG,
                                    data=_bytes,
                                    file_name=f"campaign_{CULTURAL_PROFILES.get(_mkt, {}).get('key', 'market')}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key=f"dl_cultural_img_{qc}_{_gen_n}_{_mkt}",
                                )
                            else:
                                _safe_video(_bytes)
                                st.download_button(
                                    UI_CULTURAL_DL_VID,
                                    data=_bytes,
                                    file_name=f"campaign_{CULTURAL_PROFILES.get(_mkt, {}).get('key', 'market')}.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key=f"dl_cultural_vid_{qc}_{_gen_n}_{_mkt}",
                                )
                        if _prom:
                            with st.expander(UI_CULTURAL_PROMPT_HDR, expanded=False):
                                st.code(_prom, language=None)


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

    # ── Hard failure: the council produced no usable answer ───────────────────
    # (debate_failed → every Stage-1 model failed, or Stage-4 synthesis collapsed)
    # .get() keeps old history entries — which lack these keys — rendering normally.
    if results.get("debate_failed"):
        st.error("❌ מועצת ה-AI לא הצליחה להפיק תשובה סופית. נסה שוב מאוחר יותר.")
        _failed_hard = results.get("failed_models", {})
        if _failed_hard:
            _hard_lines = [
                f"- **{MODELS.get(_mk, {}).get('label', _mk)}**: {str(_err)[:120]}"
                for _mk, _err in _failed_hard.items()
            ]
            st.markdown("**מודלים שנכשלו:**\n" + "\n".join(_hard_lines))
        _stage_errs = results.get("stage_errors", [])
        if _stage_errs:
            st.markdown("**שגיאות בשלבים:**\n" + "\n".join(f"- {_s}" for _s in _stage_errs))
        return   # do NOT render answer sections or generate a PDF

    # ── Partial failure: some models dropped out but a final answer exists ─────
    _failed_partial = results.get("failed_models", {})
    if _failed_partial:
        _part_lines = [
            f"- {MODELS.get(_mk, {}).get('label', _mk)}: {str(_err)[:120]}"
            for _mk, _err in _failed_partial.items()
        ]
        st.warning(
            "⚠️ חלק מהמודלים נכשלו והדיון רץ בלעדיהם:\n\n" + "\n".join(_part_lines)
        )

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
    if not final:
        st.warning("⚠️ הסינתזה הושלמה אך לא הוחזרה תשובה. נסה שוב.")
        return
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
        _safe_cache = safe_pdf_cache_path(results.get("pdf_cache_path", ""))
        pdf_bytes: Optional[bytes] = None
        if _safe_cache is not None:
            try:
                pdf_bytes = _safe_cache.read_bytes()
            except Exception:
                pdf_bytes = None   # cached file unreadable — fall back to regen
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
_fup_question   = st.session_state.pop("_pending_followup", None)
_fup_ctx        = st.session_state.pop("_followup_ctx", None)
_fup_imgs_carry = st.session_state.pop("_pending_followup_imgs", [])
_fup_mime_carry = st.session_state.pop("_pending_followup_mime", [])
_fup_data_carry = st.session_state.pop("_pending_followup_data", "")
_fup_academic   = st.session_state.pop("_pending_followup_academic", False)

if start_button or _fup_question is not None:
    # Resolve the effective question and context for this run
    _active_question    = _fup_question if _fup_question is not None else _effective_question
    _previous_ctx       = _fup_ctx      # None for regular questions
    _active_images      = _fup_imgs_carry if _fup_question is not None else images_bytes
    _active_images_mime = _fup_mime_carry if _fup_question is not None else images_mime
    _active_code_ctx    = _fup_data_carry if _fup_question is not None else _combined_extra_context

    if not _active_question.strip() and not _active_images and not _active_code_ctx:
        st.warning("Please provide a question, upload an image, or scan a project folder.")
        st.stop()

    all_active_keys = list(dict.fromkeys([MASTER_MODEL_KEY] + selected_keys))
    if len(all_active_keys) < 2:
        st.warning(UI_WARNING_MIN_MODELS)
        st.stop()

    # ── Cache-hit check: skip for follow-ups and Academic Mode (always fresh) ──
    # For follow-ups: carry academic mode from the previous debate (_fup_academic)
    # to avoid the widget-key conflict (Streamlit forbids writing to a widget key).
    _academic_on_now = (
        _fup_academic if _fup_question is not None
        else st.session_state.get(f"academic_mode_{st.session_state._query_counter}", False)
    )
    # Skip the cache when files are attached (data files OR a code-review
    # context block): the cache matches on question text only, so the default
    # question would otherwise replay a previous file's results for a new file.
    if _active_question.strip() and not _active_images and not st.session_state._force_rerun \
            and _fup_question is None and not _academic_on_now \
            and not _active_code_ctx and not st.session_state.code_review_context:
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
    # READ-ONLY gate: we only check whether the user is under the daily cap here.
    # The counter is incremented (charged) AFTER a successful debate, so an abort
    # or crash before/within the debate never consumes the user's quota.
    _uid_limit = (st.session_state.user or {}).get("uid")
    if _uid_limit:
        _allowed, _current_count = is_within_usage_limit(_uid_limit)
        # Refresh the cached count so the sidebar reflects today's value
        st.session_state._cached_usage = (_uid_limit, _current_count)
        if not _allowed:
            st.warning(UI_QUOTA_REACHED)
            st.stop()

    # ── Inject CSS (spinner ring + mission control dashboard) ────────────────
    st.markdown(_SPINNER_CSS + MISSION_CONTROL_CSS, unsafe_allow_html=True)

    _n_imgs = len(_active_images)
    _main_label = (
        f"{'🔄' if _fup_question is not None else '👁️'} "
        f"{'Follow-up' if _fup_question is not None else 'Vision'} Council"
        f" — analysing {_n_imgs} image{'s' if _n_imgs > 1 else ''} …"
        if _active_images
        else ("🔄 AI Council — Follow-up Debate …" if _fup_question is not None
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

        # ── Apply user-supplied API keys (BYOK) for this session ─────────────
        set_session_api_keys(st.session_state.user_api_keys)

        # ── Run the full DSAD debate ──────────────────────────────────────────
        # Wrapped so an unexpected crash (or a user abort) never charges quota
        # and never saves a broken entry to history — the gate above was
        # read-only, so nothing has been consumed yet.
        try:
            _live_results = run_council_debate(
                active_keys=selected_keys,
                question=_active_question,
                images=_active_images,
                images_mime=_active_images_mime,
                previous_context=_previous_ctx,
                code_context=_active_code_ctx,
                academic_mode=_academic_on_now,
                year_from=st.session_state.get(
                    f"academic_year_from_{st.session_state._query_counter}"
                ),
                year_to=st.session_state.get(
                    f"academic_year_to_{st.session_state._query_counter}"
                ),
                stage0_cb=stage0_cb,
                stage1_cb=stage1_cb,
                stage2_cb=stage2_cb,
                stage3_cb=stage3_cb,
                stage3b_cb=stage3b_cb,
                stage4_cb=stage4_cb,
            )
        except Exception as _debate_exc:   # noqa: BLE001
            status.update(
                label="❌ מועצת ה-AI נכשלה — אירעה שגיאה בלתי צפויה",
                state="error",
            )
            st.error(
                f"שגיאה בלתי צפויה הפסיקה את הדיון: {type(_debate_exc).__name__}. "
                f"לא חויבה מכסה — נסה שוב."
            )
            st.stop()

        # ── Final state: reflect success vs. hard failure ─────────────────────
        _debate_failed = bool(_live_results.get("debate_failed"))
        if _debate_failed:
            status.update(
                label="❌ מועצת ה-AI לא הצליחה להפיק תשובה",
                state="error",
                expanded=False,
            )
        else:
            _state["completed"] = {0, 1, 2, 3, 4, 5}
            _state["stage"]     = -1
            _refresh_ui()
            status.update(
                label="✅ Council deliberation complete — results ready!",
                state="complete",
                expanded=False,
            )

    # ── Charge quota only now that the debate actually produced results ───────
    # The pre-debate gate was READ-ONLY; the counter is incremented here so a
    # crash, abort, or all-models-failure never consumes the user's daily quota.
    _charge_uid = (st.session_state.user or {}).get("uid")
    if _charge_uid and not _debate_failed:
        _charged_count = increment_usage(_charge_uid)
        st.session_state._cached_usage = (_charge_uid, _charged_count)

    # ── Persist and re-render from session state ──────────────────────────────
    # A fully failed debate is NEVER saved and no PDF is cached. Partial
    # failures that still produced a valid final answer ARE saved as usual.
    if not _debate_failed:
        # Generate PDF once here so save_to_history can cache it to disk.
        # The cached path is stored so _display_results() serves the file
        # directly without regenerating on the first (live-run) display.
        _pdf_for_cache: Optional[bytes] = None
        try:
            _pdf_for_cache = generate_pdf(_live_results)
        except Exception:
            pass

        _current_uid  = (st.session_state.user or {}).get("uid")
        try:
            _cached_path  = save_to_history(_live_results, pdf_bytes=_pdf_for_cache, uid=_current_uid)
            if _cached_path:
                _live_results["pdf_cache_path"] = _cached_path
        except Exception:
            pass

    st.session_state.current_results = _live_results
    st.session_state.from_history    = False
    st.rerun()


# ---------------------------------------------------------------------------
# Results display — driven entirely by session state so it works
# identically for live debates and history replays
# ---------------------------------------------------------------------------
if st.session_state.current_results:
    _display_results(st.session_state.current_results)

    # ── Academic papers panel (shown only when academic_mode was active) ───
    _acad_papers = st.session_state.current_results.get("academic_papers", [])
    if _acad_papers:
        with st.expander(f"📚 מאמרים אקדמיים שנמצאו ({len(_acad_papers)})", expanded=False):
            for i, p in enumerate(_acad_papers, 1):
                _title   = p.get("title", "")
                _authors = p.get("authors", "")
                _year    = p.get("year", "")
                _journal = p.get("journal", "")
                _url     = p.get("url", "")
                _abs     = p.get("abstract", "")
                _cit     = p.get("citations")
                _src     = p.get("source", "")
                _hdr = f"**[{i}] {_title}**"
                st.markdown(_hdr)
                _meta = f"_{_authors} ({_year})"
                if _journal:
                    _meta += f". {_journal}"
                if _cit is not None:
                    _meta += f". Cited by {_cit}"
                _meta += f" — {_src}_"
                st.markdown(_meta)
                if _url:
                    st.markdown(f"🔗 [{_url}]({_url})")
                if _abs:
                    st.caption(_abs[:400] + ("…" if len(_abs) > 400 else ""))
                st.divider()

    # ── Media generation panel ─────────────────────────────────────────────
    st.divider()
    _render_media_panel(st.session_state.current_results)
    _render_cultural_panel(st.session_state.current_results)

    # ── Follow-up question panel ───────────────────────────────────────────
    st.divider()
    _fup_key      = f"followup_{st.session_state._query_counter}"
    _fup_img_key  = f"followup_img_{st.session_state._query_counter}"
    _fup_data_key = f"followup_data_{st.session_state._query_counter}"

    _fup_q = st.text_area(
        label=UI_FOLLOWUP_LABEL,
        placeholder=UI_FOLLOWUP_PLACEHOLDER,
        height=90,
        key=_fup_key,
    )

    # Image uploader
    _fup_img_files = st.file_uploader(
        UI_FOLLOWUP_IMG_LABEL,
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=_fup_img_key,
    )
    _fup_imgs:      list = []
    _fup_imgs_mime: list = []
    if _fup_img_files:
        for _ff in _fup_img_files[:4]:
            _fup_imgs.append(_ff.getvalue())
            _fup_imgs_mime.append(_ff.type or "image/jpeg")
        _fup_img_cols = st.columns(min(len(_fup_img_files), 4))
        for _fc, _ff in zip(_fup_img_cols, _fup_img_files[:4]):
            with _fc:
                st.image(_ff.getvalue(), use_container_width=True)
                st.caption(f"{_ff.name} · {len(_ff.getvalue()) // 1024} KB")

    # Data file uploader (CSV, Excel, PDF, JSON, TXT, DOCX…)
    _fup_data_files = st.file_uploader(
        UI_FOLLOWUP_DATA_LABEL,
        type=FILE_UPLOAD_ACCEPTED_TYPES,
        accept_multiple_files=True,
        key=_fup_data_key,
        help=f"עד {FILE_UPLOAD_MAX_SIZE_MB} MB לקובץ. התוכן יוזרק לכל המודלים כהקשר.",
    )
    _fup_data_ctx = ""
    if _fup_data_files:
        _fup_pairs = [(f.name, f.getvalue()) for f in _fup_data_files]
        _fup_data_ctx, _fup_data_warns = build_file_context(_fup_pairs)
        if _fup_data_ctx:
            st.success(UI_FOLLOWUP_DATA_LOADED.format(
                count=len(_fup_data_files), chars=len(_fup_data_ctx),
            ))
            if _fup_data_warns:
                st.caption(f"⚠️ {len(_fup_data_warns)} הערות טעינה")

    _fup_can_submit = (
        bool((_fup_q or "").strip()) or bool(_fup_imgs) or bool(_fup_data_ctx)
    )
    if st.button(UI_FOLLOWUP_SUBMIT_BTN, type="primary", disabled=not _fup_can_submit):
        st.session_state["_pending_followup"]          = (_fup_q or "").strip()
        st.session_state["_followup_ctx"]              = st.session_state.current_results
        st.session_state["_pending_followup_imgs"]     = _fup_imgs
        st.session_state["_pending_followup_mime"]     = _fup_imgs_mime
        st.session_state["_pending_followup_data"]     = _fup_data_ctx
        # Carry academic mode forward so the follow-up uses the same synthesis settings
        st.session_state["_pending_followup_academic"] = st.session_state.get(
            f"academic_mode_{st.session_state._query_counter}", False
        )
        st.session_state.current_results               = None
        st.session_state.from_history                  = False
        st.session_state._query_counter               += 1
        st.rerun()
