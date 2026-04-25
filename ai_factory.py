"""
ai_factory.py — Provider-Agnostic API Factory
===============================================
The factory pattern means the rest of the app never needs to know
*which* SDK is being called — it just asks for a model by key and gets
a plain-text response back.

Adding a new provider:
  1.  Add the model entry to glossary.MODELS with the correct "provider" value.
  2.  Add a matching `elif provider == "your_provider":` block in `call_model`.
  3.  That's it.  No other file needs to change.

Vision / multimodal
--------------------
All three providers support vision.  Pass `image_bytes` (raw file bytes) and
`image_mime` (e.g. "image/jpeg") to any call function.  When omitted the call
is identical to the text-only path.
"""

import base64
import os
from contextvars import ContextVar
from typing import List, Optional

import anthropic
import openai
from google import genai
from google.genai import types as genai_types

from glossary import API_TIMEOUT_SECONDS, MODELS

# ---------------------------------------------------------------------------
# Per-session API key override
# ---------------------------------------------------------------------------
# Stores user-supplied API keys scoped to the current execution context
# (thread-safe via ContextVar — each Streamlit session runs in its own thread,
# and copy_context() in logic_engine propagates this to child threads).
#
# Keys map provider name → key string, e.g.:
#   {"anthropic": "sk-ant-...", "openai": "sk-...", "google": "AIza...", "xai": "xai-..."}
#
_SESSION_KEYS: ContextVar[dict] = ContextVar("session_api_keys", default={})


def set_session_api_keys(keys: dict) -> None:
    """
    Set user-supplied API keys for the current session context.
    Call this once before run_council_debate(); the keys are inherited
    automatically by all child threads via copy_context().

    Parameters
    ----------
    keys : dict
        Mapping of provider name to API key string.
        Accepted keys: "anthropic", "openai", "google", "xai".
        Empty string or missing entry → falls back to os.environ.
    """
    _SESSION_KEYS.set({k: v for k, v in keys.items() if v and v.strip()})


def _key(env_var: str, provider: str) -> str:
    """Return the API key: session override first, then os.environ."""
    session = _SESSION_KEYS.get()
    if provider in session:
        return session[provider]
    return os.environ[env_var]


# ---------------------------------------------------------------------------
# Iron Rule — Gemini Field Agent system instruction
# Forces live-search grounding for every numeric or time-sensitive claim.
# ---------------------------------------------------------------------------
_GEMINI_FIELD_AGENT_INSTRUCTION = (
    "You are a precise data processor. "
    "When a VERIFIED CONTEXT BLOCK is present in the user message, you MUST treat "
    "its figures as ground truth for all calculations and factual claims — they "
    "come from a live pre-flight search and override your training memory. "
    "Do NOT substitute training-memory values for anything listed in the block. "
    "For facts not covered by the block, apply standard Chain-of-Thought reasoning "
    "and flag any time-sensitive claims as potentially outdated."
)


# ---------------------------------------------------------------------------
# Internal helpers — one per provider
# ---------------------------------------------------------------------------

def _call_anthropic(
    model_id: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    max_tokens: int = 4096,
    request_timeout: Optional[float] = None,
) -> str:
    """Call the Anthropic (Claude) API and return the text reply."""
    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY", "anthropic"), timeout=API_TIMEOUT_SECONDS)
    images      = images      or []
    images_mime = images_mime or []

    if images:
        content = []
        for img_bytes, img_mime in zip(images, images_mime):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img_mime,
                    "data": base64.standard_b64encode(img_bytes).decode("utf-8"),
                },
            })
        content.append({"type": "text", "text": prompt})
    else:
        content = prompt

    create_kwargs: dict = dict(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    if request_timeout is not None:
        create_kwargs["timeout"] = request_timeout

    message = client.messages.create(**create_kwargs)
    return message.content[0].text


def _call_openai(
    model_id: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    max_tokens: int = 4096,
) -> str:
    """Call the OpenAI API and return the text reply."""
    client = openai.OpenAI(api_key=_key("OPENAI_API_KEY", "openai"), timeout=API_TIMEOUT_SECONDS)
    images      = images      or []
    images_mime = images_mime or []

    if images:
        content = [{"type": "text", "text": prompt}]
        for img_bytes, img_mime in zip(images, images_mime):
            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img_mime};base64,{b64}"},
            })
    else:
        content = prompt

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_google(
    model_id: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    max_tokens: int = 4096,
) -> tuple[str, list[dict]]:
    """
    Call the Google Generative AI (Gemini) API.

    Returns
    -------
    tuple[str, list[dict]]
        (answer_text, citations) — citations always empty (from search_engine.py).
    """
    client = genai.Client(
        api_key=_key("GOOGLE_API_KEY", "google"),
        http_options=genai_types.HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000),
    )
    images      = images      or []
    images_mime = images_mime or []

    cfg = genai_types.GenerateContentConfig(
        system_instruction=_GEMINI_FIELD_AGENT_INSTRUCTION,
        max_output_tokens=max_tokens,
    )

    if images:
        contents = [
            genai_types.Part.from_bytes(data=img_bytes, mime_type=img_mime)
            for img_bytes, img_mime in zip(images, images_mime)
        ]
        contents.append(prompt)
    else:
        contents = prompt

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=cfg,
    )
    return response.text, []


def _call_xai(
    model_id: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    max_tokens: int = 4096,
) -> str:
    """Call the xAI (Grok) API using the OpenAI-compatible client."""
    client = openai.OpenAI(
        api_key=_key("XAI_API_KEY", "xai"),
        base_url="https://api.x.ai/v1",
        timeout=API_TIMEOUT_SECONDS,
    )
    images      = images      or []
    images_mime = images_mime or []

    if images:
        content = [{"type": "text", "text": prompt}]
        for img_bytes, img_mime in zip(images, images_mime):
            b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img_mime};base64,{b64}"},
            })
    else:
        content = prompt

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# API key validation — lightweight test call per provider
# ---------------------------------------------------------------------------

def validate_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Make a minimal test call to verify an API key is authentic.

    Returns
    -------
    tuple[bool, str]
        (is_valid, message)
        is_valid=True even for quota/rate-limit errors — the KEY is valid,
        the user just ran out of credits.  Callers should save the key but
        display the quota warning.
    """
    try:
        if provider == "anthropic":
            c = anthropic.Anthropic(api_key=api_key, timeout=10)
            c.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )

        elif provider == "openai":
            c = openai.OpenAI(api_key=api_key, timeout=10)
            c.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )

        elif provider == "google":
            c = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(timeout=10_000),
            )
            c.models.generate_content(
                model="gemini-1.5-flash",
                contents="hi",
                config=genai_types.GenerateContentConfig(max_output_tokens=1),
            )

        elif provider == "xai":
            c = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                timeout=10,
            )
            c.chat.completions.create(
                model="grok-3",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )

        else:
            return False, f"ספק לא מוכר: {provider}"

        return True, ""

    except Exception as exc:
        s = str(exc).lower()
        if any(w in s for w in ("401", "invalid api key", "invalid x-api-key",
                                "authentication", "unauthorized", "incorrect api key")):
            return False, "מפתח לא תקין — בדוק שהועתק במלואו"
        if any(w in s for w in ("403", "permission", "forbidden", "access denied")):
            return False, "אין הרשאה — בדוק שלמפתח יש גישה ל-API"
        if any(w in s for w in ("429", "rate limit", "quota", "exceeded", "billing")):
            # Key is real — user just ran out of credits
            return True, "מכסת ה-API הסתיימה — המפתח תקין אך המכסה מוצתה"
        return False, f"שגיאה: {type(exc).__name__}: {exc}"


def call_model_with_citations(
    model_key: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    max_tokens: int = 4096,
    request_timeout: Optional[float] = None,
) -> tuple[str, list[dict]]:
    """
    Like call_model() but also returns structured citation data.

    Returns
    -------
    tuple[str, list[dict]]
        (answer_text, citations)
        citations is a list of {"title": str, "url": str} dicts.
        Always empty for non-Google providers.
    """
    if model_key not in MODELS:
        return f"ERROR: Unknown model key '{model_key}'. Check glossary.MODELS.", []

    config   = MODELS[model_key]
    model_id = config["id"]
    provider = config["provider"]

    try:
        if provider == "google":
            return _call_google(model_id, prompt, images, images_mime, max_tokens)
        elif provider == "anthropic":
            return _call_anthropic(model_id, prompt, images, images_mime, max_tokens, request_timeout), []
        elif provider == "openai":
            return _call_openai(model_id, prompt, images, images_mime, max_tokens), []
        elif provider == "xai":
            return _call_xai(model_id, prompt, images, images_mime, max_tokens), []
        else:
            return f"ERROR: Unsupported provider '{provider}' for model '{model_key}'.", []

    except KeyError as exc:
        return (
            f"ERROR: Missing environment variable {exc}.  "
            f"Please set it before running the app."
        ), []
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}", []


def call_model(
    model_key: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Call any registered model by its glossary key and return its text reply.

    Parameters
    ----------
    model_key : str
        A key from glossary.MODELS (e.g. "claude", "gpt", "gemini").
    prompt : str
        The fully-rendered prompt string to send.
    images : list[bytes], optional
        Raw image file bytes for each uploaded visual asset.
    images_mime : list[str], optional
        MIME type per image (e.g. ["image/jpeg", "image/png"]).
    max_tokens : int
        Maximum output tokens. Default 4096. Pass 8192 for synthesis stages.

    Returns
    -------
    str
        The model's plain-text response, or an error string prefixed with
        "ERROR:" so callers can detect and display failures gracefully.
    """
    if model_key not in MODELS:
        return f"ERROR: Unknown model key '{model_key}'. Check glossary.MODELS."

    config   = MODELS[model_key]
    model_id = config["id"]
    provider = config["provider"]

    try:
        if provider == "anthropic":
            return _call_anthropic(model_id, prompt, images, images_mime, max_tokens)

        elif provider == "openai":
            return _call_openai(model_id, prompt, images, images_mime, max_tokens)

        elif provider == "xai":
            return _call_xai(model_id, prompt, images, images_mime, max_tokens)

        elif provider == "google":
            text, citations = _call_google(model_id, prompt, images, images_mime, max_tokens)
            if citations:
                footer = "\n".join(f"- [{c['title']}]({c['url']})" for c in citations)
                text += f"\n\n---\n**Grounding sources (Google Search)**\n{footer}"
            return text

        else:
            return f"ERROR: Unsupported provider '{provider}' for model '{model_key}'."

    except KeyError as exc:
        return (
            f"ERROR: Missing environment variable {exc}.  "
            f"Please set it before running the app."
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"
