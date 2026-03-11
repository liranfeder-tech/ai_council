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
from typing import List, Optional

import anthropic
import openai
from google import genai
from google.genai import types as genai_types

from glossary import MODELS


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
) -> str:
    """Call the Anthropic (Claude) API and return the text reply."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
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

    message = client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text


def _call_openai(
    model_id: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
) -> str:
    """Call the OpenAI API and return the text reply."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _call_google(
    model_id: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
) -> tuple[str, list[dict]]:
    """
    Call the Google Generative AI (Gemini) API.

    Returns
    -------
    tuple[str, list[dict]]
        (answer_text, citations) — citations always empty (from search_engine.py).
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    images      = images      or []
    images_mime = images_mime or []

    cfg = genai_types.GenerateContentConfig(
        system_instruction=_GEMINI_FIELD_AGENT_INSTRUCTION,
        max_output_tokens=4096,
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


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def call_model_with_citations(
    model_key: str,
    prompt: str,
    images: Optional[List[bytes]] = None,
    images_mime: Optional[List[str]] = None,
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
            return _call_google(model_id, prompt, images, images_mime)
        elif provider == "anthropic":
            return _call_anthropic(model_id, prompt, images, images_mime), []
        elif provider == "openai":
            return _call_openai(model_id, prompt, images, images_mime), []
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
            return _call_anthropic(model_id, prompt, images, images_mime)

        elif provider == "openai":
            return _call_openai(model_id, prompt, images, images_mime)

        elif provider == "google":
            text, citations = _call_google(model_id, prompt, images, images_mime)
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
