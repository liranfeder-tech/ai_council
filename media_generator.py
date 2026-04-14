"""
media_generator.py — AI-powered image and video generation
===========================================================
Converts a council debate's final answer into an optimised creative brief
(via Claude), then calls the appropriate generation API.

Providers
---------
Images : DALL-E 3 via OpenAI   (uses existing OPENAI_API_KEY — no extra sign-up)
Videos : Replicate              (needs REPLICATE_API_TOKEN from replicate.com)

Adding a new video provider
---------------------------
Add a new entry to VIDEO_MODELS below and the rest of the code picks it up.
"""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import openai
import requests

# ---------------------------------------------------------------------------
# Image — DALL-E 3 size presets
# ---------------------------------------------------------------------------

IMAGE_SIZES: dict[str, str] = {
    "landscape": "1792x1024",   # 16:9  — banners, YouTube thumbnails
    "portrait":  "1024x1792",   # 9:16  — Stories, Reels, mobile ads
    "square":    "1024x1024",   # 1:1   — Instagram, Twitter/X
}

# ---------------------------------------------------------------------------
# Video — Replicate model registry
# Add more entries here to support additional models without touching app.py
# ---------------------------------------------------------------------------

VIDEO_MODELS: dict[str, dict] = {
    "minimax": {
        "id":    "minimax/video-01",
        "label": "Minimax Video — 6 שניות",
        "input": lambda p, ar: {
            "prompt": p,
            "duration": 6,
            "aspect_ratio": ar,
        },
    },
    "luma": {
        "id":    "luma/dream-machine",
        "label": "Luma Dream Machine — 5 שניות",
        "input": lambda p, ar: {
            "prompt": p,
            "aspect_ratio": ar,
            "loop": False,
        },
    },
    "wan": {
        "id":    "wan-ai/wan2.1-t2v-480p",
        "label": "Wan 2.1 — מהיר (480p)",
        "input": lambda p, ar: {
            "prompt": p,
            "num_frames": 81,
            "aspect_ratio": ar,
        },
    },
}

# Aspect-ratio strings passed to Replicate models
_AR_MAP: dict[str, str] = {
    "landscape": "16:9",
    "portrait":  "9:16",
    "square":    "1:1",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _openai_key() -> str:
    """Return the OpenAI API key: session override first, then os.environ."""
    try:
        from ai_factory import _SESSION_KEYS
        session = _SESSION_KEYS.get()
        if "openai" in session and session["openai"]:
            return session["openai"]
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def _replicate_key() -> str:
    return os.environ.get("REPLICATE_API_TOKEN", "")


# ---------------------------------------------------------------------------
# Prompt generation (Claude converts the debate answer → media prompt)
# ---------------------------------------------------------------------------

def generate_image_prompt(final_answer: str, campaign_context: str = "") -> str:
    """
    Ask the council's master model to distil the debate answer into an
    optimised DALL-E 3 image prompt.

    Returns a plain-text English prompt string.
    """
    from ai_factory import call_model
    from glossary import PROMPT_MEDIA_IMAGE_BRIEF

    brief = PROMPT_MEDIA_IMAGE_BRIEF.format(
        final_answer=final_answer.strip(),
        campaign_context=campaign_context.strip() or "None provided",
    )
    result = call_model("claude", brief)
    # Strip any accidental markdown fences the model might add
    return result.strip().strip("`").strip()


def generate_video_prompt(
    final_answer: str,
    campaign_context: str = "",
    question: str = "",
) -> str:
    """
    Ask the council's master model to distil the debate answer into a
    text-to-video generation prompt (English, cinematic, product-focused).

    Returns a plain-text English prompt string.
    """
    from ai_factory import call_model
    from glossary import PROMPT_MEDIA_VIDEO_BRIEF

    brief = PROMPT_MEDIA_VIDEO_BRIEF.format(
        question=question.strip() or "See council answer below.",
        final_answer=final_answer.strip(),
        campaign_context=campaign_context.strip() or "None provided",
    )
    result = call_model("claude", brief)
    return result.strip().strip("`").strip()


# ---------------------------------------------------------------------------
# Image generation — DALL-E 3
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    size: str = "landscape",
) -> dict:
    """
    Generate a single marketing image with DALL-E 3.

    Parameters
    ----------
    prompt : str
        The creative brief / image description (English).
    size : str
        One of "landscape", "portrait", or "square".

    Returns
    -------
    dict
        bytes      – raw PNG bytes, or None on failure
        prompt_used – the prompt DALL-E actually used (may be revised by OpenAI)
        error      – error message string, or None on success
    """
    key = _openai_key()
    if not key:
        return {
            "bytes": None,
            "prompt_used": prompt,
            "error": "OPENAI_API_KEY לא מוגדר. הוסף אותו ל-.env",
        }

    size_str = IMAGE_SIZES.get(size, IMAGE_SIZES["landscape"])

    try:
        client = openai.OpenAI(api_key=key, timeout=60)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size_str,          # type: ignore[arg-type]
            quality="hd",
            response_format="b64_json",
            n=1,
        )
        item = response.data[0]
        img_bytes = base64.b64decode(item.b64_json)   # type: ignore[arg-type]
        revised   = getattr(item, "revised_prompt", None) or prompt

        return {"bytes": img_bytes, "prompt_used": revised, "error": None}

    except openai.OpenAIError as exc:
        return {"bytes": None, "prompt_used": prompt, "error": f"OpenAI: {exc}"}
    except Exception as exc:
        return {"bytes": None, "prompt_used": prompt, "error": str(exc)}


# ---------------------------------------------------------------------------
# Video generation — Replicate
# ---------------------------------------------------------------------------

def generate_video(
    prompt: str,
    model_key: str = "minimax",
    aspect_ratio: str = "landscape",
) -> dict:
    """
    Generate a short marketing clip via Replicate.

    Parameters
    ----------
    prompt : str
        The cinematic scene description (English).
    model_key : str
        Key into VIDEO_MODELS registry.
    aspect_ratio : str
        One of "landscape", "portrait", "square".

    Returns
    -------
    dict
        bytes      – raw video bytes, or None on failure
        url        – direct CDN URL to the video file
        prompt_used – the prompt that was sent
        error      – error message string, or None on success
    """
    api_token = _replicate_key()
    if not api_token:
        return {
            "bytes": None,
            "url": None,
            "prompt_used": prompt,
            "error": (
                "REPLICATE_API_TOKEN לא מוגדר ב-.env. "
                "הירשם בחינם בכתובת replicate.com והוסף את המפתח."
            ),
        }

    model_cfg = VIDEO_MODELS.get(model_key, VIDEO_MODELS["minimax"])
    model_id  = model_cfg["id"]
    ar_str    = _AR_MAP.get(aspect_ratio, "16:9")
    inputs    = model_cfg["input"](prompt, ar_str)

    try:
        import replicate  # type: ignore[import-untyped]
    except ImportError:
        return {
            "bytes": None,
            "url": None,
            "prompt_used": prompt,
            "error": "חבילת replicate לא מותקנת. הרץ: pip install replicate",
        }

    try:
        client = replicate.Client(api_token=api_token)
        output = client.run(model_id, input=inputs)

        # Replicate may return a URL string, a FileOutput, or a list
        if isinstance(output, list):
            raw = output[0]
        else:
            raw = output

        # FileOutput objects have a .url attribute; plain strings are URLs directly
        video_url: str = getattr(raw, "url", None) or str(raw)

        # Stream-download the video bytes so the user can save it locally
        resp = requests.get(video_url, timeout=180, stream=True)
        resp.raise_for_status()
        video_bytes = b"".join(resp.iter_content(chunk_size=65536))

        return {
            "bytes": video_bytes,
            "url": video_url,
            "prompt_used": prompt,
            "error": None,
        }

    except Exception as exc:
        return {
            "bytes": None,
            "url": None,
            "prompt_used": prompt,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Cultural Campaign Generator
# ---------------------------------------------------------------------------

def _generate_one_cultural_variant(
    market_name: str,
    profile: dict,
    final_answer: str,
    campaign_context: str,
    media_type: str,       # "image" or "video"
    size: str,
    video_model_key: str,
    question: str = "",
) -> dict:
    """
    Generate a single culturally-adapted image or video for one market.
    Called concurrently by generate_cultural_variants().

    Returns a result dict with keys: market, flag, bytes, url, prompt_used, error.
    """
    from ai_factory import call_model
    from glossary import PROMPT_MEDIA_CULTURAL_BRIEF

    # Step 1 — ask Claude to build a culturally-aware creative brief
    brief_prompt = PROMPT_MEDIA_CULTURAL_BRIEF.format(
        media_type="image" if media_type == "image" else "video",
        market=market_name,
        question=question.strip() or "See council answer below.",
        final_answer=final_answer.strip(),
        campaign_context=campaign_context.strip() or "None provided",
        cultural_brief=profile["brief"],
    )
    creative_prompt = call_model("claude", brief_prompt).strip().strip("`").strip()

    # Step 2 — call the appropriate generation API
    if media_type == "image":
        result = generate_image(creative_prompt, size=size)
        return {
            "market":       market_name,
            "flag":         profile.get("flag", ""),
            "bytes":        result["bytes"],
            "url":          None,
            "prompt_used":  result["prompt_used"],
            "error":        result["error"],
        }
    else:
        result = generate_video(creative_prompt, model_key=video_model_key, aspect_ratio=size)
        return {
            "market":       market_name,
            "flag":         profile.get("flag", ""),
            "bytes":        result["bytes"],
            "url":          result.get("url"),
            "prompt_used":  result["prompt_used"],
            "error":        result["error"],
        }


def generate_cultural_variants(
    final_answer: str,
    selected_markets: list[str],
    media_type: str = "image",
    size: str = "landscape",
    campaign_context: str = "",
    video_model_key: str = "minimax",
    max_workers: int = 3,
    question: str = "",
) -> list[dict]:
    """
    Generate culturally-adapted media for multiple markets in parallel.

    Parameters
    ----------
    final_answer : str
        The council debate's final answer — used as creative source material.
    selected_markets : list[str]
        Market names as they appear in CULTURAL_PROFILES keys
        (e.g. ["🇩🇪 גרמניה", "🇧🇷 ברזיל"]).
    media_type : str
        "image" or "video".
    size : str
        "landscape", "portrait", or "square".
    campaign_context : str
        Optional additional context applied to every market.
    video_model_key : str
        Replicate model key (see VIDEO_MODELS).
    max_workers : int
        Number of markets to generate in parallel.
        Capped at 3 to avoid rate-limit issues with DALL-E / Replicate.

    Returns
    -------
    list[dict]
        One result dict per market, in the same order as selected_markets.
        Each dict: {market, flag, bytes, url, prompt_used, error}
    """
    from glossary import CULTURAL_PROFILES

    results_map: dict[str, dict] = {}

    profiles_to_run = [
        (name, CULTURAL_PROFILES[name])
        for name in selected_markets
        if name in CULTURAL_PROFILES
    ]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(profiles_to_run))) as ex:
        futures = {
            ex.submit(
                _generate_one_cultural_variant,
                name, profile,
                final_answer, campaign_context,
                media_type, size, video_model_key,
                question,
            ): name
            for name, profile in profiles_to_run
        }
        for future in as_completed(futures):
            market_name = futures[future]
            try:
                results_map[market_name] = future.result()
            except Exception as exc:
                results_map[market_name] = {
                    "market":      market_name,
                    "flag":        CULTURAL_PROFILES.get(market_name, {}).get("flag", ""),
                    "bytes":       None,
                    "url":         None,
                    "prompt_used": "",
                    "error":       str(exc),
                }

    # Return in original selection order
    return [results_map[name] for name in selected_markets if name in results_map]


# ---------------------------------------------------------------------------
# Storyboard — 3 sequential clips that together tell a complete story
# ---------------------------------------------------------------------------

def _parse_storyboard_json(raw: str) -> list[dict]:
    """
    Parse Claude's JSON storyboard output.
    Strips markdown fences if present and returns the list of scene dicts.
    Falls back to an empty list on any parse error.
    """
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def generate_video_storyboard(
    final_answer: str,
    question: str = "",
    campaign_context: str = "",
    model_key: str = "minimax",
    aspect_ratio: str = "landscape",
    progress_cb=None,
) -> list[dict]:
    """
    Generate a 3-scene sequential video storyboard.

    Claude writes a 3-scene script (JSON), then each scene is rendered into
    a video clip independently. The clips are intended to be played in order.

    Parameters
    ----------
    final_answer : str
        Council debate final answer used as creative source.
    question : str
        Original user question / product description for context.
    campaign_context : str
        Optional extra campaign notes.
    model_key : str
        Replicate video model key (see VIDEO_MODELS).
    aspect_ratio : str
        "landscape", "portrait", or "square".
    progress_cb : callable, optional
        Called as progress_cb(scene_number, total=3, status="generating"|"done"|"error")
        to report per-scene progress to the UI.

    Returns
    -------
    list[dict]
        One dict per scene:
        {scene_number, title, prompt_used, bytes, url, error}
    """
    from ai_factory import call_model
    from glossary import PROMPT_MEDIA_VIDEO_STORYBOARD

    # ── Step 1: ask Claude to write the 3-scene storyboard ───────────────────
    storyboard_prompt = PROMPT_MEDIA_VIDEO_STORYBOARD.format(
        question=question.strip() or "See council answer below.",
        final_answer=final_answer.strip(),
        campaign_context=campaign_context.strip() or "None provided",
    )
    raw_json = call_model("claude", storyboard_prompt)
    scenes   = _parse_storyboard_json(raw_json)

    if not scenes:
        return [{
            "scene_number": 0,
            "title":        "שגיאה",
            "prompt_used":  raw_json,
            "bytes":        None,
            "url":          None,
            "error":        "Claude לא החזיר JSON תקין. נסה שוב.",
        }]

    # ── Step 2: generate each scene clip sequentially ────────────────────────
    # Sequential (not parallel) so Replicate queue doesn't pile up and
    # so the progress_cb can report scene-by-scene to the UI.
    results: list[dict] = []
    for scene in scenes:
        n      = scene.get("scene_number", len(results) + 1)
        title  = scene.get("title", f"סצינה {n}")
        prompt = scene.get("prompt", "")

        if progress_cb:
            progress_cb(n, total=len(scenes), status="generating")

        if not prompt:
            results.append({
                "scene_number": n,
                "title":        title,
                "prompt_used":  "",
                "bytes":        None,
                "url":          None,
                "error":        "סצינה ריקה — Claude לא סיפק prompt",
            })
            continue

        clip = generate_video(prompt, model_key=model_key, aspect_ratio=aspect_ratio)

        if progress_cb:
            progress_cb(n, total=len(scenes),
                        status="done" if not clip["error"] else "error")

        results.append({
            "scene_number": n,
            "title":        title,
            "prompt_used":  clip["prompt_used"],
            "bytes":        clip["bytes"],
            "url":          clip.get("url"),
            "error":        clip["error"],
        })

    return results
