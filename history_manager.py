"""
history_manager.py — Persistent Session History
=================================================
Saves and loads past AI council debates to/from a local history.json file.
Optionally mirrors every save to Cloud Firestore when Firebase is configured.

Storage format
--------------
history.json is a JSON array, newest entry first:

    [
        {
            "timestamp": "2026-03-11  14:30",
            "question":  "...",
            "results":   {
                "question":           str,
                "answers":            dict[str, str],
                "critiques":          dict[str, str],   ← "reviewer|target" keys
                "dialectic":          dict[str, str],   ← Stage 3 responses
                "reliability_scores": dict[str, int],   ← 0-100 per model
                "final_answer":       str,
                "fallback_used":      bool,
                "citations":          list[dict],
                "has_image":          bool,
                "pdf_cache_path":     str | absent,     ← cached PDF on disk
            }
        },
        ...
    ]

Serialisation notes
-------------------
The results dict from logic_engine.run_council_debate() contains a
`critiques` mapping with tuple keys — e.g. ("claude", "gpt").
JSON does not support non-string object keys, so those tuples are
encoded as "claude|gpt" strings and decoded back on load.

Cloud Sync (optional)
---------------------
Set FIREBASE_SERVICE_ACCOUNT=/path/to/service-account.json in .env to
enable automatic Firestore mirroring.  Requires firebase-admin package:
    pip install firebase-admin>=6.2.0
The app runs normally without this env var — the sync is silently skipped.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

HISTORY_FILE  = Path(__file__).parent / "history.json"
PDF_CACHE_DIR = Path(__file__).parent / "pdf_cache"   # cached PDFs for history replay
MAX_ENTRIES   = 50   # oldest entries are trimmed once the cap is hit

# ---------------------------------------------------------------------------
# Firebase lazy-init state (module-level singletons)
# ---------------------------------------------------------------------------
_firestore_client: Optional[Any] = None
_firestore_init_tried: bool       = False


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_results(results: dict) -> dict:
    """
    Convert a results dict to a JSON-safe form.

    Handles:
    - critiques: {(reviewer, target): text}  →  {"reviewer|target": text}
    - All other fields (dialectic, reliability_scores, pdf_cache_path, …)
      are passed through as-is — they are already JSON-serialisable.
    """
    out = dict(results)
    # critiques: {(reviewer, target): text}  →  {"reviewer|target": text}
    if "critiques" in out and out["critiques"]:
        out["critiques"] = {
            f"{k[0]}|{k[1]}": v
            for k, v in out["critiques"].items()
        }
    return out


def _deserialize_results(data: dict) -> dict:
    """Restore a results dict from its JSON-safe form."""
    out = dict(data)
    # "reviewer|target"  →  (reviewer, target)
    if "critiques" in out and out["critiques"]:
        out["critiques"] = {
            tuple(k.split("|", 1)): v
            for k, v in out["critiques"].items()
        }
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_to_history(
    results: dict,
    pdf_bytes: Optional[bytes] = None,
) -> Optional[str]:
    """
    Append a completed debate to history.json and optionally cache its PDF.

    The complete results dict — including Stage 3 dialectic responses,
    Stage 4 consensus, reliability scores, and citations — is persisted
    so the UI can fully reconstruct the debate view without any API calls.

    Parameters
    ----------
    results : dict
        The dict returned by logic_engine.run_council_debate().
    pdf_bytes : bytes, optional
        The rendered PDF bytes.  When provided, the file is written to
        pdf_cache/<timestamp>.pdf and the path is stored in the entry so
        history replays can serve it directly without regenerating.

    Returns
    -------
    Optional[str]
        Absolute path to the cached PDF file, or None if no PDF was cached.
    """
    history = _load_raw()
    now    = datetime.now()
    ts     = now.strftime("%Y-%m-%d  %H:%M")
    ts_id  = now.strftime("%Y%m%d_%H%M%S")   # filesystem-safe unique ID

    # ── Optional PDF cache ────────────────────────────────────────────────
    pdf_cache_path: Optional[str] = None
    if pdf_bytes:
        try:
            PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pdf_file = PDF_CACHE_DIR / f"{ts_id}.pdf"
            pdf_file.write_bytes(pdf_bytes)
            pdf_cache_path = str(pdf_file)
        except Exception:
            pdf_cache_path = None   # never crash the app for a cache failure

    # ── Build serialisable results dict ───────────────────────────────────
    serializable = _serialize_results(results)
    if pdf_cache_path:
        serializable["pdf_cache_path"] = pdf_cache_path

    entry = {
        "timestamp": ts,
        "question":  results.get("question", ""),
        "results":   serializable,
    }

    history.insert(0, entry)           # newest first
    history = history[:MAX_ENTRIES]    # trim

    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Optional Firestore mirror ─────────────────────────────────────────
    _sync_to_firestore(entry)

    return pdf_cache_path


def load_history() -> list[dict]:
    """
    Return all saved debates as a list, newest first.

    Each element is a dict with keys:
        "timestamp"  str
        "question"   str
        "results"    dict   ← fully restored (critiques have tuple keys)

    Corrupted or unreadable entries are silently skipped.
    """
    entries = []
    for raw in _load_raw():
        try:
            entries.append({
                "timestamp": raw["timestamp"],
                "question":  raw["question"],
                "results":   _deserialize_results(raw["results"]),
            })
        except (KeyError, TypeError, ValueError):
            continue   # skip corrupted entry
    return entries


def clear_history() -> None:
    """Delete history.json if it exists."""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


def firestore_status() -> bool:
    """
    Return True if a live Firestore connection is available.

    Triggers the lazy init on first call; subsequent calls return the cached
    result instantly.  Safe to call from UI code — never raises.
    """
    try:
        return _get_firestore() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _load_raw() -> list[dict]:
    """Load the raw JSON array from disk, returning [] on any error."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _get_firestore() -> Optional[Any]:
    """
    Lazy-initialize and return the Firestore client.

    Credential resolution order (first match wins):

    1. FIREBASE_SERVICE_ACCOUNT_JSON  — the full service-account JSON as a
       string (ideal for Heroku / Streamlit Cloud / any 12-factor environment
       where uploading a key file is not possible).  Set this secret in the
       cloud dashboard and paste the entire content of firebase_key.json.

    2. FIREBASE_SERVICE_ACCOUNT  — path to a local service-account JSON file
       (used during local development via .env → firebase_key.json).

    Returns None (silently) when neither is set, or when the firebase-admin
    package is not installed, so the app runs perfectly without cloud sync.
    """
    global _firestore_client, _firestore_init_tried
    if _firestore_init_tried:
        return _firestore_client

    _firestore_init_tried = True

    try:
        import json as _json
        import firebase_admin                       # optional dependency
        from firebase_admin import credentials, firestore as fs

        cred = None

        # ── Priority 1: JSON string (cloud / 12-factor environments) ─────────
        sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        if sa_json:
            cred = credentials.Certificate(_json.loads(sa_json))

        # ── Priority 2: File path (local development) ─────────────────────────
        if cred is None:
            sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
            if sa_path:
                cred = credentials.Certificate(sa_path)

        if cred is None:
            return None   # neither secret is set — sync disabled

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        _firestore_client = fs.client()
    except Exception:
        _firestore_client = None   # bad credentials or missing package

    return _firestore_client


def _sync_to_firestore(entry: dict) -> None:
    """
    Mirror a history entry to Firestore under the 'council_debates' collection.

    Document ID is derived from the entry timestamp so replays from any device
    see the same document.  Silently no-ops when Firebase is not configured or
    the write fails — local history.json is always the source of truth.

    To enable:
        1. pip install firebase-admin>=6.2.0
        2. Add FIREBASE_SERVICE_ACCOUNT=/path/to/service-account.json to .env
    """
    db = _get_firestore()
    if db is None:
        return
    try:
        # Build a Firestore-safe document ID from the timestamp
        doc_id = (
            entry.get("timestamp", "unknown")
            .replace(" ", "_")
            .replace(":", "-")
        )
        db.collection("council_debates").document(doc_id).set(entry)
    except Exception:
        pass   # never propagate — Firestore sync is best-effort only
