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
    uid: Optional[str] = None,
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
    _sync_to_firestore(entry, uid=uid)

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


def get_user_history(uid: str) -> list[dict]:
    """
    Fetch debate history for a specific authenticated user from Firestore.

    Falls back to the local history.json when Firestore is unavailable.
    Each returned element has the same shape as load_history() entries:
        {"timestamp": str, "question": str, "results": dict}

    Parameters
    ----------
    uid : str
        The Firebase user UID (localId) of the authenticated user.
    """
    db = _get_firestore()
    if db is not None:
        try:
            docs = (
                db.collection("users")
                .document(uid)
                .collection("debates")
                .order_by("timestamp", direction="DESCENDING")
                .limit(MAX_ENTRIES)
                .stream()
            )
            entries = []
            for doc in docs:
                raw = doc.to_dict()
                if not raw:
                    continue
                try:
                    entries.append({
                        "timestamp": raw.get("timestamp", ""),
                        "question":  raw.get("question", ""),
                        "results":   _deserialize_results(raw.get("results", {})),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            if entries:
                return entries
        except Exception:
            pass   # fall through to local history on any Firestore error

    # Fallback: local history.json (not user-scoped, but still useful)
    return load_history()


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


def _sync_to_firestore(entry: dict, uid: Optional[str] = None) -> None:
    """
    Mirror a history entry to Firestore.

    When a uid is provided, the entry is written to the user-scoped path:
        users/{uid}/debates/{doc_id}
    Otherwise it falls back to the global collection:
        council_debates/{doc_id}

    Document ID is derived from the timestamp so replays are deterministic.
    Silently no-ops when Firebase is not configured — local history.json is
    always the source of truth.
    """
    db = _get_firestore()
    if db is None:
        return
    try:
        doc_id = (
            entry.get("timestamp", "unknown")
            .replace(" ", "_")
            .replace(":", "-")
        )
        if uid:
            ref = (
                db.collection("users")
                .document(uid)
                .collection("debates")
                .document(doc_id)
            )
        else:
            ref = db.collection("council_debates").document(doc_id)
        ref.set(entry)
    except Exception:
        pass   # never propagate — Firestore sync is best-effort only
