"""
auth_manager.py — Firebase Authentication via REST API
=======================================================
Uses the Firebase Identity Toolkit REST API (not the Admin SDK) to
authenticate end-users with email + password.  The Admin SDK is a
*server-side* tool; it cannot sign users in.  The REST API is the
standard pattern for custom Python/Streamlit backends.

Required environment variable
------------------------------
    FIREBASE_WEB_API_KEY  — Web API Key from Firebase Console
                            (Project Settings → General → Web API Key).
                            Add to .env for local dev and to
                            Streamlit Cloud secrets for production.

Return contract
---------------
Both sign_in() and sign_up() return either:
  • dict  on success: {"uid": str, "email": str, "id_token": str}
  • str   on failure: a human-readable error message to display in the UI
"""

from __future__ import annotations

import os
from typing import Union

import requests

# Firebase Identity Toolkit REST endpoints
_SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
_SIGN_UP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"

# Map Firebase error codes → user-friendly messages
_FIREBASE_ERRORS: dict[str, str] = {
    "EMAIL_NOT_FOUND":              "No account found with this email. Please sign up first.",
    "INVALID_PASSWORD":             "Incorrect password. Please try again.",
    "INVALID_LOGIN_CREDENTIALS":    "Invalid email or password.",
    "USER_DISABLED":                "This account has been disabled.",
    "EMAIL_EXISTS":                 "An account with this email already exists. Please log in.",
    "WEAK_PASSWORD":                "Password must be at least 6 characters.",
    "INVALID_EMAIL":                "The email address is not valid.",
    "TOO_MANY_ATTEMPTS_TRY_LATER":  "Too many failed attempts. Please try again later.",
    "MISSING_PASSWORD":             "Please enter a password.",
}

UserInfo = dict  # {"uid": str, "email": str, "id_token": str}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _api_key() -> str | None:
    return os.environ.get("FIREBASE_WEB_API_KEY", "").strip() or None


def _parse_error(resp: requests.Response) -> str:
    try:
        raw = resp.json()["error"]["message"]
        code = raw.split(" :")[0].strip()   # Firebase sometimes appends " : details"
        return _FIREBASE_ERRORS.get(code, f"Authentication error: {code}")
    except Exception:
        return f"Authentication error (HTTP {resp.status_code})"


def _post(url: str, payload: dict) -> Union[UserInfo, str]:
    key = _api_key()
    if not key:
        return "FIREBASE_WEB_API_KEY is not set. Authentication is unavailable."
    try:
        resp = requests.post(f"{url}?key={key}", json=payload, timeout=10)
        if resp.ok:
            data = resp.json()
            return {
                "uid":      data["localId"],
                "email":    data["email"],
                "id_token": data["idToken"],
            }
        return _parse_error(resp)
    except requests.RequestException as exc:
        return f"Network error: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sign_in(email: str, password: str) -> Union[UserInfo, str]:
    """
    Sign in an existing user with email + password.

    Returns a user dict on success or an error string on failure.
    """
    return _post(
        _SIGN_IN_URL,
        {"email": email, "password": password, "returnSecureToken": True},
    )


def sign_up(email: str, password: str) -> Union[UserInfo, str]:
    """
    Create a new account with email + password.

    Returns a user dict on success or an error string on failure.
    """
    return _post(
        _SIGN_UP_URL,
        {"email": email, "password": password, "returnSecureToken": True},
    )
