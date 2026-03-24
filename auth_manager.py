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
_SIGN_IN_URL        = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
_SIGN_UP_URL        = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
_SEND_OOB_URL       = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
_LOOKUP_URL         = "https://identitytoolkit.googleapis.com/v1/accounts:lookup"
_TOKEN_REFRESH_URL  = "https://securetoken.googleapis.com/v1/token"

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

UserInfo = dict  # {"uid": str, "email": str, "id_token": str, "email_verified": bool}


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
                "uid":            data["localId"],
                "email":          data["email"],
                "id_token":       data["idToken"],
                "refresh_token":  data.get("refreshToken", ""),
                "email_verified": data.get("emailVerified", False),
            }
        return _parse_error(resp)
    except requests.RequestException as exc:
        return f"Network error: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh_session(refresh_token: str) -> Union[UserInfo, str]:
    """
    Exchange a Firebase refresh_token for a fresh id_token.

    Called on page reload to restore the session without asking the user
    to log in again.  Firebase refresh tokens do not expire unless the
    user's account is disabled or the token is explicitly revoked.

    Returns a user dict on success or an error string on failure.
    """
    key = _api_key()
    if not key or not refresh_token:
        return "Missing API key or refresh token."
    try:
        resp = requests.post(
            f"{_TOKEN_REFRESH_URL}?key={key}",
            json={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            new_id_token      = data.get("id_token") or data.get("access_token", "")
            new_refresh_token = data.get("refresh_token", refresh_token)
            uid               = data.get("user_id", "")
            # Fetch email + email_verified via /accounts:lookup
            lookup_resp = requests.post(
                f"{_LOOKUP_URL}?key={key}",
                json={"idToken": new_id_token},
                timeout=10,
            )
            if lookup_resp.ok:
                users = lookup_resp.json().get("users", [{}])
                user  = users[0] if users else {}
                return {
                    "uid":            uid or user.get("localId", ""),
                    "email":          user.get("email", ""),
                    "id_token":       new_id_token,
                    "refresh_token":  new_refresh_token,
                    "email_verified": bool(user.get("emailVerified", False)),
                }
        return _parse_error(resp)
    except requests.RequestException as exc:
        return f"Network error: {exc}"


def sign_in(email: str, password: str) -> Union[UserInfo, str]:
    """
    Sign in an existing user with email + password.

    Returns a user dict on success or an error string on failure.
    """
    result = _post(
        _SIGN_IN_URL,
        {"email": email, "password": password, "returnSecureToken": True},
    )
    if isinstance(result, dict):
        # Firebase sign-in response does NOT include emailVerified — fetch it.
        verified = get_email_verified(result["id_token"])
        result["email_verified"] = verified is True
    return result


def sign_up(email: str, password: str) -> Union[UserInfo, str]:
    """
    Create a new account with email + password.
    Automatically sends a verification email on success.

    Returns a user dict on success or an error string on failure.
    """
    result = _post(
        _SIGN_UP_URL,
        {"email": email, "password": password, "returnSecureToken": True},
    )
    if isinstance(result, dict):
        # Best-effort — never let this block the sign-up response
        send_verification_email(result["id_token"])
    return result


def send_verification_email(id_token: str) -> Union[bool, str]:
    """
    Send a verification email to the currently signed-in user.

    Returns True on success or an error string on failure.
    """
    key = _api_key()
    if not key:
        return "FIREBASE_WEB_API_KEY is not set."
    try:
        resp = requests.post(
            f"{_SEND_OOB_URL}?key={key}",
            json={"requestType": "VERIFY_EMAIL", "idToken": id_token},
            timeout=10,
        )
        return True if resp.ok else _parse_error(resp)
    except requests.RequestException as exc:
        return f"Network error: {exc}"


def get_email_verified(id_token: str) -> Union[bool, str]:
    """
    Re-fetch the emailVerified flag from Firebase for the current user.
    Used by the 'I've verified — refresh status' button.

    Returns True/False on success or an error string on failure.
    """
    key = _api_key()
    if not key:
        return "FIREBASE_WEB_API_KEY is not set."
    try:
        resp = requests.post(
            f"{_LOOKUP_URL}?key={key}",
            json={"idToken": id_token},
            timeout=10,
        )
        if resp.ok:
            users = resp.json().get("users", [])
            if users:
                return bool(users[0].get("emailVerified", False))
            return False
        return _parse_error(resp)
    except requests.RequestException as exc:
        return f"Network error: {exc}"
