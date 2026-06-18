"""Google Antigravity OAuth PKCE flow.

Antigravity is Google's internal IDE platform that provides access to Gemini
models with significantly higher rate limits than the standard Code Assist API
(``cloudcode-pa.googleapis.com``).  This module implements the OAuth PKCE flow
using the Antigravity OAuth client, storing tokens in
``~/.hermes/auth/antigravity_oauth.json``.

The resulting access token is used by ``agent.antigravity_adapter`` to talk to
``daily-cloudcode-pa.sandbox.googleapis.com`` (the Antigravity API backend).

Derived from opencode-antigravity-auth (MIT) by NoeFabris.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import logging
import os
import secrets
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_constants import get_hermes_home, secure_parent_dir

logger = logging.getLogger(__name__)


# =============================================================================
# Antigravity OAuth client credentials
# =============================================================================
# These are the public Antigravity desktop OAuth client credentials from the
# opencode-antigravity-auth plugin.  Like the Gemini CLI OAuth, these are NOT
# confidential — desktop OAuth clients use PKCE for security.
# See: https://github.com/NoeFabris/opencode-antigravity-auth

ANTIGRAVITY_CLIENT_ID = (
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
)
ANTIGRAVITY_CLIENT_SECRET = "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"

# Override via env vars
ENV_CLIENT_ID = "HERMES_ANTIGRAVITY_CLIENT_ID"
ENV_CLIENT_SECRET = "HERMES_ANTIGRAVITY_CLIENT_SECRET"


# =============================================================================
# Endpoints & constants
# =============================================================================

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v1/userinfo"

# Antigravity needs additional scopes beyond the standard Code Assist ones
OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform "
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile "
    "https://www.googleapis.com/auth/cclog "
    "https://www.googleapis.com/auth/experimentsandconfigs"
)

# Different redirect port from the gemini-cli OAuth (8085 → 51121)
DEFAULT_REDIRECT_PORT = 51121
REDIRECT_HOST = "localhost"
CALLBACK_PATH = "/oauth-callback"

REFRESH_SKEW_SECONDS = 60
TOKEN_REQUEST_TIMEOUT_SECONDS = 20.0
CALLBACK_WAIT_SECONDS = 300
LOCK_TIMEOUT_SECONDS = 30.0

_HEADLESS_ENV_VARS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "HERMES_HEADLESS")


# =============================================================================
# Error type
# =============================================================================

class AntigravityOAuthError(RuntimeError):
    """Raised for any failure in the Antigravity OAuth flow."""

    def __init__(self, message: str, *, code: str = "antigravity_oauth_error") -> None:
        super().__init__(message)
        self.code = code


# =============================================================================
# Token storage
# =============================================================================

@dataclass
class AntigravityCreds:
    access_token: str = ""
    refresh_token: str = ""
    expires_ms: int = 0         # unix MILLIseconds
    email: str = ""
    project_id: str = ""
    managed_project_id: str = ""


def _credentials_path() -> Path:
    return get_hermes_home() / "auth" / "antigravity_oauth.json"


def _lock_path() -> Path:
    return _credentials_path().with_suffix(".json.lock")


_lock_state = threading.local()


@contextlib.contextmanager
def _credentials_lock(timeout_seconds: float = LOCK_TIMEOUT_SECONDS):
    """Cross-process lock around the credentials file."""
    depth = getattr(_lock_state, "depth", 0)
    if depth > 0:
        _lock_state.depth = depth + 1
        try:
            yield
        finally:
            _lock_state.depth -= 1
        return

    lock_file_path = _lock_path()
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_file_path), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None

        if fcntl is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring Antigravity OAuth lock at {lock_file_path}."
                        )
                    time.sleep(0.05)
        else:
            acquired = True

        _lock_state.depth = 1
        yield
    finally:
        try:
            if acquired:
                try:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except ImportError:
                    pass
        finally:
            os.close(fd)
            _lock_state.depth = 0


# =============================================================================
# Client ID resolution
# =============================================================================

def _get_client_id() -> str:
    env_val = (os.getenv(ENV_CLIENT_ID) or "").strip()
    if env_val:
        return env_val
    return ANTIGRAVITY_CLIENT_ID


def _get_client_secret() -> str:
    env_val = (os.getenv(ENV_CLIENT_SECRET) or "").strip()
    if env_val:
        return env_val
    return ANTIGRAVITY_CLIENT_SECRET


def _require_client_id() -> str:
    cid = _get_client_id()
    if not cid:
        raise AntigravityOAuthError(
            "Antigravity OAuth client ID is not available.\n"
            "Set HERMES_ANTIGRAVITY_CLIENT_ID and HERMES_ANTIGRAVITY_CLIENT_SECRET "
            "in ~/.hermes/.env",
            code="antigravity_oauth_client_id_missing",
        )
    return cid


# =============================================================================
# PKCE
# =============================================================================

def _generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return verifier, challenge


# =============================================================================
# Credential persistence
# =============================================================================

def load_credentials() -> Optional[AntigravityCreds]:
    """Load stored Antigravity credentials, or None."""
    path = _credentials_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_refresh = data.get("refresh", "")
        raw_access = data.get("access", "")
        # Parse packed refresh token format: "refreshToken|projectId|managedProjectId"
        parts = raw_refresh.split("|", 2) if raw_refresh else []
        return AntigravityCreds(
            access_token=raw_access,
            refresh_token=parts[0] if len(parts) > 0 else "",
            expires_ms=int(data.get("expires", 0)),
            email=str(data.get("email", "")),
            project_id=parts[1] if len(parts) > 1 else "",
            managed_project_id=parts[2] if len(parts) > 2 else "",
        )
    except (ValueError, OSError, KeyError) as exc:
        logger.warning("Failed to load Antigravity OAuth credentials: %s", exc)
        return None


def save_credentials(
    access_token: str,
    refresh_token: str,
    expires_ms: int,
    email: str = "",
    project_id: str = "",
    managed_project_id: str = "",
) -> None:
    """Save Antigravity OAuth credentials atomically."""
    path = _credentials_path()
    secure_parent_dir(path.parent)
    packed_refresh = refresh_token
    if project_id or managed_project_id:
        packed_refresh = f"{refresh_token}|{project_id}|{managed_project_id}"
    data = {
        "refresh": packed_refresh,
        "access": access_token,
        "expires": expires_ms,
        "email": email,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))
    os.chmod(str(path), stat.S_IRUSR | stat.S_IWUSR)


def update_project_ids(project_id: str = "", managed_project_id: str = "") -> None:
    """Update stored project IDs without overwriting tokens."""
    creds = load_credentials()
    if creds is None:
        return
    save_credentials(
        access_token=creds.access_token,
        refresh_token=creds.refresh_token,
        expires_ms=creds.expires_ms,
        email=creds.email,
        project_id=project_id or creds.project_id,
        managed_project_id=managed_project_id or creds.managed_project_id,
    )


def resolve_project_id_from_env() -> str:
    """Check env for a configured project ID (GOOGLE_CLOUD_PROJECT)."""
    return (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()


# =============================================================================
# Token refresh
# =============================================================================

def _refresh_access_token(refresh_token: str) -> Tuple[str, int]:
    """Exchange a refresh token for a new access token.

    Returns (new_access_token, expires_ms).
    """
    client_id = _get_client_id()
    client_secret = _get_client_secret()
    data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AntigravityOAuthError(
            f"Token refresh failed (HTTP {exc.code}): {body}",
            code="antigravity_token_refresh_failed",
        ) from exc
    except urllib.error.URLError as exc:
        raise AntigravityOAuthError(
            f"Token refresh network error: {exc.reason}",
            code="antigravity_token_network_error",
        ) from exc

    payload = json.loads(resp.read().decode("utf-8"))
    new_access = (payload.get("access_token") or "").strip()
    if not new_access:
        raise AntigravityOAuthError(
            "Token refresh returned no access_token.",
            code="antigravity_token_refresh_empty",
        )

    expires_in = int(payload.get("expires_in", 3600))
    expires_ms = int(time.time() * 1000) + (expires_in * 1000) - (REFRESH_SKEW_SECONDS * 1000)
    return new_access, expires_ms


# =============================================================================
# Token validation & access
# =============================================================================

def access_token_expired() -> bool:
    """Check if the stored access token is expired or missing."""
    creds = load_credentials()
    if creds is None or not creds.access_token:
        return True
    # Buffer with REFRESH_SKEW_SECONDS
    return (time.time() * 1000) >= (creds.expires_ms - (REFRESH_SKEW_SECONDS * 1000))


def get_valid_access_token(*, force_refresh: bool = False) -> str:
    """Return a valid Antigravity access token, refreshing if needed.

    Uses cross-process locking so concurrent Hermes instances don't
    race on token refresh.
    """
    with _credentials_lock():
        creds = load_credentials()
        if creds is None:
            raise AntigravityOAuthError(
                "No Antigravity OAuth credentials found. Run `hermes auth add ...` "
                "or authenticate via the provider picker.",
                code="antigravity_not_authenticated",
            )
        if not creds.refresh_token:
            raise AntigravityOAuthError(
                "No Antigravity refresh token available.",
                code="antigravity_no_refresh_token",
            )
        if force_refresh or access_token_expired():
            new_access, new_expires = _refresh_access_token(creds.refresh_token)
            save_credentials(
                access_token=new_access,
                refresh_token=creds.refresh_token,
                expires_ms=new_expires,
                email=creds.email,
                project_id=creds.project_id,
                managed_project_id=creds.managed_project_id,
            )
            return new_access
        return creds.access_token


# =============================================================================
# OAuth callback server
# =============================================================================

class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP server that captures the OAuth callback code."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        code_list = params.get("code", [])
        state_list = params.get("state", [])
        error_list = params.get("error", [])

        if error_list:
            self._respond_error(f"OAuth error: {error_list[0]}")
            self.server._callback_result = ("error", error_list[0])  # type: ignore[attr-defined]
            return

        if code_list:
            code = code_list[0]
            state = state_list[0] if state_list else ""
            self._respond_success()
            self.server._callback_result = ("success", code, state)  # type: ignore[attr-defined]
        else:
            self._respond_error("No authorization code received.")
            self.server._callback_result = ("error", "no_code")  # type: ignore[attr-defined]

    def _respond_success(self) -> None:
        body = (
            "<html><body><h1>✅ Authentication successful!</h1>"
            "<p>You can close this tab and return to Hermes.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _respond_error(self, message: str) -> None:
        body = f"<html><body><h1>❌ Authentication failed</h1><p>{message}</p></body></html>"
        self.send_response(400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("OAuth callback: " + fmt, *args)


def _start_callback_server(port: int) -> http.server.HTTPServer:
    """Start the local OAuth callback server."""
    server = http.server.HTTPServer((REDIRECT_HOST, port), _CallbackHandler)
    server._callback_result = None  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def _open_browser(url: str) -> None:
    """Open a URL in the default browser."""
    import webbrowser
    logger.info("Opening browser for Antigravity OAuth: %s", url)
    webbrowser.open(url)


def _is_headless() -> bool:
    """Detect if running in a headless environment."""
    return any(os.getenv(var) for var in _HEADLESS_ENV_VARS)


# =============================================================================
# PKCE OAuth flow
# =============================================================================

def _build_authorization_url(
    client_id: str,
    redirect_uri: str,
    scope: str,
    code_challenge: str,
    state: str,
) -> str:
    params = urllib.parse.urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    })
    return f"{AUTH_ENDPOINT}?{params}"


def _exchange_code(code: str, redirect_uri: str, code_verifier: str) -> Dict[str, Any]:
    """Exchange the authorization code for tokens."""
    client_id = _get_client_id()
    client_secret = _get_client_secret()
    data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "Accept": "*/*",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AntigravityOAuthError(
            f"Token exchange failed (HTTP {exc.code}): {body}",
            code="antigravity_token_exchange_failed",
        ) from exc

    return json.loads(resp.read().decode("utf-8"))


def _fetch_user_email(access_token: str) -> str:
    """Fetch the user's email via Google's userinfo endpoint."""
    req = urllib.request.Request(
        USERINFO_ENDPOINT + "?alt=json",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=TOKEN_REQUEST_TIMEOUT_SECONDS)
        data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("email", ""))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        logger.warning("Failed to fetch Antigravity user email: %s", exc)
        return ""


# =============================================================================
# Public API
# =============================================================================

def authorize_interactive() -> Dict[str, Any]:
    """Run the full interactive Antigravity OAuth PKCE flow.

    If running in a headless/SSH environment (detected automatically), falls
    back to paste-mode: prints the auth URL for you to open in a local browser,
    then prompts you to paste the redirect URL back.

    Returns a dict with keys: provider, base_url, api_key, source,
    expires_at_ms, email, project_id.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    client_id = _require_client_id()
    client_secret = _get_client_secret()

    port = DEFAULT_REDIRECT_PORT
    redirect_uri = f"http://{REDIRECT_HOST}:{port}{CALLBACK_PATH}"

    auth_url = _build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=OAUTH_SCOPES,
        code_challenge=challenge,
        state=state,
    )

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Google Antigravity OAuth Authorization           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║ ⚠️  Terms of Service Warning:                          ║")
    print("║ Using this OAuth client with third-party software may  ║")
    print("║ violate Google's ToS. Use at your own risk.            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if _is_headless():
        code = _paste_mode_login(auth_url, verifier, redirect_uri, client_id, client_secret)
    else:
        code = _callback_server_login(
            port, auth_url, verifier, redirect_uri, state,
            client_id, client_secret,
        )

    if not code:
        raise AntigravityOAuthError(
            "No authorization code received.",
            code="antigravity_no_code",
        )

    token_data = _exchange_code(code, redirect_uri, verifier)
    access_token = (token_data.get("access_token") or "").strip()
    refresh_token = (token_data.get("refresh_token") or "").strip()
    expires_in = int(token_data.get("expires_in", 3600))
    if not refresh_token:
        raise AntigravityOAuthError(
            "No refresh token received. Make sure to grant offline access.",
            code="antigravity_no_refresh_token",
        )

    email = _fetch_user_email(access_token)
    expires_ms = int(time.time() * 1000) + (expires_in * 1000) - (REFRESH_SKEW_SECONDS * 1000)

    save_credentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_ms=expires_ms,
        email=email,
    )

    return {
        "provider": "google-antigravity",
        "base_url": "antigravity://google",
        "api_key": access_token,
        "source": "antigravity-oauth",
        "expires_at_ms": expires_ms,
        "email": email,
        "project_id": "",
    }


def _paste_mode_login(
    auth_url: str,
    verifier: str = "",
    redirect_uri: str = "",
    client_id: str = "",
    client_secret: str = "",
) -> Optional[str]:
    """Run OAuth in paste mode — no local callback server needed.

    The user opens the URL in their own browser (on a different machine),
    authorizes, and pastes the redirect URL (or code) back here.
    """
    print("Open this URL in a browser on any device:")
    print(f"  {auth_url}")
    print()
    print("After signing in, Google will redirect to a page that doesn't load.")
    print("Copy the FULL URL from your browser's address bar and paste it below.")
    print()
    return _prompt_paste_fallback()


def _callback_server_login(
    port: int,
    auth_url: str,
    verifier: str,
    redirect_uri: str,
    state: str,
    client_id: str,
    client_secret: str,
) -> Optional[str]:
    """Run OAuth with a local callback server.

    Starts an HTTP server on localhost, opens the browser, and captures
    the authorization code from the redirect.
    """
    server = _start_callback_server(port)
    try:
        print("Opening your browser to sign in to Google…")
        print(f"If it does not open automatically, visit:\n  {auth_url}")
        print()

        if not _is_headless():
            _open_browser(auth_url)

        deadline = time.monotonic() + CALLBACK_WAIT_SECONDS
        while time.monotonic() < deadline:
            if server._callback_result is not None:
                break
            time.sleep(0.5)
        else:
            # Timed out — offer paste fallback
            print("Callback server timed out. You can paste the redirect URL instead.")
            return _prompt_paste_fallback()

        result = server._callback_result
        if not result:
            return None

        if result[0] == "error":
            error_msg = result[1] if len(result) > 1 else "unknown_error"
            print(f"OAuth error received: {error_msg}")
            return None

        if result[0] == "success" and len(result) >= 3:
            code = result[1]
            returned_state = result[2]
            if returned_state != state:
                print("OAuth state mismatch. Possible CSRF attack.")
                return None
            return code

        return None
    finally:
        server.shutdown()


def _prompt_paste_fallback() -> Optional[str]:
    """Prompt the user to paste the redirect URL or authorization code."""
    print()
    print("Paste the full redirect URL from your browser's address bar,")
    print("or just the 'code=' parameter value:")
    raw = input("Callback URL or code: ").strip()
    if not raw:
        return None
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        params = urllib.parse.parse_qs(parsed.query)
        return (params.get("code") or [""])[0] or None
    if raw.startswith("?"):
        params = urllib.parse.parse_qs(raw[1:])
        return (params.get("code") or [""])[0] or None
    return raw


def get_auth_status() -> Dict[str, Any]:
    """Return a status dict for auth list / status commands."""
    creds = load_credentials()
    if creds is None:
        return {"logged_in": False, "email": "", "error": ""}
    try:
        email = creds.email or ""
        expires_at = creds.expires_ms
        return {
            "logged_in": True,
            "email": email,
            "expires_at": expires_at,
            "auth_file": str(_credentials_path()),
            "error": "",
        }
    except Exception as exc:
        return {
            "logged_in": False,
            "auth_file": str(_credentials_path()),
            "error": str(exc),
        }
