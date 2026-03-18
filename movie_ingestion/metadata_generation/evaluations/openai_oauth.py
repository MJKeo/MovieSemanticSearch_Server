"""
OpenAI OAuth2 PKCE token management for the evaluation pipeline.

Handles the full lifecycle of ChatGPT OAuth tokens for accessing the WHAM
backend (chatgpt.com/backend-api/wham/v1), which exposes Codex models
like gpt-5.4 via a ChatGPT subscription.

Flow:
  1. Browser-based PKCE authorization → access_token + refresh_token
  2. JWT decode to extract account_id and expiry (no RFC 8693 exchange needed)
  3. Token refresh when expired (single POST, no browser)

Main entry point: get_valid_auth() — returns an OpenAIOAuthTokens with
access_token and account_id needed for WHAM API calls.

Uses the Codex CLI's public OAuth client_id. The redirect URI is fixed
to http://localhost:1455/auth/callback (tied to the client_id registration).
"""

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKEN_FILE_PATH = Path("evaluation_data/openai_oauth_tokens.json")

# OpenAI OAuth endpoints (Codex CLI / Auth0-backed)
AUTH_ENDPOINT = "https://auth.openai.com/oauth/authorize"
TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"

# Codex CLI's public OAuth client_id — the only usable client_id as OpenAI
# does not currently offer third-party app registration.
CLIENT_ID = os.getenv(
    "OPENAI_OAUTH_CLIENT_ID",
    "app_EMoamEEZ73f0CkXaXp7hrann",
)

# Fixed redirect URI tied to the Codex CLI client_id registration.
# Must match exactly — a different port or path will be rejected.
REDIRECT_URI = "http://localhost:1455/auth/callback"
REDIRECT_PORT = 1455
REDIRECT_PATH = "/auth/callback"

# OpenID Connect scopes; offline_access is required for refresh tokens.
SCOPES = os.getenv(
    "OPENAI_OAUTH_SCOPES",
    "openid profile email offline_access",
)

# Refresh buffer: treat token as expired if it expires within this many seconds.
# Prevents a token from expiring mid-pipeline-run.
EXPIRY_BUFFER_SECONDS = 1200


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OpenAIOAuthTokens:
    access_token: str
    refresh_token: str
    account_id: str
    # Unix timestamp at which the access_token expires
    expires_at: float


# ---------------------------------------------------------------------------
# JWT decode (no signature verification, matching Codex CLI pattern)
# ---------------------------------------------------------------------------

def _decode_jwt_claims(token: str) -> dict:
    """Decode the payload segment of a JWT without verifying the signature.

    This matches the Codex CLI's approach — the token is already delivered
    over HTTPS from OpenAI's auth server, so signature verification is
    unnecessary for extracting claims.
    """
    payload_segment = token.split(".")[1]
    # Add padding — JWT base64url omits trailing '='
    payload_segment += "=" * (-len(payload_segment) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_segment))


def _extract_account_id(claims: dict) -> str:
    """Extract the ChatGPT account ID from JWT claims with 3-level fallback.

    Tries (in order):
      1. claims["https://api.openai.com/profile"]["accountId"]
      2. claims["chatgpt_account_id"]
      3. claims["sub"]
    """
    profile = claims.get("https://api.openai.com/profile", {})
    account_id = profile.get("accountId")
    if account_id:
        return account_id

    account_id = claims.get("chatgpt_account_id")
    if account_id:
        return account_id

    account_id = claims.get("sub")
    if account_id:
        return account_id

    raise RuntimeError(
        f"Could not extract account_id from JWT claims. "
        f"Available keys: {list(claims.keys())}"
    )


def _build_tokens_from_response(
    access_token: str,
    refresh_token: str,
) -> OpenAIOAuthTokens:
    """Decode the access_token JWT and build an OpenAIOAuthTokens instance.

    Extracts expiry (exp claim) and account_id from the JWT payload.
    """
    claims = _decode_jwt_claims(access_token)
    account_id = _extract_account_id(claims)

    # Use the JWT 'exp' claim for expiry; fall back to 1 hour from now
    expires_at = claims.get("exp", time.time() + 3600)

    return OpenAIOAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        account_id=account_id,
        expires_at=float(expires_at),
    )


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------

def load_tokens() -> Optional[OpenAIOAuthTokens]:
    """Read tokens from disk. Returns None if the file does not exist."""
    if not TOKEN_FILE_PATH.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE_PATH.read_text())
        return OpenAIOAuthTokens(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            account_id=data["account_id"],
            expires_at=float(data["expires_at"]),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Token file at {TOKEN_FILE_PATH} is malformed: {e}. "
            "Delete it and re-run to trigger a fresh auth flow."
        ) from e


def save_tokens(tokens: OpenAIOAuthTokens) -> None:
    """Atomically write tokens to disk (write-then-rename for crash safety)."""
    TOKEN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_FILE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "account_id": tokens.account_id,
        "expires_at": tokens.expires_at,
    }, indent=2))
    tmp.replace(TOKEN_FILE_PATH)


# ---------------------------------------------------------------------------
# Expiry check
# ---------------------------------------------------------------------------

def is_token_expired(
    tokens: OpenAIOAuthTokens,
    buffer_seconds: int = EXPIRY_BUFFER_SECONDS,
) -> bool:
    """Return True if the access token will expire within buffer_seconds."""
    return tokens.expires_at < (time.time() + buffer_seconds)


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

def refresh_access_token(refresh_token: str) -> OpenAIOAuthTokens:
    """Refresh credentials via a single POST to the token endpoint.

    Returns a new OpenAIOAuthTokens with the refreshed access_token,
    account_id decoded from the JWT, and updated expiry.

    Raises RuntimeError if the request fails.
    """
    response = httpx.post(
        TOKEN_ENDPOINT,
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Token refresh failed (HTTP {response.status_code}): {response.text}"
        )

    data = response.json()
    new_access_token = data.get("access_token")
    if not new_access_token:
        raise RuntimeError(
            f"Refresh response missing access_token. Keys: {list(data.keys())}"
        )
    new_refresh_token = data.get("refresh_token", refresh_token)

    return _build_tokens_from_response(new_access_token, new_refresh_token)


# ---------------------------------------------------------------------------
# Initial auth flow (PKCE)
# ---------------------------------------------------------------------------

def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a (code_verifier, code_challenge) pair for PKCE."""
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def run_initial_auth_flow() -> OpenAIOAuthTokens:
    """Perform the browser-based PKCE authorization code flow.

    Opens the user's browser to OpenAI's consent screen, starts an HTTP
    server on the fixed redirect port (1455) to capture the callback, then
    exchanges the authorization code for tokens.

    Raises RuntimeError on failure (user cancels, timeout, network error).
    """
    code_verifier, code_challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    # Build the authorization URL with OpenAI-specific parameters
    auth_params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        # OpenAI-specific params required for the Codex CLI flow
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    auth_url = f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(auth_params)}"

    # Shared state between the HTTP handler and this function
    auth_code: list[str] = []   # list so it's mutable from inside the closure
    received_state: list[str] = []
    done_event = Event()

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != REDIRECT_PATH:
                self._respond(404, "Not found")
                return

            params = dict(urllib.parse.parse_qsl(parsed.query))
            if "error" in params:
                self._respond(
                    400,
                    f"Authorization error: {params.get('error_description', params['error'])}"
                )
                done_event.set()
                return

            auth_code.append(params.get("code", ""))
            received_state.append(params.get("state", ""))
            self._respond(
                200,
                "Authorization successful — you can close this tab and return to the terminal."
            )
            done_event.set()

        def _respond(self, status: int, body: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args) -> None:
            # Suppress default access log noise
            pass

    server = HTTPServer(("127.0.0.1", REDIRECT_PORT), _CallbackHandler)
    server.timeout = 120  # Give the user 2 minutes to complete the browser flow

    print(f"\nOpening browser for OpenAI OAuth authorization...")
    print(f"If the browser does not open automatically, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    # Serve requests until the callback is received or timeout occurs
    while not done_event.is_set():
        server.handle_request()
    server.server_close()

    if not auth_code or not auth_code[0]:
        raise RuntimeError("No authorization code received — auth flow failed or was cancelled.")

    if received_state[0] != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF. Aborting.")

    # Exchange the authorization code for OIDC tokens
    response = httpx.post(
        TOKEN_ENDPOINT,
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": auth_code[0],
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed (HTTP {response.status_code}): {response.text}"
        )

    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError(
            f"Auth code exchange missing access_token. Keys: {list(data.keys())}"
        )
    pkce_refresh_token = data.get("refresh_token")
    if not pkce_refresh_token:
        raise RuntimeError(
            f"Auth code exchange missing refresh_token. Keys: {list(data.keys())}"
        )

    # Decode JWT to extract account_id and expiry — no second exchange needed
    # for the WHAM backend (it uses the raw access_token directly).
    return _build_tokens_from_response(access_token, pkce_refresh_token)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_valid_auth() -> OpenAIOAuthTokens:
    """Return valid OpenAI OAuth credentials, refreshing or re-acquiring as needed.

    Returns the full OpenAIOAuthTokens (access_token + account_id) since
    WHAM API calls require both.

    Lifecycle:
      1. Load tokens from TOKEN_FILE_PATH.
      2. If no file exists, run the browser-based initial auth flow and save.
      3. If expired (within EXPIRY_BUFFER_SECONDS), refresh and save.
      4. Return the tokens.

    Raises RuntimeError if auth or refresh fails.
    """
    tokens = load_tokens()

    if tokens is None:
        print("No OpenAI OAuth tokens found — starting initial authorization flow...")
        tokens = run_initial_auth_flow()
        save_tokens(tokens)
        print("Tokens saved. Authorization complete.")
        return tokens

    if is_token_expired(tokens):
        print("OpenAI OAuth access token is expired — refreshing...")
        tokens = refresh_access_token(tokens.refresh_token)
        save_tokens(tokens)
        print("Token refreshed and saved.")

    return tokens
