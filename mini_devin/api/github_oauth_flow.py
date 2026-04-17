"""
Minimal GitHub OAuth (web app) for self-hosted Plodder.

Env:
  GITHUB_CLIENT_ID
  GITHUB_CLIENT_SECRET
  GITHUB_OAUTH_REDIRECT_URI — must match the callback URL registered on the GitHub OAuth app
    (e.g. http://127.0.0.1:8000/api/github/oauth/callback)

Flow:
  1) GET /api/github/oauth/start → JSON ``{ "authorize_url", "state" }`` (or 501 if not configured)
  2) User opens ``authorize_url`` in a browser, approves, GitHub redirects to callback with ``code``
  3) GET /api/github/oauth/callback?code=&state= → exchanges code; stores token keyed by ``state`` (short TTL)
  4) GET /api/github/oauth/result?state= → returns ``{ "access_token": "..." }`` **once** (then deleted)

The SPA should call ``/result`` and then send the token to ``POST /api/repos/{id}/token`` or set ``GITHUB_TOKEN`` server-side.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_OAUTH_STATE_TTL_SEC = 900.0
_oauth_states: dict[str, float] = {}
_oauth_redirect_by_state: dict[str, str] = {}
_oauth_tokens: dict[str, str] = {}


def _purge_expired_states() -> None:
    now = time.time()
    dead = [s for s, exp in _oauth_states.items() if exp < now]
    for s in dead:
        _oauth_states.pop(s, None)


def oauth_start_payload(request_base_url: str) -> dict[str, Any]:
    """Build ``authorize_url`` + ``state`` or raise ``RuntimeError`` if not configured."""
    cid = (os.getenv("GITHUB_CLIENT_ID") or "").strip()
    secret = (os.getenv("GITHUB_CLIENT_SECRET") or "").strip()
    if not cid or not secret:
        raise RuntimeError("GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET must be set")

    redirect = (os.getenv("GITHUB_OAUTH_REDIRECT_URI") or "").strip()
    if not redirect:
        base = request_base_url.rstrip("/")
        redirect = f"{base}/api/github/oauth/callback"

    _purge_expired_states()
    state = secrets.token_urlsafe(24)
    _oauth_states[state] = time.time() + _OAUTH_STATE_TTL_SEC
    _oauth_redirect_by_state[state] = redirect

    params = urllib.parse.urlencode(
        {
            "client_id": cid,
            "redirect_uri": redirect,
            "scope": "repo read:user",
            "state": state,
            "allow_signup": "false",
        }
    )
    url = f"https://github.com/login/oauth/authorize?{params}"
    return {"authorize_url": url, "state": state, "redirect_uri": redirect}


def oauth_exchange_callback(code: str, state: str) -> str:
    """Exchange ``code`` for access token; store under ``state`` for one-shot retrieval. Returns token."""
    cid = (os.getenv("GITHUB_CLIENT_ID") or "").strip()
    secret = (os.getenv("GITHUB_CLIENT_SECRET") or "").strip()
    if not cid or not secret:
        raise RuntimeError("OAuth not configured")

    exp = _oauth_states.get(state)
    if exp is None or exp < time.time():
        raise ValueError("invalid or expired state")

    redirect = _oauth_redirect_by_state.get(state) or (os.getenv("GITHUB_OAUTH_REDIRECT_URI") or "").strip()
    if not redirect:
        raise ValueError("redirect_uri missing for this OAuth state — restart /api/github/oauth/start")

    body = urllib.parse.urlencode(
        {
            "client_id": cid,
            "client_secret": secret,
            "code": code,
            "redirect_uri": redirect,
        }
    ).encode()

    req = urllib.request.Request(
        "https://github.com/login/oauth/access_token",
        data=body,
        headers={"Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise ValueError(e.read().decode(errors="replace")[:500]) from e

    token = (data.get("access_token") or "").strip()
    if not token:
        raise ValueError(data.get("error_description") or data.get("error") or "no access_token in response")

    _oauth_states.pop(state, None)
    _oauth_redirect_by_state.pop(state, None)
    _oauth_tokens[state] = token
    return token


def oauth_consume_result(state: str) -> str | None:
    """Return token once for ``state``."""
    tok = _oauth_tokens.pop(state, None)
    return tok
