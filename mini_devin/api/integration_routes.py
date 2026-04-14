"""
External integration stubs (Slack, etc.).

Slack app URL verification: https://api.slack.com/events/url_verification
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from mini_devin.auth.dependencies import get_current_user
from mini_devin.auth.enterprise_deps import RequirePermission, user_to_role
from mini_devin.database.models import UserModel
from mini_devin.enterprise.rbac import Permission, role_allows

router = APIRouter(prefix="/integrations", tags=["integrations"])


def _verify_slack_signature(signing_secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    if not timestamp or not signature or not signing_secret:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(int(time.time()) - ts) > 60 * 5:
        return False
    basestring = f"v0:{timestamp}:{body.decode('utf-8', errors='replace')}".encode()
    my_sig = "v0=" + hmac.new(signing_secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_sig, signature)


@router.get("/status")
async def integrations_status() -> dict[str, Any]:
    """Whether optional integration env vars are present (no secrets returned)."""
    return {
        "slack_signing_configured": bool((os.environ.get("SLACK_SIGNING_SECRET") or "").strip()),
    }


@router.get("/me/permissions")
async def my_permissions(current_user: UserModel = Depends(get_current_user)) -> dict[str, Any]:
    """Current user's derived role and granted permissions (for UI gating)."""
    role = user_to_role(current_user)
    perms = [p.value for p in Permission if role_allows(role, p)]
    return {"role": role.value, "permissions": perms}


@router.post("/slack/events", response_model=None)
async def slack_events(request: Request) -> dict[str, Any] | JSONResponse:
    """
    Slack Events API receiver (stub).

    Handles ``url_verification`` challenges. If ``SLACK_SIGNING_SECRET`` is set,
    validates ``X-Slack-Request-Timestamp`` and ``X-Slack-Signature``.
    """
    raw = await request.body()
    secret = (os.environ.get("SLACK_SIGNING_SECRET") or "").strip()
    if secret:
        ts = request.headers.get("X-Slack-Request-Timestamp", "")
        sig = request.headers.get("X-Slack-Signature", "")
        if not _verify_slack_signature(secret, ts, raw, sig):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Slack signature")

    try:
        data: dict[str, Any] = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")

    if data.get("type") == "url_verification":
        challenge = data.get("challenge")
        if challenge is None:
            raise HTTPException(status_code=400, detail="Missing challenge")
        return JSONResponse({"challenge": challenge})

    return {
        "ok": True,
        "received_type": data.get("type"),
        "note": "Event stored nowhere; add your worker to forward to Plodder sessions.",
    }


@router.post("/slack/notify")
async def slack_notify_stub(
    _user: UserModel = Depends(RequirePermission(Permission.MANAGE_INTEGRATIONS)),
) -> dict[str, str]:
    """Authenticated stub for outbound Slack-style notifications (implement webhook POST here)."""
    return {"status": "not_implemented", "hint": "POST to your Slack incoming webhook from this handler."}
