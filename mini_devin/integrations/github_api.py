"""
Shared GitHub REST helpers (used by ``server.py`` and any other entrypoints).

Keeps a single implementation of retry + error shaping instead of duplicating
httpx logic in the monolithic server.
"""

from __future__ import annotations

import json
import time
from typing import Any, List, Optional, Tuple

import httpx


def get_github_error_suggestions(status_code: int, error_message: str) -> List[str]:
    """Actionable suggestions based on GitHub API error."""
    suggestions: List[str] = []
    error_lower = error_message.lower()

    if status_code == 401:
        suggestions.append("Check if your GitHub token is valid and not expired")
        suggestions.append("Generate a new token at github.com/settings/tokens")
    elif status_code == 403:
        if "rate limit" in error_lower:
            suggestions.append("GitHub API rate limit exceeded. Wait a few minutes and try again")
            suggestions.append("Consider using a token with higher rate limits")
        elif "abuse" in error_lower:
            suggestions.append("GitHub abuse detection triggered. Wait and retry with smaller requests")
        else:
            suggestions.append("Check if your token has the required permissions/scopes")
            suggestions.append("For Actions: enable 'actions:read' scope")
            suggestions.append("For PRs/Issues: enable 'repo' scope")
            suggestions.append("For fine-grained tokens: grant access to this specific repository")
    elif status_code == 404:
        suggestions.append("Check if the repository exists and is accessible")
        suggestions.append("Verify the owner/repo name is correct")
        suggestions.append("For private repos: ensure your token has access")
    elif status_code == 422:
        suggestions.append("Check the request parameters are valid")
        if "branch" in error_lower:
            suggestions.append("Verify the branch name exists")
        if "already exists" in error_lower:
            suggestions.append("The resource already exists")

    return suggestions


def execute_github_api(
    method: str,
    endpoint: str,
    token: str,
    data: Optional[dict] = None,
    max_retries: int = 3,
) -> Tuple[bool, Any]:
    """GitHub API request with retries. Returns ``(ok, json_or_error_dict)``."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                url = f"https://api.github.com{endpoint}"
                if method == "GET":
                    response = client.get(url, headers=headers)
                elif method == "POST":
                    response = client.post(url, headers=headers, json=data or {})
                elif method == "PATCH":
                    response = client.patch(url, headers=headers, json=data or {})
                elif method == "PUT":
                    response = client.put(url, headers=headers, json=data or {})
                elif method == "DELETE":
                    response = client.delete(url, headers=headers)
                else:
                    return False, {"error": f"Unknown method: {method}"}

                if response.status_code in [200, 201, 204]:
                    if response.status_code == 204:
                        return True, {"message": "Success"}
                    return True, response.json()
                if response.status_code == 429 or (
                    response.status_code == 403 and "rate limit" in response.text.lower()
                ):
                    retry_after = int(response.headers.get("Retry-After", 60))
                    if attempt < max_retries - 1:
                        time.sleep(min(retry_after, 30))
                        continue
                elif response.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue

                try:
                    error_data = response.json() if response.text else {"message": "Unknown error"}
                except json.JSONDecodeError:
                    error_data = {"message": response.text[:500] or "Unknown error"}
                error_msg = error_data.get("message", str(error_data))
                suggestions = get_github_error_suggestions(response.status_code, error_msg)
                return False, {
                    "error": error_msg,
                    "status_code": response.status_code,
                    "suggestions": suggestions,
                }
        except httpx.TimeoutException:
            last_error = "GitHub API request timed out"
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    return False, {"error": last_error or "Request failed after retries"}
