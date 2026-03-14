"""Routing map for Force:Object conditional edges."""

from __future__ import annotations

FORCE_OBJECT_ROUTES: dict[str, str] = {
    "Request:Review": "review_agent",
    "Inform:Status": "status_agent",
    "Fallback:Generic": "fallback_agent",
}

DEFAULT_ROUTE = "fallback_agent"
