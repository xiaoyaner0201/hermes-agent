"""Codex web search + extract plugin — bundled, auto-loaded.

Backed by OpenAI Codex's standalone web-retrieval endpoint. Auth comes from
``~/.codex/auth.json`` (created by ``codex login``) or the
``CODEX_ACCESS_TOKEN`` env var. Retrieval only — zero GPT inference tokens;
the active Hermes model does all the reasoning on the raw results.
"""

from __future__ import annotations

from plugins.web.codex.provider import CodexWebSearchProvider


def register(ctx) -> None:
    """Register the Codex provider with the plugin context."""
    ctx.register_web_search_provider(CodexWebSearchProvider())
