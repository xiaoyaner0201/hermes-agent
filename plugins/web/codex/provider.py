"""Codex standalone web search + content extraction — Hermes web backend.

Reuses OpenAI Codex's standalone web-retrieval backend
(``https://chatgpt.com/backend-api/codex/alpha/search``) with the OAuth
credentials from ``~/.codex/auth.json`` (created by ``codex login``) or the
``CODEX_ACCESS_TOKEN`` env var. **Zero GPT model inference**: the search
operation is a pure retrieval call; the active Hermes model receives the
raw results and does all the reasoning.

Config keys this provider responds to::

    web:
      backend: "codex"          # shared fallback for both capabilities
      search_backend: "codex"   # per-capability override (optional)
      extract_backend: "codex"  # per-capability override (optional)

Env vars::

    CODEX_ACCESS_TOKEN=...      # optional; overrides ~/.codex/auth.json
    CODEX_ACCOUNT_ID=...        # optional; multi-account setups

Auth resolution order (mirrors the codex-search-opencode plugin):
1. ``CODEX_ACCESS_TOKEN`` (+ optional ``CODEX_ACCOUNT_ID``)
2. ``~/.codex/auth.json`` → ``tokens.access_token`` / ``tokens.account_id``

Privacy: only the search query / opened URL is sent to the retrieval
service — never conversation history, code, or system prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List
from uuid import uuid4

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_ENDPOINT = "https://chatgpt.com/backend-api/codex/alpha/search"
_MODEL = "gpt-4o"  # selects the search backend contract; NOT an inference call
_USER_AGENT = "codex-cli/0.147.0-alpha.6.5"
_TIMEOUT_S = 25
_MAX_RETRIES = 2
_AUTH_PATH = os.path.expanduser("~/.codex/auth.json")

# Output cleanup. The retrieval backend wraps page text in private-use
# citation markers (\ue200cite\ue202<ref>\ue201) and numbers each line
# ("L0: ..."). Patterns use \uXXXX escapes in NON-raw strings so Python
# decodes them to the real codepoints at import time.
_CITE_REF_RE = re.compile("\ue200cite\ue202[^\ue201†]*\ue201")
_CITE_NUM_RE = re.compile("\ue200cite\ue202\\d+†")
_CITE_END_RE = re.compile("\ue201")
_INLINE_LINE_MARKER_RE = re.compile(" (?=L\\d+:)")
_LINE_PREFIX_RE = re.compile(r"^\s*L\d+:\s?", re.MULTILINE)
_LINE_MARKER_RE = re.compile(r"(^|\s)L\d+:")


def _clean_output(output: str) -> str:
    """Strip citation markers, the retrieval header, and line numbers."""
    text = _CITE_REF_RE.sub("", output)
    text = _CITE_NUM_RE.sub("", text)
    text = _CITE_END_RE.sub("", text)

    # Break inline "L<N>:" markers onto their own lines so the prefix strip
    # below catches every marker (the backend occasionally packs several
    # numbered lines onto one physical line after cite removal).
    text = _INLINE_LINE_MARKER_RE.sub("\n", text)

    # Cut everything before the first "L<N>:" page-line marker.
    m = _LINE_MARKER_RE.search(text)
    if m:
        text = text[m.start() :]
    text = _LINE_PREFIX_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class CodexError(Exception):
    """Base error for Codex search backend failures."""


class CodexAuthExpiredError(CodexError):
    """Token rejected by the backend (401/403) — needs a fresh `codex login`."""


def _load_codex_auth() -> tuple[str, str | None]:
    """Resolve (access_token, account_id) from env, then ~/.codex/auth.json."""
    from agent.web_search_provider import get_provider_env

    token = get_provider_env("CODEX_ACCESS_TOKEN")
    account_id = get_provider_env("CODEX_ACCOUNT_ID") or None
    if token:
        return token, account_id

    try:
        with open(_AUTH_PATH, encoding="utf-8") as f:
            data = json.load(f)
        tokens = data.get("tokens") or {}
        access = tokens.get("access_token")
        if isinstance(access, str) and access:
            acct = tokens.get("account_id")
            return access, acct if isinstance(acct, str) and acct else None
    except Exception:  # noqa: BLE001 — missing/invalid auth file => no auth
        pass
    return "", None


def _post(payload: Dict[str, Any], token: str, account_id: str | None) -> Dict[str, Any]:
    """POST to the Codex search endpoint with retries on transient failures."""
    import httpx

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = httpx.post(_ENDPOINT, json=payload, headers=headers, timeout=_TIMEOUT_S)
        except httpx.RequestError as exc:
            last_err = exc
            if attempt < _MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            break

        if resp.status_code in (502, 503, 504):
            last_err = CodexError(f"Codex search backend unavailable (HTTP {resp.status_code})")
            if attempt < _MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            break
        if resp.status_code in (401, 403):
            raise CodexAuthExpiredError(
                "Codex auth rejected (HTTP %d). Refresh with `codex login` "
                "or update CODEX_ACCESS_TOKEN." % resp.status_code
            )
        if resp.status_code == 429:
            raise CodexError("Codex search rate limited (HTTP 429) — try again later.")
        if resp.status_code >= 400:
            raise CodexError(
                f"Codex search returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()

    raise CodexError(f"Codex search request failed: {last_err}")


class CodexWebSearchProvider(WebSearchProvider):
    """Search + extract backend backed by OpenAI Codex's standalone retrieval."""

    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "Codex Search (OpenAI standalone)"

    def is_available(self) -> bool:
        """True when a Codex credential is present (cheap check, no network)."""
        from agent.web_search_provider import get_provider_env

        if get_provider_env("CODEX_ACCESS_TOKEN"):
            return True
        try:
            with open(_AUTH_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return bool((data.get("tokens") or {}).get("access_token"))
        except Exception:  # noqa: BLE001 — any failure means "not available"
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Single-query lookup against the Codex retrieval backend.

        Returns ``{"success": True, "data": {"web": [{"title", "url",
        "description", "position"}]}}`` or ``{"success": False, "error"}``.
        """
        token, account_id = _load_codex_auth()
        if not token:
            return {
                "success": False,
                "error": "Codex auth not found. Run `codex login` or set CODEX_ACCESS_TOKEN.",
            }

        payload = {
            "id": f"hermes_{uuid4().hex[:12]}",
            "model": _MODEL,
            "commands": {"search_query": [{"q": query}]},
        }
        try:
            body = _post(payload, token, account_id)
        except CodexError as exc:
            logger.warning("Codex search failed: %s", exc)
            return {"success": False, "error": str(exc)}

        count = max(1, min(int(limit), 20))
        web: List[Dict[str, Any]] = []
        for i, r in enumerate((body.get("results") or [])[:count]):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            if not title and not url:
                continue
            web.append(
                {
                    "title": title,
                    "url": url,
                    "description": (r.get("snippet") or "").strip(),
                    "position": i + 1,
                }
            )

        logger.info("Codex search '%s': %d results (limit %d)", query, len(web), limit)
        return {"success": True, "data": {"web": web}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Open each URL through the Codex retrieval backend and return page text.

        Each URL is fetched with the ``open`` command using the URL itself as
        the reference id (validated live against the endpoint). Returns the
        list shape the ``web_extract`` wrapper expects.
        """
        token, account_id = _load_codex_auth()
        if not token:
            msg = "Codex auth not found. Run `codex login` or set CODEX_ACCESS_TOKEN."
            return [{"url": u, "error": msg} for u in urls]

        out: List[Dict[str, Any]] = []
        for url in urls:
            payload = {
                "id": f"hermes_{uuid4().hex[:12]}",
                "model": _MODEL,
                "commands": {"open": [{"ref_id": url}], "response_length": "long"},
            }
            try:
                body = _post(payload, token, account_id)
            except CodexError as exc:
                logger.warning("Codex extract %s failed: %s", url, exc)
                out.append({"url": url, "error": str(exc)})
                continue

            raw = body.get("output") or ""
            results = body.get("results") or []
            title = (results[0].get("title") or "").strip() if results else ""
            ref_id = results[0].get("ref_id") if results else None
            # The web_extract wrapper consumes ``raw_content`` (falling back
            # to ``content``) as the page text, so both carry the cleaned
            # text — the cite markers / line numbers are protocol artifacts,
            # not page content.
            cleaned = _clean_output(raw)
            out.append(
                {
                    "url": url,
                    "title": title,
                    "content": cleaned,
                    "raw_content": cleaned,
                    "metadata": {"ref_id": ref_id} if ref_id else {},
                }
            )
            logger.info("Codex extract %s: %d chars (cleaned %d)", url, len(raw), len(cleaned))
        return out

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Codex Search (OpenAI standalone)",
            "badge": "free",
            "tag": "Reuses your `codex login` session — zero GPT tokens, search + extract.",
            "env_vars": [
                {
                    "key": "CODEX_ACCESS_TOKEN",
                    "prompt": "Codex access token (optional — defaults to ~/.codex/auth.json)",
                    "url": "https://github.com/openai/codex",
                },
            ],
        }
