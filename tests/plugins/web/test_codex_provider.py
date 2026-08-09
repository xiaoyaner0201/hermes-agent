"""Tests for the bundled Codex web search + extract plugin.

Covers:

- Registration: the plugin registers a ``codex`` provider with search +
  extract capabilities through the plugin registry.
- ``is_available()``: reflects ``CODEX_ACCESS_TOKEN`` or a valid
  ``~/.codex/auth.json`` — cheap file/env checks, no network.
- Auth resolution order: env var wins over the auth file; a missing or
  malformed auth file degrades to no auth.
- Output cleanup: citation markers (private-use codepoints), numbered
  ``L<N>:`` line markers, and the retrieval header are stripped.
- ``search()`` / ``extract()`` envelopes: response shapes match the
  documented ``WebSearchProvider`` contract; failures map to
  ``{"success": False, "error": ...}`` / per-URL errors.
- Transport: transient HTTP failures are retried; 401/403 and 429 map
  to actionable errors.

Per the dev skill, these tests use *real* imports from the plugin module —
only the network hop (``httpx.post``) is faked.
"""

from __future__ import annotations

import json

import httpx
import pytest

from plugins.web.codex import provider as codex_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_codex_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every Codex credential env var."""
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CODEX_ACCOUNT_ID", raising=False)


@pytest.fixture(autouse=True)
def _clean_codex_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no Codex credentials — env or auth file."""
    _clear_codex_env(monkeypatch)
    # Neutralize the real ~/.codex/auth.json on developer machines.
    monkeypatch.setattr(
        codex_provider, "_AUTH_PATH", "/nonexistent/.codex/auth.json"
    )


def _ensure_plugins_loaded() -> None:
    """Idempotently load plugins so the registry is populated."""
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()


def _write_auth_file(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    token: str = "tok-file",
    account_id: str | None = "acct-file",
    malformed: bool = False,
) -> str:
    """Write a ~/.codex/auth.json under tmp_path and point the module at it."""
    auth_dir = tmp_path / ".codex"
    auth_dir.mkdir()
    auth_path = auth_dir / "auth.json"
    if malformed:
        auth_path.write_text("{not json", encoding="utf-8")
    else:
        tokens: dict[str, str] = {"access_token": token}
        if account_id is not None:
            tokens["account_id"] = account_id
        auth_path.write_text(
            json.dumps({"tokens": tokens}), encoding="utf-8"
        )
    monkeypatch.setattr(codex_provider, "_AUTH_PATH", str(auth_path))
    return str(auth_path)


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response``."""

    def __init__(self, status_code: int, payload: dict | str) -> None:
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self._payload = payload

    def json(self) -> dict:
        if isinstance(self._payload, dict):
            return self._payload
        raise ValueError(f"not JSON: {self.text!r}")


class _FakePost:
    """Queued ``httpx.post`` stand-in: each call pops the next response."""

    def __init__(self, responses: list[tuple[int, dict | str]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, json: dict | None = None, **kwargs) -> _FakeResponse:
        self.calls.append((url, json or {}))
        status, payload = self._responses.pop(0)
        return _FakeResponse(status, payload)


# ---------------------------------------------------------------------------
# Registration + capability flags
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_codex_provider_registered(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        provider = get_provider("codex")
        assert provider is not None, "codex plugin not registered"
        assert provider.name == "codex"
        assert provider.display_name

    def test_codex_supports_search_and_extract(self) -> None:
        _ensure_plugins_loaded()
        from agent.web_search_registry import get_provider

        provider = get_provider("codex")
        assert provider is not None
        assert provider.supports_search() is True
        assert provider.supports_extract() is True


# ---------------------------------------------------------------------------
# is_available() — cheap checks only, no network
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_not_available_without_credentials(self) -> None:
        provider = codex_provider.CodexWebSearchProvider()
        assert provider.is_available() is False

    def test_available_with_env_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok-env")
        provider = codex_provider.CodexWebSearchProvider()
        assert provider.is_available() is True

    def test_available_with_auth_file(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_auth_file(tmp_path, monkeypatch)
        provider = codex_provider.CodexWebSearchProvider()
        assert provider.is_available() is True

    def test_auth_file_without_tokens_is_not_available(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_auth_file(tmp_path, monkeypatch, token="", account_id=None)
        provider = codex_provider.CodexWebSearchProvider()
        assert provider.is_available() is False

    def test_malformed_auth_file_is_not_available(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_auth_file(tmp_path, monkeypatch, malformed=True)
        provider = codex_provider.CodexWebSearchProvider()
        assert provider.is_available() is False


# ---------------------------------------------------------------------------
# Auth resolution order
# ---------------------------------------------------------------------------


class TestAuthResolution:
    def test_env_token_wins_over_auth_file(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_auth_file(tmp_path, monkeypatch, token="tok-file")
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok-env")
        monkeypatch.setenv("CODEX_ACCOUNT_ID", "acct-env")

        token, account_id = codex_provider._load_codex_auth()
        assert token == "tok-env"
        assert account_id == "acct-env"

    def test_auth_file_fallback(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_auth_file(tmp_path, monkeypatch, token="tok-file", account_id="acct-file")

        token, account_id = codex_provider._load_codex_auth()
        assert token == "tok-file"
        assert account_id == "acct-file"

    def test_no_credentials_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            codex_provider, "_AUTH_PATH", "/nonexistent/.codex/auth.json"
        )
        assert codex_provider._load_codex_auth() == ("", None)


# ---------------------------------------------------------------------------
# Output cleanup
# ---------------------------------------------------------------------------


class TestCleanOutput:
    def test_strips_citation_ref_markers(self) -> None:
        raw = "\ue200cite\ue202https://example.com\ue201L0: Body"
        assert codex_provider._clean_output(raw) == "Body"

    def test_strips_numbered_citation_markers(self) -> None:
        raw = "text \ue200cite\ue2025†more"
        assert codex_provider._clean_output(raw) == "text more"

    def test_strips_dangling_end_marker(self) -> None:
        assert codex_provider._clean_output("\ue201L0: Body") == "Body"

    def test_cuts_header_before_first_line_marker(self) -> None:
        raw = "Retrieval summary prose that is not page content.\nL0: Real page"
        assert codex_provider._clean_output(raw) == "Real page"

    def test_strips_line_number_prefixes(self) -> None:
        raw = "L0: Alpha\nL1: Beta\nL2: Gamma"
        assert codex_provider._clean_output(raw) == "Alpha\nBeta\nGamma"

    def test_splits_inline_line_markers(self) -> None:
        raw = "L0: Alpha L1: Beta"
        assert codex_provider._clean_output(raw) == "Alpha\nBeta"

    def test_collapses_excess_blank_lines(self) -> None:
        # Blank lines in the MIDDLE of content collapse to at most one blank.
        raw = "L0: Alpha\n\n\n\nBody text"
        assert codex_provider._clean_output(raw) == "Alpha\n\nBody text"
        # Blank lines immediately before a line marker are eaten by the
        # prefix strip — no leading noise after the marker.
        raw = "L0: Alpha\n\n\n\nL1: Beta"
        assert codex_provider._clean_output(raw) == "Alpha\nBeta"

    def test_combined_realistic_sample(self) -> None:
        raw = (
            "Here is what I found about the topic.\n"
            "\ue200cite\ue2021\ue201L0: Hermes Agent is an open-source agent "
            "\ue200cite\ue2022†framework L1: by Nous Research"
        )
        assert codex_provider._clean_output(raw) == (
            "Hermes Agent is an open-source agent framework\nby Nous Research"
        )


# ---------------------------------------------------------------------------
# search() envelope
# ---------------------------------------------------------------------------


class TestSearch:
    def test_success_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            codex_provider,
            "_post",
            lambda *a, **kw: {
                "results": [
                    {"title": "One", "url": "https://one.example", "snippet": "s1"},
                    {"title": "Two", "url": "https://two.example", "snippet": "s2"},
                ]
            },
        )
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        result = codex_provider.CodexWebSearchProvider().search("query")
        assert result["success"] is True
        web = result["data"]["web"]
        assert web == [
            {
                "title": "One",
                "url": "https://one.example",
                "description": "s1",
                "position": 1,
            },
            {
                "title": "Two",
                "url": "https://two.example",
                "description": "s2",
                "position": 2,
            },
        ]

    def test_limit_clamps_result_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = [
            {"title": f"R{i}", "url": f"https://r{i}.example", "snippet": ""}
            for i in range(4)
        ]
        monkeypatch.setattr(
            codex_provider, "_post", lambda *a, **kw: {"results": results}
        )
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        provider = codex_provider.CodexWebSearchProvider()
        assert len(provider.search("q", limit=2)["data"]["web"]) == 2
        # limit=0 clamps up to 1 (the backend rejects 0)
        assert len(provider.search("q", limit=0)["data"]["web"]) == 1

    def test_skips_entries_without_title_and_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            codex_provider,
            "_post",
            lambda *a, **kw: {
                "results": [
                    {"title": "", "url": "", "snippet": "empty"},
                    {"title": "Real", "url": "https://real.example", "snippet": "s"},
                ]
            },
        )
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        web = codex_provider.CodexWebSearchProvider().search("q")["data"]["web"]
        assert len(web) == 1
        assert web[0]["title"] == "Real"

    def test_missing_auth_returns_actionable_error(self) -> None:
        result = codex_provider.CodexWebSearchProvider().search("q")
        assert result["success"] is False
        assert "codex login" in result["error"]

    def test_expired_auth_returns_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a, **kw):
            raise codex_provider.CodexAuthExpiredError(
                "Codex auth rejected (HTTP 401). Refresh with `codex login`."
            )

        monkeypatch.setattr(codex_provider, "_post", _boom)
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "stale")

        result = codex_provider.CodexWebSearchProvider().search("q")
        assert result["success"] is False
        assert "codex login" in result["error"]

    def test_backend_error_maps_to_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _rate_limited(*a, **kw):
            raise codex_provider.CodexError("Codex search rate limited (HTTP 429)")

        monkeypatch.setattr(codex_provider, "_post", _rate_limited)
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        result = codex_provider.CodexWebSearchProvider().search("q")
        assert result["success"] is False
        assert "429" in result["error"]


# ---------------------------------------------------------------------------
# extract() envelope
# ---------------------------------------------------------------------------


class TestExtract:
    def test_success_cleans_output_into_both_content_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            codex_provider,
            "_post",
            lambda *a, **kw: {
                "output": "\ue200cite\ue2021\ue201L0: Page body text",
                "results": [{"title": "Page Title", "ref_id": "ref-1"}],
            },
        )
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        out = codex_provider.CodexWebSearchProvider().extract(["https://a.example"])
        assert out[0]["url"] == "https://a.example"
        assert out[0]["title"] == "Page Title"
        # The web_extract wrapper consumes raw_content first — both must be clean.
        assert out[0]["content"] == "Page body text"
        assert out[0]["raw_content"] == "Page body text"
        assert out[0]["metadata"] == {"ref_id": "ref-1"}
        assert "error" not in out[0]

    def test_per_url_error_keeps_other_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _flaky(*a, **kw):
            raise codex_provider.CodexError("Codex search backend unavailable (HTTP 503)")

        monkeypatch.setattr(codex_provider, "_post", _flaky)
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        out = codex_provider.CodexWebSearchProvider().extract(
            ["https://a.example", "https://b.example"]
        )
        assert len(out) == 2
        assert all("error" in item and "503" in item["error"] for item in out)

    def test_missing_auth_returns_per_url_error(self) -> None:
        out = codex_provider.CodexWebSearchProvider().extract(["https://a.example"])
        assert out[0]["url"] == "https://a.example"
        assert "codex login" in out[0]["error"]


# ---------------------------------------------------------------------------
# Transport: retries + error mapping (httpx.post faked)
# ---------------------------------------------------------------------------


class TestTransport:
    def test_retries_transient_failures_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePost([(503, "unavailable"), (503, "unavailable"), (200, {"results": []})])
        monkeypatch.setattr(httpx, "post", fake)
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        body = codex_provider._post({"id": "x"}, "tok", None)
        assert body == {"results": []}
        assert len(fake.calls) == 3

    def test_gives_up_after_max_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePost([(503, "unavailable")] * 3)
        monkeypatch.setattr(httpx, "post", fake)
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        with pytest.raises(codex_provider.CodexError, match="unavailable"):
            codex_provider._post({"id": "x"}, "tok", None)
        assert len(fake.calls) == 3

    def test_401_raises_auth_expired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePost([(401, "unauthorized")])
        monkeypatch.setattr(httpx, "post", fake)

        with pytest.raises(codex_provider.CodexAuthExpiredError, match="codex login"):
            codex_provider._post({"id": "x"}, "tok", None)

    def test_429_raises_rate_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePost([(429, "slow down")])
        monkeypatch.setattr(httpx, "post", fake)

        with pytest.raises(codex_provider.CodexError, match="rate limited"):
            codex_provider._post({"id": "x"}, "tok", None)

    def test_request_error_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def _flaky_post(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom", request=None)
            return _FakeResponse(200, {"results": []})

        monkeypatch.setattr(httpx, "post", _flaky_post)
        monkeypatch.setenv("CODEX_ACCESS_TOKEN", "tok")

        body = codex_provider._post({"id": "x"}, "tok", None)
        assert body == {"results": []}
        assert calls["n"] == 2
