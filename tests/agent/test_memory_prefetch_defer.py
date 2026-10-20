"""External recall deadlines and opt-in, same-conversation late-result delivery."""

import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.turn_context import _memory_turn_start_and_prefetch
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@dataclass
class Lookup:
    result: object = "memory:A"
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def finish(self):
        self.release.set()
        if self.thread is not None:
            self.thread.join(5)
            assert not self.thread.is_alive()


class ControlledProvider(MemoryProvider):
    def __init__(self, name="controlled"):
        self._name = name
        self.lookups = {}
        self.calls = []
        self.queued = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.ended = 0
        self.closed = 0

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def block(self, query, result="memory:A"):
        lookup = self.lookups[query] = Lookup(result)
        return lookup

    def prefetch(self, query, *, session_id=""):
        with self.lock:
            self.calls.append((query, session_id))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            lookup = self.lookups.get(query)
            if lookup is None:
                return f"memory:{query}"
            lookup.thread = threading.current_thread()
            lookup.entered.set()
            assert lookup.release.wait(5), "test must release the lookup"
            if isinstance(lookup.result, Exception):
                raise lookup.result
            return lookup.result
        finally:
            with self.lock:
                self.active -= 1

    def queue_prefetch(self, query, *, session_id=""):
        self.queued.append((query, session_id))

    def on_session_end(self, messages):
        self.ended += 1

    def shutdown(self):
        self.closed += 1


@pytest.fixture
def setup_manager():
    providers = []
    managers = []

    def make(*, defer=True, **kwargs):
        manager = MemoryManager(
            external_prefetch_timeout=0.05, defer_prefetch_on_timeout=defer, **kwargs,
        )
        provider = ControlledProvider()
        manager.add_provider(provider)
        managers.append(manager)
        providers.append(provider)
        return manager, provider

    yield make
    for provider in providers:
        for lookup in provider.lookups.values():
            lookup.finish()
    for manager in managers:
        manager.shutdown_all()


def turn(manager, query, session_id="conversation"):
    agent = SimpleNamespace(_memory_manager=manager, session_id=session_id, _user_turn_count=1)
    return _memory_turn_start_and_prefetch(agent, query)


def assert_earlier(text, memory="memory:A"):
    assert memory in text
    assert "earlier-turn" in text.lower()
    assert "previous question" in text.lower()
    assert "not an answer to the current question" in text.lower()


def test_immediate_current_result_is_unchanged(setup_manager):
    manager, provider = setup_manager()
    assert turn(manager, "current question") == "memory:current question"
    assert provider.calls == [("current question", "conversation")]


def test_timeout_alive_skip_then_automatic_carry_and_fresh_lookup(setup_manager):
    manager, provider = setup_manager()
    lookup = provider.block("question A")
    assert turn(manager, "question A") == ""
    assert lookup.entered.is_set()
    assert turn(manager, "question B") == ""
    assert provider.calls == [("question A", "conversation")]
    lookup.finish()

    text = turn(manager, "question C")
    assert_earlier(text)
    assert "memory:question C" in text
    assert "current question" in text.lower()
    assert turn(manager, "question D") == "memory:question D"
    assert provider.calls == [
        ("question A", "conversation"), ("question C", "conversation"), ("question D", "conversation"),
    ]
    assert provider.max_active == 1


def test_defaults_discard_late_result_without_replay():
    manager = MemoryManager(external_prefetch_timeout=0.05)
    provider = ControlledProvider()
    manager.add_provider(provider)
    lookup = provider.block("question A")
    try:
        assert turn(manager, "question A") == ""
        assert turn(manager, "question B") == ""
        lookup.finish()
        assert turn(manager, "question C") == "memory:question C"
        assert turn(manager, "question D") == "memory:question D"
    finally:
        lookup.finish()
        manager.shutdown_all()


def test_ready_carry_survives_current_timeout_and_current_is_carried_later(setup_manager):
    manager, provider = setup_manager()
    old = provider.block("question A")
    assert turn(manager, "question A") == ""
    old.finish()
    current = provider.block("question C", "memory:C")
    assert_earlier(turn(manager, "question C"))
    current.finish()
    text = turn(manager, "question D")
    assert_earlier(text, "memory:C")
    assert "memory:A" not in text
    assert "memory:question D" in text
    assert "memory:C" not in turn(manager, "question E")


@pytest.mark.parametrize("result", ["", "   ", None, RuntimeError("late failure")])
def test_late_errors_and_empty_results_are_not_memories(setup_manager, result):
    manager, provider = setup_manager()
    lookup = provider.block("question A", result)
    assert turn(manager, "question A") == ""
    lookup.finish()
    assert turn(manager, "question C") == "memory:question C"


def test_ready_carry_survives_fresh_error(setup_manager):
    manager, provider = setup_manager()
    old = provider.block("question A")
    assert turn(manager, "question A") == ""
    old.finish()
    failed = provider.block("question C", RuntimeError("current failure"))
    failed.release.set()
    assert_earlier(turn(manager, "question C"))
    assert turn(manager, "question D") == "memory:question D"


@pytest.mark.parametrize("source,target", [("one", "two"), ("", "two"), ("one", ""), ("", "")])
def test_session_ids_never_cross_carry_boundary(setup_manager, source, target):
    manager, provider = setup_manager()
    lookup = provider.block("question A")
    assert turn(manager, "question A", source) == ""
    lookup.finish()
    assert turn(manager, "question C", target) == "memory:question C"
    assert turn(manager, "question D", source) == "memory:question D"


def test_scope_change_while_alive_invalidates_carry(setup_manager):
    manager, provider = setup_manager()
    lookup = provider.block("question A")
    assert turn(manager, "question A", "one") == ""
    assert turn(manager, "question B", "two") == ""
    lookup.finish()
    assert turn(manager, "question C", "one") == "memory:question C"


def test_identical_old_and_current_text_is_not_duplicated(setup_manager):
    manager, provider = setup_manager()
    lookup = provider.block("question A", "memory:question C")
    assert turn(manager, "question A") == ""
    lookup.finish()
    assert turn(manager, "question C") == "memory:question C"


def test_concurrent_invocations_do_not_duplicate_outstanding_lookup(setup_manager):
    manager, provider = setup_manager()
    lookup = provider.block("question A")
    assert turn(manager, "question A") == ""
    barrier = threading.Barrier(4)
    results = []

    def concurrent_turn():
        barrier.wait(5)
        results.append(turn(manager, "question B"))

    threads = [threading.Thread(target=concurrent_turn) for _ in range(3)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(5)
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
        assert results == ["", "", ""]
        assert provider.calls == [("question A", "conversation")]
        assert provider.max_active == 1
    finally:
        lookup.finish()
    assert_earlier(turn(manager, "question C"))


def test_profile_context_is_bound_and_carry_cannot_cross_profiles(setup_manager, tmp_path):
    manager, provider = setup_manager()
    marker = ContextVar("memory_test_marker", default="unbound")
    seen = []
    original = provider.prefetch

    def contextual(query, **kwargs):
        seen.append(marker.get())
        return original(query, **kwargs)

    provider.prefetch = contextual
    lookup = provider.block("question A")
    token = marker.set("owning-profile")
    try:
        assert turn(manager, "question A") == ""
    finally:
        marker.reset(token)
    lookup.finish()
    home_token = set_hermes_home_override(tmp_path / "other-profile")
    try:
        assert turn(manager, "question C") == "memory:question C"
    finally:
        reset_hermes_home_override(home_token)
    assert seen[0] == "owning-profile"
    assert turn(manager, "question D") == "memory:question D"


def test_provider_instance_and_manager_isolation(setup_manager):
    manager, provider = setup_manager()
    other, _ = setup_manager()
    lookup = provider.block("question A")
    assert turn(manager, "question A") == ""
    lookup.finish()
    assert turn(other, "question C") == "memory:question C"
    replacement = ControlledProvider()  # Same name is not the same provider instance.
    assert manager._prefetch_provider(replacement, "question C", session_id="conversation") == "memory:question C"


@pytest.mark.parametrize("boundary", ["end", "switch", "shutdown"])
def test_session_boundaries_discard_late_carry_and_preserve_hooks(setup_manager, boundary):
    manager, provider = setup_manager()
    lookup = provider.block("question A")
    assert turn(manager, "question A") == ""
    if boundary == "end":
        manager.on_session_end([])
        assert provider.ended == 1
    elif boundary == "switch":
        manager.on_session_switch("conversation", reset=True)
    else:
        manager.shutdown_all()
        assert provider.closed == 1
    lookup.finish()
    assert "memory:A" not in turn(manager, "question C")


def test_builtin_path_is_inline_and_unchanged(setup_manager):
    manager, _ = setup_manager()
    builtin = ControlledProvider("builtin")
    caller = threading.current_thread()
    seen = []
    builtin.prefetch = lambda query, **kwargs: seen.append(threading.current_thread()) or "builtin memory"
    manager.add_provider(builtin)
    assert "builtin memory" in turn(manager, "question C")
    assert seen == [caller]


@pytest.mark.parametrize("defer", [False, True])
def test_async_provider_queue_contract_is_unchanged(setup_manager, defer):
    manager, provider = setup_manager(defer=defer)
    cached = {"value": ""}
    provider.queue_prefetch = lambda query, **kwargs: cached.update(value=f"async:{query}")
    provider.prefetch = lambda query, **kwargs: cached.pop("value", "")
    assert turn(manager, "first question") == ""
    manager.queue_prefetch_all("first question", session_id="conversation")
    assert manager.flush_pending(timeout=5)
    assert turn(manager, "second question") == "async:first question"
    assert turn(manager, "third question") == ""


def test_finished_worker_is_owned_by_original_waiter_until_return(setup_manager, monkeypatch):
    manager, provider = setup_manager()
    joined = threading.Event()
    return_to_caller = threading.Event()
    original_join = threading.Thread.join
    results = []

    def pause_after_join(thread, timeout=None):
        if thread.name.startswith("memory-prefetch-"):
            original_join(thread, 5)
            assert not thread.is_alive()
            joined.set()
            assert return_to_caller.wait(5)
        else:
            original_join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "join", pause_after_join)
    caller = threading.Thread(target=lambda: results.append(turn(manager, "question A")))
    caller.start()
    try:
        assert joined.wait(5)
        assert turn(manager, "question B") == ""
        assert provider.calls == [("question A", "conversation")]
    finally:
        return_to_caller.set()
        caller.join(5)
    assert results == ["memory:question A"]
    assert turn(manager, "question C") == "memory:question C"


def test_ready_carry_is_claimed_at_most_once_by_concurrent_turns(setup_manager):
    manager, provider = setup_manager()
    lookup = provider.block("question A", "unique earlier memory")
    assert turn(manager, "question A") == ""
    lookup.finish()
    start = threading.Barrier(4)
    results = []

    def consume():
        start.wait(5)
        results.append(turn(manager, "question C"))

    callers = [threading.Thread(target=consume) for _ in range(3)]
    for caller in callers:
        caller.start()
    start.wait(5)
    for caller in callers:
        caller.join(5)
        assert not caller.is_alive()
    assert sum("unique earlier memory" in text for text in results) == 1
    assert provider.max_active == 1


def test_deferred_recall_uses_egress_redaction_and_opt_in_spill(setup_manager, tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_write

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {
        "memory": {"prefetch_spill_enabled": True},
        "hooks": {"output_spill": {
            "enabled": True, "max_chars": 200, "preview_head": 32, "preview_tail": 32,
            "directory": str(tmp_path / "spill"),
        }},
    })
    manager, provider = setup_manager()
    original = provider.prefetch
    captured = []
    lookup = provider.block("question A", "late ranked memory " * 100)

    def redaction_probe(query, **kwargs):
        captured.append(query)
        return original("question A" if query.startswith("question A") else query, **kwargs)

    provider.prefetch = redaction_probe
    secret = "sk-proj-abc123def456ghi789jkl012"
    assert turn(manager, f"question A OPENAI_API_KEY={secret}") == ""
    lookup.finish()
    result = turn(manager, "question C")
    assert secret not in repr(captured)
    assert "memory prefetch output truncated" in result
    (saved,) = (tmp_path / "spill" / "conversation").glob("*.txt")
    full = saved.read_text(encoding="utf-8")
    assert_earlier(full, "late ranked memory")
    assert "memory:question C" in full
    assert "late ranked memory" not in turn(manager, "question D")


def test_carry_only_changes_current_message_composition(setup_manager):
    from agent.turn_context import _stamp_api_content_sidecar

    manager, provider = setup_manager()
    messages = [
        {"role": "system", "content": "unchanged system prompt"},
        {"role": "user", "content": "question A", "api_content": "original question A wire bytes"},
        {"role": "assistant", "content": "original answer"},
    ]
    original = [dict(message) for message in messages]
    lookup = provider.block("question A")
    assert turn(manager, "question A") == ""
    lookup.finish()
    context = turn(manager, "question C")
    messages.append({"role": "user", "content": "question C"})
    _stamp_api_content_sidecar(
        SimpleNamespace(_session_db=None), messages, 3, context, "", preflight_compressed=False,
    )
    assert_earlier(messages[3]["api_content"])
    assert messages[3]["content"] == "question C"
    assert messages[:3] == original
