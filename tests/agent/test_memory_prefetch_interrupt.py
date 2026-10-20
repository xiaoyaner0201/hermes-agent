"""An abandoned recall waiter must not strand the provider's single lookup slot."""

import asyncio
import os
import signal
import sys
import threading

import pytest

from agent.memory_manager import MemoryManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.agent.test_memory_prefetch_defer import ControlledProvider, assert_earlier, turn


@pytest.fixture
def manager_provider():
    manager = MemoryManager(external_prefetch_timeout=10, defer_prefetch_on_timeout=True)
    provider = ControlledProvider()
    manager.add_provider(provider)
    yield manager, provider
    for lookup in provider.lookups.values():
        lookup.finish()
    manager.shutdown_all()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, asyncio.CancelledError, RuntimeError])
@pytest.mark.parametrize("scope", ["same", "session", "profile"])
def test_join_exception_releases_waiter_without_duplicating_or_leaking(
    manager_provider, monkeypatch, tmp_path, error_type, scope,
):
    manager, provider = manager_provider
    lookup = provider.block("question A")
    failure = error_type("interrupted recall wait")
    original_join = threading.Thread.join

    def failed_join(thread, timeout=None):
        if thread.name.startswith("memory-prefetch-"):
            assert lookup.entered.wait(5)
            raise failure
        return original_join(thread, timeout)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "join", failed_join)
        if isinstance(failure, Exception):
            # Ordinary provider orchestration errors remain non-fatal at the public seam.
            assert manager.prefetch_all("question A", session_id="conversation") == ""
        else:
            with pytest.raises(error_type) as caught:
                manager.prefetch_all("question A", session_id="conversation")
            assert caught.value is failure

    session_id = "new-conversation" if scope == "session" else "conversation"
    if scope == "session":
        manager.on_session_switch(session_id, reset=True)
    home_token = set_hermes_home_override(tmp_path / "other-profile") if scope == "profile" else None
    try:
        # Relinquishing the waiter must not detach the worker that is still in prefetch.
        assert turn(manager, "question B", session_id) == ""
        assert provider.calls == [("question A", "conversation")]
        lookup.finish()
        result = turn(manager, "question C", session_id)
        if scope == "same":
            assert_earlier(result)
            assert "memory:question C" in result
        else:
            assert result == "memory:question C"
        assert turn(manager, "question D", session_id) == "memory:question D"
        assert provider.calls == [
            ("question A", "conversation"), ("question C", session_id), ("question D", session_id),
        ]
        assert provider.max_active == 1
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("switch_session", [False, True])
def test_real_sigint_during_join_allows_later_turns(defer, switch_session):
    manager = MemoryManager(external_prefetch_timeout=10, defer_prefetch_on_timeout=defer)
    provider = ControlledProvider()
    manager.add_provider(provider)
    lookup = provider.block("question A")
    joining = threading.Event()
    sent = threading.Event()
    previous_profile = sys.getprofile()
    previous_handler = signal.signal(signal.SIGINT, signal.default_int_handler)

    def observe_join(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "join":
            thread = frame.f_locals.get("self")
            if isinstance(thread, threading.Thread) and thread.name.startswith("memory-prefetch-"):
                joining.set()

    def send_interrupt():
        if lookup.entered.wait(5) and joining.wait(5):
            os.kill(os.getpid(), signal.SIGINT)
            sent.set()

    signaller = threading.Thread(target=send_interrupt, name="prefetch-test-interrupt")
    signaller.start()
    try:
        sys.setprofile(observe_join)
        try:
            with pytest.raises(KeyboardInterrupt):
                turn(manager, "question A")
        finally:
            sys.setprofile(previous_profile)
            signaller.join(5)
        assert sent.is_set() and not signaller.is_alive()
        assert provider.calls == [("question A", "conversation")]
        session_id = "conversation"
        if switch_session:
            session_id = "new-conversation"
            manager.on_session_switch(session_id, reset=True)
        assert turn(manager, "question while worker runs", session_id) == ""
        assert provider.calls == [("question A", "conversation")]
        lookup.finish()
        result = turn(manager, "question B", session_id)
        if defer and not switch_session:
            assert_earlier(result)
            assert "memory:question B" in result
        else:
            assert result == "memory:question B"
        assert turn(manager, "question C", session_id) == "memory:question C"
        assert provider.calls == [
            ("question A", "conversation"), ("question B", session_id), ("question C", session_id),
        ]
        assert provider.max_active == 1
    finally:
        sys.setprofile(previous_profile)
        signaller.join(5)
        signal.signal(signal.SIGINT, previous_handler)
        lookup.finish()
        manager.shutdown_all()
