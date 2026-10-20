"""Memory settings travel from a temporary YAML config through the real init seam."""

import threading
from types import SimpleNamespace

import pytest

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _init_memory
from agent.memory_manager import MemoryManager
from hermes_cli.config import atomic_config_write, load_config_readonly
from tests.agent.test_memory_prefetch_defer import ControlledProvider, assert_earlier, turn


@pytest.fixture
def runtime_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = ControlledProvider()
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: provider)
    managers = []

    def initialize(memory):
        atomic_config_write(tmp_path / "config.yaml", {"memory": {
            "provider": "controlled", "memory_enabled": False, "user_profile_enabled": False,
            **memory,
        }})
        config = load_config_readonly()
        agent = SimpleNamespace(
            enabled_toolsets=[], disabled_toolsets=[], session_id="conversation", session_cwd=None,
            _session_db=None, tools=[], _emit_startup_warning=lambda message: None,
            **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
        )
        _init_memory(agent, config, False, "api_server")
        assert agent._memory_manager is not None
        managers.append(agent._memory_manager)
        return agent._memory_manager

    yield initialize, provider
    for lookup in provider.lookups.values():
        lookup.finish()
    for manager in managers:
        manager.shutdown_all()


def test_configured_deadline_reaches_real_wait_and_carry_is_enabled(runtime_manager, monkeypatch):
    initialize, _ = runtime_manager
    manager = initialize({"external_prefetch_timeout": 12, "defer_prefetch_on_timeout": True})
    observed = []
    original_join = threading.Thread.join

    def record_join(thread, timeout=None):
        if thread.name.startswith("memory-prefetch-"):
            observed.append(timeout)
        return original_join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "join", record_join)
    assert turn(manager, "current question") == "memory:current question"
    assert observed == [12.0]


def test_configured_carry_is_used_automatically_by_turn_initialization(runtime_manager, monkeypatch):
    initialize, provider = runtime_manager
    manager = initialize({"external_prefetch_timeout": 0.05, "defer_prefetch_on_timeout": True})
    lookup = provider.block("question A")
    original_join = threading.Thread.join

    def bounded_test_wait(thread, timeout=None):
        if thread.name.startswith("memory-prefetch-") and not lookup.release.is_set():
            assert lookup.entered.wait(5)
            return original_join(thread, 0.05)
        return original_join(thread, timeout)

    # Keep this regression bounded even on a base that ignores the configured deadline.
    monkeypatch.setattr(threading.Thread, "join", bounded_test_wait)
    assert turn(manager, "question A") == ""
    lookup.finish()
    assert_earlier(turn(manager, "question C"))


def test_default_config_keeps_eight_second_deadline_and_carry_off(runtime_manager):
    initialize, _ = runtime_manager
    manager = initialize({})
    assert manager._external_prefetch_timeout == 8.0
    assert manager._defer_prefetch_on_timeout is False


@pytest.mark.parametrize("invalid", [True, False, 0, -1, "12", "invalid", float("nan"), float("inf"), [], {}])
def test_invalid_config_timeout_falls_back_without_disabling_provider(runtime_manager, invalid):
    initialize, _ = runtime_manager
    manager = initialize({"external_prefetch_timeout": invalid})
    assert manager._external_prefetch_timeout == 8.0
    assert turn(manager, "current question") == "memory:current question"


@pytest.mark.parametrize("value", [False, "false", "true", "yes", 1, 0, None, [], {}])
def test_carry_flag_requires_boolean_true(runtime_manager, value):
    initialize, _ = runtime_manager
    manager = initialize({"defer_prefetch_on_timeout": value})
    assert manager._defer_prefetch_on_timeout is False


@pytest.mark.parametrize("invalid", [True, False, 0, -1, "12", "invalid", float("nan"), float("inf"), [], {}])
def test_explicit_constructor_timeout_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="external_prefetch_timeout"):
        MemoryManager(external_prefetch_timeout=invalid)


def test_explicit_constructor_timeout_override_is_preserved():
    manager = MemoryManager(external_prefetch_timeout=0.125)
    assert manager._external_prefetch_timeout == 0.125


def test_existing_manager_keeps_startup_settings_until_recreated(runtime_manager):
    initialize, _ = runtime_manager
    first = initialize({"external_prefetch_timeout": 12, "defer_prefetch_on_timeout": True})
    second = initialize({"external_prefetch_timeout": 9, "defer_prefetch_on_timeout": False})
    assert first._external_prefetch_timeout == 12.0
    assert first._defer_prefetch_on_timeout is True
    assert second._external_prefetch_timeout == 9.0
    assert second._defer_prefetch_on_timeout is False
