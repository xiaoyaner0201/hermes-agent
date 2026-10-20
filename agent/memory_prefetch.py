"""Bounded, conversation-scoped carry-forward for timed-out external recall."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

from agent.memory_provider import MemoryProvider, spawn_context_thread
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)
EXTERNAL_PREFETCH_TIMEOUT_S = 8.0


def validate_prefetch_timeout(value: object) -> float:
    """Only a finite, positive number is a deadline; bools and numeric strings aren't."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("external_prefetch_timeout must be a finite positive number")
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise ValueError("external_prefetch_timeout must be a finite positive number") from exc
    if not math.isfinite(seconds) or seconds <= 0 or seconds > threading.TIMEOUT_MAX:
        raise ValueError("external_prefetch_timeout must be positive and within the thread wait limit")
    return seconds


def configured_prefetch_timeout(memory: dict) -> float:
    value = memory.get("external_prefetch_timeout", EXTERNAL_PREFETCH_TIMEOUT_S)
    try:
        return validate_prefetch_timeout(value)
    except ValueError:
        logger.warning("Invalid memory.external_prefetch_timeout %r; using %.1fs", value, EXTERNAL_PREFETCH_TIMEOUT_S)
        return EXTERNAL_PREFETCH_TIMEOUT_S


def _merge_recall(earlier: str, current: str) -> str:
    if not earlier or earlier.strip() == current.strip():
        return current
    carry = (
        "[Earlier-turn memory results for a previous question in this conversation; "
        "not an answer to the current question]\n" + earlier
    )
    return carry + ("\n\n[Memory results for the current question]\n" + current if current else "")


@dataclass
class _Lookup:
    provider: MemoryProvider
    query: str
    session_id: str
    home: str
    text: str = ""
    eligible: bool = True
    waiting: bool = True
    thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self.thread = spawn_context_thread(self._run, name=f"memory-prefetch-{self.provider.name}")

    def _run(self) -> None:
        try:
            result = self.provider.prefetch(self.query, session_id=self.session_id)
            self.text = result if isinstance(result, str) and result.strip() else ""
        except Exception as exc:
            logger.debug("Memory provider '%s' prefetch failed (non-fatal): %s", self.provider.name, exc, exc_info=True)


class DeferredPrefetch:
    """One outstanding lookup/result slot per provider, owned by one MemoryManager.

    A result remains in its lookup until the next eligible invocation claims it. The
    original waiter owns even an already-finished worker until it releases the slot,
    so concurrent callers cannot deliver a current result a second time as late recall.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lookups: dict[str, _Lookup] = {}
        self._closed = False

    def run(self, provider: MemoryProvider, query: str, *, session_id: str, timeout: float) -> str:
        home = hermes_home_key()
        with self._lock:
            if self._closed:
                return ""
            earlier = self._claim_completed(provider, session_id, home)
            if earlier is None:  # One lookup is still owned/running; never queue duplicates.
                return ""
            lookup = _Lookup(provider, query, session_id, home)
            self._lookups[provider.name] = lookup
            lookup.thread.start()

        try:
            lookup.thread.join(timeout)
            with self._lock:
                if lookup.thread.is_alive():
                    logger.warning(
                        "Memory provider '%s' prefetch timed out after %.1fs; "
                        "deferring its result until a later turn in the same conversation",
                        provider.name, timeout,
                    )
                    current = ""
                else:
                    self._lookups.pop(provider.name, None)
                    current = lookup.text
                return _merge_recall(earlier, current) if lookup.eligible else ""
        finally:
            # An interrupted waiter must not strand the slot; a live worker still fences it.
            with self._lock:
                lookup.waiting = False

    def _claim_completed(self, provider: MemoryProvider, session_id: str, home: str) -> str | None:
        """Called under the lock. None means busy; empty string means no eligible carry."""
        old = self._lookups.get(provider.name)
        if old is None:
            return ""
        if old.provider is not provider or old.session_id != session_id or old.home != home:
            old.eligible = False
        if old.waiting or old.thread.is_alive():
            return None
        return old.text if old.eligible and session_id else ""

    def invalidate(self, *, close: bool = False) -> None:
        """Discard carry on lifecycle boundaries without detaching a still-running worker."""
        with self._lock:
            self._closed = self._closed or close
            for lookup in self._lookups.values():
                lookup.eligible = False
            self._lookups = {
                name: lookup for name, lookup in self._lookups.items()
                if lookup.waiting or lookup.thread.is_alive()
            }
