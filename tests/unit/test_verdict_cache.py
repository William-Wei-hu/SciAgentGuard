"""Replaying paid-for answers, without letting replay masquerade as agreement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sciagentguard.adapters.agent.verdict_cache import (
    CachingClient,
    VerdictCache,
    cache_key,
)


class _CountingClient:
    """A stand-in client that records every question it was actually asked."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls: list[str] = []

    def complete(self, model_id: str, prompt: str, **kwargs: Any) -> str:
        del model_id, kwargs
        self.calls.append(prompt)
        return self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]


@pytest.fixture
def cache(tmp_path: Path) -> VerdictCache:
    return VerdictCache(path=tmp_path / "verdicts.json")


def test_a_miss_calls_through_and_stores(cache: VerdictCache) -> None:
    inner = _CountingClient(["VALID"])
    client = CachingClient(inner=inner, cache=cache, provider="stub")

    assert client.complete("m", "is this right?") == "VALID"
    assert len(inner.calls) == 1
    assert cache.get("stub", "m", "is this right?") is not None


def test_a_hit_does_not_call_the_model(cache: VerdictCache) -> None:
    """The whole point: a run that dies late must not repurchase what it already has."""

    inner = _CountingClient(["VALID"])
    client = CachingClient(inner=inner, cache=cache, provider="stub")
    client.complete("m", "is this right?")

    assert client.complete("m", "is this right?") == "VALID"
    assert len(inner.calls) == 1


def test_a_changed_question_is_a_different_entry(cache: VerdictCache) -> None:
    """Answering a new question with an old reply would be fabrication, not caching."""

    inner = _CountingClient(["first", "second"])
    client = CachingClient(inner=inner, cache=cache, provider="stub")

    assert client.complete("m", "question one") == "first"
    assert client.complete("m", "question two") == "second"
    assert len(inner.calls) == 2


def test_the_model_is_part_of_the_key() -> None:
    assert cache_key("p", "model-a", "q") != cache_key("p", "model-b", "q")
    assert cache_key("p", "m", "question one") != cache_key("p", "m", "question two")


def test_a_stored_answer_records_when_it_was_obtained(cache: VerdictCache) -> None:
    """A report has to be able to say which of its numbers are fresh."""

    CachingClient(inner=_CountingClient(["VALID"]), cache=cache, provider="stub").complete("m", "q")
    stored = cache.get("stub", "m", "q")

    assert stored is not None
    assert stored.obtained_at.endswith("+00:00")
    assert stored.provider == "stub"
    assert stored.model_id == "m"


def test_hits_and_misses_are_counted(cache: VerdictCache) -> None:
    client = CachingClient(inner=_CountingClient(["VALID"]), cache=cache, provider="stub")
    client.complete("m", "q")
    client.complete("m", "q")

    assert cache.summary == {"hits": 1, "misses": 1}


def test_a_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.json"
    path.write_text("{not json", encoding="utf-8")

    assert VerdictCache(path=path).get("stub", "m", "q") is None


# --- the trap this design exists to avoid ---------------------------------------------


def test_measuring_consistency_must_not_go_through_the_cache(cache: VerdictCache) -> None:
    """A cache reports perfect agreement whatever the model does.

    The judge answered VALID once and INVALID once on equivalent artifacts, so its consistency is
    something to measure. Measuring it through a cache would return the first answer five times
    and manufacture exactly the stability under test. The probe therefore talks to the client
    directly, and this test pins the difference.
    """

    inner = _CountingClient(["VALID", "INVALID", "INVALID"])
    cached = CachingClient(inner=inner, cache=cache, provider="stub")

    through_cache = [cached.complete("m", "same question") for _ in range(3)]
    assert through_cache == ["VALID", "VALID", "VALID"]
    assert len(inner.calls) == 1, "the cache answered without asking"

    direct = _CountingClient(["VALID", "INVALID", "INVALID"])
    unmediated = [direct.complete("m", "same question") for _ in range(3)]

    assert unmediated == ["VALID", "INVALID", "INVALID"]
    assert len(direct.calls) == 3


def test_a_replayed_answer_reports_the_original_call_cost(cache: VerdictCache) -> None:
    """The cost of a verdict is what it cost to obtain, not what it costs to remember.

    A replay returns in microseconds. Recording that as the reviewer's latency would produce a
    cost comparison arguing the exact opposite of the measurement it is meant to report.
    """

    ticks = iter([0.0, 12.0, 100.0, 100.0])
    client = CachingClient(
        inner=_CountingClient(["VALID"]),
        cache=cache,
        provider="stub",
        clock=lambda: next(ticks),
    )

    first = client.completion("m", "q")
    replayed = client.completion("m", "q")

    assert first.latency_ms == pytest.approx(12_000.0)
    assert replayed.latency_ms == pytest.approx(12_000.0), "the replay reported its own speed"
