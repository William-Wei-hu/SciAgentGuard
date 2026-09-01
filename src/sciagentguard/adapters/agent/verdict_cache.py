"""Replay model answers that have already been paid for.

A reviewer call here costs minutes and sometimes fails, so a run that dies at its last step should
not repurchase every answer before it. The cache is keyed by the exact question asked, so changing
the prompt misses rather than answering the new question with the old reply.

Two rules keep replayed answers from quietly becoming fabricated evidence. Every entry records when
the original call happened, so a report can say what is fresh and what is not. And any measurement
of a model's *consistency* must bypass the cache: a cache returns the same answer by construction,
so measuring variance through one would manufacture the consistency it set out to test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

DEFAULT_CACHE_PATH = Path(".cache/judge-verdicts.json")


@dataclass(frozen=True, slots=True)
class CachedCompletion:
    """One stored answer and the provenance of the call that produced it."""

    text: str
    provider: str
    model_id: str
    obtained_at: str
    latency_ms: float

    def as_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model_id": self.model_id,
            "obtained_at": self.obtained_at,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> CachedCompletion | None:
        text = payload.get("text")
        provider = payload.get("provider")
        model_id = payload.get("model_id")
        obtained_at = payload.get("obtained_at")
        latency = payload.get("latency_ms")
        if not isinstance(text, str) or not isinstance(provider, str):
            return None
        if not isinstance(model_id, str) or not isinstance(obtained_at, str):
            return None
        return cls(
            text=text,
            provider=provider,
            model_id=model_id,
            obtained_at=obtained_at,
            latency_ms=float(latency) if isinstance(latency, (int, float)) else 0.0,
        )


def cache_key(provider: str, model_id: str, prompt: str) -> str:
    """Hash exactly what was asked, of whom. A different question is a different key."""

    digest = hashlib.sha256()
    for part in (provider, model_id, prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass
class VerdictCache:
    """A JSON file of answers already obtained, with hit and miss counts for the report."""

    path: Path = DEFAULT_CACHE_PATH
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    def _load(self) -> dict[str, dict[str, object]]:
        if not self.path.is_file():
            return {}
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def get(self, provider: str, model_id: str, prompt: str) -> CachedCompletion | None:
        entry = self._load().get(cache_key(provider, model_id, prompt))
        if not isinstance(entry, dict):
            self.misses += 1
            return None
        completion = CachedCompletion.from_json(entry)
        if completion is None:
            self.misses += 1
            return None
        self.hits += 1
        return completion

    def put(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        text: str,
        latency_ms: float,
    ) -> CachedCompletion:
        completion = CachedCompletion(
            text=text,
            provider=provider,
            model_id=model_id,
            obtained_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
        )
        entries = self._load()
        entries[cache_key(provider, model_id, prompt)] = completion.as_json()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
        return completion

    @property
    def summary(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class Completer(Protocol):
    """The one method this wrapper needs from a client."""

    def complete(self, model_id: str, prompt: str, **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class CachingClient:
    """Wraps a client's `complete`, serving stored answers when the question is unchanged.

    Never use this to measure how consistently a model answers: it will report perfect consistency
    whatever the model does.
    """

    inner: Completer
    cache: VerdictCache
    provider: str
    clock: Callable[[], float] = field(default_factory=lambda: _monotonic)

    def complete(self, model_id: str, prompt: str, **kwargs: Any) -> str:
        return self.completion(model_id, prompt, **kwargs).text

    def completion(self, model_id: str, prompt: str, **kwargs: Any) -> CachedCompletion:
        """Return the answer together with what the original call cost.

        The latency reported is always the one measured when the model actually answered, never
        the microseconds a replay takes. A cost comparison built on replay timings would argue
        the reverse of the truth.
        """

        stored = self.cache.get(self.provider, model_id, prompt)
        if stored is not None:
            return stored

        started = self.clock()
        text = self.inner.complete(model_id, prompt, **kwargs)
        latency_ms = (self.clock() - started) * 1000.0
        if not isinstance(text, str):
            raise TypeError("the wrapped client did not return text")
        return self.cache.put(self.provider, model_id, prompt, text, latency_ms)


def _monotonic() -> float:
    from time import monotonic

    return monotonic()


__all__ = [
    "DEFAULT_CACHE_PATH",
    "CachedCompletion",
    "CachingClient",
    "Completer",
    "VerdictCache",
    "cache_key",
]
