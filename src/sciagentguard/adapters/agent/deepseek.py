"""A DeepSeek-backed judge, over the REST API through the standard library.

The judge is the baseline the contracts are measured against, so it is deliberately a different
vendor from the agent being judged: a model grading its own family invites a question this design
does not have to answer. No SDK is imported, so the package keeps no model dependency.

The verdict is read from `content` alone. These models also return `reasoning_content`, which is
long and quotes the artifact back; it is never parsed and never recorded.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from sciagentguard.adapters.agent.gemini import (
    RETRYABLE_STATUSES,
    TRANSPORT_FAILURE_STATUS,
    GeminiError,
    Transport,
    load_api_key,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
API_KEY_VARIABLE = "DEEPSEEK_API_KEY"
PROVIDER = "deepseek"
DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
# Measured, not guessed. One verdict on a real analysis artifact costs this judge about 11,700
# output tokens of reasoning before it writes a single word, so a budget sized from a "reply OK"
# probe returns finish_reason="length" and no content at all -- the same way the previous judge
# failed. 4096 and 8192 were both exhausted; this leaves headroom above what was measured.
DEFAULT_MAX_TOKENS = 32768
# And that reasoning takes time. Latencies observed across one experiment ran from 11 seconds on an
# obviously broken artifact to 700 seconds on a correct one, and a 600-second timeout turned the two
# slowest calls into failures. Because the slow calls are the hard cases, losing them is not a
# random loss: it silently removes exactly the artifacts the reviewer had most to say about.
OBSERVED_MAX_LATENCY_SECONDS = 700
REQUEST_TIMEOUT_SECONDS = 1200


class DeepSeekError(RuntimeError):
    """A DeepSeek request failed. Carries a status and a redacted reason, never credentials."""


def load_deepseek_key(
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
) -> str:
    """Read the DeepSeek key from the environment or the gitignored `.env` file."""

    try:
        return load_api_key(environ, env_file, variable=API_KEY_VARIABLE)
    except GeminiError as error:
        raise DeepSeekError(str(error)) from None


def _urllib_transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST" if body is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as error:
        return int(error.code), bytes(error.read())
    except (OSError, http.client.HTTPException) as error:
        reason = json.dumps({"error": {"message": f"transport failure: {type(error).__name__}"}})
        return TRANSPORT_FAILURE_STATUS, reason.encode("utf-8")


@dataclass(frozen=True)
class DeepSeekClient:
    """Minimal DeepSeek chat client with a per-instance request throttle."""

    api_key: str = field(repr=False)
    min_interval_seconds: float = 0.0
    base_url: str = DEFAULT_BASE_URL
    transport: Transport = _urllib_transport
    max_retries: int = 5
    retry_backoff_seconds: float = 4.0
    _clock: Callable[[], float] = time.monotonic
    _sleep: Callable[[float], None] = time.sleep
    _last_call: list[float] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise DeepSeekError("the DeepSeek API key must not be empty")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be nonnegative")

    def __repr__(self) -> str:
        return f"DeepSeekClient(base_url={self.base_url!r}, api_key=<redacted>)"

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = self._clock()
        if self._last_call:
            elapsed = now - self._last_call[-1]
            if elapsed < self.min_interval_seconds:
                self._sleep(self.min_interval_seconds - elapsed)
        self._last_call.append(self._clock())

    def complete(
        self,
        model_id: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Return the assistant's visible answer, never its reasoning trace."""

        body = json.dumps(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        status, raw = 0, b""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            status, raw = self.transport(f"{self.base_url}/chat/completions", headers, body)
            if status == 200:
                break
            if status not in RETRYABLE_STATUSES or attempt == self.max_retries:
                break
            self._sleep(self.retry_backoff_seconds * (2**attempt))
        if status != 200:
            raise DeepSeekError(f"DeepSeek returned HTTP {status}: {_reason(raw, self.api_key)}")

        try:
            payload = json.loads(raw)
        except ValueError:
            raise DeepSeekError("DeepSeek returned a body that is not JSON") from None
        return _visible_answer(payload)


@dataclass(frozen=True)
class DeepSeekJudge:
    """Ask a model whether a final artifact is scientifically valid, seeing nothing else."""

    client: DeepSeekClient
    model_id: str = DEFAULT_JUDGE_MODEL
    temperature: float = 0.0
    provider: str = PROVIDER
    source: str = "deepseek-judge"

    def verdict(self, final_artifact: Mapping[str, JsonValue]) -> bool:
        prompt = (
            "You are reviewing the final result of a high-energy physics diphoton analysis.\n"
            "You can see only this final artifact, not the code or the intermediate steps.\n\n"
            f"{json.dumps(dict(final_artifact), indent=2, sort_keys=True, default=str)}\n\n"
            "Is this result scientifically valid? Consider whether the values are physically "
            "plausible and internally consistent.\n"
            'Answer with exactly one word: "VALID" or "INVALID".'
        )
        answer = self.client.complete(self.model_id, prompt, temperature=self.temperature)
        return "INVALID" not in answer.strip().upper()


def _visible_answer(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise DeepSeekError("the response contained no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise DeepSeekError("the first choice is not an object")
    message = first.get("message")
    # `reasoning_content` is deliberately ignored: it is long, it quotes the artifact back, and a
    # verdict taken from a model's private deliberation is not the verdict it gave.
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekError(
            f"the response contained no visible content (finish_reason="
            f"{first.get('finish_reason')!r})"
        )
    return content


def _reason(raw: bytes, secret: str) -> str:
    """Summarize an error body with credentials stripped out."""

    try:
        parsed = json.loads(raw)
    except ValueError:
        return "unparseable error body"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            cleaned = message[:200]
            return cleaned.replace(secret, "<redacted>") if secret else cleaned
    return "unrecognized error body"


__all__ = [
    "API_KEY_VARIABLE",
    "DEFAULT_BASE_URL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MAX_TOKENS",
    "OBSERVED_MAX_LATENCY_SECONDS",
    "PROVIDER",
    "REQUEST_TIMEOUT_SECONDS",
    "DeepSeekClient",
    "DeepSeekError",
    "DeepSeekJudge",
    "load_deepseek_key",
]
