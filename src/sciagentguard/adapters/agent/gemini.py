"""Gemini-backed analysis agent and judge, over the REST API through the standard library.

No model SDK is imported. The base package therefore keeps no model dependency at all, and the
optional `agent` extra stays empty. The request is a plain JSON POST, which is all this needs.

The API key is read from the environment or from a gitignored `.env` file and placed in a request
header. It is never stored in a model field, never included in an exception message, and never
rendered by `repr`. `test_gemini_client.py` asserts each of those.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from sciagentguard.adapters.agent.models import AgentProposal, AgentTask

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_KEY_VARIABLE = "GEMINI_API_KEY"
PROVIDER = "google"
# Transient upstream conditions: rate limiting, capacity, and gateway errors.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
# Reasoning models spend output tokens on internal thought before writing anything, and those
# tokens count against this budget. A budget sized for the visible answer alone returns
# finishReason=MAX_TOKENS with no content at all. The limit is deliberately generous: lowering it,
# or disabling thinking, would quietly weaken whichever model is being measured.
DEFAULT_MAX_OUTPUT_TOKENS = 32768

Transport = Callable[[str, dict[str, str], bytes | None], tuple[int, bytes]]
"""(url, headers, body) -> (status, raw response). Injectable so tests never touch the network."""

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


class GeminiError(RuntimeError):
    """A Gemini request failed. Carries a status and a redacted reason, never credentials."""


def load_api_key(
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    variable: str = API_KEY_VARIABLE,
) -> str:
    """Read an API key from the environment, falling back to a gitignored `.env` file.

    The variable name is a parameter because more than one provider is involved: the agent being
    judged and the judge come from different vendors on purpose.
    """

    source = os.environ if environ is None else environ
    key = source.get(variable, "").strip()
    if key:
        return key

    path = Path(".env") if env_file is None else env_file
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == variable:
                candidate = value.strip().strip("'\"")
                if candidate:
                    return candidate

    raise GeminiError(
        f"no API key found: set {variable} in the environment, or write "
        f"{variable}=<key> into a .env file at the project root"
    )


TRANSPORT_FAILURE_STATUS = 504


def _urllib_transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
    """Perform one request, reporting transport failures as a retryable status.

    A read timeout or a dropped connection is as transient as an upstream 503, and it arrives as
    an ordinary OSError rather than an HTTPError. Returning a status instead of raising keeps one
    bounded retry policy in charge of every transient condition.
    """

    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST" if body is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return int(response.status), bytes(response.read())
    except urllib.error.HTTPError as error:
        return int(error.code), bytes(error.read())
    except (OSError, http.client.HTTPException) as error:
        # OSError covers URLError, socket timeouts, and connection resets. The text may name a
        # host but never a credential.
        reason = json.dumps({"error": {"message": f"transport failure: {type(error).__name__}"}})
        return TRANSPORT_FAILURE_STATUS, reason.encode("utf-8")


@dataclass(frozen=True)
class GeminiResponse:
    """One completion and the accounting needed to make the run reproducible evidence."""

    text: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str


@dataclass(frozen=True)
class GeminiClient:
    """Minimal Gemini REST client with a per-instance request throttle."""

    api_key: str = field(repr=False)
    min_interval_seconds: float = 0.0
    base_url: str = DEFAULT_BASE_URL
    transport: Transport = _urllib_transport
    max_retries: int = 3
    retry_backoff_seconds: float = 4.0
    _clock: Callable[[], float] = time.monotonic
    _sleep: Callable[[float], None] = time.sleep
    _last_call: list[float] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise GeminiError("the Gemini API key must not be empty")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be nonnegative")

    def __repr__(self) -> str:
        # Never let the key reach a traceback, a log line, or a debugger frame dump.
        return f"GeminiClient(base_url={self.base_url!r}, api_key=<redacted>)"

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = self._clock()
        if self._last_call:
            elapsed = now - self._last_call[-1]
            if elapsed < self.min_interval_seconds:
                self._sleep(self.min_interval_seconds - elapsed)
        self._last_call.append(self._clock())

    def _call(self, url: str, body: bytes | None) -> dict[str, Any]:
        status, raw = 0, b""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            status, raw = self.transport(url, self._headers(), body)
            if status == 200:
                break
            if status not in RETRYABLE_STATUSES or attempt == self.max_retries:
                break
            # Rate limiting and capacity are transient. Losing a whole run -- and the quota
            # already spent on it -- to one 503 would make the experiment far more expensive
            # than the retry.
            self._sleep(self.retry_backoff_seconds * (2**attempt))
        if status != 200:
            raise GeminiError(
                f"Gemini returned HTTP {status}: {_redacted_reason(raw, self.api_key)}"
            )
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise GeminiError("Gemini returned a body that is not JSON") from None
        if not isinstance(parsed, dict):
            raise GeminiError("Gemini returned a JSON value that is not an object")
        return parsed

    def list_models(self) -> tuple[str, ...]:
        """Return the model identifiers this key may call, without their `models/` prefix."""

        payload = self._call(f"{self.base_url}/models", None)
        models = payload.get("models")
        if not isinstance(models, list):
            raise GeminiError("the model listing did not contain a models array")
        names: list[str] = []
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str):
                    names.append(name.removeprefix("models/"))
        return tuple(sorted(names))

    def generate(
        self,
        model_id: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_level: str | None = None,
    ) -> GeminiResponse:
        """Request one completion and record what it cost to produce."""

        config: dict[str, JsonValue] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if seed is not None:
            config["seed"] = seed
        if thinking_level is not None:
            config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        body = json.dumps(
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
        ).encode("utf-8")

        started = self._clock()
        payload = self._call(f"{self.base_url}/models/{model_id}:generateContent", body)
        latency_ms = (self._clock() - started) * 1000.0

        usage = payload.get("usageMetadata")
        usage = usage if isinstance(usage, dict) else {}
        return GeminiResponse(
            text=_first_text(payload),
            prompt_tokens=_count(usage, "promptTokenCount"),
            output_tokens=_count(usage, "candidatesTokenCount"),
            latency_ms=latency_ms,
            model_id=model_id,
        )


def extract_code(text: str) -> str:
    """Pull Python out of a model response, tolerating fenced blocks and surrounding prose."""

    blocks = [str(block) for block in _FENCE.findall(text)]
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def build_prompt(task: AgentTask, feedback: str | None) -> str:
    """Assemble the task, the required output shape, and any feedback from a rejected attempt.

    The output shape is stated field by field. A model that is not told the exact keys will report
    correct work under its own names, which the harness must then reject as an output-contract
    failure rather than a scientific one.
    """

    sections = [
        "You are writing a complete Python analysis script.",
        "",
        f"TASK: {task.description}",
        "",
        f"INPUT: {task.input_description}",
        "",
        "RUNTIME CONTRACT:",
        "- Two globals are already defined: INPUT_PATH and OUTPUT_PATH, both strings.",
        "- You may import json, math, numpy, and uproot. There is no network access.",
        "- Write exactly one JSON object to OUTPUT_PATH and nothing else.",
        "",
        "REQUIRED OUTPUT SHAPE. Use these exact key names:",
        _SCHEMA_TEXT,
        "",
        "Reply with a single Python code block and no commentary.",
    ]
    if feedback is not None:
        sections.extend(
            [
                "",
                "A previous attempt was rejected. Here is why:",
                feedback,
                "",
                "Return a corrected script.",
            ]
        )
    return "\n".join(sections)


# Every field is defined, not merely typed. An undefined field cannot be checked: the first live
# run reported correct work in its own shape, and calling that a model error would have been
# unfair. A contract can only check what the specification pins down.
_SCHEMA_TEXT = """{
  "selection": {
    "selection_id": str,
    "input_event_ids": [int],      // every event id read from the file
    "selected_event_ids": [int],   // the events surviving all cuts
    "selected_weight": [float],    // RAW mcWeight per selected event, NOT normalized,
                                   //   aligned with selected_event_ids
    "selected_mass_gev": [float],  // diphoton mass per selected event, aligned likewise
    "selected_weight_sum": float,  // the plain sum of selected_weight
    "generated_weight_sum": float, // the per-file SumWeights CONSTANT, read once, NOT summed
    "cross_section_pb": float,     // the per-file XSection CONSTANT, read once, NOT summed
    "cutflow": [{"cut_id": str, "surviving": int}],  // non-increasing; last equals the selection
    "regions": {"signal": [int], "control": [int]}   // signal = selected events whose mass is
                                                     //   inside the window; control = those
                                                     //   outside it. Together they cover every
                                                     //   selected event exactly once
  },
  "histogram": {
    "observable": str, "bin_edges": [float],
    "bin_weight_sums": [float],    // sums of RAW weights per bin, NOT normalized
    "bin_counts": [int],
    "underflow_weight_sum": float, "overflow_weight_sum": float,
    "selected_weight_sum": float,  // the same value reported in selection
    "generated_weight_sum": float, "cross_section_pb": float,  // the same per-file constants
    "luminosity_pb_inverse": float,
    "normalization_factor": float  // cross_section_pb * luminosity_pb_inverse
                                   //   / generated_weight_sum
  },
  "yield_estimate": {
    "observable": str, "method": str, "signal_window_gev": [float, float],
    "signal_weight_sum": float,    // RAW weight inside the signal window
    "sideband_weight_sum": float, "background_estimate": float,
    "normalization_factor": float, // the same factor reported in histogram, repeated here so
                                   //   this artifact can be checked on its own
    "estimated_yield": float,      // (signal_weight_sum - background_estimate)
                                   //   * normalization_factor
    "peak_bin_index": int, "peak_bin_center_gev": float,
    "signal_bin_count": int, "sideband_bin_count": int
  }
}"""

# The cutflow must list these four stages, in this order, so the monotonicity check has endpoints.
CUTFLOW_HINT = ("all_events", "two_photons", "leading_photon_pt", "subleading_photon_pt")


@dataclass(frozen=True)
class GeminiAgent:
    """An analysis agent backed by one Gemini model."""

    client: GeminiClient
    model_id: str
    temperature: float = 0.0
    source: str = "google-gemini"

    def propose(
        self,
        task: AgentTask,
        *,
        attempt_id: str,
        feedback: str | None = None,
        seed: int | None = None,
    ) -> AgentProposal:
        prompt = build_prompt(task, feedback)
        response = self.client.generate(
            self.model_id, prompt, temperature=self.temperature, seed=seed
        )
        code = extract_code(response.text)
        if not code.strip():
            raise GeminiError("the model returned no extractable code")

        return AgentProposal(
            proposal_id=f"{task.task_id}:{attempt_id}:{self.model_id}",
            task_id=task.task_id,
            attempt_id=attempt_id,
            code=code,
            source=self.source,
            model_id=self.model_id,
            provider=PROVIDER,
            seed=seed,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            sampling_parameters={
                "temperature": self.temperature,
                "prompt_tokens": response.prompt_tokens,
                "output_tokens": response.output_tokens,
            },
            latency_ms=response.latency_ms,
        )


@dataclass(frozen=True)
class GeminiJudge:
    """A judge backed by one Gemini model, shown only the final artifact."""

    client: GeminiClient
    model_id: str
    temperature: float = 0.0
    source: str = "google-gemini-judge"

    def verdict(self, final_artifact: Mapping[str, JsonValue]) -> bool:
        prompt = (
            "You are reviewing the final result of a high-energy physics diphoton analysis.\n"
            "You can see only this final artifact, not the code or the intermediate steps.\n\n"
            f"{json.dumps(dict(final_artifact), indent=2, sort_keys=True, default=str)}\n\n"
            "Is this result scientifically valid? Consider whether the values are physically "
            "plausible and internally consistent.\n"
            'Answer with exactly one word: "VALID" or "INVALID".'
        )
        response = self.client.generate(self.model_id, prompt, temperature=self.temperature)
        return "INVALID" not in response.text.strip().upper()


def _first_text(payload: Mapping[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise GeminiError("the response contained no candidates")
    first = candidates[0]
    if not isinstance(first, dict):
        raise GeminiError("the first candidate is not an object")
    content = first.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        finish = first.get("finishReason")
        if finish == "MAX_TOKENS":
            raise GeminiError(
                "the model exhausted its output budget before writing anything; raise "
                "max_output_tokens, since reasoning tokens are drawn from the same budget"
            )
        raise GeminiError(f"the response contained no content parts (finishReason={finish!r})")
    texts = [part["text"] for part in parts if isinstance(part, dict) and "text" in part]
    return "".join(str(text) for text in texts)


def _count(usage: Mapping[str, Any], field_name: str) -> int:
    value = usage.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _redacted_reason(raw: bytes, secret: str) -> str:
    """Summarize an error body with credentials stripped out.

    An upstream error may quote the request back, key included. Anything that leaves this function
    is about to end up in a traceback, so scrub both the key we sent and anything else shaped like
    a Google API key before returning.
    """

    try:
        parsed = json.loads(raw)
    except ValueError:
        return "unparseable error body"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        status = error.get("status")
        message = error.get("message")
        if isinstance(message, str):
            trimmed = _scrub(message[:200], secret)
            return f"{status}: {trimmed}" if isinstance(status, str) else trimmed
    return "unrecognized error body"


_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")


def _scrub(text: str, secret: str) -> str:
    cleaned = text.replace(secret, "<redacted>") if secret else text
    return _KEY_SHAPE.sub("<redacted>", cleaned)


def resolve_model_id(available: Sequence[str], wanted: str) -> str:
    """Pick the API identifier matching a requested model, tolerating version suffixes."""

    if wanted in available:
        return wanted
    prefixed = [name for name in available if name.startswith(wanted)]
    if prefixed:
        return sorted(prefixed, key=len)[0]
    raise GeminiError(f"no available model matches {wanted!r}")


__all__ = [
    "API_KEY_VARIABLE",
    "CUTFLOW_HINT",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "PROVIDER",
    "RETRYABLE_STATUSES",
    "GeminiAgent",
    "GeminiClient",
    "GeminiError",
    "GeminiJudge",
    "GeminiResponse",
    "Transport",
    "build_prompt",
    "extract_code",
    "load_api_key",
    "resolve_model_id",
]
