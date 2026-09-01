"""The Gemini client must never leak the key, and must never touch the network in CI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sciagentguard.adapters.agent.gemini import (
    API_KEY_VARIABLE,
    RETRYABLE_STATUSES,
    GeminiAgent,
    GeminiClient,
    GeminiError,
    GeminiJudge,
    Transport,
    build_prompt,
    extract_code,
    load_api_key,
    resolve_model_id,
)
from sciagentguard.adapters.agent.models import AgentTask

SECRET = "AIza-TEST-SECRET-KEY-DO-NOT-LEAK"

TASK = AgentTask(
    task_id="t",
    description="Do the analysis.",
    input_description="A ROOT file.",
    expected_outputs=("selection",),
)


Call = tuple[str, dict[str, str], bytes | None]


def _transport(payload: Mapping[str, Any], status: int = 200) -> tuple[Transport, list[Call]]:
    """A stub transport that records what it was asked to send."""

    calls: list[Call] = []

    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        calls.append((url, headers, body))
        return status, json.dumps(dict(payload)).encode("utf-8")

    return transport, calls


def _completion(text: str) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 22},
    }


def _client(payload: Mapping[str, Any], status: int = 200) -> tuple[GeminiClient, list[Call]]:
    transport, calls = _transport(payload, status)
    return GeminiClient(api_key=SECRET, transport=transport), calls


def _sent_body(calls: list[Call]) -> dict[str, Any]:
    body = calls[0][2]
    assert body is not None
    parsed: dict[str, Any] = json.loads(body)
    return parsed


# --- key hygiene ---------------------------------------------------------------------


def test_the_key_never_appears_in_repr() -> None:
    client, _ = _client(_completion("x"))

    assert SECRET not in repr(client)
    assert "redacted" in repr(client)


def test_the_key_never_appears_in_an_error() -> None:
    body = {"error": {"status": "PERMISSION_DENIED", "message": f"key {SECRET} is invalid"}}
    client, _ = _client(body, status=403)

    with pytest.raises(GeminiError) as raised:
        client.list_models()

    # The upstream message quoted the key back at us; ours must not carry it onward.
    assert "403" in str(raised.value)
    assert SECRET not in str(raised.value)


def test_the_key_never_appears_in_a_proposal() -> None:
    client, _ = _client(_completion("```python\nprint(1)\n```"))
    agent = GeminiAgent(client=client, model_id="m")

    proposal = agent.propose(TASK, attempt_id="a0", seed=7)

    assert SECRET not in proposal.model_dump_json()
    assert SECRET not in repr(proposal)


def test_the_key_travels_in_a_header_not_the_url() -> None:
    client, calls = _client(_completion("code"))

    client.generate("m", "prompt")

    url, headers, _ = calls[0]
    assert SECRET not in url
    assert headers["x-goog-api-key"] == SECRET


# --- key loading ---------------------------------------------------------------------


def test_the_environment_is_preferred_over_the_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_VARIABLE}=from-file\n", encoding="utf-8")

    assert load_api_key({API_KEY_VARIABLE: "from-env"}, env_file) == "from-env"


@pytest.mark.parametrize(
    "line",
    [f"{API_KEY_VARIABLE}=plain", f'{API_KEY_VARIABLE}="quoted"', f"{API_KEY_VARIABLE}= spaced "],
)
def test_the_file_is_read_when_the_environment_is_empty(tmp_path: Path, line: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"# a comment\nOTHER=1\n{line}\n", encoding="utf-8")

    assert load_api_key({}, env_file) in {"plain", "quoted", "spaced"}


def test_a_missing_key_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(GeminiError, match="no API key found"):
        load_api_key({}, tmp_path / "absent")


def test_an_empty_key_is_rejected() -> None:
    with pytest.raises(GeminiError, match="must not be empty"):
        GeminiClient(api_key="   ")


# --- request behaviour ---------------------------------------------------------------


def test_generate_reports_tokens_and_latency() -> None:
    client, _ = _client(_completion("hello"))

    response = client.generate("m", "prompt", seed=3)

    assert response.text == "hello"
    assert response.prompt_tokens == 11
    assert response.output_tokens == 22
    assert response.latency_ms >= 0.0


def test_the_seed_and_temperature_reach_the_request() -> None:
    client, calls = _client(_completion("x"))

    client.generate("m", "prompt", temperature=0.5, seed=42)

    body = _sent_body(calls)
    assert body["generationConfig"]["seed"] == 42
    assert body["generationConfig"]["temperature"] == 0.5


def test_a_response_without_content_is_an_error() -> None:
    """An exhausted budget has its own message; every other empty response falls back to this."""

    client, _ = _client({"candidates": [{"finishReason": "SAFETY"}]})

    with pytest.raises(GeminiError, match="no content parts"):
        client.generate("m", "prompt")


def test_the_throttle_waits_between_calls() -> None:
    slept: list[float] = []
    now = [0.0]
    transport, _ = _transport(_completion("x"))
    client = GeminiClient(
        api_key=SECRET,
        min_interval_seconds=4.0,
        transport=transport,
        _clock=lambda: now[0],
        _sleep=slept.append,
    )

    client.generate("m", "a")
    now[0] = 1.0
    client.generate("m", "b")

    assert slept == [3.0]


def test_list_models_strips_the_prefix() -> None:
    client, _ = _client({"models": [{"name": "models/b"}, {"name": "models/a"}]})

    assert client.list_models() == ("a", "b")


# --- prompt and extraction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("```python\nx = 1\n```", "x = 1"),
        ("Here you go:\n```\nx = 1\n```\nHope that helps.", "x = 1"),
        ("x = 1", "x = 1"),
        (
            "```python\nshort\n```\ntext\n```python\nmuch longer block here\n```",
            "much longer block here",
        ),
    ],
)
def test_code_is_extracted_from_prose_and_fences(response: str, expected: str) -> None:
    assert extract_code(response) == expected


def test_the_prompt_states_the_exact_output_keys() -> None:
    prompt = build_prompt(TASK, None)

    # A model not told the exact keys reports correct work under its own names.
    for key in ("selection", "histogram", "yield_estimate", "INPUT_PATH", "OUTPUT_PATH"):
        assert key in prompt
    assert "rejected" not in prompt


def test_feedback_is_appended_when_an_attempt_was_rejected() -> None:
    prompt = build_prompt(TASK, "contract_id=x stage=post_histogram")

    assert "A previous attempt was rejected" in prompt
    assert "contract_id=x" in prompt


def test_a_model_returning_no_code_is_an_error() -> None:
    client, _ = _client(_completion("   "))

    with pytest.raises(GeminiError, match="no extractable code"):
        GeminiAgent(client=client, model_id="m").propose(TASK, attempt_id="a0")


# --- judge ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "accepted"),
    [("VALID", True), ("INVALID", False), ("  invalid  ", False), ("VALID.", True)],
)
def test_the_judge_reads_a_one_word_verdict(reply: str, accepted: bool) -> None:
    client, _ = _client(_completion(reply))

    assert GeminiJudge(client=client, model_id="m").verdict({"estimated_yield": 1.0}) is accepted


def test_the_judge_sees_only_the_final_artifact() -> None:
    client, calls = _client(_completion("VALID"))

    GeminiJudge(client=client, model_id="m").verdict({"estimated_yield": 7.47})

    prompt = _sent_body(calls)["contents"][0]["parts"][0]["text"]
    assert "7.47" in prompt
    assert "selection" not in prompt
    assert "histogram" not in prompt


# --- model resolution ----------------------------------------------------------------


def test_a_requested_model_resolves_through_a_version_suffix() -> None:
    available = ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite-preview-01", "gemini-3.7-flash")

    assert resolve_model_id(available, "gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"
    assert resolve_model_id(available, "gemini-3.7") == "gemini-3.7-flash"


def test_an_unavailable_model_is_an_error() -> None:
    with pytest.raises(GeminiError, match="no available model matches"):
        resolve_model_id(("gemini-3.7-flash",), "gpt-5")


# --- transient failures --------------------------------------------------------------


def test_a_transient_status_is_retried_then_succeeds() -> None:
    """Losing a run to one 503 would waste the quota already spent on it."""

    statuses = [503, 429, 200]
    slept: list[float] = []

    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        status = statuses.pop(0)
        payload = _completion("ok") if status == 200 else {"error": {"message": "busy"}}
        return status, json.dumps(payload).encode("utf-8")

    client = GeminiClient(
        api_key=SECRET, transport=transport, retry_backoff_seconds=1.0, _sleep=slept.append
    )

    assert client.generate("m", "prompt").text == "ok"
    assert slept == [1.0, 2.0]


def test_retries_are_bounded() -> None:
    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        return 503, b'{"error": {"message": "busy"}}'

    client = GeminiClient(
        api_key=SECRET,
        transport=transport,
        max_retries=2,
        retry_backoff_seconds=0.0,
        _sleep=lambda _: None,
    )

    with pytest.raises(GeminiError, match="HTTP 503"):
        client.generate("m", "prompt")


def test_a_permanent_status_is_not_retried() -> None:
    attempts = [0]

    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        attempts[0] += 1
        return 400, b'{"error": {"message": "bad request"}}'

    client = GeminiClient(api_key=SECRET, transport=transport, _sleep=lambda _: None)

    with pytest.raises(GeminiError, match="HTTP 400"):
        client.generate("m", "prompt")
    assert attempts[0] == 1


def test_a_transport_failure_is_retried_like_an_upstream_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped connection is as transient as a 503 and must not lose the run."""

    import urllib.request

    from sciagentguard.adapters.agent.gemini import TRANSPORT_FAILURE_STATUS, _urllib_transport

    assert TRANSPORT_FAILURE_STATUS in RETRYABLE_STATUSES

    def failing_open(*args: object, **kwargs: object) -> object:
        raise TimeoutError("the read operation timed out")

    monkeypatch.setattr(urllib.request, "urlopen", failing_open)
    status, body = _urllib_transport("https://example.invalid", {}, None)

    assert status == TRANSPORT_FAILURE_STATUS
    assert b"TimeoutError" in body


def test_an_exhausted_output_budget_says_what_to_change() -> None:
    """Reasoning tokens come out of the output budget, which is not obvious from the API."""

    client, _ = _client({"candidates": [{"finishReason": "MAX_TOKENS"}]})

    with pytest.raises(GeminiError, match="raise max_output_tokens"):
        client.generate("m", "prompt")


def test_a_thinking_level_reaches_the_request_only_when_asked() -> None:
    client, calls = _client(_completion("x"))
    client.generate("m", "prompt")
    assert "thinkingConfig" not in _sent_body(calls)["generationConfig"]

    client2, calls2 = _client(_completion("x"))
    client2.generate("m", "prompt", thinking_level="low")
    assert _sent_body(calls2)["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
