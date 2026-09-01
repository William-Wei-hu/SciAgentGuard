"""The DeepSeek judge must never leak the key, and never read a verdict from a reasoning trace."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sciagentguard.adapters.agent.deepseek import (
    API_KEY_VARIABLE,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekJudge,
    load_deepseek_key,
)
from sciagentguard.adapters.agent.gemini import Transport

SECRET = "sk-DEEPSEEK-TEST-SECRET-DO-NOT-LEAK"
Call = tuple[str, dict[str, str], bytes | None]


def _transport(payload: Mapping[str, Any], status: int = 200) -> tuple[Transport, list[Call]]:
    calls: list[Call] = []

    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        calls.append((url, headers, body))
        return status, json.dumps(dict(payload)).encode("utf-8")

    return transport, calls


def _reply(content: str | None, reasoning: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def _client(payload: Mapping[str, Any], status: int = 200) -> tuple[DeepSeekClient, list[Call]]:
    transport, calls = _transport(payload, status)
    return DeepSeekClient(api_key=SECRET, transport=transport), calls


# --- key hygiene ---------------------------------------------------------------------


def test_the_key_never_appears_in_repr() -> None:
    client, _ = _client(_reply("VALID"))

    assert SECRET not in repr(client)
    assert "redacted" in repr(client)


def test_the_key_never_appears_in_an_error() -> None:
    body = {"error": {"message": f"the key {SECRET} is not valid"}}
    client, _ = _client(body, status=401)

    with pytest.raises(DeepSeekError) as raised:
        client.complete("m", "prompt")

    assert "401" in str(raised.value)
    assert SECRET not in str(raised.value)


def test_the_key_travels_as_a_bearer_header() -> None:
    client, calls = _client(_reply("VALID"))

    client.complete("m", "prompt")

    url, headers, _ = calls[0]
    assert SECRET not in url
    assert headers["Authorization"] == f"Bearer {SECRET}"


def test_the_key_is_loaded_from_its_own_variable(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"GEMINI_API_KEY=other\n{API_KEY_VARIABLE}=deepseek-one\n", "utf-8")

    assert load_deepseek_key({}, env_file) == "deepseek-one"


def test_a_missing_key_names_the_variable_to_set(tmp_path: Path) -> None:
    with pytest.raises(DeepSeekError, match=API_KEY_VARIABLE):
        load_deepseek_key({}, tmp_path / "absent")


# --- the verdict comes from the visible answer ----------------------------------------


def test_a_reasoning_trace_is_never_the_verdict() -> None:
    """These models deliberate before answering. The deliberation is not the answer.

    A trace that quotes the artifact back has no place in a recorded result, and a verdict taken
    from a model's private reasoning is not the verdict it gave.
    """

    client, _ = _client(_reply(content="VALID", reasoning="This looks INVALID to me, actually."))

    assert DeepSeekJudge(client=client).verdict({"estimated_yield": 7.4}) is True


def test_a_reply_with_only_reasoning_is_an_error_not_a_verdict() -> None:
    client, _ = _client(_reply(content=None, reasoning="thinking, thinking"))

    with pytest.raises(DeepSeekError, match="no visible content"):
        DeepSeekJudge(client=client).verdict({"estimated_yield": 7.4})


@pytest.mark.parametrize(
    ("reply", "accepted"),
    [("VALID", True), ("INVALID", False), ("  invalid  ", False), ("VALID.", True)],
)
def test_the_judge_reads_a_one_word_verdict(reply: str, accepted: bool) -> None:
    client, _ = _client(_reply(reply))

    assert DeepSeekJudge(client=client).verdict({"estimated_yield": 1.0}) is accepted


def test_the_judge_sees_only_the_final_artifact() -> None:
    client, calls = _client(_reply("VALID"))

    DeepSeekJudge(client=client).verdict({"estimated_yield": 7.47})

    body = calls[0][2]
    assert body is not None
    prompt = json.loads(body)["messages"][0]["content"]
    assert "7.47" in prompt
    assert "selection" not in prompt
    assert "histogram" not in prompt


def test_the_default_judge_is_the_stronger_model() -> None:
    """The judge is the baseline we try to beat, so it is not the cheap model."""

    client, calls = _client(_reply("VALID"))

    DeepSeekJudge(client=client).verdict({"estimated_yield": 1.0})

    body = calls[0][2]
    assert body is not None
    assert json.loads(body)["model"] == "deepseek-v4-pro"


# --- transient failures ---------------------------------------------------------------


def test_a_transient_status_is_retried() -> None:
    statuses = [503, 200]
    slept: list[float] = []

    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        status = statuses.pop(0)
        payload = _reply("VALID") if status == 200 else {"error": {"message": "busy"}}
        return status, json.dumps(payload).encode("utf-8")

    client = DeepSeekClient(
        api_key=SECRET, transport=transport, retry_backoff_seconds=1.0, _sleep=slept.append
    )

    assert client.complete("m", "prompt") == "VALID"
    assert slept == [1.0]


def test_an_empty_key_is_rejected() -> None:
    with pytest.raises(DeepSeekError, match="must not be empty"):
        DeepSeekClient(api_key="  ")


def test_the_output_budget_leaves_room_for_reasoning() -> None:
    """A budget sized from a trivial probe returns no answer at all.

    One verdict on a real analysis artifact was measured at roughly 11,700 output tokens of
    reasoning before the model writes a word. At 4096 and at 8192 the response came back
    finish_reason="length" with empty content, which is how the previous judge failed too.
    """

    from sciagentguard.adapters.agent.deepseek import (
        DEFAULT_MAX_TOKENS,
        OBSERVED_MAX_LATENCY_SECONDS,
        REQUEST_TIMEOUT_SECONDS,
    )

    measured_reasoning_tokens = 11_695

    assert DEFAULT_MAX_TOKENS > measured_reasoning_tokens
    # The timeout must clear the slowest call actually seen, with room to spare. A timeout below it
    # does not lose calls at random: the slow ones are the hard artifacts, so it removes exactly
    # the cases the reviewer had most to say about.
    assert REQUEST_TIMEOUT_SECONDS > OBSERVED_MAX_LATENCY_SECONDS


def test_an_exhausted_budget_says_so_rather_than_returning_a_verdict() -> None:
    exhausted = {"choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}]}
    client, _ = _client(exhausted)

    with pytest.raises(DeepSeekError, match="no visible content"):
        client.complete("m", "prompt")
