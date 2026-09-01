"""The sandbox must confine an agent's accidents and report each one distinctly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciagentguard.adapters.agent import CodeSandbox, SandboxOutcome

WRITE_ARTIFACTS = 'import json; open(OUTPUT_PATH, "w").write(json.dumps({"ok": 1}))'


@pytest.fixture
def sandbox() -> CodeSandbox:
    return CodeSandbox(timeout_seconds=20.0, cpu_seconds=5, memory_bytes=512 * 1024**2)


@pytest.fixture
def input_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.txt"
    path.write_text("declared-input", encoding="utf-8")
    return path


def test_completed_code_returns_the_artifacts_it_wrote(
    sandbox: CodeSandbox, input_path: Path
) -> None:
    result = sandbox.run(WRITE_ARTIFACTS, input_path=input_path)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 0
    assert result.artifacts == {"ok": 1}
    assert result.produced_artifacts
    assert result.duration_ms >= 0.0


def test_proposed_code_may_read_the_declared_input(sandbox: CodeSandbox, input_path: Path) -> None:
    code = (
        "import json\n"
        "payload = open(INPUT_PATH).read()\n"
        'open(OUTPUT_PATH, "w").write(json.dumps({"read": payload}))\n'
    )

    result = sandbox.run(code, input_path=input_path)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.artifacts == {"read": "declared-input"}


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("dns", 'import socket; socket.getaddrinfo("example.com", 80)'),
        ("connect", 'import socket; socket.socket().connect(("93.184.216.34", 80))'),
        ("urllib", 'import urllib.request; urllib.request.urlopen("http://example.com")'),
        (
            "http_client",
            'import http.client\nhttp.client.HTTPConnection("example.com").request("GET", "/")\n',
        ),
        ("subprocess", 'import subprocess; subprocess.run(["echo", "hi"])'),
        ("os_system", 'import os; os.system("echo hi")'),
    ],
)
def test_egress_and_process_creation_are_denied(
    sandbox: CodeSandbox, input_path: Path, case: str, code: str
) -> None:
    result = sandbox.run(code, input_path=input_path)

    assert result.outcome is SandboxOutcome.DENIED_OPERATION, case
    assert result.artifacts is None
    assert "CONFINEMENT" in result.stderr_excerpt


def test_writes_outside_the_working_directory_are_denied(
    sandbox: CodeSandbox, tmp_path: Path, input_path: Path
) -> None:
    escape = tmp_path / "escaped.txt"
    result = sandbox.run(f'open({str(escape)!r}, "w").write("nope")', input_path=input_path)

    assert result.outcome is SandboxOutcome.DENIED_OPERATION
    assert not escape.exists()


def test_creating_a_socket_object_is_allowed_because_it_is_not_egress(
    sandbox: CodeSandbox, input_path: Path
) -> None:
    # The scientific stack opens an asyncio event loop on import, which allocates a socket.
    # Denying socket construction outright would make the sandbox unusable for real analysis.
    code = f"import socket\nsocket.socket().close()\n{WRITE_ARTIFACTS}\n"

    result = sandbox.run(code, input_path=input_path)

    assert result.outcome is SandboxOutcome.COMPLETED


def test_a_runaway_loop_is_stopped(input_path: Path) -> None:
    sandbox = CodeSandbox(timeout_seconds=30.0, cpu_seconds=1, memory_bytes=512 * 1024**2)

    result = sandbox.run("while True:\n    pass\n", input_path=input_path)

    assert result.outcome is SandboxOutcome.TIMEOUT
    assert result.artifacts is None


def test_a_raised_exception_is_reported_as_a_runtime_error(
    sandbox: CodeSandbox, input_path: Path
) -> None:
    result = sandbox.run('raise ValueError("boom")', input_path=input_path)

    assert result.outcome is SandboxOutcome.RUNTIME_ERROR
    assert result.exit_code == 2
    assert "ValueError" in result.stderr_excerpt


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("no_output", "value = 1 + 1"),
        ("not_json", 'open(OUTPUT_PATH, "w").write("not json")'),
        ("not_an_object", 'open(OUTPUT_PATH, "w").write("[1, 2, 3]")'),
    ],
)
def test_missing_or_unreadable_output_is_distinct_from_a_crash(
    sandbox: CodeSandbox, input_path: Path, case: str, code: str
) -> None:
    result = sandbox.run(code, input_path=input_path)

    assert result.outcome is SandboxOutcome.INVALID_OUTPUT, case
    assert result.exit_code == 0
    assert not result.produced_artifacts


def test_the_child_environment_carries_no_ambient_credentials(
    sandbox: CodeSandbox, input_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SHOULD_NOT_REACH_THE_CHILD")
    code = (
        'import json, os\nopen(OUTPUT_PATH, "w").write(json.dumps({"env": sorted(os.environ)}))\n'
    )

    result = sandbox.run(code, input_path=input_path)

    assert result.artifacts is not None
    assert "ANTHROPIC_API_KEY" not in json.dumps(result.artifacts)


@pytest.mark.parametrize("code", ["", "   \n\t "])
def test_empty_proposals_are_rejected_before_execution(
    sandbox: CodeSandbox, input_path: Path, code: str
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sandbox.run(code, input_path=input_path)


def test_a_missing_input_is_rejected_before_execution(sandbox: CodeSandbox, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        sandbox.run(WRITE_ARTIFACTS, input_path=tmp_path / "absent.root")


@pytest.mark.parametrize(
    "kwargs",
    [{"timeout_seconds": 0}, {"memory_bytes": 0}, {"cpu_seconds": -1}],
)
def test_limits_must_be_positive(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        CodeSandbox(**kwargs)


def test_code_guarded_by_the_main_idiom_still_runs(sandbox: CodeSandbox, input_path: Path) -> None:
    """`if __name__ == "__main__":` is the most common way to end a script.

    A namespace that does not satisfy it would skip the author's entry point entirely, and the
    run would be recorded as producing no output -- blaming the author for the harness's choice.
    """

    code = (
        "import json\n"
        "def main():\n"
        '    open(OUTPUT_PATH, "w").write(json.dumps({"ran": True}))\n'
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    result = sandbox.run(code, input_path=input_path)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.artifacts == {"ran": True}
