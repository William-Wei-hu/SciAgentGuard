"""Run agent-written analysis code in a confined subprocess.

The sandbox is a guardrail against an agent's accidents, not a security boundary against
adversarial code. It denies networking, subprocesses, and writes outside a temporary working
directory, and it bounds CPU, memory, and wall-clock time. It does not attempt to contain code
that is deliberately trying to escape: `ctypes` remains available because the scientific stack
needs it, and an audit hook can be worked around by code that sets out to do so. Never route
untrusted third-party code through this module.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

from pydantic import JsonValue

from sciagentguard.adapters.agent import _bootstrap
from sciagentguard.adapters.agent.models import SandboxOutcome, SandboxResult

_STDERR_LIMIT = 2_000
_EXIT_RUNTIME_ERROR = 2
_EXIT_CONFINEMENT = 3
_EXIT_MEMORY = 4


@dataclass(frozen=True, slots=True)
class CodeSandbox:
    """Execute proposed code with declared limits and no ambient credentials."""

    timeout_seconds: float = 60.0
    memory_bytes: int = 2 * 1024**3
    cpu_seconds: int = 60

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.cpu_seconds <= 0:
            raise ValueError("cpu_seconds must be positive")

    def run(self, code: str, *, input_path: Path) -> SandboxResult:
        """Execute ``code`` and return the artifacts it wrote, or why it failed."""

        if not code.strip():
            raise ValueError("proposed code must not be empty")
        if not input_path.is_file():
            raise ValueError("the declared sandbox input is not a file")

        start_ns = perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="sciagentguard-agent-") as directory:
            workdir = Path(directory)
            code_path = workdir / "proposal.py"
            output_path = workdir / "artifacts.json"
            config_path = workdir / "sandbox.json"
            code_path.write_text(code, encoding="utf-8")
            config_path.write_text(
                json.dumps(
                    {
                        "workdir": str(workdir),
                        "code_path": str(code_path),
                        "input_path": str(input_path.resolve()),
                        "output_path": str(output_path),
                        "memory_bytes": self.memory_bytes,
                        "cpu_seconds": self.cpu_seconds,
                    }
                ),
                encoding="utf-8",
            )

            completed, timed_out = self._execute(config_path, workdir)
            duration_ms = (perf_counter_ns() - start_ns) / 1_000_000
            if timed_out:
                return SandboxResult(
                    outcome=SandboxOutcome.TIMEOUT,
                    exit_code=None,
                    duration_ms=duration_ms,
                    stderr_excerpt=f"exceeded the {self.timeout_seconds:g}s wall-clock limit",
                )

            assert completed is not None
            stderr = _excerpt(completed.stderr)
            outcome = _classify(completed.returncode, stderr)
            if outcome is not SandboxOutcome.COMPLETED:
                return SandboxResult(
                    outcome=outcome,
                    exit_code=completed.returncode,
                    duration_ms=duration_ms,
                    stderr_excerpt=stderr,
                )

            artifacts = _read_artifacts(output_path)
            if artifacts is None:
                return SandboxResult(
                    outcome=SandboxOutcome.INVALID_OUTPUT,
                    exit_code=completed.returncode,
                    duration_ms=duration_ms,
                    stderr_excerpt=stderr or "no readable JSON artifact object was written",
                )
            return SandboxResult(
                outcome=SandboxOutcome.COMPLETED,
                exit_code=completed.returncode,
                duration_ms=duration_ms,
                stderr_excerpt=stderr,
                artifacts=artifacts,
            )

    def _execute(
        self, config_path: Path, workdir: Path
    ) -> tuple[subprocess.CompletedProcess[str] | None, bool]:
        bootstrap = Path(_bootstrap.__file__).resolve()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(bootstrap), str(config_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=workdir,
                # A minimal environment: no inherited API keys, tokens, or user configuration.
                env={"PATH": "/usr/bin:/bin", "HOME": str(workdir), "TMPDIR": str(workdir)},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, True
        return completed, False


def _classify(returncode: int, stderr: str) -> SandboxOutcome:
    if returncode == 0:
        return SandboxOutcome.COMPLETED
    if returncode == _EXIT_CONFINEMENT or "CONFINEMENT:" in stderr:
        return SandboxOutcome.DENIED_OPERATION
    if returncode == _EXIT_MEMORY or "MemoryError" in stderr:
        return SandboxOutcome.MEMORY_EXCEEDED
    if returncode == _EXIT_RUNTIME_ERROR:
        return SandboxOutcome.RUNTIME_ERROR
    # A CPU-limit kill arrives as a signal, which Python reports as a negative return code.
    if returncode < 0:
        return SandboxOutcome.TIMEOUT
    return SandboxOutcome.RUNTIME_ERROR


def _excerpt(stderr: str) -> str:
    collapsed = stderr.strip()
    if len(collapsed) <= _STDERR_LIMIT:
        return collapsed
    return f"{collapsed[:_STDERR_LIMIT]}... (truncated)"


def _read_artifacts(output_path: Path) -> dict[str, JsonValue] | None:
    if not output_path.is_file():
        return None
    try:
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    artifacts: dict[str, JsonValue] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            return None
        artifacts[key] = value
    return artifacts


__all__ = ["CodeSandbox"]
