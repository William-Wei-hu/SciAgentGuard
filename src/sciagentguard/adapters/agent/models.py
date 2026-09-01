"""Typed records exchanged with an analysis agent and its sandbox."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from sciagentguard.core.models import NonEmptyString


class SandboxOutcome(str, Enum):
    """How one sandboxed execution of agent-written code ended."""

    COMPLETED = "completed"
    RUNTIME_ERROR = "runtime_error"
    DENIED_OPERATION = "denied_operation"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    INVALID_OUTPUT = "invalid_output"


class AgentTask(BaseModel):
    """One analysis assignment given to an agent."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    task_id: NonEmptyString
    description: NonEmptyString
    input_description: NonEmptyString
    expected_outputs: tuple[NonEmptyString, ...]
    schema_version: NonEmptyString = "1.0"

    @field_validator("expected_outputs")
    @classmethod
    def require_outputs(cls, outputs: tuple[str, ...]) -> tuple[str, ...]:
        if not outputs:
            raise ValueError("a task must declare at least one expected output")
        return outputs


class AgentProposal(BaseModel):
    """Analysis code proposed by an agent, with the provenance of its generation.

    Every provenance field is optional so that a deterministic scripted agent may leave them
    empty. A model-backed agent must populate `model_id`, `provider`, `seed`, `prompt_hash`, and
    the sampling parameters, because a run that cannot state how it was generated is not evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    proposal_id: NonEmptyString
    task_id: NonEmptyString
    attempt_id: NonEmptyString
    # Deliberately a plain `str`: `NonEmptyString` strips surrounding whitespace, and silently
    # rewriting an agent's source would make the executed code differ from the recorded one.
    code: str
    source: NonEmptyString
    model_id: str | None = None
    provider: str | None = None
    seed: int | None = None
    prompt_hash: str | None = None
    sampling_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    latency_ms: float | None = None
    cost_usd: float | None = None
    schema_version: NonEmptyString = "1.0"

    @field_validator("code")
    @classmethod
    def require_code(cls, code: str) -> str:
        if not code.strip():
            raise ValueError("a proposal must contain code")
        return code

    @field_validator("latency_ms", "cost_usd")
    @classmethod
    def require_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("latency and cost must be nonnegative")
        return value


class SandboxResult(BaseModel):
    """Safe record of one sandboxed execution.

    The captured streams are truncated and are never parsed as workflow state. Artifacts are read
    from the JSON file the child wrote, never unpickled and never executed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    outcome: SandboxOutcome
    exit_code: int | None
    duration_ms: float
    stderr_excerpt: str = ""
    artifacts: dict[str, JsonValue] | None = None
    schema_version: NonEmptyString = "1.0"

    @field_validator("duration_ms")
    @classmethod
    def require_nonnegative_duration(cls, duration_ms: float) -> float:
        if duration_ms < 0:
            raise ValueError("duration_ms must be nonnegative")
        return duration_ms

    @property
    def produced_artifacts(self) -> bool:
        return self.outcome is SandboxOutcome.COMPLETED and self.artifacts is not None


__all__ = [
    "AgentProposal",
    "AgentTask",
    "SandboxOutcome",
    "SandboxResult",
]
