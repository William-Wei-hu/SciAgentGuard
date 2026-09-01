"""Enforce-mode execution across ordered scientific checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from sciagentguard.core import ContractContext, ScientificContract, ViolationSeverity
from sciagentguard.core.models import NonEmptyString
from sciagentguard.runtime.executor import ExecutionTrace, GuardedExecutor


@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """One executable step and the contracts that guard its output."""

    step: Callable[[], ContractContext]
    contracts: tuple[ScientificContract, ...]


class WorkflowTrace(BaseModel):
    """Safe aggregate of the checkpoints reached by one workflow run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workflow_id: NonEmptyString
    run_id: NonEmptyString
    attempt_id: NonEmptyString
    mode: Literal["enforce"] = "enforce"
    blocking_severities: tuple[ViolationSeverity, ...]
    blocked: bool
    checkpoints: tuple[ExecutionTrace, ...]
    timestamp: datetime
    duration_ms: float
    schema_version: NonEmptyString = "1.0"

    @field_validator("blocking_severities")
    @classmethod
    def require_blocking_severity(
        cls, severities: tuple[ViolationSeverity, ...]
    ) -> tuple[ViolationSeverity, ...]:
        if not severities:
            raise ValueError("blocking_severities must not be empty in enforce mode")
        if len(set(severities)) != len(severities):
            raise ValueError("blocking_severities must not contain duplicates")
        return severities

    @field_validator("checkpoints")
    @classmethod
    def require_checkpoints(
        cls, checkpoints: tuple[ExecutionTrace, ...]
    ) -> tuple[ExecutionTrace, ...]:
        if not checkpoints:
            raise ValueError("a workflow trace must contain at least one checkpoint")
        return checkpoints

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        return timestamp

    @field_validator("duration_ms")
    @classmethod
    def require_nonnegative_duration(cls, duration_ms: float) -> float:
        if duration_ms < 0:
            raise ValueError("duration_ms must be nonnegative")
        return duration_ms

    @model_validator(mode="after")
    def require_consistent_checkpoints(self) -> WorkflowTrace:
        identity = (self.workflow_id, self.run_id, self.attempt_id)
        for checkpoint in self.checkpoints:
            if (checkpoint.workflow_id, checkpoint.run_id, checkpoint.attempt_id) != identity:
                raise ValueError("checkpoint identity must match the workflow trace")
            if checkpoint.blocking_severities != self.blocking_severities:
                raise ValueError("checkpoint blocking severities must match the workflow trace")
        if any(checkpoint.blocked for checkpoint in self.checkpoints[:-1]):
            raise ValueError("a blocked checkpoint must be the final executed checkpoint")
        if self.blocked is not self.checkpoints[-1].blocked:
            raise ValueError("blocked must match the final checkpoint")
        return self


@dataclass(frozen=True, slots=True)
class GuardedWorkflowExecution:
    """Final workflow output paired with its safe aggregate trace."""

    output: ContractContext | None
    trace: WorkflowTrace

    def __post_init__(self) -> None:
        if self.trace.blocked and self.output is not None:
            raise ValueError("a blocked workflow must not expose output")
        if not self.trace.blocked and self.output is None:
            raise ValueError("an unblocked workflow must expose final output")
        if self.output is not None:
            output_identity = (
                self.output.workflow_id,
                self.output.run_id,
                self.output.attempt_id,
            )
            trace_identity = (self.trace.workflow_id, self.trace.run_id, self.trace.attempt_id)
            if output_identity != trace_identity:
                raise ValueError("workflow output identity must match its trace")
            if self.output.stage != self.trace.checkpoints[-1].stage:
                raise ValueError("workflow output stage must match the final checkpoint")


class GuardedWorkflowRunner:
    """Run guarded checkpoints in order and stop at the first blocking failure."""

    def __init__(self, blocking_severities: Set[ViolationSeverity] | None = None) -> None:
        self._executor = GuardedExecutor(blocking_severities)

    def execute(self, checkpoints: Iterable[WorkflowCheckpoint]) -> GuardedWorkflowExecution:
        configured_checkpoints = tuple(checkpoints)
        if not configured_checkpoints:
            raise ValueError("at least one workflow checkpoint is required")
        if not all(
            isinstance(checkpoint, WorkflowCheckpoint) for checkpoint in configured_checkpoints
        ):
            raise TypeError("workflow checkpoints must be WorkflowCheckpoint instances")

        start_ns = perf_counter_ns()
        traces: list[ExecutionTrace] = []
        final_output: ContractContext | None = None
        identity: tuple[str, str, str] | None = None

        for checkpoint in configured_checkpoints:
            execution = self._executor.execute(checkpoint.step, checkpoint.contracts)
            trace = execution.trace
            checkpoint_identity = (trace.workflow_id, trace.run_id, trace.attempt_id)
            if identity is None:
                identity = checkpoint_identity
            elif checkpoint_identity != identity:
                raise ValueError("workflow, run, and attempt identifiers must remain stable")

            traces.append(trace)
            final_output = execution.output
            if trace.blocked:
                break

        first_trace = traces[0]
        workflow_trace = WorkflowTrace(
            workflow_id=first_trace.workflow_id,
            run_id=first_trace.run_id,
            attempt_id=first_trace.attempt_id,
            blocking_severities=first_trace.blocking_severities,
            blocked=traces[-1].blocked,
            checkpoints=tuple(traces),
            timestamp=datetime.now(timezone.utc),
            duration_ms=(perf_counter_ns() - start_ns) / 1_000_000,
        )
        return GuardedWorkflowExecution(output=final_output, trace=workflow_trace)
