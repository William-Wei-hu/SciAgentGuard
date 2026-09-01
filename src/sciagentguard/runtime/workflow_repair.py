"""Bounded repair across a whole guarded workflow.

`RepairRunner` repairs one checkpoint: its trace requires every attempt to share a single stage.
An agent that rewrites its analysis does not repair a stage, it produces a new workflow, so this
module runs the bounded loop over an entire checkpoint sequence instead. It reuses the existing
`RepairOutcome` vocabulary and the same rule that only revalidation may resolve a violation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from sciagentguard.core import ContractContext, ViolationReport, ViolationSeverity
from sciagentguard.core.models import NonEmptyString
from sciagentguard.runtime.repair import RepairOutcome
from sciagentguard.runtime.workflow import (
    GuardedWorkflowRunner,
    WorkflowCheckpoint,
    WorkflowTrace,
)

AttemptFactory = Callable[[str, str | None], Sequence[WorkflowCheckpoint]]
"""Build the checkpoints of one attempt from its identifier and the feedback that preceded it."""

FeedbackFormatter = Callable[[tuple[ViolationReport, ...]], str]
"""Render blocking violations into the text handed back to the author of the workflow."""


class WorkflowRepairAttempt(BaseModel):
    """One guarded workflow run and the feedback returned after it, if any."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    trace: WorkflowTrace
    feedback: str | None = None

    @model_validator(mode="after")
    def require_feedback_only_after_a_block(self) -> WorkflowRepairAttempt:
        if self.feedback is not None and not self.trace.blocked:
            raise ValueError("an unblocked attempt must not produce repair feedback")
        return self


class WorkflowRepairTrace(BaseModel):
    """Serializable history and terminal state of a bounded workflow repair run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workflow_id: NonEmptyString
    run_id: NonEmptyString
    mode: Literal["workflow_repair"] = "workflow_repair"
    max_repair_attempts: int
    outcome: RepairOutcome
    attempts: tuple[WorkflowRepairAttempt, ...]
    timestamp: datetime
    duration_ms: float
    schema_version: NonEmptyString = "1.0"

    @field_validator("max_repair_attempts")
    @classmethod
    def require_nonnegative_limit(cls, limit: int) -> int:
        if isinstance(limit, bool) or limit < 0:
            raise ValueError("max_repair_attempts must be a nonnegative integer")
        return limit

    @field_validator("attempts")
    @classmethod
    def require_attempt(
        cls, attempts: tuple[WorkflowRepairAttempt, ...]
    ) -> tuple[WorkflowRepairAttempt, ...]:
        if not attempts:
            raise ValueError("a workflow repair trace must contain at least one attempt")
        return attempts

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
    def require_consistent_history(self) -> WorkflowRepairTrace:
        if len(self.attempts) > self.max_repair_attempts + 1:
            raise ValueError("workflow repair trace exceeds max_repair_attempts")
        if self.attempts[-1].feedback is not None:
            raise ValueError("the terminal attempt must not carry feedback")
        if any(attempt.feedback is None for attempt in self.attempts[:-1]):
            raise ValueError("only the terminal attempt may omit feedback")

        run_ids = []
        for attempt in self.attempts:
            if attempt.trace.workflow_id != self.workflow_id:
                raise ValueError("repair trace and attempt workflow_id values must match")
            if attempt.trace.run_id != self.run_id:
                raise ValueError("repair trace and attempt run_id values must match")
            run_ids.append(attempt.trace.attempt_id)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("workflow repair attempt identifiers must be unique")

        repairs = len(self.attempts) - 1
        if not self.attempts[-1].trace.blocked:
            expected = RepairOutcome.PASSED if repairs == 0 else RepairOutcome.REPAIRED
        elif repairs == self.max_repair_attempts:
            expected = RepairOutcome.EXHAUSTED
        else:
            expected = RepairOutcome.UNRESOLVED
        if self.outcome is not expected:
            raise ValueError("workflow repair outcome must match the terminal attempt")
        return self

    @property
    def repair_attempts_used(self) -> int:
        return len(self.attempts) - 1


@dataclass(frozen=True, slots=True)
class WorkflowRepairExecution:
    """Validated final output paired with the complete bounded repair history."""

    output: ContractContext | None
    trace: WorkflowRepairTrace

    def __post_init__(self) -> None:
        succeeded = self.trace.outcome in {RepairOutcome.PASSED, RepairOutcome.REPAIRED}
        if succeeded and self.output is None:
            raise ValueError("a successful repair execution must expose its validated output")
        if not succeeded and self.output is not None:
            raise ValueError("an unsuccessful repair execution must not expose workflow output")


class WorkflowRepairRunner:
    """Re-run a guarded workflow, with bounded feedback, until it revalidates or the limit."""

    def __init__(
        self,
        max_repair_attempts: int = 2,
        blocking_severities: Set[ViolationSeverity] | None = None,
    ) -> None:
        if isinstance(max_repair_attempts, bool) or max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be a nonnegative integer")
        self._max_repair_attempts = max_repair_attempts
        self._runner = GuardedWorkflowRunner(blocking_severities)

    def execute(
        self,
        attempt_factory: AttemptFactory,
        format_feedback: FeedbackFormatter,
    ) -> WorkflowRepairExecution:
        """Run attempts until the workflow revalidates or the repair budget is spent."""

        start_ns = perf_counter_ns()
        attempts: list[WorkflowRepairAttempt] = []
        output: ContractContext | None = None
        feedback: str | None = None

        for index in range(self._max_repair_attempts + 1):
            attempt_id = f"attempt-{index}"
            execution = self._runner.execute(attempt_factory(attempt_id, feedback))
            trace = execution.trace
            if not trace.blocked:
                attempts.append(WorkflowRepairAttempt(trace=trace))
                output = execution.output
                break
            if index == self._max_repair_attempts:
                attempts.append(WorkflowRepairAttempt(trace=trace))
                break
            feedback = format_feedback(_blocking_violations(trace))
            attempts.append(WorkflowRepairAttempt(trace=trace, feedback=feedback))

        repairs = len(attempts) - 1
        if not attempts[-1].trace.blocked:
            outcome = RepairOutcome.PASSED if repairs == 0 else RepairOutcome.REPAIRED
        elif repairs == self._max_repair_attempts:
            outcome = RepairOutcome.EXHAUSTED
        else:
            outcome = RepairOutcome.UNRESOLVED

        return WorkflowRepairExecution(
            output=output,
            trace=WorkflowRepairTrace(
                workflow_id=attempts[0].trace.workflow_id,
                run_id=attempts[0].trace.run_id,
                max_repair_attempts=self._max_repair_attempts,
                outcome=outcome,
                attempts=tuple(attempts),
                timestamp=datetime.now(timezone.utc),
                duration_ms=(perf_counter_ns() - start_ns) / 1_000_000,
            ),
        )


def _blocking_violations(trace: WorkflowTrace) -> tuple[ViolationReport, ...]:
    return tuple(
        result.violation
        for checkpoint in trace.checkpoints
        for result in checkpoint.results
        if result.violation is not None and result.violation.severity in trace.blocking_severities
    )


__all__ = [
    "AttemptFactory",
    "FeedbackFormatter",
    "WorkflowRepairAttempt",
    "WorkflowRepairExecution",
    "WorkflowRepairRunner",
    "WorkflowRepairTrace",
]
