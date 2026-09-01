"""Bounded repair coordination built on enforce-mode contract checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter_ns
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from sciagentguard.core import (
    ContractContext,
    ScientificContract,
    ViolationReport,
    ViolationSeverity,
)
from sciagentguard.core.models import NonEmptyString
from sciagentguard.runtime.executor import ExecutionTrace, GuardedExecutor


class RepairOutcome(str, Enum):
    """Terminal state of a bounded repair run."""

    PASSED = "passed"
    REPAIRED = "repaired"
    UNRESOLVED = "unresolved"
    EXHAUSTED = "exhausted"


class RepairAction(BaseModel):
    """Structured instruction proposed by a repair policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    action_id: NonEmptyString
    action_type: NonEmptyString
    rationale: NonEmptyString
    target_violation_ids: tuple[NonEmptyString, ...]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("target_violation_ids")
    @classmethod
    def require_unique_targets(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("a repair action must target at least one violation")
        if len(set(values)) != len(values):
            raise ValueError("target_violation_ids must not contain duplicates")
        return values


class RepairRequest(BaseModel):
    """Safe violation summary supplied to a repair policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workflow_id: NonEmptyString
    run_id: NonEmptyString
    attempt_id: NonEmptyString
    stage: NonEmptyString
    violations: tuple[ViolationReport, ...]
    remaining_attempts: int
    schema_version: NonEmptyString = "1.0"

    @field_validator("violations")
    @classmethod
    def require_unique_violations(
        cls, violations: tuple[ViolationReport, ...]
    ) -> tuple[ViolationReport, ...]:
        if not violations:
            raise ValueError("a repair request must contain at least one violation")
        identifiers = tuple(violation.violation_id for violation in violations)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("repair request violations must have unique identifiers")
        return violations

    @field_validator("remaining_attempts")
    @classmethod
    def require_remaining_attempt(cls, remaining_attempts: int) -> int:
        if isinstance(remaining_attempts, bool) or remaining_attempts < 1:
            raise ValueError("remaining_attempts must be a positive integer")
        return remaining_attempts

    @model_validator(mode="after")
    def require_consistent_violation_metadata(self) -> RepairRequest:
        for violation in self.violations:
            if violation.run_id != self.run_id:
                raise ValueError("repair request and violation run_id values must match")
            if violation.attempt_id != self.attempt_id:
                raise ValueError("repair request and violation attempt_id values must match")
            if violation.stage != self.stage:
                raise ValueError("repair request and violation stage values must match")
        return self


class RepairPolicy(Protocol):
    """Choose a bounded structured action from safe violation evidence."""

    def propose(self, request: RepairRequest) -> RepairAction | None:
        """Return an action, or decline when the policy has no supported repair."""
        ...


class RepairStep(Protocol):
    """Trusted workflow entry point that executes one structured repair action."""

    def __call__(self, action: RepairAction, *, attempt_id: str) -> ContractContext:
        """Rerun the guarded step and return a context for revalidation."""
        ...


class RepairAttempt(BaseModel):
    """One enforce-mode result and the action selected after it, if any."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    execution: ExecutionTrace
    action: RepairAction | None = None

    @model_validator(mode="after")
    def require_action_to_target_blocking_violation(self) -> RepairAttempt:
        if self.action is None:
            return self
        if not self.execution.blocked:
            raise ValueError("an unblocked attempt cannot have a repair action")

        blocking_ids = {
            result.violation.violation_id
            for result in self.execution.results
            if result.violation is not None
            and result.violation.severity in self.execution.blocking_severities
        }
        if not set(self.action.target_violation_ids).issubset(blocking_ids):
            raise ValueError("a repair action must target violations from the blocked attempt")
        return self


class RepairTrace(BaseModel):
    """Serializable history and terminal state of a bounded repair run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workflow_id: NonEmptyString
    run_id: NonEmptyString
    stage: NonEmptyString
    mode: Literal["repair"] = "repair"
    max_repair_attempts: int
    outcome: RepairOutcome
    attempts: tuple[RepairAttempt, ...]
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
    def require_attempt(cls, attempts: tuple[RepairAttempt, ...]) -> tuple[RepairAttempt, ...]:
        if not attempts:
            raise ValueError("a repair trace must contain at least one attempt")
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
    def require_consistent_history(self) -> RepairTrace:
        if len(self.attempts) > self.max_repair_attempts + 1:
            raise ValueError("repair trace exceeds max_repair_attempts")
        if self.attempts[-1].action is not None:
            raise ValueError("the terminal repair attempt must not contain an action")
        if any(attempt.action is None for attempt in self.attempts[:-1]):
            raise ValueError("only the terminal repair attempt may omit an action")

        attempt_ids: list[str] = []
        for attempt in self.attempts:
            execution = attempt.execution
            if execution.workflow_id != self.workflow_id:
                raise ValueError("repair trace and attempt workflow_id values must match")
            if execution.run_id != self.run_id:
                raise ValueError("repair trace and attempt run_id values must match")
            if execution.stage != self.stage:
                raise ValueError("repair trace and attempt stage values must match")
            attempt_ids.append(execution.attempt_id)
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("repair attempt identifiers must be unique")

        last_blocked = self.attempts[-1].execution.blocked
        repair_count = len(self.attempts) - 1
        if not last_blocked:
            expected = RepairOutcome.PASSED if repair_count == 0 else RepairOutcome.REPAIRED
        elif repair_count == self.max_repair_attempts:
            expected = RepairOutcome.EXHAUSTED
        else:
            expected = RepairOutcome.UNRESOLVED
        if self.outcome is not expected:
            raise ValueError("repair outcome must match the terminal attempt")
        return self


@dataclass(frozen=True, slots=True)
class RepairExecution:
    """Final safe output paired with a complete bounded repair trace."""

    output: ContractContext | None
    trace: RepairTrace

    def __post_init__(self) -> None:
        succeeded = self.trace.outcome in {RepairOutcome.PASSED, RepairOutcome.REPAIRED}
        if succeeded and self.output is None:
            raise ValueError("a successful repair execution must expose its validated output")
        if not succeeded and self.output is not None:
            raise ValueError("an unsuccessful repair execution must not expose workflow output")
        if self.output is None:
            return

        terminal = self.trace.attempts[-1].execution
        if self.output.workflow_id != self.trace.workflow_id:
            raise ValueError("repair output and trace workflow_id values must match")
        if self.output.run_id != self.trace.run_id:
            raise ValueError("repair output and trace run_id values must match")
        if self.output.stage != self.trace.stage:
            raise ValueError("repair output and trace stage values must match")
        if self.output.attempt_id != terminal.attempt_id:
            raise ValueError("repair output must come from the terminal attempt")


class RepairRunner:
    """Coordinate a bounded sequence of structured repairs and revalidation."""

    def __init__(
        self,
        max_repair_attempts: int = 2,
        blocking_severities: Set[ViolationSeverity] | None = None,
    ) -> None:
        if isinstance(max_repair_attempts, bool) or max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be a nonnegative integer")
        self._max_repair_attempts = max_repair_attempts
        self._executor = GuardedExecutor(blocking_severities)

    def execute(
        self,
        step: Callable[[], ContractContext],
        repair_step: RepairStep,
        contracts: Iterable[ScientificContract],
        policy: RepairPolicy,
    ) -> RepairExecution:
        configured_contracts = tuple(contracts)
        start_ns = perf_counter_ns()
        context = step()
        execution = self._executor.execute(_context_step(context), configured_contracts)
        initial_context = context
        attempts: list[RepairAttempt] = []
        repair_count = 0

        while execution.trace.blocked and repair_count < self._max_repair_attempts:
            violations = _blocking_violations(execution.trace)
            request = RepairRequest(
                workflow_id=execution.trace.workflow_id,
                run_id=execution.trace.run_id,
                attempt_id=execution.trace.attempt_id,
                stage=execution.trace.stage,
                violations=violations,
                remaining_attempts=self._max_repair_attempts - repair_count,
            )
            action = policy.propose(request)
            if action is None:
                attempts.append(RepairAttempt(execution=execution.trace))
                return self._finish(
                    output=None,
                    attempts=attempts,
                    outcome=RepairOutcome.UNRESOLVED,
                    start_ns=start_ns,
                )

            attempts.append(RepairAttempt(execution=execution.trace, action=action))
            repair_count += 1
            next_attempt_id = f"{initial_context.attempt_id}.repair-{repair_count}"
            context = repair_step(action, attempt_id=next_attempt_id)
            _validate_repair_context(context, initial_context, next_attempt_id)
            execution = self._executor.execute(_context_step(context), configured_contracts)

        attempts.append(RepairAttempt(execution=execution.trace))
        if execution.trace.blocked:
            outcome = RepairOutcome.EXHAUSTED
            output = None
        else:
            outcome = RepairOutcome.PASSED if repair_count == 0 else RepairOutcome.REPAIRED
            output = context
        return self._finish(
            output=output,
            attempts=attempts,
            outcome=outcome,
            start_ns=start_ns,
        )

    def _finish(
        self,
        *,
        output: ContractContext | None,
        attempts: list[RepairAttempt],
        outcome: RepairOutcome,
        start_ns: int,
    ) -> RepairExecution:
        first = attempts[0].execution
        trace = RepairTrace(
            workflow_id=first.workflow_id,
            run_id=first.run_id,
            stage=first.stage,
            max_repair_attempts=self._max_repair_attempts,
            outcome=outcome,
            attempts=tuple(attempts),
            timestamp=datetime.now(timezone.utc),
            duration_ms=(perf_counter_ns() - start_ns) / 1_000_000,
        )
        return RepairExecution(output=output, trace=trace)


def _blocking_violations(trace: ExecutionTrace) -> tuple[ViolationReport, ...]:
    return tuple(
        result.violation
        for result in trace.results
        if result.violation is not None and result.violation.severity in trace.blocking_severities
    )


def _context_step(context: ContractContext) -> Callable[[], ContractContext]:
    def return_context() -> ContractContext:
        return context

    return return_context


def _validate_repair_context(
    context: ContractContext,
    initial_context: ContractContext,
    expected_attempt_id: str,
) -> None:
    if not isinstance(context, ContractContext):
        raise TypeError("repair steps must return ContractContext")
    if context.workflow_id != initial_context.workflow_id:
        raise ValueError("repair steps must preserve workflow_id")
    if context.run_id != initial_context.run_id:
        raise ValueError("repair steps must preserve run_id")
    if context.stage != initial_context.stage:
        raise ValueError("repair steps must preserve stage")
    if context.attempt_id != expected_attempt_id:
        raise ValueError("repair steps must use the assigned attempt_id")
