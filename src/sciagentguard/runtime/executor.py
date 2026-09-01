"""Enforce-mode execution for one contract-guarded workflow step."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter_ns
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
    ScientificContract,
    ViolationSeverity,
)
from sciagentguard.core.models import NonEmptyString


class ExecutionTrace(BaseModel):
    """Serializable record of a single guarded checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workflow_id: NonEmptyString
    run_id: NonEmptyString
    attempt_id: NonEmptyString
    stage: NonEmptyString
    mode: Literal["enforce"] = "enforce"
    blocking_severities: tuple[ViolationSeverity, ...]
    blocked: bool
    results: tuple[ContractResult, ...]
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

    @field_validator("results")
    @classmethod
    def require_results(cls, results: tuple[ContractResult, ...]) -> tuple[ContractResult, ...]:
        if not results:
            raise ValueError("an execution trace must contain at least one contract result")
        return results

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
    def require_consistent_blocked_state(self) -> ExecutionTrace:
        expected = any(
            result.status is ContractStatus.FAIL
            and result.violation is not None
            and result.violation.severity in self.blocking_severities
            for result in self.results
        )
        if self.blocked is not expected:
            raise ValueError("blocked must match the configured severities and contract results")
        return self


@dataclass(frozen=True, slots=True)
class GuardedExecution:
    """Runtime output paired with its safe, serializable trace."""

    output: ContractContext | None
    trace: ExecutionTrace

    def __post_init__(self) -> None:
        if self.trace.blocked and self.output is not None:
            raise ValueError("a blocked execution must not expose workflow output")
        if not self.trace.blocked and self.output is None:
            raise ValueError("an unblocked execution must expose workflow output")


class GuardedExecutor:
    """Execute one step, evaluate its contracts, and withhold blocked output."""

    def __init__(self, blocking_severities: Set[ViolationSeverity] | None = None) -> None:
        configured = (
            frozenset({ViolationSeverity.ERROR, ViolationSeverity.CRITICAL})
            if blocking_severities is None
            else frozenset(blocking_severities)
        )
        if not configured:
            raise ValueError("enforce mode requires at least one blocking severity")
        self._blocking_severities = configured

    def execute(
        self,
        step: Callable[[], ContractContext],
        contracts: Iterable[ScientificContract],
    ) -> GuardedExecution:
        configured_contracts = tuple(contracts)
        if not configured_contracts:
            raise ValueError("at least one contract is required")

        contract_ids = tuple(contract.contract_id for contract in configured_contracts)
        if len(set(contract_ids)) != len(contract_ids):
            raise ValueError("contract identifiers must be unique within an execution")

        start_ns = perf_counter_ns()
        context = step()
        if not isinstance(context, ContractContext):
            raise TypeError("guarded workflow steps must return ContractContext")

        results: list[ContractResult] = []
        for contract in configured_contracts:
            if contract.stage != context.stage:
                raise ValueError(
                    f"contract {contract.contract_id!r} targets stage {contract.stage!r}, "
                    f"not {context.stage!r}"
                )
            result = contract.evaluate(context)
            if result.contract_id != contract.contract_id:
                raise ValueError(
                    f"contract {contract.contract_id!r} returned result for {result.contract_id!r}"
                )
            if result.violation is not None and result.violation.stage != context.stage:
                raise ValueError(
                    f"contract {contract.contract_id!r} reported stage "
                    f"{result.violation.stage!r}, not {context.stage!r}"
                )
            results.append(result)

        blocked = any(
            result.violation is not None and result.violation.severity in self._blocking_severities
            for result in results
        )
        ordered_severities = tuple(
            severity for severity in ViolationSeverity if severity in self._blocking_severities
        )
        trace = ExecutionTrace(
            workflow_id=context.workflow_id,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            stage=context.stage,
            blocking_severities=ordered_severities,
            blocked=blocked,
            results=tuple(results),
            timestamp=datetime.now(timezone.utc),
            duration_ms=(perf_counter_ns() - start_ns) / 1_000_000,
        )
        return GuardedExecution(output=None if blocked else context, trace=trace)
