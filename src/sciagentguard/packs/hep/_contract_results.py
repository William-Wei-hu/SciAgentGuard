"""Shared construction of bounded HEP contract results."""

from datetime import datetime, timezone
from time import perf_counter_ns

from pydantic import JsonValue

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
    ViolationReport,
    ViolationSeverity,
)


def elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000


def failed_result(
    context: ContractContext,
    *,
    contract_id: str,
    evidence: dict[str, JsonValue],
    message: str,
    likely_causes: tuple[str, ...],
    suggested_actions: tuple[str, ...],
    start_ns: int,
    affected_artifacts: tuple[str, ...] = ("events",),
) -> ContractResult:
    violation = ViolationReport(
        violation_id=f"{context.run_id}:{context.attempt_id}:{contract_id}",
        contract_id=contract_id,
        severity=ViolationSeverity.ERROR,
        stage=context.stage,
        message=message,
        evidence=evidence,
        likely_causes=likely_causes,
        suggested_actions=suggested_actions,
        affected_artifacts=affected_artifacts,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        timestamp=datetime.now(timezone.utc),
    )
    return ContractResult(
        contract_id=contract_id,
        status=ContractStatus.FAIL,
        evidence=evidence,
        duration_ms=elapsed_ms(start_ns),
        violation=violation,
    )
