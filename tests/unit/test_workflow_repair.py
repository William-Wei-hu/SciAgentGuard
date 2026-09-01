"""Bounded repair across a whole workflow: only revalidation may resolve a violation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
    ViolationReport,
    ViolationSeverity,
)
from sciagentguard.runtime import (
    RepairOutcome,
    WorkflowCheckpoint,
    WorkflowRepairRunner,
)
from sciagentguard.runtime.workflow_repair import AttemptFactory

STAGE = "post_load"


class _AlwaysPasses:
    contract_id = "test.always_passes"
    description = "A contract that never fails."
    stage = STAGE
    required_inputs = ("events",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        del context
        return ContractResult(
            contract_id=self.contract_id, status=ContractStatus.PASS, evidence={}, duration_ms=0.0
        )


class _FailsUntilRepaired:
    """Fail while the context declares itself faulty."""

    contract_id = "test.requires_repair"
    description = "A contract that fails until the workflow is rebuilt."
    stage = STAGE
    required_inputs = ("events",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        if not context.artifacts.get("faulty"):
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence={},
                duration_ms=0.0,
            )
        return ContractResult(
            contract_id=self.contract_id,
            status=ContractStatus.FAIL,
            evidence={"faulty": True},
            duration_ms=0.0,
            violation=ViolationReport(
                violation_id=f"{context.run_id}:{context.attempt_id}:{self.contract_id}",
                contract_id=self.contract_id,
                severity=ViolationSeverity.ERROR,
                stage=context.stage,
                message="The workflow is still faulty.",
                evidence={"faulty": True},
                run_id=context.run_id,
                attempt_id=context.attempt_id,
                timestamp=datetime.now(timezone.utc),
            ),
        )


def _context(attempt_id: str, *, faulty: bool) -> ContractContext:
    return ContractContext(
        workflow_id="repair-test",
        run_id="run-001",
        attempt_id=attempt_id,
        stage=STAGE,
        artifacts={"events": {}, "faulty": faulty},
    )


def _factory(
    repairs_after: int | None,
) -> tuple[AttemptFactory, list[str]]:
    """Build a workflow that stops being faulty after `repairs_after` pieces of feedback."""

    seen: list[str] = []

    def factory(attempt_id: str, feedback: str | None) -> Sequence[WorkflowCheckpoint]:
        if feedback is not None:
            seen.append(feedback)
        faulty = repairs_after is None or len(seen) < repairs_after
        context = _context(attempt_id, faulty=faulty)
        return (WorkflowCheckpoint(lambda: context, (_AlwaysPasses(), _FailsUntilRepaired())),)

    return factory, seen


def _feedback(violations: tuple[ViolationReport, ...]) -> str:
    return f"contract_id={violations[0].contract_id} stage={violations[0].stage}"


def test_a_clean_workflow_passes_without_using_the_budget() -> None:
    def factory(attempt_id: str, feedback: str | None) -> Sequence[WorkflowCheckpoint]:
        del feedback
        context = _context(attempt_id, faulty=False)
        return (WorkflowCheckpoint(lambda: context, (_AlwaysPasses(),)),)

    execution = WorkflowRepairRunner(2).execute(factory, _feedback)

    assert execution.trace.outcome is RepairOutcome.PASSED
    assert execution.trace.repair_attempts_used == 0
    assert execution.output is not None


def test_a_workflow_repaired_after_feedback_is_revalidated() -> None:
    factory, seen = _factory(repairs_after=1)

    execution = WorkflowRepairRunner(2).execute(factory, _feedback)

    assert execution.trace.outcome is RepairOutcome.REPAIRED
    assert execution.trace.repair_attempts_used == 1
    assert execution.output is not None
    assert len(seen) == 1
    assert "contract_id=test.requires_repair" in seen[0]


def test_an_unrepaired_workflow_exhausts_its_budget_and_withholds_output() -> None:
    factory, seen = _factory(repairs_after=None)

    execution = WorkflowRepairRunner(2).execute(factory, _feedback)

    assert execution.trace.outcome is RepairOutcome.EXHAUSTED
    assert execution.trace.repair_attempts_used == 2
    assert execution.output is None
    assert len(seen) == 2


def test_the_budget_is_never_exceeded() -> None:
    factory, seen = _factory(repairs_after=None)

    execution = WorkflowRepairRunner(0).execute(factory, _feedback)

    assert execution.trace.outcome is RepairOutcome.EXHAUSTED
    assert len(execution.trace.attempts) == 1
    assert seen == []


def test_only_the_terminal_attempt_omits_feedback() -> None:
    factory, _ = _factory(repairs_after=2)

    trace = WorkflowRepairRunner(2).execute(factory, _feedback).trace

    assert [attempt.feedback is None for attempt in trace.attempts] == [False, False, True]
    assert all(attempt.trace.blocked for attempt in trace.attempts[:-1])


def test_a_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        WorkflowRepairRunner(-1)
