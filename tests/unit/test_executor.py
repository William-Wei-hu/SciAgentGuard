from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
    ViolationReport,
    ViolationSeverity,
)
from sciagentguard.packs.hep import (
    MissingBranchInjector,
    RequiredBranchesContract,
    make_synthetic_hep_context,
)
from sciagentguard.runtime import ExecutionTrace, GuardedExecution, GuardedExecutor


class WrongResultIdContract:
    contract_id = "expected"
    description = "Return a result with the wrong identifier for validation tests."
    stage = "post_load"
    required_inputs: tuple[str, ...] = ()

    def evaluate(self, context: ContractContext) -> ContractResult:
        del context
        return ContractResult(
            contract_id="unexpected",
            status=ContractStatus.PASS,
            evidence={},
            duration_ms=0.0,
        )


class ExplodingContract:
    contract_id = "exploding"
    description = "Raise an unexpected software error for validation tests."
    stage = "post_load"
    required_inputs: tuple[str, ...] = ()

    def evaluate(self, context: ContractContext) -> ContractResult:
        del context
        raise RuntimeError("contract implementation failed")


class WarningContract:
    contract_id = "warning"
    description = "Return a warning-level violation for enforcement tests."
    stage = "post_load"
    required_inputs: tuple[str, ...] = ()

    def evaluate(self, context: ContractContext) -> ContractResult:
        violation = ViolationReport(
            violation_id=f"{context.run_id}:{context.attempt_id}:{self.contract_id}",
            contract_id=self.contract_id,
            severity=ViolationSeverity.WARNING,
            stage=context.stage,
            message="A warning-level invariant failed.",
            evidence={"warning": True},
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            timestamp=datetime.now(timezone.utc),
        )
        return ContractResult(
            contract_id=self.contract_id,
            status=ContractStatus.FAIL,
            evidence={"warning": True},
            duration_ms=0.0,
            violation=violation,
        )


def test_executor_returns_output_when_contracts_pass() -> None:
    context = make_synthetic_hep_context()
    execution = GuardedExecutor().execute(lambda: context, [RequiredBranchesContract()])

    assert execution.output is context
    assert execution.trace.blocked is False
    assert ExecutionTrace.model_validate_json(execution.trace.model_dump_json()) == execution.trace


def test_executor_withholds_output_when_a_blocking_contract_fails() -> None:
    context = MissingBranchInjector().inject(make_synthetic_hep_context())
    execution = GuardedExecutor().execute(lambda: context, [RequiredBranchesContract()])

    assert execution.output is None
    assert execution.trace.blocked is True


def test_executor_trace_does_not_serialize_runtime_artifacts() -> None:
    context = make_synthetic_hep_context()
    artifacts = dict(context.artifacts)
    artifacts["private_runtime_value"] = "sentinel-secret-that-must-not-be-traced"
    context = replace(context, artifacts=artifacts)

    trace_json = (
        GuardedExecutor()
        .execute(lambda: context, [RequiredBranchesContract()])
        .trace.model_dump_json()
    )

    assert "sentinel-secret-that-must-not-be-traced" not in trace_json
    assert "private_runtime_value" not in trace_json


def test_executor_rejects_empty_contracts_and_stage_mismatches() -> None:
    context = make_synthetic_hep_context()
    with pytest.raises(ValueError, match="at least one contract"):
        GuardedExecutor().execute(lambda: context, [])

    wrong_stage = replace(context, stage="analysis")
    with pytest.raises(ValueError, match="targets stage"):
        GuardedExecutor().execute(lambda: wrong_stage, [RequiredBranchesContract()])


def test_executor_rejects_duplicate_and_inconsistent_contract_ids() -> None:
    context = make_synthetic_hep_context()
    contract = RequiredBranchesContract()

    with pytest.raises(ValueError, match="must be unique"):
        GuardedExecutor().execute(lambda: context, [contract, contract])
    with pytest.raises(ValueError, match="returned result"):
        GuardedExecutor().execute(lambda: context, [WrongResultIdContract()])


def test_executor_propagates_unexpected_step_and_contract_errors() -> None:
    def failed_step() -> ContractContext:
        raise RuntimeError("workflow step failed")

    with pytest.raises(RuntimeError, match="workflow step failed"):
        GuardedExecutor().execute(failed_step, [RequiredBranchesContract()])
    with pytest.raises(RuntimeError, match="contract implementation failed"):
        GuardedExecutor().execute(make_synthetic_hep_context, [ExplodingContract()])


def test_executor_requires_a_blocking_severity() -> None:
    with pytest.raises(ValueError, match="blocking severity"):
        GuardedExecutor(set())


def test_executor_blocks_only_configured_severities() -> None:
    context = make_synthetic_hep_context()

    default_execution = GuardedExecutor().execute(lambda: context, [WarningContract()])
    warning_execution = GuardedExecutor({ViolationSeverity.WARNING}).execute(
        lambda: context, [WarningContract()]
    )

    assert default_execution.output is context
    assert default_execution.trace.blocked is False
    assert warning_execution.output is None
    assert warning_execution.trace.blocked is True


def test_trace_and_runtime_output_reject_contradictory_states() -> None:
    execution = GuardedExecutor({ViolationSeverity.ERROR}).execute(
        make_synthetic_hep_context, [RequiredBranchesContract()]
    )
    values = execution.trace.model_dump()
    values["blocked"] = True

    with pytest.raises(ValidationError, match="blocked must match"):
        ExecutionTrace.model_validate(values)
    with pytest.raises(ValueError, match="unblocked execution"):
        GuardedExecution(output=None, trace=execution.trace)
