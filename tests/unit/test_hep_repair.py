from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from sciagentguard.core import ScientificContract, SemanticFaultInjector
from sciagentguard.packs.hep import (
    DeclaredEventProvenanceContract,
    FiniteWeightsContract,
    JetPtRangeContract,
    MissingBranchInjector,
    NonfiniteWeightsInjector,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
    SyntheticHEPRepairPolicy,
    SyntheticHEPWorkflow,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    ZeroWeightsInjector,
)
from sciagentguard.runtime import RepairAction, RepairOutcome, RepairRunner


def contracts() -> tuple[ScientificContract, ...]:
    return (
        RequiredBranchesContract(),
        FiniteWeightsContract(),
        NonzeroWeightSupportContract(),
        JetPtRangeContract(),
        DeclaredEventProvenanceContract(),
    )


def test_zero_weights_are_recovered_by_reloading_the_declared_fixture() -> None:
    workflow = SyntheticHEPWorkflow(fault=ZeroWeightsInjector())
    initial = workflow.initial_step()
    initial_events = cast(Mapping[str, tuple[object, ...]], initial.artifacts["events"])

    execution = RepairRunner().execute(
        lambda: initial,
        workflow.repair_step,
        contracts(),
        SyntheticHEPRepairPolicy(),
    )

    assert execution.output is not None
    output_events = cast(Mapping[str, tuple[object, ...]], execution.output.artifacts["events"])
    assert execution.trace.outcome is RepairOutcome.REPAIRED
    assert [attempt.execution.blocked for attempt in execution.trace.attempts] == [True, False]
    assert execution.trace.attempts[0].action is not None
    assert execution.trace.attempts[0].action.action_type == (
        "hep.reload_declared_synthetic_source"
    )
    assert execution.output.attempt_id == "attempt-0.repair-1"
    assert output_events["weight"] == (1.0, 0.8, -0.2, 1.1, 0.5, 0.9)
    assert initial_events["weight"] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "fault",
    [
        MissingBranchInjector(),
        NonfiniteWeightsInjector(),
        UnitScaleErrorInjector(),
        UndeclaredSyntheticDataInjector(),
    ],
)
def test_policy_declines_faults_without_a_supported_repair(
    fault: SemanticFaultInjector,
) -> None:
    workflow = SyntheticHEPWorkflow(fault=fault)

    execution = RepairRunner().execute(
        workflow.initial_step,
        workflow.repair_step,
        contracts(),
        SyntheticHEPRepairPolicy(),
    )

    assert execution.output is None
    assert execution.trace.outcome is RepairOutcome.UNRESOLVED
    assert len(execution.trace.attempts) == 1


def test_synthetic_workflow_rejects_unknown_or_misdirected_actions() -> None:
    workflow = SyntheticHEPWorkflow()
    unknown = RepairAction(
        action_id="test-action",
        action_type="hep.unknown",
        rationale="Exercise the trusted repair boundary.",
        target_violation_ids=("test-violation",),
    )
    wrong_source = RepairAction(
        action_id="test-action",
        action_type="hep.reload_declared_synthetic_source",
        rationale="Exercise the trusted repair boundary.",
        target_violation_ids=("test-violation",),
        parameters={
            "artifact": "events",
            "source_type": "real",
            "generator": "untrusted",
        },
    )

    with pytest.raises(ValueError, match="unsupported"):
        workflow.repair_step(unknown, attempt_id="attempt-0.repair-1")
    with pytest.raises(ValueError, match="declared fixture"):
        workflow.repair_step(wrong_source, attempt_id="attempt-0.repair-1")


def test_synthetic_workflow_preserves_configured_run_identity() -> None:
    workflow = SyntheticHEPWorkflow(
        workflow_id="configured-workflow",
        run_id="configured-run",
        initial_attempt_id="configured-attempt",
    )
    initial = workflow.initial_step()

    assert initial.workflow_id == "configured-workflow"
    assert initial.run_id == "configured-run"
    assert initial.attempt_id == "configured-attempt"
