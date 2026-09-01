from collections.abc import Callable
from dataclasses import replace

import pytest

from sciagentguard.core import (
    ContractContext,
    ContractStatus,
    ScientificContract,
    SemanticFaultInjector,
)
from sciagentguard.packs.hep import (
    DeclaredEventProvenanceContract,
    EmptySelectionInjector,
    FiniteWeightsContract,
    JetPtRangeContract,
    MissingBranchInjector,
    NonfiniteWeightsInjector,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
    SplitLeakageInjector,
    SyntheticHEPWorkflow,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    WrongNormalizationInjector,
    ZeroWeightsInjector,
    make_synthetic_hep_context,
)
from sciagentguard.runtime import GuardedExecutor, GuardedWorkflowRunner, WorkflowCheckpoint


def contracts() -> tuple[ScientificContract, ...]:
    return (
        RequiredBranchesContract(),
        FiniteWeightsContract(),
        NonzeroWeightSupportContract(),
        JetPtRangeContract(),
        DeclaredEventProvenanceContract(),
    )


def test_valid_synthetic_workflow_passes_all_contracts() -> None:
    execution = GuardedExecutor().execute(make_synthetic_hep_context, contracts())

    assert execution.output is not None
    assert execution.trace.blocked is False
    assert [result.contract_id for result in execution.trace.results] == [
        "hep.schema.required_branches",
        "hep.weights.finite",
        "hep.weights.nonzero_support",
        "hep.kinematics.jet_pt_range",
        "hep.provenance.events_declared",
    ]
    assert all(result.status is ContractStatus.PASS for result in execution.trace.results)


@pytest.mark.parametrize(
    ("injector", "expected_contract_id"),
    [
        (MissingBranchInjector(), "hep.schema.required_branches"),
        (ZeroWeightsInjector(), "hep.weights.nonzero_support"),
        (NonfiniteWeightsInjector(), "hep.weights.finite"),
        (UnitScaleErrorInjector(), "hep.kinematics.jet_pt_range"),
        (UndeclaredSyntheticDataInjector(), "hep.provenance.events_declared"),
    ],
)
def test_injected_fault_is_detected_and_blocked_at_the_post_load_stage(
    injector: SemanticFaultInjector,
    expected_contract_id: str,
) -> None:
    execution = GuardedExecutor().execute(
        lambda: injector.inject(make_synthetic_hep_context()), contracts()
    )
    failed = [result for result in execution.trace.results if result.status is ContractStatus.FAIL]

    assert execution.output is None
    assert execution.trace.blocked is True
    assert [result.contract_id for result in failed] == [expected_contract_id]
    assert failed[0].violation is not None
    assert failed[0].violation.stage == "post_load"


def test_execution_trace_does_not_copy_extra_provenance_fields() -> None:
    secret = "TRACE_PROVENANCE_SECRET_SENTINEL"
    context = make_synthetic_hep_context()
    context = replace(
        context,
        provenance={
            "events": {
                "source_type": "synthetic",
                "generator": "sciagentguard.packs.hep.fixtures",
                "credential": secret,
            }
        },
    )

    execution = GuardedExecutor().execute(lambda: context, contracts())

    assert execution.trace.blocked is False
    assert secret not in execution.trace.model_dump_json()


def test_valid_synthetic_workflow_reaches_all_four_checkpoints() -> None:
    execution = GuardedWorkflowRunner().execute(SyntheticHEPWorkflow().checkpoints())

    assert execution.output is not None
    assert execution.output.stage == "post_normalization"
    assert [trace.stage for trace in execution.trace.checkpoints] == [
        "post_load",
        "post_selection",
        "post_split",
        "post_normalization",
    ]
    assert all(
        result.status is ContractStatus.PASS
        for trace in execution.trace.checkpoints
        for result in trace.results
    )


@pytest.mark.parametrize(
    ("injector", "expected_contract_id", "expected_stage"),
    [
        (MissingBranchInjector(), "hep.schema.required_branches", "post_load"),
        (ZeroWeightsInjector(), "hep.weights.nonzero_support", "post_load"),
        (NonfiniteWeightsInjector(), "hep.weights.finite", "post_load"),
        (UnitScaleErrorInjector(), "hep.kinematics.jet_pt_range", "post_load"),
        (
            UndeclaredSyntheticDataInjector(),
            "hep.provenance.events_declared",
            "post_load",
        ),
        (EmptySelectionInjector(), "hep.selection.nonempty", "post_selection"),
        (SplitLeakageInjector(), "hep.splits.disjoint_event_ids", "post_split"),
        (
            WrongNormalizationInjector(),
            "hep.normalization.yield_consistent",
            "post_normalization",
        ),
    ],
)
def test_four_stage_workflow_localizes_each_declared_fault(
    injector: SemanticFaultInjector,
    expected_contract_id: str,
    expected_stage: str,
) -> None:
    execution = GuardedWorkflowRunner().execute(SyntheticHEPWorkflow(fault=injector).checkpoints())
    failures = [
        result
        for trace in execution.trace.checkpoints
        for result in trace.results
        if result.status is ContractStatus.FAIL
    ]

    assert execution.output is None
    assert execution.trace.blocked is True
    assert execution.trace.checkpoints[-1].stage == expected_stage
    assert [result.contract_id for result in failures] == [expected_contract_id]
    assert failures[0].violation is not None
    assert failures[0].violation.stage == expected_stage


def test_post_load_failure_prevents_later_hep_steps() -> None:
    checkpoints = SyntheticHEPWorkflow(fault=MissingBranchInjector()).checkpoints()
    later_calls: list[str] = []
    instrumented: tuple[WorkflowCheckpoint, ...] = (checkpoints[0],)

    def record(step: Callable[[], ContractContext], stage: str) -> Callable[[], ContractContext]:
        def recorded_step() -> ContractContext:
            later_calls.append(stage)
            return step()

        return recorded_step

    for checkpoint in checkpoints[1:]:
        instrumented += (
            WorkflowCheckpoint(
                record(checkpoint.step, checkpoint.contracts[0].stage),
                checkpoint.contracts,
            ),
        )

    execution = GuardedWorkflowRunner().execute(instrumented)

    assert execution.trace.blocked is True
    assert later_calls == []
