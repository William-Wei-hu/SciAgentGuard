from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from sciagentguard.core import ContractContext
from sciagentguard.packs.hep import (
    MissingBranchInjector,
    RequiredBranchesContract,
    make_synthetic_hep_context,
    make_synthetic_selection_context,
)
from sciagentguard.packs.hep.contracts import NonemptySelectionContract
from sciagentguard.runtime import (
    GuardedWorkflowExecution,
    GuardedWorkflowRunner,
    WorkflowCheckpoint,
    WorkflowTrace,
)


def valid_checkpoints() -> tuple[WorkflowCheckpoint, ...]:
    return (
        WorkflowCheckpoint(make_synthetic_hep_context, (RequiredBranchesContract(),)),
        WorkflowCheckpoint(
            make_synthetic_selection_context,
            (NonemptySelectionContract(),),
        ),
    )


def test_workflow_runs_checkpoints_in_order_and_returns_the_final_context() -> None:
    stages: list[str] = []

    def record(step: str) -> ContractContext:
        stages.append(step)
        return (
            make_synthetic_hep_context()
            if step == "post_load"
            else make_synthetic_selection_context()
        )

    checkpoints = (
        WorkflowCheckpoint(lambda: record("post_load"), (RequiredBranchesContract(),)),
        WorkflowCheckpoint(
            lambda: record("post_selection"),
            (NonemptySelectionContract(),),
        ),
    )

    execution = GuardedWorkflowRunner().execute(checkpoints)

    assert stages == ["post_load", "post_selection"]
    assert execution.output is not None
    assert execution.output.stage == "post_selection"
    assert execution.trace.blocked is False
    assert [trace.stage for trace in execution.trace.checkpoints] == [
        "post_load",
        "post_selection",
    ]
    assert WorkflowTrace.model_validate_json(execution.trace.model_dump_json()) == execution.trace


def test_workflow_stops_after_the_first_blocking_checkpoint() -> None:
    downstream_called = False

    def blocked_step() -> ContractContext:
        return MissingBranchInjector().inject(make_synthetic_hep_context())

    def downstream_step() -> ContractContext:
        nonlocal downstream_called
        downstream_called = True
        return make_synthetic_selection_context()

    execution = GuardedWorkflowRunner().execute(
        (
            WorkflowCheckpoint(blocked_step, (RequiredBranchesContract(),)),
            WorkflowCheckpoint(downstream_step, (NonemptySelectionContract(),)),
        )
    )

    assert execution.output is None
    assert execution.trace.blocked is True
    assert [trace.stage for trace in execution.trace.checkpoints] == ["post_load"]
    assert downstream_called is False


def test_workflow_rejects_empty_configuration_and_identity_drift() -> None:
    with pytest.raises(ValueError, match="at least one workflow checkpoint"):
        GuardedWorkflowRunner().execute(())

    def drifted() -> ContractContext:
        return make_synthetic_selection_context(run_id="different-run")

    with pytest.raises(ValueError, match="identifiers must remain stable"):
        GuardedWorkflowRunner().execute(
            (
                WorkflowCheckpoint(make_synthetic_hep_context, (RequiredBranchesContract(),)),
                WorkflowCheckpoint(drifted, (NonemptySelectionContract(),)),
            )
        )


def test_workflow_propagates_step_and_contract_configuration_errors() -> None:
    def exploding_step() -> ContractContext:
        raise RuntimeError("checkpoint failed")

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        GuardedWorkflowRunner().execute(
            (WorkflowCheckpoint(exploding_step, (RequiredBranchesContract(),)),)
        )
    with pytest.raises(ValueError, match="at least one contract"):
        GuardedWorkflowRunner().execute((WorkflowCheckpoint(make_synthetic_hep_context, ()),))
    with pytest.raises(ValueError, match="targets stage"):
        GuardedWorkflowRunner().execute(
            (
                WorkflowCheckpoint(
                    make_synthetic_selection_context,
                    (RequiredBranchesContract(),),
                ),
            )
        )


def test_workflow_trace_never_serializes_checkpoint_contexts() -> None:
    secret = "WORKFLOW_TRACE_SECRET_SENTINEL"
    context = make_synthetic_hep_context()
    context = replace(
        context,
        artifacts={**context.artifacts, "secret": secret},
        config={**context.config, "secret": secret},
        provenance={**context.provenance, "secret": secret},
    )

    trace_json = (
        GuardedWorkflowRunner()
        .execute((WorkflowCheckpoint(lambda: context, (RequiredBranchesContract(),)),))
        .trace.model_dump_json()
    )

    assert secret not in trace_json


def test_workflow_trace_and_output_reject_contradictory_states() -> None:
    execution = GuardedWorkflowRunner().execute(valid_checkpoints())
    trace_values = execution.trace.model_dump()
    trace_values["blocked"] = True

    with pytest.raises(ValidationError, match="blocked must match"):
        WorkflowTrace.model_validate(trace_values)
    with pytest.raises(ValueError, match="unblocked workflow"):
        GuardedWorkflowExecution(output=None, trace=execution.trace)

    wrong_stage = replace(execution.output, stage="unexpected") if execution.output else None
    with pytest.raises(ValueError, match="final checkpoint"):
        GuardedWorkflowExecution(output=wrong_stage, trace=execution.trace)
