from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from sciagentguard.core import ContractContext
from sciagentguard.packs.hep import (
    MissingBranchInjector,
    RequiredBranchesContract,
    make_synthetic_hep_context,
)
from sciagentguard.runtime import (
    RepairAction,
    RepairOutcome,
    RepairPolicy,
    RepairRequest,
    RepairRunner,
    RepairTrace,
)


def blocked_context() -> ContractContext:
    return MissingBranchInjector().inject(make_synthetic_hep_context())


def action_for(request: RepairRequest, *, action_type: str = "test.restore") -> RepairAction:
    return RepairAction(
        action_id=f"{request.attempt_id}:restore",
        action_type=action_type,
        rationale="Reload the trusted test fixture.",
        target_violation_ids=(request.violations[0].violation_id,),
    )


class RecordingPolicy:
    def __init__(self) -> None:
        self.requests: list[RepairRequest] = []

    def propose(self, request: RepairRequest) -> RepairAction:
        self.requests.append(request)
        return action_for(request)


class DecliningPolicy:
    def __init__(self) -> None:
        self.requests: list[RepairRequest] = []

    def propose(self, request: RepairRequest) -> None:
        self.requests.append(request)
        return None


def restore_fixture(action: RepairAction, *, attempt_id: str) -> ContractContext:
    if action.action_type != "test.restore":
        raise ValueError("unsupported repair action")
    return make_synthetic_hep_context(attempt_id=attempt_id)


def test_valid_context_passes_without_calling_policy() -> None:
    policy = RecordingPolicy()

    execution = RepairRunner().execute(
        make_synthetic_hep_context,
        restore_fixture,
        [RequiredBranchesContract()],
        policy,
    )

    assert execution.output is not None
    assert execution.trace.outcome is RepairOutcome.PASSED
    assert len(execution.trace.attempts) == 1
    assert policy.requests == []


def test_blocked_context_is_repaired_only_after_revalidation() -> None:
    initial = blocked_context()
    policy = RecordingPolicy()

    execution = RepairRunner().execute(
        lambda: initial,
        restore_fixture,
        [RequiredBranchesContract()],
        policy,
    )

    assert execution.output is not None
    assert execution.output.attempt_id == "attempt-0.repair-1"
    assert execution.trace.outcome is RepairOutcome.REPAIRED
    assert [attempt.execution.blocked for attempt in execution.trace.attempts] == [True, False]
    assert execution.trace.attempts[0].action is not None
    assert execution.trace.attempts[1].action is None
    assert "jet_pt_gev" not in cast(dict[str, object], initial.artifacts["events"])


def test_repair_stops_at_the_configured_limit() -> None:
    policy = RecordingPolicy()

    def still_broken(action: RepairAction, *, attempt_id: str) -> ContractContext:
        del action
        context = make_synthetic_hep_context(attempt_id=attempt_id)
        return MissingBranchInjector().inject(context)

    execution = RepairRunner(max_repair_attempts=2).execute(
        blocked_context,
        still_broken,
        [RequiredBranchesContract()],
        policy,
    )

    assert execution.output is None
    assert execution.trace.outcome is RepairOutcome.EXHAUSTED
    assert len(execution.trace.attempts) == 3
    assert [request.remaining_attempts for request in policy.requests] == [2, 1]
    assert [attempt.execution.attempt_id for attempt in execution.trace.attempts] == [
        "attempt-0",
        "attempt-0.repair-1",
        "attempt-0.repair-2",
    ]


def test_zero_repair_limit_exhausts_without_calling_policy() -> None:
    policy = RecordingPolicy()

    execution = RepairRunner(max_repair_attempts=0).execute(
        blocked_context,
        restore_fixture,
        [RequiredBranchesContract()],
        policy,
    )

    assert execution.output is None
    assert execution.trace.outcome is RepairOutcome.EXHAUSTED
    assert policy.requests == []


def test_declined_repair_is_unresolved() -> None:
    policy = DecliningPolicy()

    execution = RepairRunner().execute(
        blocked_context,
        restore_fixture,
        [RequiredBranchesContract()],
        policy,
    )

    assert execution.output is None
    assert execution.trace.outcome is RepairOutcome.UNRESOLVED
    assert len(execution.trace.attempts) == 1
    assert execution.trace.attempts[0].action is None


def test_action_must_target_a_blocking_violation() -> None:
    class InvalidPolicy:
        def propose(self, request: RepairRequest) -> RepairAction:
            return RepairAction(
                action_id="invalid-target",
                action_type="test.restore",
                rationale="Target a violation that was not reported.",
                target_violation_ids=(f"{request.run_id}:missing",),
            )

    with pytest.raises(ValidationError, match="must target violations"):
        RepairRunner().execute(
            blocked_context,
            restore_fixture,
            [RequiredBranchesContract()],
            InvalidPolicy(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workflow_id", "different-workflow", "preserve workflow_id"),
        ("run_id", "different-run", "preserve run_id"),
        ("stage", "different-stage", "preserve stage"),
        ("attempt_id", "unassigned-attempt", "assigned attempt_id"),
    ],
)
def test_repair_step_must_preserve_execution_identity(field: str, value: str, message: str) -> None:
    def drifting_step(action: RepairAction, *, attempt_id: str) -> ContractContext:
        del action
        context = make_synthetic_hep_context(attempt_id=attempt_id)
        if field == "workflow_id":
            return replace(context, workflow_id=value)
        if field == "run_id":
            return replace(context, run_id=value)
        if field == "stage":
            return replace(context, stage=value)
        return replace(context, attempt_id=value)

    with pytest.raises(ValueError, match=message):
        RepairRunner().execute(
            blocked_context,
            drifting_step,
            [RequiredBranchesContract()],
            RecordingPolicy(),
        )


def test_policy_and_repair_step_exceptions_propagate() -> None:
    class ExplodingPolicy:
        def propose(self, request: RepairRequest) -> RepairAction:
            del request
            raise RuntimeError("policy failed")

    def exploding_step(action: RepairAction, *, attempt_id: str) -> ContractContext:
        del action, attempt_id
        raise RuntimeError("repair step failed")

    with pytest.raises(RuntimeError, match="policy failed"):
        RepairRunner().execute(
            blocked_context,
            restore_fixture,
            [RequiredBranchesContract()],
            ExplodingPolicy(),
        )
    with pytest.raises(RuntimeError, match="repair step failed"):
        RepairRunner().execute(
            blocked_context,
            exploding_step,
            [RequiredBranchesContract()],
            RecordingPolicy(),
        )


def test_request_and_trace_do_not_serialize_runtime_secrets() -> None:
    secret = "sentinel-secret-that-must-not-be-traced"
    context = make_synthetic_hep_context()
    context = replace(
        context,
        artifacts={**context.artifacts, "private": secret},
        config={"private": secret},
        provenance={**context.provenance, "private": secret},
    )
    context = MissingBranchInjector().inject(context)
    policy = DecliningPolicy()

    execution = RepairRunner().execute(
        lambda: context,
        restore_fixture,
        [RequiredBranchesContract()],
        policy,
    )

    assert secret not in policy.requests[0].model_dump_json()
    assert secret not in execution.trace.model_dump_json()


def test_repair_models_round_trip_and_reject_invalid_data() -> None:
    execution = RepairRunner().execute(
        blocked_context,
        restore_fixture,
        [RequiredBranchesContract()],
        RecordingPolicy(),
    )

    assert RepairTrace.model_validate_json(execution.trace.model_dump_json()) == execution.trace
    with pytest.raises(ValidationError, match="at least one violation"):
        RepairAction(
            action_id="empty-target",
            action_type="test.restore",
            rationale="No target is invalid.",
            target_violation_ids=(),
        )
    with pytest.raises(ValidationError):
        RepairAction(
            action_id="invalid-parameters",
            action_type="test.restore",
            rationale="Runtime objects are not JSON parameters.",
            target_violation_ids=("violation",),
            parameters={"object": cast(JsonValue, object())},
        )


def test_protocol_assignments_are_structurally_typed() -> None:
    policy: RepairPolicy = RecordingPolicy()

    assert isinstance(policy, RecordingPolicy)
