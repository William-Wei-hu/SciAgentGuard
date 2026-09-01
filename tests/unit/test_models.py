from datetime import datetime, timezone
from math import inf, nan

import pytest
from pydantic import ValidationError

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
    ViolationReport,
    ViolationSeverity,
)


def make_violation(*, contract_id: str = "weights.finite") -> ViolationReport:
    return ViolationReport(
        violation_id="violation-001",
        contract_id=contract_id,
        severity=ViolationSeverity.ERROR,
        stage="weighting",
        message="Event weights contain non-finite values.",
        evidence={"invalid_count": 2, "sample_indices": [3, 11]},
        likely_causes=("A division used a zero denominator.",),
        suggested_actions=("Inspect the normalization inputs.",),
        affected_artifacts=("events",),
        run_id="run-001",
        attempt_id="attempt-0",
        timestamp=datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),
    )


def test_contract_results_round_trip_through_json() -> None:
    results = (
        ContractResult(
            contract_id="schema.required_fields",
            status=ContractStatus.PASS,
            evidence={"available_fields": ["energy", "weight"]},
            duration_ms=0.4,
        ),
        ContractResult(
            contract_id="weights.finite",
            status=ContractStatus.FAIL,
            evidence={"invalid_count": 2},
            duration_ms=0.8,
            violation=make_violation(),
        ),
        ContractResult(
            contract_id="selection.nonempty",
            status=ContractStatus.NOT_APPLICABLE,
            evidence={"reason": "selection has not run"},
            duration_ms=0.1,
        ),
    )

    for result in results:
        restored = ContractResult.model_validate_json(result.model_dump_json())
        assert restored == result


@pytest.mark.parametrize("duration_ms", [-0.1, inf, nan])
def test_contract_result_rejects_invalid_duration(duration_ms: float) -> None:
    with pytest.raises(ValidationError):
        ContractResult(
            contract_id="schema.required_fields",
            status=ContractStatus.PASS,
            evidence={},
            duration_ms=duration_ms,
        )


def test_failed_result_requires_a_violation() -> None:
    with pytest.raises(ValidationError, match="must include a violation"):
        ContractResult(
            contract_id="weights.finite",
            status=ContractStatus.FAIL,
            evidence={},
            duration_ms=0.2,
        )


def test_nonfailed_result_rejects_a_violation() -> None:
    with pytest.raises(ValidationError, match="only a failed"):
        ContractResult(
            contract_id="weights.finite",
            status=ContractStatus.PASS,
            evidence={},
            duration_ms=0.2,
            violation=make_violation(),
        )


def test_result_and_violation_contract_ids_must_match() -> None:
    with pytest.raises(ValidationError, match="contract_id values must match"):
        ContractResult(
            contract_id="weights.nonzero",
            status=ContractStatus.FAIL,
            evidence={},
            duration_ms=0.2,
            violation=make_violation(contract_id="weights.finite"),
        )


def test_violation_rejects_naive_timestamp_and_extra_fields() -> None:
    report = make_violation()
    values = report.model_dump()
    values["timestamp"] = datetime(2026, 8, 27, 12, 0)

    with pytest.raises(ValidationError, match="timezone"):
        ViolationReport.model_validate(values)

    values = report.model_dump()
    values["unexpected"] = True
    with pytest.raises(ValidationError):
        ViolationReport.model_validate(values)


def test_result_rejects_extra_fields_and_non_json_evidence() -> None:
    values: dict[str, object] = {
        "contract_id": "schema.required_fields",
        "status": "pass",
        "evidence": {},
        "duration_ms": 0.1,
        "unexpected": True,
    }
    with pytest.raises(ValidationError):
        ContractResult.model_validate(values)

    values.pop("unexpected")
    values["evidence"] = {"runtime_object": object()}
    with pytest.raises(ValidationError):
        ContractResult.model_validate(values)


def test_reports_are_immutable() -> None:
    report = make_violation()

    with pytest.raises(ValidationError, match="frozen"):
        report.message = "Changed after validation."


def test_context_copies_mappings_and_normalizes_identifiers() -> None:
    artifacts: dict[str, object] = {"events": object()}
    context = ContractContext(
        workflow_id=" workflow-001 ",
        run_id="run-001",
        attempt_id="attempt-0",
        stage="load",
        artifacts=artifacts,
    )
    artifacts["late-addition"] = object()

    assert context.workflow_id == "workflow-001"
    assert "late-addition" not in context.artifacts


def test_context_rejects_empty_identifiers() -> None:
    with pytest.raises(ValueError, match="stage must not be empty"):
        ContractContext(
            workflow_id="workflow-001",
            run_id="run-001",
            attempt_id="attempt-0",
            stage="  ",
            artifacts={},
        )
