import json
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest
from pydantic import JsonValue

from sciagentguard.core import ContractStatus
from sciagentguard.packs.hep import (
    DeclaredEventProvenanceContract,
    FiniteWeightsContract,
    JetPtRangeContract,
    MissingBranchInjector,
    NonfiniteWeightsInjector,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    ZeroWeightsInjector,
    make_synthetic_hep_context,
)


def test_required_branches_contract_passes_the_valid_fixture() -> None:
    result = RequiredBranchesContract().evaluate(make_synthetic_hep_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence["missing_branches"] == []


def test_required_branches_contract_localizes_a_missing_branch() -> None:
    context = MissingBranchInjector().inject(make_synthetic_hep_context())
    result = RequiredBranchesContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence == {
        "required_branches": ["event_id", "jet_pt_gev", "weight"],
        "available_branches": ["event_id", "weight"],
        "missing_branches": ["jet_pt_gev"],
    }
    assert result.violation is not None
    assert result.violation.stage == "post_load"
    assert result.violation.suggested_actions
    assert type(result).model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "branches",
    [(), ("weight", "weight"), (" weight",)],
)
def test_required_branches_contract_rejects_invalid_configuration(
    branches: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        RequiredBranchesContract(branches)


def test_weight_contracts_pass_the_valid_fixture() -> None:
    context = make_synthetic_hep_context()

    assert FiniteWeightsContract().evaluate(context).status is ContractStatus.PASS
    assert NonzeroWeightSupportContract().evaluate(context).status is ContractStatus.PASS


def test_zero_weights_are_localized_to_weighted_support() -> None:
    context = ZeroWeightsInjector().inject(make_synthetic_hep_context())
    finite_result = FiniteWeightsContract().evaluate(context)
    support_result = NonzeroWeightSupportContract().evaluate(context)

    assert finite_result.status is ContractStatus.PASS
    assert support_result.status is ContractStatus.FAIL
    assert support_result.evidence == {"weight_count": 6, "absolute_weight_sum": 0.0}
    assert support_result.violation is not None
    assert support_result.violation.severity.value == "error"
    assert support_result.violation.suggested_actions


def test_nonfinite_weights_are_reported_without_nonfinite_json_evidence() -> None:
    context = NonfiniteWeightsInjector().inject(make_synthetic_hep_context())
    finite_result = FiniteWeightsContract().evaluate(context)
    support_result = NonzeroWeightSupportContract().evaluate(context)

    assert finite_result.status is ContractStatus.FAIL
    assert finite_result.evidence == {
        "weight_count": 6,
        "nonfinite_count": 1,
        "sample_indices": [2],
    }
    assert support_result.status is ContractStatus.NOT_APPLICABLE
    payload = json.loads(finite_result.model_dump_json())
    assert payload["evidence"] == finite_result.evidence
    assert payload["violation"]["evidence"] == finite_result.evidence


def test_weight_contracts_are_not_applicable_without_a_weight_branch() -> None:
    context = make_synthetic_hep_context()
    events = dict(cast(Mapping[str, object], context.artifacts["events"]))
    del events["weight"]
    context = replace(context, artifacts={"events": events})

    assert FiniteWeightsContract().evaluate(context).status is ContractStatus.NOT_APPLICABLE
    assert NonzeroWeightSupportContract().evaluate(context).status is ContractStatus.NOT_APPLICABLE


def test_jet_pt_range_contract_passes_the_declared_fixture() -> None:
    result = JetPtRangeContract().evaluate(make_synthetic_hep_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence["finite_observed_range"] == [33.1, 120.4]


def test_unit_scale_error_is_localized_without_copying_event_values() -> None:
    context = UnitScaleErrorInjector().inject(make_synthetic_hep_context())
    result = JetPtRangeContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence == {
        "expected_unit": "GeV",
        "actual_unit": "GeV",
        "valid_range": [0.0, 500.0],
        "value_count": 6,
        "out_of_range_count": 6,
        "out_of_range_sample_indices": [0, 1, 2, 3, 4, 5],
        "nonfinite_count": 0,
        "nonfinite_sample_indices": [],
        "finite_observed_range": [33100.0, 120400.0],
    }
    assert result.violation is not None
    assert result.violation.severity.value == "error"
    assert result.violation.stage == "post_load"
    assert result.violation.suggested_actions
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_jet_pt_range_contract_is_not_applicable_without_the_branch() -> None:
    context = MissingBranchInjector().inject(make_synthetic_hep_context())

    assert JetPtRangeContract().evaluate(context).status is ContractStatus.NOT_APPLICABLE


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"jet_pt_gev": {"expected_unit": "GeV", "valid_range": [0.0]}},
        {"jet_pt_gev": {"expected_unit": "GeV", "valid_range": [500.0, 0.0]}},
    ],
)
def test_jet_pt_range_contract_rejects_invalid_fixture_assumptions(
    config: dict[str, JsonValue],
) -> None:
    context = replace(make_synthetic_hep_context(), config=config)

    with pytest.raises(ValueError, match="config 'jet_pt_gev"):
        JetPtRangeContract().evaluate(context)


def test_numeric_contracts_reject_non_numeric_event_values() -> None:
    context = make_synthetic_hep_context()
    events = dict(cast(Mapping[str, object], context.artifacts["events"]))
    events["jet_pt_gev"] = (42.0, "unknown", 33.1, 120.4, 55.0, 91.2)
    context = replace(context, artifacts={"events": events})

    with pytest.raises(ValueError, match=r"jet_pt_gev.*index 1"):
        JetPtRangeContract().evaluate(context)


def test_event_provenance_contract_passes_the_declared_fixture() -> None:
    result = DeclaredEventProvenanceContract().evaluate(make_synthetic_hep_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence == {
        "declaration_present": True,
        "source_type_declared": True,
        "synthetic_source_declared": True,
        "generator_required": True,
        "generator_declared": True,
    }


def test_undeclared_synthetic_data_is_localized_without_exposing_provenance() -> None:
    context = UndeclaredSyntheticDataInjector().inject(make_synthetic_hep_context())
    result = DeclaredEventProvenanceContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence == {
        "declaration_present": False,
        "source_type_declared": False,
        "synthetic_source_declared": False,
        "generator_required": False,
        "generator_declared": False,
    }
    assert result.violation is not None
    assert result.violation.stage == "post_load"
    assert result.violation.severity.value == "error"
    assert result.violation.suggested_actions


def test_event_provenance_evidence_does_not_copy_extra_fields() -> None:
    secret = "PROVENANCE_SECRET_SENTINEL"
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

    result = DeclaredEventProvenanceContract().evaluate(context)

    assert result.status is ContractStatus.PASS
    assert secret not in result.model_dump_json()


@pytest.mark.parametrize(
    "declaration",
    [
        {},
        {"source_type": ""},
        {"source_type": "synthetic"},
        {"source_type": "synthetic", "generator": ""},
    ],
)
def test_event_provenance_contract_rejects_incomplete_declarations(
    declaration: dict[str, JsonValue],
) -> None:
    context = replace(make_synthetic_hep_context(), provenance={"events": declaration})

    assert DeclaredEventProvenanceContract().evaluate(context).status is ContractStatus.FAIL


def test_non_synthetic_event_provenance_does_not_require_a_generator() -> None:
    context = replace(
        make_synthetic_hep_context(),
        provenance={"events": {"source_type": "declared_test_source"}},
    )

    assert DeclaredEventProvenanceContract().evaluate(context).status is ContractStatus.PASS


def test_event_provenance_contract_rejects_a_malformed_declaration() -> None:
    context = replace(make_synthetic_hep_context(), provenance={"events": "invalid"})

    with pytest.raises(ValueError, match="provenance 'events' must be a mapping"):
        DeclaredEventProvenanceContract().evaluate(context)
