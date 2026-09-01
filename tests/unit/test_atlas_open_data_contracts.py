import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import JsonValue

from sciagentguard.core import ContractContext, ContractStatus
from sciagentguard.packs.hep import (
    AtlasDiphotonPreselectionContract,
    AtlasSourceIdentityContract,
)


def _context() -> ContractContext:
    return ContractContext(
        workflow_id="atlas-gamgam",
        run_id="run-001",
        attempt_id="attempt-0",
        stage="post_load",
        artifacts={
            "events": {
                "photon_count": (2, 3, 2),
                "photon_pt_gev": ((50.0, 30.0), (80.0, 40.0, 25.0), (65.0, 35.0)),
            }
        },
        units={"photon_pt_gev": "GeV"},
        provenance={
            "events": {
                "source_type": "simulation",
                "experiment": "ATLAS",
                "record_id": "atlas-15006",
                "doi": "10.7483/OPENDATA.ATLAS.B5BJ.3SGS",
                "file_name": "sample.root",
                "checksum": "adler32:12345678",
            }
        },
        config={"atlas_gamgam": {"minimum_photons": 2}},
    )


def _source_identity_contract() -> AtlasSourceIdentityContract:
    return AtlasSourceIdentityContract(
        expected_source_type="simulation",
        expected_record_id="atlas-15006",
        expected_doi="10.7483/OPENDATA.ATLAS.B5BJ.3SGS",
        expected_file_name="sample.root",
        expected_checksum="adler32:12345678",
    )


def test_diphoton_preselection_passes_valid_event_vectors() -> None:
    result = AtlasDiphotonPreselectionContract().evaluate(_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence == {
        "minimum_photons": 2,
        "event_count": 3,
        "observed_photon_count_range": [2, 3],
        "insufficient_photon_count": 0,
        "insufficient_sample_indices": [],
        "length_mismatch_count": 0,
        "length_mismatch_sample_indices": [],
        "nonfinite_momentum_count": 0,
        "nonfinite_sample_indices": [],
        "negative_momentum_count": 0,
        "negative_sample_indices": [],
    }


def test_diphoton_preselection_is_not_applicable_without_photon_branches() -> None:
    context = replace(_context(), artifacts={"events": {"weight": (1.0,)}})

    result = AtlasDiphotonPreselectionContract().evaluate(context)

    assert result.status is ContractStatus.NOT_APPLICABLE
    assert result.evidence["missing_branches"] == ["photon_count", "photon_pt_gev"]


def test_diphoton_preselection_localizes_bounded_structural_failures() -> None:
    context = replace(
        _context(),
        artifacts={
            "events": {
                "photon_count": (1, 3, 2),
                "photon_pt_gev": ((50.0,), (80.0, float("nan")), (65.0, -1.0, 10.0)),
            }
        },
    )

    result = AtlasDiphotonPreselectionContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence == {
        "minimum_photons": 2,
        "event_count": 3,
        "observed_photon_count_range": [1, 3],
        "insufficient_photon_count": 1,
        "insufficient_sample_indices": [0],
        "length_mismatch_count": 2,
        "length_mismatch_sample_indices": [1, 2],
        "nonfinite_momentum_count": 1,
        "nonfinite_sample_indices": [1],
        "negative_momentum_count": 1,
        "negative_sample_indices": [2],
    }
    assert result.violation is not None
    assert result.violation.stage == "post_load"
    assert result.violation.suggested_actions
    payload = json.loads(result.model_dump_json())
    assert "NaN" not in result.model_dump_json()
    assert payload["violation"]["evidence"] == result.evidence


@pytest.mark.parametrize(
    ("config", "units", "message"),
    [
        ({}, {"photon_pt_gev": "GeV"}, "config 'atlas_gamgam'"),
        (
            {"atlas_gamgam": {"minimum_photons": 1}},
            {"photon_pt_gev": "GeV"},
            "minimum_photons",
        ),
        ({"atlas_gamgam": {"minimum_photons": 2}}, {}, "must be declared as 'GeV'"),
    ],
)
def test_diphoton_preselection_rejects_invalid_contract_configuration(
    config: dict[str, JsonValue], units: dict[str, str], message: str
) -> None:
    context = replace(_context(), config=config, units=units)

    with pytest.raises(ValueError, match=message):
        AtlasDiphotonPreselectionContract().evaluate(context)


def test_source_identity_passes_without_copying_arbitrary_provenance() -> None:
    secret = "PROVENANCE_SECRET_SENTINEL"
    provenance = dict(cast(Mapping[str, JsonValue], _context().provenance["events"]))
    provenance["credential"] = secret
    context = replace(_context(), provenance={"events": provenance})

    result = _source_identity_contract().evaluate(context)

    assert result.status is ContractStatus.PASS
    assert secret not in result.model_dump_json()
    assert result.evidence["mismatched_fields"] == []


def test_source_identity_waits_for_a_declared_provenance() -> None:
    result = _source_identity_contract().evaluate(replace(_context(), provenance={}))

    assert result.status is ContractStatus.NOT_APPLICABLE


def test_source_identity_reports_field_names_without_copying_drifted_values() -> None:
    secret = "DRIFTED_SOURCE_SECRET_SENTINEL"
    provenance = dict(cast(Mapping[str, JsonValue], _context().provenance["events"]))
    provenance["record_id"] = secret
    context = replace(_context(), provenance={"events": provenance})

    result = _source_identity_contract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence["mismatched_fields"] == ["record_id"]
    assert result.violation is not None
    assert result.violation.suggested_actions
    assert secret not in result.model_dump_json()
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_closure_tolerance_admits_single_precision_accumulation_but_not_scale_errors() -> None:
    """The source stores weights in float32, so the tolerance must survive float32 summation.

    uproot returns `mcWeight` as float32, and an analysis that sums it without promoting to
    float64 carries a relative error near 1.2e-7. Rejecting that would be a false positive on
    correct work. The tolerance must still be far tighter than any real scale mistake.
    """

    import numpy as np

    weights: Any = np.full(105_503, 0.8768906, dtype=np.float32)
    single = float(np.sum(weights))
    double = float(np.sum(weights.astype(np.float64)))
    accumulation_error = abs(single - double) / abs(double)

    tolerance = 1e-6
    assert accumulation_error < tolerance, "correct float32 work must not be rejected"
    # The smallest scale mistake the closure check exists to catch is a factor of ten.
    assert tolerance < 0.1, "the tolerance must stay far below any real scale error"
