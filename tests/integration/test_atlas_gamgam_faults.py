from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.core import ContractContext, SemanticFaultInjector
from sciagentguard.packs.hep import (
    AtlasMissingEventProvenanceInjector,
    AtlasMissingPhotonMomentumInjector,
    AtlasPhotonCountMismatchInjector,
    AtlasPhotonScaleGapInjector,
    AtlasSourceIdentityDriftInjector,
    AtlasWeightScaleGapInjector,
    NonfiniteWeightsInjector,
)
from tests.integration._atlas_root import synthetic_source, write_root_file


def _context(tmp_path: Path) -> ContractContext:
    root_path = tmp_path / "test.root"
    write_root_file(root_path)
    return AtlasGamGamOpenDataAdapter(synthetic_source(root_path)).load_context(
        workflow_id="atlas-fault-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )


def _events(context: ContractContext) -> Mapping[str, tuple[object, ...]]:
    return cast(Mapping[str, tuple[object, ...]], context.artifacts["events"])


def test_missing_photon_momentum_removes_only_the_translated_branch(tmp_path: Path) -> None:
    original = _context(tmp_path)
    injected = AtlasMissingPhotonMomentumInjector().inject(original)

    assert injected is not original
    assert "photon_pt_gev" in _events(original)
    assert "photon_pt_gev" not in _events(injected)
    assert _events(injected)["event_number"] == _events(original)["event_number"]


def test_photon_count_mismatch_changes_one_count_without_changing_vectors(
    tmp_path: Path,
) -> None:
    original = _context(tmp_path)
    injected = AtlasPhotonCountMismatchInjector().inject(original)

    original_counts = cast(tuple[int, ...], _events(original)["photon_count"])

    assert original_counts[0] == 3
    assert _events(injected)["photon_count"] == (4, *original_counts[1:])
    assert _events(injected)["photon_pt_gev"] == _events(original)["photon_pt_gev"]


def test_missing_provenance_and_identity_drift_preserve_events(tmp_path: Path) -> None:
    original = _context(tmp_path)
    without_provenance = AtlasMissingEventProvenanceInjector().inject(original)
    drifted = AtlasSourceIdentityDriftInjector().inject(original)
    original_declaration = cast(Mapping[str, object], original.provenance["events"])
    drifted_declaration = cast(Mapping[str, object], drifted.provenance["events"])

    assert "events" not in without_provenance.provenance
    assert original_declaration["record_id"] == "test-atlas-gamgam"
    assert drifted_declaration["record_id"] == "atlas-record-drift"
    assert without_provenance.artifacts["events"] == original.artifacts["events"]
    assert drifted.artifacts["events"] == original.artifacts["events"]


def test_gap_probes_change_only_the_unchecked_scales(tmp_path: Path) -> None:
    original = _context(tmp_path)
    weights_scaled = AtlasWeightScaleGapInjector().inject(original)
    photons_scaled = AtlasPhotonScaleGapInjector().inject(original)

    original_weights = cast(tuple[float, ...], _events(original)["weight"])
    original_momenta = cast(tuple[tuple[float, ...], ...], _events(original)["photon_pt_gev"])

    assert _events(weights_scaled)["weight"] == tuple(value * 10.0 for value in original_weights)
    assert _events(weights_scaled)["photon_pt_gev"] == original_momenta
    assert _events(photons_scaled)["photon_pt_gev"] == tuple(
        tuple(value * 1000.0 for value in row) for row in original_momenta
    )
    assert _events(photons_scaled)["weight"] == _events(original)["weight"]
    assert photons_scaled.units == original.units


@pytest.mark.parametrize(
    "injector",
    [
        AtlasMissingEventProvenanceInjector(),
        AtlasMissingPhotonMomentumInjector(),
        AtlasPhotonCountMismatchInjector(),
        AtlasPhotonScaleGapInjector(),
        AtlasSourceIdentityDriftInjector(),
        AtlasWeightScaleGapInjector(),
    ],
)
def test_atlas_faults_are_deterministic(
    tmp_path: Path,
    injector: SemanticFaultInjector,
) -> None:
    original = _context(tmp_path)

    assert injector.inject(original, seed=1) == injector.inject(original, seed=999)


def test_atlas_faults_declare_the_expected_contract_boundaries() -> None:
    assert AtlasMissingPhotonMomentumInjector().expected_contract_ids == (
        "hep.schema.required_branches",
    )
    assert AtlasPhotonCountMismatchInjector().expected_contract_ids == (
        "hep.atlas_open_data.diphoton_preselection",
    )
    assert AtlasMissingEventProvenanceInjector().expected_contract_ids == (
        "hep.provenance.events_declared",
    )
    assert AtlasSourceIdentityDriftInjector().expected_contract_ids == (
        "hep.atlas_open_data.source_identity",
    )
    assert AtlasWeightScaleGapInjector().expected_contract_ids == ()
    assert AtlasPhotonScaleGapInjector().expected_contract_ids == ()


def test_atlas_faults_reject_invalid_preconditions(tmp_path: Path) -> None:
    context = _context(tmp_path)
    events = dict(_events(context))
    del events["photon_pt_gev"]
    missing_photons = replace(context, artifacts={"events": events})
    with pytest.raises(ValueError, match="requires"):
        AtlasMissingPhotonMomentumInjector().inject(missing_photons)
    with pytest.raises(ValueError, match="requires"):
        AtlasPhotonScaleGapInjector().inject(missing_photons)

    events = dict(_events(context))
    del events["photon_count"]
    with pytest.raises(ValueError, match="requires"):
        AtlasPhotonCountMismatchInjector().inject(replace(context, artifacts={"events": events}))

    without_provenance = replace(context, provenance={})
    with pytest.raises(ValueError, match="requires"):
        AtlasMissingEventProvenanceInjector().inject(without_provenance)
    with pytest.raises(ValueError, match="requires"):
        AtlasSourceIdentityDriftInjector().inject(without_provenance)

    nonfinite = NonfiniteWeightsInjector().inject(context)
    with pytest.raises(ValueError, match="finite weights"):
        AtlasWeightScaleGapInjector().inject(nonfinite)
