import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.runtime import GuardedWorkflowRunner, WorkflowTrace
from tests.integration._atlas_root import (
    EVENT_COUNT,
    MASS_PER_PT,
    adler32,
    synthetic_source,
    write_root_file,
)


def test_adapter_translates_root_columns_without_exposing_the_path(tmp_path: Path) -> None:
    root_path = tmp_path / "private-user-path" / "test.root"
    root_path.parent.mkdir()
    write_root_file(root_path)

    context = AtlasGamGamOpenDataAdapter(synthetic_source(root_path)).load_context(
        workflow_id="atlas-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )

    assert context.stage == "post_load"
    events = context.artifacts["events"]
    assert isinstance(events, dict)
    assert sorted(events) == [
        "channel_number",
        "cross_section_pb",
        "event_number",
        "generated_weight_sum",
        "photon_count",
        "photon_e_gev",
        "photon_eta",
        "photon_phi",
        "photon_pt_gev",
        "run_number",
        "weight",
    ]
    assert events["event_number"] == tuple(range(101, 101 + EVENT_COUNT))
    assert events["run_number"] == (11,) * EVENT_COUNT
    assert events["channel_number"] == (345318,) * EVENT_COUNT
    # The raw ROOT momenta and energies are stored in MeV and translated to GeV here. The first
    # event is a 125 GeV back-to-back pair carrying a third, softer photon.
    assert events["photon_count"][0] == 3
    leading_pt = pytest.approx(125.0 / MASS_PER_PT)
    assert events["photon_pt_gev"][0] == (leading_pt, leading_pt, 20.0)
    assert events["photon_e_gev"][0] == (leading_pt, leading_pt, 20.0)
    assert events["photon_eta"][0] == (0.0, 0.0, 0.0)
    assert events["cross_section_pb"] == (0.002,) * EVENT_COUNT
    assert events["generated_weight_sum"] == (1000.0,) * EVENT_COUNT
    assert context.units == {
        "weight": "dimensionless",
        "photon_pt_gev": "GeV",
        "photon_e_gev": "GeV",
        "photon_eta": "dimensionless",
        "photon_phi": "rad",
        "cross_section_pb": "pb",
        "generated_weight_sum": "dimensionless",
    }
    declaration = context.provenance["events"]
    assert isinstance(declaration, dict)
    assert {key: declaration[key] for key in declaration if key != "source_facts"} == {
        "source_type": "synthetic",
        "experiment": "ATLAS",
        "record_id": "test-atlas-gamgam",
        "doi": "test-doi",
        "file_name": "test.GamGam.root",
        "checksum": f"adler32:{adler32(root_path)}",
        "generator": "tests.integration._atlas_root",
    }
    # Trusted facts the loader read, for downstream contracts to check an analysis's claims
    # against. Aggregates only: nothing per-event and no local path reaches a trace.
    assert declaration["source_facts"] == {
        "event_count": EVENT_COUNT,
        "weight_min": 0.5,
        "weight_max": 1.0,
        # Magnitudes, not the signed range: a rescaled weight can sit inside a range that
        # straddles zero, so the range alone cannot show where a weight came from.
        "weight_abs_min": 0.5,
        "weight_abs_max": 1.0,
        "distinct_weight_count": 2,
        "distinct_weights": [0.5, 1.0],
        "cross_section_pb": 0.002,
        "generated_weight_sum": 1000.0,
    }
    assert str(tmp_path) not in repr(context)


@pytest.mark.parametrize("field", ["size_bytes", "adler32"])
def test_adapter_rejects_a_source_that_fails_integrity_checks(tmp_path: Path, field: str) -> None:
    root_path = tmp_path / "test.root"
    write_root_file(root_path)
    source = synthetic_source(root_path)
    invalid = (
        replace(source, size_bytes=source.size_bytes + 1)
        if field == "size_bytes"
        else replace(source, adler32="00000000")
    )

    with pytest.raises(ValueError, match=r"size mismatch|Adler-32 mismatch"):
        AtlasGamGamOpenDataAdapter(invalid).load_context(
            workflow_id="atlas-test",
            run_id="run-001",
            attempt_id="attempt-0",
        )


@pytest.mark.parametrize(
    ("tree_name", "omitted_branch", "message"),
    [
        ("events", None, "does not contain the 'mini' tree"),
        ("mini", "photon_pt", "missing required branches: photon_pt"),
    ],
)
def test_adapter_rejects_an_incompatible_root_schema(
    tmp_path: Path,
    tree_name: str,
    omitted_branch: str | None,
    message: str,
) -> None:
    root_path = tmp_path / "test.root"
    write_root_file(root_path, tree_name=tree_name, omit_branch=omitted_branch)

    with pytest.raises(ValueError, match=message):
        AtlasGamGamOpenDataAdapter(synthetic_source(root_path)).load_context(
            workflow_id="atlas-test",
            run_id="run-001",
            attempt_id="attempt-0",
        )


def test_official_source_descriptor_is_fixed_to_the_cern_record() -> None:
    source = AtlasGamGamSource.official_wph125(Path("sample.root"))

    assert source.size_bytes == 29_757_932
    assert source.adler32 == "5ac6bca3"
    assert source.source_type == "simulation"
    assert source.record_id == "atlas-15006"
    assert source.doi == "10.7483/OPENDATA.ATLAS.B5BJ.3SGS"


def test_adapter_explains_the_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_path = tmp_path / "test.root"
    write_root_file(root_path)
    modules = cast(dict[str, object], sys.modules)
    monkeypatch.setitem(modules, "awkward", None)

    with pytest.raises(ModuleNotFoundError, match="requires the 'hep' extra"):
        AtlasGamGamOpenDataAdapter(synthetic_source(root_path)).load_context(
            workflow_id="atlas-test",
            run_id="run-001",
            attempt_id="attempt-0",
        )


def test_adapter_checkpoint_runs_the_stable_contract_map_without_leaking_paths(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "LOCAL_SECRET_SENTINEL" / "test.root"
    root_path.parent.mkdir()
    write_root_file(root_path)
    checkpoint = AtlasGamGamOpenDataAdapter(synthetic_source(root_path)).checkpoint(
        workflow_id="atlas-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )

    execution = GuardedWorkflowRunner().execute((checkpoint,))

    assert execution.output is not None
    assert not execution.trace.blocked
    assert [result.contract_id for result in execution.trace.checkpoints[0].results] == [
        "hep.schema.required_branches",
        "hep.weights.finite",
        "hep.weights.nonzero_support",
        "hep.atlas_open_data.diphoton_preselection",
        "hep.provenance.events_declared",
        "hep.atlas_open_data.source_identity",
    ]
    payload = execution.trace.model_dump_json()
    assert "LOCAL_SECRET_SENTINEL" not in payload
    assert str(tmp_path) not in payload
    assert WorkflowTrace.model_validate_json(payload) == execution.trace
