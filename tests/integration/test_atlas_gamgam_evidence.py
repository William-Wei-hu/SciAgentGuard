import json
from pathlib import Path
from typing import Any, cast

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.runtime import WorkflowTrace

EVIDENCE = Path(__file__).parents[2] / "benchmarks" / "results" / "atlas_gamgam_smoke.json"


def test_committed_atlas_smoke_trace_records_the_verified_source_boundary() -> None:
    payload = EVIDENCE.read_text(encoding="utf-8")
    trace = WorkflowTrace.model_validate_json(payload)
    document = cast(dict[str, Any], json.loads(payload))
    results = {
        result["contract_id"]: result
        for checkpoint in document["checkpoints"]
        for result in checkpoint["results"]
    }

    assert not trace.blocked
    assert [checkpoint.stage for checkpoint in trace.checkpoints] == [
        "post_load",
        "post_selection",
        "post_histogram",
        "post_yield",
    ]
    assert all(
        result.status.value == "pass"
        for checkpoint in trace.checkpoints
        for result in checkpoint.results
    )
    # Derived rather than hard-coded: the committed trace must record exactly the contracts
    # the adapter declares, so adding one cannot leave a stale count passing.
    declared = {
        contract.contract_id
        for contracts in AtlasGamGamOpenDataAdapter(
            AtlasGamGamSource.official_wph125(Path("unused.root"))
        ).stage_contracts()
        for contract in contracts
    }
    assert set(results) == declared
    assert results["hep.weights.finite"]["evidence"]["weight_count"] == 113_765
    assert results["hep.weights.finite"]["evidence"]["nonfinite_count"] == 0
    diphoton = results["hep.atlas_open_data.diphoton_preselection"]["evidence"]
    assert diphoton["event_count"] == 113_765
    assert diphoton["observed_photon_count_range"] == [2, 4]
    identity = results["hep.atlas_open_data.source_identity"]["evidence"]
    assert identity["expected_record_id"] == "atlas-15006"
    assert identity["expected_checksum"] == "adler32:5ac6bca3"
    assert identity["mismatched_fields"] == []

    cutflow = results["hep.atlas_open_data.cutflow_monotonic"]["evidence"]
    assert cutflow["input_count"] == 113_765
    assert cutflow["selected_count"] == 105_503
    assert cutflow["surviving_counts"] == [113_765, 113_765, 113_210, 105_503]
    # The reconstructed diphoton spectrum peaks in the bin centred on the Higgs mass.
    assert results["hep.atlas_open_data.yield_shape"]["evidence"]["peak_bin_center_gev"] == 125.0
    assert (
        results["hep.atlas_open_data.region_disjoint"]["evidence"]["overlapping_region_pairs"] == []
    )

    for private_fragment in ("/Users/", ".cache/", "LOCAL_PATH_SECRET", "root://"):
        assert private_fragment not in payload
