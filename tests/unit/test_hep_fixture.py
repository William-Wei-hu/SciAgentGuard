from collections.abc import Mapping
from typing import cast

from sciagentguard.packs.hep import make_synthetic_hep_context


def test_synthetic_fixture_is_deterministic_and_labeled() -> None:
    first = make_synthetic_hep_context()
    second = make_synthetic_hep_context()

    assert first == second
    assert first.stage == "post_load"
    assert first.provenance["events"] == {
        "source_type": "synthetic",
        "generator": "sciagentguard.packs.hep.fixtures",
    }
    assert first.units["jet_pt_gev"] == "GeV"
    assert first.config["jet_pt_gev"] == {
        "expected_unit": "GeV",
        "valid_range": [0.0, 500.0],
    }


def test_synthetic_fixture_uses_equal_length_immutable_columns() -> None:
    context = make_synthetic_hep_context()
    events = cast(Mapping[str, tuple[object, ...]], context.artifacts["events"])

    assert set(events) == {"event_id", "jet_pt_gev", "weight"}
    assert {len(column) for column in events.values()} == {6}
    assert all(isinstance(column, tuple) for column in events.values())
