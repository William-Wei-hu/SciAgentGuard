from collections.abc import Mapping
from dataclasses import replace
from math import isnan
from typing import cast

import pytest

from sciagentguard.core import SemanticFaultInjector
from sciagentguard.packs.hep import (
    MissingBranchInjector,
    NonfiniteWeightsInjector,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    ZeroWeightsInjector,
    make_synthetic_hep_context,
)


def test_missing_branch_injector_returns_a_new_context() -> None:
    original = make_synthetic_hep_context()
    injected = MissingBranchInjector().inject(original)
    original_events = cast(Mapping[str, object], original.artifacts["events"])
    injected_events = cast(Mapping[str, object], injected.artifacts["events"])

    assert injected is not original
    assert "jet_pt_gev" in original_events
    assert "jet_pt_gev" not in injected_events
    assert injected_events["event_id"] == original_events["event_id"]


def test_missing_branch_injector_is_deterministic_and_checks_its_precondition() -> None:
    original = make_synthetic_hep_context()
    first = MissingBranchInjector().inject(original, seed=1)
    second = MissingBranchInjector().inject(original, seed=999)

    assert first == second
    with pytest.raises(ValueError, match="requires"):
        MissingBranchInjector().inject(first)


@pytest.mark.parametrize("injector", [ZeroWeightsInjector(), NonfiniteWeightsInjector()])
def test_weight_injectors_copy_the_event_artifact(
    injector: SemanticFaultInjector,
) -> None:
    original = make_synthetic_hep_context()
    original_events = cast(Mapping[str, tuple[object, ...]], original.artifacts["events"])

    injected = injector.inject(original)
    injected_events = cast(Mapping[str, tuple[object, ...]], injected.artifacts["events"])

    assert injected is not original
    assert original_events["weight"] == (1.0, 0.8, -0.2, 1.1, 0.5, 0.9)
    assert injected_events["weight"] != original_events["weight"]
    assert injected_events["event_id"] == original_events["event_id"]


def test_zero_weights_injector_is_deterministic() -> None:
    original = make_synthetic_hep_context()
    first = ZeroWeightsInjector().inject(original, seed=1)
    second = ZeroWeightsInjector().inject(original, seed=999)

    assert first == second


def test_nonfinite_weights_injector_uses_a_stable_index() -> None:
    original = make_synthetic_hep_context()
    first = NonfiniteWeightsInjector().inject(original, seed=1)
    second = NonfiniteWeightsInjector().inject(original, seed=999)
    first_events = cast(Mapping[str, tuple[object, ...]], first.artifacts["events"])
    second_events = cast(Mapping[str, tuple[object, ...]], second.artifacts["events"])

    assert isnan(cast(float, first_events["weight"][2]))
    assert isnan(cast(float, second_events["weight"][2]))
    assert first_events["weight"][:2] == second_events["weight"][:2]
    assert first_events["weight"][3:] == second_events["weight"][3:]


@pytest.mark.parametrize("injector", [ZeroWeightsInjector(), NonfiniteWeightsInjector()])
def test_weight_injectors_reject_a_missing_weight_branch(
    injector: SemanticFaultInjector,
) -> None:
    context = make_synthetic_hep_context()
    events = dict(cast(Mapping[str, object], context.artifacts["events"]))
    del events["weight"]
    context = replace(context, artifacts={"events": events})

    with pytest.raises(ValueError, match="requires"):
        injector.inject(context)


def test_unit_scale_injector_is_deterministic_and_preserves_the_source() -> None:
    original = make_synthetic_hep_context()
    original_events = cast(Mapping[str, tuple[object, ...]], original.artifacts["events"])
    first = UnitScaleErrorInjector().inject(original, seed=1)
    second = UnitScaleErrorInjector().inject(original, seed=999)
    first_events = cast(Mapping[str, tuple[object, ...]], first.artifacts["events"])

    assert first == second
    assert first is not original
    assert original_events["jet_pt_gev"] == (42.0, 78.5, 33.1, 120.4, 55.0, 91.2)
    assert first_events["jet_pt_gev"] == (
        42000.0,
        78500.0,
        33100.0,
        120400.0,
        55000.0,
        91200.0,
    )
    assert first_events["weight"] == original_events["weight"]
    assert first.units == original.units
    assert first.config == original.config


def test_unit_scale_injector_rejects_a_missing_or_non_numeric_branch() -> None:
    context = MissingBranchInjector().inject(make_synthetic_hep_context())
    with pytest.raises(ValueError, match="requires"):
        UnitScaleErrorInjector().inject(context)

    context = make_synthetic_hep_context()
    events = dict(cast(Mapping[str, object], context.artifacts["events"]))
    events["jet_pt_gev"] = ("invalid",) * 6
    context = replace(context, artifacts={"events": events})
    with pytest.raises(ValueError, match="non-numeric"):
        UnitScaleErrorInjector().inject(context)


def test_undeclared_data_injector_replaces_events_without_mutating_the_source() -> None:
    original = make_synthetic_hep_context()
    original_events = cast(Mapping[str, tuple[object, ...]], original.artifacts["events"])
    first = UndeclaredSyntheticDataInjector().inject(original, seed=1)
    second = UndeclaredSyntheticDataInjector().inject(original, seed=999)
    injected_events = cast(Mapping[str, tuple[object, ...]], first.artifacts["events"])

    assert first == second
    assert first is not original
    assert original_events["event_id"] == (1001, 1002, 1003, 1004, 1005, 1006)
    assert original.provenance["events"] == {
        "source_type": "synthetic",
        "generator": "sciagentguard.packs.hep.fixtures",
    }
    assert injected_events == {
        "event_id": (9001, 9002),
        "jet_pt_gev": (40.0, 80.0),
        "weight": (1.0, -0.5),
    }
    assert "events" not in first.provenance
    assert first.config == original.config
    assert first.units == original.units


def test_undeclared_data_injector_checks_its_preconditions() -> None:
    context = make_synthetic_hep_context()
    context = replace(context, provenance={})
    with pytest.raises(ValueError, match="requires events provenance"):
        UndeclaredSyntheticDataInjector().inject(context)

    context = replace(make_synthetic_hep_context(), artifacts={"events": "invalid"})
    with pytest.raises(ValueError, match="must be a mapping"):
        UndeclaredSyntheticDataInjector().inject(context)
