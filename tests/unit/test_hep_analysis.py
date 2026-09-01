import json
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest
from pydantic import JsonValue

from sciagentguard.core import ContractStatus
from sciagentguard.packs.hep import (
    DisjointEventSplitsContract,
    EmptySelectionInjector,
    NonemptySelectionContract,
    SplitLeakageInjector,
    WrongNormalizationInjector,
    YieldNormalizationContract,
    make_synthetic_normalization_context,
    make_synthetic_selection_context,
    make_synthetic_split_context,
)
from sciagentguard.runtime import GuardedExecutor


def test_selection_fixture_is_deterministic_and_declared_synthetic() -> None:
    first = make_synthetic_selection_context()
    second = make_synthetic_selection_context()

    assert first == second
    assert first.stage == "post_selection"
    assert first.provenance["selection"] == {
        "source_type": "synthetic",
        "generator": "sciagentguard.packs.hep.fixtures",
    }
    assert first.config["selection"] == {
        "selection_id": "jet_pt_gev_gt_50",
        "minimum_selected": 1,
    }


def test_nonempty_selection_contract_passes_the_declared_fixture() -> None:
    result = NonemptySelectionContract().evaluate(make_synthetic_selection_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence == {
        "selection_id": "jet_pt_gev_gt_50",
        "selection_stage": "post_selection",
        "input_count": 6,
        "selected_count": 4,
        "minimum_selected": 1,
    }


def test_empty_selection_is_localized_and_json_safe() -> None:
    context = EmptySelectionInjector().inject(make_synthetic_selection_context())
    result = NonemptySelectionContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence["input_count"] == 6
    assert result.evidence["selected_count"] == 0
    assert result.violation is not None
    assert result.violation.stage == "post_selection"
    assert result.violation.severity.value == "error"
    assert result.violation.affected_artifacts == ("selection",)
    assert result.violation.suggested_actions
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert json.loads(result.model_dump_json())["evidence"] == result.evidence


def test_empty_selection_injector_is_deterministic_and_non_mutating() -> None:
    original = make_synthetic_selection_context()
    original_selection = cast(Mapping[str, object], original.artifacts["selection"])
    first = EmptySelectionInjector().inject(original, seed=1)
    second = EmptySelectionInjector().inject(original, seed=999)
    injected_selection = cast(Mapping[str, object], first.artifacts["selection"])

    assert first == second
    assert first is not original
    assert original_selection["selected_event_ids"] == (1002, 1004, 1005, 1006)
    assert injected_selection["selected_event_ids"] == ()
    assert injected_selection["input_event_ids"] == original_selection["input_event_ids"]
    assert first.config == original.config
    assert first.provenance == original.provenance


def test_empty_selection_injector_checks_its_precondition() -> None:
    context = EmptySelectionInjector().inject(make_synthetic_selection_context())

    with pytest.raises(ValueError, match="requires at least one selected event"):
        EmptySelectionInjector().inject(context)


@pytest.mark.parametrize(
    "selection",
    [
        {"input_event_ids": (1001, 1001), "selected_event_ids": (1001,)},
        {"input_event_ids": (1001, 1002), "selected_event_ids": (1003,)},
        {"input_event_ids": (1001, 1002), "selected_event_ids": (1001, 1001)},
    ],
)
def test_selection_contract_rejects_invalid_event_identifiers(
    selection: dict[str, object],
) -> None:
    context = replace(make_synthetic_selection_context(), artifacts={"selection": selection})

    with pytest.raises(ValueError):
        NonemptySelectionContract().evaluate(context)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"selection": {"selection_id": "", "minimum_selected": 1}},
        {"selection": {"selection_id": "test", "minimum_selected": 0}},
    ],
)
def test_selection_contract_rejects_invalid_configuration(
    config: dict[str, JsonValue],
) -> None:
    context = replace(make_synthetic_selection_context(), config=config)

    with pytest.raises(ValueError, match="config 'selection"):
        NonemptySelectionContract().evaluate(context)


def test_selection_trace_does_not_copy_runtime_or_metadata_secrets() -> None:
    secret = "SELECTION_SECRET_SENTINEL"
    context = make_synthetic_selection_context()
    selection = dict(cast(Mapping[str, object], context.artifacts["selection"]))
    selection["private_value"] = secret
    context = replace(
        context,
        artifacts={"selection": selection},
        config={
            "selection": {
                "selection_id": "jet_pt_gev_gt_50",
                "minimum_selected": 1,
                "private_value": secret,
            }
        },
        provenance={"selection": {"source_type": "synthetic", "private_value": secret}},
    )

    trace_json = (
        GuardedExecutor()
        .execute(lambda: context, [NonemptySelectionContract()])
        .trace.model_dump_json()
    )

    assert secret not in trace_json


def test_split_fixture_is_deterministic_and_declared_synthetic() -> None:
    first = make_synthetic_split_context()
    second = make_synthetic_split_context()

    assert first == second
    assert first.stage == "post_split"
    assert first.provenance["splits"] == {
        "source_type": "synthetic",
        "generator": "sciagentguard.packs.hep.fixtures",
    }
    assert first.config["split"] == {"left": "train", "right": "test"}


def test_disjoint_event_splits_contract_passes_the_declared_fixture() -> None:
    result = DisjointEventSplitsContract().evaluate(make_synthetic_split_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence == {
        "left_split": "train",
        "right_split": "test",
        "left_count": 4,
        "right_count": 2,
        "overlap_count": 0,
        "overlap_sample_ids": [],
    }


def test_split_leakage_is_localized_and_json_safe() -> None:
    context = SplitLeakageInjector().inject(make_synthetic_split_context())
    result = DisjointEventSplitsContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence == {
        "left_split": "train",
        "right_split": "test",
        "left_count": 4,
        "right_count": 3,
        "overlap_count": 1,
        "overlap_sample_ids": [1004],
    }
    assert result.violation is not None
    assert result.violation.stage == "post_split"
    assert result.violation.severity.value == "error"
    assert result.violation.affected_artifacts == ("splits",)
    assert result.violation.suggested_actions
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_split_leakage_injector_is_deterministic_and_non_mutating() -> None:
    original = make_synthetic_split_context()
    original_splits = cast(Mapping[str, object], original.artifacts["splits"])
    first = SplitLeakageInjector().inject(original, seed=1)
    second = SplitLeakageInjector().inject(original, seed=999)
    injected_splits = cast(Mapping[str, object], first.artifacts["splits"])

    assert first == second
    assert first is not original
    assert original_splits["train"] == (1001, 1002, 1003, 1004)
    assert original_splits["test"] == (1005, 1006)
    assert injected_splits["train"] == original_splits["train"]
    assert injected_splits["test"] == (1004, 1005, 1006)
    assert first.config == original.config
    assert first.provenance == original.provenance


def test_split_leakage_injector_checks_its_preconditions() -> None:
    context = SplitLeakageInjector().inject(make_synthetic_split_context())
    with pytest.raises(ValueError, match="initially disjoint"):
        SplitLeakageInjector().inject(context)

    context = replace(
        make_synthetic_split_context(),
        artifacts={"splits": {"train": (1001,), "test": (1005, 1006)}},
    )
    with pytest.raises(ValueError, match="requires train event identifier 1004"):
        SplitLeakageInjector().inject(context)


@pytest.mark.parametrize(
    "splits",
    [
        {"train": (1001, 1001), "test": (1002,)},
        {"train": (1001,), "test": (True,)},
        {"train": (1001,), "test": "invalid"},
    ],
)
def test_split_contract_rejects_invalid_event_identifiers(
    splits: dict[str, object],
) -> None:
    context = replace(make_synthetic_split_context(), artifacts={"splits": splits})

    with pytest.raises(ValueError):
        DisjointEventSplitsContract().evaluate(context)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"split": {"left": "", "right": "test"}},
        {"split": {"left": "train", "right": "train"}},
        {"split": {"left": "train", "right": "missing"}},
    ],
)
def test_split_contract_rejects_invalid_configuration(
    config: dict[str, JsonValue],
) -> None:
    context = replace(make_synthetic_split_context(), config=config)

    with pytest.raises(ValueError):
        DisjointEventSplitsContract().evaluate(context)


def test_split_overlap_evidence_is_bounded_to_ten_identifiers() -> None:
    context = replace(
        make_synthetic_split_context(),
        artifacts={"splits": {"train": tuple(range(20)), "test": tuple(range(20))}},
    )

    result = DisjointEventSplitsContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence["overlap_count"] == 20
    assert result.evidence["overlap_sample_ids"] == list(range(10))


def test_normalization_fixture_is_deterministic_and_declared_synthetic() -> None:
    first = make_synthetic_normalization_context()
    second = make_synthetic_normalization_context()

    assert first == second
    assert first.stage == "post_normalization"
    assert first.provenance["normalization"] == {
        "source_type": "synthetic",
        "generator": "sciagentguard.packs.hep.fixtures",
    }
    assert first.units == {
        "cross_section_pb": "pb",
        "luminosity_pb_inverse": "pb^-1",
        "observed_yield": "events",
    }
    assert first.config["normalization"] == {
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-9,
    }


def test_yield_normalization_contract_passes_the_declared_fixture() -> None:
    result = YieldNormalizationContract().evaluate(make_synthetic_normalization_context())

    assert result.status is ContractStatus.PASS
    assert result.evidence == {
        "formula": (
            "selected_weight_sum * cross_section_pb * luminosity_pb_inverse / generated_weight_sum"
        ),
        "assumptions": [
            "generated_weight_sum is finite and nonzero",
            "cross_section_pb and luminosity_pb_inverse use reciprocal pb units",
        ],
        "selected_weight_sum": 4.1,
        "generated_weight_sum": 8.2,
        "cross_section_pb": 2.0,
        "luminosity_pb_inverse": 100.0,
        "expected_yield": 100.0,
        "observed_yield": 100.0,
        "absolute_difference": 0.0,
        "relative_difference": 0.0,
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-9,
    }
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_wrong_normalization_is_localized_and_json_safe() -> None:
    context = WrongNormalizationInjector().inject(make_synthetic_normalization_context())
    result = YieldNormalizationContract().evaluate(context)

    assert result.status is ContractStatus.FAIL
    assert result.evidence["expected_yield"] == 100.0
    assert result.evidence["observed_yield"] == 1000.0
    assert result.evidence["absolute_difference"] == 900.0
    assert result.evidence["relative_difference"] == 9.0
    assert result.violation is not None
    assert result.violation.stage == "post_normalization"
    assert result.violation.severity.value == "error"
    assert result.violation.affected_artifacts == ("normalization",)
    assert result.violation.suggested_actions
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_wrong_normalization_injector_is_deterministic_and_non_mutating() -> None:
    original = make_synthetic_normalization_context()
    original_values = cast(Mapping[str, object], original.artifacts["normalization"])
    first = WrongNormalizationInjector().inject(original, seed=1)
    second = WrongNormalizationInjector().inject(original, seed=999)
    injected_values = cast(Mapping[str, object], first.artifacts["normalization"])

    assert first == second
    assert first is not original
    assert original_values["observed_yield"] == 100.0
    assert injected_values["observed_yield"] == 1000.0
    assert {key: value for key, value in injected_values.items() if key != "observed_yield"} == {
        key: value for key, value in original_values.items() if key != "observed_yield"
    }
    assert first.config == original.config
    assert first.provenance == original.provenance


def test_wrong_normalization_injector_checks_its_precondition() -> None:
    context = WrongNormalizationInjector().inject(make_synthetic_normalization_context())

    with pytest.raises(ValueError, match=r"declared fixture yield of 100\.0"):
        WrongNormalizationInjector().inject(context)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"normalization": {"absolute_tolerance": -1.0, "relative_tolerance": 1e-9}},
        {
            "normalization": {
                "absolute_tolerance": 1e-12,
                "relative_tolerance": float("nan"),
            }
        },
    ],
)
def test_normalization_contract_rejects_invalid_configuration(
    config: dict[str, JsonValue],
) -> None:
    context = replace(make_synthetic_normalization_context(), config=config)

    with pytest.raises(ValueError, match="config 'normalization"):
        YieldNormalizationContract().evaluate(context)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_weight_sum", "4.1", "must be numeric"),
        ("observed_yield", float("inf"), "must be finite"),
        ("generated_weight_sum", 0.0, "must be nonzero"),
        ("cross_section_pb", -1.0, "must be nonnegative"),
        ("luminosity_pb_inverse", -1.0, "must be nonnegative"),
    ],
)
def test_normalization_contract_rejects_invalid_formula_inputs(
    field: str, value: object, message: str
) -> None:
    context = make_synthetic_normalization_context()
    values = dict(cast(Mapping[str, object], context.artifacts["normalization"]))
    values[field] = value
    context = replace(context, artifacts={"normalization": values})

    with pytest.raises(ValueError, match=message):
        YieldNormalizationContract().evaluate(context)


@pytest.mark.parametrize(
    ("observed_yield", "expected_status"),
    [
        (100.00000005, ContractStatus.PASS),
        (100.000001, ContractStatus.FAIL),
    ],
)
def test_yield_normalization_uses_the_declared_tolerances(
    observed_yield: float, expected_status: ContractStatus
) -> None:
    context = make_synthetic_normalization_context()
    values = dict(cast(Mapping[str, object], context.artifacts["normalization"]))
    values["observed_yield"] = observed_yield
    context = replace(context, artifacts={"normalization": values})

    result = YieldNormalizationContract().evaluate(context)

    assert result.status is expected_status
