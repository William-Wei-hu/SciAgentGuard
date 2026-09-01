"""Scientific contracts for the synthetic HEP event artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isclose, isfinite
from time import perf_counter_ns

from pydantic import JsonValue

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
)
from sciagentguard.packs.hep._analysis import (
    require_normalization,
    require_selection,
    require_splits,
    split_event_ids,
)
from sciagentguard.packs.hep._contract_results import elapsed_ms, failed_result
from sciagentguard.packs.hep._events import numeric_event_column, require_event_columns
from sciagentguard.packs.hep.fixtures import (
    HEP_STAGE,
    NORMALIZATION_STAGE,
    SELECTION_STAGE,
    SPLIT_STAGE,
)

REQUIRED_BRANCHES_CONTRACT_ID = "hep.schema.required_branches"
FINITE_WEIGHTS_CONTRACT_ID = "hep.weights.finite"
NONZERO_WEIGHT_SUPPORT_CONTRACT_ID = "hep.weights.nonzero_support"
JET_PT_RANGE_CONTRACT_ID = "hep.kinematics.jet_pt_range"
EVENT_PROVENANCE_CONTRACT_ID = "hep.provenance.events_declared"
NONEMPTY_SELECTION_CONTRACT_ID = "hep.selection.nonempty"
DISJOINT_EVENT_SPLITS_CONTRACT_ID = "hep.splits.disjoint_event_ids"
YIELD_NORMALIZATION_CONTRACT_ID = "hep.normalization.yield_consistent"
NORMALIZATION_FORMULA = (
    "selected_weight_sum * cross_section_pb * luminosity_pb_inverse / generated_weight_sum"
)


def _missing_weight_result(contract_id: str, start_ns: int) -> ContractResult:
    return ContractResult(
        contract_id=contract_id,
        status=ContractStatus.NOT_APPLICABLE,
        evidence={"reason": "the event artifact does not contain a weight branch"},
        duration_ms=elapsed_ms(start_ns),
    )


@dataclass(frozen=True, slots=True)
class RequiredBranchesContract:
    """Check that the event artifact exposes the configured analysis branches."""

    required_branches: tuple[str, ...] = ("event_id", "jet_pt_gev", "weight")
    contract_id: str = field(init=False, default=REQUIRED_BRANCHES_CONTRACT_ID)
    description: str = field(
        init=False, default="Required event branches are available after loading."
    )
    stage: str = field(init=False, default=HEP_STAGE)
    required_inputs: tuple[str, ...] = field(init=False, default=("events",))

    def __post_init__(self) -> None:
        if not self.required_branches:
            raise ValueError("required_branches must not be empty")
        if any(not branch or branch != branch.strip() for branch in self.required_branches):
            raise ValueError(
                "required branch names must be non-empty and have no surrounding whitespace"
            )
        if len(set(self.required_branches)) != len(self.required_branches):
            raise ValueError("required_branches must not contain duplicates")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        columns = require_event_columns(context)
        available = tuple(sorted(columns))
        missing = tuple(branch for branch in self.required_branches if branch not in columns)
        evidence: dict[str, JsonValue] = {
            "required_branches": list(self.required_branches),
            "available_branches": list(available),
            "missing_branches": list(missing),
        }
        if not missing:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The loaded event artifact is missing required analysis branches.",
            likely_causes=(
                "An input branch was renamed or removed before the post-load checkpoint.",
            ),
            suggested_actions=(
                "Compare the requested branch names with the available branch list.",
            ),
            start_ns=start_ns,
        )


class FiniteWeightsContract:
    """Require every event weight to be finite."""

    contract_id = FINITE_WEIGHTS_CONTRACT_ID
    description = "Every event weight is a finite numeric value after loading."
    stage = HEP_STAGE
    required_inputs = ("events",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        weights = numeric_event_column(context, "weight")
        if weights is None:
            return _missing_weight_result(self.contract_id, start_ns)

        invalid_indices = tuple(index for index, value in enumerate(weights) if not isfinite(value))
        evidence: dict[str, JsonValue] = {
            "weight_count": len(weights),
            "nonfinite_count": len(invalid_indices),
            "sample_indices": list(invalid_indices[:10]),
        }
        if not invalid_indices:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The event weights contain non-finite values.",
            likely_causes=("A weight calculation produced NaN or infinity before the checkpoint.",),
            suggested_actions=(
                "Inspect the reported event indices and the upstream weight calculation.",
            ),
            start_ns=start_ns,
        )


class NonzeroWeightSupportContract:
    """Require nonzero absolute weighted support for the loaded events."""

    contract_id = NONZERO_WEIGHT_SUPPORT_CONTRACT_ID
    description = "The event sample has nonzero absolute weighted support."
    stage = HEP_STAGE
    required_inputs = ("events",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        weights = numeric_event_column(context, "weight")
        if weights is None:
            return _missing_weight_result(self.contract_id, start_ns)

        invalid_indices = tuple(index for index, value in enumerate(weights) if not isfinite(value))
        if invalid_indices:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.NOT_APPLICABLE,
                evidence={
                    "reason": "finite weights are required before weighted support is evaluated",
                    "weight_count": len(weights),
                    "nonfinite_count": len(invalid_indices),
                    "sample_indices": list(invalid_indices[:10]),
                },
                duration_ms=elapsed_ms(start_ns),
            )

        absolute_weight_sum = sum(abs(value) for value in weights)
        if not isfinite(absolute_weight_sum):
            raise ValueError("the absolute event-weight sum overflowed")
        evidence: dict[str, JsonValue] = {
            "weight_count": len(weights),
            "absolute_weight_sum": absolute_weight_sum,
        }
        if absolute_weight_sum > 0.0:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The event sample has zero absolute weighted support.",
            likely_causes=(
                "All event weights were replaced with zero or a weighting step lost its scale.",
            ),
            suggested_actions=(
                "Inspect the upstream weight source before using this event sample.",
            ),
            start_ns=start_ns,
        )


def _jet_pt_assumptions(context: ContractContext) -> tuple[str, float, float]:
    assumptions = context.config.get("jet_pt_gev")
    if not isinstance(assumptions, Mapping):
        raise ValueError("config 'jet_pt_gev' must contain the fixture range assumptions")

    expected_unit = assumptions.get("expected_unit")
    if not isinstance(expected_unit, str) or not expected_unit.strip():
        raise ValueError("config 'jet_pt_gev.expected_unit' must be a non-empty string")

    valid_range = assumptions.get("valid_range")
    if (
        not isinstance(valid_range, Sequence)
        or isinstance(valid_range, (str, bytes, bytearray))
        or len(valid_range) != 2
    ):
        raise ValueError("config 'jet_pt_gev.valid_range' must contain two numeric bounds")
    lower_bound, upper_bound = valid_range
    if (
        isinstance(lower_bound, bool)
        or not isinstance(lower_bound, (int, float))
        or isinstance(upper_bound, bool)
        or not isinstance(upper_bound, (int, float))
    ):
        raise ValueError("config 'jet_pt_gev.valid_range' must contain two numeric bounds")

    lower, upper = float(lower_bound), float(upper_bound)
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError("config 'jet_pt_gev.valid_range' must contain ordered finite bounds")
    return expected_unit.strip(), lower, upper


class JetPtRangeContract:
    """Validate jet transverse momenta against this fixture's declared assumptions."""

    contract_id = JET_PT_RANGE_CONTRACT_ID
    description = "Jet transverse momenta match the declared unit and fixture-local range."
    stage = HEP_STAGE
    required_inputs = ("events",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        values = numeric_event_column(context, "jet_pt_gev")
        if values is None:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.NOT_APPLICABLE,
                evidence={"reason": "the event artifact does not contain jet_pt_gev"},
                duration_ms=elapsed_ms(start_ns),
            )

        expected_unit, lower, upper = _jet_pt_assumptions(context)
        actual_unit = context.units.get("jet_pt_gev")
        nonfinite_indices = tuple(
            index for index, value in enumerate(values) if not isfinite(value)
        )
        finite_values = tuple(value for value in values if isfinite(value))
        out_of_range_indices = tuple(
            index
            for index, value in enumerate(values)
            if isfinite(value) and not lower <= value <= upper
        )
        observed_range: JsonValue = None
        if finite_values:
            observed_range = [min(finite_values), max(finite_values)]

        evidence: dict[str, JsonValue] = {
            "expected_unit": expected_unit,
            "actual_unit": actual_unit,
            "valid_range": [lower, upper],
            "value_count": len(values),
            "out_of_range_count": len(out_of_range_indices),
            "out_of_range_sample_indices": list(out_of_range_indices[:10]),
            "nonfinite_count": len(nonfinite_indices),
            "nonfinite_sample_indices": list(nonfinite_indices[:10]),
            "finite_observed_range": observed_range,
        }
        if actual_unit == expected_unit and not out_of_range_indices and not nonfinite_indices:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="Jet transverse momenta do not match the declared fixture assumptions.",
            likely_causes=(
                "The momentum scale, unit declaration, or upstream calculation changed.",
            ),
            suggested_actions=(
                "Verify the reported indices and unit against the declared synthetic source.",
            ),
            start_ns=start_ns,
        )


class DeclaredEventProvenanceContract:
    """Require an explicit source declaration for the event artifact."""

    contract_id = EVENT_PROVENANCE_CONTRACT_ID
    description = "The event source type and required synthetic generator are declared."
    stage = HEP_STAGE
    required_inputs = ("events", "provenance")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        declaration = context.provenance.get("events")
        if declaration is not None and not isinstance(declaration, Mapping):
            raise ValueError("provenance 'events' must be a mapping")

        source_type: object = None
        generator: object = None
        if isinstance(declaration, Mapping):
            source_type = declaration.get("source_type")
            generator = declaration.get("generator")

        source_type_declared = isinstance(source_type, str) and bool(source_type.strip())
        synthetic_source_declared = (
            source_type_declared
            and isinstance(source_type, str)
            and source_type.strip() == "synthetic"
        )
        generator_required = synthetic_source_declared
        generator_declared = isinstance(generator, str) and bool(generator.strip())
        evidence: dict[str, JsonValue] = {
            "declaration_present": declaration is not None,
            "source_type_declared": source_type_declared,
            "synthetic_source_declared": synthetic_source_declared,
            "generator_required": generator_required,
            "generator_declared": generator_declared,
        }
        if source_type_declared and (not generator_required or generator_declared):
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The event artifact does not have a complete source declaration.",
            likely_causes=(
                "Source metadata was omitted or a synthetic generator was not declared.",
            ),
            suggested_actions=(
                "Declare the event source type and, for synthetic data, its generator.",
            ),
            start_ns=start_ns,
        )


def _selection_config(context: ContractContext) -> tuple[str, int]:
    config = context.config.get("selection")
    if not isinstance(config, Mapping):
        raise ValueError("config 'selection' must be a mapping")

    selection_id = config.get("selection_id")
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise ValueError("config 'selection.selection_id' must be a non-empty string")
    minimum_selected = config.get("minimum_selected")
    if (
        isinstance(minimum_selected, bool)
        or not isinstance(minimum_selected, int)
        or minimum_selected < 1
    ):
        raise ValueError("config 'selection.minimum_selected' must be a positive integer")
    return selection_id.strip(), minimum_selected


class NonemptySelectionContract:
    """Require the declared analysis selection to retain at least one event."""

    contract_id = NONEMPTY_SELECTION_CONTRACT_ID
    description = "The configured analysis selection retains the required event support."
    stage = SELECTION_STAGE
    required_inputs = ("selection", "config")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        _, input_ids, selected_ids = require_selection(context)
        selection_id, minimum_selected = _selection_config(context)
        evidence: dict[str, JsonValue] = {
            "selection_id": selection_id,
            "selection_stage": context.stage,
            "input_count": len(input_ids),
            "selected_count": len(selected_ids),
            "minimum_selected": minimum_selected,
        }
        if len(selected_ids) >= minimum_selected:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The configured event selection retained no usable events.",
            likely_causes=(
                "A selection threshold or upstream event field changed before this checkpoint.",
            ),
            suggested_actions=(
                "Inspect the declared selection and compare its input and selected counts.",
            ),
            start_ns=start_ns,
            affected_artifacts=("selection",),
        )


def _split_config(context: ContractContext) -> tuple[str, str]:
    config = context.config.get("split")
    if not isinstance(config, Mapping):
        raise ValueError("config 'split' must be a mapping")

    left = config.get("left")
    right = config.get("right")
    if not isinstance(left, str) or not left.strip():
        raise ValueError("config 'split.left' must be a non-empty string")
    if not isinstance(right, str) or not right.strip():
        raise ValueError("config 'split.right' must be a non-empty string")
    left, right = left.strip(), right.strip()
    if left == right:
        raise ValueError("config 'split' must name two different splits")
    return left, right


class DisjointEventSplitsContract:
    """Require configured analysis splits to contain disjoint event identifiers."""

    contract_id = DISJOINT_EVENT_SPLITS_CONTRACT_ID
    description = "Configured train and test splits do not share event identifiers."
    stage = SPLIT_STAGE
    required_inputs = ("splits", "config")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        left_name, right_name = _split_config(context)
        splits = require_splits(context)
        left_ids = split_event_ids(splits, left_name)
        right_ids = split_event_ids(splits, right_name)
        overlap = tuple(
            sorted(
                set(left_ids).intersection(right_ids),
                key=lambda identifier: (
                    (0, identifier) if isinstance(identifier, int) else (1, identifier)
                ),
            )
        )
        evidence: dict[str, JsonValue] = {
            "left_split": left_name,
            "right_split": right_name,
            "left_count": len(left_ids),
            "right_count": len(right_ids),
            "overlap_count": len(overlap),
            "overlap_sample_ids": list(overlap[:10]),
        }
        if not overlap:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The configured analysis splits contain overlapping event identifiers.",
            likely_causes=("An event was assigned to more than one split before this checkpoint.",),
            suggested_actions=(
                "Inspect the reported identifiers and the upstream split assignment.",
            ),
            start_ns=start_ns,
            affected_artifacts=("splits",),
        )


def _normalization_tolerances(context: ContractContext) -> tuple[float, float]:
    config = context.config.get("normalization")
    if not isinstance(config, Mapping):
        raise ValueError("config 'normalization' must be a mapping")

    values: list[float] = []
    for config_field in ("absolute_tolerance", "relative_tolerance"):
        value = config.get(config_field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config 'normalization.{config_field}' must be numeric")
        number = float(value)
        if not isfinite(number) or number < 0.0:
            raise ValueError(
                f"config 'normalization.{config_field}' must be finite and nonnegative"
            )
        values.append(number)
    return values[0], values[1]


class YieldNormalizationContract:
    """Recompute the fixture-local normalized yield from declared inputs."""

    contract_id = YIELD_NORMALIZATION_CONTRACT_ID
    description = "The observed yield matches the declared normalization formula and inputs."
    stage = NORMALIZATION_STAGE
    required_inputs = ("normalization", "config")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        (
            _,
            selected_weight_sum,
            generated_weight_sum,
            cross_section_pb,
            luminosity_pb_inverse,
            observed_yield,
        ) = require_normalization(context)
        absolute_tolerance, relative_tolerance = _normalization_tolerances(context)
        expected_yield = (
            selected_weight_sum * cross_section_pb * luminosity_pb_inverse / generated_weight_sum
        )
        if not isfinite(expected_yield):
            raise ValueError("the recomputed normalized yield must be finite")
        absolute_difference = abs(observed_yield - expected_yield)
        relative_difference: JsonValue = None
        if expected_yield != 0.0:
            relative_difference = absolute_difference / abs(expected_yield)

        evidence: dict[str, JsonValue] = {
            "formula": NORMALIZATION_FORMULA,
            "assumptions": [
                "generated_weight_sum is finite and nonzero",
                "cross_section_pb and luminosity_pb_inverse use reciprocal pb units",
            ],
            "selected_weight_sum": selected_weight_sum,
            "generated_weight_sum": generated_weight_sum,
            "cross_section_pb": cross_section_pb,
            "luminosity_pb_inverse": luminosity_pb_inverse,
            "expected_yield": expected_yield,
            "observed_yield": observed_yield,
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        }
        if isclose(
            observed_yield,
            expected_yield,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.PASS,
                evidence=evidence,
                duration_ms=elapsed_ms(start_ns),
            )

        return failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The observed yield does not match the declared normalization calculation.",
            likely_causes=(
                "A luminosity, cross-section, weight-sum, or scale factor changed upstream.",
            ),
            suggested_actions=(
                "Compare the observed yield with the reported formula inputs and recomputed value.",
            ),
            start_ns=start_ns,
            affected_artifacts=("normalization",),
        )
