"""Contracts specific to the ATLAS Open Data Gamma-Gamma boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter_ns

from pydantic import JsonValue

from sciagentguard.core import ContractContext, ContractResult, ContractStatus
from sciagentguard.packs.hep._contract_results import elapsed_ms, failed_result
from sciagentguard.packs.hep._events import require_event_columns
from sciagentguard.packs.hep.atlas_analysis import (
    ATLAS_HISTOGRAM_STAGE,
    ATLAS_YIELD_STAGE,
    require_cutflow,
    require_histogram_artifact,
    require_regions,
    require_selection_artifact,
    require_yield_artifact,
)
from sciagentguard.packs.hep.fixtures import HEP_STAGE, SELECTION_STAGE

ATLAS_DIPHOTON_PRESELECTION_CONTRACT_ID = "hep.atlas_open_data.diphoton_preselection"
ATLAS_SOURCE_IDENTITY_CONTRACT_ID = "hep.atlas_open_data.source_identity"
ATLAS_CUTFLOW_MONOTONIC_CONTRACT_ID = "hep.atlas_open_data.cutflow_monotonic"
ATLAS_REGION_DISJOINT_CONTRACT_ID = "hep.atlas_open_data.region_disjoint"
ATLAS_HISTOGRAM_CLOSURE_CONTRACT_ID = "hep.atlas_open_data.histogram_closure"
ATLAS_YIELD_SHAPE_CONTRACT_ID = "hep.atlas_open_data.yield_shape"
ATLAS_WEIGHT_PROVENANCE_CONTRACT_ID = "hep.atlas_open_data.weight_provenance"
ATLAS_REGION_COVERAGE_CONTRACT_ID = "hep.atlas_open_data.region_coverage"
ATLAS_SOURCE_CONSTANTS_CONTRACT_ID = "hep.atlas_open_data.source_constants"
ATLAS_REGION_DEFINITION_CONTRACT_ID = "hep.atlas_open_data.region_definition"
ATLAS_YIELD_CLOSURE_CONTRACT_ID = "hep.atlas_open_data.yield_closure"
ATLAS_BACKGROUND_ESTIMATE_CONTRACT_ID = "hep.atlas_open_data.background_estimate"
ATLAS_NORMALIZATION_PROVENANCE_CONTRACT_ID = "hep.atlas_open_data.normalization_provenance"

# Tolerance shared with the histogram closure check, and justified the same way: the source stores
# weights in single precision, so an analysis that accumulates in float32 differs from one that
# promotes to float64 by roughly the float32 epsilon.
SOURCE_FACT_RELATIVE_TOLERANCE = 1e-6


class AtlasDiphotonPreselectionContract:
    """Check the declared two-photon structure of a Gamma-Gamma event sample."""

    contract_id = ATLAS_DIPHOTON_PRESELECTION_CONTRACT_ID
    description = "Gamma-Gamma events contain at least two finite, non-negative photon momenta."
    stage = HEP_STAGE
    required_inputs = ("events", "config", "units")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        minimum_photons = _minimum_photons(context)
        columns = require_event_columns(context)
        missing = tuple(
            branch for branch in ("photon_count", "photon_pt_gev") if branch not in columns
        )
        if missing:
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.NOT_APPLICABLE,
                evidence={
                    "reason": "the event artifact does not contain the required photon branches",
                    "missing_branches": _json_list(missing),
                },
                duration_ms=elapsed_ms(start_ns),
            )

        if context.units.get("photon_pt_gev") != "GeV":
            raise ValueError("units 'photon_pt_gev' must be declared as 'GeV'")

        counts = _photon_counts(columns["photon_count"])
        momenta = _photon_momenta(columns["photon_pt_gev"])
        insufficient_indices: list[int] = []
        length_mismatch_indices: list[int] = []
        nonfinite_indices: list[int] = []
        negative_indices: list[int] = []

        for index, (count, values) in enumerate(zip(counts, momenta, strict=True)):
            if count < minimum_photons or len(values) < minimum_photons:
                insufficient_indices.append(index)
            if count != len(values):
                length_mismatch_indices.append(index)
            if any(not isfinite(value) for value in values):
                nonfinite_indices.append(index)
            if any(isfinite(value) and value < 0.0 for value in values):
                negative_indices.append(index)

        evidence: dict[str, JsonValue] = {
            "minimum_photons": minimum_photons,
            "event_count": len(counts),
            "observed_photon_count_range": (
                _json_list((min(counts), max(counts))) if counts else None
            ),
            "insufficient_photon_count": len(insufficient_indices),
            "insufficient_sample_indices": _json_list(insufficient_indices[:10]),
            "length_mismatch_count": len(length_mismatch_indices),
            "length_mismatch_sample_indices": _json_list(length_mismatch_indices[:10]),
            "nonfinite_momentum_count": len(nonfinite_indices),
            "nonfinite_sample_indices": _json_list(nonfinite_indices[:10]),
            "negative_momentum_count": len(negative_indices),
            "negative_sample_indices": _json_list(negative_indices[:10]),
        }
        if not any(
            (
                insufficient_indices,
                length_mismatch_indices,
                nonfinite_indices,
                negative_indices,
            )
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
            message="The Gamma-Gamma event artifact violates its declared diphoton preselection.",
            likely_causes=(
                "Photon multiplicities and momentum vectors drifted apart during data handling.",
                "A photon momentum conversion produced a negative or non-finite value.",
            ),
            suggested_actions=(
                "Inspect the bounded event indices and the ROOT-to-context conversion step.",
            ),
            start_ns=start_ns,
        )


@dataclass(frozen=True, slots=True)
class AtlasSourceIdentityContract:
    """Check a provenance declaration against one expected ATLAS source identity."""

    expected_source_type: str
    expected_record_id: str
    expected_doi: str
    expected_file_name: str
    expected_checksum: str
    contract_id: str = field(init=False, default=ATLAS_SOURCE_IDENTITY_CONTRACT_ID)
    description: str = field(
        init=False,
        default="The declared ATLAS source identity matches the verified local file.",
    )
    stage: str = field(init=False, default=HEP_STAGE)
    required_inputs: tuple[str, ...] = field(init=False, default=("provenance",))

    def __post_init__(self) -> None:
        for name in (
            "expected_source_type",
            "expected_record_id",
            "expected_doi",
            "expected_file_name",
            "expected_checksum",
        ):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        declaration = context.provenance.get("events")
        if not isinstance(declaration, Mapping):
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.NOT_APPLICABLE,
                evidence={"reason": "a declared event provenance mapping is required first"},
                duration_ms=elapsed_ms(start_ns),
            )
        source_type = declaration.get("source_type")
        if not isinstance(source_type, str) or not source_type.strip():
            return ContractResult(
                contract_id=self.contract_id,
                status=ContractStatus.NOT_APPLICABLE,
                evidence={"reason": "a declared event source type is required first"},
                duration_ms=elapsed_ms(start_ns),
            )

        expected: dict[str, str] = {
            "source_type": self.expected_source_type,
            "experiment": "ATLAS",
            "record_id": self.expected_record_id,
            "doi": self.expected_doi,
            "file_name": self.expected_file_name,
            "checksum": self.expected_checksum,
        }
        mismatched_fields = [
            field_name
            for field_name, expected_value in expected.items()
            if declaration.get(field_name) != expected_value
        ]
        evidence: dict[str, JsonValue] = {
            "expected_source_type": self.expected_source_type,
            "expected_record_id": self.expected_record_id,
            "expected_doi": self.expected_doi,
            "expected_file_name": self.expected_file_name,
            "expected_checksum": self.expected_checksum,
            "mismatched_fields": _json_list(mismatched_fields),
        }
        if not mismatched_fields:
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
            message="The event provenance does not match the expected ATLAS source identity.",
            likely_causes=(
                "The workflow loaded a different file or changed provenance after verification.",
            ),
            suggested_actions=(
                "Re-verify the local file and rebuild the context from its source descriptor.",
            ),
            start_ns=start_ns,
        )


class AtlasCutflowMonotonicContract:
    """Require the declared cutflow to shrink monotonically and match its endpoints."""

    contract_id = ATLAS_CUTFLOW_MONOTONIC_CONTRACT_ID
    description = "Each declared cut retains no more events than the cut before it."
    stage = SELECTION_STAGE
    required_inputs = ("selection",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        selection = require_selection_artifact(context)
        cutflow = require_cutflow(selection)
        input_count = len(_identifiers(selection, "input_event_ids"))
        selected_count = len(_identifiers(selection, "selected_event_ids"))

        increases = [
            cutflow[index][0]
            for index in range(1, len(cutflow))
            if cutflow[index][1] > cutflow[index - 1][1]
        ]
        negatives = [cut_id for cut_id, surviving in cutflow if surviving < 0]
        first_matches = cutflow[0][1] == input_count
        last_matches = cutflow[-1][1] == selected_count

        evidence: dict[str, JsonValue] = {
            "cut_ids": _json_list([cut_id for cut_id, _ in cutflow]),
            "surviving_counts": _json_list([surviving for _, surviving in cutflow]),
            "input_count": input_count,
            "selected_count": selected_count,
            "increasing_cuts": _json_list(increases),
            "negative_cuts": _json_list(negatives),
            "first_matches_input": first_matches,
            "last_matches_selected": last_matches,
        }
        if not increases and not negatives and first_matches and last_matches:
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
            message="The declared selection cutflow is not consistent with its own endpoints.",
            likely_causes=(
                "A cut was applied out of order or a surviving count was recorded incorrectly.",
                "The selected event list was rebuilt without updating the cutflow.",
            ),
            suggested_actions=(
                "Compare each recorded surviving count with the events entering that cut.",
            ),
            start_ns=start_ns,
            affected_artifacts=("selection",),
        )


class AtlasRegionDisjointContract:
    """Require the declared analysis regions to share no event and stay inside the selection."""

    contract_id = ATLAS_REGION_DISJOINT_CONTRACT_ID
    description = "Declared analysis regions partition selected events without overlap."
    stage = SELECTION_STAGE
    required_inputs = ("selection",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        selection = require_selection_artifact(context)
        regions = require_regions(selection)
        selected = set(_identifiers(selection, "selected_event_ids"))

        names = sorted(regions)
        overlaps: list[str] = []
        overlap_samples: list[int] = []
        for position, left in enumerate(names):
            for right in names[position + 1 :]:
                shared = sorted(set(regions[left]) & set(regions[right]))
                if shared:
                    overlaps.append(f"{left}|{right}")
                    overlap_samples.extend(shared[:10])

        outside = sorted(
            {event_id for values in regions.values() for event_id in values} - selected
        )
        duplicated = [name for name in names if len(set(regions[name])) != len(regions[name])]

        evidence: dict[str, JsonValue] = {
            "region_names": _json_list(names),
            "region_sizes": _json_list([len(regions[name]) for name in names]),
            "selected_count": len(selected),
            "overlapping_region_pairs": _json_list(overlaps),
            "overlap_sample_event_ids": _json_list(sorted(set(overlap_samples))[:10]),
            "events_outside_selection": len(outside),
            "outside_sample_event_ids": _json_list(outside[:10]),
            "regions_with_internal_duplicates": _json_list(duplicated),
        }
        if not overlaps and not outside and not duplicated:
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
            message="The declared analysis regions overlap or contain unselected events.",
            likely_causes=(
                "A region boundary was widened without removing events from the other region.",
                "Region membership was rebuilt from a stale selection.",
            ),
            suggested_actions=(
                "Recompute region membership from the current selection and compare the counts.",
            ),
            start_ns=start_ns,
            affected_artifacts=("selection",),
        )


class AtlasHistogramClosureContract:
    """Require the histogram to conserve selected weight and declare a consistent scale."""

    contract_id = ATLAS_HISTOGRAM_CLOSURE_CONTRACT_ID
    description = "Binned weights conserve the selected weight sum and the declared scale closes."
    stage = ATLAS_HISTOGRAM_STAGE
    required_inputs = ("histogram", "config")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        histogram = require_histogram_artifact(context)
        tolerance = _closure_tolerance(context)

        edges = _numbers(histogram, "bin_edges")
        weight_sums = _numbers(histogram, "bin_weight_sums")
        underflow = _number(histogram, "underflow_weight_sum")
        overflow = _number(histogram, "overflow_weight_sum")
        selected_weight_sum = _number(histogram, "selected_weight_sum")
        cross_section_pb = _number(histogram, "cross_section_pb")
        luminosity = _number(histogram, "luminosity_pb_inverse")
        generated_weight_sum = _number(histogram, "generated_weight_sum")
        declared_factor = _number(histogram, "normalization_factor")

        binned_total = sum(weight_sums) + underflow + overflow
        weight_gap = abs(binned_total - selected_weight_sum)
        weight_scale = max(abs(selected_weight_sum), 1.0)
        weight_closes = weight_gap <= tolerance * weight_scale

        edges_consistent = len(edges) == len(weight_sums) + 1
        expected_factor = (
            cross_section_pb * luminosity / generated_weight_sum
            if generated_weight_sum != 0.0
            else None
        )
        factor_gap = abs(declared_factor - expected_factor) if expected_factor is not None else None
        factor_closes = (
            expected_factor is not None
            and factor_gap is not None
            and factor_gap <= tolerance * max(abs(expected_factor), 1.0)
        )

        evidence: dict[str, JsonValue] = {
            "bin_count": len(weight_sums),
            "edge_count": len(edges),
            "edges_consistent": edges_consistent,
            "binned_weight_total": _json_number(binned_total),
            "selected_weight_sum": _json_number(selected_weight_sum),
            "weight_absolute_gap": _json_number(weight_gap),
            "relative_tolerance": tolerance,
            "declared_normalization_factor": _json_number(declared_factor),
            "recomputed_normalization_factor": (
                None if expected_factor is None else _json_number(expected_factor)
            ),
            "normalization_absolute_gap": (
                None if factor_gap is None else _json_number(factor_gap)
            ),
            "closure_relation": (
                "sum(bin_weight_sums) + underflow + overflow == selected_weight_sum"
            ),
            "normalization_relation": (
                "cross_section_pb * luminosity_pb_inverse / generated_weight_sum"
            ),
        }
        # The same constants are declared again here. Checking them at this stage too stops a
        # mistake corrected at selection from re-entering one stage later.
        facts = _source_facts(context)
        drifted_constants: list[str] = []
        if facts is not None:
            for name, declared in (
                ("cross_section_pb", cross_section_pb),
                ("generated_weight_sum", generated_weight_sum),
            ):
                if not _close(_number(facts, name), declared, SOURCE_FACT_RELATIVE_TOLERANCE):
                    drifted_constants.append(name)
        evidence["constants_matching_source"] = facts is not None
        evidence["drifted_constants"] = _json_list(drifted_constants)

        if weight_closes and edges_consistent and factor_closes and not drifted_constants:
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
            message="The histogram does not conserve the selected weight or its declared scale.",
            likely_causes=(
                "A scale factor was applied to the bins without updating the selection totals.",
                "Events were dropped or duplicated between selection and binning.",
                "A per-file constant was aggregated instead of read once.",
            ),
            suggested_actions=(
                "Recompute the binned weight total and compare it with the selected weight sum.",
                "Recompute the normalization factor from cross section, luminosity, and weights.",
            ),
            start_ns=start_ns,
            affected_artifacts=("histogram",),
        )


class AtlasYieldShapeContract:
    """Check scale-invariant shape properties of the declared yield estimate.

    This contract deliberately checks only properties that a constant rescaling cannot change:
    finiteness, non-negative support, the position of the peak bin, and the presence of an
    excess over the sideband estimate. A constant normalization error is therefore invisible
    here by construction, which is why it must be caught at the histogram checkpoint instead.
    """

    contract_id = ATLAS_YIELD_SHAPE_CONTRACT_ID
    description = "The yield estimate is finite, has an excess, and peaks in the declared window."
    stage = ATLAS_YIELD_STAGE
    required_inputs = ("yield_estimate", "config")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        estimate = require_yield_artifact(context)
        low, high = _peak_window(context)

        estimated_yield = _number(estimate, "estimated_yield")
        signal_weight_sum = _number(estimate, "signal_weight_sum")
        background_estimate = _number(estimate, "background_estimate")
        peak_center = _number(estimate, "peak_bin_center_gev")

        finite = all(
            isfinite(value)
            for value in (estimated_yield, signal_weight_sum, background_estimate, peak_center)
        )
        nonnegative_support = signal_weight_sum >= 0.0
        has_excess = signal_weight_sum > background_estimate
        peak_in_window = low <= peak_center <= high

        evidence: dict[str, JsonValue] = {
            "expected_peak_window_gev": [low, high],
            "peak_bin_center_gev": _json_number(peak_center),
            "peak_in_window": peak_in_window,
            "signal_weight_sum": _json_number(signal_weight_sum),
            "background_estimate": _json_number(background_estimate),
            "has_excess_over_sideband": has_excess,
            "all_values_finite": finite,
            "nonnegative_signal_support": nonnegative_support,
            "scale_invariance_note": (
                "these checks are invariant under a constant rescaling of the histogram"
            ),
        }
        if finite and nonnegative_support and has_excess and peak_in_window:
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
            message="The yield estimate does not satisfy its declared shape expectations.",
            likely_causes=(
                "The selection or binning moved the peak outside the declared window.",
                "The sideband estimate exceeded the signal window content.",
            ),
            suggested_actions=(
                "Inspect the binned distribution and the declared signal window together.",
            ),
            start_ns=start_ns,
            affected_artifacts=("yield_estimate",),
        )


class AtlasWeightProvenanceContract:
    """Require the reported weights to be the source's own, not a rescaling of them.

    Every other weight check in this pack asks whether an artifact agrees with itself. This one
    asks where its numbers came from. An analysis that folds the normalization factor into each
    event weight stays perfectly self-consistent -- the binned totals still match the declared
    sum -- while every weight it reports is one the source never contained.
    """

    contract_id = ATLAS_WEIGHT_PROVENANCE_CONTRACT_ID
    description = "Selected weights are raw source weights, and their declared sum is their sum."
    stage = SELECTION_STAGE
    required_inputs = ("selection", "provenance")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        selection = require_selection_artifact(context)
        facts = _source_facts(context)
        if facts is None:
            return _not_applicable(self.contract_id, start_ns)

        weights = _numbers(selection, "selected_weight")
        declared_sum = _number(selection, "selected_weight_sum")

        computed = sum(weights)
        gap = abs(computed - declared_sum)
        sum_closes = gap <= SOURCE_FACT_RELATIVE_TOLERANCE * max(abs(computed), 1.0)
        outside, mode = _weights_not_from_source(weights, facts)

        evidence: dict[str, JsonValue] = {
            "selected_count": len(weights),
            "declared_weight_sum": _json_number(declared_sum),
            "recomputed_weight_sum": _json_number(computed),
            "absolute_gap": _json_number(gap),
            "source_weight_abs_min": _json_number(_number(facts, "weight_abs_min")),
            "source_weight_abs_max": _json_number(_number(facts, "weight_abs_max")),
            "source_distinct_weight_count": facts.get("distinct_weight_count"),
            "membership_check": mode,
            "weights_not_from_source": len(outside),
            "outside_sample_indices": _json_list(outside[:10]),
            "relative_tolerance": SOURCE_FACT_RELATIVE_TOLERANCE,
            "checked_relation": (
                "selected_weight_sum == sum(selected_weight), and every selected weight is one "
                "the source actually contains"
            ),
        }
        if sum_closes and not outside:
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
            message="The reported event weights are not the source's own weights.",
            likely_causes=(
                "A normalization or scale factor was applied to the per-event weights.",
                "The declared weight sum was computed from a different list than the one reported.",
            ),
            suggested_actions=(
                "Report raw event weights and apply any scale factor only to the final estimate.",
            ),
            start_ns=start_ns,
            affected_artifacts=("selection",),
        )


class AtlasRegionCoverageContract:
    """Require the declared regions to account for every selected event.

    Disjointness alone permits an event to belong to no region at all, which silently drops it
    from the analysis while every other check still passes.
    """

    contract_id = ATLAS_REGION_COVERAGE_CONTRACT_ID
    description = "Declared analysis regions together cover every selected event."
    stage = SELECTION_STAGE
    required_inputs = ("selection",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        selection = require_selection_artifact(context)
        regions = require_regions(selection)
        selected = set(_identifiers(selection, "selected_event_ids"))
        covered = {event_id for values in regions.values() for event_id in values}

        uncovered = sorted(selected - covered)
        evidence: dict[str, JsonValue] = {
            "region_names": _json_list(sorted(regions)),
            "selected_count": len(selected),
            "covered_count": len(covered & selected),
            "uncovered_count": len(uncovered),
            "uncovered_sample_event_ids": _json_list(uncovered[:10]),
            "checked_relation": "the union of the declared regions equals the selected events",
        }
        if not uncovered:
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
            message="Some selected events belong to no declared analysis region.",
            likely_causes=(
                "A region boundary was narrowed without widening the region beside it.",
                "A region was defined by its own window rather than as the complement of another.",
            ),
            suggested_actions=(
                "Define the control region as every selected event outside the signal window.",
            ),
            start_ns=start_ns,
            affected_artifacts=("selection",),
        )


@dataclass(frozen=True, slots=True)
class AtlasSourceConstantsContract:
    """Require declared per-file constants to match what the trusted loader actually read.

    `XSection` and `SumWeights` are single values repeated on every event. Summing them across the
    sample produces a number that is internally usable -- a normalization factor computed from it
    is perfectly self-consistent -- and wrong by a factor of the event count.
    """

    artifact_name: str = "selection"
    contract_id: str = field(init=False, default=ATLAS_SOURCE_CONSTANTS_CONTRACT_ID)
    description: str = field(
        init=False,
        default="Declared source constants match the values read from the verified file.",
    )
    stage: str = field(init=False, default=SELECTION_STAGE)
    required_inputs: tuple[str, ...] = field(init=False, default=("provenance",))

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        artifact = context.artifacts.get(self.artifact_name)
        facts = _source_facts(context)
        if not isinstance(artifact, Mapping) or facts is None:
            return _not_applicable(self.contract_id, start_ns)

        mismatches: list[str] = []
        evidence: dict[str, JsonValue] = {
            "artifact": self.artifact_name,
            "relative_tolerance": SOURCE_FACT_RELATIVE_TOLERANCE,
        }
        for name in ("cross_section_pb", "generated_weight_sum"):
            expected = _number(facts, name)
            declared = artifact.get(name)
            if isinstance(declared, bool) or not isinstance(declared, (int, float)):
                mismatches.append(name)
                evidence[f"declared_{name}"] = None
            else:
                actual = float(declared)
                evidence[f"declared_{name}"] = _json_number(actual)
                if not _close(expected, actual, SOURCE_FACT_RELATIVE_TOLERANCE):
                    mismatches.append(name)
            evidence[f"source_{name}"] = _json_number(expected)
        evidence["mismatched_constants"] = _json_list(mismatches)

        if not mismatches:
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
            message="A declared per-file constant does not match the verified source.",
            likely_causes=(
                "A per-file constant repeated on every event was summed across the sample.",
                "A constant was taken from a different file or recomputed rather than read.",
            ),
            suggested_actions=(
                "Read one value of each per-file constant instead of aggregating the column.",
            ),
            start_ns=start_ns,
            affected_artifacts=(self.artifact_name,),
        )


class AtlasYieldClosureContract:
    """Require the reported yield to be what its own declared inputs imply.

    This contract exists because a model found the gap. Asked to review the final artifact, a
    judge from a different vendor objected that a signal sum of 82,383 and a background of 1,614
    cannot produce a yield of 7.47, and it was right: the scale factor relating them lived in the
    histogram artifact and never reached the one that reported the result. Every other check
    passed, because none of them looked at whether the final number followed from the numbers
    printed beside it.
    """

    contract_id = ATLAS_YIELD_CLOSURE_CONTRACT_ID
    description = "The reported yield equals its declared excess times its declared scale factor."
    stage = ATLAS_YIELD_STAGE
    required_inputs = ("yield_estimate",)

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        estimate = require_yield_artifact(context)
        if "normalization_factor" not in estimate:
            return failed_result(
                context,
                contract_id=self.contract_id,
                evidence={
                    "reason": "the artifact reports a yield without the factor that produced it",
                    "checked_relation": (
                        "estimated_yield == (signal_weight_sum - background_estimate) "
                        "* normalization_factor"
                    ),
                },
                message="The final artifact cannot be checked against its own inputs.",
                likely_causes=(
                    "The scale factor was applied but reported only at an earlier stage.",
                ),
                suggested_actions=(
                    "Report the normalization factor alongside the yield it produced.",
                ),
                start_ns=start_ns,
                affected_artifacts=("yield_estimate",),
            )

        signal = _number(estimate, "signal_weight_sum")
        background = _number(estimate, "background_estimate")
        factor = _number(estimate, "normalization_factor")
        reported = _number(estimate, "estimated_yield")
        implied = (signal - background) * factor
        gap = abs(implied - reported)
        closes = gap <= SOURCE_FACT_RELATIVE_TOLERANCE * max(abs(implied), 1.0)

        evidence: dict[str, JsonValue] = {
            "signal_weight_sum": _json_number(signal),
            "background_estimate": _json_number(background),
            "normalization_factor": _json_number(factor),
            "reported_yield": _json_number(reported),
            "implied_yield": _json_number(implied),
            "absolute_gap": _json_number(gap),
            "relative_tolerance": SOURCE_FACT_RELATIVE_TOLERANCE,
            "checked_relation": (
                "estimated_yield == (signal_weight_sum - background_estimate) "
                "* normalization_factor"
            ),
        }
        if closes:
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
            message="The reported yield does not follow from the values reported beside it.",
            likely_causes=(
                "A scale factor was applied to the yield but not to the sums it came from.",
                "The signal or background sum is not the one the yield was computed from.",
            ),
            suggested_actions=(
                "Recompute the yield from the reported excess and factor and compare.",
            ),
            start_ns=start_ns,
            affected_artifacts=("yield_estimate",),
        )


class AtlasNormalizationProvenanceContract:
    """Require the scale factor reported with a yield to follow from the verified source.

    The final artifact reports `normalization_factor` and none of the three numbers behind it, so
    nothing inside that artifact can establish it. The reviewer that proposed this said exactly
    that -- "recomputing `normalization_factor` requires the underlying luminosity, cross-section,
    or normalization inputs; these values alone cannot establish it" -- and it was right: inflating
    the factor by half and recomputing the yield from it left every contract at this stage passing.

    This is a derivation check rather than a consistency one. The cross-section and generated weight
    sum are not read from the artifact but from the provenance the verified loader attached, so an
    analysis cannot satisfy it by being self-consistent about a number it invented. The histogram
    stage checks the same relation against its own declared inputs; this one checks that the factor
    which actually reached the result is still the one the file implies.
    """

    contract_id = ATLAS_NORMALIZATION_PROVENANCE_CONTRACT_ID
    description = "The reported scale factor follows from the verified source and the luminosity."
    stage = ATLAS_YIELD_STAGE
    required_inputs = ("yield_estimate", "provenance", "config")

    _RELATION = (
        "normalization_factor == cross_section_pb * luminosity_pb_inverse / generated_weight_sum"
    )

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        estimate = require_yield_artifact(context)
        facts = _source_facts(context)
        luminosity = _declared_luminosity(context)
        if facts is None or luminosity is None:
            # Matches the other source-fact contracts: a workflow that has lost its provenance is
            # reported by the provenance contract at the checkpoint where it lost it, not by
            # every downstream check failing for a second reason.
            return _not_applicable(self.contract_id, start_ns)

        if "normalization_factor" not in estimate:
            return failed_result(
                context,
                contract_id=self.contract_id,
                evidence={
                    "reason": "the artifact reports a yield without the factor that produced it",
                    "checked_relation": self._RELATION,
                },
                message="The scale factor cannot be checked against the source it came from.",
                likely_causes=("The factor was applied but reported only at an earlier stage.",),
                suggested_actions=("Report the normalization factor alongside the yield.",),
                start_ns=start_ns,
                affected_artifacts=("yield_estimate",),
            )

        cross_section = _number(facts, "cross_section_pb")
        generated = _number(facts, "generated_weight_sum")
        declared = _number(estimate, "normalization_factor")

        evidence: dict[str, JsonValue] = {
            "declared_normalization_factor": _json_number(declared),
            "source_cross_section_pb": _json_number(cross_section),
            "source_generated_weight_sum": _json_number(generated),
            "declared_luminosity_pb_inverse": _json_number(luminosity),
            "relative_tolerance": SOURCE_FACT_RELATIVE_TOLERANCE,
            "checked_relation": self._RELATION,
        }

        if generated == 0.0:
            evidence["reason"] = "the source reports no generated weight to normalize by"
            return failed_result(
                context,
                contract_id=self.contract_id,
                evidence=evidence,
                message="The scale factor has no denominator in the verified source.",
                likely_causes=("The generated weight sum was read as zero.",),
                suggested_actions=("Check that SumWeights was read from the file, not summed.",),
                start_ns=start_ns,
                affected_artifacts=("yield_estimate",),
            )

        implied = cross_section * luminosity / generated
        gap = abs(implied - declared)
        evidence["implied_normalization_factor"] = _json_number(implied)
        evidence["absolute_gap"] = _json_number(gap)

        if _close(implied, declared, SOURCE_FACT_RELATIVE_TOLERANCE):
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
            message="The reported scale factor does not follow from the verified source.",
            likely_causes=(
                "A different luminosity or cross-section was used than the one declared.",
                "The factor reaching the result is not the one the histogram computed.",
            ),
            suggested_actions=(
                "Recompute the factor from the source cross-section, luminosity and weight sum.",
            ),
            start_ns=start_ns,
            affected_artifacts=("yield_estimate",),
        )


class AtlasBackgroundEstimateContract:
    """Require the background to be the sideband scaled by the ratio of bin counts.

    `yield_closure` checks that the reported yield follows from the excess and the scale factor,
    which leaves the excess itself unexamined: double the background, recompute the yield from it,
    and the closure still holds. Measured on the real sample, that substitution moves the result by
    two percent and no contract of the sixteen notices.

    A reviewer shown only the final artifact proposed this relation, giving the arithmetic in full.
    It was the only proposal of thirty-three that turned out to be a scientific gap rather than a
    structural check or something already covered.
    """

    contract_id = ATLAS_BACKGROUND_ESTIMATE_CONTRACT_ID
    description = "The background estimate is the sideband sum scaled by the signal/sideband ratio."
    stage = ATLAS_YIELD_STAGE
    required_inputs = ("yield_estimate",)

    _RELATION = "background_estimate == sideband_weight_sum * signal_bin_count / sideband_bin_count"
    _FIELDS = (
        "background_estimate",
        "sideband_weight_sum",
        "signal_bin_count",
        "sideband_bin_count",
    )

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        estimate = require_yield_artifact(context)

        missing = [field_name for field_name in self._FIELDS if field_name not in estimate]
        if missing:
            return failed_result(
                context,
                contract_id=self.contract_id,
                evidence={
                    "reason": "the artifact reports a background without the sums behind it",
                    "missing_fields": _json_list(missing),
                    "checked_relation": self._RELATION,
                },
                message="The background estimate cannot be checked against its own inputs.",
                likely_causes=("The sideband sums were dropped before the yield was reported.",),
                suggested_actions=(
                    "Report the sideband sum and both bin counts alongside the background.",
                ),
                start_ns=start_ns,
                affected_artifacts=("yield_estimate",),
            )

        reported = _number(estimate, "background_estimate")
        sideband = _number(estimate, "sideband_weight_sum")
        signal_bins = _number(estimate, "signal_bin_count")
        sideband_bins = _number(estimate, "sideband_bin_count")

        if sideband_bins <= 0:
            return failed_result(
                context,
                contract_id=self.contract_id,
                evidence={
                    "sideband_bin_count": _json_number(sideband_bins),
                    "checked_relation": self._RELATION,
                    "reason": "a background scaled from no sideband bins has no denominator",
                },
                message="The background estimate declares no sideband bins to have come from.",
                likely_causes=("The sideband window selected no bins.",),
                suggested_actions=("Widen the sideband window or report the estimate as absent.",),
                start_ns=start_ns,
                affected_artifacts=("yield_estimate",),
            )

        implied = sideband * signal_bins / sideband_bins
        gap = abs(implied - reported)
        closes = gap <= SOURCE_FACT_RELATIVE_TOLERANCE * max(abs(implied), 1.0)

        evidence: dict[str, JsonValue] = {
            "reported_background": _json_number(reported),
            "implied_background": _json_number(implied),
            "sideband_weight_sum": _json_number(sideband),
            "signal_bin_count": _json_number(signal_bins),
            "sideband_bin_count": _json_number(sideband_bins),
            "absolute_gap": _json_number(gap),
            "relative_tolerance": SOURCE_FACT_RELATIVE_TOLERANCE,
            "checked_relation": self._RELATION,
        }
        if closes:
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
            message="The background estimate is not the sideband scaled by the bin ratio.",
            likely_causes=(
                "The sideband was scaled by a ratio other than the one it reports.",
                "The background came from a different estimate than the sideband sum shown.",
            ),
            suggested_actions=(
                "Recompute the background from the reported sideband and bin counts and compare.",
            ),
            start_ns=start_ns,
            affected_artifacts=("yield_estimate",),
        )


class AtlasRegionDefinitionContract:
    """Require the signal region to be the mass window it claims to be.

    Disjointness and coverage are structural: any partition of the selected events satisfies both,
    including one that ignores the physics entirely. A real model, told to build a control region
    holding "every other selected event", partitioned the selection into alternating events and
    passed both checks. A region defined by a window has to contain the events in that window.
    """

    contract_id = ATLAS_REGION_DEFINITION_CONTRACT_ID
    description = "The signal region contains exactly the selected events inside the mass window."
    stage = SELECTION_STAGE
    required_inputs = ("selection", "config")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        selection = require_selection_artifact(context)
        low, high = _signal_window(context)
        regions = require_regions(selection)
        signal = set(regions.get("signal", ()))

        identifiers = _identifiers(selection, "selected_event_ids")
        masses = _numbers(selection, "selected_mass_gev")
        if len(identifiers) != len(masses):
            return _not_applicable(self.contract_id, start_ns)

        expected = {
            event_id
            for event_id, mass in zip(identifiers, masses, strict=True)
            if low <= mass < high
        }
        missing = sorted(expected - signal)
        extra = sorted(signal - expected)

        evidence: dict[str, JsonValue] = {
            "signal_window_gev": [low, high],
            "expected_signal_count": len(expected),
            "declared_signal_count": len(signal),
            "missing_from_signal": len(missing),
            "unexpected_in_signal": len(extra),
            "missing_sample_event_ids": _json_list(missing[:10]),
            "unexpected_sample_event_ids": _json_list(extra[:10]),
            "checked_relation": (
                "the signal region equals the selected events whose mass lies in the window"
            ),
        }
        if not missing and not extra:
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
            message="The signal region is not the declared mass window.",
            likely_causes=(
                "Region membership was assigned by position or index rather than by mass.",
                "A different window, or an inclusive bound, was used to define the region.",
            ),
            suggested_actions=(
                "Assign each selected event to a region from its own mass and the declared window.",
            ),
            start_ns=start_ns,
            affected_artifacts=("selection",),
        )


def _signal_window(context: ContractContext) -> tuple[float, float]:
    config = context.config.get("atlas_yield")
    if not isinstance(config, Mapping):
        raise ValueError("config 'atlas_yield' must be a mapping")
    raw = config.get("signal_window_gev")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 2:
        raise ValueError("config 'atlas_yield.signal_window_gev' must have two bounds")
    bounds: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("config 'atlas_yield.signal_window_gev' must be numeric")
        bounds.append(float(value))
    if bounds[1] <= bounds[0]:
        raise ValueError("config 'atlas_yield.signal_window_gev' must be increasing")
    return bounds[0], bounds[1]


def _declared_luminosity(context: ContractContext) -> float | None:
    """Return the run's declared luminosity, or None when the workflow does not carry one."""

    config = context.config.get("atlas_normalization")
    if not isinstance(config, Mapping):
        return None
    value = config.get("luminosity_pb_inverse")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _source_facts(context: ContractContext) -> Mapping[str, JsonValue] | None:
    """Return the loader's trusted facts about the source, if the workflow still carries them."""

    for declaration in context.provenance.values():
        if isinstance(declaration, Mapping):
            facts = declaration.get("source_facts")
            if isinstance(facts, Mapping):
                return facts
    return None


def _not_applicable(contract_id: str, start_ns: int) -> ContractResult:
    return ContractResult(
        contract_id=contract_id,
        status=ContractStatus.NOT_APPLICABLE,
        evidence={"reason": "the workflow does not carry trusted source facts"},
        duration_ms=elapsed_ms(start_ns),
    )


def _close(expected: float, actual: float, tolerance: float) -> bool:
    if not isfinite(expected) or not isfinite(actual):
        return False
    return abs(expected - actual) <= tolerance * max(abs(expected), 1.0)


def _weights_not_from_source(
    weights: Sequence[float], facts: Mapping[str, JsonValue]
) -> tuple[list[int], str]:
    """Find weights the source cannot have produced, exactly where the facts allow it.

    When the source carries few enough distinct weights to list, membership is exact. Otherwise
    the check falls back to magnitude bounds, which is weaker: a sample whose weights span orders
    of magnitude gives it little to bite on. The mode used is recorded in the evidence so a reader
    knows which of the two they are looking at.
    """

    distinct = facts.get("distinct_weights")
    tolerance = SOURCE_FACT_RELATIVE_TOLERANCE
    if isinstance(distinct, list) and distinct:
        allowed = [float(value) for value in distinct if isinstance(value, (int, float))]
        offending = [
            index
            for index, weight in enumerate(weights)
            if not any(_close(candidate, weight, tolerance) for candidate in allowed)
        ]
        return offending, "exact membership in the source's distinct weights"

    low = _number(facts, "weight_abs_min")
    high = _number(facts, "weight_abs_max")
    margin = tolerance * max(abs(low), abs(high), 1.0)
    offending = [
        index
        for index, weight in enumerate(weights)
        if not isfinite(weight) or not low - margin <= abs(weight) <= high + margin
    ]
    return offending, "magnitude within the source's absolute weight bounds"


def _json_number(value: float) -> JsonValue:
    """Return a JSON-safe number, or None when the value is not finite.

    Traces forbid inf and NaN, and a non-finite intermediate is exactly what these contracts
    are meant to report, so it must survive into the evidence as an explicit absence.
    """

    return value if isfinite(value) else None


def _identifiers(artifact: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    values = artifact.get(field_name)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"the selection artifact field {field_name!r} must be a sequence")
    identifiers: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} has a non-integer identifier at index {index}")
        identifiers.append(value)
    return tuple(identifiers)


def _number(artifact: Mapping[str, object], field_name: str) -> float:
    value = artifact.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"artifact field {field_name!r} must be numeric")
    return float(value)


def _numbers(artifact: Mapping[str, object], field_name: str) -> tuple[float, ...]:
    values = artifact.get(field_name)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"artifact field {field_name!r} must be a sequence")
    numbers: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} has a non-numeric value at index {index}")
        numbers.append(float(value))
    return tuple(numbers)


def _closure_tolerance(context: ContractContext) -> float:
    config = context.config.get("atlas_histogram")
    if not isinstance(config, Mapping):
        raise ValueError("config 'atlas_histogram' must be a mapping")
    tolerance = config.get("closure_relative_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("config 'atlas_histogram.closure_relative_tolerance' must be numeric")
    if float(tolerance) <= 0.0:
        raise ValueError("config 'atlas_histogram.closure_relative_tolerance' must be positive")
    return float(tolerance)


def _peak_window(context: ContractContext) -> tuple[float, float]:
    config = context.config.get("atlas_yield")
    if not isinstance(config, Mapping):
        raise ValueError("config 'atlas_yield' must be a mapping")
    raw = config.get("expected_peak_window_gev")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 2:
        raise ValueError("config 'atlas_yield.expected_peak_window_gev' must have two bounds")
    bounds: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("config 'atlas_yield.expected_peak_window_gev' must be numeric")
        bounds.append(float(value))
    if bounds[1] <= bounds[0]:
        raise ValueError("config 'atlas_yield.expected_peak_window_gev' must be increasing")
    return bounds[0], bounds[1]


def _minimum_photons(context: ContractContext) -> int:
    config = context.config.get("atlas_gamgam")
    if not isinstance(config, Mapping):
        raise ValueError("config 'atlas_gamgam' must be a mapping")
    minimum = config.get("minimum_photons")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum != 2:
        raise ValueError("config 'atlas_gamgam.minimum_photons' must equal 2")
    return minimum


def _json_list(values: Sequence[str | int]) -> list[JsonValue]:
    result: list[JsonValue] = []
    for value in values:
        result.append(value)
    return result


def _photon_counts(values: Sequence[object]) -> tuple[int, ...]:
    counts: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"photon_count contains a non-integer value at index {index}")
        counts.append(value)
    return tuple(counts)


def _photon_momenta(values: Sequence[object]) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for event_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"photon_pt_gev contains a non-sequence row at index {event_index}")
        converted: list[float] = []
        for photon_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "photon_pt_gev contains a non-numeric value at "
                    f"event {event_index}, photon {photon_index}"
                )
            converted.append(float(value))
        rows.append(tuple(converted))
    return tuple(rows)


__all__ = [
    "ATLAS_BACKGROUND_ESTIMATE_CONTRACT_ID",
    "ATLAS_NORMALIZATION_PROVENANCE_CONTRACT_ID",
    "AtlasBackgroundEstimateContract",
    "AtlasCutflowMonotonicContract",
    "AtlasDiphotonPreselectionContract",
    "AtlasHistogramClosureContract",
    "AtlasNormalizationProvenanceContract",
    "AtlasRegionCoverageContract",
    "AtlasRegionDefinitionContract",
    "AtlasRegionDisjointContract",
    "AtlasSourceConstantsContract",
    "AtlasSourceIdentityContract",
    "AtlasWeightProvenanceContract",
    "AtlasYieldClosureContract",
    "AtlasYieldShapeContract",
]
