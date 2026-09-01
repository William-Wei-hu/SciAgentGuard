"""Deterministic faults used to evaluate the ATLAS Gamma-Gamma boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite

from pydantic import JsonValue

from sciagentguard.core import ContractContext
from sciagentguard.packs.hep._events import (
    copy_event_columns,
    numeric_event_column,
    replace_event_columns,
)
from sciagentguard.packs.hep.atlas_analysis import (
    CONTROL_REGION,
    SIGNAL_REGION,
    replace_bin_weight_sums,
    replace_region_event_ids,
    require_histogram_artifact,
    require_regions,
    require_selection_artifact,
)
from sciagentguard.packs.hep.atlas_open_data import (
    ATLAS_DIPHOTON_PRESELECTION_CONTRACT_ID,
    ATLAS_HISTOGRAM_CLOSURE_CONTRACT_ID,
    ATLAS_REGION_DISJOINT_CONTRACT_ID,
    ATLAS_SOURCE_IDENTITY_CONTRACT_ID,
)
from sciagentguard.packs.hep.contracts import (
    EVENT_PROVENANCE_CONTRACT_ID,
    REQUIRED_BRANCHES_CONTRACT_ID,
)


class AtlasMissingPhotonMomentumInjector:
    """Remove the translated photon-momentum branch from a copied context."""

    fault_id = "atlas_missing_photon_momentum"
    taxonomy = "schema"
    description = "Remove photon_pt_gev after ROOT-to-context translation."
    preconditions = ("The events artifact contains photon_pt_gev.",)
    mutation_description = "Remove photon_pt_gev while preserving every other event branch."
    expected_contract_ids = (REQUIRED_BRANCHES_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared ROOT source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        columns = copy_event_columns(context)
        if "photon_pt_gev" not in columns:
            raise ValueError("atlas_missing_photon_momentum requires 'photon_pt_gev'")
        del columns["photon_pt_gev"]
        return replace_event_columns(context, columns)


class AtlasPhotonCountMismatchInjector:
    """Make one photon count disagree with its translated momentum vector."""

    fault_id = "atlas_photon_count_mismatch"
    taxonomy = "structure"
    description = "Increment one photon count without changing its momentum vector."
    preconditions = ("The events artifact contains a non-empty integer photon_count column.",)
    mutation_description = "Increment photon_count at zero-based event index 0 by one."
    expected_contract_ids = (ATLAS_DIPHOTON_PRESELECTION_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared ROOT source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        columns = copy_event_columns(context)
        counts = columns.get("photon_count")
        if not counts:
            raise ValueError("atlas_photon_count_mismatch requires a non-empty 'photon_count'")
        validated_counts: list[int] = []
        for index, value in enumerate(counts):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "atlas_photon_count_mismatch requires integer photon counts; "
                    f"index {index} is invalid"
                )
            validated_counts.append(value)
        changed = validated_counts
        changed[0] = changed[0] + 1
        columns["photon_count"] = tuple(changed)
        return replace_event_columns(context, columns)


class AtlasMissingEventProvenanceInjector:
    """Remove the event source declaration without modifying event data."""

    fault_id = "atlas_missing_event_provenance"
    taxonomy = "provenance"
    description = "Remove provenance for the translated events artifact."
    preconditions = ("The context contains provenance['events'].",)
    mutation_description = "Remove only provenance['events']."
    expected_contract_ids = (EVENT_PROVENANCE_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared ROOT source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        if "events" not in context.provenance:
            raise ValueError("atlas_missing_event_provenance requires provenance['events']")
        provenance = dict(context.provenance)
        del provenance["events"]
        return replace(context, provenance=provenance)


class AtlasSourceIdentityDriftInjector:
    """Change one public source identifier while preserving all event data."""

    fault_id = "atlas_source_identity_drift"
    taxonomy = "source_identity"
    description = "Replace the declared ATLAS record ID with a stable mismatched value."
    preconditions = ("The events provenance contains a non-empty record_id.",)
    mutation_description = "Set provenance['events']['record_id'] to 'atlas-record-drift'."
    expected_contract_ids = (ATLAS_SOURCE_IDENTITY_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared ROOT source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        declaration = _event_provenance(context, self.fault_id)
        record_id = declaration.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("atlas_source_identity_drift requires a non-empty record_id")
        declaration["record_id"] = "atlas-record-drift"
        provenance = dict(context.provenance)
        provenance["events"] = declaration
        return replace(context, provenance=provenance)


class AtlasWeightScaleGapInjector:
    """Probe the known absence of an integration-specific weight normalization check."""

    fault_id = "atlas_weight_scale_gap"
    taxonomy = "normalization_gap"
    description = "Multiply every finite event weight by ten."
    preconditions = ("The weight column is non-empty, finite, and has nonzero support.",)
    mutation_description = "Multiply every weight by 10.0 while preserving weight metadata."
    expected_contract_ids: tuple[str, ...] = ()
    restoration_strategy = "Discard the injected context and reload the declared ROOT source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        weights = numeric_event_column(context, "weight")
        if not weights or not all(isfinite(value) for value in weights):
            raise ValueError("atlas_weight_scale_gap requires non-empty finite weights")
        if sum(abs(value) for value in weights) == 0.0:
            raise ValueError("atlas_weight_scale_gap requires nonzero weight support")
        columns = copy_event_columns(context)
        columns["weight"] = tuple(value * 10.0 for value in weights)
        return replace_event_columns(context, columns)


class AtlasPhotonScaleGapInjector:
    """Probe the known absence of a photon momentum scale contract."""

    fault_id = "atlas_photon_scale_gap"
    taxonomy = "unit_gap"
    description = "Multiply every translated photon momentum by one thousand."
    preconditions = ("The photon_pt_gev rows are non-empty, numeric, finite, and non-negative.",)
    mutation_description = "Multiply every photon_pt_gev value by 1000.0 without changing units."
    expected_contract_ids: tuple[str, ...] = ()
    restoration_strategy = "Discard the injected context and reload the declared ROOT source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        columns = copy_event_columns(context)
        rows = columns.get("photon_pt_gev")
        if not rows:
            raise ValueError("atlas_photon_scale_gap requires non-empty 'photon_pt_gev'")

        scaled_rows: list[tuple[float, ...]] = []
        for event_index, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise ValueError(
                    "atlas_photon_scale_gap requires photon momentum sequences; "
                    f"event {event_index} is invalid"
                )
            scaled_row: list[float] = []
            for photon_index, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        "atlas_photon_scale_gap requires numeric photon momenta; "
                        f"event {event_index}, photon {photon_index} is invalid"
                    )
                number = float(value)
                if not isfinite(number) or number < 0.0:
                    raise ValueError(
                        "atlas_photon_scale_gap requires finite non-negative photon momenta"
                    )
                scaled_row.append(number * 1000.0)
            if not scaled_row:
                raise ValueError("atlas_photon_scale_gap requires non-empty photon momentum rows")
            scaled_rows.append(tuple(scaled_row))

        columns["photon_pt_gev"] = tuple(scaled_rows)
        return replace_event_columns(context, columns)


class AtlasNormalizationScaleDriftInjector:
    """Apply a wrong luminosity scale to the binned weights only.

    This is a late-invisible fault. The yield checkpoint re-derived from the scaled histogram
    stays finite and non-negative, keeps its excess over the sideband estimate, and keeps its
    peak in the declared window, because every one of those properties is invariant under a
    constant rescaling. Only the histogram checkpoint still holds the selected weight sum that
    the scaled bins are supposed to conserve.
    """

    fault_id = "atlas_normalization_scale_drift"
    taxonomy = "late_invisible_normalization"
    description = "Scale the binned weight sums without updating the selected weight total."
    preconditions = ("The histogram artifact declares non-empty bin weight sums.",)
    mutation_description = "Multiply every bin weight sum by 10.0 and leave all totals unchanged."
    expected_contract_ids = (ATLAS_HISTOGRAM_CLOSURE_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the histogram from selection."
    late_invisible_reason = (
        "The re-derived yield estimate remains finite, positive, and peaked in the declared "
        "window because the shape checks at the final checkpoint are scale-invariant."
    )

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        histogram = require_histogram_artifact(context)
        raw = histogram.get("bin_weight_sums")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
            raise ValueError("atlas_normalization_scale_drift requires non-empty bin weight sums")
        scaled: list[float] = []
        for index, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"atlas_normalization_scale_drift requires numeric bins; index {index} is not"
                )
            number = float(value)
            if not isfinite(number):
                raise ValueError("atlas_normalization_scale_drift requires finite bin weights")
            scaled.append(number * 10.0)
        return replace_bin_weight_sums(context, scaled)


class AtlasRegionOverlapInjector:
    """Duplicate signal-region events into the control region.

    This is a late-invisible fault. No downstream artifact carries region membership: the
    histogram is built from the selected masses and weights, so the re-derived histogram and
    yield are bit-for-bit identical to the valid run. Only the selection checkpoint can see it.
    """

    fault_id = "atlas_region_overlap"
    taxonomy = "late_invisible_leakage"
    description = "Add signal-region events to the control region without removing them."
    preconditions = ("The selection artifact declares a non-empty signal region.",)
    mutation_description = "Append the first signal-region event identifier to the control region."
    expected_contract_ids = (ATLAS_REGION_DISJOINT_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the selection from events."
    late_invisible_reason = (
        "Region membership is not carried past the selection checkpoint, so the histogram and "
        "yield artifacts are unchanged and the final checkpoint has nothing to inspect."
    )

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        selection = require_selection_artifact(context)
        regions = require_regions(selection)
        signal = regions.get(SIGNAL_REGION)
        control = regions.get(CONTROL_REGION)
        if not signal:
            raise ValueError("atlas_region_overlap requires a non-empty signal region")
        if control is None:
            raise ValueError("atlas_region_overlap requires a declared control region")
        return replace_region_event_ids(context, CONTROL_REGION, (*control, signal[0]))


def _event_provenance(context: ContractContext, fault_id: str) -> dict[str, JsonValue]:
    declaration = context.provenance.get("events")
    if not isinstance(declaration, Mapping):
        raise ValueError(f"{fault_id} requires a provenance mapping for events")

    copied: dict[str, JsonValue] = {}
    for key, value in declaration.items():
        if not isinstance(key, str):
            raise ValueError(f"{fault_id} requires string provenance keys")
        copied[key] = value
    return copied


__all__ = [
    "AtlasMissingEventProvenanceInjector",
    "AtlasMissingPhotonMomentumInjector",
    "AtlasNormalizationScaleDriftInjector",
    "AtlasPhotonCountMismatchInjector",
    "AtlasPhotonScaleGapInjector",
    "AtlasRegionOverlapInjector",
    "AtlasSourceIdentityDriftInjector",
    "AtlasWeightScaleGapInjector",
]
