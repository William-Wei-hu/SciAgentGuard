"""Deterministic downstream stages of the ATLAS Gamma-Gamma boundary workflow.

Each stage derives one checkpoint context from the previous one and carries forward only the
quantities that the next stage needs. Later contexts therefore do not contain earlier artifacts:
the selection table is not present at the histogram checkpoint, and neither the selection table nor
the per-event masses are present at the yield checkpoint. That information loss is the property
that makes an intermediate checkpoint able to observe something the final artifact cannot.

The yield stage is a closed-form sideband-subtracted estimate over the declared histogram. It is
not a likelihood fit, not a background model, and not a physics measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import cos, sin, sinh, sqrt

from pydantic import JsonValue

from sciagentguard.core import ContractContext
from sciagentguard.packs.hep._events import require_event_columns
from sciagentguard.packs.hep.fixtures import SELECTION_STAGE

ATLAS_HISTOGRAM_STAGE = "post_histogram"
ATLAS_YIELD_STAGE = "post_yield"

ATLAS_SELECTION_ID = "atlas_gamgam_diphoton"
SIGNAL_REGION = "signal"
CONTROL_REGION = "control"

_CUT_IDS = ("all_events", "two_photons", "leading_photon_pt", "subleading_photon_pt")


# --------------------------------------------------------------------------------------
# Stage 2: selection
# --------------------------------------------------------------------------------------


def select_diphoton_events(context: ContractContext) -> ContractContext:
    """Apply the declared diphoton selection and record its cutflow and regions."""

    columns = require_event_columns(context)
    config = _selection_settings(context)
    leading_min = config["leading_photon_pt_min_gev"]
    subleading_min = config["subleading_photon_pt_min_gev"]
    minimum_photons = _minimum_photons(context)

    event_ids = _integer_column(columns, "event_number")
    weights = _float_column(columns, "weight")
    counts = _integer_column(columns, "photon_count")
    pt_rows = _row_column(columns, "photon_pt_gev")
    eta_rows = _row_column(columns, "photon_eta")
    phi_rows = _row_column(columns, "photon_phi")
    energy_rows = _row_column(columns, "photon_e_gev")

    surviving = [len(event_ids), 0, 0, 0]
    selected_ids: list[int] = []
    selected_weights: list[float] = []
    selected_masses: list[float] = []

    for index, event_id in enumerate(event_ids):
        pts = pt_rows[index]
        if counts[index] < minimum_photons or len(pts) < minimum_photons:
            continue
        surviving[1] += 1

        order = sorted(range(len(pts)), key=lambda position: pts[position], reverse=True)
        leading, subleading = order[0], order[1]
        if pts[leading] < leading_min:
            continue
        surviving[2] += 1
        if pts[subleading] < subleading_min:
            continue
        surviving[3] += 1

        mass = _invariant_mass(
            (
                pts[leading],
                eta_rows[index][leading],
                phi_rows[index][leading],
                energy_rows[index][leading],
            ),
            (
                pts[subleading],
                eta_rows[index][subleading],
                phi_rows[index][subleading],
                energy_rows[index][subleading],
            ),
        )
        selected_ids.append(event_id)
        selected_weights.append(weights[index])
        selected_masses.append(mass)

    signal_low, signal_high = _window(context, "signal_window_gev")
    signal_ids = tuple(
        event_id
        for event_id, mass in zip(selected_ids, selected_masses, strict=True)
        if signal_low <= mass < signal_high
    )
    control_ids = tuple(
        event_id
        for event_id, mass in zip(selected_ids, selected_masses, strict=True)
        if not signal_low <= mass < signal_high
    )

    selection = {
        "selection_id": ATLAS_SELECTION_ID,
        "input_event_ids": tuple(event_ids),
        "selected_event_ids": tuple(selected_ids),
        "selected_weight": tuple(selected_weights),
        "selected_mass_gev": tuple(selected_masses),
        "selected_weight_sum": sum(selected_weights),
        "generated_weight_sum": _scalar_column(columns, "generated_weight_sum"),
        "cross_section_pb": _scalar_column(columns, "cross_section_pb"),
        "cutflow": tuple(
            {"cut_id": cut_id, "surviving": count}
            for cut_id, count in zip(_CUT_IDS, surviving, strict=True)
        ),
        "regions": {SIGNAL_REGION: signal_ids, CONTROL_REGION: control_ids},
    }

    return replace(
        context,
        stage=SELECTION_STAGE,
        artifacts={"selection": selection},
        schema={
            "selection": {
                "selection_id": ATLAS_SELECTION_ID,
                "fields": [
                    "input_event_ids",
                    "selected_event_ids",
                    "selected_weight",
                    "selected_mass_gev",
                    "cutflow",
                    "regions",
                ],
            }
        },
        units={"selected_mass_gev": "GeV", "selected_weight": "dimensionless"},
        provenance={"selection": dict(_carried_provenance(context, "events"))},
    )


# --------------------------------------------------------------------------------------
# Stage 3: histogram
# --------------------------------------------------------------------------------------


def build_mass_histogram(context: ContractContext) -> ContractContext:
    """Bin the selected diphoton masses and carry the closure inputs forward."""

    selection = require_selection_artifact(context)
    settings = _histogram_settings(context)
    bin_count = int(settings["bin_count"])
    low, high = float(settings["range_low_gev"]), float(settings["range_high_gev"])
    width = (high - low) / bin_count

    masses = _float_sequence(selection, "selected_mass_gev")
    weights = _float_sequence(selection, "selected_weight")
    bin_weight_sums = [0.0] * bin_count
    bin_counts = [0] * bin_count
    underflow = 0.0
    overflow = 0.0

    for mass, weight in zip(masses, weights, strict=True):
        if mass < low:
            underflow += weight
            continue
        if mass >= high:
            overflow += weight
            continue
        index = min(int((mass - low) / width), bin_count - 1)
        bin_weight_sums[index] += weight
        bin_counts[index] += 1

    generated_weight_sum = _float_field(selection, "generated_weight_sum")
    cross_section_pb = _float_field(selection, "cross_section_pb")
    luminosity = _luminosity_pb_inverse(context)
    if generated_weight_sum == 0.0:
        raise ValueError("the selection artifact must declare a nonzero generated_weight_sum")

    histogram = {
        "observable": "diphoton_mass_gev",
        "bin_edges": tuple(low + width * index for index in range(bin_count + 1)),
        "bin_weight_sums": tuple(bin_weight_sums),
        "bin_counts": tuple(bin_counts),
        "underflow_weight_sum": underflow,
        "overflow_weight_sum": overflow,
        "selected_weight_sum": _float_field(selection, "selected_weight_sum"),
        "generated_weight_sum": generated_weight_sum,
        "cross_section_pb": cross_section_pb,
        "luminosity_pb_inverse": luminosity,
        "normalization_factor": cross_section_pb * luminosity / generated_weight_sum,
    }

    return replace(
        context,
        stage=ATLAS_HISTOGRAM_STAGE,
        artifacts={"histogram": histogram},
        schema={
            "histogram": {
                "observable": "diphoton_mass_gev",
                "fields": ["bin_edges", "bin_weight_sums", "bin_counts", "normalization_factor"],
            }
        },
        units={"bin_edges": "GeV", "bin_weight_sums": "dimensionless"},
        provenance={"histogram": dict(_carried_provenance(context, "selection"))},
    )


# --------------------------------------------------------------------------------------
# Stage 4: yield estimate
# --------------------------------------------------------------------------------------


def estimate_yield(context: ContractContext) -> ContractContext:
    """Estimate a sideband-subtracted signal yield from the declared histogram.

    This is a closed-form arithmetic estimate over binned weight sums. It is not a likelihood
    fit and involves no optimizer, background model, or uncertainty treatment.
    """

    histogram = require_histogram_artifact(context)
    edges = _float_sequence(histogram, "bin_edges")
    weight_sums = _float_sequence(histogram, "bin_weight_sums")
    if len(edges) != len(weight_sums) + 1:
        raise ValueError("histogram bin_edges must contain exactly one more entry than bin sums")

    signal_low, signal_high = _window(context, "signal_window_gev")
    centers = tuple((edges[index] + edges[index + 1]) / 2.0 for index in range(len(weight_sums)))
    in_signal = tuple(signal_low <= center < signal_high for center in centers)

    signal_weight_sum = sum(
        weight for weight, inside in zip(weight_sums, in_signal, strict=True) if inside
    )
    sideband_weight_sum = sum(
        weight for weight, inside in zip(weight_sums, in_signal, strict=True) if not inside
    )
    signal_bins = sum(in_signal)
    sideband_bins = len(weight_sums) - signal_bins
    if signal_bins == 0 or sideband_bins == 0:
        raise ValueError("the declared signal window must cover part, but not all, of the range")

    background_estimate = sideband_weight_sum * signal_bins / sideband_bins
    peak_index = max(range(len(weight_sums)), key=lambda index: weight_sums[index])
    normalization_factor = _float_field(histogram, "normalization_factor")

    estimate = {
        "observable": "diphoton_mass_gev",
        "method": "closed-form sideband subtraction over binned weight sums",
        "signal_window_gev": (signal_low, signal_high),
        "signal_weight_sum": signal_weight_sum,
        "sideband_weight_sum": sideband_weight_sum,
        "background_estimate": background_estimate,
        # The factor is carried here, not only in the histogram. Without it the final artifact
        # states a yield its own inputs do not imply, and neither a reader nor a contract can
        # check the one against the other.
        "normalization_factor": normalization_factor,
        "estimated_yield": (signal_weight_sum - background_estimate) * normalization_factor,
        "peak_bin_index": peak_index,
        "peak_bin_center_gev": centers[peak_index],
        "signal_bin_count": signal_bins,
        "sideband_bin_count": sideband_bins,
    }

    return replace(
        context,
        stage=ATLAS_YIELD_STAGE,
        artifacts={"yield_estimate": estimate},
        schema={
            "yield_estimate": {
                "observable": "diphoton_mass_gev",
                "fields": ["estimated_yield", "peak_bin_center_gev", "signal_weight_sum"],
            }
        },
        units={"peak_bin_center_gev": "GeV", "estimated_yield": "events"},
        provenance={"yield_estimate": dict(_carried_provenance(context, "histogram"))},
    )


# --------------------------------------------------------------------------------------
# Accessors reused by contracts and injectors
# --------------------------------------------------------------------------------------


def require_selection_artifact(context: ContractContext) -> Mapping[str, object]:
    artifact = context.artifacts.get("selection")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'selection' must be a mapping")
    return artifact


def require_histogram_artifact(context: ContractContext) -> Mapping[str, object]:
    artifact = context.artifacts.get("histogram")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'histogram' must be a mapping")
    return artifact


def require_yield_artifact(context: ContractContext) -> Mapping[str, object]:
    artifact = context.artifacts.get("yield_estimate")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'yield_estimate' must be a mapping")
    return artifact


def require_cutflow(selection: Mapping[str, object]) -> tuple[tuple[str, int], ...]:
    raw = selection.get("cutflow")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raise ValueError("the selection artifact must declare a non-empty cutflow")

    stages: list[tuple[str, int]] = []
    for position, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"cutflow entry {position} must be a mapping")
        cut_id = entry.get("cut_id")
        surviving = entry.get("surviving")
        if not isinstance(cut_id, str) or not cut_id.strip():
            raise ValueError(f"cutflow entry {position} must declare a non-empty cut_id")
        if isinstance(surviving, bool) or not isinstance(surviving, int):
            raise ValueError(f"cutflow entry {position} must declare an integer surviving count")
        stages.append((cut_id.strip(), surviving))
    return tuple(stages)


def require_regions(selection: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
    raw = selection.get("regions")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("the selection artifact must declare a non-empty regions mapping")

    regions: dict[str, tuple[int, ...]] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("region names must be non-empty strings")
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise ValueError(f"region {name!r} must contain a sequence of event identifiers")
        identifiers: list[int] = []
        for position, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"region {name!r} has a non-integer identifier at {position}")
            identifiers.append(value)
        regions[name.strip()] = tuple(identifiers)
    return regions


def replace_region_event_ids(
    context: ContractContext, region: str, event_ids: Sequence[int]
) -> ContractContext:
    """Return a copy whose named region carries the supplied identifiers."""

    selection = dict(require_selection_artifact(context))
    regions = require_regions(selection)
    if region not in regions:
        raise ValueError(f"the selection artifact does not declare region {region!r}")
    regions[region] = tuple(event_ids)
    selection["regions"] = regions
    artifacts = dict(context.artifacts)
    artifacts["selection"] = selection
    return replace(context, artifacts=artifacts)


def replace_bin_weight_sums(
    context: ContractContext, bin_weight_sums: Sequence[float]
) -> ContractContext:
    """Return a copy whose histogram carries the supplied per-bin weight sums."""

    histogram = dict(require_histogram_artifact(context))
    existing = _float_sequence(histogram, "bin_weight_sums")
    if len(existing) != len(bin_weight_sums):
        raise ValueError("replacement bin weight sums must preserve the declared bin count")
    histogram["bin_weight_sums"] = tuple(float(value) for value in bin_weight_sums)
    artifacts = dict(context.artifacts)
    artifacts["histogram"] = histogram
    return replace(context, artifacts=artifacts)


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _invariant_mass(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    energy = first[3] + second[3]
    px = first[0] * cos(first[2]) + second[0] * cos(second[2])
    py = first[0] * sin(first[2]) + second[0] * sin(second[2])
    pz = first[0] * sinh(first[1]) + second[0] * sinh(second[1])
    squared = energy * energy - px * px - py * py - pz * pz
    return sqrt(squared) if squared > 0.0 else 0.0


def _settings(context: ContractContext, key: str) -> Mapping[str, JsonValue]:
    config = context.config.get(key)
    if not isinstance(config, Mapping):
        raise ValueError(f"config {key!r} must be a mapping")
    return config


def _selection_settings(context: ContractContext) -> dict[str, float]:
    config = _settings(context, "atlas_selection")
    values: dict[str, float] = {}
    for name in ("leading_photon_pt_min_gev", "subleading_photon_pt_min_gev"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config 'atlas_selection.{name}' must be numeric")
        values[name] = float(value)
    if values["leading_photon_pt_min_gev"] < values["subleading_photon_pt_min_gev"]:
        raise ValueError("the leading photon threshold must not be below the subleading threshold")
    return values


def _histogram_settings(context: ContractContext) -> dict[str, float]:
    config = _settings(context, "atlas_histogram")
    bin_count = config.get("bin_count")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 2:
        raise ValueError("config 'atlas_histogram.bin_count' must be an integer of at least two")
    low = config.get("range_low_gev")
    high = config.get("range_high_gev")
    if isinstance(low, bool) or not isinstance(low, (int, float)):
        raise ValueError("config 'atlas_histogram.range_low_gev' must be numeric")
    if isinstance(high, bool) or not isinstance(high, (int, float)):
        raise ValueError("config 'atlas_histogram.range_high_gev' must be numeric")
    if float(high) <= float(low):
        raise ValueError("config 'atlas_histogram' range must be increasing")
    return {
        "bin_count": float(bin_count),
        "range_low_gev": float(low),
        "range_high_gev": float(high),
    }


def _window(context: ContractContext, name: str) -> tuple[float, float]:
    config = _settings(context, "atlas_yield")
    raw = config.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 2:
        raise ValueError(f"config 'atlas_yield.{name}' must contain exactly two bounds")
    bounds: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config 'atlas_yield.{name}' must contain numeric bounds")
        bounds.append(float(value))
    if bounds[1] <= bounds[0]:
        raise ValueError(f"config 'atlas_yield.{name}' must be increasing")
    return bounds[0], bounds[1]


def _luminosity_pb_inverse(context: ContractContext) -> float:
    config = _settings(context, "atlas_normalization")
    value = config.get("luminosity_pb_inverse")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError("config 'atlas_normalization.luminosity_pb_inverse' must be positive")
    return float(value)


def _minimum_photons(context: ContractContext) -> int:
    config = _settings(context, "atlas_gamgam")
    minimum = config.get("minimum_photons")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError("config 'atlas_gamgam.minimum_photons' must be an integer of at least two")
    return minimum


def _integer_column(columns: Mapping[str, Sequence[object]], branch: str) -> tuple[int, ...]:
    values = columns.get(branch)
    if values is None:
        raise ValueError(f"the events artifact must contain {branch!r}")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"event branch {branch!r} has a non-integer value at index {index}")
        result.append(value)
    return tuple(result)


def _float_column(columns: Mapping[str, Sequence[object]], branch: str) -> tuple[float, ...]:
    values = columns.get(branch)
    if values is None:
        raise ValueError(f"the events artifact must contain {branch!r}")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"event branch {branch!r} has a non-numeric value at index {index}")
        # Non-finite values propagate instead of raising: a numeric pipeline carries NaN
        # forward, and the finiteness contracts exist to detect exactly that.
        result.append(float(value))
    return tuple(result)


def _scalar_column(columns: Mapping[str, Sequence[object]], branch: str) -> float:
    values = _float_column(columns, branch)
    if not values:
        raise ValueError(f"event branch {branch!r} must contain at least one value")
    if len(set(values)) != 1:
        raise ValueError(f"event branch {branch!r} must be constant across the sample")
    return values[0]


def _row_column(
    columns: Mapping[str, Sequence[object]], branch: str
) -> tuple[tuple[float, ...], ...]:
    values = columns.get(branch)
    if values is None:
        raise ValueError(f"the events artifact must contain {branch!r}")
    rows: list[tuple[float, ...]] = []
    for event_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"event branch {branch!r} has a non-sequence row at {event_index}")
        converted: list[float] = []
        for position, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"event branch {branch!r} has a non-numeric value at "
                    f"event {event_index}, position {position}"
                )
            converted.append(float(value))
        rows.append(tuple(converted))
    return tuple(rows)


def _float_sequence(artifact: Mapping[str, object], field: str) -> tuple[float, ...]:
    values = artifact.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"artifact field {field!r} must be a sequence")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"artifact field {field!r} has a non-numeric value at index {index}")
        result.append(float(value))
    return tuple(result)


def _float_field(artifact: Mapping[str, object], field: str) -> float:
    value = artifact.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"artifact field {field!r} must be numeric")
    # A non-finite value propagates rather than raising, so that the contract guarding this
    # stage reports it instead of the workflow failing with a software error.
    return float(value)


def _carried_provenance(context: ContractContext, upstream: str) -> Mapping[str, JsonValue]:
    """Carry the upstream provenance forward, or nothing when the workflow has lost it.

    Analysis code does not read provenance, so a missing declaration must not raise here. It
    propagates as an absent declaration, which is precisely the silent metadata loss that a
    provenance contract at the checkpoint where it happened is meant to catch.
    """

    declaration = context.provenance.get(upstream)
    if not isinstance(declaration, Mapping):
        return {}
    return declaration


__all__ = [
    "ATLAS_HISTOGRAM_STAGE",
    "ATLAS_SELECTION_ID",
    "ATLAS_YIELD_STAGE",
    "CONTROL_REGION",
    "SIGNAL_REGION",
    "build_mass_histogram",
    "estimate_yield",
    "replace_bin_weight_sums",
    "replace_region_event_ids",
    "require_cutflow",
    "require_histogram_artifact",
    "require_regions",
    "require_selection_artifact",
    "require_yield_artifact",
    "select_diphoton_events",
]
