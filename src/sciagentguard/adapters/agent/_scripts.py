"""Analysis scripts served by the deterministic scripted agent.

Every variant is built from one template so that a faulty variant differs from the correct one by
a small, plausible edit rather than by obvious sabotage. The faults are the kind of slip a
competent author actually makes: a luminosity quoted in the wrong unit, an inclusive bound on both
sides of a region boundary, a cut applied to the wrong quantity, a cutflow recorded before a later
cut runs.

The scripts run inside `CodeSandbox`, so they receive `INPUT_PATH` and `OUTPUT_PATH` as globals and
communicate only by writing one JSON object.
"""

from __future__ import annotations

_TEMPLATE = """
import json
import math

import uproot

LEADING_PT_MIN_GEV = 40.0
SUBLEADING_PT_MIN_GEV = 30.0
BIN_COUNT = 30
RANGE_LOW_GEV = 100.0
RANGE_HIGH_GEV = 160.0
SIGNAL_LOW_GEV = 120.0
SIGNAL_HIGH_GEV = 130.0
MEV_TO_GEV = 1e-3
{constants}

BRANCHES = [
    "eventNumber",
    "mcWeight",
    "photon_n",
    "photon_pt",
    "photon_eta",
    "photon_phi",
    "photon_E",
    "XSection",
    "SumWeights",
]


def invariant_mass(first, second):
    energy = first[3] + second[3]
    px = first[0] * math.cos(first[2]) + second[0] * math.cos(second[2])
    py = first[0] * math.sin(first[2]) + second[0] * math.sin(second[2])
    pz = first[0] * math.sinh(first[1]) + second[0] * math.sinh(second[1])
    squared = energy * energy - px * px - py * py - pz * pz
    return math.sqrt(squared) if squared > 0.0 else 0.0


with uproot.open(INPUT_PATH) as handle:
    tree = handle["mini"]
    arrays = tree.arrays(BRANCHES, library="np")

event_ids = [int(value) for value in arrays["eventNumber"]]
weights = [float(value) for value in arrays["mcWeight"]]
counts = [int(value) for value in arrays["photon_n"]]
pt_rows = [[float(v) * MEV_TO_GEV for v in row] for row in arrays["photon_pt"]]
eta_rows = [[float(v) for v in row] for row in arrays["photon_eta"]]
phi_rows = [[float(v) for v in row] for row in arrays["photon_phi"]]
energy_rows = [[float(v) * MEV_TO_GEV for v in row] for row in arrays["photon_E"]]
cross_section_pb = float(arrays["XSection"][0])
generated_weight_sum = float(arrays["SumWeights"][0])

surviving = [len(event_ids), 0, 0, 0]
selected_ids = []
selected_weights = []
selected_masses = []

for index, event_id in enumerate(event_ids):
    momenta = pt_rows[index]
    if counts[index] < 2 or len(momenta) < 2:
        continue
    surviving[1] += 1

    order = sorted(range(len(momenta)), key=lambda position: momenta[position], reverse=True)
    leading, subleading = order[0], order[1]
{selection_body}

    mass = invariant_mass(
        (momenta[leading], eta_rows[index][leading], phi_rows[index][leading],
         energy_rows[index][leading]),
        (momenta[subleading], eta_rows[index][subleading], phi_rows[index][subleading],
         energy_rows[index][subleading]),
    )
    selected_ids.append(event_id)
    selected_weights.append(weights[index])
    selected_masses.append(mass)

{cutflow_body}

signal_ids = []
control_ids = []
for event_id, mass in zip(selected_ids, selected_masses):
{region_body}

width = (RANGE_HIGH_GEV - RANGE_LOW_GEV) / BIN_COUNT
bin_weight_sums = [0.0] * BIN_COUNT
bin_counts = [0] * BIN_COUNT
underflow = 0.0
overflow = 0.0
for mass, weight in zip(selected_masses, selected_weights):
    if mass < RANGE_LOW_GEV:
        underflow += weight
        continue
    if mass >= RANGE_HIGH_GEV:
        overflow += weight
        continue
    position = min(int((mass - RANGE_LOW_GEV) / width), BIN_COUNT - 1)
    bin_weight_sums[position] += weight
    bin_counts[position] += 1

{normalization_body}

centers = [(RANGE_LOW_GEV + width * (i + 0.5)) for i in range(BIN_COUNT)]
in_signal = [SIGNAL_LOW_GEV <= center < SIGNAL_HIGH_GEV for center in centers]
signal_weight_sum = sum(w for w, inside in zip(bin_weight_sums, in_signal) if inside)
sideband_weight_sum = sum(w for w, inside in zip(bin_weight_sums, in_signal) if not inside)
signal_bins = sum(in_signal)
sideband_bins = BIN_COUNT - signal_bins
background_estimate = sideband_weight_sum * signal_bins / sideband_bins
peak_index = max(range(BIN_COUNT), key=lambda i: bin_weight_sums[i])

artifacts = {{
    "selection": {{
        "selection_id": "atlas_gamgam_diphoton",
        "input_event_ids": event_ids,
        "selected_event_ids": selected_ids,
        "selected_weight": selected_weights,
        "selected_mass_gev": selected_masses,
        "selected_weight_sum": sum(selected_weights),
        "generated_weight_sum": generated_weight_sum,
        "cross_section_pb": cross_section_pb,
        "cutflow": [
            {{"cut_id": "all_events", "surviving": surviving[0]}},
            {{"cut_id": "two_photons", "surviving": surviving[1]}},
            {{"cut_id": "leading_photon_pt", "surviving": surviving[2]}},
            {{"cut_id": "subleading_photon_pt", "surviving": surviving[3]}},
        ],
        "regions": {{"signal": signal_ids, "control": control_ids}},
    }},
    "histogram": {{
        "observable": "diphoton_mass_gev",
        "bin_edges": [RANGE_LOW_GEV + width * i for i in range(BIN_COUNT + 1)],
        "bin_weight_sums": bin_weight_sums,
        "bin_counts": bin_counts,
        "underflow_weight_sum": underflow,
        "overflow_weight_sum": overflow,
        "selected_weight_sum": sum(selected_weights),
        "generated_weight_sum": generated_weight_sum,
        "cross_section_pb": cross_section_pb,
        "luminosity_pb_inverse": LUMINOSITY_PB_INVERSE,
        "normalization_factor": normalization_factor,
    }},
    "yield_estimate": {{
        "observable": "diphoton_mass_gev",
        "method": "closed-form sideband subtraction over binned weight sums",
        "signal_window_gev": [SIGNAL_LOW_GEV, SIGNAL_HIGH_GEV],
        "signal_weight_sum": signal_weight_sum,
        "sideband_weight_sum": sideband_weight_sum,
        "background_estimate": background_estimate,
        "normalization_factor": normalization_factor,
        "estimated_yield": (signal_weight_sum - background_estimate) * normalization_factor,
        "peak_bin_index": peak_index,
        "peak_bin_center_gev": centers[peak_index],
        "signal_bin_count": signal_bins,
        "sideband_bin_count": sideband_bins,
    }},
}}

with open(OUTPUT_PATH, "w") as stream:
    json.dump(artifacts, stream)
"""

_CORRECT_SELECTION = """    if momenta[leading] < LEADING_PT_MIN_GEV:
        continue
    surviving[2] += 1
    if momenta[subleading] < SUBLEADING_PT_MIN_GEV:
        continue
    surviving[3] += 1"""

_CORRECT_CUTFLOW = ""

_CORRECT_REGIONS = """    if SIGNAL_LOW_GEV <= mass < SIGNAL_HIGH_GEV:
        signal_ids.append(event_id)
    else:
        control_ids.append(event_id)"""

_CORRECT_NORMALIZATION = (
    "normalization_factor = cross_section_pb * LUMINOSITY_PB_INVERSE / generated_weight_sum"
)

_CORRECT_CONSTANTS = "LUMINOSITY_PB_INVERSE = 10064.0"


def _script(
    *,
    constants: str = _CORRECT_CONSTANTS,
    selection_body: str = _CORRECT_SELECTION,
    cutflow_body: str = _CORRECT_CUTFLOW,
    region_body: str = _CORRECT_REGIONS,
    normalization_body: str = _CORRECT_NORMALIZATION,
) -> str:
    return _TEMPLATE.format(
        constants=constants,
        selection_body=selection_body,
        cutflow_body=cutflow_body,
        region_body=region_body,
        normalization_body=normalization_body,
    )


SCRIPTS: dict[str, str] = {
    "correct": _script(),
    # A luminosity quoted in fb^-1 where the formula expects pb^-1: the classic unit slip. Every
    # downstream shape is untouched, so only the histogram closure relation can see it.
    "luminosity_unit_slip": _script(
        constants="LUMINOSITY_PB_INVERSE = 10064.0\nQUOTED_LUMINOSITY_FB_INVERSE = 10.064",
        normalization_body=(
            "normalization_factor = ("
            "cross_section_pb * QUOTED_LUMINOSITY_FB_INVERSE / generated_weight_sum)"
        ),
    ),
    # The lower control sideband was typed with the signal window's upper bound, so every event
    # in 120-130 GeV lands in the signal and control regions at once. Nothing downstream carries
    # region membership, so nothing downstream can see it.
    "overlapping_control_window": _script(
        constants=(
            "LUMINOSITY_PB_INVERSE = 10064.0\nCONTROL_LOW_GEV = 100.0\nCONTROL_HIGH_GEV = 130.0"
        ),
        region_body="""    if SIGNAL_LOW_GEV <= mass < SIGNAL_HIGH_GEV:
        signal_ids.append(event_id)
    if CONTROL_LOW_GEV <= mass < CONTROL_HIGH_GEV:
        control_ids.append(event_id)""",
    ),
    # The cutflow is recorded from the pre-cut counters, so it claims more survivors than the
    # selection actually kept.
    "stale_cutflow": _script(
        selection_body="""    if momenta[leading] < LEADING_PT_MIN_GEV:
        continue
    surviving[2] += 1
    if momenta[subleading] < SUBLEADING_PT_MIN_GEV:
        continue""",
        cutflow_body="surviving[3] = surviving[2]",
    ),
    # The subleading threshold is applied to the leading photon, so the selection keeps nothing
    # once the thresholds are ordered the usual way.
    "empty_selection": _script(
        selection_body="""    if momenta[leading] < LEADING_PT_MIN_GEV:
        continue
    surviving[2] += 1
    if momenta[leading] < 10_000.0:
        continue
    surviving[3] += 1"""
    ),
    # Code that does not run at all. A failure to produce runnable code is counted separately
    # from a semantic failure.
    "unrunnable": "import uproot\nthis is not valid python\n",
    # Code that runs fine and writes valid JSON under the wrong keys. This is the most likely
    # non-semantic failure of a real model: it did the work, then reported it in its own shape.
    # It is neither a crash nor a scientific fault, and is counted as its own kind.
    "wrong_output_schema": _script()
    .replace('    "yield_estimate": {', '    "yield": {')
    .replace('    "histogram": {', '    "hist": {'),
}

CORRECT_SCRIPT_ID = "correct"
FAULTY_SCRIPT_IDS = tuple(script_id for script_id in SCRIPTS if script_id != CORRECT_SCRIPT_ID)

__all__ = ["CORRECT_SCRIPT_ID", "FAULTY_SCRIPT_IDS", "SCRIPTS"]
