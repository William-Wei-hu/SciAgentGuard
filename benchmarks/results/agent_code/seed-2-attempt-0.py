import json
import math
import numpy as np
import uproot

def run_analysis():
    # Open the ROOT file and get the mini tree
    file = uproot.open(INPUT_PATH)
    tree = file["mini"]

    # Define branches to read
    branches = [
        "eventNumber",
        "mcWeight",
        "photon_n",
        "photon_pt",
        "photon_eta",
        "photon_phi",
        "photon_E",
        "XSection",
        "SumWeights"
    ]

    data = tree.arrays(branches, library="np")

    event_ids = data["eventNumber"].tolist()
    mc_weights = data["mcWeight"]
    photon_n = data["photon_n"]
    photon_pt = data["photon_pt"]
    photon_eta = data["photon_eta"]
    photon_phi = data["photon_phi"]
    photon_E = data["photon_E"]

    # Per-file constants: read once (take the first entry since they are constant per file)
    cross_section_pb = float(data["XSection"][0]) if len(data["XSection"]) > 0 else 0.0
    generated_weight_sum = float(data["SumWeights"][0]) if len(data["SumWeights"]) > 0 else 1.0

    luminosity_pb_inverse = 10064.0
    normalization_factor = (cross_section_pb * luminosity_pb_inverse) / generated_weight_sum if generated_weight_sum != 0 else 0.0

    # Cutflow tracking
    total_events = len(event_ids)
    cutflow = [
        {"cut_id": "total_events", "surviving": total_events},
        {"cut_id": "photon_n >= 2", "surviving": 0},
        {"cut_id": "pt1 > 40 and pt2 > 30", "surviving": 0}
    ]

    selected_event_ids = []
    selected_weight = []
    selected_mass_gev = []

    # Selections and processing
    surviving_n2 = 0
    surviving_pts = 0

    for i in range(total_events):
        n_photons = photon_n[i]
        if n_photons < 2:
            continue
        surviving_n2 += 1

        pts = photon_pt[i]
        etas = photon_eta[i]
        phis = photon_phi[i]
        Es = photon_E[i]

        # Sort photons by pT descending to find two leading
        # Since arrays might be jagged/variable length, let's sort indices by pt
        sorted_indices = np.argsort(pts)[::-1]
        
        if len(sorted_indices) < 2:
            continue

        lead_idx = sorted_indices[0]
        sublead_idx = sorted_indices[1]

        # Convert from MeV to GeV
        pt1 = pts[lead_idx] / 1000.0
        pt2 = sublead_idx_pt = pts[sublead_idx] / 1000.0

        if pt1 <= 40.0 or pt2 <= 30.0:
            continue
        surviving_pts += 1

        # Compute diphoton invariant mass
        # Four-vector components: px, py, pz, E (in GeV)
        e1 = Es[lead_idx] / 1000.0
        eta1 = etas[lead_idx]
        phi1 = phis[lead_idx]
        
        e2 = Es[sublead_idx] / 1000.0
        eta2 = etas[sublead_idx]
        phi2 = phis[sublead_idx]

        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)

        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)

        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        E_sum = e1 + e2

        mass_sq = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
        mass = math.sqrt(max(0.0, mass_sq))

        selected_event_ids.append(int(event_ids[i]))
        selected_weight.append(float(mc_weights[i]))
        selected_mass_gev.append(float(mass))

    cutflow[1]["surviving"] = surviving_n2
    cutflow[2]["surviving"] = surviving_pts

    selected_weight_sum = float(np.sum(selected_weight)) if selected_weight else 0.0

    # Regions: signal (120-130 GeV inclusive or exclusive? standard is [120, 130])
    # Prompt says: "contains exactly those whose mass lies in 120-130 GeV, and a control region containing exactly those whose mass lies outside that window"
    signal_ids = []
    control_ids = []

    for evid, m in zip(selected_event_ids, selected_mass_gev):
        if 120.0 <= m <= 130.0:
            signal_ids.append(evid)
        else:
            control_ids.append(evid)

    regions = {
        "signal": signal_ids,
        "control": control_ids
    }

    # Histogram: 30 bins over 100-160 GeV
    obs = "diphoton_invariant_mass_gev"
    bin_edges = np.linspace(100.0, 160.0, 31).tolist()
    
    bin_weight_sums = [0.0] * 30
    bin_counts = [0] * 30
    underflow_weight_sum = 0.0
    overflow_weight_sum = 0.0

    for m, w in zip(selected_mass_gev, selected_weight):
        if m < 100.0:
            underflow_weight_sum += w
        elif m > 160.0:
            overflow_weight_sum += w
        else:
            # find bin
            # np.digitize gives bins from 1 to len(bin_edges)-1
            b_idx = np.digitize([m], bin_edges)[0] - 1
            if b_idx < 0:
                underflow_weight_sum += w
            elif b_idx >= 30:
                overflow_weight_sum += w
            else:
                bin_weight_sums[b_idx] += w
                bin_counts[b_idx] += 1

    # Yield estimate via sideband subtraction over binned weight sums
    # Signal window: 120-130 GeV. Bins covering [120, 130].
    # Let's find which bins fall fully or partially in [120, 130], or count events directly inside mass window.
    # The prompt asks for:
    # "signal_weight_sum": RAW weight inside the signal window
    # "sideband_weight_sum": ...
    # "background_estimate": ...
    # "estimated_yield": ...
    # "peak_bin_index": int, "peak_bin_center_gev": float,
    # "signal_bin_count": int, "sideband_bin_count": int

    signal_weight_sum = 0.0
    sideband_weight_sum = 0.0
    signal_bin_count = 0
    sideband_bin_count = 0

    # Determine peak bin
    peak_bin_index = int(np.argmax(bin_weight_sums)) if bin_weight_sums else 0
    peak_bin_center_gev = float((bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2.0)

    for i in range(30):
        b_low = bin_edges[i]
        b_high = bin_edges[i+1]
        b_center = (b_low + b_high) / 2.0
        # Check if bin is in signal window [120, 130]
        if b_low >= 120.0 and b_high <= 130.0:
            signal_weight_sum += bin_weight_sums[i]
            signal_bin_count += 1
        else:
            sideband_weight_sum += bin_weight_sums[i]
            sideband_bin_count += 1

    # Sideband background estimation:
    # Typically background estimate = (width of signal window / width of sideband) * sideband_weight_sum
    signal_width = 10.0 # 130 - 120
    total_range_width = 60.0 # 160 - 100
    sideband_width = total_range_width - signal_width
    background_estimate = sideband_weight_sum * (signal_width / sideband_width) if sideband_width > 0 else 0.0

    estimated_yield = (signal_weight_sum - background_estimate) * normalization_factor

    output_data = {
      "selection": {
        "selection_id": "diphoton_baseline_v1",
        "input_event_ids": [int(x) for x in event_ids],
        "selected_event_ids": selected_event_ids,
        "selected_weight": selected_weight,
        "selected_mass_gev": selected_mass_gev,
        "selected_weight_sum": selected_weight_sum,
        "generated_weight_sum": generated_weight_sum,
        "cross_section_pb": cross_section_pb,
        "cutflow": cutflow,
        "regions": regions
      },
      "histogram": {
        "observable": obs,
        "bin_edges": bin_edges,
        "bin_weight_sums": bin_weight_sums,
        "bin_counts": bin_counts,
        "underflow_weight_sum": underflow_weight_sum,
        "overflow_weight_sum": overflow_weight_sum,
        "selected_weight_sum": selected_weight_sum,
        "generated_weight_sum": generated_weight_sum,
        "cross_section_pb": cross_section_pb,
        "luminosity_pb_inverse": luminosity_pb_inverse,
        "normalization_factor": normalization_factor
      },
      "yield_estimate": {
        "observable": obs,
        "method": "sideband_subtraction",
        "signal_window_gev": [120.0, 130.0],
        "signal_weight_sum": signal_weight_sum,
        "sideband_weight_sum": sideband_weight_sum,
        "background_estimate": background_estimate,
        "estimated_yield": estimated_yield,
        "peak_bin_index": peak_bin_index,
        "peak_bin_center_gev": peak_bin_center_gev,
        "signal_bin_count": signal_bin_count,
        "sideband_bin_count": sideband_bin_count
      }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f)

if __name__ == "__main__":
    run_analysis()