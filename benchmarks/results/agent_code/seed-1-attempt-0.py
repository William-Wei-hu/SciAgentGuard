import json
import math
import numpy as np
import uproot

def compute_invariant_mass(pt1, eta1, phi1, E1, pt2, eta2, phi2, E2):
    # Convert to px, py, pz
    px1 = pt1 * math.cos(phi1)
    py1 = pt1 * math.sin(phi1)
    pz1 = pt1 * math.sinh(eta1)
    
    px2 = pt2 * math.cos(phi2)
    py2 = pt2 * math.sin(phi2)
    pz2 = pt2 * math.sinh(eta2)
    
    E_sum = E1 + E2
    px_sum = px1 + px2
    py_sum = py1 + py2
    pz_sum = pz1 + pz2
    
    m2 = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
    return math.sqrt(max(0.0, m2))

def run_analysis():
    luminosity = 10064.0
    
    file = uproot.open(INPUT_PATH)
    tree = file["mini"]
    
    branches = [
        "eventNumber", "mcWeight", "photon_n",
        "photon_pt", "photon_eta", "photon_phi", "photon_E",
        "XSection", "SumWeights"
    ]
    
    data = tree.arrays(branches, library="np")
    
    event_ids = data["eventNumber"].tolist()
    mc_weights = data["mcWeight"]
    photon_n = data["photon_n"]
    photon_pt = data["photon_pt"]
    photon_eta = data["photon_eta"]
    photon_phi = data["photon_phi"]
    photon_E = data["photon_E"]
    
    # Constants from file (read once, take first element)
    x_section = float(data["XSection"][0]) if len(data["XSection"]) > 0 else 0.0
    sum_weights = float(data["SumWeights"][0]) if len(data["SumWeights"]) > 0 else 1.0
    
    input_event_ids = [int(x) for x in event_ids]
    
    # Cutflow tracking
    total_events = len(event_ids)
    cut1_surviving = 0
    cut2_surviving = 0
    
    selected_event_ids = []
    selected_weight = []
    selected_mass_gev = []
    
    signal_event_ids = []
    control_event_ids = []
    
    for i in range(total_events):
        n_ph = photon_n[i]
        if n_ph < 2:
            continue
        cut1_surviving += 1
        
        pts = photon_pt[i]
        if len(pts) < 2:
            continue
            
        # Find two leading photons
        # Note: photon arrays in ATLAS open data are typically sorted by pT, but let's be safe and argsort
        sorted_indices = np.argsort(pts)[::-1]
        lead_idx = sorted_indices[0]
        sublead_idx = sorted_indices[1]
        
        pt1 = pts[lead_idx] / 1000.0  # MeV to GeV
        pt2 = pts[sublead_idx] / 1000.0
        
        if pt1 <= 40.0 or pt2 <= 30.0:
            continue
        cut2_surviving += 1
        
        eta1 = photon_eta[i][lead_idx]
        phi1 = photon_phi[i][lead_idx]
        E1 = photon_E[i][lead_idx] / 1000.0
        
        eta2 = photon_eta[i][sublead_idx]
        phi2 = photon_phi[i][sublead_idx]
        E2 = photon_E[i][sublead_idx] / 1000.0
        
        mass = compute_invariant_mass(pt1, eta1, phi1, E1, pt2, eta2, phi2, E2)
        
        ev_id = int(event_ids[i])
        wt = float(mc_weights[i])
        
        selected_event_ids.append(ev_id)
        selected_weight.append(wt)
        selected_mass_gev.append(mass)
        
        if 120.0 <= mass <= 130.0:
            signal_event_ids.append(ev_id)
        else:
            control_event_ids.append(ev_id)
            
    selected_weight_sum = float(np.sum(selected_weight)) if selected_weight else 0.0
    
    cutflow = [
        {"cut_id": "total_events", "surviving": total_events},
        {"cut_id": "min_two_photons", "surviving": cut1_surviving},
        {"cut_id": "photon_pt_cuts", "surviving": cut2_surviving}
    ]
    
    # Histogram definition: 30 bins over 100-160 GeV
    bin_edges = np.linspace(100.0, 160.0, 31).tolist()
    bin_counts = [0] * 30
    bin_weight_sums = [0.0] * 30
    underflow_weight_sum = 0.0
    overflow_weight_sum = 0.0
    
    for mass, wt in zip(selected_mass_gev, selected_weight):
        if mass < 100.0:
            underflow_weight_sum += wt
        elif mass > 160.0:
            overflow_weight_sum += wt
        else:
            # Find bin
            bin_idx = int((mass - 100.0) / (60.0 / 30.0))
            if bin_idx >= 30:
                bin_idx = 29
            if bin_idx < 0:
                bin_idx = 0
            bin_counts[bin_idx] += 1
            bin_weight_sums[bin_idx] += wt
            
    normalization_factor = (x_section * luminosity) / sum_weights if sum_weights != 0 else 0.0
    
    # Yield estimate via sideband subtraction
    # Signal window: 120 - 130 GeV
    # Sidebands: 100 - 120 GeV and 130 - 160 GeV
    signal_weight_sum = 0.0
    sideband_weight_sum = 0.0
    signal_bin_count = 0
    sideband_bin_count = 0
    
    peak_bin_index = -1
    max_bin_weight = -1.0
    
    for idx, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        center = (low + high) / 2.0
        w_sum = bin_weight_sums[idx]
        c_count = bin_counts[idx]
        
        if w_sum > max_bin_weight:
            max_bin_weight = w_sum
            peak_bin_index = idx
            
        if 120.0 <= low and high <= 130.0:
            signal_weight_sum += w_sum
            signal_bin_count += c_count
        elif (low >= 100.0 and high <= 120.0) or (low >= 130.0 and high <= 160.0):
            sideband_weight_sum += w_sum
            sideband_bin_count += c_count
            
    peak_bin_center_gev = float((bin_edges[peak_bin_index] + bin_edges[peak_bin_index+1]) / 2.0) if peak_bin_index >= 0 else 0.0
    
    # Sideband background estimation:
    # sideband width = (120 - 100) + (160 - 130) = 20 + 30 = 50 GeV
    # signal window width = 130 - 120 = 10 GeV
    # background estimate = sideband_weight_sum * (signal_width / sideband_width)
    background_estimate = sideband_weight_sum * (10.0 / 50.0)
    estimated_yield = (signal_weight_sum - background_estimate) * normalization_factor
    
    output_data = {
      "selection": {
        "selection_id": "diphoton_baseline",
        "input_event_ids": input_event_ids,
        "selected_event_ids": selected_event_ids,
        "selected_weight": selected_weight,
        "selected_mass_gev": selected_mass_gev,
        "selected_weight_sum": selected_weight_sum,
        "generated_weight_sum": sum_weights,
        "cross_section_pb": x_section,
        "cutflow": cutflow,
        "regions": {
            "signal": signal_event_ids,
            "control": control_event_ids
        }
      },
      "histogram": {
        "observable": "diphoton_invariant_mass_gev",
        "bin_edges": bin_edges,
        "bin_weight_sums": bin_weight_sums,
        "bin_counts": bin_counts,
        "underflow_weight_sum": underflow_weight_sum,
        "overflow_weight_sum": overflow_weight_sum,
        "selected_weight_sum": selected_weight_sum,
        "generated_weight_sum": sum_weights,
        "cross_section_pb": x_section,
        "luminosity_pb_inverse": luminosity,
        "normalization_factor": normalization_factor
      },
      "yield_estimate": {
        "observable": "diphoton_invariant_mass_gev",
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