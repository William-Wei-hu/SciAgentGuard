import zlib
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from sciagentguard.adapters import AtlasGamGamSource

# Each synthetic event holds an equal-momentum photon pair at eta = 0, opened by
# OPENING_ANGLE rather than placed exactly back to back. The opening matters: photons
# exactly pi apart have transverse momenta that cancel in the vector sum, so the invariant
# mass stops depending on their magnitude and a fault that rescales photon momenta becomes
# invisible to this fixture. For eta = 0 and an equal pair, m = pt * sqrt(2 - 2 cos dphi), so the
# momentum carrying a given mass is that mass divided by _MASS_PER_PT. The sample is a
# deterministic test fixture; it is not simulated detector data.
#
# The sample is built from three identical blocks so that any contiguous third of the file is
# itself a valid analysis input: each block peaks at 125 GeV, populates both sidebands, and
# contains two events that fail a momentum threshold. Distinct ranges of one file are distinct
# valid inputs of the same source, not independent samples.
_BLOCK_COUNT = 3
_SIGNAL_MASS_GEV = 125.0
_SIGNAL_EVENTS_PER_BLOCK = 5
_SIDEBAND_MASSES_GEV = (105.0, 111.0, 139.0)
_HALF_WEIGHT_MASS_GEV = 111.0
_THIRD_PHOTON_PT_GEV = 20.0
_MEV_PER_GEV = 1000.0
OPENING_ANGLE = 3.0 * np.pi / 4.0
MASS_PER_PT = float(np.sqrt(2.0 - 2.0 * np.cos(OPENING_ANGLE)))
_MASS_PER_PT = MASS_PER_PT

BLOCK_SIZE = _SIGNAL_EVENTS_PER_BLOCK + len(_SIDEBAND_MASSES_GEV) + 2
EVENT_COUNT = BLOCK_SIZE * _BLOCK_COUNT


def adler32(path: Path) -> str:
    checksum = 1
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum = zlib.adler32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def _pair(mass_gev: float) -> tuple[list[float], list[float], list[float]]:
    """Return the equal-momentum photon pair carrying the requested invariant mass."""

    momentum = mass_gev / _MASS_PER_PT
    return [momentum, momentum], [0.0, 0.0], [0.0, OPENING_ANGLE]


def _blocks() -> tuple[
    list[list[float]], list[list[float]], list[list[float]], list[int], list[float]
]:
    pt_rows: list[list[float]] = []
    eta_rows: list[list[float]] = []
    phi_rows: list[list[float]] = []
    counts: list[int] = []
    weights: list[float] = []

    for block in range(_BLOCK_COUNT):
        for position in range(_SIGNAL_EVENTS_PER_BLOCK):
            momenta, etas, phis = _pair(_SIGNAL_MASS_GEV)
            if block == 0 and position == 0:
                # One event carries a third, softer photon so that the leading-pair ordering
                # is exercised rather than assumed.
                momenta.append(_THIRD_PHOTON_PT_GEV)
                etas.append(0.0)
                phis.append(0.5)
            pt_rows.append(momenta)
            eta_rows.append(etas)
            phi_rows.append(phis)
            counts.append(len(momenta))
            weights.append(1.0)

        for mass in _SIDEBAND_MASSES_GEV:
            momenta, etas, phis = _pair(mass)
            pt_rows.append(momenta)
            eta_rows.append(etas)
            phi_rows.append(phis)
            counts.append(len(momenta))
            weights.append(0.5 if mass == _HALF_WEIGHT_MASS_GEV else 1.0)

        # Every event of the declared Gamma-Gamma sample carries at least two photons, so the
        # two cut events below fail a momentum threshold rather than the diphoton preselection.
        for leading, subleading in ((20.0, 15.0), (50.0, 20.0)):
            pt_rows.append([leading, subleading])
            eta_rows.append([0.0, 0.0])
            phi_rows.append([0.0, OPENING_ANGLE])
            counts.append(2)
            weights.append(1.0)

    scaled = [[value * _MEV_PER_GEV for value in row] for row in pt_rows]
    return scaled, eta_rows, phi_rows, counts, weights


def write_root_file(
    path: Path,
    *,
    tree_name: str = "mini",
    omit_branch: str | None = None,
) -> None:
    pt_rows, eta_rows, phi_rows, counts, weights = _blocks()
    # Photons are massless and sit at eta = 0, so the energy equals the transverse momentum.
    energy_rows = [list(row) for row in pt_rows]
    event_count = len(counts)

    branches: dict[str, object] = {
        "runNumber": np.full(event_count, 11, dtype=np.int32),
        "eventNumber": np.arange(101, 101 + event_count, dtype=np.int32),
        "channelNumber": np.full(event_count, 345318, dtype=np.int32),
        "mcWeight": np.array(weights, dtype=np.float64),
        "photon_n": np.array(counts, dtype=np.uint32),
        "photon_pt": ak.Array(pt_rows),
        "photon_eta": ak.Array(eta_rows),
        "photon_phi": ak.Array(phi_rows),
        "photon_E": ak.Array(energy_rows),
        "XSection": np.full(event_count, 0.002, dtype=np.float64),
        "SumWeights": np.full(event_count, 1000.0, dtype=np.float64),
    }
    if omit_branch is not None:
        del branches[omit_branch]
    with uproot.recreate(path) as root_file:
        root_file[tree_name] = branches


def synthetic_source(path: Path) -> AtlasGamGamSource:
    return AtlasGamGamSource(
        path=path,
        size_bytes=path.stat().st_size,
        adler32=adler32(path),
        source_type="synthetic",
        record_id="test-atlas-gamgam",
        doi="test-doi",
        file_name="test.GamGam.root",
        generator="tests.integration._atlas_root",
    )
