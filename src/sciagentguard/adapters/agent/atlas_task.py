"""The ATLAS diphoton task given to an agent, and the bridge back to guarded checkpoints.

The trusted adapter keeps the input boundary: it verifies the source and produces the `post_load`
context, including provenance the agent never touches. The agent owns the analysis, and its three
downstream artifacts are what the guard inspects. Configuration and units come from the trusted
context, so every contract still reads its own declared assumptions rather than the agent's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from pydantic import JsonValue

from sciagentguard.adapters.agent.models import AgentTask
from sciagentguard.core import ContractContext
from sciagentguard.packs.hep.atlas_analysis import (
    ATLAS_HISTOGRAM_STAGE,
    ATLAS_YIELD_STAGE,
)
from sciagentguard.packs.hep.fixtures import SELECTION_STAGE

ATLAS_AGENT_TASK = AgentTask(
    task_id="atlas_gamgam_diphoton_yield",
    description=(
        "Read the ATLAS Open Data Gamma-Gamma `mini` tree at INPUT_PATH and produce a diphoton "
        "analysis. Select events with at least two photons, a leading photon above 40 GeV and a "
        "subleading photon above 30 GeV. Compute the diphoton invariant mass from the two leading "
        "photons' four-vectors, converting momenta and energies from MeV to GeV. Histogram the "
        "mass in 30 bins over 100-160 GeV. Split the selected events by mass into a signal "
        "region containing exactly those whose mass lies in 120-130 GeV, and a control region "
        "containing exactly those whose mass lies outside that window, so that the two regions "
        "together contain each selected event exactly once. Normalize with "
        "cross_section_pb * luminosity_pb_inverse / generated_weight_sum, taking the cross section "
        "from XSection and the generated weight sum from SumWeights -- both are per-file constants "
        "repeated on every event, so read one value and do not sum them across events -- and an "
        "integrated luminosity of 10064 pb^-1. Report event weights as the RAW mcWeight values "
        "with no normalization applied to them; the normalization factor is reported on its own "
        "and applied only when computing the estimated yield. Estimate the signal yield by "
        "sideband subtraction over the binned weight sums. Write one JSON object to OUTPUT_PATH."
    ),
    input_description=(
        "A verified ATLAS Open Data Gamma-Gamma ROOT file containing the `mini` tree with the "
        "branches eventNumber, mcWeight, photon_n, photon_pt, photon_eta, photon_phi, photon_E, "
        "XSection, and SumWeights."
    ),
    expected_outputs=("selection", "histogram", "yield_estimate"),
)

_ARTIFACT_STAGES: tuple[tuple[str, str], ...] = (
    ("selection", SELECTION_STAGE),
    ("histogram", ATLAS_HISTOGRAM_STAGE),
    ("yield_estimate", ATLAS_YIELD_STAGE),
)


def agent_contexts(
    loaded: ContractContext,
    artifacts: Mapping[str, JsonValue],
) -> tuple[ContractContext, ...]:
    """Turn an agent's declared artifacts into the three downstream checkpoint contexts.

    Raises `ValueError` when an expected artifact is missing or is not an object. That is an
    output-contract failure of the agent, not a scientific fault, and callers must record it as
    such rather than counting it among the contract detections.
    """

    contexts: list[ContractContext] = []
    for name, stage in _ARTIFACT_STAGES:
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"the agent did not produce a {name!r} object")
        contexts.append(
            replace(
                loaded,
                stage=stage,
                artifacts={name: dict(artifact)},
                provenance={name: _agent_provenance(loaded)},
            )
        )
    return tuple(contexts)


def _agent_provenance(loaded: ContractContext) -> dict[str, JsonValue]:
    declaration = loaded.provenance.get("events")
    carried: dict[str, JsonValue] = {"produced_by": "analysis-agent"}
    if isinstance(declaration, Mapping):
        for key, value in declaration.items():
            if isinstance(key, str):
                carried[key] = value
    return carried


__all__ = ["ATLAS_AGENT_TASK", "agent_contexts"]
