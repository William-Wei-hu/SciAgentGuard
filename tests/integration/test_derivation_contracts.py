"""The contracts that check derivation must not reject correct work.

These three checks compare an analysis's numbers against the source they claim to come from, which
is a stronger claim than internal consistency and therefore a larger chance of a false positive. A
guard that blocks a correct analysis is worse than no guard, so every way of being right that this
repository knows about is asserted here.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.adapters.agent import (
    ATLAS_AGENT_TASK,
    CodeSandbox,
    ScriptedAgent,
    agent_contexts,
)
from sciagentguard.adapters.agent._scripts import CORRECT_SCRIPT_ID
from sciagentguard.core import ContractContext
from sciagentguard.packs.hep.atlas_open_data import (
    AtlasBackgroundEstimateContract,
    AtlasNormalizationProvenanceContract,
    AtlasRegionCoverageContract,
    AtlasRegionDefinitionContract,
    AtlasRegionDisjointContract,
    AtlasSourceConstantsContract,
    AtlasWeightProvenanceContract,
    AtlasYieldClosureContract,
)
from sciagentguard.runtime import GuardedWorkflowRunner
from tests.integration._atlas_root import BLOCK_SIZE, synthetic_source, write_root_file

NEW_CONTRACT_IDS = {
    "hep.atlas_open_data.weight_provenance",
    "hep.atlas_open_data.region_coverage",
    "hep.atlas_open_data.source_constants",
}


@pytest.fixture(scope="module")
def source_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("derivation") / "test.root"
    write_root_file(path)
    return path


@pytest.fixture(scope="module")
def adapter(source_path: Path) -> AtlasGamGamOpenDataAdapter:
    return AtlasGamGamOpenDataAdapter(synthetic_source(source_path))


@pytest.fixture(scope="module")
def loaded(adapter: AtlasGamGamOpenDataAdapter) -> ContractContext:
    return adapter.load_context(workflow_id="derivation", run_id="r", attempt_id="a")


@pytest.fixture(scope="module")
def correct_artifacts(source_path: Path) -> dict[str, Any]:
    proposal = ScriptedAgent(CORRECT_SCRIPT_ID).propose(ATLAS_AGENT_TASK, attempt_id="a")
    result = CodeSandbox(timeout_seconds=120.0, cpu_seconds=60).run(
        proposal.code, input_path=source_path
    )
    assert result.artifacts is not None
    return cast(dict[str, Any], copy.deepcopy(dict(result.artifacts)))


def _selection_results(loaded: ContractContext, artifacts: dict[str, Any]) -> dict[str, str]:
    """Evaluate the three new contracts against one set of agent artifacts."""

    selection_context = agent_contexts(loaded, artifacts)[0]
    contracts = (
        AtlasWeightProvenanceContract(),
        AtlasRegionCoverageContract(),
        AtlasSourceConstantsContract(),
    )
    return {
        contract.contract_id: contract.evaluate(selection_context).status.value
        for contract in contracts
    }


# --- the analyses that must pass ------------------------------------------------------


def test_the_trusted_pipeline_passes_its_own_new_contracts(
    adapter: AtlasGamGamOpenDataAdapter,
) -> None:
    execution = GuardedWorkflowRunner().execute(
        adapter.checkpoints(workflow_id="derivation", run_id="trusted", attempt_id="a")
    )

    assert not execution.trace.blocked


def test_correct_agent_code_passes_the_new_contracts(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    assert set(_selection_results(loaded, correct_artifacts).values()) == {"pass"}


def test_every_valid_range_passes(source_path: Path) -> None:
    """Milestone 5 measures false positives over three disjoint ranges; all must stay clean."""

    for index in range(3):
        ranged = AtlasGamGamOpenDataAdapter(
            synthetic_source(source_path),
            entry_start=index * BLOCK_SIZE,
            entry_stop=(index + 1) * BLOCK_SIZE,
        )
        execution = GuardedWorkflowRunner().execute(
            ranged.checkpoints(workflow_id="derivation", run_id=f"range-{index}", attempt_id="a")
        )
        assert not execution.trace.blocked, f"range {index} was rejected"


def test_single_precision_accumulation_is_not_a_violation(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The source stores weights in float32, so a float32 analysis must not be rejected."""

    nudged = copy.deepcopy(correct_artifacts)
    selection = nudged["selection"]
    selection["selected_weight"] = [float(np.float32(w)) for w in selection["selected_weight"]]
    selection["selected_weight_sum"] = float(
        np.sum(np.array(selection["selected_weight"], dtype=np.float32))
    )
    selection["generated_weight_sum"] = float(np.float32(selection["generated_weight_sum"]))
    selection["cross_section_pb"] = float(np.float32(selection["cross_section_pb"]))

    assert set(_selection_results(loaded, nudged).values()) == {"pass"}


# --- the analyses that must fail ------------------------------------------------------


def test_normalization_folded_into_the_weights_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The mistake a real model made: weights scaled, and every internal relation still closes."""

    wrong = copy.deepcopy(correct_artifacts)
    factor = wrong["histogram"]["normalization_factor"]
    selection = wrong["selection"]
    selection["selected_weight"] = [w * factor for w in selection["selected_weight"]]
    selection["selected_weight_sum"] *= factor

    results = _selection_results(loaded, wrong)

    assert results["hep.atlas_open_data.weight_provenance"] == "fail"


def test_a_declared_sum_that_is_not_the_sum_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    wrong["selection"]["selected_weight_sum"] += 1.0

    assert _selection_results(loaded, wrong)["hep.atlas_open_data.weight_provenance"] == "fail"


def test_selected_events_in_no_region_are_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    wrong["selection"]["regions"]["control"] = wrong["selection"]["regions"]["control"][1:]

    assert _selection_results(loaded, wrong)["hep.atlas_open_data.region_coverage"] == "fail"


def test_a_summed_per_file_constant_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The other mistake a real model made: SumWeights aggregated instead of read once."""

    wrong = copy.deepcopy(correct_artifacts)
    selection = wrong["selection"]
    selection["generated_weight_sum"] *= len(selection["input_event_ids"])

    assert _selection_results(loaded, wrong)["hep.atlas_open_data.source_constants"] == "fail"


# --- graceful degradation --------------------------------------------------------------


def test_the_contracts_stand_down_without_trusted_source_facts(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """A workflow that never carried source facts gets no verdict, rather than a false one."""

    stripped = replace(loaded, provenance={"events": {"source_type": "synthetic"}})
    selection_context = agent_contexts(stripped, correct_artifacts)[0]

    for contract in (AtlasWeightProvenanceContract(), AtlasSourceConstantsContract()):
        assert contract.evaluate(selection_context).status.value == "not_applicable"


def test_a_region_partition_that_ignores_mass_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The mistake a real model made: alternating events, structurally a perfect partition.

    Disjointness and coverage both pass on this, because any partition satisfies them. Only a
    contract that knows the signal region is a mass window can see that it is meaningless.
    """

    wrong = copy.deepcopy(correct_artifacts)
    selected = wrong["selection"]["selected_event_ids"]
    wrong["selection"]["regions"] = {
        "signal": [event_id for index, event_id in enumerate(selected) if index % 2 == 0],
        "control": [event_id for index, event_id in enumerate(selected) if index % 2 == 1],
    }

    selection_context = agent_contexts(loaded, wrong)[0]
    results = {
        contract.contract_id: contract.evaluate(selection_context).status.value
        for contract in (
            AtlasRegionCoverageContract(),
            AtlasRegionDisjointContract(),
            AtlasRegionDefinitionContract(),
        )
    }

    assert results["hep.atlas_open_data.region_coverage"] == "pass"
    assert results["hep.atlas_open_data.region_disjoint"] == "pass"
    assert results["hep.atlas_open_data.region_definition"] == "fail"


def test_the_correct_regions_satisfy_the_definition_check(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    selection_context = agent_contexts(loaded, correct_artifacts)[0]

    assert AtlasRegionDefinitionContract().evaluate(selection_context).status.value == "pass"


def test_a_yield_without_its_scale_factor_cannot_be_checked(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The gap a real judge found: a final result its own inputs do not imply.

    Reviewing only the final artifact, a judge objected that a signal sum of 82,383 and a
    background of 1,614 cannot yield 7.47. It was right — the scale factor relating them was
    reported one stage earlier and never reached the artifact stating the result.
    """

    stripped = copy.deepcopy(correct_artifacts)
    del stripped["yield_estimate"]["normalization_factor"]

    yield_context = agent_contexts(loaded, stripped)[2]
    result = AtlasYieldClosureContract().evaluate(yield_context)

    assert result.status.value == "fail"
    assert result.violation is not None
    assert "without the factor" in str(result.violation.evidence["reason"])


def test_a_yield_inconsistent_with_its_own_inputs_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    wrong["yield_estimate"]["estimated_yield"] *= 3.0

    yield_context = agent_contexts(loaded, wrong)[2]

    assert AtlasYieldClosureContract().evaluate(yield_context).status.value == "fail"


def test_the_correct_yield_closes(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    yield_context = agent_contexts(loaded, correct_artifacts)[2]

    assert AtlasYieldClosureContract().evaluate(yield_context).status.value == "pass"


def test_a_background_the_sidebands_do_not_imply_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The gap `yield_closure` leaves open, and the one this round's reviewer proposed.

    Doubling the background and recomputing the yield from it keeps the closure intact: the
    reported result still follows from the numbers printed beside it, and every one of the sixteen
    contracts that preceded this one accepted it. On the real sample the substitution moves the
    result by two percent.
    """

    wrong = copy.deepcopy(correct_artifacts)
    estimate = wrong["yield_estimate"]
    estimate["background_estimate"] *= 2.0
    estimate["estimated_yield"] = (
        estimate["signal_weight_sum"] - estimate["background_estimate"]
    ) * estimate["normalization_factor"]

    yield_context = agent_contexts(loaded, wrong)[2]

    assert AtlasYieldClosureContract().evaluate(yield_context).status.value == "pass"
    result = AtlasBackgroundEstimateContract().evaluate(yield_context)
    assert result.status.value == "fail"
    assert result.violation is not None
    assert (
        result.violation.evidence["reported_background"]
        != (result.violation.evidence["implied_background"])
    )


def test_a_background_without_the_sums_behind_it_cannot_be_checked(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    stripped = copy.deepcopy(correct_artifacts)
    del stripped["yield_estimate"]["sideband_weight_sum"]

    yield_context = agent_contexts(loaded, stripped)[2]
    result = AtlasBackgroundEstimateContract().evaluate(yield_context)

    assert result.status.value == "fail"
    assert result.violation is not None
    assert "sideband_weight_sum" in str(result.violation.evidence["missing_fields"])


def test_a_background_scaled_from_no_sideband_bins_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """Nothing divides by zero here: the artifact is rejected before the arithmetic runs."""

    wrong = copy.deepcopy(correct_artifacts)
    wrong["yield_estimate"]["sideband_bin_count"] = 0

    yield_context = agent_contexts(loaded, wrong)[2]
    result = AtlasBackgroundEstimateContract().evaluate(yield_context)

    assert result.status.value == "fail"
    assert result.violation is not None
    assert "no denominator" in str(result.violation.evidence["reason"])


def test_a_scale_factor_the_source_does_not_imply_is_caught(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """The check the reviewer said the final artifact could not make, made from provenance.

    The yield artifact reports `normalization_factor` and none of the three numbers behind it, so
    nothing inside it can establish the factor. The verified loader's provenance carries the
    cross-section and the generated weight sum, and the configuration carries the luminosity, which
    together determine it.
    """

    wrong = copy.deepcopy(correct_artifacts)
    estimate = wrong["yield_estimate"]
    estimate["normalization_factor"] *= 1.5
    estimate["estimated_yield"] = (
        estimate["signal_weight_sum"] - estimate["background_estimate"]
    ) * estimate["normalization_factor"]

    yield_context = agent_contexts(loaded, wrong)[2]

    # Self-consistent by construction: the two contracts that read only the artifact accept it.
    assert AtlasYieldClosureContract().evaluate(yield_context).status.value == "pass"
    assert AtlasBackgroundEstimateContract().evaluate(yield_context).status.value == "pass"
    result = AtlasNormalizationProvenanceContract().evaluate(yield_context)
    assert result.status.value == "fail"
    assert result.violation is not None
    assert (
        result.violation.evidence["declared_normalization_factor"]
        != (result.violation.evidence["implied_normalization_factor"])
    )


def test_the_correct_scale_factor_follows_from_the_verified_source(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    yield_context = agent_contexts(loaded, correct_artifacts)[2]

    assert AtlasNormalizationProvenanceContract().evaluate(yield_context).status.value == "pass"


def test_a_yield_without_a_scale_factor_cannot_be_derived_either(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    stripped = copy.deepcopy(correct_artifacts)
    del stripped["yield_estimate"]["normalization_factor"]

    yield_context = agent_contexts(loaded, stripped)[2]
    result = AtlasNormalizationProvenanceContract().evaluate(yield_context)

    assert result.status.value == "fail"
    assert result.violation is not None
    assert "without the factor" in str(result.violation.evidence["reason"])


def test_a_workflow_that_lost_its_provenance_is_not_applicable_rather_than_failing(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """Losing provenance is reported where it was lost, not by every downstream check failing."""

    yield_context = replace(agent_contexts(loaded, correct_artifacts)[2], provenance={})

    result = AtlasNormalizationProvenanceContract().evaluate(yield_context)

    assert result.status.value == "not_applicable"
    assert result.violation is None


def test_a_source_with_no_generated_weight_has_no_denominator(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    """Nothing divides by zero: the artifact is rejected before the arithmetic runs."""

    declaration = dict(cast(Any, loaded.provenance["events"]))
    facts = dict(cast(Any, declaration["source_facts"]))
    facts["generated_weight_sum"] = 0.0
    declaration["source_facts"] = facts
    zeroed = replace(loaded, provenance={"events": declaration})

    yield_context = agent_contexts(zeroed, correct_artifacts)[2]
    result = AtlasNormalizationProvenanceContract().evaluate(yield_context)

    assert result.status.value == "fail"
    assert result.violation is not None
    assert "no generated weight" in str(result.violation.evidence["reason"])


def test_the_correct_background_follows_from_its_sidebands(
    loaded: ContractContext, correct_artifacts: dict[str, Any]
) -> None:
    yield_context = agent_contexts(loaded, correct_artifacts)[2]

    assert AtlasBackgroundEstimateContract().evaluate(yield_context).status.value == "pass"
