"""The oracle must accept the trusted answer and name what a wrong one got wrong."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.adapters.agent import CodeSandbox, ScriptedAgent
from sciagentguard.adapters.agent._scripts import CORRECT_SCRIPT_ID
from sciagentguard.adapters.agent.atlas_task import ATLAS_AGENT_TASK
from sciagentguard.adapters.agent.reference import compare_to_reference
from sciagentguard.core import ContractContext
from tests.integration._atlas_root import synthetic_source, write_root_file


@pytest.fixture(scope="module")
def source_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("oracle") / "test.root"
    write_root_file(path)
    return path


@pytest.fixture(scope="module")
def contexts(source_path: Path) -> tuple[ContractContext, ...]:
    adapter = AtlasGamGamOpenDataAdapter(synthetic_source(source_path))
    return adapter.contexts(workflow_id="oracle", run_id="run-001", attempt_id="attempt-0")


@pytest.fixture(scope="module")
def correct_artifacts(source_path: Path) -> dict[str, Any]:
    """The artifacts an agent produces when its code is right."""

    proposal = ScriptedAgent(CORRECT_SCRIPT_ID).propose(ATLAS_AGENT_TASK, attempt_id="attempt-0")
    result = CodeSandbox(timeout_seconds=120.0, cpu_seconds=60).run(
        proposal.code, input_path=source_path
    )
    assert result.artifacts is not None
    return cast(dict[str, Any], copy.deepcopy(dict(result.artifacts)))


def test_the_trusted_pipeline_agrees_with_itself(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    """An oracle that cannot recognise a correct answer would make every later number noise."""

    comparison = compare_to_reference(contexts, correct_artifacts)

    assert comparison.agrees, comparison.disagreements
    assert comparison.disagreements == ()


def test_a_scaled_normalization_is_caught(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    wrong["histogram"]["normalization_factor"] *= 1000.0
    wrong["yield_estimate"]["estimated_yield"] *= 1000.0

    comparison = compare_to_reference(contexts, wrong)

    assert not comparison.agrees
    assert any("normalization_factor" in entry for entry in comparison.disagreements)


def test_overlapping_regions_are_caught(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    signal = wrong["selection"]["regions"]["signal"]
    wrong["selection"]["regions"]["control"].append(signal[0])

    comparison = compare_to_reference(contexts, wrong)

    assert not comparison.agrees
    assert any("regions.control" in entry for entry in comparison.disagreements)


def test_a_missing_artifact_is_reported_rather_than_ignored(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    del wrong["histogram"]

    comparison = compare_to_reference(contexts, wrong)

    assert not comparison.agrees
    assert any(entry.startswith("histogram:") for entry in comparison.disagreements)


def test_single_precision_accumulation_still_agrees(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    """A correct analysis that sums in float32 must not be scored as a model error."""

    import numpy as np

    nudged = copy.deepcopy(correct_artifacts)
    total = nudged["selection"]["selected_weight_sum"]
    nudged["selection"]["selected_weight_sum"] = float(np.float32(total))

    comparison = compare_to_reference(contexts, nudged)

    assert comparison.agrees, comparison.disagreements


def test_a_nonpositive_tolerance_is_rejected(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compare_to_reference(contexts, correct_artifacts, relative_tolerance=0.0)


def test_a_renamed_or_regrouped_cutflow_is_not_a_model_error(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    """Only the cutflow's endpoints and its monotonicity carry meaning.

    Calling the first stage `total_events`, or applying both momentum thresholds in one step,
    changes nothing about the analysis. Scoring either as a disagreement would report a model
    error that is not one.
    """

    renamed = copy.deepcopy(correct_artifacts)
    stages = renamed["selection"]["cutflow"]
    renamed["selection"]["cutflow"] = [
        {"cut_id": "total_events", "surviving": stages[0]["surviving"]},
        {"cut_id": "n_photon >= 2", "surviving": stages[1]["surviving"]},
        {"cut_id": "lead > 40 and sublead > 30", "surviving": stages[-1]["surviving"]},
    ]

    assert compare_to_reference(contexts, renamed).agrees


def test_a_cutflow_that_ends_at_the_wrong_count_is_still_caught(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    wrong["selection"]["cutflow"][-1]["surviving"] += 5

    comparison = compare_to_reference(contexts, wrong)

    assert not comparison.agrees
    assert any("cutflow: ends at" in entry for entry in comparison.disagreements)


def test_one_event_crossing_a_bin_edge_is_not_a_model_error(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    """Two correct implementations can put a boundary event in either of two adjacent bins.

    A mass sitting exactly on a bin edge lands on whichever side the arithmetic falls, so a
    one-event difference between neighbouring bins is a floating-point convention rather than a
    scientific disagreement.
    """

    nudged = copy.deepcopy(correct_artifacts)
    bins = nudged["histogram"]["bin_weight_sums"]
    moved = max(range(len(bins)), key=lambda index: bins[index])
    weight = correct_artifacts["selection"]["selected_weight"][0]
    bins[moved] -= weight
    bins[moved + 1] += weight

    assert compare_to_reference(contexts, nudged).agrees


def test_a_real_binning_mistake_is_still_caught(
    contexts: tuple[ContractContext, ...], correct_artifacts: dict[str, Any]
) -> None:
    wrong = copy.deepcopy(correct_artifacts)
    bins = wrong["histogram"]["bin_weight_sums"]
    peak = max(range(len(bins)), key=lambda index: bins[index])
    bins[peak] = bins[peak] / 2.0

    comparison = compare_to_reference(contexts, wrong)

    assert not comparison.agrees
    assert any("bin_weight_sums" in entry for entry in comparison.disagreements)
