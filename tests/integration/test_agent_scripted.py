"""Agent-written analysis code must reach the guard, and its faults must be localized."""

from __future__ import annotations

from pathlib import Path

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.adapters.agent import (
    ATLAS_AGENT_TASK,
    CodeSandbox,
    SandboxOutcome,
    ScriptedAgent,
    agent_contexts,
)
from sciagentguard.adapters.agent._scripts import CORRECT_SCRIPT_ID, FAULTY_SCRIPT_IDS, SCRIPTS
from sciagentguard.core import ContractContext
from sciagentguard.runtime import GuardedWorkflowRunner
from tests.integration._atlas_root import synthetic_source, write_root_file

# The scripted faults and the contract that should stop each of them. `unrunnable` is absent on
# purpose: a failure to produce runnable code is not a scientific fault.
# The stage each fault is caught at, and every contract expected to fire there. A fault may trip
# more than one: an overlapping control window both duplicates events across regions and, because
# it is defined by its own window rather than as a complement, leaves others in no region at all.
EXPECTED_DETECTION = {
    "empty_selection": ("post_selection", {"hep.selection.nonempty"}),
    "luminosity_unit_slip": ("post_histogram", {"hep.atlas_open_data.histogram_closure"}),
    "overlapping_control_window": (
        "post_selection",
        {"hep.atlas_open_data.region_disjoint", "hep.atlas_open_data.region_coverage"},
    ),
    "stale_cutflow": ("post_selection", {"hep.atlas_open_data.cutflow_monotonic"}),
}


@pytest.fixture(scope="module")
def source_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("agent") / "test.root"
    write_root_file(path)
    return path


@pytest.fixture(scope="module")
def adapter(source_path: Path) -> AtlasGamGamOpenDataAdapter:
    return AtlasGamGamOpenDataAdapter(synthetic_source(source_path))


@pytest.fixture(scope="module")
def loaded(adapter: AtlasGamGamOpenDataAdapter) -> ContractContext:
    return adapter.load_context(workflow_id="atlas-agent", run_id="run-001", attempt_id="attempt-0")


@pytest.fixture(scope="module")
def sandbox() -> CodeSandbox:
    return CodeSandbox(timeout_seconds=120.0, cpu_seconds=60, memory_bytes=4 * 1024**3)


def _guard(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    script_id: str,
    source_path: Path,
) -> tuple[SandboxOutcome, str | None, list[str]]:
    proposal = ScriptedAgent(script_id).propose(ATLAS_AGENT_TASK, attempt_id="attempt-0")
    result = sandbox.run(proposal.code, input_path=source_path)
    if not result.produced_artifacts:
        return result.outcome, None, []

    assert result.artifacts is not None
    contexts = agent_contexts(loaded, result.artifacts)
    execution = GuardedWorkflowRunner().execute(adapter.checkpoints_for((loaded, *contexts)))
    terminal = execution.trace.checkpoints[-1]
    failed = [entry.contract_id for entry in terminal.results if entry.status.value == "fail"]
    return result.outcome, terminal.stage, failed


def test_correct_agent_code_passes_every_checkpoint(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    source_path: Path,
) -> None:
    outcome, stage, failed = _guard(adapter, loaded, sandbox, CORRECT_SCRIPT_ID, source_path)

    assert outcome is SandboxOutcome.COMPLETED
    assert stage == "post_yield"
    assert failed == []


@pytest.mark.parametrize("script_id", sorted(EXPECTED_DETECTION))
def test_each_scripted_fault_is_localized_to_its_contract(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    source_path: Path,
    script_id: str,
) -> None:
    expected_stage, expected_contracts = EXPECTED_DETECTION[script_id]

    outcome, stage, failed = _guard(adapter, loaded, sandbox, script_id, source_path)

    assert outcome is SandboxOutcome.COMPLETED, "the faulty code must still run"
    assert stage == expected_stage
    assert set(failed) == expected_contracts


def test_unrunnable_code_is_a_code_failure_not_a_semantic_fault(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    source_path: Path,
) -> None:
    outcome, stage, failed = _guard(adapter, loaded, sandbox, "unrunnable", source_path)

    assert outcome is not SandboxOutcome.COMPLETED
    assert stage is None
    assert failed == []


def test_every_declared_faulty_script_has_a_declared_outcome() -> None:
    """Each faulty script is either caught by a contract or is a declared non-semantic failure."""

    non_semantic = {"unrunnable", "wrong_output_schema"}

    assert set(EXPECTED_DETECTION) | non_semantic == set(FAULTY_SCRIPT_IDS)


def test_wrong_output_shape_is_rejected_by_the_bridge_not_by_a_contract(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    source_path: Path,
) -> None:
    proposal = ScriptedAgent("wrong_output_schema").propose(
        ATLAS_AGENT_TASK, attempt_id="attempt-0"
    )
    result = sandbox.run(proposal.code, input_path=source_path)

    # The code ran perfectly; it simply reported its results under its own key names.
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.artifacts is not None
    with pytest.raises(ValueError, match="did not produce a 'histogram' object"):
        agent_contexts(loaded, result.artifacts)


def test_missing_agent_artifacts_are_an_output_contract_failure(loaded: ContractContext) -> None:
    with pytest.raises(ValueError, match="did not produce a 'histogram' object"):
        agent_contexts(loaded, {"selection": {}, "yield_estimate": {}})


def test_the_agent_repairs_only_from_structured_feedback() -> None:
    agent = ScriptedAgent("luminosity_unit_slip")
    generic = agent.propose(
        ATLAS_AGENT_TASK, attempt_id="attempt-1", feedback="the result is wrong"
    )
    structured = agent.propose(
        ATLAS_AGENT_TASK,
        attempt_id="attempt-1",
        feedback="contract_id=hep.atlas_open_data.histogram_closure stage=post_histogram",
    )

    assert generic.code == SCRIPTS["luminosity_unit_slip"]
    assert structured.code == SCRIPTS[CORRECT_SCRIPT_ID]


def test_an_unknown_script_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown script"):
        ScriptedAgent("does-not-exist")
