import json
import runpy
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import sciagentguard.packs.hep.workflow as hep_workflow
from sciagentguard.core import ContractContext
from sciagentguard.packs.hep import make_synthetic_hep_context
from sciagentguard.runtime import RepairAction

DEMO = Path(__file__).parents[2] / "examples" / "hep_guarded_demo.py"


def run_demo(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DEMO), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(result.stdout))


def test_valid_demo_emits_an_unblocked_json_trace() -> None:
    result = run_demo("--fault", "none")
    trace = parse_stdout(result)

    assert result.returncode == 0
    assert result.stderr == ""
    assert trace["blocked"] is False
    assert [checkpoint["stage"] for checkpoint in trace["checkpoints"]] == [
        "post_load",
        "post_selection",
        "post_split",
        "post_normalization",
    ]
    assert all(
        result["status"] == "pass"
        for checkpoint in trace["checkpoints"]
        for result in checkpoint["results"]
    )


@pytest.mark.parametrize(
    ("fault", "expected_contract_id", "expected_stage"),
    [
        ("missing_branch", "hep.schema.required_branches", "post_load"),
        ("zero_weights", "hep.weights.nonzero_support", "post_load"),
        ("nonfinite_weights", "hep.weights.finite", "post_load"),
        ("unit_scale_error", "hep.kinematics.jet_pt_range", "post_load"),
        ("undeclared_synthetic_data", "hep.provenance.events_declared", "post_load"),
        ("empty_selection", "hep.selection.nonempty", "post_selection"),
        ("split_leakage", "hep.splits.disjoint_event_ids", "post_split"),
        (
            "wrong_normalization",
            "hep.normalization.yield_consistent",
            "post_normalization",
        ),
    ],
)
def test_fault_demo_emits_the_localized_violation(
    fault: str, expected_contract_id: str, expected_stage: str
) -> None:
    result = run_demo("--fault", fault)
    trace = parse_stdout(result)
    failed = [
        item
        for checkpoint in trace["checkpoints"]
        for item in checkpoint["results"]
        if item["status"] == "fail"
    ]

    assert result.returncode == 1
    assert result.stderr == ""
    assert trace["blocked"] is True
    assert [item["contract_id"] for item in failed] == [expected_contract_id]
    assert trace["checkpoints"][-1]["stage"] == expected_stage
    assert failed[0]["violation"]["stage"] == expected_stage


def test_demo_can_write_its_trace_to_a_file(tmp_path: Path) -> None:
    output = tmp_path / "trace.json"
    result = run_demo("--fault", "zero_weights", "--output", str(output))

    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(output.read_text(encoding="utf-8"))["blocked"] is True


def test_demo_rejects_unknown_faults() -> None:
    result = run_demo("--fault", "unknown")

    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid choice" in result.stderr


def test_repair_demo_revalidates_zero_weights_after_a_structured_action() -> None:
    result = run_demo("--mode", "repair", "--fault", "zero_weights")
    trace = parse_stdout(result)

    assert result.returncode == 0
    assert result.stderr == ""
    assert trace["mode"] == "repair"
    assert trace["outcome"] == "repaired"
    assert [attempt["execution"]["blocked"] for attempt in trace["attempts"]] == [True, False]
    assert trace["attempts"][0]["action"]["action_type"] == ("hep.reload_declared_synthetic_source")
    assert trace["attempts"][1]["action"] is None


def test_repair_demo_does_not_call_the_policy_for_a_valid_fixture() -> None:
    result = run_demo("--mode", "repair", "--fault", "none")
    trace = parse_stdout(result)

    assert result.returncode == 0
    assert trace["outcome"] == "passed"
    assert len(trace["attempts"]) == 1
    assert trace["attempts"][0]["action"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "missing_branch",
        "nonfinite_weights",
        "unit_scale_error",
        "undeclared_synthetic_data",
        "empty_selection",
        "split_leakage",
        "wrong_normalization",
    ],
)
def test_repair_demo_leaves_unsupported_faults_unresolved(fault: str) -> None:
    result = run_demo("--mode", "repair", "--fault", fault)
    trace = parse_stdout(result)

    assert result.returncode == 1
    assert result.stderr == ""
    assert trace["outcome"] == "unresolved"
    assert len(trace["attempts"]) == 1


def test_repair_demo_can_write_its_trace_to_a_file(tmp_path: Path) -> None:
    output = tmp_path / "repair-trace.json"
    result = run_demo("--mode", "repair", "--fault", "zero_weights", "--output", str(output))

    assert result.returncode == 0
    assert result.stdout == ""
    assert json.loads(output.read_text(encoding="utf-8"))["outcome"] == "repaired"


def test_demo_json_does_not_include_runtime_or_metadata_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "CLI_SECRET_SENTINEL"

    def make_context_with_secret(
        *,
        workflow_id: str = "hep-synthetic-demo",
        run_id: str = "run-001",
        attempt_id: str = "attempt-0",
    ) -> ContractContext:
        context = make_synthetic_hep_context(
            workflow_id=workflow_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        return replace(
            context,
            artifacts={**context.artifacts, "private_artifact": secret},
            config={**context.config, "private_config": secret},
            provenance={
                "events": {
                    "source_type": "synthetic",
                    "generator": "sciagentguard.packs.hep.fixtures",
                    "credential": secret,
                }
            },
        )

    monkeypatch.setattr(hep_workflow, "make_synthetic_hep_context", make_context_with_secret)
    namespace = runpy.run_path(str(DEMO))
    main = cast(Callable[[Sequence[str] | None], int], namespace["main"])

    assert main(("--fault", "none")) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["blocked"] is False
    assert secret not in output


def test_analysis_checkpoint_repair_callback_fails_closed() -> None:
    namespace = runpy.run_path(str(DEMO))
    reject = cast(Callable[..., ContractContext], namespace["_reject_unregistered_repair"])
    action = RepairAction(
        action_id="test-action",
        action_type="unregistered.action",
        rationale="exercise the fail-closed demo boundary",
        target_violation_ids=("test-violation",),
    )

    with pytest.raises(ValueError, match="no trusted repair step is registered"):
        reject(action, attempt_id="attempt-0.repair-1")
