"""The 6B pilot must be rehearsable offline, and must never record a credential."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters.agent._scripts import SCRIPTS
from sciagentguard.adapters.agent.gemini import Transport
from tests.integration._atlas_root import synthetic_source, write_root_file

ROOT = Path(__file__).parents[2]
PILOT = ROOT / "benchmarks" / "atlas_agent_6b_pilot.py"
NAMESPACE = runpy.run_path(str(PILOT))
RUN_PILOT = cast(Callable[..., Any], NAMESPACE["run_pilot"])
RENDER_MARKDOWN = cast(Callable[[Any], str], NAMESPACE["render_markdown"])

SECRET = "AIza-PILOT-SECRET-DO-NOT-LEAK"
MODELS = {"models": [{"name": "models/gemini-3.5-flash-lite"}, {"name": "models/gemini-3.7-flash"}]}


def _stub(
    first_script: str, repair_script: str = "correct", verdict: str = "VALID"
) -> tuple[Transport, list[str]]:
    """Answer the model listing, the agent, and the judge without touching the network."""

    prompts: list[str] = []

    def transport(url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        if url.endswith("/models"):
            return 200, json.dumps(MODELS).encode("utf-8")
        assert body is not None
        prompt = json.loads(body)["contents"][0]["parts"][0]["text"]
        prompts.append(prompt)
        if "reviewing the final result" in prompt:
            text = verdict
        elif "previous attempt was rejected" in prompt:
            text = f"```python\n{SCRIPTS[repair_script]}\n```"
        else:
            text = f"```python\n{SCRIPTS[first_script]}\n```"
        payload = {
            "candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 900},
        }
        return 200, json.dumps(payload).encode("utf-8")

    return transport, prompts


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.root"
    write_root_file(path)
    return path


def _run(source_path: Path, tmp_path: Path, transport: Any, **kwargs: Any) -> Any:
    return RUN_PILOT(
        synthetic_source(source_path),
        seeds=kwargs.pop("seeds", [1]),
        timeout_seconds=120.0,
        code_dir=tmp_path / "code",
        transport=transport,
        api_key=SECRET,
        **kwargs,
    )


def test_a_faulty_model_analysis_is_blocked_and_then_repaired(
    source_path: Path, tmp_path: Path
) -> None:
    transport, _ = _stub("luminosity_unit_slip")

    report = _run(source_path, tmp_path, transport)
    run = report.runs[0]

    # The model's code ran cleanly and produced the declared shape; only the physics was wrong.
    assert run.sandbox_outcome == "completed"
    assert run.produced_artifacts
    assert run.oracle_agrees is False
    assert run.blocked_stage == "post_histogram"
    assert run.blocked_contracts == ("hep.atlas_open_data.histogram_closure",)
    assert run.arm_accepted["runtime_guarded"] is False
    for weaker in ("unguarded", "generic_data_checks", "llm_judge"):
        assert run.arm_accepted[weaker] is True, weaker

    # `final_check_only` used to accept this run, and stopped when
    # `hep.atlas_open_data.normalization_provenance` was added. That contract is declared at the
    # final stage, so this arm evaluates it, and it derives the scale factor from the loader's
    # provenance rather than from the artifact -- which is exactly what a luminosity slip breaks.
    #
    # The lesson is worth stating precisely, because it sharpens the claim this repository makes.
    # What puts a fault beyond the reach of a final check is not *when* the check runs but whether
    # it can consult trusted facts about what came before. A final-stage check restricted to the
    # artifact cannot see this fault; a final-stage check that reads provenance can. The faults
    # that remain invisible at the end -- region overlap, a stale cutflow -- are invisible because
    # no trusted fact about them survives to that stage in any form.
    assert run.arm_accepted["final_check_only"] is False
    assert report.repairs[0].resolved is True


def test_a_correct_model_analysis_passes_every_arm(source_path: Path, tmp_path: Path) -> None:
    transport, _ = _stub("correct")

    report = _run(source_path, tmp_path, transport)
    run = report.runs[0]

    assert run.oracle_agrees is True
    assert run.oracle_disagreements == ()
    assert run.blocked_stage is None
    assert all(run.arm_accepted[arm] for arm in report.arms)
    assert report.repairs == ()


def test_the_generated_code_is_preserved_for_inspection(source_path: Path, tmp_path: Path) -> None:
    transport, _ = _stub("correct")

    report = _run(source_path, tmp_path, transport, seeds=[7])
    saved = (tmp_path / "code" / "seed-7-attempt-0.py").read_text(encoding="utf-8")

    assert saved.strip() == SCRIPTS["correct"].strip()
    assert report.runs[0].code_lines > 0


def test_the_prompt_states_the_schema_and_carries_feedback_on_retry(
    source_path: Path, tmp_path: Path
) -> None:
    transport, prompts = _stub("luminosity_unit_slip")

    _run(source_path, tmp_path, transport)

    assert "yield_estimate" in prompts[0]
    retries = [prompt for prompt in prompts if "previous attempt was rejected" in prompt]
    assert retries, "a blocked run must hand the model its violations"
    assert "histogram_closure" in retries[0]


def test_the_report_records_no_credential(source_path: Path, tmp_path: Path) -> None:
    transport, _ = _stub("correct")

    report = _run(source_path, tmp_path, transport)
    payload = report.model_dump_json()

    assert SECRET not in payload
    assert SECRET not in RENDER_MARKDOWN(report)
    assert "api_key" not in payload.lower()


def test_unrunnable_model_code_is_recorded_without_arms(source_path: Path, tmp_path: Path) -> None:
    transport, _ = _stub("unrunnable")

    report = _run(source_path, tmp_path, transport)
    run = report.runs[0]

    assert not run.produced_artifacts
    assert run.oracle_agrees is None
    assert run.arm_accepted == {}
    assert report.repairs == ()


def test_a_wrong_output_shape_is_recorded_as_an_output_contract_failure(
    source_path: Path, tmp_path: Path
) -> None:
    transport, _ = _stub("wrong_output_schema")

    report = _run(source_path, tmp_path, transport)
    run = report.runs[0]

    assert run.produced_artifacts
    assert run.oracle_agrees is False
    assert any("output contract" in entry for entry in run.oracle_disagreements)
    assert run.arm_accepted == {}
