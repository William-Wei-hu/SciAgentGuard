"""The agent experiment must run offline and report code failures apart from semantic ones."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from tests.integration._atlas_root import synthetic_source, write_root_file

ROOT = Path(__file__).parents[2]
EXPERIMENT = ROOT / "benchmarks" / "atlas_agent_experiment.py"
SAVED_JSON = ROOT / "benchmarks" / "results" / "atlas_agent_results.json"
NAMESPACE = runpy.run_path(str(EXPERIMENT))
EXPERIMENT_REPORT = cast(Any, NAMESPACE["ExperimentReport"])
GENERATE_REPORT = cast(Callable[..., Any], NAMESPACE["generate_report"])
RENDER_MARKDOWN = cast(Callable[[Any], str], NAMESPACE["render_markdown"])
MAIN = cast(Callable[[list[str]], int], NAMESPACE["main"])


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    path = tmp_path / "test.root"
    write_root_file(path)
    return path


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root_path = tmp_path_factory.mktemp("LOCAL_PATH_SECRET") / "test.root"
    write_root_file(root_path)
    return GENERATE_REPORT(synthetic_source(root_path), repetitions=1, timeout_seconds=120.0)


def _run(case: Any, repetition: int, arm: str) -> Any:
    return next(run for run in case.repetitions[repetition].runs if run.baseline == arm)


def test_the_experiment_runs_offline_with_no_model_provider(report: Any) -> None:
    assert report.agent["model_id"] is None
    assert report.agent["provider"] is None
    assert "scripted" in str(report.parameters["agent_kind"])
    assert report.arms == (
        "unguarded",
        "generic_data_checks",
        "llm_judge",
        "final_check_only",
        "runtime_guarded",
    )


def test_runtime_guarding_stops_every_semantic_fault_the_agent_wrote(report: Any) -> None:
    cases = {case.case_id: case for case in report.cases}
    faults = (
        "empty_selection",
        "luminosity_unit_slip",
        "overlapping_control_window",
        "stale_cutflow",
    )

    assert _run(cases["correct"], 0, "runtime_guarded").accepted
    for case_id in faults:
        run = _run(cases[case_id], 0, "runtime_guarded")
        assert not run.accepted, case_id
        assert not run.crashed, case_id
    assert report.metrics.by_baseline["runtime_guarded"].all_fault_false_pass_rate.value == 0.0


def test_unrunnable_code_is_counted_as_a_code_failure_not_a_false_pass(report: Any) -> None:
    cases = {case.case_id: case for case in report.cases}

    assert report.code_failure_runs == 1
    for arm in report.arms:
        run = _run(cases["unrunnable"], 0, arm)
        assert run.crashed
        assert not run.accepted


def test_wrong_output_shape_is_its_own_failure_kind(report: Any) -> None:
    """Code that runs and reports in its own shape is neither a crash nor a semantic fault."""

    cases = {case.case_id: case for case in report.cases}

    assert report.schema_failure_runs == 1
    for arm in report.arms:
        run = _run(cases["wrong_output_schema"], 0, arm)
        assert run.crashed
        assert not run.accepted
        assert run.crash_type == "invalid_agent_artifacts"


def test_non_semantic_cases_are_excluded_from_the_rates(report: Any) -> None:
    """A run every arm rejects must not flatter the weakest arm by growing the denominator."""

    semantic_faults = 4
    for arm in report.arms:
        metrics = report.metrics.by_baseline[arm]
        assert metrics.all_fault_false_pass_rate.denominator == semantic_faults * 1
    assert report.metrics.by_baseline["unguarded"].all_fault_false_pass_rate.value == 1.0
    assert report.metrics.by_baseline["runtime_guarded"].all_fault_false_pass_rate.value == 0.0


def test_the_weaker_arms_miss_faults_that_runtime_guarding_catches(report: Any) -> None:
    runtime = report.metrics.by_baseline["runtime_guarded"]

    for arm in ("unguarded", "generic_data_checks", "llm_judge", "final_check_only"):
        weaker = report.metrics.by_baseline[arm]
        assert weaker.all_fault_false_pass_rate.value is not None
        assert runtime.all_fault_false_pass_rate.value is not None
        assert weaker.all_fault_false_pass_rate.value > runtime.all_fault_false_pass_rate.value


def test_no_arm_rejects_the_correct_analysis(report: Any) -> None:
    for arm in report.arms:
        metrics = report.metrics.by_baseline[arm]
        assert metrics.false_positive_rate.value == 0.0


def test_the_repair_ablation_records_both_feedback_kinds(report: Any) -> None:
    kinds = {record.feedback_kind for record in report.repair_ablation}
    structured = [r for r in report.repair_ablation if r.feedback_kind == "structured_report"]
    generic = [r for r in report.repair_ablation if r.feedback_kind == "generic_error"]

    assert kinds == {"generic_error", "structured_report"}
    assert len(structured) == len(generic) == 4
    # The scripted agent was written to repair only from structured feedback; this asserts the
    # loop wiring, not a property of any model.
    assert all(record.resolved for record in structured)
    assert not any(record.resolved for record in generic)
    assert all(record.repair_attempts_used <= 2 for record in report.repair_ablation)


def test_report_round_trip_and_markdown_carry_no_local_data(report: Any) -> None:
    payload = report.model_dump_json(indent=2)
    markdown = RENDER_MARKDOWN(report)

    assert EXPERIMENT_REPORT.model_validate_json(payload) == report
    assert "LOCAL_PATH_SECRET" not in payload
    assert str(Path.home()) not in payload
    assert '"artifacts":' not in payload
    # The caveat now names what actually produced each column, so it must track the agent rather
    # than asserting a fixed sentence that stopped being true when the judge became a model.
    assert "written by a deterministic script, not a model" in markdown
    assert "not a security boundary against adversarial code" in markdown
    # A reader skimming only the table must not mistake the scripted rule for a model.
    assert "`llm_judge` (scripted rule, no model)" in markdown
    assert "excluded from the rates" in markdown


def test_experiment_main_writes_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_path = tmp_path / "test.root"
    write_root_file(root_path)
    source = synthetic_source(root_path)
    json_output = tmp_path / "results.json"
    markdown_output = tmp_path / "results.md"
    monkeypatch.setattr(AtlasGamGamSource, "official_wph125", classmethod(lambda cls, path: source))

    exit_code = MAIN(
        [
            "--input",
            str(root_path),
            "--repetitions",
            "1",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    generated = EXPERIMENT_REPORT.model_validate(json.loads(json_output.read_text("utf-8")))
    assert exit_code == 0
    assert markdown_output.read_text(encoding="utf-8") == RENDER_MARKDOWN(generated)


def test_saved_report_is_recomputed_from_its_own_records() -> None:
    saved = EXPERIMENT_REPORT.model_validate_json(SAVED_JSON.read_text(encoding="utf-8"))
    compute = cast(Callable[..., Any], NAMESPACE["compute_metrics"])

    metric_cases = tuple(case for case in saved.cases if case.case_id in set(saved.metric_case_ids))

    assert saved.environment["sciagentguard_version"] == "0.4.0.dev1"
    assert saved.agent["provider"] is None
    assert len(metric_cases) == len(saved.metric_case_ids)
    # Recompute over the arms the report was actually scored on. An arm that failed to answer
    # somewhere is recorded but unscored, so asking for it here would raise.
    assert compute(metric_cases, saved.declared_contract_ids, saved.metric_arms) == saved.metrics
    assert saved.metrics.by_baseline["runtime_guarded"].all_fault_false_pass_rate.value == 0.0


def test_saved_report_contains_no_local_runtime_data() -> None:
    payload = SAVED_JSON.read_text(encoding="utf-8")

    for fragment in (str(ROOT), str(Path.home()), "/Users/", ".cache/"):
        assert fragment not in payload


def test_a_real_judge_can_be_injected_and_names_itself(tmp_path: Path, source_root: Path) -> None:
    """The judge is swappable, and the table must say which one produced the column.

    A reader skimming the arm table has to be able to tell a declared rule from a model without
    reading the prose, so the label is rendered from the judge's recorded identity.
    """

    class _AlwaysRejects:
        def verdict(self, final_artifact: Any) -> bool:
            del final_artifact
            return False

    report = GENERATE_REPORT(
        synthetic_source(source_root),
        repetitions=1,
        timeout_seconds=120.0,
        judge=_AlwaysRejects(),
        judge_identity={"provider": "acme", "model_id": "acme-judge-1", "source": "test"},
    )

    assert report.judge["provider"] == "acme"
    assert "acme model" in str(report.parameters["judge_kind"])
    assert "`llm_judge` (acme-judge-1)" in RENDER_MARKDOWN(report)
    cases = {case.case_id: case for case in report.cases}
    assert not _run(cases["correct"], 0, "llm_judge").accepted


def test_a_failing_judge_leaves_the_arm_absent_rather_than_accepting(
    tmp_path: Path, source_root: Path
) -> None:
    """An unavailable judge must never be recorded as having approved anything."""

    class _Unavailable:
        def verdict(self, final_artifact: Any) -> bool:
            del final_artifact
            raise RuntimeError("judge unavailable")

    report = GENERATE_REPORT(
        synthetic_source(source_root),
        repetitions=1,
        timeout_seconds=120.0,
        judge=_Unavailable(),
        judge_identity={"provider": "acme", "model_id": "acme-judge-1", "source": "test"},
    )

    cases = {case.case_id: case for case in report.cases}
    correct = cases["correct"].repetitions[0]
    recorded = {run.baseline for run in correct.runs}

    assert "llm_judge" not in recorded
    assert "runtime_guarded" in recorded
    # Dropped from the rates rather than counted: a missing verdict is neither a catch nor a miss.
    assert "llm_judge" not in report.metric_arms
    assert "runtime_guarded" in report.metric_arms
    markdown = RENDER_MARKDOWN(report)
    assert "Not scored:" in markdown
    # The verdicts that did arrive stay visible even though the arm has no overall rate.
    assert "`llm_judge`" in markdown


def test_committed_evidence_declares_the_contracts_the_code_declares() -> None:
    """Committed evidence must describe the code that is committed beside it.

    This check exists because the opposite happened: a commit carried a new contract and evidence
    that predated it, and every test passed. The saved-report test compares the evidence against
    its own records, which stays self-consistent no matter how far the code moves.
    """

    saved = EXPERIMENT_REPORT.model_validate_json(SAVED_JSON.read_text(encoding="utf-8"))
    declared = {
        contract.contract_id
        for contracts in AtlasGamGamOpenDataAdapter(
            AtlasGamGamSource.official_wph125(Path("unused.root"))
        ).stage_contracts()
        for contract in contracts
    }

    assert set(saved.declared_contract_ids) == declared, (
        "regenerate benchmarks/results/atlas_agent_results.json: it was produced by a different "
        "contract set than the one this commit declares"
    )
