from __future__ import annotations

import json
import platform
import runpy
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks" / "hep_fixture_benchmark.py"
SAVED_JSON = ROOT / "benchmarks" / "results" / "hep_fixture_results.json"
NAMESPACE = runpy.run_path(str(BENCHMARK))
BENCHMARK_REPORT = cast(Any, NAMESPACE["BenchmarkReport"])
COMPUTE_METRICS = cast(Callable[[Sequence[Any], Sequence[str]], Any], NAMESPACE["compute_metrics"])
RENDER_MARKDOWN = cast(Callable[[Any], str], NAMESPACE["render_markdown"])


def load_saved_report() -> Any:
    return BENCHMARK_REPORT.model_validate_json(SAVED_JSON.read_text(encoding="utf-8"))


def test_saved_correctness_metrics_are_recomputed_from_case_records() -> None:
    report = load_saved_report()
    recomputed = COMPUTE_METRICS(report.cases, report.declared_contract_ids)

    assert recomputed == report.metrics
    assert report.metrics.detection_recall.value == 1.0
    assert report.metrics.precision.value == 1.0
    assert report.metrics.localization_rate.value == 1.0
    assert report.metrics.unguarded_false_pass_rate.value == 1.0
    assert report.metrics.guarded_false_pass_rate.value == 0.0
    assert report.metrics.false_positive_rate.value == 0.0
    assert report.metrics.contract_coverage.value == 1.0


def test_saved_cases_distinguish_unguarded_false_passes_from_guarded_detection() -> None:
    report = load_saved_report()
    valid_case = report.cases[0]
    fault_cases = report.cases[1:]

    assert valid_case.case_id == "none"
    assert valid_case.unguarded.completed is True
    assert valid_case.guarded.blocked is False
    assert len(valid_case.guarded.checkpoints) == 4
    assert len(fault_cases) == 8
    assert all(case.unguarded.completed for case in fault_cases)
    assert all(len(case.unguarded.completed_stages) == 4 for case in fault_cases)
    assert all(case.guarded.blocked for case in fault_cases)
    for case in fault_cases:
        failures = [
            outcome
            for checkpoint in case.guarded.checkpoints
            for outcome in checkpoint.outcomes
            if outcome.status == "fail"
        ]
        assert [failure.contract_id for failure in failures] == [case.expected_contract_id]
        assert failures[0].violation_stage == case.expected_stage


def test_saved_timing_is_finite_and_recomputable_from_raw_samples() -> None:
    report = load_saved_report()
    timing = report.timing

    assert len(timing.unguarded_samples_ms) == 200
    assert len(timing.guarded_samples_ms) == 200
    assert all(value >= 0.0 for value in timing.unguarded_samples_ms)
    assert all(value >= 0.0 for value in timing.guarded_samples_ms)
    assert timing.unguarded_median_ms == statistics.median(timing.unguarded_samples_ms)
    assert timing.guarded_median_ms == statistics.median(timing.guarded_samples_ms)
    assert timing.overhead_ratio == pytest.approx(
        timing.guarded_median_ms / timing.unguarded_median_ms
    )


def test_saved_markdown_can_be_rendered_from_the_saved_json() -> None:
    markdown = RENDER_MARKDOWN(load_saved_report())

    assert markdown.startswith("# Synthetic HEP fixture results\n")
    assert "## Correctness" in markdown
    assert "## Valid-path timing" in markdown


def test_saved_json_omits_host_user_paths_and_runtime_artifacts() -> None:
    contents = SAVED_JSON.read_text(encoding="utf-8")
    parsed = cast(dict[str, Any], json.loads(contents))

    assert str(ROOT) not in contents
    assert Path.home().name not in contents
    assert platform.node() not in contents
    assert "artifacts" not in contents
    assert set(parsed["environment"]) == {
        "sciagentguard_version",
        "python_version",
        "python_implementation",
    }


def test_benchmark_cli_writes_both_outputs_without_touching_saved_results(
    tmp_path: Path,
) -> None:
    json_output = tmp_path / "result.json"
    markdown_output = tmp_path / "result.md"
    saved_json_before = SAVED_JSON.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--warmup",
            "0",
            "--repetitions",
            "2",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    report = BENCHMARK_REPORT.model_validate_json(json_output.read_text(encoding="utf-8"))
    assert len(report.timing.unguarded_samples_ms) == 2
    assert markdown_output.read_text(encoding="utf-8") == RENDER_MARKDOWN(report)
    assert SAVED_JSON.read_bytes() == saved_json_before
