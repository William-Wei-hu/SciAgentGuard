from __future__ import annotations

import json
import runpy
import statistics
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from tests.integration._atlas_root import synthetic_source, write_root_file

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks" / "atlas_gamgam_boundary_benchmark.py"
SAVED_JSON = ROOT / "benchmarks" / "results" / "atlas_gamgam_boundary_results.json"
NAMESPACE = runpy.run_path(str(BENCHMARK))
BENCHMARK_REPORT = cast(Any, NAMESPACE["BenchmarkReport"])
COMPUTE_METRICS = cast(Callable[[Any, Any], Any], NAMESPACE["compute_metrics"])
GENERATE_REPORT = cast(Callable[..., Any], NAMESPACE["generate_report"])
RENDER_MARKDOWN = cast(Callable[[Any], str], NAMESPACE["render_markdown"])
MAIN = cast(Callable[[list[str]], int], NAMESPACE["main"])


def _saved_report() -> Any:
    return BENCHMARK_REPORT.model_validate_json(SAVED_JSON.read_text(encoding="utf-8"))


@pytest.fixture
def report(tmp_path: Path) -> Any:
    root_path = tmp_path / "LOCAL_PATH_SECRET" / "test.root"
    root_path.parent.mkdir()
    write_root_file(root_path)
    source = replace(
        synthetic_source(root_path),
        generator="LOCAL_PROVENANCE_SECRET",
    )
    return GENERATE_REPORT(
        source,
        correctness_repetitions=2,
        warmup_triplets=0,
        measured_triplets=2,
    )


def _run(case: Any, repetition: int, baseline: str) -> Any:
    return next(run for run in case.repetitions[repetition].runs if run.baseline == baseline)


VALID_CASES = ("none_range_0", "none_range_1", "none_range_2")
CRASHING_CASES = ("missing_translated_branch",)
CONTRACT_DETECTED_CASES = (
    "nonfinite_weights",
    "zero_weights",
    "photon_count_mismatch",
    "missing_event_provenance",
    "source_identity_drift",
    "normalization_scale_drift",
    "region_overlap",
    # Both former gap probes are now detected. The photon-scale probe became visible once the
    # workflow gained downstream checkpoints, and the weight-scale probe once the contracts began
    # comparing an artifact's numbers against the source they claim to come from. Neither was
    # closed by a check written for it.
    "weight_scale_gap",
    "photon_scale_gap",
)
GAP_CASES: tuple[str, ...] = ()


def test_three_baselines_record_detectable_faults_and_declared_gaps(report: Any) -> None:
    cases = {case.case_id: case for case in report.cases}

    assert tuple(cases) == (
        *VALID_CASES,
        *CRASHING_CASES,
        *CONTRACT_DETECTED_CASES,
        *GAP_CASES,
    )
    for repetition in range(2):
        for case_id in VALID_CASES:
            assert all(
                _run(cases[case_id], repetition, baseline).accepted
                for baseline in NAMESPACE["_BASELINES"]
            )
        for case_id in CRASHING_CASES:
            # Removing a required column breaks the analysis code itself, so the two baselines
            # that execute every stage stop with a software error rather than a contract.
            for baseline in ("unguarded", "final_check_only"):
                run = _run(cases[case_id], repetition, baseline)
                assert run.crashed
                assert not run.accepted
                assert run.crash_type == "ValueError"
            runtime = _run(cases[case_id], repetition, "runtime_guarded")
            assert not runtime.crashed
            assert not runtime.accepted
        for case_id in CONTRACT_DETECTED_CASES:
            assert not _run(cases[case_id], repetition, "runtime_guarded").accepted
            assert not _run(cases[case_id], repetition, "runtime_guarded").crashed
        for case_id in GAP_CASES:
            assert all(
                _run(cases[case_id], repetition, baseline).accepted
                for baseline in NAMESPACE["_BASELINES"]
            )


def test_this_experiment_no_longer_declares_a_coverage_gap(report: Any) -> None:
    """No accepted probe remains -- which measures this fault taxonomy, not the contract map.

    The Milestone 6B pilot showed a real model writing wrong analyses that none of these probes
    describes and every contract here accepted, so an empty gap list must not be read as coverage.
    """

    gap_rate = report.metrics.by_baseline["runtime_guarded"].gap_probe_acceptance_rate

    assert gap_rate.denominator == 0
    assert "documents no coverage gap" in RENDER_MARKDOWN(report)
    assert "cannot find a gap its own faults do not contain" in RENDER_MARKDOWN(report)


def test_runtime_guarding_diverges_from_final_validation(report: Any) -> None:
    """Milestone 5 exit condition: the two guarded baselines must actually disagree."""

    divergence = report.metrics.runtime_vs_final_divergence
    divergent = NAMESPACE["divergent_case_ids"](report.cases)
    cases = {case.case_id: case for case in report.cases}

    assert divergence is not None
    assert divergence.numerator > 0, "runtime and final validation must disagree somewhere"
    for case_id in divergent:
        assert not _run(cases[case_id], 0, "runtime_guarded").accepted
        assert _run(cases[case_id], 0, "final_check_only").accepted

    # The two declared late-invisible faults must be among the divergent cases: they are the
    # ones whose final artifact stays plausible rather than merely unavailable.
    for case_id in NAMESPACE["LATE_INVISIBLE_CASE_IDS"]:
        assert case_id in divergent
        assert cases[case_id].late_invisible_reason


def test_late_invisible_faults_leave_the_final_artifact_plausible(report: Any) -> None:
    cases = {case.case_id: case for case in report.cases}

    for case_id in NAMESPACE["LATE_INVISIBLE_CASE_IDS"]:
        case = cases[case_id]
        assert case.injection_stage in {"post_selection", "post_histogram"}
        assert case.expected_contract_id is not None
        assert case.late_invisible_reason is not None
        for repetition in range(2):
            assert _run(case, repetition, "final_check_only").accepted
            assert not _run(case, repetition, "runtime_guarded").accepted


def test_metrics_are_recomputed_from_repetition_records(report: Any) -> None:
    assert COMPUTE_METRICS(report.cases, report.declared_contract_ids) == report.metrics
    runtime = report.metrics.by_baseline["runtime_guarded"]
    final = report.metrics.by_baseline["final_check_only"]

    assert runtime.false_positive_rate.value == 0.0
    assert final.false_positive_rate.value == 0.0
    # No probe in this set is accepted any more, so the rate has no denominator.
    assert runtime.gap_probe_acceptance_rate.denominator == 0
    assert runtime.all_fault_false_pass_rate.value < final.all_fault_false_pass_rate.value
    assert report.metrics.checkpoint_attribution is not None
    # Not every fault is caught where it was injected. Rescaled weights and rescaled photon
    # momenta are both injected at `post_load` and only become observable further down, so
    # attribution is deliberately below one: it measures where a fault becomes visible, not
    # whether it was found.
    attribution = report.metrics.checkpoint_attribution["runtime_guarded"]
    assert attribution.value is not None
    assert 0.0 < attribution.value < 1.0
    assert attribution.denominator == attribution.numerator + 4


def test_valid_cases_use_three_disjoint_ranges_of_one_source(report: Any) -> None:
    valid = [case for case in report.cases if not case.fault_injected]

    assert len(valid) == 3
    assert report.parameters["valid_range_count"] == 3
    assert report.metrics.by_baseline["runtime_guarded"].false_positive_rate.denominator == 6


def test_report_round_trip_and_markdown_preserve_safe_evidence(report: Any) -> None:
    payload = report.model_dump_json(indent=2)
    restored = BENCHMARK_REPORT.model_validate_json(payload)
    markdown = RENDER_MARKDOWN(report)

    assert restored == report
    assert "LOCAL_PATH_SECRET" not in payload
    assert "LOCAL_PROVENANCE_SECRET" not in payload
    assert str(Path.home()) not in payload
    assert '"artifacts":' not in payload
    assert "Runtime-vs-final divergence" in markdown
    assert "documents no coverage gap" in markdown
    assert "regression check" in markdown
    assert "not a proof that post-hoc validation is inherently unable" in markdown


def test_timing_is_finite_and_recomputable(report: Any) -> None:
    timing = report.timing
    unguarded_median = timing.modes["unguarded"].median_ms

    for baseline in NAMESPACE["_BASELINES"]:
        mode = timing.modes[baseline]
        assert len(mode.samples_ms) == 2
        assert all(value >= 0.0 for value in mode.samples_ms)
        assert mode.median_ms == statistics.median(mode.samples_ms)
        assert mode.ratio_to_unguarded == pytest.approx(mode.median_ms / unguarded_median)


def test_benchmark_main_writes_outputs_from_a_supplied_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "test.root"
    write_root_file(root_path)
    source = synthetic_source(root_path)
    json_output = tmp_path / "results.json"
    markdown_output = tmp_path / "results.md"

    monkeypatch.setattr(
        AtlasGamGamSource,
        "official_wph125",
        classmethod(lambda cls, path: source),
    )
    exit_code = MAIN(
        [
            "--input",
            str(root_path),
            "--correctness-repetitions",
            "1",
            "--warmup",
            "0",
            "--timing-repetitions",
            "1",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    parsed = cast(dict[str, Any], json.loads(json_output.read_text(encoding="utf-8")))
    generated = BENCHMARK_REPORT.model_validate(parsed)
    assert exit_code == 0
    assert markdown_output.read_text(encoding="utf-8") == RENDER_MARKDOWN(generated)


def test_saved_report_is_recomputed_from_the_official_source_runs() -> None:
    saved = _saved_report()

    assert saved.environment["sciagentguard_version"] == "0.4.0.dev1"
    assert saved.provenance["record_id"] == "atlas-15006"
    assert saved.provenance["event_count"] == 113_765
    assert saved.parameters["correctness_repetitions"] == 5
    assert saved.parameters["measured_triplets"] == 10
    assert saved.stages == ("post_load", "post_selection", "post_histogram", "post_yield")
    assert COMPUTE_METRICS(saved.cases, saved.declared_contract_ids) == saved.metrics
    assert saved.metrics.by_baseline["unguarded"].all_fault_false_pass_rate.value == 0.9
    assert saved.metrics.runtime_vs_final_divergence is not None
    assert saved.metrics.runtime_vs_final_divergence.numerator > 0
    assert set(NAMESPACE["LATE_INVISIBLE_CASE_IDS"]).issubset(
        set(NAMESPACE["divergent_case_ids"](saved.cases))
    )


def test_saved_report_timing_and_markdown_are_reproducible() -> None:
    saved = _saved_report()
    unguarded_median = saved.timing.modes["unguarded"].median_ms

    for baseline in NAMESPACE["_BASELINES"]:
        mode = saved.timing.modes[baseline]
        assert len(mode.samples_ms) == 10
        assert mode.median_ms == statistics.median(mode.samples_ms)
        assert mode.ratio_to_unguarded == pytest.approx(mode.median_ms / unguarded_median)


def test_saved_report_contains_no_local_runtime_data() -> None:
    payload = SAVED_JSON.read_text(encoding="utf-8")

    for private_fragment in (str(ROOT), str(Path.home()), "/Users/", ".cache/", "root://"):
        assert private_fragment not in payload
    assert '"artifacts":' not in payload
    assert set(_saved_report().environment) == {
        "sciagentguard_version",
        "python_version",
        "python_implementation",
        "numpy_version",
        "awkward_version",
        "uproot_version",
    }


def test_committed_evidence_declares_the_contracts_the_code_declares() -> None:
    """Committed evidence must describe the code committed beside it.

    A commit once carried a new contract and evidence produced before it, and every test passed:
    the saved-report check compares evidence against its own records, which stays self-consistent
    however far the code moves. This compares it against the code instead.
    """

    saved = _saved_report()
    declared = {
        contract.contract_id
        for contracts in AtlasGamGamOpenDataAdapter(
            AtlasGamGamSource.official_wph125(Path("unused.root"))
        ).stage_contracts()
        for contract in contracts
    }

    assert set(saved.declared_contract_ids) == declared, (
        "regenerate benchmarks/results/atlas_gamgam_boundary_results.json: it was produced by a "
        "different contract set than the one this commit declares"
    )
