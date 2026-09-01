from __future__ import annotations

import json
import runpy
import statistics
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import DeePTBSi64Adapter, DeePTBSi64Source
from sciagentguard.core import ContractContext
from tests.integration._deeptb_hdf5 import write_test_source

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks" / "deeptb_si64_boundary_benchmark.py"
SAVED_JSON = ROOT / "benchmarks" / "results" / "deeptb_si64_boundary_results.json"
NAMESPACE = runpy.run_path(str(BENCHMARK))
BENCHMARK_REPORT = cast(Any, NAMESPACE["BenchmarkReport"])
COMPUTE_METRICS = cast(Callable[[Any, Any], Any], NAMESPACE["compute_metrics"])
GENERATE_REPORT = cast(Callable[..., Any], NAMESPACE["generate_report"])
RENDER_MARKDOWN = cast(Callable[[Any], str], NAMESPACE["render_markdown"])
MAIN = cast(Callable[[list[str]], int], NAMESPACE["main"])


def _saved_report() -> Any:
    return BENCHMARK_REPORT.model_validate_json(SAVED_JSON.read_text(encoding="utf-8"))


def _run(case: Any, repetition: int, baseline: str) -> Any:
    return next(run for run in case.repetitions[repetition].runs if run.baseline == baseline)


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    source = write_test_source(tmp_path / "LOCAL_PATH_SECRET")
    original_load = DeePTBSi64Adapter.load_context

    def load_with_sentinels(
        self: DeePTBSi64Adapter,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
    ) -> ContractContext:
        context = original_load(
            self,
            workflow_id=workflow_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        return replace(
            context,
            artifacts={**context.artifacts, "private_matrix": ((98765.4321,),)},
            provenance={**context.provenance, "private": "PROVENANCE_SECRET_SENTINEL"},
            config={"credential": "CONFIG_SECRET_SENTINEL"},
        )

    monkeypatch.setattr(DeePTBSi64Adapter, "load_context", load_with_sentinels)
    return GENERATE_REPORT(
        source,
        correctness_repetitions=2,
        warmup_triplets=0,
        measured_triplets=2,
    )


def test_three_baselines_record_covered_faults_and_declared_gaps(report: Any) -> None:
    cases = {case.case_id: case for case in report.cases}
    detectable = (
        "missing_hamiltonian_inverse",
        "indefinite_overlap",
        "source_identity_drift",
    )
    gaps = ("atomic_species_drift", "hermitian_content_drift")

    assert tuple(cases) == ("none", *detectable, *gaps)
    for repetition in range(2):
        assert all(
            _run(cases["none"], repetition, baseline).accepted
            for baseline in NAMESPACE["_BASELINES"]
        )
        for case_id in detectable:
            assert _run(cases[case_id], repetition, "unguarded").accepted
            assert not _run(cases[case_id], repetition, "final_check_only").accepted
            assert not _run(cases[case_id], repetition, "runtime_guarded").accepted
        for case_id in gaps:
            assert all(
                _run(cases[case_id], repetition, baseline).accepted
                for baseline in NAMESPACE["_BASELINES"]
            )


def test_faults_are_localized_to_the_expected_contract_and_stage(report: Any) -> None:
    for case in report.cases:
        if not case.expected_detectable:
            continue
        run = _run(case, 0, "runtime_guarded")
        failed = tuple(
            result
            for checkpoint in run.trace.checkpoints
            for result in checkpoint.results
            if result.violation is not None
        )

        assert len(failed) == 1
        assert failed[0].contract_id == case.expected_contract_id
        assert failed[0].violation.stage == "post_hamiltonian_load"


def test_metrics_are_recomputed_from_repetition_records(report: Any) -> None:
    assert COMPUTE_METRICS(report.cases, report.declared_contract_ids) == report.metrics
    unguarded = report.metrics.by_baseline["unguarded"]
    final = report.metrics.by_baseline["final_check_only"]
    runtime = report.metrics.by_baseline["runtime_guarded"]

    assert unguarded.detection_recall.value == 0.0
    assert unguarded.all_fault_false_pass_rate.value == 1.0
    for guarded in (final, runtime):
        assert guarded.detection_recall.value == 1.0
        assert guarded.precision.value == 1.0
        assert guarded.localization_rate.value == 1.0
        assert guarded.all_fault_false_pass_rate.value == 0.4
        assert guarded.false_positive_rate.value == 0.0
        assert guarded.contract_coverage.value == 1.0
        assert guarded.gap_probe_acceptance_rate.value == 1.0
    assert report.metrics.final_runtime_agreement.value == 1.0


def test_report_round_trip_and_markdown_preserve_safe_evidence(report: Any) -> None:
    payload = report.model_dump_json(indent=2)
    restored = BENCHMARK_REPORT.model_validate_json(payload)
    markdown = RENDER_MARKDOWN(report)

    assert restored == report
    for private_fragment in (
        "LOCAL_PATH_SECRET",
        "PROVENANCE_SECRET_SENTINEL",
        "CONFIG_SECRET_SENTINEL",
        "98765.4321",
        str(Path.home()),
    ):
        assert private_fragment not in payload
        assert private_fragment not in markdown
    assert '"artifacts":' not in payload
    assert '"atomic_numbers":' not in payload
    assert "structure-to-source binding" in markdown
    assert "remained domain-specific" in markdown


def test_timing_is_finite_and_recomputable(report: Any) -> None:
    unguarded_median = report.timing.modes["unguarded"].median_ms

    for baseline in NAMESPACE["_BASELINES"]:
        mode = report.timing.modes[baseline]
        assert len(mode.samples_ms) == 2
        assert all(value >= 0.0 for value in mode.samples_ms)
        assert mode.median_ms == statistics.median(mode.samples_ms)
        assert mode.ratio_to_unguarded == pytest.approx(mode.median_ms / unguarded_median)


def test_benchmark_main_writes_outputs_from_a_supplied_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_test_source(tmp_path / "source")
    json_output = tmp_path / "results.json"
    markdown_output = tmp_path / "results.md"
    monkeypatch.setattr(
        DeePTBSi64Source,
        "official_test_sample",
        classmethod(lambda cls, path: source),
    )

    exit_code = MAIN(
        [
            "--input",
            str(source.directory),
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
    assert saved.provenance["sample_id"] == "dptb/tests/data/e3_band/data/Si64.0"
    assert saved.provenance["atom_count"] == 64
    assert saved.provenance["hamiltonian_block_count"] == 9_434
    assert saved.provenance["overlap_block_count"] == 5_562
    assert saved.parameters["correctness_repetitions"] == 5
    assert saved.parameters["measured_triplets"] == 10
    assert COMPUTE_METRICS(saved.cases, saved.declared_contract_ids) == saved.metrics
    assert saved.metrics.by_baseline["unguarded"].all_fault_false_pass_rate.value == 1.0
    assert saved.metrics.by_baseline["final_check_only"].all_fault_false_pass_rate.value == 0.4
    assert saved.metrics.by_baseline["runtime_guarded"].all_fault_false_pass_rate.value == 0.4
    assert saved.metrics.final_runtime_agreement.value == 1.0


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

    for private_fragment in (str(ROOT), str(Path.home()), "/Users/", ".cache/"):
        assert private_fragment not in payload
    assert '"artifacts":' not in payload
    assert '"atomic_numbers":' not in payload
    assert set(_saved_report().environment) == {
        "sciagentguard_version",
        "python_version",
        "python_implementation",
        "numpy_version",
        "h5py_version",
    }


def test_committed_evidence_declares_the_contracts_the_code_declares() -> None:
    """Committed evidence must describe the code committed beside it.

    The saved-report check compares the evidence against its own records, which stays
    self-consistent however far the code moves. This one compares it against the code, which is
    the check that would have caught a commit shipping a new contract with older evidence.
    """

    saved = _saved_report()
    contracts = cast(Callable[[Any], Any], NAMESPACE["_contracts"])
    declared = {
        contract.contract_id
        for contract in contracts(
            DeePTBSi64Adapter(DeePTBSi64Source.official_test_sample(Path("unused")))
        )
    }

    assert set(saved.declared_contract_ids) == declared, (
        "regenerate benchmarks/results/deeptb_si64_boundary_results.json: it was produced by a "
        "different contract set than the one this commit declares"
    )
