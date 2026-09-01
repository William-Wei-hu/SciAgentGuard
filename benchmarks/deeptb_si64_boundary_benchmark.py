"""Compare guard baselines at the DeePTB Si64 Hamiltonian-load boundary."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sciagentguard import __version__
from sciagentguard._boundary_benchmark import (
    BASELINES,
    BaselineName,
    BaselineRunRecord,
    CaseRecord,
    CorrectnessMetrics,
    RatioMetric,
    RepetitionRecord,
    TimingResult,
    compute_metrics,
    find_run,
    run_baseline,
)
from sciagentguard._boundary_benchmark import (
    measure_timing as measure_boundary_timing,
)
from sciagentguard.adapters import DeePTBSi64Adapter, DeePTBSi64Source
from sciagentguard.core import ContractContext, ScientificContract, SemanticFaultInjector
from sciagentguard.packs.materials import (
    DeePTBAtomicSpeciesDriftInjector,
    DeePTBHermitianContentDriftInjector,
    DeePTBIndefiniteOverlapInjector,
    DeePTBMissingHamiltonianInverseInjector,
    DeePTBSourceIdentityDriftInjector,
)
from sciagentguard.runtime import GuardedExecutor, GuardedWorkflowRunner

BENCHMARK_ID: Literal["deeptb_si64_boundary_comparison"] = "deeptb_si64_boundary_comparison"
DEFAULT_INPUT = Path(".cache/deeptb-si64")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/deeptb_si64_boundary_results.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/deeptb_boundary.md")
_BASELINES = BASELINES


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["deeptb_si64_boundary_comparison"] = BENCHMARK_ID
    generated_at: datetime
    provenance: dict[str, str | int]
    environment: dict[str, str]
    parameters: dict[str, str | int]
    declared_contract_ids: tuple[str, ...]
    cases: tuple[CaseRecord, ...]
    metrics: CorrectnessMetrics
    timing: TimingResult


@dataclass(frozen=True, slots=True)
class _FaultCase:
    case_id: str
    injector: SemanticFaultInjector
    expected_contract_id: str | None

    @property
    def expected_detectable(self) -> bool:
        return self.expected_contract_id is not None


FAULT_CASES = (
    _FaultCase(
        "missing_hamiltonian_inverse",
        DeePTBMissingHamiltonianInverseInjector(),
        "materials.hamiltonian.block_hermiticity",
    ),
    _FaultCase(
        "indefinite_overlap",
        DeePTBIndefiniteOverlapInjector(),
        "materials.overlap.gamma_positive_definite",
    ),
    _FaultCase(
        "source_identity_drift",
        DeePTBSourceIdentityDriftInjector(),
        "materials.deeptb.source_identity",
    ),
    _FaultCase("atomic_species_drift", DeePTBAtomicSpeciesDriftInjector(), None),
    _FaultCase("hermitian_content_drift", DeePTBHermitianContentDriftInjector(), None),
)


def _contracts(adapter: DeePTBSi64Adapter) -> tuple[ScientificContract, ...]:
    return adapter.checkpoint(
        workflow_id="deeptb-si64-boundary-benchmark",
        run_id="contract-map",
        attempt_id="attempt-0",
    ).contracts


def _context_for_run(
    base_context: ContractContext,
    case: _FaultCase | None,
    *,
    run_id: str,
) -> ContractContext:
    context = replace(base_context, run_id=run_id, attempt_id="attempt-0")
    return context if case is None else case.injector.inject(context)


def _case_record(
    base_context: ContractContext,
    contracts: tuple[ScientificContract, ...],
    case: _FaultCase | None,
    repetitions: int,
) -> CaseRecord:
    case_id = "none" if case is None else case.case_id
    records: list[RepetitionRecord] = []
    for repetition in range(repetitions):
        runs: list[BaselineRunRecord] = []
        for baseline in _BASELINES:
            context = _context_for_run(
                base_context,
                case,
                run_id=f"{case_id}-{baseline}-{repetition}",
            )
            runs.append(run_baseline(baseline, context, contracts))
        records.append(RepetitionRecord(repetition=repetition, runs=tuple(runs)))

    return CaseRecord(
        case_id=case_id,
        fault_injected=case is not None,
        expected_detectable=case.expected_detectable if case is not None else False,
        expected_contract_id=case.expected_contract_id if case is not None else None,
        expected_stage="post_hamiltonian_load" if case is not None else None,
        repetitions=tuple(records),
    )


def _timing_targets(
    adapter: DeePTBSi64Adapter,
) -> dict[BaselineName, Callable[[], None]]:
    def unguarded() -> None:
        adapter.load_context(
            workflow_id="deeptb-si64-boundary-benchmark",
            run_id="timing-unguarded",
            attempt_id="attempt-0",
        )

    def final_check_only() -> None:
        context = adapter.load_context(
            workflow_id="deeptb-si64-boundary-benchmark",
            run_id="timing-final-check",
            attempt_id="attempt-0",
        )
        execution = GuardedExecutor().execute(lambda: context, _contracts(adapter))
        if execution.trace.blocked:
            raise ValueError("the valid timing source failed final-check-only validation")

    def runtime_guarded() -> None:
        execution = GuardedWorkflowRunner().execute(
            (
                adapter.checkpoint(
                    workflow_id="deeptb-si64-boundary-benchmark",
                    run_id="timing-runtime-guarded",
                    attempt_id="attempt-0",
                ),
            )
        )
        if execution.trace.blocked:
            raise ValueError("the valid timing source failed runtime validation")

    return {
        "unguarded": unguarded,
        "final_check_only": final_check_only,
        "runtime_guarded": runtime_guarded,
    }


def measure_timing(
    adapter: DeePTBSi64Adapter,
    warmup_triplets: int,
    measured_triplets: int,
) -> TimingResult:
    return measure_boundary_timing(
        _timing_targets(adapter),
        warmup_triplets,
        measured_triplets,
    )


def generate_report(
    source: DeePTBSi64Source,
    *,
    correctness_repetitions: int = 5,
    warmup_triplets: int = 2,
    measured_triplets: int = 10,
) -> BenchmarkReport:
    if correctness_repetitions < 1:
        raise ValueError("correctness_repetitions must be positive")

    adapter = DeePTBSi64Adapter(source)
    base_context = adapter.load_context(
        workflow_id="deeptb-si64-boundary-benchmark",
        run_id="verified-source",
        attempt_id="attempt-0",
    )
    contracts = _contracts(adapter)
    declared_contract_ids = tuple(contract.contract_id for contract in contracts)
    cases = (
        _case_record(base_context, contracts, None, correctness_repetitions),
        *(
            _case_record(base_context, contracts, case, correctness_repetitions)
            for case in FAULT_CASES
        ),
    )

    return BenchmarkReport(
        generated_at=datetime.now(timezone.utc),
        provenance={
            "source_type": "public_repository_sample",
            "project": "DeePTB",
            "repository": source.repository,
            "commit": source.commit,
            "license": source.license_id,
            "sample_id": source.sample_id,
            "atom_count": source.atom_count,
            "orbitals_per_atom": source.orbitals_per_atom,
            "hamiltonian_block_count": source.hamiltonian_block_count,
            "overlap_block_count": source.overlap_block_count,
            "hamiltonian_sha256": source.hamiltonian_sha256,
            "overlap_sha256": source.overlap_sha256,
            "atomic_numbers_sha256": source.atomic_numbers_sha256,
        },
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": version("numpy"),
            "h5py_version": version("h5py"),
        },
        parameters={
            "correctness_repetitions": correctness_repetitions,
            "warmup_triplets": warmup_triplets,
            "measured_triplets": measured_triplets,
            "correctness_source_loads": 1,
            "timing_order": "rotating three-mode order",
            "timing_scope": "valid full source verification and HDF5 load",
            "checkpoint_topology": "single post_hamiltonian_load boundary",
        },
        declared_contract_ids=declared_contract_ids,
        cases=cases,
        metrics=compute_metrics(cases, declared_contract_ids),
        timing=measure_timing(adapter, warmup_triplets, measured_triplets),
    )


def _format_metric(metric: RatioMetric) -> str:
    if metric.value is None:
        return f"{metric.numerator}/{metric.denominator} (n/a)"
    return f"{metric.numerator}/{metric.denominator} ({metric.value:.1%})"


def render_markdown(report: BenchmarkReport) -> str:
    labels: tuple[tuple[str, str], ...] = (
        ("detection_recall", "Detectable-fault recall"),
        ("precision", "Detection precision"),
        ("localization_rate", "Correct localization"),
        ("all_fault_false_pass_rate", "All-fault false-pass rate"),
        ("false_positive_rate", "Valid-case false-positive rate"),
        ("contract_coverage", "Contract coverage"),
        ("gap_probe_acceptance_rate", "Gap-probe acceptance rate"),
    )
    lines = [
        "# DeePTB Si64 boundary results",
        "",
        (
            f"Generated with SciAgentGuard {report.environment['sciagentguard_version']} from "
            f"DeePTB sample `{report.provenance['sample_id']}` at commit "
            f"`{report.provenance['commit']}`. This is a verified test-data boundary, not "
            "model training, inference, or a materials prediction."
        ),
        "",
        "## Correctness",
        "",
        "| Metric | Unguarded | Final check only | Runtime guarded |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field_name, label in labels:
        values = [
            _format_metric(getattr(report.metrics.by_baseline[baseline], field_name))
            for baseline in _BASELINES
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend(
        [
            "",
            "| Case | Expected detector | Unguarded accepted | Final accepted | Runtime accepted |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case in report.cases:
        expected = case.expected_contract_id or (
            "documented gap" if case.fault_injected else "valid source"
        )
        accepted = []
        for baseline in _BASELINES:
            runs = tuple(find_run(repetition, baseline) for repetition in case.repetitions)
            accepted.append(f"{sum(run.accepted for run in runs)}/{len(runs)}")
        lines.append(
            f"| `{case.case_id}` | `{expected}` | {accepted[0]} | {accepted[1]} | {accepted[2]} |"
        )

    timing = report.timing
    lines.extend(
        [
            "",
            "## Valid-source timing",
            "",
            "| Baseline | Median | Ratio to unguarded |",
            "| --- | ---: | ---: |",
            *(
                f"| `{baseline}` | {timing.modes[baseline].median_ms:.3f} ms | "
                f"{timing.modes[baseline].ratio_to_unguarded:.3f}x |"
                for baseline in _BASELINES
            ),
            "",
            (
                f"Timing uses {report.parameters['warmup_triplets']} warm-up triplets and "
                f"{report.parameters['measured_triplets']} measured triplets in rotating order. "
                "Every sample verifies all three files and loads the valid HDF5 boundary; "
                "faulted runs are excluded. Raw samples are preserved in the JSON report."
            ),
            "",
            "## Negative results and cross-domain findings",
            "",
            (
                "Final-check-only and runtime-guarded agreed on "
                f"{_format_metric(report.metrics.final_runtime_agreement)} of runs. This is "
                "expected because the integration has one `post_hamiltonian_load` checkpoint; "
                "it does not demonstrate an intermediate-checkpoint advantage."
            ),
            "",
            (
                "Atomic-species drift and a coordinated Hermitian block-pair change pass the "
                "current contract map. They expose missing structure-to-source binding and "
                "reference-value or band-reconstruction coverage."
            ),
            "",
            (
                "The core context, contract, injector, guarded execution, trace, and comparison "
                "method transferred from HEP. ROOT/HDF5 parsing, matrix reconstruction, "
                "scientific evidence, and fault meaning remained domain-specific."
            ),
            "",
            (
                "File corruption, checksum drift, malformed HDF5 groups, block-key errors, and "
                "shape errors remain Adapter input failures and are not counted as scientific "
                "contract detections. Results apply only to the pinned Si64 sample and boundary."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--correctness-repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timing-repetitions", type=int, default=10)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = generate_report(
        DeePTBSi64Source.official_test_sample(args.input),
        correctness_repetitions=args.correctness_repetitions,
        warmup_triplets=args.warmup,
        measured_triplets=args.timing_repetitions,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
