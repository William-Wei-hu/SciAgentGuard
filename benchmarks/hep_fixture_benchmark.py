"""Measure deterministic fault detection and valid-path guard overhead."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sciagentguard import __version__
from sciagentguard.core import ContractStatus, SemanticFaultInjector
from sciagentguard.packs.hep import (
    EmptySelectionInjector,
    MissingBranchInjector,
    NonfiniteWeightsInjector,
    SplitLeakageInjector,
    SyntheticHEPWorkflow,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    WrongNormalizationInjector,
    ZeroWeightsInjector,
)
from sciagentguard.runtime import GuardedWorkflowRunner

BENCHMARK_ID = "hep_fixture_guarded_detection"
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/hep_fixture_results.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/hep_fixture.md")


class ContractOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str
    status: ContractStatus
    violation_stage: str | None


class CheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    outcomes: tuple[ContractOutcome, ...]


class UnguardedCaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completed: bool
    completed_stages: tuple[str, ...]


class GuardedCaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    blocked: bool
    checkpoints: tuple[CheckpointRecord, ...]


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    fault_injected: bool
    expected_contract_id: str | None
    expected_stage: str | None
    unguarded: UnguardedCaseRecord
    guarded: GuardedCaseRecord


class RatioMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    numerator: int
    denominator: int
    value: float


class CorrectnessMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_recall: RatioMetric
    precision: RatioMetric
    localization_rate: RatioMetric
    unguarded_false_pass_rate: RatioMetric
    guarded_false_pass_rate: RatioMetric
    false_positive_rate: RatioMetric
    contract_coverage: RatioMetric


class TimingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    unit: Literal["milliseconds"] = "milliseconds"
    timing_case: Literal["none"] = "none"
    unguarded_samples_ms: tuple[float, ...]
    guarded_samples_ms: tuple[float, ...]
    unguarded_median_ms: float
    guarded_median_ms: float
    overhead_ratio: float


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: Literal["hep_fixture_guarded_detection"] = BENCHMARK_ID
    generated_at: datetime
    provenance: dict[str, str]
    environment: dict[str, str]
    parameters: dict[str, int | str]
    declared_contract_ids: tuple[str, ...]
    cases: tuple[CaseRecord, ...]
    metrics: CorrectnessMetrics
    timing: TimingResult


@dataclass(frozen=True, slots=True)
class _FaultCase:
    case_id: str
    injector: SemanticFaultInjector
    expected_contract_id: str
    expected_stage: str


FAULT_CASES = (
    _FaultCase(
        "missing_branch",
        MissingBranchInjector(),
        "hep.schema.required_branches",
        "post_load",
    ),
    _FaultCase(
        "zero_weights",
        ZeroWeightsInjector(),
        "hep.weights.nonzero_support",
        "post_load",
    ),
    _FaultCase(
        "nonfinite_weights",
        NonfiniteWeightsInjector(),
        "hep.weights.finite",
        "post_load",
    ),
    _FaultCase(
        "unit_scale_error",
        UnitScaleErrorInjector(),
        "hep.kinematics.jet_pt_range",
        "post_load",
    ),
    _FaultCase(
        "undeclared_synthetic_data",
        UndeclaredSyntheticDataInjector(),
        "hep.provenance.events_declared",
        "post_load",
    ),
    _FaultCase(
        "empty_selection",
        EmptySelectionInjector(),
        "hep.selection.nonempty",
        "post_selection",
    ),
    _FaultCase(
        "split_leakage",
        SplitLeakageInjector(),
        "hep.splits.disjoint_event_ids",
        "post_split",
    ),
    _FaultCase(
        "wrong_normalization",
        WrongNormalizationInjector(),
        "hep.normalization.yield_consistent",
        "post_normalization",
    ),
)


def _unguarded_record(workflow: SyntheticHEPWorkflow) -> UnguardedCaseRecord:
    completed_stages = tuple(checkpoint.step().stage for checkpoint in workflow.checkpoints())
    return UnguardedCaseRecord(completed=True, completed_stages=completed_stages)


def _guarded_record(workflow: SyntheticHEPWorkflow) -> GuardedCaseRecord:
    trace = GuardedWorkflowRunner().execute(workflow.checkpoints()).trace
    checkpoints = tuple(
        CheckpointRecord(
            stage=checkpoint.stage,
            outcomes=tuple(
                ContractOutcome(
                    contract_id=result.contract_id,
                    status=result.status,
                    violation_stage=(
                        result.violation.stage if result.violation is not None else None
                    ),
                )
                for result in checkpoint.results
            ),
        )
        for checkpoint in trace.checkpoints
    )
    return GuardedCaseRecord(blocked=trace.blocked, checkpoints=checkpoints)


def _case_record(
    case_id: str,
    injector: SemanticFaultInjector | None,
    expected_contract_id: str | None,
    expected_stage: str | None,
) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        fault_injected=injector is not None,
        expected_contract_id=expected_contract_id,
        expected_stage=expected_stage,
        unguarded=_unguarded_record(SyntheticHEPWorkflow(fault=injector)),
        guarded=_guarded_record(SyntheticHEPWorkflow(fault=injector)),
    )


def _ratio(numerator: int, denominator: int) -> RatioMetric:
    if denominator <= 0:
        raise ValueError("metric denominators must be positive")
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator,
    )


def compute_metrics(
    cases: Sequence[CaseRecord], declared_contract_ids: Sequence[str]
) -> CorrectnessMetrics:
    valid_cases = tuple(case for case in cases if not case.fault_injected)
    fault_cases = tuple(case for case in cases if case.fault_injected)
    if len(valid_cases) != 1 or not fault_cases:
        raise ValueError("the benchmark requires one valid case and at least one fault case")

    failed_outcomes = tuple(
        (case, outcome)
        for case in cases
        for checkpoint in case.guarded.checkpoints
        for outcome in checkpoint.outcomes
        if outcome.status is ContractStatus.FAIL
    )
    correct_reports = sum(
        outcome.contract_id == case.expected_contract_id
        and outcome.violation_stage == case.expected_stage
        for case, outcome in failed_outcomes
        if case.fault_injected
    )
    detected_faults = sum(case.guarded.blocked for case in fault_cases)
    localized_faults = sum(
        any(
            outcome.status is ContractStatus.FAIL
            and outcome.contract_id == case.expected_contract_id
            and outcome.violation_stage == case.expected_stage
            for checkpoint in case.guarded.checkpoints
            for outcome in checkpoint.outcomes
        )
        for case in fault_cases
    )
    executed_contract_ids = {
        outcome.contract_id
        for case in cases
        for checkpoint in case.guarded.checkpoints
        for outcome in checkpoint.outcomes
    }
    covered_contracts = sum(
        contract_id in executed_contract_ids for contract_id in declared_contract_ids
    )

    return CorrectnessMetrics(
        detection_recall=_ratio(detected_faults, len(fault_cases)),
        precision=_ratio(correct_reports, len(failed_outcomes)),
        localization_rate=_ratio(localized_faults, len(fault_cases)),
        unguarded_false_pass_rate=_ratio(
            sum(case.unguarded.completed for case in fault_cases),
            len(fault_cases),
        ),
        guarded_false_pass_rate=_ratio(
            sum(not case.guarded.blocked for case in fault_cases),
            len(fault_cases),
        ),
        false_positive_rate=_ratio(
            sum(case.guarded.blocked for case in valid_cases),
            len(valid_cases),
        ),
        contract_coverage=_ratio(covered_contracts, len(declared_contract_ids)),
    )


def _elapsed_ms(function: Callable[[], None]) -> float:
    start_ns = perf_counter_ns()
    function()
    return (perf_counter_ns() - start_ns) / 1_000_000


def _run_valid_unguarded() -> None:
    for checkpoint in SyntheticHEPWorkflow().checkpoints():
        checkpoint.step()


def _run_valid_guarded() -> None:
    GuardedWorkflowRunner().execute(SyntheticHEPWorkflow().checkpoints())


def measure_timing(warmup_pairs: int, measured_pairs: int) -> TimingResult:
    if warmup_pairs < 0:
        raise ValueError("warmup_pairs must be nonnegative")
    if measured_pairs < 1:
        raise ValueError("measured_pairs must be positive")

    for index in range(warmup_pairs):
        ordered = (
            (_run_valid_unguarded, _run_valid_guarded)
            if index % 2 == 0
            else (_run_valid_guarded, _run_valid_unguarded)
        )
        for target in ordered:
            target()

    unguarded_samples: list[float] = []
    guarded_samples: list[float] = []
    for index in range(measured_pairs):
        if index % 2 == 0:
            unguarded_samples.append(_elapsed_ms(_run_valid_unguarded))
            guarded_samples.append(_elapsed_ms(_run_valid_guarded))
        else:
            guarded_samples.append(_elapsed_ms(_run_valid_guarded))
            unguarded_samples.append(_elapsed_ms(_run_valid_unguarded))

    unguarded_median = median(unguarded_samples)
    guarded_median = median(guarded_samples)
    if unguarded_median <= 0.0:
        raise ValueError("unguarded median must be positive to compute overhead")
    return TimingResult(
        unguarded_samples_ms=tuple(unguarded_samples),
        guarded_samples_ms=tuple(guarded_samples),
        unguarded_median_ms=unguarded_median,
        guarded_median_ms=guarded_median,
        overhead_ratio=guarded_median / unguarded_median,
    )


def _declared_contract_ids() -> tuple[str, ...]:
    return tuple(
        contract.contract_id
        for checkpoint in SyntheticHEPWorkflow().checkpoints()
        for contract in checkpoint.contracts
    )


def generate_report(warmup_pairs: int = 20, measured_pairs: int = 200) -> BenchmarkReport:
    declared_contract_ids = _declared_contract_ids()
    cases = (
        _case_record("none", None, None, None),
        *(
            _case_record(
                fault.case_id,
                fault.injector,
                fault.expected_contract_id,
                fault.expected_stage,
            )
            for fault in FAULT_CASES
        ),
    )
    return BenchmarkReport(
        generated_at=datetime.now(timezone.utc),
        provenance={
            "data_kind": "synthetic",
            "generator": "sciagentguard.packs.hep.fixtures",
            "scope": "deterministic fixture evaluation only",
        },
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        parameters={
            "warmup_pairs": warmup_pairs,
            "measured_pairs": measured_pairs,
            "timing_order": "paired and alternating",
            "timing_scope": "valid full workflow only",
        },
        declared_contract_ids=declared_contract_ids,
        cases=cases,
        metrics=compute_metrics(cases, declared_contract_ids),
        timing=measure_timing(warmup_pairs, measured_pairs),
    )


def _format_metric(metric: RatioMetric) -> str:
    return f"{metric.numerator}/{metric.denominator} ({metric.value:.1%})"


def render_markdown(report: BenchmarkReport) -> str:
    metric_rows = (
        ("Detection recall", report.metrics.detection_recall),
        ("Precision", report.metrics.precision),
        ("Correct localization", report.metrics.localization_rate),
        ("Unguarded false-pass rate", report.metrics.unguarded_false_pass_rate),
        ("Guarded false-pass rate", report.metrics.guarded_false_pass_rate),
        ("Valid-case false-positive rate", report.metrics.false_positive_rate),
        ("Contract coverage", report.metrics.contract_coverage),
    )
    lines = [
        "# Synthetic HEP fixture results",
        "",
        (
            f"Generated with SciAgentGuard {report.environment['sciagentguard_version']} on "
            f"{report.environment['python_implementation']} "
            f"{report.environment['python_version']}."
        ),
        "",
        "## Correctness",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        *(f"| {label} | {_format_metric(metric)} |" for label, metric in metric_rows),
        "",
        "| Case | Expected stage | Expected contract | Unguarded | Guarded |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report.cases:
        expected_stage = case.expected_stage or "all four"
        expected_contract = case.expected_contract_id or "none"
        unguarded = "completed" if case.unguarded.completed else "stopped"
        guarded = "blocked" if case.guarded.blocked else "passed"
        lines.append(
            f"| `{case.case_id}` | `{expected_stage}` | `{expected_contract}` | "
            f"{unguarded} | {guarded} |"
        )

    timing = report.timing
    lines.extend(
        [
            "",
            "## Valid-path timing",
            "",
            (
                f"After {report.parameters['warmup_pairs']} warm-up pairs, "
                f"{report.parameters['measured_pairs']} measured pairs produced a median of "
                f"{timing.unguarded_median_ms:.6f} ms unguarded and "
                f"{timing.guarded_median_ms:.6f} ms guarded "
                f"({timing.overhead_ratio:.3f}x)."
            ),
            "",
            (
                "Only the valid four-stage workflow is timed. Faulted runs are excluded because "
                "early blocking shortens execution and would make the comparison misleading."
            ),
            "",
            "## Scope",
            "",
            (
                "These results cover deterministic synthetic fixtures and deliberately injected "
                "faults. They do not establish behavior on experimental HEP data, model-generated "
                "workflows, or production infrastructure."
            ),
            "",
            (
                "The unguarded baseline means that every fixture step returned without a software "
                "exception; it does not run contracts or perform a final scientific check."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20, help="number of timing warm-up pairs")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="number of paired timing measurements",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = generate_report(args.warmup, args.repetitions)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        f"{report.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
