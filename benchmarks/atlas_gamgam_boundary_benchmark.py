"""Compare guard baselines across the four ATLAS Gamma-Gamma analysis checkpoints."""

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
    divergent_case_ids,
    find_run,
    run_baseline_plan,
)
from sciagentguard._boundary_benchmark import (
    _failed_results as _failed_contracts,
)
from sciagentguard._boundary_benchmark import (
    measure_timing as measure_boundary_timing,
)
from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.core import ContractContext
from sciagentguard.core.protocols import SemanticFaultInjector
from sciagentguard.packs.hep import (
    AtlasMissingEventProvenanceInjector,
    AtlasMissingPhotonMomentumInjector,
    AtlasNormalizationScaleDriftInjector,
    AtlasPhotonCountMismatchInjector,
    AtlasPhotonScaleGapInjector,
    AtlasRegionOverlapInjector,
    AtlasSourceIdentityDriftInjector,
    AtlasWeightScaleGapInjector,
    NonfiniteWeightsInjector,
    ZeroWeightsInjector,
)
from sciagentguard.packs.hep._events import require_event_columns

BENCHMARK_ID: Literal["atlas_gamgam_boundary_comparison"] = "atlas_gamgam_boundary_comparison"
DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/atlas_gamgam_boundary_results.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/atlas_boundary.md")

_BASELINES = BASELINES
_STAGES = ("post_load", "post_selection", "post_histogram", "post_yield")
_VALID_RANGE_COUNT = 3


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["2.0"] = "2.0"
    benchmark_id: Literal["atlas_gamgam_boundary_comparison"] = BENCHMARK_ID
    generated_at: datetime
    provenance: dict[str, str | int]
    environment: dict[str, str]
    parameters: dict[str, str | int]
    stages: tuple[str, ...]
    declared_contract_ids: tuple[str, ...]
    cases: tuple[CaseRecord, ...]
    metrics: CorrectnessMetrics
    timing: TimingResult


@dataclass(frozen=True, slots=True)
class _FaultCase:
    """One injected fault, where it enters the workflow, and where it is expected to surface.

    ``injection_stage`` and ``expected_stage`` differ whenever a fault is introduced upstream but
    is only observable further down the chain.
    """

    case_id: str
    injector: SemanticFaultInjector
    expected_contract_id: str | None
    injection_stage: str
    expected_stage: str | None = None

    @property
    def expected_detectable(self) -> bool:
        return self.expected_contract_id is not None

    @property
    def detection_stage(self) -> str | None:
        if self.expected_contract_id is None:
            return None
        return self.expected_stage or self.injection_stage

    @property
    def late_invisible_reason(self) -> str | None:
        return getattr(self.injector, "late_invisible_reason", None)


FAULT_CASES = (
    _FaultCase(
        "missing_translated_branch",
        AtlasMissingPhotonMomentumInjector(),
        "hep.schema.required_branches",
        "post_load",
    ),
    _FaultCase("nonfinite_weights", NonfiniteWeightsInjector(), "hep.weights.finite", "post_load"),
    _FaultCase("zero_weights", ZeroWeightsInjector(), "hep.weights.nonzero_support", "post_load"),
    _FaultCase(
        "photon_count_mismatch",
        AtlasPhotonCountMismatchInjector(),
        "hep.atlas_open_data.diphoton_preselection",
        "post_load",
    ),
    _FaultCase(
        "missing_event_provenance",
        AtlasMissingEventProvenanceInjector(),
        "hep.provenance.events_declared",
        "post_load",
    ),
    _FaultCase(
        "source_identity_drift",
        AtlasSourceIdentityDriftInjector(),
        "hep.atlas_open_data.source_identity",
        "post_load",
    ),
    _FaultCase(
        "normalization_scale_drift",
        AtlasNormalizationScaleDriftInjector(),
        "hep.atlas_open_data.histogram_closure",
        "post_histogram",
    ),
    _FaultCase(
        "region_overlap",
        AtlasRegionOverlapInjector(),
        "hep.atlas_open_data.region_disjoint",
        "post_selection",
    ),
    # Rescaling event weights was a documented gap while every contract asked only whether an
    # artifact agreed with itself. The weight-provenance check compares reported weights against
    # the ones the verified source actually contains, so the fault became detectable.
    _FaultCase(
        "weight_scale_gap",
        AtlasWeightScaleGapInjector(),
        "hep.atlas_open_data.weight_provenance",
        "post_load",
        expected_stage="post_selection",
    ),
    # Rescaling photon momenta was a documented gap while the workflow stopped at post_load.
    # The downstream yield-shape check now moves the reconstructed peak out of its declared
    # window, so the fault became detectable at a stage far below where it was injected.
    _FaultCase(
        "photon_scale_gap",
        AtlasPhotonScaleGapInjector(),
        "hep.atlas_open_data.yield_shape",
        "post_load",
        expected_stage="post_yield",
    ),
)

LATE_INVISIBLE_CASE_IDS = tuple(
    case.case_id for case in FAULT_CASES if case.late_invisible_reason is not None
)


def _declared_contract_ids(adapter: AtlasGamGamOpenDataAdapter) -> tuple[str, ...]:
    return tuple(
        contract.contract_id for contracts in adapter.stage_contracts() for contract in contracts
    )


def _valid_ranges(event_count: int) -> tuple[tuple[int, int], ...]:
    """Split the declared source into three disjoint contiguous ranges."""

    if event_count < _VALID_RANGE_COUNT:
        raise ValueError("the source must contain at least one event per declared valid range")
    size = event_count // _VALID_RANGE_COUNT
    bounds = [(index * size, (index + 1) * size) for index in range(_VALID_RANGE_COUNT)]
    start, _ = bounds[-1]
    bounds[-1] = (start, event_count)
    return tuple(bounds)


def _case_record(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    case: _FaultCase | None,
    case_id: str,
    repetitions: int,
) -> CaseRecord:
    records: list[RepetitionRecord] = []
    for repetition in range(repetitions):
        runs: list[BaselineRunRecord] = []
        for baseline in _BASELINES:
            base = replace(
                loaded,
                run_id=f"{case_id}-{baseline}-{repetition}",
                attempt_id="attempt-0",
            )
            checkpoints = adapter.chained_checkpoints(
                lambda bound=base: bound,
                fault=None if case is None else case.injector,
                injection_stage=None if case is None else case.injection_stage,
            )
            runs.append(run_baseline_plan(baseline, checkpoints))
        records.append(RepetitionRecord(repetition=repetition, runs=tuple(runs)))

    return CaseRecord(
        case_id=case_id,
        fault_injected=case is not None,
        expected_detectable=case.expected_detectable if case is not None else False,
        expected_contract_id=case.expected_contract_id if case is not None else None,
        expected_stage=case.detection_stage if case is not None else None,
        injection_stage=case.injection_stage if case is not None else None,
        late_invisible_reason=case.late_invisible_reason if case is not None else None,
        repetitions=tuple(records),
    )


_run_for = find_run


def _timing_targets(
    adapter: AtlasGamGamOpenDataAdapter,
) -> dict[BaselineName, Callable[[], None]]:
    def run(baseline: BaselineName) -> Callable[[], None]:
        def target() -> None:
            checkpoints = adapter.checkpoints(
                workflow_id="atlas-gamgam-boundary-benchmark",
                run_id=f"timing-{baseline}",
                attempt_id="attempt-0",
            )
            record = run_baseline_plan(baseline, checkpoints)
            if not record.accepted:
                raise ValueError(f"the valid timing source failed {baseline} validation")

        return target

    return {baseline: run(baseline) for baseline in _BASELINES}


def measure_timing(
    adapter: AtlasGamGamOpenDataAdapter,
    warmup_triplets: int,
    measured_triplets: int,
) -> TimingResult:
    return measure_boundary_timing(
        _timing_targets(adapter),
        warmup_triplets,
        measured_triplets,
    )


def generate_report(
    source: AtlasGamGamSource,
    *,
    correctness_repetitions: int = 5,
    warmup_triplets: int = 2,
    measured_triplets: int = 10,
) -> BenchmarkReport:
    if correctness_repetitions < 1:
        raise ValueError("correctness_repetitions must be positive")

    full_adapter = AtlasGamGamOpenDataAdapter(source)
    event_count = full_adapter.source_event_count()
    ranges = _valid_ranges(event_count)
    range_adapters = tuple(
        AtlasGamGamOpenDataAdapter(source, entry_start=start, entry_stop=stop)
        for start, stop in ranges
    )
    range_contexts = tuple(
        adapter.load_context(
            workflow_id="atlas-gamgam-boundary-benchmark",
            run_id=f"verified-range-{index}",
            attempt_id="attempt-0",
        )
        for index, adapter in enumerate(range_adapters)
    )

    fault_adapter = range_adapters[0]
    fault_context = range_contexts[0]
    declared_contract_ids = _declared_contract_ids(fault_adapter)

    valid_cases = tuple(
        _case_record(
            range_adapters[index],
            range_contexts[index],
            None,
            f"none_range_{index}",
            correctness_repetitions,
        )
        for index in range(len(ranges))
    )
    fault_cases = tuple(
        _case_record(
            fault_adapter,
            fault_context,
            case,
            case.case_id,
            correctness_repetitions,
        )
        for case in FAULT_CASES
    )
    cases = (*valid_cases, *fault_cases)

    return BenchmarkReport(
        generated_at=datetime.now(timezone.utc),
        provenance={
            "source_type": source.source_type,
            "experiment": "ATLAS",
            "record_id": source.record_id,
            "doi": source.doi,
            "file_name": source.file_name,
            "checksum": f"adler32:{source.adler32}",
            "size_bytes": source.size_bytes,
            "event_count": event_count,
            "fault_range_event_count": len(require_event_columns(fault_context)["event_number"]),
        },
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": version("numpy"),
            "awkward_version": version("awkward"),
            "uproot_version": version("uproot"),
        },
        parameters={
            "correctness_repetitions": correctness_repetitions,
            "warmup_triplets": warmup_triplets,
            "measured_triplets": measured_triplets,
            "valid_range_count": len(ranges),
            "valid_ranges": "; ".join(f"[{start}, {stop})" for start, stop in ranges),
            "timing_order": "rotating three-mode order",
            "timing_scope": "valid full source verification, load, and four-stage derivation",
            "checkpoint_topology": "four ordered checkpoints: " + ", ".join(_STAGES),
            "final_check_definition": (
                "every step executes, then only the contracts declared at the final stage are "
                "evaluated against the final context"
            ),
            "omitted_arms": (
                "generic_data_checks and llm_judge are deferred to Milestone 6; this experiment "
                "compares three arms"
            ),
        },
        stages=_STAGES,
        declared_contract_ids=declared_contract_ids,
        cases=cases,
        metrics=compute_metrics(cases, declared_contract_ids),
        timing=measure_timing(fault_adapter, warmup_triplets, measured_triplets),
    )


def _format_metric(metric: RatioMetric | None) -> str:
    if metric is None:
        return "n/a"
    if metric.value is None:
        return f"{metric.numerator}/{metric.denominator} (n/a)"
    return f"{metric.numerator}/{metric.denominator} ({metric.value:.1%})"


def render_markdown(report: BenchmarkReport) -> str:
    labels: tuple[tuple[str, str], ...] = (
        ("all_fault_false_pass_rate", "All-fault false-pass rate"),
        ("false_positive_rate", "Valid-case false-positive rate"),
        ("localization_rate", "Correct localization"),
        ("precision", "Detection precision"),
        ("gap_probe_acceptance_rate", "Gap-probe acceptance rate"),
        ("contract_coverage", "Contract coverage"),
        ("detection_recall", "Detectable-fault recall (regression check)"),
    )
    divergence = report.metrics.runtime_vs_final_divergence
    divergent = divergent_case_ids(report.cases)
    lines = [
        "# ATLAS Gamma-Gamma boundary results",
        "",
        (
            f"Generated with SciAgentGuard {report.environment['sciagentguard_version']} from "
            f"ATLAS record `{report.provenance['record_id']}`. The source is "
            f"{report.provenance['source_type']} data for educational use, not collision-data "
            "validation or a physics result."
        ),
        "",
        (
            f"The workflow runs {len(report.stages)} ordered checkpoints: "
            + ", ".join(f"`{stage}`" for stage in report.stages)
            + "."
        ),
        "",
        "## Headline: runtime-guarded versus final-check-only",
        "",
        (
            f"**Runtime-vs-final divergence: {_format_metric(divergence)} of fault runs.** "
            + (
                "The two guarded baselines disagree on "
                + ", ".join(f"`{case_id}`" for case_id in divergent)
                + "."
                if divergent
                else "The two guarded baselines agree on every fault run."
            )
        ),
        "",
        (
            "`final_check_only` executes every step and then evaluates only the contracts "
            "declared at the final stage, against the final context. That is what a workflow "
            "retains when validation happens after the fact: the intermediate artifacts are gone, "
            "so the contracts that guard them have nothing to run against."
        ),
        "",
        "### Late-invisible faults",
        "",
    ]
    for case in report.cases:
        if case.late_invisible_reason is None:
            continue
        lines.extend(
            [
                (
                    f"- **`{case.case_id}`** injected at `{case.injection_stage}`, expected at "
                    f"`{case.expected_contract_id}`. {case.late_invisible_reason}"
                ),
            ]
        )
    if not any(case.late_invisible_reason for case in report.cases):
        lines.append("- none declared.")

    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "| Metric | Unguarded | Final check only | Runtime guarded |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for field_name, label in labels:
        values = [
            _format_metric(getattr(report.metrics.by_baseline[baseline], field_name))
            for baseline in _BASELINES
        ]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

    attribution = report.metrics.checkpoint_attribution
    if attribution is not None:
        values = [_format_metric(attribution[baseline]) for baseline in _BASELINES]
        lines.append(f"| Checkpoint attribution | {values[0]} | {values[1]} | {values[2]} |")

    lines.extend(
        [
            "",
            (
                "Detection recall is reported only as a regression check. When a fault and the "
                "contract intended to catch it are designed together, recall is constructively "
                "near 100% and carries almost no information."
            ),
            "",
            (
                "Checkpoint attribution counts a detection at the stage where the fault was "
                "injected. It is below 100% on purpose: a rescaled weight and a rescaled photon "
                "momentum both enter at `post_load` and only become observable further down the "
                "workflow. The shortfall measures where a fault becomes visible, not whether it "
                "was found."
            ),
            "",
            (
                "| Case | Injected at | Expected detector | Unguarded accepted | "
                "Final accepted | Runtime accepted |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for case in report.cases:
        expected = case.expected_contract_id or (
            "documented gap" if case.fault_injected else "valid source"
        )
        stage = case.injection_stage or "n/a"
        accepted = []
        for baseline in _BASELINES:
            runs = tuple(_run_for(repetition, baseline) for repetition in case.repetitions)
            accepted.append(f"{sum(run.accepted for run in runs)}/{len(runs)}")
        lines.append(
            f"| `{case.case_id}` | `{stage}` | `{expected}` | "
            f"{accepted[0]} | {accepted[1]} | {accepted[2]} |"
        )

    timing = report.timing
    lines.extend(
        [
            "",
            (
                f"Valid cases use {report.parameters['valid_range_count']} disjoint event ranges "
                f"of the same declared file: {report.parameters['valid_ranges']}. These are "
                "distinct valid inputs of one source, not independent samples."
            ),
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
                "Every sample verifies the checksum, reads the valid ROOT file, and derives all "
                "four stages; faulted runs are excluded. Raw timing samples are preserved in the "
                "JSON report."
            ),
            "",
            "## Negative results and limits",
            "",
            _gap_paragraph(report),
            "",
            _uncovered_paragraph(report),
            "",
            (
                "Every divergence reported above disappears for a workflow that retains each "
                "intermediate artifact and validates all of them at the end. What this "
                "experiment measures is validation against the artifact a workflow actually "
                "keeps, so the result is an argument about information retention and early "
                "stopping, not a proof that post-hoc validation is inherently unable to catch "
                "these faults. That stronger comparison is not run here."
            ),
            "",
            (
                "One fault in this set, `missing_translated_branch`, stops the unguarded and "
                "final-check-only workflows with a plain software error rather than a contract "
                "violation. Not every semantic fault needs a contract; some simply break the "
                "code that consumes the artifact."
            ),
            "",
            (
                "`generic_data_checks` and `llm_judge`, the two remaining arms of Section 9.2 of "
                "the specification, are not run here. They arrive with Milestone 6, so this "
                "experiment does not show how runtime contracts compare with a generic dataframe "
                "validator or with a model inspecting the final artifact."
            ),
            "",
            (
                "The yield stage is a closed-form sideband-subtracted estimate over binned weight "
                "sums. It is not a likelihood fit, not a background model, and not a physics "
                "measurement. The luminosity is a configured local assumption."
            ),
            "",
            (
                "File corruption, checksum drift, missing ROOT trees, and missing raw branches "
                "remain Adapter input errors and are not counted as scientific contract "
                "detections. Results apply only to the declared file and boundary."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _gap_paragraph(report: BenchmarkReport) -> str:
    """Describe the probes that every baseline still accepts."""

    accepted_gaps = tuple(
        case.case_id
        for case in report.cases
        if case.fault_injected
        and not case.expected_detectable
        and all(_run_for(repetition, "runtime_guarded").accepted for repetition in case.repetitions)
    )
    if not accepted_gaps:
        return (
            "Every declared probe in this set is now rejected by at least one contract, so this "
            "experiment documents no coverage gap. That is a statement about this fault "
            "taxonomy, not about the contract map: the Milestone 6B pilot showed a real model "
            "writing wrong analyses that none of these probes describes, and every contract here "
            "accepted them. An experiment cannot find a gap its own faults do not contain."
        )
    listed = ", ".join(f"`{case_id}`" for case_id in accepted_gaps)
    return (
        f"The following probes remain finite, nonzero, and structurally valid, so every "
        f"baseline accepts them: {listed}. They document missing coverage rather than "
        "successful detection. A rescaled photon momentum, by contrast, was a documented gap "
        "while the workflow stopped at `post_load` and is now caught downstream by the "
        "yield-shape check, at a stage well below where it was injected."
    )


def _uncovered_paragraph(report: BenchmarkReport) -> str:
    """Name the declared contracts that no fault in this set causes to fire."""

    fired = {
        contract_id
        for case in report.cases
        for repetition in case.repetitions
        for baseline in _BASELINES
        for contract_id, _ in _failed_contracts(_run_for(repetition, baseline))
    }
    uncovered = tuple(
        contract_id for contract_id in report.declared_contract_ids if contract_id not in fired
    )
    if not uncovered:
        return "Every declared contract is exercised by at least one fault in this set."
    listed = ", ".join(f"`{contract_id}`" for contract_id in uncovered)
    return (
        f"{len(uncovered)} of {len(report.declared_contract_ids)} declared contracts never fire "
        f"in this experiment: {listed}. They pass on every case here, so their value is "
        "untested by these results."
    )


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
        AtlasGamGamSource.official_wph125(args.input),
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
