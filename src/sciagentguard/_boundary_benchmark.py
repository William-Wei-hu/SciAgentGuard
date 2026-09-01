"""Internal records and metrics shared by single-checkpoint boundary experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import median
from time import perf_counter_ns
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sciagentguard.core import ContractContext, ContractStatus, ScientificContract
from sciagentguard.runtime import (
    ExecutionTrace,
    GuardedExecutor,
    GuardedWorkflowRunner,
    WorkflowCheckpoint,
    WorkflowTrace,
)

BaselineName = Literal[
    "unguarded",
    "final_check_only",
    "runtime_guarded",
    "generic_data_checks",
    "llm_judge",
]
# The default arm set of the boundary experiments. An experiment declares its own arms; this is
# only the fallback for records that do not, so adding an arm here would silently change what
# every existing benchmark is required to contain.
BASELINES: tuple[BaselineName, ...] = (
    "unguarded",
    "final_check_only",
    "runtime_guarded",
)


class BaselineRunRecord(BaseModel):
    """One baseline's outcome for one run.

    ``crashed`` records a workflow that raised before producing a final artifact. Such a run is
    not accepted, but it was stopped by an ordinary software error rather than by a scientific
    contract, so the two outcomes must not be conflated when reading the tables.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: BaselineName
    accepted: bool
    trace: ExecutionTrace | WorkflowTrace | None
    crashed: bool = False
    crash_type: str | None = None


class RepetitionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repetition: int
    runs: tuple[BaselineRunRecord, ...]


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    fault_injected: bool
    expected_detectable: bool
    expected_contract_id: str | None
    expected_stage: str | None
    repetitions: tuple[RepetitionRecord, ...]
    injection_stage: str | None = None
    late_invisible_reason: str | None = None


class RatioMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    numerator: int
    denominator: int
    value: float | None


class BaselineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detection_recall: RatioMetric
    precision: RatioMetric
    localization_rate: RatioMetric
    all_fault_false_pass_rate: RatioMetric
    false_positive_rate: RatioMetric
    contract_coverage: RatioMetric
    gap_probe_acceptance_rate: RatioMetric


class CorrectnessMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    by_baseline: dict[BaselineName, BaselineMetrics]
    final_runtime_agreement: RatioMetric
    runtime_vs_final_divergence: RatioMetric | None = None
    checkpoint_attribution: dict[BaselineName, RatioMetric] | None = None


class TimingModeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    samples_ms: tuple[float, ...]
    median_ms: float
    ratio_to_unguarded: float


class TimingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    unit: Literal["milliseconds"] = "milliseconds"
    case_id: Literal["none"] = "none"
    modes: dict[BaselineName, TimingModeResult]


def run_baseline(
    baseline: BaselineName,
    context: ContractContext,
    contracts: tuple[ScientificContract, ...],
) -> BaselineRunRecord:
    """Run one baseline over a single guarded checkpoint."""

    return run_baseline_plan(baseline, (WorkflowCheckpoint(lambda: context, contracts),))


def run_baseline_plan(
    baseline: BaselineName,
    checkpoints: Sequence[WorkflowCheckpoint],
) -> BaselineRunRecord:
    """Run one baseline over an ordered checkpoint plan.

    ``unguarded`` executes every step and evaluates nothing. ``final_check_only`` executes every
    step and then evaluates only the contracts declared at the final stage, against the final
    context: that is all a workflow retains when validation happens after the fact.
    ``runtime_guarded`` evaluates each checkpoint's contracts as that checkpoint is reached.
    """

    if not checkpoints:
        raise ValueError("a baseline plan requires at least one checkpoint")

    if baseline == "unguarded":
        try:
            for checkpoint in checkpoints:
                checkpoint.step()
        except ValueError as error:
            return _crashed_record(baseline, error)
        return BaselineRunRecord(baseline=baseline, accepted=True, trace=None)

    if baseline == "final_check_only":
        final_context: ContractContext | None = None
        try:
            for checkpoint in checkpoints:
                final_context = checkpoint.step()
        except ValueError as error:
            return _crashed_record(baseline, error)
        if final_context is None:
            raise ValueError("the final checkpoint did not produce a context")
        resolved = final_context
        final_execution = GuardedExecutor().execute(lambda: resolved, checkpoints[-1].contracts)
        return BaselineRunRecord(
            baseline=baseline,
            accepted=not final_execution.trace.blocked,
            trace=final_execution.trace,
        )

    try:
        runtime_execution = GuardedWorkflowRunner().execute(checkpoints)
    except ValueError as error:
        return _crashed_record(baseline, error)
    return BaselineRunRecord(
        baseline=baseline,
        accepted=not runtime_execution.trace.blocked,
        trace=runtime_execution.trace,
    )


def _crashed_record(baseline: BaselineName, error: Exception) -> BaselineRunRecord:
    """Record a workflow that raised, keeping only the exception type out of the message.

    The message is deliberately discarded: it may quote artifact contents or local paths, and
    the case identifier already says which fault produced the crash.
    """

    return BaselineRunRecord(
        baseline=baseline,
        accepted=False,
        trace=None,
        crashed=True,
        crash_type=type(error).__name__,
    )


def find_run(record: RepetitionRecord, baseline: BaselineName) -> BaselineRunRecord:
    matches = tuple(run for run in record.runs if run.baseline == baseline)
    if len(matches) != 1:
        raise ValueError(f"each repetition must contain one {baseline!r} run")
    return matches[0]


def recorded_arms(cases: Sequence[CaseRecord]) -> tuple[BaselineName, ...]:
    """Return the arms actually present in the records, in the order they were recorded.

    An experiment declares its own arms rather than inheriting a global list, so that adding an
    arm for one experiment cannot silently invalidate another.
    """

    if not cases or not cases[0].repetitions:
        raise ValueError("arms can only be derived from recorded repetitions")
    arms = tuple(run.baseline for run in cases[0].repetitions[0].runs)
    if len(set(arms)) != len(arms):
        raise ValueError("a repetition must not record the same arm twice")
    return arms


def arm_coverage(cases: Sequence[CaseRecord]) -> dict[BaselineName, RatioMetric]:
    """How many of the scored runs each arm actually answered on.

    Recorded separately from the rates because a missing verdict is not a verdict. An arm that
    answered on most runs still has its per-case results preserved, and a reader can see both how
    much of the experiment it covered and what it said where it did.
    """

    total = sum(len(case.repetitions) for case in cases)
    answered: dict[BaselineName, int] = {}
    for case in cases:
        for repetition in case.repetitions:
            for run in repetition.runs:
                answered[run.baseline] = answered.get(run.baseline, 0) + 1
    return {arm: _ratio(count, total) for arm, count in answered.items()}


def scorable_arms(cases: Sequence[CaseRecord]) -> tuple[BaselineName, ...]:
    """Return the arms recorded in every repetition of every case.

    An arm that is missing anywhere cannot be scored. Counting a missing verdict as a rejection
    would credit an unavailable judge with catching a fault; counting it as an acceptance would
    blame it for one. The arm is dropped from the rates instead, and the report says so.
    """

    if not cases or not cases[0].repetitions:
        raise ValueError("arms can only be derived from recorded repetitions")
    shared: set[BaselineName] | None = None
    for case in cases:
        for repetition in case.repetitions:
            present = {run.baseline for run in repetition.runs}
            shared = present if shared is None else (shared & present)
    ordered = recorded_arms(cases)
    return tuple(arm for arm in ordered if shared is not None and arm in shared)


def compute_metrics(
    cases: Sequence[CaseRecord],
    declared_contract_ids: Sequence[str],
    arms: Sequence[BaselineName] | None = None,
) -> CorrectnessMetrics:
    if not cases or not declared_contract_ids:
        raise ValueError("metrics require cases and declared contracts")
    resolved_arms = tuple(arms) if arms is not None else recorded_arms(cases)

    comparisons = 0
    agreements = 0
    fault_runs = 0
    divergences = 0
    compares_placement = {"final_check_only", "runtime_guarded"}.issubset(resolved_arms)
    if compares_placement:
        for case in cases:
            for repetition in case.repetitions:
                final_run = find_run(repetition, "final_check_only")
                runtime_run = find_run(repetition, "runtime_guarded")
                comparisons += 1
                if final_run.accepted == runtime_run.accepted and _failed_results(
                    final_run
                ) == _failed_results(runtime_run):
                    agreements += 1
                if case.fault_injected:
                    fault_runs += 1
                    if final_run.accepted != runtime_run.accepted:
                        divergences += 1

    return CorrectnessMetrics(
        by_baseline={
            baseline: _baseline_metrics(cases, baseline, declared_contract_ids)
            for baseline in resolved_arms
        },
        final_runtime_agreement=_ratio(agreements, comparisons),
        runtime_vs_final_divergence=(
            _ratio(divergences, fault_runs) if compares_placement else None
        ),
        checkpoint_attribution={
            baseline: _checkpoint_attribution(cases, baseline) for baseline in resolved_arms
        },
    )


def divergent_case_ids(cases: Sequence[CaseRecord]) -> tuple[str, ...]:
    """Return the fault cases on which final-check-only and runtime-guarded disagree."""

    if not {"final_check_only", "runtime_guarded"}.issubset(recorded_arms(cases)):
        return ()
    divergent: list[str] = []
    for case in cases:
        if not case.fault_injected:
            continue
        disagrees = any(
            find_run(repetition, "final_check_only").accepted
            != find_run(repetition, "runtime_guarded").accepted
            for repetition in case.repetitions
        )
        if disagrees:
            divergent.append(case.case_id)
    return tuple(divergent)


def _checkpoint_attribution(
    cases: Sequence[CaseRecord],
    baseline: BaselineName,
) -> RatioMetric:
    """Measure how often a detected fault is reported at the stage where it was injected."""

    detected = 0
    attributed = 0
    for case in cases:
        if not case.expected_detectable or case.injection_stage is None:
            continue
        for repetition in case.repetitions:
            run = find_run(repetition, baseline)
            if run.accepted:
                continue
            detected += 1
            if any(stage == case.injection_stage for _, stage in _failed_results(run)):
                attributed += 1
    return _ratio(attributed, detected)


def measure_timing(
    targets: dict[BaselineName, Callable[[], None]],
    warmup_rounds: int,
    measured_rounds: int,
) -> TimingResult:
    """Time every arm once per round, rotating the order so no arm always runs first."""

    arms = tuple(targets)
    if "unguarded" not in arms:
        raise ValueError("timing requires an 'unguarded' arm to form ratios against")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds must be nonnegative")
    if measured_rounds < 1:
        raise ValueError("measured_rounds must be positive")

    orders = tuple(arms[shift:] + arms[:shift] for shift in range(len(arms)))
    for index in range(warmup_rounds):
        for baseline in orders[index % len(orders)]:
            targets[baseline]()

    samples: dict[BaselineName, list[float]] = {baseline: [] for baseline in arms}
    for index in range(measured_rounds):
        for baseline in orders[index % len(orders)]:
            samples[baseline].append(_elapsed_ms(targets[baseline]))

    unguarded_median = median(samples["unguarded"])
    if unguarded_median <= 0.0:
        raise ValueError("unguarded median must be positive to compute timing ratios")
    return TimingResult(
        modes={
            baseline: TimingModeResult(
                samples_ms=tuple(samples[baseline]),
                median_ms=median(samples[baseline]),
                ratio_to_unguarded=median(samples[baseline]) / unguarded_median,
            )
            for baseline in arms
        }
    )


def _ratio(numerator: int, denominator: int) -> RatioMetric:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("ratio counts must satisfy 0 <= numerator <= denominator")
    return RatioMetric(
        numerator=numerator,
        denominator=denominator,
        value=None if denominator == 0 else numerator / denominator,
    )


def _failed_results(run: BaselineRunRecord) -> tuple[tuple[str, str], ...]:
    trace = run.trace
    if trace is None:
        return ()
    if isinstance(trace, ExecutionTrace):
        results = tuple(trace.results)
    else:
        results = tuple(result for checkpoint in trace.checkpoints for result in checkpoint.results)
    return tuple(
        (result.contract_id, result.violation.stage)
        for result in results
        if result.status is ContractStatus.FAIL and result.violation is not None
    )


def _baseline_metrics(
    cases: Sequence[CaseRecord],
    baseline: BaselineName,
    declared_contract_ids: Sequence[str],
) -> BaselineMetrics:
    runs = tuple(
        (case, find_run(repetition, baseline)) for case in cases for repetition in case.repetitions
    )
    detectable = tuple(item for item in runs if item[0].expected_detectable)
    invalid = tuple(item for item in runs if item[0].fault_injected)
    valid = tuple(item for item in runs if not item[0].fault_injected)
    gaps = tuple(
        item for item in runs if item[0].fault_injected and not item[0].expected_detectable
    )
    reports = tuple(
        (case, contract_id, stage)
        for case, run in runs
        for contract_id, stage in _failed_results(run)
    )
    # A report is correct when the run it came from actually carried an injected fault. Whether
    # it fired the specific contract the case expected is a separate question, measured by
    # localization_rate: a real fault found by a different downstream contract is still a true
    # positive, not a precision loss.
    true_reports = tuple(item for item in reports if item[0].fault_injected)
    expected_reports = tuple(
        item
        for item in reports
        if item[0].expected_contract_id == item[1] and item[0].expected_stage == item[2]
    )
    localized = sum(
        (case.expected_contract_id, case.expected_stage) in _failed_results(run)
        for case, run in detectable
    )
    covered = {
        contract_id for case, contract_id, stage in expected_reports if case.expected_stage == stage
    }

    return BaselineMetrics(
        detection_recall=_ratio(sum(not run.accepted for _, run in detectable), len(detectable)),
        precision=_ratio(len(true_reports), len(reports)),
        localization_rate=_ratio(localized, len(detectable)),
        all_fault_false_pass_rate=_ratio(sum(run.accepted for _, run in invalid), len(invalid)),
        false_positive_rate=_ratio(sum(not run.accepted for _, run in valid), len(valid)),
        contract_coverage=_ratio(len(covered), len(declared_contract_ids)),
        gap_probe_acceptance_rate=_ratio(sum(run.accepted for _, run in gaps), len(gaps)),
    )


def _elapsed_ms(function: Callable[[], None]) -> float:
    start_ns = perf_counter_ns()
    function()
    return (perf_counter_ns() - start_ns) / 1_000_000
