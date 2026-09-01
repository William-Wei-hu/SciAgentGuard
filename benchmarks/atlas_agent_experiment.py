"""Guard a code-writing agent on the ATLAS diphoton task and compare five arms.

Milestone 6A drives this with a deterministic scripted agent and a scripted judge. It measures
that the harness works end to end; it is not evidence about how a real model writes analysis code
or answers a validity question. Milestone 6B supplies a real provider.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue

from sciagentguard import __version__
from sciagentguard._boundary_benchmark import (
    BaselineName,
    BaselineRunRecord,
    CaseRecord,
    CorrectnessMetrics,
    RatioMetric,
    RepetitionRecord,
    arm_coverage,
    compute_metrics,
    divergent_case_ids,
    run_baseline_plan,
    scorable_arms,
)
from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.adapters.agent import (
    ATLAS_AGENT_TASK,
    CodeSandbox,
    SandboxResult,
    ScriptedAgent,
    ScriptedJudge,
    agent_contexts,
    generic_data_checks,
    generic_feedback,
    structured_feedback,
)
from sciagentguard.adapters.agent._scripts import CORRECT_SCRIPT_ID
from sciagentguard.core import ContractContext
from sciagentguard.runtime import RepairOutcome, WorkflowCheckpoint, WorkflowRepairRunner

EXPERIMENT_ID: Literal["atlas_agent_guarded_comparison"] = "atlas_agent_guarded_comparison"
DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/atlas_agent_results.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/atlas_agent.md")

ARMS: tuple[BaselineName, ...] = (
    "unguarded",
    "generic_data_checks",
    "llm_judge",
    "final_check_only",
    "runtime_guarded",
)
_STAGES = ("post_load", "post_selection", "post_histogram", "post_yield")


SCHEMA_FAILURE = "invalid_agent_artifacts"


@dataclass(frozen=True, slots=True)
class _AgentCase:
    """One script the agent may write, and the contract expected to stop it.

    A case is one of three kinds. It may be scientifically faulty, in which case a contract is
    expected to stop it. It may fail to produce runnable code. Or it may run perfectly and report
    its results in the wrong shape, which is neither a crash nor a scientific fault.
    """

    case_id: str
    script_id: str
    expected_contract_id: str | None
    expected_stage: str | None
    produces_runnable_code: bool = True
    produces_declared_schema: bool = True

    @property
    def is_faulty(self) -> bool:
        return self.script_id != CORRECT_SCRIPT_ID

    @property
    def is_semantic(self) -> bool:
        """Whether this case belongs in the rates at all.

        A case that never produced a usable result is rejected by every arm, so including it
        would lower every arm's false-pass rate by the same amount and make the weakest arm look
        better than it is. Such cases are listed in the table and counted on their own.
        """

        return self.produces_runnable_code and self.produces_declared_schema


AGENT_CASES = (
    _AgentCase("correct", CORRECT_SCRIPT_ID, None, None),
    _AgentCase("empty_selection", "empty_selection", "hep.selection.nonempty", "post_selection"),
    _AgentCase(
        "luminosity_unit_slip",
        "luminosity_unit_slip",
        "hep.atlas_open_data.histogram_closure",
        "post_histogram",
    ),
    _AgentCase(
        "overlapping_control_window",
        "overlapping_control_window",
        "hep.atlas_open_data.region_disjoint",
        "post_selection",
    ),
    _AgentCase(
        "stale_cutflow",
        "stale_cutflow",
        "hep.atlas_open_data.cutflow_monotonic",
        "post_selection",
    ),
    _AgentCase("unrunnable", "unrunnable", None, None, produces_runnable_code=False),
    _AgentCase(
        "wrong_output_schema",
        "wrong_output_schema",
        None,
        None,
        produces_declared_schema=False,
    ),
)

NON_SEMANTIC_CASE_IDS = frozenset(case.case_id for case in AGENT_CASES if not case.is_semantic)

REPAIRABLE_CASES = tuple(
    case
    for case in AGENT_CASES
    if case.is_faulty and case.produces_runnable_code and case.produces_declared_schema
)


class Judge(Protocol):
    """Anything that can look at a final artifact and say whether it is scientifically valid."""

    def verdict(self, final_artifact: Mapping[str, JsonValue]) -> bool: ...


class RepairAblationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    case_id: str
    feedback_kind: str
    outcome: str
    repair_attempts_used: int
    resolved: bool


class ExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: Literal["atlas_agent_guarded_comparison"] = EXPERIMENT_ID
    generated_at: datetime
    provenance: dict[str, str | int]
    environment: dict[str, str]
    parameters: dict[str, str | int]
    agent: dict[str, str | None]
    judge: dict[str, str | None]
    # How many distinct questions the reviewer was actually asked. Cases whose final artifact
    # is identical pose one question, and one answer is reported against each of them.
    judge_distinct_questions: int | None = None
    stages: tuple[str, ...]
    arms: tuple[BaselineName, ...]
    declared_contract_ids: tuple[str, ...]
    cases: tuple[CaseRecord, ...]
    # The cases the rates were computed over. Recorded so the report can be re-derived from its
    # own contents without knowing which cases this script considers semantic.
    metric_case_ids: tuple[str, ...]
    # The arms actually scored. An arm the judge could not answer for is dropped rather than
    # counted as a verdict it never gave.
    metric_arms: tuple[BaselineName, ...]
    # How much of the experiment each arm answered on. An arm short of full coverage keeps its
    # per-case verdicts but gets no overall rate.
    arm_coverage: dict[BaselineName, RatioMetric]
    metrics: CorrectnessMetrics
    code_failure_runs: int
    schema_failure_runs: int
    sandbox_median_ms: float
    repair_ablation: tuple[RepairAblationRecord, ...]


def _final_artifact(contexts: Sequence[ContractContext]) -> dict[str, object]:
    artifact = contexts[-1].artifacts.get("yield_estimate")
    return dict(artifact) if isinstance(artifact, dict) else {}


def _arm_records(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    result: SandboxResult,
    judge: Judge,
    questions: set[str] | None = None,
) -> tuple[BaselineRunRecord, ...]:
    """Judge one agent output five ways.

    The sandbox runs once per attempt and every arm inspects the same artifacts, so the arms
    differ only in what they check, never in what the agent produced.
    """

    if not result.produced_artifacts:
        # No runnable output: no arm can accept anything. This is a code failure, and it is
        # counted separately from a semantic false pass.
        return tuple(
            BaselineRunRecord(
                baseline=arm,
                accepted=False,
                trace=None,
                crashed=True,
                crash_type=result.outcome.value,
            )
            for arm in ARMS
        )

    assert result.artifacts is not None
    try:
        contexts = agent_contexts(loaded, result.artifacts)
    except ValueError:
        return tuple(
            BaselineRunRecord(
                baseline=arm,
                accepted=False,
                trace=None,
                crashed=True,
                crash_type=SCHEMA_FAILURE,
            )
            for arm in ARMS
        )

    checkpoints = adapter.checkpoints_for((loaded, *contexts))
    final = _final_artifact(contexts)
    records: list[BaselineRunRecord] = []
    for arm in ARMS:
        if arm == "generic_data_checks":
            records.append(
                BaselineRunRecord(baseline=arm, accepted=generic_data_checks(final), trace=None)
            )
        elif arm == "llm_judge":
            if questions is not None:
                questions.add(json.dumps(final, sort_keys=True, default=str))
            try:
                verdict = judge.verdict(final)
            except Exception:
                # An unavailable judge is recorded as absent, never as an acceptance.
                continue
            records.append(BaselineRunRecord(baseline=arm, accepted=verdict, trace=None))
        else:
            records.append(run_baseline_plan(arm, checkpoints))
    return tuple(records)


def _case_record(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    judge: Judge,
    case: _AgentCase,
    input_path: Path,
    repetitions: int,
    questions: set[str] | None = None,
) -> tuple[CaseRecord, list[float], int, int]:
    records: list[RepetitionRecord] = []
    durations: list[float] = []
    code_failures = 0
    schema_failures = 0
    agent = ScriptedAgent(case.script_id)

    for repetition in range(repetitions):
        proposal = agent.propose(ATLAS_AGENT_TASK, attempt_id=f"attempt-{repetition}")
        result = sandbox.run(proposal.code, input_path=input_path)
        durations.append(result.duration_ms)
        if not result.produced_artifacts:
            code_failures += 1
        run_context = ContractContext(
            workflow_id=loaded.workflow_id,
            run_id=f"{case.case_id}-{repetition}",
            attempt_id="attempt-0",
            stage=loaded.stage,
            artifacts=loaded.artifacts,
            schema=loaded.schema,
            units=loaded.units,
            provenance=loaded.provenance,
            config=loaded.config,
        )
        runs = _arm_records(adapter, run_context, result, judge, questions)
        if any(run.crash_type == SCHEMA_FAILURE for run in runs):
            schema_failures += 1
        records.append(RepetitionRecord(repetition=repetition, runs=runs))

    return (
        CaseRecord(
            case_id=case.case_id,
            fault_injected=case.is_faulty,
            expected_detectable=case.expected_contract_id is not None,
            expected_contract_id=case.expected_contract_id,
            expected_stage=case.expected_stage,
            injection_stage=case.expected_stage,
            repetitions=tuple(records),
        ),
        durations,
        code_failures,
        schema_failures,
    )


def _run_repair_ablation(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    input_path: Path,
    max_repair_attempts: int,
) -> tuple[RepairAblationRecord, ...]:
    records: list[RepairAblationRecord] = []
    for case in REPAIRABLE_CASES:
        for kind, formatter in (
            ("generic_error", generic_feedback),
            ("structured_report", structured_feedback),
        ):
            agent = ScriptedAgent(case.script_id)

            run_id = f"repair-{case.case_id}-{kind}"

            def factory(
                attempt_id: str,
                feedback: str | None,
                bound_agent: ScriptedAgent = agent,
                bound_run_id: str = run_id,
            ) -> Sequence[WorkflowCheckpoint]:
                proposal = bound_agent.propose(
                    ATLAS_AGENT_TASK, attempt_id=attempt_id, feedback=feedback
                )
                result = sandbox.run(proposal.code, input_path=input_path)
                if not result.produced_artifacts or result.artifacts is None:
                    raise ValueError("the repair ablation requires runnable agent code")
                attempt_context = ContractContext(
                    workflow_id=loaded.workflow_id,
                    run_id=bound_run_id,
                    attempt_id=attempt_id,
                    stage=loaded.stage,
                    artifacts=loaded.artifacts,
                    schema=loaded.schema,
                    units=loaded.units,
                    provenance=loaded.provenance,
                    config=loaded.config,
                )
                contexts = agent_contexts(attempt_context, result.artifacts)
                return adapter.checkpoints_for((attempt_context, *contexts))

            execution = WorkflowRepairRunner(max_repair_attempts).execute(factory, formatter)
            records.append(
                RepairAblationRecord(
                    case_id=case.case_id,
                    feedback_kind=kind,
                    outcome=execution.trace.outcome.value,
                    repair_attempts_used=execution.trace.repair_attempts_used,
                    resolved=execution.trace.outcome
                    in {RepairOutcome.PASSED, RepairOutcome.REPAIRED},
                )
            )
    return tuple(records)


def generate_report(
    source: AtlasGamGamSource,
    *,
    repetitions: int = 3,
    max_repair_attempts: int = 2,
    timeout_seconds: float = 300.0,
    judge: Judge | None = None,
    judge_identity: dict[str, str | None] | None = None,
) -> ExperimentReport:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    adapter = AtlasGamGamOpenDataAdapter(source)
    loaded = adapter.load_context(
        workflow_id="atlas-agent-experiment",
        run_id="verified-source",
        attempt_id="attempt-0",
    )
    sandbox = CodeSandbox(
        timeout_seconds=timeout_seconds,
        cpu_seconds=int(timeout_seconds),
        memory_bytes=6 * 1024**3,
    )
    resolved_judge: Judge = ScriptedJudge() if judge is None else judge
    # Distinct prompts, not calls: late-invisible faults leave the final artifact untouched,
    # so several cases can pose one identical question to an arm that sees only that artifact.
    questions: set[str] = set()
    identity = judge_identity or {"provider": None, "model_id": None, "source": "scripted-judge"}

    cases: list[CaseRecord] = []
    semantic_cases: list[CaseRecord] = []
    durations: list[float] = []
    code_failures = 0
    schema_failures = 0
    for case in AGENT_CASES:
        record, case_durations, failures, schema_in_failures = _case_record(
            adapter, loaded, sandbox, resolved_judge, case, source.path, repetitions, questions
        )
        cases.append(record)
        if case.is_semantic:
            semantic_cases.append(record)
        durations.extend(case_durations)
        code_failures += failures
        schema_failures += schema_in_failures

    declared_contract_ids = tuple(
        contract.contract_id for contracts in adapter.stage_contracts() for contract in contracts
    )
    return ExperimentReport(
        generated_at=datetime.now(timezone.utc),
        provenance={
            "source_type": source.source_type,
            "experiment": "ATLAS",
            "record_id": source.record_id,
            "doi": source.doi,
            "file_name": source.file_name,
            "checksum": f"adler32:{source.adler32}",
        },
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": version("numpy"),
            "uproot_version": version("uproot"),
        },
        parameters={
            "repetitions": repetitions,
            "max_repair_attempts": max_repair_attempts,
            "sandbox_timeout_seconds": int(timeout_seconds),
            "checkpoint_topology": "four ordered checkpoints: " + ", ".join(_STAGES),
            "agent_kind": "deterministic scripted agent, no model provider",
            "judge_kind": (
                "declared rule, no model provider"
                if identity["provider"] is None
                else f"{identity['provider']} model, seeing only the final artifact"
            ),
        },
        agent={
            "source": ScriptedAgent(CORRECT_SCRIPT_ID).source,
            "model_id": None,
            "provider": None,
        },
        judge=identity,
        judge_distinct_questions=len(questions) or None,
        stages=_STAGES,
        arms=ARMS,
        declared_contract_ids=declared_contract_ids,
        cases=tuple(cases),
        metric_case_ids=tuple(case.case_id for case in semantic_cases),
        metric_arms=scorable_arms(semantic_cases),
        arm_coverage=arm_coverage(semantic_cases),
        metrics=compute_metrics(
            semantic_cases, declared_contract_ids, scorable_arms(semantic_cases)
        ),
        code_failure_runs=code_failures,
        schema_failure_runs=schema_failures,
        sandbox_median_ms=median(durations),
        repair_ablation=_run_repair_ablation(
            adapter, loaded, sandbox, source.path, max_repair_attempts
        ),
    )


def _format_metric(metric: RatioMetric | None) -> str:
    if metric is None:
        return "n/a"
    if metric.value is None:
        return f"{metric.numerator}/{metric.denominator} (n/a)"
    return f"{metric.numerator}/{metric.denominator} ({metric.value:.1%})"


def _metric_cases(report: ExperimentReport) -> tuple[CaseRecord, ...]:
    """Return exactly the cases the report's rates were computed over."""

    selected = set(report.metric_case_ids)
    return tuple(case for case in report.cases if case.case_id in selected)


def _identical_artifact_note(report: ExperimentReport) -> list[str]:
    """Point out cases whose final artifact is the same bytes as the correct one.

    A late-invisible fault does not reach the final artifact -- that is what makes it late
    invisible. Any arm that sees only the final artifact is therefore looking at the correct
    result, and whatever it says about those cases is a statement about the correct result. Its
    verdict cannot be read as detection, whichever way it falls.
    """

    shared = [
        case.case_id
        for case in report.cases
        if case.late_invisible_reason is not None or case.case_id in _FINAL_ARTIFACT_TWINS
    ]
    if not shared:
        return []
    return [
        (
            "> **`"
            + "`, `".join(shared)
            + "` produce a final artifact identical to the correct run's"
            + (
                f", so the reviewer was asked {report.judge_distinct_questions} distinct questions "
                f"across {len(report.metric_case_ids)} scored cases"
                if report.judge_distinct_questions is not None
                else ""
            )
            + ".** The faults are "
            "late-invisible: region membership and the cutflow never reach the yield artifact. An "
            "arm that sees only that artifact is being shown the correct result, so its verdict "
            "on these rows says nothing about the fault -- a rejection is not a catch and an "
            "acceptance is not a miss. The contracts separate them by looking at the stage where "
            "the information still exists."
        ),
        "",
    ]


# Cases whose yield artifact is byte-identical to the correct run's, verified by hashing.
_FINAL_ARTIFACT_TWINS = frozenset({"overlapping_control_window", "stale_cutflow"})


def _single_sample_note(report: ExperimentReport) -> list[str]:
    """Say plainly that the reviewer's column is one draw per cell, not a judgement.

    A separate probe asked this reviewer the identical question five times at temperature zero and
    got two different answers back. Every verdict below is therefore one sample of a process that
    does not repeat, and reading the column as a set of decisions overstates it -- in either
    direction. The contracts' column is deterministic and repeats exactly.
    """

    if report.judge["provider"] is None:
        return []
    return [
        (
            "> **Each `llm_judge` cell is a single sample, not a decision.** Asked the same "
            "question five times at temperature 0, this reviewer returned two different answers "
            "and failed to answer twice — see "
            "[the reproducibility probe](judge_reproducibility.md). Its verdicts below cannot be "
            "read as detections or as false positives; one draw from a process that does not "
            "repeat is not a measurement. The contract column is deterministic and reproduces "
            "exactly."
        ),
        "",
    ]


def _cost_note(report: ExperimentReport) -> list[str]:
    """The asymmetry that makes the two layers complements rather than competitors."""

    if report.judge["provider"] is None:
        return []
    return [
        "## What each layer costs",
        "",
        "| | contracts | reviewer |",
        "| --- | --- | --- |",
        "| one full evaluation | 276 ms for all 16 contracts across 4 checkpoints | 475 to 1894 s "
        "for one verdict |",
        "| per check | ~17 ms | one verdict is the whole answer |",
        "| failure rate | none observed | 2 of 5 probe attempts, 4 of 20 experiment calls |",
        "| repeatability | identical every run | 2 distinct answers to one question |",
        "| localisation | names the contract and the stage | none |",
        "| price | free | metered |",
        "",
        (
            "Contract timing is 20 passes over already-built contexts on this machine and this "
            "file, excluding the analysis both layers take as given. Reviewer latencies come from "
            "the probe and from cached calls, each recorded when the model actually answered. "
            "Roughly four orders of magnitude separate them, which is why the reviewer earns its "
            "place by finding checks worth writing rather than by running as one."
        ),
        "",
    ]


def _degenerate_arm_note(report: ExperimentReport) -> list[str]:
    """Warn about any arm that answered the same way every time.

    An arm that rejects every run scores perfect recall by construction, and a reader who looks
    only at the recall row would take that for a detector as good as the contracts. It is not a
    detector at all: it separates nothing. The same holds for an arm that accepts everything.
    """

    notes: list[str] = []
    for arm in report.metric_arms:
        decisions = [
            run.accepted
            for case in report.cases
            if case.case_id in set(report.metric_case_ids)
            for repetition in case.repetitions
            for run in repetition.runs
            if run.baseline == arm
        ]
        if len(decisions) < 2 or len(set(decisions)) != 1:
            continue
        verdict = "accepted" if decisions[0] else "rejected"
        unusable = "recall" if decisions[0] is False else "false-pass rate"
        notes.append(
            f"> **`{arm}` {verdict} all {len(decisions)} scored runs.** It separates nothing, so "
            f"its {unusable} is an artefact of that single answer rather than a measurement. "
            "Read its two rows together: a perfect score on one and a worthless score on the "
            "other is what a constant answer looks like."
        )
    return [*notes, ""] if notes else []


def _arm_label(report: ExperimentReport, arm: BaselineName) -> str:
    """Name an arm so a reader skimming only the table cannot mistake what produced it.

    `llm_judge` is the slot a model occupies. While that slot holds a declared rule instead, the
    table must say so, because the column header is what most readers actually read.
    """

    if arm == "llm_judge":
        provider = report.judge.get("provider")
        model = report.judge.get("model_id")
        if provider is None:
            return "`llm_judge` (scripted rule, no model)"
        return f"`llm_judge` ({model or provider})"
    return f"`{arm}`"


def render_markdown(report: ExperimentReport) -> str:
    arms = report.metric_arms
    header = " | ".join(_arm_label(report, arm) for arm in arms)
    divider = " | ".join("---:" for _ in arms)
    lines = [
        "# ATLAS agent-in-the-loop results",
        "",
        (
            f"Generated with SciAgentGuard {report.environment['sciagentguard_version']} from "
            f"ATLAS record `{report.provenance['record_id']}`."
        ),
        "",
        (
            "> **The analysis code here was written by a deterministic script, not a model.** The "
            "faults are ones this repository wrote, so nothing below is evidence about how a real "
            "model writes analysis code. What the arms do with those faults is measured."
            + (
                ""
                if report.judge["provider"] is None
                else (
                    f" The judge, however, is a real model ({report.judge['model_id']}), from a "
                    "different vendor than the agent it judges, and its verdicts are its own."
                )
            )
        ),
        "",
        "## What the agent did",
        "",
        (
            "The agent wrote complete analysis code, which ran in the sandbox and produced the "
            "selection, histogram, and yield artifacts. The guard inspected those artifacts at "
            f"{len(report.stages)} checkpoints. Faulty variants differ from the correct script by "
            "a small, plausible edit: a luminosity quoted in the wrong unit, a control window "
            "typed with the signal window's upper bound, a cutflow recorded before the last cut, "
            "a threshold applied to the wrong photon."
        ),
        "",
        (
            f"Median sandboxed execution: {report.sandbox_median_ms:.1f} ms. "
            f"Runs that produced no runnable code: {report.code_failure_runs}. "
            f"Runs that ran but reported results in the wrong shape: "
            f"{report.schema_failure_runs}. Neither kind is counted as a semantic false pass."
        ),
        "",
        "## Arm comparison",
        "",
        f"| Metric | {header} |",
        f"| --- | {divider} |",
    ]
    case_header = " | ".join(_arm_label(report, arm) for arm in report.arms)
    case_divider = " | ".join("---:" for _ in report.arms)
    lines.extend(_single_sample_note(report))
    lines.extend(_degenerate_arm_note(report))
    unscored = [arm for arm in report.arms if arm not in report.metric_arms]
    if unscored:
        described = []
        for arm in unscored:
            coverage = report.arm_coverage.get(arm)
            if coverage is None:
                described.append(f"`{arm}` answered on no run")
                continue
            missing = coverage.denominator - coverage.numerator
            described.append(
                f"`{arm}` answered on {coverage.numerator} of {coverage.denominator} runs "
                f"({missing} missing)"
            )
        lines.extend(
            [
                (
                    "> **Not scored:** "
                    + "; ".join(described)
                    + ". Its per-case verdicts are kept in the table below, but it gets no "
                    "overall rate. Scoring an arm only where it managed to answer would flatter "
                    "it here: the verdicts that went missing were the slowest calls, and the "
                    "slowest calls are the artifacts it had most to weigh."
                ),
                "",
            ]
        )
    for field_name, label in (
        ("all_fault_false_pass_rate", "All-fault false-pass rate"),
        ("false_positive_rate", "Valid-case false-positive rate"),
        ("localization_rate", "Correct localization"),
        ("detection_recall", "Detectable-fault recall"),
    ):
        values = [
            _format_metric(getattr(report.metrics.by_baseline[arm], field_name)) for arm in arms
        ]
        lines.append(f"| {label} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            (
                f"Rates are computed over the {len(report.metric_case_ids)} "
                "cases that produced a usable result. The two cases that did not -- code that "
                "does not run, and results reported in the wrong shape -- are listed below but "
                "excluded from the rates, because every arm rejects them and including them "
                "would lower every arm's false-pass rate by the same amount."
            ),
            "",
            f"| Case | Expected detector | {case_header} |",
            f"| --- | --- | {case_divider} |",
        ]
    )
    for case in report.cases:
        if case.expected_contract_id is not None:
            expected = case.expected_contract_id
        elif not case.fault_injected:
            expected = "valid analysis"
        elif case.case_id == "wrong_output_schema":
            expected = "output-contract failure"
        else:
            expected = "code failure"
        accepted = []
        for arm in report.arms:
            # Every declared arm appears here even when it has no overall score, so a verdict that
            # was paid for is not thrown away because a different run of the same arm failed.
            answered = [
                run
                for repetition in case.repetitions
                for run in repetition.runs
                if run.baseline == arm
            ]
            if not answered:
                accepted.append("n/a")
            else:
                accepted.append(f"{sum(run.accepted for run in answered)}/{len(answered)}")
        lines.append(f"| `{case.case_id}` | `{expected}` | {' | '.join(accepted)} |")

    divergent = divergent_case_ids(_metric_cases(report))
    lines.extend(_identical_artifact_note(report))
    lines.extend(
        [
            "",
            (
                "Runtime-vs-final divergence: "
                f"{_format_metric(report.metrics.runtime_vs_final_divergence)} of fault runs"
                + (
                    ", on " + ", ".join(f"`{case_id}`" for case_id in divergent) + "."
                    if divergent
                    else "."
                )
            ),
            "",
            "## Repair feedback ablation",
            "",
            (
                f"Each faulty analysis was returned to the agent with at most "
                f"{report.parameters['max_repair_attempts']} repair attempts, once with a generic "
                "error string and once with the structured violation report."
            ),
            "",
            "| Case | Feedback | Outcome | Attempts used | Resolved |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for record in report.repair_ablation:
        lines.append(
            f"| `{record.case_id}` | `{record.feedback_kind}` | `{record.outcome}` | "
            f"{record.repair_attempts_used} | {'yes' if record.resolved else 'no'} |"
        )

    lines.extend(
        [
            "",
            (
                "**The scripted agent was written to repair only from feedback that names a "
                "contract and a stage.** The ablation therefore confirms that the loop, the "
                "feedback formatting, and the revalidation all work; it does not show that "
                "structured evidence helps a real model. That question belongs to Milestone 6B."
            ),
            "",
            *_cost_note(report),
            "## Limits",
            "",
            (
                "`generic_data_checks` asserts structure only, which is the ceiling of what a "
                "dataframe validator can express without physics. `llm_judge` sees only the final "
                "artifact. Both are stand-ins for what a practitioner actually reaches for, but "
                "in this run both are deterministic fixtures."
            ),
            "",
            (
                "The sandbox denies networking, subprocesses, and writes outside its working "
                "directory, and it bounds CPU, memory, and wall-clock time. It is a guardrail "
                "against an agent's accidents, not a security boundary against adversarial code."
            ),
            "",
        ]
    )
    return "\n".join(lines)


class _ProgressJudge:
    """Report each verdict as it lands, so a slow run can be told apart from a stuck one.

    Also keeps what each verdict cost. Where the answer was replayed from cache the recorded cost
    is the original call's, not the replay's -- see `CachingClient.completion`.
    """

    def __init__(self, inner: Judge, cache: object | None = None) -> None:
        self._inner = inner
        self._count = 0
        self._cache = cache
        self.latencies_ms: list[float] = []

    def verdict(self, final_artifact: Mapping[str, JsonValue]) -> bool:
        import sys
        from time import monotonic

        self._count += 1
        started = monotonic()
        before = _cache_hits(self._cache)
        try:
            answer = self._inner.verdict(final_artifact)
        except Exception as error:
            print(
                f"judge call {self._count}: FAILED after {monotonic() - started:.0f}s "
                f"({type(error).__name__})",
                file=sys.stderr,
                flush=True,
            )
            raise
        elapsed_ms = (monotonic() - started) * 1000.0
        replayed = _cache_hits(self._cache) > before
        # A replay costs microseconds; recording that as the reviewer's latency would invert the
        # comparison this table exists to make. Fresh calls only.
        if not replayed:
            self.latencies_ms.append(elapsed_ms)
        print(
            f"judge call {self._count}: {'VALID' if answer else 'INVALID'} "
            f"after {elapsed_ms / 1000.0:.0f}s{' (replayed)' if replayed else ''}",
            file=sys.stderr,
            flush=True,
        )
        return answer


def _cache_hits(cache: object | None) -> int:
    hits = getattr(cache, "hits", 0)
    return hits if isinstance(hits, int) else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--judge",
        choices=("scripted", "deepseek"),
        default="scripted",
        help="scripted keeps the run offline; deepseek asks a real model for each verdict",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    judge: Judge | None = None
    identity: dict[str, str | None] | None = None
    if args.judge == "deepseek":
        from sciagentguard.adapters.agent.deepseek import (
            DEFAULT_JUDGE_MODEL,
            PROVIDER,
            DeepSeekClient,
            DeepSeekJudge,
            load_deepseek_key,
        )

        # This judge deliberates for minutes per verdict, and its latency varies by a factor of
        # five. Without a line per call an unattended run is indistinguishable from a hung one.
        from sciagentguard.adapters.agent.verdict_cache import CachingClient, VerdictCache

        # A verdict here costs minutes and sometimes fails. Replaying answers already paid for
        # means a run that dies late resumes instead of repurchasing everything before it.
        cache = VerdictCache()
        judge = _ProgressJudge(
            DeepSeekJudge(
                client=CachingClient(
                    inner=DeepSeekClient(api_key=load_deepseek_key()),
                    cache=cache,
                    provider=PROVIDER,
                )
            ),
            cache=cache,
        )
        identity = {
            "provider": PROVIDER,
            "model_id": DEFAULT_JUDGE_MODEL,
            "source": "deepseek-judge",
        }

    report = generate_report(
        AtlasGamGamSource.official_wph125(args.input),
        repetitions=args.repetitions,
        max_repair_attempts=args.max_repair_attempts,
        judge=judge,
        judge_identity=identity,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
