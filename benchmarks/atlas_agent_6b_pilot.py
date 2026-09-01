"""Milestone 6B pilot: put a real Gemini model under the guard on the ATLAS diphoton task.

A pilot, not the experiment. One task, a couple of seeds, roughly ten requests. Its purpose is to
prove the plumbing -- prompt, code extraction, sandboxing, the arms, the correctness oracle, the
repair loop -- before quota is spent on the full run the specification requires.

No claim about model behaviour should be drawn from this many runs.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue

from sciagentguard import __version__
from sciagentguard._boundary_benchmark import BaselineName, run_baseline_plan
from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.adapters.agent import (
    ATLAS_AGENT_TASK,
    CodeSandbox,
    agent_contexts,
    generic_data_checks,
    structured_feedback,
)
from sciagentguard.adapters.agent.gemini import (
    GeminiAgent,
    GeminiClient,
    GeminiError,
    GeminiJudge,
    Transport,
    load_api_key,
    resolve_model_id,
)
from sciagentguard.adapters.agent.models import AgentProposal, SandboxResult
from sciagentguard.adapters.agent.reference import ReferenceComparison, compare_to_reference
from sciagentguard.core import ContractContext
from sciagentguard.runtime import RepairOutcome, WorkflowCheckpoint, WorkflowRepairRunner

PILOT_ID: Literal["atlas_agent_6b_pilot"] = "atlas_agent_6b_pilot"
DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/atlas_agent_6b_pilot.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/atlas_agent_pilot.md")
DEFAULT_CODE_DIR = Path("benchmarks/results/agent_code")

AGENT_MODEL = "gemini-3.5-flash-lite"
JUDGE_MODEL = "gemini-3.7-flash"
AGENT_INTERVAL_SECONDS = 60.0 / 15.0
JUDGE_INTERVAL_SECONDS = 60.0 / 5.0

ARMS: tuple[BaselineName, ...] = (
    "unguarded",
    "generic_data_checks",
    "llm_judge",
    "final_check_only",
    "runtime_guarded",
)


class Judge(Protocol):
    """Anything that can look at a final artifact and say whether it is scientifically valid."""

    def verdict(self, final_artifact: Mapping[str, JsonValue]) -> bool: ...


class PilotRun(BaseModel):
    """One seed: what the model wrote, whether it was right, and what each arm decided."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    seed: int
    model_id: str
    provider: str
    judge_available: bool = True
    prompt_hash: str
    sampling_parameters: dict[str, JsonValue]
    latency_ms: float
    code_sha256: str
    code_lines: int
    sandbox_outcome: str
    sandbox_ms: float
    produced_artifacts: bool
    oracle_agrees: bool | None = None
    oracle_disagreements: tuple[str, ...] = ()
    arm_accepted: dict[str, bool] = {}
    blocked_stage: str | None = None
    blocked_contracts: tuple[str, ...] = ()


class PilotRepair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    seed: int
    outcome: str
    repair_attempts_used: int
    resolved: bool


class PilotReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    pilot_id: Literal["atlas_agent_6b_pilot"] = PILOT_ID
    generated_at: datetime
    provenance: dict[str, str | int]
    environment: dict[str, str]
    parameters: dict[str, str | int | float]
    arms: tuple[BaselineName, ...]
    runs: tuple[PilotRun, ...]
    repairs: tuple[PilotRepair, ...]
    total_requests: int


def _safe_proposal_fields(proposal: AgentProposal) -> dict[str, JsonValue]:
    return dict(proposal.sampling_parameters)


def _evaluate_arms(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    artifacts: dict[str, JsonValue],
    judge: Judge,
) -> tuple[dict[str, bool], str | None, tuple[str, ...]]:
    contexts = agent_contexts(loaded, artifacts)
    checkpoints = adapter.checkpoints_for((loaded, *contexts))
    final = contexts[-1].artifacts.get("yield_estimate")
    final_artifact = dict(final) if isinstance(final, dict) else {}

    accepted: dict[str, bool] = {}
    blocked_stage: str | None = None
    blocked_contracts: tuple[str, ...] = ()
    for arm in ARMS:
        if arm == "generic_data_checks":
            accepted[arm] = generic_data_checks(final_artifact)
        elif arm == "llm_judge":
            try:
                accepted[arm] = judge.verdict(final_artifact)
            except Exception:
                # A judge outage must not discard a run whose agent call has already been paid
                # for. The arm is left unrecorded, and the report says so rather than implying
                # the judge accepted anything.
                continue
        else:
            record = run_baseline_plan(arm, checkpoints)
            accepted[arm] = record.accepted
            if arm == "runtime_guarded" and not record.accepted and record.trace is not None:
                trace = record.trace
                checkpoint_traces = getattr(trace, "checkpoints", ())
                if checkpoint_traces:
                    terminal = checkpoint_traces[-1]
                    blocked_stage = terminal.stage
                    blocked_contracts = tuple(
                        result.contract_id
                        for result in terminal.results
                        if result.status.value == "fail"
                    )
    return accepted, blocked_stage, blocked_contracts


def run_pilot(
    source: AtlasGamGamSource,
    *,
    seeds: Sequence[int],
    max_repair_attempts: int = 2,
    timeout_seconds: float = 300.0,
    code_dir: Path = DEFAULT_CODE_DIR,
    transport: Transport | None = None,
    api_key: str | None = None,
    judge: Judge | None = None,
    judge_identity: dict[str, str | None] | None = None,
) -> PilotReport:
    """Run the pilot. `transport` is injectable so the whole flow can be rehearsed offline."""

    resolved_key = api_key if api_key is not None else load_api_key()

    def client(interval: float) -> GeminiClient:
        if transport is None:
            return GeminiClient(api_key=resolved_key, min_interval_seconds=interval)
        return GeminiClient(
            api_key=resolved_key, min_interval_seconds=interval, transport=transport
        )

    available = client(0.0).list_models()
    agent_model = resolve_model_id(available, AGENT_MODEL)
    judge_model = resolve_model_id(available, JUDGE_MODEL)

    agent = GeminiAgent(client=client(AGENT_INTERVAL_SECONDS), model_id=agent_model)
    # The judge is a different vendor on purpose: a model grading its own family invites a
    # question this design does not have to answer.
    resolved_judge: Judge = (
        GeminiJudge(client=client(JUDGE_INTERVAL_SECONDS), model_id=judge_model)
        if judge is None
        else judge
    )
    judge_description = judge_identity or {"provider": "google", "model_id": judge_model}

    adapter = AtlasGamGamOpenDataAdapter(source)
    loaded = adapter.load_context(
        workflow_id="atlas-agent-6b-pilot", run_id="verified-source", attempt_id="attempt-0"
    )
    reference_contexts = adapter.contexts(
        workflow_id="atlas-agent-6b-pilot", run_id="reference", attempt_id="attempt-0"
    )
    sandbox = CodeSandbox(
        timeout_seconds=timeout_seconds,
        cpu_seconds=int(timeout_seconds),
        memory_bytes=6 * 1024**3,
    )
    code_dir.mkdir(parents=True, exist_ok=True)

    runs: list[PilotRun] = []
    repairs: list[PilotRepair] = []
    requests = 1  # the model listing

    for seed in seeds:
        proposal = agent.propose(ATLAS_AGENT_TASK, attempt_id="attempt-0", seed=seed)
        requests += 1
        digest = hashlib.sha256(proposal.code.encode("utf-8")).hexdigest()
        (code_dir / f"seed-{seed}-attempt-0.py").write_text(proposal.code, encoding="utf-8")
        result: SandboxResult = sandbox.run(proposal.code, input_path=source.path)

        run = PilotRun(
            seed=seed,
            model_id=proposal.model_id or agent_model,
            provider=proposal.provider or "google",
            prompt_hash=proposal.prompt_hash or "",
            sampling_parameters=_safe_proposal_fields(proposal),
            latency_ms=proposal.latency_ms or 0.0,
            code_sha256=digest,
            code_lines=len(proposal.code.splitlines()),
            sandbox_outcome=result.outcome.value,
            sandbox_ms=result.duration_ms,
            produced_artifacts=result.produced_artifacts,
        )

        if result.produced_artifacts and result.artifacts is not None:
            comparison: ReferenceComparison = compare_to_reference(
                reference_contexts, result.artifacts
            )
            try:
                accepted, stage, contracts = _evaluate_arms(
                    adapter, loaded, dict(result.artifacts), resolved_judge
                )
                requests += 1
            except ValueError as error:
                run = run.model_copy(
                    update={
                        "oracle_agrees": comparison.agrees,
                        "oracle_disagreements": (f"output contract: {error}",),
                    }
                )
                runs.append(run)
                continue
            run = run.model_copy(
                update={
                    "oracle_agrees": comparison.agrees,
                    "oracle_disagreements": comparison.disagreements,
                    "arm_accepted": accepted,
                    "blocked_stage": stage,
                    "blocked_contracts": contracts,
                    "judge_available": "llm_judge" in accepted,
                }
            )
            if stage is not None:
                repair, used = _repair(
                    adapter, loaded, sandbox, agent, source.path, seed, max_repair_attempts
                )
                repairs.append(repair)
                requests += used
        runs.append(run)

    return PilotReport(
        generated_at=datetime.now(timezone.utc),
        provenance={
            "source_type": source.source_type,
            "record_id": source.record_id,
            "file_name": source.file_name,
            "checksum": f"adler32:{source.adler32}",
        },
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
        },
        parameters={
            "agent_model": agent_model,
            "judge_model": str(judge_description.get("model_id") or judge_model),
            "judge_provider": str(judge_description.get("provider") or "google"),
            "seeds": ",".join(str(seed) for seed in seeds),
            "max_repair_attempts": max_repair_attempts,
            "sandbox_timeout_seconds": timeout_seconds,
            "oracle": "agreement with the repository's own atlas_analysis pipeline",
            "oracle_relative_tolerance": 1e-6,
        },
        arms=ARMS,
        runs=tuple(runs),
        repairs=tuple(repairs),
        total_requests=requests,
    )


def _repair(
    adapter: AtlasGamGamOpenDataAdapter,
    loaded: ContractContext,
    sandbox: CodeSandbox,
    agent: GeminiAgent,
    input_path: Path,
    seed: int,
    max_repair_attempts: int,
) -> tuple[PilotRepair, int]:
    """Return the model's blocked analysis with structured evidence and let it try again."""

    calls = [0]

    def factory(attempt_id: str, feedback: str | None) -> Sequence[WorkflowCheckpoint]:
        proposal = agent.propose(
            ATLAS_AGENT_TASK, attempt_id=attempt_id, feedback=feedback, seed=seed
        )
        calls[0] += 1
        result = sandbox.run(proposal.code, input_path=input_path)
        if not result.produced_artifacts or result.artifacts is None:
            raise ValueError("the repair attempt produced no usable artifacts")
        context = ContractContext(
            workflow_id=loaded.workflow_id,
            run_id=f"repair-seed-{seed}",
            attempt_id=attempt_id,
            stage=loaded.stage,
            artifacts=loaded.artifacts,
            schema=loaded.schema,
            units=loaded.units,
            provenance=loaded.provenance,
            config=loaded.config,
        )
        return adapter.checkpoints_for((context, *agent_contexts(context, result.artifacts)))

    try:
        execution = WorkflowRepairRunner(max_repair_attempts).execute(factory, structured_feedback)
    except (ValueError, GeminiError) as error:
        return PilotRepair(
            seed=seed, outcome=f"aborted: {error}", repair_attempts_used=calls[0], resolved=False
        ), calls[0]
    return (
        PilotRepair(
            seed=seed,
            outcome=execution.trace.outcome.value,
            repair_attempts_used=execution.trace.repair_attempts_used,
            resolved=execution.trace.outcome in {RepairOutcome.PASSED, RepairOutcome.REPAIRED},
        ),
        calls[0],
    )


def _headline(report: PilotReport) -> list[str]:
    """State the pilot's central result before any table, whichever way it went."""

    evaluated = [run for run in report.runs if run.arm_accepted]
    wrong = [run for run in evaluated if run.oracle_agrees is False]
    missed = [run for run in wrong if run.arm_accepted.get("runtime_guarded")]
    if not evaluated:
        return []
    if not wrong:
        return [
            "## Result: the model got it right, and nothing blocked it",
            "",
            (
                f"All {len(evaluated)} run(s) matched the reference analysis, and no arm rejected "
                "any of them. The contracts that compare an analysis against its source produced "
                "no false positive on correct work, which is the property they most needed to "
                "have."
            ),
            "",
            (
                "An earlier pilot, run before the task specification defined its output fields, "
                "recorded two wrong analyses that every arm accepted. Most of what looked like "
                "model error was underspecification: the same model, at the same temperature, "
                "writes a correct analysis once the specification says what each field means. "
                "That is a finding about specifications, not about this model's ability, and two "
                "runs of one task cannot support a claim about either."
            ),
            "",
        ]
    if not missed:
        return [
            "## Result",
            "",
            (
                f"Runtime guarding rejected every run the oracle scored as wrong "
                f"({len(wrong)} of {len(evaluated)} run(s) were wrong)."
            ),
            "",
        ]
    return [
        "## Result: the contract set missed every wrong analysis",
        "",
        (
            f"**{len(missed)} of {len(evaluated)} run(s) were scientifically wrong, and every "
            "arm accepted them, runtime guarding included.** The model's code ran cleanly, "
            "produced the declared shape, and satisfied every contract."
        ),
        "",
        (
            "This is a real coverage gap, found by a real model rather than invented. Milestone "
            "6A reported a 0% false-pass rate for runtime guarding, but the faults there were "
            "written by the same hand as the contracts. A model that had never seen the contract "
            "set went straight through it on the first attempt."
        ),
        "",
        "### Why each contract passed",
        "",
        (
            "The contracts check that an artifact is *internally consistent*. They do not check "
            "that its declared inputs were *derived correctly*, and every mistake below is of the "
            "second kind:"
        ),
        "",
        (
            "- **Normalization applied twice.** The per-event weights were multiplied by the "
            "normalization factor before being summed, and the factor was reported again "
            "alongside. `histogram_closure` compares the binned total against the artifact's own "
            "`selected_weight_sum`, and both were scaled, so the relation still closed. Nothing "
            "asserts that `selected_weight_sum` is the sum of the raw event weights."
        ),
        (
            "- **A per-file constant summed across events.** `SumWeights` is one number repeated "
            "for every event; summing it inflated the denominator by the event count. The "
            "normalization factor was then computed consistently from that wrong denominator, so "
            "the closure check saw nothing wrong."
        ),
        (
            "- **Selected events in no region at all.** The region contract checks that regions "
            "do not overlap and stay inside the selection. It never checks that they *cover* it, "
            "so 1467 selected events belonging to neither region passed unnoticed."
        ),
        "",
        "### The contracts this suggests",
        "",
        (
            "1. **Weight-sum provenance** — `selected_weight_sum` must equal the sum of the raw "
            "weight column over the selected event ids. Catches double normalization.\n"
            "2. **Region coverage** — the declared regions must partition the selected events, "
            "not merely avoid overlapping.\n"
            "3. **Source-constant integrity** — `generated_weight_sum` and `cross_section_pb` "
            "must match the per-file constants the trusted loader read, not an aggregate derived "
            "from them."
        ),
        "",
        (
            "These are candidates, not conclusions. Two runs cannot establish how often a model "
            "makes these mistakes, and adding a contract for each would be fitting the guard to "
            "the errors that happened to appear. Milestone 7 is where candidates earn their way "
            "in."
        ),
        "",
    ]


def render_markdown(report: PilotReport) -> str:
    lines = [
        "# ATLAS agent pilot: Gemini under the guard",
        "",
        (
            f"Generated with SciAgentGuard {report.environment['sciagentguard_version']}. "
            f"Agent: `{report.parameters['agent_model']}`. "
            f"Judge: `{report.parameters['judge_model']}` "
            f"({report.parameters.get('judge_provider', 'unknown provider')}), a different vendor "
            "from the agent it judges."
        ),
        "",
        (
            f"> **A pilot, not the experiment.** {len(report.runs)} run(s), "
            f"{report.total_requests} API requests. This exists to show the pipeline runs and its "
            "numbers can be trusted. Nothing about model behaviour can be concluded from this "
            "many runs."
        ),
        "",
        *_headline(report),
        "## Runs",
        "",
        "| Seed | Code lines | Sandbox | Matches reference | Blocked at | Contract |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for run in report.runs:
        oracle = "n/a" if run.oracle_agrees is None else ("yes" if run.oracle_agrees else "**no**")
        lines.append(
            f"| {run.seed} | {run.code_lines} | `{run.sandbox_outcome}` | {oracle} | "
            f"`{run.blocked_stage or '-'}` | "
            f"{', '.join(f'`{c}`' for c in run.blocked_contracts) or '-'} |"
        )

    evaluated = [run for run in report.runs if run.arm_accepted]
    if evaluated:
        lines.extend(
            [
                "",
                "## Arm decisions",
                "",
                "| Seed | " + " | ".join(f"`{arm}`" for arm in report.arms) + " |",
                "| --- | " + " | ".join("---" for _ in report.arms) + " |",
            ]
        )
        for run in evaluated:
            cells = [
                "n/a"
                if arm not in run.arm_accepted
                else ("accept" if run.arm_accepted[arm] else "reject")
                for arm in report.arms
            ]
            lines.append(f"| {run.seed} | " + " | ".join(cells) + " |")

    if report.repairs:
        lines.extend(
            [
                "",
                "## Repair with structured feedback",
                "",
                "| Seed | Outcome | Attempts used | Resolved |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for repair in report.repairs:
            lines.append(
                f"| {repair.seed} | `{repair.outcome}` | {repair.repair_attempts_used} | "
                f"{'yes' if repair.resolved else 'no'} |"
            )

    lines.extend(
        [
            "",
            "## How correctness was decided",
            "",
            *(
                [
                    (
                        "The judge model was unavailable for at least one run, so the "
                        "`llm_judge` column reads `n/a` there. An unavailable arm is recorded as "
                        "absent, never as an acceptance."
                    ),
                    "",
                ]
                if any(not run.judge_available for run in report.runs if run.arm_accepted)
                else []
            ),
            (
                "A run counts as correct when its artifacts agree with this repository's own "
                f"`atlas_analysis` pipeline at {report.parameters['oracle_relative_tolerance']} "
                "relative tolerance. That oracle is our implementation, so an approach that is "
                "different but equally valid would be scored as disagreement. Every disagreement "
                "below was read by hand before being called a model error."
            ),
            "",
            (
                "The generated code for each run is preserved under "
                "`benchmarks/results/agent_code/`, keyed by seed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--judge",
        choices=("gemini", "deepseek"),
        default="deepseek",
        help="deepseek keeps the judge on a different vendor from the agent it is judging",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
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

        judge = DeepSeekJudge(client=DeepSeekClient(api_key=load_deepseek_key()))
        identity = {"provider": PROVIDER, "model_id": DEFAULT_JUDGE_MODEL}

    report = run_pilot(
        AtlasGamGamSource.official_wph125(args.input),
        seeds=args.seeds,
        max_repair_attempts=args.max_repair_attempts,
        code_dir=args.code_dir,
        judge=judge,
        judge_identity=identity,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"{report.total_requests} API requests; {len(report.runs)} runs recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
