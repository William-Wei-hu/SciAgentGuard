"""Ask one reviewer what is wrong with artifacts it is told nothing about, under two conditions.

Milestone 5 established that *detection* at intermediate checkpoints catches faults a final-only
check cannot: three faults whose final artifacts are byte-identical to the correct one prove that
blindness is structural rather than incidental. The open question is whether the same holds for
*discovery* -- whether a reviewer shown intermediate artifacts proposes checks that a reviewer shown
only the final artifact could not have proposed at all.

Two conditions answer it. Condition A shows the reviewer the yield estimate alone; condition B shows
it the selection, the histogram and the yield estimate, one call each. Both review the artifacts of
the *correct* run, and neither is shown the contract list: telling the reviewer where the gaps are
would turn the novelty gate into a restatement of the prompt, and the one finding this loop has
produced so far came from a reviewer that knew nothing about the contract set.

This script collects objections and nothing more. Every candidate it records is PENDING. Gate 1 and
gate 3 need a person; gate 2 needs a violating artifact built by hand for each objection, which is
work that follows the round rather than part of it.

The cache is enabled here, unlike in the reproducibility probe: this measures what a reviewer finds,
not how consistently it answers, and a resumed run should not repurchase answers already paid for.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from sciagentguard import __version__
from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.adapters.agent.deepseek import (
    DEFAULT_JUDGE_MODEL,
    PROVIDER,
    DeepSeekClient,
    DeepSeekError,
    load_deepseek_key,
)
from sciagentguard.adapters.agent.verdict_cache import (
    CachedCompletion,
    CachingClient,
    VerdictCache,
)
from sciagentguard.core import ContractContext
from sciagentguard.discovery import (
    CHECKS,
    OBJECTIONS,
    Candidate,
    Reviewer,
    ReviewQuestion,
    ReviewTarget,
    RoundRecord,
    collect_candidates,
    open_round,
    summarise_artifact,
)
from sciagentguard.discovery.review import SEQUENCE_SAMPLE_SIZE, SEQUENCE_SUMMARY_THRESHOLD

EXPERIMENT_ID: Literal["discovery_round"] = "discovery_round"
ROUND_INDEX = 1
DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/discovery_round.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/discovery_round_1.md")
DEFAULT_CACHE = Path(".cache/discovery-reviews.json")
# Measured, not guessed, and measured twice. A verdict prompt ("answer VALID or INVALID") fits in
# 32768 output tokens; a review prompt ("list every objection you can justify") does not. At 32768
# the reviewer exhausted the budget on reasoning and returned finish_reason="length" with no
# content at all -- twice on the selection artifact and once on the histogram.
#
# That failure is not a random loss. The budget runs out on the artifacts with the most to check,
# which are precisely the ones condition B exists to put in front of the reviewer, so a round that
# tolerated it would understate condition B by construction. 65536 and 131072 were both accepted by
# the API; the larger is used because the failures give a lower bound on what is needed, not an
# upper one.
REVIEW_MAX_TOKENS = 131072

CONDITIONS: dict[str, tuple[str, ...]] = {
    "A": ("yield_estimate",),
    "B": ("selection", "histogram", "yield_estimate"),
}
CONDITION_DESCRIPTIONS = {
    "A": "end only: the reviewer sees the final artifact and nothing else",
    "B": "stage-wise: the reviewer sees each checkpoint artifact in a separate call",
}

QUESTIONS = {question.question_id: question for question in (OBJECTIONS, CHECKS)}

# Written down before each round ran, so that whatever comes back is compared against a claim made
# in advance rather than one fitted afterwards.
PREDICTIONS = {
    "objections": (
        "Condition A raises objections only about the yield artifact's internal consistency. "
        "Condition B additionally raises objections about the cutflow, region membership and "
        "weight provenance, because those quantities exist only upstream. If A matches B, the "
        "stage-wise claim does not extend from detection to discovery, and that is the honest "
        "finding."
    ),
    "checks": (
        "Condition A proposes checks over the twelve values of the yield artifact and nothing "
        "else, because it is shown nothing else. Condition B additionally proposes checks over "
        "the cutflow, the region partition and the weight provenance. The comparison that decides "
        "the round is not the count but the membership: whether B proposes checks naming "
        "quantities that do not appear in the final artifact at all, and which A therefore could "
        "not have proposed. If the two sets coincide once duplicates are removed, the stage-wise "
        "claim does not extend from detection to discovery, and that is the honest finding."
    ),
}
PREDICTION = PREDICTIONS["objections"]


class TargetRecord(BaseModel):
    """What one review call cost and produced."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    artifact_name: str
    stage: str
    prompt_chars: int
    calls: int
    objections: int
    latency_ms: float | None = None
    error: str | None = None


class ConditionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    condition: str
    description: str
    artifacts_shown: tuple[str, ...]
    targets: tuple[TargetRecord, ...]
    round: RoundRecord

    @property
    def objections(self) -> int:
        return len(self.round.candidates)

    @property
    def failed_targets(self) -> tuple[str, ...]:
        return tuple(target.artifact_name for target in self.targets if target.error is not None)


class DiscoveryRoundReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    experiment_id: Literal["discovery_round"] = EXPERIMENT_ID
    generated_at: datetime
    environment: dict[str, str]
    parameters: dict[str, str | int | float]
    prediction: str = PREDICTION
    conditions: dict[str, ConditionRecord]


# --- building what the reviewer sees --------------------------------------------------


def build_targets(
    contexts: Sequence[ContractContext], artifact_names: Sequence[str]
) -> tuple[ReviewTarget, ...]:
    """Locate each named artifact among the ordered checkpoints and summarise it for review."""

    located: dict[str, ReviewTarget] = {}
    for context in contexts:
        for name, artifact in context.artifacts.items():
            if not isinstance(artifact, Mapping):
                continue
            located[name] = ReviewTarget(
                artifact_name=name,
                stage=context.stage,
                artifact=summarise_artifact(artifact),
            )
    missing = [name for name in artifact_names if name not in located]
    if missing:
        raise KeyError(f"no checkpoint produced the artifact(s) {missing}")
    return tuple(located[name] for name in artifact_names)


# --- running one condition ------------------------------------------------------------

Completion = Callable[[str], CachedCompletion]


def _capturing_reviewer(completion: Completion, captured: list[CachedCompletion]) -> Reviewer:
    """A reviewer that also keeps the completion, so the report can state what the call cost."""

    def reviewer(prompt: str) -> str:
        answer = completion(prompt)
        captured.append(answer)
        return answer.text

    return reviewer


def review_target(
    target: ReviewTarget,
    completion: Completion,
    *,
    round_index: int,
    max_attempts: int,
    question: ReviewQuestion = OBJECTIONS,
) -> tuple[tuple[Candidate, ...], TargetRecord]:
    """Ask the reviewer about one artifact, retrying transport failures.

    A reviewer call here runs for minutes and fails often enough that a round which aborted on the
    first failure would rarely finish. A target that never answers is recorded as an error rather
    than dropped: a condition that could not be asked is not a condition that found nothing.
    """

    prompt_chars = len(question.build(target.artifact))
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        captured: list[CachedCompletion] = []
        reviewer = _capturing_reviewer(completion, captured)
        started = monotonic()
        try:
            candidates = collect_candidates(
                [target], reviewer, round_index=round_index, question=question
            )
        except DeepSeekError as error:
            last_error = f"{type(error).__name__}: {error}"
            print(f"  {target.artifact_name}: attempt {attempt} failed -- {last_error}", flush=True)
            continue
        latency = captured[0].latency_ms if captured else (monotonic() - started) * 1000.0
        return candidates, TargetRecord(
            artifact_name=target.artifact_name,
            stage=target.stage,
            prompt_chars=prompt_chars,
            calls=attempt,
            objections=len(candidates),
            latency_ms=latency,
        )

    return (), TargetRecord(
        artifact_name=target.artifact_name,
        stage=target.stage,
        prompt_chars=prompt_chars,
        calls=max_attempts,
        objections=0,
        error=last_error,
    )


def run_condition(
    condition: str,
    targets: Sequence[ReviewTarget],
    completion: Completion,
    *,
    reviewer_identity: Mapping[str, str | None],
    round_index: int = ROUND_INDEX,
    max_attempts: int = 2,
    question: ReviewQuestion = OBJECTIONS,
    checkpoint: Callable[[ConditionRecord], None] | None = None,
) -> ConditionRecord:
    """Review every target, saving after each so a crash keeps the answers already paid for."""

    candidates: list[Candidate] = []
    records: list[TargetRecord] = []
    for target in targets:
        print(f"[{condition}] reviewing {target.artifact_name} ({target.stage})", flush=True)
        found, record = review_target(
            target,
            completion,
            round_index=round_index,
            max_attempts=max_attempts,
            question=question,
        )
        candidates.extend(found)
        records.append(record)
        print(
            f"[{condition}] {target.artifact_name}: {record.objections} objection(s)"
            + (f" after {record.latency_ms / 1000.0:.0f}s" if record.latency_ms else ""),
            flush=True,
        )
        if checkpoint is not None:
            checkpoint(
                _build_condition(
                    condition,
                    targets[: len(records)],
                    records,
                    candidates,
                    reviewer_identity,
                    round_index,
                )
            )

    return _build_condition(condition, targets, records, candidates, reviewer_identity, round_index)


def _build_condition(
    condition: str,
    targets: Sequence[ReviewTarget],
    records: Sequence[TargetRecord],
    candidates: Sequence[Candidate],
    reviewer_identity: Mapping[str, str | None],
    round_index: int,
) -> ConditionRecord:
    return ConditionRecord(
        condition=condition,
        description=CONDITION_DESCRIPTIONS.get(condition, condition),
        artifacts_shown=tuple(target.artifact_name for target in targets),
        targets=tuple(records),
        round=open_round(round_index, reviewer_identity, targets, candidates),
    )


# --- report -----------------------------------------------------------------------------


def build_report(
    conditions: Mapping[str, ConditionRecord],
    parameters: Mapping[str, str | int | float],
    prediction: str = PREDICTION,
) -> DiscoveryRoundReport:
    return DiscoveryRoundReport(
        prediction=prediction,
        generated_at=datetime.now(timezone.utc),
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
        },
        parameters=dict(parameters),
        conditions=dict(conditions),
    )


def load_conditions(path: Path) -> dict[str, ConditionRecord]:
    """Recover conditions already recorded, so A and B can be run on separate days."""

    if not path.is_file():
        return {}
    try:
        return dict(DiscoveryRoundReport.model_validate_json(path.read_text("utf-8")).conditions)
    except (OSError, ValidationError, ValueError):
        return {}


def render_markdown(report: DiscoveryRoundReport) -> str:
    lines = [
        f"# Discovery round {report.parameters.get('round_index', ROUND_INDEX)}: "
        "does stage-wise review find what end-only review cannot?",
        "",
        (
            f"Reviewer: `{report.parameters.get('model_id')}` at temperature "
            f"{report.parameters.get('temperature')}, shown one artifact per call and never shown "
            "the contract list."
        ),
        "",
        "## Prediction, recorded before the round ran",
        "",
        f"> {report.prediction}",
        "",
        "## What each condition produced",
        "",
        "| Condition | Shown | Objections | Failed calls |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in sorted(report.conditions):
        record = report.conditions[name]
        shown = ", ".join(f"`{artifact}`" for artifact in record.artifacts_shown)
        failed = len(record.failed_targets)
        lines.append(
            f"| **{name}** -- {record.description} | {shown} | {record.objections} | {failed} |"
        )

    lines.extend(
        [
            "",
            "## Per call",
            "",
            "| Condition | Artifact | Stage | Prompt | Calls | Objections | Latency |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in sorted(report.conditions):
        for target in report.conditions[name].targets:
            latency = f"{target.latency_ms / 1000.0:.0f} s" if target.latency_ms else "--"
            note = " (failed)" if target.error else ""
            lines.append(
                f"| {name} | `{target.artifact_name}`{note} | `{target.stage}` | "
                f"{target.prompt_chars:,} chars | {target.calls} | "
                f"{target.objections} | {latency} |"
            )

    failures = [
        (name, target)
        for name in sorted(report.conditions)
        for target in report.conditions[name].targets
        if target.error is not None
    ]
    if failures:
        lines.extend(
            [
                "",
                "## Calls that returned nothing",
                "",
                (
                    "A target whose calls all failed is recorded as an error rather than as "
                    "silence: a condition that could not be asked is not a condition that found "
                    "nothing. This loss is not random -- the reviewer spends its output budget on "
                    "the artifacts with the most to check, which are the ones the stage-wise "
                    "condition exists to show it -- so every failure below biases the comparison "
                    "against condition B, never towards it."
                ),
                "",
            ]
        )
        for name, target in failures:
            lines.append(f"- **{name}** / `{target.artifact_name}`: {target.error}")

    for name in sorted(report.conditions):
        record = report.conditions[name]
        lines.extend(["", f"## Condition {name}: every objection raised", ""])
        if not record.round.candidates:
            lines.append("No objection was recorded.")
            continue
        for candidate in record.round.candidates:
            lines.append(
                f"- `{candidate.candidate_id}` (`{candidate.artifact_reviewed}`) "
                f"{candidate.objection}"
            )

    lines.extend(["", "## What this round can and cannot conclude", "", _reading(report), ""])

    lines.extend(
        [
            "",
            "## Status of these candidates",
            "",
            (
                "Every candidate above is **pending**. Collection decides nothing: gate 1 asks "
                "whether an objection can be written as a deterministic check, gate 2 requires a "
                "run showing the current contracts accepting a violation of it, and gate 3 needs a "
                "maintainer who has recomputed the claim. The attrition across those three gates "
                "is the measurement this round exists to produce, and none of it has happened yet."
            ),
            "",
            "## What the reviewer was shown",
            "",
            (
                f"Sequences longer than {report.parameters.get('sequence_summary_threshold')} "
                f"elements were replaced by a labelled summary -- count, first and last "
                f"{report.parameters.get('sequence_sample_size')} values, minimum and maximum -- "
                "because the selection artifact serialises to 6.6 MB on this sample. Every number "
                "in a summary is computed from the values it replaces, and the label sits inside "
                "the JSON so the reviewer reads it in place. The histogram and the yield estimate "
                "fall under the threshold and reached the reviewer untouched."
            ),
            "",
            (
                "One round, one model, one task, one domain. This cannot show that discovery "
                "decays -- that needs later rounds, each absorbing the contracts of the one before "
                "it -- and it cannot compare the reviewer against a solver, because this setting "
                "has none."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _reading(report: DiscoveryRoundReport) -> str:
    """State what the counts support, and refuse the reading they do not support.

    A round in which no condition raised anything is the case that most invites a false summary,
    because "the reviewer found no blind spot" and "the reviewer was asked a question this artifact
    cannot answer" produce the identical number. The two are distinguished by what was shown, not
    by the count, so the distinction has to be written down rather than inferred.
    """

    conditions = report.conditions
    if not conditions:
        return "No condition has been run."

    seconds = sum(
        (target.latency_ms or 0.0) / 1000.0
        for record in conditions.values()
        for target in record.targets
    )
    calls = sum(target.calls for record in conditions.values() for target in record.targets)
    raised = {name: record.objections for name, record in sorted(conditions.items())}

    if any(raised.values()):
        return (
            "Candidates were raised, and the comparison the round exists to make is between the "
            "ones a condition could not have raised at all -- those naming quantities absent from "
            "what the other condition was shown. That comparison is made after the gates, not "
            "here: a count of candidates is not a count of blind spots."
        )

    if str(report.parameters.get("question", "objections")) != "objections":
        return (
            f"**No condition proposed anything.** {calls} call(s) and "
            f"{seconds / 60.0:.0f} minutes of deliberation produced not one check.\n\n"
            "Unlike a round that asks what contradicts what, this one could have failed its "
            "prediction: a correct artifact can still motivate checks, so a reviewer that names "
            "none has answered a question it was able to answer. Read it as a statement about "
            "this reviewer on these artifacts, not about whether the checks exist."
        )

    return (
        f"**No condition raised anything.** {calls} call(s) and "
        f"{seconds / 60.0:.0f} minutes of deliberation produced not one objection.\n\n"
        "This does not show that stage-wise review finds nothing an end-only review misses. Every "
        "artifact shown came from the *correct* run, and the question asked was which quantities "
        "**contradict each other** -- a question whose right answer on a self-consistent artifact "
        "is silence. Both conditions were therefore asked something neither could answer, and the "
        "hypothesis was never put at risk. A round that cannot fail its prediction has not tested "
        "it.\n\n"
        "What the run does establish is a property of the reviewer rather than of the contracts: "
        "shown correct artifacts, it manufactures nothing. The same reviewer, on an artifact whose "
        "arithmetic did not close, produced the objection that became `yield_closure`. It objects "
        "when there is something to object to and stays silent when there is not, which is what "
        "separates a reviewer from a complaint generator -- and it is the precondition for reading "
        "any future objection as evidence.\n\n"
        "Asking instead which relations *should be verified and cannot be confirmed from these "
        "values alone* is a different question, and the one contract discovery actually needs."
    )


# --- CLI ----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--condition", choices=("A", "B", "both"), default="both")
    parser.add_argument(
        "--question",
        choices=tuple(QUESTIONS),
        default="objections",
        help=(
            "which question to put to the reviewer. 'objections' asks what contradicts what, "
            "which a self-consistent artifact answers with silence; 'checks' asks what would have "
            "to be verified, which it can answer."
        ),
    )
    parser.add_argument("--round-index", type=int, default=ROUND_INDEX)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "rewrite the markdown from the stored JSON without contacting a provider. Replaying a "
            "round through the cache would rewrite its call counts, erasing the record of calls "
            "that exhausted their budget and returned nothing."
        ),
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        stored = DiscoveryRoundReport.model_validate_json(args.json_output.read_text("utf-8"))
        args.markdown_output.write_text(render_markdown(stored), encoding="utf-8")
        print(json.dumps({"rendered": str(args.markdown_output)}))
        return 0

    cache = VerdictCache(path=args.cache)
    client = CachingClient(
        inner=DeepSeekClient(api_key=load_deepseek_key()), cache=cache, provider=PROVIDER
    )

    question = QUESTIONS[args.question]

    def completion(prompt: str) -> CachedCompletion:
        return client.completion(
            DEFAULT_JUDGE_MODEL, prompt, temperature=0.0, max_tokens=REVIEW_MAX_TOKENS
        )

    adapter = AtlasGamGamOpenDataAdapter(AtlasGamGamSource.official_wph125(args.input))
    contexts = adapter.contexts(workflow_id="discovery-round-1", run_id="r1", attempt_id="a1")

    conditions = load_conditions(args.json_output)
    wanted = ("A", "B") if args.condition == "both" else (args.condition,)

    def save() -> None:
        report = build_report(
            conditions,
            {
                "provider": PROVIDER,
                "model_id": DEFAULT_JUDGE_MODEL,
                "temperature": 0.0,
                "round_index": args.round_index,
                "question": question.question_id,
                "max_output_tokens": REVIEW_MAX_TOKENS,
                "sequence_summary_threshold": SEQUENCE_SUMMARY_THRESHOLD,
                "sequence_sample_size": SEQUENCE_SAMPLE_SIZE,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "contracts_shown_to_reviewer": "none",
            },
            prediction=PREDICTIONS[question.question_id],
        )
        args.json_output.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")

    for name in wanted:
        targets = build_targets(contexts, CONDITIONS[name])

        def checkpoint(partial: ConditionRecord, _name: str = name) -> None:
            conditions[_name] = partial
            save()

        conditions[name] = run_condition(
            name,
            targets,
            completion,
            reviewer_identity={"provider": PROVIDER, "model_id": DEFAULT_JUDGE_MODEL},
            round_index=args.round_index,
            max_attempts=args.max_attempts,
            question=question,
            checkpoint=checkpoint,
        )
        save()

    save()
    print(
        json.dumps(
            {
                name: {
                    "objections": record.objections,
                    "failed_targets": list(record.failed_targets),
                }
                for name, record in sorted(conditions.items())
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
