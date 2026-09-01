"""Run one discovery round: ask a reviewer for objections, then test them.

The reviewer sees artifacts believed to be correct and is asked what is wrong with them. That is
the question that worked: asking a model to invent faults invites plausible guesses, while asking
it to scrutinise real output produced an arithmetic objection precise enough to become a contract.

Nothing here decides whether an objection is right. Gate 2 is settled by running the contracts
against a violating artifact; gate 3 is settled by a person.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from pydantic import JsonValue

from sciagentguard.core import ContractContext, ContractStatus, ScientificContract
from sciagentguard.discovery.models import Candidate, NoveltyEvidence, RoundRecord

Reviewer = Callable[[str], str]
"""Takes a prompt, returns the reviewer's raw reply. Injectable so tests never call a provider."""

REVIEW_PROMPT = """You are reviewing one artifact produced by a high-energy physics analysis.

Assume nothing about the code that produced it. Judge only what is written here.

{artifact}

List every objection you can justify from the values above: quantities that contradict each other,
relations that should hold and do not, or claims the artifact makes that its own numbers do not
support.

Rules for your answer:
- one objection per line, each beginning with "- "
- name the specific fields involved and, where possible, the arithmetic
- raise nothing you cannot ground in the values shown
- if you find nothing, reply with exactly: NO OBJECTIONS
"""

CHECK_PROMPT = """You are reviewing one artifact produced by a high-energy physics analysis.

Assume nothing about the code that produced it. Judge only what is written here.

{artifact}

These values are believed to be correct, and your task is not to find an error in them. It is to
say what someone would have to verify before trusting them: relations that must hold among these
quantities, and quantities whose correctness these values alone cannot establish.

Rules for your answer:
- one check per line, each beginning with "- "
- name the specific fields involved and, where possible, the arithmetic that would decide the check
- propose only checks a program could decide from data it can obtain, not matters of judgement
- if these values fully establish their own correctness, reply with exactly: NO CHECKS
"""

_BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_NOTHING = "NO OBJECTIONS"

# A reviewer cannot be shown the selection artifact as it stands: on the real sample it holds three
# 105,503-element sequences and serialises to 6.6 MB, which no prompt will carry. Long sequences are
# therefore replaced by a summary of themselves. Two properties keep that from distorting the
# result. Nothing is added -- every number in a summary is computed from the values it replaces --
# and the summary says in place that it is one, so the reviewer does not object that an array is
# shorter than the count beside it, which is an objection the truncation itself would have created.
#
# The threshold sits above the histogram's 31 bin edges on purpose: the histogram and the yield
# estimate reach the reviewer untouched, and only the per-event sequences are summarised.
SEQUENCE_SUMMARY_THRESHOLD = 32
SEQUENCE_SAMPLE_SIZE = 5


def summarise_artifact(
    artifact: Mapping[str, object],
    *,
    threshold: int = SEQUENCE_SUMMARY_THRESHOLD,
    sample: int = SEQUENCE_SAMPLE_SIZE,
) -> dict[str, JsonValue]:
    """Return the artifact with over-long sequences replaced by labelled summaries of themselves."""

    if threshold < 1 or sample < 1:
        raise ValueError("threshold and sample must be positive")
    return {str(key): _summarise(value, threshold, sample) for key, value in artifact.items()}


def _summarise(value: object, threshold: int, sample: int) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _summarise(item, threshold, sample) for key, item in value.items()}
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return cast(JsonValue, value)

    items = list(value)
    if len(items) <= threshold:
        return [_summarise(item, threshold, sample) for item in items]

    summary: dict[str, JsonValue] = {
        "note": (
            f"a summary of a {len(items)}-element sequence, not the sequence itself; "
            "the values between the samples below are not shown"
        ),
        "count": len(items),
        f"first_{sample}": [_summarise(item, threshold, sample) for item in items[:sample]],
        f"last_{sample}": [_summarise(item, threshold, sample) for item in items[-sample:]],
    }
    numbers = [
        float(item)
        for item in items
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    if len(numbers) == len(items):
        summary["minimum"] = min(numbers)
        summary["maximum"] = max(numbers)
    return summary


def build_review_prompt(artifact_name: str, artifact: Mapping[str, JsonValue]) -> str:
    del artifact_name
    return REVIEW_PROMPT.format(
        artifact=json.dumps(dict(artifact), indent=2, sort_keys=True, default=str)
    )


def parse_objections(reply: str, *, nothing_marker: str = _NOTHING) -> tuple[str, ...]:
    """Pull one candidate per bullet, tolerating the prose models wrap around lists."""

    if nothing_marker.upper() in reply.upper():
        return ()
    objections = [match.group(1) for line in reply.splitlines() if (match := _BULLET.match(line))]
    return tuple(dict.fromkeys(objections))


@dataclass(frozen=True, slots=True)
class ReviewQuestion:
    """One way of asking a reviewer for candidates, and the reply that means it has none.

    Round 1 asked which quantities contradict each other. Every artifact it was shown came from the
    correct run and was self-consistent, so silence was the right answer and the round could not
    fail its own prediction. Asking instead which relations *should be verified* is a question a
    correct artifact can answer, and it is the one contract discovery needs.
    """

    question_id: str
    template: str
    nothing_marker: str

    def build(self, artifact: Mapping[str, JsonValue]) -> str:
        return self.template.format(
            artifact=json.dumps(dict(artifact), indent=2, sort_keys=True, default=str)
        )

    def parse(self, reply: str) -> tuple[str, ...]:
        return parse_objections(reply, nothing_marker=self.nothing_marker)


OBJECTIONS = ReviewQuestion("objections", REVIEW_PROMPT, _NOTHING)
"""What contradicts what. Answerable only by an artifact that is actually wrong."""

CHECKS = ReviewQuestion("checks", CHECK_PROMPT, "NO CHECKS")
"""What would have to be verified. Answerable by a correct artifact, which is the point."""


@dataclass(frozen=True, slots=True)
class ReviewTarget:
    """One artifact to put in front of the reviewer."""

    artifact_name: str
    stage: str
    artifact: Mapping[str, JsonValue]


def collect_candidates(
    targets: Sequence[ReviewTarget],
    reviewer: Reviewer,
    *,
    round_index: int,
    question: ReviewQuestion = OBJECTIONS,
) -> tuple[Candidate, ...]:
    """Ask the reviewer about each artifact and record every candidate it raises."""

    candidates: list[Candidate] = []
    for target in targets:
        reply = reviewer(question.build(target.artifact))
        for position, objection in enumerate(question.parse(reply)):
            candidates.append(
                Candidate(
                    candidate_id=f"round{round_index}-{target.artifact_name}-{position}",
                    round_index=round_index,
                    artifact_reviewed=target.artifact_name,
                    stage=target.stage,
                    objection=objection,
                )
            )
    return tuple(candidates)


def prove_novelty(
    violating: ContractContext,
    contracts: Sequence[ScientificContract],
    summary: str,
) -> NoveltyEvidence:
    """Run the current contracts against an artifact that violates the candidate.

    This is gate 2, and it is the only gate that can be settled without a person. A candidate whose
    violation the contracts already catch is not a blind spot, however well the objection reads.
    """

    fired: list[str] = []
    evaluated: list[str] = []
    for contract in contracts:
        if contract.stage != violating.stage:
            continue
        evaluated.append(contract.contract_id)
        if contract.evaluate(violating).status is ContractStatus.FAIL:
            fired.append(contract.contract_id)
    if not evaluated:
        raise ValueError(f"no contract guards stage {violating.stage!r}")

    return NoveltyEvidence(
        violating_artifact_summary=summary,
        contracts_evaluated=tuple(evaluated),
        contracts_that_fired=tuple(fired),
        accepted_by_current_contracts=not fired,
    )


def open_round(
    round_index: int,
    reviewer_identity: Mapping[str, str | None],
    targets: Sequence[ReviewTarget],
    candidates: Sequence[Candidate],
) -> RoundRecord:
    return RoundRecord(
        round_index=round_index,
        reviewer=dict(reviewer_identity),
        artifacts_reviewed=tuple(target.artifact_name for target in targets),
        started_at=datetime.now(timezone.utc),
        candidates=tuple(candidates),
    )


__all__ = [
    "CHECKS",
    "CHECK_PROMPT",
    "OBJECTIONS",
    "REVIEW_PROMPT",
    "SEQUENCE_SAMPLE_SIZE",
    "SEQUENCE_SUMMARY_THRESHOLD",
    "ReviewQuestion",
    "ReviewTarget",
    "Reviewer",
    "build_review_prompt",
    "collect_candidates",
    "open_round",
    "parse_objections",
    "prove_novelty",
    "summarise_artifact",
]
