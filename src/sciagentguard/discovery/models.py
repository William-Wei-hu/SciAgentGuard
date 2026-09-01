"""Records of one contract-discovery round.

The measurement this module exists to produce is not a coverage percentage -- the integrity rules
forbid claiming one -- but the number of blind spots a review round confirms, and the counts of
what was rejected at each gate. Rejected candidates are kept, hallucinations included, because a
round that proposes twenty things and confirms one is a different result from a round that proposes
one and confirms it.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from sciagentguard.core.models import NonEmptyString


class CandidateKind(str, Enum):
    """How a candidate would be turned into a check.

    A candidate need not be a fault. The first finding this loop produced was a relation that
    should hold and was never checked -- there was nothing to inject, only something to verify.
    """

    FAULT = "fault"
    RELATION = "relation"


class GateOutcome(str, Enum):
    """Where a candidate stopped."""

    PENDING = "pending"
    REJECTED_NOT_EXPRESSIBLE = "rejected_not_expressible"
    REJECTED_NOT_NOVEL = "rejected_not_novel"
    REJECTED_NOT_SCIENTIFIC = "rejected_not_scientific"
    CONFIRMED = "confirmed"


class NoveltyEvidence(BaseModel):
    """The run proving the current contract set accepts a violation of the candidate.

    Gate 2 is never decided on the reviewer's word or on ours. Either this record exists and shows
    the contracts passing on a violating artifact, or the candidate does not advance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    violating_artifact_summary: NonEmptyString
    contracts_evaluated: tuple[NonEmptyString, ...]
    contracts_that_fired: tuple[NonEmptyString, ...] = ()
    accepted_by_current_contracts: bool

    @field_validator("contracts_evaluated")
    @classmethod
    def require_contracts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("novelty evidence must record the contracts that were evaluated")
        return values

    @property
    def is_novel(self) -> bool:
        return self.accepted_by_current_contracts and not self.contracts_that_fired


class HumanConfirmation(BaseModel):
    """Gate 3. A maintainer's signature, with the recomputation that justified it."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    confirmed: bool
    maintainer: NonEmptyString
    confirmed_on: datetime
    rationale: NonEmptyString
    independent_check: NonEmptyString

    @field_validator("confirmed_on")
    @classmethod
    def require_timezone(cls, moment: datetime) -> datetime:
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("confirmed_on must include timezone information")
        return moment


class Candidate(BaseModel):
    """One objection a reviewer raised, and how far it got."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    candidate_id: NonEmptyString
    round_index: int
    artifact_reviewed: NonEmptyString
    stage: NonEmptyString
    objection: NonEmptyString
    kind: CandidateKind | None = None
    outcome: GateOutcome = GateOutcome.PENDING
    novelty: NoveltyEvidence | None = None
    confirmation: HumanConfirmation | None = None
    contract_id: str | None = None
    notes: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("round_index")
    @classmethod
    def require_nonnegative_round(cls, index: int) -> int:
        if isinstance(index, bool) or index < 0:
            raise ValueError("round_index must be a nonnegative integer")
        return index

    def with_outcome(self, outcome: GateOutcome, **updates: object) -> Candidate:
        return self.model_copy(update={"outcome": outcome, **updates})


class RoundRecord(BaseModel):
    """Everything one round produced, including what it threw away."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    round_index: int
    reviewer: dict[str, str | None]
    artifacts_reviewed: tuple[NonEmptyString, ...]
    started_at: datetime
    candidates: tuple[Candidate, ...]
    schema_version: NonEmptyString = "1.0"

    @field_validator("started_at")
    @classmethod
    def require_timezone(cls, moment: datetime) -> datetime:
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("started_at must include timezone information")
        return moment

    def count(self, outcome: GateOutcome) -> int:
        return sum(1 for candidate in self.candidates if candidate.outcome is outcome)

    @property
    def confirmed_blind_spots(self) -> int:
        """The round's headline. Never expressed as a fraction of anything."""

        return self.count(GateOutcome.CONFIRMED)

    @property
    def rejections_by_gate(self) -> dict[str, int]:
        return {
            "not_expressible": self.count(GateOutcome.REJECTED_NOT_EXPRESSIBLE),
            "not_novel": self.count(GateOutcome.REJECTED_NOT_NOVEL),
            "not_scientific": self.count(GateOutcome.REJECTED_NOT_SCIENTIFIC),
            "pending": self.count(GateOutcome.PENDING),
        }


__all__ = [
    "Candidate",
    "CandidateKind",
    "GateOutcome",
    "HumanConfirmation",
    "NoveltyEvidence",
    "RoundRecord",
]
