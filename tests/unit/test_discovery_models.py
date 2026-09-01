"""The round record has to be able to describe the one real finding this loop has produced."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sciagentguard.discovery import (
    Candidate,
    CandidateKind,
    GateOutcome,
    HumanConfirmation,
    NoveltyEvidence,
    RoundRecord,
)


def _yield_closure_candidate() -> Candidate:
    """Round 0, reconstructed: the objection that produced `yield_closure`.

    A reviewer, shown a final artifact it had no other information about, objected that a signal
    sum of 82,383 and a background of 1,614 cannot produce a yield of 7.47. The scale factor
    relating them was reported one stage earlier and never reached this artifact.
    """

    return Candidate(
        candidate_id="round0-yield-closure",
        round_index=0,
        artifact_reviewed="yield_estimate",
        stage="post_yield",
        objection=(
            "The stated inputs imply a background-subtracted signal of 82382.99 - 1613.65, not "
            "7.47, so the reported estimated_yield is inconsistent with the values beside it."
        ),
        # Nothing to inject. A relation that should hold and was never checked.
        kind=CandidateKind.RELATION,
        outcome=GateOutcome.CONFIRMED,
        novelty=NoveltyEvidence(
            violating_artifact_summary="yield_estimate without normalization_factor",
            contracts_evaluated=(
                "hep.atlas_open_data.yield_shape",
                "hep.atlas_open_data.histogram_closure",
            ),
            contracts_that_fired=(),
            accepted_by_current_contracts=True,
        ),
        confirmation=HumanConfirmation(
            confirmed=True,
            maintainer="maintainer",
            confirmed_on=datetime(2026, 8, 30, tzinfo=timezone.utc),
            rationale="The final artifact stated a result its own reported inputs did not imply.",
            independent_check=(
                "Recomputed (82382.99495381117 - 1613.6540816187858) * 9.251780210572279e-05 "
                "= 7.472601895023162, matching estimated_yield exactly, with the factor absent "
                "from the artifact."
            ),
        ),
        contract_id="hep.atlas_open_data.yield_closure",
    )


def test_the_schema_can_express_the_one_finding_we_have() -> None:
    """If it cannot describe `yield_closure`, the schema is wrong rather than the finding."""

    candidate = _yield_closure_candidate()

    assert candidate.kind is CandidateKind.RELATION
    assert candidate.outcome is GateOutcome.CONFIRMED
    assert candidate.contract_id == "hep.atlas_open_data.yield_closure"
    assert candidate.novelty is not None and candidate.novelty.is_novel
    assert candidate.confirmation is not None and candidate.confirmation.confirmed


def test_a_candidate_only_counts_as_novel_when_the_contracts_actually_passed() -> None:
    """Gate 2 is decided by a run, never by anyone's claim -- the reviewer's or ours."""

    fired = NoveltyEvidence(
        violating_artifact_summary="something a contract already catches",
        contracts_evaluated=("hep.atlas_open_data.yield_closure",),
        contracts_that_fired=("hep.atlas_open_data.yield_closure",),
        accepted_by_current_contracts=False,
    )

    assert not fired.is_novel


def test_novelty_evidence_must_name_what_was_evaluated() -> None:
    with pytest.raises(ValueError, match="contracts that were evaluated"):
        NoveltyEvidence(
            violating_artifact_summary="x",
            contracts_evaluated=(),
            accepted_by_current_contracts=True,
        )


def test_a_round_counts_what_it_threw_away() -> None:
    """A round that proposes twenty and confirms one is not a round that proposes one."""

    confirmed = _yield_closure_candidate()
    rejected = Candidate(
        candidate_id="round0-vague",
        round_index=0,
        artifact_reviewed="yield_estimate",
        stage="post_yield",
        objection="The result feels too small for this luminosity.",
        outcome=GateOutcome.REJECTED_NOT_EXPRESSIBLE,
    )
    stale = Candidate(
        candidate_id="round0-known",
        round_index=0,
        artifact_reviewed="selection",
        stage="post_selection",
        objection="Regions might overlap.",
        outcome=GateOutcome.REJECTED_NOT_NOVEL,
    )

    record = RoundRecord(
        round_index=0,
        reviewer={"provider": "deepseek", "model_id": "deepseek-v4-pro"},
        artifacts_reviewed=("yield_estimate", "selection"),
        started_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        candidates=(confirmed, rejected, stale),
    )

    assert record.confirmed_blind_spots == 1
    assert record.rejections_by_gate == {
        "not_expressible": 1,
        "not_novel": 1,
        "not_scientific": 0,
        "pending": 0,
    }


def test_confirmation_requires_a_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        HumanConfirmation(
            confirmed=True,
            maintainer="m",
            confirmed_on=datetime(2026, 8, 30),
            rationale="r",
            independent_check="c",
        )
