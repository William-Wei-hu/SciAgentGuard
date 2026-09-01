"""Gate 2 must be decided by running contracts, and every candidate must be accounted for."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.core import ContractContext
from sciagentguard.discovery import Candidate, GateOutcome
from tests.integration._atlas_root import synthetic_source, write_root_file

ROOT = Path(__file__).parents[2]
GATE2 = ROOT / "benchmarks" / "discovery_gate2.py"
SAVED_ROUND = ROOT / "benchmarks" / "results" / "discovery_round2.json"
NAMESPACE = runpy.run_path(str(GATE2))

JUDGE = cast(Callable[..., Candidate], NAMESPACE["judge"])
VIOLATING = cast(Callable[..., ContractContext], NAMESPACE["violating_context"])
RENDER = cast(Callable[[Mapping[str, tuple[Candidate, ...]]], str], NAMESPACE["render_markdown"])
PROPOSITIONS = cast(Sequence[Any], NAMESPACE["PROPOSITIONS"])
MAPPING = cast(Mapping[tuple[str, int], Any], NAMESPACE["MAPPING"])
UPSTREAM = cast(type, NAMESPACE["Upstream"])
REJECTION = cast(type, NAMESPACE["Rejection"])
STAGE_INDEX = cast(Mapping[str, int], NAMESPACE["STAGE_INDEX"])
CONTRACTS_OF = cast(Callable[..., Sequence[Any]], NAMESPACE["contracts_of"])
ARTIFACT_OF = cast(Mapping[str, str], NAMESPACE["ARTIFACT_OF"])


@pytest.fixture(scope="module")
def adapter(tmp_path_factory: pytest.TempPathFactory) -> AtlasGamGamOpenDataAdapter:
    path = tmp_path_factory.mktemp("gate2") / "test.root"
    write_root_file(path)
    return AtlasGamGamOpenDataAdapter(synthetic_source(path))


@pytest.fixture(scope="module")
def contexts(adapter: AtlasGamGamOpenDataAdapter) -> tuple[ContractContext, ...]:
    return adapter.contexts(workflow_id="gate2", run_id="r", attempt_id="a")


def _candidate(artifact: str, position: int, stage: str) -> Candidate:
    return Candidate(
        candidate_id=f"round2-{artifact}-{position}",
        round_index=2,
        artifact_reviewed=artifact,
        stage=stage,
        objection="the sentence under test",
    )


# --- the violations must actually violate something -------------------------------------


@pytest.mark.parametrize("proposition", PROPOSITIONS, ids=lambda p: str(p.proposition_id))
def test_every_violation_changes_the_artifact_it_claims_to_break(
    proposition: Any, contexts: tuple[ContractContext, ...]
) -> None:
    """A mutation that leaves the artifact alone would make any check look already covered."""

    name = ARTIFACT_OF[proposition.stage]
    original = contexts[STAGE_INDEX[proposition.stage]].artifacts[name]
    mutated = VIOLATING(contexts, proposition).artifacts[name]

    assert dict(cast(Mapping[str, Any], mutated)) != dict(cast(Mapping[str, Any], original))


# --- the three verdicts ------------------------------------------------------------------


def test_the_gap_this_round_confirmed_is_no_longer_a_gap(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """The loop closing, asserted end to end.

    Sentence 0 is the background relation. When round 2 ran, no contract fired on a violation of
    it; a maintainer signed it, `background_estimate` was written, and the same violation is now
    caught. A discovery loop that cannot show this has not closed.
    """

    judged = JUDGE(_candidate("yield_estimate", 0, "post_yield"), contexts, adapter)

    assert judged.outcome is GateOutcome.REJECTED_NOT_NOVEL
    assert judged.novelty is not None
    assert "hep.atlas_open_data.background_estimate" in judged.novelty.contracts_that_fired


def test_the_earlier_measurement_is_reproducible_by_excluding_what_was_adopted(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """A gate-2 tally means nothing without the contract set it was taken against.

    The same sentence is a gap before a contract is adopted and covered after, so the round's own
    numbers stop being reproducible the moment its finding is acted on -- unless the contract can
    be excluded by name, which is what this asserts.
    """

    adopted = "hep.atlas_open_data.background_estimate"
    candidate = _candidate("yield_estimate", 0, "post_yield")

    assert JUDGE(candidate, contexts, adapter).outcome is GateOutcome.REJECTED_NOT_NOVEL
    as_found = JUDGE(candidate, contexts, adapter, without=(adopted,))
    assert as_found.novelty is not None
    assert as_found.novelty.is_novel
    assert adopted not in as_found.novelty.contracts_evaluated


def test_excluding_every_contract_of_a_stage_is_refused_rather_than_reported_as_a_gap(
    adapter: AtlasGamGamOpenDataAdapter,
) -> None:
    """`prove_novelty` needs something to have been evaluated, or 'nothing fired' means nothing."""

    remaining = CONTRACTS_OF(
        adapter,
        "post_histogram",
        ("hep.atlas_open_data.histogram_closure",),
    )

    assert remaining == ()


def test_the_one_confirmed_blind_spot_carries_a_signature_and_a_recomputation(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """Gate 3 is a person. The record has to show who, and what they checked to be sure."""

    as_found = JUDGE(
        _candidate("yield_estimate", 0, "post_yield"),
        contexts,
        adapter,
        without=("hep.atlas_open_data.background_estimate",),
    )

    assert as_found.outcome is GateOutcome.CONFIRMED
    assert as_found.contract_id == "hep.atlas_open_data.background_estimate"
    assert as_found.confirmation is not None
    assert as_found.confirmation.confirmed
    assert as_found.confirmation.maintainer
    assert "1613.6540816187858" in as_found.confirmation.independent_check


def test_passing_gate_2_alone_never_confirms_a_candidate(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """The four structural checks of round 2 passed gate 2 and were deliberately not adopted."""

    judged = JUDGE(_candidate("yield_estimate", 2, "post_yield"), contexts, adapter)

    assert judged.novelty is not None
    assert judged.novelty.is_novel
    assert judged.outcome is GateOutcome.PENDING
    assert judged.confirmation is None
    assert judged.contract_id is None


def test_a_check_the_contracts_already_make_is_rejected(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """`estimated_yield` closure is contract 16. However well phrased, it is not a new finding."""

    judged = JUDGE(_candidate("yield_estimate", 1, "post_yield"), contexts, adapter)

    assert judged.outcome is GateOutcome.REJECTED_NOT_NOVEL
    assert judged.novelty is not None
    assert "hep.atlas_open_data.yield_closure" in judged.novelty.contracts_that_fired


def test_a_check_nothing_makes_is_left_pending_for_a_person(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """Gate 2 passing is not confirmation. Gate 3 is a signature, and this is not one.

    Sentence 2 is the ordering of the signal window, which round 2's sign-off deliberately left
    unadopted. If it is ever adopted this test must move to another unadopted sentence, and that
    is the intended cost of asserting on a real gap rather than a fabricated one.
    """

    judged = JUDGE(_candidate("yield_estimate", 2, "post_yield"), contexts, adapter)

    assert judged.outcome is GateOutcome.PENDING
    assert judged.novelty is not None
    assert judged.novelty.is_novel
    assert judged.novelty.contracts_evaluated
    assert judged.notes["decidable_at_this_stage"] is True


def test_a_check_the_final_artifact_cannot_decide_is_marked_upstream_not_confirmed(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """The reviewer naming what it lacks is the round's finding, not a contract to be written."""

    judged = JUDGE(_candidate("yield_estimate", 6, "post_yield"), contexts, adapter)

    assert judged.outcome is GateOutcome.PENDING
    assert judged.notes["decidable_at_this_stage"] is False
    assert "per-bin weight sums" in str(judged.notes["needs"])
    assert judged.novelty is None


def test_a_sentence_about_the_presentation_is_not_a_check(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """The summaries are built from min, max and slices, so no run can violate that property."""

    judged = JUDGE(_candidate("selection", 8, "post_selection"), contexts, adapter)

    assert judged.outcome is GateOutcome.REJECTED_NOT_EXPRESSIBLE


def test_a_check_that_would_fail_on_correct_data_is_rejected_as_unscientific(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """This sample's event weights take both signs, so requiring nonnegative ones is wrong."""

    judged = JUDGE(_candidate("histogram", 5, "post_histogram"), contexts, adapter)

    assert judged.outcome is GateOutcome.REJECTED_NOT_SCIENTIFIC


def test_a_property_the_accessor_enforces_is_not_reported_as_an_uncovered_gap(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    """Duplicate event ids are refused before any contract runs. That is coverage, not a gap."""

    judged = JUDGE(_candidate("selection", 13, "post_selection"), contexts, adapter)

    assert judged.outcome is GateOutcome.REJECTED_NOT_NOVEL
    assert "accessor precondition" in str(judged.notes["enforced_by"])


def test_a_sentence_with_no_mapping_is_rejected_rather_than_quietly_confirmed(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    judged = JUDGE(_candidate("yield_estimate", 99, "post_yield"), contexts, adapter)

    assert judged.outcome is GateOutcome.REJECTED_NOT_EXPRESSIBLE


# --- the mapping must account for the round that was actually run -------------------------


def test_every_candidate_of_the_saved_round_has_a_reading() -> None:
    """An unmapped sentence would silently become a rejection, hiding work never done."""

    stored = json.loads(SAVED_ROUND.read_text("utf-8"))
    unmapped = [
        candidate["candidate_id"]
        for record in stored["conditions"].values()
        for candidate in record["round"]["candidates"]
        if (candidate["artifact_reviewed"], int(candidate["candidate_id"].rsplit("-", 1)[1]))
        not in MAPPING
    ]

    assert unmapped == []


def test_every_proposition_is_reachable_from_some_sentence() -> None:
    """A violation nobody proposed is one we invented, and this round records what was proposed."""

    referenced = {value for value in MAPPING.values() if isinstance(value, str)}

    assert {p.proposition_id for p in PROPOSITIONS} == referenced


def test_the_review_sheet_refuses_to_call_anything_a_contract(
    contexts: tuple[ContractContext, ...], adapter: AtlasGamGamOpenDataAdapter
) -> None:
    judged = {
        "A": (
            JUDGE(_candidate("yield_estimate", 2, "post_yield"), contexts, adapter),
            JUDGE(_candidate("yield_estimate", 6, "post_yield"), contexts, adapter),
        )
    }
    markdown = RENDER(judged)

    assert "Nothing here is a contract yet" in markdown
    assert "no signature has been given" in markdown
    assert "**gap**" in markdown
    assert "**upstream only**" in markdown
