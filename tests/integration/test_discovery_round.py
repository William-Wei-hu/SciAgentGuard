"""A discovery round, driven by a stubbed reviewer so CI never calls a provider."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.core import ContractContext
from sciagentguard.discovery import (
    ReviewTarget,
    collect_candidates,
    open_round,
    parse_objections,
    prove_novelty,
)
from tests.integration._atlas_root import synthetic_source, write_root_file


@pytest.fixture(scope="module")
def adapter(tmp_path_factory: pytest.TempPathFactory) -> AtlasGamGamOpenDataAdapter:
    path = tmp_path_factory.mktemp("discovery") / "test.root"
    write_root_file(path)
    return AtlasGamGamOpenDataAdapter(synthetic_source(path))


@pytest.fixture(scope="module")
def contexts(adapter: AtlasGamGamOpenDataAdapter) -> tuple[ContractContext, ...]:
    return adapter.contexts(workflow_id="discovery", run_id="r", attempt_id="a")


def _artifact(context: ContractContext, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], dict(cast(Mapping[str, Any], context.artifacts[name])))


# --- parsing what a reviewer actually returns -----------------------------------------


def test_objections_are_parsed_from_the_prose_models_wrap_around_them() -> None:
    reply = (
        "Looking at this artifact, I have concerns:\n\n"
        "- estimated_yield does not follow from signal_weight_sum and background_estimate\n"
        "* peak_bin_center_gev sits outside the declared window\n"
        "\nHope that helps."
    )

    assert parse_objections(reply) == (
        "estimated_yield does not follow from signal_weight_sum and background_estimate",
        "peak_bin_center_gev sits outside the declared window",
    )


def test_a_clean_review_produces_no_candidates() -> None:
    assert parse_objections("NO OBJECTIONS") == ()


def test_repeated_objections_are_recorded_once() -> None:
    reply = "- the same point\n- the same point\n"

    assert parse_objections(reply) == ("the same point",)


def test_the_prompt_shows_the_artifact_and_asks_what_is_wrong_with_it(
    contexts: tuple[ContractContext, ...],
) -> None:
    """The question that worked was 'what is wrong with this?', not 'invent a fault'."""

    from sciagentguard.discovery.review import build_review_prompt

    artifact = _artifact(contexts[3], "yield_estimate")
    prompt = build_review_prompt("yield_estimate", artifact)

    assert "estimated_yield" in prompt
    assert "objection" in prompt.lower()
    assert "invent" not in prompt.lower()
    assert "propose a fault" not in prompt.lower()


def test_the_check_question_asks_what_to_verify_rather_than_what_is_wrong(
    contexts: tuple[ContractContext, ...],
) -> None:
    """Round 1 asked what contradicts what, and a correct artifact answers that with silence."""

    from sciagentguard.discovery import CHECKS

    prompt = CHECKS.build(_artifact(contexts[3], "yield_estimate"))

    assert "estimated_yield" in prompt
    assert "verify" in prompt.lower()
    assert "believed to be correct" in prompt
    assert "not to find an error" in prompt


def test_each_question_recognises_its_own_way_of_saying_nothing() -> None:
    from sciagentguard.discovery import CHECKS, OBJECTIONS

    assert CHECKS.parse("NO CHECKS") == ()
    assert OBJECTIONS.parse("NO OBJECTIONS") == ()
    # A reviewer answering the other question's sentinel has still said something, and a bullet
    # list must not be discarded because it mentions a marker in passing.
    assert CHECKS.parse("- verify that the weights sum to the declared total") == (
        "verify that the weights sum to the declared total",
    )


def test_the_question_asked_decides_which_prompt_the_reviewer_sees(
    contexts: tuple[ContractContext, ...],
) -> None:
    from sciagentguard.discovery import CHECKS

    seen: list[str] = []

    def reviewer(prompt: str) -> str:
        seen.append(prompt)
        return "- verify the closure of bin_weight_sums against selected_weight_sum"

    targets = [ReviewTarget("histogram", "post_histogram", _artifact(contexts[2], "histogram"))]
    candidates = collect_candidates(targets, reviewer, round_index=2, question=CHECKS)

    assert "not to find an error" in seen[0]
    assert candidates[0].objection.startswith("verify the closure")
    assert candidates[0].round_index == 2


# --- gate 2, the only gate a machine may decide ---------------------------------------


def test_a_violation_the_contracts_already_catch_is_not_novel(
    adapter: AtlasGamGamOpenDataAdapter, contexts: tuple[ContractContext, ...]
) -> None:
    """A well-worded objection about something already covered must not advance."""

    from dataclasses import replace

    artifact = copy.deepcopy(_artifact(contexts[3], "yield_estimate"))
    artifact["estimated_yield"] = float(artifact["estimated_yield"]) * 5.0
    violating = replace(contexts[3], artifacts={"yield_estimate": artifact})

    evidence = prove_novelty(violating, adapter.stage_contracts()[3], "yield scaled by five")

    assert not evidence.is_novel
    assert "hep.atlas_open_data.yield_closure" in evidence.contracts_that_fired


def test_a_violation_nothing_catches_is_novel(
    adapter: AtlasGamGamOpenDataAdapter, contexts: tuple[ContractContext, ...]
) -> None:
    """`method` is a declared field that no contract reads. Changing it is a genuine gap."""

    from dataclasses import replace

    artifact = copy.deepcopy(_artifact(contexts[3], "yield_estimate"))
    artifact["method"] = "a method nobody verifies"
    violating = replace(contexts[3], artifacts={"yield_estimate": artifact})

    evidence = prove_novelty(violating, adapter.stage_contracts()[3], "method string replaced")

    assert evidence.is_novel
    assert evidence.contracts_that_fired == ()
    assert evidence.contracts_evaluated


def test_novelty_requires_a_contract_that_guards_the_stage(
    adapter: AtlasGamGamOpenDataAdapter, contexts: tuple[ContractContext, ...]
) -> None:
    with pytest.raises(ValueError, match="no contract guards stage"):
        prove_novelty(contexts[3], adapter.stage_contracts()[0], "wrong stage")


# --- a whole round --------------------------------------------------------------------


def test_a_round_records_every_objection_and_stays_pending_until_judged(
    contexts: tuple[ContractContext, ...],
) -> None:
    """Nothing is confirmed by collection alone: gate 3 needs a person, and this is not one."""

    targets = [
        ReviewTarget("yield_estimate", "post_yield", _artifact(contexts[3], "yield_estimate")),
        ReviewTarget("histogram", "post_histogram", _artifact(contexts[2], "histogram")),
    ]
    replies = iter(
        [
            "- estimated_yield is not implied by the values beside it\n- the window looks narrow",
            "NO OBJECTIONS",
        ]
    )

    candidates = collect_candidates(targets, lambda _: next(replies), round_index=1)
    record = open_round(1, {"provider": "stub", "model_id": "stub-1"}, targets, candidates)

    assert len(candidates) == 2
    assert all(candidate.outcome.value == "pending" for candidate in candidates)
    assert record.confirmed_blind_spots == 0
    assert record.rejections_by_gate["pending"] == 2
    assert record.artifacts_reviewed == ("yield_estimate", "histogram")
