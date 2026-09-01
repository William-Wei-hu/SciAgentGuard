"""The two-condition discovery round must run offline, and must keep what it has already paid for.

Every reviewer here is a stub. A test that called a provider would cost minutes, would cost money,
and would return something different on the next run.
"""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter
from sciagentguard.adapters.agent.deepseek import DeepSeekError
from sciagentguard.adapters.agent.verdict_cache import CachedCompletion
from sciagentguard.core import ContractContext
from tests.integration._atlas_root import synthetic_source, write_root_file

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks" / "discovery_round.py"
NAMESPACE = runpy.run_path(str(BENCHMARK))

BUILD_TARGETS = cast(Callable[..., Any], NAMESPACE["build_targets"])
RUN_CONDITION = cast(Callable[..., Any], NAMESPACE["run_condition"])
BUILD_REPORT = cast(Callable[..., Any], NAMESPACE["build_report"])
RENDER_MARKDOWN = cast(Callable[[Any], str], NAMESPACE["render_markdown"])
LOAD_CONDITIONS = cast(Callable[[Path], Any], NAMESPACE["load_conditions"])
CONDITIONS = cast(Mapping[str, tuple[str, ...]], NAMESPACE["CONDITIONS"])
PREDICTION = cast(str, NAMESPACE["PREDICTION"])
PREDICTIONS = cast(Mapping[str, str], NAMESPACE["PREDICTIONS"])
MAIN = cast(Callable[[list[str]], int], NAMESPACE["main"])

IDENTITY = {"provider": "stub", "model_id": "stub-reviewer"}
PARAMETERS: dict[str, str | int | float] = {
    "provider": "stub",
    "model_id": "stub-reviewer",
    "temperature": 0.0,
    "sequence_summary_threshold": 32,
    "sequence_sample_size": 5,
}


@pytest.fixture(scope="module")
def contexts(tmp_path_factory: pytest.TempPathFactory) -> tuple[ContractContext, ...]:
    path = tmp_path_factory.mktemp("discovery-round") / "test.root"
    write_root_file(path)
    adapter = AtlasGamGamOpenDataAdapter(synthetic_source(path))
    return adapter.contexts(workflow_id="discovery", run_id="r", attempt_id="a")


def _completion(replies: Sequence[str | BaseException]) -> Callable[[str], CachedCompletion]:
    stream = iter(replies)

    def completion(prompt: str) -> CachedCompletion:
        reply = next(stream)
        if isinstance(reply, BaseException):
            raise reply
        return CachedCompletion(
            text=reply,
            provider="stub",
            model_id="stub-reviewer",
            obtained_at="2026-01-01T00:00:00+00:00",
            latency_ms=1234.0,
        )

    return completion


# --- the two conditions ----------------------------------------------------------------


def test_the_two_conditions_differ_only_in_what_the_reviewer_is_shown() -> None:
    assert CONDITIONS["A"] == ("yield_estimate",)
    assert CONDITIONS["B"] == ("selection", "histogram", "yield_estimate")
    assert CONDITIONS["A"][-1] == CONDITIONS["B"][-1]


def test_condition_b_asks_once_per_stage(contexts: tuple[ContractContext, ...]) -> None:
    targets = BUILD_TARGETS(contexts, CONDITIONS["B"])
    record = RUN_CONDITION(
        "B",
        targets,
        _completion(
            [
                "- the cutflow drops events the regions still count",
                "- bin_weight_sums does not sum to selected_weight_sum",
                "- estimated_yield does not follow from the values beside it",
            ]
        ),
        reviewer_identity=IDENTITY,
    )

    assert [target.artifact_name for target in record.targets] == [
        "selection",
        "histogram",
        "yield_estimate",
    ]
    assert [target.stage for target in record.targets] == [
        "post_selection",
        "post_histogram",
        "post_yield",
    ]
    assert record.objections == 3


def test_collection_confirms_nothing(contexts: tuple[ContractContext, ...]) -> None:
    """Gates 1 and 3 need a person and gate 2 needs a violating run. None of that happens here."""

    record = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion(["- estimated_yield is not implied by its own inputs\n- the window is narrow"]),
        reviewer_identity=IDENTITY,
    )

    assert record.round.confirmed_blind_spots == 0
    assert record.round.rejections_by_gate["pending"] == 2
    assert all(candidate.outcome.value == "pending" for candidate in record.round.candidates)


def test_a_reviewer_that_objects_to_nothing_yields_no_candidates(
    contexts: tuple[ContractContext, ...],
) -> None:
    record = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion(["NO OBJECTIONS"]),
        reviewer_identity=IDENTITY,
    )

    assert record.objections == 0
    assert record.failed_targets == ()


# --- failures are recorded, not silently counted as silence -----------------------------


def test_a_target_that_never_answers_is_recorded_as_an_error(
    contexts: tuple[ContractContext, ...],
) -> None:
    """A condition that could not be asked is not a condition that found nothing."""

    record = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion([DeepSeekError("HTTP 599"), DeepSeekError("HTTP 599")]),
        reviewer_identity=IDENTITY,
        max_attempts=2,
    )

    assert record.failed_targets == ("yield_estimate",)
    assert record.targets[0].calls == 2
    assert record.targets[0].latency_ms is None
    assert "HTTP 599" in str(record.targets[0].error)
    assert record.objections == 0


def test_a_transport_failure_is_retried(contexts: tuple[ContractContext, ...]) -> None:
    record = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion([DeepSeekError("HTTP 599"), "- estimated_yield does not follow"]),
        reviewer_identity=IDENTITY,
        max_attempts=2,
    )

    assert record.failed_targets == ()
    assert record.targets[0].calls == 2
    assert record.objections == 1


def test_the_latency_reported_is_the_one_the_call_measured(
    contexts: tuple[ContractContext, ...],
) -> None:
    """A replayed answer must report what the original call cost, not the microseconds of a hit."""

    record = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion(["- something"]),
        reviewer_identity=IDENTITY,
    )

    assert record.targets[0].latency_ms == 1234.0


# --- a crash must not discard the answers already bought --------------------------------


def test_a_crash_mid_round_keeps_the_targets_already_reviewed(
    contexts: tuple[ContractContext, ...], tmp_path: Path
) -> None:
    saved = tmp_path / "discovery_round.json"
    conditions: dict[str, Any] = {}

    def checkpoint(partial: Any) -> None:
        conditions["B"] = partial
        report = BUILD_REPORT(conditions, PARAMETERS)
        saved.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="the process died"):
        RUN_CONDITION(
            "B",
            BUILD_TARGETS(contexts, CONDITIONS["B"]),
            _completion(["- the cutflow drops events", RuntimeError("the process died")]),
            reviewer_identity=IDENTITY,
            checkpoint=checkpoint,
        )

    recovered = LOAD_CONDITIONS(saved)
    assert recovered["B"].artifacts_shown == ("selection",)
    assert recovered["B"].objections == 1
    assert json.loads(saved.read_text("utf-8"))["prediction"] == PREDICTION


def test_conditions_run_on_separate_days_accumulate_in_one_report(
    contexts: tuple[ContractContext, ...], tmp_path: Path
) -> None:
    saved = tmp_path / "discovery_round.json"
    first = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion(["- estimated_yield does not follow"]),
        reviewer_identity=IDENTITY,
    )
    saved.write_text(
        f"{BUILD_REPORT({'A': first}, PARAMETERS).model_dump_json(indent=2)}\n", encoding="utf-8"
    )

    conditions = LOAD_CONDITIONS(saved)
    conditions["B"] = RUN_CONDITION(
        "B",
        BUILD_TARGETS(contexts, CONDITIONS["B"]),
        _completion(["- one", "- two", "- three"]),
        reviewer_identity=IDENTITY,
    )

    assert sorted(conditions) == ["A", "B"]
    assert conditions["A"].objections == 1
    assert conditions["B"].objections == 3


def test_an_unreadable_report_is_not_mistaken_for_an_empty_one(tmp_path: Path) -> None:
    corrupt = tmp_path / "discovery_round.json"
    corrupt.write_text("{ not json", encoding="utf-8")

    assert LOAD_CONDITIONS(corrupt) == {}
    assert LOAD_CONDITIONS(tmp_path / "absent.json") == {}


# --- what the reviewer is actually shown ------------------------------------------------


def test_the_histogram_and_the_yield_reach_the_reviewer_untouched(
    contexts: tuple[ContractContext, ...],
) -> None:
    """Only the per-event sequences are summarised; the small artifacts are shown as they are."""

    targets = {target.artifact_name: target for target in BUILD_TARGETS(contexts, CONDITIONS["B"])}

    for name, index in (("histogram", 2), ("yield_estimate", 3)):
        original = cast(Mapping[str, Any], contexts[index].artifacts[name])
        shown = targets[name].artifact
        assert dict(shown) == {key: _plain(value) for key, value in original.items()}


def test_a_long_sequence_is_shown_as_a_labelled_summary_of_itself(
    contexts: tuple[ContractContext, ...],
) -> None:
    """The selection artifact serialises to megabytes, so it cannot be shown whole."""

    from sciagentguard.discovery import summarise_artifact

    original = cast(Mapping[str, Any], contexts[1].artifacts["selection"])
    long_key = "input_event_ids"
    values = list(cast(Sequence[float], original[long_key]))
    summarised = summarise_artifact(original, threshold=4, sample=2)
    shown = cast(Mapping[str, Any], summarised[long_key])

    assert shown["count"] == len(values)
    assert shown["first_2"] == [_plain(value) for value in values[:2]]
    assert shown["last_2"] == [_plain(value) for value in values[-2:]]
    assert shown["minimum"] == min(values)
    assert shown["maximum"] == max(values)
    assert "not the sequence itself" in str(shown["note"])
    # Nothing beyond what the replaced values already say.
    assert set(shown) == {"note", "count", "first_2", "last_2", "minimum", "maximum"}


def test_the_summary_says_it_is_one_inside_the_json(
    contexts: tuple[ContractContext, ...],
) -> None:
    """Otherwise the reviewer objects that an array is shorter than the count beside it.

    This fixture holds thirty events, so its sequences fall under the default threshold and are
    shown whole; the real sample holds 113,765 and does not. The threshold is lowered here so the
    behaviour that matters on the real sample is exercised on a file a test can build.
    """

    from sciagentguard.discovery import summarise_artifact

    original = cast(Mapping[str, Any], contexts[1].artifacts["selection"])
    blob = json.dumps(summarise_artifact(original, threshold=4, sample=2), default=str)

    assert "not the sequence itself" in blob
    # The whole point of summarising: the reviewer's prompt has to fit.
    assert len(blob) < 4_000


def test_the_scalars_of_the_selection_artifact_survive_summarising(
    contexts: tuple[ContractContext, ...],
) -> None:
    """The upstream-only quantities are exactly what condition B exists to put in front of it."""

    targets = {target.artifact_name: target for target in BUILD_TARGETS(contexts, CONDITIONS["B"])}
    shown = targets["selection"].artifact
    original = cast(Mapping[str, Any], contexts[1].artifacts["selection"])

    for key in ("cutflow", "regions", "selected_weight_sum", "cross_section_pb"):
        assert dict(shown)[key] == _plain(original[key])


def test_an_artifact_no_checkpoint_produced_is_an_error(
    contexts: tuple[ContractContext, ...],
) -> None:
    with pytest.raises(KeyError, match="no checkpoint produced"):
        BUILD_TARGETS(contexts, ("a_stage_that_does_not_exist",))


# --- the report -------------------------------------------------------------------------


def test_the_report_states_the_prediction_and_that_nothing_is_confirmed(
    contexts: tuple[ContractContext, ...],
) -> None:
    conditions = {
        "A": RUN_CONDITION(
            "A",
            BUILD_TARGETS(contexts, CONDITIONS["A"]),
            _completion(["- estimated_yield does not follow from its own inputs"]),
            reviewer_identity=IDENTITY,
        )
    }
    markdown = RENDER_MARKDOWN(BUILD_REPORT(conditions, PARAMETERS))

    assert PREDICTION in markdown
    assert "pending" in markdown
    assert "estimated_yield does not follow from its own inputs" in markdown
    assert "recall" not in markdown.lower()
    assert "coverage" not in markdown.lower()


def test_the_report_says_a_failed_call_biases_against_the_stagewise_condition(
    contexts: tuple[ContractContext, ...],
) -> None:
    """The budget runs out on the richest artifacts, which are the ones condition B is about."""

    conditions = {
        "B": RUN_CONDITION(
            "B",
            BUILD_TARGETS(contexts, CONDITIONS["B"]),
            _completion(
                [
                    DeepSeekError("no visible content (finish_reason='length')"),
                    DeepSeekError("no visible content (finish_reason='length')"),
                    "- bin_weight_sums does not sum to selected_weight_sum",
                    "NO OBJECTIONS",
                ]
            ),
            reviewer_identity=IDENTITY,
            max_attempts=2,
        )
    }
    markdown = RENDER_MARKDOWN(BUILD_REPORT(conditions, PARAMETERS))

    assert "Calls that returned nothing" in markdown
    assert "finish_reason='length'" in markdown
    assert "against condition B" in markdown


def test_a_silent_round_is_not_reported_as_a_tested_hypothesis(
    contexts: tuple[ContractContext, ...],
) -> None:
    """Zero objections and zero blind spots are the same number and different findings."""

    conditions = {
        name: RUN_CONDITION(
            name,
            BUILD_TARGETS(contexts, CONDITIONS[name]),
            _completion(["NO OBJECTIONS"] * len(CONDITIONS[name])),
            reviewer_identity=IDENTITY,
        )
        for name in ("A", "B")
    }
    markdown = RENDER_MARKDOWN(BUILD_REPORT(conditions, PARAMETERS))

    assert "No condition raised anything" in markdown
    assert "has not tested it" in markdown
    assert "manufactures nothing" in markdown


def test_a_round_that_raised_objections_does_not_call_them_blind_spots(
    contexts: tuple[ContractContext, ...],
) -> None:
    conditions = {
        "A": RUN_CONDITION(
            "A",
            BUILD_TARGETS(contexts, CONDITIONS["A"]),
            _completion(["- estimated_yield does not follow from its own inputs"]),
            reviewer_identity=IDENTITY,
        )
    }
    markdown = RENDER_MARKDOWN(BUILD_REPORT(conditions, PARAMETERS))

    assert "not a count of blind spots" in markdown
    assert "No condition raised anything" not in markdown


def test_the_markdown_can_be_rebuilt_without_calling_a_provider(
    contexts: tuple[ContractContext, ...], tmp_path: Path
) -> None:
    """A replay through the cache would rewrite the call counts and erase the failed calls."""

    saved = tmp_path / "discovery_round.json"
    markdown = tmp_path / "discovery_round.md"
    record = RUN_CONDITION(
        "A",
        BUILD_TARGETS(contexts, CONDITIONS["A"]),
        _completion([DeepSeekError("finish_reason='length'"), "- one objection"]),
        reviewer_identity=IDENTITY,
        max_attempts=2,
    )
    saved.write_text(
        f"{BUILD_REPORT({'A': record}, PARAMETERS).model_dump_json(indent=2)}\n", encoding="utf-8"
    )

    assert (
        MAIN(
            [
                "--render-only",
                "--json-output",
                str(saved),
                "--markdown-output",
                str(markdown),
            ]
        )
        == 0
    )
    assert "one objection" in markdown.read_text("utf-8")
    # The call that returned nothing is still counted after the rebuild.
    assert LOAD_CONDITIONS(saved)["A"].targets[0].calls == 2


def test_a_silent_check_round_is_read_differently_from_a_silent_objection_round(
    contexts: tuple[ContractContext, ...],
) -> None:
    """Asking what to verify is a question a correct artifact can answer, so silence means more."""

    from sciagentguard.discovery import CHECKS

    conditions = {
        "B": RUN_CONDITION(
            "B",
            BUILD_TARGETS(contexts, CONDITIONS["B"]),
            _completion(["NO CHECKS"] * 3),
            reviewer_identity=IDENTITY,
            question=CHECKS,
        )
    }
    parameters = dict(PARAMETERS) | {"question": "checks"}
    markdown = RENDER_MARKDOWN(BUILD_REPORT(conditions, parameters, PREDICTIONS["checks"]))

    assert "No condition proposed anything" in markdown
    assert "could have failed its prediction" in markdown
    assert "has not tested it" not in markdown
    assert PREDICTIONS["checks"] in markdown


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return value
    return [_plain(item) for item in value]
