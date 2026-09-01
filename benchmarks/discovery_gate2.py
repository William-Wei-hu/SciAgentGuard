"""Gate 2: run the current contracts against an artifact that violates each proposed check.

A candidate advances only if this script shows the contract set *accepting* a violation of it. That
is the one gate a machine may decide, and it is decided by running contracts rather than by asking
anybody -- the reviewer's confidence and mine are both worthless here.

The mapping from a reviewer's sentence to a violating artifact is hand-written and lives in
`PROPOSITIONS` below, because turning prose into a mutation is a judgement. The judgement is
recorded so it can be disputed; the outcome of running it is not a judgement at all.

Three verdicts are possible for a candidate:

  rejected_not_expressible  no deterministic check follows from the sentence, or the sentence is
                            about how the artifact was presented rather than about the analysis
  rejected_not_novel        a contract fires on the violation, so this is already covered
  pending                   nothing fires; the gap is real and a maintainer must now confirm it

A fourth outcome is recorded in the notes rather than as a rejection: a check that cannot be decided
at the stage where it was proposed, but can be decided from an upstream artifact. Those are not
failures of the candidate. They are the reviewer stating, unprompted, what an end-only view cannot
establish -- which is the question this round was built to ask.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.core import ContractContext
from sciagentguard.discovery import (
    Candidate,
    GateOutcome,
    HumanConfirmation,
    prove_novelty,
)

DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
DEFAULT_ROUND = Path("benchmarks/results/discovery_round2.json")
DEFAULT_OUTPUT = Path("benchmarks/results/discovery_round2_gates.json")
DEFAULT_MARKDOWN = Path(".cache/reports/discovery_round_2_review_sheet.md")

STAGE_INDEX = {"post_load": 0, "post_selection": 1, "post_histogram": 2, "post_yield": 3}
ARTIFACT_OF = {
    "post_selection": "selection",
    "post_histogram": "histogram",
    "post_yield": "yield_estimate",
}

Mutate = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Proposition:
    """One check, and one artifact that breaks it."""

    proposition_id: str
    stage: str
    summary: str
    mutate: Mutate


@dataclass(frozen=True)
class Rejection:
    """A candidate that never reaches gate 2, and why."""

    reason: str
    outcome: GateOutcome = GateOutcome.REJECTED_NOT_EXPRESSIBLE


@dataclass(frozen=True)
class Upstream:
    """A check the reviewer said these values alone cannot decide, naming what it would need."""

    needs: str


def _set(artifact: dict[str, Any], key: str, value: Any) -> None:
    artifact[key] = value


PROPOSITIONS: tuple[Proposition, ...] = (
    # --- post_yield ------------------------------------------------------------------
    Proposition(
        "background_from_sidebands",
        "post_yield",
        "background_estimate == sideband_weight_sum * signal_bin_count / sideband_bin_count",
        # The yield is recomputed from the broken background so that `yield_closure` still holds:
        # otherwise this would test the contract that already exists rather than the new relation.
        lambda a: (
            (
                _set(a, "background_estimate", a["background_estimate"] * 2.0),
                _set(
                    a,
                    "estimated_yield",
                    (a["signal_weight_sum"] - a["background_estimate"]) * a["normalization_factor"],
                ),
            )
            and None
        ),
    ),
    Proposition(
        "yield_closure",
        "post_yield",
        "estimated_yield == (signal_weight_sum - background_estimate) * normalization_factor",
        lambda a: _set(a, "estimated_yield", a["estimated_yield"] * 5.0),
    ),
    Proposition(
        "window_ordered",
        "post_yield",
        "signal_window_gev[0] < signal_window_gev[1]",
        lambda a: _set(a, "signal_window_gev", (130.0, 120.0)),
    ),
    Proposition(
        "peak_inside_window",
        "post_yield",
        "signal_window_gev[0] <= peak_bin_center_gev <= signal_window_gev[1]",
        lambda a: _set(a, "peak_bin_center_gev", 200.0),
    ),
    Proposition(
        "bin_counts_positive",
        "post_yield",
        "signal_bin_count > 0, sideband_bin_count > 0, peak_bin_index >= 0",
        lambda a: _set(a, "signal_bin_count", 0),
    ),
    Proposition(
        "normalization_finite_positive",
        "post_yield",
        "normalization_factor is finite and positive",
        lambda a: _set(a, "normalization_factor", -a["normalization_factor"]),
    ),
    Proposition(
        "normalization_from_source",
        "post_yield",
        "normalization_factor == cross_section_pb * luminosity / generated_weight_sum, with the "
        "cross-section and generated weight sum taken from the verified source rather than the "
        "artifact",
        # The yield is recomputed from the inflated factor so that `yield_closure` still holds.
        lambda a: (
            (
                _set(a, "normalization_factor", a["normalization_factor"] * 1.5),
                _set(
                    a,
                    "estimated_yield",
                    (a["signal_weight_sum"] - a["background_estimate"]) * a["normalization_factor"],
                ),
            )
            and None
        ),
    ),
    # --- post_selection --------------------------------------------------------------
    Proposition(
        "cutflow_monotonic",
        "post_selection",
        "cutflow surviving counts are non-increasing",
        lambda a: _set(
            a,
            "cutflow",
            tuple(
                dict(step) | ({"surviving": 999999} if index == 2 else {})
                for index, step in enumerate(a["cutflow"])
            ),
        ),
    ),
    Proposition(
        "cutflow_head_matches_input",
        "post_selection",
        "cutflow[0].surviving == len(input_event_ids)",
        # Raised rather than lowered: lowering the head also breaks monotonicity, and a violation
        # that breaks two relations at once cannot tell you which one a contract caught.
        lambda a: _set(
            a,
            "cutflow",
            tuple(
                dict(step) | ({"surviving": step["surviving"] + 7} if index == 0 else {})
                for index, step in enumerate(a["cutflow"])
            ),
        ),
    ),
    Proposition(
        "selected_ids_within_input_range",
        "post_selection",
        "every selected_event_id lies within the range of input_event_ids",
        # The added id is mirrored into a region and into the cutflow tail so that the region
        # partition and the tail count still hold: only the range relation is left broken.
        lambda a: (
            (
                _set(a, "selected_event_ids", (*tuple(a["selected_event_ids"]), 999999999)),
                _set(
                    a,
                    "regions",
                    dict(a["regions"]) | {"signal": (*tuple(a["regions"]["signal"]), 999999999)},
                ),
                _set(
                    a,
                    "cutflow",
                    tuple(
                        dict(step) | ({"surviving": step["surviving"] + 1} if index == 3 else {})
                        for index, step in enumerate(a["cutflow"])
                    ),
                ),
                _set(a, "selected_mass_gev", (*tuple(a["selected_mass_gev"]), 125.0)),
                _set(a, "selected_weight", (*tuple(a["selected_weight"]), 0.0)),
            )
            and None
        ),
    ),
    Proposition(
        "cutflow_tail_matches_selected",
        "post_selection",
        "cutflow[-1].surviving == len(selected_event_ids)",
        lambda a: _set(
            a,
            "cutflow",
            tuple(
                dict(step) | ({"surviving": step["surviving"] - 7} if index == 3 else {})
                for index, step in enumerate(a["cutflow"])
            ),
        ),
    ),
    Proposition(
        "selected_sequences_same_length",
        "post_selection",
        "len(selected_event_ids) == len(selected_mass_gev) == len(selected_weight)",
        lambda a: _set(a, "selected_weight", tuple(a["selected_weight"])[:-1]),
    ),
    Proposition(
        "region_counts_partition_selected",
        "post_selection",
        "len(regions.signal) + len(regions.control) == len(selected_event_ids)",
        lambda a: _set(
            a,
            "regions",
            dict(a["regions"]) | {"control": tuple(a["regions"]["control"])[:-9]},
        ),
    ),
    Proposition(
        "region_ids_are_selected_ids",
        "post_selection",
        "every region member appears in selected_event_ids",
        lambda a: _set(
            a,
            "regions",
            dict(a["regions"]) | {"control": (*tuple(a["regions"]["control"])[:-1], -424242)},
        ),
    ),
    Proposition(
        "selected_ids_unique",
        "post_selection",
        "selected_event_ids contains no duplicate",
        lambda a: _set(
            a,
            "selected_event_ids",
            (*tuple(a["selected_event_ids"])[:-1], next(iter(a["selected_event_ids"]))),
        ),
    ),
    Proposition(
        "weight_sum_within_reach",
        "post_selection",
        "min(selected_weight) * n <= selected_weight_sum <= max(selected_weight) * n",
        lambda a: _set(a, "selected_weight_sum", 1.0e9),
    ),
    Proposition(
        "masses_nonnegative",
        "post_selection",
        "every selected_mass_gev is nonnegative",
        lambda a: _set(a, "selected_mass_gev", (-5.0, *tuple(a["selected_mass_gev"])[1:])),
    ),
    Proposition(
        "cross_section_positive",
        "post_selection",
        "cross_section_pb > 0",
        lambda a: _set(a, "cross_section_pb", -a["cross_section_pb"]),
    ),
    # --- post_histogram --------------------------------------------------------------
    Proposition(
        "edge_and_bin_counts_agree",
        "post_histogram",
        "len(bin_edges) == len(bin_counts) + 1 == len(bin_weight_sums) + 1",
        lambda a: _set(a, "bin_edges", tuple(a["bin_edges"])[:-1]),
    ),
    Proposition(
        "edges_strictly_increasing",
        "post_histogram",
        "bin_edges[i] < bin_edges[i + 1] for every adjacent pair",
        lambda a: _set(
            a,
            "bin_edges",
            (*tuple(a["bin_edges"])[:5], tuple(a["bin_edges"])[4], *tuple(a["bin_edges"])[6:]),
        ),
    ),
    Proposition(
        "bin_counts_nonnegative",
        "post_histogram",
        "every bin_count is a nonnegative integer",
        lambda a: _set(a, "bin_counts", (-3, *tuple(a["bin_counts"])[1:])),
    ),
    Proposition(
        "histogram_closure",
        "post_histogram",
        "sum(bin_weight_sums) + underflow + overflow == selected_weight_sum",
        lambda a: _set(
            a,
            "bin_weight_sums",
            (next(iter(a["bin_weight_sums"])) + 500.0, *tuple(a["bin_weight_sums"])[1:]),
        ),
    ),
    Proposition(
        "normalization_matches_its_inputs",
        "post_histogram",
        "generated_weight_sum * normalization_factor == cross_section_pb * luminosity_pb_inverse",
        lambda a: _set(a, "normalization_factor", a["normalization_factor"] * 1.5),
    ),
)

PROPOSITION_BY_ID = {proposition.proposition_id: proposition for proposition in PROPOSITIONS}

# The reviewer's sentences, in the order it produced them, mapped to what they mean. Written by
# hand and kept next to the sentences it interprets so a reader can disagree with any single line.
MAPPING: dict[tuple[str, int], str | Rejection | Upstream] = {
    ("yield_estimate", 0): "background_from_sidebands",
    ("yield_estimate", 1): "yield_closure",
    ("yield_estimate", 2): "window_ordered",
    ("yield_estimate", 3): "peak_inside_window",
    ("yield_estimate", 4): "bin_counts_positive",
    ("yield_estimate", 5): "normalization_finite_positive",
    ("yield_estimate", 6): Upstream("the histogram's per-bin weight sums inside the window"),
    ("yield_estimate", 7): Upstream("the histogram's per-bin weight sums over the sideband bins"),
    # The reviewer was right that these *values* cannot establish the factor, and wrong that
    # nothing can: the verified loader's provenance carries the cross-section and the generated
    # weight sum past this stage, which is the whole point of carrying them. Reclassified from
    # "upstream only" to a check once that was noticed.
    ("yield_estimate", 8): "normalization_from_source",
    ("yield_estimate", 9): Upstream("the histogram's bin edges"),
    ("yield_estimate", 10): Upstream("the histogram's bin edges and window definition"),
    ("selection", 0): "cutflow_monotonic",
    ("selection", 1): "cutflow_head_matches_input",
    ("selection", 2): "cutflow_tail_matches_selected",
    ("selection", 3): "selected_sequences_same_length",
    ("selection", 4): "cutflow_head_matches_input",
    ("selection", 5): "region_counts_partition_selected",
    ("selection", 6): "selected_ids_within_input_range",
    ("selection", 7): "region_ids_are_selected_ids",
    ("selection", 8): Rejection(
        "about the sequence summaries this round put in front of the reviewer, not about the "
        "analysis. The summaries are built by taking min, max and slices of the values they "
        "replace, so the property holds by construction and cannot be violated by any run."
    ),
    ("selection", 9): "weight_sum_within_reach",
    ("selection", 10): "masses_nonnegative",
    ("selection", 11): "cross_section_positive",
    ("selection", 12): Upstream("generator metadata outside the workflow"),
    ("selection", 13): "selected_ids_unique",
    ("histogram", 0): "edge_and_bin_counts_agree",
    ("histogram", 1): "edges_strictly_increasing",
    ("histogram", 2): "bin_counts_nonnegative",
    ("histogram", 3): "histogram_closure",
    ("histogram", 4): "normalization_matches_its_inputs",
    ("histogram", 5): Rejection(
        "conditioned on nonnegative event weights, which this sample does not have: mcWeight takes "
        "both signs, so the check it proposes would fail on correct data. Its second clause, "
        "selected_weight_sum <= generated_weight_sum, is a separate claim that needs a physicist.",
        GateOutcome.REJECTED_NOT_SCIENTIFIC,
    ),
    ("histogram", 6): Upstream("the source file's cross-section and the run's luminosity"),
    ("histogram", 7): Upstream("the selection artifact and the analysis configuration"),
}


def violating_context(contexts: Sequence[ContractContext], proposition: Proposition) -> Any:
    context = contexts[STAGE_INDEX[proposition.stage]]
    name = ARTIFACT_OF[proposition.stage]
    artifact = copy.deepcopy(dict(context.artifacts[name]))  # type: ignore[call-overload]
    proposition.mutate(artifact)
    return replace(context, artifacts={name: artifact})


# Gate 3. A maintainer's signature, and the recomputation that justified it. Nothing reaches this
# table by passing gate 2: a candidate is here because a person read it and said yes.
CONFIRMATIONS: dict[str, HumanConfirmation] = {
    "background_from_sidebands": HumanConfirmation(
        confirmed=True,
        maintainer="Zihang Wei",
        confirmed_on=datetime(2026, 8, 31, tzinfo=timezone.utc),
        rationale=(
            "The background estimate is the one quantity in the final artifact that nothing "
            "derives. Adopted as a contract; the four structural checks raised in the same round "
            "were not, to keep the contract set to scientific semantics."
        ),
        independent_check=(
            "Recomputed on the real sample: 8068.270408093929 * 5 / 25 == 1613.6540816187858, "
            "the reported background exactly. Doubling the background to 3227.308163 and "
            "recomputing the yield from it leaves all sixteen contracts passing while the result "
            "moves from 7.472602 to 7.323310, a shift of 2.0 percent."
        ),
    ),
    "normalization_from_source": HumanConfirmation(
        confirmed=True,
        maintainer="Zihang Wei",
        confirmed_on=datetime(2026, 8, 31, tzinfo=timezone.utc),
        rationale=(
            "Approved as the cheap half of the next step: turn the reviewer's statements of what "
            "it could not verify into checks the framework can. This is the one of them that "
            "needed no new plumbing, because provenance already carries what it asked for."
        ),
        independent_check=(
            "Derived from provenance and config on the real sample: 0.0019654512871056795 * "
            "10064.0 / 213799.953125 == 9.251780210572279e-05, the declared factor exactly. "
            "Inflating the factor by half and recomputing the yield from it left yield_shape, "
            "yield_closure and background_estimate all passing."
        ),
    ),
}
CONTRACT_FOR = {
    "background_from_sidebands": "hep.atlas_open_data.background_estimate",
    "normalization_from_source": "hep.atlas_open_data.normalization_provenance",
}


def contracts_of(
    adapter: AtlasGamGamOpenDataAdapter, stage: str, without: Sequence[str] = ()
) -> tuple[Any, ...]:
    """The contracts guarding one stage, optionally minus some.

    Gate 2 measures a candidate *against a contract set*, so the same sentence is a gap before a
    contract is adopted and covered after. Excluding a contract by name makes the earlier
    measurement reproducible instead of a thing one has to remember.
    """

    excluded = set(without)
    return tuple(
        contract
        for contract in adapter.stage_contracts()[STAGE_INDEX[stage]]
        if contract.contract_id not in excluded
    )


def judge(
    candidate: Candidate,
    contexts: Sequence[ContractContext],
    adapter: AtlasGamGamOpenDataAdapter,
    without: Sequence[str] = (),
) -> Candidate:
    """Decide gate 2 for one candidate by running contracts, or record why it never got there."""

    position = int(candidate.candidate_id.rsplit("-", 1)[1])
    verdict = MAPPING.get((candidate.artifact_reviewed, position))

    if isinstance(verdict, Rejection):
        return candidate.with_outcome(verdict.outcome, notes={"reason": verdict.reason})

    if isinstance(verdict, Upstream):
        return candidate.with_outcome(
            GateOutcome.PENDING,
            notes={
                "decidable_at_this_stage": False,
                "needs": verdict.needs,
                "reason": (
                    "the reviewer named, unprompted, a quantity the final artifact does not carry. "
                    "Deciding this requires the upstream artifact or its provenance."
                ),
            },
        )

    if verdict is None:
        return candidate.with_outcome(
            GateOutcome.REJECTED_NOT_EXPRESSIBLE,
            notes={"reason": "no violating artifact was written for this sentence"},
        )

    proposition = PROPOSITION_BY_ID[verdict]
    try:
        evidence = prove_novelty(
            violating_context(contexts, proposition),
            contracts_of(adapter, proposition.stage, without),
            proposition.summary,
        )
    except ValueError as error:
        # The violating artifact could not even be built: an accessor refuses it before any
        # contract evaluates. The property holds, but it is enforced by the code that reads the
        # artifact rather than by a contract, so a report that called this "no contract fired"
        # would be true and misleading at once.
        return candidate.with_outcome(
            GateOutcome.REJECTED_NOT_NOVEL,
            notes={
                "proposition": proposition.proposition_id,
                "decidable_at_this_stage": True,
                "enforced_by": "accessor precondition, not a contract",
                "reason": str(error),
            },
        )
    notes: dict[str, Any] = {
        "proposition": proposition.proposition_id,
        "decidable_at_this_stage": True,
    }
    if not evidence.is_novel:
        return candidate.with_outcome(GateOutcome.REJECTED_NOT_NOVEL, novelty=evidence, notes=notes)

    confirmation = CONFIRMATIONS.get(proposition.proposition_id)
    if confirmation is None:
        # Gate 2 passing is not confirmation. Without a signature the candidate waits.
        return candidate.with_outcome(GateOutcome.PENDING, novelty=evidence, notes=notes)

    return candidate.with_outcome(
        GateOutcome.CONFIRMED,
        novelty=evidence,
        confirmation=confirmation,
        contract_id=CONTRACT_FOR.get(proposition.proposition_id),
        notes=notes,
    )


def render_markdown(
    judged: Mapping[str, tuple[Candidate, ...]], excluded: Sequence[str] = ()
) -> str:
    heading = "# Discovery round 2: review sheet"
    if excluded:
        heading = "# Discovery round 2: review sheet, as the round found it"
    lines = [
        heading,
        "",
        (
            "Every sentence the reviewer produced, what it was taken to mean, and what happened "
            "when the current contracts were run against an artifact violating it. **Nothing here "
            "is a contract yet.** Gate 3 is a maintainer's signature, and no signature has been "
            "given."
        ),
        "",
    ]
    if excluded:
        lines[3] = (
            "Every sentence the reviewer produced, judged against the contract set as it stood "
            "**when the round ran**, with "
            + ", ".join(f"`{name}`" for name in sorted(excluded))
            + " excluded. A gate-2 tally means nothing without the contract set that produced "
            "it: the same sentence is a gap before a contract is adopted and covered after."
        )
    for condition in sorted(judged):
        lines.extend([f"## Condition {condition}", ""])
        for candidate in judged[condition]:
            notes = candidate.notes
            if candidate.outcome is GateOutcome.CONFIRMED:
                signature = candidate.confirmation
                verdict = (
                    f"**confirmed** by {signature.maintainer if signature else 'a maintainer'} "
                    f"-> `{candidate.contract_id}`"
                )
            elif candidate.outcome is GateOutcome.PENDING and not notes.get(
                "decidable_at_this_stage", True
            ):
                verdict = f"**upstream only** -- needs {notes.get('needs')}"
            elif candidate.outcome is GateOutcome.PENDING:
                verdict = "**gap** -- no contract fired on a violating artifact"
            elif candidate.outcome is GateOutcome.REJECTED_NOT_NOVEL and notes.get("enforced_by"):
                verdict = (
                    f"already enforced, but by an {notes['enforced_by']}: {notes.get('reason')}"
                )
            elif candidate.outcome is GateOutcome.REJECTED_NOT_NOVEL:
                evidence = candidate.novelty
                fired = ", ".join(
                    f"`{name}`" for name in (evidence.contracts_that_fired if evidence else ())
                )
                verdict = f"already covered by {fired}"
            else:
                verdict = f"{candidate.outcome.value} -- {notes.get('reason')}"
            lines.append(f"- `{candidate.candidate_id}` {candidate.objection}")
            lines.append(f"  - {verdict}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--round", dest="round_path", type=Path, default=DEFAULT_ROUND)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--without",
        action="append",
        default=[],
        metavar="CONTRACT_ID",
        help=(
            "exclude a contract from the run, reproducing what gate 2 said before that contract "
            "was adopted. A tally is only meaningful next to the contract set that produced it."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    adapter = AtlasGamGamOpenDataAdapter(AtlasGamGamSource.official_wph125(args.input))
    contexts = adapter.contexts(workflow_id="gate2", run_id="r1", attempt_id="a1")

    stored = json.loads(args.round_path.read_text("utf-8"))
    judged: dict[str, tuple[Candidate, ...]] = {}
    for name, record in sorted(stored["conditions"].items()):
        judged[name] = tuple(
            judge(Candidate.model_validate(raw), contexts, adapter, without=args.without)
            for raw in record["round"]["candidates"]
        )

    tally = {
        name: {
            outcome.value: sum(1 for c in candidates if c.outcome is outcome)
            for outcome in GateOutcome
            if any(c.outcome is outcome for c in candidates)
        }
        for name, candidates in judged.items()
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            {
                "contract_set": {
                    "evaluated_against": sorted(
                        contract.contract_id
                        for stage in ARTIFACT_OF
                        for contract in contracts_of(adapter, stage, args.without)
                    ),
                    "excluded": sorted(args.without),
                },
                "tally": tally,
                "conditions": {
                    name: [json.loads(c.model_dump_json()) for c in candidates]
                    for name, candidates in judged.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(judged, args.without), encoding="utf-8")
    print(json.dumps(tally, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
