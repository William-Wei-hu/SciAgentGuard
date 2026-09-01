"""Decide whether a model's analysis actually agrees with the trusted pipeline.

Milestone 6A knew which scripts were faulty because it wrote them. A real model carries no such
label, so ground truth has to come from somewhere: here, from running the same fully specified task
through `atlas_analysis.py` and comparing the answers.

This oracle is this repository's own implementation. An approach that is different but equally
correct -- another bin-edge convention, another tie-break among equal-momentum photons -- would be
scored as disagreement. Any disagreement must therefore be read before it is called a model error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

from pydantic import BaseModel, ConfigDict, JsonValue

from sciagentguard.core import ContractContext

# The same single-precision reasoning that sets the histogram closure tolerance: the source stores
# weights in float32, so an implementation that accumulates in float32 differs from one that
# promotes to float64 by roughly the float32 epsilon.
DEFAULT_RELATIVE_TOLERANCE = 1e-6


class ReferenceComparison(BaseModel):
    """Whether a model's artifacts match the trusted pipeline, and where they first diverge."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    agrees: bool
    disagreements: tuple[str, ...] = ()
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE


@dataclass(frozen=True, slots=True)
class _Reference:
    selection: Mapping[str, object]
    histogram: Mapping[str, object]
    estimate: Mapping[str, object]


def reference_from_contexts(contexts: Sequence[ContractContext]) -> _Reference:
    """Pull the three comparable artifacts out of the trusted pipeline's contexts."""

    found: dict[str, Mapping[str, object]] = {}
    for context in contexts:
        for name in ("selection", "histogram", "yield_estimate"):
            artifact = context.artifacts.get(name)
            if isinstance(artifact, Mapping):
                found[name] = artifact
    missing = [name for name in ("selection", "histogram", "yield_estimate") if name not in found]
    if missing:
        raise ValueError(f"the reference pipeline did not produce: {', '.join(missing)}")
    return _Reference(found["selection"], found["histogram"], found["yield_estimate"])


def compare_to_reference(
    contexts: Sequence[ContractContext],
    artifacts: Mapping[str, JsonValue],
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> ReferenceComparison:
    """Compare a model's artifacts against the trusted answer and name every disagreement."""

    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    reference = reference_from_contexts(contexts)
    disagreements: list[str] = []

    selection = artifacts.get("selection")
    histogram = artifacts.get("histogram")
    estimate = artifacts.get("yield_estimate")
    for name, candidate in (
        ("selection", selection),
        ("histogram", histogram),
        ("yield_estimate", estimate),
    ):
        if not isinstance(candidate, Mapping):
            disagreements.append(f"{name}: missing or not an object")

    if isinstance(selection, Mapping):
        _compare_ids(reference.selection, selection, "selected_event_ids", disagreements)
        _compare_regions(reference.selection, selection, disagreements)
        _compare_cutflow(reference.selection, selection, disagreements)
        _compare_number(
            reference.selection,
            selection,
            "selected_weight_sum",
            relative_tolerance,
            disagreements,
            "selection",
        )
    if isinstance(histogram, Mapping):
        _compare_series(
            reference.histogram,
            histogram,
            "bin_weight_sums",
            relative_tolerance,
            disagreements,
            edge_allowance=_edge_allowance(contexts),
        )
        _compare_number(
            reference.histogram,
            histogram,
            "normalization_factor",
            relative_tolerance,
            disagreements,
            "histogram",
        )
    if isinstance(estimate, Mapping):
        for field in ("estimated_yield", "peak_bin_center_gev"):
            _compare_number(
                reference.estimate,
                estimate,
                field,
                relative_tolerance,
                disagreements,
                "yield_estimate",
            )

    return ReferenceComparison(
        agrees=not disagreements,
        disagreements=tuple(disagreements),
        relative_tolerance=relative_tolerance,
    )


def _close(expected: float, actual: float, tolerance: float) -> bool:
    if not isfinite(expected) or not isfinite(actual):
        return False
    return abs(expected - actual) <= tolerance * max(abs(expected), 1.0)


def _numbers(source: Mapping[str, object], field: str) -> tuple[float, ...] | None:
    values = source.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return None
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numbers.append(float(value))
    return tuple(numbers)


def _number(source: Mapping[str, object], field: str) -> float | None:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _compare_number(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    field: str,
    tolerance: float,
    disagreements: list[str],
    prefix: str,
) -> None:
    expected = _number(reference, field)
    actual = _number(candidate, field)
    if expected is None:
        return
    if actual is None:
        disagreements.append(f"{prefix}.{field}: missing or not numeric")
        return
    if not _close(expected, actual, tolerance):
        disagreements.append(f"{prefix}.{field}: expected {expected!r}, got {actual!r}")


def _edge_allowance(contexts: Sequence[ContractContext]) -> float:
    """How much per-bin disagreement a single boundary event can legitimately explain.

    An event whose mass sits on a bin edge may land either side of it depending on how the edge
    was computed, and two correct implementations can disagree by one event in each of the two
    adjacent bins. That is a floating-point convention, not a scientific error, so the comparison
    allows a couple of event weights per bin. A real binning mistake moves far more than that.
    """

    for context in contexts:
        for declaration in context.provenance.values():
            if isinstance(declaration, Mapping):
                facts = declaration.get("source_facts")
                if isinstance(facts, Mapping):
                    largest = facts.get("weight_abs_max")
                    if isinstance(largest, (int, float)) and not isinstance(largest, bool):
                        return 2.0 * abs(float(largest))
    return 0.0


def _compare_series(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    field: str,
    tolerance: float,
    disagreements: list[str],
    edge_allowance: float = 0.0,
) -> None:
    expected = _numbers(reference, field)
    actual = _numbers(candidate, field)
    if expected is None:
        return
    if actual is None:
        disagreements.append(f"histogram.{field}: missing or not a numeric sequence")
        return
    if len(expected) != len(actual):
        disagreements.append(
            f"histogram.{field}: expected {len(expected)} entries, got {len(actual)}"
        )
        return
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if abs(left - right) > tolerance * max(abs(left), 1.0) + edge_allowance:
            disagreements.append(f"histogram.{field}[{index}]: expected {left!r}, got {right!r}")
            return


def _identifier_set(source: Mapping[str, object], field: str) -> frozenset[int] | None:
    values = source.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return None
    identifiers: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        identifiers.add(value)
    return frozenset(identifiers)


def _compare_ids(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    field: str,
    disagreements: list[str],
) -> None:
    expected = _identifier_set(reference, field)
    actual = _identifier_set(candidate, field)
    if expected is None:
        return
    if actual is None:
        disagreements.append(f"selection.{field}: missing or not a list of integers")
        return
    if expected != actual:
        disagreements.append(
            f"selection.{field}: {len(expected - actual)} missing, {len(actual - expected)} extra"
        )


def _compare_regions(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    disagreements: list[str],
) -> None:
    expected = reference.get("regions")
    actual = candidate.get("regions")
    if not isinstance(expected, Mapping):
        return
    if not isinstance(actual, Mapping):
        disagreements.append("selection.regions: missing or not an object")
        return
    for name in expected:
        if not isinstance(name, str):
            continue
        left = _identifier_set(expected, name)
        right = _identifier_set(actual, name)
        if left is None:
            continue
        if right is None:
            disagreements.append(f"selection.regions.{name}: missing or not a list of integers")
        elif left != right:
            disagreements.append(
                f"selection.regions.{name}: {len(left - right)} missing, {len(right - left)} extra"
            )


def _compare_cutflow(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    disagreements: list[str],
) -> None:
    """Compare what a cutflow means, not how its stages were named or grouped.

    An author may call the first stage `total_events` instead of `all_events`, or apply both
    momentum thresholds in one step instead of two. Neither changes the analysis. Only the
    endpoints carry meaning: the cutflow must start at every input event and end at the number of
    events actually selected, and it must never grow along the way.
    """

    expected = _cutflow(reference)
    actual = _cutflow(candidate)
    if expected is None:
        return
    if actual is None:
        disagreements.append("selection.cutflow: missing or malformed")
        return

    expected_counts = [count for _, count in expected]
    actual_counts = [count for _, count in actual]
    if actual_counts[0] != expected_counts[0]:
        disagreements.append(
            f"selection.cutflow: starts at {actual_counts[0]}, expected {expected_counts[0]}"
        )
    if actual_counts[-1] != expected_counts[-1]:
        disagreements.append(
            f"selection.cutflow: ends at {actual_counts[-1]}, expected {expected_counts[-1]}"
        )
    if any(later > earlier for earlier, later in pairwise(actual_counts)):
        disagreements.append(f"selection.cutflow: not monotonic: {actual_counts}")


def _cutflow(source: Mapping[str, object]) -> tuple[tuple[str, int], ...] | None:
    raw = source.get("cutflow")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    stages: list[tuple[str, int]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            return None
        cut_id = entry.get("cut_id")
        surviving = entry.get("surviving")
        if not isinstance(cut_id, str) or isinstance(surviving, bool):
            return None
        if not isinstance(surviving, int):
            return None
        stages.append((cut_id, surviving))
    return tuple(stages)


__all__ = [
    "DEFAULT_RELATIVE_TOLERANCE",
    "ReferenceComparison",
    "compare_to_reference",
    "reference_from_contexts",
]
