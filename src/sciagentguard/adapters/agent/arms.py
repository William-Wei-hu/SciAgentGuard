"""Comparison arms for the agent experiment, and the feedback handed back after a rejection.

`generic_data_checks` and `llm_judge` exist to be beaten or not beaten. They are deliberately
weak in different ways: the first has no domain semantics at all, and the second sees only the
final artifact. Neither is a strawman of our own invention -- they stand for the two things a
practitioner actually reaches for, a dataframe validator and a model asked to look at the answer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from pydantic import JsonValue

from sciagentguard.core import ViolationReport

_REQUIRED_YIELD_FIELDS = (
    "observable",
    "signal_window_gev",
    "signal_weight_sum",
    "background_estimate",
    "estimated_yield",
    "peak_bin_center_gev",
)


def generic_data_checks(final_artifact: Mapping[str, JsonValue]) -> bool:
    """Accept the final artifact on structure alone: present, typed, non-null, finite.

    This is the ceiling of what a generic dataframe validator can assert without knowing any
    physics. It cannot express a closure relation, a region overlap, or a normalization identity.
    """

    for field in _REQUIRED_YIELD_FIELDS:
        if field not in final_artifact:
            return False
        value = final_artifact[field]
        if value is None:
            return False
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)) and not isfinite(float(value)):
            return False
    window = final_artifact.get("signal_window_gev")
    if not isinstance(window, list) or len(window) != 2:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ScriptedJudge:
    """A stand-in for a model asked whether the final artifact is scientifically valid.

    Its rule is declared, not learned: reject a non-finite or non-positive yield, a peak outside
    the histogram range, or a signal window that is not a proper interval; otherwise accept.

    In Milestone 6A this judge is a fixture. Its verdicts say nothing about how a real model would
    answer, and must never be reported as evidence about model behaviour. Milestone 6B replaces it
    with an actual provider.
    """

    range_low_gev: float = 100.0
    range_high_gev: float = 160.0
    source: str = "scripted-judge"

    def verdict(self, final_artifact: Mapping[str, JsonValue]) -> bool:
        """Return True when the judge considers the final artifact scientifically plausible."""

        estimated_yield = _number(final_artifact.get("estimated_yield"))
        peak = _number(final_artifact.get("peak_bin_center_gev"))
        signal = _number(final_artifact.get("signal_weight_sum"))
        background = _number(final_artifact.get("background_estimate"))
        if None in (estimated_yield, peak, signal, background):
            return False
        assert estimated_yield is not None and peak is not None
        assert signal is not None and background is not None

        if estimated_yield <= 0.0:
            return False
        if not self.range_low_gev <= peak <= self.range_high_gev:
            return False
        return signal > background


def _number(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def generic_feedback(violations: tuple[ViolationReport, ...]) -> str:
    """The feedback an ordinary workflow gives: something failed, without saying what."""

    del violations
    return "The analysis was rejected. The result is not valid. Try again."


def structured_feedback(violations: tuple[ViolationReport, ...]) -> str:
    """The feedback SciAgentGuard gives: which contract, at which stage, on what evidence."""

    payload = [
        {
            "contract_id": violation.contract_id,
            "stage": violation.stage,
            "severity": violation.severity.value,
            "message": violation.message,
            "evidence": violation.evidence,
            "likely_causes": list(violation.likely_causes),
            "suggested_actions": list(violation.suggested_actions),
        }
        for violation in violations
    ]
    return json.dumps({"violations": payload}, indent=2, sort_keys=True, default=str)


FEEDBACK_KINDS = ("generic_error", "structured_report")

__all__ = [
    "FEEDBACK_KINDS",
    "ScriptedJudge",
    "generic_data_checks",
    "generic_feedback",
    "structured_feedback",
]
