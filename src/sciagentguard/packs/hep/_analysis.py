"""Internal accessors for downstream synthetic-analysis artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import TypeAlias

from sciagentguard.core import ContractContext

EventId: TypeAlias = int | str


def _event_ids(value: object, label: str) -> tuple[EventId, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence of event identifiers")

    identifiers: list[EventId] = []
    seen: set[EventId] = set()
    for index, identifier in enumerate(value):
        if isinstance(identifier, bool) or not isinstance(identifier, (int, str)):
            raise ValueError(f"{label} contains an invalid event identifier at index {index}")
        if isinstance(identifier, str) and not identifier.strip():
            raise ValueError(f"{label} contains an empty event identifier at index {index}")
        if identifier in seen:
            raise ValueError(f"{label} must not contain duplicate event identifiers")
        seen.add(identifier)
        identifiers.append(identifier)
    return tuple(identifiers)


def require_selection(
    context: ContractContext,
) -> tuple[Mapping[str, object], tuple[EventId, ...], tuple[EventId, ...]]:
    artifact = context.artifacts.get("selection")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'selection' must be a mapping")

    input_ids = _event_ids(artifact.get("input_event_ids"), "selection input_event_ids")
    selected_ids = _event_ids(artifact.get("selected_event_ids"), "selection selected_event_ids")
    if not set(selected_ids).issubset(input_ids):
        raise ValueError("selected event identifiers must be a subset of input event identifiers")
    return artifact, input_ids, selected_ids


def replace_selected_event_ids(
    context: ContractContext, selected_ids: Sequence[EventId]
) -> ContractContext:
    selection, input_ids, _ = require_selection(context)
    updated_selection = dict(selection)
    updated_selection["input_event_ids"] = input_ids
    updated_selection["selected_event_ids"] = tuple(selected_ids)
    artifacts = dict(context.artifacts)
    artifacts["selection"] = updated_selection
    return replace(context, artifacts=artifacts)


def require_splits(context: ContractContext) -> Mapping[str, object]:
    artifact = context.artifacts.get("splits")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'splits' must be a mapping")
    return artifact


def split_event_ids(splits: Mapping[str, object], split_name: str) -> tuple[EventId, ...]:
    if split_name not in splits:
        raise ValueError(f"splits artifact does not contain {split_name!r}")
    return _event_ids(splits[split_name], f"split {split_name!r}")


def replace_split_event_ids(
    context: ContractContext, split_name: str, event_ids: Sequence[EventId]
) -> ContractContext:
    splits = require_splits(context)
    updated_splits = dict(splits)
    updated_splits[split_name] = tuple(event_ids)
    artifacts = dict(context.artifacts)
    artifacts["splits"] = updated_splits
    return replace(context, artifacts=artifacts)


def _finite_number(values: Mapping[str, object], field: str) -> float:
    value = values.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"normalization field {field!r} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"normalization field {field!r} must be finite")
    return number


def require_normalization(
    context: ContractContext,
) -> tuple[Mapping[str, object], float, float, float, float, float]:
    artifact = context.artifacts.get("normalization")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'normalization' must be a mapping")

    selected_weight_sum = _finite_number(artifact, "selected_weight_sum")
    generated_weight_sum = _finite_number(artifact, "generated_weight_sum")
    cross_section_pb = _finite_number(artifact, "cross_section_pb")
    luminosity_pb_inverse = _finite_number(artifact, "luminosity_pb_inverse")
    observed_yield = _finite_number(artifact, "observed_yield")
    if generated_weight_sum == 0.0:
        raise ValueError("normalization generated_weight_sum must be nonzero")
    if cross_section_pb < 0.0:
        raise ValueError("normalization cross_section_pb must be nonnegative")
    if luminosity_pb_inverse < 0.0:
        raise ValueError("normalization luminosity_pb_inverse must be nonnegative")
    return (
        artifact,
        selected_weight_sum,
        generated_weight_sum,
        cross_section_pb,
        luminosity_pb_inverse,
        observed_yield,
    )


def replace_observed_yield(context: ContractContext, observed_yield: float) -> ContractContext:
    normalization, *_ = require_normalization(context)
    updated_normalization = dict(normalization)
    updated_normalization["observed_yield"] = observed_yield
    artifacts = dict(context.artifacts)
    artifacts["normalization"] = updated_normalization
    return replace(context, artifacts=artifacts)
