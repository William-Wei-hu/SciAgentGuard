"""Internal accessors for the synthetic column-oriented event artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TypeAlias

from sciagentguard.core import ContractContext

EventColumn: TypeAlias = Sequence[object]
EventColumns: TypeAlias = Mapping[str, EventColumn]


def require_event_columns(context: ContractContext) -> EventColumns:
    artifact = context.artifacts.get("events")
    if not isinstance(artifact, Mapping):
        raise ValueError("context artifact 'events' must be a mapping of branch names to columns")

    columns: dict[str, EventColumn] = {}
    lengths: set[int] = set()
    for branch, values in artifact.items():
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("event branch names must be non-empty strings")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"event branch {branch!r} must contain a sequence of values")
        columns[branch] = values
        lengths.add(len(values))

    if len(lengths) > 1:
        raise ValueError("event branches must contain the same number of rows")
    return columns


def copy_event_columns(context: ContractContext) -> dict[str, tuple[object, ...]]:
    return {name: tuple(values) for name, values in require_event_columns(context).items()}


def numeric_event_column(context: ContractContext, branch: str) -> tuple[float, ...] | None:
    values = require_event_columns(context).get(branch)
    if values is None:
        return None

    numeric_values: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"event branch {branch!r} contains a non-numeric value at index {index}"
            )
        numeric_values.append(float(value))
    return tuple(numeric_values)


def replace_event_columns(
    context: ContractContext, columns: Mapping[str, Sequence[object]]
) -> ContractContext:
    artifacts = dict(context.artifacts)
    artifacts["events"] = {name: tuple(values) for name, values in columns.items()}
    return replace(context, artifacts=artifacts)
