"""Typed data exchanged at scientific-contract boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Annotated, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_Value = TypeVar("_Value")


def _read_only_copy(values: Mapping[str, _Value]) -> Mapping[str, _Value]:
    return MappingProxyType(dict(values))


class ContractStatus(str, Enum):
    """Outcome of evaluating a scientific contract."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class ViolationSeverity(str, Enum):
    """Operational severity attached to a contract violation."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ViolationReport(BaseModel):
    """Portable evidence and bounded guidance for a failed contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    violation_id: NonEmptyString
    contract_id: NonEmptyString
    severity: ViolationSeverity
    stage: NonEmptyString
    message: NonEmptyString
    evidence: dict[str, JsonValue]
    likely_causes: tuple[NonEmptyString, ...] = ()
    suggested_actions: tuple[NonEmptyString, ...] = ()
    affected_artifacts: tuple[NonEmptyString, ...] = ()
    run_id: NonEmptyString
    attempt_id: NonEmptyString
    timestamp: datetime
    schema_version: NonEmptyString = "1.0"

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")
        return timestamp


class ContractResult(BaseModel):
    """Serializable result of one contract evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_id: NonEmptyString
    status: ContractStatus
    evidence: dict[str, JsonValue]
    duration_ms: float
    violation: ViolationReport | None = None

    @field_validator("duration_ms")
    @classmethod
    def require_nonnegative_duration(cls, duration_ms: float) -> float:
        if duration_ms < 0:
            raise ValueError("duration_ms must be nonnegative")
        return duration_ms

    @model_validator(mode="after")
    def require_consistent_violation(self) -> ContractResult:
        if self.status is ContractStatus.FAIL and self.violation is None:
            raise ValueError("a failed contract result must include a violation")
        if self.status is not ContractStatus.FAIL and self.violation is not None:
            raise ValueError("only a failed contract result may include a violation")
        if self.violation is not None and self.violation.contract_id != self.contract_id:
            raise ValueError("result and violation contract_id values must match")
        return self


@dataclass(frozen=True, slots=True)
class ContractContext:
    """Read-only checkpoint inputs supplied to a scientific contract.

    A context is an in-process carrier, not a trace record. Callers should log selected,
    explicitly safe fields rather than serializing the complete object.
    """

    workflow_id: str
    run_id: str
    attempt_id: str
    stage: str
    artifacts: Mapping[str, object]
    schema: Mapping[str, JsonValue] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, JsonValue] = field(default_factory=dict)
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("workflow_id", "run_id", "attempt_id", "stage"):
            value = getattr(self, name)
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value.strip())

        object.__setattr__(self, "artifacts", _read_only_copy(self.artifacts))
        object.__setattr__(self, "schema", _read_only_copy(self.schema))
        object.__setattr__(self, "units", _read_only_copy(self.units))
        object.__setattr__(self, "provenance", _read_only_copy(self.provenance))
        object.__setattr__(self, "config", _read_only_copy(self.config))
