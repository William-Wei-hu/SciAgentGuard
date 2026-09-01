"""Four-stage deterministic HEP workflow used by the fixture evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from sciagentguard.core import ContractContext, ScientificContract, SemanticFaultInjector
from sciagentguard.packs.hep.contracts import (
    DeclaredEventProvenanceContract,
    DisjointEventSplitsContract,
    FiniteWeightsContract,
    JetPtRangeContract,
    NonemptySelectionContract,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
    YieldNormalizationContract,
)
from sciagentguard.packs.hep.fixtures import (
    HEP_STAGE,
    NORMALIZATION_STAGE,
    SELECTION_STAGE,
    SPLIT_STAGE,
    make_synthetic_hep_context,
    make_synthetic_normalization_context,
    make_synthetic_selection_context,
    make_synthetic_split_context,
)
from sciagentguard.runtime.repair import RepairAction
from sciagentguard.runtime.workflow import WorkflowCheckpoint

RELOAD_SYNTHETIC_SOURCE_ACTION = "hep.reload_declared_synthetic_source"
SYNTHETIC_EVENTS_GENERATOR = "sciagentguard.packs.hep.fixtures"

_FAULT_STAGES = {
    "missing_branch": HEP_STAGE,
    "zero_weights": HEP_STAGE,
    "nonfinite_weights": HEP_STAGE,
    "unit_scale_error": HEP_STAGE,
    "undeclared_synthetic_data": HEP_STAGE,
    "empty_selection": SELECTION_STAGE,
    "split_leakage": SPLIT_STAGE,
    "wrong_normalization": NORMALIZATION_STAGE,
}


def _load_contracts() -> tuple[ScientificContract, ...]:
    return (
        RequiredBranchesContract(),
        FiniteWeightsContract(),
        NonzeroWeightSupportContract(),
        JetPtRangeContract(),
        DeclaredEventProvenanceContract(),
    )


@dataclass(frozen=True, slots=True)
class SyntheticHEPWorkflow:
    """Build the four declared checkpoints of the synthetic evaluation workflow."""

    fault: SemanticFaultInjector | None = None
    workflow_id: str = "hep-synthetic-demo"
    run_id: str = "run-001"
    initial_attempt_id: str = "attempt-0"

    def __post_init__(self) -> None:
        if self.fault is not None and self.fault.fault_id not in _FAULT_STAGES:
            raise ValueError(f"unsupported synthetic HEP fault {self.fault.fault_id!r}")

    def _context(self, stage: str) -> ContractContext:
        factory = {
            HEP_STAGE: make_synthetic_hep_context,
            SELECTION_STAGE: make_synthetic_selection_context,
            SPLIT_STAGE: make_synthetic_split_context,
            NORMALIZATION_STAGE: make_synthetic_normalization_context,
        }[stage]
        context = factory(
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            attempt_id=self.initial_attempt_id,
        )
        if self.fault is None or _FAULT_STAGES[self.fault.fault_id] != stage:
            return context
        return self.fault.inject(context)

    def initial_step(self) -> ContractContext:
        """Return the post-load checkpoint used by the bounded repair demo."""

        return self._context(HEP_STAGE)

    def checkpoints(self) -> tuple[WorkflowCheckpoint, ...]:
        """Return fresh step callables and stable contracts for all four stages."""

        return (
            WorkflowCheckpoint(lambda: self._context(HEP_STAGE), _load_contracts()),
            WorkflowCheckpoint(
                lambda: self._context(SELECTION_STAGE),
                (NonemptySelectionContract(),),
            ),
            WorkflowCheckpoint(
                lambda: self._context(SPLIT_STAGE),
                (DisjointEventSplitsContract(),),
            ),
            WorkflowCheckpoint(
                lambda: self._context(NORMALIZATION_STAGE),
                (YieldNormalizationContract(),),
            ),
        )

    def repair_step(self, action: RepairAction, *, attempt_id: str) -> ContractContext:
        if action.action_type != RELOAD_SYNTHETIC_SOURCE_ACTION:
            raise ValueError(f"unsupported synthetic HEP repair action {action.action_type!r}")
        expected_parameters = {
            "artifact": "events",
            "source_type": "synthetic",
            "generator": SYNTHETIC_EVENTS_GENERATOR,
        }
        if action.parameters != expected_parameters:
            raise ValueError("synthetic source reload parameters do not match the declared fixture")

        return make_synthetic_hep_context(
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            attempt_id=attempt_id,
        )
