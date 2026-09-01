"""Deterministic repair policy for the synthetic HEP workflow."""

from __future__ import annotations

from sciagentguard.packs.hep.contracts import NONZERO_WEIGHT_SUPPORT_CONTRACT_ID
from sciagentguard.packs.hep.workflow import (
    RELOAD_SYNTHETIC_SOURCE_ACTION,
    SYNTHETIC_EVENTS_GENERATOR,
    SyntheticHEPWorkflow,
)
from sciagentguard.runtime import RepairAction, RepairRequest


class SyntheticHEPRepairPolicy:
    """Reload the declared fixture when its event weights have no support."""

    def propose(self, request: RepairRequest) -> RepairAction | None:
        if len(request.violations) != 1:
            return None

        violation = request.violations[0]
        if violation.contract_id != NONZERO_WEIGHT_SUPPORT_CONTRACT_ID:
            return None

        return RepairAction(
            action_id=f"{request.run_id}:{request.attempt_id}:reload-synthetic-events",
            action_type=RELOAD_SYNTHETIC_SOURCE_ACTION,
            rationale=(
                "Reload the declared deterministic source instead of inferring replacement weights."
            ),
            target_violation_ids=(violation.violation_id,),
            parameters={
                "artifact": "events",
                "source_type": "synthetic",
                "generator": SYNTHETIC_EVENTS_GENERATOR,
            },
        )


__all__ = ["SyntheticHEPRepairPolicy", "SyntheticHEPWorkflow"]
