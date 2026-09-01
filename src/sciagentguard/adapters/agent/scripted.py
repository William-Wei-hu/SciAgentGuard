"""A deterministic agent that serves fixed analysis scripts.

This agent exists so the whole harness -- sandbox, arms, repair loop, evidence -- can be built and
tested offline, with no model provider and no API cost. Its behaviour is a declared fixture:

- the first attempt returns the script it was configured with;
- a later attempt returns the correct script only when the feedback it received names a contract
  identifier and a stage, and otherwise returns the same faulty script again.

That rule makes the repair ablation runnable, but it does **not** say anything about how a real
model responds to feedback. It was written to behave this way. Only Milestone 6B, with a real
provider, can produce evidence about model behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sciagentguard.adapters.agent._scripts import CORRECT_SCRIPT_ID, SCRIPTS
from sciagentguard.adapters.agent.models import AgentProposal, AgentTask

SCRIPTED_AGENT_SOURCE = "sciagentguard.adapters.agent.scripted"


@dataclass(frozen=True, slots=True)
class ScriptedAgent:
    """Serve one declared script, switching to the correct one on structured feedback."""

    script_id: str
    repairs_from_structured_feedback: bool = True
    source: str = field(init=False, default=SCRIPTED_AGENT_SOURCE)

    def __post_init__(self) -> None:
        if self.script_id not in SCRIPTS:
            known = ", ".join(sorted(SCRIPTS))
            raise ValueError(f"unknown script {self.script_id!r}; known scripts are {known}")

    def propose(
        self,
        task: AgentTask,
        *,
        attempt_id: str,
        feedback: str | None = None,
        seed: int | None = None,
    ) -> AgentProposal:
        script_id = self.script_id
        if feedback is not None and self.repairs_from_structured_feedback:
            if _names_a_contract_and_stage(feedback):
                script_id = CORRECT_SCRIPT_ID

        return AgentProposal(
            proposal_id=f"{task.task_id}:{attempt_id}:{script_id}",
            task_id=task.task_id,
            attempt_id=attempt_id,
            code=SCRIPTS[script_id],
            source=self.source,
            model_id=None,
            provider=None,
            seed=seed,
        )


def _names_a_contract_and_stage(feedback: str) -> bool:
    """Decide whether feedback is structured enough for the fixture rule to act on.

    Structured feedback quotes a contract identifier and the stage that produced it. A generic
    error string does neither.
    """

    return "contract_id" in feedback and "stage" in feedback


__all__ = ["SCRIPTED_AGENT_SOURCE", "ScriptedAgent"]
