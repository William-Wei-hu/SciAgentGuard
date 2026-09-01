"""Structural interface implemented by analysis agents."""

from __future__ import annotations

from typing import Protocol

from sciagentguard.adapters.agent.models import AgentProposal, AgentTask


class AnalysisAgent(Protocol):
    """Propose analysis code for a task, optionally revising it after feedback."""

    @property
    def source(self) -> str:
        """Stable identifier for the agent implementation."""
        ...

    def propose(
        self,
        task: AgentTask,
        *,
        attempt_id: str,
        feedback: str | None = None,
        seed: int | None = None,
    ) -> AgentProposal:
        """Return code for the task.

        ``feedback`` is the text handed back after a rejected attempt. It is the only channel
        through which a rejection reaches the agent, which is what makes the difference between a
        generic error string and a structured violation report measurable.
        """
        ...


__all__ = ["AnalysisAgent"]
