"""Structural interfaces implemented by contracts and semantic fault injectors."""

from __future__ import annotations

from typing import Protocol

from sciagentguard.core.models import ContractContext, ContractResult


class ScientificContract(Protocol):
    """A deterministic, non-mutating scientific invariant check."""

    @property
    def contract_id(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def stage(self) -> str: ...

    @property
    def required_inputs(self) -> tuple[str, ...]: ...

    def evaluate(self, context: ContractContext) -> ContractResult:
        """Evaluate the invariant at the context checkpoint."""
        ...


class SemanticFaultInjector(Protocol):
    """A deterministic, opt-in mutation used only for evaluation."""

    @property
    def fault_id(self) -> str: ...

    @property
    def taxonomy(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def preconditions(self) -> tuple[str, ...]: ...

    @property
    def mutation_description(self) -> str: ...

    @property
    def expected_contract_ids(self) -> tuple[str, ...]: ...

    @property
    def restoration_strategy(self) -> str: ...

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        """Return a new context containing the declared fault."""
        ...
