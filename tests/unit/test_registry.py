from dataclasses import replace

import pytest

from sciagentguard.core import (
    ContractContext,
    ContractRegistry,
    ContractResult,
    ContractStatus,
    FaultInjectorRegistry,
)


class PassingContract:
    def __init__(self, contract_id: str) -> None:
        self.contract_id = contract_id
        self.description = "Always passes for registry tests."
        self.stage = "test"
        self.required_inputs: tuple[str, ...] = ()

    def evaluate(self, context: ContractContext) -> ContractResult:
        return ContractResult(
            contract_id=self.contract_id,
            status=ContractStatus.PASS,
            evidence={"run_id": context.run_id},
            duration_ms=0.0,
        )


class NoOpInjector:
    def __init__(self, fault_id: str) -> None:
        self.fault_id = fault_id
        self.taxonomy = "test"
        self.description = "Copies a context without changing scientific data."
        self.preconditions: tuple[str, ...] = ()
        self.mutation_description = "No mutation; registry test double only."
        self.expected_contract_ids: tuple[str, ...] = ()
        self.restoration_strategy = "Discard the returned context."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        return replace(context)


def test_contract_registry_preserves_registration_order() -> None:
    registry = ContractRegistry()
    first = PassingContract("first")
    second = PassingContract("second")

    registry.register(first)
    registry.register(second)

    assert len(registry) == 2
    assert "first" in registry
    assert registry.get("first") is first
    assert list(registry) == [first, second]


def test_contract_registry_rejects_duplicate_and_invalid_ids() -> None:
    registry = ContractRegistry()
    registry.register(PassingContract("unique"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(PassingContract("unique"))
    with pytest.raises(ValueError, match="surrounding whitespace"):
        registry.register(PassingContract(" invalid "))
    with pytest.raises(KeyError):
        registry.get("missing")


def test_fault_injector_registry_supports_lookup_and_iteration() -> None:
    registry = FaultInjectorRegistry()
    injector = NoOpInjector("no-op")

    registry.register(injector)

    assert len(registry) == 1
    assert "no-op" in registry
    assert registry.get("no-op") is injector
    assert list(registry) == [injector]


def test_fault_injector_registry_rejects_duplicate_and_missing_ids() -> None:
    registry = FaultInjectorRegistry()
    registry.register(NoOpInjector("duplicate"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(NoOpInjector("duplicate"))
    with pytest.raises(ValueError, match="surrounding whitespace"):
        registry.register(NoOpInjector(" invalid "))
    with pytest.raises(KeyError):
        registry.get("missing")
