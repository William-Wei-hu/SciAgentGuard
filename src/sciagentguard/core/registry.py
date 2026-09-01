"""Explicit registries for contracts and evaluation-only fault injectors."""

from __future__ import annotations

from collections.abc import Iterator

from sciagentguard.core.protocols import ScientificContract, SemanticFaultInjector


def _validate_identifier(identifier: str, *, field_name: str) -> None:
    if not identifier or identifier != identifier.strip():
        raise ValueError(f"{field_name} must be non-empty and have no surrounding whitespace")


class ContractRegistry:
    """Insertion-ordered registry keyed by stable contract identifier."""

    def __init__(self) -> None:
        self._contracts: dict[str, ScientificContract] = {}

    def register(self, contract: ScientificContract) -> None:
        _validate_identifier(contract.contract_id, field_name="contract_id")
        if contract.contract_id in self._contracts:
            raise ValueError(f"contract_id already registered: {contract.contract_id}")
        self._contracts[contract.contract_id] = contract

    def get(self, contract_id: str) -> ScientificContract:
        return self._contracts[contract_id]

    def __contains__(self, contract_id: object) -> bool:
        return contract_id in self._contracts

    def __iter__(self) -> Iterator[ScientificContract]:
        return iter(self._contracts.values())

    def __len__(self) -> int:
        return len(self._contracts)


class FaultInjectorRegistry:
    """Insertion-ordered registry keyed by stable semantic-fault identifier."""

    def __init__(self) -> None:
        self._injectors: dict[str, SemanticFaultInjector] = {}

    def register(self, injector: SemanticFaultInjector) -> None:
        _validate_identifier(injector.fault_id, field_name="fault_id")
        if injector.fault_id in self._injectors:
            raise ValueError(f"fault_id already registered: {injector.fault_id}")
        self._injectors[injector.fault_id] = injector

    def get(self, fault_id: str) -> SemanticFaultInjector:
        return self._injectors[fault_id]

    def __contains__(self, fault_id: object) -> bool:
        return fault_id in self._injectors

    def __iter__(self) -> Iterator[SemanticFaultInjector]:
        return iter(self._injectors.values())

    def __len__(self) -> int:
        return len(self._injectors)
