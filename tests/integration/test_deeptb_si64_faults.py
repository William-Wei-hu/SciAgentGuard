from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from sciagentguard.adapters import DeePTBSi64Adapter
from sciagentguard.core import ContractContext, ContractStatus, SemanticFaultInjector
from sciagentguard.packs.materials import (
    DeePTBAtomicSpeciesDriftInjector,
    DeePTBHermitianContentDriftInjector,
    DeePTBIndefiniteOverlapInjector,
    DeePTBMissingHamiltonianInverseInjector,
    DeePTBSourceIdentityDriftInjector,
)
from tests.integration._deeptb_hdf5 import write_test_source

BlockKey = tuple[int, int, int, int, int]
Matrix = tuple[tuple[float | complex, ...], ...]


def _context(tmp_path: Path) -> ContractContext:
    source = write_test_source(tmp_path)
    return DeePTBSi64Adapter(source).load_context(
        workflow_id="deeptb-fault-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )


def _blocks(context: ContractContext, artifact_name: str) -> Mapping[BlockKey, Matrix]:
    return cast(Mapping[BlockKey, Matrix], context.artifacts[artifact_name])


def _results(context: ContractContext, tmp_path: Path) -> dict[str, ContractStatus]:
    source = write_test_source(tmp_path)
    contracts = (
        DeePTBSi64Adapter(source)
        .checkpoint(
            workflow_id=context.workflow_id,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
        )
        .contracts
    )
    return {contract.contract_id: contract.evaluate(context).status for contract in contracts}


def test_missing_inverse_removes_one_block_from_a_copied_mapping(tmp_path: Path) -> None:
    original = _context(tmp_path / "source")
    injected = DeePTBMissingHamiltonianInverseInjector().inject(original)

    assert (
        len(_blocks(injected, "hamiltonian_blocks"))
        == len(_blocks(original, "hamiltonian_blocks")) - 1
    )
    assert (1, 0, -1, 0, 0) in _blocks(original, "hamiltonian_blocks")
    assert (1, 0, -1, 0, 0) not in _blocks(injected, "hamiltonian_blocks")
    assert isinstance(injected.artifacts["hamiltonian_blocks"], MappingProxyType)


def test_indefinite_overlap_preserves_hermiticity_and_breaks_cholesky(
    tmp_path: Path,
) -> None:
    original = _context(tmp_path / "source")
    injected = DeePTBIndefiniteOverlapInjector().inject(original)
    original_onsite = _blocks(original, "overlap_blocks")[(0, 0, 0, 0, 0)]
    injected_onsite = _blocks(injected, "overlap_blocks")[(0, 0, 0, 0, 0)]

    assert original_onsite[0][0] == 1.0
    assert cast(float, injected_onsite[0][0]) < 0.0
    assert injected_onsite[0][1] == original_onsite[0][1]
    assert injected_onsite[1][0] == original_onsite[1][0]


def test_source_and_gap_probes_change_only_the_declared_artifact(tmp_path: Path) -> None:
    original = _context(tmp_path / "source")
    source_drift = DeePTBSourceIdentityDriftInjector().inject(original)
    species_drift = DeePTBAtomicSpeciesDriftInjector().inject(original)
    content_drift = DeePTBHermitianContentDriftInjector().inject(original)
    source_provenance = cast(Mapping[str, object], source_drift.provenance["deeptb_sample"])

    assert source_provenance["commit"] == "deeptb-commit-drift"
    assert original.artifacts["atomic_numbers"] == (14, 14)
    assert species_drift.artifacts["atomic_numbers"] == (6, 14)
    assert species_drift.provenance == original.provenance
    assert content_drift.provenance == original.provenance
    assert _blocks(content_drift, "hamiltonian_blocks")[(0, 1, 1, 0, 0)][0][0] == 0.7
    assert _blocks(content_drift, "hamiltonian_blocks")[(1, 0, -1, 0, 0)][0][0] == 0.7
    assert _blocks(original, "hamiltonian_blocks")[(0, 1, 1, 0, 0)][0][0] == 0.2


@pytest.mark.parametrize(
    "injector",
    [
        DeePTBAtomicSpeciesDriftInjector(),
        DeePTBHermitianContentDriftInjector(),
        DeePTBIndefiniteOverlapInjector(),
        DeePTBMissingHamiltonianInverseInjector(),
        DeePTBSourceIdentityDriftInjector(),
    ],
)
def test_deeptb_faults_are_deterministic(
    tmp_path: Path,
    injector: SemanticFaultInjector,
) -> None:
    original = _context(tmp_path)

    assert injector.inject(original, seed=1) == injector.inject(original, seed=999)


def test_deeptb_faults_declare_contract_boundaries() -> None:
    assert DeePTBMissingHamiltonianInverseInjector().expected_contract_ids == (
        "materials.hamiltonian.block_hermiticity",
    )
    assert DeePTBIndefiniteOverlapInjector().expected_contract_ids == (
        "materials.overlap.gamma_positive_definite",
    )
    assert DeePTBSourceIdentityDriftInjector().expected_contract_ids == (
        "materials.deeptb.source_identity",
    )
    assert DeePTBAtomicSpeciesDriftInjector().expected_contract_ids == ()
    assert DeePTBHermitianContentDriftInjector().expected_contract_ids == ()


@pytest.mark.parametrize(
    ("injector", "expected_failed"),
    [
        (
            DeePTBMissingHamiltonianInverseInjector(),
            {"materials.hamiltonian.block_hermiticity"},
        ),
        (
            DeePTBIndefiniteOverlapInjector(),
            {"materials.overlap.gamma_positive_definite"},
        ),
        (
            DeePTBSourceIdentityDriftInjector(),
            {"materials.deeptb.source_identity"},
        ),
        (DeePTBAtomicSpeciesDriftInjector(), set()),
        (DeePTBHermitianContentDriftInjector(), set()),
    ],
)
def test_faults_hit_only_the_declared_contracts(
    tmp_path: Path,
    injector: SemanticFaultInjector,
    expected_failed: set[str],
) -> None:
    context = _context(tmp_path / "context")
    statuses = _results(injector.inject(context), tmp_path / "contracts")

    assert {
        contract_id for contract_id, status in statuses.items() if status is ContractStatus.FAIL
    } == expected_failed


def test_deeptb_faults_reject_invalid_preconditions(tmp_path: Path) -> None:
    context = _context(tmp_path)
    wrong_stage = replace(context, stage="post_load")
    for injector in (
        DeePTBAtomicSpeciesDriftInjector(),
        DeePTBHermitianContentDriftInjector(),
        DeePTBIndefiniteOverlapInjector(),
        DeePTBMissingHamiltonianInverseInjector(),
        DeePTBSourceIdentityDriftInjector(),
    ):
        with pytest.raises(ValueError, match="requires stage"):
            injector.inject(wrong_stage)

    with pytest.raises(ValueError, match="requires DeePTB sample provenance"):
        DeePTBSourceIdentityDriftInjector().inject(replace(context, provenance={}))
    with pytest.raises(ValueError, match="requires silicon atomic number 14"):
        DeePTBAtomicSpeciesDriftInjector().inject(
            replace(context, artifacts={**context.artifacts, "atomic_numbers": (6, 14)})
        )
    with pytest.raises(ValueError, match="requires a non-self inverse block pair"):
        DeePTBMissingHamiltonianInverseInjector().inject(
            replace(
                context,
                artifacts={
                    **context.artifacts,
                    "hamiltonian_blocks": {(0, 0, 0, 0, 0): ((1.0, 0.0), (0.0, 1.0))},
                },
            )
        )
