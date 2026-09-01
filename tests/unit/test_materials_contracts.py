import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from sciagentguard.adapters import DeePTBSi64Adapter, DeePTBSi64Source
from sciagentguard.core import ContractContext, ContractResult, ContractStatus
from sciagentguard.packs.materials import (
    DeePTBSourceIdentityContract,
    HamiltonianBlockHermiticityContract,
    OverlapGammaPositiveDefiniteContract,
)
from tests.integration._deeptb_hdf5 import write_test_source

BlockKey = tuple[int, int, int, int, int]
Matrix = tuple[tuple[float | complex, ...], ...]


def _context(tmp_path: Path) -> tuple[ContractContext, DeePTBSi64Source]:
    source = write_test_source(tmp_path)
    context = DeePTBSi64Adapter(source).load_context(
        workflow_id="deeptb-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )
    return context, source


def _identity_contract(source: DeePTBSi64Source) -> DeePTBSourceIdentityContract:
    return DeePTBSourceIdentityContract(
        expected_repository=source.repository,
        expected_commit=source.commit,
        expected_license=source.license_id,
        expected_sample_id=source.sample_id,
        expected_files={
            source.hamiltonian_file_name: (
                source.hamiltonian_size_bytes,
                source.hamiltonian_sha256,
            ),
            source.overlap_file_name: (
                source.overlap_size_bytes,
                source.overlap_sha256,
            ),
            source.atomic_numbers_file_name: (
                source.atomic_numbers_size_bytes,
                source.atomic_numbers_sha256,
            ),
        },
    )


def _blocks(context: ContractContext, artifact: str) -> dict[BlockKey, Matrix]:
    return dict(cast(Mapping[BlockKey, Matrix], context.artifacts[artifact]))


def test_valid_deeptb_context_passes_all_materials_contracts(tmp_path: Path) -> None:
    context, source = _context(tmp_path)
    results = (
        HamiltonianBlockHermiticityContract().evaluate(context),
        OverlapGammaPositiveDefiniteContract().evaluate(context),
        _identity_contract(source).evaluate(context),
    )

    assert all(result.status is ContractStatus.PASS for result in results)
    assert results[0].evidence["max_conjugate_residual"] == 0.0
    assert results[1].evidence["matrix_dimension"] == 4
    assert results[1].evidence["cholesky_valid"] is True


@pytest.mark.parametrize("fault", ["missing", "mismatch", "nonfinite"])
def test_hermiticity_contract_localizes_bounded_failures(tmp_path: Path, fault: str) -> None:
    context, _ = _context(tmp_path)
    blocks = _blocks(context, "hamiltonian_blocks")
    if fault == "missing":
        blocks.pop((1, 0, -1, 0, 0))
    elif fault == "mismatch":
        blocks[(1, 0, -1, 0, 0)] = ((9.0, 0.0), (0.0, 9.0))
    else:
        blocks[(0, 0, 0, 0, 0)] = ((float("nan"), 0.0), (0.0, 1.0))
    changed = replace(context, artifacts={**context.artifacts, "hamiltonian_blocks": blocks})

    result = HamiltonianBlockHermiticityContract().evaluate(changed)

    assert result.status is ContractStatus.FAIL
    assert result.violation is not None
    assert result.violation.stage == "post_hamiltonian_load"
    assert result.violation.suggested_actions
    assert "NaN" not in result.model_dump_json()
    assert ContractResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("fault", ["indefinite", "asymmetric", "nonfinite"])
def test_overlap_contract_rejects_invalid_gamma_matrices(tmp_path: Path, fault: str) -> None:
    context, _ = _context(tmp_path)
    blocks = _blocks(context, "overlap_blocks")
    if fault == "indefinite":
        blocks[(0, 0, 0, 0, 0)] = ((-1.0, 0.0), (0.0, 1.0))
    elif fault == "asymmetric":
        blocks.pop((1, 0, -1, 0, 0))
    else:
        blocks[(0, 0, 0, 0, 0)] = ((float("inf"), 0.0), (0.0, 1.0))
    changed = replace(context, artifacts={**context.artifacts, "overlap_blocks": blocks})

    result = OverlapGammaPositiveDefiniteContract().evaluate(changed)

    assert result.status is ContractStatus.FAIL
    assert result.evidence["cholesky_valid"] is False
    assert result.violation is not None
    assert result.violation.suggested_actions
    assert "Infinity" not in result.model_dump_json()
    assert ContractResult.model_validate_json(result.model_dump_json()) == result


def test_source_identity_reports_allowlisted_fields_without_leaking_values(
    tmp_path: Path,
) -> None:
    context, source = _context(tmp_path)
    secret = "PROVENANCE_SECRET_SENTINEL"
    declaration = dict(cast(Mapping[str, JsonValue], context.provenance["deeptb_sample"]))
    declaration["repository"] = secret
    declaration["credential"] = secret
    changed = replace(context, provenance={"deeptb_sample": declaration})

    result = _identity_contract(source).evaluate(changed)

    assert result.status is ContractStatus.FAIL
    assert result.evidence["mismatched_fields"] == ["repository"]
    assert secret not in result.model_dump_json()


def test_source_identity_fails_when_provenance_is_missing(tmp_path: Path) -> None:
    context, source = _context(tmp_path)

    result = _identity_contract(source).evaluate(replace(context, provenance={}))

    assert result.status is ContractStatus.FAIL
    assert result.evidence["mismatched_fields"] == ["deeptb_sample"]


@pytest.mark.parametrize(
    ("fault", "contract", "message"),
    [
        (
            "schema",
            HamiltonianBlockHermiticityContract(),
            "schema 'deeptb_blocks'",
        ),
        (
            "empty_hamiltonian",
            HamiltonianBlockHermiticityContract(),
            "must not be empty",
        ),
        (
            "bad_overlap_key",
            OverlapGammaPositiveDefiniteContract(),
            "invalid block key",
        ),
    ],
)
def test_materials_contracts_reject_malformed_contexts(
    tmp_path: Path,
    fault: str,
    contract: HamiltonianBlockHermiticityContract | OverlapGammaPositiveDefiniteContract,
    message: str,
) -> None:
    context, _ = _context(tmp_path)
    if fault == "schema":
        changed = replace(context, schema={})
    elif fault == "empty_hamiltonian":
        changed = replace(context, artifacts={"hamiltonian_blocks": {}})
    else:
        changed = replace(context, artifacts={"overlap_blocks": {"bad": ((1.0,),)}})

    with pytest.raises(ValueError, match=message):
        contract.evaluate(changed)


def test_contract_evidence_never_contains_matrix_values(tmp_path: Path) -> None:
    context, source = _context(tmp_path)
    sentinel = 12345.6789
    blocks = _blocks(context, "hamiltonian_blocks")
    sentinel_matrix = ((sentinel, 0.0), (0.0, sentinel))
    blocks[(0, 1, 1, 0, 0)] = sentinel_matrix
    blocks[(1, 0, -1, 0, 0)] = sentinel_matrix
    context = replace(context, artifacts={**context.artifacts, "hamiltonian_blocks": blocks})
    payload = json.dumps(
        [
            HamiltonianBlockHermiticityContract().evaluate(context).model_dump(mode="json"),
            OverlapGammaPositiveDefiniteContract().evaluate(context).model_dump(mode="json"),
            _identity_contract(source).evaluate(context).model_dump(mode="json"),
        ]
    )

    assert str(sentinel) not in payload
