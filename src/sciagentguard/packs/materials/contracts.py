"""Scientific contracts for the pinned DeePTB Hamiltonian boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any

from pydantic import JsonValue

from sciagentguard.core import (
    ContractContext,
    ContractResult,
    ContractStatus,
    ViolationReport,
    ViolationSeverity,
)
from sciagentguard.packs.materials._blocks import (
    BlockKey,
    format_key,
    inverse_key,
    require_block_schema,
    require_blocks,
)

MATERIALS_STAGE = "post_hamiltonian_load"
ABSOLUTE_TOLERANCE = 1e-6
HAMILTONIAN_HERMITICITY_CONTRACT_ID = "materials.hamiltonian.block_hermiticity"
OVERLAP_POSITIVE_DEFINITE_CONTRACT_ID = "materials.overlap.gamma_positive_definite"
DEEPTB_SOURCE_IDENTITY_CONTRACT_ID = "materials.deeptb.source_identity"


class HamiltonianBlockHermiticityContract:
    """Check conjugate block pairs in a real-space Hamiltonian."""

    contract_id = HAMILTONIAN_HERMITICITY_CONTRACT_ID
    description = "Every Hamiltonian block has a conjugate-transpose inverse block."
    stage = MATERIALS_STAGE
    required_inputs = ("hamiltonian_blocks", "schema")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        np = _numpy()
        atom_count, orbitals_per_atom = require_block_schema(context)
        blocks = require_blocks(
            context,
            "hamiltonian_blocks",
            atom_count=atom_count,
            orbitals_per_atom=orbitals_per_atom,
        )
        nonfinite_keys = [
            key for key, matrix in blocks.items() if not bool(np.isfinite(matrix).all())
        ]
        missing_keys: list[BlockKey] = []
        inconsistent_keys: list[BlockKey] = []
        max_residual = 0.0
        visited: set[BlockKey] = set()

        for key in sorted(blocks):
            if key in visited:
                continue
            inverse = inverse_key(key)
            visited.add(key)
            if inverse not in blocks:
                missing_keys.append(key)
                continue
            visited.add(inverse)
            matrix = np.asarray(blocks[key])
            inverse_matrix = np.asarray(blocks[inverse])
            if not bool(np.isfinite(matrix).all()) or not bool(np.isfinite(inverse_matrix).all()):
                continue
            residual = float(np.max(np.abs(matrix - inverse_matrix.conj().T)))
            max_residual = max(max_residual, residual)
            if residual > ABSOLUTE_TOLERANCE:
                inconsistent_keys.append(key)

        evidence: dict[str, JsonValue] = {
            "block_count": len(blocks),
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "missing_inverse_count": len(missing_keys),
            "missing_inverse_sample_keys": _json_keys(missing_keys[:10]),
            "inconsistent_pair_count": len(inconsistent_keys),
            "inconsistent_pair_sample_keys": _json_keys(inconsistent_keys[:10]),
            "nonfinite_block_count": len(nonfinite_keys),
            "nonfinite_sample_keys": _json_keys(nonfinite_keys[:10]),
            "max_conjugate_residual": max_residual,
        }
        if not missing_keys and not inconsistent_keys and not nonfinite_keys:
            return _passed_result(self.contract_id, evidence, start_ns)
        return _failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The Hamiltonian block map violates its declared Hermiticity relation.",
            likely_causes=(
                "A predicted block was written without its inverse partner.",
                "A basis transformation or serialization step changed one side of a block pair.",
            ),
            suggested_actions=(
                "Inspect the bounded block keys before using the Hamiltonian downstream.",
            ),
            affected_artifacts=("hamiltonian_blocks",),
            start_ns=start_ns,
        )


class OverlapGammaPositiveDefiniteContract:
    """Check that the reconstructed Gamma-point overlap admits Cholesky factorization."""

    contract_id = OVERLAP_POSITIVE_DEFINITE_CONTRACT_ID
    description = "The finite Hermitian Gamma-point overlap matrix is positive definite."
    stage = MATERIALS_STAGE
    required_inputs = ("overlap_blocks", "schema")

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        np = _numpy()
        atom_count, orbitals_per_atom = require_block_schema(context)
        blocks = require_blocks(
            context,
            "overlap_blocks",
            atom_count=atom_count,
            orbitals_per_atom=orbitals_per_atom,
        )
        matrix_dimension = atom_count * orbitals_per_atom
        gamma = np.zeros((matrix_dimension, matrix_dimension), dtype=np.complex128)
        for key, block in blocks.items():
            atom_i, atom_j, _, _, _ = key
            row_start = atom_i * orbitals_per_atom
            column_start = atom_j * orbitals_per_atom
            gamma[
                row_start : row_start + orbitals_per_atom,
                column_start : column_start + orbitals_per_atom,
            ] += np.asarray(block)

        nonfinite_value_count = int(gamma.size - np.count_nonzero(np.isfinite(gamma)))
        hermitian_residual: float | None = None
        minimum_eigenvalue: float | None = None
        cholesky_valid = False
        if nonfinite_value_count == 0:
            hermitian_residual = float(np.max(np.abs(gamma - gamma.conj().T)))
            if hermitian_residual <= ABSOLUTE_TOLERANCE:
                hermitian = (gamma + gamma.conj().T) / 2.0
                minimum_eigenvalue = float(np.linalg.eigvalsh(hermitian).min())
                try:
                    np.linalg.cholesky(hermitian)
                    cholesky_valid = True
                except np.linalg.LinAlgError:
                    cholesky_valid = False

        evidence: dict[str, JsonValue] = {
            "block_count": len(blocks),
            "matrix_dimension": matrix_dimension,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "nonfinite_value_count": nonfinite_value_count,
            "hermitian_residual": hermitian_residual,
            "minimum_eigenvalue": minimum_eigenvalue,
            "cholesky_valid": cholesky_valid,
        }
        if (
            nonfinite_value_count == 0
            and hermitian_residual is not None
            and hermitian_residual <= ABSOLUTE_TOLERANCE
            and cholesky_valid
        ):
            return _passed_result(self.contract_id, evidence, start_ns)
        return _failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The Gamma-point overlap is not a finite positive-definite matrix.",
            likely_causes=(
                "Overlap blocks are missing, asymmetric, non-finite, or numerically singular.",
                "The overlap blocks and declared orbital layout came from different samples.",
            ),
            suggested_actions=(
                "Rebuild the overlap from the declared source before diagonalization.",
            ),
            affected_artifacts=("overlap_blocks",),
            start_ns=start_ns,
        )


@dataclass(frozen=True, slots=True)
class DeePTBSourceIdentityContract:
    """Check allowlisted provenance fields against one declared DeePTB source."""

    expected_repository: str
    expected_commit: str
    expected_license: str
    expected_sample_id: str
    expected_files: Mapping[str, tuple[int, str]]
    contract_id: str = field(init=False, default=DEEPTB_SOURCE_IDENTITY_CONTRACT_ID)
    description: str = field(
        init=False,
        default="The DeePTB sample provenance matches the verified local files.",
    )
    stage: str = field(init=False, default=MATERIALS_STAGE)
    required_inputs: tuple[str, ...] = field(init=False, default=("provenance",))

    def __post_init__(self) -> None:
        for name in (
            "expected_repository",
            "expected_commit",
            "expected_license",
            "expected_sample_id",
        ):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
        if not self.expected_files:
            raise ValueError("expected_files must not be empty")
        checked_files: dict[str, tuple[int, str]] = {}
        for file_name, declaration in self.expected_files.items():
            if not file_name or file_name != file_name.strip():
                raise ValueError("expected file names must be non-empty")
            size_bytes, sha256 = declaration
            if size_bytes <= 0 or len(sha256) != 64:
                raise ValueError("expected file identities must contain size and SHA-256")
            checked_files[file_name] = (size_bytes, sha256.lower())
        object.__setattr__(self, "expected_files", MappingProxyType(checked_files))

    def evaluate(self, context: ContractContext) -> ContractResult:
        start_ns = perf_counter_ns()
        declaration = context.provenance.get("deeptb_sample")
        mismatched_fields: list[str] = []
        if not isinstance(declaration, Mapping):
            mismatched_fields.append("deeptb_sample")
        else:
            expected_values = {
                "source_type": "public_repository_sample",
                "project": "DeePTB",
                "repository": self.expected_repository,
                "commit": self.expected_commit,
                "license": self.expected_license,
                "sample_id": self.expected_sample_id,
            }
            mismatched_fields.extend(
                name
                for name, expected in expected_values.items()
                if declaration.get(name) != expected
            )
            declared_files = declaration.get("files")
            if not isinstance(declared_files, Mapping):
                mismatched_fields.append("files")
            else:
                for file_name, (size_bytes, sha256) in self.expected_files.items():
                    expected_file = {"size_bytes": size_bytes, "sha256": sha256}
                    if declared_files.get(file_name) != expected_file:
                        mismatched_fields.append(f"files.{file_name}")

        evidence: dict[str, JsonValue] = {
            "expected_repository": self.expected_repository,
            "expected_commit": self.expected_commit,
            "expected_license": self.expected_license,
            "expected_sample_id": self.expected_sample_id,
            "expected_file_names": _json_strings(sorted(self.expected_files)),
            "mismatched_fields": _json_strings(mismatched_fields),
        }
        if not mismatched_fields:
            return _passed_result(self.contract_id, evidence, start_ns)
        return _failed_result(
            context,
            contract_id=self.contract_id,
            evidence=evidence,
            message="The DeePTB provenance does not match the verified source declaration.",
            likely_causes=(
                "The workflow loaded a different sample or changed provenance after verification.",
            ),
            suggested_actions=(
                "Re-verify the pinned files and rebuild the context from its source descriptor.",
            ),
            affected_artifacts=("provenance",),
            start_ns=start_ns,
        )


def _numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "materials contracts require the 'materials' extra: "
            "pip install 'sciagentguard[materials]'"
        ) from error
    return np


def _json_keys(keys: list[BlockKey]) -> list[JsonValue]:
    return [format_key(key) for key in keys]


def _json_strings(values: list[str]) -> list[JsonValue]:
    return list(values)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000


def _passed_result(
    contract_id: str, evidence: dict[str, JsonValue], start_ns: int
) -> ContractResult:
    return ContractResult(
        contract_id=contract_id,
        status=ContractStatus.PASS,
        evidence=evidence,
        duration_ms=_elapsed_ms(start_ns),
    )


def _failed_result(
    context: ContractContext,
    *,
    contract_id: str,
    evidence: dict[str, JsonValue],
    message: str,
    likely_causes: tuple[str, ...],
    suggested_actions: tuple[str, ...],
    affected_artifacts: tuple[str, ...],
    start_ns: int,
) -> ContractResult:
    violation = ViolationReport(
        violation_id=f"{context.run_id}:{context.attempt_id}:{contract_id}",
        contract_id=contract_id,
        severity=ViolationSeverity.ERROR,
        stage=context.stage,
        message=message,
        evidence=evidence,
        likely_causes=likely_causes,
        suggested_actions=suggested_actions,
        affected_artifacts=affected_artifacts,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        timestamp=datetime.now(timezone.utc),
    )
    return ContractResult(
        contract_id=contract_id,
        status=ContractStatus.FAIL,
        evidence=evidence,
        duration_ms=_elapsed_ms(start_ns),
        violation=violation,
    )


__all__ = [
    "DeePTBSourceIdentityContract",
    "HamiltonianBlockHermiticityContract",
    "OverlapGammaPositiveDefiniteContract",
]
