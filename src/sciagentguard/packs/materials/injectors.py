"""Deterministic faults for the DeePTB Si64 boundary experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from types import MappingProxyType

from pydantic import JsonValue

from sciagentguard.core import ContractContext
from sciagentguard.packs.materials._blocks import (
    BlockKey,
    ImmutableMatrix,
    copy_blocks,
    inverse_key,
    require_block_schema,
)
from sciagentguard.packs.materials.contracts import (
    DEEPTB_SOURCE_IDENTITY_CONTRACT_ID,
    HAMILTONIAN_HERMITICITY_CONTRACT_ID,
    MATERIALS_STAGE,
    OVERLAP_POSITIVE_DEFINITE_CONTRACT_ID,
)


class DeePTBMissingHamiltonianInverseInjector:
    """Remove one inverse Hamiltonian block from a copied mapping."""

    fault_id = "deeptb_missing_hamiltonian_inverse"
    taxonomy = "inverse_block"
    description = "Remove one inverse Hamiltonian block after the HDF5 boundary."
    preconditions = ("The context contains a non-self inverse Hamiltonian block pair.",)
    mutation_description = "Remove the inverse of the first sorted eligible block key."
    expected_contract_ids = (HAMILTONIAN_HERMITICITY_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared HDF5 source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        blocks = _copied_blocks(context, "hamiltonian_blocks", self.fault_id)
        key = _first_inverse_pair(blocks, self.fault_id)
        del blocks[inverse_key(key)]
        return _replace_artifact(context, "hamiltonian_blocks", blocks)


class DeePTBIndefiniteOverlapInjector:
    """Make the Gamma-point overlap indefinite through one onsite entry."""

    fault_id = "deeptb_indefinite_overlap"
    taxonomy = "positive_definiteness"
    description = "Change one onsite overlap entry so the Gamma matrix has a negative diagonal."
    preconditions = (
        "The context contains finite overlap blocks with a positive real onsite diagonal.",
    )
    mutation_description = "Shift the first sorted zero-translation onsite [0, 0] entry."
    expected_contract_ids = (OVERLAP_POSITIVE_DEFINITE_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared HDF5 source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        blocks = _copied_blocks(context, "overlap_blocks", self.fault_id)
        onsite_keys = tuple(
            key for key in sorted(blocks) if key[0] == key[1] and key[2:] == (0, 0, 0)
        )
        if not onsite_keys:
            raise ValueError(f"{self.fault_id} requires a zero-translation onsite overlap block")
        key = onsite_keys[0]
        atom_index = key[0]
        diagonal_sum = sum(
            (
                complex(matrix[0][0])
                for block_key, matrix in blocks.items()
                if block_key[:2] == (atom_index, atom_index)
            ),
            start=0j,
        )
        if (
            not isfinite(diagonal_sum.real)
            or not isfinite(diagonal_sum.imag)
            or abs(diagonal_sum.imag) > 1e-6
            or diagonal_sum.real <= 0.0
        ):
            raise ValueError(
                f"{self.fault_id} requires a finite positive real Gamma-point diagonal"
            )

        matrix = blocks[key]
        changed_value = complex(matrix[0][0]) - (2.0 * diagonal_sum.real + 1.0)
        blocks[key] = _replace_matrix_value(matrix, row=0, column=0, value=changed_value)
        return _replace_artifact(context, "overlap_blocks", blocks)


class DeePTBSourceIdentityDriftInjector:
    """Change the declared upstream commit without touching the sample artifacts."""

    fault_id = "deeptb_source_identity_drift"
    taxonomy = "source_identity"
    description = "Replace the verified DeePTB commit with a stable mismatched value."
    preconditions = ("The context contains DeePTB sample provenance with a non-empty commit.",)
    mutation_description = "Set provenance['deeptb_sample']['commit'] to 'deeptb-commit-drift'."
    expected_contract_ids = (DEEPTB_SOURCE_IDENTITY_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and reload the declared HDF5 source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        _require_stage(context, self.fault_id)
        declaration = _sample_provenance(context, self.fault_id)
        commit = declaration.get("commit")
        if not isinstance(commit, str) or not commit.strip():
            raise ValueError(f"{self.fault_id} requires a non-empty commit")
        declaration["commit"] = "deeptb-commit-drift"
        provenance = dict(context.provenance)
        provenance["deeptb_sample"] = declaration
        return replace(context, provenance=provenance)


class DeePTBAtomicSpeciesDriftInjector:
    """Probe the absence of an artifact-level Si species contract."""

    fault_id = "deeptb_atomic_species_drift"
    taxonomy = "species_consistency_gap"
    description = "Change one Si atomic number while preserving verified source provenance."
    preconditions = ("The atomic_numbers artifact starts with silicon atomic number 14.",)
    mutation_description = "Replace atomic_numbers[0] with carbon atomic number 6."
    expected_contract_ids: tuple[str, ...] = ()
    restoration_strategy = "Discard the injected context and reload the declared HDF5 source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        _require_stage(context, self.fault_id)
        raw_values = context.artifacts.get("atomic_numbers")
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes, bytearray)):
            raise ValueError(f"{self.fault_id} requires an atomic_numbers sequence")
        values = tuple(raw_values)
        if not values or values[0] != 14:
            raise ValueError(f"{self.fault_id} requires silicon atomic number 14 at index 0")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError(f"{self.fault_id} requires positive integer atomic numbers")
        return _replace_artifact(context, "atomic_numbers", (6, *values[1:]))


class DeePTBHermitianContentDriftInjector:
    """Probe value drift that preserves the checked inverse-block relation."""

    fault_id = "deeptb_hermitian_content_drift"
    taxonomy = "reference_value_gap"
    description = "Change one Hamiltonian pair while preserving conjugate Hermiticity."
    preconditions = ("The context contains a finite non-self inverse Hamiltonian block pair.",)
    mutation_description = "Add 0.5 to matching [0, 0] entries of the first sorted pair."
    expected_contract_ids: tuple[str, ...] = ()
    restoration_strategy = "Discard the injected context and reload the declared HDF5 source."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        blocks = _copied_blocks(context, "hamiltonian_blocks", self.fault_id)
        key = _first_inverse_pair(blocks, self.fault_id)
        inverse = inverse_key(key)
        value = complex(blocks[key][0][0])
        inverse_value = complex(blocks[inverse][0][0])
        if not all(
            isfinite(component)
            for component in (value.real, value.imag, inverse_value.real, inverse_value.imag)
        ):
            raise ValueError(f"{self.fault_id} requires finite paired matrix entries")
        blocks[key] = _replace_matrix_value(blocks[key], row=0, column=0, value=value + 0.5)
        blocks[inverse] = _replace_matrix_value(
            blocks[inverse], row=0, column=0, value=inverse_value + 0.5
        )
        return _replace_artifact(context, "hamiltonian_blocks", blocks)


def _require_stage(context: ContractContext, fault_id: str) -> None:
    if context.stage != MATERIALS_STAGE:
        raise ValueError(f"{fault_id} requires stage {MATERIALS_STAGE!r}")


def _copied_blocks(
    context: ContractContext,
    artifact_name: str,
    fault_id: str,
) -> dict[BlockKey, ImmutableMatrix]:
    _require_stage(context, fault_id)
    atom_count, orbitals_per_atom = require_block_schema(context)
    return copy_blocks(
        context,
        artifact_name,
        atom_count=atom_count,
        orbitals_per_atom=orbitals_per_atom,
    )


def _first_inverse_pair(blocks: Mapping[BlockKey, ImmutableMatrix], fault_id: str) -> BlockKey:
    for key in sorted(blocks):
        inverse = inverse_key(key)
        if inverse != key and inverse in blocks:
            return key
    raise ValueError(f"{fault_id} requires a non-self inverse block pair")


def _replace_matrix_value(
    matrix: ImmutableMatrix,
    *,
    row: int,
    column: int,
    value: complex,
) -> ImmutableMatrix:
    rows = [list(values) for values in matrix]
    rows[row][column] = value.real if value.imag == 0.0 else value
    return tuple(tuple(values) for values in rows)


def _replace_artifact(
    context: ContractContext,
    artifact_name: str,
    value: object,
) -> ContractContext:
    artifacts = dict(context.artifacts)
    artifacts[artifact_name] = MappingProxyType(value) if isinstance(value, dict) else value
    return replace(context, artifacts=artifacts)


def _sample_provenance(context: ContractContext, fault_id: str) -> dict[str, JsonValue]:
    declaration = context.provenance.get("deeptb_sample")
    if not isinstance(declaration, Mapping):
        raise ValueError(f"{fault_id} requires DeePTB sample provenance")
    copied: dict[str, JsonValue] = {}
    for key, value in declaration.items():
        if not isinstance(key, str):
            raise ValueError(f"{fault_id} requires string provenance keys")
        copied[key] = value
    return copied


__all__ = [
    "DeePTBAtomicSpeciesDriftInjector",
    "DeePTBHermitianContentDriftInjector",
    "DeePTBIndefiniteOverlapInjector",
    "DeePTBMissingHamiltonianInverseInjector",
    "DeePTBSourceIdentityDriftInjector",
]
