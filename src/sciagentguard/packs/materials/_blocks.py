"""Validation helpers for DeePTB block artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Complex
from typing import cast

from sciagentguard.core import ContractContext

BlockKey = tuple[int, int, int, int, int]
MatrixValue = float | complex
ImmutableMatrix = tuple[tuple[MatrixValue, ...], ...]
BlockMapping = Mapping[BlockKey, ImmutableMatrix]


def require_block_schema(context: ContractContext) -> tuple[int, int]:
    schema = context.schema.get("deeptb_blocks")
    if not isinstance(schema, Mapping):
        raise ValueError("schema 'deeptb_blocks' must be a mapping")
    atom_count = schema.get("atom_count")
    orbitals_per_atom = schema.get("orbitals_per_atom")
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise ValueError("schema 'deeptb_blocks.atom_count' must be a positive integer")
    if (
        isinstance(orbitals_per_atom, bool)
        or not isinstance(orbitals_per_atom, int)
        or orbitals_per_atom <= 0
    ):
        raise ValueError("schema 'deeptb_blocks.orbitals_per_atom' must be a positive integer")
    return atom_count, orbitals_per_atom


def require_blocks(
    context: ContractContext,
    artifact_name: str,
    *,
    atom_count: int,
    orbitals_per_atom: int,
) -> BlockMapping:
    raw_blocks = context.artifacts.get(artifact_name)
    if not isinstance(raw_blocks, Mapping):
        raise ValueError(f"artifact {artifact_name!r} must be a block mapping")
    if not raw_blocks:
        raise ValueError(f"artifact {artifact_name!r} must not be empty")

    for raw_key, raw_matrix in raw_blocks.items():
        if not _valid_key(raw_key, atom_count):
            raise ValueError(f"artifact {artifact_name!r} contains an invalid block key")
        if not isinstance(raw_matrix, Sequence) or isinstance(raw_matrix, (str, bytes, bytearray)):
            raise ValueError(f"artifact {artifact_name!r} contains a non-matrix block")
        if len(raw_matrix) != orbitals_per_atom:
            raise ValueError(f"artifact {artifact_name!r} contains a block with the wrong shape")
        for row in raw_matrix:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise ValueError(f"artifact {artifact_name!r} contains a non-matrix block")
            if len(row) != orbitals_per_atom:
                raise ValueError(
                    f"artifact {artifact_name!r} contains a block with the wrong shape"
                )
            if any(isinstance(value, bool) or not isinstance(value, Complex) for value in row):
                raise ValueError(f"artifact {artifact_name!r} contains a non-numeric matrix value")
    return cast(BlockMapping, raw_blocks)


def copy_blocks(
    context: ContractContext,
    artifact_name: str,
    *,
    atom_count: int,
    orbitals_per_atom: int,
) -> dict[BlockKey, ImmutableMatrix]:
    """Validate and copy one block mapping for deterministic fault injection."""

    return dict(
        require_blocks(
            context,
            artifact_name,
            atom_count=atom_count,
            orbitals_per_atom=orbitals_per_atom,
        )
    )


def inverse_key(key: BlockKey) -> BlockKey:
    atom_i, atom_j, rx, ry, rz = key
    return atom_j, atom_i, -rx, -ry, -rz


def format_key(key: BlockKey) -> str:
    return "_".join(str(value) for value in key)


def _valid_key(raw_key: object, atom_count: int) -> bool:
    if not isinstance(raw_key, tuple) or len(raw_key) != 5:
        return False
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_key):
        return False
    atom_i, atom_j, _, _, _ = cast(BlockKey, raw_key)
    return 0 <= atom_i < atom_count and 0 <= atom_j < atom_count
