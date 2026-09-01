"""DeePTB Si64 Hamiltonian input boundary."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import JsonValue

from sciagentguard.core import ContractContext
from sciagentguard.packs.materials import (
    DeePTBSourceIdentityContract,
    HamiltonianBlockHermiticityContract,
    OverlapGammaPositiveDefiniteContract,
)
from sciagentguard.runtime import WorkflowCheckpoint

_FRAME_NAME = "0"
_BLOCK_KEY_PATTERN = re.compile(r"(\d+)_(\d+)_(-?\d+)_(-?\d+)_(-?\d+)")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

BlockKey = tuple[int, int, int, int, int]
MatrixValue = float | complex
ImmutableMatrix = tuple[tuple[MatrixValue, ...], ...]
ImmutableBlocks = Mapping[BlockKey, ImmutableMatrix]


@dataclass(frozen=True, slots=True)
class DeePTBSi64Source:
    """A local copy of the declared DeePTB Si64 sample files."""

    directory: Path
    repository: str
    commit: str
    license_id: str
    sample_id: str
    atom_count: int
    orbitals_per_atom: int
    hamiltonian_file_name: str
    hamiltonian_size_bytes: int
    hamiltonian_sha256: str
    hamiltonian_block_count: int
    overlap_file_name: str
    overlap_size_bytes: int
    overlap_sha256: str
    overlap_block_count: int
    atomic_numbers_file_name: str
    atomic_numbers_size_bytes: int
    atomic_numbers_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            raise TypeError("directory must be a pathlib.Path")
        for name in ("repository", "commit", "license_id", "sample_id"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
        for name in ("atom_count", "orbitals_per_atom"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for prefix in ("hamiltonian", "overlap", "atomic_numbers"):
            file_name = getattr(self, f"{prefix}_file_name")
            if not file_name or Path(file_name).name != file_name:
                raise ValueError(f"{prefix}_file_name must be a file name without a directory")
            if getattr(self, f"{prefix}_size_bytes") <= 0:
                raise ValueError(f"{prefix}_size_bytes must be positive")
            checksum = getattr(self, f"{prefix}_sha256").lower()
            if _SHA256_PATTERN.fullmatch(checksum) is None:
                raise ValueError(f"{prefix}_sha256 must contain 64 hexadecimal characters")
            object.__setattr__(self, f"{prefix}_sha256", checksum)
        if self.hamiltonian_block_count <= 0 or self.overlap_block_count <= 0:
            raise ValueError("block counts must be positive")
        file_names = {
            self.hamiltonian_file_name,
            self.overlap_file_name,
            self.atomic_numbers_file_name,
        }
        if len(file_names) != 3:
            raise ValueError("source file names must be distinct")

    @classmethod
    def official_test_sample(cls, cache_directory: Path) -> DeePTBSi64Source:
        """Describe the Si64.0 sample at the pinned upstream DeePTB commit."""

        return cls(
            directory=cache_directory,
            repository="deepmodeling/DeePTB",
            commit="1dcc7f61480c373870cd5bad1d4000ac80757ff5",
            license_id="LGPL-3.0-or-later",
            sample_id="dptb/tests/data/e3_band/data/Si64.0",
            atom_count=64,
            orbitals_per_atom=4,
            hamiltonian_file_name="hamiltonians.h5",
            hamiltonian_size_bytes=3_945_920,
            hamiltonian_sha256=("3a32d5699298c6765554b982354659d4147c0400daa28514b120fd047209d5be"),
            hamiltonian_block_count=9_434,
            overlap_file_name="overlaps.h5",
            overlap_size_bytes=2_313_608,
            overlap_sha256=("df21a1f7bb1946c5f189877217998813f26c8d42d283c51849fc2c4728b794f5"),
            overlap_block_count=5_562,
            atomic_numbers_file_name="atomic_numbers.dat",
            atomic_numbers_size_bytes=192,
            atomic_numbers_sha256=(
                "faa6f6ec15575294ace0c5b5c8d742128517f01e2d67618358015da983738d2c"
            ),
        )


@dataclass(frozen=True, slots=True)
class DeePTBSi64Adapter:
    """Load one declared DeePTB Si64 sample into a contract context."""

    source: DeePTBSi64Source

    def checkpoint(
        self,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
    ) -> WorkflowCheckpoint:
        """Build the guarded Hamiltonian-load boundary for this source."""

        source = self.source
        identity_contract = DeePTBSourceIdentityContract(
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
        return WorkflowCheckpoint(
            step=lambda: self.load_context(
                workflow_id=workflow_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ),
            contracts=(
                HamiltonianBlockHermiticityContract(),
                OverlapGammaPositiveDefiniteContract(),
                identity_contract,
            ),
        )

    def load_context(
        self,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
    ) -> ContractContext:
        """Verify and load the source without retaining its local directory."""

        paths = _verified_paths(self.source)
        hamiltonian_blocks = _read_blocks(
            paths["hamiltonian"],
            atom_count=self.source.atom_count,
            orbitals_per_atom=self.source.orbitals_per_atom,
            expected_count=self.source.hamiltonian_block_count,
            label="Hamiltonian",
        )
        overlap_blocks = _read_blocks(
            paths["overlap"],
            atom_count=self.source.atom_count,
            orbitals_per_atom=self.source.orbitals_per_atom,
            expected_count=self.source.overlap_block_count,
            label="overlap",
        )
        atomic_numbers = _read_atomic_numbers(
            paths["atomic_numbers"], expected_count=self.source.atom_count
        )

        provenance: dict[str, JsonValue] = {
            "source_type": "public_repository_sample",
            "project": "DeePTB",
            "repository": self.source.repository,
            "commit": self.source.commit,
            "license": self.source.license_id,
            "sample_id": self.source.sample_id,
            "files": {
                self.source.hamiltonian_file_name: {
                    "size_bytes": self.source.hamiltonian_size_bytes,
                    "sha256": self.source.hamiltonian_sha256,
                },
                self.source.overlap_file_name: {
                    "size_bytes": self.source.overlap_size_bytes,
                    "sha256": self.source.overlap_sha256,
                },
                self.source.atomic_numbers_file_name: {
                    "size_bytes": self.source.atomic_numbers_size_bytes,
                    "sha256": self.source.atomic_numbers_sha256,
                },
            },
        }
        schema: dict[str, JsonValue] = {
            "deeptb_blocks": {
                "frame": _FRAME_NAME,
                "key_format": "atom_i_atom_j_Rx_Ry_Rz",
                "atom_count": self.source.atom_count,
                "orbitals_per_atom": self.source.orbitals_per_atom,
                "hamiltonian_block_count": self.source.hamiltonian_block_count,
                "overlap_block_count": self.source.overlap_block_count,
            }
        }

        return ContractContext(
            workflow_id=workflow_id,
            run_id=run_id,
            attempt_id=attempt_id,
            stage="post_hamiltonian_load",
            artifacts={
                "hamiltonian_blocks": hamiltonian_blocks,
                "overlap_blocks": overlap_blocks,
                "atomic_numbers": atomic_numbers,
            },
            schema=schema,
            units={"overlap_blocks": "dimensionless"},
            provenance={"deeptb_sample": provenance},
        )


def _verified_paths(source: DeePTBSi64Source) -> dict[str, Path]:
    declarations = {
        "hamiltonian": (
            source.hamiltonian_file_name,
            source.hamiltonian_size_bytes,
            source.hamiltonian_sha256,
        ),
        "overlap": (
            source.overlap_file_name,
            source.overlap_size_bytes,
            source.overlap_sha256,
        ),
        "atomic_numbers": (
            source.atomic_numbers_file_name,
            source.atomic_numbers_size_bytes,
            source.atomic_numbers_sha256,
        ),
    }
    paths: dict[str, Path] = {}
    for label, (file_name, expected_size, expected_sha256) in declarations.items():
        path = source.directory / file_name
        _verify_file(path, label, expected_size, expected_sha256)
        paths[label] = path
    return paths


def _verify_file(path: Path, label: str, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"the declared {label} source is not a file")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size} bytes, found {actual_size}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )


def _read_blocks(
    path: Path,
    *,
    atom_count: int,
    orbitals_per_atom: int,
    expected_count: int,
    label: str,
) -> ImmutableBlocks:
    try:
        import h5py
        import numpy as np
    except ModuleNotFoundError as error:
        if error.name in {"h5py", "numpy"}:
            raise ModuleNotFoundError(
                "DeePTB HDF5 support requires the 'materials' extra: "
                "pip install 'sciagentguard[materials]'"
            ) from error
        raise

    try:
        source = h5py.File(path, "r")
    except OSError as error:
        raise ValueError(f"{label} source is not a readable HDF5 file") from error

    blocks: dict[BlockKey, ImmutableMatrix] = {}
    with source:
        if set(source.keys()) != {_FRAME_NAME}:
            raise ValueError(f"{label} source must contain only frame group {_FRAME_NAME!r}")
        frame = source[_FRAME_NAME]
        if not isinstance(frame, h5py.Group):
            raise ValueError(f"{label} frame {_FRAME_NAME!r} must be an HDF5 group")
        for raw_key in frame:
            block = frame[raw_key]
            if not isinstance(block, h5py.Dataset):
                raise ValueError(f"{label} block {raw_key!r} must be an HDF5 dataset")
            key = _parse_block_key(raw_key, label, atom_count)
            values = block[()]
            expected_shape = (orbitals_per_atom, orbitals_per_atom)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{label} block {raw_key!r} must have shape {expected_shape}, "
                    f"found {values.shape}"
                )
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"{label} block {raw_key!r} must contain numeric values")
            blocks[key] = tuple(
                tuple(_python_number(value) for value in row) for row in values.tolist()
            )

    if len(blocks) != expected_count:
        raise ValueError(
            f"{label} block count mismatch: expected {expected_count}, found {len(blocks)}"
        )
    return MappingProxyType(blocks)


def _parse_block_key(raw_key: str, label: str, atom_count: int) -> BlockKey:
    match = _BLOCK_KEY_PATTERN.fullmatch(raw_key)
    if match is None:
        raise ValueError(f"{label} block key {raw_key!r} is not in i_j_Rx_Ry_Rz format")
    key = tuple(int(value) for value in match.groups())
    atom_i, atom_j, rx, ry, rz = key
    if atom_i >= atom_count or atom_j >= atom_count:
        raise ValueError(f"{label} block key {raw_key!r} references an unknown atom")
    return atom_i, atom_j, rx, ry, rz


def _python_number(value: object) -> MatrixValue:
    if isinstance(value, complex):
        return complex(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError("matrix values must be real or complex numbers")


def _read_atomic_numbers(path: Path, *, expected_count: int) -> tuple[int, ...]:
    try:
        tokens = path.read_text(encoding="utf-8").split()
        values = tuple(int(token) for token in tokens)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            "atomic numbers source must contain whitespace-separated integers"
        ) from error
    if len(values) != expected_count:
        raise ValueError(
            f"atomic number count mismatch: expected {expected_count}, found {len(values)}"
        )
    if any(value <= 0 for value in values):
        raise ValueError("atomic numbers must be positive")
    return values


__all__ = ["DeePTBSi64Adapter", "DeePTBSi64Source"]
