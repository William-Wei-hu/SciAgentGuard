import hashlib
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

from sciagentguard.adapters import DeePTBSi64Source

Matrix = list[list[float | complex]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid_hamiltonian_blocks() -> dict[str, Matrix]:
    return {
        "0_0_0_0_0": [[-1.0, 0.0], [0.0, 1.0]],
        "1_1_0_0_0": [[-0.5, 0.0], [0.0, 0.5]],
        "0_1_1_0_0": [[0.2, 0.1j], [-0.1j, 0.3]],
        "1_0_-1_0_0": [[0.2, 0.1j], [-0.1j, 0.3]],
    }


def valid_overlap_blocks() -> dict[str, Matrix]:
    return {
        "0_0_0_0_0": [[1.0, 0.0], [0.0, 1.0]],
        "1_1_0_0_0": [[1.0, 0.0], [0.0, 1.0]],
        "0_1_1_0_0": [[0.05, 0.0], [0.0, 0.05]],
        "1_0_-1_0_0": [[0.05, 0.0], [0.0, 0.05]],
    }


def write_block_file(
    path: Path,
    blocks: Mapping[str, Matrix],
    *,
    frame_name: str = "0",
) -> None:
    with h5py.File(path, "w") as source:
        frame = source.create_group(frame_name)
        for key, values in blocks.items():
            frame.create_dataset(key, data=np.asarray(values))


def write_test_source(
    directory: Path,
    *,
    hamiltonian_blocks: Mapping[str, Matrix] | None = None,
    overlap_blocks: Mapping[str, Matrix] | None = None,
    hamiltonian_frame: str = "0",
    overlap_frame: str = "0",
) -> DeePTBSi64Source:
    directory.mkdir(parents=True, exist_ok=True)
    hamiltonian_path = directory / "hamiltonians.h5"
    overlap_path = directory / "overlaps.h5"
    atomic_numbers_path = directory / "atomic_numbers.dat"
    hamiltonian = hamiltonian_blocks or valid_hamiltonian_blocks()
    overlap = overlap_blocks or valid_overlap_blocks()
    write_block_file(hamiltonian_path, hamiltonian, frame_name=hamiltonian_frame)
    write_block_file(overlap_path, overlap, frame_name=overlap_frame)
    atomic_numbers_path.write_text("14 14\n", encoding="utf-8")

    return DeePTBSi64Source(
        directory=directory,
        repository="test/DeePTB",
        commit="test-commit",
        license_id="LGPL-3.0-or-later",
        sample_id="tests/Si2.0",
        atom_count=2,
        orbitals_per_atom=2,
        hamiltonian_file_name=hamiltonian_path.name,
        hamiltonian_size_bytes=hamiltonian_path.stat().st_size,
        hamiltonian_sha256=sha256(hamiltonian_path),
        hamiltonian_block_count=len(hamiltonian),
        overlap_file_name=overlap_path.name,
        overlap_size_bytes=overlap_path.stat().st_size,
        overlap_sha256=sha256(overlap_path),
        overlap_block_count=len(overlap),
        atomic_numbers_file_name=atomic_numbers_path.name,
        atomic_numbers_size_bytes=atomic_numbers_path.stat().st_size,
        atomic_numbers_sha256=sha256(atomic_numbers_path),
    )
