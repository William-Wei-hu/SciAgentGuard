import sys
from collections.abc import MutableMapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from sciagentguard.adapters import DeePTBSi64Adapter, DeePTBSi64Source
from sciagentguard.runtime import GuardedWorkflowRunner, WorkflowTrace
from tests.integration._deeptb_hdf5 import (
    sha256,
    valid_hamiltonian_blocks,
    write_block_file,
    write_test_source,
)


def test_adapter_loads_immutable_blocks_without_exposing_the_path(tmp_path: Path) -> None:
    source = write_test_source(tmp_path / "LOCAL_SECRET_SENTINEL")

    context = DeePTBSi64Adapter(source).load_context(
        workflow_id="deeptb-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )

    assert context.stage == "post_hamiltonian_load"
    assert context.artifacts["atomic_numbers"] == (14, 14)
    hamiltonian = cast(
        MutableMapping[tuple[int, int, int, int, int], object],
        context.artifacts["hamiltonian_blocks"],
    )
    assert hamiltonian[(0, 1, 1, 0, 0)] == ((0.2, 0.1j), (-0.1j, 0.3))
    with pytest.raises(TypeError):
        hamiltonian[(0, 0, 0, 0, 0)] = ((0.0,),)
    assert context.provenance["deeptb_sample"] == {
        "source_type": "public_repository_sample",
        "project": "DeePTB",
        "repository": "test/DeePTB",
        "commit": "test-commit",
        "license": "LGPL-3.0-or-later",
        "sample_id": "tests/Si2.0",
        "files": {
            source.hamiltonian_file_name: {
                "size_bytes": source.hamiltonian_size_bytes,
                "sha256": source.hamiltonian_sha256,
            },
            source.overlap_file_name: {
                "size_bytes": source.overlap_size_bytes,
                "sha256": source.overlap_sha256,
            },
            source.atomic_numbers_file_name: {
                "size_bytes": source.atomic_numbers_size_bytes,
                "sha256": source.atomic_numbers_sha256,
            },
        },
    }
    assert str(tmp_path) not in repr(context)
    assert "LOCAL_SECRET_SENTINEL" not in repr(context)


def test_official_source_descriptor_is_pinned_to_deeptb_si64() -> None:
    source = DeePTBSi64Source.official_test_sample(Path(".cache/deeptb-si64"))

    assert source.repository == "deepmodeling/DeePTB"
    assert source.commit == "1dcc7f61480c373870cd5bad1d4000ac80757ff5"
    assert source.atom_count == 64
    assert source.orbitals_per_atom == 4
    assert source.hamiltonian_block_count == 9_434
    assert source.overlap_block_count == 5_562


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("hamiltonian_size_bytes", "hamiltonian size mismatch"),
        ("overlap_sha256", "overlap SHA-256 mismatch"),
        ("atomic_numbers_sha256", "atomic_numbers SHA-256 mismatch"),
    ],
)
def test_adapter_rejects_sources_that_fail_integrity_checks(
    tmp_path: Path, field: str, message: str
) -> None:
    source = write_test_source(tmp_path)
    if field == "hamiltonian_size_bytes":
        invalid = replace(source, hamiltonian_size_bytes=source.hamiltonian_size_bytes + 1)
    elif field == "overlap_sha256":
        invalid = replace(source, overlap_sha256="0" * 64)
    else:
        invalid = replace(source, atomic_numbers_sha256="0" * 64)

    with pytest.raises(ValueError, match=message):
        DeePTBSi64Adapter(invalid).load_context(
            workflow_id="deeptb-test",
            run_id="run-001",
            attempt_id="attempt-0",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("frame", "must contain only frame group '0'"),
        ("key", "is not in i_j_Rx_Ry_Rz format"),
        ("shape", r"must have shape \(2, 2\)"),
        ("atom", "references an unknown atom"),
        ("count", "block count mismatch"),
    ],
)
def test_adapter_rejects_invalid_block_layouts(tmp_path: Path, mutation: str, message: str) -> None:
    source = write_test_source(tmp_path)
    path = tmp_path / source.hamiltonian_file_name
    blocks = valid_hamiltonian_blocks()
    frame = "0"
    if mutation == "frame":
        frame = "1"
    elif mutation == "key":
        blocks["invalid"] = blocks.pop("0_0_0_0_0")
    elif mutation == "shape":
        blocks["0_0_0_0_0"] = [[1.0]]
    elif mutation == "atom":
        blocks["2_0_0_0_0"] = blocks.pop("0_0_0_0_0")
    elif mutation == "count":
        blocks.pop("0_1_1_0_0")
    write_block_file(path, blocks, frame_name=frame)
    changed = replace(
        source,
        hamiltonian_size_bytes=path.stat().st_size,
        hamiltonian_sha256=sha256(path),
    )

    with pytest.raises(ValueError, match=message):
        DeePTBSi64Adapter(changed).load_context(
            workflow_id="deeptb-test",
            run_id="run-001",
            attempt_id="attempt-0",
        )


def test_adapter_explains_the_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_test_source(tmp_path)
    modules = cast(dict[str, object], sys.modules)
    monkeypatch.setitem(modules, "h5py", None)

    with pytest.raises(ModuleNotFoundError, match="requires the 'materials' extra"):
        DeePTBSi64Adapter(source).load_context(
            workflow_id="deeptb-test",
            run_id="run-001",
            attempt_id="attempt-0",
        )


def test_adapter_checkpoint_runs_contracts_without_leaking_runtime_data(tmp_path: Path) -> None:
    sentinel = 12345.6789
    blocks = valid_hamiltonian_blocks()
    sentinel_matrix: list[list[float | complex]] = [[sentinel, 0.0], [0.0, sentinel]]
    blocks["0_1_1_0_0"] = sentinel_matrix
    blocks["1_0_-1_0_0"] = sentinel_matrix
    source = write_test_source(tmp_path / "LOCAL_SECRET_SENTINEL", hamiltonian_blocks=blocks)
    checkpoint = DeePTBSi64Adapter(source).checkpoint(
        workflow_id="deeptb-test",
        run_id="run-001",
        attempt_id="attempt-0",
    )

    execution = GuardedWorkflowRunner().execute((checkpoint,))

    assert execution.output is not None
    assert not execution.trace.blocked
    assert [result.contract_id for result in execution.trace.checkpoints[0].results] == [
        "materials.hamiltonian.block_hermiticity",
        "materials.overlap.gamma_positive_definite",
        "materials.deeptb.source_identity",
    ]
    payload = execution.trace.model_dump_json()
    assert "LOCAL_SECRET_SENTINEL" not in payload
    assert str(tmp_path) not in payload
    assert str(sentinel) not in payload
    assert WorkflowTrace.model_validate_json(payload) == execution.trace
