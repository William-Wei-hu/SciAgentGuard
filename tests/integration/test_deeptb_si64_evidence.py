from pathlib import Path

from sciagentguard.core import ContractStatus
from sciagentguard.runtime import WorkflowTrace

SMOKE_TRACE = Path(__file__).parents[2] / "benchmarks" / "results" / "deeptb_si64_smoke.json"


def test_committed_deeptb_smoke_trace_records_the_pinned_boundary() -> None:
    payload = SMOKE_TRACE.read_text(encoding="utf-8")
    trace = WorkflowTrace.model_validate_json(payload)

    assert trace.workflow_id == "deeptb-si64"
    assert trace.run_id == "si64-smoke"
    assert not trace.blocked
    results = {result.contract_id: result for result in trace.checkpoints[0].results}
    assert all(result.status is ContractStatus.PASS for result in results.values())
    assert results["materials.hamiltonian.block_hermiticity"].evidence["block_count"] == 9_434
    overlap = results["materials.overlap.gamma_positive_definite"].evidence
    assert overlap["block_count"] == 5_562
    assert overlap["matrix_dimension"] == 256
    assert overlap["cholesky_valid"] is True
    assert isinstance(overlap["minimum_eigenvalue"], float)
    assert overlap["minimum_eigenvalue"] > 0.0

    assert "/Users/" not in payload
    assert "williamwade" not in payload
    assert ".cache" not in payload
    assert "hamiltonian_blocks" not in payload
    assert "overlap_blocks" not in payload
