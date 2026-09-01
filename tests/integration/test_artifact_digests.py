"""The README's opening claim must be reproducible, and it is the claim this script prints."""

from __future__ import annotations

import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from tests.integration._atlas_root import write_root_file

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "benchmarks" / "artifact_digests.py"
NAMESPACE = runpy.run_path(str(SCRIPT))

COLLECT = cast(Callable[[Path], Mapping[str, str | None]], NAMESPACE["collect"])
MAIN = cast(Callable[[list[str]], int], NAMESPACE["main"])


@pytest.fixture(scope="module")
def digests(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, str | None]:
    path = tmp_path_factory.mktemp("digests") / "test.root"
    write_root_file(path)
    return COLLECT(path)


def test_two_faulty_analyses_end_at_the_correct_analysis_digest(
    digests: Mapping[str, str | None],
) -> None:
    """The repository's central claim: some faults are invisible in the final artifact.

    If this ever fails, either the faults stopped being late-invisible or the analysis started
    carrying something downstream that distinguishes them. Both change what the README may claim.
    """

    correct = digests["correct"]

    assert correct is not None
    assert digests["overlapping_control_window"] == correct
    assert digests["stale_cutflow"] == correct


def test_the_faults_that_change_the_result_are_distinguishable(
    digests: Mapping[str, str | None],
) -> None:
    """Not every fault is late-invisible, and claiming otherwise would overstate the result."""

    correct = digests["correct"]
    distinct = {digests["empty_selection"], digests["luminosity_unit_slip"]}

    assert None not in distinct
    assert correct not in distinct
    assert len(distinct) == 2


def test_an_analysis_that_produces_nothing_has_no_digest(
    digests: Mapping[str, str | None],
) -> None:
    assert digests["unrunnable"] is None
    assert digests["wrong_output_schema"] is None


def test_the_script_reports_the_collision_it_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "test.root"
    write_root_file(path)

    assert MAIN(["--input", str(path)]) == 0

    printed = capsys.readouterr().out
    assert "3 analyses share" in printed
    assert "can tell them apart" in printed
