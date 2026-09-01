import json
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import DeePTBSi64Source
from tests.integration._deeptb_hdf5 import write_test_source

DEMO = Path(__file__).parents[2] / "examples" / "deeptb_si64_guard.py"


def _main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(str(DEMO))
    return cast(Callable[[Sequence[str] | None], int], namespace["main"])


def test_demo_emits_one_safe_json_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_directory = tmp_path / "LOCAL_PATH_SECRET"
    source = write_test_source(cache_directory)
    monkeypatch.setattr(
        DeePTBSi64Source,
        "official_test_sample",
        classmethod(lambda cls, directory: source),
    )

    assert _main()(("--cache-directory", str(cache_directory))) == 0

    output = capsys.readouterr().out
    trace = cast(dict[str, Any], json.loads(output))
    assert trace["blocked"] is False
    assert len(trace["checkpoints"]) == 1
    assert "LOCAL_PATH_SECRET" not in output
    assert str(tmp_path) not in output


def test_demo_writes_the_same_trace_shape_to_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_test_source(tmp_path / "source")
    output_path = tmp_path / "trace.json"
    monkeypatch.setattr(
        DeePTBSi64Source,
        "official_test_sample",
        classmethod(lambda cls, directory: source),
    )

    assert (
        _main()(
            (
                "--cache-directory",
                str(source.directory),
                "--output",
                str(output_path),
            )
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["blocked"] is False


def test_demo_requires_a_cache_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(DEMO)])

    with pytest.raises(SystemExit, match="2"):
        _main()(None)
