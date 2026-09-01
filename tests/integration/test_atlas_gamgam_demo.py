import json
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from sciagentguard.adapters import AtlasGamGamSource
from tests.integration._atlas_root import synthetic_source, write_root_file

DEMO = Path(__file__).parents[2] / "examples" / "atlas_gamgam_guard.py"


def _main() -> Callable[[Sequence[str] | None], int]:
    namespace = runpy.run_path(str(DEMO))
    return cast(Callable[[Sequence[str] | None], int], namespace["main"])


def test_demo_emits_one_safe_json_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root_path = tmp_path / "LOCAL_PATH_SECRET" / "test.root"
    root_path.parent.mkdir()
    write_root_file(root_path)
    source = synthetic_source(root_path)
    monkeypatch.setattr(
        AtlasGamGamSource,
        "official_wph125",
        classmethod(lambda cls, path: source),
    )

    assert _main()(("--input", str(root_path))) == 0

    output = capsys.readouterr().out
    trace = cast(dict[str, Any], json.loads(output))
    assert trace["blocked"] is False
    assert [checkpoint["stage"] for checkpoint in trace["checkpoints"]] == [
        "post_load",
        "post_selection",
        "post_histogram",
        "post_yield",
    ]
    assert "LOCAL_PATH_SECRET" not in output
    assert str(tmp_path) not in output


def test_demo_writes_the_same_trace_shape_to_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = tmp_path / "test.root"
    output_path = tmp_path / "trace.json"
    write_root_file(root_path)
    source = synthetic_source(root_path)
    monkeypatch.setattr(
        AtlasGamGamSource,
        "official_wph125",
        classmethod(lambda cls, path: source),
    )

    assert _main()(("--input", str(root_path), "--output", str(output_path))) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["blocked"] is False


def test_demo_requires_an_input_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(DEMO)])

    with pytest.raises(SystemExit, match="2"):
        _main()(None)
