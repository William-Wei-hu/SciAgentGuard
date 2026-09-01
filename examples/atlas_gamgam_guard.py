"""Guard the fixed ATLAS Open Data WpH125 Gamma-Gamma ROOT sample."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.runtime import GuardedWorkflowRunner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="local path to the ROOT file")
    parser.add_argument("--output", type=Path, help="write the JSON trace to this path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = AtlasGamGamSource.official_wph125(args.input)
    checkpoints = AtlasGamGamOpenDataAdapter(source).checkpoints(
        workflow_id="atlas-gamgam-open-data",
        run_id="wph125-smoke",
        attempt_id="attempt-0",
    )
    execution = GuardedWorkflowRunner().execute(checkpoints)
    trace_json = execution.trace.model_dump_json(indent=2)

    if args.output is None:
        print(trace_json)
    else:
        args.output.write_text(f"{trace_json}\n", encoding="utf-8")
    return 1 if execution.trace.blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
