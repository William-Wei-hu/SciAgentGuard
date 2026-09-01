"""Guard the pinned DeePTB Si64 Hamiltonian and overlap sample."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sciagentguard.adapters import DeePTBSi64Adapter, DeePTBSi64Source
from sciagentguard.runtime import GuardedWorkflowRunner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-directory",
        required=True,
        type=Path,
        help="directory containing the three pinned DeePTB sample files",
    )
    parser.add_argument("--output", type=Path, help="write the JSON trace to this path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = DeePTBSi64Source.official_test_sample(args.cache_directory)
    checkpoint = DeePTBSi64Adapter(source).checkpoint(
        workflow_id="deeptb-si64",
        run_id="si64-smoke",
        attempt_id="attempt-0",
    )
    execution = GuardedWorkflowRunner().execute((checkpoint,))
    trace_json = execution.trace.model_dump_json(indent=2)

    if args.output is None:
        print(trace_json)
    else:
        args.output.write_text(f"{trace_json}\n", encoding="utf-8")
    return 1 if execution.trace.blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
