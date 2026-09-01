"""Ask one reviewer the same question repeatedly and count how many answers come back.

The same prompt has already drawn VALID from one call and INVALID from another. The two artifacts
behind that were checked and serialise identically, so it was one input and two answers rather than
two similar inputs -- but two observations are a hint, not a measurement.

The cache is bypassed deliberately and cannot be enabled here. A cache returns the same answer by
construction, so measuring determinism through one would report perfect determinism whatever the
model does.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sciagentguard import __version__
from sciagentguard.adapters import AtlasGamGamOpenDataAdapter, AtlasGamGamSource
from sciagentguard.adapters.agent.deepseek import (
    DEFAULT_JUDGE_MODEL,
    PROVIDER,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekJudge,
    load_deepseek_key,
)

PROBE_ID: Literal["judge_reproducibility"] = "judge_reproducibility"
DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/judge_reproducibility.json")
DEFAULT_MARKDOWN_OUTPUT = Path(".cache/reports/judge_reproducibility.md")


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    index: int
    verdict: str
    latency_ms: float


class ProbeReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    probe_id: Literal["judge_reproducibility"] = PROBE_ID
    generated_at: datetime
    environment: dict[str, str]
    parameters: dict[str, str | int | float]
    artifact_reviewed: str
    attempts: tuple[Attempt, ...]

    @property
    def distinct_answers(self) -> int:
        return len({attempt.verdict for attempt in self.attempts if attempt.verdict != "ERROR"})

    @property
    def answered(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.verdict != "ERROR")


def run_probe(
    source: AtlasGamGamSource,
    *,
    attempts: int,
    checkpoint: Callable[[ProbeReport], None] | None = None,
) -> ProbeReport:
    adapter = AtlasGamGamOpenDataAdapter(source)
    artifact = dict(
        adapter.contexts(workflow_id="judge-probe", run_id="r", attempt_id="a")[3].artifacts[
            "yield_estimate"
        ]
    )
    # Deliberately the bare client: no CachingClient may appear on this path.
    judge = DeepSeekJudge(client=DeepSeekClient(api_key=load_deepseek_key()))

    records: list[Attempt] = []
    for index in range(attempts):
        started = monotonic()
        try:
            verdict = "VALID" if judge.verdict(artifact) else "INVALID"
        except DeepSeekError:
            verdict = "ERROR"
        elapsed = (monotonic() - started) * 1000.0
        records.append(Attempt(index=index, verdict=verdict, latency_ms=elapsed))
        print(f"attempt {index}: {verdict} after {elapsed / 1000.0:.0f}s", flush=True)
        # Written after every attempt. Each one costs minutes, and a crash on the last of them
        # should not discard the ones already paid for -- which is exactly what happened once.
        if checkpoint is not None:
            checkpoint(_build(records, attempts))

    return _build(records, attempts)


def _build(records: Sequence[Attempt], attempts: int) -> ProbeReport:
    return ProbeReport(
        generated_at=datetime.now(timezone.utc),
        environment={
            "sciagentguard_version": __version__,
            "python_version": platform.python_version(),
        },
        parameters={
            "provider": PROVIDER,
            "model_id": DEFAULT_JUDGE_MODEL,
            "temperature": 0.0,
            "attempts": attempts,
            "cache": "bypassed; a cache would report perfect determinism whatever the model does",
        },
        artifact_reviewed="yield_estimate of the correct ATLAS analysis",
        attempts=tuple(records),
    )


def render_markdown(report: ProbeReport) -> str:
    latencies = [attempt.latency_ms for attempt in report.attempts if attempt.verdict != "ERROR"]
    verdicts = [attempt.verdict for attempt in report.attempts]
    lines = [
        "# Reviewer reproducibility",
        "",
        (
            f"`{report.parameters['model_id']}` asked the identical question "
            f"{report.parameters['attempts']} times at temperature "
            f"{report.parameters['temperature']}, with the cache bypassed."
        ),
        "",
        f"Artifact: {report.artifact_reviewed}.",
        "",
        "| Attempt | Verdict | Latency |",
        "| ---: | --- | ---: |",
    ]
    for attempt in report.attempts:
        lines.append(
            f"| {attempt.index} | `{attempt.verdict}` | {attempt.latency_ms / 1000.0:.0f} s |"
        )

    if report.answered == 0:
        summary = "No attempt returned an answer, so this probe measures nothing about determinism."
    elif report.distinct_answers > 1:
        summary = (
            f"**{report.distinct_answers} different answers to one identical question** across "
            f"{report.answered} answered attempts. The reviewer is not reproducible on this "
            "artifact, which is a property of the instrument rather than of any single verdict: "
            "a measurement that changes when nothing changed cannot be compared against anything."
        )
    else:
        summary = (
            f"One answer, `{verdicts[0]}`, across {report.answered} answered attempts. This run "
            "found no variation. It does not establish determinism -- it bounds how often "
            "variation occurs, and a larger sample would bound it more tightly."
        )

    lines.extend(
        [
            "",
            summary,
            "",
            (
                f"Latency across answered attempts: median "
                f"{median(latencies) / 1000.0:.0f} s, range "
                f"{min(latencies) / 1000.0:.0f}-{max(latencies) / 1000.0:.0f} s."
                if latencies
                else "No latency samples."
            ),
            "",
            (
                "One model, one artifact, one temperature. Nothing here generalises to other "
                "reviewers or other artifacts."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)

    def save(partial: ProbeReport) -> None:
        args.json_output.write_text(f"{partial.model_dump_json(indent=2)}\n", encoding="utf-8")
        args.markdown_output.write_text(render_markdown(partial), encoding="utf-8")

    report = run_probe(
        AtlasGamGamSource.official_wph125(args.input),
        attempts=args.attempts,
        checkpoint=save,
    )
    save(report)
    print(json.dumps({"distinct_answers": report.distinct_answers, "answered": report.answered}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
