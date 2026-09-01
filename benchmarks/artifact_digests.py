"""Fingerprint the final artifact each analysis produces.

The point of this script is a negative result. Several of the faulty analyses in this repository
end at a final artifact that is byte-identical to the one a correct analysis produces, so any check
restricted to that artifact cannot distinguish them. Printing the digests is the shortest honest way
to show it: two runs with the same fingerprint cannot be told apart by anything reading only the
fingerprinted object.

Digests cover the final `yield_estimate` artifact serialised with sorted keys, which is the object a
final-artifact-only reviewer would be handed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from sciagentguard.adapters.agent import ATLAS_AGENT_TASK, CodeSandbox, ScriptedAgent
from sciagentguard.adapters.agent._scripts import SCRIPTS

DEFAULT_INPUT = Path(".cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root")
FINAL_ARTIFACT = "yield_estimate"
DIGEST_CHARACTERS = 12


def final_digest(script_id: str, input_path: Path, sandbox: CodeSandbox) -> str | None:
    """Return the fingerprint of one analysis's final artifact, or None when it produced none."""

    proposal = ScriptedAgent(script_id=script_id).propose(ATLAS_AGENT_TASK, attempt_id="digest")
    result = sandbox.run(proposal.code, input_path=input_path)
    artifacts = result.artifacts
    if not isinstance(artifacts, dict) or FINAL_ARTIFACT not in artifacts:
        return None
    serialised = json.dumps(artifacts[FINAL_ARTIFACT], sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:DIGEST_CHARACTERS]


def collect(input_path: Path) -> dict[str, str | None]:
    sandbox = CodeSandbox()
    return {script_id: final_digest(script_id, input_path, sandbox) for script_id in SCRIPTS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    digests = collect(args.input)

    shared: dict[str, list[str]] = {}
    for script_id, digest in digests.items():
        if digest is None:
            continue
        shared.setdefault(digest, []).append(script_id)

    # Colliding analyses first, because the collision is the finding.
    order = sorted(
        digests,
        key=lambda script_id: (
            -len(shared.get(digests[script_id] or "", [])),
            digests[script_id] or "~",
            script_id,
        ),
    )
    for script_id in order:
        digest = digests[script_id]
        print(f"{script_id:<30} {digest if digest else '(no final artifact)'}")

    collisions = {digest: ids for digest, ids in shared.items() if len(ids) > 1}
    for digest, ids in collisions.items():
        print(f"\n{len(ids)} analyses share {digest}: {', '.join(sorted(ids))}")
        print("No check reading only the final artifact can tell them apart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
