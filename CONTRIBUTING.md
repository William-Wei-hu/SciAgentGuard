# Contributing

SciAgentGuard is an early prototype. Contributions should strengthen a current milestone rather
than add speculative framework layers.

## Development setup

Install [uv](https://astral.sh/uv), then create the locked development environment:

```console
uv sync --locked --extra dev --extra hep --extra materials
```

Run the same checks used in continuous integration:

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Use `uv run ruff format .` to format deliberate edits before rerunning the check.

The offline synthetic workflow can be exercised without API keys:

```console
uv run python examples/hep_guarded_demo.py --fault none
```

Regenerate the synthetic benchmark evidence and its result table together:

```console
uv run python benchmarks/hep_fixture_benchmark.py
```

Commit the generated JSON evidence. The Markdown table is a local rendering under the ignored
`.cache/reports/` directory; regenerate it from the same run and do not edit either output by hand.

The ATLAS smoke test uses a local file that is deliberately excluded from Git. Download the fixed
source with the command in the README and regenerate its trace. Never commit ROOT inputs, cache
directories, local paths, or credentials.

Regenerate the real-file boundary comparison and its Markdown table together with:

```console
uv run --locked --extra hep python benchmarks/atlas_gamgam_boundary_benchmark.py
```

Commit the generated JSON output; its Markdown rendering remains local under `.cache/reports/`. The full
benchmark intentionally requires the ignored official file and is not executed by network-isolated
CI; CI validates the same report logic against a temporary miniature ROOT source.

The DeePTB smoke test likewise keeps its three upstream sample files in the ignored cache. Download
the upstream sample pinned by commit and SHA-256 in `src/sciagentguard/adapters/deeptb_si64.py` into
`.cache/deeptb-si64`, then regenerate the safe trace. CI creates miniature HDF5 block files and
never depends on GitHub network availability.

Regenerate the DeePTB boundary comparison and its Markdown table together with:

```console
uv run --locked --extra materials python benchmarks/deeptb_si64_boundary_benchmark.py
```

Commit the generated JSON output; its Markdown rendering remains local under `.cache/reports/`. The full
comparison uses the ignored official sample; CI exercises the same faults, metrics, rendering, and
privacy boundaries with miniature HDF5 files.

## Change expectations

- Work on one milestone at a time and keep the smallest interface that satisfies its acceptance
  criteria.
- Add or update tests for every behavioral change.
- Do not swallow unexpected exceptions or treat a repair as successful before revalidation.
- Keep scientific logic in its domain pack once domain packs exist.
- Do not add an integration, dependency, or abstraction without a concrete use in the active
  milestone.
- Record non-trivial design changes in the pull request or commit description.

## Scientific claims and provenance

Every contract must document the invariant it checks, required inputs, assumptions, and known blind
spots. Fixtures and benchmark outputs must state whether they are synthetic, preserve deterministic
seeds, and include the command that generated them. Do not hand-edit generated evidence or imply
that fixture performance establishes real-world validity.

External code and data require a license review, attribution, and documented provenance.

## Checkpoints and rollback

Treat every complete, reviewable change as a recovery checkpoint:

1. keep the change focused on one purpose;
2. run the relevant tests, lint, formatting, and type checks;
3. create a descriptive commit;
4. push the commit to GitHub before starting the next logical change.

Tag completed milestones and releases so they remain easy to identify. Do not force-push or delete
the `main` branch. To investigate or recover an earlier state, create a branch from the relevant
commit or tag instead of rewriting shared history:

```console
git switch -c recovery/<name> <commit-or-tag>
```

Uncommitted work is not backed up remotely. Keep commits small enough to review and revert, but do
not create noisy commits for broken or untested intermediate states.
