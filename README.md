# SciAgentGuard

**Scientific analysis code can run without raising a single error and still be physically wrong.**
SciAgentGuard makes domain invariants executable at every checkpoint of an AI-agent analysis
workflow. It checks intermediate stages rather than final output because, in this repository, the
final output is provably not enough: three distinct faults produce a **byte-identical** final
artifact.

```console
$ uv run --extra hep python benchmarks/artifact_digests.py
correct                        bab9b172cc88
overlapping_control_window     bab9b172cc88
stale_cutflow                  bab9b172cc88
empty_selection                6db1cfcbbacb
luminosity_unit_slip           fb4355bf0a32
unrunnable                     (no final artifact)
wrong_output_schema            (no final artifact)

3 analyses share bab9b172cc88: correct, overlapping_control_window, stale_cutflow
No check reading only the final artifact can tell them apart.
```

Two of those three runs are wrong. A control region that overlaps the signal region, and a cutflow
that no longer describes the selection it reports, both leave the reported yield untouched. Any
checker restricted to the result — a schema validator, a plausibility heuristic, a language model
reading the final numbers — is blind to them by construction, not by accident.

The measurements below are on real [ATLAS Open Data](https://opendata.cern.ch/record/15006)
(113,765 simulated diphoton events, verified by size and Adler-32 checksum before use).

## Result

Seven analyses of the same dataset: one correct, four that execute cleanly but are scientifically
wrong, two that fail as code. Five checking strategies decide whether to accept each one.
**Bold** marks a wrong decision — a fault accepted, or correct work rejected.

| Analysis | No guard | Generic data checks | LLM judge | Final artifact only | **Runtime guard** |
| --- | :-: | :-: | :-: | :-: | :-: |
| `correct` | accept | accept | **reject** | accept | accept |
| `empty_selection` | **accept** | **accept** | reject | reject | reject |
| `luminosity_unit_slip` | **accept** | **accept** | reject | reject | reject |
| `overlapping_control_window` | **accept** | **accept** | reject | **accept** | reject |
| `stale_cutflow` | **accept** | **accept** | reject | **accept** | reject |
| `unrunnable` | reject | reject | reject | reject | reject |
| `wrong_output_schema` | reject | reject | reject | reject | reject |

Three things follow directly.

The runtime guard is the only column with no wrong decision. It accepts exactly the one correct
analysis and stops the other six.

The LLM judge rejects everything, including the correct analysis. It does not separate good work
from bad — it refuses all of it. Asking a language model to grade a final result is a common
proposal; as a gate, it is unusable here.

Checking only the final artifact catches the two faults that change the reported numbers and misses
the two that do not. Those are the byte-identical pair above.

Scaled up over 13 cases and 50 fault injections on the same sample:

| Strategy | Faults detected | False positives on valid runs |
| --- | ---: | ---: |
| No guard | 5 / 50 | 0 / 15 |
| Final artifact only | 20 / 50 | 0 / 15 |
| Runtime guard | **50 / 50** | **0 / 15** |

Runtime and final-artifact checking disagree on **30 of 50** fault runs. That number is a property
of where the checks sit, not of how many there are: it did not move when the contract set grew from
16 to 18.

The same structure reproduces in a second domain. Three contracts over a
[DeePTB](https://github.com/deepmodeling/DeePTB) Si64 Hamiltonian — real-space Hermiticity,
Γ-point overlap positive-definiteness, source identity — detect 15 of 15 injected materials faults
that an unguarded run accepts.

## How it works

A contract is an executable predicate over one checkpoint's artifacts, its declared units and
schema, and the provenance carried forward from the verified input. It returns structured evidence
either way, so a failure names the stage, the relation, and the numbers that broke it.

The ATLAS workflow runs four ordered checkpoints — `post_load`, `post_selection`, `post_histogram`,
`post_yield` — guarded by 18 contracts. Each stage passes forward only what the next one needs, and
that information loss is precisely what makes a late fault invisible at the end.

Two kinds of check do different work. A **consistency** contract asks whether an artifact agrees
with itself. A **derivation** contract compares what the analysis claims against facts the trusted
loader read from the source file and carried in provenance, so an analysis cannot satisfy it by
being self-consistent about a number it invented.

That distinction turned out to be sharper than "check early." A fault escapes a final check when no
trusted fact about it survives to the final stage — not merely because the check runs late. Adding
one derivation contract at the final stage moved `luminosity_unit_slip` from undetected to detected
without moving the check any earlier.

## Contract discovery

A fixed contract set only finds what its author already anticipated. SciAgentGuard closes that loop
by putting its own artifacts in front of a language model from a different vendor and asking what
would have to be verified — then refusing to take the answer on trust.

Every proposal passes three gates. It must be expressible as a deterministic check. It must survive
a run showing the current contracts **accepting** an artifact that violates it — decided by
executing contracts, never by opinion. And a maintainer must confirm it after recomputing the claim
independently.

One round produced 33 proposals across two conditions. Eighteen were already covered by existing
contracts. Thirteen could not be decided at the stage where they were raised. Two survived all three
gates and are now contracts.

One of them matters. The reviewer observed that nothing verified the background estimate against the
sidebands it came from. Doubling the background and recomputing the yield to stay self-consistent
left **all 16 contracts passing** while the physical result shifted by 2.0%.

The gate attrition is reported deliberately. Systems that synthesize invariants for programs rely on
an SMT solver, which decides novelty automatically and for free; no solver decides whether a
normalization is physically correct. Here the gate costs human attention, which makes the ratio of
proposals to confirmed blind spots a real cost rather than a footnote.

An earlier round is reported too, and it found nothing. It asked which quantities contradicted each
other, and every artifact shown was from a correct run — a question whose right answer is silence.
That round did not test its hypothesis.

## Cost

| | Contracts | LLM reviewer |
| --- | --- | --- |
| One full evaluation | 207 ms (18 contracts, 4 checkpoints) | 475 – 1894 s per verdict |
| Reproducibility | identical every run | 2 distinct answers to one question |
| Localization | names the contract and the stage | none |

Asked the same question five times at temperature 0 with caching bypassed, the reviewer answered
`INVALID`, `ERROR`, `VALID`, `ERROR`, `VALID`. A measurement that changes when nothing changed
cannot be compared against anything, which is why the reviewer proposes checks offline and never
sits in the runtime path.

## Reproduce

```console
uv sync --locked --extra dev --extra hep --extra materials
uv run pytest
```

514 tests, offline, no model provider and no network. The full ATLAS experiment needs the Open Data
file:

```console
mkdir -p .cache/atlas-open-data
curl --fail --location \
  --output .cache/atlas-open-data/mc_345318.WpH125J_Wincl_gamgam.GamGam.root \
  https://opendata.cern.ch/eos/opendata/atlas/OutreachDatasets/2020-08-19/GamGam/MC/mc_345318.WpH125J_Wincl_gamgam.GamGam.root
uv run --extra hep python benchmarks/atlas_gamgam_boundary_benchmark.py
```

Committed machine-readable evidence is regenerated, not edited. The test suite fails when a result
file was produced by a different contract set than the code declares, so stale numbers cannot
survive a change.

## What this does not show

The faults in every experiment here were injected by the maintainer. The two runs in which a real
model wrote the analysis both produced correct code, so **no semantic fault committed by a real
model has been observed yet**. Until that changes, this is a fault-injection benchmark with a
contract-discovery loop, not evidence about how models fail in practice.

Contract discovery has one round of valid data. Whether it runs dry after a few rounds is unmeasured.

The materials domain has three contracts and no discovery round, so cross-domain support is
structural rather than deep.

The invariants checked here are necessary conditions, not sufficient ones. Passing every contract
does not make an analysis correct.

## Repository

| Path | Contents |
| --- | --- |
| `src/sciagentguard/core` | contracts, contexts, results, violation reports |
| `src/sciagentguard/runtime` | checkpoint and workflow executors, bounded repair |
| `src/sciagentguard/packs/hep` | ATLAS diphoton analysis, contracts, fault injectors |
| `src/sciagentguard/packs/materials` | DeePTB Hamiltonian and overlap contracts |
| `src/sciagentguard/adapters` | verified input boundaries and the agent harness |
| `src/sciagentguard/discovery` | review rounds, novelty gates, candidate records |
| `benchmarks/` | experiments and their machine-readable results |

## Installation

Python 3.10 or newer. The base package depends only on Pydantic; the ROOT and HDF5 boundaries are
optional extras.

```console
python -m pip install -e .            # base
python -m pip install -e '.[hep]'     # ATLAS ROOT boundary
python -m pip install -e '.[materials]'
```

On macOS, an environment created inside iCloud Drive can inherit the `hidden` filesystem flag. If
Python reports that `sciagentguard` is missing after a successful sync, set
`UV_PROJECT_ENVIRONMENT=venv` and sync again.

## Provenance and integrity

The ATLAS sample is CERN Open Data under CC0; the DeePTB sample is pinned by checksum. Neither is
redistributed here. Results are simulated data processed through a closed-form estimate — this is
not a physics measurement and makes no claim about a Higgs signal.

Every interface is the maintainer's responsibility, and every claim above rests on committed,
regenerable evidence.

Licensed under Apache-2.0.
