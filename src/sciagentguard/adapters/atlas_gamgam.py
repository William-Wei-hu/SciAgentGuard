"""ATLAS Open Data Gamma-Gamma ROOT input boundary."""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from sciagentguard.core import ContractContext, ScientificContract, SemanticFaultInjector
from sciagentguard.packs.hep.atlas_analysis import (
    ATLAS_SELECTION_ID,
    build_mass_histogram,
    estimate_yield,
    select_diphoton_events,
)
from sciagentguard.packs.hep.atlas_open_data import (
    AtlasBackgroundEstimateContract,
    AtlasCutflowMonotonicContract,
    AtlasDiphotonPreselectionContract,
    AtlasHistogramClosureContract,
    AtlasNormalizationProvenanceContract,
    AtlasRegionCoverageContract,
    AtlasRegionDefinitionContract,
    AtlasRegionDisjointContract,
    AtlasSourceConstantsContract,
    AtlasSourceIdentityContract,
    AtlasWeightProvenanceContract,
    AtlasYieldClosureContract,
    AtlasYieldShapeContract,
)
from sciagentguard.packs.hep.contracts import (
    DeclaredEventProvenanceContract,
    FiniteWeightsContract,
    NonemptySelectionContract,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
)
from sciagentguard.runtime import WorkflowCheckpoint

_TREE_NAME = "mini"
_RAW_BRANCHES = (
    "runNumber",
    "eventNumber",
    "channelNumber",
    "mcWeight",
    "photon_n",
    "photon_pt",
    "photon_eta",
    "photon_phi",
    "photon_E",
    "XSection",
    "SumWeights",
)
_MEV_TO_GEV = 1e-3
_ADLER32_PATTERN = re.compile(r"[0-9a-f]{8}")
_CONTEXT_BRANCHES = (
    "run_number",
    "event_number",
    "channel_number",
    "weight",
    "photon_count",
    "photon_pt_gev",
    "photon_eta",
    "photon_phi",
    "photon_e_gev",
    "cross_section_pb",
    "generated_weight_sum",
)
# The ATLAS 2020 Open Data release documents an integrated luminosity of 10 fb^-1. It is a
# configured local assumption of this workflow, not a value derived from the file.
_LUMINOSITY_PB_INVERSE = 10_064.0
# Above this many distinct weights, the source facts carry bounds instead of the full set.
_DISTINCT_WEIGHT_LIMIT = 32


@dataclass(frozen=True, slots=True)
class AtlasGamGamSource:
    """A local Gamma-Gamma ROOT file and its declared public provenance."""

    path: Path
    size_bytes: int
    adler32: str
    source_type: str
    record_id: str
    doi: str
    file_name: str
    generator: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")

        checksum = self.adler32.lower()
        if _ADLER32_PATTERN.fullmatch(checksum) is None:
            raise ValueError("adler32 must contain exactly eight hexadecimal characters")
        object.__setattr__(self, "adler32", checksum)

        for name in ("source_type", "record_id", "doi", "file_name"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
        if Path(self.file_name).name != self.file_name:
            raise ValueError("file_name must not contain a directory")
        if self.source_type not in {"simulation", "synthetic"}:
            raise ValueError("source_type must be 'simulation' or 'synthetic'")

        if self.source_type == "synthetic":
            if self.generator is None or not self.generator.strip():
                raise ValueError("synthetic sources must declare a generator")
        elif self.generator is not None:
            raise ValueError("simulation sources must not declare a synthetic generator")

    @classmethod
    def official_wph125(cls, path: Path) -> AtlasGamGamSource:
        """Describe the fixed WpH125 sample from CERN Open Data record atlas-15006."""

        return cls(
            path=path,
            size_bytes=29_757_932,
            adler32="5ac6bca3",
            source_type="simulation",
            record_id="atlas-15006",
            doi="10.7483/OPENDATA.ATLAS.B5BJ.3SGS",
            file_name="mc_345318.WpH125J_Wincl_gamgam.GamGam.root",
        )


@dataclass(frozen=True, slots=True)
class AtlasGamGamOpenDataAdapter:
    """Translate one declared ATLAS Gamma-Gamma ROOT file into guarded checkpoints.

    ``entry_start`` and ``entry_stop`` select a contiguous event range of the same verified file.
    Distinct ranges are distinct valid inputs of one source; they are not independent samples.
    """

    source: AtlasGamGamSource
    entry_start: int = 0
    entry_stop: int | None = None

    def __post_init__(self) -> None:
        if self.entry_start < 0:
            raise ValueError("entry_start must be nonnegative")
        if self.entry_stop is not None and self.entry_stop <= self.entry_start:
            raise ValueError("entry_stop must be greater than entry_start")

    def source_event_count(self) -> int:
        """Return the number of entries in the declared ROOT tree without reading branches."""

        return _read_entry_count(self.source.path)

    def load_contracts(self) -> tuple[ScientificContract, ...]:
        """Return the contracts guarding the post-load boundary."""

        return (
            RequiredBranchesContract(_CONTEXT_BRANCHES),
            FiniteWeightsContract(),
            NonzeroWeightSupportContract(),
            AtlasDiphotonPreselectionContract(),
            DeclaredEventProvenanceContract(),
            AtlasSourceIdentityContract(
                expected_source_type=self.source.source_type,
                expected_record_id=self.source.record_id,
                expected_doi=self.source.doi,
                expected_file_name=self.source.file_name,
                expected_checksum=f"adler32:{self.source.adler32}",
            ),
        )

    def stage_contracts(self) -> tuple[tuple[ScientificContract, ...], ...]:
        """Return the contracts of each ordered checkpoint, from load to yield."""

        return (
            self.load_contracts(),
            (
                NonemptySelectionContract(),
                AtlasCutflowMonotonicContract(),
                AtlasRegionDisjointContract(),
                AtlasRegionCoverageContract(),
                AtlasRegionDefinitionContract(),
                AtlasWeightProvenanceContract(),
                AtlasSourceConstantsContract(),
            ),
            (AtlasHistogramClosureContract(),),
            (
                AtlasYieldShapeContract(),
                AtlasYieldClosureContract(),
                AtlasBackgroundEstimateContract(),
                AtlasNormalizationProvenanceContract(),
            ),
        )

    def checkpoint(
        self,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
    ) -> WorkflowCheckpoint:
        """Build the guarded post-load boundary for this source."""

        return WorkflowCheckpoint(
            step=lambda: self.load_context(
                workflow_id=workflow_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ),
            contracts=self.load_contracts(),
        )

    def contexts(
        self,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
        fault: SemanticFaultInjector | None = None,
        injection_stage: str | None = None,
    ) -> tuple[ContractContext, ...]:
        """Derive the four ordered checkpoint contexts, optionally injecting one fault.

        A fault is injected into the context of ``injection_stage``, and every later stage is
        then re-derived from the injected context. A fault therefore propagates downstream
        exactly as a real upstream mistake would.
        """

        return self.derive_contexts(
            self.load_context(
                workflow_id=workflow_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ),
            fault=fault,
            injection_stage=injection_stage,
        )

    def derive_contexts(
        self,
        loaded: ContractContext,
        *,
        fault: SemanticFaultInjector | None = None,
        injection_stage: str | None = None,
    ) -> tuple[ContractContext, ...]:
        """Derive the ordered checkpoint contexts from an already-loaded post-load context.

        Reusing one verified load keeps a repeated experiment from re-reading the source file
        for every run while still deriving each downstream stage from the stage before it.
        """

        if (fault is None) != (injection_stage is None):
            raise ValueError("a fault and an injection stage must be supplied together")

        def maybe_inject(context: ContractContext) -> ContractContext:
            if fault is None or injection_stage != context.stage:
                return context
            return fault.inject(context)

        first = maybe_inject(loaded)
        selected = maybe_inject(select_diphoton_events(first))
        histogram = maybe_inject(build_mass_histogram(selected))
        estimate = maybe_inject(estimate_yield(histogram))
        return (first, selected, histogram, estimate)

    def chained_checkpoints(
        self,
        loaded_factory: Callable[[], ContractContext],
        *,
        fault: SemanticFaultInjector | None = None,
        injection_stage: str | None = None,
    ) -> tuple[WorkflowCheckpoint, ...]:
        """Build checkpoints whose steps derive lazily, one stage at a time.

        Each step runs only when its checkpoint is reached, so a guarded run that blocks early
        never executes the later stages. A workflow whose upstream artifact was destroyed will
        therefore raise from the first downstream step it actually reaches, exactly as an
        unguarded workflow would.
        """

        if (fault is None) != (injection_stage is None):
            raise ValueError("a fault and an injection stage must be supplied together")

        chain = _StageChain(loaded_factory, fault, injection_stage)
        steps: tuple[Callable[[], ContractContext], ...] = (
            chain.load,
            chain.select,
            chain.histogram,
            chain.estimate,
        )
        return tuple(
            WorkflowCheckpoint(step=step, contracts=contracts)
            for step, contracts in zip(steps, self.stage_contracts(), strict=True)
        )

    def checkpoints_for(
        self, contexts: Sequence[ContractContext]
    ) -> tuple[WorkflowCheckpoint, ...]:
        """Pair already-derived contexts with the contracts guarding each stage."""

        return tuple(
            WorkflowCheckpoint(step=_constant_step(context), contracts=contracts)
            for context, contracts in zip(contexts, self.stage_contracts(), strict=True)
        )

    def checkpoints(
        self,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
        fault: SemanticFaultInjector | None = None,
        injection_stage: str | None = None,
    ) -> tuple[WorkflowCheckpoint, ...]:
        """Build the four ordered guarded checkpoints for this source."""

        return self.chained_checkpoints(
            lambda: self.load_context(
                workflow_id=workflow_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ),
            fault=fault,
            injection_stage=injection_stage,
        )

    def load_context(
        self,
        *,
        workflow_id: str,
        run_id: str,
        attempt_id: str,
    ) -> ContractContext:
        """Verify and load the source without exposing its local path in the context."""

        _verify_source(self.source)
        columns = _read_event_columns(
            self.source.path,
            entry_start=self.entry_start,
            entry_stop=self.entry_stop,
        )
        event_provenance: dict[str, JsonValue] = {
            "source_type": self.source.source_type,
            "experiment": "ATLAS",
            "record_id": self.source.record_id,
            "doi": self.source.doi,
            "file_name": self.source.file_name,
            "checksum": f"adler32:{self.source.adler32}",
            # Trusted facts about the source, read by this loader rather than declared by whoever
            # writes the analysis. Downstream contracts compare an analysis's claims against these
            # instead of only checking that its own numbers agree with each other. Five aggregate
            # values: no per-event data and no local paths reach a trace.
            "source_facts": _source_facts(columns),
        }
        if self.source.generator is not None:
            event_provenance["generator"] = self.source.generator
        schema: dict[str, JsonValue] = {
            "events": {
                "tree": _TREE_NAME,
                "branches": {
                    "run_number": "runNumber",
                    "event_number": "eventNumber",
                    "channel_number": "channelNumber",
                    "weight": "mcWeight",
                    "photon_count": "photon_n",
                    "photon_pt_gev": "photon_pt",
                    "photon_eta": "photon_eta",
                    "photon_phi": "photon_phi",
                    "photon_e_gev": "photon_E",
                    "cross_section_pb": "XSection",
                    "generated_weight_sum": "SumWeights",
                },
            }
        }
        config: dict[str, JsonValue] = {
            "atlas_gamgam": {
                "minimum_photons": 2,
                "raw_photon_pt_unit": "MeV",
                "weight_definition": "mcWeight",
            },
            "selection": {
                "selection_id": ATLAS_SELECTION_ID,
                "minimum_selected": 1,
            },
            "atlas_selection": {
                "leading_photon_pt_min_gev": 40.0,
                "subleading_photon_pt_min_gev": 30.0,
            },
            "atlas_histogram": {
                "bin_count": 30,
                "range_low_gev": 100.0,
                "range_high_gev": 160.0,
                # The source stores mcWeight in single precision, so an implementation that
                # accumulates in float32 -- the most natural thing to write against uproot's
                # default dtype -- carries a relative error near the float32 epsilon of 1.2e-7.
                # A tighter tolerance would reject correct analyses. This still leaves six orders
                # of magnitude of margin over the scale errors the closure check exists to catch.
                "closure_relative_tolerance": 1e-6,
            },
            "atlas_yield": {
                "signal_window_gev": [120.0, 130.0],
                "expected_peak_window_gev": [115.0, 135.0],
            },
            "atlas_normalization": {
                "luminosity_pb_inverse": _LUMINOSITY_PB_INVERSE,
            },
        }

        return ContractContext(
            workflow_id=workflow_id,
            run_id=run_id,
            attempt_id=attempt_id,
            stage="post_load",
            artifacts={"events": columns},
            schema=schema,
            units={
                "weight": "dimensionless",
                "photon_pt_gev": "GeV",
                "photon_e_gev": "GeV",
                "photon_eta": "dimensionless",
                "photon_phi": "rad",
                "cross_section_pb": "pb",
                "generated_weight_sum": "dimensionless",
            },
            provenance={"events": event_provenance},
            config=config,
        )


class _StageChain:
    """Derive one checkpoint context at a time, injecting a fault at its declared stage."""

    def __init__(
        self,
        loaded_factory: Callable[[], ContractContext],
        fault: SemanticFaultInjector | None,
        injection_stage: str | None,
    ) -> None:
        self._loaded_factory = loaded_factory
        self._fault = fault
        self._injection_stage = injection_stage
        self._current: ContractContext | None = None

    def load(self) -> ContractContext:
        return self._record(self._loaded_factory())

    def select(self) -> ContractContext:
        return self._record(select_diphoton_events(self._require_current("post_load")))

    def histogram(self) -> ContractContext:
        return self._record(build_mass_histogram(self._require_current("post_selection")))

    def estimate(self) -> ContractContext:
        return self._record(estimate_yield(self._require_current("post_histogram")))

    def _record(self, context: ContractContext) -> ContractContext:
        if self._fault is not None and self._injection_stage == context.stage:
            context = self._fault.inject(context)
        self._current = context
        return context

    def _require_current(self, expected_stage: str) -> ContractContext:
        if self._current is None:
            raise ValueError(f"the {expected_stage!r} checkpoint has not run yet")
        return self._current


def _read_entry_count(path: Path) -> int:
    _, uproot = _hep_modules()
    with uproot.open(path) as root_file:
        if _TREE_NAME not in root_file:
            raise ValueError(f"ROOT source does not contain the {_TREE_NAME!r} tree")
        count = root_file[_TREE_NAME].num_entries
    if not isinstance(count, int):
        raise ValueError("the ROOT tree did not report an integer entry count")
    return count


def _source_facts(columns: Mapping[str, Sequence[object]]) -> dict[str, JsonValue]:
    """Summarize what the verified file actually contains, for contracts to check claims against."""

    weights = [float(value) for value in columns["weight"] if isinstance(value, (int, float))]
    if not weights:
        raise ValueError("the source produced no event weights")
    cross_sections = {float(value) for value in columns["cross_section_pb"]}  # type: ignore[arg-type]
    generated = {float(value) for value in columns["generated_weight_sum"]}  # type: ignore[arg-type]
    if len(cross_sections) != 1 or len(generated) != 1:
        raise ValueError("the source cross section and generated weight sum must be constant")
    magnitudes = [abs(value) for value in weights]
    distinct = sorted({value for value in weights})
    facts: dict[str, JsonValue] = {
        "event_count": len(weights),
        "weight_min": min(weights),
        "weight_max": max(weights),
        # Magnitude bounds, not the signed range. A Monte Carlo sample carries weights of both
        # signs, so the signed range straddles zero and would admit a weight scaled to near
        # nothing. What a rescaling cannot survive is the magnitude.
        "weight_abs_min": min(magnitudes),
        "weight_abs_max": max(magnitudes),
        "distinct_weight_count": len(distinct),
        "cross_section_pb": cross_sections.pop(),
        "generated_weight_sum": generated.pop(),
    }
    if len(distinct) <= _DISTINCT_WEIGHT_LIMIT:
        # Few enough to carry in full, which makes the downstream check exact rather than a bound.
        facts["distinct_weights"] = list(distinct)
    return facts


def _constant_step(context: ContractContext) -> Callable[[], ContractContext]:
    """Return a step callable that yields one already-derived checkpoint context."""

    def step() -> ContractContext:
        return context

    return step


def _verify_source(source: AtlasGamGamSource) -> None:
    if not source.path.is_file():
        raise ValueError("the declared ATLAS Gamma-Gamma source is not a file")
    actual_size = source.path.stat().st_size
    if actual_size != source.size_bytes:
        raise ValueError(
            f"source size mismatch: expected {source.size_bytes} bytes, found {actual_size}"
        )

    checksum = 1
    with source.path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum = zlib.adler32(chunk, checksum)
    actual_checksum = f"{checksum & 0xFFFFFFFF:08x}"
    if actual_checksum != source.adler32:
        raise ValueError(
            f"source Adler-32 mismatch: expected {source.adler32}, found {actual_checksum}"
        )


def _hep_modules() -> tuple[Any, Any]:
    """Import the optional ROOT stack, or explain which extra is missing."""

    try:
        import awkward as ak
        import uproot
    except ModuleNotFoundError as error:
        if error.name in {"awkward", "numpy", "uproot"}:
            raise ModuleNotFoundError(
                "ATLAS ROOT support requires the 'hep' extra: pip install 'sciagentguard[hep]'"
            ) from error
        raise
    return ak, uproot


def _read_event_columns(
    path: Path,
    *,
    entry_start: int = 0,
    entry_stop: int | None = None,
) -> dict[str, tuple[object, ...]]:
    ak, uproot = _hep_modules()

    with uproot.open(path) as root_file:
        if _TREE_NAME not in root_file:
            raise ValueError(f"ROOT source does not contain the {_TREE_NAME!r} tree")
        tree = root_file[_TREE_NAME]
        available_branches = set(tree.keys())
        missing_branches = tuple(
            branch for branch in _RAW_BRANCHES if branch not in available_branches
        )
        if missing_branches:
            missing = ", ".join(missing_branches)
            raise ValueError(f"ROOT tree {_TREE_NAME!r} is missing required branches: {missing}")
        if entry_start >= tree.num_entries:
            raise ValueError("entry_start is beyond the end of the declared ROOT tree")
        arrays = tree.arrays(
            _RAW_BRANCHES,
            library="ak",
            entry_start=entry_start,
            entry_stop=entry_stop,
        )

    run_numbers = _integer_column(ak.to_list(arrays["runNumber"]), "runNumber")
    event_numbers = _integer_column(ak.to_list(arrays["eventNumber"]), "eventNumber")
    channel_numbers = _integer_column(ak.to_list(arrays["channelNumber"]), "channelNumber")
    weights = _numeric_column(ak.to_list(arrays["mcWeight"]), "mcWeight")
    photon_counts = _integer_column(ak.to_list(arrays["photon_n"]), "photon_n")
    photon_pt_gev = _photon_row_column(ak.to_list(arrays["photon_pt"]), "photon_pt", _MEV_TO_GEV)
    photon_eta = _photon_row_column(ak.to_list(arrays["photon_eta"]), "photon_eta", 1.0)
    photon_phi = _photon_row_column(ak.to_list(arrays["photon_phi"]), "photon_phi", 1.0)
    photon_e_gev = _photon_row_column(ak.to_list(arrays["photon_E"]), "photon_E", _MEV_TO_GEV)
    cross_section_pb = _numeric_column(ak.to_list(arrays["XSection"]), "XSection")
    generated_weight_sum = _numeric_column(ak.to_list(arrays["SumWeights"]), "SumWeights")

    columns: dict[str, tuple[object, ...]] = {
        "run_number": run_numbers,
        "event_number": event_numbers,
        "channel_number": channel_numbers,
        "weight": weights,
        "photon_count": photon_counts,
        "photon_pt_gev": photon_pt_gev,
        "photon_eta": photon_eta,
        "photon_phi": photon_phi,
        "photon_e_gev": photon_e_gev,
        "cross_section_pb": cross_section_pb,
        "generated_weight_sum": generated_weight_sum,
    }
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError("ROOT branches do not contain the same number of events")
    return columns


def _column_values(raw: object, branch: str) -> Sequence[object]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"ROOT branch {branch!r} did not produce a sequence")
    return raw


def _integer_column(raw: object, branch: str) -> tuple[int, ...]:
    values: list[int] = []
    for index, value in enumerate(_column_values(raw, branch)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"ROOT branch {branch!r} contains a non-integer at index {index}")
        values.append(value)
    return tuple(values)


def _numeric_column(raw: object, branch: str) -> tuple[float, ...]:
    values: list[float] = []
    for index, value in enumerate(_column_values(raw, branch)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"ROOT branch {branch!r} contains a non-numeric value at index {index}"
            )
        values.append(float(value))
    return tuple(values)


def _photon_row_column(raw: object, branch: str, scale: float) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for index, row in enumerate(_column_values(raw, branch)):
        values = _numeric_column(row, f"{branch}[{index}]")
        rows.append(tuple(value * scale for value in values))
    return tuple(rows)


__all__ = ["AtlasGamGamOpenDataAdapter", "AtlasGamGamSource"]
