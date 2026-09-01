"""Contracts and deterministic fault fixtures for the synthetic HEP workflow."""

from sciagentguard.packs.hep.atlas_analysis import (
    build_mass_histogram,
    estimate_yield,
    select_diphoton_events,
)
from sciagentguard.packs.hep.atlas_faults import (
    AtlasMissingEventProvenanceInjector,
    AtlasMissingPhotonMomentumInjector,
    AtlasNormalizationScaleDriftInjector,
    AtlasPhotonCountMismatchInjector,
    AtlasPhotonScaleGapInjector,
    AtlasRegionOverlapInjector,
    AtlasSourceIdentityDriftInjector,
    AtlasWeightScaleGapInjector,
)
from sciagentguard.packs.hep.atlas_open_data import (
    AtlasCutflowMonotonicContract,
    AtlasDiphotonPreselectionContract,
    AtlasHistogramClosureContract,
    AtlasRegionDisjointContract,
    AtlasSourceIdentityContract,
    AtlasYieldShapeContract,
)
from sciagentguard.packs.hep.contracts import (
    DeclaredEventProvenanceContract,
    DisjointEventSplitsContract,
    FiniteWeightsContract,
    JetPtRangeContract,
    NonemptySelectionContract,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
    YieldNormalizationContract,
)
from sciagentguard.packs.hep.fixtures import (
    make_synthetic_hep_context,
    make_synthetic_normalization_context,
    make_synthetic_selection_context,
    make_synthetic_split_context,
)
from sciagentguard.packs.hep.injectors import (
    EmptySelectionInjector,
    MissingBranchInjector,
    NonfiniteWeightsInjector,
    SplitLeakageInjector,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    WrongNormalizationInjector,
    ZeroWeightsInjector,
)
from sciagentguard.packs.hep.repair import SyntheticHEPRepairPolicy
from sciagentguard.packs.hep.workflow import SyntheticHEPWorkflow

__all__ = [
    "AtlasCutflowMonotonicContract",
    "AtlasDiphotonPreselectionContract",
    "AtlasHistogramClosureContract",
    "AtlasMissingEventProvenanceInjector",
    "AtlasMissingPhotonMomentumInjector",
    "AtlasNormalizationScaleDriftInjector",
    "AtlasPhotonCountMismatchInjector",
    "AtlasPhotonScaleGapInjector",
    "AtlasRegionDisjointContract",
    "AtlasRegionOverlapInjector",
    "AtlasSourceIdentityContract",
    "AtlasSourceIdentityDriftInjector",
    "AtlasWeightScaleGapInjector",
    "AtlasYieldShapeContract",
    "DeclaredEventProvenanceContract",
    "DisjointEventSplitsContract",
    "EmptySelectionInjector",
    "FiniteWeightsContract",
    "JetPtRangeContract",
    "MissingBranchInjector",
    "NonemptySelectionContract",
    "NonfiniteWeightsInjector",
    "NonzeroWeightSupportContract",
    "RequiredBranchesContract",
    "SplitLeakageInjector",
    "SyntheticHEPRepairPolicy",
    "SyntheticHEPWorkflow",
    "UndeclaredSyntheticDataInjector",
    "UnitScaleErrorInjector",
    "WrongNormalizationInjector",
    "YieldNormalizationContract",
    "ZeroWeightsInjector",
    "build_mass_histogram",
    "estimate_yield",
    "make_synthetic_hep_context",
    "make_synthetic_normalization_context",
    "make_synthetic_selection_context",
    "make_synthetic_split_context",
    "select_diphoton_events",
]
