"""Deterministic synthetic data used by the minimal HEP workflow."""

from __future__ import annotations

from types import MappingProxyType

from sciagentguard.core import ContractContext

HEP_STAGE = "post_load"
SELECTION_STAGE = "post_selection"
SPLIT_STAGE = "post_split"
NORMALIZATION_STAGE = "post_normalization"


def make_synthetic_hep_context(
    *,
    workflow_id: str = "hep-synthetic-demo",
    run_id: str = "run-001",
    attempt_id: str = "attempt-0",
) -> ContractContext:
    """Build a small labeled fixture; it is not experimental collision data."""

    events = MappingProxyType(
        {
            "event_id": (1001, 1002, 1003, 1004, 1005, 1006),
            "jet_pt_gev": (42.0, 78.5, 33.1, 120.4, 55.0, 91.2),
            "weight": (1.0, 0.8, -0.2, 1.1, 0.5, 0.9),
        }
    )
    return ContractContext(
        workflow_id=workflow_id,
        run_id=run_id,
        attempt_id=attempt_id,
        stage=HEP_STAGE,
        artifacts={"events": events},
        schema={
            "events": {
                "layout": "columnar",
                "branches": ["event_id", "jet_pt_gev", "weight"],
            }
        },
        units={"jet_pt_gev": "GeV"},
        provenance={
            "events": {
                "source_type": "synthetic",
                "generator": "sciagentguard.packs.hep.fixtures",
            }
        },
        config={
            "jet_pt_gev": {
                "expected_unit": "GeV",
                "valid_range": [0.0, 500.0],
            }
        },
    )


def make_synthetic_selection_context(
    *,
    workflow_id: str = "hep-synthetic-demo",
    run_id: str = "run-001",
    attempt_id: str = "attempt-0",
) -> ContractContext:
    """Build the declared result of the fixture's jet-momentum selection."""

    selection = MappingProxyType(
        {
            "input_event_ids": (1001, 1002, 1003, 1004, 1005, 1006),
            "selected_event_ids": (1002, 1004, 1005, 1006),
        }
    )
    return ContractContext(
        workflow_id=workflow_id,
        run_id=run_id,
        attempt_id=attempt_id,
        stage=SELECTION_STAGE,
        artifacts={"selection": selection},
        schema={
            "selection": {
                "fields": ["input_event_ids", "selected_event_ids"],
            }
        },
        provenance={
            "selection": {
                "source_type": "synthetic",
                "generator": "sciagentguard.packs.hep.fixtures",
            }
        },
        config={
            "selection": {
                "selection_id": "jet_pt_gev_gt_50",
                "minimum_selected": 1,
            }
        },
    )


def make_synthetic_split_context(
    *,
    workflow_id: str = "hep-synthetic-demo",
    run_id: str = "run-001",
    attempt_id: str = "attempt-0",
) -> ContractContext:
    """Build disjoint train and test splits from the synthetic event identifiers."""

    splits = MappingProxyType(
        {
            "train": (1001, 1002, 1003, 1004),
            "test": (1005, 1006),
        }
    )
    return ContractContext(
        workflow_id=workflow_id,
        run_id=run_id,
        attempt_id=attempt_id,
        stage=SPLIT_STAGE,
        artifacts={"splits": splits},
        schema={"splits": {"names": ["train", "test"]}},
        provenance={
            "splits": {
                "source_type": "synthetic",
                "generator": "sciagentguard.packs.hep.fixtures",
            }
        },
        config={"split": {"left": "train", "right": "test"}},
    )


def make_synthetic_normalization_context(
    *,
    workflow_id: str = "hep-synthetic-demo",
    run_id: str = "run-001",
    attempt_id: str = "attempt-0",
) -> ContractContext:
    """Build a fixture-local normalized yield with explicit inputs and assumptions."""

    normalization = MappingProxyType(
        {
            "selected_weight_sum": 4.1,
            "generated_weight_sum": 8.2,
            "cross_section_pb": 2.0,
            "luminosity_pb_inverse": 100.0,
            "observed_yield": 100.0,
        }
    )
    return ContractContext(
        workflow_id=workflow_id,
        run_id=run_id,
        attempt_id=attempt_id,
        stage=NORMALIZATION_STAGE,
        artifacts={"normalization": normalization},
        schema={
            "normalization": {
                "fields": [
                    "selected_weight_sum",
                    "generated_weight_sum",
                    "cross_section_pb",
                    "luminosity_pb_inverse",
                    "observed_yield",
                ]
            }
        },
        units={
            "cross_section_pb": "pb",
            "luminosity_pb_inverse": "pb^-1",
            "observed_yield": "events",
        },
        provenance={
            "normalization": {
                "source_type": "synthetic",
                "generator": "sciagentguard.packs.hep.fixtures",
            }
        },
        config={
            "normalization": {
                "absolute_tolerance": 1e-12,
                "relative_tolerance": 1e-9,
            }
        },
    )
