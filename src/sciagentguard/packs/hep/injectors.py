"""Opt-in semantic faults for the deterministic HEP fixture."""

from __future__ import annotations

from dataclasses import replace

from sciagentguard.core import ContractContext
from sciagentguard.packs.hep._analysis import (
    replace_observed_yield,
    replace_selected_event_ids,
    replace_split_event_ids,
    require_normalization,
    require_selection,
    require_splits,
    split_event_ids,
)
from sciagentguard.packs.hep._events import (
    copy_event_columns,
    numeric_event_column,
    replace_event_columns,
)
from sciagentguard.packs.hep.contracts import (
    DISJOINT_EVENT_SPLITS_CONTRACT_ID,
    EVENT_PROVENANCE_CONTRACT_ID,
    FINITE_WEIGHTS_CONTRACT_ID,
    JET_PT_RANGE_CONTRACT_ID,
    NONEMPTY_SELECTION_CONTRACT_ID,
    NONZERO_WEIGHT_SUPPORT_CONTRACT_ID,
    REQUIRED_BRANCHES_CONTRACT_ID,
    YIELD_NORMALIZATION_CONTRACT_ID,
)


class MissingBranchInjector:
    """Remove the configured transverse-momentum branch from a copied context."""

    fault_id = "missing_branch"
    taxonomy = "schema"
    description = "Remove a required analysis branch from the event artifact."
    preconditions = ("The events artifact contains jet_pt_gev.",)
    mutation_description = "Remove jet_pt_gev while preserving all other event branches."
    expected_contract_ids = (REQUIRED_BRANCHES_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the deterministic fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        columns = copy_event_columns(context)
        if "jet_pt_gev" not in columns:
            raise ValueError("missing_branch requires the event branch 'jet_pt_gev'")
        del columns["jet_pt_gev"]
        return replace_event_columns(context, columns)


class ZeroWeightsInjector:
    """Replace every event weight with zero in a copied context."""

    fault_id = "zero_weights"
    taxonomy = "numerical"
    description = "Set all event weights to zero."
    preconditions = ("The events artifact contains at least one weight.",)
    mutation_description = "Replace every value in the weight branch with 0.0."
    expected_contract_ids = (NONZERO_WEIGHT_SUPPORT_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the deterministic fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        columns = copy_event_columns(context)
        weights = columns.get("weight")
        if not weights:
            raise ValueError("zero_weights requires a non-empty event branch 'weight'")
        columns["weight"] = tuple(0.0 for _ in weights)
        return replace_event_columns(context, columns)


class NonfiniteWeightsInjector:
    """Insert one NaN weight at a stable index in a copied context."""

    fault_id = "nonfinite_weights"
    taxonomy = "numerical"
    description = "Insert a non-finite value into the event weights."
    preconditions = ("The events artifact contains at least three weights.",)
    mutation_description = "Replace the weight at zero-based index 2 with NaN."
    expected_contract_ids = (FINITE_WEIGHTS_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the deterministic fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        columns = copy_event_columns(context)
        weights = columns.get("weight")
        if weights is None or len(weights) <= 2:
            raise ValueError("nonfinite_weights requires at least three event weights")
        mutated_weights = list(weights)
        mutated_weights[2] = float("nan")
        columns["weight"] = tuple(mutated_weights)
        return replace_event_columns(context, columns)


class UnitScaleErrorInjector:
    """Scale jet transverse momenta while leaving their declared unit unchanged."""

    fault_id = "unit_scale_error"
    taxonomy = "units"
    description = "Multiply jet transverse momenta by 1000 without changing metadata."
    preconditions = ("The events artifact contains numeric jet_pt_gev values.",)
    mutation_description = "Multiply every jet_pt_gev value by 1000."
    expected_contract_ids = (JET_PT_RANGE_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the deterministic fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        values = numeric_event_column(context, "jet_pt_gev")
        if not values:
            raise ValueError("unit_scale_error requires a non-empty numeric branch 'jet_pt_gev'")
        columns = copy_event_columns(context)
        columns["jet_pt_gev"] = tuple(value * 1000.0 for value in values)
        return replace_event_columns(context, columns)


class UndeclaredSyntheticDataInjector:
    """Replace events with a valid dummy table while removing its source declaration."""

    fault_id = "undeclared_synthetic_data"
    taxonomy = "provenance"
    description = "Replace events with undeclared synthetic dummy data."
    preconditions = (
        "The events artifact is a well-formed column mapping.",
        "The context contains an events provenance declaration.",
    )
    mutation_description = "Replace events with two dummy rows and remove provenance['events']."
    expected_contract_ids = (EVENT_PROVENANCE_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the deterministic fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        copy_event_columns(context)
        if "events" not in context.provenance:
            raise ValueError("undeclared_synthetic_data requires events provenance")

        artifacts = dict(context.artifacts)
        artifacts["events"] = {
            "event_id": (9001, 9002),
            "jet_pt_gev": (40.0, 80.0),
            "weight": (1.0, -0.5),
        }
        provenance = dict(context.provenance)
        del provenance["events"]
        return replace(context, artifacts=artifacts, provenance=provenance)


class EmptySelectionInjector:
    """Remove every selected event while preserving the declared selection input."""

    fault_id = "empty_selection"
    taxonomy = "selection"
    description = "Replace the selected event identifiers with an empty sequence."
    preconditions = ("The selection artifact contains at least one selected event.",)
    mutation_description = "Set selected_event_ids to an empty tuple."
    expected_contract_ids = (NONEMPTY_SELECTION_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the selection fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        _, _, selected_ids = require_selection(context)
        if not selected_ids:
            raise ValueError("empty_selection requires at least one selected event")
        return replace_selected_event_ids(context, ())


class SplitLeakageInjector:
    """Duplicate one declared train event identifier into the test split."""

    fault_id = "split_leakage"
    taxonomy = "data_leakage"
    description = "Add train event 1004 to the test split."
    preconditions = (
        "The splits artifact contains disjoint train and test event identifiers.",
        "The train split contains synthetic event identifier 1004.",
    )
    mutation_description = "Prepend event identifier 1004 to the test split."
    expected_contract_ids = (DISJOINT_EVENT_SPLITS_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the split fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        splits = require_splits(context)
        train_ids = split_event_ids(splits, "train")
        test_ids = split_event_ids(splits, "test")
        if set(train_ids).intersection(test_ids):
            raise ValueError("split_leakage requires initially disjoint train and test splits")
        if 1004 not in train_ids:
            raise ValueError("split_leakage requires train event identifier 1004")
        return replace_split_event_ids(context, "test", (1004, *test_ids))


class WrongNormalizationInjector:
    """Scale the observed normalized yield while preserving every declared input."""

    fault_id = "wrong_normalization"
    taxonomy = "normalization"
    description = "Multiply the observed normalized yield by 10."
    preconditions = ("The normalization artifact has the declared fixture yield of 100.0.",)
    mutation_description = "Replace observed_yield 100.0 with 1000.0."
    expected_contract_ids = (YIELD_NORMALIZATION_CONTRACT_ID,)
    restoration_strategy = "Discard the injected context and rebuild the normalization fixture."

    def inject(self, context: ContractContext, *, seed: int | None = None) -> ContractContext:
        del seed
        _, _, _, _, _, observed_yield = require_normalization(context)
        if observed_yield != 100.0:
            raise ValueError("wrong_normalization requires the declared fixture yield of 100.0")
        return replace_observed_yield(context, observed_yield * 10.0)
