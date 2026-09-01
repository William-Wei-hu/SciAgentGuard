"""Run the deterministic synthetic HEP workflow and emit one JSON trace."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from sciagentguard.core import ContractContext, ScientificContract, SemanticFaultInjector
from sciagentguard.packs.hep import (
    DeclaredEventProvenanceContract,
    DisjointEventSplitsContract,
    EmptySelectionInjector,
    FiniteWeightsContract,
    JetPtRangeContract,
    MissingBranchInjector,
    NonemptySelectionContract,
    NonfiniteWeightsInjector,
    NonzeroWeightSupportContract,
    RequiredBranchesContract,
    SplitLeakageInjector,
    SyntheticHEPRepairPolicy,
    SyntheticHEPWorkflow,
    UndeclaredSyntheticDataInjector,
    UnitScaleErrorInjector,
    WrongNormalizationInjector,
    YieldNormalizationContract,
    ZeroWeightsInjector,
    make_synthetic_normalization_context,
    make_synthetic_selection_context,
    make_synthetic_split_context,
)
from sciagentguard.runtime import (
    GuardedWorkflowRunner,
    RepairAction,
    RepairOutcome,
    RepairRunner,
    RepairStep,
)

FAULT_INJECTORS: dict[str, SemanticFaultInjector] = {
    "missing_branch": MissingBranchInjector(),
    "zero_weights": ZeroWeightsInjector(),
    "nonfinite_weights": NonfiniteWeightsInjector(),
    "unit_scale_error": UnitScaleErrorInjector(),
    "undeclared_synthetic_data": UndeclaredSyntheticDataInjector(),
}
ANALYSIS_FAULT_CASES: dict[
    str, tuple[Callable[[], ContractContext], SemanticFaultInjector, ScientificContract]
] = {
    "empty_selection": (
        make_synthetic_selection_context,
        EmptySelectionInjector(),
        NonemptySelectionContract(),
    ),
    "split_leakage": (
        make_synthetic_split_context,
        SplitLeakageInjector(),
        DisjointEventSplitsContract(),
    ),
    "wrong_normalization": (
        make_synthetic_normalization_context,
        WrongNormalizationInjector(),
        YieldNormalizationContract(),
    ),
}
ALL_FAULT_INJECTORS = {
    **FAULT_INJECTORS,
    **{fault: case[1] for fault, case in ANALYSIS_FAULT_CASES.items()},
}


def configured_contracts() -> tuple[ScientificContract, ...]:
    return (
        RequiredBranchesContract(),
        FiniteWeightsContract(),
        NonzeroWeightSupportContract(),
        JetPtRangeContract(),
        DeclaredEventProvenanceContract(),
    )


def _reject_unregistered_repair(action: RepairAction, *, attempt_id: str) -> ContractContext:
    raise ValueError(
        f"no trusted repair step is registered for {action.action_type!r} at {attempt_id!r}"
    )


def configured_run(
    fault: str,
) -> tuple[Callable[[], ContractContext], RepairStep, tuple[ScientificContract, ...]]:
    analysis_case = ANALYSIS_FAULT_CASES.get(fault)
    if analysis_case is not None:
        make_context, injector, contract = analysis_case

        def initial_step() -> ContractContext:
            return injector.inject(make_context())

        return initial_step, _reject_unregistered_repair, (contract,)

    workflow = SyntheticHEPWorkflow(fault=FAULT_INJECTORS.get(fault))
    return workflow.initial_step, workflow.repair_step, configured_contracts()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fault",
        choices=("none", *FAULT_INJECTORS, *ANALYSIS_FAULT_CASES),
        default="none",
        help="deterministic evaluation-only fault to inject",
    )
    parser.add_argument(
        "--mode",
        choices=("enforce", "repair"),
        default="enforce",
        help="block on violations or run the bounded deterministic repair policy",
    )
    parser.add_argument("--output", type=Path, help="write the JSON trace to this path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    step, repair_step, contracts = configured_run(args.fault)

    if args.mode == "enforce":
        fault = None if args.fault == "none" else ALL_FAULT_INJECTORS[args.fault]
        guarded = GuardedWorkflowRunner().execute(SyntheticHEPWorkflow(fault=fault).checkpoints())
        trace_json = guarded.trace.model_dump_json(indent=2)
        succeeded = not guarded.trace.blocked
    else:
        repaired = RepairRunner().execute(
            step,
            repair_step,
            contracts,
            SyntheticHEPRepairPolicy(),
        )
        trace_json = repaired.trace.model_dump_json(indent=2)
        succeeded = repaired.trace.outcome in {RepairOutcome.PASSED, RepairOutcome.REPAIRED}
    if args.output is None:
        print(trace_json)
    else:
        args.output.write_text(f"{trace_json}\n", encoding="utf-8")
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
