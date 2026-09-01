from importlib.metadata import version

import sciagentguard
import sciagentguard.adapters
import sciagentguard.packs.hep
import sciagentguard.packs.materials
import sciagentguard.runtime


def test_installed_package_exposes_distribution_version() -> None:
    assert sciagentguard.__version__ == version("sciagentguard")
    assert sciagentguard.packs.hep.RequiredBranchesContract().contract_id == (
        "hep.schema.required_branches"
    )
    assert sciagentguard.runtime.GuardedExecutor is not None
    assert sciagentguard.adapters.AtlasGamGamOpenDataAdapter is not None
    assert sciagentguard.adapters.DeePTBSi64Adapter is not None
    assert sciagentguard.packs.materials.HamiltonianBlockHermiticityContract is not None
    assert sciagentguard.packs.hep.AtlasPhotonCountMismatchInjector is not None
