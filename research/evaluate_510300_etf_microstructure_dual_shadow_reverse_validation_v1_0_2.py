"""V1.0.2入口：评价规则与V1完全相同，仅绑定机械修正后的采集冻结件。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import research.evaluate_510300_etf_microstructure_dual_shadow_reverse_validation_v1 as base


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_REVERSE_VALIDATION_V1_0_2"
CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2.yaml"
)
FREEZE_MANIFEST = CONFIG_FILE.with_name(
    "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2_manifest.json"
)


def main() -> int:
    base.PROJECT_ID = PROJECT_ID
    base.CONFIG_FILE = CONFIG_FILE
    base.FREEZE_MANIFEST = FREEZE_MANIFEST
    base.__file__ = __file__
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
