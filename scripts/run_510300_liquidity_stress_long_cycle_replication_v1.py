"""运行510300流动性压力风险开关长周期复制。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.liquidity_stress_long_cycle_replication_v1 import main


if __name__ == "__main__":
    raise SystemExit(main())
