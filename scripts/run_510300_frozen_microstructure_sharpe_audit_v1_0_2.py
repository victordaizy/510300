"""运行冻结的V1.0.2机械修正版夏普审计。"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from frozen_microstructure_sharpe_audit_v1_0_2 import main


if __name__ == "__main__":
    raise SystemExit(main())
