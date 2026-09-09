"""一次性运行冻结的510300微观结构仓位夏普审计。"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from frozen_microstructure_sharpe_audit_v1 import main


if __name__ == "__main__":
    raise SystemExit(main())
