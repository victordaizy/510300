"""第一批基准研究及本地只读状态入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.daily_native_baseline_v1 import main

if __name__ == "__main__":
    raise SystemExit(main())
