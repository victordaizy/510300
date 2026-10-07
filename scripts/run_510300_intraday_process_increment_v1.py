"""运行已登记的510300盘中过程增量研究。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.intraday_process_increment_v1 import main

if __name__ == "__main__":
    raise SystemExit(main())
