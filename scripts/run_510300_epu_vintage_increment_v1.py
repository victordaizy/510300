"""EPU 历史版本月度研究单一入口。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.epu_vintage_increment_v1 import main

if __name__ == "__main__":
    main()
