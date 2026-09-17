"""A/H信息增量研究的单一入口。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.ah_premium_increment_v1 import main

if __name__ == "__main__":
    main()
