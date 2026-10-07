"""从项目根环境调用路线图执行入口。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.roadmap_execution_v1 import main

if __name__ == "__main__":
    main()
