"""解禁公告增量研究的固定入口。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.unlock_announcement_increment_v1 import main

if __name__ == "__main__":
    main()
