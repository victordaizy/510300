"""国家统计局全国月度数据库的命令行入口。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.nbs_monthly_collection import main

if __name__ == "__main__":
    main()
