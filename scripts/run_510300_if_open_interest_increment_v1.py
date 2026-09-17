"""在Windows直接调用时保持项目导入路径明确。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.if_open_interest_increment_v1 import main


if __name__ == "__main__":
    main()
