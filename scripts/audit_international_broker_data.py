"""运行国际化券商研究的数据底座审计，不执行收益检验。"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.international_broker_data_contract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
