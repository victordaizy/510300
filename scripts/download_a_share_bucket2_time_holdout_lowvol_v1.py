"""冻结低波动公式后下载桶2未见未来收益。"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.download_a_share_bucket1_time_holdout_formula_v1 as implementation  # noqa: E402
from scripts.freeze_a_share_bucket2_time_holdout_lowvol_v1 import verify_protocol  # noqa: E402


def main() -> int:
    implementation.verify_protocol = verify_protocol
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
