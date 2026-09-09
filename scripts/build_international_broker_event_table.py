"""从已校验的官方事件种子生成正式事件表，不触发收益检验。"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.international_broker_data_contract import (  # noqa: E402
    materialize_event_seed,
)


def main() -> int:
    output = materialize_event_seed(ROOT)
    print(f"国际化券商事件表已生成：{output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

