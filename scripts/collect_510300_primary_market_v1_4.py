"""V1.4 仅修复已批准数据 Junction 的保存路径；沿用 V1.3 免费来源合同。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_510300_primary_market_v1_2 as crosscheck
from scripts import collect_510300_primary_market_v1_3 as collector
from scripts import free_source_storage_v1_3 as storage
from scripts.priority_forward_data_paths_v1 import PATHS


def main() -> int:
    import yaml

    PATHS.validate_data_junction()
    config = yaml.safe_load(collector.CONFIG_FILE.read_text(encoding="utf-8"))
    for group in ("inputs", "outputs"):
        for value in config[group].values():
            PATHS.checked(value)
    bindings = []
    for module in (collector, crosscheck):
        for name in ("write_content_addressed_raw", "persist_record_atomic"):
            bindings.append((module, name, getattr(module, name)))
            setattr(module, name, getattr(storage, name))
    try:
        return collector.main()
    finally:
        for module, name, original in bindings:
            setattr(module, name, original)


if __name__ == "__main__":
    raise SystemExit(main())
