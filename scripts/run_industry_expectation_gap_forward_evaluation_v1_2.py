"""让冻结 V1.1 评价逻辑只读取 V1.2 内容寻址结果输入。"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_industry_expectation_gap_forward_evaluation_v1_1 as base
from scripts.industry_outcome_snapshot_v1_2 import (
    ROOT,
    load_verified_snapshot_manifest,
)


CONFIG_PATH = ROOT / "config" / "industry_expectation_gap_forward_operations_v1_2.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("行业 V1.2 运行配置顶层必须是对象")
    _, by_role = load_verified_snapshot_manifest(config)
    source_to_snapshot = {
        Path(specification["source"]).as_posix(): by_role[role]
        for role, specification in config["input_snapshot"]["roles"].items()
    }
    original_path = base._path

    def snapshot_path(relative: str) -> Path:
        normalized = Path(relative).as_posix()
        if normalized in source_to_snapshot:
            return source_to_snapshot[normalized]
        return original_path(relative)

    base._path = snapshot_path
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
