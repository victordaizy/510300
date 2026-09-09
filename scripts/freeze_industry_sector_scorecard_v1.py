"""冻结板块评分、历史胜率与频率V1。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    protocol_files = [
        CONFIG_FILE,
        ROOT / "docs" / "INDUSTRY_SECTOR_SCORECARD_V1_SPEC.md",
        ROOT / "research" / "industry_sector_scorecard_v1.py",
        ROOT / "scripts" / "build_industry_sector_scorecard_v1.py",
        ROOT / "scripts" / "freeze_industry_sector_scorecard_v1.py",
        ROOT / "tests" / "test_industry_sector_scorecard_v1.py",
    ]
    input_files = [
        ROOT / value
        for value in [
            *config["parents"].values(),
            *config["inputs"].values(),
        ]
    ]
    output_files = [
        ROOT / value
        for key, value in config["outputs"].items()
        if key != "manifest"
    ]
    all_files = protocol_files + input_files + output_files
    missing = [path.relative_to(ROOT).as_posix() for path in all_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"评分卡冻结缺少文件：{missing}")
    report = json.loads((ROOT / config["outputs"]["json"]).read_text(encoding="utf-8"))
    if report["failed_model_reference"]["status"] != "HISTORICAL_REJECTED_FROZEN":
        raise RuntimeError("评分卡没有保留失败模型边界")
    if report["definitions"]["score_conditioned_win_rate"] != "UNAVAILABLE_AWAITING_TRUE_FORWARD":
        raise RuntimeError("评分卡违规生成了高分条件胜率")
    manifest = {
        "version": config["version"],
        "status": "FROZEN_RESEARCH_ONLY",
        "as_of_date": config["as_of_date"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in protocol_files
        },
        "input_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in input_files
        },
        "output_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in output_files
        },
        "invariants": {
            "score_is_probability": False,
            "historical_base_rate_is_model_win_rate": False,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
            "failed_s1_reuse_allowed": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(manifest_path),
                "status": manifest["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

