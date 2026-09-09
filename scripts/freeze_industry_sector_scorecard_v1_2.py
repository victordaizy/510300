"""冻结板块评分卡V1.2双轴决策层。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1_2.yaml"


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
        ROOT / "docs" / "INDUSTRY_SECTOR_SCORECARD_V1_2_DUAL_AXIS_SPEC.md",
        ROOT / "research" / "industry_sector_scorecard_v1_2.py",
        ROOT / "scripts" / "build_industry_sector_scorecard_v1_2.py",
        ROOT / "scripts" / "freeze_industry_sector_scorecard_v1_2.py",
        ROOT / "tests" / "test_industry_sector_scorecard_v1_2.py",
    ]
    input_files = [ROOT / value for value in config["parents"].values()]
    output_files = [
        ROOT / value
        for key, value in config["outputs"].items()
        if key != "manifest"
    ]
    all_files = protocol_files + input_files + output_files
    missing = [
        path.relative_to(ROOT).as_posix() for path in all_files if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"V1.2冻结缺少文件：{missing}")

    parent_manifest = ROOT / config["parents"]["v1_1_manifest"]
    expected_parent_hash = config["parent_integrity"][
        "industry_sector_scorecard_v1_1_manifest_sha256"
    ]
    actual_parent_hash = sha256(parent_manifest)
    if actual_parent_hash != expected_parent_hash:
        raise RuntimeError(
            f"V1.1父清单已变化：{actual_parent_hash} != {expected_parent_hash}"
        )

    report = json.loads((ROOT / config["outputs"]["json"]).read_text(encoding="utf-8"))
    if report["parent_files_changed"] is not False:
        raise RuntimeError("V1.2没有保持V1.1父文件不变")
    if report["synthetic_combined_numeric_score"] is not None:
        raise RuntimeError("V1.2违规生成合成数值分数")
    if report["score_conditioned_win_rate"] != "UNAVAILABLE_AWAITING_TRUE_FORWARD":
        raise RuntimeError("V1.2违规生成高分条件胜率")
    if report["may_call_predictive_edge"] is not False:
        raise RuntimeError("V1.2违规宣称预测Edge")
    if report["position_mapping_enabled"] is not False:
        raise RuntimeError("V1.2违规启用仓位映射")

    manifest = {
        "version": config["version"],
        "status": "FROZEN_RESEARCH_ONLY_DUAL_AXIS",
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
        "parent_manifest_integrity": {
            parent_manifest.relative_to(ROOT).as_posix(): {
                "expected": expected_parent_hash,
                "actual": actual_parent_hash,
            }
        },
        "invariants": {
            "synthetic_combined_numeric_score_allowed": False,
            "current_score_is_probability": False,
            "historical_odds_are_current_conditional_probability": False,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
            "may_call_predictive_edge": False,
            "p1_is_buy_rating": False,
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
                "parent_manifest_unchanged": actual_parent_hash == expected_parent_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
