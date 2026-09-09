"""冻结板块评分卡V1.1胜率依赖修正与赔率输出。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1_1.yaml"


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
        ROOT / "docs" / "INDUSTRY_SECTOR_SCORECARD_V1_1_CORRECTION.md",
        ROOT / "research" / "industry_sector_scorecard_v1_1.py",
        ROOT / "scripts" / "build_industry_sector_scorecard_v1_1.py",
        ROOT / "scripts" / "freeze_industry_sector_scorecard_v1_1.py",
        ROOT / "tests" / "test_industry_sector_scorecard_v1_1.py",
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
        raise FileNotFoundError(f"V1.1冻结缺少文件：{missing}")

    integrity = config["parent_integrity"]
    expected_parent_hashes = {
        config["parents"]["v1_manifest"]: integrity[
            "industry_sector_scorecard_v1_manifest_sha256"
        ],
        config["parents"]["taxonomy_manifest"]: integrity[
            "industry_expectation_gap_etf_model_v2_manifest_sha256"
        ],
    }
    for relative, expected in expected_parent_hashes.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"冻结父清单已变化：{relative}，{actual} != {expected}")

    report = json.loads((ROOT / config["outputs"]["json"]).read_text(encoding="utf-8"))
    if report["parent_files_changed"] is not False:
        raise RuntimeError("V1.1没有保持V1父文件不变")
    if report["score_conditioned_win_rate"] != "UNAVAILABLE_AWAITING_TRUE_FORWARD":
        raise RuntimeError("V1.1违规生成了高分条件胜率")
    if report["may_call_predictive_edge"] is not False:
        raise RuntimeError("V1.1违规宣称预测Edge")
    if report["expectancy_identity_max_error"] >= 1.0e-12:
        raise RuntimeError("V1.1胜率赔率期望恒等式未通过")
    if report["governance"]["kelly_position_sizing_allowed"] is not False:
        raise RuntimeError("V1.1违规启用Kelly仓位")

    manifest = {
        "version": config["version"],
        "status": "FROZEN_RESEARCH_ONLY_DEPENDENCE_CORRECTED",
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
            relative: {"expected": expected, "actual": sha256(ROOT / relative)}
            for relative, expected in expected_parent_hashes.items()
        },
        "invariants": {
            "overlapping_months_are_independent_trials": False,
            "primary_interval": "THREE_MONTH_MOVING_BLOCK_BOOTSTRAP",
            "odds_are_gross_theoretical_sector_excess": True,
            "score_is_probability": False,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
            "may_call_predictive_edge": False,
            "kelly_position_sizing_allowed": False,
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
