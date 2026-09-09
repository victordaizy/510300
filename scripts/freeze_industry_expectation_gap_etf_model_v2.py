"""冻结行业预期差—ETF模型V2的分类、数据快照、实现与报告指纹。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "industry_expectation_gap_etf_model_v2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["outputs"]["manifest"]
    protocol_files = [
        CONFIG_FILE,
        ROOT / "docs" / "INDUSTRY_EXPECTATION_GAP_ETF_MODEL_V2_SECTOR_TAXONOMY_SPEC.md",
        ROOT / "research" / "industry_expectation_gap_etf_model_v2.py",
        ROOT / "scripts" / "acquire_industry_expectation_gap_etf_model_v2.py",
        ROOT / "scripts" / "build_industry_expectation_gap_etf_model_v2.py",
        ROOT / "scripts" / "freeze_industry_expectation_gap_etf_model_v2.py",
        ROOT / "tests" / "test_industry_expectation_gap_etf_model_v2.py",
    ]
    acquired_files = [
        ROOT / value
        for key, value in config["acquired_inputs"].items()
        if key != "acquisition_status"
    ] + [ROOT / config["acquired_inputs"]["acquisition_status"]]
    derived_files = [
        ROOT / config["outputs"]["industry_attribution"],
        ROOT / config["outputs"]["economic_bucket_daily"],
        ROOT / config["outputs"]["json"],
        ROOT / config["outputs"]["markdown"],
    ]
    parent_files = [
        ROOT / config["parent_model"],
        ROOT / config["parent_frozen_result"],
        ROOT / config["parent_historical_context"],
        ROOT / "config" / "industry_expectation_gap_etf_model_v1_manifest.json",
        ROOT / "config" / "historical_market_context_supplement_v1_manifest.json",
    ]
    all_files = protocol_files + acquired_files + derived_files + parent_files
    missing = [path.relative_to(ROOT).as_posix() for path in all_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V2冻结缺少文件：{missing}")
    acquisition_status = json.loads(
        (ROOT / config["acquired_inputs"]["acquisition_status"]).read_text(encoding="utf-8")
    )
    output_report = json.loads(
        (ROOT / config["outputs"]["json"]).read_text(encoding="utf-8")
    )
    if acquisition_status.get("status") != "PASS":
        raise RuntimeError("V2采集状态不是PASS")
    if output_report.get("original_prediction_changed") is not False:
        raise RuntimeError("V2错误改变了原冻结预测")
    manifest = {
        "version": config["version"],
        "status": "FROZEN_RESEARCH_ONLY",
        "as_of_date": config["as_of_date"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in protocol_files
        },
        "acquired_input_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in acquired_files
        },
        "derived_output_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in derived_files
        },
        "parent_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in parent_files
        },
        "invariants": {
            "original_prediction_state": output_report["original_prediction_state"],
            "original_prediction_changed": output_report["original_prediction_changed"],
            "economic_bucket_cross_sum_allowed": True,
            "theme_cross_sum_allowed": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
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

