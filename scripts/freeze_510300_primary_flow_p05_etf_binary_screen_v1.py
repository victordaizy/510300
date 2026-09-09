"""冻结510300主流量概率0.5二元筛选。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_primary_flow_p05_etf_binary_screen_v1.yaml"
SOURCE_PATHS = [
    ROOT / "research" / "primary_flow_p05_etf_binary_screen_v1.py",
    ROOT / "research" / "primary_flow_direction_challenger.py",
    ROOT / "research" / "intraday_binary_livermore_screen_v1.py",
    ROOT / "scripts" / "freeze_510300_primary_flow_p05_etf_binary_screen_v1.py",
    ROOT / "tests" / "test_primary_flow_p05_etf_binary_screen_v1.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """在结果不存在时生成一次性冻结清单。"""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["protocol"]["study_id"] != "510300_PRIMARY_FLOW_P05_ETF_BINARY_SCREEN_V1":
        raise ValueError("研究编号不匹配")
    if config["protocol"]["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("协议尚未达到冻结前实现完成状态")
    if float(config["state_rule"]["threshold"]) != 0.50:
        raise ValueError("冻结阈值必须为0.50")
    if config["state_rule"]["threshold_search"] != "FORBIDDEN":
        raise ValueError("必须禁止阈值搜索")
    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    report_path = ROOT / config["artifacts"]["report_json"]
    if manifest_path.exists():
        raise FileExistsError("冻结清单已存在，禁止覆盖")
    if report_path.exists():
        raise FileExistsError("结果已存在，不能倒序冻结")
    for path in SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"冻结源文件不存在：{path}")
    source_artifacts = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_PATHS
    }
    source_contract_names = {
        "parent_implementation",
        "execution_ledger_dependency",
    }
    input_artifacts: dict[str, str] = {}
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"冻结工件不存在：{path}")
        actual = sha256_file(path)
        if actual != contract["required_sha256"]:
            raise ValueError(
                f"固定哈希错误：{name}，期望 {contract['required_sha256']}，实际 {actual}"
            )
        relative = path.relative_to(ROOT).as_posix()
        if name in source_contract_names:
            if source_artifacts[relative] != actual:
                raise AssertionError("源依赖哈希核对不一致")
        else:
            input_artifacts[relative] = actual
    manifest = {
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_BEFORE_FIRST_SCREEN_RUN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "source_artifacts": source_artifacts,
        "input_artifacts": input_artifacts,
        "candidate_id": config["state_rule"]["candidate_id"],
        "probability_threshold": float(config["state_rule"]["threshold"]),
        "cost_scenarios": list(config["evaluation"]["cost_scenarios"]),
        "start_offsets": list(
            config["evaluation"]["start_date_perturbations_trading_days"]
        ),
        "period_ids": [period["id"] for period in config["evaluation"]["periods"]],
        "hard_gates": {
            "minimum_annualized_excess": float(
                config["objective"]["minimum_annualized_excess"]
            ),
            "minimum_rolling_242d_excess_median": float(
                config["objective"]["minimum_rolling_excess_median"]
            ),
            "every_stress_period_and_start_required": True,
        },
        "forbidden_forecast_columns": list(
            config["data_contracts"]["frozen_forecasts"][
                "forbidden_columns_as_signal"
            ]
        ),
        "threshold_or_calibration_rescue": "FORBIDDEN",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "protocol_sha256": manifest["protocol_sha256"],
                "candidate_id": manifest["candidate_id"],
                "probability_threshold": manifest["probability_threshold"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
