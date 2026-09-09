"""在读取未来收益前冻结510300宏观压力规避V1协议、实现与输入。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_macro_stress_avoidance_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_macro_stress_avoidance_v1_manifest.json"
SOURCE_PATHS = [
    ROOT / "research" / "macro_stress_avoidance_v1.py",
    ROOT / "scripts" / "download_510300_macro_stress_inputs_v1.py",
    ROOT / "scripts" / "freeze_510300_macro_stress_avoidance_v1.py",
    ROOT / "scripts" / "run_510300_macro_stress_avoidance_v1.py",
    ROOT / "tests" / "test_macro_stress_avoidance_v1.py",
]


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_protocol(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol["study_id"] != "510300_MACRO_STRESS_AVOIDANCE_V1":
        raise ValueError("研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("实现尚未达到冻结前完成状态")
    if protocol["data_start"] != "2021-01-01":
        raise ValueError("用户固定的数据起点必须为2021-01-01")
    if protocol["allowed_assets"] != ["510300.SH", "CASH_CNY"]:
        raise ValueError("可用资产必须严格为510300与人民币现金")
    if int(protocol["main_trailing_years"]) != 5:
        raise ValueError("主模型必须使用严格5年日历窗口")
    if int(protocol["robustness_trailing_years"]) != 3:
        raise ValueError("稳健性模型必须使用严格3年日历窗口")
    if float(protocol["percentile"]) != 0.90:
        raise ValueError("冻结分位数必须为90%")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("必须禁止结果后参数营救")
    if not bool(config["point_in_time"]["current_observation_excluded_from_percentile"]):
        raise ValueError("分位数必须排除当期观察")
    if not bool(config["point_in_time"]["expanding_window_forbidden"]):
        raise ValueError("必须禁止扩展窗口")
    if int(config["event_study"]["minimum_independent_events"]) != 8:
        raise ValueError("独立事件硬门必须为8次")
    if float(config["historical_gates"]["five_year_net_sharpe_minimum"]) != 1.20:
        raise ValueError("5年净夏普率硬门必须为1.20")
    if bool(config["governance"]["live_trading_authorized"]):
        raise ValueError("该研究禁止实盘授权")


def main() -> int:
    """确认没有绩效输出后生成一次性冻结清单。"""

    if MANIFEST_PATH.exists():
        raise FileExistsError("冻结清单已存在，禁止覆盖")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_protocol(config)
    for key, relative in config["artifacts"].items():
        if key == "freeze_manifest":
            continue
        path = ROOT / relative
        if path.exists():
            raise FileExistsError(f"绩效输出已存在，不能倒序冻结：{path}")
    for path in SOURCE_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"冻结实现不存在：{path}")

    input_artifacts: dict[str, str] = {}
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"冻结输入不存在：{name}，{path}")
        actual = sha256_file(path)
        expected = str(contract["required_sha256"])
        if actual != expected:
            raise ValueError(f"冻结输入哈希不匹配：{name}，实际{actual}")
        input_artifacts[path.relative_to(ROOT).as_posix()] = actual

    audit_contract = config["data_contracts"]["macro_input_audit"]
    audit = json.loads((ROOT / audit_contract["file"]).read_text(encoding="utf-8"))
    if audit.get("status") != audit_contract["required_status"]:
        raise ValueError("宏观输入审计未通过")
    if audit.get("scope") != "仅2021-01-01至2026-08-25；未采集2016-2020":
        raise ValueError("宏观输入审计范围与用户约束不匹配")
    if audit.get("governance", {}).get("performance_returns_read") is not False:
        raise ValueError("输入采集阶段不得读取绩效收益")
    if audit.get("governance", {}).get("backtest_run") is not False:
        raise ValueError("输入采集阶段不得运行回测")
    for name in ("fdr007", "usdcny_midpoint", "pmi_new_orders", "tsf_stock_yoy"):
        if int(audit["series"][name]["duplicate_keys"]) != 0:
            raise ValueError(f"宏观序列存在重复键：{name}")
        if int(audit["series"][name]["missing_first_release_values"]) != 0:
            raise ValueError(f"宏观序列存在首发值缺失：{name}")

    source_artifacts = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_PATHS
    }
    manifest: dict[str, Any] = {
        "study_id": config["protocol"]["study_id"],
        "version": config["protocol"]["version"],
        "status": "FROZEN_BEFORE_FIRST_EVENT_RUN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "source_artifacts": source_artifacts,
        "input_artifacts": input_artifacts,
        "data_scope": {
            "start": config["protocol"]["data_start"],
            "end": config["protocol"]["data_end"],
            "pre_2021_data_allowed": False,
            "full_calendar_five_year_history_required": True,
            "expanding_or_shortened_window_allowed": False,
        },
        "allowed_assets": config["protocol"]["allowed_assets"],
        "factor_count": 4,
        "factor_ids": ["FDR007_LIQUIDITY", "PMI_NEW_ORDERS", "TSF_STOCK_YOY", "USDCNY_20D"],
        "percentile": float(config["protocol"]["percentile"]),
        "current_observation_excluded_from_percentile": True,
        "event_gate": {
            "minimum_independent_events": int(
                config["event_study"]["minimum_independent_events"]
            ),
            "primary_horizon_trading_days": int(
                config["event_study"]["primary_horizon_trading_days"]
            ),
            "minimum_negative_fraction": float(
                config["event_study"]["minimum_negative_fraction"]
            ),
            "portfolio_run_before_pass": "NOT_ALLOWED",
        },
        "historical_gates": config["historical_gates"],
        "robustness_gates": config["robustness_gates"],
        "costs": config["costs"],
        "benchmarks": config["benchmarks"],
        "no_parameter_rescue": True,
        "historical_run_completed": False,
        "return_outcomes_read_after_freeze": False,
        "portfolio_backtest_status": "NOT_ALLOWED_BEFORE_EVENT_GATE",
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
    }
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    _atomic_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "protocol_sha256": manifest["protocol_sha256"],
                "source_artifact_count": len(source_artifacts),
                "input_artifact_count": len(input_artifacts),
                "return_outcomes_read_after_freeze": False,
                "live_trading_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
