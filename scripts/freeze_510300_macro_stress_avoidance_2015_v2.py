"""在首次事件收益读取前冻结510300宏观压力规避2015起点敏感性研究。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from research.macro_stress_avoidance_2015_v2 import (  # noqa: E402
    CONFIG_PATH,
    EVIDENCE_CLASS,
    ROOT,
    STUDY_ID,
    load_config,
    sha256_file,
)


MANIFEST_PATH = ROOT / "config" / "510300_macro_stress_avoidance_2015_v2_manifest.json"
SOURCE_PATHS = [
    ROOT / "research" / "macro_stress_avoidance_v1.py",
    ROOT / "research" / "macro_stress_avoidance_2015_v2.py",
    ROOT / "scripts" / "download_510300_macro_stress_inputs_v1.py",
    ROOT / "scripts" / "download_510300_macro_stress_inputs_2015_v2.py",
    ROOT / "scripts" / "build_510300_macro_stress_market_inputs_2015_v2.py",
    ROOT / "scripts" / "freeze_510300_macro_stress_avoidance_2015_v2.py",
    ROOT / "scripts" / "run_510300_macro_stress_avoidance_2015_v2.py",
    ROOT / "tests" / "test_macro_stress_avoidance_v1.py",
    ROOT / "tests" / "test_macro_stress_avoidance_2015_v2.py",
]


def canonical_hash(payload: dict[str, Any]) -> str:
    """计算与键顺序无关的内容哈希。"""

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


def _load_json(relative: str) -> dict[str, Any]:
    payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{relative}")
    return payload


def _validate_protocol(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("实现尚未达到冻结前完成状态")
    if protocol["evidence_class"] != EVIDENCE_CLASS:
        raise ValueError("事后窗口敏感性证据等级不匹配")
    if protocol["data_start"] != "2015-01-01":
        raise ValueError("数据起点必须为2015-01-01")
    if protocol["allowed_assets"] != ["510300.SH", "CASH_CNY"]:
        raise ValueError("可用资产必须严格为510300与人民币现金")
    if int(protocol["main_trailing_years"]) != 5:
        raise ValueError("主模型必须使用严格5自然年窗口")
    if float(protocol["percentile"]) != 0.90:
        raise ValueError("压力分位必须保持90%")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("必须禁止结果后参数营救")
    point_in_time = config["point_in_time"]
    if not bool(point_in_time["current_observation_excluded_from_percentile"]):
        raise ValueError("分位数必须排除当期观察")
    if not bool(point_in_time["expanding_window_forbidden"]):
        raise ValueError("必须禁止扩展窗口")
    if not bool(point_in_time["shortened_window_forbidden"]):
        raise ValueError("必须禁止缩短窗口")
    if int(config["event_study"]["minimum_independent_events"]) != 8:
        raise ValueError("独立事件硬门必须为8次")
    if float(config["historical_gates"]["five_year_net_sharpe_minimum"]) != 1.20:
        raise ValueError("5年净夏普率硬门必须为1.20")
    sensitivity = config["sensitivity_governance"]
    if sensitivity["original_2021_rejection_overridden"] is not False:
        raise ValueError("不得覆盖2021起点的冻结拒绝")
    if sensitivity["eligible_to_authorize_paper_or_live_trading"] is not False:
        raise ValueError("事后敏感性不得授权仿真或实盘")
    if bool(config["governance"]["live_trading_authorized"]):
        raise ValueError("实盘授权必须关闭")


def _validate_macro_audit(audit: dict[str, Any]) -> None:
    if audit.get("status") != "PASS" or audit.get("study_id") != STUDY_ID:
        raise ValueError("宏观输入审计未通过或研究编号不匹配")
    if audit.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("宏观输入审计的证据等级不匹配")
    if audit.get("proxy_substitution_used") is not False:
        raise ValueError("宏观输入不得使用代理替换")
    fdr = audit.get("fdr007_left_censoring", {})
    if fdr.get("first_official_valid_date") != "2017-05-31":
        raise ValueError("FDR007首个官方有效日不匹配")
    if any(fdr.get(name) is not False for name in (
        "zero_fill_used", "interpolation_used", "FR007_proxy_used"
    )):
        raise ValueError("FDR007上线前不得零填充、插值或代理")
    tsf = audit.get("tsf_stock_yoy_left_censoring", {})
    if tsf.get("official_2015_frequency") != "QUARTERLY":
        raise ValueError("2015年社融存量官方频率应为季度")
    if tsf.get("first_monthly_reference_period") != "2016-01":
        raise ValueError("社融存量首个月度参考期不匹配")
    if tsf.get("handling") != (
        "LEFT_CENSORED_NO_QUARTERLY_TO_MONTHLY_EXPANSION_"
        "NO_INTERPOLATION_NO_PROXY"
    ):
        raise ValueError("2015年社融月频左删失规则不匹配")
    for name in ("fdr007", "usdcny_midpoint", "pmi_new_orders", "tsf_stock_yoy"):
        series = audit["series"][name]
        if int(series["duplicate_keys"]) != 0:
            raise ValueError(f"宏观序列存在重复键：{name}")
        if int(series["missing_first_release_values"]) != 0:
            raise ValueError(f"宏观序列存在首发值缺失：{name}")
    governance = audit.get("governance", {})
    if governance.get("performance_metrics_computed") is not False:
        raise ValueError("宏观采集阶段不得计算绩效")
    if governance.get("backtest_run") is not False:
        raise ValueError("宏观采集阶段不得运行回测")


def _validate_market_audit(audit: dict[str, Any]) -> None:
    if audit.get("status") != "PASS" or audit.get("study_id") != STUDY_ID:
        raise ValueError("市场输入审计未通过或研究编号不匹配")
    if audit.get("missing_h00300_dates_on_510300_calendar") != 0:
        raise ValueError("H00300在510300交易日历上存在缺失")
    for name in ("510300", "H00300"):
        overlap = audit["overlap"][name]
        if overlap.get("passed") is not True:
            raise ValueError(f"市场输入重叠区对账失败：{name}")
        for field in overlap["fields"].values():
            if int(field["mismatches_above_1e_minus_9"]) != 0:
                raise ValueError(f"市场输入重叠区存在差异：{name}")
    for relative, output in audit["outputs"].items():
        if sha256_file(ROOT / relative) != output["sha256"]:
            raise ValueError(f"市场输入输出哈希漂移：{relative}")
        if int(output["duplicate_dates"]) != 0:
            raise ValueError(f"市场输入存在重复日期：{relative}")
    governance = audit.get("governance", {})
    if governance.get("performance_metrics_computed") is not False:
        raise ValueError("市场输入阶段不得计算绩效")
    if governance.get("backtest_run") is not False:
        raise ValueError("市场输入阶段不得运行回测")


def main() -> int:
    """确认未读取未来收益后，生成一次性冻结清单。"""

    if MANIFEST_PATH.exists():
        raise FileExistsError("冻结清单已存在，禁止覆盖")
    config = load_config()
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
        expected = contract.get("required_sha256")
        if expected is not None and actual != str(expected):
            raise ValueError(f"冻结输入哈希不匹配：{name}，实际{actual}")
        input_artifacts[path.relative_to(ROOT).as_posix()] = actual

    macro_audit = _load_json(config["data_contracts"]["macro_input_audit"]["file"])
    market_audit = _load_json(config["data_contracts"]["market_input_audit"]["file"])
    _validate_macro_audit(macro_audit)
    _validate_market_audit(market_audit)
    for relative, expected in macro_audit["output_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"宏观输入输出哈希漂移：{relative}")

    source_artifacts = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in SOURCE_PATHS
    }
    predecessor_result = config["inheritance"]["predecessor_result_file"]
    predecessor_manifest = config["inheritance"]["predecessor_manifest_file"]
    manifest: dict[str, Any] = {
        "study_id": STUDY_ID,
        "version": config["protocol"]["version"],
        "status": "FROZEN_BEFORE_FIRST_EVENT_RUN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "evidence_class": EVIDENCE_CLASS,
        "protocol_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "source_artifacts": source_artifacts,
        "input_artifacts": input_artifacts,
        "predecessor": {
            "result_file": predecessor_result,
            "result_sha256": sha256_file(ROOT / predecessor_result),
            "manifest_file": predecessor_manifest,
            "manifest_sha256": sha256_file(ROOT / predecessor_manifest),
            "required_decision": config["inheritance"]["predecessor_required_decision"],
            "overridden": False,
        },
        "data_scope": {
            "requested_start": "2015-01-01",
            "end": config["protocol"]["data_end"],
            "pre_2021_data_allowed": True,
            "full_calendar_five_year_history_required": True,
            "expanding_or_shortened_window_allowed": False,
            "exact_factor_history_from_2015_available": False,
            "fdr007_first_official_valid_date": "2017-05-31",
            "tsf_stock_yoy_first_monthly_reference_period": "2016-01",
        },
        "allowed_assets": config["protocol"]["allowed_assets"],
        "factor_count": 4,
        "factor_ids": [
            "FDR007_LIQUIDITY",
            "PMI_NEW_ORDERS",
            "TSF_STOCK_YOY",
            "USDCNY_20D",
        ],
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
        "post_rejection_window_sensitivity": True,
        "original_2021_rejection_overridden": False,
        "no_parameter_rescue": True,
        "historical_run_completed": False,
        "return_outcomes_read_after_freeze": False,
        "portfolio_backtest_status": "NOT_ALLOWED_BEFORE_EVENT_GATE",
        "goal_achieved": False,
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
                "evidence_class": EVIDENCE_CLASS,
                "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "protocol_sha256": manifest["protocol_sha256"],
                "source_artifact_count": len(source_artifacts),
                "input_artifact_count": len(input_artifacts),
                "return_outcomes_read_after_freeze": False,
                "goal_achieved": False,
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
