"""在不改写V1.0冻结证据的前提下重跑价格补采后的点时估值审计。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.point_in_time_valuation_v2_audit import (
    build_snapshot_coverage,
    load_config as load_base_config,
    sha256_file,
    summarize_history_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "point_in_time_valuation_v2_audit_v1_1.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for field in (
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
    ):
        if config["protocol"].get(field):
            raise ValueError(f"后补采审计禁止启用{field}")
    return config


def verify_hashes(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for name, contract in config["data_contracts"].items():
        path = ROOT / contract["file"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "dataset": name,
                "file": contract["file"],
                "expected_sha256": contract["sha256"],
                "actual_sha256": actual,
                "matches": actual == contract["sha256"],
            }
        )
    return {
        "status": "PASS" if all(row["matches"] for row in rows) else "BLOCKED_HASH_DRIFT",
        "rows": rows,
    }


def combined_price_input(snapshot_prices: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """早期优先使用专用快照表，后期继续使用当前点时成员日线。"""

    early = snapshot_prices[["date", "con_code", "raw_close"]].copy()
    later = current[["date", "con_code", "raw_close"]].copy()
    for frame in (early, later):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    combined = pd.concat([later, early], ignore_index=True)
    return combined.sort_values(["date", "con_code"]).drop_duplicates(
        ["date", "con_code"], keep="last"
    ).reset_index(drop=True)


def run_post_acquisition_audit(
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = config or load_config()
    hashes = verify_hashes(config)
    if hashes["status"] != "PASS":
        raise RuntimeError("V1.1输入哈希偏离配置")
    contracts = config["data_contracts"]
    read = lambda name: pd.read_parquet(ROOT / contracts[name]["file"])
    prior_report = json.loads(
        (ROOT / contracts["prior_audit_report"]["file"]).read_text(encoding="utf-8")
    )
    acquisition = json.loads(
        (ROOT / contracts["price_acquisition_report"]["file"]).read_text(encoding="utf-8")
    )
    if acquisition["status"] != "PASS_FIVE_YEAR_SNAPSHOT_PRICES":
        raise RuntimeError("快照价格补采报告未通过")
    weights = read("historical_weights")
    financials = read("point_in_time_financials")
    current = read("current_constituent_daily")
    snapshot_prices = read("snapshot_prices")
    index_daily = read("index_daily")
    bonds = read("government_bond_yields")
    base_config = load_base_config()
    coverage = build_snapshot_coverage(
        weights,
        financials,
        combined_price_input(snapshot_prices, current),
        index_daily,
        bonds,
        base_config,
    )
    history = summarize_history_gate(coverage, base_config)
    five = history["five_year"]
    if five["raw_ey_status"] != "PASS_LOCAL_RECONSTRUCTIBLE":
        raise RuntimeError(f"补采后五年原始EY仍未通过：{five}")
    report = deepcopy(prior_report)
    report["version"] = config["protocol"]["version"]
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["overall_status"] = "RAW_EY_5Y_DATA_GATE_PASS_NORMALIZED_MODELS_STILL_BLOCKED"
    report["formal_valuation_run_status"] = "NO_VIEW_EXCEPT_RAW_EY_MODEL_CARD_MAY_NOW_BE_FROZEN"
    report["hash_audit_v1_1"] = hashes
    report["history_gates"] = history
    report["price_acquisition"] = acquisition
    report["branch_status"] = {
        "VAL01_RAW_EY_5Y": "PASS_LOCAL_RECONSTRUCTIBLE_DATA_ONLY",
        "VAL01_NORM_EY_5Y": five["normalized_ey_status"],
        "VAL02_NORM_EY_SPREAD_5Y": five["normalized_ey_spread_status"],
        "VAL01_RAW_EY_7Y": history["seven_year"]["raw_ey_status"],
        "VAL01_NORM_EY_7Y": history["seven_year"]["normalized_ey_status"],
        "VAL02_NORM_EY_SPREAD_7Y": history["seven_year"]["normalized_ey_spread_status"],
    }
    report["completed_acquisition"] = {
        "gate": "VAL01_RAW_EY_5Y_PRICE_HISTORY",
        "status": "PASS",
        "snapshot_count": acquisition["snapshot_validation"]["snapshot_count"],
        "minimum_price_weight_coverage": acquisition["snapshot_validation"][
            "minimum_price_weight_coverage"
        ],
        "missing_price_row_count": acquisition["snapshot_validation"]["missing_price_row_count"],
        "future_price_row_count": acquisition["snapshot_validation"]["future_price_row_count"],
        "cross_source_exact_match_ratio": acquisition["cross_source_overlap"]["exact_match_ratio"],
    }
    report["remaining_blockers"] = [
        "2016-08至2017-04标准化盈利权重覆盖低于90%",
        "七年版缺2014-08至2016-07共24个月权重、价格、财务和国债历史",
        "正式收益检验前仍须冻结VAL01_RAW_EY_5Y模型卡、月度信号可得时点和扩展窗口规则",
    ]
    report["governance"] = {
        "return_calculation_performed": False,
        "ic_calculation_performed": False,
        "position_mapping_performed": False,
        "order_generation_performed": False,
        "broker_connection_performed": False,
        "prior_v1_0_files_mutated": False,
    }
    report["artifacts"] = config["artifacts"]
    return report, coverage


def render_markdown(report: dict[str, Any]) -> str:
    five = report["history_gates"]["five_year"]
    price = report["price_acquisition"]["snapshot_validation"]
    return "\n".join(
        [
            "# 沪深300点时估值V2可重建性审计 V1.1",
            "",
            "> 价格补采后，`VAL01_RAW_EY_5Y`数据闸门已经通过；标准化EY、EY利差和全部七年版本仍然NO_VIEW。本报告不包含收益、IC或仓位结果。",
            "",
            "## 已解除的硬缺口",
            "",
            f"- 官方权重快照价格：{price['snapshot_count']}个月、{price['row_count']}行。",
            f"- 最低价格权重覆盖：{price['minimum_price_weight_coverage']:.4%}。",
            f"- 缺价/未来取价：{price['missing_price_row_count']}/{price['future_price_row_count']}行。",
            f"- 与当前面板重叠收盘价完全一致率：{report['price_acquisition']['cross_source_overlap']['exact_match_ratio']:.4%}。",
            "",
            "## 分支状态",
            "",
            f"- `VAL01_RAW_EY_5Y`：`{report['branch_status']['VAL01_RAW_EY_5Y']}`。",
            f"- `VAL01_NORM_EY_5Y`：`{report['branch_status']['VAL01_NORM_EY_5Y']}`；仍有{len(five['normalized_coverage_failed_months'])}个月不足。",
            f"- `VAL02_NORM_EY_SPREAD_5Y`：`{report['branch_status']['VAL02_NORM_EY_SPREAD_5Y']}`。",
            f"- 七年原始/标准化/利差：`{report['branch_status']['VAL01_RAW_EY_7Y']}`。",
            "",
            "## 允许的下一步",
            "",
            "现在只允许冻结`VAL01_RAW_EY_5Y`模型卡，包括原始EY公式、亏损公司处理、月度权重可得时点、分位窗口、信号频率和试验登记。模型卡冻结前仍不运行未来收益、IC或仓位回测。",
            "",
        ]
    )


def write_artifacts(report: dict[str, Any], coverage: pd.DataFrame, config: dict[str, Any]) -> None:
    artifacts = config["artifacts"]
    coverage_path = ROOT / artifacts["snapshot_coverage"]
    json_path = ROOT / artifacts["report_json"]
    markdown_path = ROOT / artifacts["report_markdown"]
    for path in (coverage_path, json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_coverage = coverage_path.with_suffix(coverage_path.suffix + ".tmp")
    coverage.to_parquet(temp_coverage, index=False)
    temp_coverage.replace(coverage_path)
    report["artifacts"]["snapshot_coverage_sha256"] = sha256_file(coverage_path)
    temp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temp_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_json.replace(json_path)
    temp_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temp_markdown.write_text(render_markdown(report), encoding="utf-8")
    temp_markdown.replace(markdown_path)

