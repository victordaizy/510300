"""510300宏观压力规避：2015起点的事后窗口敏感性版本。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from research import macro_stress_avoidance_v1 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_macro_stress_avoidance_2015_v2.yaml"
STUDY_ID = "510300_MACRO_STRESS_AVOIDANCE_2015_START_V2"
EVIDENCE_CLASS = "POST_REJECTION_WINDOW_SENSITIVITY_ONLY"


sha256_file = base.sha256_file
json_default = base.json_default


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML顶层必须为对象：{path}")
    return payload


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """解析只允许变更起点、输入路径和输出路径的继承式协议。"""

    overlay = _load_yaml(path)
    inheritance = overlay["inheritance"]
    base_path = ROOT / inheritance["base_protocol_file"]
    if sha256_file(base_path) != inheritance["base_protocol_required_sha256"]:
        raise ValueError("V1基础协议哈希漂移")
    predecessor_path = ROOT / inheritance["predecessor_result_file"]
    if sha256_file(predecessor_path) != inheritance["predecessor_result_required_sha256"]:
        raise ValueError("2021起点的前序结果哈希漂移")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("decision") != inheritance["predecessor_required_decision"]:
        raise ValueError("2021起点的前序拒绝状态不匹配")

    config = copy.deepcopy(_load_yaml(base_path))
    config["protocol"].update(copy.deepcopy(overlay["protocol_overrides"]))
    config["data_contracts"] = copy.deepcopy(overlay["data_contract_overrides"])
    config["artifacts"] = copy.deepcopy(overlay["artifact_overrides"])
    config["governance"].update(copy.deepcopy(overlay["governance_overrides"]))
    config["sensitivity_governance"] = copy.deepcopy(
        overlay["sensitivity_governance"]
    )
    config["inheritance"] = copy.deepcopy(inheritance)

    protocol = config["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("2015敏感性研究编号不匹配")
    if protocol["data_start"] != "2015-01-01":
        raise ValueError("2015敏感性研究的数据起点必须为2015-01-01")
    if protocol["evidence_class"] != EVIDENCE_CLASS:
        raise ValueError("事后窗口敏感性证据等级不匹配")
    if protocol["allowed_assets"] != ["510300.SH", "CASH_CNY"]:
        raise ValueError("可用资产必须严格为510300与人民币现金")
    if int(protocol["main_trailing_years"]) != 5:
        raise ValueError("主模型必须保持严格5自然年窗口")
    if int(protocol["robustness_trailing_years"]) != 3:
        raise ValueError("稳健性模型必须保持严格3自然年窗口")
    if float(protocol["percentile"]) != 0.90:
        raise ValueError("分位数必须保持90%")
    if not bool(config["point_in_time"]["expanding_window_forbidden"]):
        raise ValueError("禁止扩展窗口")
    if not bool(config["point_in_time"]["shortened_window_forbidden"]):
        raise ValueError("禁止缩短窗口")
    sensitivity = config["sensitivity_governance"]
    if sensitivity["original_2021_rejection_overridden"] is not False:
        raise ValueError("2015敏感性研究不得覆盖2021起点的拒绝结论")
    if sensitivity["eligible_to_authorize_paper_or_live_trading"] is not False:
        raise ValueError("事后窗口敏感性研究不得授权交易")
    if bool(config["governance"]["live_trading_authorized"]):
        raise ValueError("实盘授权必须关闭")
    return config


def _sensitivity_decision(base_decision: str) -> str:
    if base_decision == "HISTORICAL_CANDIDATE_PASS":
        return (
            "POST_REJECTION_WINDOW_SENSITIVITY_HISTORICAL_GATES_PASS_"
            "NO_CANDIDATE_OR_TRADE_AUTHORIZATION"
        )
    if base_decision == "REJECTED_FROZEN_INSUFFICIENT_EVENT_SUPPORT_NO_RESCUE":
        return (
            "REJECTED_2015_START_SENSITIVITY_INSUFFICIENT_EVENT_SUPPORT_"
            "NO_RESCUE"
        )
    return (
        "REJECTED_2015_START_SENSITIVITY_HISTORICAL_OR_ROBUSTNESS_GATE_"
        "FAILED_NO_RESCUE"
    )


def evaluate_protocol(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """运行继承的冻结算法，并把证据等级与前序拒绝边界写入结果。"""

    report, artifacts = base.evaluate_protocol(config)
    mechanism_decision = str(report["decision"])
    report["decision"] = _sensitivity_decision(mechanism_decision)
    report["mechanism_decision_before_evidence_class"] = mechanism_decision
    report["evidence_class"] = EVIDENCE_CLASS
    report["data_scope"].update(
        {
            "start": "2015-01-01",
            "end": "2026-08-25",
            "pre_2021_data_used": True,
            "window_change_proposed_after_2021_rejection": True,
        }
    )
    report["predecessor"] = {
        "study_id": "510300_MACRO_STRESS_AVOIDANCE_V1",
        "decision": config["inheritance"]["predecessor_required_decision"],
        "result_file": config["inheritance"]["predecessor_result_file"],
        "result_sha256": config["inheritance"][
            "predecessor_result_required_sha256"
        ],
        "overridden": False,
    }
    report["sensitivity_governance"] = copy.deepcopy(
        config["sensitivity_governance"]
    )
    report["historical_sharpe_threshold_met"] = bool(
        report["portfolio_backtest"].get("main_gates", {}).get(
            "five_year_net_sharpe_at_least_1_20", False
        )
    )
    report["goal_achieved"] = False
    report["goal_achieved_reason"] = (
        "事后改变样本起点的敏感性结果不能覆盖冻结拒绝或构成独立确认"
    )
    report["position_mapping_enabled"] = False
    report["order_generation_enabled"] = False
    report["broker_connection_enabled"] = False
    report["live_trading_authorized"] = False
    return report, artifacts


def _number(value: Any, decimals: int = 6) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    return f"{float(value):.{decimals}f}"


def _percent(value: Any) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    return f"{float(value):.6%}"


def markdown_report(report: dict[str, Any]) -> str:
    """生成2015起点敏感性报告，明确不覆盖原拒绝结论。"""

    event = report["event_study"]
    lines = [
        "# 510300宏观压力规避：2015起点滚动敏感性结果",
        "",
        f"- 最终状态：`{report['decision']}`",
        f"- 证据等级：`{report['evidence_class']}`",
        f"- 收益评估：`{report['return_evaluation']}`",
        f"- 组合回测：`{report['portfolio_backtest_status']}`",
        "- 数据范围：2015-01-01至2026-08-25",
        "- 资产边界：仅510300.SH与人民币现金",
        "- 2021起点原拒绝是否被覆盖：`false`",
        "- 实盘授权：`false`",
        "",
        "## 严格滚动窗口可用性",
        "",
        f"- 3年模型首个有效信号日：{report['data_scope']['first_model_eligible_dates']['3y']}",
        f"- 5年模型首个有效信号日：{report['data_scope']['first_model_eligible_dates']['5y']}",
        "- 当期观察不进入90%分位；未使用扩展窗口或缩短窗口",
        "",
        "## 5年主模型事件门",
        "",
        f"- 有效交易日：{event['eligible_trading_days']}",
        f"- 原始事件起点：{event['raw_onsets']}",
        f"- 间隔20交易日后的独立事件：{event['independent_onsets']}",
        f"- 具有完整20日结果的独立事件：{event['independent_complete_primary_events']}",
        f"- 事件门状态：`{event['decision']}`",
        "",
        "| 事件硬门 | 通过 |",
        "|---|---:|",
    ]
    for name, passed in event["gates"].items():
        lines.append(f"| {name} | {'是' if passed else '否'} |")
    lines.extend(["", "## 组合层", ""])
    if report["return_evaluation"] == "NOT_ALLOWED":
        lines.extend(
            [
                "事件门未通过，因此5年、3年和双倍成本组合回测均未运行；",
                "此处没有净夏普率、净超额收益或最大回撤，NOT_ALLOWED不能解释为零收益。",
            ]
        )
    else:
        portfolio = report["portfolio_backtest"]
        five = portfolio["five_year"]
        three = portfolio["three_year"]
        double = portfolio["five_year_double_cost"]
        lines.extend(
            [
                f"- 5年净夏普率：{_number(five['strategy']['sharpe_zero_cash_rate'])}",
                f"- 5年相对H00300年化超额：{_percent(five['annualized_excess_vs_h00300'])}",
                f"- 5年相对510300含分红买入持有年化超额：{_percent(five['annualized_excess_vs_buy_hold'])}",
                f"- 3年净夏普率：{_number(three['strategy']['sharpe_zero_cash_rate'])}",
                f"- 5年双倍成本净夏普率：{_number(double['strategy']['sharpe_zero_cash_rate'])}",
                f"- 组合机制判定：`{portfolio['decision']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 证据与治理边界",
            "",
            "- 本次窗口变化是在看到2021起点事件不足后提出，只属于事后窗口敏感性。",
            "- 即使历史硬门通过，也不能把它重标为预注册确认，不能覆盖原拒绝结论。",
            "- 参数营救、真实账户仓位映射、订单生成、券商连接与实盘授权均关闭。",
            "- `goal_achieved=false`：需要独立冻结前向证据，而非再改变历史窗口。",
            "",
        ]
    )
    return "\n".join(lines)
