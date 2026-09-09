"""510300“可预测才进入”最终决策层。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


class PredictableDecisionError(ValueError):
    """最终决策层输入或冻结契约不成立。"""


PRIMARY_STATE_GROUPS = (
    "fish_middle",
    "qualitative_leads",
    "fish_tail",
    "negative",
    "mixed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent_hashes(
    root: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    """校验所有冻结父输入，防止结论随文件漂移。"""

    actual_hashes: dict[str, str] = {}
    for key, relative_path in config["parents"].items():
        path = root / str(relative_path)
        if not path.exists():
            raise PredictableDecisionError(f"父输入缺失：{relative_path}")
        actual = sha256(path)
        expected = str(config["parent_integrity"][key])
        if actual != expected:
            raise PredictableDecisionError(
                f"父输入哈希变化：{key}，{actual} != {expected}"
            )
        actual_hashes[key] = actual
    return actual_hashes


def aggregate_exclusive_sector_states(
    rows: Sequence[Mapping[str, Any]], state_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """按互斥经济大类权重汇总研究状态，不处理重叠主题。"""

    if not rows:
        raise PredictableDecisionError("互斥经济大类为空")
    entity_ids = [str(row["entity_id"]) for row in rows]
    if len(entity_ids) != len(set(entity_ids)):
        raise PredictableDecisionError("互斥经济大类存在重复实体")

    state_to_group: dict[str, str] = {}
    for group in PRIMARY_STATE_GROUPS:
        for state in state_contract[group]:
            state = str(state)
            if state in state_to_group:
                raise PredictableDecisionError(f"研究状态重复归组：{state}")
            state_to_group[state] = group

    weights = {group: 0.0 for group in PRIMARY_STATE_GROUPS}
    entities = {group: [] for group in PRIMARY_STATE_GROUPS}
    mapped_weight = 0.0
    for row in rows:
        if str(row["entity_type"]) != "ECONOMIC_BUCKET":
            raise PredictableDecisionError("指数权重汇总只允许互斥经济大类")
        weight = float(row["current_index_weight"])
        if not 0.0 <= weight <= 1.0:
            raise PredictableDecisionError(f"非法指数权重：{row['entity_id']}={weight}")
        state = str(row["integrated_forward_state"])
        if state not in state_to_group:
            raise PredictableDecisionError(f"研究状态未归组：{state}")
        group = state_to_group[state]
        weights[group] += weight
        mapped_weight += weight
        entities[group].append(
            {
                "entity_id": str(row["entity_id"]),
                "entity_name_cn": str(row["entity_name_cn"]),
                "weight": weight,
            }
        )

    if not 0.90 <= mapped_weight <= 1.02:
        raise PredictableDecisionError(f"互斥经济大类权重覆盖异常：{mapped_weight}")
    positive_structure_weight = weights["fish_middle"] + weights[
        "qualitative_leads"
    ]
    return {
        "mapped_index_weight": mapped_weight,
        "unmapped_index_weight": max(0.0, 1.0 - mapped_weight),
        "fish_middle_weight": weights["fish_middle"],
        "qualitative_leads_weight": weights["qualitative_leads"],
        "positive_structure_weight": positive_structure_weight,
        "fish_tail_watch_weight": weights["fish_tail"],
        "negative_convergence_weight": weights["negative"],
        "mixed_no_view_weight": weights["mixed"],
        "positive_structure_exceeds_negative_weight": (
            positive_structure_weight > weights["negative"]
        ),
        "entities": entities,
        "interpretation": (
            "各状态权重是指数结构描述，不是预期收益、概率或仓位分数。"
        ),
    }


def build_reliability_evidence(
    sector_prediction: Mapping[str, Any],
    driver_episode: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """抽取已经冻结的 S1、M1、T1 否证证据。"""

    s1 = sector_prediction["evaluation"]
    m1 = driver_episode["evaluation"]["fish_middle"]
    t1 = driver_episode["evaluation"]["tail_warning"]
    return [
        {
            "candidate_id": str(s1["candidate_id"]),
            "role": "板块未来贡献排序",
            "status": str(s1["status"]),
            "sample_count": int(s1["mature_oos_predictions"]),
            "primary_metric_name": "direction_accuracy",
            "primary_metric": float(s1["direction_accuracy"]),
            "required_metric": ">=0.55且其他门槛同时通过",
            "forecast_eligible": bool(
                s1["safety"]["forecast_eligible_emitted"]
            ),
        },
        {
            "candidate_id": str(m1["candidate_id"]),
            "role": "鱼中传播确认",
            "status": str(m1["status"]),
            "sample_count": int(m1["mature_event_count"]),
            "primary_metric_name": "MIDDLE20_lift",
            "primary_metric": float(m1["lift"]),
            "required_metric": ">=1.20且Bootstrap下界>1",
            "forecast_eligible": bool(
                driver_episode["evaluation"]["safety"][
                    "forecast_eligible_emitted"
                ]
            ),
        },
        {
            "candidate_id": str(t1["candidate_id"]),
            "role": "鱼尾风险预警",
            "status": str(t1["status"]),
            "sample_count": int(t1["mature_event_count"]),
            "primary_metric_name": "TAIL20_lift",
            "primary_metric": float(t1["lift"]),
            "required_metric": ">=2.00且Bootstrap下界>1",
            "forecast_eligible": False,
        },
    ]


def build_sector_statistics(
    rows: Sequence[Mapping[str, Any]], *, include_weight: bool
) -> list[dict[str, Any]]:
    """保留胜率、赔率、频率的原始含义，不生成条件概率。"""

    output: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "entity_type": str(row["entity_type"]),
            "entity_id": str(row["entity_id"]),
            "entity_name_cn": str(row["entity_name_cn"]),
            "current_score": float(row["current_score"]),
            "integrated_forward_state": str(row["integrated_forward_state"]),
            "price_phase": str(row["price_phase"]),
            "wechat_forward_direction": str(row["wechat_forward_direction"]),
            "wechat_confidence": str(row["wechat_confidence"]),
            "historical_odds_axis": str(row["historical_odds_axis"]),
            "descriptive_overlapping_win_rate": float(
                row["overlapping_relative_win_rate"]
            ),
            "independent_chain_min_win_rate": float(
                row["nonoverlap_cohort_win_rate_min"]
            ),
            "average_payoff_ratio": float(row["average_payoff_ratio"]),
            "breakeven_win_rate": float(row["breakeven_win_rate"]),
            "profit_factor": float(row["profit_factor"]),
            "mean_excess_return_60d": float(row["mean_excess_return_60d"]),
            "independent_wins_per_year": float(row["independent_wins_per_year"]),
            "score_conditioned_win_rate": None,
            "conditional_odds": None,
            "position_mapping_enabled": False,
        }
        if include_weight:
            item["current_index_weight"] = float(row["current_index_weight"])
        output.append(item)
    return output


def evaluate_entry_gates(
    config: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    expectation_gap: Mapping[str, Any],
    sector_prediction: Mapping[str, Any],
    driver_episode: Mapping[str, Any],
    daily_quality: Mapping[str, Any],
    forward_refresh: Mapping[str, Any],
    etf_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """逐门判断当前是否具备研究层面的可预测进入资格。"""

    contract = config["entry_gate_contract"]
    required_date = str(contract["data_freshness"]["required_latest_date"])
    data_checks = {
        "daily_quality_pass": daily_quality["status"]
        == contract["data_freshness"]["required_daily_quality_status"],
        "daily_latest_current": str(daily_quality["actual_last_date"]) == required_date,
        "forward_refresh_success": forward_refresh["status"]
        == contract["data_freshness"]["required_forward_refresh_status"],
        "index_market_current": str(forward_refresh["steps"]["market"]["510300_latest"])
        == required_date,
        "valuation_current": str(
            forward_refresh["steps"]["valuation_and_bonds"]["valuation_latest"]
        )
        == required_date,
        "constituent_prices_current": str(
            forward_refresh["steps"]["constituent_closes"][
                "constituent_close_date"
            ]
        )
        == required_date,
    }
    data_gate = all(data_checks.values())

    aggregation = expectation_gap["industry_aggregation"]
    expectation_gap_gate = (
        str(aggregation["state"])
        == str(contract["expectation_gap"]["required_state"])
    )

    s1_eligible = bool(
        sector_prediction["evaluation"]["safety"]["forecast_eligible_emitted"]
    )
    m1_eligible = bool(
        driver_episode["evaluation"]["safety"]["forecast_eligible_emitted"]
    )
    predictability_gate = s1_eligible or m1_eligible

    quantitative_liquidity_state = str(
        expectation_gap["market_liquidity_and_tail"]["gate_state"]
    )
    qualitative_liquidity_state = str(
        scorecard["market_forward_context"]["liquidity_state"]
    )
    liquidity_gate = (
        quantitative_liquidity_state
        == str(contract["liquidity"]["required_quantitative_gate_state"])
        and qualitative_liquidity_state
        != "MIXED_INTERNAL_RECOVERY_EXTERNAL_OUTFLOW"
    )

    tail_status = str(
        driver_episode["evaluation"]["tail_warning"]["status"]
    )
    tail_gate = tail_status in {"TRUE_FORWARD_PASS", "FORECAST_ELIGIBLE"}

    national_team_state = str(
        scorecard["market_forward_context"]["national_team_holdings_state"]
    )
    national_team_gate = national_team_state != str(
        contract["national_team"]["missing_holdings_state"]
    )

    minimum_days = int(contract["etf_execution"]["minimum_full_coverage_days"])
    execution_gate = bool(etf_readiness["eligible_for_research_evaluation"]) and int(
        etf_readiness["full_coverage_days"]
    ) >= minimum_days

    gates = {
        "G1_DATA_FRESHNESS": {
            "passed": data_gate,
            "checks": data_checks,
            "evidence": f"510300、估值和成分价格最新日={required_date}",
        },
        "G2_INDEX_EXPECTATION_GAP": {
            "passed": expectation_gap_gate,
            "state": str(aggregation["state"]),
            "net_gap_weighted_contribution": float(
                aggregation["net_gap_weighted_contribution"]
            ),
            "required_absolute_contribution": float(
                contract["expectation_gap"]["inherited_direction_threshold"]
            ),
            "observed_weight": float(
                aggregation["expectation_gap_observed_index_weight"]
            ),
        },
        "G3_TRUE_FORWARD_PREDICTABILITY": {
            "passed": predictability_gate,
            "sector_prediction_forecast_eligible": s1_eligible,
            "fish_middle_forecast_eligible": m1_eligible,
        },
        "G4_MARKET_LIQUIDITY": {
            "passed": liquidity_gate,
            "quantitative_state": quantitative_liquidity_state,
            "qualitative_state": qualitative_liquidity_state,
        },
        "G5_TAIL_AND_NATIONAL_TEAM": {
            "passed": tail_gate and national_team_gate,
            "validated_index_tail_model": tail_gate,
            "tail_model_status": tail_status,
            "national_team_evidence_available": national_team_gate,
            "national_team_state": national_team_state,
        },
        "G6_ETF_EXECUTION_RESEARCH_READINESS": {
            "passed": execution_gate,
            "status": str(etf_readiness["status"]),
            "observed_trading_days": int(etf_readiness["observed_trading_days"]),
            "full_coverage_days": int(etf_readiness["full_coverage_days"]),
            "minimum_full_coverage_days": minimum_days,
        },
    }

    blockers: list[dict[str, str]] = []
    blocker_text = {
        "G1_DATA_FRESHNESS": "数据时点或质量未通过",
        "G2_INDEX_EXPECTATION_GAP": "指数净预期差未越过冻结方向门槛",
        "G3_TRUE_FORWARD_PREDICTABILITY": "没有进入模型取得真正前瞻预测资格",
        "G4_MARKET_LIQUIDITY": "量化流动性不完整且定性流动性仅为混合",
        "G5_TAIL_AND_NATIONAL_TEAM": "指数鱼尾模型未验证且国家队点时持仓不可审计",
        "G6_ETF_EXECUTION_RESEARCH_READINESS": "ETF前瞻全覆盖样本尚未达到研究门槛",
    }
    for gate_id, result in gates.items():
        if not result["passed"]:
            blockers.append({"gate_id": gate_id, "reason": blocker_text[gate_id]})

    entry_research_eligible = all(result["passed"] for result in gates.values())
    return {
        "gates": gates,
        "blockers": blockers,
        "all_research_entry_gates_passed": entry_research_eligible,
        "current_research_view": "SECTOR_STRUCTURE_POSITIVE_NOT_INDEX_FORECAST",
        "current_entry_state": (
            "SHADOW_ENTRY_ELIGIBLE_REQUIRES_SEPARATE_POSITION_PROTOCOL"
            if entry_research_eligible
            else "WAIT_NO_PREDICTIVE_ENTRY"
        ),
        "current_action": "WAIT_NO_NEW_ENTRY",
        "existing_holding_action": "NOT_EVALUATED_NO_POSITION_DATA_READ",
    }


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染最终中文决策报告。"""

    structure = report["index_sector_structure"]
    decision = report["final_decision"]
    lines = [
        "# 510300“可预测才进入”最终决策",
        "",
        f"- 截止日：`{report['as_of_date']}`",
        f"- 模型闭环：`{report['completion_state']}`",
        f"- 当前研究判断：`{decision['current_research_view']}`",
        f"- 当前进入状态：`{decision['current_entry_state']}`",
        f"- 当前动作：`{decision['current_action']}`",
        "- 结论：行业结构偏正，但指数预期差、可预测性、流动性、鱼尾验证、国家队证据和ETF前瞻覆盖没有共同通过，因此不进入。",
        "",
        "## 从复杂到简单",
        "",
        "| 层 | 当前结果 | 是否通过 |",
        "|---|---|---:|",
    ]
    for gate_id, gate in decision["gates"].items():
        if gate_id == "G1_DATA_FRESHNESS":
            detail = gate["evidence"]
        elif gate_id == "G2_INDEX_EXPECTATION_GAP":
            detail = (
                f"{gate['state']}；净预期差{gate['net_gap_weighted_contribution']:.2%}，"
                f"门槛{gate['required_absolute_contribution']:.0%}"
            )
        elif gate_id == "G3_TRUE_FORWARD_PREDICTABILITY":
            detail = "S1与M1均未取得预测资格"
        elif gate_id == "G4_MARKET_LIQUIDITY":
            detail = (
                f"量化={gate['quantitative_state']}；定性={gate['qualitative_state']}"
            )
        elif gate_id == "G5_TAIL_AND_NATIONAL_TEAM":
            detail = (
                f"鱼尾={gate['tail_model_status']}；国家队={gate['national_team_state']}"
            )
        elif gate_id == "G6_ETF_EXECUTION_RESEARCH_READINESS":
            detail = (
                f"完整覆盖{gate['full_coverage_days']}/{gate['minimum_full_coverage_days']}日"
            )
        else:
            raise PredictableDecisionError(f"未知决策门：{gate_id}")
        lines.append(
            f"| {gate_id} | {markdown_cell(detail)} | {'是' if gate['passed'] else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 指数权重结构",
            "",
            "| 状态 | 指数权重 | 解释 |",
            "|---|---:|---|",
            f"| 鱼中共振 | {structure['fish_middle_weight']:.2%} | 当前量化、历史赔率与微信前瞻一致 |",
            f"| 定性领先定量 | {structure['qualitative_leads_weight']:.2%} | 先进制造前瞻改善，但当前量化尚未进入正向档 |",
            f"| 正向结构合计 | {structure['positive_structure_weight']:.2%} | 只描述权重，不是指数收益预测 |",
            f"| 鱼尾观察 | {structure['fish_tail_watch_weight']:.2%} | 当前为医疗健康，不是已验证指数鱼尾 |",
            f"| 负向共振 | {structure['negative_convergence_weight']:.2%} | 经营和量化证据共同偏弱 |",
            f"| 混合/无硬观点 | {structure['mixed_no_view_weight']:.2%} | 证据分化 |",
            f"| 未映射 | {structure['unmapped_index_weight']:.2%} | 冻结权重未覆盖部分 |",
            "",
            "## 当前阻断项",
            "",
        ]
    )
    for blocker in decision["blockers"]:
        lines.append(f"- `{blocker['gate_id']}`：{blocker['reason']}。")

    lines.extend(
        [
            "",
            "## 互斥经济大类：分数、胜率、赔率与频率",
            "",
            "| 板块 | 权重 | 三轴状态 | 当前分 | 历史胜率* | 非重叠最低胜率 | 赔率 | 盈亏平衡胜率 | 独立胜出/年 | 60日历史超额 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["economic_sectors"]:
        lines.append(
            f"| {markdown_cell(row['entity_name_cn'])} | {row['current_index_weight']:.2%} | "
            f"{row['integrated_forward_state']} | {row['current_score']:.1f} | "
            f"{row['descriptive_overlapping_win_rate']:.1%} | "
            f"{row['independent_chain_min_win_rate']:.1%} | "
            f"{row['average_payoff_ratio']:.2f} | {row['breakeven_win_rate']:.1%} | "
            f"{row['independent_wins_per_year']:.2f} | "
            f"{row['mean_excess_return_60d']:.2%} |"
        )
    lines.extend(
        [
            "",
            "说明：历史胜率为重叠60日窗口描述频率，不是当前条件胜率。",
            "",
            "## 重叠主题（不重复计入指数权重）",
            "",
            "| 主题 | 三轴状态 | 当前分 | 微信方向 | 历史赔率轴 | 赔率 | 独立胜出/年 |",
            "|---|---|---:|---|---|---:|---:|",
        ]
    )
    for row in report["themes"]:
        lines.append(
            f"| {markdown_cell(row['entity_name_cn'])} | {row['integrated_forward_state']} | "
            f"{row['current_score']:.1f} | {row['wechat_forward_direction']} | "
            f"{row['historical_odds_axis']} | {row['average_payoff_ratio']:.2f} | "
            f"{row['independent_wins_per_year']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 为什么现在不能称为可预测",
            "",
            "| 候选 | 作用 | 样本 | 主指标 | 预设要求 | 结果 |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in report["historical_predictability_evidence"]:
        lines.append(
            f"| {row['candidate_id']} | {row['role']} | {row['sample_count']} | "
            f"{row['primary_metric']:.3f} | {row['required_metric']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 最终边界",
            "",
            "- 当前不进入不是看空，也不自动改变已有持仓；系统没有读取用户持仓。",
            "- 当前分数和微信方向不是概率；条件胜率与条件赔率仍为`UNAVAILABLE_AWAITING_TRUE_FORWARD`。",
            "- 不再用同一历史增加指标或调整阈值。下一步只积累冻结后的20/60/120日结果。",
            "- 即使未来六道研究门全部通过，也只能得到`SHADOW_ENTRY_ELIGIBLE`；仓位与实盘必须另立协议并由用户明确授权。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "PredictableDecisionError",
    "aggregate_exclusive_sector_states",
    "build_reliability_evidence",
    "build_sector_statistics",
    "evaluate_entry_gates",
    "render_markdown",
    "sha256",
    "validate_parent_hashes",
]
