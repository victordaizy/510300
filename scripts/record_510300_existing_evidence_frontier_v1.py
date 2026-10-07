"""记录本次目标续轮的真实缺口；不训练、不采集、不累计工具调用为目标轮次。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
CHECK = MAIN / "continuation_checks/check_01.json"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_extent(relative, choices):
    path = ROOT / relative
    names = pq.read_schema(path).names
    field = next(name for name in choices if name in names)
    values = pd.read_parquet(path, columns=[field])[field].dropna()
    dates = pd.to_datetime(values.astype(str), errors="raise", utc=True)
    return {"path": relative, "field": field, "rows": len(values), "maximum": str(dates.max()), "file_modified_ns": path.stat().st_mtime_ns}


def main():
    if CHECK.exists():
        raise RuntimeError("本目标轮次已经记录，不能用重复调用增加连续轮次数。")
    status = json.loads((MAIN / "current_status.json").read_text(encoding="utf-8"))
    assert status["same_condition_consecutive_no_progress_goal_turns"] == 0
    assert status["latest_goal_turn_classification"] == "PROGRESS_FROZEN_RR3_ALL_CANDIDATE_REALIZATION_DIAGNOSTIC"
    mandate = read("config/510300_existing_data_training_mandate_v1.json")
    assert mandate["minimum_planned_net_reward_risk_ratio"] == 3
    assert mandate["new_market_data_collection_enabled"] is False
    assert mandate["target_net_sharpe"] == 1.2
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    routes = [
        ("估值回归与市场状态", "reports/research/000300_valuation_v3_state_aware_return.json", "已有后验状态诊断，依赖性和时序门槛未通过；不能把FairPE直接当已验证的目标。"),
        ("正常化估值", "reports/research/normalized_valuation_5y_final_audit_v1.json", "已登记五年模型被拒绝；七年输入需要额外历史资料，不能改窗口救回旧模型。"),
        ("PB-ROE残差", "reports/research/val04_pb_roe_residual_final_audit_v1.json", "预测稳定性与分组单调性未通过，终止在账户之前。"),
        ("盈利与可选估值", "reports/research/510300_forward_eps_optional_valuation_v1/result.json", "完整账户已计算，主压力夏普0.633，未建立独立增量；旧成本和区间不能与本轮拼接。"),
        ("估值口径一致性", "reports/research/510300_forward_eps_valuation_consistency_policy_v2/result.json", "240次拟合及账户已完成，主压力夏普0.353，历史独立验证未建立。"),
        ("支持政策公告", "reports/research/510300_equity_support_policy_event_v1/summary.json", "已有六个公告日期的固定规则账户；协议明确仅本地两年目录，缺完整同口径公告母集。"),
        ("公告后资金调整", "reports/research/510300_post_information_capital_adjustment_v1/status.json", "104个月缺98组事前预期配对，基金份额缺日级首次发布时间。其旧M1线后续训练已做，不把旧NOT_RUN误认为尚有可运行模型。"),
        ("收盘价格让步", "reports/research/510300_close_concession_source_feasibility_v1/analysis_result.json", "合格收盘和盘后观测均为零，午休样例不能补成同步收盘估值与成交证据。"),
        ("私募意向与可加仓空间", "reports/research/510300_private_manager_exploratory_training_v1/summary.json", "104次逐期拟合与20条账户已完成，真实前向事件为零。"),
        ("事前3:1目标", "reports/research/510300_rr3_target_realization_diagnostic_v1/result.json", "全部15个合格候选、两成本30路径已经完成；没有目标退出，经济支持对照缺一侧。"),
    ]
    evidence = []
    for route, relative, limitation in routes:
        obj = read(relative)
        evidence.append({"route": route, "path": relative, "status": obj.get("status"), "limitation": limitation,
                         "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()})
    coverage = [
        source_extent("reports/research/510300_sequential_patterns_regime_v1/inputs/prices.parquet", ["date", "trade_date"]),
        source_extent("data/features/000300_official_weighted_moneyflow_daily.parquet", ["date", "trade_date"]),
        source_extent("reports/research/510300_if_exhaustion_existing_training_v1/inputs/if_saved_features.parquet", ["date", "trade_date"]),
        source_extent("data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet", ["date", "trade_date"]),
        source_extent("reports/research/510300_growth_state_increment_20d_v1/inputs/pmi_new_orders.parquet", ["available_at"]),
    ]
    record = {
        "checked_at": timestamp,
        "previous_goal_turn": "PROGRESS_FROZEN_RR3_ALL_CANDIDATE_REALIZATION_DIAGNOSTIC",
        "current_goal_turn": "NO_PROGRESS_EXISTING_EVIDENCE_FRONTIER_REVALIDATED",
        "same_condition_consecutive_no_progress_goal_turns": 1,
        "blocker_id": "NO_ADMISSIBLE_UNEVALUATED_EXISTING_EVIDENCE_AND_NEW_COLLECTION_PAUSED",
        "blocker": "已列候选未发现合格、尚未执行且可凭现有资料完成的独立检验；未验证分支缺事前来源/同步价格/事件覆盖，采集仍由用户暂停。",
        "not_claimed": "不证明所有510300方法都不可能成功；未以困难、不确定或未达到数字目标本身作为阻断理由。",
        "required_external_change": "新增可核验的本地事件或来源档案，或用户改变当前资料范围，使具体缺项可以补齐；仍需先冻结问题，再验证。",
        "latest_local_source_extents": coverage,
        "candidate_route_evidence": evidence,
        "scheduler_observed_this_goal_turn": {"method": "Get-ScheduledTask", "task_name": "Codex-510300-Primary-Market-Collector", "enabled": False, "state_numeric": 1},
        "matching_research_processes_observed_this_goal_turn": [],
        "verified_wait": False,
        "new_fits": 0, "new_accounts": 0, "new_event_paths": 0,
        "goal_achieved": False, "goal_remains_active": True,
        "source_collection_resumed": False,
        "reports_and_status_writes_count_as_research_progress": False,
    }
    save(CHECK, record)
    status.update({
        "updated_at": timestamp,
        "latest_goal_turn_classification": record["current_goal_turn"],
        "same_condition_consecutive_no_progress_goal_turns": 1,
        "active_blocker_id": record["blocker_id"],
        "active_blocker_description": record["blocker"],
        "latest_continuation_check": str(CHECK.relative_to(ROOT)),
        "goal_achieved": False,
        "remaining_research_question": "需要合格的新增事件或来源证据检验经济目标兑现；当前列出的本地分支已完成或仍缺必要输入。",
    })
    save(MAIN / "current_status.json", status)
    text = """当前高夏普目标仍未完成。本续轮核对了已列出的估值、政策、资金供求、收盘价格让步和3:1目标分支，未发现可以在既有资料与约束下立即运行的新独立检验。本轮只有状态核对，没有新增训练、账户或事件路径，不记作研究进展。

已有估值模型并非尚未做过：状态回归诊断受样本依赖及时间稳定性限制，正常化估值与PB-ROE分支有冻结拒绝结果，盈利与可选估值以及估值口径一致性的账户已完成。这些资料可解释经济假设，但尚不能提供可靠的价格目标兑现概率。

尚未验证的资金与折价问题有具体输入缺项：公告后资金调整的104个月中缺98组事前预期配对，基金份额缺日级首次公开时钟；资本市场支持政策研究只有既存两年目录及六个公告日期，不能当完整母集；收盘折价来源只有午休样例，合格同步收盘及盘后观测为零。旧M1线的后续训练已完成，不能根据早期NOT_RUN再次运行。

本地价格仍到2026-09-16，成分股成交分类和IF资料到2026-08-12，DR007到2026-08-14，最新合格PMI发布日期为2026-07-31。当前联合市场观点仍为NO_VIEW。现场读取的采集任务Enabled=false，相关研究进程没有运行；这不属于等待任务完成。

真正能改变下一步的是可核验的新事件或来源档案：例如完整公告母集及发布时间、原始事前预期版本与资金首次发布记录，或决策时同步估值及相应报价。单纯重写状态名称、改变旧失败窗口、扩大目标距离或重复抽样，不补足这些事实。该判断限于目前列出的研究分支，不声称510300的所有可能方法都已穷尽。

至少3:1、事前事实解释、20万元全账户、净夏普1.2、年化10%和回撤10%的要求保持。当前记录为本次新的连续无进展检查第1轮；上一轮目标兑现诊断确有实质进展，因此没有沿用其他任务的旧阻断次数，目标暂保持未完成且活跃。
"""
    (MAIN / "后续证据缺口.md").write_text(text, encoding="utf-8")
    print(json.dumps({"本轮分类": record["current_goal_turn"], "连续无进展轮次": 1, "已核对研究分支": len(evidence), "实际本地数据边界": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
