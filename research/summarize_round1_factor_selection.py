"""汇总第一轮注册因子、单因子回测和日内事件研究的选择结论。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry.yaml"
RESEARCH_FILE = ROOT / "reports" / "research" / "registered_factor_research.json"
BACKTEST_FILE = ROOT / "reports" / "backtest" / "registered_factor_backtests.json"
INTRADAY_FILE = ROOT / "reports" / "research" / "510300_intraday_imbalance_research.json"
JSON_FILE = ROOT / "reports" / "research" / "round1_factor_selection_decision.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "round1_factor_selection_decision.md"


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    backtest = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
    intraday = json.loads(INTRADAY_FILE.read_text(encoding="utf-8"))
    eligible = [item for item in backtest["factors"] if item["candidate_status"] == "ELIGIBLE_FOR_COMBINATION_DISCUSSION"]
    watchlist_ids = {"CSI300_PE_PCTL_5Y", "CSI300_PB_PCTL_5Y"}
    watchlist = [item for item in backtest["factors"] if item["factor_id"] in watchlist_ids]
    primary_intraday = [item for item in intraday["results"] if item["role"] == "primary"]
    decision = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "round": 1,
        "governance": registry["research_status"],
        "strategy_selection_status": "NO_FACTOR_FROZEN",
        "provisional_research_candidates": [item["factor_id"] for item in eligible],
        "watchlist_insufficient_cycles": [item["factor_id"] for item in watchlist],
        "intraday_primary_status": "NO_FVG_PRIMARY_CANDIDATE",
        "blocked_tracks": registry["blocked_tracks"],
        "strategy_frozen": False,
        "reason": (
            "普通价量确认仅达到组合讨论门槛，但滚动稳定性不足；估值因子交易轮次不足；"
            "FVG主假设不确定；PV/LC和Weighted Breadth缺少合格数据。"
        ),
        "hypothesis_count": research["hypothesis_count"],
        "intraday_hypothesis_count": intraday["hypothesis_count"],
        "decision_scope": "HISTORICAL_SNAPSHOT_AT_END_OF_ROUND1",
        "superseded_by": "reports/research/round2_factor_selection_decision.json",
        "current_candidate_status": "ROUND1_PROVISIONAL_CANDIDATE_RETIRED_BY_ROUND2",
    }
    lines = [
        "# 第一轮策略因子选择决定", "",
        "> 历史快照：本报告只记录第一轮结束时的临时判断，已被第二轮选择决定取代；其中价量候选现已退役。", "",
        "## 最终状态", "",
        "- 策略冻结状态：`NO_FACTOR_FROZEN`。",
        "- 本轮完成20个日线假设、10个统一规则单因子回测和6个FVG事件假设。",
        "- 结果允许产生研究候选和观察名单，但没有足够证据冻结正式策略。", "",
        "## 第一轮当时的暂定研究候选", "",
    ]
    if eligible:
        for item in eligible:
            pseudo = item["periods"]["pseudo_oos"]
            contaminated = item["periods"]["contaminated_retrospective"]
            rolling = item["periods"]["full_retrospective"]["rolling_12m_excess"]
            lines += [
                f"### {item['factor_name_cn']}", "",
                f"- 因子ID：`{item['factor_id']}`。",
                f"- D20证据：{item['d20_evidence_rating']}（{item['d20_evidence_score']}/7）。",
                f"- pseudo-OOS超额：{pseudo['excess_vs_realistic_buy_hold']:.2%}；15bp压力超额：{item['pseudo_oos_stress_15bps']['excess_vs_realistic_buy_hold']:.2%}。",
                f"- pseudo-OOS交易次数：{pseudo['strategy']['trade_count']}。",
                f"- 受污染诊断期超额：{contaminated['excess_vs_realistic_buy_hold']:.2%}。",
                f"- 全样本滚动242日超额为正比例：{rolling['positive_ratio']:.2%}。",
                "- 判断：可以进入下一轮机制与组合讨论，但稳定性不足，不能冻结。", "",
            ]
    else:
        lines += ["- 没有日线因子通过组合讨论门槛。", ""]
    lines += ["## 观察名单", ""]
    for item in watchlist:
        pseudo = item["periods"]["pseudo_oos"]
        lines.append(
            f"- `{item['factor_id']}`：{item['factor_name_cn']}，证据{item['d20_evidence_rating']}，"
            f"但pseudo-OOS只有{pseudo['strategy']['trade_count']}笔交易，不满足至少6笔的后验样本充分性保护。"
        )
    lines += ["", "## FVG/三柱价格不平衡", ""]
    for item in primary_intraday:
        lines.append(
            f"- `{item['hypothesis_id']}` {item['event']}，4根柱：{item['evidence']['rating']}；"
            f"假设成本后均值{item['mean_hypothetical_net_return']:.4%}，相对同槽位效应{item['mean_effect_vs_same_slot_baseline']:.4%}。"
        )
    lines += [
        "- 主假设没有形成一致、成本后可执行的日内证据；FVG暂不进入策略。", "",
        "## 数据阻塞", "",
        "- PV/LC：缺少真实Level-2订单、成交、撤单和盘口历史，未用OHLCV替代。",
        "- Weighted Breadth、板块贡献和历史市值结构：缺少历史点时成分、权重和行业映射。",
        "- 当前快照只用于描述现状，不进入五年回测。", "",
        "## 下一轮唯一允许的方向", "",
        "1. 优先补历史点时成分权重和行业数据，构建Weighted Breadth与板块贡献。",
        "2. 明确定义PV、LC并获取短期真实Level-2样本，先验证字段和重构程序。",
        "3. 对普通价量确认做机制拆解，不增加任意窗口：分别检查价格方向、成交量状态及二者交互贡献。",
        "4. PE/PB只保留为慢状态观察变量，不能因少数交易收益漂亮直接升级。",
        "5. 新假设必须使用新的hypothesis_id并在运行前登记；本轮结果不得用于偷偷调整30%/70%阈值。", "",
    ]
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"第一轮选择决定：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
