"""汇总第二轮机制研究与统一回测的正式选择决定。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round2.yaml"
RESEARCH_FILE = ROOT / "reports" / "research" / "round2_mechanism_research.json"
BACKTEST_FILE = ROOT / "reports" / "backtest" / "round2_mechanism_backtests.json"
JSON_FILE = ROOT / "reports" / "research" / "round2_factor_selection_decision.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "round2_factor_selection_decision.md"


def _percentage(value: float) -> str:
    return f"{value:.2%}"


def main() -> int:
    for path in (SETTINGS_FILE, REGISTRY_FILE, RESEARCH_FILE, BACKTEST_FILE):
        if not path.exists():
            raise FileNotFoundError(f"第二轮选择汇总缺少输入：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    backtest = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
    research_d20 = {
        item["factor_id"]: item
        for item in research["results"]
        if item["target"] == registry["targets"]["primary"]
    }
    backtests = {item["factor_id"]: item for item in backtest["factors"]}
    eligible = [item["factor_id"] for item in backtest["factors"] if item["candidate_status"] != "NOT_ELIGIBLE"]
    volume = backtests["ETF_VOLUME_SHOCK_20D"]
    volume_evidence = research_d20["ETF_VOLUME_SHOCK_20D"]
    futures = backtests["IF_RELATIVE_LEAD_5D"]
    interaction_models = {
        item["target"]: item
        for item in research["incremental_models"]
        if item["model_id"] == "PRICE_VOLUME_INTERACTION"
    }
    d20_interaction = interaction_models[registry["targets"]["primary"]]
    d5_interaction = interaction_models[registry["targets"]["sensitivity"]]
    result = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "round": 2,
        "governance": registry["research_status"],
        "strategy_selection_status": "NO_ROUND2_STRATEGY_CANDIDATE",
        "strategy_frozen": False,
        "superseded_by": "reports/research/round3_factor_selection_decision.md",
        "eligible_factor_ids": eligible,
        "retired_round1_candidate": "ETF_PRICE_VOLUME_CONFIRM_20D",
        "mechanism_watchlist_only": ["ETF_VOLUME_SHOCK_20D"],
        "context_watchlist_only": ["IF_RELATIVE_LEAD_5D"],
        "rejected_factor_ids": ["ETF_TR_20D_COMPONENT", "ETF_HIGH_VOLUME_TREND_20D"],
        "reason": "第二轮没有因子通过统一持仓规则的pseudo-OOS超额与15bp压力门槛；旧价量乘积存在方向语义缺陷，修正交互项方向相反且策略失败。",
        "posthoc_effective_sample_guard": "D20 pseudo-OOS近似独立观测仅9个，首次第二轮输出后统一将WEAK/STRONG封顶为WEAK_SAMPLE_LIMITED。",
        "studywide_multiple_testing_audit": research["studywide_multiple_testing_audit"],
        "blocked_tracks": registry["blocked_tracks"],
    }
    if eligible:
        raise RuntimeError(f"汇总逻辑预期无候选，但发现：{eligible}")
    lines = [
        "# 第二轮策略因子选择决定", "",
        "> 历史快照：本报告记录第二轮结束时的结论，已由第三轮内部宽度选择决定取代。", "",
        "## 最终状态", "",
        "- 策略选择状态：`NO_ROUND2_STRATEGY_CANDIDATE`。",
        "- 策略冻结状态：`NO_FACTOR_FROZEN`。",
        "- 第二轮4个因子、8个目标假设及3层增量模型均已按登记定义运行。",
        "- 没有因子通过统一持仓规则的pseudo-OOS超额、风险改善和15bp成本压力门槛。", "",
        "## 第一轮价量候选退役", "",
        "- `ETF_PRICE_VOLUME_CONFIRM_20D`正式退役，不再作为组合候选。",
        "- 原定义把二十日收益乘以包含当日的成交量Z分数，会把“下跌×缩量”也记为正值，经济语义不唯一。",
        "- 修正后的`ETF_HIGH_VOLUME_TREND_20D`只保留放量趋势，但D20证据为REJECTED（0/7），pseudo-OOS超额"
        f"{_percentage(backtests['ETF_HIGH_VOLUME_TREND_20D']['periods']['pseudo_oos']['excess_vs_realistic_buy_hold'])}。", "",
        "## 成交量冲击：只留事件研究观察", "",
        f"- `ETF_VOLUME_SHOCK_20D`的D20原始证据得分为{volume_evidence['evidence']['score']}/7，但pseudo-OOS近似独立观测仅"
        f"{volume_evidence['evidence']['effective_sample_guard']['approx_independent_observations']}个，审计评级为`{volume_evidence['evidence']['rating']}`。",
        f"- 跨两轮全部{volume_evidence['multiple_testing']['studywide_hypothesis_count']}个日线假设的BH q值为"
        f"{volume_evidence['multiple_testing']['studywide_bh_q_value_all_daily_hypotheses']:.4f}；轮内q值不再单独承担结论。",
        f"- 统一30/70策略的pseudo-OOS超额为{_percentage(volume['periods']['pseudo_oos']['excess_vs_realistic_buy_hold'])}，"
        f"15bp压力超额为{_percentage(volume['pseudo_oos_stress_15bps']['excess_vs_realistic_buy_hold'])}。",
        "- 结论：可能存在短期事件条件差异，但不能把它当作持续仓位因子；仅保留为未来新数据上的事件研究观察项。", "",
        "## IF相对领先：不进入策略", "",
        f"- `IF_RELATIVE_LEAD_5D`的D20审计评级为`{research_d20['IF_RELATIVE_LEAD_5D']['evidence']['rating']}`。",
        f"- pseudo-OOS超额{_percentage(futures['periods']['pseudo_oos']['excess_vs_realistic_buy_hold'])}，"
        f"15bp压力超额{_percentage(futures['pseudo_oos_stress_15bps']['excess_vs_realistic_buy_hold'])}。",
        "- IF0主连换月仍是结构性数据风险；该变量只保留为环境诊断，不进入仓位规则。", "",
        "## 增量回归结论", "",
        f"- D20交互模型冻结开发参数后的pseudo-OOS R²为{d20_interaction['pseudo_oos_frozen_development_model']['oos_r_squared_vs_pseudo_mean']:.4f}。",
        f"- D5交互模型冻结开发参数后的pseudo-OOS R²为{d5_interaction['pseudo_oos_frozen_development_model']['oos_r_squared_vs_pseudo_mean']:.4f}。",
        "- 两个目标均未形成正的样本外解释增量；不能用开发期拟合改善替代策略可交易性。", "",
        "## 下一步唯一允许的研究动作", "",
        "1. 获取Tushare凭据后补五年点时权重，并进一步获取成分股历史价格、行业和点时市值，研究Weighted Breadth与板块贡献。",
        "2. 按订单流数据合同获取510300短期Level-2试样；PV/LC的数学定义必须在看到结果前冻结。",
        "3. PE/PB继续作为慢状态观察变量，等待更多真正前向周期，不调整当前窗口和30/70阈值。",
        "4. 普通日线价量轨道暂停新增变换；成交量冲击只做预先规定期限的真正前向事件记录。",
        "5. 在没有正式候选前，不启动执行优化，也不把当前快照倒填为历史。", "",
    ]
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"第二轮选择决定：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
