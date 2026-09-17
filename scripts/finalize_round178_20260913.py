"""第178轮结果交付和索引更新。"""
import json

from research.early_selected_regime_mapping_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    main_metrics = {row["cost"]: row for row in result["primary"]}
    early_metrics = {
        row["cost"]: row
        for row in result["earlier_diagnostics"]
        if row["model"] == PRIMARY
    }
    require(
        all(main_metrics[cost]["net_sharpe"] < 1.2 for cost in ["BASE", "STRESS"]),
        "第178轮主历史不应通过夏普目标",
    )
    require(
        all(main_metrics[cost]["annualized_return_excess_vs_buy_hold"] < 0 for cost in ["BASE", "STRESS"]),
        "第178轮主历史不应取得正超额",
    )
    require(
        all(early_metrics[cost]["net_sharpe"] > 1.2 for cost in ["BASE", "STRESS"]),
        "第178轮较早历史选择依据不成立",
    )
    decision = (
        "第178轮将2015年至2019年表现合格的四状态映射固定后迁移到2020年至2026年。"
        "较早历史基础／压力净夏普为1.356／1.344，均高于1.2；主历史却降为1.076／1.012，"
        "均低于1.2，且年化收益相对买入持有分别为-0.001%／-0.212%。"
        "这说明固定状态到策略的映射没有跨时期迁移能力，关闭该映射，不再局部修补。"
    )
    receipt = json.loads((OUT / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    verification = (
        "三项必要测试通过，四个完整账户核心计算2.74秒。独立程序从冻结的两个父策略决定、"
        f"指数状态和账户流水重建5,646个决定、{receipt['complete_actual_cycles']}段实际持仓及比较账户；"
        "四个目标序列与保存结果逐项一致。"
    )
    outcome_path = OUT / "candidate_outcomes.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    require(
        outcome == {
            "recorded_at": outcome["recorded_at"],
            "goal_achieved": False,
            PRIMARY: "CLOSED_EARLY_SELECTED_STATIC_MAPPING_FAILED_MAIN_MIGRATION",
        },
        "第178轮候选结论与冻结关闭结论不同",
    )
    next_path = ROOT / "docs/510300_REGIME_MAPPING_FAILURE_NEXT_20260913.md"
    document = ROOT / "deliverables/510300较早历史状态映射_第178轮_20260913/较早历史状态映射_结果及全部中文规则.md"
    deliver_round(
        ROOT,
        OUT,
        CONFIG,
        document,
        "较早历史选择的四状态固定映射",
        "CLOSED_EARLY_SELECTED_STATIC_MAPPING_FAILED_MAIN_MIGRATION",
        decision,
        verification,
        next_path,
        next_path.read_text(encoding="utf-8").splitlines(),
        "SEQUENTIAL_REGIME_STRATEGY_SELECTION_PREPARED",
        "用过去已实现账户结果按时间顺序选择状态内策略",
    )
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=179, registered=False, planned_settings=0, planned_new_accounts=0)
    for key in [
        "current_best_four_scenario_comparison_candidate",
        "latest_main_sharpe_pass_candidate",
        "latest_all_four_scenario_excess_candidate",
    ]:
        index[key]["last_compared_completed_round"] = 178
    write_json(index_path, index)
    print("第178轮交付完成。")


if __name__ == "__main__":
    main()
