"""第179轮顺序状态选择的失败关闭、交付及索引更新。"""
import json

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from scripts.prepare_round179_20260913 import CONFIG, OUT, PRIMARY, ROOT


def main():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    main_metrics = {row["cost"]: row for row in result["primary"]}
    require(all(main_metrics[cost]["net_sharpe"] < 1.2 for cost in ["BASE", "STRESS"]), "第179轮不应通过夏普目标")
    require(all(main_metrics[cost]["annualized_return"] < .10 for cost in ["BASE", "STRESS"]), "第179轮不应通过年化10%目标")
    decision = (
        "第179轮在固定市场状态内，仅依据当时已实现的两个保存账户收益，在核心、相加封顶和现金之间顺序选择。"
        "主历史基础／压力净夏普为0.269／-0.143，年化收益为0.44%／-0.22%，均远低于夏普1.2和年化10%的双门槛。"
        "高门槛使大部分时间留在现金，少量入场也没有收益优势，关闭该版本。"
    )
    verification = (
        "三项必要测试通过，四个完整账户核心计算2.36秒。独立程序从冻结的父账户日收益、父目标和指数状态"
        "重建5,646个决定、40段实际持仓及20条比较账户；四个目标序列与保存结果逐项一致。"
    )
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False, PRIMARY: "CLOSED_STRICT_SEQUENTIAL_SELECTION_LOW_EXPOSURE_AND_LOW_RETURN"}, exclusive=True)
    next_path = ROOT / "docs/510300_DIRECTIONAL_EXPOSURE_POLICY_NEXT_20260913.md"
    document = ROOT / "deliverables/510300顺序状态策略选择_第179轮_20260913/顺序状态策略选择_结果及全部中文规则.md"
    deliver_round(
        ROOT, OUT, CONFIG, document, "按状态历史实际收益顺序选择策略",
        "CLOSED_STRICT_SEQUENTIAL_SELECTION_LOW_EXPOSURE_AND_LOW_RETURN", decision, verification,
        next_path, next_path.read_text(encoding="utf-8").splitlines(), "DIRECTIONAL_EXPOSURE_POLICY_PREPARED",
        "以510300自身趋势、回撤和波动直接确定仓位",
    )
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=180, registered=False, planned_settings=0, planned_new_accounts=0, planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    for key in ["current_best_four_scenario_comparison_candidate", "latest_main_sharpe_pass_candidate", "latest_all_four_scenario_excess_candidate"]:
        index[key]["last_compared_completed_round"] = 179
    write_json(index_path, index)
    print("第179轮交付完成。")


if __name__ == "__main__":
    main()
