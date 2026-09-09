"""快速交付近邻退出结果，保留下一项在线阶段预测方案。"""
import json
from research.cycle_analogue_exit_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, digest, write_json


def main():
    preflight = ROOT / "reports/research/510300_bayesian_run_length_preflight_20260909/result.json"
    write_json(preflight, {"recorded_at": now(), "status": "EXISTING_DAILY_LOG_RETURN_COMPLETENESS_OBSERVED_NO_FORECASTS",
        "source": "reports/research/510300_adaptive_allocation_v1/features.parquet",
        "source_sha256": digest(ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"),
        "calendar_rows": 3456, "first_return_position": 1, "subsequent_finite_daily_returns": 3455,
        "first_date": "2012-05-28", "last_date": "2026-08-14", "new_predictions": 0, "new_models": 0, "new_accounts": 0,
        "evidence": "本回合此前只读日期和total_log的工具输出；只保存已完成的完整性观察，不重复计算"}, exclusive=True)
    decision = ("第122轮不同历史周期相似状态退出没有改善。主基础／压力净夏普0.521／0.476，年化5.13%／4.63%，最大回撤18.73%／20.11%；"
        "较早净夏普0.581／0.559，年化6.83%／6.53%，最大回撤13.79%／13.92%。四个账户的夏普及终值均低于第114轮局部候选，"
        "完整目标未达，关闭这个固定五邻居方法，不调邻居数量、距离、标签或确认天数挽救。")
    detail = ("七项必要测试4.56秒，114个月样本模型和四个新独立账户核心计算9.49秒，不含规则开发、文档和保存核对。"
        "原27个支持不足月份保留，零模型失败、零新参考、零未成交。主两账户各316个持仓收盘全部有模型；较早各299个，其中95个有模型、204个原无模型。"
        "已复算114个标准化模型、1230个真实持仓状态、822次预测、4110个入选邻居记录、64个完整周期及24条分年增量。"
        "主基础／压力终值较114低33900.44／34062.45元，较早低22943.68／22617.84元。"
        "下一项拟逐日对行情持续时间保持概率、自动调整旧历史影响的贝叶斯模型，与全历史累积对照；已有3455条完整日收益可直接使用，尚未计算新预测或跑账户。"
        "不增加慢数据来源，历史已反复观察，不构成独立验证。")
    next_path = ROOT / "docs/510300_BAYESIAN_RUN_LENGTH_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300不同周期相似状态退出_第122轮_20260909/不同周期相似状态退出_结果及全部中文规则.md",
        "不同历史周期的相似状态退出", "COMPLETED_DISTINCT_CYCLE_ANALOGUE_WEAKER_THAN_LOCAL_CANDIDATE_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "BAYESIAN_RUN_LENGTH_METHOD_AND_SOURCE_REVIEW_COMPLETE_IMPLEMENTATION_PENDING",
        "用全部行情持续时间的后验混合预测，比较阶段变化模型与全历史累积；现有日收益完整")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=123, registered=False, preflight_receipt=str(preflight.relative_to(ROOT)), new_models_or_accounts=0)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
