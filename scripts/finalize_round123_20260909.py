"""交付已经核对的第123轮，并保存下一项现有来源的完整性预检。"""
import json
import numpy as np
import pandas as pd
from research.bayesian_run_length_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def preflight_sources():
    folder = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
    cfg_path = ROOT / "config/510300_continuous_reference_min_variance_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    columns = ["date", "panic_reference_return", "learned_reference_return", "panic_state", "learned_state"]
    observed, frames, sources = [], {}, []
    for period in ["evaluation", "earlier_diagnostic"]:
        path = folder / f"{period}_factors.parquet"
        frame = pd.read_parquet(path, columns=columns)
        dates = pd.DatetimeIndex(frame.date)
        require(dates.is_monotonic_increasing and not dates.has_duplicates, "现有参考日期必须完整有序")
        first = int(np.flatnonzero(dates >= pd.Timestamp(cfg["reference_start"]))[0])
        values = frame[columns[1:3]].to_numpy(float)
        states = frame[columns[3:]].to_numpy(float)
        require(np.isnan(values[:first]).all() and np.isfinite(values[first:]).all(), "现有参考收益完整性不符")
        require(np.isin(states[first-1:-1], [0., 1.]).all(), "现有参考意向缺失或不在零一范围")
        monthly = [t for t in range(first, len(frame)-1) if dates[t].to_period("M") != dates[t-1].to_period("M")]
        complete = [t for t in monthly if t-first+1 >= cfg["risk_window"]]
        observed.append({"period": period, "calendar_rows": len(frame), "reference_first_index": first,
            "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
            "complete_reference_return_rows": len(frame)-first, "complete_reference_intent_rows": len(frame)-first,
            "monthly_origins": len(monthly), "origins_with_242_complete_rows": len(complete)})
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        frames[period] = frame
    early = frames["earlier_diagnostic"]
    full = frames["evaluation"].iloc[:len(early)]
    require(pd.DatetimeIndex(full.date).equals(pd.DatetimeIndex(early.date)), "较早参考不是完整历史前缀")
    # 较早终点不再产生新意向；只核对可用于交易决策的共同前缀。
    np.testing.assert_allclose(full[columns[1:]].iloc[:-1], early[columns[1:]].iloc[:-1], rtol=0, atol=0, equal_nan=True)
    sources.append({"path": str(cfg_path.relative_to(ROOT)), "sha256": digest(cfg_path)})
    receipt = ROOT / "reports/research/510300_conditional_drawdown_budget_preflight_20260909/result.json"
    write_json(receipt, {"recorded_at": now(), "status": "EXISTING_CONTINUOUS_REFERENCE_CALENDAR_COMPLETE_NO_NEW_BUDGET",
        "sources": sources, "periods": observed, "common_decision_prefix_exact": True,
        "new_optimizations": 0, "new_budgets": 0, "new_models": 0, "new_accounts": 0,
        "scope": "只检查已保存日期、完整性和共同前缀，没有计算新回撤目标、预算或业绩"}, exclusive=True)
    return receipt


def main():
    preflight = preflight_sources()
    decision = ("第123轮逐日行情持续时间预测未达标。主历史基础／压力净夏普0.023／负0.037，年化负0.79%／负1.68%，最大回撤37.13%／39.58%；"
        "较早历史净夏普0.470／0.439，年化6.69%／6.14%，最大回撤36.23%／36.75%。"
        "主历史比买入持有差，较早历史虽高于买入持有，四账户均不及第114轮局部候选。"
        "全历史累积对照的四个真实账户均与买入持有完全相同。关闭这两项固定方法，不调整先验、阶段概率、确认规则或窗口挽救。")
    detail = ("八项必要测试3.99秒。两个逐日预测模型共6910次更新，加八个新实际账户的核心计算10.68秒，不含开发、测试、文档与结果核对。"
        "两模型均完成，没有数据缺口或数值失败；八账户没有未知决策或未成交请求。"
        "已核对全部5977152个保存概率、11292个实际判断、八账户、124个完整周期及48条分年增量；四个全历史累积对照与买持逐日股数、净值相同。"
        "主基础费用路径的价格及分红贡献仅1296.60元，佣金及滑点11560.63元，净亏10264.03元；这是已保存成交路径分解，不是重新运行零费用策略。"
        "失败不只是费用，阶段识别本身没有带来足够收益。下一项直接复用两条已有连续参考，按连续回撤分配资金；来源完整性预检已完成，尚未计算新预算。"
        "历史已反复观察，完整夏普及稳定超额目标仍未达。")
    next_path = ROOT / "docs/510300_CONDITIONAL_DRAWDOWN_BUDGET_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300逐日行情持续时间预测_第123轮_20260909/逐日行情持续时间预测_结果及全部中文规则.md",
        "逐日行情持续时间预测", "COMPLETED_RUN_LENGTH_WEAKER_THAN_LOCAL_CANDIDATE_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "CONDITIONAL_DRAWDOWN_METHOD_AND_EXISTING_REFERENCE_PREFLIGHT_COMPLETE_IMPLEMENTATION_PENDING",
        "按连续回撤深度和持续时间优化两条原策略预算；复用完整参考，不补慢来源")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=124, registered=False, preflight_receipt=str(preflight.relative_to(ROOT)), new_models_or_accounts=0)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
