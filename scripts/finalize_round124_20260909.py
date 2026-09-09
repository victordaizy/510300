"""交付第124轮已核对结果，准备不增加来源的长短力度研究。"""
import json
import numpy as np
import pandas as pd
from research.conditional_drawdown_budget_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    source = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    columns = ["date", "close", "previous_close", "dividend", "volume"]
    frame = pd.read_parquet(source, columns=columns)
    require(frame.date.is_monotonic_increasing and not frame.date.duplicated().any(), "力度输入日期不完整有序")
    require(np.isfinite(frame[columns[1:]].iloc[1:].to_numpy(float)).all(), "原始力度所需后续输入不完整")
    require(frame.volume.ge(0).all() and frame.close.gt(0).all() and frame.previous_close.iloc[1:].gt(0).all(), "原始价格或成交份额范围错误")
    require(pd.isna(frame.previous_close.iloc[0]), "起始无前收盘的结构状态不同")
    np.testing.assert_allclose(frame.previous_close.iloc[1:], frame.close.iloc[:-1], atol=0, rtol=0)
    preflight = ROOT / "reports/research/510300_force_pullback_preflight_20260909/result.json"
    write_json(preflight, {"recorded_at": now(), "status": "EXISTING_PRICE_DIVIDEND_VOLUME_COMPLETE_NO_NEW_INDICATOR",
        "source": str(source.relative_to(ROOT)), "source_sha256": digest(source), "calendar_rows": len(frame),
        "first_date": str(frame.date.iloc[0].date()), "last_date": str(frame.date.iloc[-1].date()),
        "structural_first_previous_close_missing": True, "complete_subsequent_input_rows": len(frame)-1,
        "zero_volume_rows": int(frame.volume.eq(0).sum()), "previous_close_matches_past_observed_close": True,
        "new_indicators": 0, "new_predictions": 0, "new_accounts": 0,
        "scope": "仅检查原有价格、分红、成交份额及日期，没有计算力度或策略结果"}, exclusive=True)
    decision = ("第124轮连续回撤资金预算未达目标。主基础／压力净夏普0.817／0.794，年化2.05%／1.98%，最大回撤4.51%／4.63%；"
        "较早净夏普0.442／0.403，年化3.29%／2.96%，最大回撤12.44%／13.06%。"
        "主历史较第91轮退步，较早虽略有改善，四账户夏普均低于第114轮局部候选，而且两段年化收益均低于各自买入持有。"
        "关闭本固定条件回撤方法，不调整尾部、窗口、父策略、预算上限或收益项挽救。")
    detail = ("十二项必要测试5.65秒，唯一连续预算和四个新实际账户核心计算6.31秒，不含开发、测试、文档及保存核对。"
        "159个月首中75个窗口有效，225次线性规划均成功；72个原零风险窗口和12个暖启动月份保持此前预算。"
        "同一历史前缀仅优化一次，没有新统计模型或参考账户。已用保存原始及对偶解核对全部最优风险和并列区间，并检查5646个实际收盘判断和四账户。"
        "主两账户各58个持仓收盘、5次买入6次卖出，平均股票敞口约2.20%；较早各279个持仓收盘、16次买入15次卖出。没有未知实际目标或未成交请求。"
        "主基础／压力终值较91低20205.19／18443.68元，较早高4047.80／3365.34元；不能只报较早改善。"
        "下一项转向现有日线的长短量价力度回调进入和反弹转弱退出，输入完整性已检查，尚未计算新指标或账户。"
        "历史已反复观察，尚无独立高夏普及稳定超额证据。")
    next_path = ROOT / "docs/510300_FORCE_PULLBACK_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300连续回撤资金预算_第124轮_20260909/连续回撤资金预算_结果及全部中文规则.md",
        "连续回撤资金预算", "COMPLETED_CONDITIONAL_DRAWDOWN_WEAKER_MAIN_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "FORCE_PULLBACK_METHOD_AND_PRICE_VOLUME_PREFLIGHT_COMPLETE_IMPLEMENTATION_PENDING",
        "固定两期和十三期量价力度，在长线为正时回调恢复进入、反弹转弱或长线消失退出")
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=125, registered=False, preflight_receipt=str(preflight.relative_to(ROOT)), new_models_or_accounts=0)
    write_json(index_path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
