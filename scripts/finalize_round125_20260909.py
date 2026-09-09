"""交付力度策略结果并保留已查重的在线相对得失学习方案。"""
import json
import numpy as np
import pandas as pd
from research.force_pullback_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import require, now, digest, write_json


def main():
    old_receipt = ROOT / "reports/research/510300_conditional_drawdown_budget_preflight_20260909/result.json"
    old = json.loads(old_receipt.read_text(encoding="utf-8"))
    for item in old["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "既有连续参考在完整性预检后改变")
    source = ROOT / "reports/research/510300_continuous_reference_min_variance_v1/evaluation_factors.parquet"
    frame = pd.read_parquet(source, columns=["date", "panic_reference_return", "learned_reference_return"])
    first = int(np.flatnonzero(frame.date.ge("2013-06-03"))[0])
    values = frame[["panic_reference_return", "learned_reference_return"]].iloc[first:-1].to_numpy(float)
    require(np.isfinite(values).all() and (values > -1).all() and (values <= 1).all(), "原参考不能满足预定有界比较损失输入")
    month_first = [t for t in range(first, len(frame)-1) if frame.date.iloc[t].to_period("M") != frame.date.iloc[t-1].to_period("M")]
    preflight = ROOT / "reports/research/510300_adaptive_regret_experts_preflight_20260909/result.json"
    write_json(preflight, {"recorded_at": now(), "status": "UNCHANGED_CONTINUOUS_REFERENCE_LOSS_DOMAIN_AND_BIRTH_CLOCK_OBSERVED_NO_LEARNING",
        "source": str(source.relative_to(ROOT)), "source_sha256": digest(source), "reused_completeness_receipt": str(old_receipt.relative_to(ROOT)),
        "reused_completeness_receipt_sha256": digest(old_receipt), "first_reference_return_index": first,
        "initial_comparison_origin": str(frame.date.iloc[first-1].date()), "complete_nonterminal_return_rows": len(values),
        "returns_in_minus_one_exclusive_plus_one_inclusive": True, "monthly_birth_origins": len(month_first),
        "planned_initial_plus_monthly_groups": len(month_first)+1, "planned_comparison_records_per_group": 3,
        "new_loss_values": 0, "new_online_updates": 0, "new_weights": 0, "new_accounts": 0,
        "rejected_duplicate_routes": {"margin": "既有发现及长期复验失败，不重新下载", "fixed_share": "第9轮已经执行，不重开学习率及共享比例"}}, exclusive=True)
    decision = ("第125轮长短量价力度回调交易未达目标。主基础／压力净夏普负0.196／负0.366，年化负0.95%／负1.70%，最大回撤11.84%／15.30%；"
        "较早净夏普0.133／负0.025，年化0.70%／负0.44%，最大回撤14.16%／17.40%。四账户夏普均低于第114轮局部候选。"
        "关闭本固定两期、十三期力度及回调进入反弹转弱退出方法，不扫描参数、价格口径或新增止损挽救。")
    detail = ("八项必要测试5.31秒，单次因子计算及四个实际账户核心计算1.73秒，不含开发、测试、文档及核对。"
        "首行之外3455条输入完整，3443日具备两条指标；没有源缺口或数值失败。主两账户各36个完整周期、72次成交、100个持仓收盘；较早各38周期、76次成交、130持仓收盘，零未知实际判断或未成交。"
        "已经以分数价差和指数权重闭式核对3455个原始观察及6897个指数均值，核对5646实际判断、四账户和148个完整周期。"
        "主基础保存成交路径的价格与分红贡献为负1234.10元，佣金滑点11017.04元，净亏12251.14元；扣费前的路径贡献本身不足，并非只由费用造成。"
        "较早基础价格与分红贡献19189.60元，费用12045.35元；压力费用下净亏4369.80元。上述为保存路径分解，不是另跑零费用策略。"
        "两融及固定份额专家切换查到既有失败，已排除重复路线。下一项按不同起点的累计相对得失学习组合，原论文与现有参考输入检查完成，尚无新权重或账户。"
        "历史已反复观察，完整目标及独立证据仍未达。")
    next_path = ROOT / "docs/510300_ADAPTIVE_REGRET_EXPERTS_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300长短量价力度回调交易_第125轮_20260909/长短量价力度回调交易_结果及全部中文规则.md",
        "长短量价力度回调交易", "COMPLETED_FORCE_PULLBACK_WEAK_PRICE_CONTRIBUTION_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "ADAPTIVE_REGRET_METHOD_AND_EXISTING_REFERENCE_DOMAIN_PREFLIGHT_COMPLETE_IMPLEMENTATION_PENDING",
        "原两策略与现金，按不同月份起点累计相对损失及绝对差异，用自适应后悔权重组合")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=126, registered=False, preflight_receipt=str(preflight.relative_to(ROOT)), new_models_or_accounts=0)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
