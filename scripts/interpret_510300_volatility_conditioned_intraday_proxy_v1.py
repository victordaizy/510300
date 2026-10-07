"""只复算已存账本和损益拆分，不改变信号、策略或既有结果。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/research/510300_pressure_recovery_v1/volatility_conditioned_proxy_20261001"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(name: str, value: dict) -> None:
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if (OUTPUT / "acceptance_outcome.json").exists():
        raise SystemExit("本轮解释已经保存，拒绝覆盖。")
    index = pd.read_csv(OUTPUT / "FILE_INDEX.csv")
    for item in index.itertuples():
        if digest(OUTPUT / item.path) != item.sha256:
            raise ValueError(f"保存文件变化：{item.path}")
    result = pd.read_csv(OUTPUT / "05_全部情景结果.csv")
    daily = pd.read_parquet(OUTPUT / "03_全账户每日账本.parquet")
    trades = pd.read_csv(OUTPUT / "04_假设跨日交易.csv")
    decomposition, checks = [], []
    for row in result.itertuples():
        account = daily.loc[(daily.policy == row.policy) & (daily.cost == row.cost) & (daily.initial_cash_cny == row.initial_cash_cny)].sort_values("date")
        orders = trades.loc[(trades.policy == row.policy) & (trades.cost == row.cost) & (trades.initial_cash_cny == row.initial_cash_cny)]
        equity = np.r_[row.initial_cash_cny, account.equity_cny.to_numpy()]
        returns = equity[1:] / equity[:-1] - 1
        cagr = (equity[-1] / equity[0]) ** (242 / len(account)) - 1
        drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1))
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(242) if returns.std(ddof=1) > 0 else np.nan
        if not np.allclose([cagr, drawdown], [row.cagr, row.maximum_drawdown], atol=1e-10):
            raise ValueError("保存净值与保存指标不一致。")
        if not (pd.isna(sharpe) and pd.isna(row.sharpe)) and not np.isclose(sharpe, row.sharpe, atol=1e-10):
            raise ValueError("保存净值与保存夏普不一致。")
        if abs(equity[-1] - equity[0] - orders.net_pnl_cny.sum()) > 1e-5:
            raise ValueError("保存交易损益与账户总损益不一致。")
        checks.append({"policy": row.policy, "cost": row.cost, "initial_cash_cny": row.initial_cash_cny, "saved_output_recomputation": "PASS"})
        if orders.empty:
            continue
        mean = {name: float((orders[column] / orders.entry_debit * 10000).mean()) for name, column in
                {"gross_mean_bps": "gross_pnl_cny", "commission_mean_bps": "total_fee_cny", "friction_mean_bps": "friction_cny", "net_mean_bps": "net_pnl_cny"}.items()}
        decomposition.append({"policy": row.policy, "cost": row.cost, "initial_cash_cny": row.initial_cash_cny,
                              "trades": len(orders), **mean,
                              "zero_friction_same_trades_and_quantities_bps": mean["gross_mean_bps"] - mean["commission_mean_bps"],
                              "counterfactual_is_executable_policy": False})
    breakdown = pd.DataFrame(decomposition)
    breakdown.to_csv(OUTPUT / "09_费用与收益空间拆分.csv", index=False, encoding="utf-8-sig")
    primary = result.loc[result.policy.eq("SWITCH") & result.cost.eq("BASE") & result.initial_cash_cny.eq(200000)].iloc[0]
    evidence = breakdown.loc[breakdown.policy.eq("SWITCH") & breakdown.cost.eq("BASE") & breakdown.initial_cash_cny.eq(200000)].iloc[0]
    if primary.cagr >= 0:
        raise ValueError("本解释对应当前保存的亏损结果，拒绝套用到其他实验。")
    outcome = {"study_id": "510300_VOLATILITY_CONDITIONED_INTRADAY_PROXY_V1",
               "status": "COMPLETED_FIXED_CANDIDATE_REJECTED_IN_PRICE_PROXY",
               "decision": "固定切换比两个固定规则少亏，但自身没有建立正净优势。关闭这组参数，不做结果后改档、反向或退出寻优。",
               "general_regime_switching_idea_rejected": False,
               "wait_without_daily_trade_quota_retained": True,
               "primary_bottleneck": "每笔毛价格收益不足覆盖成本；同交易同数量的零摩擦解释性上界，扣佣金后均值仍略负。",
               "secondary_bottleneck": "跨日尾部仍大；2万元情景最低佣金更重。",
               "same_quantity_zero_friction_bound_bps": float(evidence.zero_friction_same_trades_and_quantities_bps),
               "zero_volume_proxy_notice": "固定恢复对照的2016-10-13 13:23分钟成交量为零，涉及两金额两费用共4行；保留原价代理并标为无成交证据，不删除重算。切换组没有此类行。",
               "original_m1_m2_expectancy": "NOT_COMPUTED", "original_m1_m2_sharpe": "NOT_COMPUTED",
               "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
               "actual_orders": 0, "position_impact": 0, "model_position_target": "UNSET"}
    save_json("acceptance_outcome.json", outcome)
    save_json("saved_output_recomputation.json", {"status": "PASS", "scenarios": checks,
                                                 "checks_do_not_establish_execution_or_profitability": True,
                                                 "interpreter_file_sha256": digest(Path(__file__))})
    text = f"""# 本轮判断：保留等待，关闭这一组具体切换规则

**按近期波动选择机会可以作为研究问题，空仓等待也应保留；本轮固定实现没有达到扣费后盈利，更没有建立原研究的夏普1.2目标。**

本地已有分钟资料覆盖2015-01-05至2026-08-20；完成预热后，共同评价区间为2016-02-18至2026-08-20，共2,555个交易日。20万元基准费用下，切换政策898次假设跨日交易，1,657天不新买入，1,345天全天无ETF交易及持仓。净年化为{primary.cagr:.2%}，夏普{primary.sharpe:.3f}，最大日终回撤{primary.maximum_drawdown:.2%}。

切换比固定恢复、固定延续少亏，说明这组规则中减少参与有帮助；这份对照不能单独证明波动率准确找到了不同策略的盈利环境。现金对照为零收益（未计利息），切换依然落后现金。按固定的三个时间区间拆开，切换净年化均为负。

## 信号、成本与账户约束

同一批20万元基准费用切换交易，按每笔入场支出标准化：

| 项目 | 每笔均值 |
|---|---:|
| 原始价格变化加分红形成的毛收益 | {evidence.gross_mean_bps:+.3f} bp |
| 往返佣金 | −{evidence.commission_mean_bps:.3f} bp |
| 假设往返价格摩擦 | −{evidence.friction_mean_bps:.3f} bp |
| 扣费净收益 | {evidence.net_mean_bps:+.3f} bp |

即使解释性地去掉全部价格摩擦，仅保留相同交易、数量和原佣金，均值也为{evidence.zero_friction_same_trades_and_quantities_bps:+.3f}bp。这不是重新运行的可执行零成本策略，没有重新投入节省的成本，也没有改变机会集合。它只说明当前毛收益余量很薄，不能把失败全归给滑点假设。

2万元基准费用情景的净年化为−14.45%，最低5元佣金在账户缩水后占比更高。20万元压力费用情景净年化为−19.05%。小账户约束会加重问题，但20万元主账户本身也未通过。

切换组最差一笔为2020-01-23入场、下一交易日退出，假设净损失约9.97%。波动分组不能消除跨假期跳空风险。最大回撤使用日终净值，不代表已涵盖全部盘中或实际执行风险。

## 数据能支持到哪里

分钟close被用作假设成交价，时间标签边界、全量成交、限价/排队和实际费用未经独立验证。固定恢复对照中还保留了2016-10-13 13:23的一条零成交量价格代理，跨金额与费用共4行，不能将其说成实际成交。切换组没有这类零量入场行。这些限制不会把当前负结果转换为正优势。

价格数据不含足够的同步IOPV、篮子和有效订单队列。本轮不替代原M1/M2；原策略成交后期望及夏普保持NOT_COMPUTED。已接触过的历史也不是独立验证集。

## 这次结束后保留什么

保留“可以等待，不要求每天交易”的研究要求；保留开盘前用已知波动描述环境的方法；**不晋升这组具体映射和入场规则，也不根据结果挑波动档、换阈值、反向或改变退出补救。**本轮没有发单，没有实际仓位调整，也不把研究中的现金状态解释为用户实际账户空仓。

原始协议、三个L2样本的首轮观察和此前旧波动率路由的拒绝记录均未改动。

完整参数、表格与曲线见[首轮技术报告](00_波动切换首轮结论.md)、[所有情景](05_全部情景结果.csv)、[费用拆分](09_费用与收益空间拆分.csv)和[假设账户曲线](波动切换_假设账户.png)。
"""
    (OUTPUT / "10_本轮判断与停止条件.md").write_text(text, encoding="utf-8")
    save_json("interpretation_receipt.json", {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                               "original_results_unchanged": True, "original_result_files": index.to_dict("records"),
                                               "interpretation_program_sha256": digest(Path(__file__))})
    final_index = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "FILE_INDEX.csv"]
    pd.DataFrame(final_index).to_csv(OUTPUT / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"保存结果复算": "16个情景通过", "损益拆分": breakdown.to_dict("records"), "本轮状态": outcome["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
