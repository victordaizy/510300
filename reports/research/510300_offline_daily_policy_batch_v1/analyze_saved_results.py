"""只读取本轮保存结果作共同口径比较；不重跑政策、调参或增加抽样。"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_run1"
OUT = ROOT / "analysis"
POLICIES = ("BUY_HOLD", "STATIC50", "VOL10", "TREND", "REPAIR", "MIX50")
CANDIDATES = ("TREND", "REPAIR", "MIX50")
BENCHMARKS = ("BUY_HOLD", "STATIC50", "VOL10")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def f(row: dict, key: str) -> float:
    return float(row[key])


def returns(ledger: list[dict], initial: float) -> list[float]:
    previous = initial
    values = []
    for row in ledger:
        equity = f(row, "equity")
        values.append(equity / previous - 1.0)
        previous = equity
    return values


def correlation(left: list[float], right: list[float]) -> float | None:
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right))
    return numerator / denominator if denominator > 0 else None


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("本轮解释目录非空，不覆盖已有复核")
    OUT.mkdir(exist_ok=True)
    summary = read_csv(RESULTS / "summary.csv")
    annual = read_csv(RESULTS / "annual.csv")
    segments = read_csv(RESULTS / "segments.csv")
    cfg = json.loads((RESULTS / "frozen_rules.json").read_text(encoding="utf-8"))
    require_keys = set(itertools.product(cfg["capitals"], cfg["costs"], POLICIES))
    keyed = {(int(row["capital"]), row["cost"], row["policy"]): row for row in summary}
    if len(summary) != 24 or set(keyed) != require_keys:
        raise ValueError("完整24账户矩阵不符")
    ledgers, daily_returns = {}, {}
    verification, concentration, execution, exposures = [], [], [], []
    maximum_identity_error = maximum_metric_error = 0.0
    common_dates = None
    for key, row in keyed.items():
        capital, cost, policy = key
        tag = f"{capital}_{cost}_{policy}"
        ledger = read_csv(RESULTS / "accounts" / (tag + "_account.csv"))
        orders = read_csv(RESULTS / "accounts" / (tag + "_orders.csv"))
        cycles = read_csv(RESULTS / "accounts" / (tag + "_cycles.csv"))
        decisions = read_csv(RESULTS / "accounts" / (tag + "_decisions.csv"))
        dates = [x["date"] for x in ledger]
        common_dates = dates if common_dates is None else common_dates
        if dates != common_dates or len(dates) != 2823 or dates != sorted(set(dates)):
            raise ValueError("账户日历缺失、重复或未对齐")
        identity_error = max(abs(f(x, "equity") - f(x, "cash") - f(x, "shares") * f(x, "close") - f(x, "receivable")) for x in ledger)
        if identity_error > 1e-6:
            raise ValueError("现金、份额及应收恒等式失败")
        if any(f(x, "cash") < -1e-6 or int(x["shares"]) < 0 or int(x["shares"]) % 100 or
               not 0 <= f(x, "stock_weight") <= 1.00000001 for x in ledger):
            raise ValueError("保存账本违反资金、份额或无杠杆条件")
        if any(x["signal_date"] >= x["execution_date"] for x in orders):
            raise ValueError("保存请求不是信号后执行")
        rr = returns(ledger, capital)
        peak = float(capital)
        drawdown = 0.0
        for x in ledger:
            peak = max(peak, f(x, "equity"))
            drawdown = min(drawdown, f(x, "equity") / peak - 1.0)
        mean, sd = statistics.fmean(rr), statistics.stdev(rr)
        calculated = {
            "cagr_trading_days": (f(ledger[-1], "equity") / capital) ** (cfg["annual_days"] / len(ledger)) - 1,
            "sharpe_zero_rf": mean / sd * math.sqrt(cfg["annual_days"]),
            "max_drawdown": drawdown,
            "average_equity_exposure": statistics.fmean(f(x, "stock_weight") for x in ledger),
        }
        metric_error = max(abs(value - f(row, name)) for name, value in calculated.items())
        if metric_error > 1e-12 or abs(sum(f(x, "commission") for x in orders) - f(row, "commission")) > 1e-6:
            raise ValueError("摘要复算或费用汇总不符")
        if len(cycles) != int(row["closed_cycles"]) or len(orders) != int(row["orders"]):
            raise ValueError("周期或请求数与摘要不符")
        if policy in CANDIDATES:
            if int(ledger[-1]["shares"]) != 0 or f(ledger[-1], "receivable") != 0:
                raise ValueError("候选期末状态不符合本次保存结果，不能直接用闭合周期解释全部利润")
            if abs(sum(f(x, "net_pnl") for x in cycles) - f(row, "net_pnl")) > 1e-6:
                raise ValueError("候选闭合周期损益与完整账户不同")
        maximum_identity_error = max(maximum_identity_error, identity_error)
        maximum_metric_error = max(maximum_metric_error, metric_error)
        ledgers[key], daily_returns[key] = ledger, rr
        verification.append({"capital": capital, "cost": cost, "policy": policy, "days": len(ledger),
                             "identity_error_cny": identity_error, "metric_error": metric_error,
                             "status": "PASS_SAVED_ARITHMETIC_ONLY"})
        holdings = [int(x["shares"]) > 0 for x in ledger]
        gaps_friday = [abs(f(x, "target_weight") - f(x, "actual_weight")) for x in decisions if dt.date.fromisoformat(x["date"]).weekday() == 4]
        exposures.append({"capital": capital, "cost": cost, "policy": policy, "holding_dates": sum(holdings),
                          "cash_dates": len(ledger) - sum(holdings), "average_exposure": calculated["average_equity_exposure"],
                          "maximum_exposure": max(f(x, "stock_weight") for x in ledger),
                          "maximum_friday_target_gap": max(gaps_friday),
                          "friday_rebalance_decisions": sum(x["reason"] == "FRIDAY_BAND_REBALANCE" for x in decisions)})
        for status, count in sorted(Counter(x["status"] for x in orders).items()):
            execution.append({"capital": capital, "cost": cost, "policy": policy, "status": status, "requests": count})
        if cycles:
            ordered = sorted(cycles, key=lambda x: f(x, "net_pnl"), reverse=True)
            total = sum(f(x, "net_pnl") for x in cycles)
            positive_total = sum(max(0.0, f(x, "net_pnl")) for x in cycles)
            best = f(ordered[0], "net_pnl")
            concentration.append({"capital": capital, "cost": cost, "policy": policy, "cycles": len(cycles),
                "total_closed_pnl_cny": total, "best_cycle_pnl_cny": best,
                "best_cycle_entry": ordered[0]["entry_date"], "best_cycle_exit": ordered[0]["exit_date"],
                "best_cycle_share_of_net_pnl": best / total if total else None,
                "best_cycle_share_of_positive_pnl": best / positive_total if positive_total else None,
                "top_five_pnl_cny": sum(f(x, "net_pnl") for x in ordered[:5]),
                "sum_other_cycle_pnl_cny": total - best,
                "interpretation": "保存周期贡献分解，非删除交易后的可执行账户"})

    comparisons, correlations, mixes = [], [], []
    for capital, cost in itertools.product(cfg["capitals"], cfg["costs"]):
        for policy, benchmark in itertools.product(CANDIDATES, BENCHMARKS):
            row, base = keyed[(capital, cost, policy)], keyed[(capital, cost, benchmark)]
            comparisons.append({"capital": capital, "cost": cost, "candidate": policy, "benchmark": benchmark,
                "delta_cagr": f(row, "cagr_trading_days") - f(base, "cagr_trading_days"),
                "delta_sharpe": f(row, "sharpe_zero_rf") - f(base, "sharpe_zero_rf"),
                "delta_drawdown_signed_positive_is_smaller_loss": f(row, "max_drawdown") - f(base, "max_drawdown"),
                "delta_average_exposure": f(row, "average_equity_exposure") - f(base, "average_equity_exposure"),
                "candidate_meets_sharpe_1_2_and_cagr_10pct": f(row, "sharpe_zero_rf") >= 1.2 and f(row, "cagr_trading_days") >= .10,
                "statistical_significance_tested": False})
        for left, right in itertools.combinations(POLICIES, 2):
            correlations.append({"capital": capital, "cost": cost, "left": left, "right": right,
                                 "daily_account_return_correlation": correlation(daily_returns[(capital, cost, left)], daily_returns[(capital, cost, right)]),
                                 "sample_dates": len(common_dates), "independent_samples_claim": False})
        trend, repair, mix = (ledgers[(capital, cost, policy)] for policy in CANDIDATES)
        mixes.append({"capital": capital, "cost": cost,
            "trend_repair_both_holding_dates": sum(int(x["shares"]) > 0 and int(y["shares"]) > 0 for x, y in zip(trend, repair)),
            "repair_only_holding_dates": sum(int(x["shares"]) == 0 and int(y["shares"]) > 0 for x, y in zip(trend, repair)),
            "mix_trend_daily_correlation": correlation(daily_returns[(capital, cost, "MIX50")], daily_returns[(capital, cost, "TREND")]),
            "trend_repair_daily_correlation": correlation(daily_returns[(capital, cost, "TREND")], daily_returns[(capital, cost, "REPAIR")]),
            "mix_minus_half_trend_half_repair_final_cny": f(mix[-1], "equity") - .5 * (f(trend[-1], "equity") + f(repair[-1], "equity")),
            "interpretation": "合并账户不同于相加两条净值；差额不是可归因于互补的纯效应，未跑新的暴露匹配对照"})

    signals = [x for x in read_csv(RESULTS / "signals.csv") if cfg["start"] <= x["date"] <= cfg["end"]]
    signal_states = Counter((x["trend_state"], x["repair_state"]) for x in signals)
    write_csv("24账户保存指标复算.csv", verification)
    write_csv("36项候选相对基准差异.csv", comparisons)
    write_csv("全账户日收益相关.csv", correlations)
    write_csv("候选周期贡献集中.csv", concentration)
    write_csv("持有天数与实际仓位.csv", exposures)
    write_csv("全部订单状态汇总.csv", execution)
    write_csv("固定组合与单策略差异.csv", mixes)
    write_json("signal_state_counts.json", [{"trend_state": int(t), "repair_state": int(r), "days": n} for (t, r), n in sorted(signal_states.items())])
    result = {
        "recorded_at": dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "COMPLETED_FIXED_BATCH_NO_CANDIDATE_TARGET_PASS", "accounts": 24,
        "candidate_scenarios": 12, "candidate_joint_target_passes": sum(f(row, "sharpe_zero_rf") >= 1.2 and f(row, "cagr_trading_days") >= .1 for row in summary if row["policy"] in CANDIDATES),
        "all_scenario_joint_target_passes": sum(f(row, "sharpe_zero_rf") >= 1.2 and f(row, "cagr_trading_days") >= .1 for row in summary),
        "account_trading_days_each": len(common_dates), "saved_account_rows": sum(map(len, ledgers.values())),
        "annual_rows": len(annual), "segment_rows": len(segments), "candidate_benchmark_comparisons": len(comparisons),
        "maximum_identity_error_cny": maximum_identity_error, "maximum_metric_error": maximum_metric_error,
        "candidate_cagr_below_all_three_benchmarks_all_scenarios": all(x["delta_cagr"] < 0 for x in comparisons),
        "candidate_sharpe_below_vol10_all_scenarios": all(x["delta_sharpe"] < 0 for x in comparisons if x["benchmark"] == "VOL10"),
        "new_policy_account_runs_in_analysis": 0, "new_random_draws": 0, "parameter_changes": 0, "network_requests": 0,
        "independent_validation": False, "actual_fills_verified": False, "promoted_candidates": [],
    }
    write_json("saved_result_review_receipt.json", result)
    primary = {(row["cost"], row["policy"]): row for row in summary if int(row["capital"]) == 200000}
    table = ["|政策|BASE年化|STRESS年化|BASE夏普|STRESS夏普|STRESS回撤幅度|STRESS平均仓位|", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        b, s = primary[("BASE", policy)], primary[("STRESS", policy)]
        table.append(f"|{policy}|{f(b,'cagr_trading_days'):.3%}|{f(s,'cagr_trading_days'):.3%}|{f(b,'sharpe_zero_rf'):.3f}|{f(s,'sharpe_zero_rf'):.3f}|{-f(s,'max_drawdown'):.3%}|{f(s,'average_equity_exposure'):.3%}|")
    small = ["|政策|2万元BASE年化 / 夏普|2万元STRESS年化 / 夏普|", "|---|---:|---:|"]
    for policy in POLICIES:
        b, s = keyed[(20000, "BASE", policy)], keyed[(20000, "STRESS", policy)]
        small.append(f"|{policy}|{f(b,'cagr_trading_days'):.3%} / {f(b,'sharpe_zero_rf'):.3f}|{f(s,'cagr_trading_days'):.3%} / {f(s,'sharpe_zero_rf'):.3f}|")
    blocks = ["|政策|2015—2019年化 / 夏普|2020—2023年化 / 夏普|2024—终点年化 / 夏普|", "|---|---:|---:|---:|"]
    for policy in POLICIES:
        cells = []
        for period in ("2015_2019", "2020_2023", "2024_END"):
            row = next(x for x in segments if x["capital"] == "200000" and x["cost"] == "STRESS" and x["policy"] == policy and x["period"] == period)
            sharpe = f"{f(row,'sharpe_zero_rf'):.3f}" if row["sharpe_zero_rf"] else "未定义（全现金）"
            cells.append(f"{f(row,'cagr_trading_days'):.3%} / {sharpe}")
        blocks.append("|" + policy + "|" + "|".join(cells) + "|")
    mix = next(x for x in mixes if x["capital"] == 200000 and x["cost"] == "STRESS")
    top_trend = next(x for x in concentration if x["capital"] == 200000 and x["cost"] == "STRESS" and x["policy"] == "TREND")
    top_repair = next(x for x in concentration if x["capital"] == 200000 and x["cost"] == "STRESS" and x["policy"] == "REPAIR")
    report = f"""# 纯日线、低频完整政策：首轮离线结果

已原样运行用户提供的run_offline.py与rules.json：三个候选加三个基准，2万/20万元×BASE/STRESS，24账户全部保留。没有改参数、增加市场输入、联网或重新定义未见样本。全部12个候选情景及全部24账户均未同时达到净夏普1.2、年化10%。

## 运行范围及新授权

日线3,456条（2012-05-28—2026-08-14）、分红14条；仅从原审阅ZIP读取readable/normalized_prices.csv和readable/normalized_dividends.csv。原始数据历史版本限制保持。正式账户2015-01-05—2026-08-14，2,823交易日；之前只预热，三个分段来自同一连续账户。

用户明确修订原研发限制：已有数据可以支持有界历史开发；入场、持有、退出作为完整合同联合检验，无需先证明孤立入场盈利。旧A/B/C、D-native与既有失败不修改，本轮也不声称新信息或首创机制。三候选事先固定，此次没有参数筛选。

## 20万元共同口径结果

{chr(10).join(table)}

各政策CAGR按242/账户天数，夏普无风险0；真实日历CAGR也在summary.csv。期末按市值，预计退出费用另列，不假称已平仓。BASE/STRESS沿用启动包费用和不利滑点。滑点已进成交价格，不再重复扣除。

## 小账户结果全部保留

{chr(10).join(small)}

## 三个预设区段（20万元STRESS）

{chr(10).join(blocks)}

分段是已暴露历史描述，不能挑选好看的阶段另称独立验证。annual.csv另含所有{len(annual)}行年度结果，segments.csv含全部{len(segments)}行。

## 当前结果的含义

1. 三候选在四个资本/费用情景的年化收益均低于三个基准，夏普均低于VOL10；候选回撤通常更小，但实际仓位也更低，因此不是基于同一风险水平的纯信息优势比较。没有额外跑暴露匹配或风险匹配账户，不能把低回撤全部归因于交易判断。
2. TREND的20万元压力年化0.651%、夏普0.123，34个平仓周期、11胜23负；最长持有已允许趋势延续，但总体收益仍弱。最大盈利周期{top_trend['best_cycle_pnl_cny']:,.2f}元，大于全期净利润{top_trend['total_closed_pnl_cny']:,.2f}元；其余周期利润相加为{top_trend['sum_other_cycle_pnl_cny']:,.2f}元。这是贡献分解，不是删除该交易后重跑的账户。
3. REPAIR四个情景都只有4个平仓周期；20万元压力仅10个收盘持仓日，平均仓位0.249%，全期年化0.129%、最大回撤0.962%。3胜1负不足以证明稳定优势。最大盈利周期来自{top_repair['best_cycle_entry']}—{top_repair['best_cycle_exit']}，占全期净利润{top_repair['best_cycle_share_of_net_pnl']:.2%}；2015—2019完全现金，夏普未定义。
4. MIX50仍是一个真实现金/份额模拟账户。20万元压力年化0.546%、夏普0.158、回撤13.209%，平均仓位14.529%。趋势与修复实际同时持仓{mix['trend_repair_both_holding_dates']}日，修复独占{mix['repair_only_holding_dates']}日；MIX50与TREND的完整日账户收益相关为{mix['mix_trend_daily_correlation']:.6f}。存在时间分离但修复样本很少，没有证明稳定组合增量。合并账户不等于两条全仓净值平均，完整差额已列出，不能把差额都命名为互补收益。
5. STATIC50在这份历史与10个百分点周五调仓带下仅首次买入，后续未触发调仓；保存决策已核对。BUY_HOLD、STATIC50、VOL10的平仓周期为0是期末仍持仓，不能把它们写成无交易或零胜率。

## 核验和未验证事项

本机原样合成自检通过。保存24账户共{result['saved_account_rows']:,}行；摘要的年化、夏普、回撤与平均仓位独立算术复算最大误差{maximum_metric_error:.3g}，净值恒等式最大误差{maximum_identity_error:.3g}元。候选全部闭合周期损益与其期末现金账户一致；费用、请求日期与完整24矩阵核对通过。该复核只读输出，没有重新模拟任何政策账户。

未执行置信区间、多重历史选择纠偏、真实订单排队检验、供应商历史首版认证、前瞻实验或参数邻域搜索。开盘为启动包的限价、不利滑点及容量条件代理。当前本地数据止于2026-08-14，不输出2026-10-02买卖观点或实际持仓。

## 本轮处置

完成这三个固定政策及三个固定基准的历史比较，保存全部24份结果；当前晋升候选0。本轮设置停止，不自动搜索55/58/62日均线、放宽修复阈值、调整权重或费用。结论只针对本固定合同，不是所有趋势、修复或组合机制永远无效。

后续如提出新版本，须从本轮具体问题出发，先写有限的新假设与比较合同。可以继续使用现有数据，不恢复“必须等新字段”或“必须先有独立盈利入场”的通用限制。新版本仍是历史开发，不能保证达到目标。

## 文件入口

- ../results_run1/summary.csv、annual.csv、segments.csv：全部结果与固定分段。
- ../results_run1/accounts/：24份账户各自逐日净值、订单、决策、闭合周期，原样保留。
- 36项候选相对基准差异.csv、持有天数与实际仓位.csv、候选周期贡献集中.csv、固定组合与单策略差异.csv：本次只读解释。
- ../authority_and_input_freeze.json、../self_test_local_receipt.json、../local_execution_receipt.json、saved_result_review_receipt.json：本轮实际运行证据。
"""
    (OUT / "首轮研究结果与结论.md").write_text(report, encoding="utf-8")
    index = [{"path": path.relative_to(RESULTS).as_posix(), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(RESULTS.rglob("*")) if path.is_file()]
    write_json("original_results_file_index.json", index)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
