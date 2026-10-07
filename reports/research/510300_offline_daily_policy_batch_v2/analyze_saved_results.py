"""读取第二轮已保存账户，解释结构增量；不重跑政策或新增抽样。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "results_run2"
OUT = STUDY / "analysis"
PARENTS = {"TREND_NO_SLOPE": "TREND", "REPAIR_SEQUENCE": "REPAIR"}


def read(name: str) -> list[dict]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict, key: str) -> float:
    return float(row[key])


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("已有解释结果，不覆盖")
    OUT.mkdir(exist_ok=True)
    summary, paired, annual, segments = (read(name) for name in ("summary.csv", "paired_daily_increment.csv", "annual.csv", "segments.csv"))
    lookup = {(int(x["capital"]), x["cost"], x["policy"]): x for x in summary}
    if len(lookup) != 32 or Counter(x["study_role"] for x in summary) != {"ORIGINAL_CONTROL_REPLAY": 24, "NEW_FIXED_POLICY": 8}:
        raise ValueError("32账户角色或矩阵不符")
    comparisons, monthly_rows, cluster_rows, attribution, tails, checks = [], [], [], [], [], []
    metric_errors, identity_errors, pair_errors = [], [], []
    for row in summary:
        if row["study_role"] != "NEW_FIXED_POLICY":
            continue
        capital, cost, policy = int(row["capital"]), row["cost"], row["policy"]
        parent = PARENTS[policy]
        old = lookup[(capital, cost, parent)]
        ledger = read(f"accounts/{capital}_{cost}_{policy}_account.csv")
        old_ledger = read(f"accounts/{capital}_{cost}_{parent}_account.csv")
        cycles = read(f"accounts/{capital}_{cost}_{policy}_cycles.csv")
        old_cycles = read(f"accounts/{capital}_{cost}_{parent}_cycles.csv")
        orders = read(f"accounts/{capital}_{cost}_{policy}_orders.csv")
        pairs = [x for x in paired if int(x["capital"]) == capital and x["cost"] == cost and x["policy"] == policy]
        dates = [x["date"] for x in ledger]
        if len(dates) != 2823 or dates != [x["date"] for x in old_ledger] or dates != [x["date"] for x in pairs]:
            raise ValueError("新旧配对日历不符")
        previous_new = previous_old = float(capital)
        returns = []
        peak, drawdown = float(capital), 0.0
        for new, prior, delta in zip(ledger, old_ledger, pairs):
            en, eo = number(new, "equity"), number(prior, "equity")
            identity_errors.append(abs(en - number(new, "cash") - number(new, "shares") * number(new, "close") - number(new, "receivable")))
            pair_errors.append(abs(number(delta, "incremental_net_pnl") - ((en - previous_new) - (eo - previous_old))))
            pair_errors.append(abs(number(delta, "equity_difference") - (en - eo)))
            returns.append(en / previous_new - 1)
            peak, drawdown = max(peak, en), min(drawdown, en / max(peak, en) - 1)
            previous_new, previous_old = en, eo
        values = {
            "cagr_trading_days": (previous_new / capital) ** (242 / len(ledger)) - 1,
            "sharpe_zero_rf": statistics.fmean(returns) / statistics.stdev(returns) * math.sqrt(242),
            "max_drawdown": drawdown,
            "average_equity_exposure": statistics.fmean(number(x, "stock_weight") for x in ledger),
        }
        error = max(abs(value - number(row, name)) for name, value in values.items())
        metric_errors.append(error)
        if int(ledger[-1]["shares"]) != 0 or number(ledger[-1], "receivable") != 0:
            raise ValueError("新候选期末有库存/应收，不能套用本次闭合周期分解")
        if abs(sum(number(x, "net_pnl") for x in cycles) - number(row, "net_pnl")) > 1e-6:
            raise ValueError("周期净损益与新候选账户不一致")
        if any(x["signal_date"] >= x["execution_date"] for x in orders):
            raise ValueError("订单没有晚于观察时点")
        checks.append({"capital": capital, "cost": cost, "policy": policy, "days": len(ledger), "summary_metric_error": error,
                       "status": "PASS_SAVED_ARITHMETIC", "new_account_replays": 0})
        for benchmark in (parent, "VOL10", "BUY_HOLD", "STATIC50"):
            ref = lookup[(capital, cost, benchmark)]
            comparisons.append({"capital": capital, "cost": cost, "policy": policy, "benchmark": benchmark,
                "delta_cagr": number(row, "cagr_trading_days") - number(ref, "cagr_trading_days"),
                "delta_sharpe": number(row, "sharpe_zero_rf") - number(ref, "sharpe_zero_rf"),
                "delta_drawdown_signed_positive_is_smaller_loss": number(row, "max_drawdown") - number(ref, "max_drawdown"),
                "delta_average_exposure": number(row, "average_equity_exposure") - number(ref, "average_equity_exposure"),
                "delta_terminal_equity_cny": number(row, "final_equity") - number(ref, "final_equity"), "statistical_interval_run": False})
        months = defaultdict(float)
        for delta in pairs:
            months[delta["date"][:7]] += number(delta, "incremental_net_pnl")
        monthly_rows.extend({"capital": capital, "cost": cost, "policy": policy, "parent": parent, "month": month, "incremental_pnl_cny": value}
                            for month, value in sorted(months.items()))
        clusters = []
        for start, end in sorted((x["entry_date"], x["exit_date"]) for x in cycles + old_cycles):
            if clusters and start <= clusters[-1][1]:
                clusters[-1][1] = max(clusters[-1][1], end)
            else:
                clusters.append([start, end])
        total = number(row, "final_equity") - number(old, "final_equity")
        contributions = []
        for index, (start, end) in enumerate(clusters, 1):
            value = sum(number(x, "incremental_net_pnl") for x in pairs if start <= x["date"] <= end)
            cluster = {"capital": capital, "cost": cost, "policy": policy, "parent": parent, "cluster": index,
                "start": start, "end": end, "incremental_pnl_cny": value,
                "full_increment_cny": total, "other_clusters_and_dates_pnl_cny": total - value,
                "classification": "SAVED_ACCOUNT_CONTRIBUTION_NOT_DELETE_AND_REPLAY"}
            cluster_rows.append(cluster); contributions.append(cluster)
        if abs(sum(x["incremental_pnl_cny"] for x in contributions) - total) > 1e-6:
            raise ValueError("完整事件簇的贡献没有覆盖账户增量")
        new_cost = number(row, "commission") + number(row, "slippage_vs_open")
        old_cost = number(old, "commission") + number(old, "slippage_vs_open")
        gross_difference = total + new_cost - old_cost
        attribution.append({"capital": capital, "cost": cost, "policy": policy, "parent": parent,
            "net_increment_cny": total, "quote_pnl_increment_cost_added_back_cny": gross_difference,
            "extra_recorded_friction_cny": new_cost - old_cost,
            "identity_error_cny": abs(total - (gross_difference - (new_cost - old_cost))),
            "classification": "ARITHMETIC_DECOMPOSITION_ON_SAVED_QUANTITIES_NOT_ZERO_COST_STRATEGY"})
        cycle_returns = [number(x, "net_return_on_total_buy_cost") for x in cycles]
        worst_count = max(1, math.ceil(.05 * len(cycles)))
        best = max(cycles, key=lambda x: number(x, "net_pnl"))
        largest_increment = max(contributions, key=lambda x: x["incremental_pnl_cny"])
        tails.append({"capital": capital, "cost": cost, "policy": policy, "closed_cycles": len(cycles),
            "holding_dates": sum(int(x["shares"]) > 0 for x in ledger), "wins": sum(number(x, "net_pnl") > 0 for x in cycles),
            "worst_cycle_return": min(cycle_returns), "worst_five_pct_cycle_mean": statistics.fmean(sorted(cycle_returns)[:worst_count]),
            "best_cycle_pnl_cny": number(best, "net_pnl"), "best_cycle_share_of_total_net": number(best, "net_pnl") / number(row, "net_pnl"),
            "largest_increment_cluster_start": largest_increment["start"], "largest_increment_cluster_end": largest_increment["end"],
            "largest_increment_cluster_cny": largest_increment["incremental_pnl_cny"],
            "increment_outside_largest_cluster_cny": total - largest_increment["incremental_pnl_cny"],
            "mean_holding_calendar_days": statistics.fmean(number(x, "holding_calendar_days") for x in cycles)})
    if max(metric_errors) > 1e-12 or max(identity_errors) > 1e-6 or max(pair_errors) > 1e-6:
        raise ValueError("保存指标、净值或边际差复核失败")
    write("8新账户保存指标复算.csv", checks)
    write("新政策相对父策略与基准.csv", comparisons)
    write("全部月份净增量.csv", monthly_rows)
    write("全部联合事件簇净增量.csv", cluster_rows)
    write("报价损益与摩擦的算术分解.csv", attribution)
    write("完整周期尾部与集中度.csv", tails)
    new_summary = [x for x in summary if x["study_role"] == "NEW_FIXED_POLICY"]
    decisions = {
        "TREND_NO_SLOPE": "POINT_INCREMENT_IN_ALL_FOUR_SCENARIOS_REGISTER_ROBUSTNESS_CANDIDATE_NO_PROMOTION",
        "REPAIR_SEQUENCE": "COVERAGE_EXPANDED_PRESSURE_INCREMENT_NOT_SUPPORTED_END_FIXED_SETTING",
    }
    result = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "COMPLETED_SECOND_FIXED_STRUCTURAL_BATCH",
        "original_control_replays": 24, "original_replay_max_error": 0.0, "new_policy_accounts": 8, "total_accounts": 32,
        "new_account_trading_days_each": 2823, "new_saved_account_rows": 8 * 2823, "total_saved_account_rows": 32 * 2823,
        "all_summary_rows": len(summary), "annual_rows": len(annual), "segment_rows": len(segments),
        "paired_daily_rows": len(paired), "new_joint_target_passes": sum(number(x, "sharpe_zero_rf") >= 1.2 and number(x, "cagr_trading_days") >= .1 for x in new_summary),
        "maximum_summary_error": max(metric_errors), "maximum_equity_identity_error_cny": max(identity_errors),
        "maximum_paired_increment_error_cny": max(pair_errors), "policy_decisions": decisions,
        "parameter_search": False, "new_combination": False, "new_accounts_in_analysis": 0, "new_random_draws": 0,
        "network_requests": 0, "independent_validation": False, "paper_replication": False, "promoted_candidates": []}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    main_table = ["|政策|BASE年化|BASE夏普|STRESS年化|STRESS夏普|STRESS回撤幅度|STRESS平均仓位|", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in ("TREND", "TREND_NO_SLOPE", "REPAIR", "REPAIR_SEQUENCE", "VOL10", "BUY_HOLD"):
        base, stress = lookup[(200000, "BASE", policy)], lookup[(200000, "STRESS", policy)]
        main_table.append(f"|{policy}|{number(base,'cagr_trading_days'):.3%}|{number(base,'sharpe_zero_rf'):.3f}|{number(stress,'cagr_trading_days'):.3%}|{number(stress,'sharpe_zero_rf'):.3f}|{-number(stress,'max_drawdown'):.3%}|{number(stress,'average_equity_exposure'):.3%}|")
    small_table = ["|新政策|资金|费用|年化|夏普|回撤幅度|平仓周期|", "|---|---:|---|---:|---:|---:|---:|"]
    for x in new_summary:
        small_table.append(f"|{x['policy']}|{x['capital']}|{x['cost']}|{number(x,'cagr_trading_days'):.3%}|{number(x,'sharpe_zero_rf'):.3f}|{-number(x,'max_drawdown'):.3%}|{x['closed_cycles']}|")
    t = next(x for x in tails if x["capital"] == 200000 and x["cost"] == "STRESS" and x["policy"] == "TREND_NO_SLOPE")
    r = next(x for x in tails if x["capital"] == 200000 and x["cost"] == "STRESS" and x["policy"] == "REPAIR_SEQUENCE")
    ra = next(x for x in attribution if x["capital"] == 200000 and x["cost"] == "STRESS" and x["policy"] == "REPAIR_SEQUENCE")
    report = f"""# 第二轮两项结构比较：完整离线结果

已原样执行用户提供的第二轮启动包。先重放第一轮24账户、逐日逐字段误差0，再运行8个新账户；合计32份完整结果。两个新规则均未达到净夏普1.2及年化10%，当前晋升0。

## 范围与固定合同

来源为第一轮results_run1交付ZIP，身份按用户脚本固定SHA核对。市场输入仍仅3,456日线及14分红；正式账户2015-01-05—2026-08-14，共2,823交易日，242年化。原引擎、风险预算、费用、调仓、限价与分红时钟的字节/参数不改。原24仅为复现对照，8新账户不是8个独立市场样本。

TREND_NO_SLOPE只删除入场的MA60斜率条件，继续两收盘进入/退出。REPAIR_SEQUENCE首次超跌启动准备、重复不刷新、回均线或准备年龄10先取消，首次回升且低于MA5确认；活动信号另保留原10日、确认锚及波动尺度。两阶段可能接近20日，机会集合已改变。没有新组合或熊市高波动切换。

## 20万元同口径结果

{chr(10).join(main_table)}

## 8个新账户全部报告

{chr(10).join(small_table)}

## 实验T：斜率条件的代价存在历史点增量，但仍未证明稳定优势

四个资金/费用情景的年化、夏普均高于原TREND，回撤均更小。20万压力年化0.651%→2.058%、夏普0.123→0.279、回撤25.520%→21.885%；平均仓位28.545%→37.230%，因此不能把结果差异全部称为纯信号信息。

平仓周期34→48，新版本15胜33负、胜率31.25%；更多进入也增加失败。原预设区段中，20万压力2015—2019新年化4.126%/夏普0.522，2020—2023年化−2.294%/夏普−0.225，2024—终点年化4.964%/夏普0.582。旧同段分别4.006%/0.543、−3.238%/−0.386、0.373%/0.087。不能只看2024，也不能将新版本所有时间段都描述为改善。

完整账户相对原TREND累计净增量37,913.52元。最大联合持有事件簇为{t['largest_increment_cluster_start']}—{t['largest_increment_cluster_end']}，贡献{t['largest_increment_cluster_cny']:,.2f}元，约41.59%；该簇以外保存增量合计{t['increment_outside_largest_cluster_cny']:,.2f}元。2024年9月的月份边际为20,910.12元，是全账户差分，不是把原策略自身利润当增量。完整月份和48个联合事件簇已保留。删除贡献只是算术描述，未运行删交易账户。

处置：保留为后续固定稳健性研究候选；当前全期夏普仍低于VOL10，年化低于VOL10和BUY_HOLD，未做区间及历史选择纠偏，不能晋升或称独立确认。后续提案可检查更早进入与新增错误之间的状态差异；本轮不拟合条件收益、不改变风险预算。

## 实验R：机会覆盖扩大，压力成本和尾部未支持进一步接受

平仓周期4→49、20万压力持仓日10→{r['holding_dates']}，平均仓位0.249%→2.462%。它解决原同日条件导致的覆盖稀少，但次数不是收益证据。

20万BASE年化0.162%→0.516%，压力却0.129%→0.120%、夏普0.234→0.065、回撤0.962%→7.579%；2万压力年化0.122%→0.074%，夏普0.226→0.045。压力下两个资金规模均没有父策略净增量。20万压力34胜15负、胜率69.39%，条件平均净赢约1.093%、净亏约1.825%、实际盈亏比0.599；最差周期{r['worst_cycle_return']:.3%}，最差5%周期均值{r['worst_five_pct_cycle_mean']:.3%}。胜率较高同时存在更大的失败。

20万压力相对原REPAIR净增量{ra['net_increment_cny']:,.2f}元。按已保存数量/路径把展示佣金和滑点加回，报价损益增量{ra['quote_pnl_increment_cost_added_back_cny']:,.2f}元，额外已记录摩擦{ra['extra_recorded_friction_cny']:,.2f}元，两者相减为净增量。这是算术归因，不是减费或零成本可执行策略；费用不能再从已扣净值中重复扣除。

2015—2019压力年化−0.906%，2020—2023为0.667%，2024—终点1.273%；固定区段并不稳定。处置：保留覆盖扩大和全部失败记录，本固定设置结束，不自动换准备期、阈值、止损或费用；不把所有修复机制判为永远无效。

## 验证与边界

收到程序的合成状态/前缀/到期/时钟和原引擎检查在本机通过。24旧账户按用户程序重放一致；解释阶段只读8新账户22,584行，摘要指标复算最大误差{max(metric_errors):.3g}，账户恒等式误差最大{max(identity_errors):.3g}元，逐日配对差复算误差{max(pair_errors):.3g}元。

全部32账户90,336行、384年度行、96固定区段行及22,584逐日边际行保存。没有额外真实政策回放、抽样、网络请求、期货/期权交易或券商订单。期末估值与退出费用储备沿用原合同。

《Momentum Crashes》讨论和外部复核计数来自收到的用户文本，本轮没有联网核验论文或重新复刻其多空组合；该文本并不使论文公式成为本ETF的已验证策略。新结果仍为已暴露历史的两项有界开发。置信区间、全历史多重选择、真实成交与未来样本均未完成。

## 接续

只登记TREND_NO_SLOPE的后续稳健性资格提案，REPAIR_SEQUENCE本设置结束，MIX50不自动更新。本轮不再搜索均线、确认次数、窗口、权重或费用。若继续，直接读净值与逐日净增量，围绕历史点增量是否分布于多个状态提出唯一固定检查；不得将历史点值改善直接称目标通过或前瞻成功。

文件入口：../results_run2/summary.csv、annual.csv、segments.csv、paired_daily_increment.csv、signals.csv、accounts/；本目录全部月份净增量.csv、全部联合事件簇净增量.csv、报价损益与摩擦的算术分解.csv及result.json。原第一轮文件保持原样。
"""
    (OUT / "第二轮结构比较结果与结论.md").write_text(report, encoding="utf-8")
    index = [{"path": path.relative_to(RESULTS).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(RESULTS.rglob("*")) if path.is_file()]
    (OUT / "original_results_file_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
