"""独立以频数加权重算抽样，交付事前状态归因及收益口径区别。"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from research.saved_state_failure_diagnostic_v1 import ROOT, OUT, CONFIG, SOURCE, STATES, CONTRASTS
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "状态归因冻结来源改变")
    table = pd.read_csv(OUT / "state_contributions_and_simultaneous_intervals.csv")
    daily = pd.read_parquet(OUT / "paired_daily_attributions.parquet")
    market = pd.read_parquet(OUT / "market_states.parquet").set_index("date")
    conventions, matrices, observed = [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        columns = []
        for cost in cfg["costs"]:
            ledgers = {model: pd.read_parquet(SOURCE / period / cost / f"{model}_ledger.parquet")
                for model in {name for pair in CONTRASTS.values() for name in pair[:2]}}
            for contrast, (left, right, _) in CONTRASTS.items():
                g = daily[daily.period.eq(period) & daily.cost.eq(cost) & daily.contrast.eq(contrast)].copy()
                a, b = ledgers[left], ledgers[right]
                require(pd.DatetimeIndex(g.date).equals(pd.DatetimeIndex(a.date)) and pd.DatetimeIndex(g.origin).equals(pd.DatetimeIndex(a.origin)), "状态归因配对日历改变")
                require(np.array_equal(g.state, market.loc[pd.DatetimeIndex(g.origin), "state"]), "状态归因使用当天而非前一收盘")
                np.testing.assert_allclose(g.net_difference, a.net_return-b.net_return, atol=1e-12, rtol=0)
                n = len(g)
                subset = table[table.period.eq(period) & table.cost.eq(cost) & table.contrast.eq(contrast)]
                subtotal = 0.
                for state in STATES:
                    mask = g.state.eq(state)
                    row = subset[subset.state.eq(state)].iloc[0]
                    require(row.observations == int(mask.sum()) and row.full_calendar_rows == n, "状态归因分母改变")
                    values = g.net_difference.to_numpy()*mask.to_numpy()
                    mu = math.fsum(g.loc[mask, "net_difference"])*cfg["annual_days"]/n
                    require(abs(mu-row.annual_mean_net_contribution) < 1e-12, "状态归因不能逐组还原")
                    require(abs(row.annual_price_contribution+row.annual_dividend_contribution-row.annual_fee_contribution-mu) < 1e-11, "状态收益分解不一致")
                    columns.append(values)
                    observed.append(mu)
                    subtotal += mu
                require(abs(subtotal-cfg["annual_days"]*math.fsum(a.net_return-b.net_return)/n) < 1e-12, "状态贡献合计不能还原全部日均差")
                if contrast == "143_VS_BUY_HOLD":
                    cagr_a = (a.equity.iloc[-1]/cfg["initial_capital"])**(cfg["annual_days"]/n)-1
                    cagr_b = (b.equity.iloc[-1]/cfg["initial_capital"])**(cfg["annual_days"]/n)-1
                    conventions.append({"period": period, "cost": cost, "strategy_cagr": cagr_a, "benchmark_cagr": cagr_b,
                        "cagr_excess": cagr_a-cagr_b, "strategy_annual_arithmetic_mean": cfg["annual_days"]*a.net_return.mean(),
                        "benchmark_annual_arithmetic_mean": cfg["annual_days"]*b.net_return.mean(), "annual_arithmetic_mean_excess": subtotal})
        matrices.append(np.column_stack(columns))
    observed = np.array(observed)
    np.testing.assert_allclose(observed, table.annual_mean_net_contribution, atol=1e-12, rtol=0)
    checks = []
    for block in cfg["blocks"]:
        saved = pd.read_parquet(OUT / f"bootstrap_contributions_block{block}.parquet").to_numpy(float)
        rng = np.random.default_rng(cfg["random_seed"]+block)
        offset = 0
        maximum_error = 0.
        for matrix in matrices:
            n, k = matrix.shape
            for begin in range(0, cfg["repetitions"], 100):
                size = min(100, cfg["repetitions"]-begin)
                starts = rng.integers(0, n, size=(size, math.ceil(n/block)))
                indices = ((starts[:, :, None]+np.arange(block)) % n).reshape(size, -1)[:, :n]
                # 用抽样日频数乘各日贡献，不使用原脚本的重复展开数组均值。
                frequencies = np.array([np.bincount(row, minlength=n) for row in indices], float)
                recomputed = frequencies @ matrix * cfg["annual_days"]/n
                error = float(abs(recomputed-saved[begin:begin+size, offset:offset+k]).max())
                maximum_error = max(maximum_error, error)
                require(error < 1e-11, "状态共同区块抽样不能按频数重算")
            offset += k
        sd = saved.std(axis=0, ddof=1)
        maxima = np.max(abs((saved[:, sd > 0]-observed[sd > 0])/sd[sd > 0]), axis=1)
        ordered = np.sort(maxima)
        rank = .95*(len(ordered)-1)
        a, b = math.floor(rank), math.ceil(rank)
        critical = float(ordered[a]+(rank-a)*(ordered[b]-ordered[a]))
        np.testing.assert_allclose(table[f"block{block}_simultaneous_lower"], observed-critical*sd, atol=1e-11, rtol=0)
        np.testing.assert_allclose(table[f"block{block}_simultaneous_upper"], observed+critical*sd, atol=1e-11, rtol=0)
        checks.append({"block": block, "recomputed_statistics": int(saved.size), "maximum_frequency_recomputation_error": maximum_error, "critical95": critical})
    require(all((table[f"block{block}_simultaneous_lower"].le(0) & table[f"block{block}_simultaneous_upper"].ge(0)).all() for block in cfg["blocks"]), "本次已读结论与同时区间不同")
    pd.DataFrame(conventions).to_csv(OUT / "return_conventions.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_PREVIOUS_CLOSE_GROUPS_FULL_CALENDAR_AND_FREQUENCY_BOOTSTRAP_CHECKED",
        "comparison_cells": len(table), "daily_rows": len(daily), "bootstrap_checks": checks, "new_accounts": 0, "new_models": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    document = ROOT / "deliverables/510300事前状态失败归因_20260909/为什么切换和增加过滤仍然失败.md"
    require(not document.parent.exists(), "状态归因交付目录已经存在")
    document.parent.mkdir(parents=True)
    lines = ["# 为什么切换策略和增加过滤仍然失败", "",
        "本次读取现有账户，没有新增策略或回测。结论是：确实能看到一些事前状态与相对表现有关，但48项比较在20日、60日区块的同时区间全部跨零；还不能据此建立有证据支持的四状态硬切换。当前夏普1.2和稳定超额目标仍未实现。", "",
        "## 先区分两种收益口径", "", "第143轮四情景的复合年化收益均高于买入持有，但年化日均收益均低于买入持有。前者反映整段实际复利，后者反映每日简单收益的平均水平；较低波动有利于复利累积，二者可以同时出现。不能把正的复合收益超额直接称为日均收益预测优势，也不能只凭日均差为负就断言不存在择时价值。", "",
        "|时期|费用|143复合年化|买持复合年化|复合年化超额|年化日均收益差|", "|---|---|---:|---:|---:|---:|"]
    for row in conventions:
        lines.append(f"|{'主历史' if row['period']=='evaluation' else '较早历史'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['strategy_cagr']:.2%}|{row['benchmark_cagr']:.2%}|{row['cagr_excess']:.2%}|{row['annual_arithmetic_mean_excess']:.2%}|")
    lines.extend(["", "## 从这次归因可以看到什么", "",
        "143在趋势向上、波动未上升状态的相对买持贡献四项均负，趋势向上、波动上升状态四项均正。以基础费用为例，前者在主／较早历史的年化日均差贡献为−1.28／−7.84个百分点，后者为+2.20／+3.08个百分点。这是保存账户的状态归因，不能直接拼成新切换账户。", "",
        "150只减预算在趋势未向上、波动未上升状态四项贡献都略正，但在较早上涨、波动未上升阶段基础费用贡献为−1.94个百分点，损失远大于那些小幅改善。这解释了为什么主历史夏普改善不能代表跨时期有效。", "",
        "151平均K线确认在三个状态的四项贡献都负，只在趋势向上、波动上升状态四项略正；所有相关区间仍跨零。不据此把失败规则改成只在这个状态启用。", "",
        "同一完整交易会跨越多个状态，日收益之间也存在关联。已有143主历史只有23个完整实际周期、较早只有10个周期，大量日线行不等于同样数量的独立机会。分组会进一步减少有效信息。", "",
        "接下来检查已有退出模型的同周期残差相关性，优先复用保存样本和模型，不继续加入未经证明的价格过滤。新的统计误差模型也必须另行冻结并由实际账户检验，不会因为本报告就自动采用。", "",
        "## 全部48项状态贡献与同时区间", "", "下面贡献及区间均为完整区间的年化日均收益差，单位为百分点；不是新策略年化收益或夏普。", ""])
    for (period, cost), group in table.groupby(["period", "cost"], sort=False):
        lines.extend([f"### {'主历史' if period=='evaluation' else '较早历史'}／{'基础费用' if cost=='BASE' else '压力费用'}", "",
            "|比较|事前状态|日数|净贡献|价格贡献|分红贡献|费用差贡献|20日区块同时区间|60日区块同时区间|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in group.to_dict("records"):
            lines.append(f"|{row['comparison_name']}|{row['state_name']}|{row['observations']}|{100*row['annual_mean_net_contribution']:.3f}|{100*row['annual_price_contribution']:.3f}|{100*row['annual_dividend_contribution']:.3f}|{100*row['annual_fee_contribution']:.3f}|[{100*row['block20_simultaneous_lower']:.3f}, {100*row['block20_simultaneous_upper']:.3f}]|[{100*row['block60_simultaneous_lower']:.3f}, {100*row['block60_simultaneous_upper']:.3f}]|")
        lines.append("")
    lines.extend(["费用差为正代表新方案费用更多，净贡献等于价格贡献加分红贡献再减费用差贡献。", "",
        f"归因核心计算{result['run_seconds']:.3f}秒，另完成独立逐组还原及192000项保存抽样统计的频数法重算。两个区块长度各同时覆盖48个比较，均未校正所有历史策略筛选，也不构成独立验证。", "",
        "## 完整计算规则", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])
    document.write_text("\n".join(lines)+"\n", encoding="utf-8")
    import shutil
    for name in ["state_contributions_and_simultaneous_intervals.csv", "full_calendar_reconciliation.csv", "cross_period_state_patterns.csv", "return_conventions.csv", "result.json", "saved_verification_receipt.json"]:
        shutil.copy2(OUT / name, document.parent / name)
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 151 and not index["goal_achieved"], "状态归因索引轮次不同")
    index["saved_state_failure_diagnostic"] = {"result": str((OUT / "result.json").relative_to(ROOT)), "document": str(document), "new_accounts": 0,
        "conclusion": "ALL_48_SIMULTANEOUS_INTERVALS_CROSS_ZERO_NO_STATE_ROUTER_PROMOTION"}
    index["next_work"] = {"status": "CYCLE_SERIAL_ERROR_INPUT_PREFLIGHT", "focus": "检查原成熟周期残差相邻相关，准备保留首行及周期权重的广义最小二乘退出学习",
        "source": "docs/510300_CYCLE_SERIAL_ERROR_NEXT_20260909.md", "candidate_round": 152, "registered": False, "planned_settings": 1, "planned_new_accounts": 4}
    index["deliveries"].append({"created_at": now(), "type": "SAVED_STATE_FAILURE_DIAGNOSTIC", "directory": str(document.parent), "main_document": str(document), "new_gpt_review_archive_created": False})
    index["updated_at"] = now()
    write_json(path, index)
    print(json.dumps({"文档": str(document), "核对": receipt, "收益口径": conventions, "下一项": index["next_work"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
