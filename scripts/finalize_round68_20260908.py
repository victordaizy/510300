"""保存现金分红比较的口径、实际账户和失败结论，不重新回测。"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.cash_distribution_funding_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table
from scripts.finalize_round66_20260907 import saved_cycles

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300现金分红与资金利率_第68轮_20260908"
DOCUMENT = OUT / "现金分红与资金利率_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_CASH_DISTRIBUTION_GAP_RECOVERY_20260908.md"


def verify(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "分红比较冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    factors = pd.read_parquet(RESEARCH / "factors.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    rates = pd.read_parquet(ROOT / cfg["rates"])
    rate_dates = pd.DatetimeIndex(rates.date)
    factor_checks = []
    for t, row in enumerate(factors.itertuples()):
        left = row.date - pd.DateOffset(years=1)
        complete = left >= pd.Timestamp(cfg["dividend_coverage_start"]) and row.date <= pd.Timestamp(cfg["dividend_coverage_end"])
        selected = dividends[dividends.ex_date.gt(left) & dividends.ex_date.le(row.date)]
        cash = float(selected.cash_dividend_per_share.sum()) if complete else np.nan
        require((np.isnan(cash) and np.isnan(row.past_year_distribution_per_share)) or abs(cash - row.past_year_distribution_per_share) < 1e-12, "事件逐条求和与保存分红不符")
        position = int(rate_dates.searchsorted(row.date, side="left")) - 1
        source_date = rate_dates[position] if position >= 0 else pd.NaT
        rate = float(rates.dr007.iloc[position]) / 100 if position >= 0 else np.nan
        age = (row.date - source_date).days if position >= 0 else np.nan
        valid = complete and np.isfinite(row.close) and np.isfinite(rate) and 0 <= age <= cfg["max_rate_age_days"]
        spread = cash / row.close - rate if valid else np.nan
        target = float(spread > 0) if valid else np.nan
        require((pd.isna(source_date) and pd.isna(row.dr_date)) or source_date == row.dr_date, "保存利率不是当时最近的较早日期")
        require((np.isnan(target) and np.isnan(row.target)) or target == row.target, "独立事件和来源查找不能还原目标")
        if valid:
            require(abs(spread - row.distribution_funding_spread) < 1e-12, "保存差额与事件现金和资金百分率不符")
        factor_checks.append({"date": row.date, "target": target, "cash_error": cash - row.past_year_distribution_per_share,
            "spread_error": spread - row.distribution_funding_spread, "rate_date": source_date})
    metrics, differences, cycles_all, groups, no_views, linear = [], [], [], [], [], []
    origins = 0
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {}
            for record in result[key]:
                if record["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{record['model']}_ledger.parquet")
                accounts[record["model"]] = ledger
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost"]:
                    require(abs(actual[field] - record[field]) < 1e-10, "指标不能由保存净值复算")
                metrics.append({"period": period, "cost": cost, "model": record["model"], "net_sharpe": actual["net_sharpe"]})
            current = accounts[PRIMARY]
            require(np.allclose(current.cash + current.shares * current.mark + current.dividend_receivable, current.equity, atol=1e-6, rtol=0), "新账户资产与净值不符")
            previous = np.r_[cfg["initial_capital"], current.equity.to_numpy()[:-1]]
            require(np.allclose(current.equity / previous - 1, current.net_return, atol=1e-13, rtol=0), "每日收益与净值不符")
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
                old = accounts[control]
                require(pd.DatetimeIndex(old.date).equals(pd.DatetimeIndex(current.date)), "对照和新账户日历不同")
                delta = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                delta["identity_error"] = delta["terminal_nav_difference"] - delta["price_difference"] - delta["dividend_difference"] + delta["commission_difference"] + delta["slippage_difference"]
                require(abs(delta["identity_error"]) < 1e-6, "账户差额经济分解未完成")
                differences.append(delta)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                target = factors.target.iloc[t]
                require(row.origin == factors.date.iloc[t] and row.execution_date == data.date.iloc[t + 1], "信号和执行时钟错位")
                require((np.isnan(target) and np.isnan(row.reference_weight)) or target == row.reference_weight, "实际请求目标不同于当时因子")
                account = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(account.shares) if account is not None else 0
                nav = float(account.equity) if account is not None else cfg["initial_capital"]
                close = float(data.close.iloc[t])
                request = 0
                if np.isfinite(target):
                    desired = math.floor(target * nav / close / cfg["lot"]) * cfg["lot"]
                    if target == 1 and shares > 0 and abs(target - shares * close / nav) < cfg["weight_band"]:
                        desired = shares
                    request = desired - shares
                require(request == row.requested_quantity, "实际份额请求不符合完整预算或无观点")
                if np.isnan(target):
                    execution = by_date.loc[row.execution_date]
                    require(execution.filled_quantity == 0, "本次真实无观点日发生意外成交")
                    no_views.append({"period": period, "cost": cost, "origin": row.origin, "execution_date": row.execution_date,
                        "rate_date": factors.dr_date.iloc[t], "rate_age": factors.dr_age_days.iloc[t], "shares_kept": int(execution.shares),
                        "source_state": factors.source_state.iloc[t]})
                origins += 1
            cycles = saved_cycles(current, dividends, cfg)
            cycles_all.extend({"period": period, "cost": cost, **cycle} for cycle in cycles)
            gains = sorted([x["net_profit"] for x in cycles], reverse=True)
            buys = current[current.filled_quantity.gt(0)]
            entry_counts = factors.set_index("date").loc[buys.origin, "past_year_distribution_count"].value_counts().to_dict()
            groups.append({"period": period, "cost": cost, "cycles": len(cycles), "positive_cycles": sum(x["net_profit"] > 0 for x in cycles),
                "negative_cycles": sum(x["net_profit"] < 0 for x in cycles), "net_profit": sum(x["net_profit"] for x in cycles),
                "gross_price_profit": sum(x["gross_price_profit"] for x in cycles), "dividend_recognized": sum(x["dividend_recognized"] for x in cycles),
                "commission": sum(x["commission"] for x in cycles), "slippage": sum(x["slippage"] for x in cycles),
                "top_two_cycle_profit": sum(gains[:2]), "top_three_cycle_profit": sum(gains[:3]),
                "additional_buys": int((current.shares_before.gt(0) & current.filled_quantity.gt(0)).sum()),
                "entries_with_one_past_year_distribution": int(entry_counts.get(1., 0)), "entries_with_two_past_year_distributions": int(entry_counts.get(2., 0))})
            x = np.column_stack([accounts["PANIC_LEARNED_HALF"].net_return, current.net_return])
            mean, covariance = x.mean(axis=0), np.cov(x, rowvar=False, ddof=1)
            solution = np.linalg.solve(covariance, mean)
            solution /= solution.sum()
            weights = [np.array([1., 0.]), np.array([0., 1.])]
            if (solution >= 0).all():
                weights.append(solution)
            values = [float(np.sqrt(cfg["annual_days"]) * w @ mean / np.sqrt(w @ covariance @ w)) for w in weights]
            best = int(np.argmax(values))
            linear.append({"period": period, "cost": cost, "correlation": float(np.corrcoef(x, rowvar=False)[0, 1]),
                "post_selected_saved_path_linear_sharpe": values[best], "original_half_weight": float(weights[best][0]),
                "cash_distribution_weight": float(weights[best][1]), "uses_entire_period_future_statistics": True,
                "tradable_strategy": False, "new_account_simulations": 0, "global_dynamic_strategy_upper_bound": False})
    for filename, rows in [("saved_factor_recomputation.csv", factor_checks), ("saved_metrics_recomputation.csv", metrics),
        ("saved_account_differences.csv", differences), ("saved_actual_cycles.csv", cycles_all), ("saved_cycle_profit_groups.csv", groups),
        ("saved_no_view_origins.csv", no_views), ("saved_two_path_future_linear_diagnostic.csv", linear)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "EVENT_FACTORS_AND_ACTUAL_CASH_ACCOUNTS_RECONCILED", "independent_factor_rows": len(factor_checks),
        "maximum_cash_sum_error": float(pd.DataFrame(factor_checks).cash_error.abs().max()), "recomputed_metrics": len(metrics),
        "account_differences": len(differences), "replayed_decision_origins": origins, "actual_cycles": len(cycles_all),
        "no_view_records": len(no_views), "no_view_records_with_actual_holdings": sum(x["shares_kept"] > 0 for x in no_views),
        "new_account_simulations": 0, "new_model_fits": 0, "security_audit_performed": False}, groups, linear


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 68)), "索引不是截至67轮，不能重复执行")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, groups, linear = verify(cfg, result)
    status = "COMPLETED_REJECTED_CASH_DISTRIBUTION_CALENDAR_CONCENTRATION"
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "主评价收益局部改善但夏普低于原学习及各半；较早仅2019两次短期交易、均在一年两次分红重叠期。结束该单策略和保存两路径固定权重尝试。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300现金分红与资金利率：第68轮", "", "## 先看结果", "",
        "**本轮已完成，目标1.2仍未实现，结束这套现金分红比较规则。** 主评价基础净夏普0.635、压力0.574；较早基础0.464、压力0.428。主评价基础年化6.87%，有局部收益改善，但最大回撤19.42%，风险调整表现仍低于原学习和原各半。较早五年只持仓18个收盘，不能把低回撤解释为稳定有效。", "",
        "从现有资料到完整四账户，不需要补齐EPS、财报或公募来源。先完成9项关键测试和来源时间核对，冻结一个条件后一次运行；没有用不同分红窗口或资金期限反复回测；组合部分仅做下文单独标明的事后数学诊断。", "",
        "|策略|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "PANIC_ONLY", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "原急跌单策略较高的主评价仅3个周期，较早只有8个且亏损，不能称为已经达标。原学习和原各半也是已反复观察的历史候选，未得到独立稳定超额证明。", "",
        "## 全部历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "主评价2020年1月2日至2026年8月14日开盘1604日；较早2015年1月5日至2019年12月31日开盘1219日。每段20万元，242日年化，含分红、佣金、滑点及全部空仓日。较早基础年化仅0.84%，低于买入持有3.90%；较低回撤主要来自长期空仓。", "",
        "## 失败原因与实际交易", "",
        "主评价32个完整周期、64笔成交、720个持仓收盘；较早2个周期、4笔成交、18个持仓收盘。两档费用没有未成交或追加。主评价32次买入中30次一年内有一次分红、2次有两次；较早两次全部落在一年内有两次分红的重叠区间。", "",
        "较早的第一次从2019年1月17日至1月24日，持仓5个收盘，基础净利润302.82元；第二次从2019年12月12日至统一终点12月31日开盘，持仓13个收盘，净利润8,280.14元。2015至2018年始终空仓。它抓到的只是很少的分配日历状态，并未证实长期识别估值机会。", "",
        "|历史及费用|完整周期|盈利周期|亏损周期|价格损益|实际分红|佣金|滑点|最终净利润|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        label = ("主评价" if g["period"] == "evaluation" else "较早") + ("／基础" if g["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{g['cycles']}|{g['positive_cycles']}|{g['negative_cycles']}|{g['gross_price_profit']:,.2f}元|{g['dividend_recognized']:,.2f}元|{g['commission']:,.2f}元|{g['slippage']:,.2f}元|{g['net_profit']:,.2f}元|")
    main = next(g for g in groups if g["period"] == "evaluation" and g["cost"] == "BASE")
    lines += ["", f"主评价基础最大的两个盈利周期合计{main['top_two_cycle_profit']:,.2f}元，占全部净利润{main['top_two_cycle_profit']/main['net_profit']:.2%}；最大的三个合计{main['top_three_cycle_profit']:,.2f}元，占{main['top_three_cycle_profit']/main['net_profit']:.2%}。其中最大一段从2025年6月19日至2026年8月14日开盘，净利润61,041.96元。这里按登记日分红权利逐周期归属，周期利润之和与完整账户净值一致，没有按净收益筛掉亏损周期。", "",
        "主评价有9个无观点决策原点，较早10个；主要来自利率超过7个自然日未更新，较早还包括最初没有已知利率。全部日期保留，没有把这些日子删掉或填成空仓。实际无观点时不下新请求，原持仓继续保留。每档费用下，主评价719个满目标原点，加上2个无观点保留持仓的执行日，再减去统一终点开盘退出的1日，得到720个持仓收盘；较早19个满目标原点减去终点1日，得到18个持仓收盘。", "",
        "## 是否值得再调组合权重", "",
        "为减少没有依据的组合回测，只对已保存的本轮净收益与原各半净收益做了一次事后线性代数比较。使用整个历史窗口的平均和协方差，求非负固定权重的数学最优值，并与两个端点比较，没有再产生真实成交账户。结果如下。", "",
        "|时期与费用|两路径相关性|事后最优线性混合夏普|", "|---|---:|---:|"]
    for row in linear:
        label = ("主评价" if row["period"] == "evaluation" else "较早") + ("／基础" if row["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{row['correlation']:.4f}|{row['post_selected_saved_path_linear_sharpe']:.3f}|")
    lines += ["", "这些值使用未来全期统计，属于不可执行诊断；两条已有账户收益线性混合也没有重算合并订单后的费用、整手和现金。它不能被称为新策略、达标结果或所有动态策略的上限。连这一有限的事后比较都未达到1.2，因此下一步不再扫描这两条路径的固定权重。", "",
        "## 全部因子与中文进出场规则", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1],
        "", "## 各年基础费用表现", "", "年份只展示差异，不能事后选择盈利年作为评价期。", "",
        "|时期|年份|净夏普|该年实际期间收益|成交笔数|", "|---|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        value = f"{row.net_sharpe:.3f}" if np.isfinite(row.net_sharpe) else "无波动，未定义"
        lines.append(f"|{'主评价' if row.period == 'evaluation' else '较早'}|{row.year}|{value}|{row.cumulative_return:.2%}|{row.trade_count}|")
    lines += ["", "2024和2025单年夏普虽然分别为1.227和1.924，完整主评价只有0.635，不能用两个年份宣称达到1.2。2021和较早2015至2018无交易、收益和波动均为零，夏普未定义，保留这一状态。", "",
        "## 已完成核对与下一步", "",
        f"9项必要测试一次通过，测试本体5.02秒。正式运行一次完成4个新账户、16个保存对照，无新增拟合。保存结果核对包括3456个独立事件与利率原点、20指标12账户差额、{receipt['replayed_decision_origins']}个份额请求和{receipt['actual_cycles']}个完整周期；事件现金求和最大差{receipt['maximum_cash_sum_error']:.3g}。两费用合计{receipt['no_view_records']}个无观点记录，其中{receipt['no_view_records_with_actual_holdings']}个实际有持仓，均保留原份额。", "",
        "登记前首次来源比较遇到微秒与纳秒时间存储精度不同；将比较对象统一为纳秒后，实际日期、时间、年龄和利率逐行相同。只修正比较方式，未改变信号、来源或账户规则；当时没有写入配置或计算新收益。测试和正式账户均没有失败，不额外计作一个已评价来源版本。", "",
        "同次工作已收尾第67轮连续趋势模型：49个季度估计全部收敛，但主基础夏普0.013、较早负0.546，下一收盘预测误差分别比价格持平大30.45%和36.46%，已经结束，不调整噪声或置信边界。详细49组参数在第67轮报告。", "",
        "下一项先检验一个现有日线即可表达的价格事件：低开后当日收盘收复前收盘，下一开盘进入；持有期间某日收盘不高于当日开盘，下一开盘退出。当天现金分红应加回开盘和收盘再判断是否真正低开、是否收复，避免除息制造虚假缺口。下一轮只做一个明确规则，尚未登记或回测；不再尝试本轮分红窗口、利率期限及固定组合权重。", "",
        "研究目标仍未完成，持续研究保持进行。只用510300及现金和现有免费来源，EPS慢源暂停，无GPT数值包，不扩大实际交易权限。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv",
        "source_receipt.json", "tests_receipt.json", "prefreeze_timestamp_comparison_note.json", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json",
        "saved_factor_recomputation.csv", "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv",
        "saved_no_view_origins.csv", "saved_two_path_future_linear_diagnostic.csv"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    columns = {"date": "日期", "origin_time": "判断时刻", "dr_date": "资金利率原日期", "dr_annual_rate": "已知年化资金利率小数",
        "dr_available_at": "资金利率可用时刻", "dr_age_days": "利率年龄自然日", "dr_valid": "利率可用", "close": "收盘价",
        "window_start_exclusive": "过去一年不含左端日", "dividend_coverage_valid": "分红窗口覆盖完整",
        "past_year_distribution_per_share": "过去一年每份现金分红", "past_year_distribution_count": "过去一年分红次数",
        "distribution_yield": "现金分红收益率", "distribution_funding_spread": "分红收益率减资金利率", "source_state": "资料状态", "target": "目标比例"}
    pd.read_parquet(RESEARCH / "factors.parquet").rename(columns=columns).to_csv(OUT / "全部逐日因子和状态.csv", index=False, encoding="utf-8-sig")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_trades.csv", OUT / f"{period}_{cost}_trades.csv")
    NEXT.write_text("""# 第68轮后：结束分红比较，转向有限的日内收复事件

第68轮研究、9测试、单配置冻结、4新账户16旧对照、保存复算和中文交付已完成。主BASE/STRESS夏普0.634675/0.574148，基础年化6.8709%、回撤19.4172%、累计55.3402%、32周期64成交720持仓收盘；早0.463786/0.427629、基础年化0.8377%、回撤1.9294%、累计4.2915%、2周期4成交18持仓。0追加、0未成交。主9/早10个无观点原点均保留，非删日或填0；每档主719满目标加2无观点保留持仓执行日减1终点退出=720持仓收盘，早19减1终点=18。

早只有2019-01-17至01-24的5个收盘、净302.82032元，2019-12-12至12-31的13个收盘、净8280.13806元，两入场均过去一年两次分红；2015—2018无成交且夏普不定义。主30入场有1次分红，2入场有2次。主最大两周期净90976.10748元，最大三109062.64540元，相对全净110680.36122元分别约82.20%/98.54%，最大2025-06-19至2026-08-14有282持仓收盘、净61041.95748。2024/2025单年夏普1.227/1.924不构成全期达标。

只做了保存本轮与R46各半两路径的事后线性代数诊断：主BASE/STRESS相关0.258517/0.258122、非负固定权重最优夏普1.027284/0.949654；早相关负0.001159/负0.001012、最优0.762281/0.714616。用了全期未来平均协方差，非新策略、无新账户，不是动态策略上限，不扫描固定权重。

资料均原有：14官方分红、2905条DR007百分率、3456行情行。DR下一股票交易日9:30可用，≤7自然日时效，独立对齐旧R7日期与率一致，不借旧整行共同feature_valid。登记前微秒/纳秒比较错误仅dr_available_at单位不同，统一纳秒后所有真实时间值相同，没有配置或收益，记录prefreeze_timestamp_comparison_note.json；无需再跑9测试或原4账户。源统计和原始送达未证实边界在source_receipt.json；无新数据下载。最终冻结代码research/cash_distribution_funding_inputs_v1.py与cash_distribution_funding_v1.py、config/510300_cash_distribution_funding_v1.json、docs/510300_CASH_DISTRIBUTION_FUNDING_V1.md，结果reports/research/510300_cash_distribution_funding_v1，中文deliverables/510300现金分红与资金利率_第68轮_20260908。3456独立事件因子、20指标12差额5646份额请求68含分红周期已核对。finalize_round68_20260908.py完成后不重跑。没有模型拟合或参考周期。

索引应截至68：342不同设置、355已评价来源版本、360登记含5旧未运行、1102主评价记录。第67轮中文交付本次已先完成，49真实拟合都收敛，保存复算0新拟合；主0.012866/负0.143978、早负0.545759/负0.675094，预测MSE高30.45%/36.46%，闭合失败。两轮结果不能混淆。

下一项69仅方向，尚未写实现、登记、测试或回测。限定一个日线价格事件：当日开盘加当天每份分红严格低于前收盘，但收盘加当天分红严格高于前收盘，称低开后收复；下一开盘进入。持仓期间只要某日收盘不高于当日开盘，下一开盘退出。入场条件本身要求收盘高于开盘，与退出不冲突；等于前收盘不新进入，等于当日开盘触发退出。重新进入等新的有效低开收复事件，不设额外等待。每份分红加回只修正除息缺口，不加在账户利润两次。无资料时无观点保留份额，首次可用无入场事件明确初始化空仓；初始与未成交的目标状态、10个百分点带宽及终点规则必须先写中文再冻结。仅现有OHLC与分红，下一开盘才行动，不在同日收盘信号价成交；不设百分比阈值或新预测模型。

限定中文和英文关键词查询research/config/docs，只命中无关third_party_gap_fill_allowed，未找到上述独立进出场的同名实现；这不能证明过去模型从未含开盘或日内因子。后续只查所需旧规则，避免重开全部历史。一次必要测试后冻结一个候选；若失败不改变缺口比例、强弱阈值或改称另一路救活。仅510300现金、原两窗口两费用、完整现金日、分红、整手、T+1和涨跌停。EPS/公募慢源继续暂停，无GPT包，无子任务，无交易。目标仍未实现，继续研究。
""", encoding="utf-8")
    record = {"round": 68, "study": result["study_id"], "title": "过去现金分红收益率与资金利率比较", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND68_COMPLETE_CASH_DISTRIBUTION_FUNDING_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计68轮，342不同设置，355已评价来源版本，360登记含5旧未运行，1102主评价记录。",
        checks="第68轮9测试、3456独立事件利率因子、20指标12差额5646请求68含分红周期；无观点保留和事后两路径比较已保存。",
        process_state_note="第67与68轮归因和中文交付均完成；第69轮仅低开收复与日内转弱退出方向，尚未登记。",
        next_work={"status": "GAP_RECOVERY_ENTRY_INTRADAY_WEAKNESS_EXIT_DIRECTION", "focus": "只用现有日线，低开收复后进入、日内转弱退出，一个候选", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_cash_distribution_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "CASH_DISTRIBUTION_FUNDING_ROUND68_CHINESE_RESULTS", "rounds": [68], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "CASH_DISTRIBUTION_RULES_AND_ACCOUNT_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt, "周期": groups, "保存两路径事后诊断": linear}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
