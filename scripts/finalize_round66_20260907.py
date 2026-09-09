"""复算波幅通道、完整成交周期和阶段差异，交付中文规则。"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.atr_trend_bands_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300真实波幅趋势通道_第66轮_20260907"
DOCUMENT = OUT / "真实波幅趋势通道_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_ATR_CONTINUOUS_TREND_20260907.md"


def saved_cycles(ledger, dividends, cfg):
    cycles, owners = [], {}
    active = None
    for row in ledger.itertuples():
        if row.filled_quantity > 0 and row.shares_before == 0:
            require(active is None, "保存账户周期重叠")
            active = {"cycle": len(cycles) + 1, "entry_date": row.date, "entry_origin": row.origin,
                "buy_debit": 0., "gross_price_profit": 0., "commission": 0., "slippage": 0.,
                "dividend_recognized": 0., "held_closes": 0, "buy_trades": 0, "sell_trades": 0}
        if row.filled_quantity != 0:
            require(active is not None, "实际成交没有归属周期")
            active["gross_price_profit"] -= row.filled_quantity * row.open_price
            active["commission"] += row.commission
            active["slippage"] += row.slippage_cost
            if row.filled_quantity > 0:
                active["buy_debit"] += row.filled_quantity * row.fill_price + row.commission
                active["buy_trades"] += 1
            else:
                active["sell_trades"] += 1
        if row.shares > 0:
            require(active is not None and row.mark_clock != "OPEN_TERMINAL", "未平仓终点不能算完成周期")
            active["held_closes"] += 1
            owners[row.date] = (active["cycle"], int(row.shares))
        if active is not None and row.filled_quantity < 0 and row.shares == 0:
            active.update(exit_date=row.date, exit_origin=row.origin, terminal_exit=row.mark_clock == "OPEN_TERMINAL")
            cycles.append(active)
            active = None
    require(active is None, "仍有未完成周期")
    dates = set(ledger.date)
    for event in dividends.itertuples():
        owner = owners.get(event.record_date)
        if owner is not None and event.ex_date in dates:
            number, shares = owner
            cycles[number - 1]["dividend_recognized"] += shares * event.cash_dividend_per_share
    for row in cycles:
        row["net_profit"] = row["gross_price_profit"] + row["dividend_recognized"] - row["commission"] - row["slippage"]
        row["cycle_net_return"] = row["net_profit"] / row["buy_debit"]
    require(abs(sum(x["net_profit"] for x in cycles) - (ledger.equity.iloc[-1] - cfg["initial_capital"])) < 1e-6, "周期利润含分红权利后仍不等于终值")
    require(abs(sum(x["dividend_recognized"] for x in cycles) - ledger.dividend_recognized.sum()) < 1e-6, "周期分红权利与实际应收不符")
    return cycles


def verify(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "通道冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "factors.parquet")
    require(pd.DatetimeIndex(f.date).equals(pd.DatetimeIndex(data.date)), "通道因子日历改变")
    tr = pd.concat([f.wealth_high - f.wealth_low, (f.wealth_high - f.wealth_close.shift()).abs(),
        (f.wealth_low - f.wealth_close.shift()).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    window = cfg["atr_window"]
    seeded = tr.copy()
    seeded.iloc[:window] = np.nan
    seeded.iloc[window] = tr.iloc[1:window + 1].mean()
    atr = seeded.ewm(alpha=1 / window, adjust=False).mean()
    require(np.allclose(tr, f.true_range, atol=1e-12, rtol=0, equal_nan=True), "真实波幅不能独立复算")
    require(np.allclose(atr, f.atr, atol=1e-12, rtol=0, equal_nan=True), "威尔德平滑不能由初值及指数递推复算")
    upper = np.where((f.basic_upper < f.upper_band.shift()) | (f.wealth_close.shift() > f.upper_band.shift()), f.basic_upper, f.upper_band.shift())
    lower = np.where((f.basic_lower > f.lower_band.shift()) | (f.wealth_close.shift() < f.lower_band.shift()), f.basic_lower, f.lower_band.shift())
    after_seed = slice(window + 1, None)
    require(np.allclose(upper[after_seed], f.upper_band.iloc[after_seed], atol=1e-12, rtol=0), "延续上轨递推错误")
    require(np.allclose(lower[after_seed], f.lower_band.iloc[after_seed], atol=1e-12, rtol=0), "延续下轨递推错误")
    expected = np.where(f.trend_state.shift().eq(0), f.wealth_close.gt(f.upper_band), f.wealth_close.ge(f.lower_band)).astype(float)
    require(np.array_equal(expected[after_seed], f.target.iloc[after_seed]), "方向切换与当前边界不符")
    require(f.target.iloc[:window].isna().all() and f.target.iloc[window] == 0, "初始化或无观点处理错误")
    metrics, differences, replay, cycle_rows, groups = [], [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {}
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                accounts[m["model"]] = ledger
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "现金份额和分红应收不等于净值")
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0), "净收益不能从完整净值复算")
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "mean_exposure"]:
                    require(abs(actual[field] - m[field]) < 1e-10, "保存指标不能复算")
                metrics.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": actual["net_sharpe"]})
            current = accounts[PRIMARY]
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
                old = accounts[control]
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(old.date)), "对照日历不同")
                d = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                d["reconciliation_error"] = d["terminal_nav_difference"] - d["price_difference"] - d["dividend_difference"] + d["commission_difference"] + d["slippage_difference"]
                require(abs(d["reconciliation_error"]) < 1e-6, "新旧账户差额没有解释完全")
                differences.append(d)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                require(row.origin == f.date.iloc[t] and row.execution_date == data.date.iloc[t + 1], "收盘判断与下一开盘时钟不符")
                require(row.reference_weight == f.target.iloc[t], "账户目标与当时通道不符")
                actual = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(actual.shares) if actual is not None else 0
                nav = float(actual.equity) if actual is not None else cfg["initial_capital"]
                close, target = float(data.close.iloc[t]), float(row.reference_weight)
                target_shares = math.floor(target * nav / close / cfg["lot"]) * cfg["lot"]
                if target == 1 and shares > 0 and abs(target - shares * close / nav) < cfg["weight_band"]:
                    target_shares = shares
                require(row.requested_quantity == target_shares - shares, "请求没有使用实际净值和整手")
            cycles = saved_cycles(current, dividends, cfg)
            cycle_rows.extend({"period": period, "cost": cost, **x} for x in cycles)
            positive = sorted((x["net_profit"] for x in cycles if x["net_profit"] > 0), reverse=True)
            groups.append({"period": period, "cost": cost, "cycles": len(cycles), "positive_cycles": len(positive),
                "negative_cycles": sum(x["net_profit"] < 0 for x in cycles), "net_profit": sum(x["net_profit"] for x in cycles),
                "largest_positive_cycle_profit": positive[0] if positive else 0., "top_two_positive_profit": sum(positive[:2]),
                "initial_target": float(decisions.reference_weight.iloc[0]), "additional_buys": int((current.shares_before.gt(0) & current.filled_quantity.gt(0)).sum())})
            replay.append({"period": period, "cost": cost, "replayed_decision_origins": len(decisions), "held_closes": int(current.shares.gt(0).sum())})
    for name, rows in [("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_state_and_request_replay.csv", replay), ("saved_actual_cycles.csv", cycle_rows), ("saved_cycle_profit_groups.csv", groups)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "SAVED_BANDS_ACCOUNTS_ACTUAL_CYCLES_AND_REQUESTS_RECONCILED",
        "recomputed_account_records": len(metrics), "account_differences": len(differences), "factor_rows_recomputed": len(f),
        "maximum_atr_recomputation_error": float((atr - f.atr).abs().max()),
        "replayed_decision_origins": sum(x["replayed_decision_origins"] for x in replay), "recomputed_closed_cycles": len(cycle_rows),
        "new_diagnostic_accounts": 0, "new_models": 0, "security_audit_performed": False}, differences, groups


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 66)), "索引不是截至65轮，不能重复更新")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, differences, groups = verify(cfg, result)
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_REJECTED_ATR_TREND_BANDS_NO_STABLE_EXCESS_OR_HIGH_SHARPE"
    decision = "完整两段两费用的夏普均明显低于原学习和原各半；主基础仅小幅超过买入持有年化，压力费用反而低于持有，较早两费用年化均低于持有。结束通道，不调窗口、倍数或初始化。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300真实波幅趋势通道：第66轮结果", "", "## 先看结论", "",
        "**这套简单通道没有实现稳定超额或高夏普，结束本方案。** 主评价基础净夏普0.351、压力费用0.320；较早基础0.309、压力0.276，均低于原学习和原各半组合。没有训练模型或补齐新来源。", "",
        "核心规则是：价格突破由近期波幅计算的上轨，下一开盘进入；收盘跌破随趋势移动的下轨，下一开盘退出。固定10日平均真实波幅、3倍带宽，运行后没有改动参数。", "",
        "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "PANIC_ONLY", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "主评价2020年1月2日至2026年8月14日开盘，共1604日；较早2015年1月5日至2019年12月31日开盘，共1219日。各20万元起始，全部空仓日和实际费用保留。原急跌单独主评价较高只有3个周期，较早8个周期，仍非稳定达标证据。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "## 为什么没有达到目标", "",
        "它约一半时间持有股票：主评价805个持仓收盘、基础平均股票仓位49.88%；较早635个持仓收盘、平均仓位52.02%。主评价20个完整周期40笔成交，较早17个完整周期34笔成交；四个账户都没有持仓追加或未成交。交易不算频繁，但价格风险暴露较大，收益不足以补偿这些波动。", "",
        "主评价基础年化3.95%，比买入持有只高0.325个百分点；压力费用下年化3.51%，反而低0.101个百分点。较早基础年化3.62%、压力3.10%，均低于买入持有。最大回撤基础主17.26%、较早23.02%，虽然小于长期满仓的回撤，也没有带来目标要求的夏普。", "",
        "|历史及费用|完整周期|盈利周期|亏损周期|周期净利润合计|最大盈利周期|最大两个盈利周期合计|", "|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        label = ("主评价" if g["period"] == "evaluation" else "较早") + ("／基础" if g["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{g['cycles']}|{g['positive_cycles']}|{g['negative_cycles']}|{g['net_profit']:,.2f}元|{g['largest_positive_cycle_profit']:,.2f}元|{g['top_two_positive_profit']:,.2f}元|")
    lines += ["", "周期利润以真实买卖价格和费用计算；分红按登记日实际持有份额归属，即使退出后才除息或到账，也不会漏记或重复算到别的周期。全部周期利润之和已与账户终值核对。", "",
        "两段开始评价时，通道已处于上行状态，因此第一次买入是在开始运行时采用已有趋势，并不是首日刚发生新突破。后面的进入才由后续状态变化产生。较早最后一个原点仍为上行，但研究终点统一开盘退出，所以636个上行决策原点对应635个持仓收盘；主评价两者均为805。", "",
        "|历史及费用|对照|终值差|价格损益差|分红差|佣金差|滑点差|", "|---|---|---:|---:|---:|---:|---:|"]
    for d in differences:
        if d["control"] not in ["REARM_RIDGE", "BUY_HOLD"] or d["cost"] != "BASE":
            continue
        label = "主评价／基础" if d["period"] == "evaluation" else "较早／基础"
        name = "原学习" if d["control"] == "REARM_RIDGE" else "买入持有"
        lines.append(f"|{label}|{name}|{d['terminal_nav_difference']:,.2f}元|{d['price_difference']:,.2f}元|{d['dividend_difference']:,.2f}元|{d['commission_difference']:,.2f}元|{d['slippage_difference']:,.2f}元|")
    lines += ["", "差额为新账户减对照；终值差等于价格差加分红差，再扣佣金差和滑点差。不能把全部差异仅归因于手续费。", "",
        "## 所有因子和完整进出场规则", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1],
        "", "## 各年基础费用结果", "", "年度拆分用于说明阶段差异。首末年可能为部分期间，不能在看到收益后只挑盈利年份交易。", "",
        "|时期|年份|净夏普|实际期间收益|复合年化|成交笔数|", "|---|---:|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for r in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        lines.append(f"|{'主评价' if r.period == 'evaluation' else '较早'}|{r.year}|{r.net_sharpe:.3f}|{r.cumulative_return:.2%}|{r.annualized_return:.2%}|{r.trade_count}|")
    lines += ["", "例如2017年单年夏普1.303，2018年却为负1.474。一个年份超过1.2不能替代整个账户历史的验收，更不能在事后据此指定下一年的交易策略。", "",
        "## 实际运行与后续方向", "",
        "11项必要测试3.56秒通过后冻结，覆盖隔夜缺口、初始均值、递推、边界触及、纯现金除息、缺失重启、未来数据不改过去和下一开盘真实成交。正式一次计算4个新账户，复用16个对照，运行无中断。", "",
        f"首行前收盘缺失原样保留，其余3455行财富高低收盘的两种因果推导已在冻结前核对。保存结果又复算3456行波幅与轨道，平均波幅最大误差{receipt['maximum_atr_recomputation_error']:.3g}；20个账户指标、12条差额、{receipt['replayed_decision_origins']}个原点目标与实际请求及{receipt['recomputed_closed_cycles']}个含分红实际周期全部核对，没有新归因账户。", "",
        "下一步不继续替换通道参数，改研究能否每天更新一个连续的趋势估计：将价格变化拆成趋势变化与短期噪声，只用当时已有价格估计，方向强于自身不确定性时进入，方向转弱时退出。拟按季度使用最近504个交易日更新噪声参数，日常逐日递推；只设计一个方案，不依据全期夏普选参数。", "",
        "机制参考[结构时间序列模型官方说明](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html)。当前环境未安装该文档对应的建模库，但已有优化库，可直接实现小型滤波与过去价格似然估计，无需补行情。当前仅完成方向和依赖检查，第67轮尚未登记、拟合或计算账户，具体初值、拟合失败和进出场边界须先写明并测试后冻结。", "",
        "EPS与其他慢源继续暂停，仍只有510300及现金，无GPT数值包、无真实交易。目标未完成，持续研究保持进行。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_state_and_request_replay.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv",
        "tests_receipt.json", "wealth_scale_receipt.json", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    pd.read_parquet(RESEARCH / "factors.parquet").to_csv(OUT / "全部逐日价格因子及通道.csv", index=False, encoding="utf-8-sig")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_trades.csv", OUT / f"{period}_{cost}_trades.csv")
    NEXT_NOTE.write_text("""# 第66轮后：连续趋势与噪声的逐日估计

第66轮10日ATR、3倍Supertrend通道结束：主BASE/STRESS夏普0.350996/0.319732、基础年化3.9507%、回撤17.2616%；较早0.308628/0.275663、基础年化3.6213%、回撤23.0201%。两段两费用都低于原R32和R46夏普。主基础较BH年化只多0.3249个百分点，压力少0.1005；早两费用均少。主20完整周期40成交805持仓收盘、早17周期34成交635持仓，0追加0未成交。两段初始目标1；早636上行原点对635收盘持仓因终点开盘统一退出，不能误判时钟漏行。主2017不属于主评价，早2017单年夏普1.303而2018负1.474，不能选盈利年。

正式研究 research/atr_trend_bands_v1.py、atr_trend_bands_inputs_v1.py，config/510300_atr_trend_bands_v1.json、docs/510300_ATR_TREND_BANDS_V1.md。报告 reports/research/510300_atr_trend_bands_v1，finalize_round66_20260907.py已完成，不重跑。11测试3.56秒通过，0中断。冻结前3455行财富高低收盘因果推导核对，首行前收盘未知保留；保存3456因子、20指标12差额5646原点请求74真实闭合周期含分红权利复算。周期利润不是直接把持仓日收益相加：按真实买卖及登记份额的分红权利归属，退出后除息不漏记。saved_actual_cycles.csv与saved_cycle_profit_groups.csv有全部明细。不改ATR窗口、3倍、初始方向、等待或止损挽救。

索引累计66轮340不同设置、353已评价来源版本、358登记含5旧未运行、1082主评价记录。1新设置4新账户16对照复用，0模型拟合。目标未完成，不称旧任一候选全局最好或独立稳定。

下一项67只有方向，尚未登记、拟合或计算。研究一个连续趋势滤波，避免继续用固定穿越阈值或对两个旧专家反复调配。价格被看成潜在水平、每天的趋势斜率与短期观测噪声；用当前及过去的对数含分红财富价格逐日修正，不用未来平滑状态。拟最多一个方案：每季度第一个收盘用最近504交易日价格估计噪声参数，其余日只递推已估计模型；趋势正向且超过自身一倍不确定性时进入，趋势非正时退出，其他情况下按明确状态保持。固定完整账户目标0/1、下一开盘、10个百分点带宽。这里只是方向，进入边界、置信尺度、初始化、估计失败时NO_VIEW与已有退出、季度更新同日是否滤波两次等必须在读新账户收益前明确并测试。

官方结构时间序列机制已读 https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html 。当前.venv没有statsmodels，已有scipy1.18.0，不因依赖开始漫长补齐；可用小型二维Kalman递推及既有优化库做高斯创新似然。优先只估计趋势变化噪声与观测噪声两个参数（平滑趋势特例，水平不另加自由噪声），不扫按夏普选择的超参数网格。任何优化仅使用该季度当时已知的训练价格，存每次参数和收盘后滤波状态；不能把全期拟合或未来平滑预测回填过去。必须先定义固定参数边界、数值收敛和失败处理，一次运行。没有拟合成功不要填零趋势或假设空仓。

限定查重在research/config/docs的Kalman、卡尔曼、UnobservedComponents、local linear trend等只有以前的方向说明，无实现命中；旧HMM、ADX、规则切换与学习候选失败保留，不宣称所有状态研究从未做过。本方向改用连续潜在趋势，而非救活旧离散家族。若无有意义政策或账户失败，结束并记录，不改训练窗、季度频率、置信边界等邻近参数挽救。

只510300与现金，原20万元、主2020-01-02至2026-08-14开盘、早2015-01-05至2019-12-31开盘、242年化、现金及无风险0、两费用、整手、分红、T+1及限价口径保持。EPS慢源补齐暂停，无GPT数值包及额外安全审计，不新子任务，不真实交易。继续有依据的方法研究。
""", encoding="utf-8")
    record = {"round": 66, "study": result["study_id"], "title": "平均真实波幅趋势通道独立进出场", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND66_COMPLETE_ATR_TREND_BANDS_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计66轮，340不同设置，353已评价来源版本，358登记含5旧未运行，1082主评价记录。",
        checks="第66轮11测试，3455因果财富行、3456波幅通道行、20指标12差额5646原点请求74含分红真实周期复算。",
        process_state_note="第66轮账户、归因和中文交付完成；第67轮为连续趋势滤波方向，尚未登记、拟合或运行。",
        next_work={"status": "CONTINUOUS_TREND_FILTER_DIRECTION_NOT_REGISTERED", "focus": "用过去价格估计趋势和噪声，连续更新进出场判断", "source": str(NEXT_NOTE.relative_to(ROOT))},
        latest_saved_atr_trend_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "ATR_TREND_BANDS_ROUND66_CHINESE_RESULTS", "rounds": [66], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "ATR_TREND_BANDS_RULES_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt, "周期利润": groups}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
