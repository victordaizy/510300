"""保存连续趋势估计的完整失败证据与中文参数表。"""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.continuous_trend_filter_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.continuous_trend_inputs_v1 import filter_likelihood, walk_forward_states
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table
from scripts.finalize_round66_20260907 import saved_cycles

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300连续趋势估计_第67轮_20260908"
DOCUMENT = OUT / "连续趋势估计_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_CONTINUOUS_TREND_CASH_DISTRIBUTION_20260908.md"


def verify(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "连续趋势冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    logs = np.log(data.wealth.to_numpy())
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    original = pd.read_parquet(RESEARCH / "daily_model_states.parquet")
    fits = json.loads((RESEARCH / "quarterly_models.json").read_text(encoding="utf-8"))["updates"]
    attempted = iter([x for x in fits if "optimizer_status" in x])
    model_checks = []

    def replay_fit(values, settings):
        saved = next(attempted)
        left, right = saved["training_start_index"], saved["training_end_index"]
        require(right == saved["origin_index"] and right - left + 1 == settings["training_observations"], "季度训练末端或窗口不符")
        np.testing.assert_array_equal(values, logs[left:right + 1])
        objective, state = filter_likelihood(values, saved["slope_noise"], saved["observation_noise"])
        error = float(np.max(np.abs(np.array(state) - saved["state"])))
        require(error < 1e-12 and abs(objective - saved["objective"]) < 1e-9, "保存参数不能还原当时滤波状态或似然")
        model_checks.append({"origin": saved["origin"], "training_start": saved["training_start"], "training_end": saved["training_end"],
            "observations": len(values), "state_error": error, "objective_error": objective - saved["objective"]})
        return {"success": saved["success"], "slope_noise": saved["slope_noise"], "observation_noise": saved["observation_noise"], "state": state}

    reproduced, _ = walk_forward_states(data, cfg["model_settings"], fitter=replay_fit)
    require(next(attempted, None) is None, "有保存模型没有被按时间核对")
    reproduced.to_parquet(RESEARCH / "saved_model_state_replay.parquet", index=False)
    pd.testing.assert_frame_equal(original, pd.read_parquet(RESEARCH / "saved_model_state_replay.parquet"), check_exact=True)
    metrics, differences, requests, cycles_all, groups, forecast, entries = [], [], [], [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {}
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                accounts[m["model"]] = ledger
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "现金、份额、分红应收与净值不符")
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0), "保存净值和净收益不符")
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost"]:
                    require(abs(actual[field] - m[field]) < 1e-10, "保存账户指标不能复算")
                metrics.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": actual["net_sharpe"]})
            current = accounts[PRIMARY]
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
                old = accounts[control]
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(old.date)), "比较账户日历不一致")
                d = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                d["reconciliation_error"] = d["terminal_nav_difference"] - d["price_difference"] - d["dividend_difference"] + d["commission_difference"] + d["slippage_difference"]
                require(abs(d["reconciliation_error"]) < 1e-6, "账户经济差额没有核对完全")
                differences.append(d)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                expected = original.target.iloc[t]
                require(row.origin == original.date.iloc[t] and row.execution_date == data.date.iloc[t + 1], "预测和执行时钟不符")
                require((pd.isna(expected) and pd.isna(row.reference_weight)) or expected == row.reference_weight, "实际目标与当时模型状态不符")
                account = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(account.shares) if account is not None else 0
                nav = float(account.equity) if account is not None else cfg["initial_capital"]
                close = float(data.close.iloc[t])
                quantity = 0
                if np.isfinite(expected):
                    target_shares = math.floor(expected * nav / close / cfg["lot"]) * cfg["lot"]
                    if expected == 1 and shares > 0 and abs(expected - shares * close / nav) < cfg["weight_band"]:
                        target_shares = shares
                    quantity = target_shares - shares
                require(row.requested_quantity == quantity, "实际份额请求与净值或无观点处理不符")
            cycles = saved_cycles(current, dividends, cfg)
            cycles_all.extend({"period": period, "cost": cost, **x} for x in cycles)
            groups.append({"period": period, "cost": cost, "cycles": len(cycles), "positive_cycles": sum(x["net_profit"] > 0 for x in cycles),
                "negative_cycles": sum(x["net_profit"] < 0 for x in cycles), "net_profit": sum(x["net_profit"] for x in cycles),
                "gross_price_profit": sum(x["gross_price_profit"] for x in cycles), "dividend_recognized": sum(x["dividend_recognized"] for x in cycles),
                "commission": sum(x["commission"] for x in cycles), "slippage": sum(x["slippage"] for x in cycles),
                "mean_held_closes": float(np.mean([x["held_closes"] for x in cycles])),
                "additional_buys": int((current.shares_before.gt(0) & current.filled_quantity.gt(0)).sum())})
            requests.append({"period": period, "cost": cost, "replayed_origins": len(decisions)})
            bought = decisions.set_index("origin").loc[current.loc[current.filled_quantity.gt(0), "origin"]]
            for row in bought.itertuples():
                t = int(row.origin_index)
                predicted_change = float(original.level.iloc[t] + original.slope.iloc[t] - logs[t])
                entries.append({"period": period, "cost": cost, "origin": row.Index, "execution_date": row.execution_date,
                    "slope": original.slope.iloc[t], "slope_z": original.slope_z.iloc[t], "expected_next_close_log_change": predicted_change,
                    "expected_next_close_change_nonpositive": predicted_change <= 0})
            if cost == "BASE":
                ix = decisions.origin_index.to_numpy(int)
                predicted = (original.level + original.slope).iloc[ix].to_numpy()
                truth, naive = logs[ix + 1], logs[ix]
                known = np.isfinite(predicted)
                model_mse, naive_mse = float(np.mean((truth[known] - predicted[known]) ** 2)), float(np.mean((truth[known] - naive[known]) ** 2))
                forecast.append({"period": period, "known_origins": int(known.sum()), "model_next_close_log_mse": model_mse,
                    "unchanged_price_log_mse": naive_mse, "relative_mse_change": model_mse / naive_mse - 1,
                    "diagnostic_only_final_day_close_not_used_in_account": True})
    for name, rows in [("saved_model_parameter_replay.csv", model_checks), ("saved_metrics_recomputation.csv", metrics),
        ("saved_account_differences.csv", differences), ("saved_request_replay.csv", requests), ("saved_actual_cycles.csv", cycles_all),
        ("saved_cycle_profit_groups.csv", groups), ("saved_next_close_forecast_diagnostic.csv", forecast), ("saved_buy_origin_forecast_mismatch.csv", entries)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "SAVED_MODELS_STATES_ACTUAL_ACCOUNTS_AND_FORECAST_DIAGNOSTIC_RECONCILED",
        "replayed_saved_models": len(model_checks), "replayed_daily_model_rows": len(original), "maximum_model_state_error": max(x["state_error"] for x in model_checks),
        "recomputed_account_records": len(metrics), "account_differences": len(differences), "replayed_decision_origins": sum(x["replayed_origins"] for x in requests),
        "recomputed_closed_cycles": len(cycles_all), "buy_origin_records": len(entries), "forecast_periods": len(forecast),
        "new_model_fits": 0, "new_account_simulations": 0, "security_audit_performed": False}, groups, fits


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require({r["round"] for r in index["completed_rounds"]} == set(range(1, 67)), "索引不是截至66轮，不能重复更新")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, groups, fits = verify(cfg, result)
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_REJECTED_CONTINUOUS_TREND_FORECAST_AND_ACCOUNT_WEAK"
    decision = "49个季度模型均收敛但两段账户弱，较早未扣费用的价格加分红已亏损，下一收盘预测误差也比价格持平更大。结束该模型，不改边界、训练窗或滤掉预测方向不一致的买入挽救。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300连续趋势估计：第67轮结果", "", "## 先看结论", "",
        "**模型能够完成估计，但没有带来可用的交易效果，结束本方案。** 主评价基础净夏普0.013、压力费用负0.144；较早基础负0.546、压力负0.675，目标1.2未实现。", "",
        "本轮每季度用过去504个交易日估计趋势变化和观测噪声，日常更新趋势；正向趋势超过自身一倍不确定性时进入，趋势非正时退出，中间保持上一目标。全部49次实际估计都收敛，失败不是因为模型没有算出来。", "",
        "|方案|主评价基础净夏普|主评价压力净夏普|较早基础净夏普|较早压力净夏普|", "|---|---:|---:|---:|---:|"]
    for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "PANIC_ONLY", "BUY_HOLD"]:
        values = [metric(result, model, period, cost)["net_sharpe"] for period, cost in [("evaluation", "BASE"), ("evaluation", "STRESS"), ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")]]
        lines.append(f"|{metric(result, model)['name']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "主评价2020年1月2日至2026年8月14日开盘1604日；较早2015年1月5日至2019年12月31日开盘1219日。各20万元，保留全部空仓日、交易费用和分红。原急跌的较高主评价仅3个周期，较早8个周期，不代表已稳定达标。", "",
        "## 完整历史表现", "", "### 主评价", "", *table(result["all_metrics"]), "### 较早历史", "", *table(result["earlier_diagnostics"]),
        "主评价基础夏普接近零，同时复合年化略为负数：夏普使用每日算术平均及波动，复合年化使用净值增长，两者口径不同。完整主评价基础累计亏2.37%，不能因夏普略为正就称策略赚钱。", "",
        "## 失败归因", "",
        "第一，模型更频繁地改变交易目标。主评价74个完整周期148笔成交，较早56个周期112笔成交；实际持仓收盘分别357和270个，平均每周期约4.82个收盘。两档费用下均无未成交、无资料缺失、无估计失败回退；这些缺失处理只在合成测试中出现，不能用来解释真实结果。", "",
        "|历史及费用|周期数|盈利周期|亏损周期|价格损益|分红|佣金|滑点|最终净利润|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        label = ("主评价" if g["period"] == "evaluation" else "较早") + ("／基础" if g["cost"] == "BASE" else "／压力")
        lines.append(f"|{label}|{g['cycles']}|{g['positive_cycles']}|{g['negative_cycles']}|{g['gross_price_profit']:,.2f}元|{g['dividend_recognized']:,.2f}元|{g['commission']:,.2f}元|{g['slippage']:,.2f}元|{g['net_profit']:,.2f}元|")
    lines += ["", "主评价基础账户的价格损益加分红为19,223.50元，实际费用23,965.14元，最终亏4,741.64元。较早的价格损益加分红在未扣这些费用前已经亏50,913.70元，扣费后亏65,405.22元。因而高换手和信号方向都存在问题，不能只把结果归因于费用。这里是同一保存账户的经济分解，没有另外模拟一个免手续费策略。", "",
        "第二，趋势斜率与从当前价格出发的预期涨幅不完全相同。下一收盘对数价格预测为当前估计水平加当前斜率；再减当前实际对数价格，才是该模型隐含的下一收盘预期变化。主评价74次买入中8次、较早56次中12次，这个预期变化非正，虽然斜率条件允许持有。两费用的日期相同。不能在看到亏损后直接删掉这些交易作为新结果。", "",
        "第三，直接核对预测也没有支持局部修补。主评价下一收盘对数价格均方误差为0.000190684，简单预测价格维持当前值为0.000146173，模型误差高30.45%；较早分别0.000379581和0.000278161，高36.46%。两段都更差，因此本轮不改成另一种预测收益门槛来挽救。此为保存模型的预测诊断，包括最后一天收盘标签；完整账户仍在最后一天开盘退出，没有用该收盘改变成交或净值。", "",
        "## 全部因子与完整中文进出场规则", "", (ROOT / cfg["rules"]).read_text(encoding="utf-8").split("\n", 1)[1],
        "", "### 每天更新量的中文展开", "",
        "以下是冻结计算的同义展开，不改变算法。预测水平为旧水平加旧斜率；预测水平方差为旧水平方差加两倍旧协方差，再加旧斜率方差。预测协方差为旧协方差加旧斜率方差，预测斜率方差为旧斜率方差加趋势噪声方差。", "",
        "创新为实际对数价格减预测水平；创新方差为预测水平方差加观测噪声方差。水平修正比例为预测水平方差除以创新方差，斜率修正比例为预测协方差除以创新方差。新水平为预测水平加水平修正比例乘创新，新斜率为旧斜率加斜率修正比例乘创新。", "",
        "更新水平方差为预测水平方差乘观测噪声方差，再除以创新方差；更新协方差同样用预测协方差乘观测噪声方差，再除以创新方差。更新斜率方差等价于预测斜率方差减去预测协方差平方除以创新方差；实现使用代数等价的噪声传播写法，减少大数相减的数值误差。独立矩阵和联合高斯分布测试已核对。", "",
        "## 全部季度参数与初始化月份", "",
        "共58次季度检查，最初9次历史不足；2014年7月起49次估计全部收敛，没有触及参数边界。共1920次优化目标求值属于内部估计过程，本轮仍只有一个策略设置。下表为全部检查；方差以对数价格模型单位记录，末端斜率是每日对数斜率。", "",
        "|季度检查日|训练起点|样本数|状态|趋势噪声方差|观测噪声方差|末端趋势|末端趋势不确定性|", "|---|---|---:|---|---:|---:|---:|---:|"]
    for f in fits:
        date, start = str(f["origin"])[:10], str(f["training_start"])[:10]
        if f["status"] != "FIT_CONVERGED":
            lines.append(f"|{date}|{start}|{f['training_end_index'] - f['training_start_index'] + 1}|历史不足，无模型|—|—|—|—|")
        else:
            lines.append(f"|{date}|{start}|{f['training_observations']}|收敛|{f['slope_noise']:.12g}|{f['observation_noise']:.12g}|{f['state'][1]:.12g}|{math.sqrt(f['state'][4]):.12g}|")
    lines += ["", "每日全部水平、斜率、协方差、创新、参数年龄和目标另存《全部逐日模型因子和状态.csv》，全部原精度参数另附季度模型文件。主评价初始目标为零，较早为一；较早首日采用此前有效模型已有的目标，第一天收盘的新季度估计只影响下一开盘。", "",
        "## 各年基础费用结果", "", "年度用于展示差异，不据此事后筛选交易年份。", "",
        "|时期|年份|净夏普|实际期间收益|复合年化|成交笔数|", "|---|---:|---:|---:|---:|---:|"]
    yearly = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for r in yearly[yearly.model.eq(PRIMARY) & yearly.cost.eq("BASE")].itertuples():
        lines.append(f"|{'主评价' if r.period == 'evaluation' else '较早'}|{r.year}|{r.net_sharpe:.3f}|{r.cumulative_return:.2%}|{r.annualized_return:.2%}|{r.trade_count}|")
    lines += ["", "## 核对与下一步", "",
        f"13项必要测试通过后冻结，正式运行无中断；49次新模型估计、4个新账户、16个保存对照。保存参数已在不重新优化的情况下还原49个末端状态和3456行逐日状态，最大状态差{receipt['maximum_model_state_error']:.3g}；20指标12差额、{receipt['replayed_decision_origins']}个原点请求和{receipt['recomputed_closed_cycles']}个含分红实际周期均已核对。", "",
        "下一步暂不继续增加价格平滑模型，先检查现有现金分红与资金利率能否形成一个定义清楚的简单条件。拟以510300已经发生的过去12个月每份现金分红除以当前价格，与已知的银行间七天资金利率比较，作为分红与资金成本参照；它不是前瞻EPS、公司股息率或已经证明的无风险利差。只用已保存资料，若需要漫长补齐就放弃。第68轮尚未登记、构建新因子或回测。", "",
        "本次有限来源检查还确认：旧第7轮全期限逆回购保存的是公开披露量，实际每日净投放仍缺到期和实际日期证据。不会将这些披露量直接改名为净流动性，也不恢复慢来源重建。已有资金利率字段采用下一交易日开盘后才可用的保守时钟，后续需单独核对来源、覆盖和时效；旧估值研究也曾有其他分红收益率口径，不能声称分红因子从未用过。", "",
        "目标仍未完成，持续研究保持进行。EPS和慢来源补齐继续暂停，无GPT数值包及真实交易。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "account_coverage.csv", "model_summary.json", "quarterly_models.json",
        "saved_model_parameter_replay.csv", "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_request_replay.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv",
        "saved_next_close_forecast_diagnostic.csv", "saved_buy_origin_forecast_mismatch.csv", "tests_receipt.json", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    pd.read_parquet(RESEARCH / "daily_model_states.parquet").to_csv(OUT / "全部逐日模型因子和状态.csv", index=False, encoding="utf-8-sig")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_trades.csv", OUT / f"{period}_{cost}_trades.csv")
    NEXT_NOTE.write_text("""# 第67轮后：现有现金分红与资金利率定义检查

第67轮CONTINUOUS_TREND_FILTER失败，主BASE/STRESS夏普0.012866/负0.143978、基础年化负0.3613%、回撤24.1136%；早负0.545759/负0.675094、基础年化负7.5613%、回撤41.6079%。主74周期148成交357持仓收盘、早56周期112成交270持仓。58季度检查、49估计全部收敛、9历史不足、0估计失败/缺失/边界，1920内部目标求值；不要把目标求值算成1920策略。两评价原点内无NO_VIEW、无未成交或旧模型回退，主初始目标0、早1。真实49拟合不可误记为0；保存复算不重新优化，新增验证拟合0。

主基础价格16228.3加分红2995.2=19223.5，佣金5919.83584、滑点18045.3，净亏4741.63584；早价格负54535.7加分红3622=负50913.7，佣金3657.72346、滑点10833.8，净亏65405.22346。因此不是仅仅费用问题。主74实际买入中8、早56中12次的“末端水平加斜率减实际对数价格”非正，但不能事后筛掉救活。模型下一收盘价格MSE主0.0001906839934对持平0.0001461729734，差30.4509%；早0.000379581496对0.000278160907差36.4611%。这已支持结束模型，不改预测数量映射、置信阈值、504日、季度频率或噪声边界。预测诊断包含最终收盘标签，账户终点仍开盘，不混淆。

代码research/continuous_trend_inputs_v1.py及continuous_trend_filter_v1.py，config/510300_continuous_trend_filter_v1.json、docs/510300_CONTINUOUS_TREND_FILTER_V1.md，结果reports/research/510300_continuous_trend_filter_v1。13测试6.52秒通过、run无中断，49保存模型及3456全状态严格复算，20指标12差额5646原点请求260完整含分红周期。完整中文报告含58次季度检查和全部49参数表，deliverables/510300连续趋势估计_第67轮_20260908。finalize_round67_20260908.py完成后不要重跑。

索引应截至67：341不同设置、354已评价来源版本、359登记含5旧未运行、1092主评价记录。此轮1设置1来源版本、4新账户16保存对照、49真实拟合、0新参考周期，目标未完成。

下一项68仅方向，未登记、未构建新因子、未计算收益。先做少量现有资料定义和覆盖检查，最多一个简单条件：过去12个月已发生的510300每份现金分红总额除以当前ETF价格，与当时可用的DR007年化利率比较。正差作为进入条件，非正退出，缺失NO_VIEW保留份额，其他账户规则先明确后冻结。该差只是ETF历史现金分配与银行间资金成本参照，不是公司前瞻EPS、成分股股息率、预期总回报或无风险利差；不计为实际融资成本，不改变现金收益0，不融资。

已有分红data/reference/510300_dividends.csv及reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json；先核完整历史覆盖、每份单位和除息日口径，过去12个月按日历一年精确定义，只使用当时已经发生的分配，不看未来公告。若源不够，放弃而不开展慢补齐。

资金源在 data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet。旧R7 reports/research/510300_total_reverse_repo_v2/features.parquet 已有dr_date、dr_available_at、dr007、dr007_known和dr_age_days；research/total_reverse_repo_v2.py的features将每个DR日值推迟到下一交易日09:30才可用、除100成小数、按自然日年龄；R7旧共同特征门还包含无关的逆回购和2015-03-01起点，不应直接将整行feature_valid作为本独立资金源可用性，也不能绕过原始DR来源约束。先从原始DR单独核对时钟、≤7自然日时效、日期完整与旧对应，保存独立因子契约，不重新下载来源。底层供应商最初历史送达时间未证明，仍属于已观测历史保守时钟回放，不能称实时独立证据。

本次限定查重：旧valuation_v2_expected_return.py的trailing_dividend_yield_annual用总回报与价格增长比值差，并非本次直接ETF过去现金分配总额除价；其他旧估值模型用分红因子，不能称未研究过分红。后续再核是否已有同一独立DR利差进出场。R7全期限逆回购的quantity_measure明确DISCLOSED_AMOUNT_NOT_ACTUAL_DAILY_NET_CASH，source_reconstruction_receipt的实际日净投放尚未重建，缺实际日期及到期证据。不能把披露量改名为净投放，也不恢复该慢源重建。用户指出的口径问题继续尊重。

所有因子、入场出场、缺失、等于阈值、日历窗、分红生效和DR发布时间先写中文，必要测试后冻结一次；不因结果改窗口、利差方向或平滑天数挽救。只510300与现金，原两历史窗口、20万元、242年化、两费用、分红、整手、T+1、方向涨跌停和完整现金日。EPS慢源暂停，无GPT数值包、不创建子任务、无真实交易。继续向目标研究。
""", encoding="utf-8")
    record = {"round": 67, "study": result["study_id"], "title": "季度估计噪声和每日连续趋势", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND67_COMPLETE_CONTINUOUS_TREND_FILTER_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计67轮，341不同设置，354已评价来源版本，359登记含5旧未运行，1092主评价记录。",
        checks="第67轮13测试、49保存模型及3456状态复算，20指标12差额5646请求260含分红周期，2段价格预测与买入方向核对。",
        process_state_note="第67轮拟合、账户、归因和中文交付完成；第68轮仅现金分红与DR007比较方向，未登记或运行。",
        next_work={"status": "CASH_DISTRIBUTION_FUNDING_SOURCE_CHECK_DIRECTION", "focus": "仅检查既有ETF历史现金分配与当时资金利率，明确定义后再决定是否登记", "source": str(NEXT_NOTE.relative_to(ROOT))},
        latest_saved_continuous_trend_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "CONTINUOUS_TREND_FILTER_ROUND67_CHINESE_RESULTS", "rounds": [67], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "CONTINUOUS_TREND_RULES_PARAMETERS_AND_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery)
    print(json.dumps({"交付": delivery, "核对": receipt, "周期": groups}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
