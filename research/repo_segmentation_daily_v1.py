"""同上午回购定盘差值的一项信息增量：两年日更分布与完整510300账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.daily_liquidity_insurance_tail_v1 as distribution
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_repo_segmentation_daily_v1"
SOURCE = ROOT / "reports/research/510300_repo_fixing_segmentation_source_v1"
PRICE = ROOT / "reports/research/510300_original_frozen_sse_completion_20260925/candidate_features.parquet"
DIVIDENDS = ROOT / "data/reference/510300_dividends.csv"
START, END = "2020-01-02", "2026-09-24"
BASE = ["pressure5", "trend20", "log_rv5_rv60"]
PRIMARY = "PRICE_AND_REPO_SEGMENTATION"
MODELS = {"HISTORY": [], "PRICE": BASE, PRIMARY: [*BASE, "fixing_segmentation_pp"]}


def implementation_checks():
    # 未成熟的退出开盘和两年以前样本均不得影响当前分布。
    count = 560
    dates = pd.bdate_range("2021-01-04", periods=count)
    data = pd.DataFrame({"date": dates, "common_known": True})
    for j, name in enumerate(MODELS[PRIMARY]):
        data[name] = np.sin(np.arange(count) / (7+j))
    labels = pd.DataFrame({"idx": range(count-5), "date": dates[:-5],
                           "exit_idx": np.arange(count-5)+5, "gross_return5": np.sin(np.arange(count-5)) / 100})
    function = predictor()
    before, receipt, models = function(count-1, data, labels)
    cutoff = dates[-1] - pd.DateOffset(years=2)
    unknown = labels.exit_idx.ge(count-1) | labels.date.lt(cutoff)
    altered = labels.copy()
    altered.loc[unknown, "gross_return5"] = -9999.
    after, _, _ = function(count-1, data, altered)
    assert receipt["status"] == "UPDATED"
    for a, b in zip(before, after):
        for key in ["mu5", "variance5", "q05", "es95"]:
            assert a[key] == b[key]
    return {"unmatured_and_older_than_two_years_ignored": True, "checked_model_sets": len(models)}


def predictor():
    old = distribution.one_day_distribution
    return FunctionType(old.__code__, {**old.__globals__, "MODELS": MODELS}, old.__name__, old.__defaults__, old.__closure__)


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮已固定，不能覆盖原研究。")
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    source = read(SOURCE / "result.json")
    assert source["status"] == "SOURCE_READY_HISTORICAL_DELIVERY_UNVERIFIED"
    assert source["last_date"][:10] == END
    checks = implementation_checks()
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_REPO_SEGMENTATION_DAILY_V1",
        "question": "同一上午全市场与银行子样本的七天定盘差，能否改善价格弱势后修复与持续恶化的区分？",
        "primary": PRIMARY, "control": "PRICE", "mean_reference": "HISTORY",
        "assets": ["510300.SH", "CASH_CNY"], "capital": 200000,
        "period": [START, END], "annual_days": 242, "targets": {"sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "single_new_information": "FR007-FDR007，单位百分点；不变换方向、不取最优期限或最近变化，不作为R007、DR007或中性利率的替代。",
        "interpretation": "该差同时包含参与者及抵押品范围的差异；不能直接识别纯非银资金成本、国家队资金或未来股票方向。",
        "known_at": "价格至前一A股收盘；官方上午定盘按当日12:00计划可用，合并到执行日09:00；仅最近7自然日的最新完整同日差可用。历史首版和首次实际送达未认证。",
        "price_features": {"pressure5": "负的过去五日含分红对数收益除以rv20乘sqrt(5)",
                           "trend20": "过去二十日含分红对数收益除以rv20乘sqrt(20)",
                           "log_rv5_rv60": "五日/六十日均方根波动之比取对数"},
        "training": "每个交易日只用向前两个日历年内的原点；五日退出开盘严格早于当前执行日。最少252个共同成熟原点，价格和新增信息使用完全相同训练池。",
        "model": "复用已有固定126近邻经验分布：训练均值和样本标准差标准化、截断[-5,5]、欧氏距离，并列以较早原点优先。HISTORY为同池均值和分布；不拟合或扫描k、缩放、窗口。",
        "label": "执行日开盘买入至第五个后续交易日开盘的含已获股息权益收益，不含开盘前涨幅；重叠标签不是独立机会。",
        "account": "两个候选各BASE/STRESS成本，连续四账户；允许五日价格弱势时新增库存。每日枚举100份目标，最大化w*mu5-2*w^2*var5-调整成本及退出储备；决策均使用压力成本。",
        "risk": {"maximum_position_fraction": .5, "five_day_ES95_fraction": .025,
                 "gap_stress": -.1, "gap_budget": .05, "drawdown_headroom_share": .5, "drawdown_stop": .1,
                 "fees": "先扣估计调整费用再算风险预算；尾部和跳空损失另加未来退出费用。开盘只能缩小买入数量。"},
        "execution": "固定09:00判断、09:30开盘成交，100份、tick0.001、T+1、分红权益分账、现金和涨跌停限制；旧账户引擎复用。缺信息最长保留5日，不新增；10%回撤后本轮不恢复。",
        "terminal": "最后日期照常决策、收盘计价，有仓位计压力退出储备，不利用未来截止日提前平仓。",
        "costs": distribution.COSTS, "new_final_accounts": 4,
        "comparison": "主比较新增信息减价格：20交易日循环区块、2000次、seed2026092507；披露全部空仓日、固定2020-2023和2024-末端，以及全部滚动两年。",
        "prediction_comparison": "共同成熟原点的五日均值MSE、5%分位损失；全体及当时五日价格弱势分别报告；不因某个子组更好更换主方案。",
        "monitor": "固定每5个原点的一相位；只汇入截至当时已退出的标签；至少60个非重叠五日标签，最近60个分位突破率超过15%只记录报警，不据结果改启停。",
        "development_gate": "主压力账户满足1.2/10%/10%，全账户收益增量95%下界>0且两个固定时期增量均正，才保留开发候选；历史选择和独立验证缺失仍须披露。",
        "duplicate_boundary": "旧FDR绝对水平/政策利差、DR007减政策利率及操作量研究保留。此次只有同一时钟FR007-FDR007的新字段差值；既有价格分布方法不优化。",
        "implementation_checks": checks, "historical_prices_previously_researched": True,
        "new_parameter_grid": 0, "orders_authorized": False, "goal_achieved": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    paths = [Path(__file__), Path(distribution.__file__), Path(inventory.__file__), Path(engine.__file__),
             ROOT / "research/strategy_review_diagnostics_v1.py", ROOT / "research/selected_mix_daily_two_year_v1.py",
             PRICE, DIVIDENDS, SOURCE / "fixing_segmentation.parquet", SOURCE / "result.json", SOURCE / "protocol.json"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
         "before_new_candidate_returns_or_accounts": True}, True)
    print("定盘差值的单项增量与四个完整账户已固定，未开始新绩效计算。", flush=True)


def design():
    raw = pd.read_parquet(PRICE)
    dividends = engine.normalize_dividends(pd.read_csv(DIVIDENDS))
    d = engine.build_features(engine.normalize_prices(raw), dividends)
    d = d[d.date.le(END)].reset_index(drop=True)
    for window in [5, 20, 60]:
        d[f"rv{window}_day"] = np.sqrt(d.total_log.pow(2).rolling(window).mean())
    x = d[["date"]].copy()
    x["idx"] = np.arange(len(x))
    x["pressure5"] = (-d.total_log.rolling(5).sum() / (d.rv20_day * np.sqrt(5))).shift(1)
    x["trend20"] = (d.total_log.rolling(20).sum() / (d.rv20_day * np.sqrt(20))).shift(1)
    x["log_rv5_rv60"] = np.log(d.rv5_day / d.rv60_day).shift(1)
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    x["decision_time"] = x.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)
    source = pd.read_parquet(SOURCE / "fixing_segmentation.parquet")
    source = source.rename(columns={"date": "fixing_date", "segmentation_pp": "fixing_segmentation_pp"})
    source["available_at"] = pd.to_datetime(source.available_at).dt.as_unit("ns")
    source["fixing_date"] = pd.to_datetime(source.fixing_date).astype("datetime64[ns]")
    x = pd.merge_asof(x.sort_values("decision_time"), source.sort_values("available_at"),
                      left_on="decision_time", right_on="available_at", direction="backward")
    age = (x.date - x.fixing_date).dt.days
    x["common_known"] = x[MODELS[PRIMARY]].notna().all(axis=1) & age.between(0, 7)
    known = x.common_known
    assert x.loc[known, "available_at"].le(x.loc[known, "decision_time"]).all()
    assert x.loc[known, "fixing_date"].lt(x.loc[known, "date"]).all()
    labels = distribution.labels(d, dividends)
    return d, dividends, x, labels


def adjusted_limits(e, target, ref, context, cost):
    change = target - int(context["shares_before_decision"])
    price = e.fill_price(ref, 1 if change > 0 else -1, cost, .001)
    adjustment = abs(change) * abs(price-ref) + e.commission(change, price, cost)
    limits = distribution.risk_limits(context["decision_equity"] - adjustment, context["peak_equity"], context["es95"])
    return limits, adjustment


def risk_valid(e, quantity, ref, context, cost):
    if quantity == 0:
        return True
    limits, _ = adjusted_limits(e, quantity, ref, context, cost)
    return distribution.risk_valid(e, quantity, ref, limits, cost)


def plan(e, account, ref, peak, prediction, pressure):
    nav = account.value(ref)
    cost = distribution.COSTS["STRESS"]
    context = {"decision_equity": nav, "peak_equity": peak, "shares_before_decision": account.shares,
               "es95": max(float(prediction.es95), 1e-8)}
    maximum = int(.5 * nav / ref / 100) * 100
    if pressure <= 0:
        maximum = min(maximum, account.shares)
    maximum = min(maximum, account.shares + e.affordable_quantity(account.cash, e.fill_price(ref, 1, cost, .001), cost, 100))
    best = None
    for target in range(0, maximum+1, 100):
        if not risk_valid(e, target, ref, context, cost):
            continue
        limits, adjustment = adjusted_limits(e, target, ref, context, cost)
        exit_price = e.fill_price(ref, -1, cost, .001)
        exit_cost = target * (ref-exit_price) + e.commission(target, exit_price, cost)
        weight = target * ref / nav
        score = weight * prediction.mu5 - 2*weight*weight*prediction.variance5 - (adjustment+exit_cost)/nav
        candidate = {**context, **limits, "target_shares": target, "requested_quantity": target-account.shares,
                     "planned_weight": weight, "score": score, "estimated_adjustment_cost": adjustment,
                     "reserved_exit_cost": exit_cost, "reference_price": ref,
                     "mu5": prediction.mu5, "variance5": prediction.variance5}
        if best is None or score > best["score"]+1e-12:
            best = candidate
    assert best is not None
    return best


def learn(x, labels):
    function = predictor()
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, receipt, m = function(int(i), x, labels)
        predictions.extend(p)
        receipts.append(receipt)
        models.extend(m)
        if len(receipts) % 300 == 0:
            print(f"两年滚动共同分布已计算至{x.date.iloc[i].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    return pred, models


def forecast_evaluation(pred, labels, x):
    scored = pred.merge(labels[["idx", "exit_idx", "exit_date", "gross_return5"]], on="idx", how="left", validate="many_to_one")
    scored["squared_error"] = (scored.mu5-scored.gross_return5).pow(2)
    residual = scored.gross_return5-scored.q05
    scored["quantile_loss"] = residual*(.05-residual.lt(0).astype(float))
    scored["pressure5"] = x.pressure5.iloc[scored.idx.to_numpy(int)].to_numpy()
    scored.to_parquet(OUT / "results/scored_predictions.parquet", index=False)
    result = []
    for group_name, mask in [("ALL", scored.gross_return5.notna()),
                              ("PRIOR_FIVE_DAY_DECLINE", scored.gross_return5.notna() & scored.pressure5.gt(0))]:
        subset = scored[mask]
        aggregate = subset.groupby("model", sort=False).agg(n=("idx", "size"), MSE=("squared_error", "mean"),
                                                          quantile_loss=("quantile_loss", "mean"))
        result.append({"subset": group_name, "models": aggregate.reset_index().to_dict("records"),
                       "mse_improvement_fraction": 1-aggregate.loc[PRIMARY, "MSE"]/aggregate.loc["PRICE", "MSE"]})
    # 监控只使用退出日严格早于当前执行日的五日固定相位标签。
    anchor = int(pred.idx.min())
    monitor = []
    for model, values in scored.groupby("model", sort=False):
        phase = values[(values.idx-anchor).mod(5).eq(0) & values.gross_return5.notna()]
        for row in values.itertuples():
            prior = phase[phase.exit_idx.lt(row.idx)].tail(60)
            rate = float((prior.gross_return5 < prior.q05).mean()) if len(prior) else None
            monitor.append({"idx": row.idx, "date": row.date, "model": model, "mature_nonoverlap_n": len(prior),
                            "last_mature_exit_idx": int(prior.exit_idx.max()) if len(prior) else None,
                            "tail_breach_rate": rate, "alert": bool(len(prior) >= 60 and rate > .15),
                            "used_to_change_accounts": False})
    pd.DataFrame(monitor).to_parquet(OUT / "results/mature_only_monitor.parquet", index=False)
    return result, scored, monitor


def account_verification(ledger, decisions):
    np.testing.assert_allclose(ledger.equity, ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable-ledger.terminal_exit_reserve, atol=1e-6)
    assert ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all()
    assert (-ledger.filled_quantity.clip(upper=0)).le(ledger.sellable_before).all()
    assert ledger.accounting_error.abs().max() < 1e-6
    assert decisions.loc[decisions.filled_quantity.gt(0), "pressure5"].gt(0).all()
    for row in decisions[decisions.filled_quantity.gt(0)].to_dict("records"):
        quantity = int(ledger.loc[ledger.idx.eq(row["idx"]), "shares"].iloc[0])
        assert risk_valid(engine, quantity, row["actual_open"], row, distribution.COSTS["STRESS"])
    return {"cash_dividend_T1": True, "buy_only_after_known_price_weakness": True, "risk_and_adjustment_cost_budget": True}


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    d, dividends, x, labels = design()
    d.to_parquet(OUT / "inputs/market.parquet", index=False)
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    labels.to_parquet(OUT / "inputs/mature_labels.parquet", index=False)
    pred, models = learn(x, labels)
    forecast, scored, monitor = forecast_evaluation(pred, labels, x)
    # 只比较与旧价格基准共享的分布方法；不修改导入模块的全局变量。
    proxy = SimpleNamespace(START=START, END=END, COSTS=distribution.COSTS, plan=plan, risk_valid=risk_valid)
    accounts, checks, pending_cycles = {}, [], []
    for model in ["PRICE", PRIMARY]:
        chosen = pred[pred.model.eq(model)].copy()
        chosen["model"] = "PRICE"
        for cost in distribution.COSTS:
            ledger, decisions = inventory.simulate(proxy, engine, d, x, dividends, chosen, "PRICE_DOWN_ONLY", cost)
            ledger["policy"], decisions["policy"] = model, model
            folder = OUT / "accounts" / cost / model
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            closed, pending = cycles(ledger)
            pd.DataFrame(closed).to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "pending_cycle.json", pending, True)
            accounts[model, cost] = ledger
            check = account_verification(ledger, decisions)
            checks.append({"model": model, "cost": cost, **check})
            pending_cycles.append({"model": model, "cost": cost, "closed": len(closed), "open": pending is not None})
            m = metrics(ledger)
            print(f"{model}/{cost}：压力预算连续账户夏普{m['sharpe']:.6f}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    account_metrics, rolling = summarize_accounts(OUT, d, accounts)
    rng = np.random.default_rng(2026092507)
    comparisons = []
    for cost in distribution.COSTS:
        left, right = accounts[PRIMARY, cost], accounts["PRICE", cost]
        periods = []
        for start, end in [(START, "2023-12-31"), ("2024-01-01", END)]:
            mask = left.date.between(start, end)
            periods.append({"start": start, "end": end, "annual_arithmetic_difference":
                            float((left.loc[mask, "net_return"]-right.loc[mask, "net_return"]).mean()*242)})
        comparisons.append({"cost": cost, **paired_interval(left, right, rng), "fixed_periods": periods})
    # 保存分布重算和未来样本截断，不重复四个完整账户。
    lookup = pred.set_index(["idx", "model"])
    sample = labels.set_index("idx")
    for record in models:
        i = record["idx"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i]-pd.DateOffset(years=2)).all()
        assert pool.exit_idx.lt(i).all()
        stats = distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(stats[key], lookup.loc[(i, record["model"]), key], rtol=0, atol=1e-14)
    function = predictor()
    for day in ["2022-12-30", "2024-12-31"]:
        i = int(x.date.searchsorted(day, side="right"))-1
        restricted = labels[labels.date.le(x.date.iloc[i])].copy()
        restricted.loc[restricted.exit_idx.ge(i), "gross_return5"] = -9999.
        p, _, _ = function(i, x.iloc[:i+1], restricted)
        for row in p:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == lookup.loc[(i, row["model"]), key]
    primary = next(row for row in account_metrics if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    stress_pair = next(row for row in comparisons if row["cost"] == "STRESS")
    increment_gate = stress_pair["lower_95"] > 0 and all(row["annual_arithmetic_difference"] > 0 for row in stress_pair["fixed_periods"])
    result = {"at": now(), "study_id": "510300_REPO_SEGMENTATION_DAILY_V1",
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment_gate else "FROZEN_NO_QUALIFIED_REPO_SEGMENTATION_STRATEGY",
              "primary": primary, "all_accounts": account_metrics, "comparisons": comparisons,
              "forecast_evaluation": forecast, "daily_distribution_updates": len(models),
              "unique_prediction_days": int(pred.idx.nunique()), "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "new_full_accounts": 4, "new_fitted_regression_coefficients": 0, "new_parameter_searches": 0,
              "completed_and_pending_cycles": pending_cycles, "account_checks": checks,
              "saved_distributions_recomputed": len(models), "future_exclusion_checks": 2,
              "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in rolling if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
              "observation_monitor_alert_counts": pd.DataFrame(monitor).groupby("model").alert.sum().to_dict(),
              "increment_gate_met": increment_gate, "historical_first_vintage_verified": False,
              "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("同上午定盘差值的两年日更预测、四个连续账户与成熟后监控已全部完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回购定盘分化信息的固定日更研究")
    parser.add_argument("command", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    if args.command == "check":
        print(implementation_checks())
    else:
        {"freeze": freeze, "run": run}[args.command]()
