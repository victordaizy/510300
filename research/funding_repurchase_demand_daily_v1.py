"""已确认回购披露、资金意外与价格状态的固定增量及完整账户试验。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.corporate_repurchase_disclosed_demand_v1 as demand
import research.funding_calendar_innovation_daily_v1 as funding
import research.repo_segmentation_daily_v1 as parent
import research.daily_liquidity_insurance_tail_v1 as distribution
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_funding_repurchase_demand_daily_v1"
STUDY = "510300_FUNDING_REPURCHASE_DEMAND_DAILY_V1"
FEATURE = "confirmed_repurchase_intensity20"
FUNDING = "PRICE_AND_FUNDING"
DEMAND = "PRICE_AND_REPURCHASE"
PRIMARY = "PRICE_FUNDING_AND_REPURCHASE"
MODELS = {"HISTORY": [], "PRICE": parent.BASE,
          FUNDING: [*parent.BASE, "funding_innovation"],
          DEMAND: [*parent.BASE, FEATURE],
          PRIMARY: [*parent.BASE, "funding_innovation", FEATURE]}
TRADED = ["PRICE", FUNDING, DEMAND, PRIMARY]
START, END = parent.START, parent.END


def information():
    x = pd.read_parquet(funding.OUT / "inputs/decision_information.parquet")
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    r = pd.read_parquet(demand.OUT / "results/decision_information.parquet")
    r["date"] = pd.to_datetime(r.date).astype("datetime64[ns]")
    pd.testing.assert_series_equal(x.date, r.date)
    x = x.merge(r, on="date", validate="one_to_one")
    x["funding_information_known"] = x.common_known
    x["common_known"] &= x.repurchase_information_known & x[FEATURE].notna()
    known = x.common_known
    assert x.idx.tolist() == list(range(len(x)))
    assert x.loc[known, "available_at"].le(x.loc[known, "decision_time"]).all()
    assert x.loc[known, "source_date"].lt(x.loc[known, "date"]).all()
    assert np.isfinite(x.loc[known, [*parent.BASE, FEATURE]].to_numpy(float)).all()
    return x


def innovation_series(i, x, nested, source_count):
    assert nested.decision_idx.eq(i).all()
    assert nested.source_idx.le(int(x.source_idx.iloc[i])).all()
    residual = np.full(source_count, np.nan)
    residual[nested.source_idx.to_numpy(int)] = (nested.actual - nested.CALENDAR).to_numpy(float)
    mapped = x.source_idx.fillna(-1).to_numpy(int)
    innovation = np.full(len(x), np.nan)
    valid = mapped >= 0
    innovation[valid] = residual[mapped[valid]]
    return innovation


def fit_day(i, x, nested, source_count, labels):
    day = x.date.iloc[i]
    lower = day - pd.DateOffset(years=2)
    receipt = {"idx": i, "date": day, "lower_bound": lower,
               "status": "NO_VIEW_SOURCE", "n_train": 0, "latest_exit_idx": None}
    if not bool(x.common_known.iloc[i]) or nested is None:
        return [], [], receipt
    innovations = innovation_series(i, x, nested, source_count)
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)].copy()
    ids = pool.idx.to_numpy(int)
    pool = pool.loc[x.common_known.iloc[ids].to_numpy(bool) & np.isfinite(innovations[ids])]
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None)
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        receipt["status"] = "NO_VIEW_TRAINING"
        return [], [], receipt
    receipt["status"] = "UPDATED"
    ids = pool.idx.to_numpy(int)
    values = x.loc[ids, [*parent.BASE, FEATURE]].copy()
    values["funding_innovation"] = innovations[ids]
    current = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    predictions, saved = [], []
    for name, columns in MODELS.items():
        mean, scale = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            assert np.isfinite(raw).all()
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1.
            z = np.clip((raw - mean) / scale, -5, 5)
            live = np.clip((np.array([current[c] for c in columns]) - mean) / scale, -5, 5)
            selected = np.lexsort((ids, np.sum((z - live)**2, axis=1)))[:126]
        else:
            selected = np.arange(len(pool))
        chosen = pool.iloc[selected]
        predictions.append({**receipt, "model": name, "n_selected": len(chosen),
                            **distribution.empirical_statistics(chosen.gross_return5)})
        saved.append({"idx": i, "model": name, "features": columns,
                      "mean": mean.tolist(), "scale": scale.tolist(),
                      "training_indices": ids.tolist(), "selected_indices": chosen.idx.tolist(),
                      "current_values": {c: float(current[c]) for c in MODELS[PRIMARY]}})
    return predictions, saved, receipt


def implementation_check():
    dates = pd.bdate_range("2022-01-03", periods=580)
    n, i = len(dates), len(dates) - 1
    x = pd.DataFrame({"date": dates, "common_known": True, "source_idx": np.arange(n)})
    for j, col in enumerate([*parent.BASE, FEATURE]):
        x[col] = np.sin(np.arange(n) / (j + 5))
    nested = pd.DataFrame({"decision_idx": i, "source_idx": np.arange(n),
                           "actual": np.sin(np.arange(n) / 13), "CALENDAR": 0.})
    labels = pd.DataFrame({"idx": np.arange(n), "date": dates,
                           "exit_idx": np.arange(n) + 5, "gross_return5": np.cos(np.arange(n)) / 100})
    before, models, receipt = fit_day(i, x, nested, n, labels)
    assert receipt["status"] == "UPDATED"
    forbidden = labels.exit_idx.ge(i) | labels.date.lt(dates[i] - pd.DateOffset(years=2))
    changed = labels.copy()
    changed.loc[forbidden, "gross_return5"] = -9999.
    after, rebuilt, _ = fit_day(i, x, nested, n, changed)
    assert before == after and models == rebuilt
    assert all(m["training_indices"] == models[0]["training_indices"] for m in models)
    return {"future_and_expired_labels_do_not_change_prediction": True,
            "all_models_share_mature_training_pool": True, "checked_models": len(models)}


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮已固定，不能覆盖规则或原结果。")
    checks = implementation_check()
    x = information()
    assert read(demand.OUT / "result.json")["status"] == "CONFIRMED_DISCLOSED_DEMAND_SERIES_READY_FOR_FIXED_TEST"
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "实际买入已经发生且被明确披露后，这项信息能否改善相同价格状态和资金意外下的未来五日分布及510300完整账户？",
        "mechanism_hypothesis": "企业已披露的资本承接可能与其他卖压的持续性不同；披露也可能滞后于买入结束或反映价格下跌后的被动反应，方向由固定历史条件分布估计。",
        "feature": {"name": FEATURE, "source_protocol": (demand.OUT / "protocol.json").relative_to(ROOT).as_posix(),
                    "meaning": "过去20个决策日可确认的正向披露增量/股票此前20日成交额，再乘前一期保存的指数权重并求和。",
                    "unknown": "未识别金额或身份仍未知；无新增已确认披露不是实际回购为零。",
                    "clock": "公告已发布日结束后，最早下一交易日09:00使用；不回填至回购实际发生日。",
                    "fixed_transform": "直接使用已冻结的强度，不搜索窗口、方向、分位阈值或公司子集。"},
        "funding": "只读复用每个决策日两年窗内重建的FDR007日历模型残差；不使用旧窗参数、不新增资金回归。它不代表NFCI或中性利率差。",
        "models": MODELS, "traded_models": TRADED,
        "training": "最近两个日历年的共同成熟原点，五日退出开盘严格早于当前日期；至少252个原点；训练均值与样本标准差标准化后截断5，固定126近邻，同距以原点编号排序；HISTORY为同池分布。",
        "account": "四模型各BASE/STRESS，共8个新完整账户；复用固定每日库存引擎、仅五日价格弱势时允许增仓。没有复用覆盖期不同的旧价格账户。",
        "capital": 200000, "assets": ["510300.SH", "CASH_CNY"], "period": [START, END],
        "costs": distribution.COSTS, "annual_days": 242, "cash_and_risk_free_rate": 0,
        "risk": {"maximum_position": .5, "five_day_ES95_budget": .025,
                 "gap_return": -.1, "gap_budget": .05, "drawdown_reserve_fraction": .5, "drawdown_stop": .1,
                 "execution": "09:00决策，开盘只可缩量；T+1、100份、现金股息分账、费用、末端退出储备沿用原规则。"},
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparisons": "主模型相对三个对照均报告完整账户配对收益差，20日循环区块2000次seed2026092603；同一算法另列从首个共同预测日至最后预测后五个交易日的固定可用期诊断，不替代全期目标。",
        "availability": "资料缺失或训练不足均保留NO_VIEW。全期保留对应现金日，不把没有实际运行信息的2020-2023写成策略平稳期。",
        "calendar_year_diagnostics": "逐年公布共同预测日与完整账户；不足126个预测日的年份标记证据不足，不宣称跨状态稳定。",
        "development_gate": "主STRESS全期三目标均过、相对三个对照配对差95%下界均正、至少两个日历年各126个预测日且各自增量均正，才列开发候选；缺独立验证仍不表示总目标完成。",
        "monitor": "固定每5日一相位的成熟预测，最近60条q05跌穿率超过15%仅记警报；不足60条为NOT_ASSESSABLE，不把零警报当稳定。",
        "new_parameter_grid": 0, "new_funding_regression_fits": 0, "new_accounts": 8,
        "historical_prices_previously_researched": True, "historical_first_delivery_authenticated": False,
        "new_independent_forward_observations": 0, "orders_authorized": False, "goal_achieved": False,
        "source_coverage": {"common_dates": int(x.common_known.sum()),
                            "first": x.loc[x.common_known, "date"].min(), "last": x.loc[x.common_known, "date"].max()},
        "implementation_checks": checks}
    save(OUT / "protocol.json", protocol, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    paths = [Path(__file__), Path(demand.__file__), Path(funding.__file__), Path(parent.__file__),
             Path(distribution.__file__), Path(inventory.__file__), Path(engine.__file__),
             ROOT / "research/selected_mix_reappraisal_v1.py", ROOT / "research/selected_mix_daily_two_year_v1.py",
             ROOT / "research/strategy_review_diagnostics_v1.py", parent.DIVIDENDS,
             parent.OUT / "inputs/market.parquet", parent.OUT / "inputs/mature_labels.parquet",
             funding.OUT / "inputs/decision_information.parquet", funding.OUT / "inputs/funding_features.parquet",
             funding.OUT / "results/nested_funding_predictions.parquet", funding.OUT / "result.json",
             demand.OUT / "results/decision_information.parquet", demand.OUT / "results/confirmed_disclosure_events.json",
             demand.OUT / "protocol.json", demand.OUT / "result.json"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
                               "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
                               "before_new_candidate_return_calculation": True}, True)
    print(f"价格、资金、回购的共同样本和8个账户已固定，共同信息日{x.common_known.sum()}个。", flush=True)


def verify_predictions(pred, models, x, labels, groups, source_count):
    lookup, sample = pred.set_index(["idx", "model"]), labels.set_index("idx", drop=False)
    pools = {}
    for record in models:
        i = record["idx"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i] - pd.DateOffset(years=2)).all() and pool.exit_idx.lt(i).all()
        assert x.common_known.iloc[pool.idx.to_numpy(int)].all()
        assert pools.setdefault(i, record["training_indices"]) == record["training_indices"]
        stats = distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in stats:
            np.testing.assert_allclose(stats[key], lookup.loc[(i, record["model"]), key], atol=1e-14, rtol=0)
    dates = sorted(pools)
    checked = []
    for i in sorted(set([dates[0], dates[len(dates) // 2], dates[-1]])):
        changed = labels.copy()
        excluded = changed.exit_idx.ge(i) | changed.date.lt(x.date.iloc[i] - pd.DateOffset(years=2))
        changed.loc[excluded, "gross_return5"] = -9999.
        rows, _, _ = fit_day(i, x.iloc[:i + 1], groups[i], source_count, changed)
        for row in rows:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == lookup.loc[(i, row["model"]), key]
        checked.append(x.date.iloc[i])
    return {"saved_distributions_recomputed": len(models), "common_pool_dates": len(pools),
            "future_and_expired_label_perturbation_dates": checked}


def forecast_evaluation(pred, labels, x):
    scored = pred.merge(labels[["idx", "gross_return5", "exit_idx", "exit_date"]], on="idx", how="left", validate="many_to_one")
    scored = scored.merge(x[["idx", "pressure5"]], on="idx", how="left", validate="many_to_one")
    scored.to_parquet(OUT / "results/scored_predictions.parquet", index=False)
    summaries = []
    for subset, frame in [("ALL", scored), ("PRIOR_FIVE_DAY_DECLINE", scored[scored.pressure5.gt(0)])]:
        rows = []
        for name, part in frame[frame.gross_return5.notna()].groupby("model", sort=False):
            error = part.gross_return5 - part.q05
            rows.append({"model": name, "n": len(part), "MSE": float(((part.gross_return5 - part.mu5)**2).mean()),
                         "quantile_loss": float(np.maximum(.05 * error, -.95 * error).mean()),
                         "q05_breach_fraction": float(part.gross_return5.lt(part.q05).mean())})
        lookup = {r["model"]: r for r in rows}
        summaries.append({"subset": subset, "models": rows,
                          "primary_mse_improvement": {c: 1 - lookup[PRIMARY]["MSE"] / lookup[c]["MSE"] for c in MODELS if c != PRIMARY}})
    first = int(x.index[x.date.ge(START)][0])
    monitors = []
    for name, frame in scored[(scored.idx - first) % 5 == 0].groupby("model"):
        part = frame[frame.gross_return5.notna()].sort_values("exit_idx").copy()
        part["breach"] = part.gross_return5.lt(part.q05)
        part["breach_rate60"] = part.breach.rolling(60, min_periods=60).mean()
        part["monitor_assessable"] = part.breach_rate60.notna()
        part["alert"] = part.breach_rate60.gt(.15)
        monitors.append(part)
    monitor = pd.concat(monitors, ignore_index=True)
    monitor.to_parquet(OUT / "results/mature_monitor.parquet", index=False)
    return summaries, scored, monitor


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    x = information()
    market = pd.read_parquet(parent.OUT / "inputs/market.parquet")
    labels = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    source = pd.read_parquet(funding.OUT / "inputs/funding_features.parquet", columns=["source_idx"])
    nested = pd.read_parquet(funding.OUT / "results/nested_funding_predictions.parquet")
    groups = {int(i): p for i, p in nested.groupby("decision_idx", sort=False)}
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = fit_day(int(i), x, groups.get(int(i)), len(source), labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"价格、资金、回购条件分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    if pred.empty:
        raise RuntimeError("没有达到预设共同成熟样本数；不得放宽样本数补造策略。")
    verification = verify_predictions(pred, models, x, labels, groups, len(source))
    save(OUT / "results/prediction_verification.json", verification, True)
    forecast, scored, monitor = forecast_evaluation(pred, labels, x)
    proxy = SimpleNamespace(START=START, END=END, COSTS=distribution.COSTS, plan=parent.plan, risk_valid=parent.risk_valid)
    accounts, account_checks, cycle_rows = {}, [], []
    for name in TRADED:
        selected = pred[pred.model.eq(name)].copy()
        selected["model"] = "PRICE"
        for cost in distribution.COSTS:
            ledger, decisions = inventory.simulate(proxy, engine, market, x, dividends, selected, "PRICE_DOWN_ONLY", cost)
            ledger["policy"], decisions["policy"] = name, name
            pd.testing.assert_series_equal(ledger.date.reset_index(drop=True), market.loc[market.date.ge(START), "date"].reset_index(drop=True))
            account_checks.append({"model": name, "cost": cost, **parent.account_verification(ledger, decisions)})
            folder = OUT / "accounts" / cost / name
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            closed, pending = cycles(ledger)
            closed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "pending_cycle.json", pending, True)
            cycle_rows.append({"model": name, "cost": cost, "closed": len(closed), "open": pending is not None})
            accounts[name, cost] = ledger
            summary = metrics(ledger)
            sh = "未定义" if summary["sharpe"] is None else f"{summary['sharpe']:.6f}"
            print(f"{name}/{cost}：夏普{sh}、年化{summary['annual_return']:.3%}、回撤{abs(summary['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    rng = np.random.default_rng(2026092603)
    comparisons, active_metrics, calendar_counts = [], [], []
    ready = pred[["idx", "date"]].drop_duplicates().sort_values("idx")
    active_start = ready.date.min()
    active_end = market.date.iloc[min(int(ready.idx.max()) + 5, len(market) - 1)]
    year_counts = ready.groupby(ready.date.dt.year).size().to_dict()
    for year, count in year_counts.items():
        calendar_counts.append({"year": int(year), "prediction_days": int(count), "at_least_126": count >= 126})
    for cost in distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for control in TRADED[:-1]:
            right = accounts[control, cost]
            mask = left.date.between(active_start, active_end)
            years = []
            for year, count in year_counts.items():
                period = left.date.dt.year.eq(year)
                years.append({"year": int(year), "prediction_days": int(count),
                              "annual_arithmetic_difference": float((left.loc[period, "net_return"] - right.loc[period, "net_return"]).mean() * 242),
                              "evidence_status": "AT_LEAST_126_PREDICTION_DAYS" if count >= 126 else "INSUFFICIENT_YEAR_COVERAGE"})
            comparisons.append({"cost": cost, "control": control, "full_account": paired_interval(left, right, rng),
                                "fixed_available_period": paired_interval(left.loc[mask], right.loc[mask], rng), "years": years})
        for name in TRADED:
            ledger = accounts[name, cost]
            mask = ledger.date.between(active_start, active_end)
            begin = int(np.flatnonzero(mask)[0])
            capital = float(ledger.equity.iloc[begin - 1]) if begin else 200000.
            active_metrics.append({"model": name, "cost": cost, "start": active_start, "end": active_end,
                                   "only_coverage_diagnostic": True, **metrics(ledger.loc[mask], capital)})
    primary = next(r for r in measures if r["policy"] == PRIMARY and r["cost"] == "STRESS")
    increment = all(r["full_account"]["lower_95"] > 0 for r in comparisons if r["cost"] == "STRESS")
    enough_years = sum(count >= 126 for count in year_counts.values()) >= 2
    era_positive = enough_years and all(y["annual_arithmetic_difference"] > 0
        for r in comparisons if r["cost"] == "STRESS" for y in r["years"] if y["prediction_days"] >= 126)
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000., atol=1e-6, rtol=0)
    candidate = primary["historical_point_targets_met"] and increment and era_positive
    result = {"at": now(), "study_id": STUDY,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if candidate else "FROZEN_NO_QUALIFIED_DISCLOSED_REPURCHASE_STRATEGY",
              "primary": primary, "all_accounts": measures, "comparisons": comparisons,
              "available_period_diagnostics": active_metrics, "forecast_evaluation": forecast,
              "increment_gate_met": increment, "calendar_coverage": calendar_counts, "cross_year_evidence_sufficient": enough_years,
              "new_full_accounts": len(accounts), "new_conditional_distributions": len(pred), "new_funding_regression_fits": 0,
              "unique_prediction_days": len(ready), "first_prediction_date": active_start, "last_prediction_date": ready.date.max(),
              "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "schedule_statuses": schedule.status.value_counts().to_dict(),
              "training_rows_min": int(pred.n_train.min()), "training_rows_max": int(pred.n_train.max()),
              "account_checks": account_checks, "completed_and_pending_cycles": cycle_rows, "distribution_checks": verification,
              "primary_cash_attribution": {"actual_shares_gross_pnl": gross, "commission_slippage_and_terminal_reserve": expense,
                                           "net_profit": float(ledger.equity.iloc[-1] - 200000.)},
              "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
              "mature_monitor": [{"model": name, "nonoverlap_mature_rows": len(part),
                                  "assessable_rows": int(part.monitor_assessable.sum()), "alerts": int(part.alert.sum()),
                                  "status": "ASSESSED" if part.monitor_assessable.any() else "NOT_ASSESSABLE"}
                                 for name, part in monitor.groupby("model")],
              "new_parameter_searches": 0, "new_market_downloads": 0, "historical_first_vintage_authenticated": False,
              "new_independent_forward_observations": 0, "current_market_view": "NO_VIEW", "goal_status": "active",
              "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print("回购披露与资金增量试验、8个完整账户及可用期诊断已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="已确认回购披露与宏观资金的固定账户增量")
    parser.add_argument("command", choices=["check", "freeze", "run"])
    args = parser.parse_args()
    if args.command == "check":
        print(implementation_check())
        x = information()
        print({"共同信息日": int(x.common_known.sum()), "新候选收益已计算": False})
    elif args.command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
