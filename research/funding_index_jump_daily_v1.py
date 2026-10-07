"""固定盘中变差与跳跃代理的增量，复用资金残差并运行两年日更账户。"""
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

import research.index_jump_measurement_core_v1 as core
import research.funding_calendar_innovation_daily_v1 as funding
import research.funding_afternoon_response_daily_v1 as previous
import research.repo_segmentation_daily_v1 as parent
import research.daily_liquidity_insurance_tail_v1 as distribution
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_funding_index_jump_daily_v1"
MINUTE = ROOT / "data/raw/market/000300_1m_tushare_raw.parquet"
QUALITY = ROOT / "reports/data_quality/000300_1m_proxy_quality.json"
PREVIOUS_COMPLETION = ROOT / "reports/research/510300_funding_afternoon_response_saved_completion_v1/result.json"
STUDY = "510300_FUNDING_INDEX_JUMP_DAILY_V1"
RV = "PRICE_AND_INTRADAY_RV"
JUMP = "PRICE_RV_AND_JUMP"
FUNDING_RV = "PRICE_FUNDING_AND_RV"
PRIMARY = "PRICE_FUNDING_RV_AND_JUMP"
MODELS = {"HISTORY": [], "PRICE": parent.BASE,
    RV: [*parent.BASE, "log_intraday_rv_ratio"],
    JUMP: [*parent.BASE, "log_intraday_rv_ratio", "jump_fraction"],
    FUNDING_RV: [*parent.BASE, "log_intraday_rv_ratio", "funding_innovation"],
    PRIMARY: [*parent.BASE, "log_intraday_rv_ratio", "funding_innovation", "jump_fraction"]}
NEW_MODELS = [RV, JUMP, FUNDING_RV, PRIMARY]
START, END = parent.START, parent.END
METHOD_URL = "https://shephard.scholars.harvard.edu/sites/g/files/omnuum7741/files/power.pdf"


def design():
    market = pd.read_parquet(parent.OUT / "inputs/market.parquet")
    x = pd.read_parquet(funding.OUT / "inputs/decision_information.parquet")
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    x["source_date"] = pd.to_datetime(x.source_date).astype("datetime64[ns]")
    minute = pd.read_parquet(MINUTE, columns=["ts_code", "trade_time", "close"])
    measurement = core.scale_measurement(core.measure(minute), market)
    x = x.merge(measurement, on="source_date", how="left", validate="many_to_one")
    x["common_known"] &= x.feature_known.eq(True)
    prior_x = pd.read_parquet(previous.OUT / "inputs/decision_information.parquet")
    # 只有共同样本逐日相同才允许复用旧价格账户；不能用覆盖变化制造增量。
    pd.testing.assert_series_equal(x.common_known, prior_x.common_known, check_names=False)
    pd.testing.assert_series_equal(x.date, prior_x.date, check_names=False)
    for column in parent.BASE:
        pd.testing.assert_series_equal(x[column], prior_x[column], check_names=False)
    valid = x.common_known
    assert x.loc[valid, "variation_available_at"].le(x.loc[valid, "decision_time"]).all()
    assert x.loc[valid, "source_date"].lt(x.loc[valid, "date"]).all()
    nested = pd.read_parquet(funding.OUT / "results/nested_funding_predictions.parquet")
    source = pd.read_parquet(funding.OUT / "inputs/funding_features.parquet")
    labels = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    return market, x, measurement, nested, source, labels, dividends


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("指数跳跃增量已经固定，不能覆盖原记录。")
    tests = core.implementation_checks()
    quality = read(QUALITY)
    assert quality["status"] == "PASS"
    market, x, measurement, nested, source, labels, dividends = design()
    assert len(measurement) == 1211 and measurement.feature_known.all()
    assert measurement.complete_positive_endpoints.all()
    assert read(funding.OUT / "result.json")["status"] == "FROZEN_NO_QUALIFIED_FUNDING_INNOVATION_STRATEGY"
    assert PREVIOUS_COMPLETION.exists()
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "在相同价格弱势、盘中波动规模及资金模型意外下，指数波动集中程度代理能否改善五日分布与完整账户？",
        "mechanism_hypothesis": "持续价格调整和少数集中变化可能对应不同的后续路径；跳跃不能直接识别消息、知情交易或被迫卖出。对收益改善只作待检验假设。",
        "measurement": {"asset": "000300.SH", "price": "分钟标签close", "sampling": core.SESSIONS,
            "returns_each_session": 22, "returns_total": 44,
            "RV": "两段内部五分钟对数收益平方之和",
            "BV": "每段(pi/2)*(22/21)*相邻收益绝对值乘积之和，再相加",
            "finite_sample_adjustment": "22/21仅补每段22收益只有21相邻积，在独立同方差正态模型下校正项数；并非对任意真实市场无偏。",
            "jump_fraction": "max(RV-BV,0)/RV；不是显著性跳跃检验，不识别跳跃方向",
            "rv_control": "log(RV/源日前20个510300含分红日对数收益平方均值)",
            "excluded": "隔夜、午休、开盘初段及收盘末段均未测量，不能称全天总跳跃风险",
            "zero_or_invalid": "端点缺失、非正价格或RV为0保留NO_VIEW，不填补"},
        "source_clock": "源日15:30计划可用，最早下一A股交易日09:00判断、09:30开盘；历史首次送达及分钟标签起止语义未认证。",
        "funding": "复用当前两年窗内重构的FDR007日历模型残差，不携带旧窗参数，不重估资金回归。",
        "models": MODELS, "primary_increment": f"{PRIMARY}减{FUNDING_RV}，控制细粒度RV后检验跳跃信息；另比较减{JUMP}以检验资金增量。",
        "other_controls": [RV, "PRICE"],
        "training": "六模型共同成熟池，最近两日历年且退出开盘严格早于当前判断；至少252原点，固定126近邻，训练内标准化、截断5，距离同分按原点编号。HISTORY使用完整共同池。",
        "reuse": "PRICE和HISTORY必须与旧下午响应研究逐日训练集、选中集和预测值一致，才只读复用PRICE两成本账户；四个新模型共8新账户。",
        "account": "沿用原每日滚动库存引擎而非按上一轮点值选择期限；新增只在五日已知价格弱势，T+1、整手、费用、现金股息分账与账户尾部限制均不变。",
        "assets": ["510300.SH", "CASH_CNY"], "capital": 200000,
        "period": [START, END], "annual_days": 242, "cash_and_risk_free_rate": 0,
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparison": "完整账户全部日期，20日循环区块2000次，seed2026092602；固定2020-2023及2024-末端，另保存所有滚动两年。",
        "development_gate": "主压力账户同时满足三目标，相对四对照的收益差95%下界均正且两固定时期均正，才可列开发候选；仍需独立验证。",
        "monitor": "固定五日间隔的成熟预测，最近60条q05跌穿率超过15%记警报，纯诊断不临时改账户；尾部预算与回撤规则继续执行。",
        "coverage": "全期账户保留分钟资料缺失和训练不足现金日；不以实际可预测短区间替代完整目标。",
        "duplicates": "旧下午价格、日内隔夜、成交压力、成分广度、限价和五日期限结果不重跑；本轮只改变指数RV与双幂差代理信息。",
        "new_parameter_grid": 0, "new_funding_regression_fits": 0, "new_full_accounts": 8,
        "reused_full_accounts": 2, "new_market_downloads": 0,
        "historical_first_vintage_authenticated": False, "new_independent_forward_observations": 0,
        "source_quality_warnings": quality["warnings"], "method_source": METHOD_URL,
        "goal_achieved": False, "orders_authorized": False}
    save(OUT / "protocol.json", protocol, True)
    save(OUT / "inputs/preflight_checks.json", {"at": now(), **tests,
        "source_dates": len(measurement), "complete_46_endpoints_dates": int(measurement.complete_positive_endpoints.sum()),
        "common_coverage_equals_saved_price_control": True, "new_candidate_returns_computed": False}, True)
    save(OUT / "inputs/method_source.json", {"url": METHOD_URL, "accessed_at": now(),
        "title": "Power and bipower variation with stochastic volatility and jumps",
        "authors": ["Ole E. Barndorff-Nielsen", "Neil Shephard"], "draft_date": "2003-11-02",
        "verified_sections": "第3页双幂定义；第8-9页正态绝对矩及概率极限；方法摘要",
        "use": "RV与归一化BV之差在相应过程假设下用于跳跃二次变差测量；有限采样代理不能认证新闻来源或交易收益。",
        "normalization": "mu1=E|N(0,1)|=sqrt(2/pi)，故1/mu1^2=pi/2；22/21为本方案预先固定的项数修正。"}, True)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    for module in [Path(__file__), Path(core.__file__)]:
        shutil.copy2(module, OUT / "code" / module.name)
    paths = [Path(__file__), Path(core.__file__), Path(funding.__file__), Path(previous.__file__),
        Path(parent.__file__), Path(distribution.__file__), Path(inventory.__file__), Path(engine.__file__),
        ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
        MINUTE, QUALITY, parent.DIVIDENDS, PREVIOUS_COMPLETION,
        funding.OUT / "result.json", funding.OUT / "inputs/decision_information.parquet",
        funding.OUT / "inputs/funding_features.parquet", funding.OUT / "results/nested_funding_predictions.parquet",
        parent.OUT / "inputs/market.parquet", parent.OUT / "inputs/mature_labels.parquet",
        previous.OUT / "inputs/decision_information.parquet", previous.OUT / "results/predictions.parquet",
        previous.OUT / "results/saved_distributions.json", previous.OUT / "results/update_receipts.parquet"]
    for cost in distribution.COSTS:
        for filename in ["ledger.parquet", "decisions.parquet", "cycles.parquet", "pending_cycle.json"]:
            paths.append(previous.OUT / "accounts" / cost / "PRICE" / filename)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "before_new_candidate_returns": True}, True)
    print("盘中RV与跳跃代理增量已固定；将生成8个新账户，复用2个共同价格对照。", flush=True)


def fit_day(i, x, nested, source_count, labels):
    day = x.date.iloc[i]
    lower = day - pd.DateOffset(years=2)
    receipt = {"idx": i, "date": day, "lower_bound": lower, "status": "NO_VIEW_SOURCE", "n_train": 0}
    if not bool(x.common_known.iloc[i]) or nested is None:
        return [], [], receipt
    assert nested.decision_idx.eq(i).all()
    residual = np.full(source_count, np.nan)
    residual[nested.source_idx.to_numpy(int)] = (nested.actual - nested.CALENDAR).to_numpy(float)
    mapped = x.source_idx.fillna(-1).to_numpy(int)
    innovations = np.full(len(x), np.nan)
    valid = mapped >= 0
    innovations[valid] = residual[mapped[valid]]
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)].copy()
    indices = pool.idx.to_numpy(int)
    keep = x.common_known.iloc[indices].to_numpy(bool) & np.isfinite(innovations[indices])
    pool = pool.loc[keep]
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None)
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        receipt["status"] = "NO_VIEW_TRAINING"
        return [], [], receipt
    receipt["status"] = "UPDATED"
    indices = pool.idx.to_numpy(int)
    values = x.loc[indices, [*parent.BASE, "log_intraday_rv_ratio", "jump_fraction"]].copy()
    values["funding_innovation"] = innovations[indices]
    current = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    predictions, models = [], []
    for name, columns in MODELS.items():
        mean, scale = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            assert np.isfinite(raw).all()
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1.
            standardized = np.clip((raw - mean) / scale, -5, 5)
            live = np.clip((np.array([current[k] for k in columns]) - mean) / scale, -5, 5)
            selected = np.lexsort((indices, np.sum((standardized - live)**2, axis=1)))[:126]
        else:
            selected = np.arange(len(pool))
        chosen = pool.iloc[selected]
        predictions.append({**receipt, "model": name, "n_selected": len(chosen),
            **distribution.empirical_statistics(chosen.gross_return5)})
        models.append({"idx": i, "model": name, "features": columns, "mean": mean.tolist(), "scale": scale.tolist(),
            "training_indices": indices.tolist(), "selected_indices": chosen.idx.tolist(),
            "current_funding_innovation": float(innovations[i]),
            "current_log_intraday_rv_ratio": float(current["log_intraday_rv_ratio"]),
            "current_jump_fraction": float(current["jump_fraction"])})
    return predictions, models, receipt


def verify_distributions(pred, models, receipts, labels, x, groups, source_count):
    old_pred = pd.read_parquet(previous.OUT / "results/predictions.parquet")
    columns = ["idx", "date", "model", "n_train", "n_selected", "mu5", "variance5", "q05", "es95", "win_probability", "selected_tail_count"]
    current = pred[pred.model.isin(["HISTORY", "PRICE"])][columns].sort_values(["idx", "model"]).reset_index(drop=True)
    old = old_pred[old_pred.model.isin(["HISTORY", "PRICE"])][columns].sort_values(["idx", "model"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(current, old, check_exact=True)
    old_schedule = pd.read_parquet(previous.OUT / "results/update_receipts.parquet")
    pd.testing.assert_frame_equal(receipts, old_schedule, check_exact=True)
    old_models = {(r["idx"], r["model"]): r for r in read(previous.OUT / "results/saved_distributions.json") if r["model"] in ["HISTORY", "PRICE"]}
    lookup, sample = pred.set_index(["idx", "model"]), labels.set_index("idx", drop=False)
    for record in models:
        i, name = record["idx"], record["model"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i] - pd.DateOffset(years=2)).all() and pool.exit_idx.lt(i).all()
        assert x.common_known.iloc[pool.idx.to_numpy(int)].all()
        if name in ["HISTORY", "PRICE"]:
            previous_record = old_models[i, name]
            for column in ["training_indices", "selected_indices", "features", "mean", "scale"]:
                assert record[column] == previous_record[column]
        values = distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in ["mu5", "variance5", "q05", "es95", "win_probability"]:
            np.testing.assert_allclose(values[key], lookup.loc[(i, name), key], atol=1e-14, rtol=0)
    checked_dates = []
    for day in ["2023-06-30", "2025-09-24"]:
        i = int(x.date.searchsorted(day, side="right")) - 1
        changed = labels.copy()
        excluded = changed.exit_idx.ge(i) | changed.date.lt(x.date.iloc[i] - pd.DateOffset(years=2))
        changed.loc[excluded, "gross_return5"] = -9999.
        rebuilt, _, _ = fit_day(i, x.iloc[:i + 1], groups.get(i), source_count, changed)
        assert len(rebuilt) == len(MODELS)
        for row in rebuilt:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == lookup.loc[(i, row["model"]), key]
        checked_dates.append(day)
    return {"saved_distributions_recomputed": len(models), "unchanged_price_history_controls": len(current),
        "same_source_training_status_and_samples": True, "future_and_expired_label_perturbation_dates": checked_dates}


def evaluate_forecasts(pred, labels, x):
    scored = pred.merge(labels[["idx", "gross_return5", "exit_idx", "exit_date"]], on="idx", how="left", validate="many_to_one")
    scored = scored.merge(x[["idx", "pressure5"]], on="idx", how="left", validate="many_to_one")
    scored.to_parquet(OUT / "results/scored_predictions.parquet", index=False)
    results = []
    for subset, frame in [("ALL", scored), ("PRIOR_FIVE_DAY_DECLINE", scored[scored.pressure5.gt(0)])]:
        measures = []
        for name, part in frame[frame.gross_return5.notna()].groupby("model", sort=False):
            error = part.gross_return5 - part.q05
            measures.append({"model": name, "n": len(part), "MSE": float(((part.gross_return5 - part.mu5)**2).mean()),
                "quantile_loss": float(np.maximum(.05 * error, -.95 * error).mean()),
                "q05_breach_fraction": float(part.gross_return5.lt(part.q05).mean())})
        lookup = {row["model"]: row for row in measures}
        results.append({"subset": subset, "models": measures,
            "primary_mse_improvement": {control: 1 - lookup[PRIMARY]["MSE"] / lookup[control]["MSE"] for control in ["HISTORY", "PRICE", RV, JUMP, FUNDING_RV]}})
    monitors = []
    first = int(x.index[x.date.ge(START)][0])
    for name, frame in scored[(scored.idx - first) % 5 == 0].groupby("model"):
        part = frame[frame.gross_return5.notna()].sort_values("exit_idx").copy()
        part["breach"] = part.gross_return5.lt(part.q05)
        part["breach_rate60"] = part.breach.rolling(60, min_periods=60).mean()
        part["alert"] = part.breach_rate60.gt(.15)
        monitors.append(part)
    monitor = pd.concat(monitors, ignore_index=True)
    monitor.to_parquet(OUT / "results/mature_monitor.parquet", index=False)
    return results, scored, monitor


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    market, x, measurement, nested, source, labels, dividends = design()
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    measurement.to_parquet(OUT / "inputs/index_variation.parquet", index=False)
    groups = {int(i): part for i, part in nested.groupby("decision_idx", sort=False)}
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = fit_day(int(i), x, groups.get(int(i)), len(source), labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"指数波动结构与资金分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred, schedule = pd.DataFrame(predictions), pd.DataFrame(receipts)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    verification = verify_distributions(pred, models, schedule, labels, x, groups, len(source))
    save(OUT / "results/prediction_verification.json", verification, True)
    print("共同样本、旧价格预测和时间隔离检查通过，开始8个新账户。", flush=True)
    forecast, scored, monitor = evaluate_forecasts(pred, labels, x)
    accounts, checks, cycle_rows, reused = {}, [], [], []
    for cost in distribution.COSTS:
        folder = previous.OUT / "accounts" / cost / "PRICE"
        ledger = pd.read_parquet(folder / "ledger.parquet")
        decisions = pd.read_parquet(folder / "decisions.parquet")
        accounts["PRICE", cost] = ledger
        checks.append({"model": "PRICE", "cost": cost, "reused": True, **parent.account_verification(ledger, decisions)})
        reused.append({"model": "PRICE", "cost": cost, "ledger": (folder / "ledger.parquet").relative_to(ROOT).as_posix(),
            "sha256": digest(folder / "ledger.parquet")})
    save(OUT / "results/reused_accounts.json", reused, True)
    proxy = SimpleNamespace(START=START, END=END, COSTS=distribution.COSTS, plan=parent.plan, risk_valid=parent.risk_valid)
    for name in NEW_MODELS:
        selected = pred[pred.model.eq(name)].copy()
        selected["model"] = "PRICE"
        for cost in distribution.COSTS:
            ledger, decisions = inventory.simulate(proxy, engine, market, x, dividends, selected, "PRICE_DOWN_ONLY", cost)
            ledger["policy"], decisions["policy"] = name, name
            folder = OUT / "accounts" / cost / name
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            closed, pending = cycles(ledger)
            closed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "pending_cycle.json", pending, True)
            accounts[name, cost] = ledger
            checks.append({"model": name, "cost": cost, "reused": False, **parent.account_verification(ledger, decisions)})
            cycle_rows.append({"model": name, "cost": cost, "closed": len(closed), "open": pending is not None})
            summary = metrics(ledger)
            sh = "未定义" if summary["sharpe"] is None else f"{summary['sharpe']:.6f}"
            print(f"{name}/{cost}：夏普{sh}、年化{summary['annual_return']:.3%}、回撤{abs(summary['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    rng = np.random.default_rng(2026092602)
    comparisons = []
    for cost in distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for control in [FUNDING_RV, JUMP, RV, "PRICE"]:
            right = accounts[control, cost]
            pd.testing.assert_series_equal(left.date, right.date)
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi,
                    "annual_arithmetic_difference": float((left.loc[mask, "net_return"] - right.loc[mask, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "control": control, **paired_interval(left, right, rng), "fixed_periods": periods})
    primary = next(row for row in measures if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    increment = all(row["lower_95"] > 0 and all(era["annual_arithmetic_difference"] > 0 for era in row["fixed_periods"])
        for row in comparisons if row["cost"] == "STRESS")
    updated = schedule[schedule.status.eq("UPDATED")]
    ready = pred[["idx", "date"]].drop_duplicates()
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000., atol=1e-6, rtol=0)
    result = {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_INDEX_JUMP_STRATEGY",
        "primary": primary, "all_accounts": measures, "comparisons": comparisons,
        "forecast_evaluation": forecast, "increment_gate_met": increment,
        "new_full_accounts": 8, "reused_full_accounts": 2, "new_funding_regression_fits": 0,
        "new_conditional_distributions": int(pred.model.isin(NEW_MODELS).sum()),
        "recomputed_control_distributions": int(pred.model.isin(["HISTORY", "PRICE"]).sum()),
        "reused_nested_funding_predictions_each_model": len(nested),
        "unique_prediction_days": len(ready), "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
        "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
        "source_variation_dates": len(measurement), "source_first_date": measurement.source_date.min(), "source_last_date": measurement.source_date.max(),
        "schedule_statuses": schedule.status.value_counts().to_dict(),
        "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
        "jump_fraction_summary": measurement.jump_fraction.describe(percentiles=[.05, .5, .95]).to_dict(),
        "zero_jump_proxy_dates": int(measurement.jump_fraction.eq(0).sum()),
        "account_checks": checks, "completed_and_pending_cycles": cycle_rows, "distribution_checks": verification,
        "primary_cash_attribution": {"actual_shares_gross_pnl": gross, "commission_slippage_and_terminal_reserve": expense,
            "net_profit": float(ledger.equity.iloc[-1] - 200000.)},
        "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in rolling if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
        "monitor_alerts": monitor.groupby("model").alert.sum().to_dict(),
        "new_parameter_searches": 0, "new_market_downloads": 0, "historical_first_vintage_authenticated": False,
        "new_independent_forward_observations": 0, "current_market_view": "NO_VIEW",
        "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print("指数波动结构的固定增量研究和8个新账户完成，结果已保存。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指数双幂变差与资金信息的固定增量检验")
    parser.add_argument("command", choices=["check", "freeze", "run"])
    args = parser.parse_args()
    if args.command == "check":
        print(core.implementation_checks())
        _, x, measurement, _, _, _, _ = design()
        print({"共同信息日期": int(x.common_known.sum()), "分钟测量日期": len(measurement), "测量全部可用": bool(measurement.feature_known.all())})
    elif args.command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
