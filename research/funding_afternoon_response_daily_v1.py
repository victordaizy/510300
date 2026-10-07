"""上午资金模型意外与下午价格反应的固定增量研究，只交易510300。"""
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

import research.funding_calendar_innovation_daily_v1 as prior
import research.repo_segmentation_daily_v1 as parent
import research.daily_liquidity_insurance_tail_v1 as distribution
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_funding_afternoon_response_daily_v1"
MINUTE = ROOT / "data/raw/market/510300_1m_tushare_raw.parquet"
QUALITY = ROOT / "reports/data_quality/510300_1m_tushare_quality.json"
STUDY = "510300_FUNDING_AFTERNOON_RESPONSE_DAILY_V1"
FUNDING = "PRICE_AND_FUNDING"
RESPONSE = "PRICE_AND_AFTERNOON"
PRIMARY = "PRICE_FUNDING_AND_AFTERNOON"
MODELS = {"HISTORY": [], "PRICE": parent.BASE,
          FUNDING: [*parent.BASE, "funding_innovation"],
          RESPONSE: [*parent.BASE, "afternoon_response"],
          PRIMARY: [*parent.BASE, "funding_innovation", "afternoon_response"]}
START, END = parent.START, parent.END


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("资金与下午响应方案已经固定，不覆盖原记录。")
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    quality = read(QUALITY)
    assert quality["coverage_status"] == "PASS" and digest(MINUTE) == quality["sha256"]
    prior_result = read(prior.OUT / "result.json")
    assert prior_result["status"] == "FROZEN_NO_QUALIFIED_FUNDING_INNOVATION_STRATEGY"
    protocol = {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "同样的上午资金模型意外，与午后继续下跌或企稳上涨共同观察时，能否更好地区分下一可交易阶段的结果？",
        "information": "沿用已保存、当前两年窗内嵌套计算的FDR007模型意外，只增加同一源日13:05至14:55标签close价格的对数变化，除以源日前20日含分红日收益均方根波动。",
        "funding_clock": "上午定盘按源日12:00计划可用；实际首次送达未认证。下午窗口在该计划时间之后。",
        "response_clock": "下午两个价格仅作为信号，源日15:30后视为可用，最早下一A股09:00判断、09:30开盘执行；不把确认期涨幅算进交易收益。",
        "minute_source": "已有第三方代理分钟档案，2021-08-12至2026-08-12共1211日；两个指定标签均存在。原供应商未明确定义标签是bar起点还是终点，不宣称精确秒级事件反应；两者均位于午后，仅用于次日信息。",
        "causal_limit": "下午价格受多种消息影响，不能证明它由资金意外引起；模型意外也不是市场一致预期差。本轮只检验预测和账户增量。",
        "models": MODELS,
        "primary_increment": "PRIMARY减RESPONSE，识别已有相同价格反应时资金信息的额外作用。另比较PRIMARY减FUNDING及PRIMARY减PRICE。",
        "factorial_diagnostic": "完整账户每日收益的PRIMARY-RESPONSE-FUNDING+PRICE，保留为联合增量描述；账户路径不同，不当成结构因果交互。",
        "training": "所有5个分布模型使用完全相同且已具备两个信息的训练池；最近两日历年、至少252个成熟原点、退出开盘严格早于当前09:00；沿用固定126近邻、训练内标准化截断5。",
        "reuse": "复用上一轮572489组嵌套资金预测，不重估其1144978次资金回归。只生成本轮新的条件分布和账户。旧六账户不作为本轮共同覆盖对照。",
        "coverage": "账户仍从2020-01-02持续到2026-09-24，包括分钟历史开始前的NO_VIEW、样本不足及末端缺少分钟信息的所有现金日；另描述实际可预测区间，不靠缩短评价期宣布通过。",
        "assets": ["510300.SH", "CASH_CNY"], "capital": 200000, "period": [START, END],
        "annual_days": 242, "cash_and_risk_free_rate": 0,
        "targets": {"sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "account": "四方案各BASE/STRESS共8个完整账户。只在已知五日价格弱势时允许新增库存；原目标效用、50%上限、ES95的2.5%、10%跳空的5%与峰值余量一半、T+1与压力费用均不变。",
        "comparison": "完整同日收益20日循环区块2000次，seed2026092601；固定2020-2023、2024-末端及所有滚动两年。",
        "development_gate": "主压力账户三目标通过；相对RESPONSE、FUNDING和PRICE的增量95%下界都正且两个固定时期均正，才留作开发候选；资料认证和独立验证仍未完成。",
        "duplicate_boundary": "不恢复旧收盘30分钟成交压力策略，不重做LPR与日终股债20日实验；本次检验上午资金残差加午后价格共同状态，单独保留下午价格对照。",
        "new_parameter_grid": 0, "new_funding_regression_fits": 0, "new_market_downloads": 0,
        "source_quality_warnings": quality["warnings"], "new_full_accounts": 8,
        "first_vintage_authenticated": False, "new_independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False}
    save(OUT / "protocol.json", protocol, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    paths = [Path(__file__), Path(prior.__file__), Path(parent.__file__), Path(distribution.__file__),
             Path(inventory.__file__), Path(engine.__file__), parent.DIVIDENDS, MINUTE, QUALITY,
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             prior.OUT / "result.json", prior.OUT / "protocol.json", prior.OUT / "inputs/decision_information.parquet",
             prior.OUT / "inputs/funding_features.parquet", prior.OUT / "results/nested_funding_predictions.parquet",
             parent.OUT / "inputs/market.parquet", parent.OUT / "inputs/mature_labels.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_candidate_returns": True}, True)
    print("上午资金与下午价格的四组信息对照及八账户已固定。", flush=True)


def design():
    market = pd.read_parquet(parent.OUT / "inputs/market.parquet")
    x = pd.read_parquet(prior.OUT / "inputs/decision_information.parquet")
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    x["source_date"] = pd.to_datetime(x.source_date).astype("datetime64[ns]")
    minute = pd.read_parquet(MINUTE, columns=["ts_code", "trade_time", "close"])
    assert set(minute.ts_code) == {"510300.SH"}
    minute["trade_time"] = pd.to_datetime(minute.trade_time).astype("datetime64[ns]")
    minute["clock"] = minute.trade_time.dt.strftime("%H:%M:%S")
    minute["source_date"] = minute.trade_time.dt.normalize()
    chosen = minute[minute.clock.isin(["13:05:00", "14:55:00"])].copy()
    assert not chosen.duplicated(["source_date", "clock"]).any()
    response = chosen.pivot(index="source_date", columns="clock", values="close").reset_index()
    assert len(response) == 1211 and response[["13:05:00", "14:55:00"]].gt(0).all().all()
    vol = pd.DataFrame({"source_date": market.date, "source_prior_rv20": np.sqrt(market.total_log.pow(2).rolling(20).mean()).shift(1)})
    response = response.merge(vol, on="source_date", how="left", validate="one_to_one")
    response["afternoon_log_return"] = np.log(response["14:55:00"]/response["13:05:00"])
    response["afternoon_response"] = response.afternoon_log_return/response.source_prior_rv20
    response["response_available_at"] = pd.to_datetime(response.source_date).dt.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15, minutes=30)
    x = x.merge(response, on="source_date", how="left", validate="many_to_one")
    x["common_known"] &= x.afternoon_response.notna() & np.isfinite(x.afternoon_response)
    assert x.loc[x.common_known, "response_available_at"].le(x.loc[x.common_known, "decision_time"]).all()
    assert x.loc[x.common_known, "source_date"].lt(x.loc[x.common_known, "date"]).all()
    nested = pd.read_parquet(prior.OUT / "results/nested_funding_predictions.parquet")
    source = pd.read_parquet(prior.OUT / "inputs/funding_features.parquet")
    labels = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    return market, x, response, nested, source, labels, dividends


def fit_day(i, x, nested, source_count, labels):
    day = x.date.iloc[i]
    lower = day-pd.DateOffset(years=2)
    receipt = {"idx": i, "date": day, "lower_bound": lower, "status": "NO_VIEW_SOURCE", "n_train": 0}
    if not bool(x.common_known.iloc[i]) or nested is None:
        return [], [], receipt
    assert nested.decision_idx.eq(i).all()
    residual = np.full(source_count, np.nan)
    residual[nested.source_idx.to_numpy(int)] = (nested.actual-nested.CALENDAR).to_numpy(float)
    mapped = x.source_idx.fillna(-1).to_numpy(int)
    innovations = np.full(len(x), np.nan)
    valid = mapped >= 0
    innovations[valid] = residual[mapped[valid]]
    pool = labels[labels.date.ge(lower) & labels.exit_idx.lt(i)].copy()
    index = pool.idx.to_numpy(int)
    keep = x.common_known.iloc[index].to_numpy(bool) & np.isfinite(innovations[index])
    pool = pool.loc[keep]
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None)
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        receipt["status"] = "NO_VIEW_TRAINING"
        return [], [], receipt
    receipt["status"] = "UPDATED"
    indices = pool.idx.to_numpy(int)
    values = x.loc[indices, [*parent.BASE, "afternoon_response"]].copy()
    values["funding_innovation"] = innovations[indices]
    current = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    predictions, models = [], []
    for name, columns in MODELS.items():
        mean, sd = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            sd[sd < 1e-12] = 1.
            z = np.clip((raw-mean)/sd, -5, 5)
            live = np.clip((np.array([current[k] for k in columns])-mean)/sd, -5, 5)
            selected = np.lexsort((indices, np.sum((z-live)**2, axis=1)))[:126]
        else:
            selected = np.arange(len(pool))
        chosen = pool.iloc[selected]
        predictions.append({**receipt, "model": name, "n_selected": len(chosen),
                            **distribution.empirical_statistics(chosen.gross_return5)})
        models.append({"idx": i, "model": name, "features": columns, "mean": mean.tolist(), "scale": sd.tolist(),
                       "training_indices": indices.tolist(), "selected_indices": chosen.idx.tolist(),
                       "current_funding_innovation": float(innovations[i]), "current_afternoon_response": current["afternoon_response"]})
    return predictions, models, receipt


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    market, x, response, nested, source, labels, dividends = design()
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    response.to_parquet(OUT / "inputs/afternoon_response.parquet", index=False)
    groups = {int(i): part for i, part in nested.groupby("decision_idx", sort=False)}
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r = fit_day(int(i), x, groups.get(int(i)), len(source), labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts)%300 == 0:
            print(f"资金与下午价格联合分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    schedule = pd.DataFrame(receipts)
    schedule.to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    original = prior.forecast_evaluation
    evaluate = FunctionType(original.__code__, {**original.__globals__, "OUT": OUT, "PRIMARY": PRIMARY, "RAW": RESPONSE},
                            original.__name__, original.__defaults__, original.__closure__)
    forecast, scored, monitor = evaluate(pred, labels, x)
    for record in forecast:
        record["primary_mse_improvement_vs_afternoon_only"] = record.pop("primary_mse_improvement_vs_raw")
    proxy = SimpleNamespace(START=START, END=END, COSTS=distribution.COSTS, plan=parent.plan, risk_valid=parent.risk_valid)
    accounts, checks, cycle_rows = {}, [], []
    for name in ["PRICE", FUNDING, RESPONSE, PRIMARY]:
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
            checks.append({"model": name, "cost": cost, **parent.account_verification(ledger, decisions)})
            cycle_rows.append({"model": name, "cost": cost, "closed": len(closed), "open": pending is not None})
            summary = metrics(ledger)
            sharpe = "未定义" if summary["sharpe"] is None else f"{summary['sharpe']:.6f}"
            print(f"{name}/{cost}：夏普{sharpe}、年化{summary['annual_return']:.3%}、回撤{abs(summary['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    rng = np.random.default_rng(2026092601)
    comparisons, factorial = [], []
    for cost in distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for control in [RESPONSE, FUNDING, "PRICE"]:
            right = accounts[control, cost]
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"]-right.loc[mask, "net_return"]).mean()*242)})
            comparisons.append({"cost": cost, "control": control, **paired_interval(left, right, rng), "fixed_periods": periods})
        daily = left.net_return-accounts[RESPONSE, cost].net_return-accounts[FUNDING, cost].net_return+accounts["PRICE", cost].net_return
        factorial.append({"cost": cost, "annual_arithmetic_difference": float(daily.mean()*242),
                          **paired_interval(pd.DataFrame({"net_return": daily}), pd.DataFrame({"net_return": np.zeros(len(daily))}), rng),
                          "causal_identification": False})
    lookup, sample = pred.set_index(["idx", "model"]), labels.set_index("idx")
    for record in models:
        i = record["idx"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i]-pd.DateOffset(years=2)).all() and pool.exit_idx.lt(i).all()
        assert x.common_known.iloc[pool.idx.to_numpy(int)].all()
        values = distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(values[key], lookup.loc[(i, record["model"]), key], atol=1e-14, rtol=0)
    verified_dates = []
    for day in ["2023-06-30", "2025-09-24"]:
        i = int(x.date.searchsorted(day, side="right"))-1
        restricted = labels[labels.date.le(x.date.iloc[i])].copy()
        restricted.loc[restricted.exit_idx.ge(i), "gross_return5"] = -9999.
        p, _, _ = fit_day(i, x.iloc[:i+1], groups.get(i), len(source), restricted)
        assert len(p) == len(MODELS)
        for row in p:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == lookup.loc[(i, row["model"]), key]
        verified_dates.append(day)
    primary = next(row for row in measures if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    increment = all(row["lower_95"] > 0 and all(era["annual_arithmetic_difference"] > 0 for era in row["fixed_periods"]) for row in comparisons if row["cost"] == "STRESS")
    updated = schedule[schedule.status.eq("UPDATED")]
    ready_days = pred[["idx", "date"]].drop_duplicates()
    result = {"at": now(), "study_id": STUDY,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_FUNDING_RESPONSE_STRATEGY",
              "primary": primary, "all_accounts": measures, "comparisons": comparisons, "factorial_diagnostic": factorial,
              "forecast_evaluation": forecast, "new_full_accounts": 8, "new_funding_regression_fits": 0,
              "reused_nested_funding_predictions_each_model": len(nested), "daily_distribution_updates": len(models),
              "unique_prediction_days": len(ready_days), "first_prediction_date": ready_days.date.min(), "last_prediction_date": ready_days.date.max(),
              "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "source_response_pairs": len(response), "source_first_date": response.source_date.min(), "source_last_date": response.source_date.max(),
              "schedule_statuses": schedule.status.value_counts().to_dict(), "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
              "account_checks": checks, "completed_and_pending_cycles": cycle_rows,
              "saved_distributions_recomputed": len(models), "future_label_exclusion_dates": verified_dates,
              "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in rolling if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
              "monitor_alerts": monitor.groupby("model").alert.sum().to_dict(), "increment_gate_met": increment,
              "new_parameter_searches": 0, "new_market_downloads": 0, "historical_first_vintage_authenticated": False,
              "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("资金与下午价格的固定联合信息研究及八账户已完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上午资金意外与下午价格反应的增量研究")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
