"""区分利率水平和事前模型意外：两年日更的六个510300完整账户。"""
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

import research.funding_calendar_innovation_core_v1 as funding
import research.repo_segmentation_daily_v1 as parent
import research.daily_liquidity_insurance_tail_v1 as distribution
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_funding_calendar_innovation_daily_v1"
SOURCE = parent.SOURCE
STUDY = "510300_FUNDING_CALENDAR_INNOVATION_DAILY_V1"
PRIMARY = "PRICE_AND_FUNDING_INNOVATION"
RAW = "PRICE_AND_RAW_FDR"
MODELS = {"HISTORY": [], "PRICE": parent.BASE, RAW: [*parent.BASE, "raw_fdr"],
          PRIMARY: [*parent.BASE, "funding_innovation"]}
START, END = parent.START, parent.END
MECHANISM_URL = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3066656/61ed5a0548c542428584bccc244c9d5c/2016110818532756152.pdf"


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本研究已固定，不能覆盖。")
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    verified = funding.checks()
    protocol = {
        "at": now(), "study_id": STUDY, "primary": PRIMARY, "level_control": RAW, "price_control": "PRICE",
        "question": "资金利率中的月初、月末、季末可预期成分与其余意外变化分开后，能否改善下跌后的510300条件收益与账户？",
        "mechanism": "央行报告说明资金需求受季末、税期和节假日影响。本轮只建模自然月初、月末、季末，不声称穷尽税期、节假日、监管与政策冲击。",
        "primary_source_for_mechanism_only": MECHANISM_URL,
        "source": "复用官方FDR007上午定盘，2017-05-31至2026-09-24共2328日；不是DR007全天利率，也不是中性利率。源日12时计划可用，A股次开盘执行，年龄最多7自然日。历史首版/首次送达未认证。",
        "funding_forecast": {"features": funding.ALL_COLUMNS, "lags": "前一个定盘、前20个定盘均值",
            "calendar": "自然月前3日、自然月末5日、季末月末5日；不读取未来实际开休市长度。",
            "regression": "训练内标准化岭回归，alpha固定10、截距不惩罚，无截断或非负修正。",
            "minimum_training_rows": 126,
            "forecast_controls": ["PERSISTENCE_LAST_RATE", "AR_TWO_INPUTS", "AR_PLUS_THREE_CALENDAR_INPUTS"],
            "strict_two_year_memory": "对每个A股决策日重置两年原始利率窗，前20行只作滞后输入。窗内每个源日j只用j之前的利率拟合，至少126行；历史残差同样重新嵌套计算。不携带两年前模型参数，不使用j本身或后续数值预测j。",
            "innovation": "FDR007实际值减上述日历模型事前预测，百分点。属于模型意外，不是市场一致预期差，也不识别纯资金供给冲击。"},
        "equity_prediction": "价格、原始利率、利率意外三方案和HISTORY使用完全相同的成熟训练池；最近两日历年，至少252行，退出开盘严格早于当前09:00。沿用标准化截断5、固定126近邻经验分布。",
        "accounts": "三个方案各BASE/STRESS，共6连续账户。仅五日价格下跌时增加库存；沿用原固定库存效用与扣调仓费后的尾部预算，不修改入场或退出窗口。",
        "assets": ["510300.SH", "CASH_CNY"], "initial_capital": 200000,
        "period": [START, END], "annual_days": 242, "cash_and_riskfree_rate": 0,
        "targets": {"sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "risk": "目标仓位50%；5日ES95预算2.5%；10%跳空预算5%且至多使用距90%权益峰值余量一半；调仓及退出费用先计入。10%回撤后下一可卖开盘退出，本轮不恢复。",
        "costs": distribution.COSTS,
        "comparison": "主方案对原始利率及价格，完整同日账户20日循环区块2000次，seed2026092511；固定2020-2023/2024-末端及全部滚动两年。",
        "prediction_evaluation": "全部成熟及当时五日下跌样本，均值MSE和5%分位损失。资金预测按每个源日首次进入A股决策时的预测计一次，嵌套重复拟合不当新增独立观察。",
        "development_gate": "主压力三目标联合通过，相对两个对照的增量95%下界都正且两个固定时期均正；日历资金预测MSE同时优于持续性和AR对照。仍需独立前向验证。",
        "monitor": "每五个A股原点的固定相位，只收已成熟标签，最近60个5%分位突破率大于15%记录，不根据结果反调启停。",
        "duplicate_boundary": "旧日历直接择时、FR-FDR水平、央行操作量相对20日均值以及已失败的库存/HMM用途保留。本次只检验在当前两年窗口内部因果重建的资金利率预测残差，不重搜旧参数。",
        "new_market_downloads": 0, "new_final_accounts": 6, "parameter_searches": 0,
        "previously_researched_history": True, "independent_forward_observations": 0,
        "implementation_checks": verified, "orders_authorized": False, "goal_achieved": False,
    }
    save(OUT / "protocol.json", protocol, True)
    for code in [Path(__file__), Path(funding.__file__)]:
        shutil.copy2(code, OUT / "code" / code.name)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    paths = [Path(__file__), Path(funding.__file__), Path(parent.__file__), Path(distribution.__file__),
             Path(inventory.__file__), Path(engine.__file__), parent.DIVIDENDS,
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             parent.OUT / "inputs/market.parquet", parent.OUT / "inputs/decision_information.parquet",
             parent.OUT / "inputs/mature_labels.parquet", SOURCE / "fixing_segmentation.parquet", SOURCE / "result.json"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
         "before_new_candidate_returns": True}, True)
    print("资金利率事前模型意外、两项对照及六个账户已固定。", flush=True)


def inputs():
    market = pd.read_parquet(parent.OUT / "inputs/market.parquet")
    x = pd.read_parquet(parent.OUT / "inputs/decision_information.parquet", columns=["date", "idx", "decision_time", *parent.BASE])
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    x["decision_time"] = pd.to_datetime(x.decision_time).dt.as_unit("ns")
    source = funding.prepare(pd.read_parquet(SOURCE / "fixing_segmentation.parquet"))
    source["available_at"] = pd.to_datetime(source.available_at).dt.as_unit("ns")
    merge = source[["date", "source_idx", "available_at", "fdr007_percent"]].rename(columns={"date": "source_date", "fdr007_percent": "raw_fdr"})
    x = pd.merge_asof(x.sort_values("decision_time"), merge.sort_values("available_at"),
                      left_on="decision_time", right_on="available_at", direction="backward")
    x["common_known"] = x[parent.BASE].notna().all(axis=1) & x.raw_fdr.notna() & (x.date-x.source_date).dt.days.between(1, 7)
    assert x.loc[x.common_known, "available_at"].le(x.loc[x.common_known, "decision_time"]).all()
    labels = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(parent.DIVIDENDS))
    return market, x, source, labels, dividends


def fit_day(i, x, source, sufficient, labels):
    today = x.date.iloc[i]
    lower = today - pd.DateOffset(years=2)
    receipt = {"idx": i, "date": today, "lower_bound": lower, "status": "NO_VIEW_SOURCE"}
    if not bool(x.common_known.iloc[i]):
        return [], [], receipt, pd.DataFrame(), None
    source_i = int(x.source_idx.iloc[i])
    curve = funding.forecast_curve(source, sufficient, lower, source_i)
    if not len(curve["source_indices"]):
        return [], [], {**receipt, "status": "NO_VIEW_FUNDING_TRAINING"}, pd.DataFrame(), None
    residual = np.full(len(source), np.nan)
    selected_sources = curve["source_indices"]
    residual[selected_sources] = sufficient["y"][selected_sources] - curve["CALENDAR"]
    mapped = x.source_idx.fillna(-1).to_numpy(int)
    innovations = np.full(len(x), np.nan)
    valid = mapped >= 0
    innovations[valid] = residual[mapped[valid]]
    eligible = labels.date.ge(lower) & labels.exit_idx.lt(i)
    pool = labels.loc[eligible].copy()
    positions = pool.idx.to_numpy(int)
    keep = x.common_known.iloc[positions].to_numpy(bool) & np.isfinite(innovations[positions])
    pool = pool.loc[keep]
    receipt.update(n_train=len(pool), latest_exit_idx=int(pool.exit_idx.max()) if len(pool) else None,
                   funding_raw_start=curve["raw_start"], funding_train_start=curve["train_start"],
                   funding_nested_targets=len(selected_sources), current_source_idx=source_i)
    nested = pd.DataFrame({"decision_idx": i, "source_idx": selected_sources,
                           "n_train": selected_sources-curve["train_start"],
                           "train_start": curve["train_start"], "raw_start": curve["raw_start"],
                           "AR": curve["AR"], "CALENDAR": curve["CALENDAR"],
                           "actual": sufficient["y"][selected_sources]})
    current = {"decision_idx": i, "decision_date": today, "source_idx": source_i,
               "source_date": source.date.iloc[source_i], "n_train": source_i-curve["train_start"],
               "train_start": curve["train_start"], "raw_start": curve["raw_start"],
               "actual": float(sufficient["y"][source_i]), "PERSISTENCE": float(source.lag1.iloc[source_i]),
               "AR": float(curve["AR"][-1]), "CALENDAR": float(curve["CALENDAR"][-1]),
               "innovation": float(residual[source_i]),
               "AR_coefficients": curve["AR_beta"][-1].tolist(),
               "calendar_coefficients": curve["CALENDAR_beta"][-1].tolist()}
    if len(pool) < 252 or not np.isfinite(innovations[i]):
        return [], [], {**receipt, "status": "NO_VIEW_EQUITY_TRAINING"}, nested, current
    receipt["status"] = "UPDATED"
    indices = pool.idx.to_numpy(int)
    values = x.loc[indices, [*parent.BASE, "raw_fdr"]].copy()
    values["funding_innovation"] = innovations[indices]
    current_values = {**x.iloc[i].to_dict(), "funding_innovation": innovations[i]}
    predictions, models = [], []
    for name, columns in MODELS.items():
        mean, scale = np.array([]), np.array([])
        if columns:
            raw = values[columns].to_numpy(float)
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1.
            z = np.clip((raw-mean)/scale, -5, 5)
            live = np.clip((np.array([current_values[k] for k in columns])-mean)/scale, -5, 5)
            distance = np.sum((z-live)**2, axis=1)
            selected = np.lexsort((indices, distance))[:126]
        else:
            selected = np.arange(len(pool))
        chosen = pool.iloc[selected]
        summary = distribution.empirical_statistics(chosen.gross_return5)
        predictions.append({**receipt, "model": name, "n_selected": len(chosen), **summary})
        models.append({"idx": i, "model": name, "features": columns, "mean": mean.tolist(), "scale": scale.tolist(),
                       "training_indices": indices.tolist(), "selected_indices": chosen.idx.tolist(),
                       "current_funding_innovation": float(innovations[i]), "current_raw_fdr": current_values["raw_fdr"]})
    return predictions, models, receipt, nested, current


def forecast_evaluation(pred, labels, x):
    scored = pred.merge(labels[["idx", "gross_return5", "exit_idx", "exit_date"]], on="idx", how="left", validate="many_to_one")
    scored = scored.merge(x[["idx", "pressure5"]], on="idx", how="left", validate="many_to_one")
    scored.to_parquet(OUT / "results/scored_predictions.parquet", index=False)
    results, monitors = [], []
    for subset, frame in [("ALL", scored), ("PRIOR_FIVE_DAY_DECLINE", scored[scored.pressure5.gt(0)])]:
        frame = frame[frame.gross_return5.notna()]
        measures = []
        for model, part in frame.groupby("model"):
            error = part.gross_return5-part.q05
            measures.append({"model": model, "n": len(part), "MSE": float(((part.gross_return5-part.mu5)**2).mean()),
                             "quantile_loss": float(np.maximum(.05*error, -.95*error).mean())})
        lookup = {row["model"]: row for row in measures}
        results.append({"subset": subset, "models": measures,
                        "primary_mse_improvement_vs_price": 1-lookup[PRIMARY]["MSE"]/lookup["PRICE"]["MSE"],
                        "primary_mse_improvement_vs_raw": 1-lookup[PRIMARY]["MSE"]/lookup[RAW]["MSE"]})
    first = int(x.index[x.date.ge(START)][0])
    for name, frame in scored[(scored.idx-first)%5 == 0].groupby("model"):
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
    market, x, source, labels, dividends = inputs()
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    source.to_parquet(OUT / "inputs/funding_features.parquet", index=False)
    sufficient = funding.prefixes(source)
    predictions, models, receipts, nested_frames, current_funding = [], [], [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, m, r, nested, current = fit_day(int(i), x, source, sufficient, labels)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if not nested.empty:
            nested_frames.append(nested)
        if current is not None:
            current_funding.append(current)
        if len(receipts)%200 == 0:
            print(f"资金意外与指数分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    nested = pd.concat(nested_frames, ignore_index=True)
    nested.to_parquet(OUT / "results/nested_funding_predictions.parquet", index=False)
    current_frame = pd.DataFrame(current_funding)
    current_frame.to_parquet(OUT / "results/current_funding_predictions.parquet", index=False)
    unique = current_frame.sort_values("decision_idx").drop_duplicates("source_idx", keep="first")
    funding_eval = {name: {"n": len(unique), "MSE": float(((unique.actual-unique[name])**2).mean()),
                          "MAE": float((unique.actual-unique[name]).abs().mean())}
                    for name in ["PERSISTENCE", "AR", "CALENDAR"]}
    funding_gate = all(funding_eval["CALENDAR"]["MSE"] < funding_eval[control]["MSE"] for control in ["PERSISTENCE", "AR"])
    forecast, scored, monitor = forecast_evaluation(pred, labels, x)
    accounts, account_checks, cycle_rows = {}, [], []
    proxy = SimpleNamespace(START=START, END=END, COSTS=distribution.COSTS, plan=parent.plan, risk_valid=parent.risk_valid)
    for model in ["PRICE", RAW, PRIMARY]:
        selected = pred[pred.model.eq(model)].copy()
        selected["model"] = "PRICE"
        for cost in distribution.COSTS:
            ledger, decisions = inventory.simulate(proxy, engine, market, x, dividends, selected, "PRICE_DOWN_ONLY", cost)
            ledger["policy"], decisions["policy"] = model, model
            folder = OUT / "accounts" / cost / model
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            closed, pending = cycles(ledger)
            closed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "pending_cycle.json", pending, True)
            accounts[model, cost] = ledger
            account_checks.append({"model": model, "cost": cost, **parent.account_verification(ledger, decisions)})
            cycle_rows.append({"model": model, "cost": cost, "closed": len(closed), "open": pending is not None})
            measures = metrics(ledger)
            print(f"{model}/{cost}：夏普{measures['sharpe']:.6f}、年化{measures['annual_return']:.3%}、回撤{abs(measures['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    comparisons = []
    rng = np.random.default_rng(2026092511)
    for cost in distribution.COSTS:
        for control in [RAW, "PRICE"]:
            left, right = accounts[PRIMARY, cost], accounts[control, cost]
            periods = []
            for lo, hi in [(START, "2023-12-31"), ("2024-01-01", END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi, "annual_arithmetic_difference": float((left.loc[mask, "net_return"]-right.loc[mask, "net_return"]).mean()*242)})
            comparisons.append({"cost": cost, "control": control, **paired_interval(left, right, rng), "fixed_periods": periods})
    saved_lookup = pred.set_index(["idx", "model"])
    sample = labels.set_index("idx")
    for record in models:
        i = record["idx"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i]-pd.DateOffset(years=2)).all() and pool.exit_idx.lt(i).all()
        stats = distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(stats[key], saved_lookup.loc[(i, record["model"]), key], atol=1e-14, rtol=0)
    assert nested.source_idx.sub(nested.train_start).ge(126).all()
    assert nested.train_start.sub(nested.raw_start).eq(20).all()
    for day in ["2022-12-30", "2025-09-24"]:
        i = int(x.date.searchsorted(day, side="right"))-1
        restricted = labels[labels.date.le(x.date.iloc[i])].copy()
        restricted.loc[restricted.exit_idx.ge(i), "gross_return5"] = -9999.
        p, _, _, _, _ = fit_day(i, x.iloc[:i+1], source, sufficient, restricted)
        assert len(p) == len(MODELS)
        for row in p:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == saved_lookup.loc[(i, row["model"]), key]
    primary = next(row for row in measures if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    stress = [row for row in comparisons if row["cost"] == "STRESS"]
    increment_gate = all(row["lower_95"] > 0 and all(era["annual_arithmetic_difference"] > 0 for era in row["fixed_periods"]) for row in stress)
    passed = primary["historical_point_targets_met"] and increment_gate and funding_gate
    result = {"at": now(), "study_id": STUDY,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if passed else "FROZEN_NO_QUALIFIED_FUNDING_INNOVATION_STRATEGY",
              "primary": primary, "all_accounts": measures, "comparisons": comparisons,
              "funding_forecast_evaluation": funding_eval, "funding_forecast_gate": funding_gate,
              "forecast_evaluation": forecast, "new_full_accounts": 6,
              "daily_distribution_updates": len(models), "unique_prediction_days": int(pred.idx.nunique()),
              "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "nested_funding_forecasts_each_model": len(nested), "funding_regression_fits": len(nested)*2,
              "unique_funding_evaluation_days": len(unique), "nested_forecasts_are_independent_observations": False,
              "saved_distributions_recomputed": len(models), "future_label_exclusion_checks": 2,
              "account_checks": account_checks, "completed_and_pending_cycles": cycle_rows,
              "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in rolling if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
              "monitor_alerts": monitor.groupby("model").alert.sum().to_dict(),
              "increment_gate_met": increment_gate, "source_first_vintage_verified": False,
              "new_market_downloads": 0, "new_parameter_searches": 0,
              "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("利率水平与事前模型意外的固定六账户研究完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="资金利率可预期成分与模型意外研究")
    parser.add_argument("command", choices=["check", "freeze", "run"])
    args = parser.parse_args()
    if args.command == "check":
        print(funding.checks())
    elif args.command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
