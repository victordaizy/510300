"""510300 事先可知日历条件的完整账户研究，仅历史模拟。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.adaptive_allocation_v1 import ROOT, identity, model_for, eligible_training, target_request, summarize, save_account
from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, fill_price, choose_order, execute_order, normalize_dividends,
    digest, now, require, write_json, block_indices, return_metrics, interval,
)

STUDY = "510300_CALENDAR_LIQUIDITY_TIMING_V1"
OUT = ROOT / "reports/research/510300_calendar_liquidity_timing_v1"
CONFIG = ROOT / "config/510300_calendar_liquidity_timing_v1.json"
MANIFEST = ROOT / "config/510300_calendar_liquidity_timing_v1_manifest.json"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
CALENDAR_COLUMNS = [
    "weekday_mon", "weekday_tue", "weekday_wed", "weekday_thu", "weekday_fri",
    "month_phase_sin", "month_phase_cos", "year_phase_sin", "year_phase_cos",
    "month_ordinal_scaled", "remaining_calendar_days_scaled", "month_first3",
    "month_last5_calendar", "quarter_first3", "quarter_last5_calendar",
    "reopen_gap_log", "post_break_first3", "month_edge",
]
RULES = {
    "K1_PRIMARY_MONTH_START3": "主方案：每月前三个交易日",
    "K2_MONTH_EDGE": "月末五个自然日与月初三日",
    "K3_QUARTER_START3": "每季度首月前三个交易日",
    "K4_POST_LONG_BREAK3": "长休市之后前三个交易日",
    "K5_EQUAL_EVENT_VOTE": "四类日历条件等权",
}
ENSEMBLES = {"K6_CALENDAR_CONSENSUS": "CALENDAR", "K7_PRICE_CONSENSUS": "PRICE", "K8_FULL_CONSENSUS": "FULL"}


def specification() -> dict:
    config = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config.update({"study_id": STUDY, "primary": "K1_PRIMARY_MONTH_START3", "model_start": "2019-12-31",
                   "calendar_first_month_sessions": 3, "calendar_month_end_natural_days": 5,
                   "long_break_minimum_date_gap_days": 4, "post_break_sessions": 3,
                   "calendar_columns": CALENDAR_COLUMNS, "rule_names": RULES, "meta_names": ENSEMBLES,
                   "decision_clock": "EXECUTION_DAY_09_00_ASIA_SHANGHAI",
                   "calendar_uses_no_sessions_after_execution_date": True,
                   "calendar_source_first_delivery_proven": False,
                   "diagnostic_only_zero_cost_models": ["K1_PRIMARY_MONTH_START3", "K8_FULL_CONSENSUS"],
                   "models": [{"id": f"{group}_{kind}_H1", "group": group, "kind": kind, "horizon": 1}
                              for group in ("CALENDAR", "PRICE", "FULL") for kind in ("RIDGE", "ET")],
                   "evidence_class": "PREQUENTIAL_REPLAY_OF_ALREADY_OBSERVED_HISTORY"})
    return config


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "第六轮已冻结，禁止覆盖")
    OUT.mkdir(parents=True, exist_ok=True)
    config = specification()
    write_json(CONFIG, config, exclusive=True)
    paths = {CONFIG, Path(__file__), ROOT / "docs/510300_CALENDAR_LIQUIDITY_TIMING_V1_PROTOCOL.md",
             ROOT / "tests/test_calendar_liquidity_timing_v1.py", ROOT / "config/510300_research_authority_v6.json",
             ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
             PARENT / "features.parquet", PARENT / "input_receipt.json",
             ROOT / "reports/research/510300_overnight_global_information_v1/labels.parquet",
             ROOT / "research/overnight_global_information_v1.py",
             ROOT / "docs/A_SHARE_HS_CROSS_SECTIONAL_RETURN_SEASONALITY_V1_PROTOCOL.md",
             ROOT / "reports/research/A_SHARE_HS_CROSS_SECTIONAL_RETURN_SEASONALITY_V1.md"}
    parent_manifest = ROOT / "config/510300_adaptive_allocation_v1_manifest.json"
    paths.add(parent_manifest)
    paths.update(ROOT / item["path"] for item in json.loads(parent_manifest.read_text(encoding="utf-8"))["files"])
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [identity(p) for p in sorted(paths)],
               "own_candidate_returns_read_before_freeze": False, "underlying_history_previously_observed": True,
               "prior_rounds_known": 5, "registered_candidates": 14,
               "old_cross_sectional_same_month_return_strategy_preserved": True,
               "method_references": [
                   "https://business.purdue.edu/faculty/mcconnell/publications/Equity-Returns-at-the-Turn-of-the-Month.pdf",
                   "https://doi.org/10.1111/j.1540-6261.1990.tb02435.x"]}, exclusive=True)
    print(json.dumps({"状态": "第六轮已冻结", "清单哈希": digest(MANIFEST)}, ensure_ascii=False), flush=True)


def calendar_features(open_dates: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(open_dates)).normalize()
    require(len(dates) > 0 and dates.is_monotonic_increasing and dates.is_unique, "日历必须唯一且递增")
    require((dates.dayofweek < 5).all(), "日历含周末开市，需另行核对")
    frame = pd.DataFrame({"execution_date": dates})
    month = dates.to_period("M")
    ordinal = frame.groupby(month, sort=False).cumcount().to_numpy() + 1
    gap = pd.Series(dates).diff().dt.days.fillna(1).to_numpy(int)
    since_break = np.full(len(dates), 1000000, dtype=int)
    last_break = -1000000
    for index in range(len(dates)):
        if gap[index] >= config["long_break_minimum_date_gap_days"]:
            last_break = index
        since_break[index] = index - last_break + 1
    remaining = dates.days_in_month - dates.day
    for number, name in enumerate(CALENDAR_COLUMNS[:5]):
        frame[name] = (dates.dayofweek == number).astype(float)
    phase = 2 * np.pi * (dates.day - 1) / dates.days_in_month
    year_phase = 2 * np.pi * (dates.month - 1) / 12
    frame["month_phase_sin"], frame["month_phase_cos"] = np.sin(phase), np.cos(phase)
    frame["year_phase_sin"], frame["year_phase_cos"] = np.sin(year_phase), np.cos(year_phase)
    frame["month_ordinal_scaled"] = np.minimum(ordinal / 22, 1)
    frame["remaining_calendar_days_scaled"] = remaining / 31
    frame["month_first3"] = (ordinal <= config["calendar_first_month_sessions"]).astype(float)
    frame["month_last5_calendar"] = (remaining < config["calendar_month_end_natural_days"]).astype(float)
    frame["quarter_first3"] = (np.isin(dates.month, [1, 4, 7, 10]) & (frame.month_first3 == 1)).astype(float)
    frame["quarter_last5_calendar"] = (np.isin(dates.month, [3, 6, 9, 12]) & (frame.month_last5_calendar == 1)).astype(float)
    frame["reopen_gap_log"] = np.log1p(gap)
    frame["post_break_first3"] = (since_break <= config["post_break_sessions"]).astype(float)
    frame["month_edge"] = ((frame.month_first3 == 1) | (frame.month_last5_calendar == 1)).astype(float)
    frame["month_session_ordinal"] = ordinal
    frame["previous_session_gap_days"] = gap
    frame["sessions_since_last_long_break"] = since_break
    frame["decision_time"] = dates + pd.Timedelta(hours=9)
    frame["last_market_session_used"] = dates
    return frame


def prepare(data: pd.DataFrame, calendar: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    features = calendar_features(dates, config)
    result = data.copy()
    result["execution_date"] = result.date.shift(-1)
    result = result.merge(features, on="execution_date", how="left", validate="many_to_one", sort=False)
    require(pd.DatetimeIndex(result.date).equals(pd.DatetimeIndex(data.date)), "日历合并改变原行情顺序")
    result["feature_valid"] &= np.isfinite(result[CALENDAR_COLUMNS]).all(axis=1)
    valid = result.execution_date.notna()
    require((result.loc[valid, "decision_time"] == result.loc[valid, "execution_date"] + pd.Timedelta(hours=9)).all(), "日历信息未对齐当天九点决策")
    require((result.loc[valid, "last_market_session_used"] <= result.loc[valid, "execution_date"]).all(), "使用了决策日后的市场日历")
    return result, features


def rule_targets(data: pd.DataFrame) -> dict[str, np.ndarray]:
    targets = {"K1_PRIMARY_MONTH_START3": data.month_first3.to_numpy(float),
               "K2_MONTH_EDGE": data.month_edge.to_numpy(float),
               "K3_QUARTER_START3": data.quarter_first3.to_numpy(float),
               "K4_POST_LONG_BREAK3": data.post_break_first3.to_numpy(float)}
    targets["K5_EQUAL_EVENT_VOTE"] = np.mean(np.column_stack(list(targets.values())), axis=1)
    return targets


def train_all(data: pd.DataFrame, labels: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    price_columns = json.loads((PARENT / "input_receipt.json").read_text(encoding="utf-8"))["features"]
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0]) - 1
    quarter = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [first] + [t for t in range(first + 1, len(data) - 1) if quarter[t] != quarter[t - 1]] + [len(data) - 1]
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(labels.date)), "成熟收益与因子日期不一致")
    y, predictions, receipts = labels.Y1.to_numpy(float), {}, []
    (OUT / "final_models").mkdir(exist_ok=True)
    for number, item in enumerate(config["models"], 1):
        columns = (price_columns if item["group"] in ("PRICE", "FULL") else []) + (CALENDAR_COLUMNS if item["group"] in ("CALENDAR", "FULL") else [])
        x, prediction = data[columns].to_numpy(float), np.full(len(data), np.nan)
        for t, end in zip(cuts[:-1], cuts[1:]):
            train = eligible_training(data, t, 1, None)
            require(len(train) >= config["minimum_train_samples"] and np.isfinite(x[train]).all() and np.isfinite(y[train]).all(), "有效成熟训练样本不足")
            pred_ix = np.arange(t, end)[data.feature_valid.iloc[t:end].to_numpy()]
            seed = config["random_seed"] + int(data.date.iloc[t].strftime("%Y%m%d"))
            model = model_for(item["kind"], seed)
            with threadpool_limits(limits=1):
                model.fit(x[train], y[train])
                if len(pred_ix):
                    prediction[pred_ix] = model.predict(x[pred_ix])
            receipts.append({"model": item["id"], "group": item["group"], "fit_origin": data.date.iloc[t],
                             "train_rows": len(train), "first_train_origin": data.date.iloc[train[0]],
                             "last_train_origin": data.date.iloc[train[-1]], "last_label_exit": data.date.iloc[train[-1] + 2],
                             "last_prediction_origin": data.date.iloc[end - 1], "random_seed": seed,
                             "training_origin_sha256": hashlib.sha256(train.astype(np.int64).tobytes()).hexdigest(),
                             "training_array_sha256": hashlib.sha256(x[train].tobytes() + y[train].tobytes()).hexdigest()})
        predictions[item["id"]] = prediction
        joblib.dump({"model": model, "features": columns, "specification": item, "last_fit_receipt": receipts[-1]},
                     OUT / "final_models" / f"{item['id']}.joblib", compress=3)
        print(f"日历条件模型已完成 {number}/{len(config['models'])}：{label(item['id'])}", flush=True)
    for key, group in ENSEMBLES.items():
        predictions[key] = np.mean(np.column_stack([predictions[f"{group}_{kind}_H1"] for kind in ("RIDGE", "ET")]), axis=1)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "predictions.parquet", index=False)
    return predictions


def simulate_policy(data: pd.DataFrame, dividends: pd.DataFrame, config: dict, cost: dict,
                    model_id: str, targets: np.ndarray | None = None,
                    prediction: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    values = {name: data[name].to_numpy() for name in ("open", "close", "previous_close", "dividend", "variance60", "feature_valid")}
    events, account = dividends.to_dict("records"), Account(config["initial_capital"])
    previous_nav, previous_mark = config["initial_capital"], float(values["close"][anchor])
    records, decisions = [], []

    def decide(t: int) -> dict:
        close = float(values["close"][t])
        if model_id == "BUY_HOLD":
            quantity = affordable_quantity(account.cash, fill_price(close, 1, cost, config["tick"]), cost, config["lot"]) if t == anchor else 0
            decision = {"requested_quantity": quantity, "reference_weight": 1.0, "action": "买入持有", "view": "有效"}
        elif not values["feature_valid"][t] or (targets is not None and not np.isfinite(targets[t])) or (prediction is not None and not np.isfinite(prediction[t])):
            decision = {"requested_quantity": 0, "reference_weight": None, "action": "无有效判断，保持已有份额", "view": "NO_VIEW"}
        elif targets is not None:
            decision = {**target_request(account, close, float(targets[t]), config), "view": "有效"}
        elif prediction is not None:
            decision = {**choose_order(account, close, float(prediction[t]), float(values["variance60"][t]), cost, config), "view": "有效"}
        else:
            raise ValueError("缺少预先登记的规则或预测")
        decisions.append({"origin": dates[t], "decision_time": dates[t + 1] + pd.Timedelta(hours=9),
                          "execution_date": dates[t + 1], "simulation_only": True, **decision})
        return decision

    pending = decide(anchor)
    for day in range(first, last + 1):
        date, op, cl = dates[day], float(values["open"][day]), float(values["close"][day])
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for key, event in enumerate(events):
            if event["ex_date"] == date:
                value = account.entitlements.get(key, 0) * event["cash_dividend_per_share"]
                account.receivables[key] = value
                recognized += value
            if event["payment_date"] < date and key in account.receivables:
                value = account.receivables.pop(key)
                account.cash += value
                paid += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        pretrade = account.value(op)
        execution = execute_order(account, quantity, op, float(values["previous_close"][day]), float(values["dividend"][day]), day, cost, config)
        mark = op if terminal else cl
        if not terminal:
            for key, event in enumerate(events):
                if event["payment_date"] == date and key in account.receivables:
                    value = account.receivables.pop(key)
                    account.cash += value
                    paid += value
                if event["record_date"] == date:
                    account.entitlements[key] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"日历账户财富不守恒：{model_id}，{date}")
        account.assert_valid()
        records.append({"date": date, "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / pretrade, "view_at_order_origin": pending["view"], **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)


def label(key: str) -> str:
    if key in RULES:
        return RULES[key]
    names = {"K6_CALENDAR_CONSENSUS": "日历信息双模型共识", "K7_PRICE_CONSENSUS": "价格信息双模型共识",
             "K8_FULL_CONSENSUS": "日历加价格双模型共识", "BUY_HOLD": "买入持有"}
    if key in names:
        return names[key]
    group, kind, _ = key.split("_")
    return f"{dict(CALENDAR='日历信息', PRICE='价格信息', FULL='日历加价格')[group]}、{'岭回归' if kind == 'RIDGE' else '极随机树'}"


def run(expected: str) -> None:
    require(digest(MANIFEST) == expected, "第六轮冻结清单哈希不符")
    for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        path = ROOT / item["path"]
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], f"冻结文件变化：{item['path']}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    data, calendar = prepare(pd.read_parquet(PARENT / "features.parquet"), pd.read_parquet(ROOT / config["inputs"]["calendar"]), config)
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    labels = pd.read_parquet(ROOT / "reports/research/510300_overnight_global_information_v1/labels.parquet")
    data.to_parquet(OUT / "features.parquet", index=False)
    calendar.to_parquet(OUT / "date_only_calendar_features.parquet", index=False)
    predictions, targets = train_all(data, labels, config), rule_targets(data)
    pd.DataFrame({"date": data.date, **targets}).to_parquet(OUT / "rule_targets.parquet", index=False)
    valid = (data.execution_date >= config["evaluation_start"]) & (data.execution_date < config["data_cutoff"])
    write_json(OUT / "feature_receipt.json", {"status": "PASS_DATE_ONLY_CAUSAL_CALENDAR_FEATURES", "calendar_features": CALENDAR_COLUMNS,
               "valid_evaluation_origins": int(data.loc[valid, "feature_valid"].sum()),
               "rule_positive_target_days": {key: int((value[valid.to_numpy()] > 0).sum()) for key, value in targets.items()},
               "future_session_count_used": False, "historical_calendar_first_delivery_proven": False})
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for cost_id, cost in config["costs"].items():
        ledgers = {}
        for key in list(targets) + list(predictions) + ["BUY_HOLD"]:
            ledger, decisions = simulate_policy(data, dividends, config, cost, key, targets.get(key), predictions.get(key))
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            ledgers[key] = ledger
        baseline = summarize(ledgers["BUY_HOLD"], config)
        for key, ledger in ledgers.items():
            item = {"cost": cost_id, "model": key, **summarize(ledger, config)}
            item["annualized_return_excess_vs_buy_hold"] = item["annualized_return"] - baseline["annualized_return"]
            item["meets_point_target"] = item["net_sharpe"] is not None and item["net_sharpe"] >= 1.2
            metrics.append(item)
            for year, part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(part, config)})
            for name, start, end in (("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                                     ("2024—终点", "2024-01-01", config["data_cutoff"])):
                part = ledger.loc[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": key, "era": name, **summarize(part, config)})
        primary = ledgers[config["primary"]].net_return.to_numpy()
        benchmark = ledgers["BUY_HOLD"].net_return.to_numpy()
        increment = ledgers["K8_FULL_CONSENSUS"].net_return.to_numpy() - ledgers["K7_PRICE_CONSENSUS"].net_return.to_numpy()
        samples = {"primary_sharpe": [], "primary_annualized_arithmetic_excess_vs_buy_hold": [],
                   "full_vs_price_annualized_arithmetic_increment": []}
        rng = np.random.default_rng(config["random_seed"])
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(primary), config["bootstrap_day_block"])
            sh = return_metrics(primary[ix], config["annual_days"])["net_sharpe"]
            samples["primary_sharpe"].append(np.nan if sh is None else sh)
            samples["primary_annualized_arithmetic_excess_vs_buy_hold"].append(float((primary[ix] - benchmark[ix]).mean() * 242))
            samples["full_vs_price_annualized_arithmetic_increment"].append(float(increment[ix].mean() * 242))
        uncertainty[cost_id] = {**{key + "_95_interval": interval(value) for key, value in samples.items()}, "multiple_search_adjusted": False}
        print(f"第六轮 {cost_id} 的 15 个完整评价账户已完成", flush=True)
    diagnostics = {}
    for key in config["diagnostic_only_zero_cost_models"]:
        ledger, decisions = simulate_policy(data, dividends, config, {"commission": 0.0, "minimum": 0.0, "slippage": 0.0}, key, targets.get(key), predictions.get(key))
        save_account(OUT / "diagnostic/ZERO_COST", key, ledger, decisions)
        diagnostics[key] = summarize(ledger, config)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    primary = frame.loc[frame.model == config["primary"]]
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    reached = bool(primary.meets_point_target.all())
    both = frame.pivot(index="model", columns="cost", values="meets_point_target").all(axis=1)
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(),
              "number_of_candidates": 14, "benchmark_count": 1, "evaluation_accounts": 30, "diagnostic_accounts": 2,
              "trained_model_count": 6, "quarterly_fits": len(pd.read_csv(OUT / "training_receipts.csv")),
              "primary_point_target_in_both_costs": reached, "any_candidate_point_target_in_both_costs": both[both].index.tolist(),
              "zero_cost_diagnostics": diagnostics, "uncertainty": uncertainty,
              "historical_calendar_first_delivery_proven": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
              "position_impact": 0, "validated_live_strategy": "NONE"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    lines = ["# 第六轮：日历资金节奏的实际结果", "",
             f"预先指定的月初三日主方案，基础夏普 **{primary.iloc[0].net_sharpe:.4f}**，压力夏普 **{primary.iloc[1].net_sharpe:.4f}**。目标为一点二。", "",
             "共五个确定规则、六个底层预测模型、三个双模型共识，十四个登记候选。所有候选、两种费用和买入持有基准均保留。", "",
             "|费用|策略中文名称|夏普率|年化收益|最大回撤|成交次数|", "|---|---|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"|{'基础' if row.cost == 'BASE' else '压力'}|{label(row.model)}|{row.net_sharpe:.4f}|{row.annualized_return:.2%}|{row.max_drawdown:.2%}|{row.trade_count}|")
    lines += ["", f"事后最好候选为“{label(best.model)}”，基础夏普 {best.net_sharpe:.4f}。该结果不替换主方案。", "",
              "日历因子只使用决策当天日期和此前已发生交易日。月末条件用自然日计算，没有读取本月未来实际交易日总数或未来休市长度。旧个股同月收益排序的失败结论保留。"]
    for key, value in diagnostics.items():
        lines += ["", f"“{label(key)}”零费用诊断夏普 {value['net_sharpe']:.4f}，只作费用归因，不用于验收。"]
    (OUT / "第六轮日历条件_全部实际结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日历资金节奏研究：先冻结后实际执行")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        require(bool(args.manifest_sha256), "执行必须提供冻结清单身份")
        run(args.manifest_sha256)
    else:
        parser.error("请指定冻结或执行")
