"""使用已保存的正确来源账户解释持仓、预测误差和损益集中，不重跑策略。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity
from research.adaptive_allocation_v1 import summarize
from research.forward_eps_monthly_policy_v1 import mature_training

OUT = ROOT / "reports/research/510300_forward_eps_corrected_account_attribution_v1_1"
R11 = "510300_forward_eps_monthly_policy_v2_csi"
R15 = "510300_forward_eps_utility_exit_v1"
R16 = "510300_forward_eps_revision_distribution_policy_v1"
R17 = "510300_forward_eps_report_valuation_policy_v1"
R18 = "510300_forward_eps_optional_valuation_v1"
ACCOUNTS = {
    "EPS": (R11, "E3_FORWARD_EPS", "原前瞻EPS三因子"),
    "PRICE_EPS": (R11, "E1_PRICE_FORWARD_EPS", "价格加前瞻EPS"),
    "PRICE": (R11, "E2_MATCHED_PRICE", "共同月份价格对照"),
    "DAILY_TREND_EXIT": (R15, "Z2_FORECAST_EXPIRY_AND_TREND", "原EPS加期限与每日趋势退出"),
    "REVISION_DISTRIBUTION": (R16, "D1_EPS_DISTRIBUTION", "EPS加利润修正分布"),
    "VALUATION": (R17, "V1_EPS_REPORTED_VALUATION", "EPS加研报估值空间"),
    "MATCHED_EPS": (R17, "V2_MATCHED_EPS", "估值共同月份EPS对照"),
    "OPTIONAL_VALUATION": (R18, "O1_EPS_OPTIONAL_VALUATION", "原EPS加可选估值修正"),
    "BUY_HOLD": (R11, "BUY_HOLD", "买入持有"),
}
PAIRS = [("EPS", "BUY_HOLD"), ("PRICE_EPS", "EPS"), ("PRICE_EPS", "PRICE"),
         ("DAILY_TREND_EXIT", "EPS"), ("REVISION_DISTRIBUTION", "EPS"),
         ("VALUATION", "MATCHED_EPS"), ("OPTIONAL_VALUATION", "EPS")]
COMPONENTS = ["mean_size", "time_variation", "dividend_entitlement", "cost", "net"]


def decompose(ledger: pd.DataFrame, market: pd.DataFrame, initial: float, start: pd.Timestamp) -> pd.DataFrame:
    before = ledger.equity.shift(1).fillna(initial)
    keep = ledger.date.ge(start)
    selected = ledger.loc[keep].copy()
    previous_nav = before.loc[keep].to_numpy(float)
    previous_close = market.close.shift(1).loc[selected.date].to_numpy(float)
    ex = market.dividend.loc[selected.date].to_numpy(float)
    op = selected.open.to_numpy(float)
    mark = selected.mark.to_numpy(float)
    q_on = selected.shares_before.to_numpy(float)
    q_day = selected.shares.to_numpy(float)
    w_on = q_on * previous_close / previous_nav
    w_day = q_day * op / previous_nav
    r_on = (op - previous_close + ex) / previous_close
    r_day = (mark - op) / op
    values = pd.DataFrame({"date": selected.date.to_numpy(), "previous_nav": previous_nav,
        "q_on": q_on, "q_day": q_day, "w_on": w_on, "w_day": w_day,
        "r_on": r_on, "r_day": r_day, "unit_gain_on": op - previous_close + ex,
        "unit_gain_day": mark - op,
        "dividend_residual_cny": selected.dividend_recognized.to_numpy() - q_on * ex,
        "transaction_cost_paid_cny": selected.commission.to_numpy() + selected.slippage_cost.to_numpy()})
    for suffix, left, right, gain_left, gain_right, dividend, friction in [
        ("return", w_on, w_day, r_on, r_day, values.dividend_residual_cny.to_numpy() / previous_nav, values.transaction_cost_paid_cny.to_numpy() / previous_nav),
        ("cny", q_on, q_day, values.unit_gain_on.to_numpy(), values.unit_gain_day.to_numpy(), values.dividend_residual_cny.to_numpy(), values.transaction_cost_paid_cny.to_numpy()),
    ]:
        values["mean_size_" + suffix] = left.mean() * gain_left + right.mean() * gain_right
        values["time_variation_" + suffix] = (left - left.mean()) * gain_left + (right - right.mean()) * gain_right
        values["dividend_entitlement_" + suffix] = dividend
        values["cost_" + suffix] = -friction
        values["net_" + suffix] = values[[x + "_" + suffix for x in COMPONENTS[:-1]]].sum(axis=1)
    np.testing.assert_allclose(values.net_return, selected.net_return.to_numpy(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(values.net_cny, selected.pnl.to_numpy(), atol=1e-7, rtol=0)
    return values


def forecast_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / "reports/research" / R11
    monthly = pd.read_parquet(base / "monthly_features_and_mature_labels.parquet")
    signals = pd.read_parquet(base / "signals.parquet")
    records = []
    for _, origin in monthly.iterrows():
        location = int(origin.origin_index)
        if not bool(signals.event_mask.iloc[location]):
            continue
        train = mature_training(monthly, origin.origin)
        for name in ["E1_PRICE_FORWARD_EPS", "E2_MATCHED_PRICE", "E3_FORWARD_EPS"]:
            prediction = signals.loc[location, name]
            if not np.isfinite(prediction):
                continue
            record = {"model": name, "origin": origin.origin, "forecast": float(prediction),
                "actual_y60": float(origin.Y60) if np.isfinite(origin.Y60) else np.nan,
                "label_exit_date": origin.label_exit_date, "training_months": len(train),
                "training_mean_y60": float(train.Y60.mean()),
                "trend_above_120_day_mean": bool(origin.sma120 > 0), "calendar_target_roll_month": origin.origin.month == 1,
                "label_overlap_warning": "月度预测的60交易日标签存在重叠，不按独立样本解释"}
            if np.isfinite(record["actual_y60"]):
                record.update({"forecast_error": prediction - record["actual_y60"],
                    "squared_error": (prediction - record["actual_y60"]) ** 2,
                    "training_mean_squared_error": (record["training_mean_y60"] - record["actual_y60"]) ** 2,
                    "direction_correct": bool(np.sign(prediction) == np.sign(record["actual_y60"]))})
            records.append(record)
    facts = pd.DataFrame(records)
    summaries = []
    known = facts.loc[facts.actual_y60.notna()]
    for model, group in known.groupby("model", sort=True):
        for label, part in [("全部已兑现预测", group),
                            ("判断时位于120日均线之上", group.loc[group.trend_above_120_day_mean]),
                            ("判断时不高于120日均线", group.loc[~group.trend_above_120_day_mean]),
                            ("一月预测目标年度切换", group.loc[group.calendar_target_roll_month]),
                            ("其他月份", group.loc[~group.calendar_target_roll_month])]:
            summaries.append({"model": model, "group": label, "known_forecasts": len(part),
                "positive_forecasts": int(part.forecast.gt(0).sum()), "positive_actuals": int(part.actual_y60.gt(0).sum()),
                "direction_correct_count": int(part.direction_correct.sum()),
                "mean_absolute_error": float(part.forecast_error.abs().mean()) if len(part) else None,
                "mean_squared_error": float(part.squared_error.mean()) if len(part) else None,
                "training_mean_baseline_squared_error": float(part.training_mean_squared_error.mean()) if len(part) else None,
                "mean_predicted_return": float(part.forecast.mean()) if len(part) else None,
                "mean_actual_return": float(part.actual_y60.mean()) if len(part) else None})
    return facts, pd.DataFrame(summaries)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "config/510300_forward_eps_monthly_policy_v2.json"
    config = read(config_path)
    price_path = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    inputs = [config_path, price_path, Path(__file__),
        ROOT / "reports/research" / R11 / "signals.parquet",
        ROOT / "reports/research" / R11 / "monthly_features_and_mature_labels.parquet"]
    inputs.extend(ROOT / "reports/research/510300_forward_eps_corrected_account_attribution_v1" / name for name in ["result.json", "protocol.json", "verification_issue.json"])
    inputs.extend(ROOT / "reports/research" / folder / "evaluation" / cost / (name + "_ledger.parquet")
                  for folder, name, _ in ACCOUNTS.values() for cost in config["costs"])
    for folder in sorted({x[0] for x in ACCOUNTS.values()}):
        inputs.extend(ROOT / "reports/research" / folder / file for file in ["result.json", "saved_numerical_verification.json"])
    save(OUT / "protocol.json", {"recorded_at": now(), "source_returns_already_observed": True,
        "new_strategy_returns_generated": False, "methods_registered": 0,
        "descriptive_attribution_not_new_tradable_strategy": True, "costs": list(config["costs"]),
        "accounts": ACCOUNTS, "pairs": PAIRS, "fixed_periods": ["完整原评价期间", "原EPS第一次可用预测的下一交易日起"],
        "components": COMPONENTS, "no_new_bootstrap_or_significance_claim": True,
        "source_files": [identity(p) for p in inputs]}, exclusive=True)
    data = pd.read_parquet(price_path)
    market = data.set_index("date")
    signals = pd.read_parquet(ROOT / "reports/research" / R11 / "signals.parquet")
    first_index = int(np.flatnonzero(np.isfinite(signals.E3_FORWARD_EPS))[0])
    common_start = data.date.iloc[first_index + 1]
    periods = {"FULL_EVALUATION": pd.Timestamp(config["evaluation_start"]), "AFTER_FIRST_EPS_MODEL": common_start}
    summaries, annual, trades, pairs, concentration = [], [], [], [], []
    decomposed = {}
    maximum_rate_error = maximum_cny_error = 0.0
    for cost in config["costs"]:
        for key, (folder, name, chinese) in ACCOUNTS.items():
            ledger = pd.read_parquet(ROOT / "reports/research" / folder / "evaluation" / cost / (name + "_ledger.parquet"))
            assert ledger.date.iloc[0] == pd.Timestamp(config["evaluation_start"])
            assert ledger.date.iloc[-1] == pd.Timestamp(config["data_cutoff"])
            for period, start in periods.items():
                part = ledger.loc[ledger.date.ge(start)]
                frame = decompose(ledger, market, config["initial_capital"], start)
                decomposed[(cost, period, key)] = frame
                (OUT / "decomposition_ledgers").mkdir(exist_ok=True)
                frame.to_parquet(OUT / "decomposition_ledgers" / f"{cost}_{period}_{key}.parquet", index=False)
                summary = {"cost_scenario": cost, "period": period, "model": key, "策略": chinese,
                    "days": len(frame), "start": str(frame.date.min().date()), "end": str(frame.date.max().date()),
                    "mean_overnight_weight": float(frame.w_on.mean()), "mean_intraday_weight": float(frame.w_day.mean()),
                    "diagnostic_window_only_not_new_goal_evaluation": period != "FULL_EVALUATION",
                    **summarize(part, config)}
                for component in COMPONENTS:
                    summary[component + "_annual_arithmetic"] = float(frame[component + "_return"].mean() * config["annual_days"])
                    summary[component + "_cumulative_cny"] = float(frame[component + "_cny"].sum())
                summaries.append(summary)
                maximum_rate_error = max(maximum_rate_error, float(np.max(np.abs(frame.net_return.to_numpy() - part.net_return.to_numpy()))))
                maximum_cny_error = max(maximum_cny_error, float(np.max(np.abs(frame.net_cny.to_numpy() - part.pnl.to_numpy()))))
            for year, part in ledger.groupby(ledger.date.dt.year):
                annual.append({"cost_scenario": cost, "model": key, "策略": chinese, "year": int(year),
                    "pnl_cny": float(part.pnl.sum()), "holding_days": int(part.shares.gt(0).sum()),
                    "cash_only_days": int((part.shares.eq(0) & part.shares_before.eq(0)).sum()), **summarize(part, config)})
            for row in ledger.loc[ledger.filled_quantity.ne(0)].to_dict("records"):
                if row["mark_clock"] == "OPEN_TERMINAL":
                    action = "统一终点清算"
                elif row["filled_quantity"] > 0:
                    action = "进入" if row["shares_before"] == 0 else "增加"
                else:
                    action = "退出" if row["shares"] == 0 else "减少"
                trades.append({"cost_scenario": cost, "model": key, "策略": chinese, "动作": action, **row})
            positive = ledger.loc[ledger.pnl.gt(0), "pnl"].sort_values(ascending=False)
            concentration.append({"cost_scenario": cost, "model": key, "策略": chinese,
                "full_pnl_cny": float(ledger.pnl.sum()), "sum_positive_daily_pnl": float(positive.sum()),
                "five_largest_positive_daily_pnl": float(positive.head(5).sum()),
                "largest_five_share_of_positive_pnl": float(positive.head(5).sum() / positive.sum()) if positive.sum() else None,
                "post_observation_diagnostic_not_dates_removed": True})
        for period in periods:
            for left, right in PAIRS:
                a = decomposed[(cost, period, left)]
                b = decomposed[(cost, period, right)]
                assert a.date.tolist() == b.date.tolist()
                values = {component: float((a[component + "_return"] - b[component + "_return"]).mean() * config["annual_days"]) for component in COMPONENTS}
                np.testing.assert_allclose(sum(values[x] for x in COMPONENTS[:-1]), values["net"], atol=1e-12, rtol=0)
                pairs.append({"cost_scenario": cost, "period": period, "left": left, "right": right, "days": len(a), **values})
    forecasts, forecast_summary = forecast_diagnostic()
    outputs = {"账户规模与时变损益分解.csv": pd.DataFrame(summaries), "同期间策略增量分解.csv": pd.DataFrame(pairs),
        "逐年持仓和损益.csv": pd.DataFrame(annual), "已有账户全部进出场.csv": pd.DataFrame(trades),
        "盈利集中程度.csv": pd.DataFrame(concentration), "已有预测与兑现结果.csv": forecasts,
        "不同市场状态下预测误差.csv": forecast_summary}
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": "510300_FORWARD_EPS_CORRECTED_ACCOUNT_ATTRIBUTION_V1_1", "completed_at": now(),
        "status": "SAVED_CORRECTED_SOURCE_ACCOUNT_DIAGNOSTIC_COMPLETE_NO_NEW_STRATEGY",
        "source_accounts": len(ACCOUNTS) * len(config["costs"]), "account_period_decompositions": len(summaries),
        "paired_decompositions": len(pairs), "first_eps_prediction": str(data.date.iloc[first_index].date()),
        "first_eps_model_execution_date": str(common_start.date()), "annual_records": len(annual),
        "saved_filled_trades": len(trades), "saved_forecast_records": len(forecasts),
        "maximum_daily_return_reconstruction_error": maximum_rate_error,
        "maximum_daily_cny_reconstruction_error": maximum_cny_error,
        "new_models_fit": 0, "new_accounts_generated": 0, "new_strategy_configurations": 0,
        "new_return_labels_generated": 0, "new_downloads": 0, "new_random_samples": 0,
        "goal_evaluation_window_changed": False, "goal_achieved": False,
        "descriptive_diagnostic_not_causal_or_independent_alpha_proof": True,
        "pairs": pairs, "forecast_summary": forecast_summary.to_dict("records")}
    save(OUT / "result.json", result, exclusive=True)
    print("完成已有18账户、36账户期间的损益拆分；没有新增模型或账户。", flush=True)
    print(pd.DataFrame(pairs).query("cost_scenario == 'BASE' and period == 'AFTER_FIRST_EPS_MODEL'").to_string(index=False), flush=True)
    print(forecast_summary.loc[forecast_summary.group.eq("全部已兑现预测")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
