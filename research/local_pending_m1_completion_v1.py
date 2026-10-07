"""用已有9月16日行情补齐待观察事件的保存模型输出，不拟合、不运行账户。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_local_pending_m1_completion_v1"
PARENT = "reports/research/510300_new_m1_consensus_training_v1"
LATEST = "reports/research/510300_fixed_daily_continuation_v1/2026-09-16"
SOURCES = {
    "mandate.json": "config/510300_existing_data_training_mandate_v1.json",
    "parent_protocol.json": PARENT + "/protocol.json",
    "parent_summary.json": PARENT + "/summary.json",
    "parent_branch_decision.json": PARENT + "/branch_decision.json",
    "pending.csv": PARENT + "/results/待行情事件.csv",
    "mature_event_features.csv": PARENT + "/results/事件特征与五日标签.csv",
    "saved_final_models.json": PARENT + "/models/期末已训练模型.json",
    "parent_delivery_receipt.json": PARENT + "/delivery_receipt.json",
    "common_market.parquet": "reports/research/510300_private_manager_intent_v1/inputs/market.parquet",
    "latest_prices.parquet": LATEST + "/candidate_prices.parquet",
    "latest_admission.json": LATEST + "/admission_receipt.json",
    "latest_dividend_coverage.json": LATEST + "/candidate_dividend_coverage.json",
    "latest_completed.json": "reports/research/510300_fixed_daily_continuation_v1/latest_completed.json",
    "dividends.csv": "data/reference/510300_dividends.csv",
    "calendar.csv": "data/reference/sse_trade_calendar_2026.csv",
    "calendar_metadata.json": "data/reference/sse_trade_calendar_2026.metadata.json",
    "recent_target_reassessment.json": "reports/research/510300_sparse_sharpe_support_bound_v2/summary.json",
    "historical_target_reassessment.json": "reports/research/510300_saved_candidates_sharpe13_v1/target_change_result.json",
    "collector_pause.json": "reports/research/510300_forward_host_diagnosis_20260922/user_pause_receipt.json",
    "absorption_protocol.json": "reports/research/510300_absorption_available_members_training_v1/protocol.json",
    "absorption_monthly.csv": "reports/research/510300_absorption_available_members_training_v1/results/月末训练特征与标签.csv",
    "if_protocol.json": "reports/research/510300_if_exhaustion_existing_training_v1/protocol.json",
    "if_summary.json": "reports/research/510300_if_exhaustion_existing_training_v1/summary.json",
    "lpr_summary.json": "reports/research/510300_lpr_expectation_exploratory_training_v1/summary.json",
    "private_summary.json": "reports/research/510300_private_manager_exploratory_training_v1/summary.json",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def freeze(out):
    require(not (out / "freeze.json").exists(), "已保存本轮输入，不覆盖。")
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        shutil.copy2(ROOT / source, out / "inputs" / name)
    (out / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), out / "code" / Path(__file__).name)
    mandate = load(out / "inputs/mandate.json")
    require(mandate["target_net_sharpe"] == 1.3 and not mandate["new_market_data_collection_enabled"], "用户目标或不采集要求变化。")
    protocol = {"study_id": "510300_LOCAL_PENDING_M1_COMPLETION_V1", "frozen_at": now(),
        "user_goal_net_sharpe": mandate["target_net_sharpe"], "capital_cny": mandate["capital_cny"],
        "target_max_drawdown": mandate["target_max_drawdown"], "executable_assets": mandate["executable_assets"],
        "scope": "发现本地9月16日行情后，重建原待行情事件特征并代入原保存模型；不拟合模型或回测账户。",
        "evidence_class": "RETROSPECTIVE_SAVED_MODEL_OUTPUT_NOT_PRECOMMITTED_FORECAST",
        "parent_model_creation_after_historical_decision": True,
        "parent_terminal_decision_preserved": True, "new_model_fits": 0, "new_accounts": 0,
        "new_downloads": 0, "orders_authorized": False, "goal_achieved": False}
    save(out / "protocol.json", protocol)
    paths = [out / "protocol.json", *sorted((out / "inputs").iterdir()), *sorted((out / "code").iterdir())]
    save(out / "freeze.json", {"frozen_at": now(), "source_paths": SOURCES,
        "identities": {p.relative_to(out).as_posix(): digest(p) for p in paths if p.is_file()}})
    print("保存模型、待处理事件、本地行情及日历已固定。")


def check_inputs(out):
    for name, expected in load(out / "freeze.json")["identities"].items():
        require(digest(out / name) == expected, f"输入已变化：{name}")


def build_market(out):
    common = pd.read_parquet(out / "inputs/common_market.parquet").copy()
    prices = pd.read_parquet(out / "inputs/latest_prices.parquet").copy()
    for frame in [common, prices]:
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        require(not frame.date.duplicated().any() and frame.date.is_monotonic_increasing, "行情日期重复或无序。")
    old_end = common.date.max()
    prefix = prices[prices.date <= old_end].reset_index(drop=True)
    require(prefix.date.equals(common.date.reset_index(drop=True)), "两份行情旧日期前缀不同。")
    maximum_difference = float(np.max(np.abs(prefix[["open", "high", "low", "close"]].to_numpy() - common[["open", "high", "low", "close"]].to_numpy())))
    require(maximum_difference < 1e-12, "旧行情价差需要先解决，不能直接接入。")
    require(load(out / "inputs/latest_admission.json")["dividend_coverage_through"] >= str(prices.date.max().date()), "新增日期分红覆盖不足。")
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends["ex_date"] = pd.to_datetime(dividends.ex_date)
    dividend_by_date = dividends.groupby("ex_date").cash_dividend_per_share.sum().to_dict()
    tail = prices[prices.date > old_end][["date", "open", "high", "low", "close"]].copy()
    tail["dividend"] = tail.date.map(dividend_by_date).fillna(0.0)
    previous_close = float(common.close.iloc[-1])
    previous_wealth = float(common.wealth.iloc[-1])
    returns, wealth = [], []
    for row in tail.itertuples():
        total_return = (row.close + row.dividend) / previous_close - 1
        previous_wealth *= 1 + total_return
        returns.append(total_return)
        wealth.append(previous_wealth)
        previous_close = row.close
    tail["total_simple"], tail["wealth"] = returns, wealth
    columns = ["date", "open", "high", "low", "close", "dividend", "total_simple", "wealth"]
    result = pd.concat([common[columns], tail[columns]], ignore_index=True)
    return result, {"common_rows": len(common), "latest_rows": len(prices), "additional_rows": len(tail),
        "common_market_end": str(old_end.date()), "latest_market_end": str(prices.date.max().date()),
        "additional_dates": tail.date.dt.strftime("%Y-%m-%d").tolist(), "old_ohlc_maximum_absolute_difference": maximum_difference}


def features_for(raw, market):
    dates = pd.DatetimeIndex(market.date).tz_localize("Asia/Shanghai")
    opens, closes = dates + pd.Timedelta(hours=9, minutes=30), dates + pd.Timedelta(hours=15)
    published = pd.Timestamp(raw["available_at_upper_bound"])
    observation_i = int(opens.searchsorted(published, side="right"))
    require(observation_i < len(market), "观察日行情仍缺失。")
    pre_i = int(closes.searchsorted(published, side="left")) - 1
    require(20 <= pre_i < observation_i, "公告前价格范围非法。")
    pre, observed = market.iloc[pre_i], market.iloc[observation_i]
    sigma = float(market.iloc[observation_i - 19:observation_i + 1].total_simple.std(ddof=1))
    forecast_upper = pd.Timestamp(raw["forecast_date"]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    require(forecast_upper < published, "预期发布时间晚于实际公布。")
    return {"stat_month": raw["stat_month"], "published_at": published.isoformat(),
        "observation_date": str(observed.date.date()), "historical_decision_at": closes[observation_i].isoformat(),
        "forecast_available_upper": forecast_upper.isoformat(), "reference_close": float(observed.close),
        "vol20": sigma, "price_pre20_return": float(pre.wealth / market.iloc[pre_i - 20].wealth - 1),
        "initial_response": float((observed.close + market.iloc[pre_i + 1:observation_i + 1].dividend.sum()) / pre.close - 1),
        "price_log_sigma20": math.log(sigma), "spread_surprise_proxy_pp": float(raw["spread_surprise_proxy_pp"])}


def predict_saved(model, features):
    x = np.asarray([features[col] for col in model["columns"]], dtype=float)
    z = (x - np.asarray(model["center"])) / np.asarray(model["scale"])
    z[~np.asarray(model["active"], dtype=bool)] = 0.0
    return float(model["intercept"] + z @ np.asarray(model["coefficients_standardized"]))


def calculate(out):
    check_inputs(out)
    market, boundary = build_market(out)
    parent_cfg = load(out / "inputs/parent_protocol.json")
    pending = pd.read_csv(out / "inputs/pending.csv")
    require(len(pending) == 1, "原待处理月份数量不同。")
    raw = pending.iloc[0].to_dict()
    features = features_for(raw, market)
    mature = pd.read_csv(out / "inputs/mature_event_features.csv")
    maximum_feature_error = 0.0
    for row in mature.to_dict("records"):
        derived = features_for(row, market)
        for key in ["reference_close", "vol20", "price_pre20_return", "initial_response", "price_log_sigma20", "spread_surprise_proxy_pp"]:
            maximum_feature_error = max(maximum_feature_error, abs(derived[key] - float(row[key])))
    require(maximum_feature_error < 1e-12, "原16事件特征未复现。")
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(out / "inputs/calendar.csv").trade_date))
    observation = pd.Timestamp(features["observation_date"])
    following = calendar[calendar > observation][:parent_cfg["horizon_sessions"]]
    require(len(following) == parent_cfg["horizon_sessions"], "日历不足完整评价期。")
    required_dates = following.strftime("%Y-%m-%d").tolist()
    available_dates = set(market.date.dt.strftime("%Y-%m-%d"))
    missing = [d for d in required_dates if d not in available_dates]
    model_created = pd.Timestamp(load(out / "inputs/parent_summary.json")["completed_at"])
    require(model_created > pd.Timestamp(features["historical_decision_at"]), "重建输出性质需重新检查。")
    features.update({"planned_entry_date": required_dates[0], "planned_exit_date": required_dates[-1],
        "target5": None, "label_status": "PENDING_COMPLETE_HORIZON", "missing_price_dates": "|".join(missing)})
    require(len(missing) > 0, "标签已成熟，本轮不应继续标记待行情。")
    final_models = load(out / "inputs/saved_final_models.json")
    rows = []
    for name, model in final_models.items():
        require(pd.Timestamp(model["maximum_training_label_end_at"]) < pd.Timestamp(features["historical_decision_at"]), "训练标签晚于历史决策时点。")
        require(model["n_train"] == 16 and len(model["train_months"]) == 16, "保存模型训练范围变化。")
        rows.append({"model": name, "stat_month": raw["stat_month"],
            "historical_decision_at": features["historical_decision_at"], "parent_model_created_at": model_created.isoformat(),
            "prediction_gross_5d_return": predict_saved(model, features),
            "n_train": model["n_train"], "maximum_training_label_end_at": model["maximum_training_label_end_at"],
            "entry_date": required_dates[0], "exit_date": required_dates[-1],
            "actual_return": None, "label_status": "PENDING_COMPLETE_HORIZON",
            "evidence_class": "RETROSPECTIVE_RECONSTRUCTION_NOT_PRECOMMITTED_FORECAST"})
    require(len(rows) == 3, "保存模型数量变化。")
    observed_only = market[market.date <= observation].copy()
    trimmed = features_for(raw, observed_only)
    for model in final_models.values():
        require(abs(predict_saved(model, features) - predict_saved(model, trimmed)) < 1e-14, "输出使用了观察日之后的价格。")
    boundary.update({"old_mature_events_reproduced": len(mature), "maximum_old_feature_error": maximum_feature_error,
        "prediction_independent_of_post_observation_rows": True,
        "pending_month": raw["stat_month"], "historical_observation_date": features["observation_date"],
        "required_horizon_dates": required_dates, "observed_horizon_dates": [d for d in required_dates if d in available_dates],
        "missing_horizon_dates": missing, "parent_models_created_at": model_created.isoformat(),
        "local_market_dates_after_model_creation": int((market.date > model_created.tz_localize(None).normalize()).sum()),
        "original_latest_continuation_model": load(out / "inputs/latest_completed.json")["all_metrics"][0]["model"],
        "legacy_continuation_observed_days_preserved": load(out / "inputs/latest_completed.json")["new_observed_trading_days"],
        "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0,
        "new_independent_forward_predictions": 0, "goal_achieved": False, "orders_authorized": False})
    return pd.DataFrame([features]), pd.DataFrame(rows), boundary


def run(out):
    require(not (out / "summary.json").exists(), "本轮已完成，不覆盖。")
    features, predictions, boundary = calculate(out)
    (out / "results").mkdir(exist_ok=True)
    features.to_csv(out / "results/补齐的历史事件特征.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    predictions.to_csv(out / "results/保存模型的历史重建输出.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    save(out / "data_boundary.json", boundary)
    save(out / "summary.json", {"study_id": "510300_LOCAL_PENDING_M1_COMPLETION_V1", "completed_at": now(),
        "status": "COMPLETED_PENDING_FEATURE_AND_SAVED_MODEL_OUTPUT_RECONSTRUCTION",
        "current_target_net_sharpe": 1.3, "reconstructed_events": len(features), "saved_model_outputs": len(predictions),
        "old_mature_events_reproduced": boundary["old_mature_events_reproduced"],
        "label_status": "PENDING_COMPLETE_HORIZON", "missing_dates": boundary["missing_horizon_dates"],
        "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0, "new_independent_forward_predictions": 0,
        "goal_achieved": False, "orders_authorized": False,
        "previous_goal_turn_classification": "PROGRESS_FULL_EARLIER_ARCHIVE_REASSESSED_AT_1_3"})
    print(json.dumps(boundary, ensure_ascii=False, indent=2))
    print(predictions[["model", "prediction_gross_5d_return", "label_status"]].to_string(index=False))


def verify(out):
    features, predictions, boundary = calculate(out)
    for name, expected in [("补齐的历史事件特征.csv", features), ("保存模型的历史重建输出.csv", predictions)]:
        actual = pd.read_csv(out / "results" / name)
        expected = expected.replace({None: np.nan})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    require(boundary == load(out / "data_boundary.json"), "数据边界复算不同。")
    return {"status": "PASS_SAVED_MODELS_FEATURES_DATES_AND_NO_POST_OBSERVATION_DEPENDENCE",
        "old_events_reproduced": boundary["old_mature_events_reproduced"], "new_reconstructed_event": 1,
        "saved_model_outputs": len(predictions), "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0,
        "new_independent_forward_predictions": 0}


def main():
    parser = argparse.ArgumentParser(description="本地待行情事件的保存模型输出重建。")
    parser.add_argument("command", choices=["freeze", "calculate", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "calculate": run}[args.command](args.root)


if __name__ == "__main__":
    main()
