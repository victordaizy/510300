"""按历史发布版本重建 EPU 月度信息，固定预测、账户与保存材料重算。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, build_features as etf_features, commission,
    digest, execute_order, fill_price, normalize_dividends, normalize_prices,
    now, require, return_metrics, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_epu_vintage_increment_v1.json"
PROTOCOL = ROOT / "docs/510300_EPU_VINTAGE_INCREMENT_V1_PROTOCOL.md"


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def outpath(config: dict) -> Path:
    return ROOT / config["output"]


def available_clock(date, config: dict) -> pd.Timestamp:
    naive = pd.Timestamp(date).normalize() + pd.Timedelta(days=config["source_available_calendar_days_after_vintage_midnight"])
    return naive.tz_localize(config["source_timezone"]).tz_convert("Asia/Shanghai")


def parse_vintages(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.rename(columns={"period_start_date": "observation_month", config["series_id"]: "epu_value"}).copy()
    for col in ("observation_month", "realtime_start_date", "realtime_end_date"):
        frame[col] = pd.to_datetime(frame[col], errors="raise")
    require(not frame.duplicated(["observation_month", "realtime_start_date"]).any(), "版本主键重复")
    require(np.isfinite(frame.epu_value).all() and frame.epu_value.gt(0).all(), "EPU 数值非法")
    require((frame.observation_month.dt.day == 1).all(), "统计期不是月初")
    require(frame.observation_month.lt(frame.realtime_start_date).all(), "统计月与发布顺序异常")
    for _, group in frame.groupby("observation_month"):
        group = group.sort_values("realtime_start_date")
        for previous, current in zip(group.to_dict("records"), group.to_dict("records")[1:]):
            require(previous["realtime_end_date"] == current["realtime_start_date"] - pd.Timedelta(days=1), "修订区间不连续")
        require(pd.isna(group.realtime_end_date.iloc[-1]), "末版本不是当前版本")
    frame["available_at"] = frame.realtime_start_date.map(lambda x: available_clock(x, config))
    return frame.sort_values(["observation_month", "realtime_start_date"]).reset_index(drop=True)


def asof_state(vintages: pd.DataFrame, decision: pd.Timestamp) -> pd.DataFrame:
    admitted = vintages.loc[vintages.available_at.le(decision)]
    return admitted.sort_values(["observation_month", "realtime_start_date"]).groupby("observation_month", sort=True).tail(1).sort_values("observation_month")


def epu_at(vintages: pd.DataFrame, decision: pd.Timestamp, config: dict) -> dict:
    state = asof_state(vintages, decision)
    base = {"feature_status": "NO_VIEW_EPU_HISTORY_MISSING"}
    if state.empty:
        return base
    latest = state.iloc[-1]
    period = latest.observation_month.to_period("M")
    expected = pd.period_range(period - 12, period, freq="M")
    by_period = state.assign(period=state.observation_month.dt.to_period("M")).set_index("period")
    if not expected.isin(by_period.index).all():
        return base
    used = by_period.loc[expected].copy()
    values = used.epu_value.to_numpy(float)
    lag = decision.tz_localize(None).to_period("M").ordinal - period.ordinal
    evidence = [{"observation_month": x.observation_month.strftime("%Y-%m-%d"),
                 "vintage_date": x.realtime_start_date.strftime("%Y-%m-%d"), "value": float(x.epu_value)}
                for x in used.itertuples()]
    return {"feature_status": "PASS" if 1 <= lag <= config["maximum_observation_lag_months"] else "NO_VIEW_EPU_TOO_OLD",
            "epu_observation_month": latest.observation_month, "epu_latest_value": float(latest.epu_value),
            "epu_latest_value_vintage": latest.realtime_start_date,
            "epu_latest_value_available_at": latest.available_at,
            "epu_all_feature_inputs_available_at": used.available_at.max(),
            "epu_observation_lag_months": lag,
            "EPU_LOG_RELATIVE_PREVIOUS12": float(np.log(values[-1]) - np.log(values[:-1]).mean()),
            "EPU_LOG_CHANGE1": float(np.log(values[-1] / values[-2])),
            "epu_input_versions_json": json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}


def load_inputs(config: dict, *, snapshot: bool) -> tuple[dict, dict]:
    base = outpath(config) / "frozen_inputs" if snapshot else ROOT
    files = {key: base / path for key, path in config["inputs"].items()}
    receipt = json.loads(files["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(files["dividend_coverage"].read_text(encoding="utf-8"))
    require(receipt["status"] == "PASS" and digest(files["prices"]) == receipt["canonical_price"]["output_sha256"], "行情准入或版本不符")
    require(coverage["complete_history_confirmed"] and coverage["coverage_end"] >= config["data_cutoff"], "分红历史不完整")
    require(digest(files["dividends"]) == coverage["distribution_file_sha256"], "分红哈希不符")
    prices = normalize_prices(pd.read_parquet(files["prices"]))
    prices = prices.loc[prices.date.le(config["data_cutoff"])].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(files["dividends"]))
    cal = pd.read_parquet(files["calendar"])
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open, "trade_date"]))
    dates = dates[(dates >= prices.date.iloc[0]) & (dates <= pd.Timestamp(config["data_cutoff"]))].sort_values()
    require(pd.DatetimeIndex(prices.date).equals(dates) and prices.date.iloc[-1] == pd.Timestamp(config["data_cutoff"]), "价格日期与日历不符")
    source_receipt = json.loads(files["vintage_receipt"].read_text(encoding="utf-8"))
    require(source_receipt["http_status"] == 200 and source_receipt["tls_verify"], "历史版本源没有成功回执")
    require(digest(files["vintage_zip"]) == source_receipt["sha256"], "原始版本 ZIP 漂移")
    with zipfile.ZipFile(files["vintage_zip"]) as z:
        require(z.read("obs._by_real-time_period.csv") == files["vintages"].read_bytes(), "CSV 不是原始 ZIP 成员")
        require(z.read("README.txt") == files["vintage_readme"].read_bytes(), "原始说明不符")
    vintages = parse_vintages(pd.read_csv(files["vintages"]), config)
    form = json.loads(files["download_form"].read_text(encoding="utf-8"))
    listed = next(f["options"] for f in form["fields"] if f["name"] == "form[selected_vintage_dates][]")
    require(set(vintages.realtime_start_date.dt.strftime("%Y-%m-%d")) == {x["value"] for x in listed}, "下载版本集合与表单不符")
    matrix = pd.read_csv(files["crosscheck_matrix"])
    require(matrix.shape[1] == 3, "交叉验证矩阵不是两个版本")
    cross_errors = []
    for column in matrix.columns[1:]:
        date = pd.to_datetime(column.rsplit("_", 1)[1], format="%Y%m%d")
        state = vintages.loc[vintages.realtime_start_date.le(date)].groupby("observation_month").tail(1).set_index("observation_month")
        actual = matrix.set_index(pd.to_datetime(matrix.iloc[:, 0]))[column]
        actual = pd.to_numeric(actual, errors="coerce").dropna()
        require(set(actual.index) == set(state.index), "不同导出方式的可见月份不一致")
        cross_errors.extend(np.abs(actual - state.loc[actual.index, "epu_value"]).tolist())
    require(max(cross_errors) <= 0.00001, "历史区间与独立版本矩阵不一致")
    info = {"status": "PASS_ARCHIVAL_MONTHLY_VINTAGES_WITH_EXPLICIT_ARRIVAL_ASSUMPTION", "epu_rows": len(vintages),
            "epu_observation_months": vintages.observation_month.nunique(), "vintage_dates": vintages.realtime_start_date.nunique(),
            "first_vintage": vintages.realtime_start_date.min(), "last_vintage": vintages.realtime_start_date.max(),
            "revision_rows": int(vintages.realtime_end_date.notna().sum()), "crosscheck_values": len(cross_errors),
            "crosscheck_max_absolute_error": max(cross_errors), "etf_rows": len(prices), "dividend_events": len(dividends),
            "historical_first_http_delivery_proven": False, "source_provenance": {key: identity(path) for key, path in files.items()}}
    return {"prices": prices, "dividends": dividends, "vintages": vintages}, info


def price_features(data: dict) -> pd.DataFrame:
    feature = etf_features(data["prices"], data["dividends"], 20)
    for span in (20, 60):
        feature["ETF_R" + str(span)] = np.expm1(feature.total_log.rolling(span, min_periods=span).sum())
    return feature


def monthly_label(feature: pd.DataFrame, dividends: pd.DataFrame, entry: int, end: int, config: dict) -> dict:
    buy_day, sell_day = feature.date.iloc[entry], feature.date.iloc[end]
    eligible = dividends.loc[dividends.record_date.ge(buy_day) & dividends.record_date.lt(sell_day) & dividends.ex_date.le(sell_day)]
    distribution = float(eligible.cash_dividend_per_share.sum())
    cost = config["costs"]["BASE"]
    buy, sell = (fill_price(float(feature.open.iloc[i]), side, cost, config["tick"]) for i, side in ((entry, 1), (end, -1)))
    quantity = affordable_quantity(config["initial_capital"], buy, cost, config["lot"])
    require(quantity > 0, "月度标签无法买入一手")
    paid = quantity * buy + commission(quantity, buy, cost)
    received = quantity * (sell + distribution) - commission(quantity, sell, cost)
    net = (received - paid) / config["initial_capital"]
    marked = [(quantity * (float(row.close) + float(eligible.loc[eligible.ex_date.le(row.date), "cash_dividend_per_share"].sum())) - paid) / config["initial_capital"]
              for row in feature.iloc[entry:end].itertuples()]
    return {"label_status": "MATURE", "Y_NET_MONTH": net,
            "Y_GROSS_MONTH": (float(feature.open.iloc[end]) + distribution) / float(feature.open.iloc[entry]) - 1,
            "label_dividend_per_share": distribution, "label_quantity": quantity,
            "label_min_marked_return": min([0.0, net] + marked),
            "entry_gap_total_return": (float(feature.open.iloc[entry]) + float(feature.dividend.iloc[entry])) / float(feature.close.iloc[entry - 1]) - 1}


def build_samples(data: dict, feature: pd.DataFrame, config: dict, *, with_labels: bool) -> pd.DataFrame:
    first_indices = feature.groupby(feature.date.dt.to_period("M"), sort=True).head(1).index.to_list()
    rows = []
    for position, entry in enumerate(first_indices):
        date = feature.date.iloc[entry]
        if date < pd.Timestamp(config["sample_entry_start"]) or entry == 0:
            continue
        end = first_indices[position + 1] if position + 1 < len(first_indices) else None
        origin = feature.iloc[entry - 1]
        decision = (date + pd.Timedelta(hours=9)).tz_localize("Asia/Shanghai")
        row = {"entry_index": entry, "origin_index": entry - 1, "origin": origin.date, "entry_date": date,
               "exit_index": end, "exit_date": feature.date.iloc[end] if end is not None else pd.NaT,
               "decision_at": decision, "entry_month": str(date.to_period("M")),
               "holding_trading_days": end - entry if end is not None else None,
               "evaluation_origin": date >= pd.Timestamp(config["evaluation_entry_start"]),
               "label_status": "MATURE_NOT_READ" if end is not None else "CENSORED_AFTER_CUTOFF",
               "Y_NET_MONTH": np.nan, **{c: float(origin[c]) for c in config["M0"]},
               **epu_at(data["vintages"], decision, config)}
        if not np.isfinite([row[x] for x in config["M0"]]).all():
            row["feature_status"] = "NO_VIEW_ETF_FEATURES_MISSING"
        if with_labels and end is not None:
            row.update(monthly_label(feature, data["dividends"], entry, end, config))
        rows.append(row)
    return pd.DataFrame(rows)


def mature_training(samples: pd.DataFrame, origin) -> pd.DataFrame:
    return samples.loc[samples.exit_date.lt(pd.Timestamp(origin)) & samples.feature_status.eq("PASS")
                       & samples.label_status.eq("MATURE") & samples.Y_NET_MONTH.notna()]


def fit_model(train: pd.DataFrame, columns: list[str], config: dict) -> dict:
    x, y = train[columns].to_numpy(float), train.Y_NET_MONTH.to_numpy(float)
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = np.clip((x - mean) / std, -config["standardized_clip"], config["standardized_clip"])
    design = np.c_[np.ones(len(z)), z]
    penalty = np.diag([0.0] + [config["ridge_lambda"]] * len(columns))
    beta = np.linalg.solve(design.T @ design / len(y) + penalty, design.T @ y / len(y))
    return {"features": columns, "mean": mean, "std": std, "coefficients": beta, "train_rows": len(train),
            "train_first_origin": train.origin.min(), "train_last_exit": train.exit_date.max(),
            "train_origin_sha256": hashlib.sha256(train.origin_index.to_numpy(np.int64).tobytes()).hexdigest()}


def predict(row: dict, model: dict, config: dict) -> float:
    x = np.array([row[c] for c in model["features"]])
    z = np.clip((x - np.asarray(model["mean"])) / np.asarray(model["std"]), -config["standardized_clip"], config["standardized_clip"])
    return float(np.r_[1.0, z] @ np.asarray(model["coefficients"]))


def rolling_predictions(samples: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list]:
    models, records = [], []
    for item in samples.loc[samples.evaluation_origin].to_dict("records"):
        train = mature_training(samples, item["origin"])
        require(len(train) >= config["minimum_train_rows"], "成熟月度训练样本不足")
        for name, cols in {"M0": config["M0"], "M1": config["M0"] + config["M1_additions"]}.items():
            model = fit_model(train, cols, config)
            model.update(model_key=item["entry_month"] + ":" + name, model_id=name, refit_origin=item["origin"])
            models.append(model)
            item["model_key_" + name] = model["model_key"]
            item["prediction_" + name] = predict(item, model, config) if item["feature_status"] == "PASS" else np.nan
        item["prediction_status"] = item["feature_status"]
        records.append(item)
    return pd.DataFrame(records), models


def month_indices(count: int, config: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = config["bootstrap_month_block"]
    starts = rng.integers(0, count, size=(config["bootstrap_repetitions"], math.ceil(count / block)))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % count).reshape(len(starts), -1)[:, :count]


def prediction_group(frame: pd.DataFrame, name: str) -> dict:
    valid = frame.dropna(subset=["Y_NET_MONTH", "prediction_M0", "prediction_M1"])
    row = {"era": name, "calendar_months": len(frame), "paired_months": len(valid)}
    for model in ("M0", "M1"):
        y, p = valid.Y_NET_MONTH, valid["prediction_" + model]
        row[model + "_mse"] = float(((y - p) ** 2).mean()) if len(valid) else None
        row[model + "_direction_accuracy"] = float((y.gt(0) == p.gt(0)).mean()) if len(valid) else None
        for part, mask in (("participate", p.gt(0)), ("avoid", p.le(0))):
            row[model + "_" + part + "_months"] = int(mask.sum())
            row[model + "_" + part + "_mean_net_label"] = float(y.loc[mask].mean()) if mask.any() else None
            row[model + "_" + part + "_worst_marked_label"] = float(valid.loc[mask, "label_min_marked_return"].min()) if mask.any() else None
    if len(valid):
        row["mse_improvement"] = row["M0_mse"] - row["M1_mse"]
        row["relative_mse_improvement"] = row["mse_improvement"] / row["M0_mse"]
    return row


def prediction_evaluation(predictions: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DataFrame]:
    mature = predictions.loc[predictions.label_status.eq("MATURE")].copy()
    indices = month_indices(len(mature), config, config["random_seed"])
    loss = ((mature.Y_NET_MONTH - mature.prediction_M0) ** 2 - (mature.Y_NET_MONTH - mature.prediction_M1) ** 2).to_numpy()
    draws = np.nanmean(loss[indices], axis=1)
    rows = [prediction_group(mature, "ALL")]
    rows.extend(prediction_group(mature.loc[mature.entry_date.between(start, end)], name) for name, start, end in config["eras"])
    positive = sum(r["era"] in config["complete_eras_for_gate"] and r["paired_months"] >= config["minimum_era_rows"] and r.get("mse_improvement", -1) > 0 for r in rows)
    bounds = np.quantile(draws, [.025, .975])
    report = {**rows[0], "mature_calendar_months": len(mature), "censored_months": int(predictions.label_status.ne("MATURE").sum()),
              "no_view_mature_months": int(mature.prediction_status.ne("PASS").sum()),
              "distinct_used_latest_value_vintages": int(mature.loc[mature.prediction_status.eq("PASS"), "epu_latest_value_vintage"].nunique()),
              "positive_complete_eras": positive, "paired_mse_improvement_95pct_ci": bounds,
              "predictive_increment_gate_pass": bool(bounds[0] > 0 and positive >= 2), "bootstrap_unit": "CONSECUTIVE_CALENDAR_MONTHS_WITH_MISSING_MONTHS_PRESERVED"}
    return report, pd.DataFrame(rows), indices, pd.DataFrame({"mse_improvement": draws})


def request_quantity(account: Account, reference: pd.Series, weight: float, cost: dict, config: dict) -> int:
    target = math.floor(weight * account.value(float(reference.close)) / float(reference.close) / config["lot"]) * config["lot"]
    if target > account.shares:
        px = fill_price(float(reference.close), 1, cost, config["tick"])
        target = min(target, account.shares + affordable_quantity(account.cash, px, cost, config["lot"]))
    return int(target - account.shares)


def simulate_account(feature: pd.DataFrame, dividends: pd.DataFrame, predictions: pd.DataFrame,
                     cost: dict, config: dict, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = predictions.loc[predictions.label_status.eq("MATURE")]
    first, last = int(full.entry_index.iloc[0]), int(full.exit_index.iloc[-1])
    monthly = full.set_index("entry_index")
    account = Account(config["initial_capital"])
    events, ledgers, trades, decisions = dividends.to_dict("records"), [], [], []
    previous_nav, previous_mark = config["initial_capital"], float(feature.open.iloc[first])
    final_month = str(full.entry_month.iloc[-1])
    for day in range(first, last + 1):
        market, request, mu, status = feature.iloc[day], 0, None, "NO_SCHEDULED_REBALANCE"
        date = market.date
        if day in monthly.index:
            reference, row = feature.iloc[day - 1], monthly.loc[day]
            vol_weight = min(1.0, config["annual_volatility_budget"] / max(1e-12, math.sqrt(float(reference.RV20) * config["annual_days"])))
            if model == "BUY_HOLD":
                weight, status = 1.0, "BUY_HOLD"
                if day == first:
                    request = request_quantity(account, reference, weight, cost, config)
            elif model == "VOL_ONLY":
                weight, status = vol_weight, "VOL_ONLY"
                request = request_quantity(account, reference, weight, cost, config)
            else:
                status = row.prediction_status
                mu = float(row["prediction_" + model]) if status == "PASS" else None
                weight = vol_weight if mu is not None and mu > 0 else 0.0
                request = request_quantity(account, reference, weight, cost, config)
            decisions.append({"origin": reference.date, "execution_date": date, "requested_quantity": request,
                              "target_weight": weight, "net_prediction": mu, "information_status": status,
                              "reference_close": reference.close, "reference_cash": account.cash, "reference_shares": account.shares,
                              "reference_equity": account.value(float(reference.close)), "reference_RV20": reference.RV20})
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for key, event in enumerate(events):
            if event["ex_date"] == date:
                amount = account.entitlements.get(key, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[key] = amount
                    recognized += amount
            if event["payment_date"] < date and key in account.receivables:
                amount = account.receivables.pop(key)
                account.cash += amount
                paid += amount
        if day == last:
            request = -account.shares
        execution = execute_order(account, request, float(market.open), float(market.previous_close), float(market.dividend), day, cost, config)
        if request:
            trades.append({"date": date, "terminal": day == last, **execution})
        mark = float(market.open if day == last else market.close)
        if day != last:
            for key, event in enumerate(events):
                if event["payment_date"] == date and key in account.receivables:
                    amount = account.receivables.pop(key)
                    account.cash += amount
                    paid += amount
                if event["record_date"] == date:
                    account.entitlements[key] = account.shares
        nav = account.value(mark)
        pnl = old_shares * (float(market.open) - previous_mark) + account.shares * (mark - float(market.open))
        error = nav - previous_nav - pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "完整账户财富不守恒")
        account.assert_valid()
        ledgers.append({"date": date, "bootstrap_month": final_month if day == last else str(date.to_period("M")),
                        "mark": mark, "mark_clock": "OPEN_TERMINAL" if day == last else "CLOSE", "open_price": market.open,
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "price_pnl": pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid, "commission": execution["commission"],
                        "slippage_cost": execution["slippage_cost"], "accounting_error": error,
                        "exposure": account.shares * mark / nav, "execution_status": execution["status"]})
        previous_nav, previous_mark = nav, mark
    require(account.shares == 0, "终点未成功卖出，不能伪装现金终态")
    trade_columns = ["date", "terminal", "requested_quantity", "filled_quantity", "open_price", "fill_price", "commission", "slippage_cost", "notional", "status", "cash_before", "shares_before", "cash_after", "shares_after"]
    return pd.DataFrame(ledgers), pd.DataFrame(trades).reindex(columns=trade_columns), pd.DataFrame(decisions)


def economic_evaluation(ledgers: dict, indices: np.ndarray, config: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summaries, draws, aggregates = {}, {}, []
    for cost in config["costs"]:
        one = ledgers[cost + "_M1"]
        for baseline in ("M0", "VOL_ONLY"):
            zero = ledgers[cost + "_" + baseline]
            require(one.date.equals(zero.date), "账户日期不配对")
            frame = pd.DataFrame({"month": one.bootstrap_month, "delta": one.net_return - zero.net_return})
            monthly = frame.groupby("month", sort=True).delta.agg(["sum", "count"])
            require(monthly.shape[0] == indices.shape[1], "账户与预测月份区块不一致")
            values = monthly["sum"].to_numpy()[indices].sum(axis=1) / monthly["count"].to_numpy()[indices].sum(axis=1)
            key = cost + "_M1_MINUS_" + baseline
            draws[key] = values
            summaries[key] = {"mean_daily_increment": float(frame.delta.mean()),
                              "annualized_arithmetic_increment": float(frame.delta.mean() * config["annual_days"]),
                              "paired_daily_increment_95pct_ci": np.quantile(values, [.025, .975]),
                              "not_exposure_neutral_alpha": True}
            aggregates.extend({"comparison": key, "month": month, "delta_sum": row["sum"], "days": int(row["count"])} for month, row in monthly.iterrows())
    return summaries, pd.DataFrame(draws), pd.DataFrame(aggregates)


def preflight(config: dict) -> dict:
    out = outpath(config)
    require(not (out / "freeze_manifest.json").exists(), "已经冻结，禁止覆盖预检")
    data, report = load_inputs(config, snapshot=False)
    feature = price_features(data)
    samples = build_samples(data, feature, config, with_labels=False)
    evaluation = samples.loc[samples.evaluation_origin]
    first = evaluation.iloc[0]
    train = samples.loc[samples.exit_date.lt(first.origin) & samples.feature_status.eq("PASS")]
    require(len(train) >= config["minimum_train_rows"], "预检：首月成熟训练不足")
    full = evaluation.loc[evaluation.label_status.eq("MATURE_NOT_READ")]
    require(len(full) >= 36, "预检：完整评价月份不足三年")
    report.update(study_id=config["study_id"], created_at=now(), new_future_labels_computed=False, new_models_fitted=False,
                  new_accounts_computed=False, monthly_samples=len(samples), evaluation_months=len(evaluation),
                  mature_evaluation_months=len(full), first_model_training_months=len(train),
                  evaluation_feature_status_counts=evaluation.feature_status.value_counts().to_dict(),
                  planned_account_first=full.entry_date.iloc[0], planned_account_terminal=full.exit_date.iloc[-1],
                  untraded_tail_days=len(feature) - 1 - int(full.exit_index.iloc[-1]))
    write_json(out / "data_preflight.json", report, exclusive=True)
    samples.to_parquet(out / "preflight_schedule_without_labels.parquet", index=False)
    return report


def freeze(config: dict) -> dict:
    out = outpath(config)
    require(not (out / "freeze_manifest.json").exists(), "冻结记录已存在")
    pre = json.loads((out / "data_preflight.json").read_text(encoding="utf-8"))
    require(not pre["new_future_labels_computed"], "预检已经读取未来标签")
    snapshots = []
    for key, rel in config["inputs"].items():
        source, target = ROOT / rel, out / "frozen_inputs" / rel
        require(digest(source) == pre["source_provenance"][key]["sha256"], "预检后输入漂移")
        require(not target.exists(), "冻结输入已存在")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(digest(source) == digest(target), "输入复制不一致")
        snapshots.append(identity(target))
    protected = [CONFIG, PROTOCOL, Path(__file__), ROOT / config["authority"],
                 ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "scripts/run_510300_epu_vintage_increment_v1.py",
                 ROOT / "tests/test_epu_vintage_increment_v1.py", out / "USER_REQUEST.md", out / "deduplication.json",
                 out / "pre_freeze_tests.json", out / "data_preflight.json", out / "preflight_schedule_without_labels.parquet"]
    manifest = {"study_id": config["study_id"], "created_at": now(), "status": "FROZEN_BEFORE_NEW_LABELS_MODELS_AND_ACCOUNTS",
                "protected_files": [identity(p) for p in protected], "input_snapshots": snapshots,
                "models_at_freeze": 0, "accounts_at_freeze": 0, "historical_runs_allowed": 1}
    write_json(out / "freeze_manifest.json", manifest, exclusive=True)
    return manifest


def verify_frozen(config: dict) -> dict:
    manifest = json.loads((outpath(config) / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["protected_files"] + manifest["input_snapshots"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结文件漂移：" + item["path"])
    return manifest


def run(config: dict) -> dict:
    manifest, out = verify_frozen(config), outpath(config)
    write_json(out / "run_claim.json", {"started_at": now(), "pid": os.getpid(), "freeze_sha256": digest(out / "freeze_manifest.json")}, exclusive=True)
    try:
        data, source = load_inputs(config, snapshot=True)
        feature = price_features(data)
        samples = build_samples(data, feature, config, with_labels=True)
        predictions, models = rolling_predictions(samples, config)
        evaluation, groups, indices, pred_draws = prediction_evaluation(predictions, config)
        for name, frame in (("features", feature), ("vintages", data["vintages"]), ("samples", samples), ("predictions", predictions), ("prediction_bootstrap_draws", pred_draws)):
            frame.to_parquet(out / (name + ".parquet"), index=False)
        write_json(out / "models.json", {"models": models})
        np.savez_compressed(out / "month_bootstrap_indices.npz", indices=indices)
        groups.to_csv(out / "prediction_comparison.csv", index=False, encoding="utf-8-sig")
        account_dir = out / "accounts"
        account_dir.mkdir()
        accounts, era_accounts, ledgers = [], [], {}
        for cost_name, cost in config["costs"].items():
            for model in config["account_models"]:
                ledger, trades, decisions = simulate_account(feature, data["dividends"], predictions, cost, config, model)
                stem = cost_name + "_" + model
                ledger.to_parquet(account_dir / (stem + "_ledger.parquet"), index=False)
                decisions.to_parquet(account_dir / (stem + "_decisions.parquet"), index=False)
                trades.to_csv(account_dir / (stem + "_trades.csv"), index=False, encoding="utf-8-sig")
                accounts.append({"cost": cost_name, "model": model, "days": len(ledger), "first": ledger.date.iloc[0], "last": ledger.date.iloc[-1],
                                 "trade_records": len(trades), "terminal_equity": ledger.equity.iloc[-1], "mean_exposure": ledger.exposure.mean(),
                                 "commission": ledger.commission.sum(), "slippage_cost": ledger.slippage_cost.sum(),
                                 **return_metrics(ledger.net_return.to_numpy(), config["annual_days"])})
                for name, start, end in config["eras"]:
                    part = ledger.loc[ledger.date.between(start, end)]
                    era_accounts.append({"cost": cost_name, "model": model, "era": name, "days": len(part),
                                         **return_metrics(part.net_return.to_numpy(), config["annual_days"])})
                ledgers[stem] = ledger
        pd.DataFrame(accounts).to_csv(out / "account_comparison.csv", index=False, encoding="utf-8-sig")
        era_frame = pd.DataFrame(era_accounts)
        era_frame.to_csv(out / "account_era_comparison.csv", index=False, encoding="utf-8-sig")
        economic, account_draws, monthly = economic_evaluation(ledgers, indices, config)
        account_draws.to_parquet(out / "account_bootstrap_draws.parquet", index=False)
        monthly.to_csv(out / "economic_monthly_aggregates.csv", index=False, encoding="utf-8-sig")
        positive_eras = 0
        for name in config["complete_eras_for_gate"]:
            part = era_frame.loc[era_frame.cost.eq("STRESS") & era_frame.era.eq(name)].set_index("model")
            if all(part.loc["M1", "annualized_arithmetic_mean"] > part.loc[b, "annualized_arithmetic_mean"] for b in ("M0", "VOL_ONLY")):
                positive_eras += 1
        stress = next(a for a in accounts if a["cost"] == "STRESS" and a["model"] == "M1")
        economic_gate = (all(economic["STRESS_M1_MINUS_" + b]["paired_daily_increment_95pct_ci"][0] > 0 for b in ("M0", "VOL_ONLY"))
                         and positive_eras >= 2 and stress["annualized_return"] > 0)
        continuation = bool(evaluation["predictive_increment_gate_pass"] or economic_gate)
        historical_target = bool(stress["annualized_return"] >= config["goal"]["net_cagr_at_least"] and stress["net_sharpe"] is not None and stress["net_sharpe"] >= config["goal"]["net_sharpe_at_least"])
        result = {"study_id": config["study_id"], "status": "HISTORICAL_EPU_INCREMENT_CANDIDATE_ONLY" if continuation else "REJECTED_FROZEN_NO_RELIABLE_EPU_INCREMENT",
                  "completed_at": now(), "freeze_created_at": manifest["created_at"], "evidence_class": config["evidence_class"],
                  "evaluation": evaluation, "economic_increment": economic, "positive_complete_economic_eras": positive_eras,
                  "economic_increment_gate_pass": bool(economic_gate), "continuation_gate_pass": continuation,
                  "accounts": accounts, "model_fits": len(models), "new_accounts": len(accounts),
                  "historical_stress_point_target_pass": historical_target, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
                  "historical_first_http_delivery_proven": False, "availability_scope": config["availability_scope"],
                  "source_first_vintage": source["first_vintage"], "existing_terminal_decisions_preserved": True,
                  "post_result_parameter_rescue": False, "network_calls_by_runner": 0, "position_impact": 0,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
        write_json(out / "result.json", result, exclusive=True)
        paths = [p for p in out.rglob("*") if p.is_file() and "frozen_inputs" not in p.relative_to(out).parts and p.name not in {"run_claim.json", "execution_receipt.json"}]
        write_json(out / "execution_receipt.json", {"status": "COMPLETED_ONCE", "completed_at": now(), "files": [identity(p) for p in sorted(paths)],
                   "new_model_fits": len(models), "new_accounts": len(accounts), "new_historical_runs": 1}, exclusive=True)
        return result
    except Exception as exc:
        write_json(out / "failure_receipt.json", {"status": "PROGRAM_FAILED_CLAIM_PRESERVED", "at": now(), "error_type": type(exc).__name__, "error": str(exc)}, exclusive=True)
        raise


def verify_saved(config: dict) -> dict:
    verify_frozen(config)
    out = outpath(config)
    execution = json.loads((out / "execution_receipt.json").read_text(encoding="utf-8"))
    for item in execution["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "保存文件漂移：" + item["path"])
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    samples, predictions = (pd.read_parquet(out / (name + ".parquet")) for name in ("samples", "predictions"))
    vintages = pd.read_parquet(out / "vintages.parquet")
    for row in samples.to_dict("records"):
        reconstructed = epu_at(vintages, row["decision_at"], config)
        require(reconstructed["epu_input_versions_json"] == row["epu_input_versions_json"], "历史版本选择不可重算")
        require(reconstructed["feature_status"] == row["feature_status"], "信息可用状态不一致")
        require(row["epu_all_feature_inputs_available_at"] <= row["decision_at"], "EPU 特征输入晚于决策")
        require(row["origin"] < row["entry_date"], "价格原点没有早于开盘")
        for col in config["M1_additions"]:
            require(abs(reconstructed[col] - row[col]) < 1e-12, "保存 EPU 特征不符")
    models = json.loads((out / "models.json").read_text(encoding="utf-8"))["models"]
    lookup = {model["model_key"]: model for model in models}
    for model in models:
        train = mature_training(samples, model["refit_origin"])
        require(len(train) == model["train_rows"] and train.exit_date.max() < pd.Timestamp(model["refit_origin"]), "成熟标签数量或截止时钟不符")
        require(hashlib.sha256(train.origin_index.to_numpy(np.int64).tobytes()).hexdigest() == model["train_origin_sha256"], "训练原点集合不符")
    errors = []
    for row in predictions.to_dict("records"):
        for model in ("M0", "M1"):
            if row["prediction_status"] == "PASS":
                errors.append(abs(predict(row, lookup[row["model_key_" + model]], config) - row["prediction_" + model]))
            else:
                require(pd.isna(row["prediction_" + model]), "NO_VIEW 月仍生成了预测")
    require(max(errors) < 1e-12, "保存模型不能重算预测")
    mature = predictions.loc[predictions.label_status.eq("MATURE")]
    indices = np.load(out / "month_bootstrap_indices.npz")["indices"]
    loss = ((mature.Y_NET_MONTH - mature.prediction_M0) ** 2 - (mature.Y_NET_MONTH - mature.prediction_M1) ** 2).to_numpy()
    draws = pd.read_parquet(out / "prediction_bootstrap_draws.parquet").mse_improvement.to_numpy()
    require(np.allclose(np.nanmean(loss[indices], axis=1), draws, atol=1e-16, rtol=0), "月度配对预测区块不符")
    require(np.allclose(np.quantile(draws, [.025, .975]), result["evaluation"]["paired_mse_improvement_95pct_ci"], atol=1e-16, rtol=0), "预测区间不符")
    ledgers = {}
    for item in result["accounts"]:
        stem = item["cost"] + "_" + item["model"]
        ledger = pd.read_parquet(out / "accounts" / (stem + "_ledger.parquet"))
        require(ledger.accounting_error.abs().max() < 1e-6 and ledger.shares.iloc[-1] == 0, "财富守恒或终点清算不符")
        require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-7, rtol=0), "每日净值恒等式不符")
        previous = np.r_[config["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
        require(np.allclose(ledger.equity.to_numpy() / previous - 1, ledger.net_return, atol=1e-14, rtol=0), "每日净值收益不符")
        for key, value in return_metrics(ledger.net_return.to_numpy(), config["annual_days"]).items():
            require((value is None and item[key] is None) or (value is not None and np.isclose(value, item[key], atol=1e-12)), "账户指标不可重算")
        decisions = pd.read_parquet(out / "accounts" / (stem + "_decisions.parquet"))
        require(len(decisions) == len(mature) and decisions.execution_date.equals(mature.entry_date.reset_index(drop=True)), "月度账户决策日期不符")
        if item["model"] in ("M0", "M1"):
            mask = decisions.information_status.ne("PASS")
            require(decisions.loc[mask, "target_weight"].eq(0).all(), "缺少信息时未按冻结规则取现金")
        ledgers[stem] = ledger
    economic, reconstructed_draws, monthly = economic_evaluation(ledgers, indices, config)
    saved_draws = pd.read_parquet(out / "account_bootstrap_draws.parquet")
    require(np.allclose(reconstructed_draws.to_numpy(), saved_draws.to_numpy(), atol=1e-16, rtol=0), "完整月份账户重采样不符")
    for key, value in economic.items():
        require(np.allclose(value["paired_daily_increment_95pct_ci"], result["economic_increment"][key]["paired_daily_increment_95pct_ci"], atol=1e-16, rtol=0), "账户增量区间不符")
    receipt = {"status": "PASS_SAVED_ARCHIVAL_CLOCK_MODELS_FULL_ACCOUNTS_AND_MONTH_BLOCK_RECOMPUTATION", "verified_at": now(),
               "archival_monthly_rows_checked": len(samples), "models_checked": len(models), "prediction_rows": len(predictions),
               "max_prediction_error": max(errors), "accounts_checked": len(ledgers), "new_model_fits": 0, "new_accounts": 0,
               "new_random_samples": 0, "network_calls": 0, "security_audit": False}
    write_json(out / "saved_verification_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="EPU 历史版本月度研究")
    parser.add_argument("action", choices=["preflight", "freeze", "run", "verify", "status"])
    action = parser.parse_args().action
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if action == "status":
        path = outpath(config) / "result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "NOT_RUN", "claim_exists": (outpath(config) / "run_claim.json").exists()}
    else:
        result = {"preflight": preflight, "freeze": freeze, "run": run, "verify": verify_saved}[action](config)
    from research.intraday_overnight_increment_v1 import safe_json
    print(json.dumps(safe_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
