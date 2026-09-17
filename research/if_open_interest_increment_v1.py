"""IF总持仓的配对增量诊断；只交易510300的账户阶段由冻结主门控制。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, build_features as etf_features, commission,
    digest, execute_order, fill_price, normalize_dividends, normalize_prices,
    now, require, return_metrics, safe_json, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_if_open_interest_increment_v1.json"
PROTOCOL = ROOT / "docs/510300_IF_OPEN_INTEREST_INCREMENT_V1_PROTOCOL.md"


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def output_path(config: dict) -> Path:
    return ROOT / config["output"]


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def load_inputs(config: dict, *, snapshot: bool) -> tuple[dict, dict]:
    base = output_path(config) / "frozen_inputs" if snapshot else ROOT
    files = {name: base / rel for name, rel in config["inputs"].items()}
    price_receipt = json.loads(files["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(files["dividend_coverage"].read_text(encoding="utf-8"))
    receipt = json.loads(files["if_receipt"].read_text(encoding="utf-8"))
    require(price_receipt["status"] == "PASS", "ETF价格缺少既有准入回执")
    require(digest(files["prices"]) == price_receipt["canonical_price"]["output_sha256"], "ETF价格版本不匹配")
    require(coverage["complete_history_confirmed"], "分红覆盖未确认")
    require(coverage["coverage_end"] >= config["data_cutoff"], "分红覆盖不足")
    require(digest(files["dividends"]) == coverage["distribution_file_sha256"], "分红版本不匹配")
    require(receipt["status"] == "SUCCESS_CFFEX_IF_CONTRACT_HISTORY_ACQUIRED", "官方IF历史获取未成功")
    for key in ("if_daily", "if_expiry"):
        require(digest(files[key]) == receipt["outputs"][config["inputs"][key]], f"{key}与官方回执不匹配")

    prices = normalize_prices(pd.read_parquet(files["prices"]))
    prices = prices.loc[prices.date <= pd.Timestamp(config["data_cutoff"])].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(files["dividends"]))
    calendar = pd.read_parquet(files["calendar"])
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    expected = dates[(dates >= prices.date.iloc[0]) & (dates <= pd.Timestamp(config["data_cutoff"]))].sort_values()
    require(pd.DatetimeIndex(prices.date).equals(expected), "ETF行情与独立开市日历不同")
    require(str(prices.date.iloc[-1].date()) == config["data_cutoff"], "ETF终点未覆盖")
    require(len(dividends) == coverage["event_count"], "分红事件数不匹配")
    for event in dividends.itertuples():
        if prices.date.iloc[0] <= event.ex_date <= prices.date.iloc[-1]:
            require(event.record_date in expected and event.ex_date in expected, "分红日期缺少交易日")
            require(expected[expected.get_loc(event.ex_date) - 1] == event.record_date, "登记日与除息日不相邻")

    spot = pd.read_parquet(files["spot"])
    spot["date"] = pd.to_datetime(spot.date).dt.normalize()
    require(not spot.date.duplicated().any(), "现货指数日期重复")
    require(set(spot.symbol) == {"000300.SH"}, "现货观察标的不符")
    require(np.isfinite(spot.close).all() and spot.close.gt(0).all(), "现货价格非法")
    spot = spot.set_index("date").reindex(expected).reset_index(names="date")
    require(spot.close.notna().all(), "现货缺少ETF交易日")

    daily = pd.read_parquet(files["if_daily"])
    expiry = pd.read_parquet(files["if_expiry"])
    daily["date"] = pd.to_datetime(daily.date).dt.normalize()
    for name in ("first_date", "last_history_date", "expiry_date"):
        expiry[name] = pd.to_datetime(expiry[name]).dt.normalize()
    require(not daily.duplicated(["date", "symbol"]).any(), "IF逐合约键重复")
    require(daily.symbol.str.fullmatch(r"IF\d{4}").all(), "IF数据混入其他品种或主连")
    require(not expiry.symbol.duplicated().any(), "到期表代码重复")
    require(set(daily.symbol) == set(expiry.symbol), "合约与到期表不一致")
    require(len(daily) == receipt["daily"]["rows"], "IF行数不符")
    require(np.isfinite(daily[["close", "open_interest"]]).all().all(), "IF输入有非有限值")
    require(daily.close.gt(0).all() and daily.open_interest.ge(0).all(), "IF价格或持仓值非法")
    require(daily.source_url.str.contains("cffex.com.cn", regex=False).all(), "IF来源域不符")
    require(str(daily.date.max().date()) == config["if_data_cutoff"], "IF截止日不符")
    counts = daily.groupby("date").size()
    require(counts.eq(4).all(), "IF日记录并非完整四合约，禁止用不完整总量")
    if_expected = expected[(expected >= pd.Timestamp(config["if_comparable_price_start"])) & (expected <= pd.Timestamp(config["if_data_cutoff"]))]
    require(set(if_expected).issubset(set(daily.date)), "IF缺少必要交易日")
    joined = daily.merge(expiry[["symbol", "expiry_date"]], on="symbol", validate="many_to_one")
    require(joined.date.le(joined.expiry_date).all(), "存在到期日之后的合约记录")
    data = {"prices": prices, "dividends": dividends, "spot": spot, "if_daily": daily,
            "if_expiry": expiry, "calendar": expected}
    info = {"status": "PASS_EXISTING_SOURCES_WITH_ASSUMED_ONE_DAY_LAG",
            "if_rows": len(daily), "if_contracts": daily.symbol.nunique(), "if_days": daily.date.nunique(),
            "if_first": daily.date.min(), "if_last": daily.date.max(), "etf_rows": len(prices),
            "etf_last": prices.date.max(), "dividend_events": len(dividends),
            "same_day_publication_timestamp_proven": False, "historical_lag_is_assumption": True,
            "source_provenance": {name: identity(path) for name, path in files.items()}}
    return data, info


def if_daily_features(daily: pd.DataFrame, expiry: pd.DataFrame, spot: pd.DataFrame,
                      comparable_start: str) -> pd.DataFrame:
    """按同一真实合约计算价差收益；持仓总量保留换月日并显式标识。"""
    daily = daily.sort_values(["date", "symbol"]).copy()
    dates = pd.DatetimeIndex(sorted(daily.date.unique()))
    closes = daily.pivot(index="date", columns="symbol", values="close").reindex(dates)
    present = daily.pivot(index="date", columns="symbol", values="open_interest").reindex(dates)
    aggregate = present.sum(axis=1, min_count=1)
    oi_change = np.log(aggregate.where(aggregate > 0)).diff()
    spot_close = spot.set_index("date").close.reindex(dates)
    expires = expiry.set_index("symbol").expiry_date.to_dict()
    start = pd.Timestamp(comparable_start)
    rows = []
    old_symbols: set[str] = set()
    for index, date in enumerate(dates):
        symbols = set(present.columns[present.iloc[index].notna()])
        has_expiry = any(expires[symbol] == date for symbol in symbols)
        week = date.to_period("W-SUN")
        expiry_week = any(pd.Timestamp(expires[symbol]).to_period("W-SUN") == week for symbol in symbols)
        roll = bool(index and symbols != old_symbols) or has_expiry
        relative, contract_count = np.nan, 0
        price_start = pd.NaT
        if index >= 5 and dates[index - 5] >= start:
            price_start = dates[index - 5]
            eligible = [symbol for symbol in sorted(symbols) if expires[symbol] > date
                        and closes[symbol].iloc[index - 5:index + 1].notna().all()]
            if eligible and np.isfinite([spot_close.iloc[index], spot_close.iloc[index - 5]]).all():
                same_contract = np.log(closes.loc[date, eligible].to_numpy(float)
                                       / closes.loc[price_start, eligible].to_numpy(float))
                relative = float(same_contract.mean() - np.log(spot_close.iloc[index] / spot_close.iloc[index - 5]))
                contract_count = len(eligible)
        rows.append({"date": date, "IF_TOTAL_OI": float(aggregate.iloc[index]),
                     "IF_OI_LOG_CHANGE1": float(oi_change.iloc[index]), "IF_REL5": relative,
                     "IF_EXPIRY_WEEK": float(expiry_week), "IF_ROLL_BOUNDARY": float(roll),
                     "IF_PRICE_START": price_start, "IF_PRICE_CONTRACTS": contract_count})
        old_symbols = symbols
    return pd.DataFrame(rows)


def build_features(data: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature = etf_features(data["prices"], data["dividends"], 20)
    feature["ETF_R1"] = feature.total_simple
    feature["ETF_R5"] = np.expm1(feature.total_log.rolling(5, min_periods=5).sum())
    contract = if_daily_features(data["if_daily"], data["if_expiry"], data["spot"], config["if_comparable_price_start"])
    aligned = contract.set_index("date").reindex(pd.DatetimeIndex(feature.date))
    require(config["if_availability_lag_trading_days"] == 1, "禁止更改IF信息延迟")
    for name in ("IF_REL5", "IF_EXPIRY_WEEK", "IF_ROLL_BOUNDARY", "IF_OI_LOG_CHANGE1"):
        feature[name + "_LAG1"] = aligned[name].shift(1).to_numpy()
    feature["if_source_date"] = pd.Series(feature.date).shift(1)
    feature["if_source_present"] = aligned.IF_TOTAL_OI.notna().shift(1, fill_value=False).to_numpy()
    feature["if_price_start_date"] = aligned.IF_PRICE_START.shift(1).to_numpy()
    feature["IF_OI_LOG_CHANGE1_LAG1_X_ETF_R5_SIGN"] = feature.IF_OI_LOG_CHANGE1_LAG1 * np.sign(feature.ETF_R5)
    cols = config["M0"] + config["M1_additions"]
    finite = np.isfinite(feature[cols].to_numpy(float)).all(axis=1)
    feature["feature_status"] = np.where(finite & feature.if_source_present, "PASS", "NO_VIEW_INPUT_NOT_AVAILABLE")
    feature["origin_index"] = np.arange(len(feature))
    return feature, contract


def five_day_label(feature: pd.DataFrame, dividends: pd.DataFrame, origin: int, config: dict) -> dict:
    entry, end = origin + 1, origin + 1 + int(config["horizon"])
    if end >= len(feature):
        return {"label_status": "CENSORED_AFTER_CUTOFF", "Y5_NET_BASE": np.nan,
                "exit_index": end, "exit_date": pd.NaT}
    entry_date, exit_date = feature.date.iloc[entry], feature.date.iloc[end]
    eligible = dividends.loc[dividends.record_date.ge(entry_date) & dividends.record_date.lt(exit_date)
                             & dividends.ex_date.le(exit_date)]
    cash_dividend = float(eligible.cash_dividend_per_share.sum())
    cost = config["costs"]["BASE"]
    buy = fill_price(float(feature.open.iloc[entry]), 1, cost, config["tick"])
    sell = fill_price(float(feature.open.iloc[end]), -1, cost, config["tick"])
    quantity = affordable_quantity(config["initial_capital"], buy, cost, config["lot"])
    require(quantity > 0, "独立标签账户无法买入一手")
    paid = quantity * buy + commission(quantity, buy, cost)
    received = quantity * (sell + cash_dividend) - commission(quantity, sell, cost)
    net = (received - paid) / config["initial_capital"]
    marks = feature.iloc[entry:end][["date", "close"]]
    marked_values = []
    for row in marks.itertuples():
        accrued = float(eligible.loc[eligible.ex_date.le(row.date), "cash_dividend_per_share"].sum())
        marked_values.append((quantity * (float(row.close) + accrued) - paid) / config["initial_capital"])
    return {"label_status": "MATURE", "entry_index": entry, "exit_index": end,
            "entry_date": entry_date, "exit_date": exit_date, "Y5_NET_BASE": net,
            "Y5_GROSS_QUOTED": (float(feature.open.iloc[end]) + cash_dividend) / float(feature.open.iloc[entry]) - 1,
            "label_dividend_per_share": cash_dividend, "label_quantity": quantity,
            "label_min_marked_return": min([0.0, net] + marked_values),
            "entry_gap_total_return": (float(feature.open.iloc[entry]) + float(feature.dividend.iloc[entry])) / float(feature.close.iloc[origin]) - 1}


def build_samples(feature: pd.DataFrame, dividends: pd.DataFrame, config: dict, *, with_labels: bool) -> pd.DataFrame:
    rows = []
    for origin in np.flatnonzero(feature.date.ge(pd.Timestamp(config["train_origin_start"]))):
        entry, end = origin + 1, origin + 1 + config["horizon"]
        row = feature.iloc[origin][["origin_index", "date", "feature_status", "if_source_date",
                                   "if_price_start_date"] + config["M0"] + config["M1_additions"]].to_dict()
        row["origin"] = row.pop("date")
        row.update({"entry_index": entry, "exit_index": end,
                    "entry_date": feature.date.iloc[entry] if entry < len(feature) else pd.NaT,
                    "exit_date": feature.date.iloc[end] if end < len(feature) else pd.NaT,
                    "evaluation_origin": bool(entry < len(feature) and feature.date.iloc[entry] >= pd.Timestamp(config["evaluation_entry_start"])),
                    "label_status": "MATURE_NOT_READ" if end < len(feature) else "CENSORED_AFTER_CUTOFF"})
        if with_labels:
            row.update(five_day_label(feature, dividends, int(origin), config))
        rows.append(row)
    return pd.DataFrame(rows)


def mature_training(samples: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    return samples.loc[samples.exit_date.lt(origin) & samples.feature_status.eq("PASS")
                       & samples.label_status.eq("MATURE") & samples.Y5_NET_BASE.notna()]


def ridge_fit(frame: pd.DataFrame, cols: list[str], config: dict) -> dict:
    x = frame[cols].to_numpy(float)
    y = frame.Y5_NET_BASE.to_numpy(float)
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = np.clip((x - mean) / std, -config["standardized_clip"], config["standardized_clip"])
    design = np.c_[np.ones(len(z)), z]
    penalty = np.diag([0.0] + [config["ridge_lambda"]] * len(cols))
    beta = np.linalg.solve(design.T @ design / len(y) + penalty, design.T @ y / len(y))
    return {"features": cols, "mean": mean, "std": std, "coefficients": beta,
            "train_rows": len(frame), "train_first_origin": frame.origin.min(),
            "train_last_origin": frame.origin.max(), "train_last_exit": frame.exit_date.max(),
            "train_origin_sha256": hashlib.sha256(frame.origin_index.to_numpy(np.int64).tobytes()).hexdigest()}


def ridge_predict(row: dict | pd.Series, model: dict, config: dict) -> float:
    x = np.array([row[col] for col in model["features"]], dtype=float)
    z = np.clip((x - np.asarray(model["mean"])) / np.asarray(model["std"]),
                -config["standardized_clip"], config["standardized_clip"])
    return float(np.r_[1.0, z] @ np.asarray(model["coefficients"]))


def rolling_predictions(samples: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    cols = {"M0": config["M0"], "M1": config["M0"] + config["M1_additions"]}
    records, models = [], []
    last_month, current = None, {}
    for item in samples.loc[samples.evaluation_origin].to_dict("records"):
        month = str(pd.Timestamp(item["entry_date"]).to_period("M"))
        if month != last_month:
            train = mature_training(samples, pd.Timestamp(item["origin"]))
            require(len(train) >= config["minimum_train_rows"], "月首成熟训练行数不足")
            current = {}
            for name, columns in cols.items():
                model = ridge_fit(train, columns, config)
                model.update({"model_id": name, "model_key": month + ":" + name,
                              "refit_origin": item["origin"], "entry_month": month})
                require(model["train_last_exit"] < item["origin"], "训练标签越过模型可得时钟")
                models.append(model)
                current[name] = model
            last_month = month
        item["prediction_status"] = item["feature_status"]
        for name in cols:
            item["prediction_" + name] = (ridge_predict(item, current[name], config)
                                           if item["feature_status"] == "PASS" else np.nan)
            item["model_key_" + name] = current[name]["model_key"]
        records.append(item)
    return pd.DataFrame(records), models


def group_metrics(frame: pd.DataFrame, label: str) -> dict:
    paired = frame.dropna(subset=["Y5_NET_BASE", "prediction_M0", "prediction_M1"])
    row = {"group": label, "calendar_origins": len(frame), "paired_origins": len(paired)}
    if not len(paired):
        return row
    y = paired.Y5_NET_BASE.to_numpy(float)
    for model in ("M0", "M1"):
        prediction = paired["prediction_" + model].to_numpy(float)
        participate = prediction > 0
        row.update({model + "_mse": float(np.mean((y - prediction) ** 2)),
                    model + "_mae": float(np.mean(np.abs(y - prediction))),
                    model + "_positive_prediction_fraction": float(participate.mean()),
                    model + "_positive_prediction_realized_mean": float(y[participate].mean()) if participate.any() else None,
                    model + "_positive_prediction_realized_loss_fraction": float((y[participate] < 0).mean()) if participate.any() else None,
                    model + "_positive_prediction_min_marked_mean": float(paired.loc[participate, "label_min_marked_return"].mean()) if participate.any() else None})
    row["mse_improvement"] = row["M0_mse"] - row["M1_mse"]
    row["relative_mse_improvement"] = row["mse_improvement"] / row["M0_mse"]
    row["mean_entry_gap_total_return"] = float(paired.entry_gap_total_return.mean())
    return row


def paired_evaluation(predictions: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DataFrame]:
    frame = predictions.loc[predictions.label_status.eq("MATURE")].copy()
    valid = frame.prediction_status.eq("PASS") & frame[["prediction_M0", "prediction_M1"]].notna().all(axis=1)
    require(valid.any(), "没有配对预测可供评价")
    loss = (frame.Y5_NET_BASE - frame.prediction_M0) ** 2 - (frame.Y5_NET_BASE - frame.prediction_M1) ** 2
    loss = loss.where(valid).to_numpy(float)
    rng = np.random.default_rng(config["random_seed"])
    count, block = len(frame), config["bootstrap_day_block"]
    starts = rng.integers(0, count, size=(config["bootstrap_repetitions"], math.ceil(count / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % count).reshape(len(starts), -1)[:, :count].astype(np.int32)
    draws = np.nanmean(loss[indices], axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    groups = [group_metrics(frame, "ALL")]
    era_rows = []
    for name, start, end in config["eras"]:
        era = group_metrics(frame.loc[frame.entry_date.between(pd.Timestamp(start), pd.Timestamp(end))], name)
        era_rows.append(era)
        groups.append(era)
    for name, field, value in [("EXPIRY_WEEK", "IF_EXPIRY_WEEK_LAG1", 1),
                               ("NON_EXPIRY_WEEK", "IF_EXPIRY_WEEK_LAG1", 0),
                               ("ROLL_BOUNDARY", "IF_ROLL_BOUNDARY_LAG1", 1),
                               ("NON_ROLL_BOUNDARY", "IF_ROLL_BOUNDARY_LAG1", 0)]:
        groups.append(group_metrics(frame.loc[frame[field].eq(value)], name))
    positive = sum(era.get("mse_improvement", 0) > 0 for era in era_rows)
    gates = {"positive_paired_95pct_lower_bound": bool(lower > 0),
             "at_least_two_positive_eras": positive >= config["primary_gate"]["minimum_positive_eras"],
             "era_sample_coverage": all(era["paired_origins"] >= config["primary_gate"]["minimum_rows_each_era"] for era in era_rows),
             "complete_mature_origin_coverage": bool(valid.all())}
    result = {"primary_gate_pass": all(gates.values()), "gates": gates, "overall": groups[0],
              "paired_mse_improvement_95pct_ci": [float(lower), float(upper)], "positive_eras": positive,
              "mature_evaluation_origins": len(frame), "paired_origins": int(valid.sum()),
              "censored_origins": int(predictions.label_status.ne("MATURE").sum()),
              "bootstrap": {"replicates": len(draws), "block_trading_days": block, "seed": config["random_seed"],
                            "preserved_full_calendar_before_resampling": True,
                            "project_level_research_selection_corrected": False}}
    return result, pd.DataFrame(groups), indices, pd.DataFrame({"replicate": np.arange(len(draws)), "mse_improvement": draws})


def account_request(account: Account, market: pd.Series, prediction: float, cost: dict, config: dict) -> tuple[int, float]:
    """以决策日ETF价格和风险决定份额；不得借账户空仓压低波动来放大。"""
    weight = min(1.0, config["annual_volatility_budget"] / math.sqrt(config["annual_days"] * float(market.RV20))) if prediction > 0 else 0.0
    target = math.floor(weight * account.value(float(market.close)) / float(market.close) / config["lot"]) * config["lot"]
    if target > account.shares:
        px = fill_price(float(market.close), 1, cost, config["tick"])
        target = min(target, account.shares + affordable_quantity(account.cash, px, cost, config["lot"]))
    return int(target - account.shares), float(weight)


def simulate_account(feature: pd.DataFrame, dividends: pd.DataFrame, predictions: pd.DataFrame,
                     cost: dict, config: dict, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = int(np.flatnonzero(feature.date.ge(pd.Timestamp(config["evaluation_entry_start"])))[0])
    last = first + ((len(feature) - 1 - first) // config["horizon"]) * config["horizon"]
    by_origin = predictions.set_index("origin_index")
    account = Account(config["initial_capital"])
    events = dividends.to_dict("records")
    ledgers, trades, decisions = [], [], []
    previous_nav, previous_mark = config["initial_capital"], float(feature.open.iloc[first])
    pending = 0
    for day in range(first, last + 1):
        market = feature.iloc[day]
        date = market.date
        if day == first or (day - first) % config["horizon"] == 0 and day < last:
            origin = day - 1
            reference = feature.iloc[origin]
            if model == "BUY_HOLD":
                if day == first:
                    pending = affordable_quantity(account.cash, fill_price(float(reference.close), 1, cost, config["tick"]), cost, config["lot"])
                weight, mu = 1.0, None
            else:
                require(origin in by_origin.index and by_origin.loc[origin, "prediction_status"] == "PASS", "账户原点缺少预测")
                mu = float(by_origin.loc[origin, "prediction_" + model])
                pending, weight = account_request(account, reference, mu, cost, config)
            decisions.append({"origin": reference.date, "execution_date": date, "requested_quantity": pending,
                              "target_weight": weight, "net_prediction": mu, "reference_close": reference.close})
        old_shares = account.shares
        recognized, paid = 0.0, 0.0
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
        request = -account.shares if day == last else pending
        execution = execute_order(account, request, float(market.open), float(market.previous_close),
                                  float(market.dividend), day, cost, config)
        if request:
            trades.append({"date": date, "terminal": day == last, **execution})
        pending = 0
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
        price_pnl = old_shares * (float(market.open) - previous_mark) + account.shares * (mark - float(market.open))
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "完整账户财富不守恒")
        account.assert_valid()
        ledgers.append({"date": date, "mark": mark, "mark_clock": "OPEN_TERMINAL" if day == last else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1.0, "price_pnl": price_pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid,
                        "commission": execution["commission"], "slippage_cost": execution["slippage_cost"],
                        "accounting_error": error, "exposure": account.shares * mark / nav,
                        "execution_status": execution["status"]})
        previous_nav, previous_mark = nav, mark
    require(account.shares == 0, "终点仍有无法卖出的份额，账户终态不可静默视为现金")
    return pd.DataFrame(ledgers), pd.DataFrame(trades), pd.DataFrame(decisions)


def preflight(config: dict) -> dict:
    out = output_path(config)
    require(not (out / "freeze_manifest.json").exists(), "已冻结，不能覆盖预检")
    data, report = load_inputs(config, snapshot=False)
    feature, contract = build_features(data, config)
    samples = build_samples(feature, data["dividends"], config, with_labels=False)
    mature = samples.loc[samples.evaluation_origin & samples.label_status.eq("MATURE_NOT_READ")]
    report.update({"study_id": config["study_id"], "created_at": now(), "new_future_labels_computed": False,
                   "new_models_fitted": False, "new_accounts_computed": False,
                   "planned_mature_evaluation_origins": len(mature),
                   "planned_mature_feature_pass": int(mature.feature_status.eq("PASS").sum()),
                   "planned_first_origin": mature.origin.min(), "planned_last_origin": mature.origin.max(),
                   "training_feature_pass": int(samples.loc[~samples.evaluation_origin, "feature_status"].eq("PASS").sum()),
                   "max_abs_log_oi_change": float(contract.IF_OI_LOG_CHANGE1.abs().max())})
    require(mature.feature_status.eq("PASS").all(), "成熟评价日有缺失特征，预检不通过")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "data_preflight.json", report, exclusive=True)
    samples.to_parquet(out / "preflight_schedule_without_labels.parquet", index=False)
    return report


def freeze(config: dict) -> dict:
    out = output_path(config)
    require(not (out / "freeze_manifest.json").exists(), "冻结清单已经存在")
    prior = json.loads((out / "data_preflight.json").read_text(encoding="utf-8"))
    require(prior["new_future_labels_computed"] is False, "预检已经读取标签")
    snapshot_records = []
    for name, rel in config["inputs"].items():
        source = ROOT / rel
        require(digest(source) == prior["source_provenance"][name]["sha256"], "预检后输入发生漂移")
        target = out / "frozen_inputs" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "输入快照已存在，禁止覆盖")
        shutil.copyfile(source, target)
        require(digest(target) == digest(source), "快照复制不一致")
        snapshot_records.append(identity(target))
    protected = [CONFIG, PROTOCOL, ROOT / config["authority"], Path(__file__),
                 ROOT / "research/intraday_overnight_increment_v1.py",
                 ROOT / "scripts/run_510300_if_open_interest_increment_v1.py",
                 ROOT / "tests/test_if_open_interest_increment_v1.py",
                 out / "data_preflight.json", out / "preflight_schedule_without_labels.parquet",
                 out / "USER_PROPOSAL.txt", out / "USER_REQUEST.md", out / "SOURCE_NOTES.md",
                 out / "deduplication.json", out / "pre_freeze_tests.json"]
    receipt = {"study_id": config["study_id"], "status": "FROZEN_BEFORE_LABELS_AND_MODELS",
               "created_at": now(), "protected_files": [identity(path) for path in protected],
               "input_snapshots": snapshot_records, "historical_runs_allowed": 1,
               "labels_at_freeze": 0, "models_at_freeze": 0, "accounts_at_freeze": 0}
    write_json(out / "freeze_manifest.json", receipt, exclusive=True)
    return receipt


def verify_frozen(config: dict) -> dict:
    manifest = json.loads((output_path(config) / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["protected_files"] + manifest["input_snapshots"]:
        path = ROOT / item["path"]
        require(path.is_file() and digest(path) == item["sha256"], "冻结文件不匹配：" + item["path"])
    return manifest


def run(config: dict) -> dict:
    manifest = verify_frozen(config)
    out = output_path(config)
    write_json(out / "run_claim.json", {"started_at": now(), "freeze_sha256": digest(out / "freeze_manifest.json"),
                                        "pid": __import__("os").getpid()}, exclusive=True)
    try:
        data, source_report = load_inputs(config, snapshot=True)
        feature, contract = build_features(data, config)
        samples = build_samples(feature, data["dividends"], config, with_labels=True)
        predictions, models = rolling_predictions(samples, config)
        evaluation, groups, indices, draws = paired_evaluation(predictions, config)
        feature.to_parquet(out / "features.parquet", index=False)
        contract.to_parquet(out / "if_source_features.parquet", index=False)
        samples.to_parquet(out / "samples.parquet", index=False)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        write_json(out / "models.json", {"models": models})
        groups.to_csv(out / "prediction_comparison.csv", index=False, encoding="utf-8-sig")
        draws.to_parquet(out / "bootstrap_draws.parquet", index=False)
        np.savez_compressed(out / "bootstrap_indices.npz", indices=indices)
        accounts = []
        if evaluation["primary_gate_pass"]:
            account_dir = out / "accounts"
            account_dir.mkdir()
            for cost_name, cost in config["costs"].items():
                for model in ("M0", "M1", "BUY_HOLD"):
                    ledger, trades, decisions = simulate_account(feature, data["dividends"], predictions, cost, config, model)
                    stem = cost_name + "_" + model
                    ledger.to_parquet(account_dir / (stem + "_ledger.parquet"), index=False)
                    trades.to_csv(account_dir / (stem + "_trades.csv"), index=False, encoding="utf-8-sig")
                    decisions.to_parquet(account_dir / (stem + "_decisions.parquet"), index=False)
                    metrics = return_metrics(ledger.net_return.to_numpy(), config["annual_days"])
                    accounts.append({"cost": cost_name, "model": model, "first": ledger.date.iloc[0], "last": ledger.date.iloc[-1],
                                     "days": len(ledger), "trade_records": len(trades), **metrics})
            pd.DataFrame(accounts).to_csv(out / "account_comparison.csv", index=False, encoding="utf-8-sig")
        status = "HISTORICAL_INCREMENT_PASS_ACCOUNT_DIAGNOSTIC_ONLY" if evaluation["primary_gate_pass"] else "REJECTED_FROZEN_NO_INCREMENT_UNDER_REGISTERED_TEST"
        result = {"study_id": config["study_id"], "status": status, "completed_at": now(),
                  "evidence_class": config["evidence_class"], "evaluation": evaluation,
                  "model_fits": len(models), "refit_months": len(models) // 2,
                  "account_stage": "COMPUTED_HISTORICAL_DIAGNOSTIC" if accounts else "NOT_RUN_PRIMARY_INCREMENT_GATE_FAILED",
                  "accounts": accounts, "net_sharpe": "SEE_ACCOUNTS" if accounts else "NOT_COMPUTED",
                  "net_cagr": "SEE_ACCOUNTS" if accounts else "NOT_COMPUTED",
                  "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
                  "availability_timestamp_proven": False, "availability_assumption": config["historical_availability_assumption"],
                  "existing_rejected_studies_reopened": False, "new_final_full_history_model": False,
                  "futures_orders": 0, "broker_calls": 0, "position_impact": 0,
                  "no_parameter_rescue": True, "freeze_created_at": manifest["created_at"],
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
        write_json(out / "result.json", result, exclusive=True)
        protected = [p for p in out.iterdir() if p.is_file() and p.name not in {"run_claim.json", "execution_receipt.json"}]
        if accounts:
            protected.extend(p for p in (out / "accounts").iterdir() if p.is_file())
        write_json(out / "execution_receipt.json", {"status": "COMPLETED_ONCE", "completed_at": now(),
                                                   "files": [identity(path) for path in sorted(protected)],
                                                   "new_historical_runs": 1, "new_model_fits": len(models),
                                                   "new_accounts": len(accounts), "network_calls_by_runner": 0}, exclusive=True)
        return result
    except Exception as exc:
        write_json(out / "failure_receipt.json", {"status": "PROGRAM_FAILED_CLAIM_PRESERVED", "at": now(),
                                                  "error_type": type(exc).__name__, "error": str(exc)}, exclusive=True)
        raise


def verify_saved(config: dict) -> dict:
    verify_frozen(config)
    out = output_path(config)
    receipt = json.loads((out / "execution_receipt.json").read_text(encoding="utf-8"))
    for item in receipt["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "保存结果哈希不同：" + item["path"])
    samples = pd.read_parquet(out / "samples.parquet")
    predictions = pd.read_parquet(out / "predictions.parquet")
    models = json.loads((out / "models.json").read_text(encoding="utf-8"))["models"]
    by_model = {model["model_key"]: model for model in models}
    for model in models:
        refit = pd.Timestamp(model["refit_origin"])
        train = mature_training(samples, refit)
        require(len(train) == model["train_rows"], "成熟训练数量不一致")
        require(hashlib.sha256(train.origin_index.to_numpy(np.int64).tobytes()).hexdigest() == model["train_origin_sha256"], "训练原点集合不一致")
        require(pd.Timestamp(model["train_last_exit"]) < refit, "发现重叠标签泄漏")
    max_error = 0.0
    for row in predictions.loc[predictions.prediction_status.eq("PASS")].to_dict("records"):
        require(pd.Timestamp(row["if_source_date"]) < pd.Timestamp(row["origin"]) < pd.Timestamp(row["entry_date"]), "信息日期顺序不同")
        for name in ("M0", "M1"):
            prediction = ridge_predict(row, by_model[row["model_key_" + name]], config)
            max_error = max(max_error, abs(prediction - row["prediction_" + name]))
    require(max_error < 1e-12, "保存预测不能由保存模型重算")
    frame = predictions.loc[predictions.label_status.eq("MATURE")]
    loss = ((frame.Y5_NET_BASE - frame.prediction_M0) ** 2 - (frame.Y5_NET_BASE - frame.prediction_M1) ** 2).to_numpy()
    indices = np.load(out / "bootstrap_indices.npz")["indices"]
    draws = pd.read_parquet(out / "bootstrap_draws.parquet")
    recomputed = np.nanmean(loss[indices], axis=1)
    require(np.allclose(recomputed, draws.mse_improvement.to_numpy(), atol=1e-16, rtol=0), "保存区块统计不能重算")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    require(np.allclose(np.quantile(recomputed, [.025, .975]), result["evaluation"]["paired_mse_improvement_95pct_ci"], atol=1e-16, rtol=0), "保存区间不同")
    if not result["evaluation"]["primary_gate_pass"]:
        require(not (out / "accounts").exists() and not result["accounts"], "主门失败仍生成了账户")
    else:
        for item in result["accounts"]:
            ledger = pd.read_parquet(out / "accounts" / (item["cost"] + "_" + item["model"] + "_ledger.parquet"))
            require(ledger.accounting_error.abs().max() < 1e-6, "保存账户不守恒")
            metrics = return_metrics(ledger.net_return.to_numpy(), config["annual_days"])
            for key, value in metrics.items():
                require(value is None and item[key] is None or np.isclose(value, item[key], atol=1e-12), "保存账户绩效不同")
    verified = {"status": "PASS_SAVED_MODEL_CLOCK_BOOTSTRAP_AND_STAGE_RECOMPUTATION", "verified_at": now(),
                "models_checked": len(models), "prediction_rows": len(predictions), "max_prediction_error": max_error,
                "bootstrap_draws_checked": len(draws), "new_fits": 0, "new_accounts": 0,
                "new_random_samples": 0, "network_calls": 0, "security_audit": False}
    write_json(out / "saved_verification_receipt.json", verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="IF持仓增量的冻结研究入口")
    parser.add_argument("mode", choices=["preflight", "freeze", "run", "verify", "status"])
    args = parser.parse_args()
    config = read_config()
    if args.mode == "status":
        path = output_path(config) / "result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "NOT_RUN"}
    else:
        result = {"preflight": preflight, "freeze": freeze, "run": run, "verify": verify_saved}[args.mode](config)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.mode in {"freeze", "preflight"}:
        result = {key: value for key, value in result.items() if key not in {"protected_files", "input_snapshots", "source_provenance"}}
    print(json.dumps(safe_json(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
