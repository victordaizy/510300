"""A/H溢价信息的固定配对预测、连续现金账户及保存结果重算。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import (
    build_features as etf_features, digest, normalize_dividends, normalize_prices,
    now, require, return_metrics, write_json,
)
from research.if_open_interest_increment_v1 import (
    five_day_label, group_metrics, mature_training, ridge_predict,
    rolling_predictions, simulate_account,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_ah_premium_increment_v1.json"
PROTOCOL = ROOT / "docs/510300_AH_PREMIUM_INCREMENT_V1_PROTOCOL.md"


def read_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def output_path(config: dict) -> Path:
    return ROOT / config["output"]


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def parse_official_chart(payload: dict) -> pd.DataFrame:
    require(payload["indexCode"] == "01044.00", "官网指数身份不是A/H溢价")
    frame = pd.DataFrame(payload["indexLevels-5y"], columns=["timestamp_ms", "close"])
    frame["date"] = (pd.to_datetime(frame.timestamp_ms, unit="ms", utc=True)
                     .dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None).dt.normalize())
    require(not frame.date.duplicated().any(), "官网图表日期重复")
    require(frame.date.is_monotonic_increasing, "官网图表日期不递增")
    require(np.isfinite(frame.close).all() and frame.close.gt(0).all(), "A/H比值不合法")
    require(frame.date.max() == pd.Timestamp(payload["lastUpdate"]).normalize(), "官网日期解析与末日标签不符")
    return frame[["date", "close", "timestamp_ms"]]


def align_source(decisions: pd.Series, source: pd.DataFrame, prefix: str, config: dict) -> pd.DataFrame:
    require(not source.date.duplicated().any(), "外部来源日期重复")
    right = source[["date", "close"]].sort_values("date").copy()
    right = right.rename(columns={"date": prefix + "_source_date", "close": prefix + "_close"})
    clock = prefix + "_available_at"
    right[clock] = right[prefix + "_source_date"] + pd.Timedelta(days=1, hours=config["external_available_hour_next_calendar_day"])
    left = pd.DataFrame({"decision_at": decisions, "row_index": np.arange(len(decisions))})
    valid = left.decision_at.notna()
    joined = pd.merge_asof(left.loc[valid].sort_values("decision_at"), right, left_on="decision_at",
                           right_on=clock, direction="backward", allow_exact_matches=True)
    joined = joined.set_index("row_index").reindex(np.arange(len(decisions)))
    joined[prefix + "_age_days"] = (joined.decision_at.dt.normalize() - joined[prefix + "_source_date"]).dt.days
    joined[prefix + "_source_ok"] = (joined[clock].notna() & joined[clock].le(joined.decision_at)
        & joined[prefix + "_age_days"].between(1, config["maximum_source_age_calendar_days"]))
    return joined.drop(columns="decision_at").reset_index(drop=True)


def load_inputs(config: dict, *, snapshot: bool) -> tuple[dict, dict]:
    base = output_path(config) / "frozen_inputs" if snapshot else ROOT
    files = {name: base / rel for name, rel in config["inputs"].items()}
    price_receipt = json.loads(files["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(files["dividend_coverage"].read_text(encoding="utf-8"))
    require(price_receipt["status"] == "PASS", "ETF行情缺少准入回执")
    require(digest(files["prices"]) == price_receipt["canonical_price"]["output_sha256"], "ETF行情版本漂移")
    require(coverage["complete_history_confirmed"] and coverage["coverage_end"] >= config["data_cutoff"], "分红覆盖不足")
    require(digest(files["dividends"]) == coverage["distribution_file_sha256"], "分红版本漂移")
    prices = normalize_prices(pd.read_parquet(files["prices"]))
    prices = prices.loc[prices.date.le(pd.Timestamp(config["data_cutoff"]))].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(files["dividends"]))
    calendar = pd.read_parquet(files["calendar"])
    expected = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    expected = expected[(expected >= prices.date.iloc[0]) & (expected <= pd.Timestamp(config["data_cutoff"]))].sort_values()
    require(pd.DatetimeIndex(prices.date).equals(expected), "ETF价格与开市日历不符")
    require(prices.date.iloc[-1] == pd.Timestamp(config["data_cutoff"]), "ETF行情未到固定截止日")
    require(len(dividends) == coverage["event_count"], "分红事件数不符")
    ah = parse_official_chart(json.loads(files["ah_chart"].read_text(encoding="utf-8")))
    source_receipt = json.loads(files["ah_source_receipt"].read_text(encoding="utf-8"))
    chart_receipts = [r for r in source_receipt if r["url"] == "https://www.hsi.com.hk/data/eng/indexes/01044.00/chart.json"]
    require(len(chart_receipts) == 1 and digest(files["ah_chart"]) == chart_receipts[0]["sha256"], "A/H原始响应与回执不一致")
    cross = pd.read_parquet(files["ah_sina_crosscheck"])[["date", "close"]]
    overlap = ah.merge(cross, on="date", suffixes=("_hsi", "_sina"), validate="one_to_one")
    error = float((overlap.close_hsi - overlap.close_sina).abs().max())
    require(len(overlap) >= 40 and error < 1e-10, "官网与新浪共同日线未一致")
    require(float(ah.set_index("date").loc[pd.Timestamp("2026-08-31"), "close"]) == 123.53, "与已读官方8月事实表不一致")
    hsi = pd.read_parquet(files["hsi_prices"])
    hsi_receipt = json.loads(files["hsi_receipt"].read_text(encoding="utf-8"))
    require(hsi_receipt["markets"]["^HSI"]["passed"], "HSI控制变量缺少历史准入")
    require(digest(files["hsi_prices"]) == hsi_receipt["markets"]["^HSI"]["sha256"], "HSI控制变量与原回执不同")
    hsi["date"] = pd.to_datetime(hsi.date).dt.normalize()
    require(set(hsi.symbol) == {"^HSI"}, "HSI控制变量身份不符")
    require(not hsi.date.duplicated().any() and np.isfinite(hsi.close).all() and hsi.close.gt(0).all(), "HSI控制变量不合法")
    info = {"status": "PASS_FREE_OFFICIAL_CHART_AND_RECENT_CROSSCHECK_WITH_ASSUMED_CLOCK",
            "ah_rows": len(ah), "ah_first": ah.date.min(), "ah_last": ah.date.max(),
            "crosscheck_rows": len(overlap), "crosscheck_max_absolute_error": error,
            "historical_first_delivery_timestamp_proven": False, "etf_rows": len(prices),
            "dividend_events": len(dividends), "hsi_control_rows": len(hsi),
            "source_provenance": {key: identity(path) for key, path in files.items()}}
    return {"prices": prices, "dividends": dividends,
            "ah": ah.loc[ah.date.le(config["data_cutoff"])].reset_index(drop=True),
            "hsi": hsi.loc[hsi.date.le(config["data_cutoff"])].reset_index(drop=True)}, info


def build_features(data: dict, config: dict) -> pd.DataFrame:
    feature = etf_features(data["prices"], data["dividends"], 20)
    feature["ETF_R1"] = feature.total_simple
    for window in (5, 20):
        feature["ETF_R" + str(window)] = np.expm1(feature.total_log.rolling(window, min_periods=window).sum())
    feature["decision_at"] = feature.date.shift(-1) + pd.Timedelta(hours=config["decision_hour_entry_day"])
    for prefix in ("ah", "hsi"):
        aligned = align_source(feature.decision_at, data[prefix], prefix, config)
        feature = pd.concat([feature, aligned], axis=1)
    for window in (5, 20):
        feature["HSI_R" + str(window)] = feature.hsi_close / feature.hsi_close.shift(window) - 1
    feature["AH_LOG_LEVEL"] = np.log(feature.ah_close / 100)
    feature["AH_LOG_CHANGE5"] = np.log(feature.ah_close).diff(5)
    available = feature.ah_source_ok & feature.hsi_source_ok
    finite = np.isfinite(feature[config["M0"] + config["M1_additions"]].to_numpy(float)).all(axis=1)
    feature["feature_status"] = np.where(available & finite, "PASS", "NO_VIEW_INPUT_NOT_AVAILABLE")
    feature["origin_index"] = np.arange(len(feature))
    return feature


def build_samples(feature: pd.DataFrame, dividends: pd.DataFrame, config: dict, *, with_labels: bool) -> pd.DataFrame:
    columns = ["origin_index", "date", "decision_at", "feature_status", "ah_source_date", "hsi_source_date",
               "ah_available_at", "hsi_available_at", "ah_age_days", "hsi_age_days"] + config["M0"] + config["M1_additions"]
    rows = []
    for origin in np.flatnonzero(feature.date.ge(config["train_origin_start"])):
        entry, end = origin + 1, origin + 1 + config["horizon"]
        row = feature.iloc[origin][columns].to_dict()
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


def block_indices(count: int, config: dict, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = config["bootstrap_day_block"]
    starts = rng.integers(0, count, size=(config["bootstrap_repetitions"], math.ceil(count / block)))
    return ((starts[:, :, None] + np.arange(block)) % count).reshape(len(starts), -1)[:, :count].astype(np.int32)


def prediction_evaluation(predictions: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DataFrame]:
    frame = predictions.loc[predictions.label_status.eq("MATURE")].copy()
    require(frame.prediction_status.eq("PASS").all(), "成熟评价原点未完整配对")
    loss = ((frame.Y5_NET_BASE - frame.prediction_M0) ** 2 - (frame.Y5_NET_BASE - frame.prediction_M1) ** 2).to_numpy()
    indices = block_indices(len(frame), config, seed=config["random_seed"])
    draws = loss[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [.025, .975])
    groups = [group_metrics(frame, "ALL")]
    for name, start, end in config["eras"]:
        groups.append(group_metrics(frame.loc[frame.entry_date.between(start, end)], name))
    positive = sum(g["mse_improvement"] > 0 for g in groups[1:])
    gate = lower > 0 and positive >= 2 and all(g["paired_origins"] >= config["minimum_era_rows"] for g in groups[1:])
    result = {"overall": groups[0], "paired_mse_improvement_95pct_ci": [lower, upper],
              "positive_eras": positive, "predictive_increment_gate_pass": bool(gate),
              "mature_evaluation_origins": len(frame), "censored_origins": int(predictions.label_status.ne("MATURE").sum()),
              "overlapping_labels": True, "researcher_selection_bias_corrected": False}
    return result, pd.DataFrame(groups), indices, pd.DataFrame({"mse_improvement": draws})


def preflight(config: dict) -> dict:
    out = output_path(config)
    require(not (out / "freeze_manifest.json").exists(), "已冻结，不能覆盖预检")
    data, report = load_inputs(config, snapshot=False)
    feature = build_features(data, config)
    samples = build_samples(feature, data["dividends"], config, with_labels=False)
    evaluation = samples.loc[samples.evaluation_origin]
    require(evaluation.feature_status.eq("PASS").all(), "评价日期缺少共同可用信息")
    first = evaluation.iloc[0]
    training = samples.loc[samples.exit_date.lt(first.origin) & samples.feature_status.eq("PASS")]
    require(len(training) >= config["minimum_train_rows"], "第一次更新的成熟训练行不足")
    entry = int(first.entry_index)
    terminal = entry + ((len(feature) - 1 - entry) // config["horizon"]) * config["horizon"]
    report.update({"study_id": config["study_id"], "created_at": now(), "new_future_labels_computed": False,
                   "new_models_fitted": False, "new_accounts_computed": False,
                   "first_model_training_rows": len(training), "evaluation_origins": len(evaluation),
                   "mature_evaluation_origins": int(evaluation.label_status.eq("MATURE_NOT_READ").sum()),
                   "planned_account_first": feature.date.iloc[entry], "planned_account_terminal": feature.date.iloc[terminal],
                   "untraded_tail_days_after_terminal": len(feature) - 1 - terminal,
                   "max_ah_source_age_days": int(evaluation.ah_age_days.max()),
                   "max_hsi_source_age_days": int(evaluation.hsi_age_days.max())})
    write_json(out / "data_preflight.json", report, exclusive=True)
    samples.to_parquet(out / "preflight_schedule_without_labels.parquet", index=False)
    return report


def freeze(config: dict) -> dict:
    out = output_path(config)
    require(not (out / "freeze_manifest.json").exists(), "冻结记录已存在")
    pre = json.loads((out / "data_preflight.json").read_text(encoding="utf-8"))
    require(not pre["new_future_labels_computed"], "预检已经计算标签")
    snapshots = []
    for key, rel in config["inputs"].items():
        source, target = ROOT / rel, out / "frozen_inputs" / rel
        require(digest(source) == pre["source_provenance"][key]["sha256"], "预检之后输入漂移")
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "输入快照已存在")
        shutil.copyfile(source, target)
        require(digest(target) == digest(source), "复制不一致")
        snapshots.append(identity(target))
    protected = [CONFIG, PROTOCOL, Path(__file__), ROOT / config["authority"],
                 ROOT / "research/if_open_interest_increment_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
                 ROOT / "scripts/run_510300_ah_premium_increment_v1.py", ROOT / "tests/test_ah_premium_increment_v1.py",
                 out / "USER_REQUEST.md", out / "deduplication.json", out / "pre_freeze_tests.json",
                 out / "data_preflight.json", out / "preflight_schedule_without_labels.parquet"]
    record = {"study_id": config["study_id"], "status": "FROZEN_BEFORE_NEW_LABELS_MODELS_AND_ACCOUNTS",
              "created_at": now(), "historical_runs_allowed": 1, "protected_files": [identity(p) for p in protected],
              "input_snapshots": snapshots, "models_at_freeze": 0, "accounts_at_freeze": 0}
    write_json(out / "freeze_manifest.json", record, exclusive=True)
    return record


def verify_frozen(config: dict) -> dict:
    manifest = json.loads((output_path(config) / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["protected_files"] + manifest["input_snapshots"]:
        path = ROOT / item["path"]
        require(path.is_file() and digest(path) == item["sha256"], "冻结文件不一致：" + item["path"])
    return manifest


def run(config: dict) -> dict:
    manifest = verify_frozen(config)
    out = output_path(config)
    write_json(out / "run_claim.json", {"started_at": now(), "pid": os.getpid(),
               "freeze_sha256": digest(out / "freeze_manifest.json")}, exclusive=True)
    try:
        data, source = load_inputs(config, snapshot=True)
        feature = build_features(data, config)
        samples = build_samples(feature, data["dividends"], config, with_labels=True)
        predictions, models = rolling_predictions(samples, config)
        evaluation, groups, pred_indices, pred_draws = prediction_evaluation(predictions, config)
        for name, frame in [("features", feature), ("samples", samples), ("predictions", predictions),
                            ("prediction_bootstrap_draws", pred_draws)]:
            frame.to_parquet(out / (name + ".parquet"), index=False)
        write_json(out / "models.json", {"models": models})
        groups.to_csv(out / "prediction_comparison.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(out / "prediction_bootstrap_indices.npz", indices=pred_indices)
        account_dir = out / "accounts"
        account_dir.mkdir()
        accounts, ledgers, era_accounts = [], {}, []
        for cost_name, cost in config["costs"].items():
            for model in config["account_models"]:
                ledger, trades, decisions = simulate_account(feature, data["dividends"], predictions, cost, config, model)
                stem = cost_name + "_" + model
                ledger.to_parquet(account_dir / (stem + "_ledger.parquet"), index=False)
                trades.to_csv(account_dir / (stem + "_trades.csv"), index=False, encoding="utf-8-sig")
                decisions.to_parquet(account_dir / (stem + "_decisions.parquet"), index=False)
                metrics = return_metrics(ledger.net_return.to_numpy(), config["annual_days"])
                accounts.append({"cost": cost_name, "model": model, "first": ledger.date.iloc[0], "last": ledger.date.iloc[-1],
                                 "days": len(ledger), "trade_records": len(trades), "terminal_equity": ledger.equity.iloc[-1],
                                 "mean_exposure": ledger.exposure.mean(), "commission": ledger.commission.sum(),
                                 "slippage_cost": ledger.slippage_cost.sum(), **metrics})
                for name, start, end in config["eras"]:
                    part = ledger.loc[ledger.date.between(start, end)]
                    era_accounts.append({"cost": cost_name, "model": model, "era": name, "days": len(part),
                                         **return_metrics(part.net_return.to_numpy(), config["annual_days"])})
                ledgers[stem] = ledger
        pd.DataFrame(accounts).to_csv(out / "account_comparison.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(era_accounts).to_csv(out / "account_era_comparison.csv", index=False, encoding="utf-8-sig")
        account_indices = block_indices(len(ledgers["BASE_M0"]), config, seed=config["random_seed"] + 1)
        np.savez_compressed(out / "account_bootstrap_indices.npz", indices=account_indices)
        economic, account_draws = {}, {}
        for cost_name in config["costs"]:
            a, b = ledgers[cost_name + "_M0"], ledgers[cost_name + "_M1"]
            require(a.date.equals(b.date), "配对账户日期不一致")
            daily = (b.net_return - a.net_return).to_numpy()
            draws = daily[account_indices].mean(axis=1)
            account_draws[cost_name + "_mean_daily_increment"] = draws
            bounds = np.quantile(draws, [.025, .975])
            economic[cost_name] = {"mean_daily_increment": daily.mean(), "paired_daily_increment_95pct_ci": bounds,
                                  "annualized_arithmetic_increment": daily.mean() * config["annual_days"],
                                  "increment_is_not_exposure_neutral_regression": True}
        pd.DataFrame(account_draws).to_parquet(out / "account_bootstrap_draws.parquet", index=False)
        stress = next(a for a in accounts if a["cost"] == "STRESS" and a["model"] == "M1")
        continuation = (evaluation["predictive_increment_gate_pass"]
                        and economic["STRESS"]["paired_daily_increment_95pct_ci"][0] > 0 and stress["annualized_return"] > 0)
        historical_target = (stress["annualized_return"] >= config["goal"]["net_cagr_at_least"]
                             and stress["net_sharpe"] is not None
                             and stress["net_sharpe"] >= config["goal"]["net_sharpe_at_least"])
        result = {"study_id": config["study_id"], "status": "HISTORICAL_INCREMENT_CANDIDATE_ONLY" if continuation else "REJECTED_FROZEN_NO_RELIABLE_AH_INCREMENT",
                  "completed_at": now(), "freeze_created_at": manifest["created_at"], "evidence_class": config["evidence_class"],
                  "evaluation": evaluation, "economic_increment": economic, "accounts": accounts,
                  "model_fits": len(models), "refit_months": len(models) // 2, "new_accounts": len(accounts),
                  "continuation_gate_pass": bool(continuation), "historical_stress_point_target_pass": bool(historical_target),
                  "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
                  "historical_first_delivery_timestamp_proven": False, "historical_availability_assumption": config["historical_availability_assumption"],
                  "limited_source_history_start": source["ah_first"], "existing_rejected_studies_reopened": False,
                  "post_result_parameter_rescue": False, "network_calls_by_runner": 0, "position_impact": 0,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
        write_json(out / "result.json", result, exclusive=True)
        files = [p for p in out.rglob("*") if p.is_file() and "frozen_inputs" not in p.relative_to(out).parts
                 and p.name not in {"run_claim.json", "execution_receipt.json"}]
        write_json(out / "execution_receipt.json", {"status": "COMPLETED_ONCE", "completed_at": now(),
                   "files": [identity(p) for p in sorted(files)], "new_model_fits": len(models),
                   "new_accounts": len(accounts), "new_historical_runs": 1}, exclusive=True)
        return result
    except Exception as exc:
        write_json(out / "failure_receipt.json", {"status": "PROGRAM_FAILED_CLAIM_PRESERVED", "at": now(),
                   "error_type": type(exc).__name__, "error": str(exc)}, exclusive=True)
        raise


def verify_saved(config: dict) -> dict:
    verify_frozen(config)
    out = output_path(config)
    execution = json.loads((out / "execution_receipt.json").read_text(encoding="utf-8"))
    for item in execution["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "保存文件漂移：" + item["path"])
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    samples = pd.read_parquet(out / "samples.parquet")
    predictions = pd.read_parquet(out / "predictions.parquet")
    models = json.loads((out / "models.json").read_text(encoding="utf-8"))["models"]
    lookup = {m["model_key"]: m for m in models}
    for model in models:
        train = mature_training(samples, pd.Timestamp(model["refit_origin"]))
        require(len(train) == model["train_rows"], "保存成熟训练数量不符")
        require(hashlib.sha256(train.origin_index.to_numpy(np.int64).tobytes()).hexdigest() == model["train_origin_sha256"], "训练原点集合不符")
        require(pd.Timestamp(model["train_last_exit"]) < pd.Timestamp(model["refit_origin"]), "训练标签时钟泄漏")
    errors = []
    for row in predictions.to_dict("records"):
        for prefix in ("ah", "hsi"):
            require(row[prefix + "_source_date"] < row["entry_date"], "外部源日没有早于入场")
            require(row[prefix + "_available_at"] <= row["decision_at"] < row["entry_date"] + pd.Timedelta(hours=9, minutes=30), "数据到达、决定和成交顺序错误")
        for model in ("M0", "M1"):
            errors.append(abs(ridge_predict(row, lookup[row["model_key_" + model]], config) - row["prediction_" + model]))
    require(max(errors) < 1e-12, "保存预测不可由保存模型重算")
    frame = predictions.loc[predictions.label_status.eq("MATURE")]
    loss = ((frame.Y5_NET_BASE - frame.prediction_M0) ** 2 - (frame.Y5_NET_BASE - frame.prediction_M1) ** 2).to_numpy()
    indices = np.load(out / "prediction_bootstrap_indices.npz")["indices"]
    draws = pd.read_parquet(out / "prediction_bootstrap_draws.parquet").mse_improvement.to_numpy()
    require(np.allclose(loss[indices].mean(axis=1), draws, atol=1e-16, rtol=0), "保存预测区块不可重算")
    require(np.allclose(np.quantile(draws, [.025, .975]), result["evaluation"]["paired_mse_improvement_95pct_ci"], atol=1e-16, rtol=0), "保存预测区间不符")
    ledgers = {}
    for item in result["accounts"]:
        stem = item["cost"] + "_" + item["model"]
        ledger = pd.read_parquet(out / "accounts" / (stem + "_ledger.parquet"))
        require(ledger.accounting_error.abs().max() < 1e-6 and ledger.shares.iloc[-1] == 0, "保存账户财富或清算终态不符")
        require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-7, rtol=0), "现金份额应收与净值不符")
        metrics = return_metrics(ledger.net_return.to_numpy(), config["annual_days"])
        for key, value in metrics.items():
            require(value is None and item[key] is None or value is not None and np.isclose(value, item[key], atol=1e-12), "保存账户指标不符")
        ledgers[stem] = ledger
    account_indices = np.load(out / "account_bootstrap_indices.npz")["indices"]
    account_draws = pd.read_parquet(out / "account_bootstrap_draws.parquet")
    for cost_name in config["costs"]:
        delta = (ledgers[cost_name + "_M1"].net_return - ledgers[cost_name + "_M0"].net_return).to_numpy()
        values = delta[account_indices].mean(axis=1)
        require(np.allclose(values, account_draws[cost_name + "_mean_daily_increment"], atol=1e-16, rtol=0), "保存账户增量区块不符")
        require(np.allclose(np.quantile(values, [.025, .975]), result["economic_increment"][cost_name]["paired_daily_increment_95pct_ci"], atol=1e-16, rtol=0), "保存账户增量区间不符")
    verified = {"status": "PASS_SAVED_CLOCK_MODELS_ACCOUNTS_AND_PAIRED_BLOCK_RECOMPUTATION", "verified_at": now(),
                "models_checked": len(models), "prediction_rows": len(predictions), "max_prediction_error": max(errors),
                "accounts_checked": len(ledgers), "new_model_fits": 0, "new_accounts": 0, "new_random_samples": 0,
                "network_calls": 0, "security_audit": False}
    write_json(out / "saved_verification_receipt.json", verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="A/H溢价信息的冻结增量研究")
    parser.add_argument("mode", choices=["preflight", "freeze", "run", "verify", "status"])
    args = parser.parse_args()
    config = read_config()
    if args.mode == "status":
        path = output_path(config) / "result.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "NOT_RUN"}
    else:
        result = {"preflight": preflight, "freeze": freeze, "run": run, "verify": verify_saved}[args.mode](config)
    from research.intraday_overnight_increment_v1 import safe_json
    print(json.dumps(safe_json(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
