"""固定一个报告增长分歧变量的历史顺序预测与条件完整账户试验。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.adaptive_allocation_v1 import factors, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import normalize_prices, normalize_dividends, holding_total_return, return_metrics

CONFIG = ROOT / "config/510300_eps_growth_disagreement_increment_v1.json"
PROTOCOL = ROOT / "docs/510300_EPS_GROWTH_DISAGREEMENT_INCREMENT_V1_PROTOCOL.md"
OUT = ROOT / "reports/research/510300_eps_growth_disagreement_increment_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def frozen_paths(config):
    source = ROOT / config["source_feasibility"]
    paths = [CONFIG, PROTOCOL, Path(__file__), ROOT / "tests/test_eps_growth_disagreement_increment_v1.py",
             ROOT / "scripts/prepare_eps_growth_disagreement_increment_sources_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/event_clock_account_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "config/510300_research_authority_v6.json",
             ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
             ROOT / config["old_monthly_features"], source / "company_pairs.parquet",
             source / "monthly_source_feasibility.parquet", source / "result.json", source / "claim.json",
             OUT / "source_check_claim.json", OUT / "source_validation.json", OUT / "source_fact_checks.parquet",
             OUT / "source_sample_visual_verification.json", OUT / "synthetic_test_receipt.json"]
    paths += [ROOT / path for path in config["inputs"].values()]
    paths += [ROOT / row["path"] for row in read(OUT / "source_validation.json")["files"]]
    return sorted(set(paths))


def freeze():
    config = read(CONFIG)
    if read(OUT / "source_validation.json")["status"] != "PASS_USED_EPS_ROWS_RECOMPUTED_CLOCKS_AND_WITHIN_REPORT_UNITS_CHECKED":
        raise ValueError("参与来源的数学与时钟核对未通过")
    if read(OUT / "synthetic_test_receipt.json")["status"] != "PASS_SYNTHETIC_CLOCK_NESTED_RIDGE_AND_CALENDAR_BLOCK_TESTS":
        raise ValueError("必要合成测试未通过")
    if not read(OUT / "source_sample_visual_verification.json")["all_selected_pages_inspected"]:
        raise ValueError("固定原件样本尚未核实")
    files = [identity(path) for path in frozen_paths(config)]
    save(OUT / "protocol_freeze.json", {"frozen_at": now(), "study_id": config["study_id"],
         "old_history_and_source_statistics_seen": True, "new_candidate_predictions_seen": False,
         "new_candidate_account_returns_seen": False, "candidate_count": 1, "model_information_sets": 2,
         "files": files})
    print(json.dumps({"status": "单变量试验协议与全部输入已冻结", "files": len(files), "sha256": identity(OUT / "protocol_freeze.json")["sha256"]}, ensure_ascii=False), flush=True)


def check_frozen():
    for entry in read(OUT / "protocol_freeze.json")["files"]:
        if identity(ROOT / entry["path"]) != entry:
            raise ValueError("冻结输入或代码改变：" + entry["path"])
    return read(CONFIG)


def load_market(config):
    coverage = read(ROOT / config["inputs"]["dividend_coverage"])
    if not coverage["complete_history_confirmed"] or coverage["coverage_end"] < config["data_cutoff"]:
        raise ValueError("分红覆盖不足")
    dividend_path = ROOT / config["inputs"]["dividends"]
    if identity(dividend_path)["sha256"] != coverage["distribution_file_sha256"]:
        raise ValueError("分红与冻结覆盖回执不符")
    if read(ROOT / config["inputs"]["price_receipt"])["status"] != "PASS":
        raise ValueError("行情来源回执未通过")
    dividends = normalize_dividends(pd.read_csv(dividend_path))
    if len(dividends) != coverage["event_count"]:
        raise ValueError("分红事件数不符")
    prices = normalize_prices(pd.read_parquet(ROOT / config["inputs"]["prices"]))
    prices = prices.loc[prices.date <= pd.Timestamp(config["data_cutoff"])].reset_index(drop=True)
    calendar = pd.read_parquet(ROOT / config["inputs"]["calendar"])
    dates = pd.to_datetime(calendar.loc[calendar.is_open.eq(1), "trade_date"])
    dates = dates.loc[dates.between(prices.date.iloc[0], prices.date.iloc[-1])]
    if list(dates) != list(prices.date) or prices.date.iloc[-1] != pd.Timestamp(config["data_cutoff"]):
        raise ValueError("行情与冻结交易日历不连续或截点不符")
    data, _ = factors(prices, dividends)
    return data, dividends


def source_features(config):
    source = ROOT / config["source_feasibility"]
    pairs = pd.read_parquet(source / "company_pairs.parquet")
    previous = pd.read_parquet(source / "monthly_source_feasibility.parquet")
    monthly = pd.read_parquet(ROOT / config["old_monthly_features"])
    rows = []
    for origin, group in pairs.groupby("origin", sort=True):
        available = group.loc[group.paired_growth_available]
        row = {"origin": origin, "paired_company_count": len(available), "paired_company_fraction": len(available) / len(group),
               "paired_mean_growth_median": np.nan, "paired_age_gap_median": np.nan,
               "paired_mean_age_median": np.nan, "median_absolute_growth_disagreement": np.nan}
        if len(available):
            row.update({"paired_mean_growth_median": float(((available.eps_growth_guosen + available.eps_growth_soochow) / 2).median()),
                        "paired_age_gap_median": float(abs(available.report_age_days_guosen - available.report_age_days_soochow).median()),
                        "paired_mean_age_median": float(((available.report_age_days_guosen + available.report_age_days_soochow) / 2).median()),
                        "median_absolute_growth_disagreement": float(available.absolute_growth_disagreement.median())})
        rows.append(row)
    aggregated = pd.DataFrame(rows)
    expected = previous[["origin", "paired_company_count", "median_absolute_growth_disagreement"]].copy()
    pd.testing.assert_frame_equal(aggregated[expected.columns].reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, check_exact=True)
    monthly = monthly.merge(aggregated, on="origin", validate="one_to_one")
    columns = config["models"]["M1"]
    monthly["all_features_valid"] = monthly.common_sources_valid & monthly.paired_company_count.ge(config["minimum_paired_companies"]) & np.isfinite(monthly[columns]).all(axis=1)
    if monthly.all_features_valid.tolist() != previous.eligible_for_next_protocol.tolist():
        raise ValueError("新控制量意外改变了上一轮固定的有效源月")
    return monthly


def make_samples(monthly, data, dividends, config):
    samples = monthly.copy()
    indices = pd.DatetimeIndex(data.date).get_indexer(samples.origin)
    if (indices < 0).any():
        raise ValueError("来源原点不是行情交易日")
    samples["origin_index"] = indices
    samples["entry_index"] = indices + 1
    samples["exit_index"] = indices + 1 + config["horizon"]
    samples["Y60"] = np.nan
    samples["label_dividend_per_share"] = np.nan
    samples["label_exit_date"] = pd.NaT
    samples["label_status"] = "NO_VIEW_SOURCE"
    for index, row in samples.iterrows():
        if not row.all_features_valid:
            continue
        if row.exit_index >= len(data):
            samples.loc[index, "label_status"] = "CENSORED_AFTER_FROZEN_CUTOFF"
            continue
        value, distribution = holding_total_return(data, dividends, int(row.entry_index), int(row.exit_index))
        samples.loc[index, "Y60"] = value
        samples.loc[index, "label_dividend_per_share"] = distribution
        samples.loc[index, "label_exit_date"] = data.date.iloc[int(row.exit_index)]
        samples.loc[index, "label_status"] = "MATURE"
    return samples


def mature_training(samples, origin):
    return samples.loc[samples.all_features_valid & samples.origin.lt(origin) & samples.label_exit_date.notna() & samples.label_exit_date.le(origin) & np.isfinite(samples.Y60)].copy()


def fit_model(train, columns, alpha):
    x, y = train[columns].to_numpy(float), train.Y60.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("训练数据不完整")
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    z = (x - mean) / scale
    intercept = float(y.mean())
    slopes = np.linalg.solve(z.T @ z + alpha * np.eye(len(columns)), z.T @ (y - intercept))
    return {"columns": columns, "mean": mean.tolist(), "scale": scale.tolist(), "intercept": intercept,
            "slopes": slopes.tolist(), "alpha": alpha, "training_samples": len(train)}


def predict(row, model):
    x = np.asarray([row[column] for column in model["columns"]], dtype=float)
    return float(model["intercept"] + ((x - model["mean"]) / model["scale"]) @ np.asarray(model["slopes"]))


def rolling_predictions(samples, config):
    rows, receipts = [], []
    for _, source in samples.loc[samples.origin.ge(pd.Timestamp(config["evaluation_start"]))].iterrows():
        train = mature_training(samples, source.origin)
        valid = bool(source.all_features_valid and len(train) >= config["minimum_train_samples"])
        status = "TRAINED_PAIRED_MODELS" if valid else "NO_VIEW_SOURCE" if not source.all_features_valid else "NO_VIEW_INSUFFICIENT_MATURE_TRAINING_MONTHS"
        row = {"origin": source.origin, "origin_index": int(source.origin_index), "label_exit_date": source.label_exit_date,
               "Y60": source.Y60, "source_eligible": bool(source.all_features_valid), "training_samples": len(train),
               "status": status, "M0": np.nan, "M1": np.nan}
        if valid:
            for name, columns in config["models"].items():
                model = fit_model(train, columns, config["ridge_alpha"])
                value = predict(source, model)
                row[name] = value
                receipts.append({"model_id": name, "origin": source.origin.strftime("%Y-%m-%d"),
                    "training_origins": train.origin.dt.strftime("%Y-%m-%d").tolist(),
                    "latest_training_label_exit": train.label_exit_date.max().strftime("%Y-%m-%d"),
                    "training_label_sha256": hashlib.sha256(train.Y60.to_numpy(dtype="<f8").tobytes()).hexdigest(),
                    "fit": model, "prediction": value})
        rows.append(row)
    return pd.DataFrame(rows), receipts


def group_metrics(frame, label):
    valid = frame.loc[np.isfinite(frame[["Y60", "M0", "M1"]]).all(axis=1)]
    result = {"group": label, "n": len(valid)}
    if len(valid) == 0:
        result.update({"M0_mse": None, "M1_mse": None, "mse_improvement": None, "relative_mse_improvement": None})
        return result
    y = valid.Y60.to_numpy(float)
    for model in ("M0", "M1"):
        p = valid[model].to_numpy(float)
        actual, predicted = y > 0, p > 0
        result[model + "_mse"] = float(np.mean((y - p) ** 2))
        result[model + "_mae"] = float(np.mean(abs(y - p)))
        result[model + "_direction_accuracy"] = float(np.mean(actual == predicted))
        result[model + "_true_up"] = int((actual & predicted).sum())
        result[model + "_false_up"] = int((~actual & predicted).sum())
        result[model + "_true_nonup"] = int((~actual & ~predicted).sum())
        result[model + "_false_nonup"] = int((actual & ~predicted).sum())
    result["actual_up_count"] = int((y > 0).sum())
    result["actual_down_count"] = int((y < 0).sum())
    result["mse_improvement"] = result["M0_mse"] - result["M1_mse"]
    result["relative_mse_improvement"] = result["mse_improvement"] / result["M0_mse"] if result["M0_mse"] > 0 else None
    return result


def block_draw_indices(count, block, repetitions, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, count, size=(repetitions, math.ceil(count / block)))
    return ((starts[:, :, None] + np.arange(block)) % count).reshape(repetitions, -1)[:, :count]


def evaluate_predictions(predictions, config, saved_indices=None):
    valid = predictions.loc[np.isfinite(predictions[["Y60", "M0", "M1"]]).all(axis=1)].copy()
    if not len(valid):
        raise ValueError("没有可评价的共同成熟预测，不能形成误差区间")
    groups = [group_metrics(valid, "ALL"), group_metrics(valid.loc[valid.Y60.gt(0)], "ACTUAL_UP"), group_metrics(valid.loc[valid.Y60.lt(0)], "ACTUAL_DOWN")]
    for year in sorted(valid.origin.dt.year.unique()):
        groups.append(group_metrics(valid.loc[valid.origin.dt.year.eq(year)], str(year)))
    valid["month"] = valid.origin.dt.to_period("M").astype(str)
    valid["loss_improvement"] = (valid.Y60 - valid.M0) ** 2 - (valid.Y60 - valid.M1) ** 2
    months = pd.period_range(valid.origin.min().to_period("M"), valid.origin.max().to_period("M"), freq="M").astype(str)
    calendar = pd.DataFrame({"month": months}).merge(valid[["month", "origin", "loss_improvement"]], on="month", how="left", validate="one_to_one")
    gate = config["prediction_gate"]
    indices = saved_indices if saved_indices is not None else block_draw_indices(len(calendar), gate["bootstrap_month_block"], gate["bootstrap_repetitions"], gate["random_seed"])
    draws = np.nanmean(calendar.loss_improvement.to_numpy()[indices], axis=1)
    if not np.isfinite(draws).all():
        raise ValueError("固定重采样产生无有效月的样本")
    lower, upper = np.quantile(draws, [.025, .975]).tolist()
    year_checks = []
    for year in gate["required_complete_years"]:
        row = next((x for x in groups if x["group"] == str(year)), {"n": 0, "mse_improvement": None})
        year_checks.append({"year": year, "n": row["n"], "mse_improvement": row["mse_improvement"],
                            "passed": bool(row["n"] >= gate["minimum_paired_origins_each_required_year"] and row["mse_improvement"] is not None and row["mse_improvement"] > 0)})
    conditions = {"minimum_mature_pairs": len(valid) >= gate["minimum_paired_mature_origins"],
                  "positive_95_interval_lower": lower > 0, "positive_both_complete_years": all(x["passed"] for x in year_checks)}
    evaluation = {"overall": groups[0], "mature_paired_origins": len(valid), "first_origin": valid.origin.min().strftime("%Y-%m-%d"),
                  "last_origin": valid.origin.max().strftime("%Y-%m-%d"), "calendar_months_in_bootstrap": len(calendar),
                  "missing_calendar_months_preserved": int(calendar.loss_improvement.isna().sum()),
                  "paired_mse_improvement_95_interval": [lower, upper], "required_year_checks": year_checks,
                  "conditions": conditions, "prediction_increment_gate_pass": all(conditions.values()),
                  "groups_are_diagnostics_not_alternative_selection_rules": True}
    return evaluation, pd.DataFrame(groups), calendar, indices, pd.DataFrame({"replicate": range(len(draws)), "mse_improvement": draws})


def accounts(data, dividends, samples, predictions, config):
    mask = np.zeros(len(data), dtype=bool)
    mask[samples.origin_index.to_numpy(int)] = True
    arrays = {name: np.full(len(data), np.nan) for name in ("M0", "M1")}
    for name in arrays:
        arrays[name][predictions.origin_index.to_numpy(int)] = predictions[name].to_numpy(float)
    metrics, checks, uncertainty, draw_frames = [], [], {}, []
    for cost_name, cost in config["costs"].items():
        ledgers = {}
        for name in ("M0", "M1", "BUY_HOLD"):
            ledger, decisions = simulate_event_account(data, dividends, config, cost, config["evaluation_start"], name,
                prediction=arrays.get(name), horizon=config["horizon"], event_mask=mask)
            if ledger.terminal_unliquidated.iloc[-1]:
                raise ValueError("冻结终点无法完整清算")
            if name != "BUY_HOLD" and decisions.loc[~decisions.origin_index.isin(np.flatnonzero(mask)), "requested_quantity"].ne(0).any():
                raise ValueError("月末时钟之外产生新订单")
            save_account(OUT / "accounts" / cost_name, name, ledger, decisions)
            ledgers[name] = ledger
            metric = {"cost": cost_name, "model": name, **summarize(ledger, config)}
            metric["joint_point_target"] = bool(metric["net_sharpe"] is not None and metric["net_sharpe"] >= 1.2 and metric["annualized_return"] >= .10)
            metrics.append(metric)
            checks.append({"cost": cost_name, "model": name, "daily_rows": len(ledger), "max_accounting_error": float(ledger.accounting_error.abs().max()), "fully_liquidated": True})
        indices = block_draw_indices(len(ledgers["M1"]), config["account_bootstrap_day_block"], config["account_bootstrap_repetitions"], config["prediction_gate"]["random_seed"])
        np.savez_compressed(OUT / f"{cost_name}_account_bootstrap_indices.npz", indices=indices)
        differences = ledgers["M1"].net_return.to_numpy() - ledgers["M0"].net_return.to_numpy()
        draws = differences[indices].mean(axis=1) * config["annual_days"]
        draw_frames.append(pd.DataFrame({"cost": cost_name, "replicate": range(len(draws)), "annualized_mean_return_increment": draws}))
        uncertainty[cost_name] = {"M1_minus_M0_annualized_mean_daily_return_95_interval": np.quantile(draws, [.025, .975]).tolist()}
    pd.DataFrame(metrics).to_csv(OUT / "account_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(draw_frames, ignore_index=True).to_parquet(OUT / "account_bootstrap_draws.parquet", index=False)
    save(OUT / "account_execution_checks.json", {"rows": checks})
    return {"status": "SIX_COMPLETE_SIMULATION_ACCOUNTS_COMPUTED", "new_accounts_generated": 6, "metrics": metrics, "uncertainty": uncertainty,
            "stress_M1_joint_point_target": next(x["joint_point_target"] for x in metrics if x["cost"] == "STRESS" and x["model"] == "M1")}


def run():
    config = check_frozen()
    save(OUT / "RUN_STARTED.json", {"started_at": now(), "freeze": identity(OUT / "protocol_freeze.json"), "new_candidate_run": 1})
    try:
        data, dividends = load_market(config)
        monthly = source_features(config)
        samples = make_samples(monthly, data, dividends, config)
        predictions, receipts = rolling_predictions(samples, config)
        evaluation, groups, calendar, indices, draws = evaluate_predictions(predictions, config)
        for name, frame in [("market_features", data), ("samples", samples), ("predictions", predictions), ("prediction_groups", groups), ("prediction_calendar_loss", calendar), ("prediction_bootstrap_draws", draws)]:
            frame.to_parquet(OUT / (name + ".parquet"), index=False)
        predictions.to_csv(OUT / "全部月末预测与成熟状态.csv", index=False, encoding="utf-8-sig")
        groups.to_csv(OUT / "分时期与涨跌诊断.csv", index=False, encoding="utf-8-sig")
        save(OUT / "training_receipts.json", {"rows": receipts})
        np.savez_compressed(OUT / "prediction_bootstrap_indices.npz", indices=indices)
        save(OUT / "prediction_evaluation.json", evaluation)
        print(json.dumps({"status": "预测和固定增量检验完成", "fits": len(receipts), "evaluation": evaluation}, ensure_ascii=False), flush=True)
        if evaluation["prediction_increment_gate_pass"]:
            economics = accounts(data, dividends, samples, predictions, config)
            status = "HISTORICAL_INCREMENT_AND_ACCOUNTS_COMPUTED_INDEPENDENT_VALIDATION_NOT_ESTABLISHED"
        else:
            economics = {"status": "NOT_RUN_PREDICTION_GATE", "new_accounts_generated": 0, "metrics": None,
                         "reason": evaluation["conditions"], "stress_M1_joint_point_target": None}
            status = "REJECTED_FROZEN_NO_RELIABLE_GROWTH_DISAGREEMENT_INCREMENT"
        save(OUT / "account_stage_result.json", economics)
        result = {"study_id": config["study_id"], "completed_at": now(), "status": status,
                  "source_eligible_months": int(samples.all_features_valid.sum()), "recomputed_mature_return_labels": int(samples.Y60.notna().sum()),
                  "new_label_definition": False, "new_model_fits": len(receipts), "model_information_sets": 2, "new_candidates": 1,
                  "prediction_rows_including_no_view": len(predictions), "predicted_origins": int(predictions.M1.notna().sum()),
                  "predicted_but_not_yet_mature_origins": int((predictions.M1.notna() & predictions.Y60.isna()).sum()),
                  "evaluation": evaluation, "account_stage": economics, "new_accounts_generated": economics["new_accounts_generated"],
                  "new_network_requests": 0, "old_strategy_or_factor_modified": False,
                  "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "goal_achieved": False, "position_impact": 0}
        save(OUT / "result.json", result)
        print(json.dumps({"status": status, "fits": len(receipts), "accounts": economics["new_accounts_generated"], "goal_achieved": False}, ensure_ascii=False), flush=True)
    except Exception as exc:
        save(OUT / "PROGRAM_FAILURE.json", {"failed_at": now(), "status": "PROGRAM_FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise


def equal_frame(expected, path):
    actual = pd.read_parquet(path)
    for column in expected.columns:
        if pd.api.types.is_datetime64_any_dtype(expected[column]):
            expected[column] = expected[column].astype("datetime64[ns]")
            actual[column] = actual[column].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True), check_exact=True)


def verify(receipt_path):
    config = check_frozen()
    data, dividends = load_market(config)
    samples = make_samples(source_features(config), data, dividends, config)
    equal_frame(data, OUT / "market_features.parquet")
    equal_frame(samples, OUT / "samples.parquet")
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    receipts = read(OUT / "training_receipts.json")["rows"]
    largest_residual, largest_prediction_error = 0.0, 0.0
    for record in receipts:
        origin = pd.Timestamp(record["origin"])
        train = mature_training(samples, origin)
        if train.origin.dt.strftime("%Y-%m-%d").tolist() != record["training_origins"] or train.label_exit_date.max().strftime("%Y-%m-%d") != record["latest_training_label_exit"]:
            raise ValueError("保存模型的训练时钟或样本集合不符")
        if hashlib.sha256(train.Y60.to_numpy(dtype="<f8").tobytes()).hexdigest() != record["training_label_sha256"]:
            raise ValueError("训练标签身份改变")
        model = record["fit"]
        if model["columns"] != config["models"][record["model_id"]] or model["alpha"] != config["ridge_alpha"]:
            raise ValueError("模型列或正则化不符")
        x, y = train[model["columns"]].to_numpy(float), train.Y60.to_numpy(float)
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
        scale = np.where(scale == 0, 1.0, scale)
        np.testing.assert_allclose(mean, model["mean"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(scale, model["scale"], rtol=0, atol=1e-14)
        if abs(y.mean() - model["intercept"]) > 1e-14:
            raise ValueError("截距不等于训练标签均值")
        z = (x - mean) / scale
        residual = float(np.max(abs((z.T @ z + model["alpha"] * np.eye(len(mean))) @ np.asarray(model["slopes"]) - z.T @ (y - y.mean()))))
        largest_residual = max(largest_residual, residual)
        if residual > 1e-10:
            raise ValueError("保存系数不满足原岭回归方程")
        row = samples.loc[samples.origin.eq(origin)].iloc[0]
        value = predict(row, model)
        saved_value = float(predictions.loc[predictions.origin.eq(origin), record["model_id"]].iloc[0])
        error = max(abs(value - saved_value), abs(value - record["prediction"]))
        largest_prediction_error = max(largest_prediction_error, error)
        if error > 1e-14:
            raise ValueError("保存预测与模型不符")
    expected_count = 2 * int(predictions.M1.notna().sum())
    if len(receipts) != expected_count or predictions.M0.notna().tolist() != predictions.M1.notna().tolist():
        raise ValueError("两个信息集没有共同预测原点")
    indices = np.load(OUT / "prediction_bootstrap_indices.npz")["indices"]
    evaluation, groups, calendar, _, draws = evaluate_predictions(predictions, config, saved_indices=indices)
    if evaluation != read(OUT / "prediction_evaluation.json") or evaluation != read(OUT / "result.json")["evaluation"]:
        raise ValueError("保存增量结论无法复算")
    for frame, name in [(groups, "prediction_groups"), (calendar, "prediction_calendar_loss"), (draws, "prediction_bootstrap_draws")]:
        equal_frame(frame, OUT / (name + ".parquet"))
    economics = read(OUT / "account_stage_result.json")
    if not evaluation["prediction_increment_gate_pass"]:
        if economics["status"] != "NOT_RUN_PREDICTION_GATE" or economics["new_accounts_generated"] != 0 or (OUT / "accounts").exists():
            raise ValueError("未通过预测门却运行了账户")
    else:
        for metric in economics["metrics"]:
            ledger = pd.read_parquet(OUT / "accounts" / metric["cost"] / (metric["model"] + "_ledger.parquet"))
            recomputed = summarize(ledger, config)
            for key, value in recomputed.items():
                if isinstance(value, (int, float)) and value is not None:
                    if not np.isclose(value, metric[key], rtol=0, atol=1e-10):
                        raise ValueError("保存账户绩效不一致")
                elif value != metric[key]:
                    raise ValueError("保存账户状态不一致")
    save(receipt_path, {"status": "PASS_FROZEN_INPUTS_LABEL_CLOCKS_SAVED_MODELS_PREDICTIONS_AND_BOOTSTRAP_RECOMPUTED",
         "completed_at": now(), "saved_models_checked": len(receipts), "largest_normal_equation_residual": largest_residual,
         "maximum_prediction_error": largest_prediction_error, "bootstrap_replicates_checked": len(draws),
         "new_model_fits": 0, "new_random_draws": 0, "new_accounts": 0, "goal_achieved": False})
    print("冻结输入、标签时钟、保存系数、全部预测和固定重采样结果已复算通过。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["freeze", "run", "verify"], required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
    elif args.mode == "run":
        run()
    elif args.receipt is not None:
        verify(args.receipt)
    else:
        raise ValueError("离线复核必须指定新回执路径")
