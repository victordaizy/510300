"""公告日粒度的唯一信息增量、滚动预测与510300完整账户。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import build_features as etf_features, digest, normalize_dividends, normalize_prices, now, require, return_metrics, write_json
from research.if_open_interest_increment_v1 import five_day_label, rolling_predictions, simulate_account, mature_training, ridge_predict
from research.ah_premium_increment_v1 import block_indices, prediction_evaluation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/"config/510300_unlock_announcement_increment_v1.json"
PROTOCOL = ROOT/"docs/510300_UNLOCK_ANNOUNCEMENT_INCREMENT_V1_PROTOCOL.md"


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def outpath(config):
    return ROOT/config["output"]


def load_inputs(config, snapshot=False):
    base = outpath(config)/"frozen_inputs" if snapshot else ROOT
    paths = {k: base/v for k,v in config["inputs"].items()}
    price_receipt = json.loads(paths["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(paths["dividend_coverage"].read_text(encoding="utf-8"))
    source = json.loads(paths["source_completion"].read_text(encoding="utf-8"))
    admitted = json.loads(paths["source_admission"].read_text(encoding="utf-8"))
    membership = json.loads(paths["membership_admission"].read_text(encoding="utf-8"))
    require(price_receipt["status"] == "PASS" and digest(paths["prices"]) == price_receipt["canonical_price"]["output_sha256"], "ETF价格版本不符")
    require(coverage["complete_history_confirmed"] and coverage["coverage_end"] >= config["data_cutoff"], "分红覆盖不足")
    require(digest(paths["dividends"]) == coverage["distribution_file_sha256"], "分红版本不符")
    require(source["status"] == "PASS_COMPLETE_DAILY_DOCUMENT_METADATA_WITH_RECORDED_INTRADAY_TIMESTAMP_DRIFT" and source["complete_months"] == 140, "公告日来源未完整")
    require(digest(paths["announcements"]) == source["output_files"]["eligible_original_disclosures.parquet"]["sha256"], "合格公告归档版本不符")
    require(admitted["status"] == "PASS_SOURCE_METADATA_AND_PIT_MEMBERSHIP_DIAGNOSTIC_ONLY" and digest(paths["source_completion"]) == admitted["source_completion"]["sha256"], "公告准入回执不符")
    require(digest(paths["members"]) == admitted["membership_sha256"], "成员名单与准入版本不同")
    require(membership["membership_admission"]["status"] == "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION", "成员名单无既有准入")
    prices = normalize_prices(pd.read_parquet(paths["prices"]))
    prices = prices.loc[prices.date.le(config["data_cutoff"])].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(paths["dividends"]))
    calendar = pd.read_parquet(paths["calendar"])
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    dates = dates[(dates >= prices.date.iloc[0]) & (dates <= pd.Timestamp(config["data_cutoff"]))].sort_values()
    require(pd.DatetimeIndex(prices.date).equals(dates) and dates[-1] == pd.Timestamp(config["data_cutoff"]), "交易日历与行情不符")
    require(len(dividends) == coverage["event_count"], "分红事件数不符")
    announcements = pd.read_parquet(paths["announcements"])
    require(not announcements.announcementId.duplicated().any() and announcements.title_kind.eq("ELIGIBLE_ORIGINAL_DISCLOSURE").all(), "公告身份重复或混入其他标题类别")
    members = pd.read_parquet(paths["members"])[["membership_date", "symbol"]]
    members["membership_date"] = pd.to_datetime(members.membership_date)
    require(not members.duplicated().any() and members.groupby("membership_date").size().eq(300).all(), "逐日成员名单不符合300股合同")
    return {"prices": prices, "dividends": dividends, "announcements": announcements, "members": members}, {"source_provenance": {k: identity(p) for k,p in paths.items()}, "eligible_announcements": len(announcements), "membership_sessions": members.membership_date.nunique(), "source_status": source["status"], "historical_first_delivery_proven": False}


def announcement_density(dates, announcements, members, window, delay):
    """仅已可用公告进入过去窗口，当前成员资格逐行确定。"""
    dates = pd.DatetimeIndex(dates)
    origins = dates+pd.Timedelta(hours=15)
    member_codes = members.symbol.str.split(".").str[0]
    codes = pd.Index(sorted(member_codes.unique()))
    events = announcements[["secCode", "announcement_date"]].drop_duplicates().copy()
    events["available_at_assumed"] = pd.to_datetime(events.announcement_date)+pd.Timedelta(days=delay)
    events["available_origin_index"] = origins.searchsorted(pd.DatetimeIndex(events.available_at_assumed))
    event_codes = codes.get_indexer(events.secCode)
    event_indices = events.available_origin_index.to_numpy()
    relevant = (event_codes >= 0) & (event_indices < len(dates))
    active = np.zeros((len(dates), len(codes)), dtype=np.uint8)
    active[event_indices[relevant], event_codes[relevant]] = 1
    rolling = pd.DataFrame(active).rolling(window, min_periods=window).max().to_numpy()
    member_mask = np.zeros(active.shape, dtype=bool)
    member_indices = dates.get_indexer(members.membership_date)
    require((member_indices >= 0).all(), "成员日期越过特征日历")
    member_mask[member_indices, codes.get_indexer(member_codes)] = True
    count = np.nansum(rolling*member_mask, axis=1)
    valid_members = member_mask.sum(axis=1) == 300
    density = np.where(valid_members, count/300.0, np.nan)
    events["enters_feature_before_cutoff"] = relevant
    return density, count, events


def build_features(data, config):
    feature = etf_features(data["prices"], data["dividends"], 20)
    feature["ETF_R1"] = feature.total_simple
    for window in (5, 20):
        feature["ETF_R"+str(window)] = np.expm1(feature.total_log.rolling(window, min_periods=window).sum())
    month = feature.date.dt.month
    feature["CALENDAR_SIN"] = np.sin(2*np.pi*month/12)
    feature["CALENDAR_COS"] = np.cos(2*np.pi*month/12)
    density, count, events = announcement_density(feature.date, data["announcements"], data["members"], config["announcement_window_sessions"], config["announcement_calendar_day_delay"])
    feature["CSI300_UNLOCK_DENSITY20"] = density
    feature["ANN20_MEMBER_COUNT"] = count
    feature["decision_at"] = feature.date+pd.Timedelta(hours=15)
    first_possible = (pd.Timestamp(config["source_start"])+pd.Timedelta(days=config["announcement_calendar_day_delay"]))
    first_index = pd.DatetimeIndex(feature.decision_at).searchsorted(first_possible)
    warmed = np.arange(len(feature)) >= first_index+config["announcement_window_sessions"]-1
    finite = np.isfinite(feature[config["M0"]+config["M1_additions"]].to_numpy()).all(axis=1)
    feature["feature_status"] = np.where(warmed & finite, "PASS", "NO_VIEW_INPUT_NOT_AVAILABLE")
    feature["origin_index"] = np.arange(len(feature))
    feature["announcement_latest_date_allowed"] = feature.date-pd.Timedelta(days=config["announcement_calendar_day_delay"])
    return feature, events


def build_samples(feature, dividends, config, with_labels=False):
    cols = ["origin_index", "date", "decision_at", "feature_status", "ANN20_MEMBER_COUNT", "announcement_latest_date_allowed"]+config["M0"]+config["M1_additions"]
    rows = []
    for origin in np.flatnonzero(feature.date.ge(config["train_origin_start"])):
        entry, end = origin+1, origin+1+config["horizon"]
        row = feature.iloc[origin][cols].to_dict()
        row["origin"] = row.pop("date")
        row.update(entry_index=entry, exit_index=end, entry_date=feature.date.iloc[entry] if entry<len(feature) else pd.NaT, exit_date=feature.date.iloc[end] if end<len(feature) else pd.NaT, evaluation_origin=bool(entry<len(feature) and feature.date.iloc[entry]>=pd.Timestamp(config["evaluation_entry_start"])), label_status="MATURE_NOT_READ" if end<len(feature) else "CENSORED_AFTER_CUTOFF")
        if with_labels:
            row.update(five_day_label(feature, dividends, int(origin), config))
        rows.append(row)
    return pd.DataFrame(rows)


def preflight(config):
    out = outpath(config)
    require(not (out/"freeze_manifest.json").exists(), "已冻结，不能覆盖预检")
    data, source = load_inputs(config)
    feature, events = build_features(data, config)
    samples = build_samples(feature, data["dividends"], config)
    evaluation = samples.loc[samples.evaluation_origin]
    require(evaluation.feature_status.eq("PASS").all(), "评价日期存在缺少共同信息")
    first = evaluation.iloc[0]
    training = samples.loc[samples.exit_date.lt(first.origin) & samples.feature_status.eq("PASS")]
    require(len(training)>=config["minimum_train_rows"], "首模型成熟训练不足")
    start = int(first.entry_index)
    terminal = start+((len(feature)-1-start)//config["horizon"])*config["horizon"]
    report = {"study_id": config["study_id"], "status": "PASS_SOURCE_CLOCK_AND_SCHEDULE_BEFORE_NEW_LABELS", "created_at": now(), **source, "first_training_rows": len(training), "evaluation_origins": len(evaluation), "mature_evaluation_origins": int(evaluation.label_status.eq("MATURE_NOT_READ").sum()), "account_first": str(feature.date.iloc[start].date()), "account_terminal": str(feature.date.iloc[terminal].date()), "untraded_tail_days": len(feature)-1-terminal, "density_min": float(evaluation.CSI300_UNLOCK_DENSITY20.min()), "density_max": float(evaluation.CSI300_UNLOCK_DENSITY20.max()), "density_unique_values": int(evaluation.CSI300_UNLOCK_DENSITY20.nunique()), "new_return_labels": 0, "new_models": 0, "new_accounts": 0}
    write_json(out/"data_preflight.json", report, exclusive=True)
    samples.to_parquet(out/"preflight_schedule_without_labels.parquet", index=False)
    events.to_parquet(out/"announcement_availability_lineage.parquet", index=False)
    return report


def freeze(config):
    out = outpath(config)
    pre = json.loads((out/"data_preflight.json").read_text(encoding="utf-8"))
    require(pre["new_return_labels"] == 0 and not (out/"freeze_manifest.json").exists(), "冻结前已计算标签或记录已存在")
    snapshots = []
    for key, rel in config["inputs"].items():
        original, destination = ROOT/rel, out/"frozen_inputs"/rel
        require(digest(original) == pre["source_provenance"][key]["sha256"], "输入在预检后变化")
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "输入快照已存在")
        shutil.copyfile(original, destination)
        snapshots.append(identity(destination))
    protected = [CONFIG, PROTOCOL, Path(__file__), ROOT/config["authority"], ROOT/"research/if_open_interest_increment_v1.py", ROOT/"research/ah_premium_increment_v1.py", ROOT/"research/intraday_overnight_increment_v1.py", ROOT/"scripts/run_510300_unlock_announcement_increment_v1.py", ROOT/"tests/test_unlock_announcement_increment_v1.py"]
    protected += [out/name for name in ["USER_REQUEST.md", "deduplication.json", "pre_freeze_tests.json", "data_preflight.json", "preflight_schedule_without_labels.parquet", "announcement_availability_lineage.parquet"]]
    record = {"study_id": config["study_id"], "status": "FROZEN_BEFORE_NEW_LABELS_MODELS_AND_ACCOUNTS", "created_at": now(), "historical_runs_allowed": 1, "protected_files": [identity(p) for p in protected], "input_snapshots": snapshots, "new_models_at_freeze": 0, "new_accounts_at_freeze": 0}
    write_json(out/"freeze_manifest.json", record, exclusive=True)
    return record


def verify_frozen(config):
    record = json.loads((outpath(config)/"freeze_manifest.json").read_text(encoding="utf-8"))
    for item in record["protected_files"]+record["input_snapshots"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "冻结对象漂移："+item["path"])
    return record


def direction_groups(predictions, config):
    mature = predictions.loc[predictions.label_status.eq("MATURE")]
    rows = []
    for label, start, end in [["ALL", config["evaluation_entry_start"], config["data_cutoff"]]]+config["eras"]:
        part = mature.loc[mature.entry_date.between(start, end)]
        for model in ("M0", "M1"):
            positive = part["prediction_"+model] > 0
            actual_up, actual_down = part.Y5_NET_BASE.gt(0), part.Y5_NET_BASE.le(0)
            rows.append({"era": label, "model": model, "origins": len(part), "accuracy": float(positive.eq(actual_up).mean()), "up_recall": float(positive.loc[actual_up].mean()), "down_recall": float((~positive).loc[actual_down].mean()), "positive_prediction_count": int(positive.sum()), "negative_prediction_count": int((~positive).sum()), "positive_prediction_realized_mean": float(part.loc[positive, "Y5_NET_BASE"].mean()) if positive.any() else None, "negative_prediction_realized_mean": float(part.loc[~positive, "Y5_NET_BASE"].mean()) if (~positive).any() else None})
    return pd.DataFrame(rows)


def run(config):
    manifest = verify_frozen(config)
    out = outpath(config)
    write_json(out/"run_claim.json", {"started_at": now(), "pid": os.getpid(), "freeze_sha256": digest(out/"freeze_manifest.json")}, exclusive=True)
    try:
        data, source = load_inputs(config, snapshot=True)
        feature, events = build_features(data, config)
        samples = build_samples(feature, data["dividends"], config, with_labels=True)
        predictions, models = rolling_predictions(samples, config)
        predictions["prediction_VOL_ONLY"] = 1.0
        evaluation, groups, pred_indices, pred_draws = prediction_evaluation(predictions, config)
        for name, frame in [("features", feature), ("samples", samples), ("predictions", predictions), ("prediction_bootstrap_draws", pred_draws)]:
            frame.to_parquet(out/(name+".parquet"), index=False)
        write_json(out/"models.json", {"models": models})
        groups.to_csv(out/"prediction_comparison.csv", index=False, encoding="utf-8-sig")
        direction_groups(predictions, config).to_csv(out/"direction_comparison.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(out/"prediction_bootstrap_indices.npz", indices=pred_indices)
        accountdir = out/"accounts"
        accountdir.mkdir()
        accounts, ledgers, era_accounts = [], {}, []
        for cost_name, cost in config["costs"].items():
            for model in config["account_models"]:
                ledger, trades, decisions = simulate_account(feature, data["dividends"], predictions, cost, config, model)
                stem = cost_name+"_"+model
                ledger.to_parquet(accountdir/(stem+"_ledger.parquet"), index=False)
                trades.to_csv(accountdir/(stem+"_trades.csv"), index=False, encoding="utf-8-sig")
                decisions.to_parquet(accountdir/(stem+"_decisions.parquet"), index=False)
                metrics = return_metrics(ledger.net_return.to_numpy(), config["annual_days"])
                accounts.append({"cost": cost_name, "model": model, "first": ledger.date.iloc[0], "last": ledger.date.iloc[-1], "days": len(ledger), "trade_records": len(trades), "terminal_equity": ledger.equity.iloc[-1], "mean_exposure": ledger.exposure.mean(), "commission": ledger.commission.sum(), "slippage_cost": ledger.slippage_cost.sum(), **metrics})
                for name, start, end in config["eras"]:
                    part = ledger.loc[ledger.date.between(start, end)]
                    era_accounts.append({"cost": cost_name, "model": model, "era": name, "days": len(part), **return_metrics(part.net_return.to_numpy(), config["annual_days"])})
                ledgers[stem] = ledger
        pd.DataFrame(accounts).to_csv(out/"account_comparison.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(era_accounts).to_csv(out/"account_era_comparison.csv", index=False, encoding="utf-8-sig")
        indices = block_indices(len(ledgers["BASE_M0"]), config, seed=config["random_seed"]+1)
        np.savez_compressed(out/"account_bootstrap_indices.npz", indices=indices)
        economics, draws, era_rows = {}, {}, []
        for cost_name in config["costs"]:
            economics[cost_name] = {}
            m1 = ledgers[cost_name+"_M1"]
            for reference in ("M0", "VOL_ONLY"):
                ref = ledgers[cost_name+"_"+reference]
                require(ref.date.equals(m1.date), "配对账户日历不符")
                delta = (m1.net_return-ref.net_return).to_numpy()
                values = delta[indices].mean(axis=1)
                draws[cost_name+"_M1_MINUS_"+reference] = values
                economic = {"mean_daily_increment": delta.mean(), "annualized_arithmetic_increment": delta.mean()*config["annual_days"], "paired_daily_increment_95pct_ci": np.quantile(values, [.025, .975]), "exposure_neutral_alpha": False}
                positive = 0
                for name, start, end in config["eras"]:
                    mask = m1.date.between(start, end).to_numpy()
                    point = float(delta[mask].mean())
                    positive += point > 0
                    era_rows.append({"cost": cost_name, "reference": reference, "era": name, "mean_daily_increment": point})
                economic["positive_eras"] = positive
                economics[cost_name][reference] = economic
        pd.DataFrame(draws).to_parquet(out/"account_bootstrap_draws.parquet", index=False)
        pd.DataFrame(era_rows).to_csv(out/"economic_era_comparison.csv", index=False, encoding="utf-8-sig")
        stress = next(a for a in accounts if a["cost"] == "STRESS" and a["model"] == "M1")
        pred_gate = evaluation["predictive_increment_gate_pass"] and stress["annualized_return"]>0 and economics["STRESS"]["M0"]["mean_daily_increment"]>0
        era_frame = pd.DataFrame(era_rows)
        eras_both = era_frame.loc[era_frame.cost.eq("STRESS")].pivot(index="era", columns="reference", values="mean_daily_increment").gt(0).all(axis=1).sum()
        economic_gate = all(economics["STRESS"][ref]["paired_daily_increment_95pct_ci"][0]>0 for ref in ("M0", "VOL_ONLY")) and eras_both>=2
        continuation = bool(pred_gate or economic_gate)
        target = stress["annualized_return"]>=config["goal"]["net_cagr_at_least"] and stress["net_sharpe"] is not None and stress["net_sharpe"]>=config["goal"]["net_sharpe_at_least"]
        result = {"study_id": config["study_id"], "status": "HISTORICAL_UNLOCK_INFORMATION_CANDIDATE_ONLY" if continuation else "REJECTED_FROZEN_NO_RELIABLE_UNLOCK_ANNOUNCEMENT_INCREMENT", "completed_at": now(), "freeze_created_at": manifest["created_at"], "evaluation": evaluation, "economic_increment": economics, "accounts": accounts, "model_fits": len(models), "refit_months": len(models)//2, "new_accounts": len(accounts), "prediction_continuation_gate": bool(pred_gate), "economic_continuation_gate": bool(economic_gate), "stress_positive_eras_vs_both_controls": int(eras_both), "continuation_gate_pass": continuation, "historical_stress_point_target_pass": bool(target), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "historical_first_delivery_timestamp_proven": False, "historical_availability_assumption": config["historical_availability_assumption"], "researcher_selection_bias_corrected": False, "post_result_parameter_rescue": False, "network_calls_by_runner": 0, "position_impact": 0}
        write_json(out/"result.json", result, exclusive=True)
        files = [p for p in out.rglob("*") if p.is_file() and "frozen_inputs" not in p.relative_to(out).parts and p.name not in {"run_claim.json", "execution_receipt.json"}]
        write_json(out/"execution_receipt.json", {"status": "COMPLETED_ONCE", "completed_at": now(), "files": [identity(p) for p in sorted(files)], "new_model_fits": len(models), "new_accounts": len(accounts), "new_historical_runs": 1}, exclusive=True)
        return result
    except Exception as error:
        write_json(out/"failure_receipt.json", {"status": "PROGRAM_FAILED_CLAIM_PRESERVED", "at": now(), "error_type": type(error).__name__, "error": str(error)}, exclusive=True)
        raise


def verify_saved(config):
    verify_frozen(config)
    out = outpath(config)
    receipt = json.loads((out/"execution_receipt.json").read_text(encoding="utf-8"))
    for item in receipt["files"]:
        require(digest(ROOT/item["path"]) == item["sha256"], "保存文件变化")
    result = json.loads((out/"result.json").read_text(encoding="utf-8"))
    data, _ = load_inputs(config, snapshot=True)
    recreated, _ = build_features(data, config)
    feature = pd.read_parquet(out/"features.parquet")
    require(np.allclose(recreated.CSI300_UNLOCK_DENSITY20, feature.CSI300_UNLOCK_DENSITY20, equal_nan=True, atol=0, rtol=0), "公告时钟及当前成员特征重算不同")
    samples = pd.read_parquet(out/"samples.parquet")
    predictions = pd.read_parquet(out/"predictions.parquet")
    models = json.loads((out/"models.json").read_text(encoding="utf-8"))["models"]
    lookup = {model["model_key"]: model for model in models}
    for model in models:
        train = mature_training(samples, pd.Timestamp(model["refit_origin"]))
        require(len(train) == model["train_rows"] and hashlib.sha256(train.origin_index.to_numpy(np.int64).tobytes()).hexdigest() == model["train_origin_sha256"], "成熟训练原点集合不符")
        require(pd.Timestamp(model["train_last_exit"])<pd.Timestamp(model["refit_origin"]), "训练时钟越界")
    errors = []
    for row in predictions.to_dict("records"):
        require(row["announcement_latest_date_allowed"]+pd.Timedelta(days=config["announcement_calendar_day_delay"])<=row["decision_at"]<row["entry_date"]+pd.Timedelta(hours=9, minutes=30), "公告、决定、执行时钟不符")
        for model in ("M0", "M1"):
            errors.append(abs(ridge_predict(row, lookup[row["model_key_"+model]], config)-row["prediction_"+model]))
    require(max(errors)<1e-12, "预测与保存模型不一致")
    mature = predictions.loc[predictions.label_status.eq("MATURE")]
    loss = ((mature.Y5_NET_BASE-mature.prediction_M0)**2-(mature.Y5_NET_BASE-mature.prediction_M1)**2).to_numpy()
    indices = np.load(out/"prediction_bootstrap_indices.npz")["indices"]
    draws = pd.read_parquet(out/"prediction_bootstrap_draws.parquet").mse_improvement.to_numpy()
    require(np.allclose(loss[indices].mean(axis=1), draws, atol=1e-16, rtol=0), "保存预测区块不可重算")
    require(np.allclose(np.quantile(draws,[.025,.975]),result["evaluation"]["paired_mse_improvement_95pct_ci"],atol=1e-16,rtol=0), "预测区间不同")
    ledgers = {}
    for item in result["accounts"]:
        stem = item["cost"]+"_"+item["model"]
        ledger = pd.read_parquet(out/"accounts"/(stem+"_ledger.parquet"))
        require(ledger.accounting_error.abs().max()<1e-6 and ledger.shares.iloc[-1]==0, "账户守恒或清算不符")
        require(np.allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,ledger.equity,atol=1e-7,rtol=0), "账户资产合计不符")
        for key,value in return_metrics(ledger.net_return.to_numpy(),config["annual_days"]).items():
            require(value is None and item[key] is None or value is not None and np.isclose(value,item[key],atol=1e-12), "账户指标重算不同")
        ledgers[stem] = ledger
    indices = np.load(out/"account_bootstrap_indices.npz")["indices"]
    saved_draws = pd.read_parquet(out/"account_bootstrap_draws.parquet")
    for cost in config["costs"]:
        for ref in ("M0", "VOL_ONLY"):
            delta = (ledgers[cost+"_M1"].net_return-ledgers[cost+"_"+ref].net_return).to_numpy()
            values = delta[indices].mean(axis=1)
            require(np.allclose(values,saved_draws[cost+"_M1_MINUS_"+ref],atol=1e-16,rtol=0), "账户配对区块重算不同")
            require(np.allclose(np.quantile(values,[.025,.975]),result["economic_increment"][cost][ref]["paired_daily_increment_95pct_ci"],atol=1e-16,rtol=0), "账户增量区间不同")
    verified = {"status": "PASS_SAVED_ANNOUNCEMENT_CLOCK_FEATURE_MODEL_ACCOUNT_RECOMPUTATION", "verified_at": now(), "models_checked": len(models), "prediction_rows": len(predictions), "max_prediction_error": max(errors), "accounts_checked": len(ledgers), "new_labels": 0, "new_model_fits": 0, "new_accounts": 0, "new_random_samples": 0, "network_calls": 0}
    write_json(out/"saved_verification_receipt.json",verified)
    return verified


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["preflight","freeze","run","verify","status"])
    mode=parser.parse_args().mode
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    if mode=="status":
        path=outpath(config)/"result.json"
        value=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status":"NOT_RUN"}
    else:
        value={"preflight":preflight,"freeze":freeze,"run":run,"verify":verify_saved}[mode](config)
    from research.intraday_overnight_increment_v1 import safe_json
    print(json.dumps(safe_json(value),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
