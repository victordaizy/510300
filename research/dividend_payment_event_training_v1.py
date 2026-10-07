"""已有分红现金发放事件的五日探索训练；仅交易510300和现金。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

import sparse_event_training_core_v3 as core
import dividend_payment_training_helpers_v1 as helper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_dividend_payment_event_training_v1"


def freeze(out):
    core.require(not (out / "freeze.json").exists(), "本轮输入已固定，不覆盖。")
    (out / "inputs/official").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    copies = {
        "market.parquet": "reports/research/510300_private_manager_intent_v1/inputs/market.parquet",
        "dividends.csv": "data/reference/510300_dividends.csv",
        "dividend_coverage.json": "data/reference/510300_dividends_coverage.json",
        "prior_yield_funding_config.json": "config/510300_cash_distribution_funding_v1.json",
        "prior_yield_funding_result.json": "reports/research/510300_cash_distribution_funding_v1/result.json",
        "prior_calendar_result.json": "reports/research/510300_calendar_liquidity_timing_v1/result.json",
        "mandate.json": "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, relative in copies.items():
        shutil.copy2(ROOT / relative, out / "inputs" / name)
    coverage = core.load(out / "inputs/dividend_coverage.json")
    core.require(core.digest(out / "inputs/dividends.csv") == coverage["distribution_file_sha256"], "已有分红表与来源回执不同。")
    sources = []
    for row in coverage["official_source_snapshots"]:
        source = ROOT / row["saved_file"]
        core.require(core.digest(source) == row["sha256"], "已存分红原文发生变化。")
        destination = out / "inputs/official" / source.name
        shutil.copy2(source, destination)
        sources.append({**row, "package_path": destination.relative_to(out).as_posix()})
    core.save(out / "inputs/official_sources.json", sources)
    for source in [Path(__file__), Path(core.__file__), Path(helper.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    files = [out / "protocol.json", *[p for p in (out / "inputs").rglob("*") if p.is_file()], *list((out / "code").iterdir())]
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_this_round_event_labels_models_and_accounts": True,
        "historical_prices_previously_seen": True, "payment_dates_previously_known": True,
        "identities": {p.relative_to(out).as_posix(): core.digest(p) for p in files}})
    print("14次既有分红、原文和固定五日规则已保存，开始训练。", flush=True)


def check_inputs(out):
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"固定输入变化：{name}")


def make_events(out, cfg):
    market = core.read_market(out)
    market = market[market.date <= cfg["account_end"]].reset_index(drop=True)
    market["past5"] = np.expm1(np.log1p(market.return1).rolling(5).sum())
    calendar = pd.DatetimeIndex(market.date)
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends[dividends.symbol == "510300.SH"].sort_values("payment_date")
    rows = []
    for row in dividends.itertuples():
        payment = pd.Timestamp(row.payment_date)
        decision_i = int(calendar.searchsorted(payment, side="right")) - 1
        entry_i = decision_i + 1
        exit_i = entry_i + cfg["horizon_sessions"] - 1
        ex_i = int(calendar.get_loc(pd.Timestamp(row.ex_date)))
        core.require(ex_i > 0 and exit_i < len(market), "既有分红事件缺少完整行情。")
        core.require(calendar[entry_i] > payment and payment >= pd.Timestamp(row.ex_date), "现金发放与入场时钟错误。")
        directory_dates = re.findall(r"/(\d{4}-\d{2}-\d{2})/", row.source)
        filename_dates = re.findall(r"_(\d{8})_", row.source)
        source_days = [pd.Timestamp(value) for value in directory_dates]
        source_days.extend(pd.to_datetime(value, format="%Y%m%d") for value in filename_dates)
        core.require(bool(source_days), "已有来源缺少可定位的公告日期。")
        source_date = max(source_days)
        core.require(source_date < calendar[decision_i], "公告日期必须早于决策日。")
        known = market.iloc[decision_i]
        cum_dividend_close = float(market.close.iloc[ex_i - 1])
        block = market.iloc[entry_i:exit_i + 1]
        target5 = (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1
        rows.append({"month": row.payment_date, "payment_date": row.payment_date, "record_date": row.record_date,
            "ex_date": row.ex_date, "source_publication_day_proxy": str(source_date.date()), "source": row.source,
            "cash_dividend_per_share": row.cash_dividend_per_share,
            "decision_at": (calendar[decision_i].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=1)).isoformat(),
            "entry_i": entry_i, "exit_i": exit_i, "entry_date": str(calendar[entry_i].date()), "exit_date": str(calendar[exit_i].date()),
            "label_end_at": (calendar[exit_i].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).isoformat(),
            "reference_close": float(known.close), "vol20": float(known.vol20), "past5": float(known.past5), "logvol20": float(known.logvol20),
            "dividend_yield": float(row.cash_dividend_per_share) / cum_dividend_close,
            "ex_to_payment_total_return": (float(known.close) + float(row.cash_dividend_per_share)) / cum_dividend_close - 1,
            "target5": target5, "own_right_to_this_dividend": False})
    frame = pd.DataFrame(rows)
    core.require(len(frame) == 14 and np.isfinite(frame[["past5", "logvol20", "dividend_yield", "ex_to_payment_total_return", "target5"]].to_numpy()).all(), "事件范围或特征缺失。")
    return frame, market


def run(out):
    core.require(not (out / "summary.json").exists(), "本轮已完成，不重复运行。")
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    events, market = make_events(out, cfg)
    predictions, models, final = helper.train(events, market, cfg)
    scores, comparisons = helper.score(predictions)
    for name, frame in {"全部事件与五日标签": events, "逐期预测": predictions,
            "预测指标": scores, "增量比较": comparisons}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    core.save(out / "models/逐期模型.json", models)
    core.save(out / "models/期末已训练模型.json", final)
    start_i = int(predictions.entry_i.min())
    accounts = helper.run_accounts(out, cfg, events, market, predictions, start_i)
    passing = []
    for (model, policy), group in accounts[accounts.policy != "BENCHMARK"].groupby(["model", "policy"]):
        if group.meets_numeric_targets.all():
            passing.append({"model": model, "policy": policy})
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(), "status": "EXPLORATORY_DIVIDEND_PAYMENT_EVENT_TRAINED",
        "all_events": len(events), "warmup_events": cfg["min_train"], "predicted_events_per_model": len(predictions) // len(cfg["models"]),
        "prequential_ridge_fits": sum(m["model_type"] == "RIDGE" for m in models),
        "prequential_mean_updates": sum(m["model_type"] == "MATURE_MEAN" for m in models),
        "final_ridge_models": 2, "final_mean_references": 1, "prediction_rows": len(predictions), "account_scenarios": len(accounts),
        "account_start": str(market.date.iloc[start_i].date()), "account_end": cfg["account_end"],
        "complete_calendar_sessions": len(market) - start_i, "numeric_targets_pass_both_costs": passing,
        "goal_achieved": False, "independent_forward_events": 0, "new_downloads": 0, "orders_authorized": False,
        "previous_goal_turn_classification": "PROGRESS_HAZARD_MODELS_TRAINED_AND_INCREMENT_REJECTED",
        "conclusion": "分红发放后固定五日事件完成实际训练和完整账户。" + ("存在历史数值达线组合，尚不构成独立有效证据。" if passing else "没有组合同时达到两档成本下净夏普1.5、最大回撤不超过10%。")}
    core.save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(scores.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(accounts[["model", "policy", "scenario", "net_sharpe", "max_drawdown", "opportunities", "ending_equity_cny"]].to_string(index=False))


def verify(out):
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    events, market = make_events(out, cfg)
    pd.testing.assert_frame_equal(events, pd.read_csv(out / "results/全部事件与五日标签.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    predictions = pd.read_csv(out / "results/逐期预测.csv")
    models = core.load(out / "models/逐期模型.json")
    final = core.load(out / "models/期末已训练模型.json")
    indexed = events.set_index("month")
    for model in [*models, *final.values()]:
        sample = indexed.loc[model["train_months"]]
        limit = model.get("decision_at", model.get("as_of"))
        core.require((sample.label_end_at < limit).all(), "训练使用了未成熟标签。")
        x, y = sample[model["columns"]].to_numpy(float), sample.target5.to_numpy(float)
        center, scale, beta = [np.asarray(model[k]) for k in ["center", "scale", "coefficients_standardized"]]
        raw_scale = x.std(axis=0, ddof=0)
        core.require(np.allclose(center, x.mean(axis=0), rtol=0, atol=1e-12), "训练中心不同。")
        core.require(np.allclose(scale, np.where(raw_scale >= 1e-12, raw_scale, 1), rtol=0, atol=1e-12), "训练标准化不同。")
        core.require(abs(model["intercept"] - y.mean()) < 1e-12, "训练成熟均值不同。")
        z = (x - center) / scale
        z[:, ~np.asarray(model["active"], bool)] = 0
        core.require(np.allclose((z.T @ z / len(sample) + cfg["ridge_mean_loss_penalty"] * np.eye(len(beta))) @ beta,
            z.T @ (y - model["intercept"]) / len(sample), atol=1e-12, rtol=0), "保存岭参数不满足原方程。")
        if "prediction_month" in model:
            row = predictions[(predictions.model == model["model"]) & (predictions.month == model["prediction_month"])].iloc[0]
            core.require(abs(core.predict(model, indexed.loc[row.month]) - row.prediction) < 1e-12, "保存参数不能复现预测。")
            errors = predictions[(predictions.model == row.model) & (predictions.label_end_at < row.decision_at)]
            core.require(len(errors) == row.mature_prequential_error_count, "误差校准时钟错误。")
            if len(errors):
                core.require(abs(math.sqrt(float((errors.prediction - errors.target5).pow(2).mean())) - row.prior_prediction_rmse) < 1e-12, "过去预测误差不同。")
    scores, comparisons = helper.score(predictions)
    for name, frame in [("预测指标", scores), ("增量比较", comparisons)]:
        pd.testing.assert_frame_equal(frame, pd.read_csv(out / "results" / f"{name}.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    n_accounts = helper.verify_accounts(out, cfg, market)
    decisions = pd.read_csv(out / "results/逐事件交易判断.csv")
    core.require(decisions[decisions.model == "EVENT_ONLY"].shape[0] == 16, "事件对照必须按同一8个可预测事件评价两档成本。")
    return {"status": "PASS_SAVED_DIVIDEND_EVENTS_MODELS_AND_ACCOUNTS", "event_labels": len(events),
        "saved_models_and_means": len(models) + len(final), "prediction_rows": len(predictions),
        "account_scenarios": n_accounts, "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="已有分红现金发放事件实际训练。")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "train": run}[args.command](args.root)


if __name__ == "__main__":
    main()
