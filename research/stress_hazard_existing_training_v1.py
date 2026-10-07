"""使用已有修补样本实际训练压力危险率；仅作探索，不生成交易概率或订单。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

import sparse_event_training_core_v3 as io
import stress_hazard_local_logit_v1 as logistic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_stress_hazard_existing_training_v1"


def freeze(out):
    io.require(not (out / "freeze.json").exists(), "本轮已冻结，不覆盖。")
    copies = {
        "samples.parquet": "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/g1_sample_eligibility.parquet",
        "market.parquet": "data/raw/market/510300_daily_2015_v2.parquet",
        "dividends.csv": "data/reference/510300_dividends.csv",
        "parent_protocol.yaml": "config/510300_stress_transmission_hazard_v2.yaml",
        "parent_quarter_protocol.yaml": "config/510300_stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1.yaml",
        "parent_model_design.yaml": "config/510300_stress_transmission_hazard_v2_g2_counterfactual_v1.yaml",
        "parent_terminal_status.json": "reports/research/510300_stress_transmission_hazard_v2_g1b_historical_final_status_v1.json",
        "parent_remediation_receipt.json": "reports/audit/510300_stress_transmission_hazard_v2_g1_historical_remediation_build_v1.json",
        "parent_version_lock.json": "config/510300_stress_transmission_hazard_v2_g0_1_post_remediation_version_lock_v1_manifest.json",
        "parent_era_audit.csv": "reports/research/510300_stress_transmission_hazard_v2_g1b_prequential_era_audit_v1.csv",
        "parent_event_audit.csv": "reports/research/510300_stress_transmission_hazard_v2_g1b_prequential_event_audit_v1.csv",
        "mandate.json": "config/510300_existing_data_training_mandate_v1.json",
    }
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    for name, relative in copies.items():
        shutil.copy2(ROOT / relative, out / "inputs" / name)
    io.require(io.digest(out / "inputs/samples.parquet") == "ba0b9ded2ef60e517a673863bf30da717ad0c2819dca07ce9d36915cefe76e25", "既有修补样本身份变化。")
    for source in [Path(__file__), Path(io.__file__), Path(logistic.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    for relative in ["research/stress_transmission_hazard_v2_g2_counterfactual_v1.py", "research/stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1.py"]:
        source = ROOT / relative
        shutil.copy2(source, out / "code" / ("parent_reference_" + source.name))
    paths = [out / "protocol.json", *list((out / "inputs").iterdir()), *list((out / "code").iterdir())]
    io.save(out / "freeze.json", {"frozen_at": io.now(), "before_this_round_fits": True,
        "old_sample_counts_known": True, "historical_prices_seen_in_previous_research": True,
        "identities": {p.relative_to(out).as_posix(): io.digest(p) for p in paths}})
    print("已有输入与原模型规则已保存，开始探索训练。", flush=True)


def check_inputs(out):
    for name, expected in io.load(out / "freeze.json")["identities"].items():
        io.require(io.digest(out / name) == expected, f"冻结文件变化：{name}")


def prepare(out, cfg):
    full = pd.read_parquet(out / "inputs/samples.parquet").sort_values("origin_date").reset_index(drop=True)
    for col in ["origin_date", "entry_date", "horizon_end_date"]:
        full[col] = pd.to_datetime(full[col]).dt.normalize()
    full["event_id"] = full.event_id.fillna("").astype(str)
    io.require(not full.origin_date.duplicated().any(), "信息原点日期重复。")
    positives = full[full.bad10 == 1]
    events = positives.groupby("event_id", sort=True).agg(
        event_start=("origin_date", "min"), event_end=("horizon_end_date", "max"),
        eligible_origins=("b2_vs_b1_eligible", "sum")).reset_index()
    event_map = events.set_index("event_id")
    vintages = pd.DatetimeIndex(full.groupby(full.origin_date.dt.to_period("Q"), sort=True).head(1).origin_date)
    common = full[full.b2_vs_b1_eligible].copy().reset_index(drop=True)
    io.require(full.b2_vs_b1_eligible.equals(full.b3_vs_b2_eligible), "B2与B3共同样本不同；不得暗中换样本。")
    all_features = sorted({column for columns in cfg["models"].values() for column in columns})
    io.require(np.isfinite(common[all_features].to_numpy(float)).all(), "模型输入含缺失。")
    common["weight"] = common.b2_vs_b1_model_weight.astype(float)
    common["sample_id"] = common.origin_date.dt.strftime("%Y-%m-%d")
    common["event_start"] = common.event_id.map(event_map.event_start)
    common["event_end"] = common.event_id.map(event_map.event_end)
    common["assignment_date"] = common.origin_date.where(common.bad10 == 0, common.event_start)
    common["model_vintage"] = [vintages[vintages.searchsorted(day, side="right") - 1] for day in common.assignment_date]
    common["era"] = [era_for(day, cfg) for day in common.assignment_date]
    common["block_event"] = np.where(common.bad10 == 1, common.event_id, "NEG_" + common.sample_id)
    common["block_year"] = common.assignment_date.dt.year.astype(str)
    io.require((common.weight > 0).all(), "样本权重必须为正。")
    event_weight = common[common.bad10 == 1].groupby("event_id").weight.sum()
    io.require(np.allclose(event_weight, 1, atol=1e-12, rtol=0), "同一压力事件的权重总和不是1。")
    io.require((common.loc[common.bad10 == 0, "weight"] == 1).all(), "非事件日权重不同。")
    io.require(len(full) == 2813 and len(common) == 1279 and len(event_weight) == 31, "原样本计数不同。")
    return full, common, events, vintages


def era_for(day, cfg):
    day = str(pd.Timestamp(day).date())
    for era in cfg["eras"]:
        if era["start"] <= day <= era["end"]:
            return era["id"]
    raise ValueError(f"日期未落入固定时期：{day}")


def training_sample(common, vintage):
    closed_positive = (common.bad10 == 1) & (common.event_end < vintage)
    matured_negative = (common.bad10 == 0) & (common.horizon_end_date < vintage)
    return common[closed_positive | matured_negative].copy()


def train(common, vintages, cfg):
    snapshots, forecasts, vintage_rows = [], [], []
    for vintage in vintages:
        sample = training_sample(common, vintage)
        n_events = sample.loc[sample.bad10 == 1, "event_id"].nunique()
        n_negative = int((sample.bad10 == 0).sum())
        available = n_events >= 1 and n_negative >= 1
        vintage_rows.append({"vintage": str(vintage.date()), "training_rows": len(sample),
            "training_events": n_events, "training_negative_origins": n_negative, "trained": available})
        evaluation = common[common.model_vintage == vintage]
        for name, columns in cfg["models"].items():
            model = None
            if available:
                x, y, w = sample[columns].to_numpy(float), sample.bad10.to_numpy(float), sample.weight.to_numpy(float)
                weighted_rate = float(np.average(y, weights=w))
                if columns:
                    fit = logistic.fit_nonnegative_logistic(x, y, w, l2_penalty=cfg["l2_penalty"])
                    params = asdict(fit)
                else:
                    intercept = math.log(weighted_rate / (1 - weighted_rate))
                    params = {"intercept": intercept, "slopes": [], "objective": float(np.average(np.logaddexp(0, intercept) - y * intercept, weights=w)), "iterations": 0, "converged": True}
                model = {**params, "model": name, "columns": columns, "vintage": str(vintage.date()),
                    "training_sample_ids": sample.sample_id.tolist(), "training_events": n_events,
                    "maximum_training_label_end": str(sample.horizon_end_date.max().date()),
                    "weighted_training_rate": weighted_rate, "unweighted_training_origin_rate": float(y.mean()),
                    "model_type": "NONNEGATIVE_LOGIT" if columns else "WEIGHTED_BASE_RATE"}
                snapshots.append(model)
            for row in evaluation.itertuples():
                score = float(expit(model["intercept"] + np.asarray([getattr(row, c) for c in columns]) @ np.asarray(model["slopes"]))) if model else None
                forecasts.append({"model": name, "origin_date": str(row.origin_date.date()),
                    "entry_date": str(row.entry_date.date()), "horizon_end_date": str(row.horizon_end_date.date()),
                    "model_vintage": str(vintage.date()), "event_id": row.event_id, "bad10": int(row.bad10),
                    "weight": row.weight, "era": row.era, "block_event": row.block_event, "block_year": row.block_year,
                    "risk_score": score, "status": "PREQUENTIAL_SCORE" if model else "TRAINING_WARMUP_NO_SCORE"})
    return snapshots, pd.DataFrame(forecasts), pd.DataFrame(vintage_rows)


def metrics(predictions, cfg):
    rows = []
    for model in cfg["models"]:
        data = predictions[(predictions.model == model) & predictions.risk_score.notna()]
        for segment in ["ALL", *[e["id"] for e in cfg["eras"]]]:
            sample = data if segment == "ALL" else data[data.era == segment]
            row = {"model": model, "segment": segment, "origins": len(sample),
                "positive_events": sample.loc[sample.bad10 == 1, "event_id"].nunique(), "status": "EVALUATED" if len(sample) else "NO_PREDICTIONS"}
            if len(sample):
                y, p, w = sample.bad10.to_numpy(float), sample.risk_score.to_numpy(float), sample.weight.to_numpy(float)
                p = np.clip(p, 1e-12, 1 - 1e-12)
                loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
                row.update({"weighted_log_loss": float(np.average(loss, weights=w)),
                    "weighted_brier": float(np.average((p - y) ** 2, weights=w)),
                    "unweighted_log_loss": float(loss.mean()), "unweighted_brier": float(((p - y) ** 2).mean()),
                    "weighted_observed_rate": float(np.average(y, weights=w)), "unweighted_observed_origin_rate": float(y.mean()),
                    "weighted_average_score": float(np.average(p, weights=w)), "score_min": float(p.min()), "score_max": float(p.max()),
                    "weighted_roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else None,
                    "weighted_pr_auc": float(average_precision_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else None})
            rows.append(row)
    return pd.DataFrame(rows)


def compare(metrics_frame, cfg):
    comparisons = []
    for candidate, reference in cfg["comparisons"]:
        for segment in ["ALL", *[e["id"] for e in cfg["eras"]]]:
            a = metrics_frame[(metrics_frame.model == reference) & (metrics_frame.segment == segment)].iloc[0]
            b = metrics_frame[(metrics_frame.model == candidate) & (metrics_frame.segment == segment)].iloc[0]
            improvement = a.get("weighted_log_loss", np.nan) - b.get("weighted_log_loss", np.nan)
            comparisons.append({"candidate": candidate, "reference": reference, "segment": segment,
                "origins": b.origins, "positive_events": b.positive_events,
                "log_loss_improvement": improvement,
                "relative_log_loss_improvement": improvement / a.get("weighted_log_loss", np.nan)})
    return pd.DataFrame(comparisons)


def bootstrap(predictions, out, cfg):
    usable = predictions[predictions.risk_score.notna()].copy()
    p = usable.risk_score.clip(1e-12, 1 - 1e-12)
    usable["loss"] = -(usable.bad10 * np.log(p) + (1 - usable.bad10) * np.log1p(-p))
    losses = usable.pivot(index="origin_date", columns="model", values="loss")
    base = usable[usable.model == "B0"].set_index("origin_date").loc[losses.index]
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    distributions, summaries, count_files = [], [], {}
    for block_name in ["block_event", "block_year"]:
        groups = base.groupby(block_name, sort=True)
        ids = list(groups.groups)
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=cfg["bootstrap_repetitions"]).astype(np.uint16)
        count_files[block_name] = counts
        count_files[block_name + "_ids"] = np.array(ids)
        grouped_weight = groups.weight.sum().reindex(ids).to_numpy(float)
        for candidate, reference in cfg["comparisons"]:
            difference = (losses[reference] - losses[candidate]) * base.weight
            grouped_difference = difference.groupby(base[block_name]).sum().reindex(ids).to_numpy(float)
            estimates = (counts @ grouped_difference) / (counts @ grouped_weight)
            summaries.append({"candidate": candidate, "reference": reference, "unit": block_name,
                "blocks": len(ids), "repetitions": len(estimates), "q10": float(np.quantile(estimates, 0.10)),
                "q50": float(np.quantile(estimates, 0.50)), "q90": float(np.quantile(estimates, 0.90)),
                "positive_fraction": float((estimates > 0).mean())})
            distributions.extend({"candidate": candidate, "reference": reference, "unit": block_name,
                "replicate": i, "log_loss_improvement": value} for i, value in enumerate(estimates))
    (out / "results").mkdir(exist_ok=True)
    np.savez_compressed(out / "results/固定重抽样计数.npz", **count_files)
    return pd.DataFrame(summaries), pd.DataFrame(distributions)


def check_labels(out, full):
    market = pd.read_parquet(out / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.normalize()
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends[dividends.symbol == "510300.SH"].copy()
    dividends["date"] = pd.to_datetime(dividends.ex_date)
    cash = dividends.set_index("date").cash_dividend_per_share
    market["dividend"] = market.date.map(cash).fillna(0)
    calendar = pd.DatetimeIndex(market.date)
    for row in full.itertuples():
        i, j, k = [int(calendar.get_loc(getattr(row, c))) for c in ["origin_date", "entry_date", "horizon_end_date"]]
        io.require(j == i + 1 and k == i + 10, "原标签持有期限不是下一开盘起10交易日。")
        block = market.iloc[j:k + 1]
        entitlement = np.cumsum(np.r_[0, block.dividend.to_numpy(float)[1:]])
        path = (block.close.to_numpy(float) + entitlement) / float(block.open.iloc[0]) - 1
        io.require(int(path.min() <= -0.04) == int(row.bad10), "原BAD10标签与已有行情不符。")
    return len(full)


def run(out):
    io.require(not (out / "summary.json").exists(), "本轮已完成，不重复训练。")
    check_inputs(out)
    cfg = io.load(out / "protocol.json")
    full, common, events, vintages = prepare(out, cfg)
    checked = check_labels(out, full)
    snapshots, predictions, vintage_rows = train(common, vintages, cfg)
    io.save(out / "models/季度已训练模型.json", snapshots)
    last = {name: [m for m in snapshots if m["model"] == name][-1] for name in cfg["models"]}
    io.save(out / "models/末次季度模型引用.json", {name: {"vintage": m["vintage"], "status": "EXISTING_SNAPSHOT_NOT_NEW_FIT_NO_CURRENT_SIGNAL"} for name, m in last.items()})
    scores = metrics(predictions, cfg)
    comparisons = compare(scores, cfg)
    intervals, draws = bootstrap(predictions, out, cfg)
    for name, frame in {"共同训练样本": common, "全部压力事件": events, "季度训练计数": vintage_rows,
            "逐期风险分数": predictions, "预测指标": scores, "模型增量比较": comparisons,
            "重抽样区间": intervals, "固定重抽样结果": draws}.items():
        io.export(out / "results" / f"{name}.csv", frame)
    summary = {"study_id": cfg["study_id"], "completed_at": io.now(), "status": "EXPLORATORY_QUARTERLY_HAZARD_MODELS_TRAINED",
        "all_origins": len(full), "label_recomputed_origins": checked, "common_origins": len(common),
        "common_pressure_events": common.loc[common.bad10 == 1, "event_id"].nunique(),
        "trained_quarter_vintages": int(vintage_rows.trained.sum()),
        "logistic_fits": sum(m["model_type"] == "NONNEGATIVE_LOGIT" for m in snapshots),
        "weighted_base_rate_snapshots": sum(m["model_type"] == "WEIGHTED_BASE_RATE" for m in snapshots),
        "saved_parameter_snapshots": len(snapshots), "forecast_rows_including_warmup": len(predictions),
        "scored_origins_per_model": int(predictions[(predictions.model == "B0") & predictions.risk_score.notna()].shape[0]),
        "predicted_independent_events": predictions.loc[(predictions.model == "B0") & predictions.risk_score.notna() & (predictions.bad10 == 1), "event_id"].nunique(),
        "account_scenarios": 0, "net_sharpe": None, "max_drawdown": None,
        "account_status": "NOT_RUN_RISK_SCORE_IS_NOT_RETURN_OR_ENTRY_MODEL", "new_downloads": 0,
        "independent_forward_events": 0, "goal_achieved": False, "orders_authorized": False,
        "previous_goal_turn_classification": "PROGRESS_IF_EXHAUSTION_TRAINING_AND_ACCOUNTS_COMPLETED"}
    io.save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(comparisons.to_string(index=False))
    print(intervals.to_string(index=False))


def verify(out):
    check_inputs(out)
    cfg = io.load(out / "protocol.json")
    full, common, events, vintages = prepare(out, cfg)
    check_labels(out, full)
    models = io.load(out / "models/季度已训练模型.json")
    predictions = pd.read_csv(out / "results/逐期风险分数.csv", dtype={"block_year": str}).fillna({"event_id": ""})
    index = common.set_index("sample_id")
    max_gradient = 0.0
    for model in models:
        sample = training_sample(common, pd.Timestamp(model["vintage"]))
        io.require(sample.sample_id.tolist() == model["training_sample_ids"], "训练集合与成熟时钟不一致。")
        x, y, w = sample[model["columns"]].to_numpy(float), sample.bad10.to_numpy(float), sample.weight.to_numpy(float)
        slopes = np.asarray(model["slopes"])
        linear = model["intercept"] + x @ slopes
        probability = expit(linear)
        residual = w * (probability - y) / w.sum()
        gradient = np.r_[residual.sum(), x.T @ residual + slopes]
        projected = gradient.copy()
        if len(slopes):
            projected[1:] = np.where(slopes > 1e-9, gradient[1:], np.minimum(gradient[1:], 0))
        max_gradient = max(max_gradient, float(np.abs(projected).max()))
        io.require(np.abs(projected).max() < 1e-6, "保存参数未满足固定受约束优化的一阶条件。")
        objective = float(np.average(np.logaddexp(0, linear) - y * linear, weights=w) + 0.5 * np.dot(slopes, slopes))
        io.require(abs(objective - model["objective"]) < 1e-10, "保存目标函数不同。")
        selected = predictions[(predictions.model == model["model"]) & (predictions.model_vintage == model["vintage"])]
        if len(selected):
            features = index.loc[selected.origin_date, model["columns"]].to_numpy(float)
            expected = expit(model["intercept"] + features @ slopes)
            io.require(np.allclose(expected, selected.risk_score, atol=1e-12, rtol=0), "保存参数不能复现风险分数。")
    scored = predictions[predictions.risk_score.notna()]
    positive = scored[(scored.model == "B0") & (scored.bad10 == 1)]
    io.require((positive.groupby("event_id").model_vintage.nunique() == 1).all(), "一个压力事件跨模型版本。")
    expected_scores = metrics(predictions, cfg)
    saved_scores = pd.read_csv(out / "results/预测指标.csv")
    pd.testing.assert_frame_equal(expected_scores, saved_scores, check_dtype=False, rtol=1e-10, atol=1e-12)
    saved_comparisons = pd.read_csv(out / "results/模型增量比较.csv")
    pd.testing.assert_frame_equal(compare(expected_scores, cfg), saved_comparisons, check_dtype=False, rtol=1e-10, atol=1e-12)
    usable = scored.copy()
    p = usable.risk_score.clip(1e-12, 1 - 1e-12)
    usable["loss"] = -(usable.bad10 * np.log(p) + (1 - usable.bad10) * np.log1p(-p))
    losses = usable.pivot(index="origin_date", columns="model", values="loss")
    base = usable[usable.model == "B0"].set_index("origin_date").loc[losses.index]
    draws = pd.read_csv(out / "results/固定重抽样结果.csv")
    with np.load(out / "results/固定重抽样计数.npz", allow_pickle=False) as saved:
        for unit in ["block_event", "block_year"]:
            ids, counts = saved[unit + "_ids"], saved[unit]
            io.require(np.all(counts.sum(axis=1) == len(ids)), "固定重抽样组数不同。")
            weights = base.groupby(unit).weight.sum().reindex(ids).to_numpy(float)
            for candidate, reference in cfg["comparisons"]:
                component = ((losses[reference] - losses[candidate]) * base.weight).groupby(base[unit]).sum().reindex(ids).to_numpy(float)
                expected = (counts @ component) / (counts @ weights)
                actual = draws[(draws.unit == unit) & (draws.candidate == candidate) & (draws.reference == reference)].sort_values("replicate").log_loss_improvement
                io.require(np.allclose(expected, actual, atol=1e-12, rtol=0), "固定重抽样保存结果不能对账。")
    return {"status": "PASS_SAVED_HAZARD_LABELS_MODELS_PREDICTIONS_AND_RESAMPLES", "label_rows": len(full),
        "parameter_snapshots": len(models), "max_projected_gradient": max_gradient,
        "scored_prediction_rows": len(scored), "bootstrap_result_rows": len(draws),
        "new_model_fits": 0, "new_accounts": 0, "new_random_samples": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="已有压力危险率样本的实际探索训练。")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "train": run}[args.command](args.root)


if __name__ == "__main__":
    main()
