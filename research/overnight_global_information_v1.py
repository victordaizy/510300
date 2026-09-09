"""按实际跨时区收盘边界对齐免费海外信息，进行历史可行性检验。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.adaptive_allocation_v1 import ROOT, identity, simulate, save_account, summarize, eligible_training
from research.intraday_overnight_increment_v1 import (
    digest, now, write_json, require, holding_total_return, normalize_dividends,
    block_indices, return_metrics, interval,
)
import research.return_classification_v1 as classification

STUDY = "510300_OVERNIGHT_GLOBAL_INFORMATION_V1"
OUT = ROOT / "reports/research/510300_overnight_global_information_v1"
CONFIG = ROOT / "config/510300_overnight_global_information_v1.json"
MANIFEST = ROOT / "config/510300_overnight_global_information_v1_manifest.json"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
SOURCES = {
    "GSPC": ("data/raw/cross_market_chart_ml_v1/GSPC_daily.parquet", "US"),
    "IXIC": ("data/raw/cross_market_chart_ml_v1/IXIC_daily.parquet", "US"),
    "HSI": ("data/raw/cross_market_chart_ml_v1/HSI_daily.parquet", "ASIA_DELAY"),
    "N225": ("data/raw/cross_market_chart_ml_v1/N225_daily.parquet", "ASIA_DELAY"),
    "VIX": ("data/raw/us_china_overnight_v1/VIX_daily.parquet", "US"),
}
NAMES = {"D1_PRIMARY_EXTERNAL_H1": "主方案：盘后一日预测共识", "D2_PRICE_H1": "中国因子一日预测共识",
         "D3_EXTERNAL_ALL": "盘后全部期限共识", "D4_PRICE_ALL": "中国因子全部期限共识", "BUY_HOLD": "买入持有"}


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "第三轮已经冻结，禁止覆盖")
    config = json.loads((ROOT / "config/510300_return_classification_v1.json").read_text(encoding="utf-8"))
    config.update({"study_id": STUDY, "primary": "D1_PRIMARY_EXTERNAL_H1", "maximum_source_age_days": 7,
                   "decision_clock": "NEXT_CHINA_SESSION_09_00_ASIA_SHANGHAI", "source_paths": SOURCES,
                   "historical_vendor_first_delivery_proven": False})
    config["models"] = [{"id": f"C_{kind}_H{h}_EXPANDING_MAGNITUDE", "kind": kind,
                          "horizon": h, "train_window": None, "weighting": "MAGNITUDE"}
                         for kind in ("LOGIT", "ET") for h in (1, 5, 20)]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, config, exclusive=True)
    paths = [CONFIG, Path(__file__), ROOT / "research/return_classification_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "docs/510300_OVERNIGHT_GLOBAL_INFORMATION_V1_PROTOCOL.md",
             ROOT / "tests/test_overnight_global_information_v1.py", PARENT / "features.parquet",
             PARENT / "input_receipt.json", ROOT / config["inputs"]["dividends"]]
    paths.extend(ROOT / source[0] for source in SOURCES.values())
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [identity(p) for p in paths],
                         "own_candidate_returns_read_before_freeze": False,
                         "history_previously_observed": True, "prior_round_one_results_known": True}, exclusive=True)
    print(json.dumps({"状态": "第三轮方案已冻结", "清单哈希": digest(MANIFEST)}, ensure_ascii=False), flush=True)


def available_times(source_dates: pd.Series, clock: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(source_dates)).normalize()
    if clock == "US":
        return (dates + pd.Timedelta(hours=17)).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
    return (dates + pd.Timedelta(days=1, hours=8)).tz_localize("Asia/Shanghai").tz_convert("UTC").tz_localize(None)


def align_one(data: pd.DataFrame, source: pd.DataFrame, source_id: str, clock: str) -> tuple[pd.DataFrame, list[str]]:
    source = source.copy().sort_values("date").reset_index(drop=True)
    require(not pd.to_datetime(source.date).duplicated().any(), f"{source_id} 日期重复")
    close = source.close.astype(float)
    require(np.isfinite(close).all() and (close > 0).all(), f"{source_id} 收盘价格不完整")
    right = pd.DataFrame({"source_date": pd.to_datetime(source.date), "available_at": available_times(source.date, clock)})
    log_return = np.log(close).diff()
    columns = []
    for h in (1, 5, 20):
        name = f"{source_id}_mom{h}"
        right[name] = log_return.rolling(h).sum()
        columns.append(name)
    for suffix, values in (("vol20", log_return.rolling(20).std(ddof=1) * np.sqrt(242)),
                            ("dd60", close / close.rolling(60).max() - 1)):
        name = f"{source_id}_{suffix}"
        right[name] = values
        columns.append(name)
    if source_id == "VIX":
        right["VIX_log_level"] = np.log(close)
        columns.append("VIX_log_level")
    execution_dates = pd.to_datetime(data.date.shift(-1))
    decision = pd.DatetimeIndex(execution_dates + pd.Timedelta(hours=9)).tz_localize("Asia/Shanghai").tz_convert("UTC").tz_localize(None)
    left = pd.DataFrame({"origin_index": np.arange(len(data)), "decision_time": decision})
    merged = pd.merge_asof(left.dropna(subset=["decision_time"]).sort_values("decision_time"),
                          right.sort_values("available_at"), left_on="decision_time", right_on="available_at",
                          direction="backward", allow_exact_matches=True)
    merged[f"{source_id}_age_days"] = (merged.decision_time - merged.available_at).dt.total_seconds() / 86400
    columns.append(f"{source_id}_age_days")
    valid = merged.available_at.notna()
    require((merged.loc[valid, "available_at"] <= merged.loc[valid, "decision_time"]).all(), "海外收盘穿越决策时间")
    merged["source_id"] = source_id
    return merged.set_index("origin_index").reindex(range(len(data))), columns


def prepare(config: dict) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    data = pd.read_parquet(PARENT / "features.parquet")
    price_columns = json.loads((PARENT / "input_receipt.json").read_text(encoding="utf-8"))["features"]
    data["price_feature_valid"] = data.feature_valid
    external_columns, alignments = [], []
    valid = data.feature_valid.to_numpy().copy()
    for key, (path, clock) in SOURCES.items():
        aligned, columns = align_one(data, pd.read_parquet(ROOT / path), key, clock)
        for col in columns:
            data[col] = aligned[col].to_numpy()
        valid &= np.isfinite(data[columns]).all(axis=1).to_numpy()
        valid &= (data[f"{key}_age_days"] <= config["maximum_source_age_days"]).to_numpy()
        external_columns.extend(columns)
        alignments.append(aligned.reset_index()[["origin_index", "source_id", "decision_time", "source_date", "available_at", f"{key}_age_days"]].rename(columns={f"{key}_age_days": "age_days"}))
    data["feature_valid"] = valid
    alignment = pd.concat(alignments, ignore_index=True)
    alignment.to_parquet(OUT / "source_clock_alignment.parquet", index=False)
    data.to_parquet(OUT / "features.parquet", index=False)
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    labels = {"date": data.date}
    for h in (1, 5, 20):
        y = np.full(len(data), np.nan)
        for t in range(len(data) - h - 1):
            y[t] = holding_total_return(data, dividends, t + 1, t + h + 1)[0]
        labels[f"Y{h}"] = y
    pd.DataFrame(labels).to_parquet(OUT / "labels.parquet", index=False)
    write_json(OUT / "source_receipt.json", {"state": "HISTORICAL_MARKET_CLOCK_ALIGNMENT_ONLY",
               "price_features": price_columns, "external_features": external_columns,
               "total_feature_count": len(price_columns) + len(external_columns),
               "valid_feature_rows": int(valid.sum()), "historical_vendor_first_delivery_proven": False,
               "no_view_evaluation_origins": data.loc[(data.date >= "2019-12-31") & ~data.feature_valid, "date"].tolist(),
               "clock_sources": ["https://www.nyse.com/trade/trading-information",
                                  "https://datashop.cboe.com/new-product-launch-high-level-option-sentiment-10222021"]})
    return data, price_columns, external_columns, dividends


def binary_targets(probabilities: np.ndarray) -> np.ndarray:
    held, result = 0.0, np.zeros(len(probabilities))
    for t, value in enumerate(probabilities):
        if np.isfinite(value):
            held = float(value > 0.5)
        result[t] = held
    return result


def label(model_id: str) -> str:
    if model_id in NAMES:
        return NAMES[model_id]
    group, original = model_id.split("__", 1)
    return ("含盘后信息／" if group == "EXTERNAL" else "仅中国因子／") + classification.label(original)


def train_group(data: pd.DataFrame, columns: list[str], config: dict, group: str) -> dict[str, np.ndarray]:
    folder = OUT / group
    (folder / "final_models").mkdir(parents=True, exist_ok=True)
    x = data[columns].to_numpy(float)
    labels = pd.read_parquet(OUT / "labels.parquet")
    start = int(np.flatnonzero(data.date >= pd.Timestamp(config["shadow_start"]))[0]) - 1
    quarters = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [start] + [t for t in range(start + 1, len(data) - 1) if quarters[t] != quarters[t - 1]] + [len(data) - 1]
    results, receipts = {}, []
    for item in config["models"]:
        y = labels[f"Y{item['horizon']}"] .to_numpy(float)
        prediction = np.full(len(data), np.nan)
        for t, end in zip(cuts[:-1], cuts[1:]):
            train = eligible_training(data, t, item["horizon"], item["train_window"])
            require(len(train) >= config["minimum_train_samples"], "盘后模型成熟样本不足")
            require(np.isfinite(x[train]).all() and np.isfinite(y[train]).all(), "盘后训练输入不完整")
            weights = np.abs(y[train]) / np.abs(y[train]).mean()
            scaler = StandardScaler().fit(x[train])
            seed = config["random_seed"] + int(data.date.iloc[t].strftime("%Y%m%d"))
            model = classification.classifier(item["kind"], seed)
            eligible_prediction = np.arange(t, end)[data.feature_valid.iloc[t:end].to_numpy()]
            with threadpool_limits(limits=1):
                model.fit(scaler.transform(x[train]), (y[train] > 0).astype(int), sample_weight=weights)
                if len(eligible_prediction):
                    prediction[eligible_prediction] = model.predict_proba(scaler.transform(x[eligible_prediction]))[:, 1]
            receipts.append({"model": item["id"], "group": group, "fit_origin": data.date.iloc[t],
                             "fit_decision_time": data.date.iloc[t + 1] + pd.Timedelta(hours=9),
                             "train_rows": len(train), "first_train_origin": data.date.iloc[train[0]],
                             "last_train_origin": data.date.iloc[train[-1]],
                             "last_label_exit": data.date.iloc[train[-1] + item["horizon"] + 1],
                             "last_prediction_origin": data.date.iloc[end - 1], "random_seed": seed,
                             "train_array_sha256": hashlib.sha256(x[train].tobytes() + y[train].tobytes() + weights.tobytes()).hexdigest()})
        results[item["id"]] = prediction
        joblib.dump({"model": model, "scaler": scaler, "features": columns, "specification": item,
                     "last_fit_receipt": receipts[-1]}, folder / "final_models" / f"{item['id']}.joblib", compress=3)
        print(f"第三轮 {group} 分类模型完成：{item['id']}", flush=True)
    pd.DataFrame(receipts).to_csv(folder / "training_receipts.csv", index=False, encoding="utf-8-sig")
    return results


def run(expected: str) -> None:
    require(digest(MANIFEST) == expected, "第三轮清单哈希不符")
    for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], f"第三轮冻结文件变化：{item['path']}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    data, price_columns, extra, dividends = prepare(config)
    predictions, targets, horizons = {}, {}, {}
    for group, columns in (("PRICE", price_columns), ("EXTERNAL", price_columns + extra)):
        training_data = data.copy()
        if group == "PRICE":
            training_data["feature_valid"] = training_data.price_feature_valid
        result = train_group(training_data, columns, config, group)
        for item in config["models"]:
            model_id = group + "__" + item["id"]
            probability = result[item["id"]]
            if group == "EXTERNAL":
                probability[~data.feature_valid.to_numpy()] = np.nan
            predictions[model_id] = probability
            targets[model_id] = binary_targets(probability)
            horizons[model_id] = item["horizon"]
    for model_id, group, only_h1 in (("D1_PRIMARY_EXTERNAL_H1", "EXTERNAL", True), ("D2_PRICE_H1", "PRICE", True),
                                    ("D3_EXTERNAL_ALL", "EXTERNAL", False), ("D4_PRICE_ALL", "PRICE", False)):
        values = [value for key, value in predictions.items() if key.startswith(group + "__") and (not only_h1 or "_H1_" in key)]
        probability = np.mean(np.column_stack(values), axis=1)
        targets[model_id] = classification.consensus(probability, config["consensus_upper"], config["consensus_lower"])
        horizons[model_id] = 1 if only_h1 else 5
    pd.DataFrame({"date": data.date, **targets}).to_parquet(OUT / "targets.parquet", index=False)
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "probabilities.parquet", index=False)
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for cost_id, cost in config["costs"].items():
        ledgers = {}
        for model_id in list(targets) + ["BUY_HOLD"]:
            ledger, decisions = simulate(data, dividends, config, cost, config["evaluation_start"], model_id,
                          targets=targets.get(model_id), rebalance=horizons.get(model_id, 1))
            ledger["valuation_anchor_date"] = ledger.origin
            ledger["decision_time_asia_shanghai"] = (pd.to_datetime(ledger.date) + pd.Timedelta(hours=9)).dt.tz_localize("Asia/Shanghai")
            decisions["valuation_anchor_date"] = decisions.origin
            decisions["decision_time_asia_shanghai"] = (pd.to_datetime(decisions.execution_date) + pd.Timedelta(hours=9)).dt.tz_localize("Asia/Shanghai")
            save_account(OUT / "evaluation" / cost_id, model_id, ledger, decisions)
            ledgers[model_id] = ledger
        baseline = summarize(ledgers["BUY_HOLD"], config)
        for model_id, ledger in ledgers.items():
            item = {"cost": cost_id, "model": model_id, **summarize(ledger, config)}
            item["annualized_return_excess_vs_buy_hold"] = item["annualized_return"] - baseline["annualized_return"]
            item["meets_point_target"] = item["net_sharpe"] is not None and item["net_sharpe"] >= 1.2
            metrics.append(item)
            for year, part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": model_id, "year": int(year), **summarize(part, config)})
            for name, start, end in (("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                                     ("2024—终点", "2024-01-01", config["data_cutoff"])):
                part = ledger.loc[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": model_id, "era": name, **summarize(part, config)})
        pd.DataFrame({"date": ledgers["BUY_HOLD"].date, **{k: v.net_return.to_numpy() for k, v in ledgers.items()}}).to_parquet(OUT / f"{cost_id}_all_evaluation_returns.parquet", index=False)
        a, b = ledgers[config["primary"]].net_return.to_numpy(), ledgers["BUY_HOLD"].net_return.to_numpy()
        rng = np.random.default_rng(config["random_seed"])
        sharpe, excess = [], []
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(a), config["bootstrap_day_block"])
            value = return_metrics(a[ix], config["annual_days"])["net_sharpe"]
            sharpe.append(np.nan if value is None else value)
            excess.append(float((a[ix] - b[ix]).mean() * config["annual_days"]))
        uncertainty[cost_id] = {"primary_sharpe_95_interval": interval(sharpe),
                               "primary_annualized_arithmetic_excess_95_interval": interval(excess), "multiple_search_adjusted": False}
        print(f"第三轮 {cost_id} 的 17 个完整评价账户已完成", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    primary = frame.loc[frame.model == config["primary"]]
    reached = bool(primary.loc[primary.cost == "BASE", "meets_point_target"].iloc[0])
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(),
              "number_of_candidates": 16, "benchmark_count": 1, "trained_model_count": 12,
              "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
              "historical_vendor_first_delivery_proven": False, "uncertainty": uncertainty,
              "position_impact": 0, "validated_live_strategy": "NONE"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    lines = ["# 510300 免费全球盘后信息：第三轮实际结果", "",
             "16 个候选与一个基准分别运行基础和压力费用，共 34 个完整评价账户。海外指数只用于观察，所有账户只持有 510300 与现金。", "",
             f"主方案基础夏普为 {primary.loc[primary.cost == 'BASE', 'net_sharpe'].iloc[0]:.4f}。事后最好候选为“{label(best['model'])}”，夏普 {best['net_sharpe']:.4f}。", "",
             "| 策略 | 基础夏普 | 压力夏普 | 年化收益 | 年化超额 | 最大回撤 | 成交笔数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model_id in [config["primary"], "BUY_HOLD"] + [k for k in targets if k != config["primary"]]:
        row = frame.loc[(frame.cost == "BASE") & (frame.model == model_id)].iloc[0]
        stress = frame.loc[(frame.cost == "STRESS") & (frame.model == model_id)].iloc[0]
        fmt = lambda value: "无定义" if pd.isna(value) else f"{value:.4f}"
        lines.append(f"| {label(model_id)} | {fmt(row.net_sharpe)} | {fmt(stress.net_sharpe)} | {row.annualized_return:.2%} | {row.annualized_return_excess_vs_buy_hold:.2%} | {row.max_drawdown:.2%} | {row.trade_count} |")
    lines.extend(["", "本次按历史市场收盘时钟匹配海外日线，并非实时免费接口已经验证。历史首次交付时刻与数据修订情况没有证据，因此即使历史数值达标，也必须另外验证可取得性和独立表现。",
                  "", f"本轮结论状态：{result['status']}。已保存所有时间匹配、模型训练、账户及分年分段资料。"])
    (OUT / "510300免费全球盘后信息_实际结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行免费全球盘后信息研究")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    require(args.freeze != args.run, "必须选择冻结或运行之一")
    if args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256), "缺少冻结清单哈希")
        run(args.expected_manifest_sha256)
