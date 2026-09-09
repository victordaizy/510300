"""涨跌分类与幅度加权的预登记研究，沿用完整账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.adaptive_allocation_v1 import ROOT, identity, eligible_training, simulate, summarize, save_account
from research.intraday_overnight_increment_v1 import now, digest, write_json, require, block_indices, return_metrics, interval

STUDY = "510300_RETURN_CLASSIFICATION_V1"
OUT = ROOT / "reports/research/510300_return_classification_v1"
CONFIG = ROOT / "config/510300_return_classification_v1.json"
MANIFEST = ROOT / "config/510300_return_classification_v1_manifest.json"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "第二轮方案已经冻结，禁止覆盖")
    config = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config.update({"study_id": STUDY, "primary": "C1_WEIGHTED_CONSENSUS", "consensus_upper": 0.52,
                   "consensus_lower": 0.48, "rule_names": {}, "meta_names": {},
                   "historical_method_authorization": "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md"})
    config["models"] = [{"id": f"C_{kind}_H{horizon}_{window}_{weight}", "kind": kind,
                          "horizon": horizon, "train_window": None if window == "EXPANDING" else 756,
                          "weighting": weight}
                         for kind in ("LOGIT", "HGB", "ET") for horizon in (5, 20)
                         for window in ("EXPANDING", "ROLL756") for weight in ("EQUAL", "MAGNITUDE")]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, config, exclusive=True)
    parent_manifest = ROOT / "config/510300_adaptive_allocation_v1_manifest.json"
    paths = [CONFIG, Path(__file__), ROOT / "docs/510300_RETURN_CLASSIFICATION_V1_PROTOCOL.md",
             parent_manifest, ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", PARENT / "features.parquet",
             PARENT / "labels.parquet", PARENT / "input_receipt.json", ROOT / config["inputs"]["dividends"],
             ROOT / "tests/test_return_classification_v1.py"]
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [identity(p) for p in paths],
               "own_candidate_returns_read_before_freeze": False, "prior_round_results_known": True,
               "prior_round": "510300_ADAPTIVE_ALLOCATION_V1"}, exclusive=True)
    print(json.dumps({"状态": "第二轮方案已冻结", "清单哈希": digest(MANIFEST)}, ensure_ascii=False), flush=True)


def classifier(kind: str, seed: int):
    if kind == "LOGIT":
        return LogisticRegression(C=0.01, max_iter=1000, random_state=seed)
    if kind == "HGB":
        return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.05, max_leaf_nodes=7,
                      min_samples_leaf=60, l2_regularization=10, early_stopping=False, random_state=seed)
    if kind == "ET":
        return ExtraTreesClassifier(n_estimators=96, max_depth=3, min_samples_leaf=40,
                      max_features=0.75, bootstrap=False, random_state=seed, n_jobs=1)
    raise ValueError("未知分类算法")


def train_all(data: pd.DataFrame, columns: list[str], config: dict) -> dict[str, np.ndarray]:
    x = data[columns].to_numpy(float)
    labels = pd.read_parquet(PARENT / "labels.parquet")
    start = int(np.flatnonzero(data.date >= pd.Timestamp(config["shadow_start"]))[0]) - 1
    quarters = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [start] + [t for t in range(start + 1, len(data) - 1) if quarters[t] != quarters[t - 1]] + [len(data) - 1]
    predictions, receipts = {}, []
    (OUT / "final_models").mkdir(exist_ok=True)
    for number, item in enumerate(config["models"], 1):
        y = labels[f"Y{item['horizon']}"] .to_numpy(float)
        prediction = np.full(len(data), np.nan)
        for t, end in zip(cuts[:-1], cuts[1:]):
            train = eligible_training(data, t, item["horizon"], item["train_window"])
            require(len(train) >= config["minimum_train_samples"], "分类模型的已成熟样本不足")
            require(np.isfinite(y[train]).all(), "训练标签不完整")
            classes = (y[train] > 0).astype(int)
            require(len(np.unique(classes)) == 2, "训练资料缺少一种涨跌状态")
            weights = np.ones(len(train)) if item["weighting"] == "EQUAL" else np.abs(y[train]) / np.abs(y[train]).mean()
            scaler = StandardScaler().fit(x[train])
            seed = config["random_seed"] + int(data.date.iloc[t].strftime("%Y%m%d"))
            model = classifier(item["kind"], seed)
            with threadpool_limits(limits=1):
                model.fit(scaler.transform(x[train]), classes, sample_weight=weights)
                prediction[t:end] = model.predict_proba(scaler.transform(x[t:end]))[:, 1]
            receipts.append({"model": item["id"], "fit_origin": data.date.iloc[t], "train_rows": len(train),
                             "first_train_origin": data.date.iloc[train[0]], "last_train_origin": data.date.iloc[train[-1]],
                             "last_label_exit": data.date.iloc[train[-1] + item["horizon"] + 1],
                             "last_prediction_origin": data.date.iloc[end - 1], "random_seed": seed,
                             "train_array_sha256": hashlib.sha256(x[train].tobytes() + y[train].tobytes() + weights.tobytes()).hexdigest()})
        predictions[item["id"]] = prediction
        joblib.dump({"model": model, "scaler": scaler, "features": columns, "specification": item,
                     "last_fit_receipt": receipts[-1]}, OUT / "final_models" / f"{item['id']}.joblib", compress=3)
        print(f"第二轮分类模型完成 {number}/{len(config['models'])}：{item['id']}", flush=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "probabilities.parquet", index=False)
    return predictions


def consensus(probabilities: np.ndarray, upper: float, lower: float) -> np.ndarray:
    result = np.zeros(len(probabilities))
    held = 0.0
    for t, p in enumerate(probabilities):
        if np.isfinite(p):
            if p > upper:
                held = 1.0
            elif p < lower:
                held = 0.0
        result[t] = held
    return result


def label(model_id: str) -> str:
    special = {"BUY_HOLD": "买入持有", "C1_WEIGHTED_CONSENSUS": "主方案：幅度加权概率共识",
               "C2_EQUAL_CONSENSUS": "普通训练概率共识"}
    if model_id in special:
        return special[model_id]
    _, kind, horizon, window, weight = model_id.split("_")
    kind_cn = {"LOGIT": "逻辑回归", "HGB": "浅层梯度提升", "ET": "极随机树"}[kind]
    return f"{kind_cn}／{horizon[1:]}日／{'全部过去样本' if window == 'EXPANDING' else '最近756日'}／{'幅度加权' if weight == 'MAGNITUDE' else '普通训练'}"


def run(expected: str) -> None:
    require(digest(MANIFEST) == expected, "第二轮冻结清单不符")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], f"第二轮冻结文件变化：{item['path']}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    data = pd.read_parquet(PARENT / "features.parquet")
    from research.intraday_overnight_increment_v1 import normalize_dividends
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    columns = json.loads((PARENT / "input_receipt.json").read_text(encoding="utf-8"))["features"]
    predictions = train_all(data, columns, config)
    targets = {key: (value > 0.5).astype(float) for key, value in predictions.items()}
    for model_id, weighting in (("C1_WEIGHTED_CONSENSUS", "MAGNITUDE"), ("C2_EQUAL_CONSENSUS", "EQUAL")):
        matrix = np.column_stack([value for key, value in predictions.items() if key.endswith(weighting)])
        mean_probability = np.mean(matrix, axis=1)
        targets[model_id] = consensus(mean_probability, config["consensus_upper"], config["consensus_lower"])
    pd.DataFrame({"date": data.date, **targets}).to_parquet(OUT / "targets.parquet", index=False)
    horizons = {item["id"]: item["horizon"] for item in config["models"]}
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for cost_id, cost in config["costs"].items():
        ledgers = {}
        for model_id in list(targets) + ["BUY_HOLD"]:
            ledger, decisions = simulate(data, dividends, config, cost, config["evaluation_start"], model_id,
                       targets=targets.get(model_id), rebalance=horizons.get(model_id, 5))
            save_account(OUT / "evaluation" / cost_id, model_id, ledger, decisions)
            ledgers[model_id] = ledger
        baseline = summarize(ledgers["BUY_HOLD"], config)
        for model_id, ledger in ledgers.items():
            item = {"cost": cost_id, "model": model_id, **summarize(ledger, config)}
            item["annualized_return_excess_vs_buy_hold"] = item["annualized_return"] - baseline["annualized_return"]
            item["meets_point_target"] = item["net_sharpe"] is not None and item["net_sharpe"] >= 1.2
            metrics.append(item)
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": model_id, "year": int(year), **summarize(group, config)})
            for name, start, end in (("2020—2021", "2020-01-01", "2021-12-31"),
                                     ("2022—2023", "2022-01-01", "2023-12-31"),
                                     ("2024—终点", "2024-01-01", config["data_cutoff"])):
                group = ledger.loc[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": model_id, "era": name, **summarize(group, config)})
        pd.DataFrame({"date": ledgers["BUY_HOLD"].date,
                      **{k: v.net_return.to_numpy() for k, v in ledgers.items()}}).to_parquet(OUT / f"{cost_id}_all_evaluation_returns.parquet", index=False)
        a = ledgers[config["primary"]].net_return.to_numpy()
        b = ledgers["BUY_HOLD"].net_return.to_numpy()
        rng = np.random.default_rng(config["random_seed"])
        sharpe, excess = [], []
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(a), config["bootstrap_day_block"])
            value = return_metrics(a[ix], config["annual_days"])["net_sharpe"]
            sharpe.append(np.nan if value is None else value)
            excess.append(float((a[ix] - b[ix]).mean() * config["annual_days"]))
        uncertainty[cost_id] = {"primary_sharpe_95_interval": interval(sharpe),
                               "primary_annualized_arithmetic_excess_95_interval": interval(excess),
                               "multiple_search_adjusted": False}
        print(f"第二轮 {cost_id} 的 27 个完整评价账户已完成", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    primary = frame.loc[frame.model == config["primary"]]
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    reached = bool(primary.loc[primary.cost == "BASE", "meets_point_target"].iloc[0])
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(),
              "number_of_candidates": 26, "benchmark_count": 1, "trained_model_count": 24,
              "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "uncertainty": uncertainty,
              "position_impact": 0, "validated_live_strategy": "NONE"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    lines = ["# 510300 涨跌分类与幅度加权：第二轮实际结果", "",
             "本轮计算 24 个分类模型、两个概率共识组合及买入持有基准，基础和压力费用分别运行，共 54 个完整评价账户。所有因子和规则在研究协议中以中文写明。", "",
             f"预先指定主方案基础夏普为 {primary.loc[primary.cost == 'BASE', 'net_sharpe'].iloc[0]:.4f}；本轮事后最好候选为“{label(best['model'])}”，基础夏普 {best['net_sharpe']:.4f}。", "",
             "| 策略 | 基础夏普 | 压力夏普 | 年化收益 | 年化超额 | 最大回撤 | 成交笔数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model_id in [config["primary"], "BUY_HOLD"] + [key for key in targets if key != config["primary"]]:
        row = frame.loc[(frame.cost == "BASE") & (frame.model == model_id)].iloc[0]
        stress = frame.loc[(frame.cost == "STRESS") & (frame.model == model_id)].iloc[0]
        fmt = lambda value: "无定义" if pd.isna(value) else f"{value:.4f}"
        lines.append(f"| {label(model_id)} | {fmt(row.net_sharpe)} | {fmt(stress.net_sharpe)} | {row.annualized_return:.2%} | {row.annualized_return_excess_vs_buy_hold:.2%} | {row.max_drawdown:.2%} | {row.trade_count} |")
    lines.extend(["", "本轮为已经多次研究过的历史重放，不是独立样本。区块重抽样没有校正历次策略搜索；区间、全部逐日账户、训练时点、分年度和分阶段指标均保留。",
                  "", f"本轮结论状态：{result['status']}。目标是否已经实现，必须按真实结果判断；失败模型不能靠修改本轮参数改写结论。"])
    (OUT / "510300涨跌分类与幅度加权_实际结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结或运行涨跌分类研究")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    require(args.freeze != args.run, "必须选择冻结或运行之一")
    if args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256), "缺少清单哈希")
        run(args.expected_manifest_sha256)
