"""按当时市场状态持续更新机制权重，并运行510300完整模拟账户。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import ROOT, save_account, simulate, summarize
from research.intraday_overnight_increment_v1 import block_indices, interval, normalize_dividends, now, require, return_metrics, write_json

STUDY = "510300_CONTEXTUAL_EXPERT_TRACKING_V1"
OUT = ROOT / "reports/research/510300_contextual_expert_tracking_v1"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
CONFIG = ROOT / "config/510300_contextual_expert_tracking_v1.json"
MANIFEST = ROOT / "config/510300_contextual_expert_tracking_v1_manifest.json"
FAMILIES = ["CASH", "BUY_HOLD", "F_TREND", "F_REVERSAL", "F_MIXED", "F_PREDICTION"]
META = ["A1_PRIMARY_CONTEXT4_BLEND", "A2_GLOBAL", "A3_CONTEXT4_ONLY", "A4_CONTEXT2_BLEND", "A5_CONTEXT4_HARD", "A6_EQUAL_FAMILIES"]
NAMES = {"CASH": "现金", "BUY_HOLD": "买入持有", "F_TREND": "趋势机制", "F_REVERSAL": "反转机制",
         "F_MIXED": "趋势震荡混合机制", "F_PREDICTION": "预测模型机制",
         "A1_PRIMARY_CONTEXT4_BLEND": "主方案：四状态与总体混合", "A2_GLOBAL": "仅总体学习", "A3_CONTEXT4_ONLY": "仅四状态学习",
         "A4_CONTEXT2_BLEND": "趋势两状态与总体混合", "A5_CONTEXT4_HARD": "四状态硬选择", "A6_EQUAL_FAMILIES": "六机制固定等权"}


def physical(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    actual = Path(r"E:\ResearchData\New project 8") / rel
    return actual if rel.parts[0] == "data" and actual.exists() else path


def sha(path: Path) -> str:
    return hashlib.sha256(physical(path).read_bytes()).hexdigest()


def ident(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": physical(path).stat().st_size, "sha256": sha(path)}


def groups(config: dict) -> dict:
    rules = list(config["rule_names"])
    return {"F_TREND": [k for k in rules if k not in ("R09_RSI2", "R10_RSI2_TREND", "R11_REGIME")],
            "F_REVERSAL": ["R09_RSI2", "R10_RSI2_TREND"], "F_MIXED": ["R11_REGIME"],
            "F_PREDICTION": [k["id"] for k in config["models"]]}


def prepare_signals() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "source_receipt.json").exists(), "本轮信号来源已保存")
    old = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(PARENT / "features.parquet")
    dates = pd.DatetimeIndex(data.date)
    first = int(np.flatnonzero(data.date >= old["shadow_start"])[0]) - 1
    family_groups = groups(old)
    streams = {}
    for members in family_groups.values():
        for member in members:
            d = pd.read_parquet(PARENT / "shadow/BASE" / (member + "_decisions.parquet"))
            require(not d.origin.duplicated().any(), "底层决策日期重复")
            require((d.execution_date > d.origin).all(), "底层信号执行时间不在决策之后")
            stream = d.set_index("origin").reference_weight.reindex(dates).to_numpy(float)
            require(np.isfinite(stream[first:-1]).all(), "底层连续可用信号缺失：" + member)
            require(((stream[first:-1] >= 0) & (stream[first:-1] <= 1)).all(), "底层目标仓位越界")
            streams[member] = stream
    require(data.feature_valid.iloc[first:-1].all(), "状态来源有缺失")
    prior_fits = pd.read_csv(PARENT / "training_receipts.csv")
    require((pd.to_datetime(prior_fits.last_label_exit) <= pd.to_datetime(prior_fits.fit_origin)).all(), "旧模型存在未到期训练标签")
    result = {"date": data.date, "CASH": np.zeros(len(data)), "BUY_HOLD": np.ones(len(data))}
    for family, members in family_groups.items():
        result[family] = np.mean(np.column_stack([streams[m] for m in members]), axis=1)
    pd.DataFrame(result).to_parquet(OUT / "family_raw_targets.parquet", index=False)
    write_json(OUT / "source_receipt.json", {"prepared_at": now(), "status": "PASS_CONTINUOUS_PAST_TRAINED_SIGNALS_AND_STATE_INPUTS",
               "first_signal_origin": data.date.iloc[first], "last_signal_origin": data.date.iloc[-2], "groups": family_groups,
               "family_count_including_cash_and_hold": len(FAMILIES), "original_active_components": len(streams),
               "prior_training_receipt_rows": len(prior_fits), "prior_label_maturity_verified": True,
               "new_family_or_meta_returns_read": False}, exclusive=True)
    print("第九轮来源准备完成：24个既有输出按四个主动机制分组，另含现金与买入持有", flush=True)


def fixed_share(weights: np.ndarray, reward: np.ndarray | None, eta: float, alpha: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if reward is not None:
        logits = np.log(weights) + eta * np.clip(reward, -3, 3)
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
    weights = (1 - alpha) * weights + alpha / len(weights)
    return weights / weights.sum()


def online_tracking(data: pd.DataFrame, rewards: np.ndarray, teacher_targets: np.ndarray, config: dict, start: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    n, k = rewards.shape
    require(teacher_targets.shape == (n, k) and k == len(FAMILIES), "机制信号形状不符")
    eta, alpha = config["learning_rate"], config["share_fraction"]
    prior = np.ones(k) / k
    overall = prior.copy()
    local4, local2 = np.tile(prior, (4, 1)), np.tile(prior, (2, 1))
    mass4, mass2 = np.zeros(4), np.zeros(2)
    states2 = (data.sma120.to_numpy() > 0).astype(int)
    states4 = 2 * states2 + (data.vol_ratio.to_numpy() > 1).astype(int)
    variance = data.variance60.to_numpy(float)
    predictions = {m: np.full(n, np.nan) for m in META}
    weight_records, update_records = [], []
    for t in range(start, n - 1):
        scale = np.nan
        observed = pd.NaT
        previous_state = None
        if t > start:
            require(np.isfinite(rewards[t]).all() and (rewards[t] > -1).all(), "机制回报未结算或非法")
            scale = max(np.sqrt(variance[t - 1]), config["minimum_daily_risk_scale"])
            reward = np.clip(np.log1p(rewards[t]) / scale, -3, 3)
            overall = fixed_share(overall, reward, eta, alpha)
            for state in range(4):
                local4[state] = fixed_share(local4[state], reward if state == states4[t - 1] else None, eta, alpha)
            for state in range(2):
                local2[state] = fixed_share(local2[state], reward if state == states2[t - 1] else None, eta, alpha)
            mass4 *= 1 - alpha
            mass2 *= 1 - alpha
            mass4[states4[t - 1]] += 1
            mass2[states2[t - 1]] += 1
            previous_state = int(states4[t - 1])
            observed = data.date.iloc[t]
        current4, current2 = states4[t], states2[t]
        confidence4 = mass4[current4] / (mass4[current4] + config["context_prior_observations"])
        confidence2 = mass2[current2] / (mass2[current2] + config["context_prior_observations"])
        main = (1 - confidence4) * overall + confidence4 * local4[current4]
        trend = (1 - confidence2) * overall + confidence2 * local2[current2]
        hard = np.eye(k)[int(np.argmax(main))]
        weights = dict(zip(META, [main, overall, local4[current4], trend, hard, prior]))
        require(np.isfinite(teacher_targets[t]).all(), "当时机制次日目标缺失")
        for model, w in weights.items():
            require(abs(w.sum() - 1) < 1e-12 and (w >= 0).all(), "机制分配权重不守恒")
            predictions[model][t] = np.clip(float(w @ teacher_targets[t]), 0, 1)
            weight_records.append({"date": data.date.iloc[t], "model": model, "state4": int(current4), "state2": int(current2),
                                   "confidence4": float(confidence4), "confidence2": float(confidence2), "target": predictions[model][t],
                                   **{name: float(w[j]) for j, name in enumerate(FAMILIES)}})
        update_records.append({"origin": data.date.iloc[t], "decision_time": data.date.iloc[t] + pd.Timedelta(hours=15),
                               "latest_observed_return_date": observed, "updated_previous_state4": previous_state,
                               "current_state4": int(current4), "previous_day_risk_scale": scale,
                               "current_state_effective_observations": float(mass4[current4]),
                               "all_local_state_weights": json.dumps(local4.tolist()), "overall_weights": json.dumps(overall.tolist())})
    return predictions, pd.DataFrame(weight_records), pd.DataFrame(update_records)


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "第九轮已经冻结")
    old = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config = {k: v for k, v in old.items() if k not in ("models", "rule_names", "meta_names")}
    config.update({"study_id": STUDY, "primary": META[0], "families": FAMILIES, "combinations": META,
                   "learning_rate": float(np.sqrt(2 * np.log(len(FAMILIES)) / 126)), "share_fraction": 1 / 126,
                   "context_prior_observations": 20.0, "minimum_daily_risk_scale": .005,
                   "teacher_cost": "BASE", "execution_rebalance_sessions": 5, "underlying_components_promoted": False,
                   "new_source_financials_included": False, "forecast_historical_outputs_retrained": False})
    write_json(CONFIG, config, exclusive=True)
    paths = [CONFIG, Path(__file__), ROOT / "docs/510300_CONTEXTUAL_EXPERT_TRACKING_V1_PROTOCOL.md",
             ROOT / "tests/test_contextual_expert_tracking_v1.py", ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
             ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_adaptive_allocation_v1.json",
             ROOT / "config/510300_adaptive_allocation_v1_manifest.json", ROOT / "docs/510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             PARENT / "features.parquet", PARENT / "input_receipt.json", PARENT / "training_receipts.csv", PARENT / "predictions.parquet",
             ROOT / config["inputs"]["dividends"], OUT / "family_raw_targets.parquet", OUT / "source_receipt.json"]
    for members in groups(old).values():
        paths.extend(PARENT / "shadow/BASE" / (m + "_decisions.parquet") for m in members)
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [ident(p) for p in paths],
                          "own_family_and_meta_returns_read_before_freeze": False, "underlying_history_previously_observed": True}, exclusive=True)
    print(json.dumps({"状态": "第九轮已冻结", "清单哈希": sha(MANIFEST)}, ensure_ascii=False), flush=True)


def run(expected: str) -> None:
    require(sha(MANIFEST) == expected, "第九轮清单哈希不符")
    for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        require(sha(ROOT / row["path"]) == row["sha256"], "冻结输入已变化：" + row["path"])
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    data = pd.read_parquet(PARENT / "features.parquet")
    raw_targets = pd.read_parquet(OUT / "family_raw_targets.parquet")
    dates = pd.DatetimeIndex(data.date)
    start = int(np.flatnonzero(data.date >= config["shadow_start"])[0]) - 1
    dividends = normalize_dividends(pd.read_csv(physical(ROOT / config["inputs"]["dividends"])))
    rewards = np.full((len(data), len(FAMILIES)), np.nan)
    teacher_targets = np.full_like(rewards, np.nan)
    for j, key in enumerate(FAMILIES):
        ledger, decisions = simulate(data, dividends, config, config["costs"]["BASE"], config["shadow_start"], key,
                                     targets=raw_targets[key].to_numpy(), rebalance=config["execution_rebalance_sessions"])
        save_account(OUT / "mechanism_shadow/BASE", key, ledger, decisions)
        rewards[dates.get_indexer(ledger.date), j] = ledger.net_return.to_numpy()
        teacher_targets[dates.get_indexer(decisions.origin), j] = decisions.reference_weight.to_numpy()
    pd.DataFrame({"date": data.date, **{k: rewards[:, j] for j, k in enumerate(FAMILIES)}}).to_parquet(OUT / "teacher_returns.parquet", index=False)
    pd.DataFrame({"date": data.date, **{k: teacher_targets[:, j] for j, k in enumerate(FAMILIES)}}).to_parquet(OUT / "teacher_targets.parquet", index=False)
    predictions, weight_trace, updates = online_tracking(data, rewards, teacher_targets, config, start)
    weight_trace.to_parquet(OUT / "daily_mechanism_weights.parquet", index=False)
    updates.to_parquet(OUT / "online_update_receipts.parquet", index=False)
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "targets.parquet", index=False)
    print("六个机制的基础账户与逐日状态权重学习已完成", flush=True)
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for costid, cost in config["costs"].items():
        accounts = {}
        for key in META + FAMILIES:
            target = predictions[key] if key in predictions else raw_targets[key].to_numpy()
            ledger, decisions = simulate(data, dividends, config, cost, config["evaluation_start"], key,
                                         targets=target, rebalance=config["execution_rebalance_sessions"])
            accounts[key] = ledger
            save_account(OUT / "evaluation" / costid, key, ledger, decisions)
        baseline = summarize(accounts["BUY_HOLD"], config)
        for key, ledger in accounts.items():
            row = {"cost": costid, "model": key, **summarize(ledger, config)}
            row["annualized_return_excess_vs_buy_hold"] = row["annualized_return"] - baseline["annualized_return"]
            row["meets_point_target"] = row["net_sharpe"] is not None and row["net_sharpe"] >= config["high_sharpe_target"]
            metrics.append(row)
            for year, part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": costid, "model": key, "year": int(year), **summarize(part, config)})
            for name, left, right in (("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", config["data_cutoff"])):
                eras.append({"cost": costid, "model": key, "era": name, **summarize(ledger.loc[(ledger.date >= left) & (ledger.date <= right)], config)})
        rr = pd.DataFrame({"date": accounts["BUY_HOLD"].date, **{k: v.net_return.to_numpy() for k, v in accounts.items()}})
        rr.to_parquet(OUT / f"{costid}_all_evaluation_returns.parquet", index=False)
        rng = np.random.default_rng(config["random_seed"])
        samples = {"primary_sharpe": [], "increment_vs_equal": [], "increment_vs_global": [], "increment_vs_hard": []}
        for _ in range(config["bootstrap_repetitions"]):
            take = block_indices(rng, len(rr), config["bootstrap_day_block"])
            r = rr[config["primary"]].to_numpy()[take]
            samples["primary_sharpe"].append(return_metrics(r, config["annual_days"])["net_sharpe"])
            for name, key in (("equal", "A6_EQUAL_FAMILIES"), ("global", "A2_GLOBAL"), ("hard", "A5_CONTEXT4_HARD")):
                samples[f"increment_vs_{name}"].append(float((r - rr[key].to_numpy()[take]).mean() * config["annual_days"]))
        uncertainty[costid] = {k + "_95_interval": interval(v) for k, v in samples.items()}
        print(f"第九轮 {costid} 的12条完整账户已完成", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    primary = frame.loc[frame.model == config["primary"]]
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    reached = bool(primary.loc[primary.cost == "BASE", "meets_point_target"].iloc[0])
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(), "number_of_candidates": 11,
              "evaluation_accounts": 24, "mechanism_shadow_accounts": 6, "underlying_predictions_retrained": False,
              "uncertainty": uncertainty, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
              "position_impact": 0, "original_components_promoted": False}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第九轮机制分组、状态权重跟踪和完整账户")
    parser.add_argument("--prepare-signals", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    require(sum((args.prepare_signals, args.freeze, args.run)) == 1, "选择信号准备、冻结或运行之一")
    if args.prepare_signals:
        prepare_signals()
    elif args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256), "缺少冻结清单哈希")
        run(args.expected_manifest_sha256)
