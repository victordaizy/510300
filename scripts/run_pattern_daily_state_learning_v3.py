"""V3用普通日的成熟样本学习状态收益，固定比较背景状态与转变信息。"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_sequential_patterns_2021_2026_v2"
OUT = ROOT / "reports/research/510300_pattern_daily_state_learning_v3"
sys.path.insert(0, str(PARENT / "code"))
from sequential_patterns_regime_v1 import FAMILIES, COSTS, account, digest, metrics, now, save_csv, save_json

FEATURES = {
    "BACKGROUND": ["trend_z", "relative_log_volatility"],
    "PHASE_VOLUME": ["trend_z", "relative_log_volatility", "trend_z_change3", "log_volume_ratio"],
}


def design(d):
    x = d[["date", "trend_z"]].copy()
    x["relative_log_volatility"] = np.log(d.rv20 / d.rv20.shift(1).rolling(252, min_periods=126).median())
    x["trend_z_change3"] = d.trend_z - d.trend_z.shift(3)
    x["log_volume_ratio"] = np.log(d.volume_ratio)
    return x


def fit_state(x, y, indices, columns, i, date, model):
    values = x.loc[indices, columns].to_numpy(float)
    means, scales = values.mean(axis=0), values.std(axis=0, ddof=1)
    scales[scales < 1e-12] = 1.0
    raw = np.clip((values - means) / scales, -5, 5)
    centered = raw - raw.mean(axis=0)
    ymean = float(y.mean())
    beta = np.linalg.solve(centered.T @ centered + 10.0 * np.eye(len(columns)), centered.T @ (y - ymean))
    intercept = ymean - float(raw.mean(axis=0) @ beta)
    return {"model": model, "fit_idx": i, "as_of_date": date, "actual_created_at": now(), "columns": columns,
            "n_train": len(indices), "train_signal_indices": indices.tolist(), "x_mean": means.tolist(),
            "x_scale": scales.tolist(), "coefficients": beta.tolist(), "intercept": intercept,
            "mature_mean_reference": ymean, "ridge_alpha": 10.0, "input_clip": 5.0}


def predict(model, row):
    z = np.clip((row[model["columns"]].to_numpy(float) - model["x_mean"]) / model["x_scale"], -5, 5)
    return float(model["intercept"] + z @ model["coefficients"])


def setup():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("V3已经冻结。")
    sources = {
        "inputs/features.parquet": PARENT / "inputs/features.parquet",
        "inputs/signals.parquet": PARENT / "inputs/signals.parquet",
        "inputs/labels.parquet": PARENT / "inputs/labels.parquet",
        "inputs/controls.parquet": PARENT / "inputs/controls.parquet",
        "inputs/dividends.csv": PARENT / "inputs/dividends.csv",
        "inputs/prices.parquet": PARENT / "inputs/prices.parquet",
        "inputs/calendar.csv": PARENT / "inputs/calendar.csv",
        "inputs/historical_calendar.csv": PARENT / "inputs/historical_calendar.csv",
        "inputs/v2_comparison.csv": PARENT / "comparison.csv",
        "inputs/v2_summary.json": PARENT / "summary.json",
        "inputs/gate_diagnostic.json": ROOT / "reports/research/510300_pattern_gate_phase_diagnostic_v1/summary.json",
        "code/sequential_patterns_regime_v1.py": PARENT / "code/sequential_patterns_regime_v1.py",
        "code/run_pattern_daily_state_learning_v3.py": Path(__file__),
    }
    for relative, src in sources.items():
        dest = OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    protocol = {"study_id": "510300_PATTERN_DAILY_STATE_LEARNING_V3", "frozen_at": now(),
                "hypothesis": "稀少形态案例无法及时估计优势；从全部普通日已成熟样本学习背景状态与转变信息，再用于固定三形态的确认日。",
                "period": ["2021-01-04", "2026-09-16"], "capital_cny": 200000,
                "models": {"MATURE_MEAN": "同训练池净收益均值", **FEATURES},
                "training_pool": "此前504个交易日普通日固定五日持有的已成熟压力净收益；不以形态是否成功筛样。",
                "label_availability": "普通日模拟在下一开盘入场，持有五日后可执行开盘退出；退出发生后才允许使用。",
                "min_train": 252, "refit_every_sessions": 21, "ridge_alpha": 10.0,
                "scaling": "仅训练样本均值/样本标准差，标准化后截断到±5；截距不惩罚。",
                "decision": "每个确认日用对应日预测压力净收益>0启用，之后每日预测<=0则下一可卖开盘退出；形态失效和原5日期限仍执行。",
                "important_change": "这是新训练表示，不再使用每形态6/4或3/2数量门和固定状态映射，也不把不适用于新表示的旧置信下界移植为门槛。",
                "evidence_grade": "探索性模型，正预测不是高置信度；须比较预测误差、完整账户和独立前向证据。",
                "preserved": ["三种形态", "仅510300与现金", "原两档成本", "T+1", "完整账户", "年度次数下限取消"],
                "acceptance": {"net_cagr": .10, "net_sharpe": 1.2, "max_drawdown": .10,
                               "phase_information": "PHASE_VOLUME对BACKGROUND预测误差与净账户增量须分别检验，不以复杂度本身作为优势。"},
                "uncertainty": "对成熟预测损失及固定账户收益用相同20日移动区块、2000次、种子20260924；保存抽样，不据结果选择模型。",
                "one_predeclared_candidate_family": True, "parameter_grids": 0, "no_rescue_after_result": True,
                "independence": "本版本创建于2026-09-24，数据结束更早且被反复使用，全部为历史顺序重放。",
                "new_market_collection": False, "orders_authorized": False, "current_view": "NO_VIEW"}
    save_json(OUT / "protocol.json", protocol)
    save_json(OUT / "freeze.json", {"frozen_at": now(), "before_first_fit_and_account": True,
               "hashes": {r: digest(OUT / r) for r in sources} | {"protocol.json": digest(OUT / "protocol.json")}})


def run():
    if (OUT / "summary.json").exists():
        raise RuntimeError("V3已有结果，禁止改参数重跑。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for name, h in frozen["hashes"].items():
        assert digest(OUT / name) == h
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    x = design(d)
    controls = pd.read_parquet(OUT / "inputs/controls.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    signals = pd.read_parquet(OUT / "inputs/signals.parquet")
    start, end = int(d.index[d.date >= "2021-01-04"][0]), len(d) - 1
    models, forecasts, schedule = [], [], []
    active_models = {}
    mean = None
    for i in range(start, end + 1):
        if (i - start) % 21 == 0:
            pool = controls[(controls.exit_idx <= i) & (controls.signal_idx >= i - 504)].copy()
            indices = pool.signal_idx.astype(int).to_numpy()
            mask = np.isfinite(x.loc[indices, FEATURES["PHASE_VOLUME"]].to_numpy(float)).all(axis=1)
            pool, indices = pool.iloc[np.flatnonzero(mask)], indices[mask]
            if len(indices) < 252:
                raise RuntimeError("固定日样本训练数量不足，不能自动减门槛。")
            y = pool.net_return.to_numpy(float)
            assert int(pool.exit_idx.max()) <= i
            mean = float(y.mean())
            for name, columns in FEATURES.items():
                model = fit_state(x, y, indices, columns, i, d.date.iloc[i], name)
                model["latest_mature_exit_idx"] = int(pool.exit_idx.max())
                models.append(model)
                active_models[name] = model
            schedule.append({"fit_idx": i, "date": d.date.iloc[i], "n_train": len(indices),
                             "mature_mean": mean, "latest_mature_exit_idx": int(pool.exit_idx.max())})
        rows = {"MATURE_MEAN": mean} | {name: predict(model, x.iloc[i]) for name, model in active_models.items()}
        forecasts.append({"idx": i, "date": d.date.iloc[i], "fit_idx": active_models["BACKGROUND"]["fit_idx"], **rows})
    out = OUT / "results"
    out.mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    for model in models:
        save_json(OUT / "models" / f"{model['model']}_{model['as_of_date']}.json", model)
    pred = pd.DataFrame(forecasts)
    pred.to_parquet(out / "daily_forecasts.parquet", index=False)
    save_csv(out / "daily_forecasts.csv", pred)
    save_csv(out / "fit_schedule.csv", schedule)
    x.to_parquet(out / "model_features.parquet", index=False)
    evaluation = pred.merge(controls[["signal_idx", "exit_idx", "net_return"]], left_on="idx", right_on="signal_idx", how="left")
    for model in ("MATURE_MEAN", *FEATURES):
        evaluation[f"{model}_squared_error"] = (evaluation[model] - evaluation.net_return) ** 2
    save_csv(out / "prediction_evaluation.csv", evaluation)
    signal_predictions = signals.merge(pred, left_on="signal_idx", right_on="idx", how="inner")
    save_csv(out / "signal_predictions.csv", signal_predictions)
    accounts, ledgers = [], {}
    for model in ("MATURE_MEAN", *FEATURES):
        dec = pd.DataFrame([{"idx": int(row.idx), "family": family, "FULL": bool(getattr(row, model) > 0)}
                            for row in pred.itertuples() for family in FAMILIES])
        save_csv(out / f"decisions_{model}.csv", dec)
        for cost in COSTS:
            ledger, trades, rejected, terminal = account(d, div, signals, dec, start, end, "FULL", cost)
            folder = out / "accounts" / model / cost
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "daily.parquet", index=False)
            save_csv(folder / "daily.csv", ledger)
            save_csv(folder / "trades.csv", trades if len(trades) else pd.DataFrame(columns=["signal_id", "entry_idx", "exit_idx", "entry_date", "exit_date", "net_pnl", "holding_sessions"]))
            save_csv(folder / "rejected.csv", rejected if len(rejected) else pd.DataFrame(columns=["date", "signal_id", "reason"]))
            m = metrics(ledger, trades, terminal)
            save_json(folder / "metrics.json", m)
            accounts.append(dict(model=model, cost=cost, **m))
            if cost == "STRESS":
                r = ledger.daily_return.to_numpy(float).copy()
                previous = ledger.equity_cny.iloc[-2]
                r[-1] = (ledger.equity_cny.iloc[-1] - terminal["terminal_haircut_cny"]) / previous - 1
                ledgers[model] = r
    result = pd.DataFrame(accounts)
    save_csv(out / "account_comparison.csv", result)
    # 联合抽样保留时间相关性；不把正态近似分数当样本外确定性。
    n = len(pred)
    rng = np.random.default_rng(20260924)
    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / 20))))
    draws = ((starts[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(out / "bootstrap_indices.npz", indices=draws)
    delta_loss = (evaluation.BACKGROUND_squared_error - evaluation.PHASE_VOLUME_squared_error).to_numpy(float)
    loss_boot = np.nanmean(delta_loss[draws], axis=1)
    returns = np.stack([ledgers["BACKGROUND"], ledgers["PHASE_VOLUME"]], axis=1)
    sample = returns[draws]
    cagr = np.prod(1 + sample, axis=1) ** (252 / n) - 1
    std = sample.std(axis=1, ddof=1)
    sharpe = np.divide(sample.mean(axis=1) * np.sqrt(252), std, out=np.full_like(std, np.nan), where=std > 1e-14)
    intervals = []
    for name, value in (("MSE_BACKGROUND_MINUS_PHASE", loss_boot), ("CAGR_PHASE_MINUS_BACKGROUND", cagr[:, 1] - cagr[:, 0]), ("SHARPE_PHASE_MINUS_BACKGROUND", sharpe[:, 1] - sharpe[:, 0])):
        valid = np.isfinite(value)
        intervals.append({"measure": name, "lower95": np.quantile(value[valid], .025) if valid.any() else None,
                          "upper95": np.quantile(value[valid], .975) if valid.any() else None,
                          "finite_draws": int(valid.sum()), "total_draws": 2000,
                          "status": "COMPUTED" if valid.all() else "PARTIAL_UNDEFINED_DRAWS" if valid.any() else "NOT_COMPUTED"})
    save_csv(out / "paired_intervals.csv", intervals)
    performance = [{"model": model, "mature_evaluations": int(evaluation.net_return.notna().sum()),
                    "mse": float(evaluation[f"{model}_squared_error"].mean()),
                    "positive_forecast_days": int((pred[model] > 0).sum()),
                    "positive_confirmed_signals": int((signal_predictions[model] > 0).sum())}
                   for model in ("MATURE_MEAN", *FEATURES)]
    save_csv(out / "prediction_comparison.csv", performance)
    phase = result[result.model == "PHASE_VOLUME"]
    summary = {"completed_at": now(), "study_id": "510300_PATTERN_DAILY_STATE_LEARNING_V3", "new_fitted_models": len(models),
               "refit_dates": len(schedule), "new_accounts": len(accounts), "shape_confirmations": len(signal_predictions),
               "model_columns": FEATURES, "prediction_performance": performance, "phase_accounts": phase.to_dict("records"),
               "numeric_phase_account_pass": bool(phase.numerical_target_pass.all()), "paired_intervals": intervals,
               "parameter_grids": 0, "market_downloads": 0, "independent_forward_observations": 0,
               "validated_goal_achieved": False, "status": "FROZEN_V3_NUMERICAL_FAILURE" if not phase.numerical_target_pass.all() else "NUMERICAL_PASS_UNVALIDATED",
               "current_view": "NO_VIEW", "position_target": "UNSET"}
    save_json(OUT / "summary.json", summary)
    print(result[["model", "cost", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles"]].to_string(index=False))
    print(pd.DataFrame(performance).to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="固定V3日样本状态学习与研究账户。")
    p.add_argument("command", choices=["freeze", "run"])
    args = p.parse_args()
    setup() if args.command == "freeze" else run()
