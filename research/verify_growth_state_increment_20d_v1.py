"""独立复算冻结增长实验的标签、系数、预测、区块样本和判据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def self_test() -> dict:
    from growth_state_increment_20d_v1 import label, panel_and_origins, walk_forward
    cfg = {"lookback": 20, "horizon": 20, "risk_floor": 1e-8, "minimum_train_months": 1,
           "last_decision_date": "2026-07-31", "ridge_penalty": 1., "capital": [200000, 20000],
           "cost": {"commission": .0002, "minimum": 5., "stress_slippage": .001, "lot": 100, "annual_days": 242}}
    prices = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4), "open": [10.] * 4,
                           "close": [10., 10., 9.8, 9.6], "dividend": [0., .5, .2, 0.]})
    result, path = label(prices, 0, 3)
    np.testing.assert_allclose(result["return_20d"], -.02, atol=1e-14)
    np.testing.assert_allclose(result["earned_dividend"], .2, atol=1e-14)
    np.testing.assert_allclose(path, [0., 0., -.02], atol=1e-14)
    np.testing.assert_allclose(result["downside_variance_20d"], .0004 / 3, atol=1e-14)
    rows = []
    for i, day in enumerate(["2020-01-01", "2020-01-20", "2020-01-21"]):
        rows.append({"origin_id": str(i), "date": day, "decision_at": day + "T15:00:00",
            "exit_at": ("2020-01-20" if i == 0 else "2020-12-31") + "T15:00:00", "return_20d": .01 if i == 0 else 99.,
            "downside_variance_20d": .0001, "is_release": True, "is_weekly": True,
            "price_m20": .01, "price_log_sigma20": -4., "price_log_down20": -9., "growth_level": 1., "growth_change": .1,
            "stale_growth_level": np.nan, "stale_growth_change": np.nan, "close": 4., "down20": .0001})
    pred, models = walk_forward(pd.DataFrame(rows), cfg)
    assert len(pred) == 1 and pred.iloc[0].date == "2020-01-21"
    assert all(m["training_ids"] == ["0"] for m in models)
    np.testing.assert_allclose(pred.prediction_P1, [.01], atol=1e-14)
    with tempfile.TemporaryDirectory(prefix="growth_clock_synthetic_") as directory:
        root = Path(directory)
        (root / "inputs").mkdir()
        days = pd.bdate_range("2014-12-01", "2026-09-11")
        close = 10. * np.exp(.005 * np.sin(np.arange(len(days)) / 3))
        frame = pd.DataFrame({"date": days, "open": close, "close": close, "dividend": 0.})
        frame["total_simple"] = frame.close.pct_change()
        frame.to_parquet(root / "inputs/market.parquet", index=False)
        months = pd.period_range("2015-01", "2026-07", freq="M")
        macro_dates = [str(m.end_time.date()) + "T09:30:00+08:00" for m in months]
        macro_dates[1] = "2015-02-27T15:01:00+08:00"
        macro = pd.DataFrame({"reference_period": months.astype(str), "first_release_value": np.arange(139) / 10 + 49., "available_at": macro_dates})
        macro.to_parquet(root / "inputs/pmi_new_orders.parquet", index=False)
        pd.DataFrame({"trade_date": days, "is_open": True}).to_parquet(root / "inputs/calendar.parquet", index=False)
        panel, origins = panel_and_origins(root, cfg)
        assert panel.loc[panel.date == "2015-02-27", "reference_period"].iloc[0] == "2015-01"
        assert panel.loc[panel.date == "2015-03-02", "reference_period"].iloc[0] == "2015-02"
        assert origins.loc[(origins.reference_period == "2015-02") & origins.is_release, "date"].iloc[0] == "2015-03-02"
    return {"status": "PASS_SYNTHETIC_CLOCK_MATURITY_AND_DIVIDEND_CASES", "cases": ["入场日除息不享有", "后续分红进入现金权利与下行财富路径", "同日收盘刚成熟标签不进入训练", "未成熟极端未来标签不能影响预测", "15时01分公布移到下一交易日收盘"], "real_labels_read": 0}


def verify(study: Path) -> dict:
    cfg = read_json(study / "protocol.json")
    frozen = read_json(study / "freeze_receipt.json")
    count = 0

    def require(condition, message: str) -> None:
        nonlocal count
        if not bool(condition):
            raise ValueError("复核失败：" + message)
        count += 1

    def close(left, right, message: str, atol: float = 2e-11) -> None:
        require(np.allclose(left, right, atol=atol, rtol=1e-9, equal_nan=True), message)

    frozen_items = list(frozen["files"])
    correction_path = study / "calendar_correction_receipt.json"
    if correction_path.exists():
        frozen_items += read_json(correction_path)["additional_frozen_files"]
    for item in frozen_items:
        data = (study / item["path"]).read_bytes()
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], "冻结输入或代码：" + item["path"])
    prices = pd.read_parquet(study / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    prices["date"] = pd.to_datetime(prices.date)
    macro = pd.read_parquet(study / "inputs/pmi_new_orders.parquet").sort_values("reference_period").reset_index(drop=True)
    macro_at = pd.to_datetime(macro.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    origins = pd.read_csv(study / "results/全部周度与公布原点及20日标签.csv")
    monthly = pd.read_csv(study / "results/全部月度训练原点.csv").set_index("origin_id")
    models = read_json(study / "results/固定模型与训练集合.json")
    pred = pd.read_csv(study / "results/全部逐期预测.csv")
    primary = pd.read_csv(study / "results/月度主要评价及损失.csv")
    summary = read_json(study / "results/summary.json")
    log_returns = np.log1p(prices.total_simple.to_numpy())
    dates = pd.DatetimeIndex(prices.date)
    for row in origins.itertuples():
        j = int(row.origin_index)
        require(dates[j].date().isoformat() == row.date, "观察原点日期")
        expected_at = dates[j] + pd.Timedelta(hours=15)
        require(pd.Timestamp(row.decision_at) == expected_at, "观察时钟")
        admitted = np.flatnonzero((macro_at <= expected_at).to_numpy())
        require(len(admitted) >= 2, "增长变化具备前月")
        k = int(admitted[-1])
        require(row.reference_period == macro.reference_period.iloc[k], "最新已知月份")
        close(row.growth_level, macro.first_release_value.iloc[k] - 50, "增长水平")
        close(row.growth_change, macro.first_release_value.iloc[k] - macro.first_release_value.iloc[k - 1], "增长变化")
        close(row.price_m20, np.expm1(log_returns[j - 19:j + 1].sum()), "价格动量")
        trailing = prices.total_simple.iloc[j - 19:j + 1].to_numpy()
        close(row.price_log_sigma20, np.log(trailing.std(ddof=1)), "价格波动")
        down20 = max(cfg["risk_floor"], np.minimum(trailing, 0).dot(np.minimum(trailing, 0)) / 20)
        close(row.price_log_down20, np.log(down20), "历史下行方差")
        wealth, cash, daily = float(prices.open.iloc[j + 1]), 0., []
        for t in range(j + 1, j + 21):
            if t > j + 1:
                cash += float(prices.dividend.iloc[t])
            next_value = float(prices.close.iloc[t]) + cash
            daily.append(next_value / wealth - 1)
            wealth = next_value
        close(row.return_20d, wealth / prices.open.iloc[j + 1] - 1, "含分红二十日标签")
        close(row.downside_variance_20d, sum(min(x, 0.) ** 2 for x in daily) / 20, "可成交后下行方差")
        require(row.entry_date == dates[j + 1].date().isoformat() and row.exit_date == dates[j + 20].date().isoformat(), "首尾20个交易日")
    lookup = {m["fit_id"]: m for m in models}
    require(len(lookup) == len(models), "拟合身份唯一")
    for m in models:
        tr = monthly.loc[m["training_ids"]]
        require((pd.to_datetime(tr.exit_at) < pd.Timestamp(m["first_used_at"])).all(), "固定训练集成熟")
        x = tr[m["columns"]].to_numpy(float)
        y_raw = tr.return_20d.to_numpy() if m["model"].startswith("P") else tr.downside_variance_20d.to_numpy()
        y = y_raw if m["model"].startswith("P") else np.log(np.maximum(y_raw, cfg["risk_floor"]))
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.
        z = (x - mean) / scale
        # 独立通过增广最小二乘求解，与生产脚本的正规方程不同。
        design = np.vstack([z / np.sqrt(len(tr)), np.sqrt(cfg["ridge_penalty"]) * np.eye(len(mean))])
        target = np.r_[(y - y.mean()) / np.sqrt(len(tr)), np.zeros(len(mean))]
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        close(m["beta"], beta, "固定岭回归系数复算")
        close(m["mean"], mean, "训练均值")
        close(m["scale"], scale, "训练标准化")
        close(m["intercept"], y.mean(), "训练截距")
        smear = 1. if m["model"].startswith("P") else np.exp(y - y.mean() - z @ beta).mean()
        close(m["smearing"], smear, "训练内风险均值修正")
    for row in pred.to_dict("records"):
        for key in ("P0", "P1", "R0", "R1"):
            model = lookup[row["fit_" + key]]
            tr = monthly.loc[model["training_ids"]]
            expected_ids = monthly.index[pd.to_datetime(monthly.exit_at) < pd.Timestamp(row["decision_at"])].tolist()
            require(model["training_ids"] == expected_ids, "只用全部已成熟月度训练集")
            values = np.array([row[c] for c in model["columns"]])
            linear = model["intercept"] + np.dot((values - model["mean"]) / model["scale"], model["beta"])
            result = np.exp(linear) * model["smearing"] if key.startswith("R") else linear
            close(row["prediction_" + key], result, "逐期预测等式")
            if key in ("P1", "R1") and row["refresh_eligible"]:
                values[-2:] = [row["stale_growth_level"], row["stale_growth_change"]]
                linear = model["intercept"] + np.dot((values - model["mean"]) / model["scale"], model["beta"])
                stale = np.exp(linear) * model["smearing"] if key.startswith("R") else linear
                close(row["prediction_" + key + "_stale"], stale, "同系数及同目标刷新对照")
        close(row["prediction_MEAN"], tr.return_20d.mean(), "成熟收益均值")
        close(row["prediction_RMEAN"], tr.downside_variance_20d.mean(), "成熟下行方差均值")
    require(primary.origin_id.tolist() == pred.loc[pred.is_release, "origin_id"].tolist(), "主要评价每月一次")
    metrics = pd.read_csv(study / "results/八项基准汇总.csv").set_index("model")
    for key in metrics.index:
        yhat = primary["prediction_" + key]
        risk = key in ("R0", "R1", "RMEAN", "DOWN20")
        y = primary.downside_variance_20d if risk else primary.return_20d
        loss = np.log(yhat) + y / yhat if risk else (y - yhat) ** 2
        close(primary["loss_" + key], loss, "每月主要损失")
        close(metrics.loc[key, "mean_loss"], loss.mean(), "八项基准平均损失")
    indices = np.load(study / "results/固定区块抽样索引.npz")["indices"]
    draws = pd.read_csv(study / "results/固定区块损失改善抽样.csv")
    n = len(primary)
    require(indices.shape == (cfg["bootstrap"]["draws"], n), "固定抽样索引维度")
    require(indices.min() >= 0 and indices.max() < n, "抽样索引范围")
    for block in range(0, n, cfg["bootstrap"]["month_block"]):
        require((np.diff(indices[:, block:min(n, block + cfg["bootstrap"]["month_block"])], axis=1) == 1).all(), "连续月块")
    for item in summary["comparisons"]:
        ref, cand = item["reference"], item["candidate"]
        difference = (primary["loss_" + ref] - primary["loss_" + cand]).to_numpy()
        samples = difference[indices].mean(axis=1)
        close(draws[ref + "_minus_" + cand], samples, "完整固定损失抽样")
        close(item["mean_improvement"], difference.mean(), "平均增量")
        close(item["one_sided_95pct_lower"], np.quantile(samples, .05), "预定单侧下界")
        close(item["first_half_improvement"], difference[:n // 2].mean(), "前半增量")
        close(item["second_half_improvement"], difference[n // 2:].mean(), "后半增量")
    for candidate in ("P1", "R1"):
        comparisons = [x for x in summary["comparisons"] if x["candidate"] == candidate]
        passed = all(x["mean_improvement"] > 0 and x["one_sided_95pct_lower"] > 0 for x in comparisons) and comparisons[0]["first_half_improvement"] > 0 and comparisons[0]["second_half_improvement"] > 0
        require(summary["verdicts"][candidate].startswith("PASS") == passed, "独立通道冻结判据")
    require(summary["model_fits"] == len(models) and summary["new_accounts"] == 0 and summary["independent_forward_events"] == 0, "运行计数")
    return {"status": "PASS_INDEPENDENT_SAVED_LABEL_MODEL_AND_GATE_RECOMPUTATION", "checks": count,
        "label_origins": len(origins), "monthly_training_origins": len(monthly), "review_predictions": len(pred), "primary_evaluation_months": n,
        "fixed_models_recomputed_for_verification": len(models), "new_research_models": 0, "new_accounts": 0,
        "new_random_draws": 0, "verdicts": summary["verdicts"],
        "limit": "离线独立复算保存实验，不构成独立市场样本或全部历史原文认证。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增长状态二十日实验的合成边界测试及只读独立复算")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/research/510300_growth_state_increment_20d_v1")
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else verify(args.study_dir), ensure_ascii=False, indent=2))
