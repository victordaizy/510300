"""只读复核已保存的旧实验诊断、逐日信息对齐和政策案例；不拟合新模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(study: Path) -> dict:
    checks: list[str] = []

    def require(ok: bool, name: str) -> None:
        if not ok:
            raise ValueError("复核未通过：" + name)
        checks.append(name)

    def close(actual, expected, name: str, atol: float = 2e-12) -> None:
        require(bool(np.allclose(actual, expected, rtol=1e-10, atol=atol, equal_nan=True)), name)

    freeze = read_json(study / "scope_freeze.json")
    for item in freeze["inputs"]:
        data = (study / "inputs" / item["input"]).read_bytes()
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], "冻结输入身份：" + item["input"])

    events = pd.read_csv(study / "inputs/old_events.csv").set_index("event_id")
    source_pred = pd.read_csv(study / "inputs/old_predictions.csv").set_index("event_id")
    pred = pd.read_csv(study / "results/旧五日实验_补充基准逐事件.csv").set_index("event_id")
    models = read_json(study / "inputs/old_models.json")
    lookup = {(r["event_id"], r["arm"]): r for r in models}
    require(len(pred) == 24 and len(models) == 48 and len(events) == 76, "旧样本数量")
    require(pred.index.equals(source_pred.index), "旧预测事件及次序")
    pd.testing.assert_frame_equal(pred[source_pred.columns], source_pred, check_exact=False, atol=1e-14, rtol=1e-12)
    checks.append("旧预测原列保留")
    for event_id, row in pred.iterrows():
        ids = lookup[(event_id, "A")]["training_events"]
        tr = events.loc[ids]
        require(bool((pd.to_datetime(tr.exit_at, utc=True) < pd.Timestamp(row.observation_at)).all()), "训练标签成熟：" + event_id)
        require(len(ids) == row.training_events and event_id not in ids, "训练集合：" + event_id)
        close(row.prediction_mature_mean, tr.residual_5d_gross_return.mean(), "成熟训练均值：" + event_id)
        close(row.actual_5d_return, events.loc[event_id, "residual_5d_gross_return"], "已保存五日标签：" + event_id)
        for arm in ("A", "B"):
            m = lookup[(event_id, arm)]
            require(m["training_events"] == ids, "A/B共同训练集合：" + event_id + arm)
            x = events.loc[event_id, m["columns"]].to_numpy(dtype=float)
            prediction = float(m["intercept"] + ((x - np.asarray(m["mean"])) / np.asarray(m["scale"])) @ np.asarray(m["beta"]))
            close(prediction, row["prediction_" + arm], "已冻结系数预测等式：" + event_id + arm)
    close(pred.prediction_zero, np.zeros(len(pred)), "零收益基准")
    diag = read_json(study / "results/旧五日诊断.json")
    metrics = pd.read_csv(study / "results/旧五日实验_补充基准汇总.csv").set_index("model")
    for model in ("A", "B", "mature_mean", "zero"):
        values = pred["prediction_" + model]
        mse = float(((values - pred.actual_5d_return) ** 2).mean())
        close(metrics.loc[model, "mse"], mse, "平方误差：" + model)
        close(metrics.loc[model, "rmse_pp"], np.sqrt(mse) * 100, "百分点RMSE：" + model)
        require(int(metrics.loc[model, "positive_predictions"]) == int((values > 0).sum()), "正预测数量：" + model)
        require(int(metrics.loc[model, "negative_predictions"]) == int((values < 0).sum()), "负预测数量：" + model)
    close(metrics.loc["B", "mse"] / metrics.loc["A", "mse"] - 1, diag["relative_B_MSE_increase"], "相对MSE变化")
    require(bool((pred.prediction_A < 0).all() and (pred.prediction_B < 0).all()), "A和B全部为负")
    draws = pd.read_csv(study / "results/旧五日诊断_固定抽样均值.csv").iloc[:, 0]
    require(len(draws) == 10000, "保存抽样次数")
    close(np.quantile(draws, [.05, .95]), diag["A_minus_B_error_90pct_two_sided"], "保存抽样双侧区间")
    require(diag["A_minus_B_error_90pct_two_sided"][0] < 0 < diag["A_minus_B_error_90pct_two_sided"][1], "区间跨零")

    market = pd.read_parquet(study / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    panel = pd.read_parquet(study / "results/全部日线_增长与操作利率已知记录.parquet")
    market["date"] = pd.to_datetime(market.date)
    require(len(panel) == len(market) == 3476, "全部日线点保留")
    pd.testing.assert_frame_equal(panel[market.columns], market, check_exact=True)
    checks.append("逐日行情原列完全一致")
    csv_panel = pd.read_csv(study / "results/全部日线_增长与操作利率已知记录.csv")
    require(bool(np.array_equal(pd.to_datetime(csv_panel.date).to_numpy(dtype="datetime64[ns]"), panel.date.to_numpy(dtype="datetime64[ns]"))), "CSV/Parquet日期一致")
    for col in panel.select_dtypes(include=[np.number]).columns:
        close(csv_panel[col], panel[col], "CSV/Parquet数值一致：" + col)
    require(panel.date.is_monotonic_increasing and panel.date.is_unique, "日线顺序及唯一性")
    decision = panel.date + pd.Timedelta(hours=15)
    require(decision.equals(panel.decision_at), "每日15时判断时钟")
    pmi = pd.read_parquet(study / "inputs/pmi_new_orders.parquet").sort_values("available_at").reset_index(drop=True)
    rates = pd.read_parquet(study / "inputs/operation_rate_records.parquet").sort_values("published_at").reset_index(drop=True)
    pmi_at = pd.to_datetime(pmi.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    rate_at = pd.to_datetime(rates.published_at)
    p_idx = np.searchsorted(pmi_at.to_numpy(), decision.to_numpy(), side="right") - 1
    r_idx = np.searchsorted(rate_at.to_numpy(), decision.to_numpy(), side="right") - 1

    def expected(values: np.ndarray, indexes: np.ndarray, cutoff: str) -> np.ndarray:
        out = np.full(len(panel), np.nan)
        admitted = (indexes >= 0) & (panel.date <= pd.Timestamp(cutoff)).to_numpy()
        out[admitted] = values[indexes[admitted]]
        return out

    close(panel.first_release_value, expected(pmi.first_release_value.to_numpy(), p_idx, "2026-07-31"), "增长状态只取当时已公布值")
    close(panel.pmi_change_1m, expected(pmi.first_release_value.diff().to_numpy(), p_idx, "2026-07-31"), "增长月度变化")
    close(panel.pmi_above50, panel.first_release_value - 50, "增长相对50位置")
    close(panel.seven_day_rate_percent, expected(rates.seven_day_rate_percent.to_numpy(), r_idx, "2026-08-14"), "操作利率只取当时已记录值")
    close(panel.total_return_drawdown, market.wealth / market.wealth.cummax() - 1, "完整含分红回撤")
    ledger = pd.read_csv(study / "results/宏观信息事件账本_继承来源.csv")
    require(len(pmi) == 139 and len(rates) == 25 and len(ledger) == 164, "宏观事件数量")
    require(ledger.event_id.is_unique and ledger.surprise.isna().all() and ledger.expectation.isna().all(), "事件唯一且未伪填预期")
    require((ledger.family == "增长").sum() == 139 and (ledger.family == "利率条件").sum() == 25, "分通道事件数量")
    identities = pd.read_csv(study / "evidence/继承原文身份核对.csv")
    require(len(identities) == 164 and identities.hash_match.all() and identities.expected_sha256.equals(identities.current_sha256), "保存的原文身份核对回执一致")

    clock = read_json(study / "results/九月案例_时钟与价格核对.json")
    prices = market.set_index("date")
    old = events.loc["MONEY_2024-08"]
    old_open = prices.loc[pd.Timestamp(old.entry_date), "open"]
    old_close = prices.loc[pd.Timestamp(old.exit_date), "close"]
    update_open = prices.loc[pd.Timestamp("2024-09-25"), "open"]
    close(clock["old_window_return"], old_close / old_open - 1, "旧五日案例价格等式")
    close(clock["old_window_return"], old.residual_5d_gross_return, "旧标签没有改变")
    close(clock["illustrative_update_open_to_old_exit"], old_close / update_open - 1, "动态成交时钟示意等式")
    require(clock["is_strategy_return"] is False and clock["is_policy_causal_contribution"] is False, "案例不冒充新策略或政策因果")
    policy = pd.read_csv(study / "results/政策宣布与实施_九月案例.csv")
    require(len(policy) == 3 and policy.expectation.isna().all() and policy.surprise.isna().all(), "案例政策没有假定预期")
    announce = pd.Timestamp(policy.loc[policy.kind == "宣布", "available_at"].iloc[0])
    require(announce < pd.Timestamp(clock["update_observation"]) < pd.Timestamp(clock["update_first_execution"]), "宣布至观察再到可成交顺序")
    require(policy.loc[policy.kind == "实施", "available_at"].iloc[0] == "2024-09-27T23:59:59+08:00", "日期精度采用保守上界")
    case_rows = pd.read_csv(study / "results/九月案例_全部交易日.csv", parse_dates=["date"])
    expected_rows = market[(market.date >= "2024-09-13") & (market.date <= "2024-09-30")]
    require(case_rows.date.tolist() == expected_rows.date.tolist(), "案例保留全部交易日")
    close(case_rows.close, expected_rows.close, "案例每日收盘")

    summary = read_json(study / "results/summary.json")
    require(all(summary[k] == 0 for k in ("new_twenty_day_models", "new_accounts", "new_20d_labels", "independent_forward_events")), "新二十日研究及账户未运行")
    require(summary["whole_macro_program_complete"] is False and summary["full_account_target_established"] is False, "未宣称完整宏观目标达成")
    require(summary["capital_main"] == 200000 and summary["capital_cost_comparison"] == 20000, "账户本金设定")
    require(freeze["old_five_day_result"] == "REJECTED_FROZEN_NO_RELIABLE_MESSAGE_INCREMENT", "旧失败裁决保留")
    return {"status": "PASS_SAVED_NUMERIC_TIMING_AND_INPUT_IDENTITY", "checks": len(checks), "check_names": checks,
            "old_coefficient_prediction_equations": 48, "mature_mean_equations": 24, "daily_rows": len(panel),
            "macro_ledger_rows": len(ledger), "new_model_fits": 0, "new_accounts": 0, "new_20d_labels": 0,
            "limits": "保存输出数值与时钟关系复核；不重新获取或认证164份原始公开版本，不重抽样、不重新拟合，不证明科学有效性或独立样本。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线复核宏观重订阶段的保存结果")
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/research/510300_macro_dynamic_reframe_v1")
    args = parser.parse_args()
    print(json.dumps(verify(args.study_dir.resolve()), ensure_ascii=False, indent=2))
