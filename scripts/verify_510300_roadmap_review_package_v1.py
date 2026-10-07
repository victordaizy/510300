"""读取审阅包保存文件，复算区块统计和净周期；不产生新抽样或拟合。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_file(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(package: Path) -> dict:
    project = package / "project"
    report = project / "reports/research/510300_roadmap_execution_v1"
    first = project / "reports/research/510300_daily_native_baseline_v1"
    parent = project / "reports/research/510300_intraday_process_increment_v1"
    frozen = json_file(report / "freeze.json")
    for row in frozen["files"]:
        path = project / PureWindowsPath(row["path"]).as_posix()
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "冻结文件身份不符：" + row["path"])
    index = json_file(report / "file_index.json")
    for row in index["files"]:
        body = (report / PureWindowsPath(row["path"]).as_posix()).read_bytes()
        require(len(body) == row["bytes"] and hashlib.sha256(body).hexdigest() == row["sha256"], "报告索引不符：" + row["path"])

    previous_path = project / "scripts/verify_510300_daily_native_review_package_v1.py"
    spec = importlib.util.spec_from_file_location("first_batch_saved_verifier", previous_path)
    require(spec is not None and spec.loader is not None, "第一批复算入口无法读取")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    old_result = previous.verify(package)

    result = json_file(report / "result.json")
    tasks = pd.read_csv(report / "48项执行台账.csv")
    require(len(tasks) == 48 and tasks["任务ID"].nunique() == 48, "48项台账存在遗漏或重复")
    completed = int(tasks["本次状态"].eq("COMPLETED_IN_STATED_SCOPE").sum())
    partial = int(tasks["本次状态"].eq("PARTIAL_WITH_EXPLICIT_LIMIT").sum())
    require((completed, partial, 48 - completed - partial) == (21, 11, 16), "任务状态计数不符")
    require(not result["all_48_scientific_tasks_complete"] and not result["financial_goal_achieved"], "未完成条件被消除")

    saved_intervals = json_file(report / "uncertainty_intervals.json")
    paired = pd.read_parquet(report / "paired_account_returns.parquet")
    native = pd.read_parquet(first / "native_accounts.parquet")
    common = pd.read_parquet(parent / "08_完整账户逐日账本.parquet")
    common = common.loc[common.model.eq("D")]
    cfg = json_file(project / "config/510300_roadmap_execution_v1.json")
    with np.load(report / "bootstrap_indices.npz", allow_pickle=False) as saved:
        account_index, prediction_index = saved["account"], saved["prediction"]
    for values, dates in ((account_index, 471), (prediction_index, 364)):
        require(values.shape == (5000, dates), "保存区块索引维度不符")
        require(np.issubdtype(values.dtype, np.integer) and values.min() >= 0 and values.max() < dates, "保存区块索引越界")
        length = cfg["bootstrap"]["block_length"]
        for start in range(0, dates, length):
            block = values[:, start:min(start + length, dates)]
            require(np.all(np.diff(block, axis=1) % dates == 1), "保存索引不符合循环连续区块")

    interval_errors = []

    def check_interval(values: np.ndarray, indices: np.ndarray, multiplier: float, row: dict) -> None:
        distribution = values[indices].mean(axis=1) * multiplier
        bounds = np.quantile(distribution, [0.025, 0.975, 0.005, 0.995])
        actual = [values.mean() * multiplier, *bounds]
        keys = ["point", "nominal_95_lower", "nominal_95_upper", "local_five_comparison_lower", "local_five_comparison_upper"]
        interval_errors.extend(abs(float(value) - row[key]) for value, key in zip(actual, keys))
        require(row["dates"] == len(values), "统计样本日期数不符")

    monthly = pd.read_csv(report / "all_monthly_marginal_pnl.csv")
    monthly_error = 0.0
    for (capital, cost), part in paired.groupby(["capital", "cost"], sort=True):
        part = part.sort_values("date").reset_index(drop=True)
        left = native.loc[native.capital.eq(capital) & native.cost.eq(cost), ["date", "net_return", "equity"]]
        right = common.loc[common.capital.eq(capital) & common.cost.eq(cost), ["date", "net_return", "equity"]]
        reconstructed = left.merge(right, on="date", suffixes=("_native", "_common"), validate="one_to_one").sort_values("date").reset_index(drop=True)
        columns = ["date", "net_return_native", "equity_native", "net_return_common", "equity_common"]
        pd.testing.assert_frame_equal(part[columns], reconstructed[columns], check_exact=True)
        difference = (reconstructed.net_return_native - reconstructed.net_return_common).to_numpy()
        require(np.array_equal(difference, part.daily_return_difference.to_numpy()), "配对收益与原账本不符")
        selected = [row for row in saved_intervals if row["capital"] == capital and row["cost"] == cost]
        require(len(selected) == 1, "账户区间不存在或重复")
        check_interval(difference, account_index, cfg["primary_annual_days"], selected[0])
        cny = reconstructed.equity_native - reconstructed.equity_common
        marginal = cny.diff().fillna(cny.iloc[0])
        actual = marginal.groupby(reconstructed.date.dt.to_period("M")).sum()
        reference = monthly.loc[monthly.capital.eq(capital) & monthly.cost.eq(cost)].set_index("month")
        require(set(actual.index.astype(str)) == set(reference.index), "月份贡献覆盖不符")
        monthly_error = max(monthly_error, max(abs(float(value) - reference.loc[str(month), "delta_cny"]) for month, value in actual.items()))

    predictions = pd.read_parquet(report / "paired_predictions.parquet").sort_values("date").reset_index(drop=True)
    old_pred = pd.read_parquet(parent / "06_滚动预测.parquet")
    new_pred = pd.read_parquet(first / "native_forecasts.parquet")
    reconstructed_pred = old_pred.loc[old_pred.prediction_D.notna() & old_pred.return_h2.notna(), ["date", "prediction_D", "return_h2"]].merge(
        new_pred[["date", "prediction_D"]], on="date", suffixes=("_common", "_native"), validate="one_to_one").sort_values("date").reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions, reconstructed_pred, check_exact=True)
    errors = (predictions.prediction_D_common - predictions.return_h2) ** 2 - (predictions.prediction_D_native - predictions.return_h2) ** 2
    mse_row = [row for row in saved_intervals if row["comparison"] == "MSE_COMMON_MINUS_NATIVE"]
    require(len(mse_row) == 1, "MSE比较不存在或重复")
    check_interval(errors.to_numpy(), prediction_index, 1.0, mse_row[0])
    require(len(saved_intervals) == 5 and max(interval_errors) < 1e-12 and monthly_error < 1e-6, "区间或月份复算不符")

    cycles = pd.read_parquet(first / "native_cycles.parquet")
    payoffs = pd.read_csv(report / "payoff_distribution.csv")
    payoff_errors = []
    for row in payoffs.itertuples():
        selected = cycles.loc[cycles.capital.eq(row.capital) & cycles.cost.eq(row.cost)]
        values = selected.net_return.to_numpy()
        p, q = float((values > 0).mean()), float((values < 0).mean())
        gain = float(values[values > 0].mean()) if p else 0.0
        loss = float(-values[values < 0].mean()) if q else 0.0
        actual = {
            "profit_probability": p, "loss_probability": q, "flat_probability": float((values == 0).mean()),
            "mean_gain_net": gain, "mean_loss_net": loss, "pG_minus_qL_net": p * gain - q * loss,
            "equal_weight_cycle_mean_net": float(values.mean()), "minimum_cycle_net": float(values.min()),
            "worst_5pct_cycle_mean_net": float(np.sort(values)[:max(1, math.ceil(len(values) * 0.05))].mean()),
        }
        require(row.cycles == len(values) and not row.cost_deducted_again, "净周期计数或费用处理不符")
        payoff_errors.extend(abs(value - getattr(row, key)) for key, value in actual.items())
        for key, field in (("total_net_pnl", "net_pnl"), ("total_commission", "commission"), ("total_slippage", "slippage")):
            require(abs(float(selected[field].sum()) - getattr(row, key)) < 1e-6, "周期金额合计不符")
    require(len(payoffs) == 4 and max(payoff_errors) < 1e-12, "周期净分布复算失败")

    collector = json_file(report / "collector_current_status.json")
    for source in collector["sources"]:
        if source["status"] == "PUBLIC_ROWS_RECEIVED":
            raw = report / "raw_vintages" / collector["observation_day"] / source["raw_path"]
            require(hashlib.sha256(raw.read_bytes()).hexdigest() == source["sha256"], "公开原响应身份不符")
    trigger = json_file(report / "scheduled_trigger_verification.json")
    require(trigger["last_task_result"] == 0, "本地休市分支触发未通过")
    return {
        "status": "PASS_EXTRACTED_ROADMAP_SAVED_RECOMPUTATION", "first_batch_saved_recomputation": old_result,
        "roadmap_frozen_files": len(frozen["files"]), "roadmap_report_indexed_files": len(index["files"]),
        "tasks": 48, "task_status_counts": [completed, partial, 48 - completed - partial],
        "paired_account_days": 471, "paired_prediction_days": 364, "saved_intervals_recomputed": 5,
        "saved_draws_per_statistic": 5000, "maximum_interval_error": max(interval_errors),
        "maximum_monthly_cny_error": monthly_error, "maximum_payoff_error": max(payoff_errors),
        "source_responses_with_rows": collector["sources_with_rows"], "raw_bytes": collector["raw_bytes"],
        "new_fits": 0, "new_account_simulations": 0, "new_random_draws": 0, "network_requests": 0,
        "independent_financial_validation": False, "external_review_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="复算48任务审阅包中的已保存统计，不启动研究或采集")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    body = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
