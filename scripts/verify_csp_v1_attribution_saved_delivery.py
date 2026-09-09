"""读取归因ZIP或解压目录复算保存统计；不生成新账户、不拟合、不重新抽样。"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

CURRENT = "project/reports/research/510300_csp_v1_static_vs_timing_attribution_v1/"
PARENT = "project/reports/research/510300_conditional_score_policy_v1/"


def main() -> None:
    parser = argparse.ArgumentParser(description="仅复算保存账本；可核对已保存区块样本，不重新随机抽样")
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-saved-samples", action="store_true", help="额外使用NumPy核对已经保存的2,000组索引与统计")
    args = parser.parse_args()
    archive = zipfile.ZipFile(args.zip) if args.zip else None

    def read(name):
        return archive.read(name) if archive else (args.root / name).read_bytes()

    def rows(name):
        return list(csv.DictReader(io.StringIO(read(name).decode("utf-8-sig"))))

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    def recalc(values):
        mu = statistics.mean(values) * 242
        sigma = statistics.stdev(values) * math.sqrt(242)
        wealth = peak = 1.0
        dd = 0.0
        for value in values:
            wealth *= 1 + value
            peak = max(peak, wealth)
            dd = min(dd, wealth / peak - 1)
        return {"cumulative_return": wealth - 1, "annualized_return": wealth ** (242 / len(values)) - 1,
                "annualized_arithmetic_mean": mu, "annualized_volatility": sigma,
                "net_sharpe": mu / sigma, "max_drawdown": dd, "utility_gamma5": mu - 2.5 * sigma ** 2,
                "ending_equity": 200000 * wealth}

    try:
        result = json.loads(read(CURRENT + "result.json"))
        prior = json.loads(read(PARENT + "result.json"))
        saved_stats = json.loads(read(CURRENT + "statistics.json"))
        check(result["state"] == "COMPLETED_DIAGNOSTIC_ONLY_NO_PROMOTION", "研究终态不符")
        returns, dates_by_key, checks = {}, {}, {}
        for key, expected in {**prior["economics"], **result["economics"]}.items():
            prefix = CURRENT + "accounts/" if key.startswith(("C0_", "C1_")) else PARENT + "evaluation/"
            ledger = rows(prefix + key + "_ledger.csv")
            dates = [row["date"] for row in ledger]
            check(len(dates) == 1604 and len(set(dates)) == 1604, f"日期条数错误：{key}")
            check(dates[0] == "2020-01-02" and dates[-1] == "2026-08-14", "起止日期错误")
            check(all(float(row["cash"]) >= -1e-7 and int(float(row["shares"])) % 100 == 0 for row in ledger), "现金/整手错误")
            values = [float(row["net_return"]) for row in ledger]
            actual = recalc(values)
            actual["total_friction"] = sum(float(row["commission"]) + float(row["slippage_cost"]) for row in ledger)
            errors = {metric: abs(value - expected[metric]) for metric, value in actual.items()}
            check(max(errors.values()) < 1e-7, f"保存账本复算不符：{key}/{errors}")
            returns[key], dates_by_key[key] = values, dates
            checks[key] = {"rows": len(ledger), "maximum_metric_error": max(errors.values())}
        for key, expected in result["ideal_economics"].items():
            ledger = rows(CURRENT + "ideal/" + key + ".csv")
            dates_by_key[key] = [row["date"] for row in ledger]
            values = [float(row["net_return"]) for row in ledger]
            returns[key] = values
            actual = recalc(values)
            errors = {metric: abs(value - expected[metric]) for metric, value in actual.items() if metric in expected}
            check(max(errors.values()) < 1e-10, f"理想路径统计不符：{key}")
            match = result["ideal_matching"][key]
            reference = returns[match["reference"]]
            check(max(abs(a - match["ratio"] * b) for a, b in zip(values, reference)) < 1e-12, "理想净收益缩放不符")
            checks[key] = {"rows": len(ledger), "maximum_metric_error": max(errors.values()), "tradable": False}
        reference_dates = dates_by_key["FULL_BASE"]
        check(all(dates == reference_dates for dates in dates_by_key.values()), "联合日历不一致")
        sample_check = {"performed": False}
        if args.check_saved_samples:
            import numpy as np
            with np.load(io.BytesIO(read(CURRENT + "bootstrap_indices.npz")), allow_pickle=False) as packed:
                indices = packed["indices"]
            with np.load(io.BytesIO(read(CURRENT + "bootstrap_samples.npz")), allow_pickle=False) as packed:
                samples, keys, names = packed["samples"], packed["path_order"].tolist(), packed["statistic_order"].tolist()
            check(keys == saved_stats["path_order"] and names == saved_stats["statistic_order"], "抽样列顺序不符")
            check(indices.shape == (2000, 1604), "保存区块预算不符")
            data = np.column_stack([returns[key] for key in keys])
            maximum = 0.0
            for rep, ix in enumerate(indices):
                check((ix >= 0).all() and (ix < 1604).all(), "保存区块索引超界")
                chosen = data[ix]
                mu = chosen.mean(axis=0) * 242
                sigma = chosen.std(axis=0, ddof=1) * math.sqrt(242)
                logs = np.log1p(chosen).sum(axis=0)
                actual = np.column_stack([mu / sigma, np.expm1(logs * 242 / 1604),
                                          mu - 2.5 * sigma ** 2, mu, 200000 * np.exp(logs)])
                error = float(np.max(np.abs(actual - samples[rep])))
                maximum = max(maximum, error)
            check(maximum < 1e-7, f"原保存索引对应的抽样统计不符：{maximum}")
            interval_error = 0.0
            for pair, expected in saved_stats["paired_increments"].items():
                left, right = pair.split("_MINUS_")
                delta = samples[:, keys.index(left)] - samples[:, keys.index(right)]
                quantiles = np.quantile(delta, [.025, .975], axis=0)
                for k, name in enumerate(names):
                    interval_error = max(interval_error, float(np.max(np.abs(quantiles[:, k] - expected[name]["interval95"]))))
            check(interval_error < 1e-10, "保存区间不符")
            sample_check = {"performed": True, "saved_repetitions_checked": 2000,
                            "maximum_sample_error": maximum, "maximum_interval_error": interval_error,
                            "new_random_samples": 0}
        print(json.dumps({"status": "PASS_SAVED_DELIVERY_RECOMPUTATION", "paths": checks,
                          "saved_sample_check": sample_check, "new_account_runs": 0,
                          "new_fits": 0, "new_random_samples": 0}, ensure_ascii=False, indent=2))
    finally:
        if archive:
            archive.close()


if __name__ == "__main__":
    main()
