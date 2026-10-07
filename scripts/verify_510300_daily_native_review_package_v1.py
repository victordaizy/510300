"""只读取解压副本并复算保存结果；不训练、不回放账户、不抽样。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(package: Path) -> dict:
    project = package / "project"
    report = project / "reports/research/510300_daily_native_baseline_v1"
    path_map = pd.read_csv(package / "06_SOURCE_PATH_MAP.csv")
    lookup = dict(zip(path_map.original_path, path_map.package_path))
    frozen = json.loads((report / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        require(item["path"] in lookup, "冻结文件缺少相对路径映射")
        value = (package / lookup[item["path"]]).read_bytes()
        require(hashlib.sha256(value).hexdigest() == item["sha256"], "冻结文件身份不符：" + item["path"])
    index = json.loads((report / "file_index.json").read_text(encoding="utf-8"))
    for item in index["files"]:
        value = (report / item["path"]).read_bytes()
        require(len(value) == item["bytes"] and hashlib.sha256(value).hexdigest() == item["sha256"], "本批报告索引不符：" + item["path"])

    original_verifier = project / "scripts/verify_510300_intraday_process_review_package_v1.py"
    spec = importlib.util.spec_from_file_location("original_saved_verifier", original_verifier)
    require(spec is not None and spec.loader is not None, "无法读取原复算入口")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.verify(package)

    sys.path.insert(0, str(project))
    from research import daily_native_baseline_v1 as engine

    require(Path(engine.__file__).resolve().is_relative_to(project.resolve()), "新基准模块不属于解压副本")
    cfg = json.loads((project / "config/510300_daily_native_baseline_v1.json").read_text(encoding="utf-8"))
    parent = json.loads((project / cfg["parent_config"]).read_text(encoding="utf-8"))
    ledger = pd.read_parquet(report / "native_accounts.parquet")
    metrics = pd.read_csv(report / "10_D_native完整账户指标.csv")
    cycles = pd.read_parquet(report / "native_cycles.parquet")
    native = engine.verify_accounts(ledger, metrics, cycles, cfg)

    panel = pd.read_parquet(report / "daily_asof_panel.parquet").set_index("date")
    forecasts = pd.read_parquet(report / "native_forecasts.parquet")
    model_rows = json.loads((report / "native_saved_models.json").read_text(encoding="utf-8"))
    models = {pd.Timestamp(row["fit_date"]): row for row in model_rows}
    model_cfg = copy.deepcopy(parent)
    model_cfg["model"].update(cfg["model"])
    differences = []
    for row in forecasts.loc[forecasts.prediction_D.notna()].itertuples():
        model = models[pd.Timestamp(row.fit_date)]
        require(pd.Timestamp(model["last_training_label_exit"]) <= pd.Timestamp(row.fit_date) <= row.date,
                "保存模型使用了未来成熟日期")
        value = engine.old.predict(model, panel.loc[row.date], model_cfg)
        differences.append(abs(value - row.prediction_D))
    require(len(differences) == 706 and max(differences) < 1e-12, "新基准保存预测复算失败")

    original_report = project / cfg["parent_output"]
    old_predictions = pd.read_parquet(original_report / "06_滚动预测.parquet")
    pair = old_predictions.loc[old_predictions.prediction_D.notna() & old_predictions.return_h2.notna(),
                               ["date", "prediction_D", "return_h2"]].merge(
        forecasts[["date", "prediction_D"]], on="date", validate="one_to_one", suffixes=("_common", "_native"))
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    require(len(pair) == summary["paired_original_forecast_days"] == 364, "共同预测日期数不符")
    for label in ("common", "native"):
        mse = float(np.mean((pair["prediction_D_" + label] - pair.return_h2) ** 2))
        require(abs(mse - summary["paired_mse_" + label]) < 1e-12, "配对MSE点值不符")

    accounts = pd.read_parquet(original_report / "08_完整账户逐日账本.parquet")
    selected = accounts.loc[accounts.capital.eq(200000) & accounts.cost.eq("BASE") & accounts.model.isin(["C", "D"])]
    paired_accounts = selected.pivot(index="date", columns="model", values="equity").sort_index()
    difference = paired_accounts.C - paired_accounts.D
    daily_difference = difference.diff().fillna(difference.iloc[0])
    september = float(daily_difference.loc[daily_difference.index.to_period("M") == pd.Period("2024-09")].sum())
    total = float(difference.iloc[-1])
    require(abs(september) < 1e-8 and abs(total - 3729.8021) < 1e-6, "C−D边际归因修正不符")

    return {
        "status": "PASS_EXTRACTED_FIRST_BATCH_SAVED_RECOMPUTATION",
        "native_frozen_files": len(frozen["files"]),
        "native_report_indexed_files": len(index["files"]),
        "original_saved_recomputation": original,
        "native_saved_recomputation": native,
        "native_predictions_recomputed": len(differences),
        "maximum_native_prediction_error": max(differences),
        "paired_prediction_days": len(pair),
        "c_minus_d_total_cny": total,
        "c_minus_d_september_2024_cny": september,
        "new_fits": 0,
        "new_account_simulations": 0,
        "new_random_draws": 0,
        "network_requests": 0,
        "confidence_intervals_recomputed": False,
        "independent_validation_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="复算第一批审阅包内保存账户和预测")
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
