"""仅从审阅包副本核对冻结身份、CSV导出和保存统计，不拟合、不回测、不抽样。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(package: Path) -> dict:
    project = package / "project"
    report = project / "reports/research/510300_intraday_process_increment_v1"
    cfg = json.loads((project / "config/510300_intraday_process_increment_v1.json").read_text(encoding="utf-8"))
    paths = pd.read_csv(package / "06_SOURCE_PATH_MAP.csv")
    lookup = dict(zip(paths.original_path, paths.package_path))
    frozen = json.loads((report / "freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        require(row["path"] in lookup, "冻结来源缺少包内映射：" + row["path"])
        require(digest(package / lookup[row["path"]]) == row["sha256"], "冻结来源身份不符：" + row["path"])

    exports = json.loads((package / "readable/CSV_EXPORTS.json").read_text(encoding="utf-8"))
    for row in exports:
        frame = pd.read_parquet(package / row["source"])
        actual = pd.read_csv(package / row["csv"])
        require(len(frame) == len(actual) == row["rows"], "CSV导出行数不符")
        require(list(frame.columns) == list(actual.columns), "CSV导出列不符")
        for column in frame.select_dtypes(include="number").columns:
            require(np.allclose(frame[column].astype(float), actual[column].astype(float), rtol=1e-12, atol=1e-10, equal_nan=True), "CSV数值导出不符：" + column)

    sys.path.insert(0, str(project))
    import research.intraday_process_increment_v1 as engine
    require(Path(engine.__file__).resolve().is_relative_to(project.resolve()), "加载的账户复算代码不属于解压副本")
    # 原复算函数会保存新回执，因此只对临时副本调用，原包内结果不被改写。
    with tempfile.TemporaryDirectory(prefix="只读账本复算_", dir=package.parent) as temporary:
        work = Path(temporary)
        require(work.resolve().is_relative_to(package.parent.resolve()), "临时复算目录越界")
        for name in ("08_完整账户逐日账本.parquet", "12_全部账户指标.csv", "09_全部模拟请求.csv", "11_完成持有周期.csv"):
            (work / name).write_bytes((report / name).read_bytes())
        account = engine.verify_saved(work, cfg)

    predictions = pd.read_parquet(report / "06_滚动预测.parquet")
    columns = ["prediction_" + key for key in ("D", "A", "B", "C", "A_CONTROL")]
    available = predictions.loc[predictions[columns].notna().all(axis=1) & predictions.return_h2.notna()]
    comparison = json.loads((report / "predictive_comparison.json").read_text(encoding="utf-8"))
    require(len(available) == comparison["days"] == 364, "共同预测日数不符")
    mse = {key: float(((available["prediction_" + key] - available.return_h2) ** 2).mean()) for key in ("D", "A", "B", "C", "A_CONTROL")}
    errors = [abs(mse["D"] - comparison["baseline_mse"])]
    for key in ("A", "B", "C"):
        expected = comparison["comparisons"][key]
        errors.extend([abs(mse[key] - expected["mse"]), abs(mse["D"] - mse[key] - expected["mse_reduction"])])
    errors.append(abs(mse["A_CONTROL"] - mse["A"] - comparison["A_persistence_beyond_rebound"]["mse_reduction"]))
    require(max(errors) < 1e-12, "保存预测点统计不符")

    cycles = pd.read_csv(report / "11_完成持有周期.csv")
    concentration = pd.read_csv(report / "16_全账户收益集中度.csv")
    for key, group in cycles.groupby(["capital", "cost", "model"]):
        saved = concentration.loc[concentration.capital.eq(key[0]) & concentration.cost.eq(key[1]) & concentration.model.eq(key[2])].iloc[0]
        total = float(group.net_pnl.sum())
        require(abs(total - saved.net_pnl) < 1e-6, "集中度净利润不符")
        require(abs(group.net_pnl.max() / total - saved.best_share_of_total_net) < 1e-12, "最大周期集中度不符")
        require(abs(group.net_pnl.nlargest(3).sum() / total - saved.top3_share_of_total_net) < 1e-12, "前三周期集中度不符")

    return {"verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "PASS_EXTRACTED_PACKAGE_SAVED_RESULTS_RECOMPUTATION", "frozen_files": len(frozen["files"]), "readable_csv_exports": len(exports), "account_recomputation": account, "forecast_days": len(available), "maximum_prediction_point_error": max(errors), "concentration_accounts": len(concentration), "new_model_fits": 0, "new_account_simulations": 0, "new_random_draws": 0, "network_requests": 0, "confidence_intervals_recomputed": False, "interval_note": "原实验保存配置、种子和区间，未保存逐次bootstrap索引；本次不重抽，因此不宣称已复算区间。", "external_review_performed": False, "independent_financial_validation": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="从解压审阅包核对原保存结果")
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
