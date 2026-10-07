"""用包内保存的全部原始图表输入，在临时目录重绘宏观背景图。"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="重绘包内两张历史宏观背景图，不改变冻结输入")
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    study, out = args.study_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="510300_context_plot_") as temp:
        stage = Path(temp)
        (stage / "inputs").mkdir()
        (stage / "results").mkdir()
        mapping = {
            "inputs/market_daily.parquet": "inputs/market_daily.parquet",
            "inputs/official_releases.csv": "inputs/official_releases.csv",
            "inputs/policy_context_events.json": "inputs/policy_context_events.json",
            "inputs/money_market_daily.parquet": "results/510300与当时已知货币数据_全部日线.parquet",
            "inputs/prior_six_cases.csv": "results/事前预期差与510300全部事件.csv",
        }
        for source, destination in mapping.items():
            shutil.copy2(study / source, stage / destination)
        source_code = study / "code/plot_money_expectations_policy_v1.py"
        subprocess.run([sys.executable, str(source_code), "--study-dir", str(stage)], check=True)
        for name in ("510300_货币预期与政策全景", "510300_M1M2_完整历史走势"):
            for suffix in ("png", "svg"):
                shutil.copy2(stage / "figures" / f"{name}.{suffix}", out / f"{name}.{suffix}")
    print("两张宏观背景图已由包内全部日频输入重绘。")


if __name__ == "__main__":
    main()
