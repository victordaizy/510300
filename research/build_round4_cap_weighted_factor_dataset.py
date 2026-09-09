"""将第四轮流通市值近似加权因子与既有可执行收益标签点时对齐。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round4.yaml"
BREADTH_FILE = ROOT / "data" / "features" / "000300_circulating_cap_weighted_breadth_daily.parquet"
BASE_DATASET_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_round4_cap_weighted_dataset.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "round4_cap_weighted_factor_dataset.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    for path in (SETTINGS_FILE, REGISTRY_FILE, BREADTH_FILE, BASE_DATASET_FILE):
        if not path.exists():
            raise FileNotFoundError(f"第四轮因子数据集输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    breadth = pd.read_parquet(BREADTH_FILE)
    base = pd.read_parquet(BASE_DATASET_FILE)
    breadth["date"] = pd.to_datetime(breadth["date"])
    base["date"] = pd.to_datetime(base["date"])
    target_columns = [
        "date",
        "exec_total_return_5d_net",
        "exec_total_return_5d_gross",
        "exec_mae_5d",
        "exec_mfe_5d",
        "label_end_date_5d",
        "exec_total_return_20d_net",
        "exec_total_return_20d_gross",
        "exec_mae_20d",
        "exec_mfe_20d",
        "label_end_date_20d",
    ]
    missing_targets = sorted(set(target_columns).difference(base.columns))
    if missing_targets:
        raise ValueError(f"既有可执行标签缺少字段：{missing_targets}")
    dataset = breadth.merge(base[target_columns], on="date", how="inner", validate="one_to_one")
    if len(dataset) != len(breadth):
        raise ValueError(f"第四轮因子与标签日期未完整对齐：{len(breadth)}->{len(dataset)}")
    for horizon in (5, 20):
        dataset[f"label_end_date_{horizon}d"] = pd.to_datetime(dataset[f"label_end_date_{horizon}d"])
    factor_columns = [item["column"] for item in registry["factor_definitions"]]
    if dataset[factor_columns].isna().any().any():
        raise ValueError("第四轮登记因子与标签合并后出现空值")
    dataset.sort_values("date", inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS_APPROXIMATION_NOT_OFFICIAL_WEIGHT",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "registration_timing": registry["registration_timing"],
        "row_count": int(len(dataset)),
        "first_date": str(dataset["date"].min().date()),
        "last_date": str(dataset["date"].max().date()),
        "factor_count": len(factor_columns),
        "hypothesis_count": len(registry["hypotheses"]),
        "factor_columns": factor_columns,
        "target_definition": "t日收盘后形成因子，t+1开盘买入，t+h收盘卖出；净收益含分红、滑点、佣金、最低佣金和整手约束",
        "weighting_semantics": "DAILY_CIRCULATING_MARKET_CAP_APPROXIMATION_NOT_OFFICIAL_CSI_WEIGHT",
        "breadth_sha256": _sha256(BREADTH_FILE),
        "base_dataset_sha256": _sha256(BASE_DATASET_FILE),
        "registry_sha256": _sha256(REGISTRY_FILE),
        "output_sha256": _sha256(OUTPUT_FILE),
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
