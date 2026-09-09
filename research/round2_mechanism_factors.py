"""构建第二轮价量机制拆解与IF环境因子。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round2.yaml"
INPUT_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_round2_mechanism_dataset.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "round2_mechanism_dataset.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_round2_factors(dataset: pd.DataFrame) -> pd.DataFrame:
    """只用当日及此前数据构建第二轮因子，不修改第一轮数据集。"""

    data = dataset.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    log_volume = np.log1p(pd.to_numeric(data["etf_volume"], errors="coerce"))
    prior_mean = log_volume.shift(1).rolling(20, min_periods=20).mean()
    prior_std = log_volume.shift(1).rolling(20, min_periods=20).std(ddof=1)
    volume_shock = ((log_volume - prior_mean) / prior_std.replace(0.0, np.nan)).clip(-4.0, 4.0)

    data["round2_etf_total_return_20d"] = data["factor_etf_total_return_20d"]
    data["round2_etf_log_volume_shock_20d"] = volume_shock
    data["round2_etf_high_volume_trend_20d"] = (
        data["round2_etf_total_return_20d"] * volume_shock.clip(lower=0.0)
    )
    data["round2_if_minus_index_return_5d"] = data["context_if_minus_index_return_5d"]
    data["round2_legacy_price_volume_product_20d"] = data["factor_etf_price_volume_confirmation_20d"]
    data["signal_asof_date"] = data["date"]
    if data["date"].duplicated().any() or (data["signal_asof_date"] != data["date"]).any():
        raise ValueError("第二轮因子日期重复或信息截止日异常")
    return data


def main() -> int:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"缺少第一轮注册数据集：{INPUT_FILE}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    dataset = build_round2_factors(pd.read_parquet(INPUT_FILE))
    registered_columns = [item["column"] for item in registry["factor_definitions"]]
    missing = [column for column in registered_columns if column not in dataset]
    if missing:
        raise ValueError(f"第二轮注册因子没有生成：{missing}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "row_count": int(len(dataset)),
        "first_date": str(dataset["date"].min().date()),
        "last_date": str(dataset["date"].max().date()),
        "input_sha256": _sha256(INPUT_FILE),
        "registry_sha256": _sha256(REGISTRY_FILE),
        "output_sha256": _sha256(OUTPUT_FILE),
        "registered_factor_count": len(registered_columns),
        "registered_hypothesis_count": len(registry["hypotheses"]),
        "derivation_note": registry["derivation_note"],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
