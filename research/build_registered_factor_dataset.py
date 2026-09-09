"""生成正式注册的日线因子与可执行Target数据集。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.registered_factors import (
    ExecutableTargetCosts,
    add_executable_targets,
    build_registered_daily_factors,
)


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry.yaml"
OUTPUT_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
REPORT_FILE = ROOT / "reports" / "research" / "registered_factor_dataset.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    paths = {
        "etf": ROOT / "data/raw/market/510300_daily_raw.parquet",
        "index": ROOT / "data/raw/market/000300_daily_feature_warmup.parquet",
        "official_pe": ROOT / "data/raw/valuation/000300_pe_official_raw.parquet",
        "valuation": ROOT / "data/raw/valuation/000300_valuation_daily_raw.parquet",
        "dividends": ROOT / "data/reference/510300_dividends.csv",
        "if": ROOT / "data/raw/futures/IF0_daily_raw.parquet",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"注册因子数据缺失：{missing}")
    etf = pd.read_parquet(paths["etf"])
    index = pd.read_parquet(paths["index"])
    pe = pd.read_parquet(paths["official_pe"])
    valuation = pd.read_parquet(paths["valuation"])
    dividends = pd.read_csv(paths["dividends"], parse_dates=["record_date", "ex_date", "payment_date"])
    futures = pd.read_parquet(paths["if"])
    factors = build_registered_daily_factors(etf, index, pe, valuation, dividends, futures)
    costs_config = settings["backtest"]
    target_costs = ExecutableTargetCosts(
        initial_cash=float(costs_config["initial_cash"]),
        commission_rate=float(costs_config["commission_rate"]),
        minimum_commission_cny=float(costs_config["minimum_commission_cny"]),
        stamp_duty_rate=float(costs_config["stamp_duty_rate"]),
        slippage_bps=float(costs_config["slippage_bps_base"]),
        lot_size=int(costs_config["lot_size"]),
    )
    target_prices = add_executable_targets(etf, dividends, horizons=(5, 20), costs=target_costs)
    target_columns = [
        "date",
        "exec_total_return_5d_net", "exec_total_return_5d_gross", "exec_mae_5d", "exec_mfe_5d", "label_end_date_5d",
        "exec_total_return_20d_net", "exec_total_return_20d_gross", "exec_mae_20d", "exec_mfe_20d", "label_end_date_20d",
    ]
    dataset = factors.merge(target_prices[target_columns], on="date", how="inner", validate="one_to_one")
    start = pd.Timestamp(settings["project"]["start_date"])
    end = pd.Timestamp(settings["project"]["end_date"])
    dataset = dataset.loc[dataset["date"].between(start, end)].copy().reset_index(drop=True)
    registered_columns = [item["column"] for item in registry["factor_definitions"]]
    absent = [column for column in registered_columns if column not in dataset]
    if absent:
        raise ValueError(f"注册因子没有生成：{absent}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "row_count": int(len(dataset)),
        "first_date": str(dataset["date"].min().date()),
        "last_date": str(dataset["date"].max().date()),
        "registered_factor_count": len(registered_columns),
        "registered_hypothesis_count": len(registry["hypotheses"]),
        "target_definition": "t日收盘后信号，t+1开盘买入，t+h收盘卖出；含分红、滑点、佣金、最低佣金和整手约束",
        "source_hashes": {name: _sha256(path) for name, path in paths.items()},
        "registry_sha256": _sha256(REGISTRY_FILE),
        "output_sha256": _sha256(OUTPUT_FILE),
        "blocked_tracks": registry["blocked_tracks"],
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
