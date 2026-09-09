"""在三个已解盲代码桶上交叉检查分层低波动第一名。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import _benchmark_series, summarize  # noqa: E402
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.develop_a_share_bucket12_stratified_lowvol_v1 import select_one_per_bucket  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


OUTPUT_JSON = ROOT / "reports/discovery/a_share_bucket012_stratified_lowvol_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_BUCKET012_STRATIFIED_LOWVOL_DEVELOPMENT_V1.md"


def main() -> int:
    contract0 = yaml.safe_load((ROOT / "config/a_share_hash_holdout_alpha_v2.yaml").read_text(encoding="utf-8"))
    contract1 = yaml.safe_load((ROOT / "config/a_share_bucket1_time_holdout_formula_v1.yaml").read_text(encoding="utf-8"))
    contract2 = yaml.safe_load((ROOT / "config/a_share_bucket2_time_holdout_lowvol_v1.yaml").read_text(encoding="utf-8"))
    f1 = pd.read_parquet(ROOT / contract1["paths"]["features"])
    common_dates = set(pd.to_datetime(f1["date"].unique()))
    feature_specs = (
        (0, ROOT / contract0["paths"]["features"]),
        (1, ROOT / contract1["paths"]["features"]),
        (2, ROOT / contract2["paths"]["features"]),
    )
    features = []
    for bucket, path in feature_specs:
        frame = pd.read_parquet(path)
        frame = frame.loc[pd.to_datetime(frame["date"]).isin(common_dates)].copy()
        frame["split_bucket"] = bucket
        features.append(frame)
    all_features = pd.concat(features, ignore_index=True)
    panel_paths = (
        ROOT / contract0["paths"]["holdout_panel"],
        ROOT / contract1["paths"]["holdout_panel"],
        ROOT / contract2["paths"]["holdout_panel"],
    )
    panel = pd.concat([pd.read_parquet(path) for path in panel_paths], ignore_index=True)
    if panel[["date", "con_code"]].duplicated().any():
        raise RuntimeError("三个代码桶的执行行情出现重复证券日期")
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    reference = pd.read_parquet(ROOT / contract2["paths"]["base_ledger"])
    calendar = pd.DatetimeIndex(pd.to_datetime(reference["date"]))
    benchmark_raw = pd.read_parquet(ROOT / contract2["paths"]["benchmark"])
    benchmark = _benchmark_series(benchmark_raw, calendar, 20000.0)
    master = pd.read_parquet(ROOT / contract2["inputs"]["master"])
    master = master.loc[master["split_bucket"].isin([0, 1, 2])].copy()
    gap = float(contract2["universe"]["maximum_open_total_return_gap_for_trade"])
    results = []
    for retention_rank in (1, 2, 3):
        targets = select_one_per_bucket(all_features, retention_rank)
        target_counts = targets.groupby("signal_date").size()
        if target_counts.min() != 3 or target_counts.max() != 3:
            raise RuntimeError("每个信号日必须恰好从三个代码桶各选一只")
        base_ledger, base_trades = run_small_account_open_backtest(
            execution, targets, calendar, 20000.0, _costs(contract2, "base_slippage_bps_per_leg"), gap
        )
        stress_ledger, stress_trades = run_small_account_open_backtest(
            execution, targets, calendar, 20000.0, _costs(contract2, "stress_slippage_bps_per_leg"), gap
        )
        base = summarize(base_ledger, base_trades, benchmark, 20000.0)
        stress = summarize(stress_ledger, stress_trades, benchmark, 20000.0)
        bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract2)
        periods = _periods(base_ledger, benchmark, contract2)
        selected = set(targets["con_code"].astype(str))
        delisted = master.loc[
            master["ts_code"].astype(str).isin(selected)
            & master["delist_date"].notna()
            & master["delist_date"].le(pd.Timestamp(contract2["periods"]["evaluation_end"])),
            "ts_code",
        ].astype(str).tolist()
        period_excess = {key: value["annualized_excess"] for key, value in periods.items()}
        gates = {
            "base_annualized_excess_20pct": base["annualized_excess"] >= 0.20,
            "stress_annualized_excess_20pct": stress["annualized_excess"] >= 0.20,
            "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > 0.0,
            "positive_predefined_periods": sum(value > 0 for value in period_excess.values()) >= 2,
            "all_trades_at_least_5000": base_trades.empty or bool(base_trades["notional"].ge(5000.0 - 1e-8).all()),
            "no_selected_stock_delisted": not delisted,
        }
        row = _json_safe(
            {
                "retention_rank_within_bucket": retention_rank,
                "base": base,
                "stress": stress,
                "bootstrap_annualized_excess": bootstrap,
                "period_annualized_excess": period_excess,
                "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else np.nan,
                "maximum_position_count": int(base_ledger["position_count"].max()),
                "selected_delisted_codes": delisted,
                "gates": gates,
                "all_gates_pass": all(gates.values()),
            }
        )
        results.append(row)
        print(
            f"完成三桶保留{retention_rank}：超额{base['annualized_excess']:.2%}，"
            f"Bootstrap下界{bootstrap['interval_95pct'][0]:.2%}",
            flush=True,
        )
    report = _json_safe(
        {
            "project_id": "A_SHARE_BUCKET012_STRATIFIED_LOWVOL_DEVELOPMENT_V1",
            "status": "DEVELOPMENT_CROSS_BUCKET_CHECK_COMPLETE",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "factor_count": 1,
            "holdings": 3,
            "future_bucket3_returns_read": False,
            "future_bucket4_returns_read": False,
            "results": results,
            "governance": {"development_only": True, "order_generation": "DISABLED", "broker_connection": "DISABLED"},
        }
    )
    lines = [
        "# 桶0+1+2分层低波动交叉检查 V1",
        "",
        "|桶内保留名次|基础超额|压力超额|波动|回撤|Bootstrap下界|最小成交|全部门槛|",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"|{item['retention_rank_within_bucket']}|{item['base']['annualized_excess']:.2%}|"
            f"{item['stress']['annualized_excess']:.2%}|{item['base']['strategy']['annualized_volatility']:.2%}|"
            f"{item['base']['strategy']['maximum_drawdown']:.2%}|"
            f"{item['bootstrap_annualized_excess']['interval_95pct'][0]:.2%}|"
            f"{item['minimum_trade_notional_cny']:.0f}|{'PASS' if item['all_gates_pass'] else 'FAIL'}|"
        )
    lines += ["", "仅为已解盲开发证据；桶3、桶4未来收益仍未读取；不生成订单。", ""]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text("\n".join(lines), OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
