"""用已解盲桶1与桶2检验分层低波动第一名的可交易组合。"""

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
from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS  # noqa: E402
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


CONFIG1 = ROOT / "config/a_share_bucket1_time_holdout_formula_v1.yaml"
CONFIG2 = ROOT / "config/a_share_bucket2_time_holdout_lowvol_v1.yaml"
OUTPUT_JSON = ROOT / "reports/discovery/a_share_bucket12_stratified_lowvol_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_BUCKET12_STRATIFIED_LOWVOL_DEVELOPMENT_V1.md"


def select_one_per_bucket(features: pd.DataFrame, retention_rank: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for bucket, bucket_features in features.groupby("split_bucket", sort=True):
        ready = bucket_features.loc[
            bucket_features["signal_output"].eq("SIGNAL_READY")
            & bucket_features["average_amount20"].ge(1_000_000_000.0)
            & bucket_features["raw_close"].ge(10.0)
        ].copy()
        ready["score"] = ready[FEATURE_COLUMNS[4]]
        ready.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)
        ready["score_rank"] = ready.groupby("date", sort=True).cumcount() + 1
        previous: str | None = None
        for _, frame in ready.groupby("date", sort=True):
            ranks = dict(zip(frame["con_code"].astype(str), frame["score_rank"], strict=True))
            selected = (
                previous
                if previous is not None and ranks.get(previous, retention_rank + 1) <= retention_rank
                else str(frame.iloc[0]["con_code"])
            )
            day = frame.loc[frame["con_code"].astype(str).eq(selected), ["date", "con_code", "score"]].copy()
            day["bucket"] = int(bucket)
            rows.append(day)
            previous = selected
    targets = pd.concat(rows, ignore_index=True).rename(columns={"date": "signal_date"})
    targets.sort_values(["signal_date", "bucket"], inplace=True)
    targets["selection_rank"] = targets.groupby("signal_date", sort=True).cumcount() + 1
    targets["regime"] = "BUCKET12_STRATIFIED_LOWVOL20"
    return targets.reset_index(drop=True)


def result_row(
    retention_rank: int,
    base: dict,
    stress: dict,
    bootstrap: dict,
    periods: dict,
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    selected_delisted: list[str],
) -> dict:
    period_excess = {key: value["annualized_excess"] for key, value in periods.items()}
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= 0.20,
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= 0.20,
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > 0.0,
        "positive_predefined_periods": sum(value > 0 for value in period_excess.values()) >= 2,
        "all_trades_at_least_5000": trades.empty or bool(trades["notional"].ge(5000.0 - 1e-8).all()),
        "no_selected_stock_delisted": not selected_delisted,
    }
    return _json_safe(
        {
            "retention_rank_within_bucket": retention_rank,
            "factor_count": 1,
            "holdings": 2,
            "base": base,
            "stress": stress,
            "bootstrap_annualized_excess": bootstrap,
            "period_annualized_excess": period_excess,
            "minimum_trade_notional_cny": float(trades["notional"].min()) if not trades.empty else np.nan,
            "maximum_position_count": int(ledger["position_count"].max()),
            "selected_delisted_codes": selected_delisted,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
    )


def render(report: dict) -> str:
    lines = [
        "# 桶1+2分层低波动开发 V1",
        "",
        "- 两个已解盲代码桶各选低波动20日排名第一的股票。",
        "- 两只持仓共同使用20000元，执行T+1、100股、最低5000元成交及真实最低佣金。",
        "- 桶3与桶4的2024—2026未来收益均未读取。",
        "",
        "|桶内保留名次|基础超额|压力超额|波动|回撤|Bootstrap下界|成交腿|最小成交|门槛|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        lines.append(
            f"|{item['retention_rank_within_bucket']}|{item['base']['annualized_excess']:.2%}|"
            f"{item['stress']['annualized_excess']:.2%}|{item['base']['strategy']['annualized_volatility']:.2%}|"
            f"{item['base']['strategy']['maximum_drawdown']:.2%}|"
            f"{item['bootstrap_annualized_excess']['interval_95pct'][0]:.2%}|"
            f"{item['base']['trade_rows']}|{item['minimum_trade_notional_cny']:.0f}|"
            f"{'PASS' if item['all_gates_pass'] else 'FAIL'}|"
        )
    lines += [
        "",
        f"- 建议冻结的桶内保留名次：`{report['recommended_retention_rank']}`",
        "- 这是开发证据，不是桶3+4独立盲测结果；不生成仓位或订单。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    contract1 = yaml.safe_load(CONFIG1.read_text(encoding="utf-8"))
    contract2 = yaml.safe_load(CONFIG2.read_text(encoding="utf-8"))
    frames = []
    panels = []
    for bucket, contract in ((1, contract1), (2, contract2)):
        features = pd.read_parquet(ROOT / contract["paths"]["features"])
        features["split_bucket"] = bucket
        frames.append(features)
        panels.append(pd.read_parquet(ROOT / contract["paths"]["holdout_panel"]))
    all_features = pd.concat(frames, ignore_index=True)
    panel = pd.concat(panels, ignore_index=True)
    if panel[["date", "con_code"]].duplicated().any():
        raise RuntimeError("桶1与桶2执行行情出现重复证券日期")
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    reference = pd.read_parquet(ROOT / contract2["paths"]["base_ledger"])
    calendar = pd.DatetimeIndex(pd.to_datetime(reference["date"]))
    benchmark_raw = pd.read_parquet(ROOT / contract2["paths"]["benchmark"])
    benchmark = _benchmark_series(benchmark_raw, calendar, 20000.0)
    master = pd.read_parquet(ROOT / contract2["inputs"]["master"])
    master = master.loc[master["split_bucket"].isin([1, 2])].copy()
    gap = float(contract2["universe"]["maximum_open_total_return_gap_for_trade"])
    results = []
    for retention_rank in (1, 2, 3, 5):
        targets = select_one_per_bucket(all_features, retention_rank)
        if targets.groupby("signal_date").size().min() != 2 or targets.groupby("signal_date").size().max() != 2:
            raise RuntimeError("每个信号日必须恰好从两个代码桶各选一只")
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
        selected_delisted = master.loc[
            master["ts_code"].astype(str).isin(selected)
            & master["delist_date"].notna()
            & master["delist_date"].le(pd.Timestamp(contract2["periods"]["evaluation_end"])),
            "ts_code",
        ].astype(str).tolist()
        row = result_row(
            retention_rank, base, stress, bootstrap, periods, base_ledger, base_trades, selected_delisted
        )
        results.append(row)
        print(
            f"完成桶内保留{retention_rank}：基础超额{base['annualized_excess']:.2%}，"
            f"Bootstrap下界{bootstrap['interval_95pct'][0]:.2%}",
            flush=True,
        )
    passing = [item for item in results if item["all_gates_pass"]]
    recommended = max(passing, key=lambda item: item["bootstrap_annualized_excess"]["interval_95pct"][0]) if passing else None
    report = _json_safe(
        {
            "project_id": "A_SHARE_BUCKET12_STRATIFIED_LOWVOL_DEVELOPMENT_V1",
            "status": "DEVELOPMENT_CANDIDATE_FOUND" if recommended else "NO_DEVELOPMENT_CANDIDATE",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "future_bucket3_returns_read": False,
            "future_bucket4_returns_read": False,
            "results": results,
            "recommended_retention_rank": recommended["retention_rank_within_bucket"] if recommended else None,
            "governance": {
                "development_only": True,
                "paper_signal": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
            },
        }
    )
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text(render(report), OUTPUT_MD)
    print(json.dumps({"状态": report["status"], "建议保留名次": report["recommended_retention_rank"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
