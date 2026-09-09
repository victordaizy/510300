"""按学术原始定义检验A股60日形成、跳过5日的月度反转。"""

from __future__ import annotations

import hashlib
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
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


CONFIG = ROOT / "config/a_share_bucket34_time_holdout_stratified_lowvol_v1.yaml"
OUTPUT_JSON = ROOT / "reports/discovery/a_share_academic_reversal_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_ACADEMIC_REVERSAL_DEVELOPMENT_V1.md"
SOURCE_REVERSAL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6872158"
SOURCE_FACTOR_TIMING = "https://doi.org/10.1002/ise3.86"


def code_stratum(code: str) -> int:
    return hashlib.sha256(str(code).encode("utf-8")).digest()[0] % 3


def build_targets(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    evaluation_start: str,
    evaluation_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """冻结为60日形成期、跳过5日、20日调仓，每个哈希层选一个最弱者。"""

    evaluation = calendar[
        (calendar >= pd.Timestamp(evaluation_start)) & (calendar <= pd.Timestamp(evaluation_end))
    ]
    signal_dates = set(evaluation[::20])
    source = panel[["date", "con_code", "raw_close", "total_return_close", "volume", "amount"]].copy()
    source["date"] = pd.to_datetime(source["date"])
    source.sort_values(["con_code", "date"], inplace=True)
    pieces = []
    for _, group in source.groupby("con_code", sort=False):
        group = group.reset_index(drop=True)
        total_close = group["total_return_close"]
        calculated = pd.DataFrame(
            {
                "reversal_60_skip_5": -np.log(total_close.shift(5) / total_close.shift(60)),
                "average_amount20": group["amount"].rolling(20, min_periods=20).mean(),
            }
        )
        selected = group["date"].isin(signal_dates)
        if selected.any():
            pieces.append(
                pd.concat(
                    [group.loc[selected].reset_index(drop=True), calculated.loc[selected].reset_index(drop=True)],
                    axis=1,
                )
            )
    features = pd.concat(pieces, ignore_index=True)
    meta = master[["ts_code", "list_date", "delist_date"]].copy()
    meta["list_date"] = pd.to_datetime(meta["list_date"], errors="coerce")
    meta["delist_date"] = pd.to_datetime(meta["delist_date"], errors="coerce")
    features = features.merge(meta, left_on="con_code", right_on="ts_code", how="left", validate="many_to_one")
    features["stratum"] = features["con_code"].astype(str).map(code_stratum)
    features["eligible"] = (
        features["reversal_60_skip_5"].replace([np.inf, -np.inf], np.nan).notna()
        & features["average_amount20"].ge(50_000_000.0)
        & features["raw_close"].ge(10.0)
        & features["raw_close"].mul(100.0).le(6500.0)
        & features["volume"].gt(0.0)
        & features["date"].ge(features["list_date"])
        & (features["delist_date"].isna() | features["date"].lt(features["delist_date"]))
    )
    ready = features.loc[features["eligible"]].copy()
    ready.sort_values(
        ["date", "stratum", "reversal_60_skip_5", "con_code"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    targets = ready.groupby(["date", "stratum"], sort=True).head(1).copy()
    targets.rename(columns={"date": "signal_date", "reversal_60_skip_5": "score"}, inplace=True)
    targets.sort_values(["signal_date", "stratum"], inplace=True)
    targets["selection_rank"] = targets.groupby("signal_date", sort=True).cumcount() + 1
    targets["regime"] = "ACADEMIC_REVERSAL_60_SKIP5_MONTHLY"
    counts = targets.groupby("signal_date").size()
    if counts.min() != 3 or counts.max() != 3:
        raise RuntimeError("每个信号日必须从三个哈希层各选一只")
    return features.reset_index(drop=True), targets[
        ["signal_date", "con_code", "score", "stratum", "selection_rank", "regime"]
    ].reset_index(drop=True)


def evaluate(
    label: str,
    panel: pd.DataFrame,
    master: pd.DataFrame,
    benchmark_raw: pd.DataFrame,
    evaluation_start: str,
    evaluation_end: str,
    period_specs: list[dict],
    base_contract: dict,
) -> dict:
    benchmark_raw = benchmark_raw.copy()
    benchmark_raw["date"] = pd.to_datetime(benchmark_raw["date"])
    calendar_all = pd.DatetimeIndex(sorted(benchmark_raw["date"].unique()))
    features, targets = build_targets(
        panel, master, calendar_all, evaluation_start, evaluation_end
    )
    calendar = calendar_all[
        (calendar_all >= pd.Timestamp(evaluation_start)) & (calendar_all <= pd.Timestamp(evaluation_end))
    ]
    selected_codes = set(targets["con_code"].astype(str))
    execution = panel.loc[
        panel["con_code"].astype(str).isin(selected_codes),
        ["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"],
    ].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    contract = json.loads(json.dumps(base_contract))
    contract["evaluation"]["predefined_periods"] = period_specs
    contract["evaluation"]["positive_predefined_periods_minimum"] = 2
    contract["evaluation"]["random_seed"] = 20260819
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution, targets, calendar, 20000.0, _costs(contract, "base_slippage_bps_per_leg"), gap
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution, targets, calendar, 20000.0, _costs(contract, "stress_slippage_bps_per_leg"), gap
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], 20000.0)
    base = summarize(base_ledger, base_trades, benchmark, 20000.0)
    stress = summarize(stress_ledger, stress_trades, benchmark, 20000.0)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = _periods(base_ledger, benchmark, contract)
    selected_delisted = master.loc[
        master["ts_code"].astype(str).isin(selected_codes)
        & master["delist_date"].notna()
        & master["delist_date"].le(pd.Timestamp(evaluation_end)),
        "ts_code",
    ].astype(str).tolist()
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= 0.20,
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= 0.20,
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > 0.0,
        "positive_predefined_periods": sum(value["annualized_excess"] > 0 for value in periods.values()) >= 2,
        "all_trades_at_least_5000": base_trades.empty or bool(base_trades["notional"].ge(5000.0 - 1e-8).all()),
        "no_selected_stock_delisted": not selected_delisted,
    }
    return _json_safe(
        {
            "label": label,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "feature_rows": len(features),
            "signal_dates": int(targets["signal_date"].nunique()),
            "selected_codes": len(selected_codes),
            "base": base,
            "stress": stress,
            "bootstrap_annualized_excess": bootstrap,
            "predefined_periods": periods,
            "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else np.nan,
            "selected_delisted_codes": selected_delisted,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        }
    )


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    old_panel = pd.read_parquet(ROOT / "data/raw/a_share_hash_holdout_v2/training_panel.parquet")
    old_benchmark = pd.read_parquet(ROOT / "data/raw/a_share_hash_holdout_v2/training_H00300.parquet")
    old_master = master.loc[master["split_bucket"].isin([1, 2, 3, 4])].copy()
    recent_paths = (
        "data/raw/a_share_hash_holdout_v2/holdout_panel.parquet",
        "data/raw/a_share_bucket1_time_holdout_formula_v1/holdout_panel.parquet",
        "data/raw/a_share_bucket2_time_holdout_lowvol_v1/holdout_panel.parquet",
        "data/raw/a_share_bucket34_time_holdout_stratified_lowvol_v1/holdout_panel.parquet",
    )
    recent_panel = pd.concat([pd.read_parquet(ROOT / path) for path in recent_paths], ignore_index=True)
    if recent_panel[["date", "con_code"]].duplicated().any():
        raise RuntimeError("2024—2026开发面板出现重复证券日期")
    recent_benchmark = pd.read_parquet(ROOT / contract["paths"]["benchmark"])
    results = [
        evaluate(
            "DEVELOPMENT_2015_2023",
            old_panel,
            old_master,
            old_benchmark,
            "2015-01-05",
            "2023-11-30",
            [
                {"id": "P1_2015_2017", "start": "2015-01-05", "end": "2017-12-29"},
                {"id": "P2_2018_2020", "start": "2018-01-02", "end": "2020-12-31"},
                {"id": "P3_2021_2023", "start": "2021-01-04", "end": "2023-11-30"},
            ],
            contract,
        ),
        evaluate(
            "DEVELOPMENT_2024_2026",
            recent_panel,
            master,
            recent_benchmark,
            "2024-01-02",
            "2026-08-14",
            contract["evaluation"]["predefined_periods"],
            contract,
        ),
    ]
    report = _json_safe(
        {
            "project_id": "A_SHARE_ACADEMIC_REVERSAL_DEVELOPMENT_V1",
            "status": "DEVELOPMENT_COMPLETE",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source_rules": {
                "formation_trading_days": 60,
                "skip_recent_trading_days": 5,
                "rebalance_trading_days": 20,
                "minimum_20d_average_amount_cny": 50_000_000.0,
                "sources": [SOURCE_REVERSAL, SOURCE_FACTOR_TIMING],
            },
            "factor_count": 1,
            "results": results,
            "qualifies_for_untouched_2005_2013_backward_holdout": all(
                item["all_gates_pass"] for item in results
            ),
            "governance": {
                "development_only": True,
                "untouched_2005_2013_returns_read": False,
                "paper_signal": "DISABLED",
                "order_generation": "DISABLED",
            },
        }
    )
    lines = [
        "# A股学术短期反转开发 V1",
        "",
        "- 固定规则：过去60日形成、跳过最近5日、每20日调仓。",
        "- 三个代码哈希层各选最弱一只，共3只；20日平均成交额至少5000万元；总资金20000元。",
        "",
        "|开发区间|策略年化|H00300年化|基础超额|压力超额|波动|回撤|Bootstrap下界|门槛|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"|{item['label']}|{item['base']['strategy']['cagr']:.2%}|{item['base']['benchmark']['cagr']:.2%}|"
            f"{item['base']['annualized_excess']:.2%}|{item['stress']['annualized_excess']:.2%}|"
            f"{item['base']['strategy']['annualized_volatility']:.2%}|{item['base']['strategy']['maximum_drawdown']:.2%}|"
            f"{item['bootstrap_annualized_excess']['interval_95pct'][0]:.2%}|"
            f"{'PASS' if item['all_gates_pass'] else 'FAIL'}|"
        )
    lines += [
        "",
        f"- 是否有资格冻结到2005—2013向后时间盲测：`{report['qualifies_for_untouched_2005_2013_backward_holdout']}`",
        "- 开发结果不生成仓位或订单。",
        "",
    ]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text("\n".join(lines), OUTPUT_MD)
    print(
        json.dumps(
            {
                item["label"]: {
                    "基础超额": item["base"]["annualized_excess"],
                    "压力超额": item["stress"]["annualized_excess"],
                    "Bootstrap下界": item["bootstrap_annualized_excess"]["interval_95pct"][0],
                    "全部门槛": item["all_gates_pass"],
                }
                for item in results
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
