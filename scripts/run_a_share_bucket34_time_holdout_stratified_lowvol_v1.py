"""一次性打开桶3+4未见未来收益上的分层低波动严格结果。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import _benchmark_series, summarize  # noqa: E402
from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, ModelRules, build_features  # noqa: E402
from research.small_account_cross_sectional import run_small_account_open_backtest  # noqa: E402
from scripts.freeze_a_share_bucket34_time_holdout_stratified_lowvol_v1 import (  # noqa: E402
    sha256,
    tree_sha256,
    verify_protocol,
)
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import (  # noqa: E402
    _costs,
    _json_safe,
    _paired_bootstrap,
    _periods,
)


def rules(contract: dict) -> ModelRules:
    periods, universe = contract["periods"], contract["universe"]
    return ModelRules(
        signal_start=pd.Timestamp(periods["evaluation_start"]),
        signal_end=pd.Timestamp(periods["evaluation_end"]),
        rebalance_step=int(periods["rebalance_every_trading_days"]),
        horizon=int(periods["target_horizon_trading_days"]),
        minimum_amount=float(universe["minimum_20d_average_amount_cny"]),
        maximum_one_lot=float(universe["maximum_signal_price_for_one_lot_cny"]),
        learning_rate=0.01,
        max_iter=1,
        max_leaf_nodes=2,
        min_samples_leaf=2,
        l2_regularization=0.0,
        max_bins=2,
        random_state=20260819,
        target_clip=(-0.5, 0.5),
        half_life_days=1095.75,
    )


def select_targets(features: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """每个未来哈希桶独立选低波动20日排名第一的证券。"""

    rows: list[pd.DataFrame] = []
    for bucket, frame in features.groupby("split_bucket", sort=True):
        ready = frame.loc[
            frame["signal_output"].eq("SIGNAL_READY")
            & frame["raw_close"].ge(float(contract["universe"]["minimum_signal_price_cny"]))
        ].copy()
        ready["score"] = ready[FEATURE_COLUMNS[4]]
        ready.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)
        selected = ready.groupby("date", sort=True).head(1).copy()
        selected["split_bucket"] = int(bucket)
        rows.append(selected[["date", "con_code", "score", "split_bucket"]])
    targets = pd.concat(rows, ignore_index=True).rename(columns={"date": "signal_date"})
    targets.sort_values(["signal_date", "split_bucket", "con_code"], inplace=True)
    targets["selection_rank"] = targets.groupby("signal_date", sort=True).cumcount() + 1
    targets["regime"] = "BUCKET34_STRATIFIED_LOWVOL20"
    return targets.reset_index(drop=True)


def render(report: dict) -> str:
    base, stress = report["base_cost"], report["stress_cost"]
    lines = [
        "# 全A桶3+4代码与时间盲测分层低波动公式 V1",
        "",
        f"- 状态：`{report['status']}`",
        "- 因子：低波动20日排名（1个）",
        "- 持仓：桶3与桶4各第一名，共2只",
        "- 桶3与桶4未来收益均在公式和评估器冻结后下载",
        "",
        "|成本|策略年化|H00300年化|年化超额|波动|最大回撤|成交腿|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|"
        f"{base['annualized_excess']:.2%}|{base['strategy']['annualized_volatility']:.2%}|"
        f"{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|"
        f"{stress['annualized_excess']:.2%}|{stress['strategy']['annualized_volatility']:.2%}|"
        f"{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|",
        "",
        f"- Bootstrap年化超额95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交额：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in report["gates"].items())
    lines += ["", "结果后禁止在桶3或桶4上修改公式补救；本结果不生成仓位、订单或券商连接。", ""]
    return "\n".join(lines)


def main() -> int:
    contract, protocol = verify_protocol()
    paths, periods = contract["paths"], contract["periods"]
    status = json.loads((ROOT / paths["data_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or not status.get("protocol_frozen_before_download"):
        raise RuntimeError("桶3+4未来数据未通过冻结后审计")
    if status.get("holdout_buckets") != [3, 4]:
        raise RuntimeError("数据状态的未来留出桶不是3和4")
    if status["protocol_manifest_sha256"] != sha256(ROOT / paths["protocol_manifest"]):
        raise RuntimeError("桶3+4数据下载后协议清单变化")
    checks = {
        contract["inputs"]["master"]: status["hashes"]["master"],
        paths["holdout_panel"]: status["hashes"]["panel"],
        paths["benchmark"]: status["hashes"]["benchmark"],
    }
    changed = [name for name, expected in checks.items() if sha256(ROOT / name) != expected]
    if tree_sha256(ROOT / paths["checkpoint_directory"]) != status["hashes"]["checkpoint_tree"]:
        changed.append(paths["checkpoint_directory"])
    if changed:
        raise RuntimeError(f"桶3+4未来输入变化：{changed}")
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    master = master.loc[master["split_bucket"].isin([3, 4])].copy()
    panel = pd.read_parquet(ROOT / paths["holdout_panel"])
    benchmark_raw = pd.read_parquet(ROOT / paths["benchmark"])
    panel_codes = set(panel["con_code"].astype(str))
    master_buckets = master.set_index("ts_code")["split_bucket"].astype(int)
    actual_buckets = set(pd.Series(list(panel_codes)).map(master_buckets).dropna().astype(int))
    if actual_buckets != {3, 4}:
        raise RuntimeError(f"执行面板实际代码桶错误：{sorted(actual_buckets)}")
    feature_frames: list[pd.DataFrame] = []
    calendars: list[pd.DatetimeIndex] = []
    for bucket in (3, 4):
        bucket_master = master.loc[master["split_bucket"].eq(bucket)].copy()
        bucket_codes = set(bucket_master["ts_code"].astype(str))
        bucket_panel = panel.loc[panel["con_code"].astype(str).isin(bucket_codes)].copy()
        bucket_features, bucket_calendar = build_features(bucket_panel, bucket_master, benchmark_raw, rules(contract))
        bucket_features["split_bucket"] = bucket
        feature_frames.append(bucket_features)
        calendars.append(bucket_calendar)
    if not calendars[0].equals(calendars[1]):
        raise RuntimeError("桶3与桶4交易日历不一致")
    features = pd.concat(feature_frames, ignore_index=True)
    targets = select_targets(features, contract)
    target_counts = targets.groupby("signal_date").size()
    if target_counts.min() != 2 or target_counts.max() != 2:
        raise RuntimeError("每个信号日必须从桶3和桶4各选一只")
    if targets.groupby(["signal_date", "split_bucket"]).size().ne(1).any():
        raise RuntimeError("同一信号日的桶内选择数量不是1")
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    start, end = pd.Timestamp(periods["evaluation_start"]), pd.Timestamp(periods["evaluation_end"])
    calendar = calendars[0][(calendars[0] >= start) & (calendars[0] <= end)]
    initial = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution, targets, calendar, initial, _costs(contract, "base_slippage_bps_per_leg"), gap
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution, targets, calendar, initial, _costs(contract, "stress_slippage_bps_per_leg"), gap
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial)
    base = summarize(base_ledger, base_trades, benchmark, initial)
    stress = summarize(stress_ledger, stress_trades, benchmark, initial)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    period_results = _periods(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in period_results.values())
    selected_codes = set(targets["con_code"].astype(str))
    selected_delisted = master.loc[
        master["ts_code"].astype(str).isin(selected_codes)
        & master["delist_date"].notna()
        & master["delist_date"].le(end),
        "ts_code",
    ].astype(str).tolist()
    minimum_trade = float(base_trades["notional"].min()) if not base_trades.empty else np.nan
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= float(contract["evaluation"]["annualized_excess_minimum"]),
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
        "no_selected_stock_delisted_during_holdout": len(selected_delisted) == 0,
    }
    result_status = (
        "STRICT_BUCKET34_CODE_AND_TIME_HOLDOUT_PASS_AWAITING_FORWARD"
        if all(gates.values())
        else "STRICT_BUCKET34_CODE_AND_TIME_HOLDOUT_REJECTED_FROZEN"
    )
    report = _json_safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "status": result_status,
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "formula_frozen_at": protocol["frozen_at"],
            "future_returns_downloaded_before_freeze": False,
            "future_holdout_buckets": [3, 4],
            "factor_count": 1,
            "row_counts": {
                "features": len(features),
                "ready": int(features["signal_output"].eq("SIGNAL_READY").sum()),
                "signal_dates": int(targets["signal_date"].nunique()),
            },
            "base_cost": base,
            "stress_cost": stress,
            "bootstrap_annualized_excess": bootstrap,
            "predefined_periods": period_results,
            "positive_predefined_periods": positive_periods,
            "gates": gates,
            "execution_audit": {
                "minimum_trade_notional_cny": minimum_trade,
                "maximum_position_count": int(base_ledger["position_count"].max()),
                "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()),
                "selected_delisted_codes": selected_delisted,
                "t_plus_one_enforced": True,
            },
            "data_audit": status,
            "safety": contract["governance"],
        }
    )
    for frame, key in (
        (features, "features"),
        (targets, "targets"),
        (base_ledger, "base_ledger"),
        (base_trades, "base_trades"),
        (stress_ledger, "stress_ledger"),
        (stress_trades, "stress_trades"),
    ):
        atomic_parquet(frame, ROOT / paths[key])
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / paths["result_json"])
    atomic_text(render(report), ROOT / paths["result_markdown"])
    print(
        json.dumps(
            {
                "状态": result_status,
                "策略年化": base["strategy"]["cagr"],
                "H00300年化": base["benchmark"]["cagr"],
                "基础年化超额": base["annualized_excess"],
                "压力年化超额": stress["annualized_excess"],
                "Bootstrap下界": bootstrap["interval_95pct"][0],
                "正超额子期": positive_periods,
                "选中退市股": selected_delisted,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
