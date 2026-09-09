"""一次性打开桶1未见未来收益上的四因子公式结果。"""

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
from scripts.freeze_a_share_bucket1_time_holdout_formula_v1 import sha256, tree_sha256, verify_protocol  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402
from scripts.run_csi300_small_account_external_2015_2021 import _costs, _json_safe, _paired_bootstrap, _periods  # noqa: E402


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
    formula = contract["formula"]
    ready = features.loc[
        features["signal_output"].eq("SIGNAL_READY")
        & features["raw_close"].ge(float(contract["universe"]["minimum_signal_price_cny"]))
    ].copy()
    ready["score"] = (
        float(formula["drawdown_position_60_rank_weight"]) * ready[FEATURE_COLUMNS[6]]
        + float(formula["low_volatility_20_rank_weight"]) * ready[FEATURE_COLUMNS[4]]
        + float(formula["low_volatility_60_rank_weight"]) * ready[FEATURE_COLUMNS[5]]
        + float(formula["trend_efficiency_60_rank_weight"]) * ready[FEATURE_COLUMNS[9]]
    )
    ready.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)
    ready["score_rank"] = ready.groupby("date").cumcount() + 1
    retention = int(contract["selection"]["retention_score_rank"])
    previous = None
    rows = []
    for date, frame in ready.groupby("date", sort=True):
        ranks = dict(zip(frame["con_code"].astype(str), frame["score_rank"], strict=True))
        selected = previous if previous is not None and ranks.get(previous, retention + 1) <= retention else str(frame.iloc[0]["con_code"])
        row = frame.loc[frame["con_code"].astype(str).eq(selected), ["date", "con_code", "score", "score_rank"]].copy()
        row["selection_rank"] = 1
        rows.append(row)
        previous = selected
    targets = pd.concat(rows, ignore_index=True).rename(columns={"date": "signal_date"})
    targets["regime"] = "BUCKET1_TIME_HOLDOUT_FOUR_FACTOR"
    return targets


def render(report: dict) -> str:
    base, stress = report["base_cost"], report["stress_cost"]
    lines = [
        "# 全A桶1代码与时间盲测四因子公式 V1",
        "",
        f"- 状态：`{report['status']}`",
        "- 因子数：4",
        "- 桶1未来收益在公式冻结后下载",
        "",
        "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|",
        "|---|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|",
        "",
        f"- Bootstrap年化超额95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交额：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in report["gates"].items())
    lines += ["", "结果不生成仓位、订单或券商连接。", ""]
    return "\n".join(lines)


def main() -> int:
    contract, protocol = verify_protocol()
    paths, periods = contract["paths"], contract["periods"]
    status = json.loads((ROOT / paths["data_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or not status.get("protocol_frozen_before_download"):
        raise RuntimeError("桶1未来数据未通过冻结后审计")
    if status["protocol_manifest_sha256"] != sha256(ROOT / paths["protocol_manifest"]):
        raise RuntimeError("桶1数据下载后协议清单变化")
    checks = {
        contract["inputs"]["master"]: status["hashes"]["master"],
        paths["holdout_panel"]: status["hashes"]["panel"],
        paths["benchmark"]: status["hashes"]["benchmark"],
    }
    changed = [name for name, expected in checks.items() if sha256(ROOT / name) != expected]
    if tree_sha256(ROOT / paths["checkpoint_directory"]) != status["hashes"]["checkpoint_tree"]:
        changed.append(paths["checkpoint_directory"])
    if changed:
        raise RuntimeError(f"桶1未来输入变化：{changed}")
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    master = master.loc[master["split_bucket"].eq(int(contract["split"]["future_holdout_bucket"]))].copy()
    panel = pd.read_parquet(ROOT / paths["holdout_panel"])
    benchmark_raw = pd.read_parquet(ROOT / paths["benchmark"])
    features, calendar_all = build_features(panel, master, benchmark_raw, rules(contract))
    targets = select_targets(features, contract)
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    start, end = pd.Timestamp(periods["evaluation_start"]), pd.Timestamp(periods["evaluation_end"])
    calendar = calendar_all[(calendar_all >= start) & (calendar_all <= end)]
    initial = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(execution, targets, calendar, initial, _costs(contract, "base_slippage_bps_per_leg"), gap)
    stress_ledger, stress_trades = run_small_account_open_backtest(execution, targets, calendar, initial, _costs(contract, "stress_slippage_bps_per_leg"), gap)
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial)
    base, stress = summarize(base_ledger, base_trades, benchmark, initial), summarize(stress_ledger, stress_trades, benchmark, initial)
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
    result_status = "STRICT_BUCKET1_CODE_AND_TIME_HOLDOUT_PASS_AWAITING_FORWARD" if all(gates.values()) else "STRICT_BUCKET1_CODE_AND_TIME_HOLDOUT_REJECTED_FROZEN"
    report = _json_safe({
        "project_id": contract["protocol"]["project_id"],
        "status": result_status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "formula_frozen_at": protocol["frozen_at"],
        "future_returns_downloaded_before_freeze": False,
        "factor_count": int(contract["formula"]["factor_count"]),
        "row_counts": {"features": len(features), "ready": int(features["signal_output"].eq("SIGNAL_READY").sum()), "signal_dates": int(targets["signal_date"].nunique())},
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
    })
    for frame, key in (
        (features, "features"), (targets, "targets"), (base_ledger, "base_ledger"), (base_trades, "base_trades"),
        (stress_ledger, "stress_ledger"), (stress_trades, "stress_trades"),
    ):
        atomic_parquet(frame, ROOT / paths[key])
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / paths["result_json"])
    atomic_text(render(report), ROOT / paths["result_markdown"])
    print(json.dumps({
        "状态": result_status,
        "策略年化": base["strategy"]["cagr"],
        "H00300年化": base["benchmark"]["cagr"],
        "基础年化超额": base["annualized_excess"],
        "压力年化超额": stress["annualized_excess"],
        "Bootstrap下界": bootstrap["interval_95pct"][0],
        "正超额子期": positive_periods,
        "选中退市股": selected_delisted,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
