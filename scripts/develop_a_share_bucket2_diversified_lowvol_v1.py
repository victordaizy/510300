"""在已解盲的桶2上开发低波动分散化候选，不读取桶3未来收益。"""

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


CONFIG = ROOT / "config/a_share_bucket2_time_holdout_lowvol_v1.yaml"
OUTPUT_JSON = ROOT / "reports/discovery/a_share_bucket2_diversified_lowvol_development_v1.json"
OUTPUT_MD = ROOT / "reports/discovery/A_SHARE_BUCKET2_DIVERSIFIED_LOWVOL_DEVELOPMENT_V1.md"

SCORES = {
    "LV20": {FEATURE_COLUMNS[4]: 1.0},
    "LV60": {FEATURE_COLUMNS[5]: 1.0},
    "LV20_60_EQ": {FEATURE_COLUMNS[4]: 0.5, FEATURE_COLUMNS[5]: 0.5},
    "LV20_75_LV60_25": {FEATURE_COLUMNS[4]: 0.75, FEATURE_COLUMNS[5]: 0.25},
    "LV20_25_LV60_75": {FEATURE_COLUMNS[4]: 0.25, FEATURE_COLUMNS[5]: 0.75},
}


def select_targets(
    features: pd.DataFrame,
    weights: dict[str, float],
    holdings: int,
    retention_rank: int,
) -> pd.DataFrame:
    """按冻结的点时特征排名，并允许原持仓在保留名次内继续持有。"""

    ready = features.loc[
        features["signal_output"].eq("SIGNAL_READY") & features["raw_close"].ge(10.0)
    ].copy()
    ready["score"] = sum(weight * ready[column] for column, weight in weights.items())
    ready.sort_values(["date", "score", "con_code"], ascending=[True, False, True], inplace=True)
    ready["score_rank"] = ready.groupby("date", sort=True).cumcount() + 1
    previous: list[str] = []
    rows: list[pd.DataFrame] = []
    for _, frame in ready.groupby("date", sort=True):
        ranks = dict(zip(frame["con_code"].astype(str), frame["score_rank"], strict=True))
        retained = [code for code in previous if ranks.get(code, retention_rank + 1) <= retention_rank]
        retained = sorted(retained, key=lambda code: (ranks[code], code))[:holdings]
        selected = list(retained)
        if len(selected) < holdings:
            for code in frame["con_code"].astype(str):
                if code not in selected:
                    selected.append(code)
                if len(selected) == holdings:
                    break
        order = {code: rank for rank, code in enumerate(selected, start=1)}
        day = frame.loc[frame["con_code"].astype(str).isin(selected), ["date", "con_code", "score"]].copy()
        day["selection_rank"] = day["con_code"].astype(str).map(order)
        rows.append(day)
        previous = selected
    targets = pd.concat(rows, ignore_index=True).rename(columns={"date": "signal_date"})
    targets["regime"] = "BUCKET2_DIVERSIFIED_LOWVOL_DEVELOPMENT"
    return targets.sort_values(["signal_date", "selection_rank"]).reset_index(drop=True)


def compact_result(
    name: str,
    holdings: int,
    retention_rank: int,
    weights: dict[str, float],
    base: dict,
    stress: dict,
    bootstrap: dict,
    periods: dict,
    trades: pd.DataFrame,
    ledger: pd.DataFrame,
    selected_delisted: list[str],
) -> dict:
    period_excess = {name: item["annualized_excess"] for name, item in periods.items()}
    return _json_safe(
        {
            "candidate": name,
            "factor_count": len(weights),
            "weights": weights,
            "holdings": holdings,
            "retention_rank": retention_rank,
            "base_strategy_cagr": base["strategy"]["cagr"],
            "base_benchmark_cagr": base["benchmark"]["cagr"],
            "base_annualized_excess": base["annualized_excess"],
            "stress_annualized_excess": stress["annualized_excess"],
            "annualized_volatility": base["strategy"]["annualized_volatility"],
            "maximum_drawdown": base["strategy"]["maximum_drawdown"],
            "bootstrap_median": bootstrap["median"],
            "bootstrap_lower_95pct": bootstrap["interval_95pct"][0],
            "bootstrap_upper_95pct": bootstrap["interval_95pct"][1],
            "period_annualized_excess": period_excess,
            "positive_periods": sum(value > 0 for value in period_excess.values()),
            "trade_rows": len(trades),
            "minimum_trade_notional_cny": float(trades["notional"].min()) if not trades.empty else np.nan,
            "maximum_position_count": int(ledger["position_count"].max()),
            "selected_delisted_codes": selected_delisted,
            "strict_development_gates_pass": (
                base["annualized_excess"] >= 0.20
                and stress["annualized_excess"] >= 0.20
                and bootstrap["interval_95pct"][0] > 0.0
                and sum(value > 0 for value in period_excess.values()) >= 2
                and (trades.empty or bool(trades["notional"].ge(5000.0 - 1e-8).all()))
                and not selected_delisted
            ),
        }
    )


def render(report: dict) -> str:
    lines = [
        "# 桶2低波动分散化开发 V1",
        "",
        "- 作用：仅在已经解盲的桶2上开发下一只盲测候选。",
        "- 桶3未来收益：未读取、未下载。",
        "- 账户：20000元；T+1；100股；单腿至少5000元；佣金最低5元。",
        "",
        "|候选|持仓|保留名次|基础超额|压力超额|波动|回撤|Bootstrap下界|成交腿|开发门槛|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        lines.append(
            f"|{item['candidate']}|{item['holdings']}|{item['retention_rank']}|"
            f"{item['base_annualized_excess']:.2%}|{item['stress_annualized_excess']:.2%}|"
            f"{item['annualized_volatility']:.2%}|{item['maximum_drawdown']:.2%}|"
            f"{item['bootstrap_lower_95pct']:.2%}|{item['trade_rows']}|"
            f"{'PASS' if item['strict_development_gates_pass'] else 'FAIL'}|"
        )
    lines += [
        "",
        f"- 建议冻结候选：`{report['recommended_candidate_id']}`",
        "- 选择规则：先要求开发门槛全过，再优先3只持仓和低波动20/60平台化组合，最后比较Bootstrap下界。",
        "- 本报告是开发证据，不是独立样本成功证据，也不生成仓位或订单。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = contract["paths"]
    features = pd.read_parquet(ROOT / paths["features"])
    panel = pd.read_parquet(ROOT / paths["holdout_panel"])
    benchmark_raw = pd.read_parquet(ROOT / paths["benchmark"])
    master = pd.read_parquet(ROOT / contract["inputs"]["master"])
    master = master.loc[master["split_bucket"].eq(2)].copy()
    base_reference = pd.read_parquet(ROOT / paths["base_ledger"])
    calendar = pd.DatetimeIndex(pd.to_datetime(base_reference["date"]))
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    benchmark = _benchmark_series(benchmark_raw, calendar, float(contract["account"]["initial_cash_cny"]))
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    results: list[dict] = []
    for score_name, weights in SCORES.items():
        for holdings in (1, 2, 3):
            for retention_rank in (holdings, holdings * 2):
                targets = select_targets(features, weights, holdings, retention_rank)
                base_ledger, base_trades = run_small_account_open_backtest(
                    execution, targets, calendar, 20000.0, _costs(contract, "base_slippage_bps_per_leg"), gap
                )
                stress_ledger, stress_trades = run_small_account_open_backtest(
                    execution, targets, calendar, 20000.0, _costs(contract, "stress_slippage_bps_per_leg"), gap
                )
                base = summarize(base_ledger, base_trades, benchmark, 20000.0)
                stress = summarize(stress_ledger, stress_trades, benchmark, 20000.0)
                bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
                periods = _periods(base_ledger, benchmark, contract)
                selected_codes = set(targets["con_code"].astype(str))
                selected_delisted = master.loc[
                    master["ts_code"].astype(str).isin(selected_codes)
                    & master["delist_date"].notna()
                    & master["delist_date"].le(pd.Timestamp(contract["periods"]["evaluation_end"])),
                    "ts_code",
                ].astype(str).tolist()
                results.append(
                    compact_result(
                        score_name,
                        holdings,
                        retention_rank,
                        weights,
                        base,
                        stress,
                        bootstrap,
                        periods,
                        base_trades,
                        base_ledger,
                        selected_delisted,
                    )
                )
                print(
                    f"完成 {score_name} 持仓{holdings} 保留{retention_rank}："
                    f"超额{base['annualized_excess']:.2%}，Bootstrap下界{bootstrap['interval_95pct'][0]:.2%}",
                    flush=True,
                )
    passing = [item for item in results if item["strict_development_gates_pass"]]
    platform = [
        item for item in passing
        if item["candidate"] == "LV20_60_EQ" and item["holdings"] == 3
    ]
    eligible = platform or [item for item in passing if item["holdings"] == 3] or passing
    recommended = max(eligible, key=lambda item: item["bootstrap_lower_95pct"]) if eligible else None
    report = _json_safe(
        {
            "project_id": "A_SHARE_BUCKET2_DIVERSIFIED_LOWVOL_DEVELOPMENT_V1",
            "status": "DEVELOPMENT_CANDIDATE_FOUND" if recommended else "NO_DEVELOPMENT_CANDIDATE",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "future_bucket3_returns_read": False,
            "candidate_count": len(results),
            "passing_candidate_count": len(passing),
            "recommended_candidate_id": (
                f"{recommended['candidate']}_H{recommended['holdings']}_R{recommended['retention_rank']}"
                if recommended else None
            ),
            "recommended_candidate": recommended,
            "selection_rule_frozen_in_code_before_run": True,
            "results": sorted(
                results,
                key=lambda item: (item["strict_development_gates_pass"], item["bootstrap_lower_95pct"]),
                reverse=True,
            ),
            "governance": {
                "development_only": True,
                "future_bucket3_untouched": True,
                "paper_signal": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
            },
        }
    )
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), OUTPUT_JSON)
    atomic_text(render(report), OUTPUT_MD)
    print(
        json.dumps(
            {
                "状态": report["status"],
                "通过候选数": len(passing),
                "建议候选": report["recommended_candidate_id"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
