"""按每 20 个市场交易日纠正并重新冻结沪深 A 股信号；不读取组合收益。"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1_4_1_signal_freeze_calendar_corrected.yaml"
V14_PATH = ROOT / "scripts/freeze_a_share_hs_concentrated_low_risk_trend_v1_4_signal.py"
V14_SPEC = importlib.util.spec_from_file_location("a_share_hs_signal_v14", V14_PATH)
assert V14_SPEC and V14_SPEC.loader
V14 = importlib.util.module_from_spec(V14_SPEC)
sys.modules[V14_SPEC.name] = V14
V14_SPEC.loader.exec_module(V14)


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def freeze_signals_calendar_corrected(
    contract: dict[str, Any],
    features: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_dates = pd.DatetimeIndex(
        features.loc[features["entry_eligible"].fillna(False), "date"].unique()
    ).sort_values()
    if len(eligible_dates) == 0:
        raise RuntimeError("没有任何满足冻结入场条件的信号日")
    first_candidate_date = eligible_dates[0]
    anchor_position = int(calendar_dates.searchsorted(first_candidate_date))
    if anchor_position >= len(calendar_dates) or calendar_dates[anchor_position] != first_candidate_date:
        raise RuntimeError("首个候选日不属于冻结市场日历")
    frequency = int(contract["selection"]["frequency_trading_days"])
    maximum_feature_date = pd.Timestamp(features["date"].max()).normalize()
    signal_dates = calendar_dates[anchor_position::frequency]
    signal_dates = signal_dates[signal_dates <= maximum_feature_date]
    return_panel = features.pivot_table(
        index="date", columns="ts_code", values="log_return", aggfunc="last"
    ).sort_index()
    features.drop(columns=["log_return"], inplace=True)
    events: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        snapshot = features.loc[features["date"].eq(signal_date)].copy()
        return_history = return_panel.loc[:signal_date].tail(
            int(contract["selection"]["correlation_window_days"])
        )
        selected20 = V14.ENGINE.select_low_risk_names(
            snapshot,
            return_history,
            maximum_names=int(contract["selection"]["maximum_names"]),
            max_names_per_industry=int(contract["selection"]["max_names_per_industry"]),
            max_pairwise_correlation=float(contract["selection"]["max_pairwise_correlation"]),
            min_pair_observations=int(contract["selection"]["correlation_min_valid_observations"]),
        )
        candidate_count = int(snapshot["entry_eligible"].fillna(False).sum()) if not snapshot.empty else 0
        top3 = selected20[:3]
        for rank, code in enumerate(selected20, start=1):
            row = snapshot.loc[snapshot["ts_code"].eq(code)].sort_values("ts_code").iloc[0]
            selected_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "rank": rank,
                    "stable_risk_score": float(row["stable_risk_score"]),
                    "ts_vol_percentile": float(row["ts_vol_percentile"]),
                    "momentum_60_5": float(row["momentum_60_5"]),
                    "industry_l1": str(row["industry_l1"]),
                    "candidate_count": candidate_count,
                }
            )
        events.append(
            {
                "signal_date": signal_date,
                "candidate_count": candidate_count,
                "top1": json.dumps(selected20[:1], ensure_ascii=False),
                "top3": json.dumps(top3, ensure_ascii=False),
                "top5": json.dumps(selected20[:5], ensure_ascii=False),
                "top20": json.dumps(selected20[:20], ensure_ascii=False),
                "top3_count": len(top3),
                "cash_weight": 1.0 - sum(V14.ENGINE.target_weights(top3, variant=3).values()),
            }
        )
    return pd.DataFrame(events), pd.DataFrame(selected_rows)


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    features, calendar_dates, market_receipts = V14.build_signal_panel(contract)
    events, selected = freeze_signals_calendar_corrected(contract, features, calendar_dates)
    root = project_path(contract["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    events_path = project_path(contract["outputs"]["signal_events"])
    selected_path = project_path(contract["outputs"]["selected_symbols"])
    events.to_parquet(events_path, index=False)
    selected.to_parquet(selected_path, index=False)
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt = {
        "contract_id": contract["contract_id"],
        "model_id": contract["model_id"],
        "supersedes_signal_contract_id": contract["supersedes_signal_contract_id"],
        "frozen_at": frozen_at,
        "status": "SIGNAL_FROZEN_BEFORE_RETURN_READ",
        "view_status": "NO_VIEW",
        "schedule_mode": "EVERY_20_MARKET_TRADING_DAYS",
        "portfolio_return_values_read": False,
        "positions_generated": False,
        "orders_generated": False,
        "signal_dates": len(events),
        "zero_candidate_signal_dates": int(events["candidate_count"].eq(0).sum()),
        "selected_symbol_count": int(selected["ts_code"].nunique()) if not selected.empty else 0,
        "candidate_count_min": int(events["candidate_count"].min()),
        "candidate_count_max": int(events["candidate_count"].max()),
        "market_inputs": market_receipts,
        "artifacts": {
            "signal_events": {
                "path": events_path.relative_to(ROOT).as_posix(),
                "sha256": V14.sha256_file(events_path),
            },
            "selected_symbols": {
                "path": selected_path.relative_to(ROOT).as_posix(),
                "sha256": V14.sha256_file(selected_path),
            },
        },
    }
    receipt_path = project_path(contract["outputs"]["receipt"])
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "A_SHARE_HS_SIGNAL_FREEZE_V1_4_1",
        "contract_id": contract["contract_id"],
        "model_id": contract["model_id"],
        "audited_at": frozen_at,
        "status": receipt["status"],
        "view_status": "NO_VIEW",
        "schedule_mode": receipt["schedule_mode"],
        "portfolio_return_values_read": False,
        "signals_generated": True,
        "positions_generated": False,
        "orders_generated": False,
        "signal_dates": receipt["signal_dates"],
        "zero_candidate_signal_dates": receipt["zero_candidate_signal_dates"],
        "selected_symbol_count": receipt["selected_symbol_count"],
        "candidate_count_min": receipt["candidate_count_min"],
        "candidate_count_max": receipt["candidate_count_max"],
        "artifacts": receipt["artifacts"],
        "conclusion": "信号已按每20个市场交易日重新冻结；零候选日明确持有现金。",
    }
    report_path = project_path(contract["outputs"]["report_json"])
    report_md_path = project_path(contract["outputs"]["report_markdown"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md_path.write_text(
        "# 沪深 A 股 V1.4.1 日历纠正信号冻结\n\n"
        f"- 状态：`{report['status']}`\n"
        "- 调仓频率：每 20 个市场交易日\n"
        f"- 信号日：{report['signal_dates']}\n"
        f"- 零候选现金日：{report['zero_candidate_signal_dates']}\n"
        f"- 入选证券全集：{report['selected_symbol_count']} 只\n"
        "- 组合收益读取：否\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "状态": report["status"],
                "信号日": report["signal_dates"],
                "零候选日": report["zero_candidate_signal_dates"],
                "入选证券": report["selected_symbol_count"],
                "组合收益读取": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
