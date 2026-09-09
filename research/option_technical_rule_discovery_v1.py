"""在冻结发现期内比较四种经典技术方向规则。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.option_trade_envelope_v1 import (  # noqa: E402
    assert_input_hashes,
    build_schedule,
    evaluate_leg,
    select_contract,
)
from scripts.freeze_510300_option_trade_envelope_v1 import (  # noqa: E402
    verify_protocol as verify_vehicle_protocol,
)


CONFIG = ROOT / "config" / "510300_option_technical_rule_discovery_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_sources(cutoff: pd.Timestamp, vehicle_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    eod = pd.read_parquet(
        ROOT / vehicle_config["inputs"]["option_eod"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    risk = pd.read_parquet(
        ROOT / vehicle_config["inputs"]["option_risk"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    benchmark = pd.read_parquet(
        ROOT / vehicle_config["inputs"]["benchmark"]["path"],
        filters=[("date", "<=", cutoff)],
    )
    eod["trade_date"] = pd.to_datetime(eod["trade_date"]).dt.normalize()
    risk["trade_date"] = pd.to_datetime(risk["trade_date"]).dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    if max(eod["trade_date"].max(), risk["trade_date"].max(), benchmark["date"].max()) > cutoff:
        raise RuntimeError("发现期加载越过冻结截止日")
    if eod.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("期权日线存在重复键")
    if risk.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("期权风险指标存在重复键")
    panel = eod.merge(
        risk[["trade_date", "contract_code", "delta", "implied_volatility"]],
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
    )
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"]).dt.normalize()
    panel["dte"] = (panel["expiry_date"] - panel["trade_date"]).dt.days
    panel["premium_per_contract"] = panel["close"] * panel["contract_unit"]
    benchmark = (
        benchmark[["date", "close"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return panel, benchmark


def build_rule_sides(benchmark: pd.DataFrame) -> pd.DataFrame:
    rules = benchmark.copy().sort_values("date").reset_index(drop=True)
    close = rules["close"].astype(float)
    momentum120 = close / close.shift(120) - 1.0
    ma20 = close.rolling(20, min_periods=20).mean()
    ma120 = close.rolling(120, min_periods=120).mean()
    rules["R1_TSMOM_120"] = np.where(momentum120.ge(0), "C", "P")
    rules.loc[momentum120.isna(), "R1_TSMOM_120"] = None
    rules["R2_MA_20_120"] = np.where(ma20.ge(ma120), "C", "P")
    rules.loc[ma120.isna(), "R2_MA_20_120"] = None
    prior_high55 = close.shift(1).rolling(55, min_periods=55).max()
    prior_low20 = close.shift(1).rolling(20, min_periods=20).min()
    state: str | None = None
    donchian: list[str | None] = []
    for price, high, low, average in zip(close, prior_high55, prior_low20, ma120):
        if pd.notna(high) and price > high:
            state = "C"
        elif pd.notna(low) and price < low:
            state = "P"
        elif state is None and pd.notna(average):
            state = "C" if price >= average else "P"
        donchian.append(state)
    rules["R3_DONCHIAN_55_20"] = donchian
    ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    histogram = macd - signal
    rules["R4_MACD_12_26_9"] = np.where(histogram.ge(0), "C", "P")
    rules.loc[histogram.isna(), "R4_MACD_12_26_9"] = None
    return rules


def evaluate_rule(
    rule_id: str,
    rule_sides: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    vehicle_config: dict,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schedule = build_schedule(benchmark, vehicle_config)
    sides = rule_sides.set_index("date")[rule_id]
    benchmark_close = benchmark.set_index("date")["close"]
    panel_by_date = {date: rows for date, rows in panel.groupby("trade_date", sort=False)}
    indexed_panel = panel.set_index(["trade_date", "contract_code"]).sort_index()
    initial = float(vehicle_config["execution_envelope"]["initial_cash_cny"])
    capital = initial
    minimum_passes: list[bool] = []
    correct: list[bool] = []
    ledger: list[dict[str, Any]] = []
    for dates in schedule:
        side = sides.get(dates["signal_date"], None)
        before = capital
        if side not in ("C", "P"):
            ledger.append(
                {
                    **{key: str(value.date()) for key, value in dates.items()},
                    "status": "NO_RULE_WARMUP",
                    "capital_before": before,
                    "capital_after": capital,
                }
            )
            continue
        signal_rows = panel_by_date.get(dates["signal_date"], pd.DataFrame())
        selection = select_contract(signal_rows, str(side), vehicle_config)
        leg = evaluate_leg(
            selection,
            dates["entry_date"],
            dates["exit_date"],
            indexed_panel,
            capital,
            vehicle_config,
        )
        capital = max(0.0, capital + leg.pnl_cny)
        realized_up = benchmark_close.loc[dates["exit_date"]] >= benchmark_close.loc[dates["entry_date"]]
        correct.append((side == "C") == bool(realized_up))
        if leg.executed:
            minimum_passes.append(leg.entry_minimum_pass)
        ledger.append(
            {
                **{key: str(value.date()) for key, value in dates.items()},
                "side": side,
                "contract_code": None if selection is None else selection["contract_code"],
                "quantity": None if selection is None else selection["quantity"],
                "status": leg.status,
                "direction_correct": correct[-1],
                "capital_before": before,
                "pnl_cny": leg.pnl_cny,
                "capital_after": capital,
                "entry_notional_cny": leg.entry_notional_cny,
                "entry_minimum_pass": leg.entry_minimum_pass,
            }
        )
    if not schedule:
        raise RuntimeError("发现期没有策略日程")
    start = schedule[0]["entry_date"]
    end = schedule[-1]["exit_date"]
    days = max((end - start).days, 1)
    strategy_cagr = -1.0 if capital <= 0 else (capital / initial) ** (365.2425 / days) - 1.0
    benchmark_total = float(benchmark_close.loc[end] / benchmark_close.loc[start] - 1.0)
    benchmark_cagr = (1.0 + benchmark_total) ** (365.2425 / days) - 1.0
    all_minimum = bool(minimum_passes) and all(minimum_passes)
    metrics = {
        "rule_id": rule_id,
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "signal_count": len(schedule),
        "direction_observation_count": len(correct),
        "direction_accuracy": float(np.mean(correct)) if correct else None,
        "initial_cash_cny": initial,
        "final_equity_cny": capital,
        "strategy_total_return": capital / initial - 1.0,
        "strategy_cagr": strategy_cagr,
        "benchmark_total_return": benchmark_total,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess": strategy_cagr - benchmark_cagr,
        "all_opening_trades_at_least_5000": all_minimum,
        "executed_opening_count": len(minimum_passes),
    }
    return metrics, ledger


def run_discovery(config: dict, vehicle_config: dict) -> dict:
    cutoff = pd.Timestamp(config["dates"]["discovery_end"])
    panel, benchmark = load_sources(cutoff, vehicle_config)
    rules = build_rule_sides(benchmark)
    candidates = []
    ledgers = {}
    for rule_id in sorted(config["candidates"]):
        metrics, ledger = evaluate_rule(rule_id, rules, panel, benchmark, vehicle_config)
        candidates.append(metrics)
        ledgers[rule_id] = ledger
    ranked = sorted(candidates, key=lambda item: (-item["annualized_excess"], item["rule_id"]))
    winner = ranked[0]
    threshold = float(config["selection"]["annualized_excess_minimum"])
    gate = (
        winner["annualized_excess"] >= threshold
        and winner["all_opening_trades_at_least_5000"]
    )
    return {
        "loaded_option_max_date": str(panel["trade_date"].max().date()),
        "loaded_benchmark_max_date": str(benchmark["date"].max().date()),
        "candidate_family_size": len(candidates),
        "candidates": ranked,
        "selected_rule_id": winner["rule_id"],
        "selected_rule_metrics": winner,
        "selected_rule_ledger": ledgers[winner["rule_id"]],
        "selection_gate": {
            "winner_annualized_excess_at_least_20pct": winner["annualized_excess"] >= threshold,
            "minimum_opening_trade_gate": winner["all_opening_trades_at_least_5000"],
            "eligible_for_frozen_pseudo_confirmation": gate,
        },
    }


def write_outputs(result: dict, config: dict, manifest: dict) -> None:
    payload = {
        "project_id": config["protocol"]["project_id"],
        "status": (
            "DISCOVERY_WINNER_SELECTED_CONFIRMATION_NOT_AUTHORIZED"
            if result["selection_gate"]["eligible_for_frozen_pseudo_confirmation"]
            else "DISCOVERY_RULE_FAMILY_REJECTED_FROZEN"
        ),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "MULTIPLE_RULE_DISCOVERY_ONLY_POST_DISCOVERY_UNREAD",
        **result,
        "safety": config["governance"],
    }
    json_path = ROOT / config["paths"]["result_json"]
    markdown_path = ROOT / config["paths"]["result_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    rows = [
        f"|{item['rule_id']}|{item['direction_accuracy']:.2%}|{item['strategy_cagr']:.2%}|"
        f"{item['benchmark_cagr']:.2%}|{item['annualized_excess']:.2%}|{item['final_equity_cny']:.2f}|"
        for item in payload["candidates"]
    ]
    markdown_path.write_text(
        "\n".join(
            [
                "# 510300期权经典技术规则发现期V1",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 证据：`{payload['evidence_label']}`",
                f"- 发现期加载截止：{payload['loaded_option_max_date']}",
                f"- 候选数：{payload['candidate_family_size']}（多重比较已披露）",
                "",
                "|规则|方向准确率|策略年化|基准年化|年化超额|期末权益|",
                "|---|---:|---:|---:|---:|---:|",
                *rows,
                "",
                f"- 自动入选：`{payload['selected_rule_id']}`",
                f"- 年化超额至少20%：{payload['selection_gate']['winner_annualized_excess_at_least_20pct']}",
                f"- 开仓至少5000元：{payload['selection_gate']['minimum_opening_trade_gate']}",
                f"- 允许另行冻结2023伪确认：{payload['selection_gate']['eligible_for_frozen_pseudo_confirmation']}",
                "",
                "2023伪确认、2024验证和2025后最终留出均未在本协议读取。",
                "发现期胜者不等于有效策略，必须先冻结胜者再逐级外推。历史成交价包络仍不等于历史买卖盘。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    from scripts.freeze_510300_option_technical_rule_discovery_v1 import verify_protocol

    config, manifest = verify_protocol()
    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    result = run_discovery(config, vehicle_config)
    write_outputs(result, config, manifest)
    concise = {key: value for key, value in result.items() if key != "selected_rule_ledger"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
