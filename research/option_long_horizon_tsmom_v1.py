"""40交易日持有的510300深度实值期权时间序列动量V1。"""

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
    evaluate_leg,
    select_contract,
)
from research.option_technical_rule_discovery_v1 import load_sources  # noqa: E402
from scripts.freeze_510300_option_trade_envelope_v1 import (  # noqa: E402
    verify_protocol as verify_vehicle_protocol,
)


CONFIG = ROOT / "config" / "510300_option_long_horizon_tsmom_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def build_schedule(benchmark: pd.DataFrame, config: dict) -> list[dict[str, pd.Timestamp]]:
    start = pd.Timestamp(config["dates"]["first_signal"])
    end = pd.Timestamp(config["dates"]["development_end"])
    holding = int(config["schedule"]["holding_trading_days"])
    dates = benchmark.loc[benchmark["date"].between(start, end), "date"].tolist()
    schedule: list[dict[str, pd.Timestamp]] = []
    index = 0
    while index + 1 + holding < len(dates):
        schedule.append(
            {
                "signal_date": dates[index],
                "entry_date": dates[index + 1],
                "exit_date": dates[index + 1 + holding],
            }
        )
        index += holding + 1
    if schedule and schedule[-1]["exit_date"] > end:
        raise RuntimeError("长周期日程越过开发期")
    return schedule


def momentum_sides(benchmark: pd.DataFrame) -> pd.Series:
    ordered = benchmark.sort_values("date").copy()
    momentum = ordered["close"] / ordered["close"].shift(120) - 1.0
    sides = pd.Series(np.where(momentum.ge(0), "C", "P"), index=ordered["date"])
    sides.loc[momentum.isna().to_numpy()] = None
    return sides


def run_development(config: dict, vehicle_config: dict) -> dict:
    cutoff = pd.Timestamp(config["dates"]["development_end"])
    panel, benchmark = load_sources(cutoff, vehicle_config)
    schedule = build_schedule(benchmark, config)
    sides = momentum_sides(benchmark)
    benchmark_close = benchmark.set_index("date")["close"]
    panel_by_date = {date: rows for date, rows in panel.groupby("trade_date", sort=False)}
    indexed_panel = panel.set_index(["trade_date", "contract_code"]).sort_index()
    initial = float(config["execution_envelope"]["initial_cash_cny"])
    capitals = {"TSMOM_120": initial, "PERFECT_DIRECTION_ORACLE": initial}
    ledgers: list[dict[str, Any]] = []
    minimum_passes: dict[str, list[bool]] = {name: [] for name in capitals}
    direction_correct: list[bool] = []
    for dates in schedule:
        realized_up = benchmark_close.loc[dates["exit_date"]] >= benchmark_close.loc[dates["entry_date"]]
        strategy_side = sides.get(dates["signal_date"], None)
        oracle_side = "C" if realized_up else "P"
        row: dict[str, Any] = {
            **{key: str(value.date()) for key, value in dates.items()},
            "strategy_side": strategy_side,
            "oracle_side": oracle_side,
        }
        for scenario, side in (
            ("TSMOM_120", strategy_side),
            ("PERFECT_DIRECTION_ORACLE", oracle_side),
        ):
            before = capitals[scenario]
            if side not in ("C", "P"):
                row[scenario] = {
                    "status": "NO_RULE_WARMUP",
                    "capital_before": before,
                    "capital_after": before,
                }
                continue
            signal_rows = panel_by_date.get(dates["signal_date"], pd.DataFrame())
            selection = select_contract(signal_rows, str(side), config)
            leg = evaluate_leg(
                selection,
                dates["entry_date"],
                dates["exit_date"],
                indexed_panel,
                before,
                config,
            )
            after = max(0.0, before + leg.pnl_cny)
            capitals[scenario] = after
            if leg.executed:
                minimum_passes[scenario].append(leg.entry_minimum_pass)
            row[scenario] = {
                "side": side,
                "contract_code": None if selection is None else selection["contract_code"],
                "quantity": None if selection is None else selection["quantity"],
                "status": leg.status,
                "capital_before": before,
                "pnl_cny": leg.pnl_cny,
                "capital_after": after,
                "entry_notional_cny": leg.entry_notional_cny,
                "entry_minimum_pass": leg.entry_minimum_pass,
            }
        if strategy_side in ("C", "P"):
            direction_correct.append((strategy_side == "C") == bool(realized_up))
        ledgers.append(row)
    if not schedule:
        raise RuntimeError("长周期开发期没有日程")
    start = schedule[0]["entry_date"]
    end = schedule[-1]["exit_date"]
    days = max((end - start).days, 1)
    benchmark_total = float(benchmark_close.loc[end] / benchmark_close.loc[start] - 1.0)
    benchmark_cagr = (1.0 + benchmark_total) ** (365.2425 / days) - 1.0
    scenarios = {}
    for name, capital in capitals.items():
        strategy_cagr = -1.0 if capital <= 0 else (capital / initial) ** (365.2425 / days) - 1.0
        scenarios[name] = {
            "initial_cash_cny": initial,
            "final_equity_cny": capital,
            "total_return": capital / initial - 1.0,
            "cagr": strategy_cagr,
            "benchmark_cagr": benchmark_cagr,
            "annualized_excess": strategy_cagr - benchmark_cagr,
            "executed_opening_count": len(minimum_passes[name]),
            "all_opening_trades_at_least_5000": bool(minimum_passes[name])
            and all(minimum_passes[name]),
        }
    strategy = scenarios["TSMOM_120"]
    threshold = float(config["evaluation"]["annualized_excess_minimum"])
    pass_gate = (
        strategy["annualized_excess"] >= threshold
        and strategy["all_opening_trades_at_least_5000"]
    )
    return {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "loaded_option_max_date": str(panel["trade_date"].max().date()),
        "loaded_benchmark_max_date": str(benchmark["date"].max().date()),
        "signal_count": len(schedule),
        "strategy_direction_accuracy": float(np.mean(direction_correct))
        if direction_correct
        else None,
        "benchmark_total_return": benchmark_total,
        "benchmark_cagr": benchmark_cagr,
        "scenarios": scenarios,
        "development_gate": {
            "strategy_annualized_excess_at_least_20pct": strategy["annualized_excess"]
            >= threshold,
            "minimum_opening_trade_gate": strategy["all_opening_trades_at_least_5000"],
            "eligible_for_frozen_pseudo_confirmation": pass_gate,
        },
        "ledger": ledgers,
    }


def write_outputs(result: dict, config: dict, manifest: dict) -> None:
    payload = {
        "project_id": config["protocol"]["project_id"],
        "status": (
            "DEVELOPMENT_PASS_CONFIRMATION_NOT_AUTHORIZED"
            if result["development_gate"]["eligible_for_frozen_pseudo_confirmation"]
            else "DEVELOPMENT_REJECTED_FROZEN"
        ),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "LONG_HORIZON_DEVELOPMENT_ONLY_POST_DEVELOPMENT_UNREAD",
        "factor_count": 1,
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
        f"|{name}|{item['cagr']:.2%}|{item['benchmark_cagr']:.2%}|"
        f"{item['annualized_excess']:.2%}|{item['final_equity_cny']:.2f}|"
        for name, item in payload["scenarios"].items()
    ]
    markdown_path.write_text(
        "\n".join(
            [
                "# 510300期权长周期时间序列动量V1开发结果",
                "",
                f"- 状态：`{payload['status']}`",
                f"- 证据：`{payload['evidence_label']}`",
                f"- 因子数：{payload['factor_count']} / 10",
                f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
                f"- 非重叠40日周期：{payload['signal_count']}",
                f"- 时间序列动量方向准确率：{payload['strategy_direction_accuracy']:.2%}",
                "",
                "|情景|策略年化|基准年化|年化超额|期末权益|",
                "|---|---:|---:|---:|---:|",
                *rows,
                "",
                f"- 策略年化超额至少20%：{payload['development_gate']['strategy_annualized_excess_at_least_20pct']}",
                f"- 开仓至少5000元：{payload['development_gate']['minimum_opening_trade_gate']}",
                f"- 允许另行冻结2023伪确认：{payload['development_gate']['eligible_for_frozen_pseudo_confirmation']}",
                "",
                "完美方向只作为车辆上界。2023、2024与最终留出均未读取。",
                "历史最高/最低成交价包络仍不构成买卖盘证明，且不授权仓位、订单或实盘。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    from scripts.freeze_510300_option_long_horizon_tsmom_v1 import verify_protocol

    config, manifest = verify_protocol()
    from scripts.freeze_510300_option_technical_rule_discovery_v1 import (
        verify_protocol as verify_data_loader_protocol,
    )

    verify_data_loader_protocol()
    vehicle_config, _ = verify_vehicle_protocol()
    assert_input_hashes(vehicle_config)
    result = run_development(config, vehicle_config)
    write_outputs(result, config, manifest)
    concise = {key: value for key, value in result.items() if key != "ledger"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
