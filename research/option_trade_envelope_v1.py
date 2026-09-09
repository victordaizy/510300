"""510300期权日线成交价不利包络的开发期车辆可行性检验。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_option_trade_envelope_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_input_hashes(config: dict) -> None:
    for name, item in config["inputs"].items():
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入不存在：{name}={path}")
        if sha256(path) != str(item["sha256"]).lower():
            raise RuntimeError(f"冻结输入哈希漂移：{name}")


def load_development_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在Parquet层过滤，禁止把验证或最终留出行载入内存。"""

    cutoff = pd.Timestamp(config["dates"]["development_end"])
    eod = pd.read_parquet(
        ROOT / config["inputs"]["option_eod"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    risk = pd.read_parquet(
        ROOT / config["inputs"]["option_risk"]["path"],
        filters=[("trade_date", "<=", cutoff)],
    )
    benchmark = pd.read_parquet(
        ROOT / config["inputs"]["benchmark"]["path"],
        filters=[("date", "<=", cutoff)],
    )
    eod["trade_date"] = pd.to_datetime(eod["trade_date"]).dt.normalize()
    risk["trade_date"] = pd.to_datetime(risk["trade_date"]).dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    for name, frame, column in (
        ("option_eod", eod, "trade_date"),
        ("option_risk", risk, "trade_date"),
        ("benchmark", benchmark, "date"),
    ):
        if frame.empty or frame[column].max() > cutoff:
            raise RuntimeError(f"{name}越过冻结开发期截止日")
    if eod.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("期权日线存在重复合约日期")
    if risk.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("期权风险指标存在重复合约日期")
    panel = eod.merge(
        risk[["trade_date", "contract_code", "delta", "implied_volatility"]],
        on=["trade_date", "contract_code"],
        how="left",
        validate="one_to_one",
    )
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"]).dt.normalize()
    panel["dte"] = (panel["expiry_date"] - panel["trade_date"]).dt.days
    panel["premium_per_contract"] = panel["close"] * panel["contract_unit"]
    benchmark = benchmark[["date", "close"]].dropna().drop_duplicates("date")
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    return panel, benchmark


def build_schedule(benchmark: pd.DataFrame, config: dict) -> list[dict[str, pd.Timestamp]]:
    start = pd.Timestamp(config["dates"]["feature_start"])
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
        raise RuntimeError("车辆日程越过开发期截止日")
    return schedule


def select_contract(
    signal_rows: pd.DataFrame,
    option_type: str,
    config: dict,
) -> dict[str, Any] | None:
    rule = config["contract_selection"]
    rows = signal_rows.copy()
    rows = rows.loc[
        rows["option_type"].eq(option_type)
        & ~rows["is_adjusted"].astype(bool)
        & rows["contract_unit"].eq(int(rule["contract_unit_required"]))
        & rows["dte"].between(
            int(rule["minimum_calendar_days_to_expiry"]),
            int(rule["maximum_calendar_days_to_expiry"]),
        )
        & rows["delta"].abs().between(
            float(rule["absolute_delta_minimum"]),
            float(rule["absolute_delta_maximum"]),
        )
        & rows["volume"].ge(float(rule["minimum_signal_day_volume"]))
        & rows["open_interest"].ge(float(rule["minimum_signal_day_open_interest"]))
        & rows["premium_per_contract"].between(
            float(rule["minimum_signal_premium_per_contract_cny"]),
            float(rule["maximum_signal_premium_per_contract_cny"]),
        )
    ].copy()
    if rows.empty:
        return None
    rows["absolute_delta_distance"] = (
        rows["delta"].abs() - float(rule["target_absolute_delta"])
    ).abs()
    rows["expiry_distance"] = (
        rows["dte"] - int(rule["target_calendar_days_to_expiry"])
    ).abs()
    rows = rows.sort_values(
        ["absolute_delta_distance", "expiry_distance", "volume", "contract_code"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    row = rows.iloc[0]
    quantity = math.ceil(
        float(rule["target_total_signal_premium_cny"])
        / float(row["premium_per_contract"])
    )
    quantity = min(quantity, int(rule["maximum_contracts"]))
    return {
        "contract_code": str(row["contract_code"]),
        "option_type": option_type,
        "quantity": int(quantity),
        "signal_close": float(row["close"]),
        "signal_premium": float(row["premium_per_contract"] * quantity),
        "signal_volume": float(row["volume"]),
        "signal_open_interest": float(row["open_interest"]),
        "signal_delta": float(row["delta"]),
        "signal_dte": int(row["dte"]),
        "contract_unit": int(row["contract_unit"]),
    }


@dataclass(frozen=True)
class LegResult:
    status: str
    pnl_cny: float
    entry_notional_cny: float
    exit_notional_cny: float
    entry_minimum_pass: bool
    executed: bool


def evaluate_leg(
    selection: dict[str, Any] | None,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    indexed_panel: pd.DataFrame,
    capital_cny: float,
    config: dict,
) -> LegResult:
    if selection is None:
        return LegResult("NO_CONTRACT", 0.0, 0.0, 0.0, True, False)
    code = selection["contract_code"]
    quantity = int(selection["quantity"])
    unit = int(selection["contract_unit"])
    fee = float(config["execution_envelope"]["option_fee_cny_per_contract_per_leg"])
    minimum = float(config["execution_envelope"]["minimum_opening_trade_notional_cny"])
    try:
        entry = indexed_panel.loc[(entry_date, code)]
    except KeyError:
        return LegResult("NO_ENTRY_ROW", 0.0, 0.0, 0.0, True, False)
    if isinstance(entry, pd.DataFrame):
        raise ValueError("入场键不唯一")
    entry_price = float(entry["high"])
    entry_volume = float(entry["volume"])
    if not math.isfinite(entry_price) or entry_price <= 0 or entry_volume <= 0:
        return LegResult("NO_ENTRY_TRADE", 0.0, 0.0, 0.0, True, False)
    entry_notional = entry_price * unit * quantity
    minimum_pass = entry_notional >= minimum
    buy_fee = fee * quantity
    if entry_notional + buy_fee > capital_cny:
        return LegResult(
            "INSUFFICIENT_CASH", 0.0, entry_notional, 0.0, minimum_pass, False
        )
    try:
        exit_row = indexed_panel.loc[(exit_date, code)]
    except KeyError:
        exit_row = None
    if isinstance(exit_row, pd.DataFrame):
        raise ValueError("退出键不唯一")
    if exit_row is None:
        exit_notional = 0.0
        status = "MISSING_EXIT_MARKED_ZERO"
    else:
        exit_price = float(exit_row["low"])
        exit_volume = float(exit_row["volume"])
        if not math.isfinite(exit_price) or exit_price <= 0 or exit_volume <= 0:
            exit_notional = 0.0
            status = "NO_EXIT_TRADE_MARKED_ZERO"
        else:
            exit_notional = exit_price * unit * quantity
            status = "EXECUTED_WORST_TRADE_ENVELOPE"
    sell_fee = fee * quantity
    pnl = exit_notional - entry_notional - buy_fee - sell_fee
    return LegResult(status, pnl, entry_notional, exit_notional, minimum_pass, True)


def cagr(final_value: float, initial_value: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    if final_value <= 0:
        return -1.0
    return (final_value / initial_value) ** (365.2425 / days) - 1.0


def run_development(config: dict, panel: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    schedule = build_schedule(benchmark, config)
    panel_by_date = {date: rows for date, rows in panel.groupby("trade_date", sort=False)}
    indexed = panel.set_index(["trade_date", "contract_code"]).sort_index()
    benchmark_close = benchmark.set_index("date")["close"]
    initial = float(config["execution_envelope"]["initial_cash_cny"])
    capitals = {
        "ALWAYS_CALL": initial,
        "PERFECT_UNDERLYING_DIRECTION_ORACLE": initial,
        "PERFECT_OPTION_PNL_ORACLE": initial,
    }
    ledgers: list[dict[str, Any]] = []
    entry_minimum_passes: list[bool] = []
    for period, dates in enumerate(schedule, start=1):
        signal_rows = panel_by_date.get(dates["signal_date"], pd.DataFrame())
        call_selection = select_contract(signal_rows, "C", config)
        put_selection = select_contract(signal_rows, "P", config)
        underlying_return = (
            float(benchmark_close.loc[dates["exit_date"]])
            / float(benchmark_close.loc[dates["entry_date"]])
            - 1.0
        )
        period_row: dict[str, Any] = {
            "period": period,
            **{key: str(value.date()) for key, value in dates.items()},
            "underlying_total_return": underlying_return,
            "call_contract": None if call_selection is None else call_selection["contract_code"],
            "put_contract": None if put_selection is None else put_selection["contract_code"],
        }
        for scenario in capitals:
            capital_before = capitals[scenario]
            call_leg = evaluate_leg(
                call_selection,
                dates["entry_date"],
                dates["exit_date"],
                indexed,
                capital_before,
                config,
            )
            put_leg = evaluate_leg(
                put_selection,
                dates["entry_date"],
                dates["exit_date"],
                indexed,
                capital_before,
                config,
            )
            if scenario == "ALWAYS_CALL":
                side, chosen = "C", call_leg
            elif scenario == "PERFECT_UNDERLYING_DIRECTION_ORACLE":
                side, chosen = ("C", call_leg) if underlying_return >= 0 else ("P", put_leg)
            else:
                side, chosen = (
                    ("C", call_leg)
                    if call_leg.pnl_cny >= put_leg.pnl_cny
                    else ("P", put_leg)
                )
            capital_after = max(0.0, capital_before + chosen.pnl_cny)
            capitals[scenario] = capital_after
            if chosen.executed:
                entry_minimum_passes.append(chosen.entry_minimum_pass)
            period_row[scenario] = {
                "side": side,
                "status": chosen.status,
                "capital_before": capital_before,
                "pnl_cny": chosen.pnl_cny,
                "capital_after": capital_after,
                "entry_notional_cny": chosen.entry_notional_cny,
                "exit_notional_cny": chosen.exit_notional_cny,
                "entry_minimum_pass": chosen.entry_minimum_pass,
            }
        ledgers.append(period_row)
    if not schedule:
        raise RuntimeError("开发期没有可评估日程")
    start = schedule[0]["entry_date"]
    end = schedule[-1]["exit_date"]
    benchmark_initial = float(benchmark_close.loc[start])
    benchmark_final = float(benchmark_close.loc[end])
    benchmark_cagr = cagr(benchmark_final, benchmark_initial, start, end)
    scenarios = {}
    for name, final_value in capitals.items():
        strategy_cagr = cagr(final_value, initial, start, end)
        scenarios[name] = {
            "initial_cash_cny": initial,
            "final_equity_cny": final_value,
            "total_return": final_value / initial - 1.0,
            "cagr": strategy_cagr,
            "benchmark_cagr": benchmark_cagr,
            "annualized_excess": strategy_cagr - benchmark_cagr,
        }
    oracle = scenarios["PERFECT_UNDERLYING_DIRECTION_ORACLE"]
    threshold = float(config["evaluation"]["research_value_threshold"])
    return {
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "period_count": len(schedule),
        "loaded_option_max_date": str(panel["trade_date"].max().date()),
        "loaded_benchmark_max_date": str(benchmark["date"].max().date()),
        "benchmark": {
            "initial_close": benchmark_initial,
            "final_close": benchmark_final,
            "total_return": benchmark_final / benchmark_initial - 1.0,
            "cagr": benchmark_cagr,
        },
        "scenarios": scenarios,
        "execution_audit": {
            "executed_leg_count": len(entry_minimum_passes),
            "all_opening_trades_at_least_5000": all(entry_minimum_passes),
            "minimum_entry_notional_pass_ratio": (
                float(sum(entry_minimum_passes) / len(entry_minimum_passes))
                if entry_minimum_passes
                else None
            ),
            "entry_price": "NEXT_DAY_DAILY_HIGH",
            "exit_price": "EXIT_DAY_DAILY_LOW",
            "historical_bid_ask_available": False,
        },
        "research_gate": {
            "perfect_direction_oracle_excess_at_least_20pct": oracle["annualized_excess"]
            >= threshold,
            "minimum_opening_trade_gate": all(entry_minimum_passes),
            "eligible_for_strategy_model_research": oracle["annualized_excess"] >= threshold
            and all(entry_minimum_passes),
        },
        "ledger": ledgers,
    }


def write_outputs(result: dict, config: dict, manifest: dict) -> None:
    payload = {
        "project_id": config["protocol"]["project_id"],
        "status": (
            "DEV_ORACLE_VEHICLE_FEASIBLE_NOT_A_STRATEGY"
            if result["research_gate"]["eligible_for_strategy_model_research"]
            else "DEV_ORACLE_VEHICLE_REJECTED"
        ),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "evidence_label": "DEVELOPMENT_ORACLE_ONLY_NO_VALIDATION_OR_HOLDOUT_READ",
        "deployable_strategy": False,
        **result,
        "safety": config["governance"],
    }
    json_path = ROOT / config["paths"]["result_json"]
    markdown_path = ROOT / config["paths"]["result_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    scenario_lines = []
    for name, item in payload["scenarios"].items():
        scenario_lines.append(
            f"|{name}|{item['cagr']:.2%}|{item['benchmark_cagr']:.2%}|"
            f"{item['annualized_excess']:.2%}|{item['final_equity_cny']:.2f}|"
        )
    markdown = "\n".join(
        [
            "# 510300期权不利成交价包络V1开发期结果",
            "",
            f"- 状态：`{payload['status']}`",
            f"- 证据：`{payload['evidence_label']}`",
            f"- 区间：{payload['period_start']} 至 {payload['period_end']}",
            f"- 非重叠周期：{payload['period_count']}",
            "- 2024验证期与2025后最终留出期：未读取",
            "",
            "|情景|策略年化|基准年化|年化超额|期末权益|",
            "|---|---:|---:|---:|---:|",
            *scenario_lines,
            "",
            "## 解释",
            "",
            "`PERFECT_UNDERLYING_DIRECTION_ORACLE`使用未来十日指数方向，",
            "`PERFECT_OPTION_PNL_ORACLE`直接选择事后损益更高的一侧；二者都不可能实盘，",
            "只回答车辆是否存在足够上界。入场按次日最高成交价、退出按退出日最低成交价并加费用。",
            "由于历史买卖盘仍不可见，本结果不是历史可执行证明，也不授权策略、仓位或订单。",
            "",
            "## 门槛",
            "",
            f"- 完美方向上界年化超额至少20%：{payload['research_gate']['perfect_direction_oracle_excess_at_least_20pct']}",
            f"- 所有实际开仓至少5000元：{payload['research_gate']['minimum_opening_trade_gate']}",
            f"- 可进入不超过10因子的模型研究：{payload['research_gate']['eligible_for_strategy_model_research']}",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_510300_option_trade_envelope_v1 import verify_protocol

    config, manifest = verify_protocol()
    assert_input_hashes(config)
    panel, benchmark = load_development_data(config)
    result = run_development(config, panel, benchmark)
    write_outputs(result, config, manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "ledger"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
