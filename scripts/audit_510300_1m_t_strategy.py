"""审计510300底仓做T研究所需的一分钟数据与执行前提。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "intraday_t_strategy.yaml"
JSON_REPORT_FILE = (
    PROJECT_ROOT / "reports" / "data_quality" / "510300_1m_t_strategy_readiness.json"
)
MARKDOWN_REPORT_FILE = (
    PROJECT_ROOT / "reports" / "data_quality" / "510300_1m_t_strategy_readiness.md"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def _expected_record_times() -> list[time]:
    morning = pd.date_range("2026-01-01 09:30", "2026-01-01 11:30", freq="1min")
    afternoon = pd.date_range("2026-01-01 13:01", "2026-01-01 15:00", freq="1min")
    return [value.time() for value in morning.append(afternoon)]


def _round_up_to_tick(value: float, tick: float) -> float:
    return math.ceil((value - 1e-12) / tick) * tick


def _buy_cost(notional: float, commission_rate: float, minimum_commission: float) -> float:
    return notional + max(notional * commission_rate, minimum_commission)


def _relative_error(actual: pd.Series, reference: pd.Series) -> pd.Series:
    denominator = reference.abs().replace(0.0, np.nan)
    output = (actual - reference).abs() / denominator
    zero_reference = reference.eq(0.0)
    output.loc[zero_reference & actual.eq(0.0)] = 0.0
    output.loc[zero_reference & actual.ne(0.0)] = np.inf
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    session = report["session_integrity"]
    reconciliation = report["daily_reconciliation"]
    liquidity = report["liquidity"]
    affordability = report["account_executability"]
    error_lines = (
        "\n".join(f"- `{item['code']}`：{item['message']}" for item in report["errors"])
        if report["errors"]
        else "- 无"
    )
    warning_lines = (
        "\n".join(f"- `{item['code']}`：{item['message']}" for item in report["warnings"])
        if report["warnings"]
        else "- 无"
    )
    return f"""# 510300 一分钟底仓做T数据就绪审计

结论：**{report['status']}**。本报告只判断数据和执行约束是否足以开始回测，不包含任何策略收益结果。

## 核心证据

| 项目 | 结果 |
|---|---:|
| 数据区间 | {coverage['first_trade_time']} 至 {coverage['last_trade_time']} |
| 一分钟记录数 | {coverage['row_count']:,} |
| 交易日数 | {coverage['trading_day_count']:,} |
| 每日241条记录的交易日 | {session['complete_241_record_days']:,}/{coverage['trading_day_count']:,} |
| 时间集合完全一致的交易日 | {session['exact_expected_time_set_days']:,}/{coverage['trading_day_count']:,} |
| 重复时间戳 | {coverage['duplicate_timestamp_rows']:,} |
| 零成交量记录 | {coverage['zero_volume_rows']:,} |
| 日线对账交易日 | {reconciliation['matched_days']:,} |
| OHLC最大绝对误差 | {reconciliation['maximum_ohlc_absolute_error']:.12f} 元 |
| 成交量最大相对误差 | {reconciliation['maximum_volume_relative_error']:.12%} |
| 成交额最大相对误差 | {reconciliation['maximum_amount_relative_error']:.12%} |
| 信号时段满足1500份/5%量比的记录占比 | {liquidity['maximum_block_eligible_record_coverage']:.4%} |
| 首日建仓后最多可买入的做T数量 | {affordability['initial_maximum_affordable_t_shares']:,} 份 |
| T+1底仓覆盖最大卖出块 | {'是' if affordability['base_inventory_covers_maximum_t_block'] else '否'} |

## 错误

{error_lines}

## 警告与限制

{warning_lines}

## 方法边界

- 原始字段仅标注为 `trade_time`，没有可靠声明它是分钟起点还是终点。因此信号只使用当前记录及以前的数据，成交统一放到下一条记录开盘价。
- 510300不是普通意义上的当日新买当日卖。回测必须通过2500份旧底仓完成“买新卖旧”或“先卖旧后买回”，且每天收盘恢复2500份。
- 零成交记录不会被填充或伪造成可成交记录；策略的5%量比规则会自动跳过流动性不足的信号。
- 本审计通过不代表策略能实现20%年化收益，只代表可以进入成本完整、无明显前视的历史回测。
"""


def main() -> int:
    config = _load_config()
    data_config = config["data"]
    account_config = config["account"]
    cost_config = config["costs"]
    entry_config = config["entry"]
    inventory_config = config["inventory_constraints"]

    paths = {
        "minute": _resolve(data_config["minute_file"]),
        "minute_metadata": _resolve(data_config["minute_metadata_file"]),
        "daily": _resolve(data_config["daily_file"]),
        "dividend": _resolve(data_config["dividend_file"]),
        "strategy_config": CONFIG_FILE,
    }
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for name, path in paths.items():
        if not path.exists():
            errors.append({"code": f"MISSING_{name.upper()}_FILE", "message": str(path)})
    if errors:
        report = {
            "status": "FAIL",
            "checked_at": datetime.now(ZoneInfo(data_config["timezone"])).isoformat(),
            "errors": errors,
            "warnings": warnings,
        }
        JSON_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        JSON_REPORT_FILE.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    minute = pd.read_parquet(paths["minute"])
    daily = pd.read_parquet(paths["daily"])
    dividends = pd.read_csv(paths["dividend"], encoding="utf-8-sig")
    metadata = json.loads(paths["minute_metadata"].read_text(encoding="utf-8"))

    minute_required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
    daily_required = {"date", "open", "high", "low", "close", "volume", "amount"}
    dividend_required = {
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
    }
    for label, actual, required in [
        ("MINUTE", set(minute.columns), minute_required),
        ("DAILY", set(daily.columns), daily_required),
        ("DIVIDEND", set(dividends.columns), dividend_required),
    ]:
        missing = sorted(required - actual)
        if missing:
            errors.append(
                {
                    "code": f"{label}_COLUMNS_MISSING",
                    "message": f"缺少字段：{', '.join(missing)}",
                }
            )
    if errors:
        report = {
            "status": "FAIL",
            "checked_at": datetime.now(ZoneInfo(data_config["timezone"])).isoformat(),
            "errors": errors,
            "warnings": warnings,
        }
        JSON_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        JSON_REPORT_FILE.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    minute = minute.copy()
    daily = daily.copy()
    minute["trade_time"] = pd.to_datetime(minute["trade_time"])
    minute["trade_date"] = minute["trade_time"].dt.normalize()
    daily["trade_date"] = pd.to_datetime(daily["date"]).dt.normalize()
    price_fields = ["open", "high", "low", "close"]

    duplicate_rows = int(minute.duplicated(["ts_code", "trade_time"], keep=False).sum())
    if duplicate_rows:
        errors.append(
            {"code": "DUPLICATE_MINUTE_TIMESTAMPS", "message": f"发现{duplicate_rows}条重复记录"}
        )
    if not minute["trade_time"].is_monotonic_increasing:
        errors.append({"code": "MINUTE_NOT_SORTED", "message": "一分钟数据未按时间升序排列"})
    if minute[list(minute_required)].isna().any().any():
        errors.append({"code": "MINUTE_NULL_VALUE", "message": "一分钟核心字段存在空值"})
    if daily[list(daily_required)].isna().any().any():
        errors.append({"code": "DAILY_NULL_VALUE", "message": "日线核心字段存在空值"})
    if minute["ts_code"].nunique() != 1 or minute["ts_code"].iloc[0] != config["strategy"]["symbol"]:
        errors.append({"code": "SYMBOL_MISMATCH", "message": "一分钟数据标的与策略注册标的不一致"})

    numeric_values = minute[price_fields + ["vol", "amount"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        errors.append({"code": "NON_FINITE_VALUE", "message": "一分钟数据存在无穷或非数值"})
    if minute[price_fields].le(0.0).any().any():
        errors.append({"code": "NON_POSITIVE_PRICE", "message": "价格字段存在非正数"})
    if minute[["vol", "amount"]].lt(0.0).any().any():
        errors.append({"code": "NEGATIVE_VOLUME_OR_AMOUNT", "message": "成交量或成交额存在负数"})

    invalid_ohlc = (
        minute["high"].lt(minute[["open", "close"]].max(axis=1) - 1e-12)
        | minute["low"].gt(minute[["open", "close"]].min(axis=1) + 1e-12)
        | minute["high"].lt(minute["low"] - 1e-12)
    )
    if invalid_ohlc.any():
        errors.append(
            {"code": "INVALID_OHLC", "message": f"发现{int(invalid_ohlc.sum())}条OHLC逻辑错误"}
        )

    tick = float(cost_config["price_tick_cny"])
    tick_error = np.abs(minute[price_fields].to_numpy(dtype=float) / tick - np.rint(
        minute[price_fields].to_numpy(dtype=float) / tick
    ))
    off_tick_count = int((tick_error > 1e-7).sum())
    if off_tick_count:
        errors.append(
            {"code": "PRICE_OFF_TICK", "message": f"发现{off_tick_count}个价格不符合{tick:.3f}元最小价位"}
        )

    zero_volume = minute["vol"].eq(0)
    zero_amount = minute["amount"].eq(0)
    inconsistent_zero = zero_volume.ne(zero_amount)
    if inconsistent_zero.any():
        errors.append(
            {
                "code": "VOLUME_AMOUNT_ZERO_MISMATCH",
                "message": f"发现{int(inconsistent_zero.sum())}条成交量与成交额零值不一致记录",
            }
        )
    if zero_volume.any():
        warnings.append(
            {
                "code": "ZERO_VOLUME_RECORDS",
                "message": f"共有{int(zero_volume.sum())}条零成交记录，策略必须跳过这些记录",
            }
        )

    expected_times = _expected_record_times()
    expected_time_set = set(expected_times)
    expected_rows = int(data_config["expected_rows_per_full_day"])
    counts = minute.groupby("trade_date", sort=True).size()
    complete_days = int(counts.eq(expected_rows).sum())
    time_set_mismatches: list[dict[str, Any]] = []
    for trade_date, frame in minute.groupby("trade_date", sort=True):
        actual_times = frame["trade_time"].dt.time.tolist()
        if actual_times != expected_times:
            actual_set = set(actual_times)
            time_set_mismatches.append(
                {
                    "trade_date": trade_date.date().isoformat(),
                    "row_count": len(actual_times),
                    "missing_times": sorted(value.isoformat() for value in expected_time_set - actual_set),
                    "extra_times": sorted(value.isoformat() for value in actual_set - expected_time_set),
                }
            )
    exact_time_days = int(minute["trade_date"].nunique() - len(time_set_mismatches))
    if complete_days != minute["trade_date"].nunique():
        errors.append(
            {
                "code": "INCOMPLETE_241_RECORD_DAY",
                "message": f"仅{complete_days}/{minute['trade_date'].nunique()}个交易日有{expected_rows}条记录",
            }
        )
    if time_set_mismatches:
        errors.append(
            {
                "code": "SESSION_TIME_SET_MISMATCH",
                "message": f"{len(time_set_mismatches)}个交易日的时间集合或顺序不一致",
            }
        )

    aggregate = minute.groupby("trade_date", as_index=False).agg(
        open_minute=("open", "first"),
        high_minute=("high", "max"),
        low_minute=("low", "min"),
        close_minute=("close", "last"),
        volume_minute=("vol", "sum"),
        amount_minute=("amount", "sum"),
    )
    reconciled = aggregate.merge(
        daily[["trade_date", "open", "high", "low", "close", "volume", "amount"]],
        on="trade_date",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched_days = int(reconciled["_merge"].ne("both").sum())
    if unmatched_days:
        errors.append(
            {"code": "MINUTE_DAILY_DATE_MISMATCH", "message": f"发现{unmatched_days}个无法一一对账的日期"}
        )
    matched = reconciled.loc[reconciled["_merge"].eq("both")].copy()
    ohlc_errors: dict[str, float] = {}
    for field in price_fields:
        difference = (matched[f"{field}_minute"] - matched[field]).abs()
        ohlc_errors[field] = float(difference.max()) if len(difference) else float("inf")
        if (difference > tick + 1e-12).any():
            errors.append(
                {
                    "code": f"DAILY_{field.upper()}_RECONCILIATION_FAILED",
                    "message": f"{field}存在超过一个最小价位的日线对账误差",
                }
            )
    volume_relative = _relative_error(matched["volume_minute"], matched["volume"])
    amount_relative = _relative_error(matched["amount_minute"], matched["amount"])
    reconciliation_tolerance = 1e-4
    if (volume_relative > reconciliation_tolerance).any():
        errors.append({"code": "DAILY_VOLUME_RECONCILIATION_FAILED", "message": "成交量日线对账误差超限"})
    if (amount_relative > reconciliation_tolerance).any():
        errors.append({"code": "DAILY_AMOUNT_RECONCILIATION_FAILED", "message": "成交额日线对账误差超限"})

    record_times = minute["trade_time"].dt.time
    signal_window = pd.Series(False, index=minute.index)
    for window in entry_config["signal_windows"]:
        start = time.fromisoformat(window["start"])
        end = time.fromisoformat(window["end"])
        signal_window |= record_times.between(start, end, inclusive="both")
    eligible_records = minute.loc[signal_window]
    maximum_t_shares = int(account_config["maximum_t_shares"])
    volume_ratio = float(entry_config["maximum_order_to_signal_record_volume_ratio"])
    required_signal_volume = math.ceil(maximum_t_shares / volume_ratio)
    liquidity_coverage = float(eligible_records["vol"].ge(required_signal_volume).mean())
    minimum_liquidity_coverage = float(entry_config["minimum_eligible_record_liquidity_coverage"])
    if liquidity_coverage < minimum_liquidity_coverage:
        errors.append(
            {
                "code": "INSUFFICIENT_SIGNAL_WINDOW_LIQUIDITY",
                "message": f"最大交易块的可执行记录覆盖率{liquidity_coverage:.2%}低于{minimum_liquidity_coverage:.2%}",
            }
        )

    first_open = float(minute.iloc[0]["open"])
    base_slippage = float(cost_config["base_slippage_bps_per_leg"]) / 10_000.0
    acquisition_price = _round_up_to_tick(first_open * (1.0 + base_slippage), tick)
    commission_rate = float(cost_config["commission_rate"])
    minimum_commission = float(cost_config["minimum_commission_cny"])
    base_shares = int(account_config["base_target_shares"])
    initial_cash = float(account_config["initial_cash_cny"])
    acquisition_notional = acquisition_price * base_shares
    acquisition_total = _buy_cost(acquisition_notional, commission_rate, minimum_commission)
    remaining_cash = initial_cash - acquisition_total
    affordable_t_shares = 0
    for shares in range(maximum_t_shares, 0, -int(account_config["lot_size"])):
        t_notional = acquisition_price * shares
        if _buy_cost(t_notional, commission_rate, minimum_commission) <= remaining_cash + 1e-9:
            affordable_t_shares = shares
            break
    minimum_t_shares = int(account_config["minimum_t_shares"])
    if acquisition_total > initial_cash:
        errors.append({"code": "CORE_ACQUISITION_UNAFFORDABLE", "message": "2万元无法完成2500份初始底仓建仓"})
    if affordable_t_shares < minimum_t_shares:
        errors.append(
            {
                "code": "INITIAL_BUY_FIRST_T_BLOCK_UNAFFORDABLE",
                "message": f"初始剩余现金仅支持{affordable_t_shares}份，低于最小做T数量{minimum_t_shares}份",
            }
        )
    inventory_covers_max = base_shares >= maximum_t_shares
    if not inventory_covers_max:
        errors.append({"code": "BASE_INVENTORY_TOO_SMALL", "message": "底仓无法覆盖最大先卖后买交易块"})
    if not bool(inventory_config["enforce_t_plus_one"]):
        errors.append({"code": "T_PLUS_ONE_NOT_ENFORCED", "message": "策略配置未强制T+1"})
    if inventory_config["end_of_day_required_shares"] != base_shares:
        errors.append({"code": "END_OF_DAY_INVENTORY_MISMATCH", "message": "日终目标份额与底仓份额不一致"})
    if data_config["earliest_execution"] != "NEXT_RECORD_OPEN":
        errors.append({"code": "SAME_RECORD_EXECUTION_RISK", "message": "策略未强制下一条记录开盘成交"})

    metadata_row_count = int(metadata.get("row_count", -1))
    metadata_hash = metadata.get("files", {}).get("parquet", {}).get("sha256")
    actual_minute_hash = _sha256_file(paths["minute"])
    if metadata_row_count != len(minute):
        errors.append({"code": "METADATA_ROW_COUNT_MISMATCH", "message": "元数据行数与Parquet不一致"})
    if metadata_hash != actual_minute_hash:
        errors.append({"code": "METADATA_HASH_MISMATCH", "message": "元数据哈希与Parquet不一致"})

    report: dict[str, Any] = {
        "status": "FAIL" if errors else "PASS",
        "scope": "510300底仓-VWAP日内均值回归V1数据与执行就绪审计",
        "contains_strategy_outcomes": False,
        "checked_at": datetime.now(ZoneInfo(data_config["timezone"])).isoformat(),
        "strategy_id": config["strategy"]["strategy_id"],
        "files": {
            name: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "coverage": {
            "row_count": int(len(minute)),
            "trading_day_count": int(minute["trade_date"].nunique()),
            "first_trade_time": minute["trade_time"].min().isoformat(),
            "last_trade_time": minute["trade_time"].max().isoformat(),
            "duplicate_timestamp_rows": duplicate_rows,
            "zero_volume_rows": int(zero_volume.sum()),
            "zero_volume_days": int(minute.loc[zero_volume, "trade_date"].nunique()),
            "off_tick_price_count": off_tick_count,
            "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        },
        "session_integrity": {
            "expected_rows_per_day": expected_rows,
            "complete_241_record_days": complete_days,
            "exact_expected_time_set_days": exact_time_days,
            "expected_first_time": expected_times[0].isoformat(),
            "expected_last_morning_time": time(11, 30).isoformat(),
            "expected_first_afternoon_time": time(13, 1).isoformat(),
            "expected_last_time": expected_times[-1].isoformat(),
            "mismatch_samples": time_set_mismatches[:10],
        },
        "daily_reconciliation": {
            "matched_days": int(len(matched)),
            "unmatched_days": unmatched_days,
            "ohlc_maximum_absolute_error_by_field": ohlc_errors,
            "maximum_ohlc_absolute_error": max(ohlc_errors.values()),
            "maximum_volume_relative_error": float(volume_relative.max()),
            "maximum_amount_relative_error": float(amount_relative.max()),
            "relative_tolerance": reconciliation_tolerance,
        },
        "liquidity": {
            "signal_window_record_count": int(len(eligible_records)),
            "maximum_t_shares": maximum_t_shares,
            "maximum_order_to_signal_record_volume_ratio": volume_ratio,
            "required_signal_record_volume_shares": required_signal_volume,
            "maximum_block_eligible_record_count": int(eligible_records["vol"].ge(required_signal_volume).sum()),
            "maximum_block_eligible_record_coverage": liquidity_coverage,
            "minimum_required_coverage": minimum_liquidity_coverage,
            "signal_window_volume_quantiles_shares": {
                str(key): float(value)
                for key, value in eligible_records["vol"].quantile([0.0, 0.01, 0.05, 0.5, 0.95]).items()
            },
        },
        "account_executability": {
            "initial_cash_cny": initial_cash,
            "first_record_raw_open_cny": first_open,
            "core_acquisition_execution_price_cny": acquisition_price,
            "core_acquisition_notional_cny": acquisition_notional,
            "core_acquisition_total_cost_cny": acquisition_total,
            "cash_after_core_acquisition_cny": remaining_cash,
            "initial_maximum_affordable_t_shares": affordable_t_shares,
            "minimum_t_shares": minimum_t_shares,
            "maximum_t_shares": maximum_t_shares,
            "base_inventory_covers_maximum_t_block": inventory_covers_max,
            "end_of_day_required_shares": int(inventory_config["end_of_day_required_shares"]),
            "same_day_new_shares_sellable": bool(inventory_config["same_day_new_shares_sellable"]),
        },
        "timestamp_semantics_guard": {
            "provider_semantics": data_config["timestamp_semantics"],
            "signal_observation": data_config["signal_observation"],
            "execution": data_config["earliest_execution"],
            "status": "PASS" if data_config["earliest_execution"] == "NEXT_RECORD_OPEN" else "FAIL",
        },
        "dividends": {
            "event_count": int(len(dividends)),
            "first_record_date": str(dividends["record_date"].min()),
            "last_payment_date": str(dividends["payment_date"].max()),
            "accounting": account_config["dividend_accounting"],
        },
        "errors": errors,
        "warnings": warnings,
    }
    report = _json_ready(report)
    JSON_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = JSON_REPORT_FILE.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(JSON_REPORT_FILE)
    temporary_markdown = MARKDOWN_REPORT_FILE.with_suffix(".md.tmp")
    temporary_markdown.write_text(_render_markdown(report), encoding="utf-8")
    temporary_markdown.replace(MARKDOWN_REPORT_FILE)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
