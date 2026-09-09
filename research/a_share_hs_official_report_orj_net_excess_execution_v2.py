"""ORJ V2 历史执行验证的冻结纯函数。

本模块只处理机械规则、成本、组合聚合与推断，不读取真实研究文件，
以便在冻结协议前用合成数据验证。V2 是已看过父研究结果后的非盲测验，
即使通过也不构成前瞻样本外确认或实盘授权。
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "config"
    / "a_share_hs_official_report_orj_net_excess_execution_v2.yaml"
)

RESOLVED_EXECUTION_STATUSES = {
    "FILLED_RESOLVED",
    "NO_FILL_ENTRY_SUSPENDED_CASH",
    "NO_FILL_ENTRY_UPPER_LIMIT_CASH",
}
NO_FILL_EXECUTION_STATUSES = {
    "NO_FILL_ENTRY_SUSPENDED_CASH",
    "NO_FILL_ENTRY_UPPER_LIMIT_CASH",
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取并验证 V2 配置的顶层合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "study",
        "parent",
        "periods",
        "inputs",
        "selection",
        "execution",
        "price_limits",
        "costs",
        "benchmarks",
        "inference",
        "classification",
        "not_acceptance_gates",
        "artifacts",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"ORJ V2 配置缺少顶层字段：{missing}")
    if payload["study"]["research_scope"] != (
        "HISTORICAL_EXECUTION_VALIDATION_NOT_BLIND"
    ):
        raise ValueError("ORJ V2 必须明确标记为非盲历史执行验证")
    if payload["selection"].get("future_label_field_for_selection_forbidden") != (
        "analysis_eligible"
    ):
        raise ValueError("ORJ V2 必须禁止用未来标签完整性筛选信号")
    return payload


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """计算 JSON 可序列化对象的稳定内容哈希。"""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    """拒绝字段不完整的数据帧。"""

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段：{missing}")


def finite_positive(value: Any) -> bool:
    """判断值是否为有限正数。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0.0)


def true_value(value: Any) -> bool:
    """安全解释可能含缺失值的布尔字段。"""

    if value is None or pd.isna(value):
        return False
    return bool(value)


def select_high_orj(
    signal_ledger: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """仅用起点可知字段，按事件日选择高 ORJ 前 20%。"""

    required = {
        "ts_code",
        "event_market_date",
        "event_market_day_index",
        "orj",
        "signal_eligible",
        "analysis_eligible",
    }
    require_columns(signal_ledger, required, name="ORJ 信号账本")
    work = signal_ledger.copy()
    work["event_market_date"] = pd.to_datetime(
        work["event_market_date"], errors="coerce"
    ).dt.normalize()
    periods = config["periods"]
    start = pd.Timestamp(periods["descriptive_start"])
    end = pd.Timestamp(periods["primary_end"])
    eligible = work.loc[
        work["signal_eligible"].fillna(False).astype(bool)
        & work["event_market_date"].between(start, end, inclusive="both")
    ].copy()
    if eligible.empty:
        return eligible.assign(
            event_date_candidate_count=pd.Series(dtype="int64"),
            selected_tail_count=pd.Series(dtype="int64"),
            high_orj_rank=pd.Series(dtype="int64"),
            evaluation_period_v2=pd.Series(dtype="object"),
            execution_event_id=pd.Series(dtype="object"),
        )
    numeric = pd.to_numeric(eligible["orj"], errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("signal_eligible=True 的记录存在非有限 ORJ")
    eligible["orj"] = numeric
    duplicates = eligible.duplicated(["ts_code", "event_market_date"], keep=False)
    if duplicates.any():
        examples = eligible.loc[
            duplicates, ["ts_code", "event_market_date"]
        ].head(10)
        raise ValueError(
            "修正后的父账本仍存在股票-事件日重复："
            + examples.to_dict(orient="records").__repr__()
        )

    selection = config["selection"]
    minimum = int(selection["minimum_candidates_per_event_date"])
    fraction = float(selection["high_orj_fraction"])
    pieces: list[pd.DataFrame] = []
    for _, group in eligible.groupby("event_market_date", sort=True, observed=True):
        size = len(group)
        if size < minimum:
            continue
        tail_count = int(np.floor(size * fraction))
        if tail_count < 1:
            continue
        ordered = group.sort_values(
            ["orj", "ts_code"],
            ascending=[False, True],
            kind="mergesort",
        ).head(tail_count).copy()
        ordered["event_date_candidate_count"] = int(size)
        ordered["selected_tail_count"] = int(tail_count)
        ordered["high_orj_rank"] = np.arange(1, tail_count + 1, dtype=int)
        pieces.append(ordered)
    if not pieces:
        return eligible.iloc[0:0].assign(
            event_date_candidate_count=pd.Series(dtype="int64"),
            selected_tail_count=pd.Series(dtype="int64"),
            high_orj_rank=pd.Series(dtype="int64"),
            evaluation_period_v2=pd.Series(dtype="object"),
            execution_event_id=pd.Series(dtype="object"),
        )
    selected = pd.concat(pieces, ignore_index=True)
    primary_start = pd.Timestamp(periods["primary_start"])
    primary_end = pd.Timestamp(periods["primary_end"])
    selected["evaluation_period_v2"] = np.where(
        selected["event_market_date"].between(
            primary_start, primary_end, inclusive="both"
        ),
        "PRIMARY_2021_2025",
        "DESCRIPTIVE_2016_2020",
    )
    selected["execution_event_id"] = (
        selected["ts_code"].astype(str)
        + "|"
        + selected["event_market_date"].dt.strftime("%Y-%m-%d")
    )
    return selected.sort_values(
        ["event_market_date", "high_orj_rank", "ts_code"],
        kind="mergesort",
    ).reset_index(drop=True)


def price_limit_fraction(
    ts_code: str,
    trade_date: Any,
    security_status: str,
    config: Mapping[str, Any],
) -> float:
    """按 2016—2025 沪深普通 A 股规则返回当日涨跌幅比例。"""

    code = str(ts_code).upper()
    date = pd.Timestamp(trade_date).normalize()
    status = str(security_status).upper()
    if status not in {"NORMAL", "ST"}:
        raise ValueError(f"无法为非正常交易状态计算价格限制：{status}")
    rules = config["price_limits"]
    local = code.split(".")[0]
    is_star = code.endswith(".SH") and local.startswith(("688", "689"))
    is_chinext = code.endswith(".SZ") and local.startswith(("300", "301"))
    if is_star:
        return float(rules["star_board_fraction"])
    if is_chinext and date >= pd.Timestamp("2020-08-24"):
        return float(rules["chinext_fraction_from_2020_08_24"])
    if is_chinext:
        key = (
            "chinext_fraction_before_2020_08_24_st"
            if status == "ST"
            else "chinext_fraction_before_2020_08_24_normal"
        )
        return float(rules[key])
    key = "main_board_st_fraction" if status == "ST" else "main_board_normal_fraction"
    return float(rules[key])


def rounded_limit_price(pre_close: float, fraction: float, *, upper: bool) -> float:
    """按 0.01 元和 ROUND_HALF_UP 计算涨停或跌停价。"""

    if not finite_positive(pre_close):
        return float("nan")
    multiplier = Decimal("1") + (
        Decimal(str(fraction)) if upper else -Decimal(str(fraction))
    )
    value = (Decimal(str(float(pre_close))) * multiplier).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return float(value)


def at_price_limit(
    observed_price: float,
    pre_close: float,
    fraction: float,
    *,
    upper: bool,
) -> bool:
    """判断观察价是否处于按规则四舍五入后的涨跌停价。"""

    if not finite_positive(observed_price) or not finite_positive(pre_close):
        return False
    limit = rounded_limit_price(pre_close, fraction, upper=upper)
    tolerance = 0.005000001
    if upper:
        return bool(float(observed_price) >= limit - tolerance)
    return bool(float(observed_price) <= limit + tolerance)


def transfer_fee_rate(trade_date: Any, config: Mapping[str, Any]) -> float:
    """返回交易日适用的双向过户费率。"""

    costs = config["costs"]
    if pd.Timestamp(trade_date).normalize() >= pd.Timestamp("2022-04-29"):
        return float(costs["transfer_fee_rate_each_side_from_2022_04_29"])
    return float(costs["transfer_fee_rate_each_side_before_2022_04_29"])


def sell_stamp_duty_rate(trade_date: Any, config: Mapping[str, Any]) -> float:
    """返回卖出日适用的证券交易印花税率。"""

    costs = config["costs"]
    if pd.Timestamp(trade_date).normalize() >= pd.Timestamp("2023-08-28"):
        return float(costs["sell_stamp_duty_from_2023_08_28"])
    return float(costs["sell_stamp_duty_before_2023_08_28"])


def calculate_filled_trade_economics(
    *,
    entry_raw_open: float,
    entry_total_return_open: float,
    exit_total_return_close: float,
    entry_date: Any,
    exit_date: Any,
    slippage_fraction_each_side: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """计算一手交易在公司行动总回报口径下的真实费用后经济结果。"""

    values = [entry_raw_open, entry_total_return_open, exit_total_return_close]
    if not all(finite_positive(value) for value in values):
        raise ValueError("成交经济计算要求有效的开盘原价与总回报价格")
    execution = config["execution"]
    costs = config["costs"]
    shares = int(execution["shares_per_signal"])
    commission_rate = float(costs["commission_rate_each_side"])
    commission_minimum = float(costs["commission_minimum_cny_each_order"])
    slippage = float(slippage_fraction_each_side)
    raw_entry_notional = shares * float(entry_raw_open)
    effective_buy_notional = raw_entry_notional * (1.0 + slippage)
    buy_commission = max(commission_minimum, effective_buy_notional * commission_rate)
    buy_transfer_fee = effective_buy_notional * transfer_fee_rate(entry_date, config)
    committed_capital = effective_buy_notional + buy_commission + buy_transfer_fee

    gross_terminal_value = raw_entry_notional * (
        float(exit_total_return_close) / float(entry_total_return_open)
    )
    effective_sell_notional = gross_terminal_value * (1.0 - slippage)
    sell_commission = max(
        commission_minimum, effective_sell_notional * commission_rate
    )
    sell_transfer_fee = effective_sell_notional * transfer_fee_rate(exit_date, config)
    stamp_duty = effective_sell_notional * sell_stamp_duty_rate(exit_date, config)
    final_cash = (
        effective_sell_notional
        - sell_commission
        - sell_transfer_fee
        - stamp_duty
    )
    net_pnl = final_cash - committed_capital
    net_return = final_cash / committed_capital - 1.0
    total_explicit_and_slippage_cost = (
        effective_buy_notional
        - raw_entry_notional
        + buy_commission
        + buy_transfer_fee
        + gross_terminal_value
        - effective_sell_notional
        + sell_commission
        + sell_transfer_fee
        + stamp_duty
    )
    return {
        "shares": float(shares),
        "raw_entry_notional": float(raw_entry_notional),
        "effective_buy_notional": float(effective_buy_notional),
        "buy_commission": float(buy_commission),
        "buy_transfer_fee": float(buy_transfer_fee),
        "committed_capital": float(committed_capital),
        "gross_terminal_value": float(gross_terminal_value),
        "effective_sell_notional": float(effective_sell_notional),
        "sell_commission": float(sell_commission),
        "sell_transfer_fee": float(sell_transfer_fee),
        "sell_stamp_duty": float(stamp_duty),
        "final_cash": float(final_cash),
        "net_pnl": float(net_pnl),
        "net_return": float(net_return),
        "total_explicit_and_slippage_cost": float(
            total_explicit_and_slippage_cost
        ),
    }


def classify_independent_execution(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """不考虑同股重叠时，机械判定成交、现金未成交或 NO_VIEW。"""

    shares = int(config["execution"]["shares_per_signal"])
    entry_raw_open = row.get("entry_raw_open")
    entry_pre_close = row.get("entry_pre_close")
    planned_price = (
        float(entry_raw_open)
        if finite_positive(entry_raw_open)
        else float(entry_pre_close)
        if finite_positive(entry_pre_close)
        else float("nan")
    )
    planned_notional = shares * planned_price if finite_positive(planned_price) else np.nan
    base = {
        "independent_execution_status": "",
        "planned_notional": float(planned_notional),
        "entry_filled": False,
        "execution_resolved_before_overlap": False,
        "entry_limit_fraction": np.nan,
        "entry_limit_price": np.nan,
        "exit_limit_fraction": np.nan,
        "exit_limit_price": np.nan,
    }
    if not true_value(row.get("entry_row_present")):
        base["independent_execution_status"] = "NO_VIEW_MISSING_ENTRY_MARKET_ROW"
        return base
    entry_status = row.get("entry_security_status")
    if entry_status is None or pd.isna(entry_status):
        base["independent_execution_status"] = "NO_VIEW_MISSING_ENTRY_SECURITY_STATUS"
        return base
    if str(entry_status).upper() not in {"NORMAL", "ST"}:
        base["independent_execution_status"] = "NO_VIEW_NONTRADABLE_ENTRY_STATUS"
        return base
    if true_value(row.get("entry_is_suspended")) or not true_value(
        row.get("entry_observed_traded_row")
    ):
        if finite_positive(planned_notional):
            base["independent_execution_status"] = (
                "NO_FILL_ENTRY_SUSPENDED_CASH"
            )
            base["execution_resolved_before_overlap"] = True
        else:
            base["independent_execution_status"] = (
                "NO_VIEW_ENTRY_SUSPENDED_WITHOUT_PLANNED_PRICE"
            )
        return base
    if not all(
        finite_positive(value)
        for value in (
            entry_raw_open,
            entry_pre_close,
            row.get("entry_total_return_open"),
        )
    ):
        base["independent_execution_status"] = "NO_VIEW_INVALID_ENTRY_PRICE"
        return base
    entry_fraction = price_limit_fraction(
        str(row.get("ts_code")),
        row.get("entry_date"),
        str(entry_status),
        config,
    )
    base["entry_limit_fraction"] = entry_fraction
    base["entry_limit_price"] = rounded_limit_price(
        float(entry_pre_close), entry_fraction, upper=True
    )
    if at_price_limit(
        float(entry_raw_open),
        float(entry_pre_close),
        entry_fraction,
        upper=True,
    ):
        base["independent_execution_status"] = "NO_FILL_ENTRY_UPPER_LIMIT_CASH"
        base["execution_resolved_before_overlap"] = True
        return base

    base["entry_filled"] = True
    if not true_value(row.get("exit_row_present")):
        base["independent_execution_status"] = "FILLED_UNRESOLVED_MISSING_EXIT_ROW"
        return base
    exit_status = row.get("exit_security_status")
    if exit_status is None or pd.isna(exit_status):
        base["independent_execution_status"] = (
            "FILLED_UNRESOLVED_MISSING_EXIT_SECURITY_STATUS"
        )
        return base
    if str(exit_status).upper() not in {"NORMAL", "ST"}:
        base["independent_execution_status"] = (
            "FILLED_UNRESOLVED_NONTRADABLE_EXIT_STATUS"
        )
        return base
    if true_value(row.get("exit_is_suspended")) or not true_value(
        row.get("exit_observed_traded_row")
    ):
        base["independent_execution_status"] = "FILLED_UNRESOLVED_EXIT_SUSPENDED"
        return base
    exit_raw_close = row.get("exit_raw_close")
    exit_pre_close = row.get("exit_pre_close")
    if not all(
        finite_positive(value)
        for value in (
            exit_raw_close,
            exit_pre_close,
            row.get("exit_total_return_close"),
        )
    ):
        base["independent_execution_status"] = "FILLED_UNRESOLVED_INVALID_EXIT_PRICE"
        return base
    exit_fraction = price_limit_fraction(
        str(row.get("ts_code")),
        row.get("exit_date"),
        str(exit_status),
        config,
    )
    base["exit_limit_fraction"] = exit_fraction
    base["exit_limit_price"] = rounded_limit_price(
        float(exit_pre_close), exit_fraction, upper=False
    )
    if at_price_limit(
        float(exit_raw_close),
        float(exit_pre_close),
        exit_fraction,
        upper=False,
    ):
        base["independent_execution_status"] = (
            "FILLED_UNRESOLVED_EXIT_LOWER_LIMIT"
        )
        return base
    base["independent_execution_status"] = "FILLED_RESOLVED"
    base["execution_resolved_before_overlap"] = True
    return base


def classify_execution_frame(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """为事件账本增加独立成交状态。"""

    required = {
        "execution_event_id",
        "ts_code",
        "entry_date",
        "exit_date",
        "entry_row_present",
        "exit_row_present",
    }
    require_columns(frame, required, name="ORJ 执行输入")
    records = [
        classify_independent_execution(row, config)
        for row in frame.to_dict(orient="records")
    ]
    classified = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1
    )
    return classified


def apply_same_security_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    """按一手持仓的实际区间跳过同股重叠信号。"""

    required = {
        "execution_event_id",
        "ts_code",
        "entry_market_day_index",
        "exit_market_day_index",
        "independent_execution_status",
    }
    require_columns(frame, required, name="ORJ 独立成交账本")
    work = frame.copy().reset_index(drop=True)
    work["_original_order"] = np.arange(len(work), dtype=int)
    work = work.sort_values(
        ["ts_code", "entry_market_day_index", "execution_event_id"],
        kind="mergesort",
    )
    work["execution_status"] = work["independent_execution_status"]
    work["allocation_included"] = True
    work["overlap_reference_event_id"] = None
    work["overlap_reference_exit_market_day_index"] = np.nan

    for _, indices in work.groupby("ts_code", sort=False).groups.items():
        active_until: int | None = None
        active_event: str | None = None
        unresolved_event: str | None = None
        for index in indices:
            entry_index = int(work.at[index, "entry_market_day_index"])
            if unresolved_event is not None:
                work.at[index, "execution_status"] = (
                    "SKIPPED_AFTER_UNRESOLVED_SAME_SECURITY_STATE"
                )
                work.at[index, "allocation_included"] = False
                work.at[index, "overlap_reference_event_id"] = unresolved_event
                continue
            if active_until is not None and entry_index <= active_until:
                work.at[index, "execution_status"] = (
                    "SKIPPED_SAME_SECURITY_ACTIVE_POSITION"
                )
                work.at[index, "allocation_included"] = False
                work.at[index, "overlap_reference_event_id"] = active_event
                work.at[index, "overlap_reference_exit_market_day_index"] = (
                    active_until
                )
                continue
            status = str(work.at[index, "independent_execution_status"])
            event_id = str(work.at[index, "execution_event_id"])
            if status == "FILLED_RESOLVED":
                active_until = int(work.at[index, "exit_market_day_index"])
                active_event = event_id
            elif status.startswith("NO_VIEW_") or status.startswith(
                "FILLED_UNRESOLVED_"
            ):
                unresolved_event = event_id
    return work.sort_values("_original_order").drop(columns="_original_order").reset_index(
        drop=True
    )


def attach_execution_economics(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """为重叠处理后的事件增加基础与压力成本经济结果。"""

    required = {
        "execution_status",
        "allocation_included",
        "planned_notional",
        "entry_raw_open",
        "entry_total_return_open",
        "exit_total_return_close",
        "entry_date",
        "exit_date",
    }
    require_columns(frame, required, name="ORJ 重叠处理账本")
    work = frame.copy()
    scenarios = config["costs"]["slippage_scenarios"]
    for scenario in scenarios:
        for field in (
            "committed_capital",
            "net_pnl",
            "net_return",
            "final_cash",
            "gross_terminal_value",
            "total_explicit_and_slippage_cost",
        ):
            work[f"{scenario}_{field}"] = np.nan
    work["execution_resolved"] = False
    for index, row in work.iterrows():
        if not true_value(row["allocation_included"]):
            continue
        status = str(row["execution_status"])
        if status in NO_FILL_EXECUTION_STATUSES:
            if not finite_positive(row["planned_notional"]):
                raise ValueError(f"现金未成交记录缺少计划本金：{row['execution_event_id']}")
            work.at[index, "execution_resolved"] = True
            for scenario in scenarios:
                capital = float(row["planned_notional"])
                work.at[index, f"{scenario}_committed_capital"] = capital
                work.at[index, f"{scenario}_net_pnl"] = 0.0
                work.at[index, f"{scenario}_net_return"] = 0.0
                work.at[index, f"{scenario}_final_cash"] = capital
                work.at[index, f"{scenario}_gross_terminal_value"] = capital
                work.at[
                    index, f"{scenario}_total_explicit_and_slippage_cost"
                ] = 0.0
            continue
        if status != "FILLED_RESOLVED":
            continue
        work.at[index, "execution_resolved"] = True
        for scenario, scenario_config in scenarios.items():
            economics = calculate_filled_trade_economics(
                entry_raw_open=float(row["entry_raw_open"]),
                entry_total_return_open=float(row["entry_total_return_open"]),
                exit_total_return_close=float(row["exit_total_return_close"]),
                entry_date=row["entry_date"],
                exit_date=row["exit_date"],
                slippage_fraction_each_side=float(
                    scenario_config["each_side_fraction"]
                ),
                config=config,
            )
            for field in (
                "committed_capital",
                "net_pnl",
                "net_return",
                "final_cash",
                "gross_terminal_value",
                "total_explicit_and_slippage_cost",
            ):
                work.at[index, f"{scenario}_{field}"] = economics[field]
    return work


def attach_benchmark_returns(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """机械构造开盘对齐的行业与沪深300全收益基准。"""

    required = {
        "industry_valid_path_rows",
        "industry_minimum_peer_count",
        "industry_log_return_sum_t1_t20",
        "industry_peer_log_return_t1_close_to_close",
        "industry_peer_log_return_t1_open_to_close",
        "industry_intraday_peer_count",
        "h00300_entry_close",
        "h00300_exit_close",
        "csi300_entry_open",
        "csi300_entry_close",
    }
    require_columns(frame, required, name="ORJ 基准输入")
    work = frame.copy()
    minimum_peers = int(
        config["benchmarks"]["industry"]["minimum_peers_each_day"]
    )
    work["industry_benchmark_status"] = "NO_VIEW_INCOMPLETE_INDUSTRY_PATH"
    work["industry_return"] = np.nan
    industry_valid = (
        pd.to_numeric(work["industry_valid_path_rows"], errors="coerce").eq(20)
        & pd.to_numeric(
            work["industry_minimum_peer_count"], errors="coerce"
        ).ge(minimum_peers)
        & pd.to_numeric(
            work["industry_intraday_peer_count"], errors="coerce"
        ).ge(minimum_peers)
    )
    industry_components = [
        "industry_log_return_sum_t1_t20",
        "industry_peer_log_return_t1_close_to_close",
        "industry_peer_log_return_t1_open_to_close",
    ]
    component_values = work[industry_components].apply(
        pd.to_numeric, errors="coerce"
    )
    industry_valid &= np.isfinite(component_values.to_numpy(dtype=float)).all(axis=1)
    aligned_log = (
        component_values["industry_log_return_sum_t1_t20"]
        - component_values["industry_peer_log_return_t1_close_to_close"]
        + component_values["industry_peer_log_return_t1_open_to_close"]
    )
    work.loc[industry_valid, "industry_return"] = np.expm1(
        aligned_log.loc[industry_valid]
    )
    work.loc[industry_valid, "industry_benchmark_status"] = (
        "PASS_ALIGNED_POINT_IN_TIME_INDUSTRY_BENCHMARK"
    )

    work["broad_market_benchmark_status"] = "NO_VIEW_INCOMPLETE_H00300_PATH"
    work["broad_market_return"] = np.nan
    broad_columns = [
        "h00300_entry_close",
        "h00300_exit_close",
        "csi300_entry_open",
        "csi300_entry_close",
    ]
    broad_values = work[broad_columns].apply(pd.to_numeric, errors="coerce")
    broad_valid = np.isfinite(broad_values.to_numpy(dtype=float)).all(axis=1) & (
        broad_values.to_numpy(dtype=float) > 0.0
    ).all(axis=1)
    broad_return = (
        broad_values["h00300_exit_close"]
        / broad_values["h00300_entry_close"]
        * broad_values["csi300_entry_close"]
        / broad_values["csi300_entry_open"]
        - 1.0
    )
    work.loc[broad_valid, "broad_market_return"] = broad_return.loc[broad_valid]
    work.loc[broad_valid, "broad_market_benchmark_status"] = (
        "PASS_H00300_TOTAL_RETURN_WITH_000300_ENTRY_INTRADAY_ALIGNMENT"
    )
    return work


def build_decision_date_metrics(
    event_outcomes: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """按决策日和已承诺本金聚合组合净收益与两个净超额。"""

    required = {
        "event_market_date",
        "evaluation_period_v2",
        "allocation_included",
        "execution_resolved",
        "execution_status",
        "industry_return",
        "broad_market_return",
    }
    require_columns(event_outcomes, required, name="ORJ 事件结果")
    scenarios = list(config["costs"]["slippage_scenarios"])
    benchmarks = {
        "industry": "industry_return",
        "broad_market": "broad_market_return",
    }
    date_coverage_required = float(
        config["inference"]["date_minimum_resolved_allocation_fraction"]
    )
    rows: list[dict[str, Any]] = []
    for event_date, raw_group in event_outcomes.groupby(
        "event_market_date", sort=True, observed=True
    ):
        group = raw_group.loc[raw_group["allocation_included"].fillna(False)].copy()
        allocation_count = len(group)
        if allocation_count == 0:
            continue
        period_values = group["evaluation_period_v2"].dropna().astype(str).unique()
        if len(period_values) != 1:
            raise ValueError(f"同一决策日出现多个评价区间：{event_date}")
        for scenario in scenarios:
            capital_column = f"{scenario}_committed_capital"
            pnl_column = f"{scenario}_net_pnl"
            return_column = f"{scenario}_net_return"
            require_columns(
                group,
                {capital_column, pnl_column, return_column},
                name=f"ORJ {scenario} 事件结果",
            )
            for benchmark, benchmark_column in benchmarks.items():
                capital = pd.to_numeric(group[capital_column], errors="coerce")
                net_pnl = pd.to_numeric(group[pnl_column], errors="coerce")
                strategy_return = pd.to_numeric(
                    group[return_column], errors="coerce"
                )
                benchmark_return = pd.to_numeric(
                    group[benchmark_column], errors="coerce"
                )
                resolved = (
                    group["execution_resolved"].fillna(False).astype(bool)
                    & np.isfinite(capital.to_numpy(dtype=float))
                    & capital.gt(0.0)
                    & np.isfinite(net_pnl.to_numpy(dtype=float))
                    & np.isfinite(strategy_return.to_numpy(dtype=float))
                    & np.isfinite(benchmark_return.to_numpy(dtype=float))
                    & benchmark_return.gt(-1.0)
                )
                resolved_count = int(resolved.sum())
                resolution_fraction = resolved_count / allocation_count
                eligible = bool(
                    resolution_fraction >= date_coverage_required
                    and resolved_count > 0
                )
                committed = float(capital.loc[resolved].sum()) if resolved_count else np.nan
                pnl = float(net_pnl.loc[resolved].sum()) if resolved_count else np.nan
                if eligible and finite_positive(committed):
                    portfolio_return = pnl / committed
                    weighted_benchmark_return = float(
                        np.average(
                            benchmark_return.loc[resolved].to_numpy(dtype=float),
                            weights=capital.loc[resolved].to_numpy(dtype=float),
                        )
                    )
                    net_excess = (
                        (1.0 + portfolio_return)
                        / (1.0 + weighted_benchmark_return)
                        - 1.0
                    )
                    event_excess = (
                        (1.0 + strategy_return.loc[resolved])
                        / (1.0 + benchmark_return.loc[resolved])
                        - 1.0
                    )
                    event_excess_win_rate = float((event_excess > 0.0).mean())
                else:
                    portfolio_return = np.nan
                    weighted_benchmark_return = np.nan
                    net_excess = np.nan
                    event_excess_win_rate = np.nan
                rows.append(
                    {
                        "event_market_date": pd.Timestamp(event_date).normalize(),
                        "year": int(pd.Timestamp(event_date).year),
                        "evaluation_period_v2": period_values[0],
                        "cost_scenario": scenario,
                        "benchmark": benchmark,
                        "selected_signal_count": int(len(raw_group)),
                        "allocation_count": int(allocation_count),
                        "resolved_allocation_count": resolved_count,
                        "resolution_fraction": float(resolution_fraction),
                        "filled_resolved_count": int(
                            group["execution_status"].eq("FILLED_RESOLVED").sum()
                        ),
                        "no_fill_cash_count": int(
                            group["execution_status"].isin(
                                NO_FILL_EXECUTION_STATUSES
                            ).sum()
                        ),
                        "no_view_count": int(
                            group["execution_status"].str.startswith(
                                ("NO_VIEW_", "FILLED_UNRESOLVED_"), na=False
                            ).sum()
                        ),
                        "decision_date_eligible": eligible,
                        "resolved_committed_capital_cny": committed,
                        "resolved_net_pnl_cny": pnl,
                        "portfolio_net_return": float(portfolio_return),
                        "benchmark_return": float(weighted_benchmark_return),
                        "net_excess_return": float(net_excess),
                        "event_excess_win_rate_descriptive": float(
                            event_excess_win_rate
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows).sort_values(
        ["event_market_date", "cost_scenario", "benchmark"], kind="mergesort"
    ).reset_index(drop=True)


def moving_block_bootstrap_mean(
    values: pd.Series,
    *,
    repetitions: int,
    block_length: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """在有序决策日上对均值执行确定性移动块 Bootstrap。"""

    series = values.dropna().sort_index()
    array = series.to_numpy(dtype=float)
    size = len(array)
    if size < block_length:
        return {
            "status": "NO_VIEW_INSUFFICIENT_DECISION_DATES",
            "decision_dates": size,
            "point_estimate": float(array.mean()) if size else np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "repetitions": int(repetitions),
            "block_length": int(block_length),
            "seed": int(seed),
            "confidence_level": float(confidence_level),
        }
    starts = np.arange(size - block_length + 1, dtype=int)
    blocks_needed = int(np.ceil(size / block_length))
    offsets = np.arange(block_length, dtype=int)
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts_chosen = generator.choice(starts, size=blocks_needed, replace=True)
        indices = (starts_chosen[:, None] + offsets[None, :]).reshape(-1)[:size]
        estimates[repetition] = array[indices].mean()
    alpha = 1.0 - confidence_level
    return {
        "status": "PASS_BOOTSTRAP_COMPLETED",
        "decision_dates": int(size),
        "point_estimate": float(array.mean()),
        "ci_lower": float(np.quantile(estimates, alpha / 2.0)),
        "ci_upper": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "repetitions": int(repetitions),
        "block_length": int(block_length),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
    }


def build_annual_summary(decision_date_metrics: pd.DataFrame) -> pd.DataFrame:
    """生成完整年度的决策日等权描述统计。"""

    required = {
        "year",
        "evaluation_period_v2",
        "cost_scenario",
        "benchmark",
        "decision_date_eligible",
        "net_excess_return",
        "portfolio_net_return",
        "benchmark_return",
    }
    require_columns(decision_date_metrics, required, name="ORJ 决策日指标")
    eligible = decision_date_metrics.loc[
        decision_date_metrics["decision_date_eligible"].fillna(False)
    ].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in eligible.groupby(
        ["year", "evaluation_period_v2", "cost_scenario", "benchmark"],
        sort=True,
        observed=True,
    ):
        excess = pd.to_numeric(group["net_excess_return"], errors="coerce").dropna()
        rows.append(
            {
                "year": int(keys[0]),
                "evaluation_period_v2": keys[1],
                "cost_scenario": keys[2],
                "benchmark": keys[3],
                "decision_dates": int(len(excess)),
                "mean_net_excess_return": float(excess.mean()),
                "median_net_excess_return": float(excess.median()),
                "positive_decision_date_fraction": float((excess > 0.0).mean()),
                "mean_portfolio_net_return": float(
                    pd.to_numeric(
                        group["portfolio_net_return"], errors="coerce"
                    ).mean()
                ),
                "mean_benchmark_return": float(
                    pd.to_numeric(group["benchmark_return"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _path_inference(
    decision_date_metrics: pd.DataFrame,
    *,
    scenario: str,
    benchmark: str,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """汇总单个成本-基准路径的覆盖、Bootstrap、两半与年度结果。"""

    inference = config["inference"]
    primary = decision_date_metrics.loc[
        decision_date_metrics["evaluation_period_v2"].eq("PRIMARY_2021_2025")
        & decision_date_metrics["cost_scenario"].eq(scenario)
        & decision_date_metrics["benchmark"].eq(benchmark)
    ].copy()
    allocations = int(primary["allocation_count"].sum())
    resolved = int(primary["resolved_allocation_count"].sum())
    overall_resolution_fraction = resolved / allocations if allocations else 0.0
    eligible = primary.loc[
        primary["decision_date_eligible"].fillna(False)
        & pd.to_numeric(primary["net_excess_return"], errors="coerce").notna()
    ].sort_values("event_market_date", kind="mergesort")
    series = pd.Series(
        eligible["net_excess_return"].to_numpy(dtype=float),
        index=pd.to_datetime(eligible["event_market_date"]),
    )
    bootstrap = moving_block_bootstrap_mean(
        series,
        repetitions=int(inference["bootstrap_repetitions"]),
        block_length=int(inference["bootstrap_block_length_dates"]),
        seed=int(seed),
        confidence_level=float(inference["confidence_level"]),
    )
    count = len(eligible)
    earlier_count = int(np.ceil(count / 2.0))
    earlier_mean = (
        float(eligible.iloc[:earlier_count]["net_excess_return"].mean())
        if earlier_count
        else np.nan
    )
    later_mean = (
        float(eligible.iloc[earlier_count:]["net_excess_return"].mean())
        if count - earlier_count
        else np.nan
    )
    configured_years = [int(year) for year in config["periods"]["primary_complete_years"]]
    annual_means = (
        eligible.groupby("year", observed=True)["net_excess_return"].mean().to_dict()
    )
    annual_means = {
        str(year): float(annual_means[year])
        for year in configured_years
        if year in annual_means
    }
    positive_years = sum(value > 0.0 for value in annual_means.values())
    positive_year_fraction = (
        positive_years / len(configured_years) if configured_years else np.nan
    )
    evidence_checks = {
        "overall_resolution_fraction_at_least_required": bool(
            overall_resolution_fraction
            >= float(inference["overall_minimum_resolved_allocation_fraction"])
        ),
        "minimum_primary_decision_dates_met": bool(
            count >= int(inference["minimum_primary_decision_dates"])
        ),
        "all_complete_years_observed": bool(
            len(annual_means) == len(configured_years)
        ),
        "bootstrap_completed": bootstrap["status"] == "PASS_BOOTSTRAP_COMPLETED",
    }
    stability_checks = {
        "mean_strictly_positive": bool(
            np.isfinite(bootstrap["point_estimate"])
            and bootstrap["point_estimate"] > 0.0
        ),
        "bootstrap_lower_strictly_positive": bool(
            np.isfinite(bootstrap["ci_lower"]) and bootstrap["ci_lower"] > 0.0
        ),
        "earlier_half_strictly_positive": bool(
            np.isfinite(earlier_mean) and earlier_mean > 0.0
        ),
        "later_half_strictly_positive": bool(
            np.isfinite(later_mean) and later_mean > 0.0
        ),
        "positive_complete_year_fraction_met": bool(
            np.isfinite(positive_year_fraction)
            and positive_year_fraction
            >= float(inference["minimum_positive_complete_year_fraction"])
        ),
    }
    evidence_pass = all(evidence_checks.values())
    economic_pass = evidence_pass and all(stability_checks.values())
    if economic_pass:
        status = "PASS_HISTORICAL_NET_EXCESS_PATH_NOT_BLIND"
    elif evidence_pass:
        status = "REJECT_HISTORICAL_NET_EXCESS_PATH"
    else:
        status = "NO_VIEW_INSUFFICIENT_EXECUTION_EVIDENCE"
    return {
        "status": status,
        "cost_scenario": scenario,
        "benchmark": benchmark,
        "selected_allocations": allocations,
        "resolved_allocations": resolved,
        "overall_resolution_fraction": float(overall_resolution_fraction),
        "eligible_primary_decision_dates": int(count),
        "bootstrap": bootstrap,
        "chronological_halves": {
            "earlier_decision_dates": int(earlier_count),
            "later_decision_dates": int(count - earlier_count),
            "earlier_mean_net_excess_return": float(earlier_mean),
            "later_mean_net_excess_return": float(later_mean),
        },
        "annual_mean_net_excess_returns": annual_means,
        "positive_complete_years": int(positive_years),
        "positive_complete_year_fraction": float(positive_year_fraction),
        "evidence_checks": evidence_checks,
        "stability_checks": stability_checks,
        "evidence_pass": bool(evidence_pass),
        "economic_pass": bool(economic_pass),
    }


def evaluate_gate(
    decision_date_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """按冻结规则分类历史 Alpha 或强 Beta 候选。"""

    seed = int(config["inference"]["base_seed"])
    descriptive: dict[str, Any] = {}
    offsets = {
        ("base", "industry"): 0,
        ("base", "broad_market"): 1,
        ("stress", "industry"): 10,
        ("stress", "broad_market"): 11,
    }
    for scenario in config["costs"]["slippage_scenarios"]:
        for benchmark in ("industry", "broad_market"):
            key = f"{scenario}__{benchmark}"
            descriptive[key] = _path_inference(
                decision_date_metrics,
                scenario=scenario,
                benchmark=benchmark,
                seed=seed + offsets[(scenario, benchmark)],
                config=config,
            )
    alpha_path = descriptive["stress__industry"]
    strong_beta_path = descriptive["stress__broad_market"]
    alpha_pass = bool(alpha_path["economic_pass"])
    strong_beta_pass = bool(strong_beta_path["economic_pass"])
    if alpha_pass and strong_beta_pass:
        historical_status = (
            "HISTORICAL_ALPHA_AND_STRONG_BETA_CANDIDATE_NOT_BLIND"
        )
    elif alpha_pass:
        historical_status = "HISTORICAL_ALPHA_CANDIDATE_NOT_BLIND"
    elif strong_beta_pass:
        historical_status = "HISTORICAL_STRONG_BETA_CANDIDATE_NOT_BLIND"
    elif alpha_path["evidence_pass"] and strong_beta_path["evidence_pass"]:
        historical_status = "REJECTED_ORJ_NET_EXCESS_EXECUTION_V2"
    else:
        historical_status = "NO_VIEW_INSUFFICIENT_ORJ_EXECUTION_EVIDENCE"
    return {
        "study_id": config["study"]["study_id"],
        "historical_status": historical_status,
        "historical_alpha_candidate": alpha_pass,
        "historical_strong_beta_candidate": strong_beta_pass,
        "stress_industry_path": alpha_path,
        "stress_broad_market_path": strong_beta_path,
        "all_cost_benchmark_paths": descriptive,
        "not_acceptance_gates": list(config["not_acceptance_gates"]),
        "classification_boundary": {
            "not_blind": True,
            "not_forward_oos_confirmed": True,
            "not_shadow_authorized": True,
            "not_live_authorized": True,
        },
    }


def descriptive_trade_statistics(
    event_outcomes: pd.DataFrame,
    *,
    scenario: str,
    benchmark_column: str,
    primary_only: bool = True,
) -> dict[str, Any]:
    """计算不参与门控的胜率、盈亏比、利润因子和集中度。"""

    frame = event_outcomes.copy()
    if primary_only:
        frame = frame.loc[frame["evaluation_period_v2"].eq("PRIMARY_2021_2025")]
    strategy_return = pd.to_numeric(
        frame[f"{scenario}_net_return"], errors="coerce"
    )
    benchmark_return = pd.to_numeric(frame[benchmark_column], errors="coerce")
    resolved = (
        frame["allocation_included"].fillna(False)
        & frame["execution_resolved"].fillna(False)
        & strategy_return.notna()
        & benchmark_return.notna()
        & benchmark_return.gt(-1.0)
    )
    excess = (
        (1.0 + strategy_return.loc[resolved])
        / (1.0 + benchmark_return.loc[resolved])
        - 1.0
    )
    wins = excess.loc[excess > 0.0]
    losses = excess.loc[excess < 0.0]
    mean_win = float(wins.mean()) if len(wins) else np.nan
    mean_loss_abs = float(-losses.mean()) if len(losses) else np.nan
    payoff = (
        mean_win / mean_loss_abs
        if finite_positive(mean_loss_abs) and np.isfinite(mean_win)
        else np.nan
    )
    profit_factor = (
        float(wins.sum() / -losses.sum())
        if len(losses) and float(losses.sum()) < 0.0
        else np.nan
    )
    return {
        "resolved_events": int(len(excess)),
        "positive_excess_events": int((excess > 0.0).sum()),
        "negative_excess_events": int((excess < 0.0).sum()),
        "event_excess_win_rate": float((excess > 0.0).mean())
        if len(excess)
        else np.nan,
        "mean_positive_event_excess": mean_win,
        "mean_negative_event_excess_abs": mean_loss_abs,
        "payoff_ratio": float(payoff),
        "profit_factor": float(profit_factor),
        "mean_event_excess": float(excess.mean()) if len(excess) else np.nan,
        "median_event_excess": float(excess.median()) if len(excess) else np.nan,
        "explicitly_not_an_acceptance_gate": True,
    }


def json_ready(value: Any) -> Any:
    """把 NumPy、时间和非有限值转换为严格 JSON 可写对象。"""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
