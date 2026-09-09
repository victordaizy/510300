"""510300全市场换手率可见度V1的冻结研究实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import NormalDist
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)


CONFIG_PATH = ROOT / "config" / "510300_aggregate_turnover_visibility_v1.yaml"
MANIFEST_PATH = (
    ROOT / "config" / "510300_aggregate_turnover_visibility_v1_manifest.json"
)


class ContractError(RuntimeError):
    """冻结协议、输入数据或点时约束不成立。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        ("protocol", "project_id"): "510300_AGGREGATE_TURNOVER_VISIBILITY_V1",
        ("scope", "execution_asset"): "510300.SH",
        ("dates", "evaluation_start"): "2015-01-05",
        ("dates", "evaluation_end"): "2026-08-14",
        ("dates", "factor_warmup_start"): "2014-12-01",
        ("dates", "factor_last_complete_month_end"): "2026-07-31",
        ("data_contract", "expected_trading_dates"): 2836,
        ("data_contract", "expected_complete_months"): 140,
        ("data_contract", "historical_availability_status"): (
            "HISTORICAL_ARCHIVE_RECONSTRUCTION_NOT_TIMESTAMP_PROOF"
        ),
        ("factor", "expected_predictive_direction"): "positive",
        ("external_allocation", "intercept"): -0.019,
        ("external_allocation", "turnover_coefficient"): 0.258,
        ("external_allocation", "risk_aversion_gamma"): 5.0,
        ("external_allocation", "fixed_monthly_standard_deviation"): 0.0901,
        ("external_allocation", "lower_weight_bound"): 0.0,
        ("external_allocation", "upper_weight_bound"): 1.0,
        ("portfolio", "initial_capital_cny"): 20000.0,
        ("portfolio", "lot_size_shares"): 100,
        ("portfolio", "cash_annual_rate"): 0.0,
        ("portfolio", "base_costs", "commission_rate_per_leg"): 0.0003,
        ("portfolio", "base_costs", "minimum_commission_cny_per_leg"): 5.0,
        ("portfolio", "base_costs", "slippage_bps_per_leg"): 5.0,
        ("portfolio", "stress_costs", "commission_rate_per_leg"): 0.0003,
        ("portfolio", "stress_costs", "minimum_commission_cny_per_leg"): 5.0,
        ("portfolio", "stress_costs", "slippage_bps_per_leg"): 10.0,
        ("portfolio_gates", "base_net_sharpe_minimum"): 1.20,
    }
    for keys, value in expected.items():
        observed: Any = config
        for key in keys:
            observed = observed[key]
        if observed != value:
            raise ContractError(f"冻结字段{'.'.join(keys)}异常：{observed!r}")
    if config["scope"]["allowed_holdings"] != ["510300.SH", "CASH_CNY"]:
        raise ContractError("只允许持有510300.SH或人民币现金")
    forbidden_true = [
        config["scope"]["leverage_allowed"],
        config["scope"]["short_selling_allowed"],
        config["scope"]["derivatives_execution_allowed"],
        config["scope"]["live_trading_authorized"],
        config["governance"]["order_generation"],
        config["governance"]["broker_connection"],
        config["governance"]["position_change"],
        config["governance"]["live_trading_authorized"],
    ]
    if any(forbidden_true):
        raise ContractError("研究边界不得授权杠杆、卖空、衍生品或实盘执行")
    if config["protocol"]["one_shot"] is not True:
        raise ContractError("候选必须是一次性冻结检验")
    if config["protocol"]["parameter_rescue_after_result"] != "forbidden":
        raise ContractError("结果后参数救援必须被禁止")
    if config["protocol"]["alternate_szse_scope_rescue_after_result"] != "forbidden":
        raise ContractError("深市代理口径不得在结果后替换")
    if config["data_contract"]["tls_certificate_verification"] != "required":
        raise ContractError("官方数据传输必须启用TLS证书校验")
    if config["data_contract"]["szse"]["transport"] != (
        "HTTPS_ONLY_TLS_VERIFICATION_REQUIRED"
    ):
        raise ContractError("深交所正式数据源不得降级到HTTP")
    if config["protocol"]["historical_archive_decision"]["status"] != (
        "HISTORICAL_ARCHIVE_RECONSTRUCTION_NOT_TIMESTAMP_PROOF"
    ):
        raise ContractError("历史归档的可得性限制未被锁定")
    variance = float(config["external_allocation"]["fixed_monthly_variance"])
    standard_deviation = float(
        config["external_allocation"]["fixed_monthly_standard_deviation"]
    )
    if not math.isclose(variance, standard_deviation**2, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("论文固定月度方差与标准差不一致")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError(f"冻结清单不存在：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("冻结清单状态不是FROZEN_BEFORE_FIRST_RESULT")
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("冻结清单项目标识不一致")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("冻结后配置哈希发生变化")
    for relative, expected_hash in manifest.get("tracked_files", {}).items():
        path = _project_path(relative)
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ContractError(f"冻结实现文件发生变化：{relative}")
    for relative, expected_hash in manifest.get("input_files", {}).items():
        path = _project_path(relative)
        if not path.exists() or sha256_file(path) != expected_hash:
            raise ContractError(f"冻结输入文件发生变化：{relative}")
    if manifest.get("candidate_outcomes_read_before_freeze") is not False:
        raise ContractError("冻结清单未证明结果在冻结前不可见")
    if manifest.get("portfolio_returns_read_before_freeze") is not False:
        raise ContractError("冻结清单未证明组合收益在冻结前不可见")
    return manifest


def _strip_jsonp(payload: bytes | str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    opening = text.find("(")
    closing = text.rfind(")")
    if opening < 1 or closing <= opening:
        raise ContractError("上交所响应不是有效JSON或JSONP")
    return json.loads(text[opening + 1 : closing])


def _number(value: Any, *, label: str) -> float:
    if value is None:
        raise ContractError(f"{label}为空")
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--"}:
        raise ContractError(f"{label}为空或占位符")
    parsed = float(text)
    if not math.isfinite(parsed):
        raise ContractError(f"{label}不是有限数")
    return parsed


def _sum_selected_rows(
    rows: Iterable[dict[str, Any]],
    *,
    key_name: str,
    selected: set[str],
    trade_field: str,
    market_cap_field: str,
    label: str,
) -> tuple[float, float, list[str]]:
    selected_rows = [row for row in rows if str(row.get(key_name)) in selected]
    observed = sorted(str(row.get(key_name)) for row in selected_rows)
    required_present = selected.intersection(observed)
    if not required_present:
        raise ContractError(f"{label}没有任何指定A股类别")
    trade_amount = sum(
        _number(row.get(trade_field), label=f"{label}.{trade_field}")
        for row in selected_rows
    )
    market_cap = sum(
        _number(row.get(market_cap_field), label=f"{label}.{market_cap_field}")
        for row in selected_rows
    )
    if trade_amount <= 0 or market_cap <= 0:
        raise ContractError(f"{label}成交额或总市值不是正数")
    return trade_amount, market_cap, observed


def _collapse_identical_category_rows(
    rows: Iterable[dict[str, Any]],
    *,
    key_name: str,
    selected: set[str],
    value_fields: tuple[str, ...],
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    """折叠官方源中同类别且关键数值逐字相同的重复行。"""

    selected_rows = [row for row in rows if str(row.get(key_name)) in selected]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault(str(row.get(key_name)), []).append(row)
    collapsed: list[dict[str, Any]] = []
    duplicate_count = 0
    for category, category_rows in sorted(grouped.items()):
        signatures = {
            tuple(str(row.get(field)) for field in value_fields)
            for row in category_rows
        }
        if len(signatures) != 1:
            raise ContractError(f"{label}类别{category}存在数值不一致的重复行")
        collapsed.append(category_rows[0])
        duplicate_count += len(category_rows) - 1
    return collapsed, duplicate_count


def parse_sse_legacy_daily(
    payload: bytes | str,
    expected_date: str,
    selected_product_types: Iterable[str] = ("1", "48"),
) -> dict[str, Any]:
    document = _strip_jsonp(payload)
    rows = list(document.get("result") or [])
    if not rows:
        raise ContractError(f"上交所旧日度接口{expected_date}无数据")
    returned_dates = {
        str(row.get("CAL_DATE", ""))[:10] for row in rows if row.get("CAL_DATE")
    }
    if returned_dates != {expected_date}:
        raise ContractError(
            f"上交所旧日度接口日期不一致：期望{expected_date}，得到{returned_dates}"
        )
    trade, market_cap, observed = _sum_selected_rows(
        rows,
        key_name="PRODUCT_TYPE",
        selected=set(selected_product_types),
        trade_field="TX_AMOUNT_FULL",
        market_cap_field="MKT_VALUE_FULL",
        label=f"SSE_LEGACY_DAILY_{expected_date}",
    )
    return {
        "date": expected_date,
        "trade_amount_cny_100m": trade,
        "market_cap_cny_100m": market_cap,
        "selected_categories_observed": observed,
    }


def parse_sse_current_daily(
    payload: bytes | str,
    expected_date: str,
    selected_product_codes: Iterable[str] = ("01", "03"),
) -> dict[str, Any]:
    document = _strip_jsonp(payload)
    rows = list(document.get("result") or [])
    if not rows:
        raise ContractError(f"上交所新日度接口{expected_date}无数据")
    expected_compact = expected_date.replace("-", "")
    returned_dates = {
        str(row.get("TRADE_DATE", ""))[:8]
        for row in rows
        if row.get("TRADE_DATE")
    }
    if returned_dates != {expected_compact}:
        raise ContractError(
            f"上交所新日度接口日期不一致：期望{expected_compact}，得到{returned_dates}"
        )
    trade, market_cap, observed = _sum_selected_rows(
        rows,
        key_name="PRODUCT_CODE",
        selected=set(selected_product_codes),
        trade_field="TRADE_AMT",
        market_cap_field="TOTAL_VALUE",
        label=f"SSE_CURRENT_DAILY_{expected_date}",
    )
    return {
        "date": expected_date,
        "trade_amount_cny_100m": trade,
        "market_cap_cny_100m": market_cap,
        "selected_categories_observed": observed,
    }


def _szse_tab(document: Any, tab_key: str) -> dict[str, Any]:
    if not isinstance(document, list):
        raise ContractError("深交所响应顶层不是数组")
    matches = [
        item
        for item in document
        if str((item.get("metadata") or {}).get("tabkey")) == tab_key
    ]
    if len(matches) != 1:
        raise ContractError(f"深交所响应中{tab_key}数量不是1")
    return matches[0]


def parse_szse_daily(
    payload: bytes | str,
    expected_date: str,
    tab_key: str = "tab1",
) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    item = _szse_tab(json.loads(text), tab_key)
    conditions = list((item.get("metadata") or {}).get("conditions") or [])
    date_conditions = [row for row in conditions if row.get("name") == "txtQueryDate"]
    if len(date_conditions) != 1 or date_conditions[0].get("defaultValue") != expected_date:
        raise ContractError(f"深交所日度接口日期未回显为{expected_date}")
    rows = list(item.get("data") or [])
    by_name = {str(row.get("zbmc")): row.get("brsz") for row in rows}
    required = {"股票总市值（亿元）", "股票成交金额（亿元）"}
    if missing := required.difference(by_name):
        raise ContractError(f"深交所日度接口缺少指标：{sorted(missing)}")
    trade = _number(by_name["股票成交金额（亿元）"], label="SZSE_DAILY_TRADE")
    market_cap = _number(by_name["股票总市值（亿元）"], label="SZSE_DAILY_MCAP")
    if trade <= 0 or market_cap <= 0:
        raise ContractError("深交所日度成交额或总市值不是正数")
    return {
        "date": expected_date,
        "trade_amount_cny_100m": trade,
        "market_cap_cny_100m": market_cap,
    }


def parse_sse_legacy_monthly(
    payload: bytes | str,
    expected_month: str,
    selected_product_types: Iterable[str] = ("1", "48"),
) -> dict[str, Any]:
    document = _strip_jsonp(payload)
    rows = list(document.get("result") or [])
    if not rows:
        raise ContractError(f"上交所旧月度接口{expected_month}无数据")
    selected = set(selected_product_types)
    collapsed, duplicate_count = _collapse_identical_category_rows(
        rows,
        key_name="PRODUCT_TYPE",
        selected=selected,
        value_fields=("TX_AMOUNT", "MKT_VALUE"),
        label=f"SSE_LEGACY_MONTHLY_{expected_month}",
    )
    trade, market_cap, observed = _sum_selected_rows(
        collapsed,
        key_name="PRODUCT_TYPE",
        selected=selected,
        trade_field="TX_AMOUNT",
        market_cap_field="MKT_VALUE",
        label=f"SSE_LEGACY_MONTHLY_{expected_month}",
    )
    return {
        "factor_month": expected_month,
        "trade_amount_cny_100m": trade,
        "market_cap_cny_100m": market_cap,
        "selected_categories_observed": observed,
        "identical_duplicate_rows_collapsed": duplicate_count,
    }


def parse_sse_current_monthly(
    payload: bytes | str,
    expected_month: str,
    selected_product_codes: Iterable[str] = ("01", "03"),
) -> dict[str, Any]:
    document = _strip_jsonp(payload)
    rows = list(document.get("result") or [])
    if not rows:
        raise ContractError(f"上交所新月度接口{expected_month}无数据")
    selected = set(selected_product_codes)
    collapsed, duplicate_count = _collapse_identical_category_rows(
        rows,
        key_name="PRODUCT_CODE",
        selected=selected,
        value_fields=("TRADE_AMT", "TOTAL_VALUE"),
        label=f"SSE_CURRENT_MONTHLY_{expected_month}",
    )
    trade, market_cap, observed = _sum_selected_rows(
        collapsed,
        key_name="PRODUCT_CODE",
        selected=selected,
        trade_field="TRADE_AMT",
        market_cap_field="TOTAL_VALUE",
        label=f"SSE_CURRENT_MONTHLY_{expected_month}",
    )
    return {
        "factor_month": expected_month,
        "trade_amount_cny_100m": trade,
        "market_cap_cny_100m": market_cap,
        "selected_categories_observed": observed,
        "identical_duplicate_rows_collapsed": duplicate_count,
    }


def parse_szse_monthly(payload: bytes | str, expected_month: str) -> dict[str, Any]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    item = _szse_tab(json.loads(text), "tab1")
    conditions = list((item.get("metadata") or {}).get("conditions") or [])
    date_conditions = [row for row in conditions if row.get("name") == "txtQueryDate"]
    if len(date_conditions) != 1 or date_conditions[0].get("defaultValue") != expected_month:
        raise ContractError(f"深交所月度接口月份未回显为{expected_month}")
    rows = list(item.get("data") or [])
    matches = [row for row in rows if row.get("zbmc") == "成交金额（亿元）"]
    if len(matches) != 1:
        raise ContractError("深交所月度接口缺少唯一成交金额行")
    trade = _number(matches[0].get("gp"), label="SZSE_MONTHLY_TRADE")
    if trade <= 0:
        raise ContractError("深交所月度成交额不是正数")
    return {"factor_month": expected_month, "trade_amount_cny_100m": trade}


def build_monthly_factor(
    daily: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "date",
        "sse_a_trade_amount_cny_100m",
        "sse_a_market_cap_cny_100m",
        "szse_stock_trade_amount_cny_100m",
        "szse_stock_market_cap_cny_100m",
    }
    if missing := required.difference(daily.columns):
        raise ContractError(f"官方日度输入缺少字段：{sorted(missing)}")
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ContractError("官方日度输入日期为空或重复")
    numeric_columns = sorted(required.difference({"date"}))
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if frame[numeric_columns].isna().any().any() or (frame[numeric_columns] <= 0).any().any():
        raise ContractError("官方日度输入存在空值、零值或负值")
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["factor_month"] = frame["date"].dt.to_period("M").astype(str)
    frame["aggregate_trade_amount_cny_100m"] = (
        frame["sse_a_trade_amount_cny_100m"]
        + frame["szse_stock_trade_amount_cny_100m"]
    )
    frame["aggregate_market_cap_cny_100m"] = (
        frame["sse_a_market_cap_cny_100m"]
        + frame["szse_stock_market_cap_cny_100m"]
    )
    monthly_trade = (
        frame.groupby("factor_month", sort=True)["aggregate_trade_amount_cny_100m"]
        .sum()
        .rename("aggregate_monthly_trade_amount_cny_100m")
    )
    month_end = (
        frame.groupby("factor_month", sort=True, as_index=False)
        .tail(1)
        .set_index("factor_month")
    )
    output = pd.DataFrame(monthly_trade).join(
        month_end[["date", "aggregate_market_cap_cny_100m"]], how="inner"
    )
    output = output.rename(
        columns={
            "date": "month_end_date",
            "aggregate_market_cap_cny_100m": "aggregate_month_end_market_cap_cny_100m",
        }
    ).reset_index()
    output["aggregate_turnover_ratio"] = (
        output["aggregate_monthly_trade_amount_cny_100m"]
        / output["aggregate_month_end_market_cap_cny_100m"]
    )
    allocation = config["external_allocation"]
    output["external_forecast_monthly_return"] = (
        float(allocation["intercept"])
        + float(allocation["turnover_coefficient"])
        * output["aggregate_turnover_ratio"]
    )
    raw_weight = output["external_forecast_monthly_return"] / (
        float(allocation["risk_aversion_gamma"])
        * float(allocation["fixed_monthly_variance"])
    )
    output["target_weight_before_lot_rounding"] = raw_weight.clip(
        lower=float(allocation["lower_weight_bound"]),
        upper=float(allocation["upper_weight_bound"]),
    )
    return output.sort_values("month_end_date").reset_index(drop=True)


def build_monthly_factor_from_components(
    components: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """从冻结的交易所月度组成项复算论文定义的总换手率与外部仓位。"""

    component_columns = [
        "sse_a_monthly_trade_amount_cny_100m",
        "sse_a_month_end_market_cap_cny_100m",
        "szse_stock_monthly_trade_amount_cny_100m",
        "szse_stock_month_end_market_cap_cny_100m",
    ]
    aggregate_columns = [
        "aggregate_monthly_trade_amount_cny_100m",
        "aggregate_month_end_market_cap_cny_100m",
    ]
    required = {"factor_month", "month_end_date", *component_columns, *aggregate_columns}
    if missing := required.difference(components.columns):
        raise ContractError(f"官方月度组成输入缺少字段：{sorted(missing)}")

    frame = components.copy()
    frame["factor_month"] = frame["factor_month"].astype(str)
    frame["month_end_date"] = pd.to_datetime(frame["month_end_date"], errors="coerce")
    if frame["month_end_date"].isna().any():
        raise ContractError("官方月度组成输入存在无效月末日期")
    if frame["factor_month"].duplicated().any() or frame["month_end_date"].duplicated().any():
        raise ContractError("官方月度组成输入存在重复月份或重复月末日期")
    if not (
        frame["factor_month"]
        == frame["month_end_date"].dt.to_period("M").astype(str)
    ).all():
        raise ContractError("官方月度组成输入的月份与月末日期不一致")

    numeric_columns = component_columns + aggregate_columns
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if frame[numeric_columns].isna().any().any():
        raise ContractError("官方月度组成输入存在非数值或空值")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ContractError("官方月度组成输入存在非有限数")
    if (frame[numeric_columns] <= 0).any().any():
        raise ContractError("官方月度组成输入存在零值或负值")

    frame = frame.sort_values("factor_month").reset_index(drop=True)
    contract = config["data_contract"]
    expected_months = pd.period_range(
        start=str(contract["expected_first_factor_month"]),
        end=str(contract["expected_last_factor_month"]),
        freq="M",
    ).astype(str).tolist()
    if len(frame) != int(contract["expected_complete_months"]):
        raise ContractError(
            f"官方月度组成输入行数异常：{len(frame)}，"
            f"预期{contract['expected_complete_months']}"
        )
    if frame["factor_month"].tolist() != expected_months:
        raise ContractError("官方月度组成输入月份不连续或边界不一致")

    rebuilt_trade = (
        frame["sse_a_monthly_trade_amount_cny_100m"]
        + frame["szse_stock_monthly_trade_amount_cny_100m"]
    )
    rebuilt_market_cap = (
        frame["sse_a_month_end_market_cap_cny_100m"]
        + frame["szse_stock_month_end_market_cap_cny_100m"]
    )
    if not np.allclose(
        frame["aggregate_monthly_trade_amount_cny_100m"].to_numpy(dtype=float),
        rebuilt_trade.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-9,
        equal_nan=False,
    ):
        raise ContractError("官方月度组成输入的总成交额与两市组成项不一致")
    if not np.allclose(
        frame["aggregate_month_end_market_cap_cny_100m"].to_numpy(dtype=float),
        rebuilt_market_cap.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-9,
        equal_nan=False,
    ):
        raise ContractError("官方月度组成输入的总市值与两市组成项不一致")

    output = frame[
        [
            "factor_month",
            "month_end_date",
            "aggregate_monthly_trade_amount_cny_100m",
            "aggregate_month_end_market_cap_cny_100m",
        ]
    ].copy()
    output["aggregate_turnover_ratio"] = (
        output["aggregate_monthly_trade_amount_cny_100m"]
        / output["aggregate_month_end_market_cap_cny_100m"]
    )
    allocation = config["external_allocation"]
    output["external_forecast_monthly_return"] = (
        float(allocation["intercept"])
        + float(allocation["turnover_coefficient"])
        * output["aggregate_turnover_ratio"]
    )
    raw_weight = output["external_forecast_monthly_return"] / (
        float(allocation["risk_aversion_gamma"])
        * float(allocation["fixed_monthly_variance"])
    )
    output["target_weight_before_lot_rounding"] = raw_weight.clip(
        lower=float(allocation["lower_weight_bound"]),
        upper=float(allocation["upper_weight_bound"]),
    )
    return output.reset_index(drop=True)


def _load_market(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = set(config["inputs"]["etf_daily"]["required_columns"])
    if missing := required.difference(frame.columns):
        raise ContractError(f"510300行情缺少字段：{sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    price_columns = ["open", "high", "low", "close"]
    frame[price_columns] = frame[price_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ContractError("510300行情日期为空或重复")
    if frame[price_columns].isna().any().any() or (frame[price_columns] <= 0).any().any():
        raise ContractError("510300行情价格为空、零或负数")
    if "symbol" in frame.columns and set(frame["symbol"].dropna().astype(str)) != {
        "510300.SH"
    }:
        raise ContractError("510300行情文件包含其他证券")
    required_dates = {
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    }
    if not required_dates.issubset(set(frame["date"])):
        raise ContractError("510300行情未精确覆盖冻结评价起止日")
    return frame


def _load_dividends(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(config["inputs"]["dividends"]["required_columns"])
    if missing := required.difference(frame.columns):
        raise ContractError(f"分红输入缺少字段：{sorted(missing)}")
    for column in ["record_date", "ex_date", "payment_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["cash_dividend_per_share"] = pd.to_numeric(
        frame["cash_dividend_per_share"], errors="coerce"
    )
    if frame[["record_date", "ex_date", "payment_date"]].isna().any().any():
        raise ContractError("分红日期存在空值")
    if frame["cash_dividend_per_share"].isna().any() or (
        frame["cash_dividend_per_share"] < 0
    ).any():
        raise ContractError("分红金额存在空值或负数")
    if set(frame["symbol"].astype(str)) != {"510300.SH"}:
        raise ContractError("分红输入包含其他证券")
    event_columns = [
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
    ]
    if frame[event_columns].duplicated().any():
        raise ContractError("分红输入存在重复事件")
    if not (
        (frame["record_date"] <= frame["ex_date"])
        & (frame["ex_date"] <= frame["payment_date"])
    ).all():
        raise ContractError("分红登记日、除息日或发放日顺序异常")
    return frame.sort_values("ex_date").reset_index(drop=True)


def _load_benchmark(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if {"date", "close"}.difference(frame.columns):
        raise ContractError("H00300基准缺少date或close")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if frame[["date", "close"]].isna().any().any():
        raise ContractError("H00300基准日期或价格存在空值")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or (frame["close"] <= 0).any():
        raise ContractError("H00300基准日期重复或价格无效")
    required_dates = {
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    }
    if not required_dates.issubset(set(frame["date"])):
        raise ContractError("H00300基准未精确覆盖冻结评价起止日")
    return frame


def _validate_etf_input_audit(config: dict[str, Any]) -> None:
    audit_path = _project_path(config["inputs"]["etf_input_audit"]["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != config["inputs"]["etf_input_audit"]["required_status"]:
        raise ContractError("既有510300行情输入审计不是PASS")
    if audit.get("candidate_outcomes_computed") is not False:
        raise ContractError("既有510300行情输入审计越界计算了候选结果")
    if audit.get("portfolio_returns_computed") is not False:
        raise ContractError("既有510300行情输入审计越界计算了组合收益")

    market_path = _project_path(config["inputs"]["etf_daily"]["path"])
    canonical = audit.get("canonical_price") or {}
    if canonical.get("output_path") != config["inputs"]["etf_daily"]["path"]:
        raise ContractError("既有行情审计绑定的510300文件路径不一致")
    if canonical.get("output_sha256") != sha256_file(market_path):
        raise ContractError("既有行情审计绑定的510300文件哈希不一致")
    if pd.Timestamp(canonical.get("source_last_date")) < pd.Timestamp(
        config["dates"]["evaluation_end"]
    ):
        raise ContractError("既有行情审计的源数据末日早于冻结评价末日")

    audited_hashes = audit.get("input_hashes") or {}
    for key in ["dividends", "benchmark_total_return"]:
        relative = config["inputs"][key]["path"]
        if audited_hashes.get(relative) != sha256_file(_project_path(relative)):
            raise ContractError(f"既有行情审计绑定的输入哈希不一致：{relative}")


def load_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_path = _project_path(config["inputs"]["input_audit"]["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ContractError("换手率输入审计不是PASS")
    if audit.get("candidate_outcomes_computed") is not False:
        raise ContractError("输入审计越界读取了候选结果")
    if audit.get("portfolio_returns_computed") is not False:
        raise ContractError("输入审计越界读取了组合收益")
    _validate_etf_input_audit(config)
    market = _load_market(_project_path(config["inputs"]["etf_daily"]["path"]), config)
    dividends = _load_dividends(
        _project_path(config["inputs"]["dividends"]["path"]), config
    )
    benchmark = _load_benchmark(
        _project_path(config["inputs"]["benchmark_total_return"]["path"]),
        config,
    )
    components = pd.read_parquet(
        _project_path(config["inputs"]["official_monthly_components"]["path"])
    )
    monthly = pd.read_parquet(_project_path(config["inputs"]["monthly_factor"]["path"]))
    rebuilt = build_monthly_factor_from_components(components, config)
    monthly = monthly.sort_values("month_end_date").reset_index(drop=True)
    if list(monthly.columns) != list(rebuilt.columns):
        raise ContractError("冻结月度因子列结构与复算结果不一致")
    for column in monthly.columns:
        if column in {"factor_month"}:
            if not monthly[column].astype(str).equals(rebuilt[column].astype(str)):
                raise ContractError(f"月度因子文本列复算不一致：{column}")
        elif column == "month_end_date":
            if not pd.to_datetime(monthly[column]).equals(pd.to_datetime(rebuilt[column])):
                raise ContractError("月末日期复算不一致")
        else:
            left = pd.to_numeric(monthly[column], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=False):
                raise ContractError(f"月度因子数值列复算不一致：{column}")
    monthly["month_end_date"] = pd.to_datetime(monthly["month_end_date"])
    return market, dividends, benchmark, components, monthly


def build_signal_schedule(
    market: pd.DataFrame,
    monthly: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    trading_dates = market["date"].sort_values().tolist()
    date_position = {date: index for index, date in enumerate(trading_dates)}
    rows: list[dict[str, Any]] = []
    for factor in monthly.itertuples(index=False):
        decision_date = pd.Timestamp(factor.month_end_date)
        if decision_date not in date_position:
            raise ContractError(f"因子月末{decision_date.date()}不是510300交易日")
        index = date_position[decision_date]
        if index + 1 >= len(trading_dates):
            raise ContractError("最后一个因子月末没有下一交易日")
        execution_date = pd.Timestamp(trading_dates[index + 1])
        factor_month = str(factor.factor_month)
        if factor_month != str(decision_date.to_period("M")):
            raise ContractError("因子月份与决策月不一致")
        target = float(factor.target_weight_before_lot_rounding)
        forecast = float(factor.external_forecast_monthly_return)
        rows.append(
            {
                "factor_month": factor_month,
                "decision_date": decision_date,
                "execution_date": execution_date,
                "aggregate_turnover_ratio": float(factor.aggregate_turnover_ratio),
                "external_forecast_monthly_return": forecast,
                "target_position": target,
                "signal_reason": (
                    "论文固定系数预测非正，目标为现金"
                    if target == 0.0
                    else "论文固定系数与固定方差映射的510300目标权重"
                ),
            }
        )
    signals = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    if signals["decision_date"].duplicated().any():
        raise ContractError("信号决策日重复")
    if not signals["target_position"].between(0.0, 1.0).all():
        raise ContractError("目标仓位超出0到1")
    return signals


def _structural_period(date: pd.Timestamp, config: dict[str, Any]) -> str | None:
    for name, bounds in config["dates"]["structural_periods"].items():
        if pd.Timestamp(bounds["start"]) <= date <= pd.Timestamp(bounds["end"]):
            return str(name)
    return None


def add_mechanism_targets(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    complete = signals.sort_values("execution_date").copy()
    complete["exit_execution_date"] = complete["execution_date"].shift(-1)
    complete = complete.loc[
        complete["execution_date"].between(start, end)
        & complete["exit_execution_date"].notna()
        & (complete["exit_execution_date"] <= end)
    ].copy()
    open_by_date = market.set_index("date")["open"]
    future_returns: list[float] = []
    dividends_per_share: list[float] = []
    for row in complete.itertuples(index=False):
        entry_date = pd.Timestamp(row.execution_date)
        exit_date = pd.Timestamp(row.exit_execution_date)
        if entry_date not in open_by_date.index or exit_date not in open_by_date.index:
            raise ContractError("机制目标进入日或退出日缺少开盘价")
        entitled = float(
            dividends.loc[
                (dividends["record_date"] >= entry_date)
                & (dividends["record_date"] < exit_date),
                "cash_dividend_per_share",
            ].sum()
        )
        entry_open = float(open_by_date.loc[entry_date])
        exit_open = float(open_by_date.loc[exit_date])
        dividends_per_share.append(entitled)
        future_returns.append((exit_open + entitled) / entry_open - 1.0)
    complete["entitled_cash_dividend_per_share"] = dividends_per_share
    complete["future_interval_total_return"] = future_returns
    complete["structural_period"] = [
        _structural_period(pd.Timestamp(date), config)
        for date in complete["execution_date"]
    ]
    complete["calendar_year"] = complete["execution_date"].dt.year
    return complete.reset_index(drop=True)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def ordinary_least_squares_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(x) != len(y) or np.var(x) <= 0:
        return None
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.pinv(design.T @ design) @ design.T @ y
    return _optional_float(coefficients[1])


def newey_west_slope(
    factor: pd.Series,
    future_return: pd.Series,
    *,
    max_lag: int = 3,
) -> dict[str, float | int | None]:
    frame = pd.DataFrame({"x": factor, "y": future_return}).dropna()
    if len(frame) < max_lag + 4:
        return {
            "observations": int(len(frame)),
            "intercept": None,
            "slope": None,
            "standard_error": None,
            "t_stat": None,
            "max_lag": int(max_lag),
        }
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ y
    residual = y - design @ coefficients
    score = design * residual[:, None]
    meat = score.T @ score
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    variance = float(covariance[1, 1])
    standard_error = math.sqrt(variance) if variance > 0 else None
    slope = float(coefficients[1])
    return {
        "observations": int(len(frame)),
        "intercept": _optional_float(coefficients[0]),
        "slope": slope,
        "standard_error": _optional_float(standard_error),
        "t_stat": _optional_float(slope / standard_error if standard_error else None),
        "max_lag": int(max_lag),
    }


def circular_block_bootstrap_slope(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    valid = frame[["aggregate_turnover_ratio", "future_interval_total_return"]].dropna()
    valid = valid.reset_index(drop=True)
    repetitions = int(config["bootstrap"]["repetitions"])
    block_length = int(config["bootstrap"]["block_length_months"])
    confidence = float(config["bootstrap"]["confidence_level"])
    seed = int(config["bootstrap"]["random_seed"])
    n = len(valid)
    if n < block_length or repetitions < 1:
        return {
            "repetitions": repetitions,
            "block_length_months": block_length,
            "valid_draws": 0,
            "confidence_interval": [None, None],
            "confidence_level": confidence,
            "seed": seed,
        }
    x = valid["aggregate_turnover_ratio"].to_numpy(dtype=float)
    y = valid["future_interval_total_return"].to_numpy(dtype=float)
    block_count = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        slope = ordinary_least_squares_slope(x[indices], y[indices])
        if slope is not None:
            slopes.append(slope)
    alpha = (1.0 - confidence) / 2.0
    interval = (
        [float(np.quantile(slopes, alpha)), float(np.quantile(slopes, 1.0 - alpha))]
        if slopes
        else [None, None]
    )
    return {
        "repetitions": repetitions,
        "block_length_months": block_length,
        "valid_draws": int(len(slopes)),
        "confidence_interval": interval,
        "confidence_level": confidence,
        "seed": seed,
    }


def evaluate_mechanism_gate(
    mechanism: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    columns = [
        "execution_date",
        "structural_period",
        "aggregate_turnover_ratio",
        "future_interval_total_return",
    ]
    valid = mechanism[columns].dropna().copy()
    max_lag = int(config["mechanism_gates"]["newey_west_max_lag_months"])
    regression = newey_west_slope(
        valid["aggregate_turnover_ratio"],
        valid["future_interval_total_return"],
        max_lag=max_lag,
    )
    bootstrap = circular_block_bootstrap_slope(valid, config)
    lower = bootstrap["confidence_interval"][0]
    structural: dict[str, dict[str, Any]] = {}
    for name in config["dates"]["structural_periods"]:
        subset = valid.loc[valid["structural_period"] == name]
        result = newey_west_slope(
            subset["aggregate_turnover_ratio"],
            subset["future_interval_total_return"],
            max_lag=max_lag,
        )
        structural[name] = result
    gate_config = config["mechanism_gates"]
    gates = {
        "minimum_complete_monthly_targets": len(valid)
        >= int(gate_config["minimum_complete_monthly_targets"]),
        "full_sample_slope_positive": regression["slope"] is not None
        and float(regression["slope"]) > 0.0,
        "newey_west_one_sided_t": regression["t_stat"] is not None
        and float(regression["t_stat"])
        >= float(gate_config["newey_west_one_sided_t_minimum"]),
        "bootstrap_90pct_lower_positive": lower is not None and float(lower) > 0.0,
        "every_structural_period_slope_positive": bool(structural)
        and all(
            details["slope"] is not None and float(details["slope"]) > 0.0
            for details in structural.values()
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "observations": int(len(valid)),
        "turnover_minimum": _optional_float(valid["aggregate_turnover_ratio"].min()),
        "turnover_maximum": _optional_float(valid["aggregate_turnover_ratio"].max()),
        "future_return_mean": _optional_float(valid["future_interval_total_return"].mean()),
        "newey_west_regression": regression,
        "bootstrap": bootstrap,
        "structural_periods": structural,
        "gates": gates,
    }


def _engine_targets(signals: pd.DataFrame) -> pd.DataFrame:
    return (
        signals[
            ["decision_date", "execution_date", "target_position", "signal_reason"]
        ]
        .rename(columns={"decision_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def _costs(config: dict[str, Any], name: str) -> BacktestCosts:
    specification = config["portfolio"][name]
    return BacktestCosts(
        commission_rate=float(specification["commission_rate_per_leg"]),
        minimum_commission_cny=float(specification["minimum_commission_cny_per_leg"]),
        stamp_duty_rate=float(specification["stamp_duty_rate"]),
        slippage_bps=float(specification["slippage_bps_per_leg"]),
        lot_size=int(config["portfolio"]["lot_size_shares"]),
        cash_annual_rate=float(config["portfolio"]["cash_annual_rate"]),
    )


def _engine_start_date(market: pd.DataFrame, evaluation_start: pd.Timestamp) -> pd.Timestamp:
    earlier = market.loc[market["date"] < evaluation_start, "date"]
    if earlier.empty:
        raise ContractError("评价期前缺少一个交易日用于执行2014年12月信号")
    return pd.Timestamp(earlier.max())


def _run_account(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    costs: BacktestCosts,
    *,
    delayed_start_months: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    engine_start = _engine_start_date(market, start)
    prepared = targets.copy()
    if delayed_start_months > 0:
        cutoff = start + pd.DateOffset(months=int(delayed_start_months))
        prepared.loc[prepared["execution_date"] < cutoff, "target_position"] = 0.0
        prepared.loc[prepared["execution_date"] < cutoff, "signal_reason"] = (
            f"延迟起点稳健性：前{int(delayed_start_months)}个月持有现金"
        )
    ledger, trades = run_long_cash_backtest(
        prices=market,
        dividends=dividends,
        targets=prepared,
        initial_cash=float(config["portfolio"]["initial_capital_cny"]),
        costs=costs,
        start_date=engine_start,
        end_date=end,
    )
    ledger = ledger.loc[ledger["date"].between(start, end)].reset_index(drop=True)
    if not trades.empty:
        trades = trades.loc[trades["date"].between(start, end)].reset_index(drop=True)
    return ledger, trades


def _run_buy_hold(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    engine_start = _engine_start_date(market, start)
    targets = pd.DataFrame(
        {
            "date": [engine_start],
            "execution_date": [start],
            "target_position": [1.0],
            "signal_reason": ["510300买入持有基准"],
        }
    )
    return _run_account(market, dividends, targets, config, _costs(config, "base_costs"))


def _benchmark_cagr(
    benchmark: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    subset = benchmark.loc[benchmark["date"].between(start_date, end_date)].copy()
    if len(subset) < 2:
        raise ContractError("H00300基准区间不足两行")
    if subset["date"].iloc[0] != start_date or subset["date"].iloc[-1] != end_date:
        raise ContractError("H00300基准区间没有精确落在冻结评价起止日")
    elapsed_days = max((subset["date"].iloc[-1] - subset["date"].iloc[0]).days, 1)
    total_return = float(subset["close"].iloc[-1] / subset["close"].iloc[0] - 1.0)
    return float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


def _return_series_summary(returns: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "observations": 0,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_zero_cash_rate": None,
        }
    total = float(np.prod(1.0 + values) - 1.0)
    annualized_return = (
        float((1.0 + total) ** (242.0 / len(values)) - 1.0) if total > -1.0 else -1.0
    )
    volatility = (
        float(np.std(values, ddof=1) * math.sqrt(242.0)) if len(values) > 1 else None
    )
    mean_return = float(np.mean(values) * 242.0)
    sharpe = mean_return / volatility if volatility and volatility > 0 else None
    return {
        "observations": int(len(values)),
        "total_return": total,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": _optional_float(sharpe),
    }


def probabilistic_sharpe_probability(
    returns: pd.Series,
    benchmark_annual_sharpe: float,
) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return None
    daily_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    benchmark_daily = float(benchmark_annual_sharpe) / math.sqrt(242.0)
    series = pd.Series(values)
    skewness = float(series.skew())
    raw_kurtosis = float(series.kurt()) + 3.0
    denominator_squared = (
        1.0
        - skewness * daily_sharpe
        + ((raw_kurtosis - 1.0) / 4.0) * daily_sharpe**2
    )
    if denominator_squared <= 0:
        return None
    z_score = (
        (daily_sharpe - benchmark_daily)
        * math.sqrt(len(values) - 1)
        / math.sqrt(denominator_squared)
    )
    return float(NormalDist().cdf(z_score))


def deflated_sharpe_probability(
    returns: pd.Series,
    total_trial_count: int,
) -> dict[str, float | int | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return {
            "total_trial_count": int(total_trial_count),
            "expected_maximum_null_sharpe_annualized": None,
            "probability": None,
        }
    trial_count = max(int(total_trial_count), 1)
    if trial_count == 1:
        expected_maximum_daily = 0.0
    else:
        gamma = 0.5772156649015329
        normal = NormalDist()
        trial_std_daily = 1.0 / math.sqrt(len(values) - 1)
        expected_maximum_daily = trial_std_daily * (
            (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + gamma * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        )
    probability = probabilistic_sharpe_probability(
        pd.Series(values), expected_maximum_daily * math.sqrt(242.0)
    )
    return {
        "total_trial_count": trial_count,
        "expected_maximum_null_sharpe_annualized": float(
            expected_maximum_daily * math.sqrt(242.0)
        ),
        "probability": probability,
    }


def _annual_contribution_analysis(
    strategy_ledger: pd.DataFrame,
    buy_hold_ledger: pd.DataFrame,
) -> dict[str, Any]:
    merged = strategy_ledger[["date", "daily_return"]].merge(
        buy_hold_ledger[["date", "daily_return"]],
        on="date",
        suffixes=("_strategy", "_buy_hold"),
        validate="one_to_one",
    )
    merged["year"] = merged["date"].dt.year
    annual_rows: list[dict[str, Any]] = []
    for year, subset in merged.groupby("year", sort=True):
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        buy_hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        annual_rows.append(
            {
                "year": int(year),
                "strategy_return": strategy_return,
                "buy_hold_return": buy_hold_return,
                "excess_return": strategy_return - buy_hold_return,
            }
        )
    positives = [max(float(row["excess_return"]), 0.0) for row in annual_rows]
    positive_sum = float(sum(positives))
    maximum_share = max(positives) / positive_sum if positive_sum > 0 else None
    delete_year: dict[str, float] = {}
    for year in sorted(merged["year"].unique()):
        subset = merged.loc[merged["year"] != int(year)]
        strategy_return = float((1.0 + subset["daily_return_strategy"]).prod() - 1.0)
        buy_hold_return = float((1.0 + subset["daily_return_buy_hold"]).prod() - 1.0)
        delete_year[str(int(year))] = strategy_return - buy_hold_return
    return {
        "annual_rows": annual_rows,
        "maximum_single_positive_year_share_of_positive_excess": maximum_share,
        "delete_calendar_year_total_return_excess_vs_buy_hold": delete_year,
    }


def evaluate_portfolio(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    targets = _engine_targets(signals)
    base_ledger, base_trades = _run_account(
        market, dividends, targets, config, _costs(config, "base_costs")
    )
    stress_ledger, stress_trades = _run_account(
        market, dividends, targets, config, _costs(config, "stress_costs")
    )
    buy_hold_ledger, buy_hold_trades = _run_buy_hold(market, dividends, config)
    initial_cash = float(config["portfolio"]["initial_capital_cny"])
    base = summarize_backtest(base_ledger, base_trades, initial_cash)
    stress = summarize_backtest(stress_ledger, stress_trades, initial_cash)
    buy_hold = summarize_backtest(buy_hold_ledger, buy_hold_trades, initial_cash)
    h00300_cagr = _benchmark_cagr(
        benchmark,
        pd.Timestamp(config["dates"]["evaluation_start"]),
        pd.Timestamp(config["dates"]["evaluation_end"]),
    )
    structural: dict[str, dict[str, Any]] = {}
    for name, bounds in config["dates"]["structural_periods"].items():
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"])
        strategy_returns = base_ledger.loc[
            base_ledger["date"].between(start, end), "daily_return"
        ]
        buy_hold_returns = buy_hold_ledger.loc[
            buy_hold_ledger["date"].between(start, end), "daily_return"
        ]
        strategy_summary = _return_series_summary(strategy_returns)
        buy_hold_summary = _return_series_summary(buy_hold_returns)
        structural[name] = {
            "strategy": strategy_summary,
            "buy_hold": buy_hold_summary,
            "annualized_excess_vs_buy_hold": (
                float(strategy_summary["annualized_return"])
                - float(buy_hold_summary["annualized_return"])
                if strategy_summary["annualized_return"] is not None
                and buy_hold_summary["annualized_return"] is not None
                else None
            ),
        }
    delayed: dict[str, dict[str, Any]] = {}
    for months in config["portfolio"]["delayed_start_months"]:
        ledger, trades = _run_account(
            market,
            dividends,
            targets,
            config,
            _costs(config, "base_costs"),
            delayed_start_months=int(months),
        )
        details = summarize_backtest(ledger, trades, initial_cash)
        details["annualized_excess_vs_buy_hold"] = float(details["cagr"]) - float(
            buy_hold["cagr"]
        )
        delayed[str(int(months))] = details
    annual = _annual_contribution_analysis(base_ledger, buy_hold_ledger)
    total_trials = int(
        manifest["selection_bias_control"]["total_trial_count_including_current"]
    )
    dsr = deflated_sharpe_probability(base_ledger["daily_return"], total_trials)
    psr_above_1 = probabilistic_sharpe_probability(base_ledger["daily_return"], 1.0)
    drawdown_ratio = (
        abs(float(base["max_drawdown"])) / abs(float(buy_hold["max_drawdown"]))
        if float(buy_hold["max_drawdown"]) < 0
        else None
    )
    base_excess_buy_hold = float(base["cagr"]) - float(buy_hold["cagr"])
    base_excess_h00300 = float(base["cagr"]) - h00300_cagr
    gate_config = config["portfolio_gates"]
    gates = {
        "base_net_sharpe": base["sharpe_zero_cash_rate"] is not None
        and float(base["sharpe_zero_cash_rate"])
        >= float(gate_config["base_net_sharpe_minimum"]),
        "stress_net_sharpe": stress["sharpe_zero_cash_rate"] is not None
        and float(stress["sharpe_zero_cash_rate"])
        >= float(gate_config["stress_net_sharpe_minimum"]),
        "annualized_excess_vs_510300_buy_hold_positive": base_excess_buy_hold > 0.0,
        "annualized_excess_vs_h00300_total_return_positive": base_excess_h00300 > 0.0,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio is not None
        and drawdown_ratio
        <= float(gate_config["maximum_drawdown_ratio_vs_buy_hold_maximum"]),
        "every_structural_period_net_sharpe": all(
            details["strategy"]["sharpe_zero_cash_rate"] is not None
            and float(details["strategy"]["sharpe_zero_cash_rate"])
            >= float(gate_config["every_structural_period_net_sharpe_minimum"])
            for details in structural.values()
        ),
        "every_structural_period_excess_vs_buy_hold_positive": all(
            details["annualized_excess_vs_buy_hold"] is not None
            and float(details["annualized_excess_vs_buy_hold"]) > 0.0
            for details in structural.values()
        ),
        "every_delayed_start_net_sharpe": all(
            details["sharpe_zero_cash_rate"] is not None
            and float(details["sharpe_zero_cash_rate"])
            >= float(gate_config["every_delayed_start_net_sharpe_minimum"])
            for details in delayed.values()
        ),
        "every_delayed_start_excess_vs_buy_hold_positive": all(
            float(details["annualized_excess_vs_buy_hold"]) > 0.0
            for details in delayed.values()
        ),
        "delete_any_calendar_year_excess_vs_buy_hold_positive": all(
            float(value) > 0.0
            for value in annual[
                "delete_calendar_year_total_return_excess_vs_buy_hold"
            ].values()
        ),
        "maximum_single_positive_year_share": annual[
            "maximum_single_positive_year_share_of_positive_excess"
        ]
        is not None
        and float(
            annual["maximum_single_positive_year_share_of_positive_excess"]
        )
        <= float(
            gate_config["maximum_single_positive_year_share_of_positive_excess"]
        ),
        "deflated_sharpe_probability": dsr["probability"] is not None
        and float(dsr["probability"])
        >= float(gate_config["deflated_sharpe_probability_minimum"]),
        "probabilistic_sharpe_above_1p0": psr_above_1 is not None
        and float(psr_above_1)
        >= float(gate_config["probabilistic_sharpe_above_1p0_minimum"]),
    }
    result = {
        "passed": bool(all(gates.values())),
        "base": base,
        "stress": stress,
        "buy_hold_510300": buy_hold,
        "h00300_total_return_cagr": h00300_cagr,
        "base_annualized_excess_vs_510300_buy_hold": base_excess_buy_hold,
        "base_annualized_excess_vs_h00300": base_excess_h00300,
        "maximum_drawdown_ratio_vs_buy_hold": drawdown_ratio,
        "structural_periods": structural,
        "delayed_starts": delayed,
        "annual_contribution": annual,
        "deflated_sharpe": dsr,
        "probabilistic_sharpe_above_1p0": psr_above_1,
        "gates": gates,
    }
    tables = {
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "stress_ledger": stress_ledger,
        "stress_trades": stress_trades,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
    }
    return result, tables


def _input_snapshot(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = [
        "etf_daily",
        "etf_input_audit",
        "dividends",
        "benchmark_total_return",
        "official_source_archive",
        "official_monthly_components",
        "monthly_factor",
        "input_audit",
        "backtest_engine",
    ]
    output: dict[str, dict[str, Any]] = {}
    for key in keys:
        path = _project_path(config["inputs"][key]["path"])
        output[path.relative_to(ROOT).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return output


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    signals: pd.DataFrame,
    mechanism: pd.DataFrame,
    mechanism_result: dict[str, Any],
    portfolio_result: dict[str, Any] | None,
) -> dict[str, Any]:
    mechanism_passed = bool(mechanism_result["passed"])
    portfolio_evaluated = portfolio_result is not None
    if not mechanism_passed:
        status = config["adjudication"]["mechanism_fail_status"]
        return_evaluation = "NOT_ALLOWED"
        net_sharpe: float | str = "NOT_COMPUTED"
        historical_target_achieved = False
    else:
        if portfolio_result is None:
            raise ContractError("机制通过后缺少组合评价")
        net_sharpe = float(portfolio_result["base"]["sharpe_zero_cash_rate"])
        historical_target_achieved = bool(portfolio_result["passed"])
        status = (
            config["adjudication"]["portfolio_pass_status"]
            if historical_target_achieved
            else config["adjudication"]["portfolio_fail_status"]
        )
        return_evaluation = "ALLOWED_AND_COMPLETED"
    return {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "paper_transfer": {
            "title": config["literature"]["primary_paper"]["title"],
            "paper_sample_end": config["literature"]["primary_paper"]["sample_end"],
            "evaluation_start": config["dates"]["evaluation_start"],
            "intercept": config["external_allocation"]["intercept"],
            "turnover_coefficient": config["external_allocation"][
                "turnover_coefficient"
            ],
            "risk_aversion_gamma": config["external_allocation"][
                "risk_aversion_gamma"
            ],
            "fixed_monthly_standard_deviation": config["external_allocation"][
                "fixed_monthly_standard_deviation"
            ],
            "local_return_fitting": False,
        },
        "signal_summary": {
            "rows": int(len(signals)),
            "first_decision_date": signals["decision_date"].min().date().isoformat(),
            "last_decision_date": signals["decision_date"].max().date().isoformat(),
            "minimum_target_weight": float(signals["target_position"].min()),
            "maximum_target_weight": float(signals["target_position"].max()),
            "mean_target_weight": float(signals["target_position"].mean()),
            "cash_target_months": int((signals["target_position"] == 0.0).sum()),
            "full_weight_months": int((signals["target_position"] == 1.0).sum()),
        },
        "mechanism_gate_passed": mechanism_passed,
        "mechanism": mechanism_result,
        "mechanism_table_rows": int(len(mechanism)),
        "portfolio_evaluated": portfolio_evaluated,
        "portfolio": portfolio_result,
        "selection_bias_control": manifest["selection_bias_control"],
        "input_snapshot": _input_snapshot(config),
        "adjudication": {
            "return_evaluation": return_evaluation,
            "net_sharpe": net_sharpe,
            "target_net_sharpe": float(config["adjudication"]["target_net_sharpe"]),
            "historical_target_achieved": historical_target_achieved,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "no_rescue": True,
        },
        "execution_boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    mechanism = report["mechanism"]
    regression = mechanism["newey_west_regression"]
    bootstrap = mechanism["bootstrap"]
    lines = [
        "# 510300 全市场换手率可见度 V1",
        "",
        f"- 项目标识：`{report['project_id']}`",
        f"- 最终状态：`{report['status']}`",
        f"- 机制门：`{'PASS' if report['mechanism_gate_passed'] else 'FAIL'}`",
        f"- 组合收益是否获准计算：`{report['portfolio_evaluated']}`",
        f"- RETURN_EVALUATION：`{report['adjudication']['return_evaluation']}`",
        f"- NET_SHARPE：`{report['adjudication']['net_sharpe']}`",
        "- 实盘授权：`false`",
        "",
        "## 冻结候选",
        "",
        "因子为两市官方月度成交额之和除以对应月末总市值。上交所严格取主板A股加科创板；深交所取官方Stocks合计，含极小的B股/存托凭证口径，该代理在结果后不得更换。",
        "",
        "2014至2025深交所数值来自官方年鉴重建；这不是年鉴在历史月末已发布的时间戳证明。真实前瞻运行必须在下一交易日开盘前独立捕获官方输入，缺失即NO_VIEW。",
        "",
        "论文固定预测式为 `forecast = -0.019 + 0.258 × turnover`；仓位为 `clip(forecast / (5 × 0.0901²), 0, 1)`。2015年前510300收益不参与拟合或评价。",
        "",
        "## 机制门结果",
        "",
        f"- 完整月度目标：{mechanism['observations']}",
        f"- Newey-West 斜率：{regression['slope']}",
        f"- Newey-West t 值：{regression['t_stat']}",
        f"- 90% 移动块自助区间：{bootstrap['confidence_interval']}",
        "",
        "机制门明细：",
        "",
    ]
    for name, passed in mechanism["gates"].items():
        lines.append(f"- `{name}`：`{passed}`")
    lines.extend(["", "## 组合评价", ""])
    portfolio = report["portfolio"]
    if portfolio is None:
        lines.append("机制门失败，按冻结协议禁止计算策略收益、夏普率或任何替代仓位。")
    else:
        base = portfolio["base"]
        stress = portfolio["stress"]
        buy_hold = portfolio["buy_hold_510300"]
        lines.extend(
            [
                f"- 基准成本净夏普率：{base['sharpe_zero_cash_rate']}",
                f"- 10bp滑点压力净夏普率：{stress['sharpe_zero_cash_rate']}",
                f"- 510300买入持有夏普率：{buy_hold['sharpe_zero_cash_rate']}",
                f"- 基准成本CAGR：{base['cagr']}",
                f"- 最大回撤：{base['max_drawdown']}",
                "",
                "组合门明细：",
                "",
            ]
        )
        for name, passed in portfolio["gates"].items():
            lines.append(f"- `{name}`：`{passed}`")
    lines.extend(
        [
            "",
            "## 判定边界",
            "",
            "历史通过也只能进入独立前瞻确认，不能生成订单、连接券商或改变仓位；本报告始终为研究用途。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def run_study(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    market, dividends, benchmark, _components, monthly = load_inputs(config)
    signals = build_signal_schedule(market, monthly, config)
    mechanism = add_mechanism_targets(signals, market, dividends, config)
    mechanism_result = evaluate_mechanism_gate(mechanism, config)
    portfolio_result: dict[str, Any] | None = None
    portfolio_tables: dict[str, pd.DataFrame] = {}
    if mechanism_result["passed"]:
        portfolio_result, portfolio_tables = evaluate_portfolio(
            market, dividends, benchmark, signals, config, manifest
        )
    report = build_report(
        config,
        manifest,
        signals,
        mechanism,
        mechanism_result,
        portfolio_result,
    )
    if write:
        result_path = _project_path(config["paths"]["result_json"])
        markdown_path = _project_path(config["paths"]["result_markdown"])
        if result_path.exists() or markdown_path.exists():
            raise FileExistsError("冻结候选结果已经存在，禁止覆盖")
        _atomic_parquet(_project_path(config["paths"]["signal_table"]), signals)
        _atomic_parquet(_project_path(config["paths"]["mechanism_table"]), mechanism)
        if portfolio_result is not None:
            for key, frame in portfolio_tables.items():
                _atomic_parquet(_project_path(config["paths"][key]), frame)
        _atomic_json(result_path, report)
        _atomic_text(markdown_path, render_markdown(report))
    return report
