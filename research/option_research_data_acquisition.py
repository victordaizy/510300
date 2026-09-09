"""510300 与 IO 期权研究数据的标准化和质量检查。

这里只处理公开或已授权的原始行情，不构造历史买卖盘，不计算择时收益。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OPTION_MASTER_REQUIRED = {
    "ts_code",
    "opt_code",
    "call_put",
    "exercise_price",
    "maturity_date",
    "list_date",
    "delist_date",
    "per_unit",
    "opt_multiplier",
}

OPTION_DAILY_REQUIRED = {
    "ts_code",
    "trade_date",
    "exchange",
    "pre_settle",
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "vol",
    "amount",
    "oi",
}

SSE_RISK_REQUIRED = {
    "TRADE_DATE",
    "SECURITY_ID",
    "CONTRACT_ID",
    "CONTRACT_SYMBOL",
    "CONTRACT_TYPE",
    "DELTA_VALUE",
    "THETA_VALUE",
    "GAMMA_VALUE",
    "VEGA_VALUE",
    "RHO_VALUE",
    "IMPLC_VOLATLTY",
}

CFFEX_COLUMNS = {
    "合约代码": "contract_code",
    "今开盘": "open",
    "最高价": "high",
    "最低价": "low",
    "成交量": "volume",
    "成交金额": "turnover_10k_cny",
    "持仓量": "open_interest",
    "持仓变化": "open_interest_change",
    "今收盘": "close",
    "今结算": "settlement",
    "前结算": "previous_settlement",
    "涨跌1": "change_close",
    "涨跌2": "change_settlement",
    "Delta": "delta",
}


def _missing_columns(data: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(data.columns))


def trading_date_chunks(
    dates: Iterable[pd.Timestamp | str], size: int
) -> list[list[pd.Timestamp]]:
    """把交易日排序去重后分块。"""

    if size <= 0:
        raise ValueError("交易日分块大小必须大于0")
    normalized = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in dates
            if not pd.isna(value)
        }
    )
    return [normalized[index : index + size] for index in range(0, len(normalized), size)]


def normalize_contract_master(
    raw: pd.DataFrame, retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """筛选并标准化 510300 期权合约主表，保留调整合约。"""

    missing = _missing_columns(raw, OPTION_MASTER_REQUIRED)
    if missing:
        raise ValueError(f"期权合约主表缺少字段：{missing}")
    selected = raw.loc[raw["opt_code"].astype(str).eq("OP510300.SH")].copy()
    if selected.empty:
        raise ValueError("合约主表没有返回 OP510300.SH")
    for column in ("exercise_price", "per_unit", "opt_multiplier"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    for column in ("list_date", "maturity_date", "delist_date"):
        selected[column] = pd.to_datetime(
            selected[column], format="%Y%m%d", errors="coerce"
        ).dt.normalize()
    selected["call_put"] = selected["call_put"].astype(str).str.upper().str.strip()
    selected["contract_unit"] = selected["opt_multiplier"].where(
        selected["opt_multiplier"].gt(0), selected["per_unit"]
    )
    selected["is_adjusted"] = (
        selected["contract_unit"].ne(10000)
        | selected["per_unit"].ne(10000)
    )
    result = pd.DataFrame(
        {
            "contract_code": selected["ts_code"].astype(str).str.strip(),
            "underlying_code": "510300.SH",
            "option_type": selected["call_put"],
            "list_date": selected["list_date"],
            "expiry_date": selected["maturity_date"],
            "delist_date": selected["delist_date"],
            "strike": selected["exercise_price"],
            "contract_unit": selected["contract_unit"],
            "is_adjusted": selected["is_adjusted"].astype(bool),
            "contract_status": "上市",
            "source": "tushare_proxy.opt_basic",
            "retrieved_at": pd.Timestamp(retrieved_at),
            "provider_name": selected.get("name", pd.Series(index=selected.index, dtype="object")),
            "provider_opt_code": selected["opt_code"].astype(str),
            "provider_per_unit": selected["per_unit"],
            "provider_opt_multiplier": selected["opt_multiplier"],
        }
    )
    required_non_null = [
        "contract_code",
        "option_type",
        "list_date",
        "expiry_date",
        "strike",
        "contract_unit",
    ]
    if result[required_non_null].isna().any().any():
        raise ValueError("标准化合约主表存在关键空值")
    invalid_type = ~result["option_type"].isin(["C", "P"])
    if invalid_type.any():
        values = sorted(result.loc[invalid_type, "option_type"].unique().tolist())
        raise ValueError(f"合约主表存在未知期权类型：{values}")
    if result["contract_code"].duplicated().any():
        raise ValueError("合约主表存在重复合约代码")
    return result.sort_values(["list_date", "contract_code"]).reset_index(drop=True)


def normalize_option_daily(
    raw: pd.DataFrame,
    master: pd.DataFrame,
    underlying: pd.DataFrame,
    retrieved_at: pd.Timestamp | str,
) -> pd.DataFrame:
    """标准化已成交的 510300 期权日行情，不生成不存在的买卖盘。"""

    missing = _missing_columns(raw, OPTION_DAILY_REQUIRED)
    if missing:
        raise ValueError(f"期权日行情缺少字段：{missing}")
    codes = set(master["contract_code"].astype(str))
    selected = raw.loc[raw["ts_code"].astype(str).isin(codes)].copy()
    if selected.empty:
        raise ValueError("期权日行情中没有510300合约")
    selected["trade_date"] = pd.to_datetime(
        selected["trade_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    numeric = [
        "pre_settle",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "settle",
        "vol",
        "amount",
        "oi",
    ]
    for column in numeric:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    master_join = master[
        [
            "contract_code",
            "option_type",
            "expiry_date",
            "strike",
            "contract_unit",
            "is_adjusted",
        ]
    ]
    selected = selected.rename(columns={"ts_code": "contract_code"}).merge(
        master_join, on="contract_code", how="left", validate="many_to_one"
    )
    close_reference = underlying[["date", "close"]].copy()
    close_reference["date"] = pd.to_datetime(
        close_reference["date"], errors="coerce"
    ).dt.normalize()
    close_reference["close"] = pd.to_numeric(close_reference["close"], errors="coerce")
    close_reference = close_reference.drop_duplicates("date", keep="last").rename(
        columns={"date": "trade_date", "close": "underlying_close"}
    )
    selected = selected.merge(
        close_reference, on="trade_date", how="left", validate="many_to_one"
    )
    selected["contract_status"] = "上市"
    selected["source"] = "tushare_proxy.opt_daily"
    selected["retrieved_at"] = pd.Timestamp(retrieved_at)
    selected = selected.rename(
        columns={
            "pre_settle": "previous_settlement",
            "pre_close": "previous_close",
            "settle": "settlement",
            "vol": "volume",
            "amount": "turnover_10k_cny",
            "oi": "open_interest",
        }
    )
    ordered = [
        "trade_date",
        "contract_code",
        "option_type",
        "expiry_date",
        "strike",
        "contract_unit",
        "contract_status",
        "is_adjusted",
        "previous_settlement",
        "previous_close",
        "open",
        "high",
        "low",
        "close",
        "settlement",
        "volume",
        "turnover_10k_cny",
        "open_interest",
        "underlying_close",
        "exchange",
        "source",
        "retrieved_at",
    ]
    result = selected[ordered].copy()
    if result[["trade_date", "contract_code"]].isna().any().any():
        raise ValueError("标准化期权日行情缺少交易日或合约代码")
    if result[["trade_date", "contract_code"]].duplicated().any():
        raise ValueError("期权日行情存在重复交易日和合约代码")
    if result["option_type"].isna().any():
        raise ValueError("期权日行情包含合约主表无法识别的代码")
    return result.sort_values(["trade_date", "contract_code"]).reset_index(drop=True)


def normalize_underlying_daily(
    raw: pd.DataFrame, retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """把 Tushare ETF 日线转换为项目统一价格和数量单位。"""

    required = {
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    }
    missing = _missing_columns(raw, required)
    if missing:
        raise ValueError(f"510300日线缺少字段：{missing}")
    selected = raw.loc[raw["ts_code"].astype(str).eq("510300.SH")].copy()
    selected["date"] = pd.to_datetime(
        selected["trade_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    for column in ("open", "high", "low", "close", "vol", "amount"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    result = pd.DataFrame(
        {
            "date": selected["date"],
            "open": selected["open"],
            "high": selected["high"],
            "low": selected["low"],
            "close": selected["close"],
            "volume": selected["vol"] * 100.0,
            "amount": selected["amount"] * 1000.0,
            "symbol": "510300.SH",
            "source": "tushare_proxy.fund_daily",
            "volume_unit": "share",
            "amount_unit": "CNY",
            "retrieved_at": pd.Timestamp(retrieved_at),
        }
    )
    if result[["date", "open", "high", "low", "close"]].isna().any().any():
        raise ValueError("510300标准化日线存在关键空值")
    return result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def merge_underlying_reference(
    existing: pd.DataFrame, downloaded: pd.DataFrame, price_tolerance: float = 0.000001
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """校验重叠价格后，只补充缺失交易日。"""

    left = existing.copy()
    right = downloaded.copy()
    for frame in (left, right):
        if "retrieved_at" in frame.columns:
            frame["retrieved_at"] = frame["retrieved_at"].astype(str)
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    overlap = left[["date", "close"]].merge(
        right[["date", "close"]], on="date", suffixes=("_existing", "_downloaded")
    )
    differences = (overlap["close_existing"] - overlap["close_downloaded"]).abs()
    maximum_difference = float(differences.max()) if len(differences) else np.nan
    if len(differences) and maximum_difference > price_tolerance:
        raise ValueError(
            f"510300重叠收盘价不一致，最大绝对差={maximum_difference:.8f}"
        )
    existing_dates = set(left["date"].dropna())
    additions = right.loc[~right["date"].isin(existing_dates)].copy()
    result = pd.concat([left, additions], ignore_index=True, sort=False)
    result = result.sort_values("date").drop_duplicates("date", keep="first").reset_index(
        drop=True
    )
    return result, {
        "overlap_day_count": int(len(overlap)),
        "added_day_count": int(len(additions)),
        "maximum_overlap_close_difference": maximum_difference,
    }


def normalize_sse_risk_indicators(
    records: list[dict[str, object]], retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """标准化上交所逐合约希腊字母和隐含波动率。"""

    raw = pd.DataFrame(records)
    missing = _missing_columns(raw, SSE_RISK_REQUIRED)
    if missing:
        raise ValueError(f"上交所风险指标缺少字段：{missing}")
    selected = raw.loc[
        raw["CONTRACT_ID"].astype(str).str.startswith("510300", na=False)
    ].copy()
    if selected.empty:
        raise ValueError("上交所风险指标中没有510300合约")
    option_mapping = {"认购": "C", "认沽": "P", "C": "C", "P": "P"}
    result = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                selected["TRADE_DATE"], errors="coerce"
            ).dt.normalize(),
            "contract_code": selected["SECURITY_ID"].astype(str).str.strip()
            + ".SH",
            "exchange_contract_id": selected["CONTRACT_ID"].astype(str).str.strip(),
            "contract_symbol": selected["CONTRACT_SYMBOL"].astype(str).str.strip(),
            "option_type": selected["CONTRACT_TYPE"].astype(str).str.strip().map(option_mapping),
        }
    )
    numeric_mapping = {
        "DELTA_VALUE": "delta",
        "THETA_VALUE": "theta",
        "GAMMA_VALUE": "gamma",
        "VEGA_VALUE": "vega",
        "RHO_VALUE": "rho",
        "IMPLC_VOLATLTY": "implied_volatility",
    }
    for source, target in numeric_mapping.items():
        result[target] = pd.to_numeric(selected[source], errors="coerce")
    result["source"] = "sse.commonQuery.option_risk_indicator"
    result["retrieved_at"] = pd.Timestamp(retrieved_at)
    if result[["trade_date", "contract_code", "option_type"]].isna().any().any():
        raise ValueError("上交所风险指标存在关键空值")
    if result[["trade_date", "contract_code"]].duplicated().any():
        raise ValueError("上交所风险指标存在重复交易日和合约代码")
    return result.sort_values(["trade_date", "contract_code"]).reset_index(drop=True)


def parse_cffex_daily_csv(
    content: bytes, trade_date: pd.Timestamp | str, retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """解析中金所历史日行情 CSV，仅保留 IO 期权。"""

    raw = pd.read_csv(io.BytesIO(content), encoding="gb18030")
    raw = raw.dropna(axis=1, how="all")
    missing = sorted(set(CFFEX_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"中金所日行情缺少字段：{missing}")
    selected = raw.rename(columns=CFFEX_COLUMNS)
    selected["contract_code"] = selected["contract_code"].astype(str).str.strip()
    selected = selected.loc[
        selected["contract_code"].str.match(r"^IO\d{4}-[CP]-\d+(?:\.\d+)?$")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    parsed = selected["contract_code"].str.extract(
        r"^IO(?P<contract_month>\d{4})-(?P<option_type>[CP])-(?P<strike>\d+(?:\.\d+)?)$"
    )
    numeric = [
        "open",
        "high",
        "low",
        "volume",
        "turnover_10k_cny",
        "open_interest",
        "open_interest_change",
        "close",
        "settlement",
        "previous_settlement",
        "change_close",
        "change_settlement",
        "delta",
    ]
    for column in numeric:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["trade_date"] = pd.Timestamp(trade_date).normalize()
    selected["contract_month"] = parsed["contract_month"].values
    selected["option_type"] = parsed["option_type"].values
    selected["strike"] = pd.to_numeric(parsed["strike"], errors="coerce").values
    selected["contract_unit"] = 100.0
    selected["turnover_cny"] = selected["turnover_10k_cny"] * 10000.0
    selected["source"] = "cffex_official.history_daily"
    selected["retrieved_at"] = pd.Timestamp(retrieved_at)
    ordered = [
        "trade_date",
        "contract_code",
        "contract_month",
        "option_type",
        "strike",
        "contract_unit",
        "open",
        "high",
        "low",
        "close",
        "settlement",
        "previous_settlement",
        "volume",
        "turnover_10k_cny",
        "turnover_cny",
        "open_interest",
        "open_interest_change",
        "change_close",
        "change_settlement",
        "delta",
        "source",
        "retrieved_at",
    ]
    return selected[ordered].sort_values("contract_code").reset_index(drop=True)


def parse_cffex_month_zip(
    content: bytes,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    retrieved_at: pd.Timestamp | str,
) -> pd.DataFrame:
    """解析一个中金所月度压缩包。"""

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            match = re.search(r"(?P<date>\d{8})_1\.csv$", name)
            if not match:
                continue
            trade_date = pd.to_datetime(match.group("date"), format="%Y%m%d")
            if not start_date <= trade_date <= end_date:
                continue
            frame = parse_cffex_daily_csv(
                archive.read(name), trade_date=trade_date, retrieved_at=retrieved_at
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "contract_code"]
    ).reset_index(drop=True)


def dataset_quality(
    data: pd.DataFrame,
    expected_dates: Iterable[pd.Timestamp | str],
    code_column: str = "contract_code",
) -> dict[str, object]:
    """生成不依赖收益标签的数据覆盖证据。"""

    expected = {
        pd.Timestamp(value).normalize()
        for value in expected_dates
        if not pd.isna(value)
    }
    if data.empty:
        return {
            "row_count": 0,
            "contract_count": 0,
            "trading_day_count": 0,
            "first_trade_date": None,
            "last_trade_date": None,
            "date_coverage": 0.0,
            "missing_dates": [str(value.date()) for value in sorted(expected)],
        }
    actual_dates = set(pd.to_datetime(data["trade_date"]).dt.normalize())
    missing_dates = sorted(expected.difference(actual_dates))
    return {
        "row_count": int(len(data)),
        "contract_count": int(data[code_column].nunique()),
        "trading_day_count": int(len(actual_dates)),
        "first_trade_date": str(min(actual_dates).date()),
        "last_trade_date": str(max(actual_dates).date()),
        "date_coverage": float(len(expected & actual_dates) / max(len(expected), 1)),
        "missing_dates": [str(value.date()) for value in missing_dates],
    }


def validate_zip(content: bytes) -> None:
    """拒绝把错误网页缓存成中金所压缩包。"""

    if not zipfile.is_zipfile(io.BytesIO(content)):
        raise ValueError("中金所月度响应不是有效ZIP文件")


def safe_file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0
