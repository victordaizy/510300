"""尾部风险研究补充数据的标准化与跨源检查。"""

from __future__ import annotations

import re

import pandas as pd


def _number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str).str.replace(",", "", regex=False).replace({"-": None, "": None}),
        errors="coerce",
    )


def normalize_cffex_proxy_daily(
    raw: pd.DataFrame, retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """标准化代理返回的中金所IO期权日线。"""

    required = {
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
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"中金所代理日线缺少字段：{missing}")
    data = raw.loc[raw["ts_code"].astype(str).str.startswith("IO")].copy()
    if data.empty:
        raise ValueError("中金所代理日线没有IO合约")
    data["contract_code"] = data["ts_code"].astype(str).str.replace(
        r"\.CFX$", "", regex=True
    )
    parsed = data["contract_code"].str.extract(
        r"^IO(?P<contract_month>\d{4})-(?P<option_type>[CP])-(?P<strike>\d+(?:\.\d+)?)$"
    )
    if parsed.isna().any().any():
        examples = data.loc[parsed.isna().any(axis=1), "contract_code"].head(10).tolist()
        raise ValueError(f"无法解析IO合约代码：{examples}")
    data["trade_date"] = pd.to_datetime(
        data["trade_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    numeric_mapping = {
        "pre_settle": "previous_settlement",
        "pre_close": "previous_close",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "settle": "settlement",
        "vol": "volume",
        "amount": "turnover_10k_cny",
        "oi": "open_interest",
    }
    result = pd.DataFrame(
        {
            "trade_date": data["trade_date"],
            "contract_code": data["contract_code"],
            "contract_month": parsed["contract_month"].values,
            "option_type": parsed["option_type"].values,
            "strike": pd.to_numeric(parsed["strike"], errors="coerce").values,
            "exchange": data["exchange"].astype(str),
        }
    )
    for source, target in numeric_mapping.items():
        result[target] = pd.to_numeric(data[source], errors="coerce").values
    result["source"] = "tushare_proxy.opt_daily.CFFEX"
    result["retrieved_at"] = str(pd.Timestamp(retrieved_at))
    if result[["trade_date", "contract_code"]].duplicated().any():
        raise ValueError("中金所代理日线存在重复键")
    return result.sort_values(["trade_date", "contract_code"]).reset_index(drop=True)


def normalize_sse_daily_statistics(
    records: list[dict[str, object]], retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """标准化上交所股票期权每日汇总统计，仅保留510300。"""

    raw = pd.DataFrame(records)
    required = {
        "CONTRACT_VOLUME",
        "CALL_VOLUME",
        "LEAVES_QTY",
        "CP_RATE",
        "PUT_VOLUME",
        "TRADE_DATE",
        "TOTAL_MONEY",
        "TOTAL_VOLUME",
        "SECURITY_CODE",
        "LEAVES_CALL_QTY",
        "LEAVES_PUT_QTY",
        "SECURITY_ABBR",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"上交所每日汇总缺少字段：{missing}")
    data = raw.loc[raw["SECURITY_CODE"].astype(str).eq("510300")].copy()
    if len(data) != 1:
        raise ValueError(f"上交所每日汇总中510300记录数应为1，实际为{len(data)}")
    result = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(data["TRADE_DATE"], errors="coerce").dt.normalize(),
            "underlying_code": "510300.SH",
            "security_abbreviation": data["SECURITY_ABBR"].astype(str),
            "source": "sse.commonQuery.option_daily_statistics",
            "retrieved_at": str(pd.Timestamp(retrieved_at)),
        }
    )
    mapping = {
        "CONTRACT_VOLUME": "contract_count",
        "CALL_VOLUME": "call_volume",
        "PUT_VOLUME": "put_volume",
        "TOTAL_VOLUME": "total_volume",
        "TOTAL_MONEY": "turnover_10k_cny",
        "LEAVES_QTY": "open_interest",
        "LEAVES_CALL_QTY": "call_open_interest",
        "LEAVES_PUT_QTY": "put_open_interest",
        "CP_RATE": "put_call_ratio_percent",
    }
    for source, target in mapping.items():
        result[target] = _number(data[source]).values
    return result.reset_index(drop=True)


SINA_FIELDS = [
    "summary_bid_volume",
    "summary_bid_price",
    "last_price",
    "summary_ask_price",
    "summary_ask_volume",
    "open_interest",
    "pct_change",
    "strike",
    "previous_close",
    "open",
    "upper_limit",
    "lower_limit",
    "ask5",
    "ask5_volume",
    "ask4",
    "ask4_volume",
    "ask3",
    "ask3_volume",
    "ask2",
    "ask2_volume",
    "ask1",
    "ask1_volume",
    "bid1",
    "bid1_volume",
    "bid2",
    "bid2_volume",
    "bid3",
    "bid3_volume",
    "bid4",
    "bid4_volume",
    "bid5",
    "bid5_volume",
    "quote_timestamp",
    "main_contract_flag",
    "status_code",
    "underlying_type",
    "underlying_code",
    "contract_name",
    "amplitude",
    "high",
    "low",
    "volume",
    "turnover_cny",
]


def parse_sina_batch_quotes(response_text: str) -> dict[str, list[str]]:
    """解析新浪批量期权行情响应，空字符串代表该合约已不再保留。"""

    result: dict[str, list[str]] = {}
    pattern = re.compile(r'var\s+hq_str_CON_OP_(?P<code>\d+)="(?P<data>[^"]*)";')
    for match in pattern.finditer(response_text):
        data = match.group("data")
        if data:
            result[match.group("code")] = data.split(",")
    return result


def normalize_sina_five_day_minutes(
    contract_code: str,
    day_records: list[list[dict[str, object]]],
    retrieved_at: pd.Timestamp | str,
) -> pd.DataFrame:
    """标准化新浪长期保留的单合约最后五个交易日分钟成交序列。"""

    frames = [pd.DataFrame(records) for records in day_records if records]
    if not frames:
        return pd.DataFrame(
            columns=[
                "contract_code",
                "trade_date",
                "minute_timestamp",
                "price",
                "average_price",
                "volume",
                "source",
                "retrieved_at",
                "history_usage",
            ]
        )
    data = pd.concat(frames, ignore_index=True)
    data = data.rename(
        columns={"d": "date", "i": "time", "p": "price", "a": "avg_price", "v": "volume"}
    )
    required = {"date", "time", "price", "avg_price", "volume"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{contract_code}新浪五日分钟数据缺少字段：{missing}")
    data[["date", "time"]] = data[["date", "time"]].ffill()
    trade_date = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    timestamp = pd.to_datetime(
        trade_date.dt.strftime("%Y-%m-%d") + " " + data["time"].astype(str),
        errors="coerce",
    )
    result = pd.DataFrame(
        {
            "contract_code": f"{contract_code}.SH",
            "trade_date": trade_date,
            "minute_timestamp": timestamp,
            "price": pd.to_numeric(data["price"], errors="coerce"),
            "average_price": pd.to_numeric(data["avg_price"], errors="coerce"),
            "volume": pd.to_numeric(data["volume"], errors="coerce"),
            "source": "sina.StockOptionDaylineService.getFiveDayLine",
            "retrieved_at": str(pd.Timestamp(retrieved_at)),
            "history_usage": "LAST_FIVE_TRADING_DAYS_PER_CONTRACT_ONLY",
        }
    )
    result = result.dropna(subset=["trade_date", "minute_timestamp"])
    result = result.drop_duplicates(["contract_code", "minute_timestamp"], keep="last")
    return result.sort_values("minute_timestamp").reset_index(drop=True)


def normalize_sina_option_quote(
    contract_code: str, values: list[str], retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """标准化新浪单份股票期权五档盘口。"""

    if len(values) < len(SINA_FIELDS):
        raise ValueError(f"{contract_code}新浪盘口字段不足：{len(values)}")
    row = dict(zip(SINA_FIELDS, values[: len(SINA_FIELDS)]))
    result = pd.DataFrame([row])
    result.insert(0, "contract_code", f"{contract_code}.SH")
    numeric = [
        value
        for value in SINA_FIELDS
        if value
        not in {
            "quote_timestamp",
            "main_contract_flag",
            "status_code",
            "underlying_type",
            "underlying_code",
            "contract_name",
        }
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["quote_timestamp"] = pd.to_datetime(
        result["quote_timestamp"], errors="coerce"
    )
    result["underlying_code"] = result["underlying_code"].astype(str) + ".SH"
    result["source"] = "sina.option_sse_spot_price"
    result["retrieved_at"] = str(pd.Timestamp(retrieved_at))
    if result["quote_timestamp"].isna().any():
        raise ValueError(f"{contract_code}新浪盘口时间戳无效")
    return result


def normalize_citic_industry_intervals(
    raw: pd.DataFrame, universe: set[str], retrieved_at: pd.Timestamp | str
) -> pd.DataFrame:
    """标准化中信行业点时区间，作为申万之外的独立对照。"""

    required = {
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "in_date",
        "out_date",
        "is_new",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"中信行业区间缺少字段：{missing}")
    data = raw.loc[raw["ts_code"].astype(str).isin(universe)].copy()
    data["in_date"] = pd.to_datetime(
        data["in_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    data["out_date"] = pd.to_datetime(
        data["out_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    data = data.dropna(subset=["ts_code", "l1_code", "l1_name", "in_date"])
    result = pd.DataFrame(
        {
            "con_code": data["ts_code"].astype(str),
            "industry_l1": data["l1_name"].astype(str),
            "industry_l1_code": data["l1_code"].astype(str),
            "industry_l2": data["l2_name"].astype(str),
            "industry_l2_code": data["l2_code"].astype(str),
            "industry_l3": data["l3_name"].astype(str),
            "industry_l3_code": data["l3_code"].astype(str),
            "in_date": data["in_date"],
            "out_date": data["out_date"],
            "is_current": data["is_new"].astype(str).str.upper().eq("Y"),
            "classification_usage": "POINT_IN_TIME_INTERVAL_CITIC_ALTERNATIVE",
            "source": "tushare_proxy.ci_index_member",
            "retrieved_at": str(pd.Timestamp(retrieved_at)),
        }
    )
    return result.drop_duplicates(
        ["con_code", "industry_l1_code", "in_date", "out_date"], keep="last"
    ).sort_values(["con_code", "in_date", "industry_l1_code"]).reset_index(drop=True)
