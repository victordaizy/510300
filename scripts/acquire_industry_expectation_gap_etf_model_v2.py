"""补齐行业预期差—ETF模型V2截至2026-08-18的可得数据。

所有输出均写入V2独立文件，不覆盖V1原始数据或冻结结果。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_000300_official_snapshot import (  # noqa: E402
    normalize_industry,
    request_json,
)
from scripts.download_510300_option_research_data import load_proxy  # noqa: E402
from scripts.download_csi300_component_moneyflow_tushare import (  # noqa: E402
    normalize_moneyflow,
)
from scripts.download_daily_flow_data_tushare import (  # noqa: E402
    normalize_fund_share,
    normalize_margin,
)


CONFIG_FILE = ROOT / "config" / "industry_expectation_gap_etf_model_v2.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")

CITIC_L1_CODE_TO_NAME = {
    "CI005001.CI": "石油石化",
    "CI005002.CI": "煤炭",
    "CI005003.CI": "有色金属",
    "CI005004.CI": "电力及公用事业",
    "CI005005.CI": "钢铁",
    "CI005006.CI": "基础化工",
    "CI005007.CI": "建筑",
    "CI005008.CI": "建材",
    "CI005009.CI": "轻工制造",
    "CI005010.CI": "机械",
    "CI005011.CI": "电力设备及新能源",
    "CI005012.CI": "国防军工",
    "CI005013.CI": "汽车",
    "CI005014.CI": "商贸零售",
    "CI005015.CI": "消费者服务",
    "CI005016.CI": "家电",
    "CI005017.CI": "纺织服装",
    "CI005018.CI": "医药",
    "CI005019.CI": "食品饮料",
    "CI005020.CI": "农林牧渔",
    "CI005021.CI": "银行",
    "CI005022.CI": "非银行金融",
    "CI005023.CI": "房地产",
    "CI005024.CI": "交通运输",
    "CI005025.CI": "电子",
    "CI005026.CI": "通信",
    "CI005027.CI": "计算机",
    "CI005028.CI": "传媒",
    "CI005029.CI": "综合",
    "CI005030.CI": "综合金融",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _query_with_retry(call, label: str, attempts: int = 4) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = call()
            if result is None or result.empty:
                raise ValueError("返回空数据")
            return result
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{label}失败：{type(error).__name__}: {error}")


def normalize_constituent_extension(
    daily_by_date: dict[pd.Timestamp, pd.DataFrame],
    factor_by_date: dict[pd.Timestamp, pd.DataFrame],
    symbols: list[str],
    prior_rows: pd.DataFrame,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """把按日全市场行情和复权因子标准化为300只成分股增量面板。"""

    previous = prior_rows.copy()
    previous["date"] = pd.to_datetime(previous["date"], errors="coerce")
    previous = previous.sort_values(["con_code", "date"]).groupby(
        "con_code", as_index=False
    ).tail(1)
    previous = previous.set_index("con_code")
    rows: list[dict[str, Any]] = []
    for date in sorted(daily_by_date):
        raw = daily_by_date[date].copy()
        factors = factor_by_date[date].copy()
        required_daily = {
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        }
        required_factor = {"ts_code", "trade_date", "adj_factor"}
        if missing := sorted(required_daily.difference(raw.columns)):
            raise ValueError(f"{date.date()}日线缺少字段：{missing}")
        if missing := sorted(required_factor.difference(factors.columns)):
            raise ValueError(f"{date.date()}复权因子缺少字段：{missing}")
        raw = raw.loc[raw["ts_code"].astype(str).isin(symbols)].copy()
        factors = factors.loc[factors["ts_code"].astype(str).isin(symbols)].copy()
        raw = raw.drop_duplicates("ts_code", keep="last").set_index("ts_code")
        factors = factors.drop_duplicates("ts_code", keep="last").set_index("ts_code")
        current_rows: list[dict[str, Any]] = []
        for symbol in symbols:
            has_trade = symbol in raw.index
            prior = previous.loc[symbol] if symbol in previous.index else None
            factor = (
                pd.to_numeric(factors.loc[symbol, "adj_factor"], errors="coerce")
                if symbol in factors.index
                else (pd.to_numeric(prior["adj_factor"], errors="coerce") if prior is not None else np.nan)
            )
            if pd.isna(factor) or float(factor) <= 0:
                raise ValueError(f"{symbol}在{date.date()}缺少有效复权因子")
            if has_trade:
                quote = raw.loc[symbol]
                price_values = {
                    name: float(pd.to_numeric(quote[name], errors="coerce"))
                    for name in ("open", "high", "low", "close")
                }
                if any(not np.isfinite(value) or value <= 0 for value in price_values.values()):
                    raise ValueError(f"{symbol}在{date.date()}存在非法OHLC")
                volume = float(pd.to_numeric(quote["vol"], errors="coerce")) * 100.0
                amount = float(pd.to_numeric(quote["amount"], errors="coerce")) * 1000.0
                price_source = "tushare_proxy.daily_trade_date_batch"
                adjustment_source = "tushare_proxy.adj_factor_trade_date_batch"
            else:
                if prior is None:
                    raise ValueError(f"{symbol}在{date.date()}停牌且没有前值")
                close = float(prior["raw_close"])
                price_values = {name: close for name in ("open", "high", "low", "close")}
                volume = 0.0
                amount = 0.0
                price_source = "tushare_proxy.daily_forward_fill_suspension_v2"
                adjustment_source = "tushare_proxy.adj_factor_current_or_forward_fill_v2"
            item = {
                "date": pd.Timestamp(date),
                "con_code": symbol,
                "raw_open": price_values["open"],
                "raw_high": price_values["high"],
                "raw_low": price_values["low"],
                "raw_close": price_values["close"],
                "total_return_open": price_values["open"] * float(factor),
                "total_return_high": price_values["high"] * float(factor),
                "total_return_low": price_values["low"] * float(factor),
                "total_return_close": price_values["close"] * float(factor),
                "volume": volume,
                "amount": amount,
                "adj_factor": float(factor),
                "price_source": price_source,
                "adjustment_source": adjustment_source,
                "retrieved_at": retrieved_at,
                "is_index_member": True,
                "is_suspended": not has_trade,
                "outstanding_share": np.nan,
                "total_market_cap_cny": np.nan,
                "market_cap_asof_date": pd.NaT,
            }
            current_rows.append(item)
        current = pd.DataFrame(current_rows)
        rows.extend(current_rows)
        previous = current.set_index("con_code")
    result = pd.DataFrame(rows).sort_values(["date", "con_code"]).reset_index(drop=True)
    counts = result.groupby("date")["con_code"].nunique()
    if counts.empty or not counts.eq(300).all():
        raise ValueError(f"成分股增量每日覆盖异常：{counts.to_dict()}")
    if result[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股增量存在重复证券日期")
    if result["total_return_close"].isna().any() or result["total_return_close"].le(0).any():
        raise ValueError("成分股增量总收益价格无效")
    return result


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(config["as_of_date"])
    retrieved_at = datetime.now(TIMEZONE)
    acquired = {name: ROOT / path for name, path in config["acquired_inputs"].items()}
    acquired.pop("acquisition_status")
    status_file = ROOT / config["acquired_inputs"]["acquisition_status"]
    token = ""
    try:
        historical_daily = pd.read_parquet(ROOT / config["inputs"]["historical_constituent_daily"])
        historical_daily["date"] = pd.to_datetime(historical_daily["date"])
        last_historical_date = historical_daily["date"].max()
        calendar_files = sorted((ROOT / "data" / "reference").glob("sse_trade_calendar_*.csv"))
        if not calendar_files:
            raise FileNotFoundError("缺少上交所交易日历")
        calendar = pd.read_csv(calendar_files[-1])
        date_column = "date" if "date" in calendar.columns else "trade_date"
        trading_dates = pd.to_datetime(calendar[date_column], errors="coerce")
        extension_dates = sorted(
            pd.Timestamp(value)
            for value in trading_dates.loc[
                trading_dates.gt(last_historical_date) & trading_dates.le(as_of)
            ].dropna().unique()
        )
        if not extension_dates:
            raise ValueError("没有需要补齐的成分股交易日")

        weights = pd.read_parquet(ROOT / config["inputs"]["historical_weights"])
        weights["trade_date"] = pd.to_datetime(weights["trade_date"])
        latest_weight_date = weights.loc[weights["trade_date"].le(as_of), "trade_date"].max()
        symbols = sorted(
            weights.loc[weights["trade_date"].eq(latest_weight_date), "con_code"]
            .astype(str)
            .unique()
        )
        if len(symbols) != 300:
            raise ValueError(f"最新权重快照成分数不是300：{len(symbols)}")

        pro, token, endpoint = load_proxy()
        daily_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        factor_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        for date in extension_dates:
            text_date = date.strftime("%Y%m%d")
            daily_by_date[date] = _query_with_retry(
                lambda text_date=text_date: pro.daily(trade_date=text_date),
                f"{text_date}全市场日线",
            )
            time.sleep(0.8)
            factor_by_date[date] = _query_with_retry(
                lambda text_date=text_date: pro.adj_factor(trade_date=text_date),
                f"{text_date}全市场复权因子",
            )
            time.sleep(0.8)
        extension = normalize_constituent_extension(
            daily_by_date,
            factor_by_date,
            symbols,
            historical_daily,
            retrieved_at,
        )
        atomic_parquet(extension, acquired["constituent_extension"])

        flow_start = pd.Timestamp("2026-08-13")
        flow_dates = sorted(
            pd.Timestamp(value)
            for value in trading_dates.loc[
                trading_dates.between(flow_start, as_of)
            ].dropna().unique()
        )
        flow_parts: list[pd.DataFrame] = []
        for date in flow_dates:
            text_date = date.strftime("%Y%m%d")
            part = _query_with_retry(
                lambda text_date=text_date: pro.moneyflow(trade_date=text_date),
                f"{text_date}全市场订单分类资金流",
            )
            flow_parts.append(part.loc[part["ts_code"].astype(str).isin(symbols)])
            time.sleep(0.8)
        moneyflow = normalize_moneyflow(pd.concat(flow_parts, ignore_index=True))
        moneyflow["source"] = "tushare_proxy.moneyflow_trade_date_batch"
        moneyflow["retrieved_at"] = retrieved_at
        atomic_parquet(moneyflow, acquired["component_moneyflow_extension"])

        fund_raw = _query_with_retry(
            lambda: pro.query(
                "fund_share",
                ts_code="510300.SH",
                start_date=flow_start.strftime("%Y%m%d"),
                end_date=as_of.strftime("%Y%m%d"),
            ),
            "510300基金份额",
        )
        time.sleep(0.8)
        margin_raw = _query_with_retry(
            lambda: pro.query(
                "margin_detail",
                ts_code="510300.SH",
                start_date=flow_start.strftime("%Y%m%d"),
                end_date=as_of.strftime("%Y%m%d"),
            ),
            "510300融资融券",
        )
        fund_share = normalize_fund_share(fund_raw, retrieved_at)
        margin = normalize_margin(margin_raw, retrieved_at)
        atomic_parquet(fund_share, acquired["fund_share_extension"])
        atomic_parquet(margin, acquired["margin_extension"])

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        cics_payloads = {
            level: request_json(
                session,
                f"/index/weight/industry-weight-two-new/000300?cicsType={level}",
            )
            for level in range(1, 5)
        }
        cics = normalize_industry(cics_payloads, retrieved_at.isoformat())
        if cics["date"].max() != as_of:
            raise ValueError(f"中证CICS快照日期为{cics['date'].max().date()}，不是{as_of.date()}")
        atomic_parquet(cics, acquired["cics_snapshot"])

        citic_parts: list[pd.DataFrame] = []
        for date in extension_dates:
            text_date = date.strftime("%Y%m%d")
            part = _query_with_retry(
                lambda text_date=text_date: pro.ci_daily(trade_date=text_date),
                f"{text_date}中信行业指数",
            )
            citic_parts.append(
                part.loc[part["ts_code"].astype(str).isin(CITIC_L1_CODE_TO_NAME)].copy()
            )
            time.sleep(0.8)
        citic = pd.concat(citic_parts, ignore_index=True)
        citic["industry_l1"] = citic["ts_code"].astype(str).map(CITIC_L1_CODE_TO_NAME)
        citic["source"] = "tushare_proxy.ci_daily"
        citic["retrieved_at"] = retrieved_at
        atomic_parquet(citic, acquired["citic_index_crosscheck"])

        v3_file = ROOT / "data" / "raw" / "forward" / "v3_forward_2_constituent_close.parquet"
        close_crosscheck: dict[str, Any] = {"status": "NOT_AVAILABLE"}
        if v3_file.exists():
            v3 = pd.read_parquet(v3_file)
            v3["date"] = pd.to_datetime(v3["date"])
            comparison = extension.loc[extension["date"].eq(as_of), ["con_code", "raw_close"]].merge(
                v3.loc[v3["date"].eq(as_of), ["con_code", "raw_close"]],
                on="con_code",
                suffixes=("_tushare", "_sina"),
                how="inner",
                validate="one_to_one",
            )
            difference = (comparison["raw_close_tushare"] - comparison["raw_close_sina"]).abs()
            close_crosscheck = {
                "status": "PASS" if len(comparison) == 300 and float(difference.max()) <= 0.011 else "WARN",
                "comparable_count": int(len(comparison)),
                "maximum_absolute_close_difference": float(difference.max()) if len(difference) else None,
                "exact_match_ratio": float(difference.eq(0).mean()) if len(difference) else None,
                "independent_source": "sina.hq.batch_quote",
            }

        files = {name: path for name, path in acquired.items()}
        report = {
            "version": config["version"],
            "status": "PASS",
            "as_of_date": config["as_of_date"],
            "retrieved_at": retrieved_at.isoformat(),
            "api_host": endpoint.split("//", 1)[-1],
            "credential_persisted": False,
            "constituent_extension": {
                "dates": [date.date().isoformat() for date in extension_dates],
                "rows": int(len(extension)),
                "symbols_per_date": {
                    str(date.date()): int(count)
                    for date, count in extension.groupby("date")["con_code"].nunique().items()
                },
                "suspended_rows": int(extension["is_suspended"].sum()),
                "latest_weight_snapshot_date": latest_weight_date.date().isoformat(),
                "sina_close_crosscheck": close_crosscheck,
            },
            "component_moneyflow_extension": {
                "dates": sorted(moneyflow["date"].dt.date.astype(str).unique().tolist()),
                "rows": int(len(moneyflow)),
                "symbol_count": int(moneyflow["ts_code"].nunique()),
                "semantics": "ACTIVE_ORDER_CLASSIFICATION_NOT_ETF_FLOW_NOT_INVESTOR_IDENTITY",
            },
            "fund_share_latest": fund_share["date"].max().date().isoformat(),
            "margin_latest": margin["date"].max().date().isoformat(),
            "cics_snapshot_date": cics["date"].max().date().isoformat(),
            "cics_l1_weight_sum_percent": float(
                cics.loc[cics["industry_level"].eq(1), "weight_pct"].sum()
            ),
            "citic_index_dates": sorted(
                pd.to_datetime(citic["trade_date"]).dt.date.astype(str).unique().tolist()
            ),
            "files": {
                path.relative_to(ROOT).as_posix(): {
                    "rows": int(len(pd.read_parquet(path))),
                    "sha256": sha256(path),
                }
                for path in files.values()
            },
        }
        atomic_json(report, status_file)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        message = str(exc).replace(token, "[REDACTED]") if token else str(exc)
        atomic_json(
            {
                "version": config.get("version"),
                "status": "FAILED",
                "as_of_date": config.get("as_of_date"),
                "retrieved_at": retrieved_at.isoformat(),
                "error": f"{type(exc).__name__}: {message}",
            },
            status_file,
        )
        print(f"V2数据补齐失败：{type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

