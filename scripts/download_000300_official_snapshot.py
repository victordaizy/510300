"""下载中证指数官网的沪深300估值历史与当前权重截面。

本脚本仅保存提供方原始含义，不将当前截面回填为历史点时数据。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "constituents"
VALUATION_DIR = PROJECT_ROOT / "data" / "raw" / "valuation"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "000300_official_snapshot_quality.json"
RAW_SNAPSHOT_FILE = OUTPUT_DIR / "000300_official_snapshot_raw.json"

BASE_URL = "https://www.csindex.com.cn/csindex-home"
FACTSHEET_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/"
    "detail/files/zh_CN/000300factsheet.pdf"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(session: requests.Session, path: str, maximum_retries: int = 3) -> dict[str, Any]:
    final_error: Exception | None = None
    url = f"{BASE_URL}{path}"
    for attempt in range(1, maximum_retries + 1):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code")) != "200" or not payload.get("success"):
                raise RuntimeError(f"提供方返回失败状态：{payload.get('code')} {payload.get('msg')}")
            return payload
        except Exception as exc:
            final_error = exc
            if attempt < maximum_retries:
                time.sleep(attempt)
    raise RuntimeError(f"中证指数接口请求失败 {url}：{final_error}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_parquet_atomic(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def download_binary(session: requests.Session, url: str, path: Path) -> None:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError("官方指数事实表不是有效PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(path)


def normalize_industry(payloads: dict[int, dict[str, Any]], retrieved_at: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for level, payload in payloads.items():
        body = payload["data"]
        frame = pd.DataFrame(body.get("industryWeightList") or [])
        if frame.empty:
            raise RuntimeError(f"中证行业{level}级权重为空")
        frame = frame.rename(
            columns={
                "tradeDate": "date",
                "csiType": "industry_name_cn",
                "csiTypeEn": "industry_name_en",
                "weightPct": "weight_pct_display",
                "preciseWeight": "weight_pct",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
        frame["industry_level"] = level
        frame["weight_pct_display"] = pd.to_numeric(frame["weight_pct_display"], errors="raise")
        frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="raise")
        frame["index_code"] = "000300"
        frame["source"] = f"csindex.industry-weight-two-new.cicsType={level}"
        frame["retrieved_at"] = retrieved_at
        frames.append(
            frame[
                [
                    "date",
                    "index_code",
                    "industry_level",
                    "industry_name_cn",
                    "industry_name_en",
                    "weight_pct_display",
                    "weight_pct",
                    "source",
                    "retrieved_at",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def normalize_top10(payload: dict[str, Any], retrieved_at: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload["data"].get("weightList") or [])
    if len(frame) != 10:
        raise RuntimeError(f"官方前十大权重条数异常：{len(frame)}")
    frame = frame.rename(
        columns={
            "tradeDate": "date",
            "securityCode": "stock_code",
            "securityName": "stock_name",
            "securityNameEn": "stock_name_en",
            "marketNameCn": "exchange_cn",
            "marketNameEn": "exchange_en",
            "cics1stName": "industry_l1_cn",
            "cics1stNameEn": "industry_l1_en",
            "cics2ndName": "industry_l2_cn",
            "cics2ndNameEn": "industry_l2_en",
            "weight": "weight_pct_display",
            "preciseWeight": "weight_pct",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    frame["weight_pct_display"] = pd.to_numeric(frame["weight_pct_display"], errors="raise")
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="raise")
    frame["source"] = "csindex.top10new"
    frame["retrieved_at"] = retrieved_at
    columns = [
        "date",
        "indexCode",
        "stock_code",
        "stock_name",
        "stock_name_en",
        "exchange_cn",
        "exchange_en",
        "industry_l1_cn",
        "industry_l1_en",
        "industry_l2_cn",
        "industry_l2_en",
        "weight_pct_display",
        "weight_pct",
        "source",
        "retrieved_at",
    ]
    return frame[columns].rename(columns={"indexCode": "index_code"})


def normalize_market_weights(
    payload: dict[str, Any],
    category_type: str,
    source: str,
    retrieved_at: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(payload["data"].get("marketWeightList") or [])
    if frame.empty:
        raise RuntimeError(f"官方{category_type}权重为空")
    frame = frame.rename(
        columns={
            "tradeDate": "date",
            "marketNameCn": "category_name_cn",
            "marketNameEn": "category_name_en",
            "weight": "weight_pct",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="raise")
    frame["index_code"] = "000300"
    frame["category_type"] = category_type
    frame["source"] = source
    frame["retrieved_at"] = retrieved_at
    return frame[
        [
            "date",
            "index_code",
            "category_type",
            "category_name_cn",
            "category_name_en",
            "weight_pct",
            "source",
            "retrieved_at",
        ]
    ]


def normalize_official_pe(payload: dict[str, Any], retrieved_at: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload.get("data") or [])
    required = {"tradeDate", "peg"}
    missing = required.difference(frame.columns)
    if frame.empty or missing:
        raise RuntimeError(f"官方PE历史为空或缺少字段：{sorted(missing)}")
    frame = frame.rename(
        columns={
            "tradeDate": "date",
            "indexName": "index_name_cn",
            "indexNameEn": "index_name_en",
            "peg": "pe_official",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    frame["pe_official"] = pd.to_numeric(frame["pe_official"], errors="raise")
    frame["index_code"] = "000300"
    frame["provider_original_field"] = "peg"
    frame["source"] = "csindex.indexCsiDsPe"
    frame["retrieved_at"] = retrieved_at
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if (frame["pe_official"] <= 0).any():
        raise RuntimeError("官方PE存在非正值")
    return frame[
        [
            "date",
            "index_code",
            "index_name_cn",
            "index_name_en",
            "pe_official",
            "provider_original_field",
            "source",
            "retrieved_at",
        ]
    ]


def normalize_feature_snapshot(payload: dict[str, Any], retrieved_at: str) -> pd.DataFrame:
    """规范化中证官网指数特征截面。

    ``calMkv`` 保留为“计算用市值”语义，单位与官方事实表的
    “指数市值（亿元）”一致，不改称为成分股全口径总市值。
    """

    body = payload["data"]
    return pd.DataFrame(
        [
            {
                "date": pd.to_datetime(body["tradeDate"], format="%Y%m%d", errors="raise"),
                "index_code": body["indexCode"],
                "constituent_count": int(float(body["consNum"])),
                "calculated_market_cap_sum_cny_100m": float(body["calMkvSum"]),
                "calculated_market_cap_max_cny_100m": float(body["calMkvMax"]),
                "calculated_market_cap_min_cny_100m": float(body["calMkvMin"]),
                "calculated_market_cap_average_cny_100m": float(body["calMkvAvg"]),
                "calculated_market_cap_median_cny_100m": float(body["calMkvMedian"]),
                "unit": "CNY_100_million",
                "provider_field_prefix": "calMkv",
                "source": "csindex.index-feature",
                "retrieved_at": retrieved_at,
            }
        ]
    )


def parse_factsheet_snapshot(path: Path, retrieved_at: str) -> pd.DataFrame:
    """从官方事实表提取带有截面日和单位的基本面数值。"""

    with pdfplumber.open(path) as document:
        text = "\n".join((page.extract_text() or "") for page in document.pages)

    def required(pattern: str, name: str) -> str:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            raise RuntimeError(f"官方事实表缺少可解析字段：{name}")
        return match.group(1)

    date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not date_match:
        raise RuntimeError("官方事实表缺少可解析截面日")
    snapshot_date = pd.Timestamp(
        year=int(date_match.group(1)),
        month=int(date_match.group(2)),
        day=int(date_match.group(3)),
    )
    market_pattern = (
        r"成分股\s+个股总市值最大\s+(\d+)\s+"
        r"(\d+)\s+总市值\s+个股总市值最小\s+(\d+)\s+"
        r"指数市值\s+(\d+)\s+个股总市值平均\s+(\d+)"
    )
    market_match = re.search(market_pattern, text, flags=re.MULTILINE)
    if not market_match:
        raise RuntimeError("官方事实表市值表无法解析")
    maximum, constituent_sum, minimum, index_market_cap, average = map(
        float, market_match.groups()
    )
    return pd.DataFrame(
        [
            {
                "date": snapshot_date,
                "index_code": "000300",
                "rolling_pe": float(required(r"滚动市盈率\s+([\d.]+)", "滚动市盈率")),
                "pb": float(required(r"市净率\s+([\d.]+)", "市净率")),
                "dividend_yield_pct": float(required(r"股息率\s+([\d.]+)%", "股息率")),
                "volatility_1y_annualized_pct": float(
                    required(r"1年年化\s+([\d.]+)%\s+滚动市盈率", "1年年化波动率")
                ),
                "volatility_3y_annualized_pct": float(
                    required(r"3年年化\s+([\d.]+)%\s+市净率", "3年年化波动率")
                ),
                "volatility_5y_annualized_pct": float(
                    required(r"5年年化\s+([\d.]+)%\s+股息率", "5年年化波动率")
                ),
                "constituent_total_market_cap_cny_100m": constituent_sum,
                "index_calculated_market_cap_cny_100m": index_market_cap,
                "constituent_market_cap_max_cny_100m": maximum,
                "constituent_market_cap_min_cny_100m": minimum,
                "constituent_market_cap_average_cny_100m": average,
                "unit_market_cap": "CNY_100_million",
                "source": "csindex.000300factsheet.pdf",
                "retrieved_at": retrieved_at,
            }
        ]
    )


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    timezone = ZoneInfo(config["project"]["timezone"])
    retrieved_at = datetime.now(timezone).isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    paths = {
        "basic": "/indexInfo/index-basic-info/000300",
        "feature": "/indexInfo/index-feature/000300",
        "top10": "/index/weight/top10new/000300",
        "exchange": "/index/weight/market-weight/000300",
        "board": "/index/weight/market-weight-add/000300",
        "performance": "/perf/get-index-yield-item/000300",
        "annualized": "/perf/get-index-yield-item-nianHua/000300",
        "pe": "/perf/indexCsiDsPe?indexCode=000300",
    }
    payloads = {name: request_json(session, path) for name, path in paths.items()}
    industry_payloads = {
        level: request_json(session, f"/index/weight/industry-weight-two-new/000300?cicsType={level}")
        for level in range(1, 5)
    }
    raw_snapshot = {
        "retrieved_at": retrieved_at,
        "endpoint_base": BASE_URL,
        "payloads": payloads,
        "industry_payloads": {str(level): value for level, value in industry_payloads.items()},
    }
    write_json_atomic(RAW_SNAPSHOT_FILE, raw_snapshot)

    industry = normalize_industry(industry_payloads, retrieved_at)
    top10 = normalize_top10(payloads["top10"], retrieved_at)
    exchange = normalize_market_weights(
        payloads["exchange"], "exchange", "csindex.market-weight", retrieved_at
    )
    board = normalize_market_weights(
        payloads["board"], "listing_board", "csindex.market-weight-add", retrieved_at
    )
    market = pd.concat([exchange, board], ignore_index=True)
    official_pe = normalize_official_pe(payloads["pe"], retrieved_at)
    feature_snapshot = normalize_feature_snapshot(payloads["feature"], retrieved_at)

    industry_file = OUTPUT_DIR / "000300_industry_weights_current.parquet"
    top10_file = OUTPUT_DIR / "000300_top10_weights_current.parquet"
    market_file = OUTPUT_DIR / "000300_market_weights_current.parquet"
    pe_file = VALUATION_DIR / "000300_pe_official_raw.parquet"
    feature_file = OUTPUT_DIR / "000300_feature_current.parquet"
    factsheet_file = REFERENCE_DIR / "000300_factsheet.pdf"
    write_parquet_atomic(industry_file, industry)
    write_parquet_atomic(top10_file, top10)
    write_parquet_atomic(market_file, market)
    write_parquet_atomic(pe_file, official_pe)
    write_parquet_atomic(feature_file, feature_snapshot)
    download_binary(session, FACTSHEET_URL, factsheet_file)
    factsheet_snapshot = parse_factsheet_snapshot(factsheet_file, retrieved_at)
    factsheet_snapshot_file = OUTPUT_DIR / "000300_factsheet_metrics_current.parquet"
    write_parquet_atomic(factsheet_snapshot_file, factsheet_snapshot)

    industry_sums = {
        str(level): float(group["weight_pct"].sum())
        for level, group in industry.groupby("industry_level", sort=True)
    }
    market_sums = {
        category: float(group["weight_pct"].sum())
        for category, group in market.groupby("category_type", sort=True)
    }
    dates = {
        "feature": pd.to_datetime(payloads["feature"]["data"]["tradeDate"], format="%Y%m%d").date().isoformat(),
        "industry": industry["date"].max().date().isoformat(),
        "top10": top10["date"].max().date().isoformat(),
        "market": market["date"].max().date().isoformat(),
        "pe_last": official_pe["date"].max().date().isoformat(),
    }
    sum_checks = all(abs(value - 100.0) <= 0.05 for value in industry_sums.values()) and all(
        abs(value - 100.0) <= 0.05 for value in market_sums.values()
    )
    snapshot_dates_match = len({dates["feature"], dates["industry"], dates["top10"], dates["market"]}) == 1
    status = "PASS" if sum_checks and snapshot_dates_match else "WARN"
    factsheet_date = factsheet_snapshot.iloc[0]["date"]
    factsheet_pe = float(factsheet_snapshot.iloc[0]["rolling_pe"])
    official_pe_on_factsheet_date = official_pe.loc[
        official_pe["date"].eq(factsheet_date), "pe_official"
    ]
    factsheet_pe_match = (
        len(official_pe_on_factsheet_date) == 1
        and abs(float(official_pe_on_factsheet_date.iloc[0]) - factsheet_pe) < 1e-12
    )
    if not factsheet_pe_match:
        status = "WARN"
    files = [
        industry_file,
        top10_file,
        market_file,
        pe_file,
        feature_file,
        factsheet_snapshot_file,
        factsheet_file,
        RAW_SNAPSHOT_FILE,
    ]
    report = {
        "status": status,
        "checked_at": retrieved_at,
        "snapshot_dates": dates,
        "snapshot_dates_match": snapshot_dates_match,
        "industry_level_weight_sums_pct": industry_sums,
        "market_weight_sums_pct": market_sums,
        "top10_row_count": int(len(top10)),
        "top10_precise_weight_sum_pct": float(top10["weight_pct"].sum()),
        "provider_top10_display_sum_pct": float(payloads["top10"]["data"]["top10Sum"]),
        "official_pe": {
            "row_count": int(len(official_pe)),
            "first_date": official_pe["date"].min().date().isoformat(),
            "last_date": official_pe["date"].max().date().isoformat(),
            "provider_original_field": "peg",
            "semantic_basis": "接口路径indexCsiDsPe；不自行改写为静态PE或TTM PE。",
            "factsheet_rolling_pe_date": factsheet_date.date().isoformat(),
            "factsheet_rolling_pe": factsheet_pe,
            "official_api_pe_on_factsheet_date": (
                float(official_pe_on_factsheet_date.iloc[0])
                if len(official_pe_on_factsheet_date) == 1
                else None
            ),
            "factsheet_semantic_cross_check": factsheet_pe_match,
        },
        "factsheet_metrics": {
            **factsheet_snapshot.iloc[0].drop(labels=["date", "retrieved_at"]).to_dict(),
            "date": factsheet_date.date().isoformat(),
        },
        "feature_snapshot": payloads["feature"]["data"],
        "annualized_provider_fields": payloads["annualized"]["data"],
        "files": {
            path.relative_to(PROJECT_ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "limitations": [
            "行业、市场、板块和前十大权重是当前截面，不是历史点时序列。",
            "官方PE接口原字段名为peg；已用事实表的滚动市盈率做同日语义交叉验证，仍保留原字段名。",
            "年化字段保留在原始快照中，未经产品页语义复核前不当作策略因子。",
        ],
    }
    write_json_atomic(REPORT_FILE, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
