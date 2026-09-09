"""下载尾部风险研究仍可取得的冗余源、官方统计和盘口快照。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.return_tail_auxiliary_acquisition import active_members, industry_coverage
from research.return_tail_supplemental_acquisition import (
    normalize_cffex_proxy_daily,
    normalize_citic_industry_intervals,
    normalize_sina_five_day_minutes,
    normalize_sina_option_quote,
    normalize_sse_daily_statistics,
    parse_sina_batch_quotes,
)
from scripts.download_510300_option_research_data import (
    CALENDAR_FILE,
    OPTION_DAILY_FILE,
    PROXY_DAILY_FIELDS,
    RateLimiter,
    atomic_json,
    atomic_parquet,
    load_proxy,
    now,
    retry_call,
    sanitized_error,
    trading_date_chunks,
)


OPTION_DIR = ROOT / "data" / "raw" / "return_tail" / "options"
CFFEX_PROXY_CACHE = OPTION_DIR / ".acquisition_cache" / "tushare_cffex_daily"
CFFEX_PROXY_FILE = OPTION_DIR / "IO_tushare_eod.parquet"
CFFEX_OFFICIAL_FILE = OPTION_DIR / "IO_eod_chain.parquet"
SSE_STATS_CACHE = OPTION_DIR / ".acquisition_cache" / "sse_daily_statistics"
SSE_STATS_FILE = OPTION_DIR / "510300_sse_daily_statistics.parquet"
SINA_SNAPSHOT_DIR = OPTION_DIR / "forward_snapshots"
SINA_RETAINED_TERMINAL_FILE = OPTION_DIR / "sina_retained_terminal_orderbook.parquet"
SINA_FIVE_DAY_CACHE = OPTION_DIR / ".acquisition_cache" / "sina_five_day_minutes"
SINA_FIVE_DAY_FILE = OPTION_DIR / "510300_sina_terminal_five_day_minutes.parquet"
MEMBERSHIP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
CITIC_RAW_FILE = ROOT / "data" / "raw" / "reference" / "a_share_citic_industry_intervals_raw.parquet"
CITIC_FILE = ROOT / "data" / "raw" / "reference" / "a_share_citic_industry_point_in_time.parquet"
STATUS_FILE = ROOT / "reports" / "data_quality" / "return_tail_supplemental_acquisition.json"


def _load_dates() -> list[pd.Timestamp]:
    calendar = pd.read_parquet(CALENDAR_FILE)
    return sorted(pd.to_datetime(calendar["trade_date"]).dt.normalize().tolist())


def download_cffex_proxy(
    pro: Any, token: str, limiter: RateLimiter, dates: list[pd.Timestamp]
) -> dict[str, Any]:
    CFFEX_PROXY_CACHE.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    downloaded = 0
    reused = 0
    chunks = trading_date_chunks(dates, 10)
    for index, chunk in enumerate(chunks, start=1):
        left, right = chunk[0], chunk[-1]
        path = CFFEX_PROXY_CACHE / f"{left:%Y%m%d}_{right:%Y%m%d}.parquet"
        if path.exists():
            raw = pd.read_parquet(path)
            actual_dates = set(
                pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce").dt.normalize()
            )
            valid = set(chunk).issubset(actual_dates)
        else:
            raw = pd.DataFrame()
            valid = False
        if valid:
            reused += 1
        else:
            raw = retry_call(
                lambda left=left, right=right: pro.opt_daily(
                    exchange="CFFEX",
                    start_date=left.strftime("%Y%m%d"),
                    end_date=right.strftime("%Y%m%d"),
                    fields=PROXY_DAILY_FIELDS,
                ),
                limiter,
                f"下载中金所代理期权日线{left:%Y%m%d}-{right:%Y%m%d}",
                token,
            )
            if raw is None or raw.empty or len(raw) >= 14900:
                raise RuntimeError(f"中金所代理期权块为空或可能截断：{left.date()}至{right.date()}")
            raw = raw.loc[raw["ts_code"].astype(str).str.startswith("IO")].copy()
            raw["retrieved_at"] = str(now())
            atomic_parquet(raw, path)
            downloaded += 1
        frames.append(raw.drop(columns=["retrieved_at"], errors="ignore"))
        if index == 1 or index % 20 == 0 or index == len(chunks):
            print(f"IO代理日线进度：{index}/{len(chunks)}块", flush=True)
    data = normalize_cffex_proxy_daily(pd.concat(frames, ignore_index=True), now())
    atomic_parquet(data, CFFEX_PROXY_FILE)
    official = pd.read_parquet(CFFEX_OFFICIAL_FILE)
    comparison = official.merge(
        data,
        on=["trade_date", "contract_code"],
        how="outer",
        indicator=True,
        suffixes=("_official", "_proxy"),
        validate="one_to_one",
    )
    matched = comparison["_merge"].eq("both")
    metrics: dict[str, Any] = {
        "matched_key_count": int(matched.sum()),
        "official_only_key_count": int(comparison["_merge"].eq("left_only").sum()),
        "proxy_only_key_count": int(comparison["_merge"].eq("right_only").sum()),
    }
    for column in ("close", "settlement", "volume", "open_interest"):
        difference = (
            pd.to_numeric(comparison.loc[matched, f"{column}_official"], errors="coerce")
            - pd.to_numeric(comparison.loc[matched, f"{column}_proxy"], errors="coerce")
        ).abs()
        metrics[f"{column}_maximum_absolute_difference"] = float(difference.max())
        metrics[f"{column}_exact_match_ratio"] = float(difference.eq(0).mean())
    return {
        "status": "PASS",
        "file": CFFEX_PROXY_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(data)),
        "contract_count": int(data["contract_code"].nunique()),
        "trading_day_count": int(data["trade_date"].nunique()),
        "downloaded_chunk_count": downloaded,
        "reused_chunk_count": reused,
        "cross_source": metrics,
    }


def _sse_stats_request(session: requests.Session, trade_date: pd.Timestamp) -> list[dict[str, object]]:
    response = session.get(
        "http://query.sse.com.cn/commonQuery.do",
        params={
            "isPagination": "false",
            "tradeDate": trade_date.strftime("%Y%m%d"),
            "sqlId": "COMMON_SSE_ZQPZ_YSP_QQ_SJTJ_MRTJ_CX",
        },
        timeout=45,
    )
    response.raise_for_status()
    records = response.json().get("result")
    if not isinstance(records, list) or not records:
        raise RuntimeError("上交所每日汇总响应为空")
    return records


def download_sse_statistics(dates: list[pd.Timestamp], workers: int) -> dict[str, Any]:
    SSE_STATS_CACHE.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(0.20)
    thread_state = threading.local()
    frames: dict[pd.Timestamp, pd.DataFrame] = {}
    pending: list[pd.Timestamp] = []

    def session() -> requests.Session:
        value = getattr(thread_state, "session", None)
        if value is None:
            value = requests.Session()
            value.headers.update(
                {"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0 510300-audit/1.0"}
            )
            thread_state.session = value
        return value

    for date in dates:
        path = SSE_STATS_CACHE / f"{date:%Y%m%d}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if len(frame) == 1 and pd.Timestamp(frame.loc[0, "trade_date"]).normalize() == date:
                frames[date] = frame
                continue
        pending.append(date)

    def fetch(date: pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame]:
        records = retry_call(
            lambda: _sse_stats_request(session(), date),
            limiter,
            f"下载上交所每日汇总{date:%Y%m%d}",
            maximum_attempts=6,
        )
        frame = normalize_sse_daily_statistics(records, now())
        atomic_parquet(frame, SSE_STATS_CACHE / f"{date:%Y%m%d}.parquet")
        return date, frame

    completed = len(frames)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="上交所汇总") as pool:
        futures = {pool.submit(fetch, date): date for date in pending}
        for future in as_completed(futures):
            date, frame = future.result()
            frames[date] = frame
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(dates):
                print(f"上交所每日汇总进度：{completed}/{len(dates)}", flush=True)
    data = pd.concat([frames[date] for date in sorted(frames)], ignore_index=True)
    atomic_parquet(data, SSE_STATS_FILE)
    eod = pd.read_parquet(OPTION_DAILY_FILE)
    grouped = eod.groupby("trade_date", as_index=False).agg(
        calculated_contract_count=("contract_code", "nunique"),
        calculated_total_volume=("volume", "sum"),
        calculated_open_interest=("open_interest", "sum"),
    )
    option_group = eod.groupby(["trade_date", "option_type"], as_index=False)["volume"].sum()
    pivot = option_group.pivot(index="trade_date", columns="option_type", values="volume").reset_index()
    pivot = pivot.rename(columns={"C": "calculated_call_volume", "P": "calculated_put_volume"})
    comparison = data.merge(grouped, on="trade_date", validate="one_to_one").merge(
        pivot, on="trade_date", validate="one_to_one"
    )
    checks: dict[str, Any] = {}
    pairs = {
        "total_volume": "calculated_total_volume",
        "open_interest": "calculated_open_interest",
        "call_volume": "calculated_call_volume",
        "put_volume": "calculated_put_volume",
        "contract_count": "calculated_contract_count",
    }
    for official, calculated in pairs.items():
        valid = comparison[official].notna() & comparison[calculated].notna()
        difference = (comparison.loc[valid, official] - comparison.loc[valid, calculated]).abs()
        checks[f"{official}_compared_days"] = int(valid.sum())
        checks[f"{official}_exact_match_ratio"] = float(difference.eq(0).mean())
        checks[f"{official}_maximum_absolute_difference"] = float(difference.max())

    expiry_dates = set(
        eod.loc[eod["trade_date"].eq(eod["expiry_date"]), "trade_date"]
        .dropna()
        .tolist()
    )
    open_interest_valid = comparison["open_interest"].notna() & comparison[
        "calculated_open_interest"
    ].notna()
    open_interest_difference = (
        comparison["open_interest"] - comparison["calculated_open_interest"]
    ).abs()
    comparison["is_expiry_day"] = comparison["trade_date"].isin(expiry_dates)
    for label, mask in {
        "expiry_day": comparison["is_expiry_day"],
        "non_expiry_day": ~comparison["is_expiry_day"],
    }.items():
        segment = open_interest_valid & mask
        segment_difference = open_interest_difference.loc[segment]
        checks[f"open_interest_{label}_compared_days"] = int(segment.sum())
        checks[f"open_interest_{label}_exact_match_ratio"] = float(
            segment_difference.eq(0).mean()
        )
        checks[f"open_interest_{label}_maximum_absolute_difference"] = float(
            segment_difference.max()
        )
    non_expiry_mismatches = comparison.loc[
        open_interest_valid
        & ~comparison["is_expiry_day"]
        & open_interest_difference.ne(0),
        ["trade_date"],
    ].copy()
    non_expiry_mismatches["year"] = non_expiry_mismatches["trade_date"].dt.year
    checks["open_interest_non_expiry_mismatch_days_by_year"] = {
        str(year): int(count)
        for year, count in non_expiry_mismatches.groupby("year").size().items()
    }
    checks["open_interest_interpretation"] = (
        "上交所汇总与合约日线持仓量存在时点或供应商口径差异；"
        "到期日差异尤为集中，且2022年存在持续的非到期日差异，"
        "因此持仓量不作为同口径验证通过项"
    )
    return {
        "status": "PASS_WITH_OPEN_INTEREST_SOURCE_DIVERGENCE",
        "file": SSE_STATS_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(data)),
        "first_date": str(data["trade_date"].min().date()),
        "last_date": str(data["trade_date"].max().date()),
        "cross_source": checks,
    }


def _sina_request(session: requests.Session, code: str) -> list[str]:
    response = session.get(
        f"https://hq.sinajs.cn/list=CON_OP_{code}",
        headers={
            "Referer": "https://stock.finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        timeout=30,
    )
    response.raise_for_status()
    match = re.search(r'="(?P<data>.*)"', response.text)
    if not match or not match.group("data"):
        raise RuntimeError(f"新浪盘口{code}响应为空")
    return match.group("data").split(",")


def download_sina_snapshot(workers: int) -> dict[str, Any]:
    eod = pd.read_parquet(OPTION_DAILY_FILE)
    latest_date = pd.Timestamp(eod["trade_date"].max()).normalize()
    codes = sorted(
        eod.loc[eod["trade_date"].eq(latest_date), "contract_code"]
        .astype(str)
        .str.replace(".SH", "", regex=False)
        .unique()
    )
    limiter = RateLimiter(0.15)
    thread_state = threading.local()

    def session() -> requests.Session:
        value = getattr(thread_state, "session", None)
        if value is None:
            value = requests.Session()
            thread_state.session = value
        return value

    def fetch(code: str) -> pd.DataFrame:
        values = retry_call(
            lambda: _sina_request(session(), code),
            limiter,
            f"下载新浪盘口{code}",
            maximum_attempts=6,
        )
        return normalize_sina_option_quote(code, values, now())

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="新浪盘口") as pool:
        futures = {pool.submit(fetch, code): code for code in codes}
        for completed, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if completed == 1 or completed % 25 == 0 or completed == len(codes):
                print(f"新浪全链盘口进度：{completed}/{len(codes)}", flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values("contract_code")
    quote_dates = pd.to_datetime(data["quote_timestamp"]).dt.normalize()
    file_date = pd.Timestamp(quote_dates.mode().iloc[0])
    SINA_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SINA_SNAPSHOT_DIR / f"{file_date:%Y%m%d}_sina_orderbook.parquet"
    existing_file_reused = path.exists()
    if existing_file_reused:
        immutable_data = pd.read_parquet(path)
    else:
        atomic_parquet(data, path)
        immutable_data = data
    atomic_parquet(data, SINA_SNAPSHOT_DIR / "latest_sina_orderbook.parquet")
    return {
        "status": "STALE_MARKET_CLOSED_SNAPSHOT" if file_date < pd.Timestamp(now()).tz_localize(None).normalize() else "CURRENT_SNAPSHOT",
        "file": path.relative_to(ROOT).as_posix(),
        "row_count": int(len(immutable_data)),
        "quote_date": str(file_date.date()),
        "earliest_quote_timestamp": pd.Timestamp(immutable_data["quote_timestamp"].min()).isoformat(),
        "latest_quote_timestamp": pd.Timestamp(immutable_data["quote_timestamp"].max()).isoformat(),
        "positive_bid1_ratio": float(immutable_data["bid1"].gt(0).mean()),
        "positive_ask1_ratio": float(immutable_data["ask1"].gt(0).mean()),
        "inverted_quote_count": int(
            (immutable_data["bid1"] > immutable_data["ask1"]).sum()
        ),
        "append_only_date_file": True,
        "existing_file_reused_without_overwrite": existing_file_reused,
    }


def _sina_batch_request(session: requests.Session, codes: list[str]) -> dict[str, list[str]]:
    symbols = ",".join(f"CON_OP_{code}" for code in codes)
    response = session.get(
        f"https://hq.sinajs.cn/list={symbols}",
        headers={
            "Referer": "https://stock.finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        timeout=60,
    )
    response.raise_for_status()
    return parse_sina_batch_quotes(response.text)


def download_sina_retained_terminal_snapshots() -> dict[str, Any]:
    """获取新浪仍保留的已到期合约最后一笔五档行情，不冒充逐日历史。"""

    master = pd.read_parquet(OPTION_DIR / "510300_contract_master.parquet")
    latest_trade_date = pd.Timestamp(
        pd.read_parquet(OPTION_DAILY_FILE, columns=["trade_date"])["trade_date"].max()
    ).normalize()
    expired = master.loc[
        pd.to_datetime(master["expiry_date"]).lt(latest_trade_date),
        ["contract_code", "expiry_date"],
    ].copy()
    expired["plain_code"] = (
        expired["contract_code"].astype(str).str.replace(".SH", "", regex=False)
    )
    codes = sorted(expired["plain_code"].unique())
    batch_size = 80
    batches = [codes[index : index + batch_size] for index in range(0, len(codes), batch_size)]
    limiter = RateLimiter(0.25)
    session = requests.Session()
    returned: dict[str, list[str]] = {}
    for index, batch in enumerate(batches, start=1):
        part = retry_call(
            lambda batch=batch: _sina_batch_request(session, batch),
            limiter,
            f"下载新浪已到期盘口第{index}批",
            maximum_attempts=6,
        )
        returned.update(part)
        if index == 1 or index % 10 == 0 or index == len(batches):
            print(f"新浪已到期盘口进度：{index}/{len(batches)}批", flush=True)

    frames: list[pd.DataFrame] = []
    malformed_codes: list[str] = []
    retrieved_at = now()
    for code, values in returned.items():
        try:
            frames.append(normalize_sina_option_quote(code, values, retrieved_at))
        except ValueError:
            malformed_codes.append(code)
    if not frames:
        raise RuntimeError("新浪未保留任何已到期510300期权盘口")
    data = pd.concat(frames, ignore_index=True).sort_values("contract_code")
    data = data.merge(
        expired[["contract_code", "expiry_date"]].drop_duplicates("contract_code"),
        on="contract_code",
        how="left",
        validate="one_to_one",
    )
    data["snapshot_usage"] = "RETAINED_TERMINAL_ONLY_NOT_DAILY_HISTORY"
    atomic_parquet(data, SINA_RETAINED_TERMINAL_FILE)

    expired["expiry_year"] = pd.to_datetime(expired["expiry_date"]).dt.year.astype(str)
    retained_codes = set(data["contract_code"].str.replace(".SH", "", regex=False))
    coverage_by_year: dict[str, dict[str, int | float]] = {}
    for year, group in expired.groupby("expiry_year"):
        retained = int(group["plain_code"].isin(retained_codes).sum())
        total = int(len(group))
        coverage_by_year[str(year)] = {
            "expired_contract_count": total,
            "retained_snapshot_count": retained,
            "coverage_ratio": float(retained / max(total, 1)),
        }
    return {
        "status": "PARTIAL_RETAINED_TERMINAL_SNAPSHOTS_NOT_DAILY_HISTORY",
        "file": SINA_RETAINED_TERMINAL_FILE.relative_to(ROOT).as_posix(),
        "expired_contract_count": int(len(expired)),
        "retained_snapshot_count": int(len(data)),
        "empty_response_count": int(len(expired) - len(data) - len(malformed_codes)),
        "malformed_response_count": int(len(malformed_codes)),
        "earliest_quote_timestamp": pd.Timestamp(data["quote_timestamp"].min()).isoformat(),
        "latest_quote_timestamp": pd.Timestamp(data["quote_timestamp"].max()).isoformat(),
        "positive_bid1_ratio": float(data["bid1"].gt(0).mean()),
        "positive_ask1_ratio": float(data["ask1"].gt(0).mean()),
        "coverage_by_expiry_year": coverage_by_year,
        "frozen_daily_history_substitution_allowed": False,
    }


def _sina_five_day_request(
    session: requests.Session, code: str
) -> list[list[dict[str, object]]]:
    response = session.get(
        "https://stock.finance.sina.com.cn/futures/api/openapi.php/"
        "StockOptionDaylineService.getFiveDayLine",
        params={"symbol": f"CON_OP_{code}"},
        headers={
            "Referer": "https://stock.finance.sina.com.cn/option/quotes.html",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result") or {}
    data = result.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"新浪五日分钟{code}响应结构无效")
    return data


def download_sina_five_day_minutes(workers: int) -> dict[str, Any]:
    """下载全部510300期权合约各自最后五个交易日的分钟成交序列。"""

    master = pd.read_parquet(OPTION_DIR / "510300_contract_master.parquet")
    codes = sorted(
        master["contract_code"]
        .astype(str)
        .str.replace(".SH", "", regex=False)
        .unique()
    )
    SINA_FIVE_DAY_CACHE.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(0.20)
    thread_state = threading.local()

    def session() -> requests.Session:
        value = getattr(thread_state, "session", None)
        if value is None:
            value = requests.Session()
            thread_state.session = value
        return value

    def fetch(code: str) -> tuple[str, pd.DataFrame, bool]:
        cache_file = SINA_FIVE_DAY_CACHE / f"{code}.parquet"
        if cache_file.exists():
            return code, pd.read_parquet(cache_file), True
        records = retry_call(
            lambda: _sina_five_day_request(session(), code),
            limiter,
            f"下载新浪五日分钟{code}",
            maximum_attempts=6,
        )
        frame = normalize_sina_five_day_minutes(code, records, now())
        if frame.empty:
            raise RuntimeError(f"新浪五日分钟{code}为空")
        atomic_parquet(frame, cache_file)
        return code, frame, False

    frames: list[pd.DataFrame] = []
    downloaded_count = 0
    reused_count = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="新浪五日分钟") as pool:
        futures = {pool.submit(fetch, code): code for code in codes}
        for completed, future in enumerate(as_completed(futures), start=1):
            _, frame, reused = future.result()
            frames.append(frame)
            if reused:
                reused_count += 1
            else:
                downloaded_count += 1
            if completed == 1 or completed % 100 == 0 or completed == len(codes):
                print(f"新浪五日分钟进度：{completed}/{len(codes)}个合约", flush=True)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["trade_date", "contract_code", "minute_timestamp"])
    duplicate_key_count = int(
        data[["contract_code", "minute_timestamp"]].duplicated().sum()
    )
    if duplicate_key_count:
        raise RuntimeError(f"新浪五日分钟存在{duplicate_key_count}个重复键")
    atomic_parquet(data, SINA_FIVE_DAY_FILE)
    days_per_contract = data.groupby("contract_code")["trade_date"].nunique()
    return {
        "status": "PASS_TERMINAL_FIVE_DAY_MINUTES_NOT_FULL_HISTORY",
        "file": SINA_FIVE_DAY_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(data)),
        "contract_count": int(data["contract_code"].nunique()),
        "expected_contract_count": int(len(codes)),
        "contract_coverage_ratio": float(data["contract_code"].nunique() / max(len(codes), 1)),
        "trading_day_count": int(data["trade_date"].nunique()),
        "first_trade_date": str(data["trade_date"].min().date()),
        "last_trade_date": str(data["trade_date"].max().date()),
        "five_day_contract_ratio": float(days_per_contract.eq(5).mean()),
        "price_positive_ratio": float(data["price"].gt(0).mean()),
        "duplicate_key_count": duplicate_key_count,
        "downloaded_contract_count": downloaded_count,
        "reused_contract_count": reused_count,
        "historical_bid_ask_available": False,
        "frozen_daily_history_substitution_allowed": False,
    }


def download_citic_industry(
    pro: Any, token: str, limiter: RateLimiter
) -> dict[str, Any]:
    historical = retry_call(
        lambda: pro.ci_index_member(is_new="N"),
        limiter,
        "下载中信历史行业成员",
        token,
    )
    current_parts: list[pd.DataFrame] = []
    for index, number in enumerate(range(1, 50), start=1):
        l1_code = f"CI005{number:03d}.CI"
        part = retry_call(
            lambda l1_code=l1_code: pro.ci_index_member(l1_code=l1_code, is_new="Y"),
            limiter,
            f"下载中信当前行业{l1_code}",
            token,
        )
        if part is not None and not part.empty:
            current_parts.append(part)
        if index % 10 == 0 or index == 49:
            print(f"中信行业进度：{index}/49个一级代码", flush=True)
    raw = pd.concat([historical, *current_parts], ignore_index=True).drop_duplicates()
    raw["source"] = "tushare_proxy.ci_index_member"
    raw["retrieved_at"] = str(now())
    atomic_parquet(raw, CITIC_RAW_FILE)
    membership = pd.read_parquet(MEMBERSHIP_FILE)
    universe = set(membership["symbol"].astype(str))
    industry = normalize_citic_industry_intervals(raw, universe, now())
    members = active_members(membership, _load_dates())
    coverage = industry_coverage(industry, members)
    atomic_parquet(industry, CITIC_FILE)
    return {
        "status": "PASS" if coverage["coverage_ratio"] >= 0.99 else "PARTIAL",
        "file": CITIC_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(industry)),
        "symbol_count": int(industry["con_code"].nunique()),
        "frozen_sw_statistic_substitution_allowed": False,
        **coverage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载尾部风险研究补充数据")
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "cffex-proxy",
            "sse-stats",
            "sina-snapshot",
            "sina-terminal-snapshots",
            "sina-five-day-minutes",
            "citic-industry",
        ],
        default="all",
    )
    parser.add_argument("--interval", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 0.5:
        raise ValueError("代理请求间隔不得低于0.5秒")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("并发数必须介于1至4")
    dates = _load_dates()
    previous: dict[str, Any] = {}
    if STATUS_FILE.exists():
        try:
            previous = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    phases = dict(previous.get("phases", {}))
    needs_proxy = args.phase in {"all", "cffex-proxy", "citic-industry"}
    pro = token = endpoint = None
    limiter = None
    if needs_proxy:
        pro, token, endpoint = load_proxy()
        limiter = RateLimiter(args.interval)
    if args.phase in {"all", "cffex-proxy"}:
        assert pro is not None and token is not None and limiter is not None
        phases["cffex_proxy_daily"] = download_cffex_proxy(pro, token, limiter, dates)
    if args.phase in {"all", "sse-stats"}:
        phases["sse_daily_statistics"] = download_sse_statistics(dates, args.workers)
    if args.phase in {"all", "sina-snapshot"}:
        phases["sina_orderbook_snapshot"] = download_sina_snapshot(args.workers)
    if args.phase in {"all", "sina-terminal-snapshots"}:
        phases["sina_retained_terminal_snapshots"] = (
            download_sina_retained_terminal_snapshots()
        )
    if args.phase in {"all", "sina-five-day-minutes"}:
        phases["sina_terminal_five_day_minutes"] = download_sina_five_day_minutes(
            args.workers
        )
    if args.phase in {"all", "citic-industry"}:
        assert pro is not None and token is not None and limiter is not None
        phases["citic_industry"] = download_citic_industry(pro, token, limiter)
    status = {
        "status": "ACQUISITION_COMPLETE_WITH_DOCUMENTED_UNAVAILABLE_INPUTS",
        "checked_at": now().isoformat(),
        "proxy_host": endpoint.split("//", 1)[-1] if endpoint else previous.get("proxy_host", "fast.xiaodefa.cn"),
        "token_persisted_in_outputs": False,
        "unavailable": {
            "full_continuous_option_minute_history": (
                "代理返回无opt_mins接口权限；新浪仅补齐每个合约最后五个交易日"
            ),
            "historical_sse_bid_ask_snapshots": "公开日线、分钟线和风险指标接口均不提供历史盘口快照",
            "immutable_consensus_vintages": "report_rc为单篇研报记录，不是供应商历史一致预期快照",
        },
        "resolved_elsewhere": {
            "sw_industry_membership": "按一级行业拆分抓取后，成员日覆盖99.667%，已通过99%门槛",
        },
        "phases": phases,
    }
    atomic_json(status, STATUS_FILE)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"补充数据下载失败：{sanitized_error(exc)}", file=sys.stderr, flush=True)
        raise SystemExit(1)
