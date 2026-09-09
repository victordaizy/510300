"""下载并审计 510300 与 IO 期权研究数据，支持断点续传。

安全边界：只读市场数据；不连接券商、不生成订单、不映射实盘仓位。
代理令牌仅从当前环境或项目 .env 读取，绝不写入任何输出文件。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import tushare as ts
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.option_research_data_acquisition import (
    dataset_quality,
    merge_underlying_reference,
    normalize_contract_master,
    normalize_option_daily,
    normalize_sse_risk_indicators,
    normalize_underlying_daily,
    parse_cffex_month_zip,
    trading_date_chunks,
    validate_zip,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
REGISTRY_FILE = PROJECT_ROOT / "config" / "return_tail_hypothesis_registry.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
OPTION_DIR = PROJECT_ROOT / "data" / "raw" / "return_tail" / "options"
CACHE_DIR = OPTION_DIR / ".acquisition_cache"
TUSHARE_CACHE_DIR = CACHE_DIR / "tushare_sse_daily"
SSE_RISK_CACHE_DIR = CACHE_DIR / "sse_risk"
CFFEX_CACHE_DIR = CACHE_DIR / "cffex_monthly"
MASTER_RAW_FILE = OPTION_DIR / "510300_tushare_contracts_raw.parquet"
MASTER_FILE = OPTION_DIR / "510300_contract_master.parquet"
OPTION_DAILY_FILE = OPTION_DIR / "510300_tushare_eod.parquet"
RISK_FILE = OPTION_DIR / "510300_sse_risk_indicators.parquet"
CFFEX_FILE = OPTION_DIR / "IO_eod_chain.parquet"
UNDERLYING_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
CALENDAR_FILE = OPTION_DIR / "sse_trading_calendar.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_option_acquisition.json"
PROXY_DAILY_FIELDS = (
    "ts_code,trade_date,exchange,pre_settle,pre_close,open,high,low,close,"
    "settle,vol,amount,oi"
)
PROXY_MASTER_FIELDS = (
    "ts_code,exchange,name,per_unit,opt_code,opt_type,call_put,exercise_price,"
    "maturity_date,list_date,delist_date,opt_multiplier"
)


class RateLimiter:
    """线程安全的最小调用间隔控制器。"""

    def __init__(self, minimum_interval_seconds: float) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("请求间隔不得小于0")
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self.minimum_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


def now() -> datetime:
    return datetime.now(TIMEZONE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def sanitized_error(exc: Exception, secret: str = "") -> str:
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[令牌已隐藏]")
    return message[:1000]


def retry_call(
    operation: Callable[[], Any],
    limiter: RateLimiter,
    operation_name: str,
    secret: str = "",
    maximum_attempts: int = 6,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        limiter.wait()
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == maximum_attempts:
                break
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"{operation_name}第{attempt}次失败，{delay}秒后重试："
                f"{sanitized_error(exc, secret)}",
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(
        f"{operation_name}连续失败{maximum_attempts}次：{sanitized_error(last_error, secret)}"
    ) from last_error


def registry_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    contract = registry["data_contracts"]["options"]
    return pd.Timestamp(contract["listing_date"]), pd.Timestamp(contract["cutoff_date"])


def load_proxy() -> tuple[Any, str, str]:
    load_dotenv(ENV_FILE, override=False)
    token = os.getenv("TUSHARE_PROXY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("缺少TUSHARE_PROXY_TOKEN；未读取或输出任何令牌内容")
    endpoint = os.getenv("TUSHARE_PROXY_URL", "https://fast.xiaodefa.cn").strip().rstrip("/")
    if endpoint not in {"https://fast.xiaodefa.cn", "https://tt.xiaodefa.cn"}:
        raise ValueError("代理地址不在教程列出的允许名单内")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = endpoint
    return pro, token, endpoint


def download_calendar(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    data = retry_call(
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            is_open="1",
            fields="exchange,cal_date,is_open,pretrade_date",
        ),
        limiter,
        "下载上交所交易日历",
        token,
    )
    if data is None or data.empty:
        raise RuntimeError("上交所交易日历为空")
    result = data.copy()
    result["trade_date"] = pd.to_datetime(
        result["cal_date"], format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    result["source"] = "tushare_proxy.trade_cal"
    result["retrieved_at"] = now()
    result = result.sort_values("trade_date").reset_index(drop=True)
    atomic_parquet(result, CALENDAR_FILE)
    return result


def download_underlying(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    raw = retry_call(
        lambda: pro.fund_daily(
            ts_code="510300.SH",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        ),
        limiter,
        "下载510300日线对照",
        token,
    )
    downloaded = normalize_underlying_daily(raw, now())
    existing = pd.read_parquet(UNDERLYING_FILE) if UNDERLYING_FILE.exists() else downloaded.iloc[0:0]
    merged, evidence = merge_underlying_reference(existing, downloaded)
    atomic_parquet(merged, UNDERLYING_FILE)
    return merged, evidence


def download_master(
    pro: Any, token: str, limiter: RateLimiter
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = retry_call(
        lambda: pro.opt_basic(exchange="SSE", fields=PROXY_MASTER_FIELDS),
        limiter,
        "下载上交所期权合约主表",
        token,
    )
    if raw is None or raw.empty:
        raise RuntimeError("上交所期权合约主表为空")
    selected_raw = raw.loc[raw["opt_code"].astype(str).eq("OP510300.SH")].copy()
    selected_raw["source"] = "tushare_proxy.opt_basic"
    selected_raw["retrieved_at"] = now()
    master = normalize_contract_master(raw, now())
    atomic_parquet(selected_raw, MASTER_RAW_FILE)
    atomic_parquet(master, MASTER_FILE)
    return selected_raw, master


def _cached_tushare_chunk_is_valid(path: Path, expected_dates: list[pd.Timestamp]) -> bool:
    if not path.exists():
        return False
    try:
        data = pd.read_parquet(path, columns=["trade_date", "ts_code"])
        actual = set(pd.to_datetime(data["trade_date"], format="%Y%m%d", errors="coerce").dt.normalize())
        return set(expected_dates).issubset(actual) and data["ts_code"].notna().all()
    except Exception:
        return False


def download_option_daily(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    dates: list[pd.Timestamp],
    master: pd.DataFrame,
    underlying: pd.DataFrame,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    TUSHARE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    master_codes = set(master["contract_code"].astype(str))
    chunks = trading_date_chunks(dates, chunk_size)
    frames: list[pd.DataFrame] = []
    downloaded_count = 0
    reused_count = 0
    for index, chunk in enumerate(chunks, start=1):
        left, right = chunk[0], chunk[-1]
        path = TUSHARE_CACHE_DIR / f"{left:%Y%m%d}_{right:%Y%m%d}.parquet"
        if _cached_tushare_chunk_is_valid(path, chunk):
            raw = pd.read_parquet(path)
            reused_count += 1
        else:
            raw = retry_call(
                lambda left=left, right=right: pro.opt_daily(
                    exchange="SSE",
                    start_date=left.strftime("%Y%m%d"),
                    end_date=right.strftime("%Y%m%d"),
                    fields=PROXY_DAILY_FIELDS,
                ),
                limiter,
                f"下载上交所期权日行情{left:%Y%m%d}-{right:%Y%m%d}",
                token,
            )
            if raw is None or raw.empty:
                raise RuntimeError(f"{left.date()}至{right.date()}期权日行情为空")
            if len(raw) >= 14900:
                raise RuntimeError(
                    f"{left.date()}至{right.date()}返回{len(raw)}行，接近接口上限，拒绝接受潜在截断"
                )
            actual_dates = set(
                pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce").dt.normalize()
            )
            missing_dates = sorted(set(chunk).difference(actual_dates))
            if missing_dates:
                raise RuntimeError(
                    "期权日行情缺失交易日："
                    + ",".join(str(value.date()) for value in missing_dates)
                )
            raw = raw.loc[raw["ts_code"].astype(str).isin(master_codes)].copy()
            if raw.empty:
                raise RuntimeError(f"{left.date()}至{right.date()}没有510300期权")
            raw["retrieved_at"] = now()
            atomic_parquet(raw, path)
            downloaded_count += 1
        frames.append(raw.drop(columns=["retrieved_at"], errors="ignore"))
        if index == 1 or index % 10 == 0 or index == len(chunks):
            print(
                f"510300期权成交日线进度：{index}/{len(chunks)}块，"
                f"当前{left.date()}至{right.date()}，{len(raw)}行",
                flush=True,
            )
    combined_raw = pd.concat(frames, ignore_index=True)
    normalized = normalize_option_daily(combined_raw, master, underlying, now())
    atomic_parquet(normalized, OPTION_DAILY_FILE)
    return normalized, {
        "downloaded_chunk_count": downloaded_count,
        "reused_chunk_count": reused_count,
        "total_chunk_count": len(chunks),
    }


def _sse_risk_request(session: requests.Session, trade_date: pd.Timestamp) -> list[dict[str, object]]:
    response = session.get(
        "http://query.sse.com.cn/commonQuery.do",
        params={
            "isPagination": "false",
            "trade_date": trade_date.strftime("%Y%m%d"),
            "sqlId": "SSE_ZQPZ_YSP_GGQQZSXT_YSHQ_QQFXZB_DATE_L",
            "contractSymbol": "",
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("result")
    if not isinstance(records, list) or not records:
        raise RuntimeError("上交所风险指标响应没有result记录")
    return records


def download_sse_risk(
    dates: list[pd.Timestamp], minimum_interval_seconds: float, workers: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    SSE_RISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(minimum_interval_seconds)
    thread_state = threading.local()
    frames_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    pending_dates: list[pd.Timestamp] = []
    downloaded_count = 0
    reused_count = 0

    def thread_session() -> requests.Session:
        session = getattr(thread_state, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Referer": "https://www.sse.com.cn/",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "510300-research-audit/1.0"
                    ),
                }
            )
            thread_state.session = session
        return session

    for trade_date in dates:
        path = SSE_RISK_CACHE_DIR / f"{trade_date:%Y%m%d}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            valid_cache = (
                not frame.empty
                and set(pd.to_datetime(frame["trade_date"]).dt.normalize()) == {trade_date}
            )
        else:
            valid_cache = False
            frame = pd.DataFrame()
        if valid_cache:
            frames_by_date[trade_date] = frame
            reused_count += 1
        else:
            pending_dates.append(trade_date)

    def fetch_one(trade_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.DataFrame]:
        path = SSE_RISK_CACHE_DIR / f"{trade_date:%Y%m%d}.parquet"
        session = thread_session()
        records = retry_call(
            lambda: _sse_risk_request(session, trade_date),
            limiter,
            f"下载上交所风险指标{trade_date:%Y%m%d}",
            maximum_attempts=6,
        )
        frame = normalize_sse_risk_indicators(records, now())
        atomic_parquet(frame, path)
        return trade_date, frame

    completed_count = reused_count
    if reused_count:
        print(
            f"上交所隐含波动率复用缓存：{reused_count}/{len(dates)}个交易日",
            flush=True,
        )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="上交所风险指标") as pool:
        futures = {pool.submit(fetch_one, value): value for value in pending_dates}
        for future in as_completed(futures):
            trade_date, frame = future.result()
            frames_by_date[trade_date] = frame
            downloaded_count += 1
            completed_count += 1
            if (
                completed_count == reused_count + 1
                or completed_count % 50 == 0
                or completed_count == len(dates)
            ):
                print(
                    f"上交所隐含波动率进度：{completed_count}/{len(dates)}个交易日，"
                    f"刚完成{trade_date.date()}，{len(frame)}行",
                    flush=True,
                )
    frames = [frames_by_date[value] for value in sorted(frames_by_date)]
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "contract_code"]
    ).reset_index(drop=True)
    if result[["trade_date", "contract_code"]].duplicated().any():
        raise RuntimeError("合并后的上交所风险指标存在重复键")
    atomic_parquet(result, RISK_FILE)
    return result, {
        "downloaded_day_count": downloaded_count,
        "reused_day_count": reused_count,
        "total_day_count": len(dates),
    }


def _month_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Period]:
    return list(pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M"))


def _download_cffex_zip(
    session: requests.Session, month: pd.Period, limiter: RateLimiter
) -> bytes:
    month_text = str(month).replace("-", "")
    url = f"http://www.cffex.com.cn/sj/historysj/{month_text}/zip/{month_text}.zip"
    limiter.wait()
    response = session.get(url, timeout=60)
    response.raise_for_status()
    validate_zip(response.content)
    return response.content


def download_cffex(
    start: pd.Timestamp, end: pd.Timestamp, minimum_interval_seconds: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    CFFEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(minimum_interval_seconds)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 510300-research-audit/1.0"})
    months = _month_periods(start, end)
    frames: list[pd.DataFrame] = []
    downloaded_count = 0
    reused_count = 0
    for index, month in enumerate(months, start=1):
        month_text = str(month).replace("-", "")
        path = CFFEX_CACHE_DIR / f"{month_text}.zip"
        if path.exists():
            content = path.read_bytes()
            try:
                validate_zip(content)
                reused_count += 1
            except ValueError:
                content = b""
        else:
            content = b""
        if not content:
            last_error: Exception | None = None
            for attempt in range(1, 6):
                try:
                    content = _download_cffex_zip(session, month, limiter)
                    atomic_bytes(content, path)
                    downloaded_count += 1
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 5:
                        raise RuntimeError(
                            f"下载中金所{month_text}月度历史数据失败：{sanitized_error(exc)}"
                        ) from exc
                    time.sleep(min(2 ** (attempt - 1), 20))
            if not content and last_error is not None:
                raise last_error
        frame = parse_cffex_month_zip(content, start, end, now())
        if not frame.empty:
            frames.append(frame)
        if index == 1 or index % 12 == 0 or index == len(months):
            print(
                f"中金所IO日线进度：{index}/{len(months)}个月，当前{month_text}，{len(frame)}行",
                flush=True,
            )
    if not frames:
        raise RuntimeError("中金所月度历史数据中没有IO期权")
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "contract_code"]
    ).reset_index(drop=True)
    if result[["trade_date", "contract_code"]].duplicated().any():
        raise RuntimeError("合并后的中金所IO日线存在重复键")
    atomic_parquet(result, CFFEX_FILE)
    return result, {
        "downloaded_month_count": downloaded_count,
        "reused_month_count": reused_count,
        "total_month_count": len(months),
    }


def file_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"file": path.relative_to(PROJECT_ROOT).as_posix(), "exists": False}
    return {
        "file": path.relative_to(PROJECT_ROOT).as_posix(),
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def create_report(
    start: pd.Timestamp,
    end: pd.Timestamp,
    calendar: pd.DataFrame,
    underlying_evidence: dict[str, float | int] | None,
    proxy_host: str | None,
    run_evidence: dict[str, Any],
) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    if REPORT_FILE.exists():
        try:
            previous = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    previous_sources = previous.get("sources", {})
    if proxy_host is None:
        proxy_host = previous_sources.get("proxy_host")
    if proxy_host is None:
        proxy_host = os.getenv(
            "TUSHARE_PROXY_URL", "https://fast.xiaodefa.cn"
        ).strip().rstrip("/").split("//", 1)[-1]
    merged_run_evidence = dict(previous.get("run_evidence", {}))
    merged_run_evidence.update(run_evidence)
    if underlying_evidence is None:
        underlying_evidence = previous.get("underlying_merge")

    dates = list(pd.to_datetime(calendar["trade_date"]).dt.normalize())
    datasets: dict[str, Any] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in (
        ("contract_master", MASTER_FILE),
        ("tushare_eod", OPTION_DAILY_FILE),
        ("sse_risk_indicators", RISK_FILE),
        ("cffex_io_eod", CFFEX_FILE),
    ):
        evidence = file_evidence(path)
        if path.exists():
            data = pd.read_parquet(path)
            loaded[name] = data
            if name == "contract_master":
                evidence["row_count"] = int(len(data))
                evidence["contract_count"] = int(data["contract_code"].nunique())
                evidence["adjusted_contract_count"] = int(data["is_adjusted"].sum())
                evidence["first_list_date"] = str(pd.to_datetime(data["list_date"]).min().date())
                evidence["last_expiry_date"] = str(pd.to_datetime(data["expiry_date"]).max().date())
            else:
                evidence.update(dataset_quality(data, dates))
        datasets[name] = evidence
    dataset_checks: dict[str, Any] = {}
    if "contract_master" in loaded and "tushare_eod" in loaded:
        master = loaded["contract_master"]
        eod = loaded["tushare_eod"]
        master_codes = set(master["contract_code"].astype(str))
        eod_codes = set(eod["contract_code"].astype(str))
        dataset_checks["tushare_eod"] = {
            "duplicate_key_count": int(
                eod[["trade_date", "contract_code"]].duplicated().sum()
            ),
            "contract_master_coverage": float(
                len(master_codes & eod_codes) / max(len(eod_codes), 1)
            ),
            "underlying_close_null_ratio": float(eod["underlying_close"].isna().mean()),
            "adjusted_quote_row_count": int(eod["is_adjusted"].fillna(False).sum()),
            "close_null_ratio": float(
                pd.to_numeric(eod["close"], errors="coerce").isna().mean()
            ),
            "close_positive_ratio": float(pd.to_numeric(eod["close"], errors="coerce").gt(0).mean()),
            "invalid_close_with_positive_volume_count": int(
                (
                    pd.to_numeric(eod["close"], errors="coerce").fillna(0).le(0)
                    & pd.to_numeric(eod["volume"], errors="coerce").gt(0)
                ).sum()
            ),
            "open_interest_nonnegative_ratio": float(
                pd.to_numeric(eod["open_interest"], errors="coerce").ge(0).mean()
            ),
        }
        if UNDERLYING_FILE.exists():
            underlying_reference = pd.read_parquet(UNDERLYING_FILE)
            underlying_reference["date"] = pd.to_datetime(
                underlying_reference["date"], errors="coerce"
            ).dt.normalize()
            eod_underlying = (
                eod.groupby("trade_date", as_index=False)["underlying_close"]
                .first()
                .rename(columns={"trade_date": "date", "underlying_close": "option_close"})
            )
            comparison = underlying_reference[["date", "close"]].merge(
                eod_underlying, on="date", how="inner", validate="one_to_one"
            )
            close_difference = (
                pd.to_numeric(comparison["close"], errors="coerce")
                - pd.to_numeric(comparison["option_close"], errors="coerce")
            ).abs()
            dataset_checks["underlying_crosscheck"] = {
                "matched_day_count": int(len(comparison)),
                "maximum_absolute_close_difference": float(close_difference.max()),
                "source_day_counts": {
                    str(key): int(value)
                    for key, value in (
                        underlying_reference.loc[
                            underlying_reference["date"].between(start, end)
                        ]
                        .groupby("source")["date"]
                        .nunique()
                        .items()
                    )
                },
            }
    if "sse_risk_indicators" in loaded:
        risk = loaded["sse_risk_indicators"]
        implied_volatility = pd.to_numeric(risk["implied_volatility"], errors="coerce")
        delta = pd.to_numeric(risk["delta"], errors="coerce")
        risk_checks: dict[str, Any] = {
            "duplicate_key_count": int(
                risk[["trade_date", "contract_code"]].duplicated().sum()
            ),
            "implied_volatility_null_ratio": float(implied_volatility.isna().mean()),
            "implied_volatility_nonpositive_ratio": float(implied_volatility.le(0).mean()),
            "positive_iv_and_delta_usable_ratio": float(
                (implied_volatility.gt(0) & delta.notna()).mean()
            ),
        }
        if "tushare_eod" in loaded:
            eod_data = loaded["tushare_eod"]
            eod_keys = pd.MultiIndex.from_frame(eod_data[["trade_date", "contract_code"]])
            risk_keys = pd.MultiIndex.from_frame(risk[["trade_date", "contract_code"]])
            matched_count = int(len(eod_keys.intersection(risk_keys)))
            risk_checks["matched_tushare_eod_key_count"] = matched_count
            risk_checks["tushare_eod_key_match_ratio"] = float(
                matched_count / max(len(eod_keys), 1)
            )
            risk_checks["risk_key_match_ratio"] = float(
                matched_count / max(len(risk_keys), 1)
            )
            risk_with_trade = risk.merge(
                eod_data[["trade_date", "contract_code", "volume", "open_interest"]],
                on=["trade_date", "contract_code"],
                how="left",
                validate="one_to_one",
            )
            zero_iv = pd.to_numeric(
                risk_with_trade["implied_volatility"], errors="coerce"
            ).fillna(0).le(0)
            risk_checks["nonpositive_iv_with_positive_volume_count"] = int(
                (zero_iv & pd.to_numeric(risk_with_trade["volume"], errors="coerce").gt(0)).sum()
            )
            eod_without_risk = eod_data.merge(
                risk[["trade_date", "contract_code"]],
                on=["trade_date", "contract_code"],
                how="left",
                indicator=True,
                validate="one_to_one",
            )
            eod_without_risk = eod_without_risk.loc[
                eod_without_risk["_merge"].eq("left_only")
            ]
            on_expiry = pd.to_datetime(
                eod_without_risk["trade_date"], errors="coerce"
            ).eq(pd.to_datetime(eod_without_risk["expiry_date"], errors="coerce"))
            risk_checks["eod_keys_without_risk_count"] = int(len(eod_without_risk))
            risk_checks["eod_keys_without_risk_on_expiry_ratio"] = float(
                on_expiry.mean() if len(on_expiry) else 1.0
            )
        dataset_checks["sse_risk_indicators"] = risk_checks
    if "cffex_io_eod" in loaded:
        io_data = loaded["cffex_io_eod"]
        dataset_checks["cffex_io_eod"] = {
            "duplicate_key_count": int(
                io_data[["trade_date", "contract_code"]].duplicated().sum()
            ),
            "delta_null_ratio": float(io_data["delta"].isna().mean()),
            "close_positive_ratio": float(
                pd.to_numeric(io_data["close"], errors="coerce").gt(0).mean()
            ),
            "open_interest_nonnegative_ratio": float(
                pd.to_numeric(io_data["open_interest"], errors="coerce").ge(0).mean()
            ),
        }
    eod_coverage = datasets.get("tushare_eod", {}).get("date_coverage", 0.0)
    risk_coverage = datasets.get("sse_risk_indicators", {}).get("date_coverage", 0.0)
    cffex_coverage = datasets.get("cffex_io_eod", {}).get("date_coverage", 0.0)
    complete_available = all(
        datasets.get(name, {}).get("exists", False)
        for name in ("contract_master", "tushare_eod", "sse_risk_indicators", "cffex_io_eod")
    )
    report = {
        "status": "ACQUIRED_WITH_HISTORICAL_QUOTE_LIMITATION" if complete_available else "PARTIAL",
        "checked_at": now().isoformat(),
        "scope": {
            "underlying": "510300.SH",
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "expected_trading_day_count": int(len(dates)),
        },
        "safety": {
            "paper_research_only": True,
            "broker_connection_enabled": False,
            "order_generation_enabled": False,
            "token_persisted_in_outputs": False,
        },
        "sources": {
            "proxy_host": proxy_host,
            "sse_risk_endpoint": "query.sse.com.cn/commonQuery.do",
            "cffex_history_service": "www.cffex.com.cn/sj/historysj/YYYYMM/zip/YYYYMM.zip",
        },
        "datasets": datasets,
        "dataset_checks": dataset_checks,
        "cache_evidence": {
            "tushare_daily_chunk_count": int(
                len(list(TUSHARE_CACHE_DIR.glob("*.parquet")))
            ),
            "sse_risk_day_count": int(len(list(SSE_RISK_CACHE_DIR.glob("*.parquet")))),
            "cffex_month_zip_count": int(len(list(CFFEX_CACHE_DIR.glob("*.zip")))),
        },
        "run_evidence": merged_run_evidence,
        "underlying_merge": underlying_evidence,
        "quality_gates": {
            "tushare_eod_full_date_coverage": bool(eod_coverage >= 0.999),
            "sse_risk_full_date_coverage": bool(risk_coverage >= 0.999),
            "cffex_io_full_date_coverage": bool(cffex_coverage >= 0.999),
            "adjusted_contracts_preserved": bool(
                datasets.get("contract_master", {}).get("adjusted_contract_count", 0) > 0
            ),
            "historical_bid1_ask1_available": False,
            "historical_close_snapshot_timestamp_available": False,
        },
        "research_boundary": {
            "frozen_quote_contract_file_created": False,
            "reason": (
                "已授权日行情只包含成交OHLC、结算价、成交量和持仓量；上交所公开风险指标"
                "包含隐含波动率及希腊字母，但两者都不含历史bid1、ask1和收盘快照时间。"
                "因此不创建510300_eod_chain.parquet，也不宣称通过冻结的数据门槛。"
            ),
            "exploratory_inputs_now_available_for": ["O2", "O3", "O4"],
            "frozen_protocol_blocked_hypotheses": ["O1", "O2", "O3", "O4"],
            "return_test_authorized": False,
        },
    }
    atomic_json(report, REPORT_FILE)
    return report


def parse_args() -> argparse.Namespace:
    default_start, default_end = registry_dates()
    parser = argparse.ArgumentParser(description="下载并审计510300与IO期权研究数据")
    parser.add_argument(
        "--phase",
        choices=["all", "tushare", "sse-risk", "cffex", "report"],
        default="all",
        help="执行阶段，默认全部执行",
    )
    parser.add_argument("--start-date", default=str(default_start.date()))
    parser.add_argument("--end-date", default=str(default_end.date()))
    parser.add_argument("--chunk-trading-days", type=int, default=10)
    parser.add_argument("--proxy-interval", type=float, default=0.65)
    parser.add_argument("--sse-interval", type=float, default=0.20)
    parser.add_argument("--sse-workers", type=int, default=4)
    parser.add_argument("--cffex-interval", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    if end < start:
        raise ValueError("结束日期不得早于开始日期")
    if args.proxy_interval < 0.5:
        raise ValueError("代理接口请求间隔不得低于0.5秒")
    if args.chunk_trading_days < 1 or args.chunk_trading_days > 10:
        raise ValueError("每块交易日数必须介于1至10，避免接口行数截断")
    if args.sse_workers < 1 or args.sse_workers > 4:
        raise ValueError("上交所风险指标并发数必须介于1至4")

    run_evidence: dict[str, Any] = {}
    underlying_evidence: dict[str, float | int] | None = None
    proxy_host: str | None = None
    calendar: pd.DataFrame

    needs_proxy = args.phase in {"all", "tushare"}
    if needs_proxy:
        pro, token, endpoint = load_proxy()
        proxy_host = endpoint.split("//", 1)[-1]
        limiter = RateLimiter(args.proxy_interval)
        calendar = download_calendar(pro, token, limiter, start, end)
        underlying, underlying_evidence = download_underlying(
            pro, token, limiter, start, end
        )
        _, master = download_master(pro, token, limiter)
        dates = list(pd.to_datetime(calendar["trade_date"]).dt.normalize())
        option_daily, evidence = download_option_daily(
            pro,
            token,
            limiter,
            dates,
            master,
            underlying,
            args.chunk_trading_days,
        )
        run_evidence["tushare"] = evidence
        print(
            f"510300成交日线完成：{len(option_daily)}行，"
            f"{option_daily['trade_date'].nunique()}个交易日",
            flush=True,
        )
    else:
        if not CALENDAR_FILE.exists():
            raise RuntimeError("缺少交易日历，请先执行--phase tushare")
        calendar = pd.read_parquet(CALENDAR_FILE)

    dates = list(
        pd.to_datetime(calendar["trade_date"], errors="coerce")
        .dropna()
        .dt.normalize()
    )
    dates = [value for value in dates if start <= value <= end]

    if args.phase in {"all", "sse-risk"}:
        risk, evidence = download_sse_risk(
            dates, args.sse_interval, args.sse_workers
        )
        run_evidence["sse_risk"] = evidence
        print(
            f"上交所隐含波动率完成：{len(risk)}行，{risk['trade_date'].nunique()}个交易日",
            flush=True,
        )

    if args.phase in {"all", "cffex"}:
        cffex, evidence = download_cffex(start, end, args.cffex_interval)
        run_evidence["cffex"] = evidence
        print(
            f"中金所IO成交日线完成：{len(cffex)}行，{cffex['trade_date'].nunique()}个交易日",
            flush=True,
        )

    report = create_report(
        start,
        end,
        calendar,
        underlying_evidence,
        proxy_host,
        run_evidence,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"期权研究数据下载失败：{sanitized_error(exc)}", file=sys.stderr, flush=True)
        raise SystemExit(1)
