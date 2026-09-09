"""利用已授权代理补齐成分股日线、点时行业区间和历史研报原始记录。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.return_tail_auxiliary_acquisition import (
    active_members,
    build_constituent_panel,
    industry_coverage,
    normalize_analyst_reports,
    normalize_industry_intervals,
    normalize_stock_history,
)
from scripts.download_510300_option_research_data import (
    RateLimiter,
    atomic_json,
    atomic_parquet,
    load_proxy,
    now,
    retry_call,
    sanitized_error,
)


MEMBERSHIP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
CALENDAR_FILE = ROOT / "data" / "raw" / "return_tail" / "options" / "sse_trading_calendar.parquet"
DAILY_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
DAILY_BACKUP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily_pre_tushare_backfill.parquet"
STOCK_CACHE_DIR = ROOT / "data" / "raw" / "constituents" / ".tushare_stock_history_cache"
INDUSTRY_FILE = ROOT / "data" / "raw" / "reference" / "a_share_sw_industry_static.parquet"
INDUSTRY_BACKUP_FILE = ROOT / "data" / "raw" / "reference" / "a_share_sw_industry_static_pre_pit.parquet"
INDUSTRY_RAW_FILE = ROOT / "data" / "raw" / "reference" / "a_share_sw_industry_intervals_raw.parquet"
INDUSTRY_PARTIAL_FILE = ROOT / "data" / "raw" / "reference" / "a_share_sw_industry_point_in_time_partial.parquet"
INDUSTRY_CACHE_DIR = ROOT / "data" / "raw" / "reference" / ".sw_industry_member_cache"
REPORT_CACHE_DIR = ROOT / "data" / "raw" / "return_tail" / "earnings" / ".report_rc_cache"
REPORT_SYMBOL_CACHE_DIR = ROOT / "data" / "raw" / "return_tail" / "earnings" / ".report_rc_symbol_cache"
ANALYST_REPORT_FILE = ROOT / "data" / "raw" / "return_tail" / "earnings" / "csi300_analyst_reports_point_in_time.parquet"
STATUS_FILE = ROOT / "reports" / "data_quality" / "return_tail_auxiliary_acquisition.json"


def _calendar(pro: Any, token: str, limiter: RateLimiter, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    raw = retry_call(
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            is_open="1",
            fields="cal_date,is_open",
        ),
        limiter,
        "下载成分股扩展交易日历",
        token,
    )
    return sorted(pd.to_datetime(raw["cal_date"], format="%Y%m%d").dt.normalize().tolist())


def _download_symbol(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    path = STOCK_CACHE_DIR / f"{symbol.replace('.', '_')}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if {
            "date",
            "con_code",
            "total_return_close",
            "raw_close",
        }.issubset(cached.columns):
            return cached
    daily = retry_call(
        lambda: pro.daily(
            ts_code=symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        ),
        limiter,
        f"下载{symbol}日线",
        token,
    )
    factor = retry_call(
        lambda: pro.adj_factor(
            ts_code=symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        ),
        limiter,
        f"下载{symbol}复权因子",
        token,
    )
    result = normalize_stock_history(daily, factor, symbol, now())
    atomic_parquet(result, path)
    return result


def download_constituents(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    workers: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    intervals = pd.read_parquet(MEMBERSHIP_FILE)
    project_calendar = pd.read_parquet(CALENDAR_FILE)
    output_dates = sorted(
        pd.to_datetime(project_calendar["trade_date"]).dt.normalize().tolist()
    )
    warmup_start = start - pd.Timedelta(days=31)
    extended_dates = _calendar(pro, token, limiter, warmup_start, end)
    relevant = intervals.loc[
        pd.to_datetime(intervals["opt_in"], errors="coerce").le(end)
        & (
            pd.to_datetime(intervals["opt_out"], errors="coerce").isna()
            | pd.to_datetime(intervals["opt_out"], errors="coerce").gt(start)
        )
    ]
    symbols = sorted(relevant["symbol"].astype(str).unique())
    STOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="成分股日线") as pool:
        futures = {
            pool.submit(_download_symbol, pro, token, limiter, symbol, warmup_start, end): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                failures[symbol] = sanitized_error(exc, token)
            if completed == 1 or completed % 50 == 0 or completed == len(symbols):
                print(
                    f"成分股全历史进度：{completed}/{len(symbols)}，失败{len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"仍有{len(failures)}只成分股失败：{dict(list(failures.items())[:10])}")
    histories = pd.concat(frames, ignore_index=True)
    panel = build_constituent_panel(histories, relevant, extended_dates, output_dates)
    comparison_file = DAILY_BACKUP_FILE if DAILY_BACKUP_FILE.exists() else DAILY_FILE
    previous = pd.read_parquet(comparison_file) if comparison_file.exists() else pd.DataFrame()
    overlap_checks: dict[str, Any] = {}
    if not previous.empty:
        previous_members = previous.loc[previous["is_index_member"].fillna(False).astype(bool)]
        comparison = previous_members[["date", "con_code", "raw_close"]].merge(
            panel[["date", "con_code", "raw_close"]],
            on=["date", "con_code"],
            suffixes=("_previous", "_tushare"),
        )
        difference = (
            pd.to_numeric(comparison["raw_close_previous"], errors="coerce")
            - pd.to_numeric(comparison["raw_close_tushare"], errors="coerce")
        ).abs()
        comparable = difference.notna()
        overlap_checks = {
            "overlap_row_count": int(len(comparison)),
            "comparable_close_row_count": int(comparable.sum()),
            "noncomparable_close_row_count": int((~comparable).sum()),
            "maximum_raw_close_difference": float(difference.loc[comparable].max()),
            "raw_close_exact_match_ratio": float(
                difference.loc[comparable].eq(0).mean()
            ),
        }
        if not DAILY_BACKUP_FILE.exists():
            DAILY_BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DAILY_FILE, DAILY_BACKUP_FILE)
    atomic_parquet(panel, DAILY_FILE)
    counts = panel.groupby("date")["con_code"].nunique()
    return {
        "status": "PASS",
        "file": DAILY_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(panel)),
        "symbol_count": int(panel["con_code"].nunique()),
        "trading_day_count": int(panel["date"].nunique()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "minimum_members_per_day": int(counts.min()),
        "maximum_members_per_day": int(counts.max()),
        "suspended_member_rows": int(panel["is_suspended"].sum()),
        "valid_total_return_close_ratio": float(panel["total_return_close"].notna().mean()),
        "cache_file_count": int(len(list(STOCK_CACHE_DIR.glob("*.parquet")))),
        "previous_file_backup": DAILY_BACKUP_FILE.relative_to(ROOT).as_posix(),
        "cross_source_overlap": overlap_checks,
    }


def download_industry(
    pro: Any, token: str, limiter: RateLimiter, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    INDUSTRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    classification_frames: list[pd.DataFrame] = []
    for source in ("SW2014", "SW2021"):
        classification = retry_call(
            lambda source=source: pro.index_classify(level="L1", src=source),
            limiter,
            f"下载{source}申万一级行业目录",
            token,
        )
        if classification is None or classification.empty:
            raise RuntimeError(f"{source}申万一级行业目录为空")
        classification = classification.copy()
        classification["classification_source"] = source
        classification_frames.append(classification)
    classifications = pd.concat(classification_frames, ignore_index=True)
    level_one_codes = sorted(classifications["index_code"].dropna().astype(str).unique())

    frames: list[pd.DataFrame] = []
    downloaded_query_count = 0
    reused_query_count = 0
    total_query_count = len(level_one_codes) * 2
    completed_query_count = 0
    for level_one_code in level_one_codes:
        safe_code = level_one_code.replace(".", "_")
        for current_flag in ("N", "Y"):
            cache_file = INDUSTRY_CACHE_DIR / f"{safe_code}_{current_flag}.parquet"
            if cache_file.exists():
                part = pd.read_parquet(cache_file)
                reused_query_count += 1
            else:
                part = retry_call(
                    lambda level_one_code=level_one_code, current_flag=current_flag: (
                        pro.index_member_all(
                            l1_code=level_one_code,
                            is_new=current_flag,
                        )
                    ),
                    limiter,
                    f"下载申万行业{level_one_code}成员区间is_new={current_flag}",
                    token,
                )
                if part is None:
                    part = pd.DataFrame()
                atomic_parquet(part, cache_file)
                downloaded_query_count += 1
            if not part.empty:
                frames.append(part)
            completed_query_count += 1
            if completed_query_count == 1 or completed_query_count % 10 == 0:
                print(
                    f"申万行业成员进度：{completed_query_count}/{total_query_count}个查询",
                    flush=True,
                )
    if not frames:
        raise RuntimeError("按一级行业拆分后申万行业成员区间仍全部为空")
    raw = pd.concat(frames, ignore_index=True).drop_duplicates()
    raw["source"] = "tushare_proxy.index_member_all"
    raw["retrieved_at"] = str(now())
    atomic_parquet(raw, INDUSTRY_RAW_FILE)
    intervals = pd.read_parquet(MEMBERSHIP_FILE)
    universe = set(intervals["symbol"].astype(str))
    industry = normalize_industry_intervals(raw, universe, now())
    calendar = pd.read_parquet(CALENDAR_FILE)
    dates = pd.to_datetime(calendar["trade_date"]).dt.normalize()
    members = active_members(intervals, dates)
    coverage = industry_coverage(industry, members)
    if coverage["coverage_ratio"] < 0.99 or coverage["overlap_ratio"] > 0.001:
        atomic_parquet(industry, INDUSTRY_PARTIAL_FILE)
        return {
            "status": "PARTIAL_PRE_2021_CLASSIFICATION_GAP",
            "file": INDUSTRY_PARTIAL_FILE.relative_to(ROOT).as_posix(),
            "configured_static_file_replaced": False,
            "row_count": int(len(industry)),
            "symbol_count": int(industry["con_code"].nunique()),
            "level_one_code_count": int(len(level_one_codes)),
            "downloaded_query_count": downloaded_query_count,
            "reused_query_count": reused_query_count,
            "reason": "按申万一级行业拆分抓取后，严格成员日覆盖仍不足99%或区间重叠超限。",
            **coverage,
        }
    if INDUSTRY_FILE.exists() and not INDUSTRY_BACKUP_FILE.exists():
        INDUSTRY_BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(INDUSTRY_FILE, INDUSTRY_BACKUP_FILE)
    atomic_parquet(industry, INDUSTRY_FILE)
    return {
        "status": "PASS",
        "file": INDUSTRY_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(industry)),
        "symbol_count": int(industry["con_code"].nunique()),
        "first_in_date": str(industry["in_date"].min().date()),
        "last_in_date": str(industry["in_date"].max().date()),
        "backup": INDUSTRY_BACKUP_FILE.relative_to(ROOT).as_posix(),
        "level_one_code_count": int(len(level_one_codes)),
        "downloaded_query_count": downloaded_query_count,
        "reused_query_count": reused_query_count,
        **coverage,
    }


def _report_ranges(start: pd.Timestamp, end: pd.Timestamp, days: int = 7) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    left = start
    while left <= end:
        right = min(left + pd.Timedelta(days=days - 1), end)
        ranges.append((left, right))
        left = right + pd.Timedelta(days=1)
    return ranges


def _download_report_range(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    left: pd.Timestamp,
    right: pd.Timestamp,
) -> list[pd.DataFrame]:
    path = REPORT_CACHE_DIR / f"{left:%Y%m%d}_{right:%Y%m%d}.parquet"
    if path.exists():
        return [pd.read_parquet(path)]
    raw = retry_call(
        lambda: pro.report_rc(
            start_date=left.strftime("%Y%m%d"), end_date=right.strftime("%Y%m%d")
        ),
        limiter,
        f"下载分析师研报{left:%Y%m%d}-{right:%Y%m%d}",
        token,
    )
    if raw is None:
        raw = pd.DataFrame()
    if len(raw) >= 5000:
        if left == right:
            raise RuntimeError(f"{left.date()}单日研报达到5000行，接口可能截断")
        midpoint = left + pd.Timedelta(days=(right - left).days // 2)
        return _download_report_range(pro, token, limiter, left, midpoint) + _download_report_range(
            pro, token, limiter, midpoint + pd.Timedelta(days=1), right
        )
    raw["retrieved_at"] = str(now())
    atomic_parquet(raw, path)
    return [raw]


def _download_report_symbol(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    symbol: str,
    left: pd.Timestamp,
    right: pd.Timestamp,
) -> list[pd.DataFrame]:
    safe_symbol = symbol.replace(".", "_")
    path = REPORT_SYMBOL_CACHE_DIR / f"{safe_symbol}_{left:%Y%m%d}_{right:%Y%m%d}.parquet"
    if path.exists():
        return [pd.read_parquet(path)]
    raw = retry_call(
        lambda: pro.report_rc(
            ts_code=symbol,
            start_date=left.strftime("%Y%m%d"),
            end_date=right.strftime("%Y%m%d"),
        ),
        limiter,
        f"下载{symbol}历史研报{left:%Y%m%d}-{right:%Y%m%d}",
        token,
    )
    if raw is None:
        raw = pd.DataFrame()
    if len(raw) >= 5000:
        if left == right:
            raise RuntimeError(f"{symbol}在{left.date()}单日研报达到5000行，接口可能截断")
        midpoint = left + pd.Timedelta(days=(right - left).days // 2)
        return _download_report_symbol(
            pro, token, limiter, symbol, left, midpoint
        ) + _download_report_symbol(
            pro, token, limiter, symbol, midpoint + pd.Timedelta(days=1), right
        )
    raw["retrieved_at"] = str(now())
    atomic_parquet(raw, path)
    return [raw]


def download_reports(
    pro: Any,
    token: str,
    limiter: RateLimiter,
    workers: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    REPORT_SYMBOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_parquet(MEMBERSHIP_FILE)
    symbols = sorted(
        intervals.loc[
            pd.to_datetime(intervals["opt_in"], errors="coerce").le(end)
            & (
                pd.to_datetime(intervals["opt_out"], errors="coerce").isna()
                | pd.to_datetime(intervals["opt_out"], errors="coerce").gt(start)
            ),
            "symbol",
        ]
        .astype(str)
        .unique()
    )
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="历史研报") as pool:
        futures = {
            pool.submit(
                _download_report_symbol, pro, token, limiter, symbol, start, end
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            frames.extend(future.result())
            if completed == 1 or completed % 50 == 0 or completed == len(symbols):
                print(f"历史研报进度：{completed}/{len(symbols)}只证券", flush=True)
    nonempty = [frame for frame in frames if not frame.empty]
    raw = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    universe = set(intervals["symbol"].astype(str))
    reports = normalize_analyst_reports(raw, universe, now())
    atomic_parquet(reports, ANALYST_REPORT_FILE)
    return {
        "status": "ACQUIRED_NOT_CONSENSUS_VINTAGE",
        "file": ANALYST_REPORT_FILE.relative_to(ROOT).as_posix(),
        "row_count": int(len(reports)),
        "symbol_count": int(reports["ts_code"].nunique()),
        "report_day_count": int(reports["report_date"].nunique()),
        "first_report_date": str(reports["report_date"].min().date()),
        "last_report_date": str(reports["report_date"].max().date()),
        "eps_non_null_ratio": float(reports["eps"].notna().mean()),
        "consensus_vintage_file_created": False,
        "reason": "历史单篇研报可审计，但不是供应商逐日一致预期vintage，不得解除E1停止条件。",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载尾部风险辅助研究数据")
    parser.add_argument("--phase", choices=["all", "constituents", "industry", "reports"], default="all")
    parser.add_argument("--start-date", default="2019-12-23")
    parser.add_argument("--end-date", default="2026-08-14")
    parser.add_argument("--interval", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 0.5:
        raise ValueError("代理请求间隔不得低于0.5秒")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("并发数必须介于1至4")
    start, end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    pro, token, endpoint = load_proxy()
    limiter = RateLimiter(args.interval)
    previous: dict[str, Any] = {}
    if STATUS_FILE.exists():
        try:
            previous = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    phases = dict(previous.get("phases", {}))
    if args.phase in {"all", "constituents"}:
        phases["constituents"] = download_constituents(
            pro, token, limiter, args.workers, start, end
        )
    if args.phase in {"all", "industry"}:
        phases["industry"] = download_industry(pro, token, limiter, start, end)
    if args.phase in {"all", "reports"}:
        phases["analyst_reports"] = download_reports(
            pro, token, limiter, args.workers, start, end
        )
    status = {
        "status": "COMPLETE_WITH_E1_PROTOCOL_LIMITATION",
        "checked_at": now().isoformat(),
        "proxy_host": endpoint.split("//", 1)[-1],
        "token_persisted_in_outputs": False,
        "phases": phases,
    }
    atomic_json(status, STATUS_FILE)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"辅助数据下载失败：{sanitized_error(exc)}", file=sys.stderr, flush=True)
        raise SystemExit(1)
