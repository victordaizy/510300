"""采集 510300 状态路由 V1 所需的中债国债收益率长历史。

该脚本只补齐折现率输入，不读取 510300 未来收益，不覆盖共享的旧数据文件。
早期窗口通过中债官方曲线接口（由 AKShare 适配）取得，并与现有 2016 年以来
批次做重叠逐日校验；任何不一致都会停止，不做插值或来源替换。
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "macro" / "china_government_bond_yields_daily.parquet"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "regime_transition_router_v1" / "cgb_curve"
)
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
OUTPUT_FILE = OUTPUT_ROOT / "cgb_1y_10y_daily_2012_2026.parquet"
REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "510300_regime_transition_router_v1_cgb.json"
)

START_DATE = pd.Timestamp("2012-01-01")
OVERLAP_END_DATE = pd.Timestamp("2016-11-30")
MAX_WINDOW_CALENDAR_DAYS = 89
CURVE_NAME_COLUMN = "曲线名称"
DATE_COLUMN = "日期"
ONE_YEAR_COLUMN = "1年"
TEN_YEAR_COLUMN = "10年"
GOVERNMENT_CURVE_NAME = "中债国债收益率曲线"
SOURCE_ID = "chinabond.via_akshare.bond_china_yield"


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_download_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    maximum_calendar_days: int = MAX_WINDOW_CALENDAR_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """生成连续、互不重叠的闭区间。"""

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError("结束日期不能早于开始日期")
    if maximum_calendar_days < 1:
        raise ValueError("单窗口自然日数必须为正")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.Timedelta(days=maximum_calendar_days), end)
        windows.append((cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def normalize_curve_response(
    raw: pd.DataFrame,
    *,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """筛选国债曲线并保留统一单位的 1 年、10 年收益率。"""

    required = {
        CURVE_NAME_COLUMN,
        DATE_COLUMN,
        ONE_YEAR_COLUMN,
        TEN_YEAR_COLUMN,
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"中债响应缺少字段：{sorted(missing)}")
    result = raw.loc[
        raw[CURVE_NAME_COLUMN].eq(GOVERNMENT_CURVE_NAME),
        [DATE_COLUMN, ONE_YEAR_COLUMN, TEN_YEAR_COLUMN],
    ].copy()
    result.columns = ["date", "cgb_1y", "cgb_10y"]
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result[["cgb_1y", "cgb_10y"]] = result[["cgb_1y", "cgb_10y"]].apply(
        pd.to_numeric,
        errors="raise",
    )
    result.drop_duplicates("date", keep="last", inplace=True)
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.empty:
        raise ValueError("中债响应没有国债曲线记录")
    if result[["cgb_1y", "cgb_10y"]].isna().any(axis=None):
        raise ValueError("中债国债收益率含缺失值")
    if result[["cgb_1y", "cgb_10y"]].le(0.0).any(axis=None):
        raise ValueError("中债国债收益率含非正值，需人工核查单位")
    result["source"] = SOURCE_ID
    result["retrieved_at"] = retrieved_at
    result["available_at_rule"] = "曲线日期当日17:30后；下一交易日开盘最早可执行"
    return result


def _checkpoint_path(start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return CHECKPOINT_ROOT / f"{start:%Y%m%d}_{end:%Y%m%d}.parquet"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def acquire_early_curve(
    fetcher: Callable[..., pd.DataFrame],
    *,
    retrieved_at: datetime,
    sleep_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """断点采集早期曲线；失败窗口最多重试三次。"""

    pieces: list[pd.DataFrame] = []
    checkpoints_reused = 0
    checkpoints_created = 0
    errors: list[str] = []
    windows = build_download_windows(START_DATE, OVERLAP_END_DATE)
    for sequence, (window_start, window_end) in enumerate(windows, start=1):
        checkpoint = _checkpoint_path(window_start, window_end)
        if checkpoint.exists():
            normalized = pd.read_parquet(checkpoint)
            normalized["date"] = pd.to_datetime(
                normalized["date"], errors="raise"
            ).dt.normalize()
            checkpoints_reused += 1
        else:
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    raw = fetcher(
                        start_date=window_start.strftime("%Y%m%d"),
                        end_date=window_end.strftime("%Y%m%d"),
                    )
                    normalized = normalize_curve_response(
                        raw,
                        retrieved_at=retrieved_at,
                    )
                    break
                except Exception as exception:  # pragma: no cover - 依赖外部网络
                    last_error = exception
                    errors.append(
                        f"{window_start.date()}至{window_end.date()}第{attempt}次："
                        f"{type(exception).__name__}: {exception}"
                    )
                    if attempt < 3:
                        time.sleep(2.0 * attempt)
            else:  # pragma: no cover - 依赖外部网络
                raise RuntimeError(
                    f"中债窗口{window_start.date()}至{window_end.date()}采集失败："
                    f"{type(last_error).__name__}: {last_error}"
                )
            _atomic_parquet(normalized, checkpoint)
            checkpoints_created += 1
        pieces.append(normalized)
        print(
            f"中债早期曲线进度 {sequence}/{len(windows)}："
            f"{window_start.date()} 至 {window_end.date()}",
            flush=True,
        )
        if sleep_seconds > 0 and sequence < len(windows):
            time.sleep(sleep_seconds)
    combined = pd.concat(pieces, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise").dt.normalize()
    combined.drop_duplicates("date", keep="last", inplace=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined, {
        "window_count": len(windows),
        "checkpoints_reused": checkpoints_reused,
        "checkpoints_created": checkpoints_created,
        "retry_error_count": len(errors),
        "retry_errors": errors,
    }


def combine_with_existing(
    early: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """重叠校验后拼接早期与现有批次。"""

    early = early.copy()
    existing = existing.copy()
    for frame in (early, existing):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["date"].duplicated().any():
            raise ValueError("国债曲线输入存在重复日期")
    required = {"date", "cgb_1y", "cgb_10y", "source", "retrieved_at"}
    if missing := required.difference(existing.columns):
        raise ValueError(f"现有国债曲线缺少字段：{sorted(missing)}")

    overlap = early[["date", "cgb_1y", "cgb_10y"]].merge(
        existing[["date", "cgb_1y", "cgb_10y"]],
        on="date",
        how="inner",
        suffixes=("_early", "_existing"),
        validate="one_to_one",
    )
    if len(overlap) < 20:
        raise ValueError(f"早期批次与现有批次重叠不足：{len(overlap)}行")
    differences: dict[str, float] = {}
    for column in ["cgb_1y", "cgb_10y"]:
        maximum = float(
            (
                pd.to_numeric(overlap[f"{column}_early"], errors="raise")
                - pd.to_numeric(overlap[f"{column}_existing"], errors="raise")
            )
            .abs()
            .max()
        )
        differences[column] = maximum
        if maximum > 1e-10:
            raise ValueError(f"国债曲线重叠字段{column}最大差异{maximum}超过容差")

    first_existing = existing["date"].min()
    early_only = early.loc[early["date"].lt(first_existing)].copy()
    if "available_at_rule" not in existing.columns:
        existing["available_at_rule"] = "曲线日期当日17:30后；下一交易日开盘最早可执行"
    columns = [
        "date",
        "cgb_1y",
        "cgb_10y",
        "source",
        "retrieved_at",
        "available_at_rule",
    ]
    combined = pd.concat(
        [early_only[columns], existing[columns]],
        ignore_index=True,
    )
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    if combined["date"].duplicated().any():
        raise ValueError("拼接后的国债曲线存在重复日期")
    if combined[["cgb_1y", "cgb_10y"]].isna().any(axis=None):
        raise ValueError("拼接后的国债曲线存在缺失值")
    if combined["date"].min() > START_DATE + pd.Timedelta(days=10):
        raise ValueError("国债曲线首日距冻结起点超过10个自然日")
    if combined["date"].max() < pd.Timestamp("2026-08-12"):
        raise ValueError("国债曲线未覆盖状态研究截止日")
    return combined, {
        "overlap_rows": int(len(overlap)),
        "maximum_absolute_difference_pct_point": differences,
        "first_existing_date": first_existing.date().isoformat(),
        "early_only_rows": int(len(early_only)),
    }


def main() -> int:
    """执行采集、拼接和原子落盘。"""

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"现有国债曲线不存在：{SOURCE_FILE}")
    import akshare as ak

    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    source_sha256_before = sha256_file(SOURCE_FILE)
    existing = pd.read_parquet(SOURCE_FILE)
    early, acquisition = acquire_early_curve(
        ak.bond_china_yield,
        retrieved_at=retrieved_at,
    )
    combined, overlap = combine_with_existing(early, existing)
    _atomic_parquet(combined, OUTPUT_FILE)
    payload = {
        "project_id": "510300_REGIME_TRANSITION_ROUTER_V1",
        "status": "PASS_CGB_LONG_HISTORY_ACQUISITION",
        "generated_at_asia_shanghai": retrieved_at.isoformat(),
        "source_contract": {
            "source": SOURCE_ID,
            "compiler": "中央国债登记结算有限责任公司",
            "unit": "百分数",
            "publication_time_local": "17:30",
            "tls_certificate_bypass_used": False,
            "proxy_or_interpolation_used": False,
            "shared_source_file_overwritten": False,
            "checkpoint_semantics": "标准化官方响应；未保存HTTP报文头",
        },
        "existing_source_file": {
            "path": SOURCE_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "sha256_before": source_sha256_before,
            "sha256_after": sha256_file(SOURCE_FILE),
            "unchanged": source_sha256_before == sha256_file(SOURCE_FILE),
        },
        "acquisition": acquisition,
        "overlap_validation": overlap,
        "output": {
            "path": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(OUTPUT_FILE),
            "rows": int(len(combined)),
            "first_date": combined["date"].min().date().isoformat(),
            "last_date": combined["date"].max().date().isoformat(),
            "duplicate_dates": int(combined["date"].duplicated().sum()),
            "missing_values": int(
                combined[["cgb_1y", "cgb_10y"]].isna().sum().sum()
            ),
        },
        "boundaries": {
            "market_price_read": False,
            "future_return_read": False,
            "strategy_result_read": False,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(payload, REPORT_FILE)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
