"""以中证指数官网、东方财富和新浪构建零Token的V3_FORWARD_2输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.refresh_v3_forward_inputs as shared
from research.v3_forward_2_validation import (
    FINANCIALS_FILE,
    FORWARD_CONSTITUENT_CLOSE_FILE,
    WEIGHTS_FILE,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
STATUS_FILE = ROOT / "paper" / "v3_forward_2_input_refresh_status.json"
WEIGHT_SNAPSHOT_LOG = (
    ROOT / "data" / "raw" / "forward" / "v3_forward_2_weight_snapshot_log.parquet"
)
MEMBERSHIP_SNAPSHOT_LOG = (
    ROOT / "data" / "raw" / "forward" / "v3_forward_2_membership_snapshot_log.parquet"
)
FINANCIAL_CHECKPOINT_DIR = (
    ROOT / "data" / "raw" / "forward" / "v3_forward_2_financial_checkpoints"
)
EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
LEGACY_FINANCIALS_FILE = (
    ROOT / "data" / "raw" / "fundamentals" / "csi300_financials_point_in_time.parquet"
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _persist_status(status: dict[str, Any]) -> None:
    """原子保存进度；外部超时终止后仍能看到最后完成或正在运行的步骤。"""

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(STATUS_FILE)


def _timed_step(
    status: dict[str, Any],
    name: str,
    action: Callable[[], Any],
) -> Any:
    started = time.perf_counter()
    status["active_step"] = name
    status["steps"][name] = {
        "status": "RUNNING",
        "started_at": datetime.now(TIMEZONE).isoformat(),
    }
    _persist_status(status)
    try:
        result = action()
    except Exception as exc:
        status["steps"][name].update(
            {
                "status": "FAILED",
                "duration_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        status["failed_step"] = name
        _persist_status(status)
        raise
    detail = result[1] if isinstance(result, tuple) else result
    if not isinstance(detail, dict):
        detail = {"result": detail}
    status["steps"][name] = {
        "status": "SUCCESS",
        "duration_seconds": time.perf_counter() - started,
        **detail,
    }
    status["last_completed_step"] = name
    status["active_step"] = None
    _persist_status(status)
    return result


def _retry(call: Callable[[], Any], label: str, attempts: int = 4) -> Any:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = call()
            if result is None or (hasattr(result, "empty") and result.empty):
                raise ValueError("返回空数据")
            return result
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{label}失败：{type(error).__name__}: {error}")


def _con_code(code: object, exchange: object) -> str:
    text = str(exchange)
    if "上海" in text or "Shanghai" in text:
        suffix = "SH"
    elif "深圳" in text or "Shenzhen" in text:
        suffix = "SZ"
    else:
        raise ValueError(f"无法识别交易所：{exchange}")
    return f"{str(code).zfill(6)}.{suffix}"


def normalize_csindex_weights(
    raw: pd.DataFrame, observed_date: pd.Timestamp, retrieved_at: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"日期", "指数代码", "成分券代码", "交易所", "权重"}
    if missing := required - set(raw.columns):
        raise ValueError(f"中证权重缺少字段：{sorted(missing)}")
    data = raw.copy()
    data["trade_date"] = pd.to_datetime(data["日期"], errors="coerce")
    data["weight"] = pd.to_numeric(data["权重"], errors="coerce")
    data["con_code"] = [
        _con_code(code, exchange)
        for code, exchange in zip(data["成分券代码"], data["交易所"], strict=True)
    ]
    data = data.dropna(subset=["trade_date", "weight"])
    if data["trade_date"].nunique() != 1:
        raise ValueError("中证权重返回多个官方日期")
    official_date = pd.Timestamp(data["trade_date"].iloc[0]).normalize()
    if official_date > observed_date:
        raise ValueError("中证权重官方日期晚于抓取日")
    if data["con_code"].nunique() != 300:
        raise ValueError(f"中证权重成分数不是300：{data['con_code'].nunique()}")
    if not 98.0 <= data["weight"].sum() <= 102.0:
        raise ValueError(f"中证权重之和异常：{data['weight'].sum():.4f}")
    canonical = pd.DataFrame(
        {
            "index_code": "000300.SH",
            "con_code": data["con_code"],
            "trade_date": official_date,
            "weight": data["weight"].astype(float),
            "source": "csindex.via_akshare.index_stock_cons_weight_csindex",
            "retrieved_at": retrieved_at,
        }
    ).sort_values("con_code").reset_index(drop=True)
    snapshot = canonical.copy()
    snapshot.insert(0, "observed_date", observed_date)
    return canonical, snapshot


def normalize_csindex_membership(
    raw: pd.DataFrame, observed_date: pd.Timestamp, retrieved_at: datetime
) -> pd.DataFrame:
    required = {"日期", "指数代码", "成分券代码", "交易所"}
    if missing := required - set(raw.columns):
        raise ValueError(f"中证成分目录缺少字段：{sorted(missing)}")
    data = raw.copy()
    data["source_date"] = pd.to_datetime(data["日期"], errors="coerce")
    data["con_code"] = [
        _con_code(code, exchange)
        for code, exchange in zip(data["成分券代码"], data["交易所"], strict=True)
    ]
    if data["con_code"].nunique() != 300:
        raise ValueError(f"中证成分目录数量不是300：{data['con_code'].nunique()}")
    if data["source_date"].max().normalize() != observed_date:
        raise ValueError(
            f"中证成分目录日期为{data['source_date'].max().date()}，不是目标日{observed_date.date()}"
        )
    return pd.DataFrame(
        {
            "observed_date": observed_date,
            "source_date": data["source_date"],
            "con_code": data["con_code"],
            "source": "csindex.via_akshare.index_stock_cons_csindex",
            "retrieved_at": retrieved_at,
        }
    ).sort_values("con_code").reset_index(drop=True)


def _append_immutable_daily_snapshot(
    path: Path, snapshot: pd.DataFrame, date_column: str, key_columns: list[str]
) -> str:
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not existing.empty:
        existing[date_column] = pd.to_datetime(existing[date_column])
        target = pd.Timestamp(snapshot[date_column].iloc[0])
        same = existing.loc[existing[date_column].eq(target)].copy()
        if not same.empty:
            compare_columns = [column for column in snapshot.columns if column != "retrieved_at"]
            left = same[compare_columns].sort_values(key_columns).reset_index(drop=True)
            right = snapshot[compare_columns].sort_values(key_columns).reset_index(drop=True)
            identical = len(left) == len(right)
            for column in compare_columns:
                if not identical:
                    break
                if "date" in column:
                    identical = bool(
                        np.array_equal(
                            pd.to_datetime(left[column]).to_numpy(dtype="datetime64[ns]"),
                            pd.to_datetime(right[column]).to_numpy(dtype="datetime64[ns]"),
                        )
                    )
                elif pd.api.types.is_numeric_dtype(left[column]) or pd.api.types.is_numeric_dtype(
                    right[column]
                ):
                    identical = bool(
                        np.allclose(
                            pd.to_numeric(left[column], errors="coerce"),
                            pd.to_numeric(right[column], errors="coerce"),
                            rtol=0.0,
                            atol=5e-7,
                            equal_nan=True,
                        )
                    )
                else:
                    identical = left[column].astype(str).equals(right[column].astype(str))
            if identical:
                return "IDEMPOTENT"
            raise ValueError(f"{path.name}同日快照内容变化，禁止覆盖")
    combined = pd.concat([existing, snapshot], ignore_index=True)
    _atomic_parquet(combined.sort_values([date_column, *key_columns]), path)
    return "APPENDED"


def refresh_csindex_snapshots(
    asof: pd.Timestamp, retrieved_at: datetime
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_weights = _retry(
        lambda: ak.index_stock_cons_weight_csindex(symbol="000300"),
        "中证沪深300权重",
    )
    raw_membership = _retry(
        lambda: ak.index_stock_cons_csindex(symbol="000300"),
        "中证沪深300成分目录",
    )
    weights, weight_snapshot = normalize_csindex_weights(
        raw_weights, asof, retrieved_at
    )
    membership = normalize_csindex_membership(raw_membership, asof, retrieved_at)
    if set(weights["con_code"]) != set(membership["con_code"]):
        raise ValueError("中证权重与成分目录的300只证券集合不一致")
    existing = pd.read_parquet(WEIGHTS_FILE) if WEIGHTS_FILE.exists() else pd.DataFrame()
    canonical = pd.concat([existing, weights], ignore_index=True)
    canonical = canonical.drop_duplicates(["trade_date", "con_code"], keep="last")
    _atomic_parquet(canonical.sort_values(["trade_date", "con_code"]), WEIGHTS_FILE)
    weight_action = _append_immutable_daily_snapshot(
        WEIGHT_SNAPSHOT_LOG, weight_snapshot, "observed_date", ["con_code"]
    )
    membership_action = _append_immutable_daily_snapshot(
        MEMBERSHIP_SNAPSHOT_LOG, membership, "observed_date", ["con_code"]
    )
    return weights, {
        "official_weight_date": str(weights["trade_date"].max().date()),
        "observed_date": str(asof.date()),
        "constituent_count": 300,
        "weight_snapshot_action": weight_action,
        "membership_snapshot_action": membership_action,
    }


def _secid(con_code: str) -> str:
    code, exchange = con_code.split(".")
    return f"{1 if exchange == 'SH' else 0}.{code}"


def fetch_total_shares(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start in range(0, len(symbols), 20):
        batch = symbols[start : start + 20]

        def request_batch() -> dict[str, Any]:
            response = requests.get(
                EASTMONEY_QUOTE_URL,
                params={
                    "secids": ",".join(_secid(symbol) for symbol in batch),
                    "fields": "f2,f12,f13,f14,f20",
                    "fltt": "2",
                    "invt": "2",
                },
                timeout=(10, 35),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("data") is None:
                raise ValueError("东方财富批量行情没有data")
            return payload

        payload = _retry(request_batch, "东方财富总股本批量行情")
        for item in payload["data"]["diff"]:
            code = str(item["f12"]).zfill(6)
            suffix = "SH" if int(item["f13"]) == 1 else "SZ"
            price = pd.to_numeric(item["f2"], errors="coerce")
            market_cap = pd.to_numeric(item["f20"], errors="coerce")
            if pd.isna(price) or pd.isna(market_cap) or price <= 0 or market_cap <= 0:
                continue
            shares = round(float(market_cap) / float(price) / 100.0) * 100.0
            rows.append(
                {
                    "con_code": f"{code}.{suffix}",
                    "total_shares": shares,
                }
            )
    result = pd.DataFrame(rows).drop_duplicates("con_code", keep="last")
    if result["con_code"].nunique() < math.ceil(len(symbols) * 0.98):
        raise ValueError(
            f"东方财富总股本覆盖不足：{result['con_code'].nunique()}/{len(symbols)}"
        )
    return result


def _quarter_periods(asof: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    return list(pd.date_range(end=asof, periods=count, freq="QE"))


def _created_on_target_date(path: Path, asof: pd.Timestamp) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, TIMEZONE)
    return modified.date() == asof.date()


def _fallback_total_shares_from_seed(symbols: list[str]) -> pd.DataFrame:
    seed = pd.read_parquet(LEGACY_FINANCIALS_FILE).copy()
    seed = seed.loc[seed["con_code"].astype(str).isin(symbols)]
    seed["report_period"] = pd.to_datetime(seed["report_period"])
    seed["available_at"] = pd.to_datetime(seed["available_at"])
    seed = seed.dropna(subset=["total_shares"]).sort_values(
        ["con_code", "report_period", "available_at"]
    )
    latest = seed.drop_duplicates("con_code", keep="last")
    return latest[["con_code", "total_shares"]].assign(
        total_shares_source="V1切换日前最近已知股本"
    )


def fetch_eastmoney_financial_candidates(
    symbols: list[str], asof: pd.Timestamp, period_count: int
) -> pd.DataFrame:
    symbol_codes = {symbol.split(".")[0]: symbol for symbol in symbols}
    pieces: list[pd.DataFrame] = []
    performance_pieces: list[pd.DataFrame] = []
    for period in _quarter_periods(asof, period_count):
        text = period.strftime("%Y%m%d")
        daily_checkpoint_dir = FINANCIAL_CHECKPOINT_DIR / asof.strftime("%Y%m%d")
        income_checkpoint = daily_checkpoint_dir / f"income_{text}.parquet"
        balance_checkpoint = daily_checkpoint_dir / f"balance_{text}.parquet"
        performance_checkpoint = daily_checkpoint_dir / f"performance_{text}.parquet"
        FINANCIAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        daily_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        legacy_income = FINANCIAL_CHECKPOINT_DIR / f"income_{text}.parquet"
        legacy_balance = FINANCIAL_CHECKPOINT_DIR / f"balance_{text}.parquet"
        legacy_performance = FINANCIAL_CHECKPOINT_DIR / f"performance_{text}.parquet"
        if not income_checkpoint.exists() and _created_on_target_date(legacy_income, asof):
            income = pd.read_parquet(legacy_income)
        elif income_checkpoint.exists():
            income = pd.read_parquet(income_checkpoint)
        else:
            income = _retry(
                lambda text=text: ak.stock_lrb_em(date=text), f"东方财富利润表{text}"
            )
            _atomic_parquet(income, income_checkpoint)
        if not balance_checkpoint.exists() and _created_on_target_date(legacy_balance, asof):
            balance = pd.read_parquet(legacy_balance)
        elif balance_checkpoint.exists():
            balance = pd.read_parquet(balance_checkpoint)
        else:
            balance = _retry(
                lambda text=text: ak.stock_zcfz_em(date=text), f"东方财富资产负债表{text}"
            )
            _atomic_parquet(balance, balance_checkpoint)
        if not performance_checkpoint.exists() and _created_on_target_date(
            legacy_performance, asof
        ):
            performance = pd.read_parquet(legacy_performance)
        elif performance_checkpoint.exists():
            performance = pd.read_parquet(performance_checkpoint)
        else:
            performance = _retry(
                lambda text=text: ak.stock_yjbb_em(date=text), f"东方财富业绩报表{text}"
            )
            _atomic_parquet(performance, performance_checkpoint)
        income = income.loc[income["股票代码"].astype(str).isin(symbol_codes)].copy()
        balance = balance.loc[balance["股票代码"].astype(str).isin(symbol_codes)].copy()
        income["公告日期"] = pd.to_datetime(income["公告日期"], errors="coerce")
        balance["公告日期"] = pd.to_datetime(balance["公告日期"], errors="coerce")
        income = income.loc[income["公告日期"].le(asof)].sort_values("公告日期")
        balance = balance.loc[balance["公告日期"].le(asof)].sort_values("公告日期")
        income = income.drop_duplicates("股票代码", keep="last").rename(
            columns={
                "股票代码": "code",
                "净利润": "net_profit_parent_cny",
                "营业总收入": "revenue_cny",
                "公告日期": "income_announcement_date",
            }
        )
        balance = balance.drop_duplicates("股票代码", keep="last").rename(
            columns={
                "股票代码": "code",
                "股东权益合计": "equity_parent_cny",
                "公告日期": "balance_announcement_date",
            }
        )
        merged = income[
            ["code", "revenue_cny", "net_profit_parent_cny", "income_announcement_date"]
        ].merge(
            balance[["code", "equity_parent_cny", "balance_announcement_date"]],
            on="code",
            how="outer",
            validate="one_to_one",
        )
        merged["con_code"] = merged["code"].map(symbol_codes)
        merged["report_period"] = period.normalize()
        merged["announcement_date"] = merged[
            ["income_announcement_date", "balance_announcement_date"]
        ].max(axis=1)
        pieces.append(merged.drop(columns="code"))
        performance = performance.loc[
            performance["股票代码"].astype(str).isin(symbol_codes)
        ].copy()
        performance["最新公告日期"] = pd.to_datetime(
            performance["最新公告日期"], errors="coerce"
        )
        performance = performance.loc[performance["最新公告日期"].le(asof)]
        performance["con_code"] = performance["股票代码"].astype(str).map(symbol_codes)
        performance["report_period"] = period.normalize()
        performance["calculated_total_shares"] = (
            pd.to_numeric(performance["净利润-净利润"], errors="coerce")
            / pd.to_numeric(performance["每股收益"], errors="coerce")
        )
        performance_pieces.append(
            performance[
                ["con_code", "report_period", "calculated_total_shares"]
            ].replace([np.inf, -np.inf], np.nan)
        )
    candidates = pd.concat(pieces, ignore_index=True)
    candidates = candidates.dropna(subset=["con_code", "announcement_date"])
    performance = pd.concat(performance_pieces, ignore_index=True).dropna(
        subset=["calculated_total_shares"]
    )
    performance = performance.loc[performance["calculated_total_shares"].gt(0)]
    performance = performance.sort_values(["con_code", "report_period"]).drop_duplicates(
        "con_code", keep="last"
    )
    shares = _fallback_total_shares_from_seed(symbols).merge(
        performance[["con_code", "calculated_total_shares"]],
        on="con_code",
        how="outer",
        validate="one_to_one",
    )
    use_calculated = shares["calculated_total_shares"].notna()
    shares.loc[use_calculated, "total_shares"] = shares.loc[
        use_calculated, "calculated_total_shares"
    ]
    shares.loc[use_calculated, "total_shares_source"] = "东方财富净利润/每股收益推算"
    shares = shares.drop(columns="calculated_total_shares")
    if shares["total_shares"].notna().mean() < 0.98:
        raise ValueError("总股本有效覆盖低于98%")
    candidates = candidates.merge(shares, on="con_code", how="left", validate="many_to_one")
    latest_period = candidates.groupby("con_code")["report_period"].transform("max")
    candidates.loc[candidates["report_period"].ne(latest_period), "total_shares"] = np.nan
    candidates["source"] = (
        "eastmoney.via_akshare.stock_lrb_em+stock_zcfz_em;"
        "shares=eastmoney.performance_profit/eps_or_frozen_pre_switch_value"
    )
    return candidates.reset_index(drop=True)


def _financial_content_hash(row: pd.Series) -> str:
    values: list[object] = [
        str(pd.Timestamp(row["announcement_date"]).date()),
        *[
            None if pd.isna(row[column]) else round(float(row[column]), 6)
            for column in (
                "revenue_cny",
                "net_profit_parent_cny",
                "equity_parent_cny",
                "total_shares",
            )
        ],
    ]
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def append_financial_revisions(
    existing: pd.DataFrame,
    candidates: pd.DataFrame,
    observed_date: pd.Timestamp,
    retrieved_at: datetime,
) -> tuple[pd.DataFrame, int]:
    data = existing.copy()
    if not data.empty:
        for column in ("report_period", "available_at", "announcement_date"):
            data[column] = pd.to_datetime(data[column])
    candidates = candidates.copy()
    candidates["content_hash"] = candidates.apply(_financial_content_hash, axis=1)
    latest_hash: dict[tuple[str, pd.Timestamp], str] = {}
    if not data.empty:
        latest = data.sort_values("available_at").drop_duplicates(
            ["con_code", "report_period"], keep="last"
        )
        latest_hash = {
            (str(row.con_code), pd.Timestamp(row.report_period)): str(row.content_hash)
            for row in latest.itertuples(index=False)
        }
    rows: list[dict[str, object]] = []
    for row in candidates.to_dict("records"):
        key = (str(row["con_code"]), pd.Timestamp(row["report_period"]))
        if latest_hash.get(key) == row["content_hash"]:
            continue
        is_revision = key in latest_hash
        available_at = (
            observed_date
            if is_revision
            else max(pd.Timestamp(row["announcement_date"]).normalize(), observed_date)
        )
        rows.append(
            {
                **row,
                "available_at": available_at,
                "retrieved_at": retrieved_at,
                "revision_detected": is_revision,
            }
        )
    if rows:
        data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)
        data = data.sort_values(["con_code", "report_period", "available_at"])
    return data.reset_index(drop=True), len(rows)


def seed_frozen_historical_financials(asof: pd.Timestamp) -> pd.DataFrame:
    """继承切换日前已冻结历史；切换后的新增与修订不再依赖原供应商。"""

    if not LEGACY_FINANCIALS_FILE.exists():
        raise FileNotFoundError("缺少V1已冻结历史财务种子")
    seed = pd.read_parquet(LEGACY_FINANCIALS_FILE).copy()
    for column in ("report_period", "available_at", "announcement_date"):
        seed[column] = pd.to_datetime(seed[column], errors="coerce")
    seed = seed.loc[seed["available_at"].le(asof)].copy()
    seed["content_hash"] = seed.apply(_financial_content_hash, axis=1)
    seed["revision_detected"] = False
    seed["historical_seed"] = True
    return seed.reset_index(drop=True)


def refresh_financials(
    symbols: list[str], asof: pd.Timestamp, retrieved_at: datetime
) -> dict[str, object]:
    existing = (
        pd.read_parquet(FINANCIALS_FILE)
        if FINANCIALS_FILE.exists()
        else seed_frozen_historical_financials(asof)
    )
    existing_periods = (
        pd.to_datetime(existing["report_period"]).nunique() if not existing.empty else 0
    )
    full_refresh = existing_periods < 16 or asof.dayofweek == 0
    period_count = 16 if full_refresh else 5
    daily_checkpoint_dir = FINANCIAL_CHECKPOINT_DIR / asof.strftime("%Y%m%d")
    cached_before = len(list(daily_checkpoint_dir.glob("*.parquet"))) if daily_checkpoint_dir.exists() else 0
    candidates = fetch_eastmoney_financial_candidates(symbols, asof, period_count)
    combined, appended = append_financial_revisions(
        existing, candidates, asof, retrieved_at
    )
    _atomic_parquet(combined, FINANCIALS_FILE)
    eligible = combined.loc[pd.to_datetime(combined["available_at"]).le(asof)]
    covered_current_symbols = set(eligible["con_code"].astype(str)).intersection(symbols)
    coverage = len(covered_current_symbols) / len(symbols)
    if coverage < 0.98:
        raise ValueError(f"东方财富点时财务证券覆盖率仅{coverage:.2%}")
    return {
        "refresh_scope_quarters": period_count,
        "revision_or_new_event_count": appended,
        "eligible_symbol_coverage": float(coverage),
        "latest_announcement_date": str(
            pd.to_datetime(eligible["announcement_date"]).max().date()
        ),
        "checkpoint_directory": daily_checkpoint_dir.relative_to(ROOT).as_posix(),
        "checkpoint_files_before_step": cached_before,
        "checkpoint_files_after_step": (
            len(list(daily_checkpoint_dir.glob("*.parquet")))
            if daily_checkpoint_dir.exists()
            else 0
        ),
        "checkpoint_cache_reused": cached_before > 0,
    }


def refresh_v2_constituent_closes(
    symbols: list[str], asof: pd.Timestamp, retrieved_at: datetime
) -> dict[str, object]:
    if FORWARD_CONSTITUENT_CLOSE_FILE.exists():
        existing = pd.read_parquet(FORWARD_CONSTITUENT_CLOSE_FILE)
        existing["date"] = pd.to_datetime(existing["date"])
        today = existing.loc[
            existing["date"].eq(asof) & existing["con_code"].astype(str).isin(symbols)
        ]
        if today["con_code"].nunique() == len(symbols):
            return {
                "constituent_close_date": str(asof.date()),
                "constituent_close_count": len(symbols),
                "suspended_count": int(today["is_suspended"].sum()),
                "snapshot_action": "IDEMPOTENT",
            }
    with _v2_constituent_paths():
        detail = _retry(
            lambda: shared.refresh_constituent_closes(asof, retrieved_at),
            "新浪成分股收盘快照",
        )
    return {**detail, "snapshot_action": "APPENDED"}


@contextmanager
def _v2_constituent_paths() -> Iterator[None]:
    original_weights = shared.WEIGHTS_FILE
    original_closes = shared.FORWARD_CONSTITUENT_CLOSE_FILE
    try:
        shared.WEIGHTS_FILE = WEIGHTS_FILE
        shared.FORWARD_CONSTITUENT_CLOSE_FILE = FORWARD_CONSTITUENT_CLOSE_FILE
        yield
    finally:
        shared.WEIGHTS_FILE = original_weights
        shared.FORWARD_CONSTITUENT_CLOSE_FILE = original_closes


def refresh_all(asof: pd.Timestamp | None = None) -> dict[str, object]:
    target = pd.Timestamp(asof or datetime.now(TIMEZONE).date()).normalize()
    retrieved_at = datetime.now(TIMEZONE)
    status: dict[str, Any] = {
        "asof": str(target.date()),
        "retrieved_at": retrieved_at.isoformat(),
        "status": "RUNNING",
        "token_or_api_key_used": False,
        "steps": {},
        "active_step": None,
    }
    overall_started = time.perf_counter()
    _persist_status(status)
    try:
        market = _timed_step(
            status,
            "market",
            lambda: shared.refresh_market_series(target, retrieved_at),
        )
        target_text = str(target.date())
        if any(
            market[key] != target_text
            for key in ("000300_latest", "510300_latest", "H00300_latest")
        ):
            status["status"] = "SKIPPED_NON_TRADING_OR_PROVIDER_NOT_READY"
            status["message"] = "三条核心收盘序列尚未共同到达目标日，不生成信号。"
            return status
        _timed_step(
            status,
            "valuation_and_bonds",
            lambda: shared.refresh_valuation_and_bonds(target, retrieved_at),
        )
        weights, weight_detail = _timed_step(
            status,
            "csindex_weights_and_membership",
            lambda: refresh_csindex_snapshots(target, retrieved_at),
        )
        symbols = sorted(weights["con_code"].astype(str).unique())
        _timed_step(
            status,
            "eastmoney_financials",
            lambda: refresh_financials(symbols, target, retrieved_at),
        )
        _timed_step(
            status,
            "constituent_closes",
            lambda: refresh_v2_constituent_closes(symbols, target, retrieved_at),
        )
        status["status"] = "SUCCESS"
        return status
    except Exception as exc:
        status["status"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["duration_seconds"] = time.perf_counter() - overall_started
        status["active_step"] = None if status["status"] != "RUNNING" else status["active_step"]
        _persist_status(status)


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新V3_FORWARD_2零Token前瞻输入")
    parser.add_argument("--date", help="目标日期YYYY-MM-DD；仅允许当日运行")
    args = parser.parse_args()
    try:
        result = refresh_all(pd.Timestamp(args.date) if args.date else None)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"V3_FORWARD_2输入刷新失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
