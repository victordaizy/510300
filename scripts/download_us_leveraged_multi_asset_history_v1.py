"""采集美国杠杆多资产V1的价格、公司行动、汇率与基准输入。"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "us_leveraged_multi_asset_absolute_momentum_v1.yaml"


class AcquisitionContractError(ValueError):
    """输入无法满足冻结采集合同。"""


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取候选合同，并验证采集阶段不可越过的核心边界。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcquisitionContractError("候选合同必须是YAML对象")
    expected = ["UPRO", "TQQQ", "TMF", "UGL"]
    if payload["universe"]["fixed_tickers"] != expected:
        raise AcquisitionContractError("固定产品集合发生变化")
    if payload["objective"]["initial_capital_cny"] != 500000.0:
        raise AcquisitionContractError("初始资金必须固定为50万元")
    if payload["objective"]["minimum_annualized_net_excess"] != 0.40:
        raise AcquisitionContractError("年化净超额硬门必须为40个百分点")
    if payload["objective"]["minimum_strategy_net_sharpe"] != 1.50:
        raise AcquisitionContractError("净夏普硬门必须为1.50")
    if payload["objective"]["user_transaction_fee_rate_per_leg"] != 0.0001:
        raise AcquisitionContractError("用户单边交易费必须为万分之一")
    if payload["data_sources"]["strategy_total_return_or_rank_may_be_computed_during_acquisition"]:
        raise AcquisitionContractError("采集阶段不得计算策略收益或排名")
    return payload


def sha256_file(path: Path) -> str:
    """计算单个产物的SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """原子写入Parquet，避免中断留下半成品。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入JSON状态。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _numeric(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise AcquisitionContractError(f"{label}缺少字段：{column}")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[columns].isna().any().any():
        bad = result.loc[result[columns].isna().any(axis=1)].head(3)
        raise AcquisitionContractError(f"{label}存在无法解析的数值：{bad.to_dict('records')}")
    return result


def normalize_security_history(
    ticker: str,
    raw_frame: pd.DataFrame,
    factor_frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """把未复权日线与因子事件规范为可交易价格和单证券信号输入。"""

    raw_required = ["date", "open", "high", "low", "close", "volume"]
    missing = sorted(set(raw_required) - set(raw_frame.columns))
    if missing:
        raise AcquisitionContractError(f"{ticker}未复权日线缺少字段：{missing}")
    raw = raw_frame.loc[:, raw_required].copy()
    raw["date"] = (
        pd.to_datetime(raw["date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    )
    raw = _numeric(raw, ["open", "high", "low", "close", "volume"], f"{ticker}未复权日线")
    raw = raw.loc[raw["date"].between(start, end)].dropna(subset=["date"])
    raw = raw.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if raw.empty:
        raise AcquisitionContractError(f"{ticker}在冻结日期范围内没有日线")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any():
        raise AcquisitionContractError(f"{ticker}存在非正价格")
    if (raw["volume"] < 0).any():
        raise AcquisitionContractError(f"{ticker}存在负成交量")
    if (raw["high"] < raw[["open", "close", "low"]].max(axis=1)).any():
        raise AcquisitionContractError(f"{ticker}存在最高价小于开盘、收盘或最低价")
    if (raw["low"] > raw[["open", "close", "high"]].min(axis=1)).any():
        raise AcquisitionContractError(f"{ticker}存在最低价大于开盘、收盘或最高价")

    factor_required = ["date", "qfq_factor", "adjust"]
    missing = sorted(set(factor_required) - set(factor_frame.columns))
    if missing:
        raise AcquisitionContractError(f"{ticker}复权因子表缺少字段：{missing}")
    factors = factor_frame.loc[:, factor_required].copy()
    factors["date"] = (
        pd.to_datetime(factors["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    factors = _numeric(factors, ["qfq_factor", "adjust"], f"{ticker}复权因子表")
    factors = factors.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    factors = factors.sort_values("date").reset_index(drop=True)
    if factors.empty or factors.iloc[0]["date"] > raw.iloc[0]["date"]:
        raise AcquisitionContractError(f"{ticker}复权因子无法覆盖首个日线日期")
    if (factors["qfq_factor"] <= 0).any():
        raise AcquisitionContractError(f"{ticker}存在非正复权因子")

    panel = pd.merge_asof(
        raw,
        factors.rename(columns={"date": "factor_effective_date"}),
        left_on="date",
        right_on="factor_effective_date",
        direction="backward",
        allow_exact_matches=True,
    )
    if panel[["qfq_factor", "adjust", "factor_effective_date"]].isna().any().any():
        raise AcquisitionContractError(f"{ticker}存在未映射复权因子的日线")

    previous_factor = panel["qfq_factor"].shift(1)
    previous_adjust = panel["adjust"].shift(1)
    panel["split_ratio_at_open"] = (panel["qfq_factor"] / previous_factor).fillna(1.0)
    panel.loc[np.isclose(panel["split_ratio_at_open"], 1.0, rtol=0.0, atol=1e-12), "split_ratio_at_open"] = 1.0
    adjustment_change = (panel["adjust"] - previous_adjust).fillna(0.0)
    panel["cash_distribution_per_post_event_share_usd"] = adjustment_change / panel["qfq_factor"]
    tolerance = 1e-9
    if (panel["split_ratio_at_open"] <= 0).any():
        raise AcquisitionContractError(f"{ticker}存在非正拆分比率")
    if (panel["cash_distribution_per_post_event_share_usd"] < -tolerance).any():
        bad = panel.loc[
            panel["cash_distribution_per_post_event_share_usd"] < -tolerance,
            ["date", "qfq_factor", "adjust", "cash_distribution_per_post_event_share_usd"],
        ].head(3)
        raise AcquisitionContractError(f"{ticker}现金分配推导为负：{bad.to_dict('records')}")
    panel["cash_distribution_per_post_event_share_usd"] = panel[
        "cash_distribution_per_post_event_share_usd"
    ].clip(lower=0.0)

    panel["split_adjusted_close"] = panel["close"] * panel["qfq_factor"]
    adjusted_distribution = (
        panel["cash_distribution_per_post_event_share_usd"] * panel["qfq_factor"]
    )
    gross = (panel["split_adjusted_close"] + adjusted_distribution) / panel[
        "split_adjusted_close"
    ].shift(1)
    gross.iloc[0] = 1.0
    if (~np.isfinite(gross)).any() or (gross <= 0).any():
        raise AcquisitionContractError(f"{ticker}单证券总收益指数存在非正或非有限增长因子")
    panel["signal_total_return_index"] = 100.0 * gross.cumprod()
    panel["raw_dollar_turnover_usd"] = panel["close"] * panel["volume"]
    panel["ticker"] = ticker
    panel = panel.rename(
        columns={
            "open": "raw_open",
            "high": "raw_high",
            "low": "raw_low",
            "close": "raw_close",
            "volume": "raw_volume",
        }
    )
    panel["source_raw"] = "akshare.stock_us_daily(adjust='')"
    panel["source_factor"] = "akshare.stock_us_daily(adjust='qfq-factor')"
    columns = [
        "ticker",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_dollar_turnover_usd",
        "factor_effective_date",
        "qfq_factor",
        "adjust",
        "split_ratio_at_open",
        "cash_distribution_per_post_event_share_usd",
        "split_adjusted_close",
        "signal_total_return_index",
        "source_raw",
        "source_factor",
    ]
    result = panel.loc[:, columns].copy()
    if result[["ticker", "date"]].duplicated().any():
        raise AcquisitionContractError(f"{ticker}规范面板存在重复主键")
    return result


def normalize_boc_usd_cny(frame: pd.DataFrame) -> pd.DataFrame:
    """规范新浪财经中行牌价表中的美元中行折算价。"""

    if frame.shape[1] < 6:
        raise AcquisitionContractError("中行美元历史牌价少于六列")
    date_column = frame.columns[0]
    conversion_column = frame.columns[5]
    result = frame[[date_column, conversion_column]].rename(
        columns={date_column: "date", conversion_column: "cny_per_usd"}
    )
    result["date"] = (
        pd.to_datetime(result["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    result["cny_per_usd"] = pd.to_numeric(result["cny_per_usd"], errors="coerce")
    result = result.dropna().drop_duplicates("date", keep="last").sort_values("date")
    if not result.empty and float(result["cny_per_usd"].median()) > 20.0:
        result["cny_per_usd"] = result["cny_per_usd"] / 100.0
    result = result.loc[result["cny_per_usd"] > 0].reset_index(drop=True)
    if result.empty:
        raise AcquisitionContractError("中行美元历史牌价没有非缺失正值")
    result["source"] = "AkShare currency_boc_sina，新浪财经中行牌价历史表"
    return result


def fetch_boc_usd_cny(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """按年获取美元中行折算价，避免单次页面过大。"""

    import importlib

    module = importlib.import_module(ak.currency_boc_sina.__module__)
    mapping = module._currency_boc_sina_map(  # noqa: SLF001
        start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    )
    usd_symbol = next((key for key, value in mapping.items() if value == "USD"), None)
    if usd_symbol is None:
        raise AcquisitionContractError("AkShare中行牌价映射中没有USD")
    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                frames.append(
                    ak.currency_boc_sina(
                        symbol=usd_symbol,
                        start_date=year_start.strftime("%Y%m%d"),
                        end_date=year_end.strftime("%Y%m%d"),
                    )
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(float(attempt * 2))
        if last_error is not None:
            raise RuntimeError(
                f"中行美元牌价{year}年下载失败：{type(last_error).__name__}: {last_error}"
            )
    if not frames:
        raise AcquisitionContractError("中行美元牌价下载结果为空")
    return normalize_boc_usd_cny(pd.concat(frames, ignore_index=True))


def normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    """规范本地H00300全收益快照。"""

    if not {"date", "close"}.issubset(frame.columns):
        raise AcquisitionContractError("H00300快照缺少date或close")
    result = frame.copy()
    result["date"] = (
        pd.to_datetime(result["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"])
    result = result.loc[result["close"] > 0].drop_duplicates("date", keep="last")
    result = result.sort_values("date").reset_index(drop=True)
    if result.empty:
        raise AcquisitionContractError("H00300快照没有有效记录")
    result["symbol"] = "H00300"
    result["name"] = "沪深300全收益指数"
    result["source"] = "项目既有H00300全收益不可变快照"
    return result[["date", "symbol", "name", "close", "source"]]


def partition_inputs(
    panel: pd.DataFrame,
    fx: pd.DataFrame,
    benchmark: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """按冻结日期物理切分可见期与封存复验期。"""

    periods = contract["historical_partition"]
    visible_end = pd.Timestamp(periods["visible_end"])
    replication_start = pd.Timestamp(periods["formula_family_replication_start"])
    replication_end = pd.Timestamp(periods["formula_family_replication_end"])
    visible_panel = panel.loc[panel["date"].le(visible_end)].copy()
    sealed_panel = panel.loc[panel["date"].between(replication_start, replication_end)].copy()
    visible_fx = fx.loc[fx["date"].le(visible_end)].copy()
    sealed_fx = fx.loc[fx["date"].between(replication_start, replication_end)].copy()
    visible_benchmark = benchmark.loc[benchmark["date"].le(visible_end)].copy()
    sealed_benchmark = benchmark.loc[
        benchmark["date"].between(replication_start, replication_end)
    ].copy()
    if visible_panel.empty or sealed_panel.empty:
        raise AcquisitionContractError("价格面板的可见期或封存期为空")
    if visible_fx.empty or sealed_fx.empty:
        raise AcquisitionContractError("汇率的可见期或封存期为空")
    if visible_benchmark.empty or sealed_benchmark.empty:
        raise AcquisitionContractError("基准的可见期或封存期为空")
    if visible_panel["date"].max() >= sealed_panel["date"].min():
        raise AcquisitionContractError("价格可见期与封存期日期重叠")
    return {
        "visible_panel": visible_panel,
        "sealed_panel": sealed_panel,
        "visible_fx": visible_fx,
        "sealed_fx": sealed_fx,
        "visible_benchmark": visible_benchmark,
        "sealed_benchmark": sealed_benchmark,
    }


def fetch_security_with_retry(
    ticker: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    attempts: int = 3,
) -> pd.DataFrame:
    """下载单只产品，瞬时网络失败时有限重试。"""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = ak.stock_us_daily(symbol=ticker, adjust="")
            factors = ak.stock_us_daily(symbol=ticker, adjust="qfq-factor")
            return normalize_security_history(ticker, raw, factors, start=start, end=end)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt * 2))
    raise RuntimeError(f"{ticker}下载失败：{type(last_error).__name__}: {last_error}")


def build_product_master(contract: dict[str, Any]) -> pd.DataFrame:
    """把固定产品身份写成独立输入表。"""

    rows: list[dict[str, Any]] = []
    for ticker in contract["universe"]["fixed_tickers"]:
        item = contract["universe"]["products"][ticker]
        rows.append(
            {
                "ticker": ticker,
                "risk_source": item["risk_source"],
                "stated_daily_multiple": float(item["stated_daily_multiple"]),
                "issuer": item["issuer"],
                "inception_date": pd.Timestamp(item["inception_date"]),
                "fixed_before_history_acquisition": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    contract = load_contract()
    periods = contract["historical_partition"]
    start = pd.Timestamp(periods["acquisition_start"])
    end = pd.Timestamp(periods["formula_family_replication_end"])
    input_paths = contract["inputs"]
    status_path = ROOT / input_paths["source_status"]

    try:
        product_master = build_product_master(contract)
        parts: list[pd.DataFrame] = []
        for ticker in contract["universe"]["fixed_tickers"]:
            print(f"正在采集 {ticker} 未复权日线与公司行动因子", flush=True)
            parts.append(fetch_security_with_retry(ticker, start=start, end=end))
        panel = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])
        if panel[["ticker", "date"]].duplicated().any():
            raise AcquisitionContractError("合并价格面板存在重复主键")
        if set(panel["ticker"].unique()) != set(contract["universe"]["fixed_tickers"]):
            raise AcquisitionContractError("合并价格面板未覆盖全部固定产品")

        print("正在采集新浪财经中行美元历史折算价", flush=True)
        fx = fetch_boc_usd_cny(start, end)
        benchmark_source = ROOT / contract["data_sources"]["benchmark_source_snapshot"]
        benchmark = normalize_benchmark(pd.read_parquet(benchmark_source))
        benchmark = benchmark.loc[benchmark["date"].le(end)].copy()
        partitions = partition_inputs(panel, fx, benchmark, contract)

        paths = {
            "product_master": ROOT / input_paths["product_master"],
            "daily_panel": ROOT / input_paths["daily_panel"],
            "visible_panel": ROOT / input_paths["visible_panel"],
            "sealed_replication_panel": ROOT / input_paths["sealed_replication_panel"],
            "fx_full": ROOT / input_paths["fx_full"],
            "visible_fx": ROOT / input_paths["visible_fx"],
            "sealed_replication_fx": ROOT / input_paths["sealed_replication_fx"],
            "visible_benchmark": ROOT / input_paths["visible_benchmark"],
            "sealed_replication_benchmark": ROOT / input_paths["sealed_replication_benchmark"],
        }
        frames = {
            "product_master": product_master,
            "daily_panel": panel,
            "visible_panel": partitions["visible_panel"],
            "sealed_replication_panel": partitions["sealed_panel"],
            "fx_full": fx,
            "visible_fx": partitions["visible_fx"],
            "sealed_replication_fx": partitions["sealed_fx"],
            "visible_benchmark": partitions["visible_benchmark"],
            "sealed_replication_benchmark": partitions["sealed_benchmark"],
        }
        for key, path in paths.items():
            atomic_parquet(path, frames[key])

        visible = partitions["visible_panel"]
        sealed = partitions["sealed_panel"]
        status: dict[str, Any] = {
            "schema_version": "1.0.0",
            "status": "PASS_INPUT_ACQUISITION_ONLY",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": contract["protocol"]["candidate_id"],
            "strategy_total_return_or_rank_computed": False,
            "portfolio_nav_or_target_gate_computed": False,
            "security_total_return_index_computed": True,
            "fixed_tickers": contract["universe"]["fixed_tickers"],
            "panel_rows": int(len(panel)),
            "visible_panel_rows": int(len(visible)),
            "sealed_panel_rows": int(len(sealed)),
            "rows_by_ticker": {
                str(key): int(value) for key, value in panel.groupby("ticker").size().items()
            },
            "visible_date_range": [
                visible["date"].min().date().isoformat(),
                visible["date"].max().date().isoformat(),
            ],
            "sealed_date_range": [
                sealed["date"].min().date().isoformat(),
                sealed["date"].max().date().isoformat(),
            ],
            "split_event_rows": int(panel["split_ratio_at_open"].ne(1.0).sum()),
            "cash_distribution_event_rows": int(
                panel["cash_distribution_per_post_event_share_usd"].gt(0).sum()
            ),
            "sources": {
                "us_daily_raw": "akshare.stock_us_daily(adjust='')，底层新浪历史日线",
                "us_qfq_factor": "akshare.stock_us_daily(adjust='qfq-factor')，底层新浪复权因子",
                "fx": "AkShare currency_boc_sina，新浪财经中行美元历史折算价",
                "benchmark": benchmark_source.relative_to(ROOT).as_posix(),
            },
            "hashes": {key: sha256_file(path) for key, path in paths.items()},
        }
        atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_INPUT_ACQUISITION",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": contract["protocol"]["candidate_id"],
            "strategy_total_return_or_rank_computed": False,
            "portfolio_nav_or_target_gate_computed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json(status_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
