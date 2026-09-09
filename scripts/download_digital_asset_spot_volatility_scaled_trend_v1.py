"""下载数字资产现货波动缩放趋势V1的Coinbase日线并分区。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "digital_asset_spot_volatility_scaled_trend_v1.yaml"


class DigitalAssetDataError(ValueError):
    """Coinbase或外生输入不满足冻结合同。"""


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DigitalAssetDataError("数字资产候选配置必须是YAML对象")
    if payload["universe"]["fixed_products"] != ["BTC-USD", "ETH-USD"]:
        raise DigitalAssetDataError("固定数字资产产品发生变化")
    if payload["information_timing"]["candle_granularity_seconds"] != 86400:
        raise DigitalAssetDataError("Coinbase日线粒度必须为86400秒")
    if payload["data_sources"]["strategy_return_or_rank_may_be_computed_during_acquisition"]:
        raise DigitalAssetDataError("采集阶段不得计算策略收益或排名")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def normalize_candles(
    product_id: str,
    rows: list[list[Any]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """规范Coinbase `[time, low, high, open, close, volume]` 日线。"""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            raise DigitalAssetDataError(f"{product_id}蜡烛结构异常：{row}")
        timestamp, low, high, open_price, close, volume = row
        date = pd.Timestamp(datetime.fromtimestamp(int(timestamp), tz=timezone.utc)).tz_localize(None)
        normalized.append(
            {
                "product_id": product_id,
                "date": date.normalize(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    frame = pd.DataFrame(normalized)
    if frame.empty:
        raise DigitalAssetDataError(f"{product_id}日线为空")
    frame = frame.loc[frame["date"].between(start, end)].copy()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "open", "high", "low", "close", "volume"]].isna().any().any():
        raise DigitalAssetDataError(f"{product_id}日线包含无法解析值")
    frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise DigitalAssetDataError(f"{product_id}存在非正价格")
    if (frame["volume"] <= 0.0).any():
        raise DigitalAssetDataError(f"{product_id}存在非正成交量或无成交日")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DigitalAssetDataError(f"{product_id}最高价逻辑错误")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DigitalAssetDataError(f"{product_id}最低价逻辑错误")
    expected = pd.date_range(start, end, freq="D")
    missing = expected.difference(pd.DatetimeIndex(frame["date"]))
    if len(missing):
        sample = [date.date().isoformat() for date in missing[:10]]
        raise DigitalAssetDataError(
            f"{product_id}缺少{len(missing)}个UTC日线桶，样例={sample}"
        )
    frame["dollar_turnover_usd"] = frame["close"] * frame["volume"]
    frame["source"] = "Coinbase Exchange公开Get product candles接口"
    return frame


def fetch_product_candles(
    product_id: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    endpoint_template: str,
) -> pd.DataFrame:
    """按不超过250日的窗口下载，避免300根上限。"""

    endpoint = endpoint_template.format(product_id=product_id)
    cursor = start
    rows: list[list[Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "evidence-first-quant-research/1.0"})
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=249), end)
        parameters = {
            "granularity": 86400,
            "start": cursor.strftime("%Y-%m-%dT00:00:00Z"),
            "end": (chunk_end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
        }
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = session.get(endpoint, params=parameters, timeout=45)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise DigitalAssetDataError(
                        f"{product_id}接口返回非数组：{str(payload)[:200]}"
                    )
                rows.extend(payload)
                last_error = None
                break
            except (requests.RequestException, ValueError, DigitalAssetDataError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(float(attempt * 2))
        if last_error is not None:
            raise RuntimeError(
                f"{product_id} {cursor.date()}至{chunk_end.date()}下载失败：{last_error}"
            )
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.12)
    return normalize_candles(product_id, rows, start=start, end=end)


def normalize_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "close"}.issubset(frame.columns):
        raise DigitalAssetDataError("H00300快照缺少date或close")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").astype("datetime64[ns]")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).loc[lambda x: x["close"].gt(0.0)]
    result = result.drop_duplicates("date", keep="last").sort_values("date")
    result["symbol"] = "H00300"
    result["name"] = "沪深300全收益指数"
    result["source"] = "项目既有H00300全收益快照"
    return result[["date", "symbol", "name", "close", "source"]].reset_index(drop=True)


def normalize_fx(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"date", "cny_per_usd"}.issubset(frame.columns):
        raise DigitalAssetDataError("官方USD/CNY中间价缺少字段")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").astype("datetime64[ns]")
    result["cny_per_usd"] = pd.to_numeric(result["cny_per_usd"], errors="coerce")
    result = result.dropna(subset=["date", "cny_per_usd"])
    result = result.loc[result["cny_per_usd"].gt(0.0)]
    return result.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def main() -> int:
    contract = load_contract()
    periods = contract["historical_partition"]
    start = pd.Timestamp(periods["acquisition_start"])
    end = pd.Timestamp(periods["formula_family_replication_end"])
    inputs = contract["inputs"]
    status_path = ROOT / inputs["source_status"]
    try:
        parts: list[pd.DataFrame] = []
        for product_id in contract["universe"]["fixed_products"]:
            print(f"正在下载 {product_id} Coinbase UTC日线", flush=True)
            parts.append(
                fetch_product_candles(
                    product_id,
                    start=start,
                    end=end,
                    endpoint_template=contract["data_sources"]["api_endpoint_template"],
                )
            )
        panel = pd.concat(parts, ignore_index=True).sort_values(["date", "product_id"])
        if panel[["product_id", "date"]].duplicated().any():
            raise DigitalAssetDataError("合并数字资产面板存在重复主键")
        expected_rows = len(pd.date_range(start, end, freq="D")) * 2
        if len(panel) != expected_rows:
            raise DigitalAssetDataError(f"数字资产面板行数不是完整双产品日历：{len(panel)}/{expected_rows}")

        visible_end = pd.Timestamp(periods["visible_end"])
        sealed_start = pd.Timestamp(periods["formula_family_replication_start"])
        visible_panel = panel.loc[panel["date"].le(visible_end)].copy()
        sealed_panel = panel.loc[panel["date"].between(sealed_start, end)].copy()
        if visible_panel["date"].max() >= sealed_panel["date"].min():
            raise DigitalAssetDataError("数字资产可见期与封存期重叠")

        fx = normalize_fx(pd.read_parquet(ROOT / contract["data_sources"]["fx_full"]))
        visible_fx = fx.loc[fx["date"].le(visible_end)].copy()
        sealed_fx = fx.loc[fx["date"].between(sealed_start, end)].copy()
        benchmark = normalize_benchmark(
            pd.read_parquet(ROOT / contract["data_sources"]["benchmark_source_snapshot"])
        )
        visible_benchmark = benchmark.loc[benchmark["date"].le(visible_end)].copy()
        sealed_benchmark = benchmark.loc[
            benchmark["date"].between(sealed_start, end)
        ].copy()
        if visible_fx.empty or sealed_fx.empty or visible_benchmark.empty or sealed_benchmark.empty:
            raise DigitalAssetDataError("汇率或基准的可见/封存分区为空")

        product_master = pd.DataFrame(
            [
                {
                    "product_id": product_id,
                    "base_currency": product_id.split("-")[0],
                    "quote_currency": "USD",
                    "venue": "COINBASE_EXCHANGE",
                    "market_type": "SPOT",
                    "fixed_before_history_acquisition": True,
                }
                for product_id in contract["universe"]["fixed_products"]
            ]
        )
        paths = {
            "product_master": ROOT / inputs["product_master"],
            "visible_panel": ROOT / inputs["visible_panel"],
            "sealed_replication_panel": ROOT / inputs["sealed_replication_panel"],
            "visible_fx": ROOT / inputs["visible_fx"],
            "sealed_replication_fx": ROOT / inputs["sealed_replication_fx"],
            "visible_benchmark": ROOT / inputs["visible_benchmark"],
            "sealed_replication_benchmark": ROOT / inputs["sealed_replication_benchmark"],
        }
        frames = {
            "product_master": product_master,
            "visible_panel": visible_panel,
            "sealed_replication_panel": sealed_panel,
            "visible_fx": visible_fx,
            "sealed_replication_fx": sealed_fx,
            "visible_benchmark": visible_benchmark,
            "sealed_replication_benchmark": sealed_benchmark,
        }
        for key, path in paths.items():
            atomic_parquet(path, frames[key])
        status = {
            "schema_version": "1.0.0",
            "status": "PASS_INPUT_ACQUISITION_ONLY",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": contract["protocol"]["candidate_id"],
            "strategy_total_return_or_rank_computed": False,
            "portfolio_nav_or_target_gate_computed": False,
            "fixed_products": contract["universe"]["fixed_products"],
            "panel_rows": int(len(panel)),
            "visible_panel_rows": int(len(visible_panel)),
            "sealed_panel_rows": int(len(sealed_panel)),
            "visible_date_range": [
                visible_panel["date"].min().date().isoformat(),
                visible_panel["date"].max().date().isoformat(),
            ],
            "sealed_date_range": [
                sealed_panel["date"].min().date().isoformat(),
                sealed_panel["date"].max().date().isoformat(),
            ],
            "complete_utc_daily_calendar": True,
            "source": contract["data_sources"]["api_endpoint_template"],
            "provider_documented_missing_no_tick_bucket_risk": True,
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
