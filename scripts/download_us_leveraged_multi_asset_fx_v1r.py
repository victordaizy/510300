"""下载V1R所需的中国外汇交易中心官方USD/CNY中间价。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "us_leveraged_multi_asset_absolute_momentum_v1r_fx_source_correction.yaml"
PAGE_SIZE = 50


class FxCorrectionDataError(ValueError):
    """官方中间价输入不满足V1R修正合同。"""


def load_override(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FxCorrectionDataError("V1R修正配置必须是YAML对象")
    correction = payload.get("correction", {})
    if correction.get("source_to") != "CHINAMONEY_CFETS_OFFICIAL_USD_CNY_CENTRAL_PARITY":
        raise FxCorrectionDataError("V1R官方汇率来源发生变化")
    if int(correction.get("maximum_fx_staleness_calendar_days_to", 0)) != 14:
        raise FxCorrectionDataError("V1R最长汇率陈旧期必须为14天")
    if correction.get("strategy_return_or_rank_may_be_computed_during_acquisition") is not False:
        raise FxCorrectionDataError("V1R采集阶段不得计算策略收益或排名")
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


def request_json(endpoint: str, parameters: dict[str, Any], *, attempts: int = 4) -> dict[str, Any]:
    """以空POST获取中汇网官方JSON。"""

    url = endpoint + "?" + urlencode(parameters, safe="/,")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://www.chinamoney.com.cn",
        "Referer": "https://www.chinamoney.com.cn/chinese/bkccpr/",
        "X-Requested-With": "XMLHttpRequest",
    }
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = Request(url, data=b"", headers=headers, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                body = response.read()
                content_type = str(response.headers.get("Content-Type", ""))
            if "json" not in content_type.lower() or body.lstrip().startswith(b"<"):
                raise FxCorrectionDataError("中国外汇交易中心返回非JSON内容")
            payload = json.loads(body.decode("utf-8"))
            if str(payload.get("head", {}).get("rep_code")) != "200":
                raise FxCorrectionDataError(f"中国外汇交易中心业务响应失败：{payload.get('head')}")
            return payload
        except (HTTPError, URLError, TimeoutError, FxCorrectionDataError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"中国外汇交易中心请求连续失败：{last_error}")


def parse_usd_cny_pages(pages: list[dict[str, Any]]) -> pd.DataFrame:
    """解析单币种分页响应并拒绝重复、缺失或非正中间价。"""

    rows: list[dict[str, Any]] = []
    for page in pages:
        if str(page.get("head", {}).get("rep_code")) != "200":
            raise FxCorrectionDataError("官方中间价分页响应失败")
        currency = str(page.get("data", {}).get("currency", ""))
        if currency != "USD/CNY":
            raise FxCorrectionDataError(f"官方中间价币种漂移：{currency}")
        for record in page.get("records", []):
            values = record.get("values", [])
            if len(values) != 1:
                raise FxCorrectionDataError(f"USD/CNY记录字段数异常：{record}")
            rows.append({"date": record.get("date"), "cny_per_usd": values[0]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FxCorrectionDataError("官方USD/CNY中间价为空")
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    frame["cny_per_usd"] = pd.to_numeric(frame["cny_per_usd"], errors="coerce")
    if frame[["date", "cny_per_usd"]].isna().any().any():
        raise FxCorrectionDataError("官方USD/CNY中间价包含无法解析值")
    if frame["date"].duplicated().any():
        raise FxCorrectionDataError("官方USD/CNY中间价日期重复")
    if frame["cny_per_usd"].le(0.0).any():
        raise FxCorrectionDataError("官方USD/CNY中间价包含非正值")
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["source"] = "中国外汇交易中心人民币汇率中间价历史查询"
    return frame


def fetch_or_load_receipt(override: dict[str, Any]) -> dict[str, Any]:
    """获取2008—2026单币种完整分页，并缓存原始收据。"""

    correction_data = override["correction_data"]
    receipt_path = ROOT / correction_data["raw_receipt"]
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        pages = receipt.get("pages", [])
        if pages:
            parsed = parse_usd_cny_pages(pages)
            if int(receipt.get("complete_record_count", -1)) == len(parsed):
                return receipt
    endpoint = correction_data["source_endpoint"]
    pages: list[dict[str, Any]] = []
    for year in range(2008, 2027):
        end_date = "2026-08-14" if year == 2026 else f"{year}-12-31"
        base = {
            "startDate": f"{year}-01-01",
            "endDate": end_date,
            "currency": "USD/CNY",
            "pageSize": PAGE_SIZE,
        }
        first = request_json(endpoint, {**base, "pageNum": 1})
        page_total = int(first.get("data", {}).get("pageTotal", 0))
        total = int(first.get("data", {}).get("total", 0))
        if page_total < 1 or total < 1:
            raise FxCorrectionDataError(f"{year}年官方USD/CNY中间价为空")
        year_pages = [first]
        for page_number in range(2, page_total + 1):
            year_pages.append(request_json(endpoint, {**base, "pageNum": page_number}))
            time.sleep(0.05)
        raw_count = sum(len(page.get("records", [])) for page in year_pages)
        if raw_count != total:
            raise FxCorrectionDataError(
                f"{year}年官方USD/CNY分页不完整：{raw_count}/{total}"
            )
        pages.extend(year_pages)
        print(f"官方USD/CNY中间价已获取至 {year} 年，共 {raw_count} 条", flush=True)
    parsed = parse_usd_cny_pages(pages)
    receipt = {
        "schema_version": "1.0.0",
        "source": endpoint,
        "source_page": correction_data["source_page"],
        "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "currency": "USD/CNY",
        "first_date": parsed["date"].min().date().isoformat(),
        "last_date": parsed["date"].max().date().isoformat(),
        "complete_record_count": int(len(parsed)),
        "pages": pages,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def strict_lag_coverage(
    visible_panel: pd.DataFrame,
    visible_fx: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """只验证共同美国交易日的严格滞后汇率日期覆盖。"""

    panel = visible_panel[["ticker", "date"]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    counts = panel.groupby("date", observed=True)["ticker"].nunique()
    dates = pd.DataFrame({"date": counts.loc[counts.eq(4)].index})
    dates = dates.loc[dates["date"].between(start, end)].sort_values("date")
    right = visible_fx[["date", "cny_per_usd"]].copy()
    right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
    right = right.sort_values("date")
    mapped = pd.merge_asof(
        dates,
        right.rename(columns={"date": "fx_source_date"}),
        left_on="date",
        right_on="fx_source_date",
        direction="backward",
        allow_exact_matches=False,
    )
    mapped["age"] = (mapped["date"] - mapped["fx_source_date"]).dt.days
    if mapped[["fx_source_date", "cny_per_usd", "age"]].isna().any().any():
        raise FxCorrectionDataError("可见期存在无法映射的严格滞后官方中间价")
    worst = mapped.loc[mapped["age"].idxmax()]
    return {
        "evaluation_us_trading_days": int(len(mapped)),
        "maximum_fx_age_calendar_days": int(mapped["age"].max()),
        "worst_us_date": worst["date"].date().isoformat(),
        "worst_fx_source_date": worst["fx_source_date"].date().isoformat(),
        "strictly_lagged": bool((mapped["fx_source_date"] < mapped["date"]).all()),
    }


def main() -> int:
    override = load_override()
    base = yaml.safe_load((ROOT / override["base_candidate_config"]).read_text(encoding="utf-8"))
    inputs = base["inputs"]
    corrected_inputs = override["overrides"]["inputs"]
    status_path = ROOT / corrected_inputs["source_status"]
    try:
        receipt = fetch_or_load_receipt(override)
        fx = parse_usd_cny_pages(receipt["pages"])
        start = pd.Timestamp(base["historical_partition"]["acquisition_start"])
        end = pd.Timestamp(base["historical_partition"]["formula_family_replication_end"])
        fx = fx.loc[fx["date"].between(start, end)].copy()
        visible_end = pd.Timestamp(base["historical_partition"]["visible_end"])
        sealed_start = pd.Timestamp(
            base["historical_partition"]["formula_family_replication_start"]
        )
        visible_fx = fx.loc[fx["date"].le(visible_end)].copy()
        sealed_fx = fx.loc[fx["date"].between(sealed_start, end)].copy()
        paths = {
            "fx_full": ROOT / corrected_inputs["fx_full"],
            "visible_fx": ROOT / corrected_inputs["visible_fx"],
            "sealed_replication_fx": ROOT / corrected_inputs["sealed_replication_fx"],
        }
        atomic_parquet(paths["fx_full"], fx)
        atomic_parquet(paths["visible_fx"], visible_fx)
        atomic_parquet(paths["sealed_replication_fx"], sealed_fx)
        visible_panel_path = ROOT / inputs["visible_panel"]
        coverage = strict_lag_coverage(
            pd.read_parquet(visible_panel_path, columns=["ticker", "date"]),
            visible_fx,
            start=pd.Timestamp(base["historical_partition"]["visible_start"]),
            end=visible_end,
        )
        maximum_allowed = int(
            override["overrides"]["currency"]["maximum_fx_staleness_calendar_days"]
        )
        if coverage["maximum_fx_age_calendar_days"] > maximum_allowed:
            raise FxCorrectionDataError(
                f"官方中间价最长陈旧{coverage['maximum_fx_age_calendar_days']}天，超过{maximum_allowed}天"
            )
        inherited_paths = {
            "product_master": ROOT / inputs["product_master"],
            "visible_panel": visible_panel_path,
            "sealed_replication_panel": ROOT / inputs["sealed_replication_panel"],
            "visible_benchmark": ROOT / inputs["visible_benchmark"],
            "sealed_replication_benchmark": ROOT / inputs["sealed_replication_benchmark"],
        }
        receipt_path = ROOT / override["correction_data"]["raw_receipt"]
        status = {
            "schema_version": "1.0.0",
            "status": "PASS_FX_SOURCE_CORRECTION_INPUT_ACQUISITION_ONLY",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": override["protocol"]["candidate_id"],
            "strategy_total_return_or_rank_computed": False,
            "portfolio_nav_or_target_gate_computed": False,
            "security_price_inputs_reused_without_change": True,
            "official_fx_rows": int(len(fx)),
            "visible_fx_rows": int(len(visible_fx)),
            "sealed_fx_rows": int(len(sealed_fx)),
            "visible_strict_lag_coverage": coverage,
            "maximum_allowed_fx_staleness_calendar_days": maximum_allowed,
            "source": override["correction_data"]["source_endpoint"],
            "hashes": {
                "raw_receipt": sha256_file(receipt_path),
                **{key: sha256_file(path) for key, path in paths.items()},
                **{
                    f"inherited_{key}": sha256_file(path)
                    for key, path in inherited_paths.items()
                },
            },
        }
        atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_FX_SOURCE_CORRECTION_INPUT_ACQUISITION",
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_id": override["protocol"]["candidate_id"],
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
