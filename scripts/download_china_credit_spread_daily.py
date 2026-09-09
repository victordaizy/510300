"""下载并审计中债3年AAA中短期票据信用利差日线。"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning


ROOT = Path(__file__).resolve().parents[1]
MARKET_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
EXISTING_GOVERNMENT_FILE = (
    ROOT / "data" / "raw" / "macro" / "china_government_bond_yields_daily.parquet"
)
OUTPUT_FILE = ROOT / "data" / "raw" / "macro" / "china_credit_spread_3y_daily.parquet"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "macro" / "china_credit_spread_3y_checkpoints"
REPORT_FILE = ROOT / "reports" / "data_quality" / "china_credit_spread_3y_status.json"

SOURCE_URL = "https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/historyQuery"
GOVERNMENT_CURVE = "中债国债收益率曲线"
CP_NOTE_AAA_CURVE = "中债中短期票据收益率曲线(AAA)"
REQUIRED_COLUMNS = {"曲线名称", "日期", "1年", "3年", "10年"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_download_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """生成严格少于一年的闭区间，满足上游接口约束。"""

    if end < start:
        raise ValueError("结束日期不能早于开始日期")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.normalize()
    normalized_end = end.normalize()
    while cursor <= normalized_end:
        window_end = min(cursor + pd.Timedelta(days=364), normalized_end)
        windows.append((cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def request_history(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, bool, list[str]]:
    """优先验证TLS；仅在本机代理握手失败时使用可审计的降级路径。"""

    params = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "gjqx": "0",
        "qxId": "ycqx",
        "locale": "cn_ZH",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    errors: list[str] = []
    for verify_tls in (True, False):
        for attempt in range(2):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InsecureRequestWarning)
                    response = requests.get(
                        SOURCE_URL,
                        params=params,
                        headers=headers,
                        timeout=45,
                        verify=verify_tls,
                    )
                response.raise_for_status()
                tables = pd.read_html(StringIO(response.text.replace("&nbsp", "")), header=0)
                if len(tables) < 2:
                    raise ValueError("官方历史页面未返回收益率明细表")
                return tables[1], verify_tls, errors
            except Exception as exception:
                errors.append(
                    f"verify_tls={verify_tls},attempt={attempt + 1},"
                    f"{type(exception).__name__}: {exception}"
                )
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(
        f"中债历史窗口{start.date()}至{end.date()}下载失败：{' | '.join(errors)}"
    )


def normalize_curves(
    raw: pd.DataFrame, retrieved_at: datetime, tls_verified: bool
) -> pd.DataFrame:
    """提取同日同期限的国债与AAA中票曲线，并将百分数利差换算为基点。"""

    if missing := REQUIRED_COLUMNS - set(raw.columns):
        raise ValueError(f"收益率曲线缺少字段：{sorted(missing)}")
    frame = raw.loc[
        raw["曲线名称"].isin([GOVERNMENT_CURVE, CP_NOTE_AAA_CURVE]),
        ["曲线名称", "日期", "1年", "3年", "10年"],
    ].copy()
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
    for column in ("1年", "3年", "10年"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["日期", "3年"])
    if frame.empty:
        raise ValueError("窗口内没有国债与AAA中票的3年曲线")
    if frame.duplicated(["日期", "曲线名称"]).any():
        raise ValueError("同日同曲线存在重复记录")

    government = frame.loc[frame["曲线名称"].eq(GOVERNMENT_CURVE)].set_index("日期")
    credit = frame.loc[frame["曲线名称"].eq(CP_NOTE_AAA_CURVE)].set_index("日期")
    result = pd.DataFrame(index=government.index.intersection(credit.index).sort_values())
    result["cgb_1y"] = government.reindex(result.index)["1年"]
    result["cgb_3y"] = government.reindex(result.index)["3年"]
    result["cgb_10y"] = government.reindex(result.index)["10年"]
    result["cpnote_aaa_3y"] = credit.reindex(result.index)["3年"]
    result = result.dropna(subset=["cgb_3y", "cpnote_aaa_3y"])
    if result.empty:
        raise ValueError("国债与AAA中票没有可配对的同日3年期限")
    result["credit_spread_3y_bp"] = (
        result["cpnote_aaa_3y"] - result["cgb_3y"]
    ) * 100.0
    if (result[["cgb_3y", "cpnote_aaa_3y"]] <= 0).any().any():
        raise ValueError("收益率存在非正值，需人工核查单位")
    if (result["credit_spread_3y_bp"] <= 0).any():
        raise ValueError("AAA中票相对国债出现非正利差，需人工核查曲线映射")
    result = result.reset_index(names="date")
    result["source"] = "chinabond.official.historyQuery"
    result["publication_time_local"] = "17:30"
    result["retrieved_at"] = retrieved_at
    result["transport_tls_verified"] = bool(tls_verified)
    return result


def official_checkpoint_results(data: pd.DataFrame) -> list[dict[str, object]]:
    """核对官方公开表的固定日期数值；这些检查点在完整下载前已记录。"""

    checkpoints = [
        ("2026-07-30", 1.2813, 1.6812),
        ("2026-07-31", 1.2786, 1.6802),
        ("2026-08-18", 1.2493, 1.6556),
    ]
    indexed = data.set_index("date")
    results: list[dict[str, object]] = []
    for date_text, expected_cgb, expected_credit in checkpoints:
        date = pd.Timestamp(date_text)
        if date not in indexed.index:
            results.append({"date": date_text, "passed": False, "reason": "日期缺失"})
            continue
        row = indexed.loc[date]
        actual_cgb = float(row["cgb_3y"])
        actual_credit = float(row["cpnote_aaa_3y"])
        passed = abs(actual_cgb - expected_cgb) <= 1e-8 and abs(actual_credit - expected_credit) <= 1e-8
        results.append(
            {
                "date": date_text,
                "expected_cgb_3y_pct": expected_cgb,
                "actual_cgb_3y_pct": actual_cgb,
                "expected_cpnote_aaa_3y_pct": expected_credit,
                "actual_cpnote_aaa_3y_pct": actual_credit,
                "passed": passed,
            }
        )
    return results


def cross_check_existing_government(data: pd.DataFrame) -> dict[str, object]:
    """与既有独立下载批次的1年/10年国债曲线逐日核对。"""

    existing = pd.read_parquet(EXISTING_GOVERNMENT_FILE)
    existing["date"] = pd.to_datetime(existing["date"], errors="raise")
    merged = data[["date", "cgb_1y", "cgb_10y"]].merge(
        existing[["date", "cgb_1y", "cgb_10y"]],
        on="date",
        how="inner",
        suffixes=("_new", "_existing"),
        validate="one_to_one",
    )
    one_year_error = (merged["cgb_1y_new"] - merged["cgb_1y_existing"]).abs()
    ten_year_error = (merged["cgb_10y_new"] - merged["cgb_10y_existing"]).abs()
    passed = len(merged) > 2000 and one_year_error.max() <= 1e-8 and ten_year_error.max() <= 1e-8
    return {
        "passed": bool(passed),
        "overlap_rows": int(len(merged)),
        "maximum_absolute_cgb_1y_error_pct_point": float(one_year_error.max()),
        "maximum_absolute_cgb_10y_error_pct_point": float(ten_year_error.max()),
        "limitation": "两批下载均来自中债官方接口；这是批次一致性核对，不是独立供应商核对。",
    }


def main() -> int:
    if OUTPUT_FILE.exists() or REPORT_FILE.exists():
        raise FileExistsError("信用利差正式输出已存在，拒绝覆盖")
    market = pd.read_parquet(MARKET_FILE, columns=["date"])
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    start = market["date"].min().normalize()
    end = market["date"].max().normalize()
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    pieces: list[pd.DataFrame] = []
    request_errors: list[str] = []

    for index, (window_start, window_end) in enumerate(
        build_download_windows(start, end), start=1
    ):
        checkpoint = CHECKPOINT_DIR / f"{window_start:%Y%m%d}_{window_end:%Y%m%d}.parquet"
        if checkpoint.exists():
            normalized = pd.read_parquet(checkpoint)
            print(f"复用信用利差检查点 {checkpoint.name}", flush=True)
        else:
            print(
                f"下载中债信用利差 {window_start.date()} 至 {window_end.date()} "
                f"({index}/{len(build_download_windows(start, end))})",
                flush=True,
            )
            raw, tls_verified, errors = request_history(window_start, window_end)
            request_errors.extend(errors)
            normalized = normalize_curves(raw, retrieved_at, tls_verified)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(checkpoint, index=False)
        pieces.append(normalized)
        time.sleep(0.5)

    data = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if data["date"].min() > start + pd.Timedelta(days=7):
        raise ValueError("信用利差首日距510300行情起点超过7天")
    if end - data["date"].max() > pd.Timedelta(days=7):
        raise ValueError("信用利差末日距510300行情终点超过7天")
    checkpoints = official_checkpoint_results(data)
    government_cross_check = cross_check_existing_government(data)
    if not all(bool(item["passed"]) for item in checkpoints):
        raise ValueError("官方固定日期数值核对失败")
    if not government_cross_check["passed"]:
        raise ValueError("既有国债曲线批次一致性核对失败")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS_DISCOVERY_ONLY",
        "checked_at": retrieved_at.isoformat(),
        "source_url": SOURCE_URL,
        "source_compiler": "中央国债登记结算有限责任公司",
        "curve_pair": [GOVERNMENT_CURVE, CP_NOTE_AAA_CURVE],
        "tenor": "3年",
        "source_unit": "百分数",
        "spread_unit": "基点",
        "publication_time_local": "17:30",
        "row_count": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "credit_spread_3y_bp_min": float(data["credit_spread_3y_bp"].min()),
        "credit_spread_3y_bp_max": float(data["credit_spread_3y_bp"].max()),
        "tls_verified_rows": int(data["transport_tls_verified"].sum()),
        "tls_unverified_rows": int((~data["transport_tls_verified"]).sum()),
        "transport_limitation": (
            "本机代理对部分官方请求发生TLS握手EOF；降级请求未验证TLS证书。"
            "已用官方固定日期值及既有下载批次逐日核对内容，但仍不构成独立供应商验证。"
        ),
        "request_error_log": request_errors,
        "official_checkpoint_results": checkpoints,
        "existing_government_batch_cross_check": government_cross_check,
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(OUTPUT_FILE),
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
