"""校验510300分红事件：官方来源快照、累计分红差分交叉核对和日期逻辑。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
SOURCE_DIR = PROJECT_ROOT / "data" / "reference" / "510300_dividend_sources"
SINA_FILE = PROJECT_ROOT / "data" / "raw" / "fund" / "510300_dividend_cumulative_sina.parquet"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_dividends_quality.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def main() -> int:
    checked_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    events = pd.read_csv(EVENT_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    errors: list[str] = []
    if events["ex_date"].duplicated().any():
        errors.append("除息日重复")
    if not (events["record_date"] < events["ex_date"]).all():
        errors.append("登记日必须早于除息日")
    if not (events["ex_date"] <= events["payment_date"]).all():
        errors.append("发放日不得早于除息日")
    if (events["cash_dividend_per_share"] <= 0).any():
        errors.append("现金分红存在非正值")

    source_records: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for index, row in events.iterrows():
        response = session.get(row["source"], timeout=45)
        response.raise_for_status()
        extension = Path(urlparse(row["source"]).path).suffix.lower()
        if extension not in {".pdf", ".html", ".shtml"}:
            extension = ".bin"
        target = SOURCE_DIR / f"{row['ex_date']:%Y-%m-%d}_{index + 1}{extension}"
        atomic_bytes(response.content, target)
        pdf_signature_valid = None
        if extension == ".pdf":
            pdf_signature_valid = response.content.startswith(b"%PDF")
            if not pdf_signature_valid:
                errors.append(f"{row['ex_date'].date()}的PDF签名无效")
        source_records.append(
            {
                "ex_date": row["ex_date"].date().isoformat(),
                "url": row["source"],
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "bytes": len(response.content),
                "saved_file": target.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(target),
                "pdf_signature_valid": pdf_signature_valid,
            }
        )

    cumulative = ak.fund_etf_dividend_sina(symbol="sh510300").copy()
    cumulative.columns = ["ex_date", "cumulative_dividend_per_share"]
    cumulative["ex_date"] = pd.to_datetime(cumulative["ex_date"], errors="raise")
    cumulative["cumulative_dividend_per_share"] = pd.to_numeric(
        cumulative["cumulative_dividend_per_share"], errors="raise"
    )
    cumulative = cumulative.sort_values("ex_date").reset_index(drop=True)
    cumulative["cash_dividend_implied"] = cumulative["cumulative_dividend_per_share"].diff()
    cumulative["source"] = "akshare.fund_etf_dividend_sina"
    cumulative["retrieved_at"] = checked_at
    atomic_parquet(cumulative, SINA_FILE)

    comparison = events.merge(
        cumulative[["ex_date", "cash_dividend_implied", "cumulative_dividend_per_share"]],
        on="ex_date",
        how="left",
        validate="one_to_one",
    )
    comparison["absolute_difference"] = (
        comparison["cash_dividend_per_share"] - comparison["cash_dividend_implied"]
    ).abs()
    if comparison["cash_dividend_implied"].isna().any():
        errors.append("新浪累计分红序列缺少已知除息日")
    if (comparison["absolute_difference"] > 1e-9).any():
        errors.append("官方事件金额与新浪累计分红差分不一致")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "checked_at": checked_at,
        "event_count_in_research_window": int(len(events)),
        "first_ex_date": events["ex_date"].min().date().isoformat(),
        "last_ex_date": events["ex_date"].max().date().isoformat(),
        "official_source_snapshots": source_records,
        "sina_cumulative_series": {
            "row_count": int(len(cumulative)),
            "first_date": cumulative["ex_date"].min().date().isoformat(),
            "last_date": cumulative["ex_date"].max().date().isoformat(),
            "file": SINA_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(SINA_FILE),
        },
        "cross_check": comparison[
            [
                "ex_date",
                "cash_dividend_per_share",
                "cash_dividend_implied",
                "absolute_difference",
            ]
        ].assign(ex_date=lambda frame: frame["ex_date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        "errors": errors,
        "accounting_note": "登记日锁定分红权；除息日转入应收并纳入财富；发放日仅将应收转为现金。",
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
