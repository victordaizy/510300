"""构建并核验510300自上市以来的完整现金分红事件表。"""

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


ROOT = Path(__file__).resolve().parents[1]
EVENT_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
ATTESTATION_FILE = ROOT / "data" / "reference" / "510300_dividends_coverage.json"
SOURCE_DIR = ROOT / "data" / "reference" / "510300_dividend_sources" / "full_history"
TIMEZONE = ZoneInfo("Asia/Shanghai")


EVENTS = (
    ("2012-12-17", "2012-12-18", "2012-12-24", 0.033, "https://www.sse.com.cn/disclosure/fund/announcement/c/2012-12-12/510300_20121212_1.pdf"),
    ("2014-01-20", "2014-01-21", "2014-01-27", 0.048, "https://www.sse.com.cn/disclosure/fund/announcement/c/2014-01-14/510300_20140115_1.pdf"),
    ("2015-01-19", "2015-01-20", "2015-01-23", 0.035, "https://www.sse.com.cn/disclosure/fund/announcement/c/2015-01-13/510300_20150114_1.pdf"),
    ("2016-01-19", "2016-01-20", "2016-01-25", 0.051, "https://www.sse.com.cn/disclosure/fund/announcement/c/2016-01-14/510300_20160114_1.pdf"),
    ("2017-01-20", "2017-01-23", "2017-01-26", 0.055, "https://www.sse.com.cn/disclosure/fund/announcement/c/2017-01-16/510300_20170117_1.pdf"),
    ("2018-01-22", "2018-01-23", "2018-01-26", 0.046, "https://www.sse.com.cn/disclosure/fund/announcement/c/2018-01-17/510300_20180117_1.pdf"),
    ("2019-01-15", "2019-01-16", "2019-01-21", 0.059, "https://www.sse.com.cn/disclosure/fund/announcement/c/2019-01-10/510300_20190110_1.pdf"),
    ("2019-12-10", "2019-12-11", "2019-12-16", 0.062, "https://www.sse.com.cn/disclosure/fund/announcement/c/2019-12-05/510300_20191205_1.pdf"),
    ("2021-01-15", "2021-01-18", "2021-01-21", 0.072, "https://www.sse.com.cn/disclosure/fund/announcement/c/2021-01-11/510300_20210111_1.pdf"),
    ("2022-01-18", "2022-01-19", "2022-01-24", 0.075, "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-01-12/510300_20220112_1_pWPNE0aG.pdf"),
    ("2023-01-13", "2023-01-16", "2023-01-19", 0.064, "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-01-09/510300_20230109_0ED5.pdf"),
    ("2024-01-17", "2024-01-18", "2024-01-23", 0.069, "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2024-01-11/510300_20240111_QQFV.pdf"),
    ("2025-06-17", "2025-06-18", "2025-06-27", 0.088, "https://www.sse.com.cn/assortment/options/disclo/update/c/c_20250611_10781544.shtml"),
    ("2026-01-16", "2026-01-19", "2026-01-27", 0.123, "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-01-12/510300_20260112_VTCZ.pdf"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    retrieved_at = datetime.now(TIMEZONE).isoformat()
    frame = pd.DataFrame(
        EVENTS,
        columns=[
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
            "source",
        ],
    )
    frame.insert(0, "symbol", "510300.SH")
    for column in ("record_date", "ex_date", "payment_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame["ex_date"].duplicated().any():
        raise ValueError("完整分红表出现重复除息日")
    if not (frame["record_date"] < frame["ex_date"]).all():
        raise ValueError("权益登记日必须早于除息日")
    if not (frame["ex_date"] <= frame["payment_date"]).all():
        raise ValueError("支付日不得早于除息日")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    snapshots: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        response = session.get(row.source, timeout=45)
        response.raise_for_status()
        extension = Path(urlparse(row.source).path).suffix.lower()
        if extension not in {".pdf", ".shtml", ".html"}:
            extension = ".bin"
        target = SOURCE_DIR / f"{row.ex_date.date().isoformat()}{extension}"
        _atomic_bytes(response.content, target)
        if extension == ".pdf" and not response.content.startswith(b"%PDF"):
            raise ValueError(f"{row.ex_date.date()}官方PDF签名无效")
        snapshots.append(
            {
                "ex_date": row.ex_date.date().isoformat(),
                "url": row.source,
                "saved_file": target.relative_to(ROOT).as_posix(),
                "bytes": len(response.content),
                "sha256": sha256(target),
            }
        )

    cumulative = ak.fund_etf_dividend_sina(symbol="sh510300").copy()
    cumulative.columns = ["ex_date", "cumulative_dividend_per_share"]
    cumulative["ex_date"] = pd.to_datetime(cumulative["ex_date"], errors="raise")
    cumulative["cumulative_dividend_per_share"] = pd.to_numeric(
        cumulative["cumulative_dividend_per_share"], errors="raise"
    )
    cumulative = cumulative.sort_values("ex_date").reset_index(drop=True)
    cumulative["cash_dividend_implied"] = cumulative[
        "cumulative_dividend_per_share"
    ].diff()
    cumulative.loc[0, "cash_dividend_implied"] = cumulative.loc[
        0, "cumulative_dividend_per_share"
    ]
    comparison = frame.merge(
        cumulative[["ex_date", "cash_dividend_implied"]],
        on="ex_date",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    comparison["absolute_difference"] = (
        comparison["cash_dividend_per_share"]
        - comparison["cash_dividend_implied"]
    ).abs()
    if not comparison["_merge"].eq("both").all():
        raise ValueError("官方事件与新浪累计分红日期集合不一致")
    if not comparison["absolute_difference"].le(1e-9).all():
        raise ValueError("官方事件金额与新浪累计分红差分不一致")

    csv_text = frame.assign(
        record_date=lambda data: data["record_date"].dt.strftime("%Y-%m-%d"),
        ex_date=lambda data: data["ex_date"].dt.strftime("%Y-%m-%d"),
        payment_date=lambda data: data["payment_date"].dt.strftime("%Y-%m-%d"),
    ).to_csv(index=False, lineterminator="\n")
    _atomic_text(csv_text, EVENT_FILE)
    attestation = {
        "complete_history_confirmed": True,
        "coverage_start": "2012-05-28",
        "coverage_end": "2026-08-14",
        "event_count": int(len(frame)),
        "first_ex_date": frame["ex_date"].min().date().isoformat(),
        "last_ex_date": frame["ex_date"].max().date().isoformat(),
        "distribution_file": EVENT_FILE.relative_to(ROOT).as_posix(),
        "distribution_file_sha256": sha256(EVENT_FILE),
        "retrieved_at": retrieved_at,
        "official_sources": frame["source"].tolist(),
        "official_source_snapshots": snapshots,
        "secondary_cross_check": {
            "source": "akshare.fund_etf_dividend_sina(sh510300)",
            "event_count": int(len(cumulative)),
            "date_set_matches": True,
            "maximum_absolute_amount_difference": float(
                comparison["absolute_difference"].max()
            ),
        },
    }
    _atomic_text(
        json.dumps(attestation, ensure_ascii=False, indent=2), ATTESTATION_FILE
    )
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

