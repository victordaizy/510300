"""依据上交所2026年休市公告构造并交叉核验交易日历。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "reference" / "sse_trade_calendar_2026.csv"
METADATA = ROOT / "data" / "reference" / "sse_trade_calendar_2026.metadata.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
OFFICIAL_SOURCE = "https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml"


OFFICIAL_CLOSURES = (
    ("2026-01-01", "2026-01-03", "元旦"),
    ("2026-02-15", "2026-02-23", "春节"),
    ("2026-04-04", "2026-04-06", "清明节"),
    ("2026-05-01", "2026-05-05", "劳动节"),
    ("2026-06-19", "2026-06-21", "端午节"),
    ("2026-09-25", "2026-09-27", "中秋节"),
    ("2026-10-01", "2026-10-07", "国庆节"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def official_calendar() -> pd.DataFrame:
    weekdays = pd.bdate_range("2026-01-01", "2026-12-31")
    closed: set[pd.Timestamp] = set()
    for start, end, _ in OFFICIAL_CLOSURES:
        closed.update(pd.date_range(start, end).normalize())
    trading_days = [date for date in weekdays if date.normalize() not in closed]
    frame = pd.DataFrame({"trade_date": trading_days})
    frame["calendar_year"] = 2026
    frame["week_period"] = frame["trade_date"].dt.to_period("W-FRI").astype(str)
    frame["is_week_last_trade_date"] = frame["trade_date"].eq(
        frame.groupby("week_period")["trade_date"].transform("max")
    )
    frame["source"] = OFFICIAL_SOURCE
    return frame


def main() -> int:
    frame = official_calendar()
    secondary = ak.tool_trade_date_hist_sina().copy()
    secondary["trade_date"] = pd.to_datetime(
        secondary["trade_date"], errors="raise"
    ).dt.normalize()
    secondary_dates = set(
        secondary.loc[
            secondary["trade_date"].dt.year.eq(2026), "trade_date"
        ].tolist()
    )
    official_dates = set(frame["trade_date"].tolist())
    only_official = sorted(official_dates - secondary_dates)
    only_secondary = sorted(secondary_dates - official_dates)
    if only_official or only_secondary:
        raise RuntimeError(
            "上交所休市公告推导日历与新浪交易日历不一致："
            f"only_official={only_official}, only_secondary={only_secondary}"
        )
    csv_text = frame.assign(
        trade_date=lambda data: data["trade_date"].dt.strftime("%Y-%m-%d")
    ).to_csv(index=False, lineterminator="\n")
    atomic_text(OUTPUT, csv_text)
    metadata = {
        "status": "PASS",
        "year": 2026,
        "coverage_start": str(frame["trade_date"].min().date()),
        "coverage_end": str(frame["trade_date"].max().date()),
        "trading_day_count": int(len(frame)),
        "week_count": int(frame["week_period"].nunique()),
        "official_source": OFFICIAL_SOURCE,
        "official_closures": [
            {"start": start, "end": end, "name": name}
            for start, end, name in OFFICIAL_CLOSURES
        ],
        "secondary_cross_check": {
            "source": "akshare.tool_trade_date_hist_sina",
            "date_set_matches": True,
            "only_official": [],
            "only_secondary": [],
        },
        "file": OUTPUT.relative_to(ROOT).as_posix(),
        "sha256": sha256(OUTPUT),
        "retrieved_at": datetime.now(TIMEZONE).isoformat(),
    }
    atomic_text(METADATA, json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
