"""下载并规范化中金所IF前20会员成交与多空持仓排名历史。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_cffex_member_position_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "845f94c3eed3ef6991ff78312c1578b18baffcc2dd8039e915af4fe0d74ac739"
)
DATE_SCHEDULE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "futures"
    / "cffex_if_contract_history_v1_0_1"
    / "cffex_if_contract_daily.parquet"
)
DATE_SCHEDULE_SHA256 = (
    "a4da6ec1b61d98cc59e9c9ae1850bebe6372a3fc6e04935cccecb094a28dc8a2"
)
START_DATE = pd.Timestamp("2010-04-16")
END_DATE = pd.Timestamp("2026-08-12")
SOURCE_URL = "http://www.cffex.com.cn/sj/ccpm/{yyyymm}/{dd}/IF_1.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

RAW_DIR = ROOT / "data" / "raw" / "futures" / "cffex_if_member_positions_v1"
DAILY_DIR = RAW_DIR / "daily_files"
OUTPUT_PATH = RAW_DIR / "cffex_if_member_position_rank.parquet"
AUDIT_PATH = (
    ROOT / "reports" / "data_quality" / "cffex_if_member_positions_v1.json"
)


def sha256_bytes(payload: bytes) -> str:
    """计算字节串SHA-256。"""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """转换NumPy、Pandas值以便写入JSON。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"无法JSON序列化：{type(value).__name__}")


def atomic_bytes(payload: bytes, path: Path) -> None:
    """原子写入二进制文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def source_url(date: pd.Timestamp) -> str:
    """生成单日官方排名文件地址。"""

    return SOURCE_URL.format(yyyymm=date.strftime("%Y%m"), dd=date.strftime("%d"))


def looks_like_rank_file(payload: bytes, date: pd.Timestamp) -> bool:
    """核对响应含固定日期与IF合约排名行。"""

    try:
        text = payload.decode("gb18030")
    except UnicodeDecodeError:
        return False
    date_text = date.strftime("%Y%m%d")
    return bool(
        date_text in text
        and re.search(rf"^{date_text},IF\d{{4}}\s*,\d+,", text, flags=re.MULTILINE)
    )


def download_day(date: pd.Timestamp) -> tuple[pd.Timestamp, bytes, str]:
    """下载或复用单日官方排名文件，最多尝试三次。"""

    date_text = date.strftime("%Y%m%d")
    path = DAILY_DIR / date_text[:4] / f"{date_text}.csv"
    if path.exists():
        cached = path.read_bytes()
        if looks_like_rank_file(cached, date):
            return date, cached, "CACHE"
    url = source_url(date)
    errors: list[str] = []
    for attempt in range(1, 6):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(10, 45),
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.content
            if not looks_like_rank_file(payload, date):
                raise RuntimeError("响应不含该日IF会员排名行")
            atomic_bytes(payload, path)
            return date, payload, "NETWORK"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"第{attempt}次：{type(exc).__name__}: {exc}")
            if attempt < 5:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"{date_text}下载失败；{' | '.join(errors)}")


def parse_numeric(value: str) -> tuple[int, bool]:
    """将官方数值单元格转整数；空白或横线按合同记为0。"""

    text = value.strip().replace(",", "")
    if text in {"", "-", "--", "null", "None", "nan"}:
        return 0, True
    return int(float(text)), False


def parse_day(
    date: pd.Timestamp,
    payload: bytes,
    payload_hash: str,
) -> tuple[pd.DataFrame, int]:
    """解析两种历史表头结构下的12列排名明细。"""

    text = payload.decode("gb18030")
    date_text = date.strftime("%Y%m%d")
    records: list[dict[str, Any]] = []
    blank_or_dash_count = 0
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 12 or row[0].strip() != date_text:
            continue
        symbol = row[1].strip()
        rank_text = row[2].strip()
        if not re.fullmatch(r"IF\d{4}", symbol) or not rank_text.isdigit():
            continue
        rank = int(rank_text)
        if rank < 1 or rank > 20:
            continue
        numeric: list[int] = []
        for position in [4, 5, 7, 8, 10, 11]:
            value, was_blank = parse_numeric(row[position])
            numeric.append(value)
            blank_or_dash_count += int(was_blank)
        records.append(
            {
                "date": date,
                "symbol": symbol,
                "rank": rank,
                "volume_member": row[3].strip(),
                "volume": numeric[0],
                "volume_change": numeric[1],
                "long_member": row[6].strip(),
                "long_open_interest": numeric[2],
                "long_open_interest_change": numeric[3],
                "short_member": row[9].strip(),
                "short_open_interest": numeric[4],
                "short_open_interest_change": numeric[5],
                "source_url": source_url(date),
                "raw_file_sha256": payload_hash,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"{date_text}没有可解析的IF排名明细")
    if frame.duplicated(["date", "symbol", "rank"]).any():
        raise ValueError(f"{date_text}出现合约排名重复")
    if frame[["volume", "long_open_interest", "short_open_interest"]].lt(0).any().any():
        raise ValueError(f"{date_text}出现负持仓或负成交量")
    return frame, blank_or_dash_count


def write_failure(error: Exception, *, stage: str, completed_dates: int) -> None:
    """保留外部源或程序失败状态。"""

    atomic_json(
        {
            "status": "EXTERNAL_SOURCE_OR_PROGRAM_FAILED",
            "stage": stage,
            "failed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_dates": completed_dates,
            "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
            "performance_returns_read": False,
        },
        AUDIT_PATH,
    )


def main() -> int:
    """正式获取全部IF会员持仓排名并生成固定输入。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("正式获取前固定的会员持仓候选合同发生漂移")
    if sha256_file(DATE_SCHEDULE_PATH) != DATE_SCHEDULE_SHA256:
        raise ValueError("中金所IF官方交易日输入发生漂移")
    if AUDIT_PATH.exists() and OUTPUT_PATH.exists():
        prior = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        if prior.get("status") == "SUCCESS_CFFEX_IF_MEMBER_POSITIONS_ACQUIRED":
            print(
                json.dumps(
                    {
                        "status": "ALREADY_ACQUIRED",
                        "output_sha256": sha256_file(OUTPUT_PATH),
                        "audit_sha256": sha256_file(AUDIT_PATH),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return 0

    schedule = pd.read_parquet(DATE_SCHEDULE_PATH, columns=["date"])
    schedule["date"] = pd.to_datetime(schedule["date"], errors="raise").dt.normalize()
    dates = sorted(
        value
        for value in schedule["date"].unique()
        if START_DATE <= pd.Timestamp(value) <= END_DATE
    )
    dates = [pd.Timestamp(value) for value in dates]
    if not dates or dates[0] != START_DATE or dates[-1] != END_DATE:
        raise ValueError("IF官方排名下载交易日首末日期不匹配")
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[pd.Timestamp, tuple[bytes, str]] = {}
    completed_count = 0
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(download_day, date): date for date in dates}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                date, payload, mode = future.result()
                downloaded[date] = (payload, mode)
                if completed_count % 250 == 0 or completed_count == len(dates):
                    print(
                        f"会员排名获取进度：{completed_count}/{len(dates)}",
                        flush=True,
                    )
    except Exception as exc:  # noqa: BLE001
        write_failure(
            exc,
            stage="DAILY_POSITION_RANK_DOWNLOAD",
            completed_dates=completed_count,
        )
        raise

    try:
        frames: list[pd.DataFrame] = []
        file_hashes: dict[str, str] = {}
        blank_or_dash_count = 0
        for index, date in enumerate(dates, start=1):
            payload, _mode = downloaded[date]
            payload_hash = sha256_bytes(payload)
            file_hashes[date.strftime("%Y%m%d")] = payload_hash
            frame, missing_count = parse_day(date, payload, payload_hash)
            frames.append(frame)
            blank_or_dash_count += missing_count
            if index % 500 == 0 or index == len(dates):
                print(f"会员排名解析进度：{index}/{len(dates)}", flush=True)
        ranks = pd.concat(frames, ignore_index=True)
        ranks.sort_values(["date", "symbol", "rank"], kind="mergesort", inplace=True)
        ranks.reset_index(drop=True, inplace=True)
        if ranks.duplicated(["date", "symbol", "rank"]).any():
            raise ValueError("合并后的IF会员排名存在日期代码名次重复")
        if ranks["date"].min() != START_DATE or ranks["date"].max() != END_DATE:
            raise ValueError("合并后的IF会员排名首末日期不匹配")
        if ranks["date"].nunique() != len(dates):
            raise ValueError("合并后的IF会员排名缺少官方交易日")
        ranks["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        atomic_parquet(ranks, OUTPUT_PATH)
        counts = ranks.groupby(["date", "symbol"], sort=False)["rank"].nunique()
        audit = {
            "status": "SUCCESS_CFFEX_IF_MEMBER_POSITIONS_ACQUIRED",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_contract": CANDIDATE_CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
            "source": {
                "provider": "中国金融期货交易所",
                "daily_url_template": SOURCE_URL,
                "date_schedule": DATE_SCHEDULE_PATH.relative_to(ROOT).as_posix(),
                "date_schedule_sha256": DATE_SCHEDULE_SHA256,
                "daily_file_count": len(dates),
                "daily_file_sha256": file_hashes,
                "network_download_count": int(
                    sum(mode == "NETWORK" for _payload, mode in downloaded.values())
                ),
                "cache_reuse_count": int(
                    sum(mode == "CACHE" for _payload, mode in downloaded.values())
                ),
            },
            "normalized": {
                "rows": int(len(ranks)),
                "unique_dates": int(ranks["date"].nunique()),
                "unique_contracts": int(ranks["symbol"].nunique()),
                "first_date": ranks["date"].min().date().isoformat(),
                "last_date": ranks["date"].max().date().isoformat(),
                "minimum_rank_rows_per_date_contract": int(counts.min()),
                "maximum_rank_rows_per_date_contract": int(counts.max()),
                "date_contracts_with_fewer_than_5_ranks": int(counts.lt(5).sum()),
                "blank_or_dash_numeric_cells_interpreted_as_zero": int(
                    blank_or_dash_count
                ),
                "negative_position_level_rows": int(
                    ranks[["long_open_interest", "short_open_interest"]]
                    .lt(0)
                    .any(axis=1)
                    .sum()
                ),
            },
            "performance_returns_read": False,
            "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "output_sha256": sha256_file(OUTPUT_PATH),
        }
        atomic_json(audit, AUDIT_PATH)
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "rows": audit["normalized"]["rows"],
                    "unique_dates": audit["normalized"]["unique_dates"],
                    "unique_contracts": audit["normalized"]["unique_contracts"],
                    "minimum_rank_rows_per_date_contract": audit["normalized"][
                        "minimum_rank_rows_per_date_contract"
                    ],
                    "output_sha256": audit["output_sha256"],
                    "audit_sha256": sha256_file(AUDIT_PATH),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        write_failure(exc, stage="PARSE_AND_NORMALIZE", completed_dates=len(dates))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
