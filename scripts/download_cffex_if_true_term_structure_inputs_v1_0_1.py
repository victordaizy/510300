"""下载并规范化中金所IF逐合约日行情及样本末日合约到期元数据。"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
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
    ROOT
    / "config"
    / "510300_if_true_term_structure_binary_screen_v1_0_1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "0d406748722608296935ec7b398e9e4a1fab153b6d411dd83fb70bbe654fa4c1"
)
START_DATE = pd.Timestamp("2010-04-16")
END_DATE = pd.Timestamp("2026-08-12")
ARCHIVE_URL = "http://www.cffex.com.cn/sj/historysj/{yyyymm}/zip/{yyyymm}.zip"
ACTIVE_METADATA_URL = "http://www.cffex.com.cn/sj/jycs/202608/12/index.xml"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

RAW_DIR = ROOT / "data" / "raw" / "futures" / "cffex_if_contract_history_v1_0_1"
ARCHIVE_DIR = RAW_DIR / "monthly_archives"
CONTRACT_DAILY_PATH = RAW_DIR / "cffex_if_contract_daily.parquet"
EXPIRY_PATH = RAW_DIR / "cffex_if_contract_expiry.parquet"
ACTIVE_METADATA_PATH = RAW_DIR / "active_contract_metadata_20260812.xml"
AUDIT_PATH = (
    ROOT / "reports" / "data_quality" / "cffex_if_true_term_structure_inputs_v1_0_1.json"
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


def month_ids(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """生成闭区间月份标识。"""

    return [value.strftime("%Y%m") for value in pd.period_range(start, end, freq="M")]


def valid_zip(payload: bytes) -> bool:
    """核对响应确为至少含一个CSV的ZIP。"""

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return bool(
                archive.testzip() is None
                and any(name.lower().endswith(".csv") for name in archive.namelist())
            )
    except zipfile.BadZipFile:
        return False


def download_month(yyyymm: str) -> tuple[str, bytes, str]:
    """下载或复用单个月包，最多尝试三次。"""

    path = ARCHIVE_DIR / f"{yyyymm}.zip"
    if path.exists():
        cached = path.read_bytes()
        if valid_zip(cached):
            return yyyymm, cached, "CACHE"
    url = ARCHIVE_URL.format(yyyymm=yyyymm)
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(10, 60),
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.content
            if not valid_zip(payload):
                raise RuntimeError("响应不是有效中金所月度ZIP")
            atomic_bytes(payload, path)
            return yyyymm, payload, "NETWORK"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"第{attempt}次：{type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"{yyyymm}下载失败；{' | '.join(errors)}")


def parse_month(yyyymm: str, payload: bytes, archive_hash: str) -> pd.DataFrame:
    """按稳定列位置解析月包内IF逐合约记录。"""

    rows: list[pd.DataFrame] = []
    source_url = ARCHIVE_URL.format(yyyymm=yyyymm)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member_name in sorted(archive.namelist()):
            matched = re.search(r"(\d{8})_1\.csv$", Path(member_name).name)
            if matched is None:
                continue
            date = pd.Timestamp(datetime.strptime(matched.group(1), "%Y%m%d"))
            if date < START_DATE or date > END_DATE:
                continue
            member = archive.read(member_name)
            try:
                frame = pd.read_csv(io.BytesIO(member), encoding="gb18030")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"无法解析{yyyymm}/{member_name}：{exc}") from exc
            if frame.shape[1] not in {14, 15}:
                raise ValueError(
                    f"{yyyymm}/{member_name}列数异常：{frame.shape[1]}"
                )
            symbols = frame.iloc[:, 0].astype(str).str.strip()
            frame = frame.loc[symbols.str.fullmatch(r"IF\d{4}", na=False)].copy()
            if frame.empty:
                raise ValueError(f"交易日文件没有IF逐合约记录：{member_name}")
            result = pd.DataFrame(
                {
                    "symbol": frame.iloc[:, 0].astype(str).str.strip(),
                    "date": date,
                    "open": pd.to_numeric(frame.iloc[:, 1], errors="coerce"),
                    "high": pd.to_numeric(frame.iloc[:, 2], errors="coerce"),
                    "low": pd.to_numeric(frame.iloc[:, 3], errors="coerce"),
                    "volume": pd.to_numeric(frame.iloc[:, 4], errors="coerce"),
                    "turnover": pd.to_numeric(frame.iloc[:, 5], errors="coerce"),
                    "open_interest": pd.to_numeric(frame.iloc[:, 6], errors="coerce"),
                    "close": pd.to_numeric(frame.iloc[:, 8], errors="coerce"),
                    "settle": pd.to_numeric(frame.iloc[:, 9], errors="coerce"),
                    "pre_settle": pd.to_numeric(frame.iloc[:, 10], errors="coerce"),
                    "archive_yyyymm": yyyymm,
                    "archive_sha256": archive_hash,
                    "archive_member": member_name,
                    "source_url": source_url,
                }
            )
            rows.append(result)
    if not rows:
        raise ValueError(f"月包{yyyymm}在固定日期范围内没有IF记录")
    return pd.concat(rows, ignore_index=True)


def download_active_metadata() -> tuple[bytes, dict[str, pd.Timestamp]]:
    """下载样本末日交易参数并提取仍存续IF合约到期日。"""

    errors: list[str] = []
    payload = b""
    for attempt in range(1, 4):
        try:
            response = requests.get(
                ACTIVE_METADATA_URL,
                headers={"User-Agent": USER_AGENT},
                timeout=(10, 60),
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.content
            root = ET.fromstring(payload)
            mapping: dict[str, pd.Timestamp] = {}
            for record in root.findall(".//INDEX"):
                values = {child.tag: child.text for child in record}
                symbol = (values.get("INSTRUMENT_ID") or "").strip()
                if re.fullmatch(r"IF\d{4}", symbol):
                    expiry = pd.to_datetime(
                        values.get("END_TRADING_DAY"), format="%Y%m%d", errors="raise"
                    ).normalize()
                    mapping[symbol] = expiry
            if not mapping:
                raise ValueError("样本末日交易参数没有IF合约")
            atomic_bytes(payload, ACTIVE_METADATA_PATH)
            return payload, mapping
        except Exception as exc:  # noqa: BLE001
            errors.append(f"第{attempt}次：{type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"样本末日合约元数据下载失败；{' | '.join(errors)}")


def build_expiry_table(
    daily: pd.DataFrame,
    active_mapping: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """为每个历史IF合约建立公开到期日，不把样本截止日当到期日。"""

    grouped = daily.groupby("symbol", sort=True)["date"].agg(["min", "max"])
    grouped.rename(columns={"min": "first_date", "max": "last_history_date"}, inplace=True)
    active_symbols = set(
        daily.loc[daily["date"].eq(END_DATE), "symbol"].astype(str).unique()
    )
    if active_symbols != set(active_mapping):
        raise ValueError(
            "样本末日日包与交易参数IF合约集合不一致："
            f"日包={sorted(active_symbols)}，元数据={sorted(active_mapping)}"
        )
    rows: list[dict[str, Any]] = []
    for symbol, values in grouped.iterrows():
        active = symbol in active_symbols
        expiry = (
            active_mapping[symbol]
            if active
            else pd.Timestamp(values["last_history_date"])
        )
        contract_month = pd.Timestamp(
            year=2000 + int(symbol[2:4]), month=int(symbol[4:6]), day=1
        )
        if expiry.year != contract_month.year or expiry.month != contract_month.month:
            raise ValueError(f"{symbol}到期日不在合约月份：{expiry.date()}")
        rows.append(
            {
                "symbol": symbol,
                "first_date": pd.Timestamp(values["first_date"]),
                "last_history_date": pd.Timestamp(values["last_history_date"]),
                "expiry_date": expiry,
                "expiry_source": (
                    "CFFEX_TRADING_PARAMETER_END_TRADING_DAY_AT_CEILING"
                    if active
                    else "CFFEX_HISTORY_LAST_APPEARANCE"
                ),
                "active_at_ceiling": active,
            }
        )
    result = pd.DataFrame(rows).sort_values("symbol", kind="mergesort")
    result.reset_index(drop=True, inplace=True)
    if result["symbol"].duplicated().any():
        raise AssertionError("合约到期表出现重复代码")
    return result


def write_failure(error: Exception, *, stage: str) -> None:
    """保留外部源或程序失败状态。"""

    atomic_json(
        {
            "status": "EXTERNAL_SOURCE_OR_PROGRAM_FAILED",
            "stage": stage,
            "failed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
            "performance_returns_read": False,
        },
        AUDIT_PATH,
    )


def main() -> int:
    """正式获取全部官方逐合约历史并生成固定输入。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("正式获取前固定的V1.0.1候选合同发生漂移")
    if AUDIT_PATH.exists() and CONTRACT_DAILY_PATH.exists() and EXPIRY_PATH.exists():
        prior = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        if prior.get("status") == "SUCCESS_CFFEX_IF_CONTRACT_HISTORY_ACQUIRED":
            print(
                json.dumps(
                    {
                        "status": "ALREADY_ACQUIRED",
                        "contract_daily_sha256": sha256_file(CONTRACT_DAILY_PATH),
                        "expiry_sha256": sha256_file(EXPIRY_PATH),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return 0

    months = month_ids(START_DATE, END_DATE)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, tuple[bytes, str]] = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(download_month, month): month for month in months}
            for completed_count, future in enumerate(as_completed(futures), start=1):
                yyyymm, payload, mode = future.result()
                downloaded[yyyymm] = (payload, mode)
                if completed_count % 12 == 0 or completed_count == len(months):
                    print(
                        f"月包获取进度：{completed_count}/{len(months)}",
                        flush=True,
                    )
    except Exception as exc:  # noqa: BLE001
        write_failure(exc, stage="MONTHLY_ARCHIVE_DOWNLOAD")
        raise

    try:
        frames: list[pd.DataFrame] = []
        archive_hashes: dict[str, str] = {}
        for yyyymm in months:
            payload, _mode = downloaded[yyyymm]
            archive_hash = sha256_bytes(payload)
            archive_hashes[yyyymm] = archive_hash
            frames.append(parse_month(yyyymm, payload, archive_hash))
        daily = pd.concat(frames, ignore_index=True)
        daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
        daily.sort_values(["date", "symbol"], kind="mergesort", inplace=True)
        daily.reset_index(drop=True, inplace=True)
        if daily.duplicated(["date", "symbol"]).any():
            raise ValueError("中金所IF逐合约数据存在日期代码重复")
        if daily["date"].min() != START_DATE or daily["date"].max() != END_DATE:
            raise ValueError("中金所IF逐合约首末日期不匹配固定合同")
        if not daily["symbol"].str.fullmatch(r"IF\d{4}").all():
            raise ValueError("中金所逐合约输入混入非IF代码")
        daily["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

        metadata_payload, active_mapping = download_active_metadata()
        expiry = build_expiry_table(daily, active_mapping)
        atomic_parquet(daily, CONTRACT_DAILY_PATH)
        atomic_parquet(expiry, EXPIRY_PATH)
        audit = {
            "status": "SUCCESS_CFFEX_IF_CONTRACT_HISTORY_ACQUIRED",
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "candidate_contract": CANDIDATE_CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
            "source": {
                "provider": "中国金融期货交易所",
                "monthly_archive_url_template": ARCHIVE_URL,
                "active_metadata_url": ACTIVE_METADATA_URL,
                "active_metadata_sha256": sha256_bytes(metadata_payload),
                "archive_count": len(months),
                "archive_sha256": archive_hashes,
                "network_download_count": int(
                    sum(mode == "NETWORK" for _payload, mode in downloaded.values())
                ),
                "cache_reuse_count": int(
                    sum(mode == "CACHE" for _payload, mode in downloaded.values())
                ),
            },
            "daily": {
                "rows": int(len(daily)),
                "unique_dates": int(daily["date"].nunique()),
                "unique_contracts": int(daily["symbol"].nunique()),
                "first_date": daily["date"].min().date().isoformat(),
                "last_date": daily["date"].max().date().isoformat(),
                "nonpositive_or_missing_close_rows": int(
                    (daily["close"].isna() | daily["close"].le(0.0)).sum()
                ),
                "negative_or_missing_open_interest_rows": int(
                    (daily["open_interest"].isna() | daily["open_interest"].lt(0.0)).sum()
                ),
            },
            "expiry": {
                "rows": int(len(expiry)),
                "active_at_ceiling_count": int(expiry["active_at_ceiling"].sum()),
                "active_symbols": expiry.loc[
                    expiry["active_at_ceiling"], "symbol"
                ].tolist(),
                "active_expiries": {
                    symbol: date.date().isoformat()
                    for symbol, date in active_mapping.items()
                },
            },
            "performance_returns_read": False,
            "outputs": {
                CONTRACT_DAILY_PATH.relative_to(ROOT).as_posix(): sha256_file(
                    CONTRACT_DAILY_PATH
                ),
                EXPIRY_PATH.relative_to(ROOT).as_posix(): sha256_file(EXPIRY_PATH),
                ACTIVE_METADATA_PATH.relative_to(ROOT).as_posix(): sha256_file(
                    ACTIVE_METADATA_PATH
                ),
            },
        }
        atomic_json(audit, AUDIT_PATH)
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "rows": audit["daily"]["rows"],
                    "unique_dates": audit["daily"]["unique_dates"],
                    "unique_contracts": audit["daily"]["unique_contracts"],
                    "active_symbols": audit["expiry"]["active_symbols"],
                    "contract_daily_sha256": audit["outputs"][
                        CONTRACT_DAILY_PATH.relative_to(ROOT).as_posix()
                    ],
                    "expiry_sha256": audit["outputs"][
                        EXPIRY_PATH.relative_to(ROOT).as_posix()
                    ],
                    "audit_sha256": sha256_file(AUDIT_PATH),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        write_failure(exc, stage="PARSE_NORMALIZE_AND_EXPIRY")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
