"""合并510300基金份额与收盘折溢价历史，供固定二元规则筛选使用。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_etf_share_premium_level_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "33b62f3917287cbf9dbc194bc7027cac3b5c29c9e409955ed557be8d0f91c658"
)
SEAM_DATE = pd.Timestamp("2021-08-12")

INPUTS = {
    "historical_share": {
        "path": ROOT
        / "data"
        / "external_validation"
        / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2"
        / "snapshots"
        / "20260828T125705_0800"
        / "510300_fund_share_daily.parquet",
        "sha256": "1b816e788a4e030d4b944f1a38951188da422bc5f6f3088fc51c4406fcd3b787",
    },
    "historical_nav": {
        "path": ROOT
        / "data"
        / "external_validation"
        / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2"
        / "snapshots"
        / "20260828T125705_0800"
        / "510300_nav_daily.parquet",
        "sha256": "823b4ea5cafa0b6d7713a5777d1765e9a40b0cacd81085a4c6e0d3a2078ca5b3",
    },
    "current_share": {
        "path": ROOT / "data" / "raw" / "flow" / "510300_fund_share_daily_tushare.parquet",
        "sha256": "e34ef2ee9ead0616aef2705a93b1b32c2e10be7841134c59126071b126ba71f4",
    },
    "current_nav": {
        "path": ROOT / "data" / "raw" / "fund" / "510300_nav_daily_raw.parquet",
        "sha256": "d5f353a8d7bbcb19b5007e4dcb619d0d854111d1f9641fc368e31b52e5c5369d",
    },
}

OUTPUT_PATH = (
    ROOT / "data" / "raw" / "flow" / "510300_etf_share_premium_level_full_v1.parquet"
)
AUDIT_PATH = (
    ROOT / "reports" / "data_quality" / "510300_etf_share_premium_level_full_v1.json"
)


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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
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


def verify_fixed_inputs() -> dict[str, str]:
    """核对候选合同和四个既有原始输入未发生漂移。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("全历史合并前固定的候选合同发生漂移")
    hashes: dict[str, str] = {}
    for name, contract in INPUTS.items():
        path = contract["path"]
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
        actual = sha256_file(path)
        if actual != contract["sha256"]:
            raise ValueError(f"固定输入哈希漂移：{name}")
        hashes[path.relative_to(ROOT).as_posix()] = actual
    return hashes


def normalize_share(frame: pd.DataFrame, *, segment: str) -> pd.DataFrame:
    """规范化份额输入并保留来源字段。"""

    required = {"date", "fund_shares", "source"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{segment}份额输入缺少字段：{sorted(missing)}")
    result = frame[["date", "fund_shares", "source", "retrieved_at"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["fund_shares"] = pd.to_numeric(result["fund_shares"], errors="raise")
    result.rename(
        columns={"source": "share_source", "retrieved_at": "share_retrieved_at"},
        inplace=True,
    )
    result["share_retrieved_at"] = result["share_retrieved_at"].map(
        lambda value: None if pd.isna(value) else str(value)
    )
    result["share_segment"] = segment
    result.sort_values("date", kind="mergesort", inplace=True)
    if result["date"].duplicated().any():
        raise ValueError(f"{segment}份额日期重复")
    if result["fund_shares"].isna().any() or result["fund_shares"].le(0.0).any():
        raise ValueError(f"{segment}基金份额缺失或非正")
    return result.reset_index(drop=True)


def normalize_nav(frame: pd.DataFrame, *, segment: str) -> pd.DataFrame:
    """规范化收盘折溢价输入并保留双来源字段。"""

    required = {
        "date",
        "close_premium_to_nav",
        "source_primary",
        "source_secondary",
        "retrieved_at",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"{segment}净值输入缺少字段：{sorted(missing)}")
    result = frame[
        [
            "date",
            "close_premium_to_nav",
            "source_primary",
            "source_secondary",
            "retrieved_at",
        ]
    ].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["close_premium_to_nav"] = pd.to_numeric(
        result["close_premium_to_nav"], errors="coerce"
    )
    result.rename(columns={"retrieved_at": "nav_retrieved_at"}, inplace=True)
    result["nav_retrieved_at"] = result["nav_retrieved_at"].map(
        lambda value: None if pd.isna(value) else str(value)
    )
    result["nav_segment"] = segment
    result.sort_values("date", kind="mergesort", inplace=True)
    if result["date"].duplicated().any():
        raise ValueError(f"{segment}净值日期重复")
    if result["close_premium_to_nav"].dropna().abs().ge(0.10).any():
        raise ValueError(f"{segment}折溢价绝对值达到10%，疑似单位或数据异常")
    return result.reset_index(drop=True)


def seam_value(frame: pd.DataFrame, column: str) -> float:
    """提取唯一拼接日数值。"""

    values = frame.loc[frame["date"].eq(SEAM_DATE), column]
    if len(values) != 1:
        raise ValueError(f"拼接日{SEAM_DATE.date()}的{column}不是唯一值")
    return float(values.iloc[0])


def build_full_history() -> tuple[pd.DataFrame, dict[str, Any]]:
    """完成无填充的历史/当前拼接，并返回逐日联合特征和审计。"""

    source_hashes = verify_fixed_inputs()
    historical_share = normalize_share(
        pd.read_parquet(INPUTS["historical_share"]["path"]), segment="HISTORICAL_SSE"
    )
    historical_nav = normalize_nav(
        pd.read_parquet(INPUTS["historical_nav"]["path"]), segment="HISTORICAL_DUAL"
    )
    current_share = normalize_share(
        pd.read_parquet(INPUTS["current_share"]["path"]), segment="CURRENT_TUSHARE"
    )
    current_nav = normalize_nav(
        pd.read_parquet(INPUTS["current_nav"]["path"]), segment="CURRENT_DUAL"
    )

    historical_share_seam = seam_value(historical_share, "fund_shares")
    current_share_seam = seam_value(current_share, "fund_shares")
    historical_premium_seam = seam_value(historical_nav, "close_premium_to_nav")
    current_premium_seam = seam_value(current_nav, "close_premium_to_nav")
    share_seam_equal = bool(
        np.isclose(historical_share_seam, current_share_seam, rtol=0.0, atol=0.0)
    )
    premium_seam_equal = bool(
        np.isclose(
            historical_premium_seam,
            current_premium_seam,
            rtol=0.0,
            atol=1e-15,
        )
    )
    if not share_seam_equal or not premium_seam_equal:
        raise ValueError("历史与当前输入在拼接日不一致")

    full_share = pd.concat(
        [
            historical_share.loc[historical_share["date"].lt(SEAM_DATE)],
            current_share.loc[current_share["date"].ge(SEAM_DATE)],
        ],
        ignore_index=True,
    )
    full_nav = pd.concat(
        [
            historical_nav.loc[historical_nav["date"].lt(SEAM_DATE)],
            current_nav.loc[current_nav["date"].ge(SEAM_DATE)],
        ],
        ignore_index=True,
    )
    if full_share["date"].duplicated().any() or full_nav["date"].duplicated().any():
        raise ValueError("拼接后出现重复日期")

    nav_missing_premium_dates = sorted(
        full_nav.loc[full_nav["close_premium_to_nav"].isna(), "date"].tolist()
    )
    full_nav = full_nav.loc[full_nav["close_premium_to_nav"].notna()].copy()
    share_dates = set(full_share["date"])
    nav_dates = set(full_nav["date"])
    share_only_dates = sorted(share_dates - nav_dates)
    nav_only_dates = sorted(nav_dates - share_dates)
    outer = full_share.merge(
        full_nav,
        on="date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    full = outer.loc[outer["_merge"].eq("both")].drop(columns="_merge").copy()
    full.sort_values("date", kind="mergesort", inplace=True)
    full.reset_index(drop=True, inplace=True)
    if full.empty or full["date"].duplicated().any():
        raise ValueError("联合特征为空或日期重复")
    if full[["fund_shares", "close_premium_to_nav"]].isna().any().any():
        raise ValueError("联合特征存在缺失，不允许填充")
    full["feature_asof"] = full["date"] + pd.Timedelta(hours=15)
    full["execution_earliest"] = "NEXT_TRADING_DAY_OPEN"

    audit = {
        "status": "SUCCESS_FIXED_CANDIDATE_HISTORY_MERGE_NO_FILL",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_contract": CANDIDATE_CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "source_hashes": source_hashes,
        "seam": {
            "date": SEAM_DATE.date().isoformat(),
            "historical_fund_shares": historical_share_seam,
            "current_fund_shares": current_share_seam,
            "fund_shares_exact_equal": share_seam_equal,
            "historical_close_premium_to_nav": historical_premium_seam,
            "current_close_premium_to_nav": current_premium_seam,
            "premium_equal_at_1e_minus_15": premium_seam_equal,
            "historical_rows_selected_strictly_before_seam": True,
            "current_rows_selected_on_or_after_seam": True,
        },
        "input_rows": {
            "historical_share": int(len(historical_share)),
            "historical_nav": int(len(historical_nav)),
            "current_share": int(len(current_share)),
            "current_nav": int(len(current_nav)),
        },
        "merged_rows": int(len(full)),
        "first_date": full["date"].min().date().isoformat(),
        "last_date": full["date"].max().date().isoformat(),
        "share_only_dates_excluded_without_fill": [
            value.date().isoformat() for value in share_only_dates
        ],
        "nav_missing_premium_dates_excluded_without_fill": [
            value.date().isoformat() for value in nav_missing_premium_dates
        ],
        "nav_only_dates_excluded_without_fill": [
            value.date().isoformat() for value in nav_only_dates
        ],
        "share_only_dates_are_weekends": bool(
            share_only_dates and all(value.dayofweek >= 5 for value in share_only_dates)
        ),
        "nav_missing_premium_dates_are_weekends": bool(
            nav_missing_premium_dates
            and all(value.dayofweek >= 5 for value in nav_missing_premium_dates)
        ),
        "future_510300_return_columns_read": 0,
        "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
    }
    return full, audit


def main() -> int:
    """构建一次可审计的联合特征输入。"""

    full, audit = build_full_history()
    atomic_parquet(full, OUTPUT_PATH)
    audit["output_sha256"] = sha256_file(OUTPUT_PATH)
    atomic_json(audit, AUDIT_PATH)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "rows": audit["merged_rows"],
                "first_date": audit["first_date"],
                "last_date": audit["last_date"],
                "excluded_without_fill": audit[
                    "share_only_dates_excluded_without_fill"
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


if __name__ == "__main__":
    raise SystemExit(main())
