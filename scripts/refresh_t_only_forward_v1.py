"""刷新 T_ONLY 前瞻行情；严格禁止覆盖父协议冻结输入。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.t_only_robustness_forward_v1 import load_stress_config
from scripts.download_510300_daily import fetch_from_sina, validate


TIMEZONE = ZoneInfo("Asia/Shanghai")
MANIFEST = ROOT / "config" / "t_only_forward_v1_freeze_manifest.json"
DATA_GATE = ROOT / "reports" / "data_quality" / "t_only_forward_v1_data_gate.json"
QUARANTINE = ROOT / "data" / "quarantine" / "t_only_forward_v1_overlap_mismatches.csv"


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


def verify_manifest() -> dict:
    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：T_ONLY 前瞻冻结清单缺失")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for section in (
        "protocol_files",
        "implementation_files",
        "parent_files",
        "parent_input_files",
    ):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：冻结文件指纹变化：{relative_path}")
    return manifest


def dividend_secondary_cross_check(dividend_file: Path) -> dict:
    official = pd.read_csv(dividend_file)
    official["ex_date"] = pd.to_datetime(official["ex_date"], errors="raise")
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
    comparison = official[["ex_date", "cash_dividend_per_share"]].merge(
        cumulative[["ex_date", "cash_dividend_implied"]],
        on="ex_date",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    comparison["difference"] = (
        comparison["cash_dividend_per_share"]
        - comparison["cash_dividend_implied"]
    ).abs()
    date_set_matches = bool(comparison["_merge"].eq("both").all())
    amount_matches = bool(comparison["difference"].dropna().le(1e-9).all())
    return {
        "status": "PASS" if date_set_matches and amount_matches else "FAIL_NEW_EVENT_REQUIRES_OFFICIAL_REVIEW",
        "source": "akshare.fund_etf_dividend_sina(sh510300)",
        "official_event_count": int(len(official)),
        "secondary_event_count": int(len(cumulative)),
        "date_set_matches": date_set_matches,
        "amount_matches": amount_matches,
        "maximum_absolute_amount_difference": (
            float(comparison["difference"].max())
            if comparison["difference"].notna().any()
            else None
        ),
    }


def main() -> int:
    manifest = verify_manifest()
    config = load_stress_config()
    forward = config["forward_data"]
    frozen_path = ROOT / forward["frozen_market_file"]
    output_path = ROOT / forward["refreshed_market_file"]
    metadata_path = ROOT / forward["metadata_file"]
    dividend_path = ROOT / forward["dividend_file"]
    now = datetime.now(TIMEZONE)
    requested_end = now.date()
    if now.hour < 16:
        requested_end -= timedelta(days=1)
    retrieved_at = now.isoformat()

    frozen = pd.read_parquet(frozen_path).copy()
    frozen["date"] = pd.to_datetime(frozen["date"], errors="raise").dt.normalize()
    frozen = frozen.sort_values("date").drop_duplicates("date", keep="last")
    start_date = str(frozen["date"].min().date())
    end_date = str(requested_end)
    downloaded = fetch_from_sina("510300.SH", start_date, end_date)
    downloaded = downloaded.sort_values("date").reset_index(drop=True)
    validate(downloaded, start_date, end_date)

    overlap_columns = list(forward["overlap_columns"])
    overlap = frozen[["date", *overlap_columns]].merge(
        downloaded[["date", *overlap_columns]],
        on="date",
        how="left",
        suffixes=("_frozen", "_downloaded"),
        validate="one_to_one",
    )
    mismatch = pd.Series(False, index=overlap.index)
    for column in overlap_columns:
        left = pd.to_numeric(overlap[f"{column}_frozen"], errors="coerce")
        right = pd.to_numeric(overlap[f"{column}_downloaded"], errors="coerce")
        mismatch |= ~np.isclose(
            left,
            right,
            rtol=0.0,
            atol=float(forward["overlap_numeric_tolerance"]),
            equal_nan=False,
        )
    mismatches = overlap.loc[mismatch].copy()
    if not mismatches.empty:
        QUARANTINE.parent.mkdir(parents=True, exist_ok=True)
        mismatches.to_csv(QUARANTINE, index=False)
        gate = {
            "status": "NO_VIEW_OVERLAP_MISMATCH",
            "retrieved_at": retrieved_at,
            "mismatch_rows": int(len(mismatches)),
            "quarantine_file": QUARANTINE.relative_to(ROOT).as_posix(),
            "frozen_input_unchanged": sha256(frozen_path)
            == manifest["parent_input_files"][forward["frozen_market_file"]],
        }
        atomic_text(DATA_GATE, json.dumps(gate, ensure_ascii=False, indent=2))
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 2

    frozen_last = frozen["date"].max()
    new_rows = downloaded.loc[downloaded["date"] > frozen_last].copy()
    new_rows["retrieved_at"] = retrieved_at
    combined = pd.concat([frozen, new_rows], ignore_index=True, sort=False)
    combined = combined.sort_values("date").drop_duplicates("date", keep="first")
    combined["symbol"] = "510300.SH"
    combined["source"] = "akshare.fund_etf_hist_sina"
    combined["volume_unit"] = "share"
    combined["amount_unit"] = "CNY"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(output_path)

    dividend_check = dividend_secondary_cross_check(dividend_path)
    actual_last = pd.Timestamp(combined["date"].max()).normalize()
    coverage_effective_through = (
        str(actual_last.date()) if dividend_check["status"] == "PASS" else None
    )
    status = "PASS" if dividend_check["status"] == "PASS" else "NO_VIEW_DIVIDEND_REVIEW_REQUIRED"
    metadata = {
        "status": status,
        "symbol": "510300.SH",
        "source": forward["source"],
        "retrieved_at": retrieved_at,
        "requested_end": end_date,
        "actual_first_date": str(pd.Timestamp(combined["date"].min()).date()),
        "actual_last_date": str(actual_last.date()),
        "row_count": int(len(combined)),
        "new_rows_after_frozen_cutoff": int(len(new_rows)),
        "frozen_cutoff": str(frozen_last.date()),
        "overlap_rows_verified": int(len(frozen)),
        "overlap_mismatch_rows": 0,
        "file": output_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(output_path),
        "frozen_input_unchanged": sha256(frozen_path)
        == manifest["parent_input_files"][forward["frozen_market_file"]],
        "dividend_secondary_cross_check": dividend_check,
        "dividend_coverage_effective_through": coverage_effective_through,
    }
    atomic_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    atomic_text(DATA_GATE, json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
