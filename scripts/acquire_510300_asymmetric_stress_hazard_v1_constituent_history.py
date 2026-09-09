"""补齐旧路线 G2 所需的 2015—2020 点时成分股总收益历史。"""

from __future__ import annotations

import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import sha256_file, strict_json_text
from scripts.download_a_share_hash_holdout_training_v1 import download_daily
from scripts.download_csi300_all_etf_momentum_v1 import credentials


MEMBERSHIP_PATH = (
    ROOT
    / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/"
    "000300_daily_pit_membership_20150101_20260814.parquet"
)
OUTPUT_DIR = (
    ROOT
    / "data/curated/"
    "510300_asymmetric_stress_hazard_v1_original_route_g2"
)
CHECKPOINT_DIR = (
    ROOT
    / "data/raw/"
    "510300_asymmetric_stress_hazard_v1_original_route_g2/"
    "constituent_history_checkpoints"
)
OUTPUT_PATH = OUTPUT_DIR / "constituent_history_20141101_20200228.parquet"
MANIFEST_PATH = OUTPUT_DIR / "constituent_history_manifest.json"

DOWNLOAD_START = "2014-11-01"
DOWNLOAD_END = "2020-02-28"
RESEARCH_START = pd.Timestamp("2015-01-05")
RESEARCH_END = pd.Timestamp(DOWNLOAD_END)


def select_required_symbols(membership: pd.DataFrame) -> list[str]:
    required = {"membership_date", "index_code", "symbol"}
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise ValueError(f"点时成分文件缺少字段：{missing}")
    work = membership[list(required)].copy()
    work["membership_date"] = pd.to_datetime(
        work["membership_date"], errors="coerce"
    ).dt.normalize()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    work = work.loc[
        work["index_code"].astype(str).eq("000300")
        & work["membership_date"].between(RESEARCH_START, RESEARCH_END)
    ]
    if work.empty or work[["membership_date", "symbol"]].isna().any().any():
        raise ValueError("研究区间内没有合法的点时成分")
    return sorted(work["symbol"].astype(str).unique())


def validate_downloaded_panel(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "date",
        "con_code",
        "pre_close",
        "raw_close",
        "total_return_close",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"下载面板缺少字段：{missing}")
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["con_code"] = work["con_code"].astype("string").str.strip().str.upper()
    if work[["date", "con_code"]].isna().any().any():
        raise ValueError("下载面板含非法日期或证券代码")
    if work.duplicated(["date", "con_code"]).any():
        raise ValueError("下载面板存在证券日期重复")
    numeric = work[["pre_close", "raw_close", "total_return_close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric <= 0).any().any():
        raise ValueError("下载面板价格含缺失、非有限值、零值或负值")

    member = membership[["membership_date", "index_code", "symbol"]].copy()
    member["membership_date"] = pd.to_datetime(
        member["membership_date"], errors="coerce"
    ).dt.normalize()
    member["symbol"] = member["symbol"].astype("string").str.strip().str.upper()
    member = member.loc[
        member["index_code"].astype(str).eq("000300")
        & member["membership_date"].between(RESEARCH_START, RESEARCH_END)
    ]
    calendar = pd.DatetimeIndex(sorted(member["membership_date"].unique()))
    prices = (
        work.loc[work["date"].le(RESEARCH_END), ["date", "con_code", "total_return_close"]]
        .pivot(index="date", columns="con_code", values="total_return_close")
        .reindex(calendar)
        .ffill()
    )
    long_prices = (
        prices.rename_axis("membership_date")
        .reset_index()
        .melt(
            id_vars="membership_date",
            var_name="symbol",
            value_name="price",
        )
    )
    checked = member.merge(
        long_prices,
        on=["membership_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    missing_member_days = int(checked["price"].isna().sum())
    daily_valid = checked.groupby("membership_date")["price"].count()
    full_dates = daily_valid.index[daily_valid.eq(300)]
    return {
        "row_count": int(len(work)),
        "symbol_count": int(work["con_code"].nunique()),
        "first_date": work["date"].min().date().isoformat(),
        "last_date": work["date"].max().date().isoformat(),
        "validated_member_day_count": int(len(checked)),
        "missing_member_day_count_after_suspension_forward_fill": missing_member_days,
        "date_level_no_view_required": missing_member_days > 0,
        "first_full_300_member_price_date": (
            full_dates.min().date().isoformat() if len(full_dates) else None
        ),
    }


def _select_api() -> tuple[object, str]:
    secret, preferred = credentials()
    ts.set_token(secret)
    alternate = (
        "https://tt.xiaodefa.cn"
        if preferred == "https://fast.xiaodefa.cn"
        else "https://fast.xiaodefa.cn"
    )
    failures: list[str] = []
    for endpoint in dict.fromkeys([preferred, alternate]):
        api = ts.pro_api()
        api._DataApi__http_url = endpoint
        try:
            probe = api.trade_cal(
                exchange="SSE",
                start_date="20150105",
                end_date="20150105",
                fields="exchange,cal_date,is_open",
            )
            if probe is not None and not probe.empty:
                return api, endpoint
        except Exception as exc:
            failures.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"授权行情节点不可用：{failures}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="并发请求数；默认 2，避免触发供应商分钟频率冷却",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers 必须位于 1 至 4")
    if not MEMBERSHIP_PATH.is_file():
        raise FileNotFoundError(f"点时成分输入不存在：{MEMBERSHIP_PATH}")
    membership = pd.read_parquet(MEMBERSHIP_PATH)
    symbols = select_required_symbols(membership)
    api, endpoint = _select_api()
    print(f"需补齐 {len(symbols)} 只历史成分股；已有断点自动复用。", flush=True)

    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_daily,
                api,
                symbol,
                DOWNLOAD_START,
                DOWNLOAD_END,
                CHECKPOINT_DIR,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = f"{type(exc).__name__}: {exc}"
            if completed % 40 == 0 or completed == len(futures):
                print(
                    f"历史成分进度 {completed}/{len(futures)}，"
                    f"成功 {len(results)}，失败 {len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"仍有 {len(failures)} 只下载失败："
            f"{dict(list(failures.items())[:20])}"
        )

    panel = (
        pd.concat(results.values(), ignore_index=True)
        .sort_values(["date", "con_code"], kind="stable")
        .reset_index(drop=True)
    )
    metrics = validate_downloaded_panel(panel, membership)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".parquet.tmp")
    panel.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT_PATH)
    manifest = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "dataset_id": "ORIGINAL_ROUTE_G2_CONSTITUENT_HISTORY_20141101_20200228",
        "status": "PASS_PROVIDER_HISTORY_WITH_DATE_LEVEL_NO_VIEW",
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "download_start": DOWNLOAD_START,
        "download_end": DOWNLOAD_END,
        "provider_api": "tushare.daily",
        "provider_endpoint": endpoint,
        "total_return_method": "RAW_CLOSE_LINKED_BY_PROVIDER_PRE_CLOSE",
        "suspension_rule": "FORWARD_FILL_LAST_TOTAL_RETURN_CLOSE_ON_MARKET_SESSION",
        "membership_path": MEMBERSHIP_PATH.relative_to(ROOT).as_posix(),
        "membership_sha256": sha256_file(MEMBERSHIP_PATH),
        "output_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(OUTPUT_PATH),
        "output_bytes": OUTPUT_PATH.stat().st_size,
        "metrics": metrics,
        "credential_persisted": False,
    }
    MANIFEST_PATH.write_text(
        strict_json_text(manifest, pretty=True) + "\n", encoding="utf-8"
    )
    print(strict_json_text(manifest, pretty=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
