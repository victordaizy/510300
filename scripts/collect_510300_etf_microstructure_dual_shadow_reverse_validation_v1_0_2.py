"""V1.0.2机械修正：净值硬门只约束入模字段与信号交易日。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.collect_510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_1 as correction_v101


base = correction_v101.base
PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_REVERSE_VALIDATION_V1_0_2"
CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2.yaml"
)
FREEZE_MANIFEST = CONFIG_FILE.with_name(
    "510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2_manifest.json"
)


def _collect_nav_corrected(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dates: list[pd.Timestamp],
    close_by_date: dict[pd.Timestamp, float],
    raw_dir: Path,
    workers: int,
    tolerance: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    east_rows, east_receipts, east_pages = (
        correction_v101._collect_paginated_nav_corrected(
            kind="nav_eastmoney",
            url=base.EASTMONEY_NAV_URL,
            raw_dir=raw_dir,
            start=start,
            end=end,
            workers=workers,
        )
    )
    sina_rows, sina_receipts, sina_pages = (
        correction_v101._collect_paginated_nav_corrected(
            kind="nav_sina",
            url=base.SINA_NAV_URL,
            raw_dir=raw_dir,
            start=start,
            end=end,
            workers=workers,
        )
    )
    east = pd.DataFrame(east_rows).rename(
        columns={
            "FSRQ": "date",
            "DWJZ": "nav_eastmoney",
            "LJJZ": "acc_nav_eastmoney",
        }
    )
    sina = pd.DataFrame(sina_rows).rename(
        columns={"fbrq": "date", "jjjz": "nav_sina", "ljjz": "acc_nav_sina"}
    )
    east = east[["date", "nav_eastmoney", "acc_nav_eastmoney"]].copy()
    sina = sina[["date", "nav_sina", "acc_nav_sina"]].copy()
    for frame, columns in [
        (east, ["nav_eastmoney", "acc_nav_eastmoney"]),
        (sina, ["nav_sina", "acc_nav_sina"]),
    ]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["date"].duplicated().any():
            raise ValueError("净值来源存在重复日期")
    merged = east.merge(
        sina,
        on="date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        bad = merged.loc[~merged["_merge"].eq("both"), ["date", "_merge"]]
        raise ValueError(f"两来源净值日期不一致：{bad.head(20).to_dict('records')}")
    unit_difference = (merged["nav_eastmoney"] - merged["nav_sina"]).abs()
    accumulated_difference = (
        merged["acc_nav_eastmoney"] - merged["acc_nav_sina"]
    ).abs()
    if (unit_difference > tolerance).any():
        bad = merged.loc[
            unit_difference.gt(tolerance),
            ["date", "nav_eastmoney", "nav_sina"],
        ].copy()
        bad["difference"] = unit_difference.loc[bad.index]
        raise ValueError(f"两来源单位净值差异超过冻结容差：{bad.head(20).to_dict('records')}")

    expected = pd.DatetimeIndex(dates)
    available = pd.DatetimeIndex(merged["date"])
    missing = expected.difference(available)
    if len(missing):
        raise ValueError(f"交易日净值缺失：{missing.strftime('%Y-%m-%d').tolist()[:20]}")
    signal_mask = merged["date"].isin(expected)
    if (accumulated_difference.loc[signal_mask] > tolerance).any():
        bad = merged.loc[
            signal_mask & accumulated_difference.gt(tolerance),
            ["date", "acc_nav_eastmoney", "acc_nav_sina"],
        ].copy()
        bad["difference"] = accumulated_difference.loc[bad.index]
        raise ValueError(
            f"信号交易日两来源累计净值差异超过冻结容差：{bad.head(20).to_dict('records')}"
        )
    non_signal_anomaly_mask = (~signal_mask) & accumulated_difference.gt(tolerance)
    non_signal_anomalies: list[dict[str, Any]] = []
    for index, row in merged.loc[non_signal_anomaly_mask].iterrows():
        non_signal_anomalies.append(
            {
                "date": row["date"].date().isoformat(),
                "acc_nav_eastmoney": float(row["acc_nav_eastmoney"]),
                "acc_nav_sina": float(row["acc_nav_sina"]),
                "absolute_difference": float(accumulated_difference.loc[index]),
                "candidate_input": False,
                "signal_trading_day": False,
            }
        )

    output = merged.loc[signal_mask].copy()
    output["unit_nav"] = output["nav_eastmoney"]
    output["accumulated_nav"] = output["acc_nav_eastmoney"]
    output["close"] = output["date"].map(close_by_date)
    output["close_premium_to_nav"] = output["close"] / output["unit_nav"] - 1.0
    output["close_premium_bps"] = output["close_premium_to_nav"] * 10_000.0
    output["symbol"] = "510300.SH"
    output["source_primary"] = "eastmoney.f10.lsjz"
    output["source_secondary"] = "sina.CaihuiFundInfoService.getNav"
    output.sort_values("date", inplace=True)
    output.reset_index(drop=True, inplace=True)
    audit = {
        "eastmoney_source_rows": int(len(east)),
        "sina_source_rows": int(len(sina)),
        "signal_calendar_rows": int(len(output)),
        "eastmoney_pages": east_pages,
        "sina_pages": sina_pages,
        "maximum_unit_nav_absolute_difference_all_source_dates": float(
            unit_difference.max()
        ),
        "maximum_accumulated_nav_absolute_difference_signal_dates": float(
            accumulated_difference.loc[signal_mask].max()
        ),
        "maximum_accumulated_nav_absolute_difference_all_source_dates": float(
            accumulated_difference.max()
        ),
        "non_trading_or_extra_source_dates": int(len(available.difference(expected))),
        "non_signal_accumulated_nav_anomalies": non_signal_anomalies,
        "candidate_input_uses_unit_nav_only": True,
    }
    print(
        f"净值采集完成：交易日{len(output)}行，单位净值最大差异{unit_difference.max():.8f}，"
        f"非信号日累计净值异常{len(non_signal_anomalies)}条。",
        flush=True,
    )
    return output, [*east_receipts, *sina_receipts], audit


def main() -> int:
    base.PROJECT_ID = PROJECT_ID
    base.CONFIG_FILE = CONFIG_FILE
    base.FREEZE_MANIFEST = FREEZE_MANIFEST
    base.__file__ = __file__
    base._collect_paginated_nav = correction_v101._collect_paginated_nav_corrected
    base._collect_nav = _collect_nav_corrected
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
