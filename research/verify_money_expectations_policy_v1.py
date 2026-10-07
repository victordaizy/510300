"""只读核验保存的事件收益、信息时钟和交付索引。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="只读核验本轮保存数字")
    ap.add_argument("--study-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.study_dir.resolve()
    m = pd.read_parquet(root / "inputs/market_daily.parquet").sort_values("date").reset_index(drop=True)
    m["date"] = pd.to_datetime(m.date)
    computed = (m.close + m.dividend) / m.close.shift(1) - 1
    np.testing.assert_allclose(computed.iloc[1:], m.total_simple.iloc[1:], atol=1e-12, rtol=0)
    np.testing.assert_allclose((m.wealth / m.wealth.shift(1) - 1).iloc[1:], m.total_simple.iloc[1:], atol=1e-12, rtol=0)
    verified = 0
    for name, datecol in [("事前预期差与510300全部事件.csv", "release_date"),
                          ("政策冲击与510300全部事件.csv", "AnnouncementDate"),
                          ("公布后回述预期_不并入事前样本.csv", "release_day")]:
        frame = pd.read_csv(root / "results" / name)
        for _, row in frame.iterrows():
            if pd.isna(row.get("base_date")):
                continue
            day = pd.Timestamp(row[datecol]).normalize()
            base = m.index[m.date <= day][-1]
            assert str(m.at[base, "date"].date()) == row.base_date
            for h in (1, 5, 20):
                if pd.isna(row.get(f"return_{h}d")):
                    continue
                part = m.iloc[base + 1:base + h + 1]
                assert len(part) == h and (part.date > day).all()
                # 独立使用逐日总收益乘积，不复用分析脚本的财富指数比值。
                ret = float(np.prod(1 + part.total_simple) - 1)
                np.testing.assert_allclose(ret, row[f"return_{h}d"], atol=2e-11, rtol=0)
                assert str(part.iloc[-1].date.date()) == row[f"end_{h}d"]
                verified += 1
    e = pd.read_csv(root / "results/事前预期差与510300全部事件.csv")
    assert len(e) == 6 and e.stat_month.is_unique
    assert (pd.to_datetime(e.report_date) < pd.to_datetime(e.release_date)).all()
    assert (e.definition_version == "M1_NEW2025_M2_MMF2018_ECNY").all()
    np.testing.assert_allclose(e.spread_surprise_proxy_pp,
                              (e.M1_actual_pp - e.M2_actual_pp) - (e.M1_forecast_pp - e.M2_forecast_pp), atol=1e-12)
    daily = pd.read_parquet(root / "results/510300与当时已知货币数据_全部日线.parquet")
    assert len(daily) == 2111 and daily.date.is_unique
    admitted = daily.dropna(subset=["available_at"])
    assert (admitted.available_at <= admitted.close_time).all()
    official = pd.read_csv(root / "inputs/official_releases.csv")
    raw_verified = 0
    for _, row in official.iterrows():
        raw = (root / row.raw_path_in_package).resolve()
        assert root in raw.parents and raw.is_file()
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row.source_sha256
        raw_verified += 1
    indexed = 0
    index = root / "FILE_INDEX.csv"
    if index.exists():
        table = pd.read_csv(index)
        assert table.path.is_unique
        for _, row in table.iterrows():
            p = (root / row.path).resolve()
            assert root in p.parents
            assert p.stat().st_size == row.bytes
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row.sha256
            indexed += 1
    print(json.dumps({"status": "PASS_SAVED_NUMERICS_AND_INFORMATION_CLOCK",
                      "verified_event_horizon_returns": verified,
                      "official_source_files_verified": raw_verified,
                      "daily_points": len(daily), "indexed_files_checked": indexed,
                      "network_calls": 0, "model_fits": 0, "account_runs": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
