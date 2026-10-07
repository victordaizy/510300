"""用保存的行情独立复核时间顺序、旧标签和波动窗口；不运行预测模型。"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_policy_information_timing_v9"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-10):
    assert np.allclose(a, b, atol=tol, rtol=tol, equal_nan=True), (a, b)


def verify():
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == freeze["protocol_sha256"]
    parent = pd.read_csv(OUT / "inputs/parent56.csv")
    full = pd.read_csv(OUT / "results/56个月_原调查新信息时序与价格波动.csv")
    cases = pd.read_csv(OUT / "results/七个调查偏差病例_完整时序与传导.csv")
    extension = pd.read_csv(OUT / "results/56个月_货币基数与波动窗口构成.csv").set_index("month")
    missing = pd.read_csv(OUT / "results/28个月_调查缺失原状.csv")
    mother = pd.read_csv(OUT / "inputs/survey84.csv")
    market = pd.read_csv(OUT / "inputs/market.csv")
    money = pd.read_csv(OUT / "inputs/monthly_context.csv")
    credit = pd.read_csv(OUT / "inputs/credit.csv")
    orders = pd.read_parquet(OUT / "inputs/orders.parquet")
    paths = pd.read_csv(OUT / "results/56个月_第一完整反应日至原终点的逐日路径.csv")
    pd.testing.assert_frame_equal(parent, full[parent.columns], check_dtype=False, atol=1e-12, rtol=1e-12)
    assert len(full) == 56 and len(cases) == 7 and len(paths) == 1176 and len(missing) == 28
    assert set(full.month).isdisjoint(set(missing.month)) and set(full.month) | set(missing.month) == set(mother.month)
    r = (market.close + market.dividend) / market.close.shift(1) - 1
    down = np.minimum(r, 0) ** 2
    monetary_time = pd.to_datetime(money.published_at, utc=True)
    credit_time = pd.to_datetime(credit.known_at, utc=True)
    orders_time = pd.to_datetime(orders.available_at, utc=True)
    dates = pd.DatetimeIndex(pd.to_datetime(market.date)).tz_localize("Asia/Shanghai")
    errors = []
    for row in full.to_dict("records"):
        at = pd.Timestamp(row["announcement_at"])
        review = int(np.flatnonzero(dates + pd.Timedelta(hours=9, minutes=30) > at)[0])
        entry, end = int(row["entry_i"]), int(row["exit_i"])
        assert entry == review + 1 and end == entry + 19
        assert market.iloc[review].date == row["first_reaction_date"]
        assert market.iloc[entry].date == row["entry_date"] and market.iloc[end].date == row["exit_date"]
        rebuilt = (market.iloc[end].close + market.iloc[entry + 1:end + 1].dividend.sum()) / market.iloc[entry].open - 1
        close(rebuilt, row["target20"])
        errors.append(abs(rebuilt - row["target20"]))
        close(row["first_gap_return"] + row["first_intraday_contribution"] + row["first_dividend_contribution"], row["first_total_return"])
        for frame, times, period, known_period, known_at in [
            (money, monetary_time, "stat_month", "known_money_stat_month", "known_money_published_at"),
            (credit, credit_time, "stat_month", "known_credit_stat_month", "known_credit_known_at"),
            (orders, orders_time, "reference_period", "known_orders_reference_period", "known_orders_available_at"),
        ]:
            available = frame.loc[times <= at]
            index = times.loc[available.index].idxmax()
            assert str(frame.loc[index, period]) == str(row[known_period])
            assert pd.Timestamp(row[known_at]) <= at
        ext = extension.loc[row["month"]]
        for prefix, i in [("pre", review - 1), ("review", review)]:
            recent = float(down.iloc[i - 4:i + 1].sum())
            previous = float(down.iloc[i - 9:i - 4].sum())
            exited = float(down.iloc[i - 24:i - 19].sum())
            current_variance = float(down.iloc[i - 19:i + 1].mean() * 252)
            old_variance = float(down.iloc[i - 24:i - 4].mean() * 252)
            close(ext[prefix + "_v_recent5_down_sum"], recent)
            close(ext[prefix + "_v_previous5_down_sum"], previous)
            close(ext[prefix + "_v_exited5_down_sum"], exited)
            close(ext[prefix + "_v_down2"], current_variance)
            close(ext[prefix + "_v_d5_down2"], current_variance - old_variance)
            close(current_variance - old_variance, 252 / 20 * (recent - exited))
        series = paths[paths.month == row["month"]]
        assert len(series) == 21 and series.session_from_reaction.tolist() == list(range(21))
        for point in series.to_dict("records"):
            i = review + point["session_from_reaction"]
            assert market.iloc[i].date == point["date"]
            ret = (market.iloc[i].close + market.iloc[review + 1:i + 1].dividend.sum()) / market.iloc[review].open - 1
            close(ret, point["first_open_to_close_return"])
    july = full.set_index("month").loc["2024-07"]
    survey = max(pd.Timestamp(july.survey_published_at), pd.Timestamp(july.survey_modified_at))
    assert survey < pd.Timestamp(july.intervening_notice_at) < pd.Timestamp(july.announcement_at)
    assert july.minutes_notice_to_lpr == 60
    assert july.updated_numeric_consensus_after_intervening_notice == "UNKNOWN"
    close([july.t1_surprise_min_bp, july.t5_surprise_min_bp], [-10, -10])
    source_receipts = list((OUT / "sources").glob("*.receipt.json"))
    assert len(source_receipts) == 4
    for file in source_receipts:
        rec = json.loads(file.read_text(encoding="utf-8"))
        assert digest(ROOT / rec["path"]) == rec["sha256"]
    added = pd.read_csv(OUT / "results/定向补充的窗口内后续政策.csv")
    assert added.month.tolist() == ["2023-08"]
    facts = json.loads((OUT / "results/日期价格事实.json").read_text(encoding="utf-8"))
    sep = facts["september2024_path"]
    close(sep["original_target20"], sep["before_september24_node_return"] + sep["after_september23_close_contribution_same_initial_capital"])
    for i in ["2023-06", "2023-08"]:
        item = full.set_index("month").loc[i]
        assert pd.isna(item.known_credit_corporate_long_ytd_yoy_change_yi)
        assert pd.isna(item.known_credit_household_long_ytd_yoy_change_yi)
    parent_summary = json.loads((OUT / "inputs/parent_summary.json").read_text(encoding="utf-8"))
    assert parent_summary["status"] == "EXPLORATORY_TRAINED" and parent_summary["goal_achieved"] is False
    assert parent_summary["candidate_roundtrips_across_costs"] == 0
    result = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_LABELS_CLOCKS_AND_ROLLING_IDENTITIES",
        "old_labels_unchanged": len(full), "label_max_absolute_error": max(errors), "verified_daily_paths": len(paths),
        "verified_asof_connections": len(full) * 3, "downside_windows_recomputed": len(full) * 2,
        "source_hashes_verified": len(source_receipts), "missing_surveys_preserved": len(missing),
        "old_training_failure_preserved": True, "independent_validation": False, "goal_achieved": False}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    verify()
