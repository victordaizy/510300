"""核对保存的第六轮连接、原收益恒等式和预期量纲，不重跑预测模型。"""
from pathlib import Path
import json
import re
import shutil
import sys

import numpy as np
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.information_change_transmission_v6 import OUT, read, save, sha, now


def run():
    record = {"at": now(), "study_id": "510300_INFORMATION_CHANGE_TRANSMISSION_V6"}
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha(OUT / "protocol.json") == frozen["protocol_sha256"]
    receipts = json.loads((OUT / "input_receipts.json").read_text(encoding="utf-8"))
    for row in receipts["items"]:
        assert sha(OUT / "inputs" / row["name"]) == row["sha256"]
        assert sha(ROOT / row["source"]) == row["sha256"]
    record["unchanged_input_files"] = len(receipts["items"])
    panels = {}
    label_cells = 0
    for name, title, expected_rows in [("monthly.csv", "104个月", 104), ("weekly.csv", "445周", 445)]:
        parent = read(name)
        panel = pd.read_csv(OUT / "results" / (title + "_原因盈利定价与预期差完整连接.csv"))
        assert len(panel) == expected_rows and panel.origin_id.is_unique
        pd.testing.assert_frame_equal(panel[parent.columns], parent, check_dtype=False, check_exact=False, rtol=3e-13, atol=3e-13)
        future = [c for c in parent if c.startswith(("E0_", "E1_", "delay_change_", "first_reaction_"))]
        label_cells += int(parent[future].notna().sum().sum())
        input_only = pd.read_csv(OUT / "results" / (title + "_不含后续价格标签_财务仍为回顾版本.csv"))
        assert not set(future).intersection(input_only.columns)
        assert not any(c.startswith(("policy_node", "contribution_after", "original_exit")) for c in input_only)
        usable = panel[panel.expectation_surprise_usable]
        cut = pd.to_datetime(usable.snapshot_at, utc=True)
        assert (pd.to_datetime(usable.expectation_available_at_upper_bound, utc=True) <= cut).all()
        assert (pd.to_datetime(usable.available_at_upper_bound, utc=True) <= cut).all()
        # 日历上的预计公布日不替换实际数据发布时间；报告日期按当天结束计。
        forecast_upper = pd.to_datetime(usable.expectation_forecast_date).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        assert (forecast_upper <= pd.to_datetime(usable.expectation_published_at, utc=True)).all()
        np.testing.assert_allclose(usable.expectation_expected_spread_proxy_pp, usable.expectation_expected_m1_pp - usable.expectation_expected_m2_pp, atol=1e-12)
        np.testing.assert_allclose(usable.expectation_spread_surprise_proxy_pp, usable.m1_yoy_pp - usable.m2_yoy_pp - usable.expectation_expected_spread_proxy_pp, atol=1e-12)
        panels[name] = panel
    record["preserved_original_origins"] = 549
    record["preserved_nonempty_outcome_cells"] = label_cells
    monthly = panels["monthly.csv"].set_index("stat_month")
    assert not monthly.loc["2026-08", "expectation_surprise_usable"]
    assert monthly.loc[["2020-06", "2025-07"], "expectation_spread_surprise_proxy_pp"].isna().all()
    full_window = pd.read_csv(OUT / "results/两个固定历史段_完整28个月_新信息与传导.csv")
    expected = list(pd.period_range("2020-04", "2021-02", freq="M").astype(str)) + list(pd.period_range("2024-04", "2025-08", freq="M").astype(str))
    assert full_window.stat_month.tolist() == expected
    record["complete_fixed_context_months"] = len(full_window)
    paths = pd.read_csv(OUT / "results/原E0二十日_全部逐日贡献与波动变化.csv")
    splits = pd.read_csv(OUT / "results/原E0二十日_已记录后续政策与路径分段.csv").set_index("stat_month")
    market = read("market.csv").set_index("date")
    max_error = 0.0
    for month, group in paths.groupby("stat_month", sort=False):
        assert group.day_number.tolist() == list(range(1, 21))
        original = monthly.loc[month]
        assert group.date.iloc[0] == original.E0_20_entry_date
        assert group.date.iloc[-1] == original.E0_20_exit_date
        quotes = market.loc[group.date]
        entry_open = float(quotes.open.iloc[0])
        dividends = quotes.dividend.to_numpy(dtype=float).copy()
        dividends[0] = 0.0
        eq = (quotes.close.to_numpy(dtype=float) + dividends.cumsum()) / entry_open - 1
        daily_contributions = np.diff(np.r_[0.0, eq])
        gaps = np.r_[0.0, (quotes.open.to_numpy()[1:] - quotes.close.to_numpy()[:-1] + dividends[1:]) / entry_open]
        intraday = (quotes.close.to_numpy() - quotes.open.to_numpy()) / entry_open
        for saved, independently_computed in [(group.path_return, eq), (group.daily_contribution_to_entry_return, daily_contributions), (group.overnight_contribution_to_entry_return, gaps), (group.intraday_contribution_to_entry_return, intraday)]:
            np.testing.assert_allclose(saved, independently_computed, atol=3e-13, rtol=3e-13)
        err = abs(daily_contributions.sum() - original.E0_20_return)
        assert err < 3e-13
        max_error = max(max_error, err)
    record["original_20_session_paths"] = len(splits)
    record["independently_recomputed_daily_rows"] = len(paths)
    record["maximum_original_return_identity_error"] = max_error
    assert len(splits) == 103 and len(paths) == 2060
    policy = read("policy_nodes.csv")
    policy["known"] = pd.to_datetime(policy.source_available_upper, utc=True)
    links = pd.read_csv(OUT / "results/原E0二十日_全部后续政策节点.csv")
    for month, summary in splits.iterrows():
        cutoff = pd.Timestamp(monthly.loc[month, "snapshot_at"])
        finish = pd.Timestamp(summary.original_exit_date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        expected_nodes = policy[(policy.known > cutoff) & (policy.known <= finish)]
        actual = links[links.stat_month.eq(month)]
        assert set(actual.node_id) == set(expected_nodes.node_id)
        assert len(actual) == summary.recorded_policy_node_count
        if len(expected_nodes):
            first = expected_nodes.known.min()
            assert set(summary.first_node_ids_same_clock.split(";")) == set(expected_nodes.loc[expected_nodes.known.eq(first), "node_id"])
            path = paths[paths.stat_month.eq(month)]
            times = pd.to_datetime(path.date).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
            before = path[times < first]
            if len(before):
                pre = float(before.path_return.iloc[-1])
                assert summary.split_close_date == before.date.iloc[-1]
                np.testing.assert_allclose([summary.contribution_before_node, summary.contribution_after_boundary_to_exit], [pre, float(path.path_return.iloc[-1]) - pre], atol=3e-13)
    record["policy_catalog_nodes"] = len(policy)
    record["post_origin_policy_links"] = len(links)
    record["policy_links_are_causal_estimates"] = False
    states = pd.read_csv(OUT / "results/原E0二十日_每日收盘已经公开的货币状态.csv")
    assert len(states) == len(paths)
    original = read("monthly.csv")[["stat_month", "available_at_upper_bound", "m1_yoy_pp", "m2_yoy_pp"]].copy()
    original["clock"] = pd.to_datetime(original.available_at_upper_bound, utc=True)
    original = original.sort_values("clock").reset_index(drop=True)
    available = original.clock.astype("int64").to_numpy()
    closes = pd.to_datetime(states.close_information_cutoff, utc=True).astype("int64").to_numpy()
    indices = np.searchsorted(available, closes, side="right") - 1
    assert (indices >= 0).all()
    expected = original.iloc[indices].reset_index(drop=True)
    assert states.latest_published_stat_month.tolist() == expected.stat_month.tolist()
    np.testing.assert_allclose(states[["m1_yoy_pp", "m2_yoy_pp"]], expected[["m1_yoy_pp", "m2_yoy_pp"]], atol=1e-12)
    record["daily_publication_clock_rows_verified"] = len(states)
    source_records = json.loads((OUT / "sources/三个既定病例的事前报告.json").read_text(encoding="utf-8"))
    credit = pd.read_csv(OUT / "results/三例同一调查_贷款社融预期与实际.csv")
    assert len(credit) == 6
    for source in source_records:
        pdf = OUT / "sources" / source["filename"]
        assert sha(pdf) == source["sha256"]
        with pdfplumber.open(pdf) as doc:
            text = doc.pages[source["M1"]["page"] - 1].extract_text()
        china = text.rsplit("China", 1)[-1]
        for metric, label in [("人民币贷款新增", "New Yuan Loans CNY"), ("社会融资增量", "Aggregate Financing CNY")]:
            row = credit[(credit.stat_month == source["stat_month"]) & (credit.metric == metric)].iloc[0]
            line = next(s for s in china.splitlines() if label in s)
            first_billion = float(re.findall(r"([\d,.]+)b", line)[0].replace(",", ""))
            assert row.forecast_yi == first_billion * 10
            assert row.actual_minus_forecast_yi == row.actual_yi - row.forecast_yi
            assert row.difference_lower_yi == row.actual_minus_forecast_yi - row.combined_display_rounding_halfwidth_yi
            assert row.difference_upper_yi == row.actual_minus_forecast_yi + row.combined_display_rounding_halfwidth_yi
            if metric == "人民币贷款新增":
                assert row.actual_yi == monthly.loc[row.stat_month, "loan_rmb_total_ytd_yi"]
    assert not credit[(credit.metric == "社会融资增量") & credit.stat_month.isin(["2021-01", "2024-08"])].visible_using_this_actual_source_at_original_snapshot.any()
    record["same_source_credit_forecasts_verified"] = len(credit)
    verdict = read("prior_verdict.json")
    assert verdict[0]["status"] == "REJECTED_FROZEN_NO_RELIABLE_MESSAGE_INCREMENT"
    assert verdict[1]["status"] == "NOT_RUN_SAMPLE_GATE"
    record.update(status="PASS_SAVED_JOIN_AND_PATH_RECOMPUTATION", new_models=0, new_accounts=0, historical_first_vintages_fully_verified=False, independent_validation=False, goal_achieved=False)
    save("verification.json", record)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
