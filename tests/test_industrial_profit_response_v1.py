"""检查利润口径、舍入、公告后成交及未成交标签的关键边界。"""
import numpy as np
import pandas as pd
import pytest

from research.industrial_profit_response_v1 import parse_profit, build_events


def source_html(body_growth="增长1.79倍", table_growth="178.9"):
    text = f"<html><p>全国规模以上工业企业实现利润总额11140.1亿元，同比{body_growth}（按可比口径计算），比2019年增长72.1%，两年平均增长31.2%。</p><table><tr><th>分组</th><th>营业收入</th><th>同比增长</th><th>营业成本</th><th>同比增长</th><th>利润总额</th><th>同比增长</th></tr><tr><td>总计</td><td>168726.6</td><td>45.5</td><td>139909.0</td><td>43.5</td><td>11140.1</td><td>{table_growth}</td></tr></table></html>"
    return text.encode("utf-8")


def market_and_release():
    dates = pd.bdate_range("2024-01-02", periods=36)
    market = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.2,
                           "low": 9.8, "previous_close": 10., "dividend": 0.})
    release = {"stat_month": "2023-11", "release_date": "2024-01-05",
               "known_at": "2024-01-05T23:59:59+08:00", "profit_ytd_reported_yoy_pct": 4.}
    market.loc[4, "close"] = 10.1
    return market, release


def test_multiplier_rounding_keeps_original_table_and_annual_yoy():
    row = parse_profit(source_html())
    assert row["profit_ytd_reported_yoy_pct"] == 178.9
    assert row["body_yoy_pct"] == 179.
    assert parse_profit(source_html("下降4.1%", "-4.1"))["profit_ytd_reported_yoy_pct"] == -4.1


def test_conflicting_body_and_table_cannot_enter_source_panel():
    with pytest.raises(AssertionError, match="舍入"):
        parse_profit(source_html(table_growth="72.1"))


def test_day_only_release_waits_for_next_full_session_before_entry():
    market, release = market_and_release()
    row = build_events(market, [release]).iloc[0]
    assert row.observation_date == pd.Timestamp("2024-01-08")
    assert row.entry_date == pd.Timestamp("2024-01-09")
    assert row.selected and row.status == "MATURE"
    altered = market.copy()
    altered.loc[5:, "close"] = 1.
    future = build_events(altered, [release]).iloc[0]
    assert future.selected == row.selected and future.equity_response == row.equity_response


def test_unfilled_has_zero_cost_and_censored_exit_is_not_zero_return():
    market, release = market_and_release()
    market.loc[5, "open"] = 11.
    unfilled = build_events(market, [release]).iloc[0]
    assert unfilled.status == "UNFILLED_UPPER_LIMIT"
    assert unfilled.gross_return == 0. and unfilled.stress_proportional_proxy_return == 0.
    market.loc[5, "open"] = 10.
    censored = build_events(market.iloc[:10], [release]).iloc[0]
    assert censored.status == "CENSORED_EXIT_AFTER_SAMPLE" and np.isnan(censored.gross_return)
