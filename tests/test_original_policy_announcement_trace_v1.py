"""来源日期、实际利率及重复宣布的因果边界。"""
import numpy as np
import pandas as pd
import pytest

from research import original_policy_announcement_trace_inputs_v1 as inputs


def test_reprint_publication_is_not_embedded_decision_date():
    raw = '<html><title>央行决定</title><p class="artlabel">新华社 2019-01-05</p><article class="bdycont">人民银行4日决定降准。</article></html>'.encode()
    result = inputs.source_document(raw)
    assert result["source_available_upper"] == inputs.clock("2019-01-05 23:59:59")
    assert "4日" in result["text"]


def test_actual_operation_rate_and_prior_unknown():
    raw = '<html><title>公开市场交易公告</title><p>2017-03-16 09:46:29</p><div id="zoom"><table><tr><td>7天</td><td>800亿元</td><td>2. 45 %</td></tr></table></div></html>'.encode()
    record = {"published_at": "2017-03-16 09:46:29", "notice_date": "2017-03-16", "seven_day_rate_percent": 2.45, "previous_rate_percent": np.nan}
    result = inputs.observed_rate(raw, record)
    assert np.isnan(result["change_vs_previous_observed_bp"])
    assert result["change_status"] == "PRIOR_OPERATION_RATE_UNKNOWN"
    record["seven_day_rate_percent"] = 2.35
    with pytest.raises(ValueError, match="7天中标利率"):
        inputs.observed_rate(raw, record)


def test_fx_and_assessment_are_not_domestic_rrr_cut():
    assert inputs.navigation_family("外汇存款准备金率上调") == "外汇存款准备金"
    assert inputs.navigation_family("2016年度支持三农考核") == "定向准备金年度考核"
    assert inputs.navigation_family("存款准备金考核制度由时点法改平均法") == "准备金考核制度"


def calendar():
    return pd.DataFrame({"date": pd.to_datetime(["2024-09-23", "2024-09-24", "2024-09-25"]),
        "decision_time": ["2024-09-23 16:00:00", "2024-09-24 16:00:00", "2024-09-25 16:00:00"]})


def node_table():
    return pd.DataFrame([
        {"node_id": "R01", "source_available_upper": "2024-09-24 09:19:36", "common_source_id": "直播", "economic_identity": "降准降息计划", "information_role": "ANNOUNCEMENT"},
        {"node_id": "C01", "source_available_upper": "2024-09-24 11:42:50", "common_source_id": "直播", "economic_identity": "股票工具计划", "information_role": "ANNOUNCEMENT"},
        {"node_id": "R02", "source_available_upper": "2024-09-25 23:59:59", "common_source_id": "实施转载", "economic_identity": "降准降息计划", "information_role": "IMPLEMENTATION_CONFIRMATION"},
    ])


def test_common_source_is_not_independent_three_votes():
    data = inputs.daily_known(calendar(), node_table())
    day = data.iloc[1]
    assert day.new_recorded_nodes == 2
    assert day.new_unique_common_sources == 1
    assert day.new_recorded_announcement_nodes == 2
    assert not day.empty_record_means_policy_absent
    assert data.iloc[2].new_recorded_nodes == 0


def test_future_document_cannot_change_previous_observations():
    data, nodes = calendar(), node_table()
    full = inputs.daily_known(data, nodes)
    prefix = inputs.daily_known(data.iloc[:2], nodes.iloc[:2])
    pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True), prefix, check_exact=True)


def test_implementation_is_not_new_announcement():
    data, nodes = calendar(), node_table()
    nodes.loc[2, "source_available_upper"] = "2024-09-25 09:20:30"
    result = inputs.daily_known(data, nodes).iloc[2]
    assert result.new_recorded_nodes == 1
    assert result.new_recorded_announcement_nodes == 0
