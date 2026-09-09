from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.stress_transmission_hazard_v2_g1_historical_remediation_v1 import (
    combine_official_intervals,
    parse_sse_official_response,
    parse_szse_official_document,
    remediate_official_suspensions,
)


def _parent_rows() -> pd.DataFrame:
    rows = []
    for symbol in ("000001.SZ", "600000.SH"):
        for date, observed, close, state, usable, ret in (
            ("2020-01-02", True, 10.0, "TRADED_VALID", True, 0.0),
            ("2020-01-03", False, np.nan, "SUPPLIER_MISSING_OR_CONFLICT", False, np.nan),
            ("2020-01-06", True, 10.2, "TRADED_VALID", True, 0.02),
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "previous_unadjusted_close": 10.0 if observed else np.nan,
                    "unadjusted_close": close,
                    "source_observed": observed,
                    "supplier_conflict": not observed,
                    "official_suspension": False,
                    "suspension_evidence_id": "",
                    "corporate_action_status": "NONE_CONFIRMED" if observed else "UNRESOLVED",
                    "corporate_action_evidence_id": "NO_ACTION" if observed else "MISSING",
                    "cash_distribution_per_pre_event_share": 0.0 if observed else np.nan,
                    "post_to_pre_share_ratio": 1.0 if observed else np.nan,
                    "subscription_cash_outflow_per_pre_event_share": 0.0 if observed else np.nan,
                    "constituent_return_state": state,
                    "daily_total_shareholder_return": ret,
                    "return_is_usable": usable,
                    "state_reason": (
                        "UNADJUSTED_PRICE_PLUS_RESOLVED_ACTION_LEDGER"
                        if observed
                        else "SUPPLIER_CONFLICT_FAIL_CLOSED"
                    ),
                    "daily_source": "TEST",
                    "previous_observed_date": pd.NaT,
                    "daily_pre_close": np.nan,
                    "action_reference_gap_cny": np.nan,
                    "action_resolution_reason": "TEST",
                }
            )
    return pd.DataFrame(rows)


def _membership() -> pd.DataFrame:
    # 核心函数要求每日300只；测试用无关占位证券补齐，且它们都有父行。
    parent = _parent_rows()
    dates = sorted(parent["date"].unique())
    symbols = ["000001.SZ", "600000.SH"] + [f"{i:06d}.SZ" for i in range(100000, 100298)]
    members = pd.DataFrame(
        [
            {"membership_date": date, "index_code": "000300", "symbol": symbol}
            for date in dates
            for symbol in symbols
        ]
    )
    extras = []
    for date in dates:
        for symbol in symbols[2:]:
            extras.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "previous_unadjusted_close": 10.0,
                    "unadjusted_close": 10.0,
                    "source_observed": True,
                    "supplier_conflict": False,
                    "official_suspension": False,
                    "suspension_evidence_id": "",
                    "corporate_action_status": "NONE_CONFIRMED",
                    "corporate_action_evidence_id": "NO_ACTION",
                    "cash_distribution_per_pre_event_share": 0.0,
                    "post_to_pre_share_ratio": 1.0,
                    "subscription_cash_outflow_per_pre_event_share": 0.0,
                    "constituent_return_state": "TRADED_VALID",
                    "daily_total_shareholder_return": 0.0,
                    "return_is_usable": True,
                    "state_reason": "UNADJUSTED_PRICE_PLUS_RESOLVED_ACTION_LEDGER",
                    "daily_source": "TEST",
                    "previous_observed_date": pd.NaT,
                    "daily_pre_close": 10.0,
                    "action_reference_gap_cny": 0.0,
                    "action_resolution_reason": "TEST",
                }
            )
    return members, pd.concat([parent, pd.DataFrame(extras)], ignore_index=True)


def test_sse_parser_preserves_official_interval() -> None:
    payload = {
        "result": [
            {
                "productCode": "600000",
                "startStopDate": "20200103",
                "endStopDate": "20200106",
                "stopTime": "",
                "stopReason": "重要事项未公告",
                "endStopReason": "刊登重要公告",
                "controlType": "TR",
                "type": "LXTP",
            }
        ]
    }
    frame = parse_sse_official_response(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        source_url="https://query.sse.com.cn/commonSoaQuery.do?test=1",
        retrieved_at="2026-09-03T12:00:00+08:00",
    )
    assert frame.loc[0, "symbol"] == "600000.SH"
    assert frame.loc[0, "start_at"] == pd.Timestamp("2020-01-03 09:30")
    assert frame.loc[0, "resume_at"] == pd.Timestamp("2020-01-06 09:30")
    assert bool(frame.loc[0, "full_day_scope"])


def test_sse_parser_accepts_same_day_wh_but_rejects_pm_as_full_day() -> None:
    payload = {
        "result": [
            {
                "productCode": "600000",
                "startStopDate": "20200103",
                "endStopDate": "20200103",
                "stopTime": "WH",
                "stopReason": "整日停牌",
                "endStopReason": "整日停牌",
                "type": "LSTP",
            },
            {
                "productCode": "600001",
                "startStopDate": "20200103",
                "endStopDate": "20200103",
                "stopTime": "PM",
                "stopReason": "午间停牌",
                "endStopReason": "午间停牌",
                "type": "LSTP",
            },
        ]
    }
    frame = parse_sse_official_response(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        source_url="https://query.sse.com.cn/commonSoaQuery.do?test=same-day",
        retrieved_at="2026-09-03T12:00:00+08:00",
    ).set_index("symbol")
    assert frame.loc["600000.SH", "resume_at"] == pd.Timestamp("2020-01-03 15:00")
    assert bool(frame.loc["600000.SH", "full_day_scope"])
    assert frame.loc["600001.SH", "start_at"] == pd.Timestamp("2020-01-03 13:00")
    assert frame.loc["600001.SH", "resume_at"] == pd.Timestamp("2020-01-03 15:00")
    assert not bool(frame.loc["600001.SH", "full_day_scope"])


def test_sse_parser_excludes_non_equity_products_before_date_validation() -> None:
    payload = {
        "result": [
            {
                "productCode": "191016",
                "startStopDate": "20180522",
                "endStopDate": "20180521",
                "stopTime": "",
                "stopReason": "重要公告",
                "endStopReason": "重要公告",
                "type": "LXTP",
            },
            {
                "productCode": "600000",
                "startStopDate": "20180522",
                "endStopDate": "20180522",
                "stopTime": "WH",
                "stopReason": "整日停牌",
                "endStopReason": "整日停牌",
                "type": "LSTP",
            },
        ]
    }
    frame = parse_sse_official_response(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        source_url="https://query.sse.com.cn/commonSoaQuery.do?test=product-scope",
        retrieved_at="2026-09-03T12:00:00+08:00",
    )
    assert frame["symbol"].tolist() == ["600000.SH"]


def test_szse_parser_preserves_leading_zero_and_open_interval() -> None:
    html = """
    <html><body><h1>证券停牌情况</h1><table>
      <tr><th>代码</th><th>证券简称</th><th>停牌原因</th><th>停牌时间</th><th>复牌时间</th></tr>
      <tr><td>000001</td><td>平安银行</td><td>重大事项</td><td>2020/01/03 09:30</td><td>9999/12/31 00:00</td></tr>
    </table></body></html>
    """.encode("gb18030")
    frame = parse_szse_official_document(
        html,
        source_url="https://docs.static.szse.cn/test.html",
        report_month="2020-01",
        retrieved_at="2026-09-03T12:00:00+08:00",
    )
    assert frame.loc[0, "symbol"] == "000001.SZ"
    assert bool(frame.loc[0, "open_ended"])
    assert pd.isna(frame.loc[0, "resume_at"])


def test_full_scope_match_promotes_only_official_rows() -> None:
    membership, parent = _membership()
    sse = parse_sse_official_response(
        json.dumps(
            {
                "result": [
                    {
                        "productCode": "600000",
                        "startStopDate": "20200103",
                        "endStopDate": "20200106",
                        "stopTime": "",
                        "stopReason": "测试",
                        "endStopReason": "复牌",
                    }
                ]
            }
        ).encode(),
        source_url="https://query.sse.com.cn/commonSoaQuery.do?test=1",
        retrieved_at="2026-09-03T12:00:00+08:00",
    )
    szse = parse_szse_official_document(
        """<h1>证券停牌情况</h1><table><tr><th>代码</th><th>简称</th><th>原因</th><th>停牌</th><th>复牌</th></tr>
        <tr><td>000001</td><td>平安</td><td>测试</td><td>2020/01/03 09:30</td><td>2020/01/06 09:30</td></tr></table>""".encode("gb18030"),
        source_url="https://docs.static.szse.cn/test.html",
        report_month="2020-01",
        retrieved_at="2026-09-03T12:00:00+08:00",
    )
    result = remediate_official_suspensions(
        classified_returns=parent,
        membership=membership,
        dividend_actions=pd.DataFrame(columns=["ts_code", "ex_date"]),
        official_intervals=combine_official_intervals([sse, szse]),
        dividend_source_sha256="a" * 64,
        historical_cutoff="2020-01-06",
        expected_target_rows=2,
    )
    promoted = result.remediated_classified_returns.loc[
        result.remediated_classified_returns["date"].eq(pd.Timestamp("2020-01-03"))
        & result.remediated_classified_returns["symbol"].isin(["000001.SZ", "600000.SH"])
    ]
    assert promoted["constituent_return_state"].eq("OFFICIAL_SUSPENSION").all()
    assert promoted["daily_total_shareholder_return"].eq(0.0).all()
    assert result.metrics["target_member_day_count"] == 2
    assert result.metrics["promoted_official_suspension_member_day_count"] == 2
    assert not result.target_evidence["event_or_label_used_for_selection"].any()


def test_action_candidate_blocks_zero_return_promotion() -> None:
    membership, parent = _membership()
    szse = parse_szse_official_document(
        """<h1>证券停牌情况</h1><table><tr><th>代码</th><th>简称</th><th>原因</th><th>停牌</th><th>复牌</th></tr>
        <tr><td>000001</td><td>平安</td><td>测试</td><td>2020/01/03 09:30</td><td>2020/01/06 09:30</td></tr></table>""".encode("gb18030"),
        source_url="https://docs.static.szse.cn/test.html",
        report_month="2020-01",
        retrieved_at="2026-09-03T12:00:00+08:00",
    )
    result = remediate_official_suspensions(
        classified_returns=parent,
        membership=membership,
        dividend_actions=pd.DataFrame([{"ts_code": "000001.SZ", "ex_date": "20200103"}]),
        official_intervals=szse,
        dividend_source_sha256="b" * 64,
        historical_cutoff="2020-01-06",
        expected_target_rows=2,
    )
    row = result.remediated_classified_returns.loc[
        result.remediated_classified_returns["date"].eq(pd.Timestamp("2020-01-03"))
        & result.remediated_classified_returns["symbol"].eq("000001.SZ")
    ].iloc[0]
    assert row["constituent_return_state"] == "SUPPLIER_MISSING_OR_CONFLICT"
    assert pd.isna(row["daily_total_shareholder_return"])
    evidence = result.target_evidence.loc[
        result.target_evidence["symbol"].eq("000001.SZ")
    ].iloc[0]
    assert evidence["block_reason"] == "DIVIDEND_ACTION_CANDIDATE_REQUIRES_SEPARATE_RECONCILIATION"
