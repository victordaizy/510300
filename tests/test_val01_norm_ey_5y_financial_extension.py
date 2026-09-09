"""VAL01_NORM_EY_5Y 财务历史扩展专项测试。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

from research.val01_norm_ey_5y_financial_extension import (
    ROOT,
    build_extension_events,
    build_quarter_periods,
    classify_failure,
    combine_financial_archives,
    find_continuity_gaps,
    load_config,
    load_target_universe,
    resolve_transport,
    sanitized_transport,
    validate_api_response,
    validate_api_history_response,
    verify_frozen_inputs,
)


def test_协议固定13个季度且禁止收益_ic_交易(tmp_path: Path) -> None:
    config = load_config()
    periods = build_quarter_periods(pd.Timestamp("2012-06-30"), pd.Timestamp("2015-06-30"))
    assert len(periods) == 13
    assert periods[0] == "20120630"
    assert periods[-1] == "20150630"
    for field in (
        "return_calculation_enabled",
        "ic_calculation_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
    ):
        assert config["protocol"][field] is False
    config["protocol"]["ic_calculation_enabled"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="禁止启用"):
        load_config(invalid)


def test_目标证券固定为60个月516只() -> None:
    weights, symbols = load_target_universe(load_config())
    assert weights["trade_date"].nunique() == 60
    assert len(symbols) == 516
    assert weights["trade_date"].min() == pd.Timestamp("2016-08-31")
    assert weights["trade_date"].max() == pd.Timestamp("2021-07-30")


def test_冻结原财务和132个检查点哈希一致() -> None:
    result = verify_frozen_inputs(load_config())
    assert result["status"] == "PASS"
    assert result["checkpoint_file_count"] == 132
    assert result["checkpoint_matches"] is True


def test_代理传输强制https且报告不含令牌() -> None:
    config = load_config()
    transport = resolve_transport(
        config,
        {
            "TUSHARE_PROXY_TOKEN": "secret-token",
            "TUSHARE_PROXY_API_URL": "https://secure.example.test",
            "TUSHARE_PROXY_TOKEN_EXPIRES_AT": "2099-01-01T00:00:00+08:00",
        },
    )
    public = sanitized_transport(transport)
    assert transport["token"] == "secret-token"
    assert "token" not in public
    assert public["token_saved_or_echoed"] is False
    with pytest.raises(ValueError, match="HTTPS"):
        resolve_transport(
            config,
            {
                "TUSHARE_PROXY_TOKEN": "secret-token",
                "TUSHARE_PROXY_API_URL": "http://insecure.example.test",
                "TUSHARE_PROXY_TOKEN_EXPIRES_AT": "2099-01-01T00:00:00+08:00",
            },
        )


def test_代理超速错误被归入冷却类别() -> None:
    error = RuntimeError("访问频率已超速，将触发 6 分钟冷却")
    assert classify_failure(error) == "BLOCKED_API_RATE_LIMIT_COOLDOWN"


def test_vip响应必须字段齐全且报告期一致() -> None:
    config = load_config()
    valid = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20120831"],
            "f_ann_date": ["20120831"],
            "end_date": ["20120630"],
            "report_type": ["1"],
            "revenue": [1.0],
            "n_income_attr_p": [0.1],
        }
    )
    result = validate_api_response(valid, "income_vip", "20120630", config)
    assert len(result) == 1
    invalid = valid.drop(columns="f_ann_date")
    with pytest.raises(ValueError, match="缺少字段"):
        validate_api_response(invalid, "income_vip", "20120630", config)
    wrong_period = valid.assign(end_date="20120930")
    with pytest.raises(ValueError, match="混入其他报告期"):
        validate_api_response(wrong_period, "income_vip", "20120630", config)


def test_单证券历史响应允许多报告期但拒绝无效日期() -> None:
    config = load_config()
    history = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20120831", "20130430"],
            "f_ann_date": ["20120831", "20130430"],
            "end_date": ["20120630", "20121231"],
            "report_type": ["1", "1"],
            "revenue": [1.0, 2.0],
            "n_income_attr_p": [0.1, 0.2],
        }
    )
    result = validate_api_history_response(history, "income_vip", config)
    assert len(result) == 2
    invalid = history.assign(end_date=["20120630", "not-a-date"])
    with pytest.raises(ValueError, match="无效报告期"):
        validate_api_history_response(invalid, "income_vip", config)


def test_连续性缺口只从证券首次出现后开始() -> None:
    periods = ["20120630", "20120930", "20121231", "20130331"]
    frames = {
        "20120630": pd.DataFrame({"ts_code": ["A"]}),
        "20120930": pd.DataFrame({"ts_code": []}),
        "20121231": pd.DataFrame({"ts_code": ["A", "B"]}),
        "20130331": pd.DataFrame({"ts_code": ["A", "B"]}),
    }
    anchor = pd.DataFrame({"ts_code": ["A", "B", "C"], "end_date": ["20150930"] * 3})
    gaps = find_continuity_gaps(frames, anchor, {"A", "B", "C"}, periods)
    assert gaps == [("A", "20120930")]


def _api_frames() -> dict[str, pd.DataFrame]:
    return {
        "income_vip": pd.DataFrame(
            {
                "ts_code": ["A"], "ann_date": ["20120831"], "f_ann_date": ["20120831"],
                "end_date": ["20120630"], "report_type": ["1"], "revenue": [10.0],
                "n_income_attr_p": [1.0],
            }
        ),
        "balancesheet_vip": pd.DataFrame(
            {
                "ts_code": ["A"], "ann_date": ["20120831"], "f_ann_date": ["20120831"],
                "end_date": ["20120630"], "report_type": ["1"],
                "total_hldr_eqy_exc_min_int": [5.0], "total_share": [2.0],
            }
        ),
        "fina_indicator_vip": pd.DataFrame(
            {
                "ts_code": ["A"], "ann_date": ["20120831"], "end_date": ["20120630"],
                "eps": [0.5], "dt_eps": [0.5], "bps": [2.5], "roe": [20.0],
            }
        ),
    }


def test_扩展事件点时键唯一且只含目标证券() -> None:
    events = build_extension_events(
        _api_frames(), {"A"}, datetime(2026, 8, 19, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    assert len(events) == 1
    assert events[["con_code", "report_period", "available_at"]].duplicated().sum() == 0
    assert set(events["con_code"]) == {"A"}
    assert events.loc[0, "available_at"] == pd.Timestamp("2012-08-31")


def test_合并档案拒绝报告期重叠() -> None:
    extension = build_extension_events(
        _api_frames(), {"A"}, datetime(2026, 8, 19, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    base = extension.copy()
    base["report_period"] = pd.Timestamp("2015-09-30")
    combined = combine_financial_archives(base, extension, "2015-09-30")
    assert len(combined) == 2
    overlap = extension.copy()
    overlap["report_period"] = pd.Timestamp("2015-09-30")
    with pytest.raises(ValueError, match="重叠"):
        combine_financial_archives(base, overlap, "2015-09-30")
