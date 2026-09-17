"""验证免费输入适配的真实保存响应兼容和不产生无效请求的边界。"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from research import new_daily_input_adapter_v1 as adapter
from research.intraday_overnight_increment_v1 import write_json

ROOT = adapter.ROOT
SAVED = ROOT / 'reports/research/510300_post_selection_data_feasibility_v1'
ACCEPTED = ROOT / 'reports/research/510300_post_selection_extension_inputs_v1'


@pytest.fixture(scope='module')
def tables():
    return {name: adapter.parse_prices(name, (SAVED / (name + '.raw')).read_bytes(), '2026-08-01', '2026-09-11') for name in ['sina', 'tencent']}


def test_weekend_and_before_close_do_not_schedule_new_data():
    dates = pd.to_datetime(['2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15'])
    for clock in ['2026-09-13T11:00:00+08:00', '2026-09-14T15:04:59+08:00']:
        added, last, next_day = adapter.completed_dates(dates, '2026-09-11', clock)
        assert len(added) == 0 and last == '2026-09-11' and next_day == '2026-09-14'
    added, last, next_day = adapter.completed_dates(dates, '2026-09-11', '2026-09-14T15:05:00+08:00')
    assert len(added) == 1 and last == '2026-09-14' and next_day == '2026-09-15'


@pytest.mark.parametrize('name', ['sina', 'tencent'])
def test_saved_raw_parser_matches_previous_table(tables, name):
    old = pd.read_parquet(SAVED / (name + '_daily.parquet'))
    pd.testing.assert_frame_equal(tables[name], old, check_dtype=False, check_exact=True)


def test_actual_saved_prices_keep_original_rounding(tables):
    old = pd.read_parquet(ROOT / 'data/raw/market/510300_daily_downside_risk_v1.parquet')
    dates = tables['sina'].loc[tables['sina'].date.gt('2026-08-14'), 'date']
    comparison = adapter.compare_prices(tables['sina'], tables['tencent'], old, dates)
    assert len(comparison) == 180 and comparison.loc[comparison.field.eq('volume'), 'absolute_difference'].max() == 49
    assert comparison.loc[comparison.field.eq('amount'), 'absolute_difference'].max() == 48


@pytest.mark.parametrize('field, difference', [('high', .001), ('volume', 200.)])
def test_price_and_precision_failures_rejected(tables, field, difference):
    other = tables['tencent'].copy()
    other.loc[0, field] += difference
    old = pd.read_parquet(ROOT / 'data/raw/market/510300_daily_downside_risk_v1.parquet')
    dates = tables['sina'].loc[tables['sina'].date.gt('2026-08-14'), 'date']
    with pytest.raises(ValueError, match='精度差异'):
        adapter.compare_prices(tables['sina'], other, old, dates)


def test_actual_append_matches_all_previously_accepted_inputs(tables):
    old = pd.read_parquet(ROOT / 'data/raw/market/510300_daily_downside_risk_v1.parquet')
    original_features = pd.read_parquet(ROOT / 'reports/research/510300_adaptive_allocation_v1/features.parquet')
    prices, features = adapter.append_inputs(old, original_features, tables['sina'],
        pd.read_csv(ROOT / 'data/reference/510300_dividends.csv'), adapter.read(SAVED / 'sina_receipt.json')['retrieved_at'], 'sina.klc_saved_211')
    pd.testing.assert_frame_equal(prices, pd.read_parquet(ACCEPTED / 'candidate_prices.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(features, pd.read_parquet(ACCEPTED / 'candidate_features.parquet'), check_exact=True)


def test_no_new_date_never_calls_network_or_accounts(tmp_path, monkeypatch):
    settings = {'output_root': str(tmp_path / 'output'), 'initial_source': {'prices': str(ACCEPTED / 'candidate_prices.parquet')},
        'calendar': 'data/reference/sse_trade_calendar_2026.csv'}
    target = tmp_path / 'settings.json'
    write_json(target, settings, exclusive=True)
    monkeypatch.setattr(adapter, 'now', lambda: '2026-09-13T11:00:00+08:00')
    def unexpected(*args, **kwargs):
        raise AssertionError('没有新收盘时不能调用网络或账户')
    monkeypatch.setattr(adapter, 'collect_and_admit', unexpected)
    monkeypatch.setattr(adapter, 'run_accounts', unexpected)
    result = adapter.run(target)
    assert result['status'] == 'NO_NEW_COMPLETE_TRADING_DAY' and result['network_requests'] == result['new_account_rows'] == 0


def test_valid_source_is_reused_across_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    calls = []
    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=b'cached response', status_code=200, url='https://example.test/data', headers={}, raise_for_status=lambda: None)
    monkeypatch.setattr(adapter.requests, 'get', fake_get)
    first = adapter.fetch_once(tmp_path, 'sina', 'https://example.test/data', None, 'https://example.test/', 1)
    adapter.accept_source(tmp_path, first)
    second = adapter.fetch_once(tmp_path, 'sina', 'https://example.test/data', None, 'https://example.test/', 2)
    assert first == second and len(calls) == 1 and calls[0]['verify'] is True


def test_failed_attempt_is_preserved_and_only_later_attempt_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    clock = ['2026-09-14T15:10:00+08:00']
    monkeypatch.setattr(adapter, 'now', lambda: clock[0])
    calls = []
    def failed_get(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout('测试超时')
    monkeypatch.setattr(adapter.requests, 'get', failed_get)
    first = adapter.fetch_once(tmp_path, 'sina', 'https://example.test/data', None, 'https://example.test/', 1)
    assert adapter.fetch_once(tmp_path, 'sina', 'https://example.test/data', None, 'https://example.test/', 1) == first
    with pytest.raises(ValueError, match='五分钟'):
        adapter.fetch_once(tmp_path, 'sina', 'https://example.test/data', None, 'https://example.test/', 2)
    clock[0] = '2026-09-14T15:16:00+08:00'
    second = adapter.fetch_once(tmp_path, 'sina', 'https://example.test/data', None, 'https://example.test/', 2)
    assert len(calls) == 2 and first['status'] == second['status'] == 'EXTERNAL_FREE_SOURCE_FAILED'
