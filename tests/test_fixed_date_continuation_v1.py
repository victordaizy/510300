"""只验证新入口的日历边界、月首缺项和同目录活进程锁。"""
import pandas as pd
import pytest

from research.fixed_date_continuation_v1 import dates_to_append, check_monthly_records, exclusive_run


def fixture():
    dates = pd.to_datetime(['2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15', '2026-09-16'])
    return pd.DataFrame({'date': dates[:4]}), dates


def test_single_day_and_variable_length_with_weekend():
    data, calendar = fixture()
    one, next_date = dates_to_append(data.iloc[:3], calendar, '2026-09-11', '2026-09-14', '2026-09-14T15:06:00+08:00', '2026-09-14')
    assert len(one) == 1 and next_date == '2026-09-15'
    two, next_date = dates_to_append(data, calendar, '2026-09-11', '2026-09-15', '2026-09-15T15:06:00+08:00', '2026-09-15')
    assert len(two) == 2 and next_date == '2026-09-16'


@pytest.mark.parametrize('problem, message', [('missing', '新增日期缺失'), ('unfinished', '尚未完整收盘'), ('dividend', '分红覆盖不足')])
def test_incomplete_new_input_rejected(problem, message):
    data, calendar = fixture()
    stamp, coverage = '2026-09-15T15:06:00+08:00', '2026-09-15'
    if problem == 'missing':
        data = data.drop(index=2)
    elif problem == 'unfinished':
        stamp = '2026-09-15T14:59:00+08:00'
    else:
        coverage = '2026-09-14'
    with pytest.raises(ValueError, match=message):
        dates_to_append(data, calendar, '2026-09-11', '2026-09-15', stamp, coverage)


def test_monthly_record_is_required_without_daily_refitting():
    data = pd.DataFrame({'date': pd.to_datetime(['2026-08-31', '2026-09-01', '2026-09-02'])})
    record = {'fit_origin': '2026-09-01', 'fit_index': 1, 'latest_exit_index': 0}
    with pytest.raises(ValueError, match='缺少原定月首记录'):
        check_monthly_records(data, data.date.iloc[1:], [], [record])
    check_monthly_records(data, data.date.iloc[1:], [record], [record])
    check_monthly_records(data, data.date.iloc[2:], [], [])


def test_live_lock_and_released_file_are_distinguished(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(ValueError, match='持有运行锁'):
            with exclusive_run(tmp_path):
                raise AssertionError('同一目录不应允许并发进入')
    with exclusive_run(tmp_path):
        assert (tmp_path / 'run.lock').is_file()
