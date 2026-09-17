"""既有免费日线和官方分红的按日适配，接纳后调用固定账户增量入口。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import requests

from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.adaptive_allocation_v1 import factors
from research.fixed_date_continuation_v1 import run as run_accounts, exclusive_run, dates_to_append, ROOT
from research.official_dividend_coverage_refresh_v1 import check_http_clock, compare_manager, ledger_rows, parse_manager, validate_announcements


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def location(value):
    return (ROOT / value).resolve()


def relative(value):
    return str(Path(value).resolve().relative_to(ROOT))


def completed_dates(calendar, previous, moment):
    """15:05后才接纳当日收盘；周末和旧截止日不发起网络请求。"""
    clock = pd.Timestamp(moment)
    require(clock.tzinfo is not None, '实际时间必须包含时区')
    clock = clock.tz_convert('Asia/Shanghai')
    dates = pd.DatetimeIndex(calendar).sort_values()
    require(dates.is_unique and len(dates) > 0 and clock.date() <= dates[-1].date(), '官方日历缺失或已过期')
    last_day = clock.tz_localize(None).normalize()
    complete = dates[(dates < last_day) | ((dates == last_day) & (clock.hour * 60 + clock.minute >= 905))]
    require(len(complete) > 0, '官方日历没有已完成日期')
    additions = complete[complete > pd.Timestamp(previous)]
    next_days = dates[dates > complete[-1]]
    require(len(next_days) > 0, '官方日历缺少下一交易日')
    return additions, str(complete[-1].date()), str(next_days[0].date())


def parse_prices(name, body, start, end):
    """沿用211解析口径，只返回指定日期表，不写旧输出。"""
    if name == 'sina':
        objects = ak.fund_etf_hist_sina.__globals__
        vm = objects['py_mini_racer'].MiniRacer()
        vm.eval(objects['hk_js_decode'])
        encoded = body.decode('utf-8').split('=')[1].split(';')[0].replace('"', '')
        data = pd.DataFrame(vm.call('d', encoded))
    elif name == 'tencent':
        text = body.decode('utf-8')
        value = ak.stock_zh_a_hist_tx.__globals__['demjson'].decode(text[text.find('{'):])
        rows = value['data']['sh510300']['day']
        data = pd.DataFrame([{'date': r[0], 'open': r[1], 'close': r[2], 'high': r[3], 'low': r[4],
            'volume': float(r[5]) * 100, 'amount': float(r[8]) * 10000 if len(r) > 8 and r[8] not in ['', None] else np.nan} for r in rows])
    else:
        raise ValueError('未登记行情来源：' + name)
    require(all(c in data for c in ['date', 'open', 'close', 'high', 'low', 'volume']), '行情必要字段缺失')
    data['date'] = pd.to_datetime(data.date, errors='raise').dt.tz_localize(None).dt.normalize()
    if 'amount' not in data:
        data['amount'] = np.nan
    for field in ['open', 'close', 'high', 'low', 'volume', 'amount']:
        data[field] = pd.to_numeric(data[field], errors='coerce')
    data = data[data.date.between(start, end)].sort_values('date').reset_index(drop=True)
    require(len(data) > 0 and not data.date.duplicated().any(), '行情为空或日期重复')
    numbers = data[['open', 'close', 'high', 'low', 'volume', 'amount']]
    require(np.isfinite(numbers.to_numpy()).all() and (numbers.iloc[:, :4] > 0).all().all() and (numbers.iloc[:, 4:] >= 0).all().all(), '行情字段缺失或非法')
    require((data.high >= data[['open', 'close', 'low']].max(axis=1)).all() and
            (data.low <= data[['open', 'close', 'high']].min(axis=1)).all(), '开高低收关系错误')
    data['symbol'], data['source'] = '510300.SH', name
    return data


def compare_prices(sina, tencent, old_prices, expected):
    require(pd.DatetimeIndex(sina.date).equals(pd.DatetimeIndex(tencent.date)), '两行情来源日期不同')
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    rows = []
    for field in fields:
        a, b = sina[field].to_numpy(float), tencent[field].to_numpy(float)
        tolerance = 50 if field in ['volume', 'amount'] else 0
        difference = np.abs(a - b)
        require(np.isfinite(a).all() and np.isfinite(b).all() and (difference <= tolerance).all(), '行情超过原精度差异：' + field)
        if tolerance:
            require(np.array_equal(np.round(a / 100) * 100, np.rint(b)) and (np.abs(b - np.rint(b)) <= 1e-6).all(), '腾讯百单位舍入不能复现：' + field)
        rows.extend({'date': day, 'field': field, 'sina': x, 'tencent': y, 'absolute_difference': error, 'tolerance': tolerance}
            for day, x, y, error in zip(sina.date, a, b, difference))
    last = old_prices.date.iloc[-1]
    extra = sina[sina.date > last]
    require(pd.DatetimeIndex(extra.date).equals(pd.DatetimeIndex(expected)), '新行情未覆盖全部新增官方交易日')
    overlap = old_prices.merge(sina, on='date', suffixes=('_old', '_new'))
    require(len(overlap) > 0, '缺少旧行情重叠锚点')
    for field in fields:
        require(np.array_equal(overlap[field + '_old'], overlap[field + '_new']), '已接纳行情被来源修订：' + field)
    return pd.DataFrame(rows)


def append_inputs(old_prices, old_features, sina, dividends, retrieved_at, source_name='sina.klc_daily_incremental'):
    extra = sina[sina.date > old_prices.date.iloc[-1]].copy()
    require(len(extra) > 0, '没有可追加的新价格')
    for column in old_prices.columns:
        if column not in extra:
            extra[column] = old_prices[column].iloc[-1]
    extra['source'], extra['source_original'] = source_name, source_name
    extra['retrieved_at'] = retrieved_at
    extra['correction_applied'], extra['correction_reason'] = False, ''
    # 整数成交字段必须原样可表示，不能静默截掉小数。
    for field in ['volume', 'amount']:
        if pd.api.types.is_integer_dtype(old_prices[field].dtype):
            require((extra[field].to_numpy(float) == np.rint(extra[field].to_numpy(float))).all(), '原整数字段出现不可直接接纳的小数：' + field)
    extra = extra[old_prices.columns].astype(old_prices.dtypes.to_dict())
    prices = pd.concat([old_prices, extra], ignore_index=True)
    features, names = factors(prices, dividends)
    pd.testing.assert_frame_equal(prices.iloc[:len(old_prices)].reset_index(drop=True), old_prices, check_exact=True)
    pd.testing.assert_frame_equal(features.iloc[:len(old_features)].reset_index(drop=True), old_features, check_exact=True)
    require(features.feature_valid.iloc[len(old_features):].all(), '新增因素不完整')
    return prices, features


def fetch_once(directory, name, url, params, referer, attempt):
    definition = {'url': url, 'params': params, 'referer': referer}
    valid = directory / (name + '_valid.json')
    target = directory / f'{name}_attempt{attempt}.json'
    if valid.exists():
        receipt = read(location(read(valid)['receipt']))
        require(receipt['request'] == definition and digest(location(receipt['raw_file'])) == receipt['raw_sha256'], '已通过来源的请求或内容不同')
        return receipt
    if target.exists():
        return read(target)
    require(attempt >= 1, '请求尝试编号必须为正')
    earlier = sorted(directory.glob(name + '_attempt*.json'))
    if earlier:
        previous = max((read(p) for p in earlier), key=lambda r: r['attempt'])
        require(attempt > previous['attempt'], '失败重试编号必须晚于该来源上次尝试')
        require(pd.Timestamp(now()) - pd.Timestamp(previous['finished_at']) >= pd.Timedelta(minutes=5), '失败来源尚未到五分钟重试间隔')
    receipt = {'source': name, 'request': definition, 'attempt': attempt, 'started_at': now(),
        'tls_verified': True, 'request_count': 1, 'cost_cny': 0, 'receipt_file': relative(target)}
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0', 'Referer': referer},
            timeout=(8, 20), verify=True, allow_redirects=False)
        body = response.content
        receipt.update(retrieved_at=now(), http_status=response.status_code, final_url=response.url,
            headers=dict(response.headers), raw_bytes=len(body))
        require(len(body) <= 5_000_000, '来源响应超出原体积上限')
        raw = directory / f'{name}_attempt{attempt}.raw'
        with raw.open('xb') as stream:
            stream.write(body)
        receipt.update(raw_file=relative(raw), raw_sha256=digest(raw))
        response.raise_for_status()
        require(response.status_code == 200, '来源不是直接成功响应')
        receipt['status'] = 'HTTP_OK_NOT_YET_ADMITTED'
    except Exception as exc:
        receipt.update(status='EXTERNAL_FREE_SOURCE_FAILED', error_type=type(exc).__name__, error=str(exc))
    receipt['finished_at'] = now()
    write_json(target, receipt, exclusive=True)
    return receipt


def accept_source(directory, receipt):
    target = directory / (receipt['source'] + '_valid.json')
    value = {'receipt': receipt['receipt_file'], 'raw_sha256': receipt['raw_sha256'], 'validated_at': now()}
    if target.exists():
        require(read(target)['raw_sha256'] == value['raw_sha256'], '已通过来源被改写')
    else:
        write_json(target, value, exclusive=True)


def raw(receipt):
    require(receipt['status'] == 'HTTP_OK_NOT_YET_ADMITTED', '来源请求失败：' + receipt['source'])
    target = location(receipt['raw_file'])
    require(digest(target) == receipt['raw_sha256'], '原始来源内容改变')
    return target.read_bytes()


def sse_params(official, end, number):
    return {'isPagination': 'true', 'pageHelp.pageSize': official['page_size'], 'pageHelp.pageNo': number,
        'pageHelp.beginPage': number, 'pageHelp.cacheSize': 1, 'pageHelp.endPage': number, 'type': 'inParams',
        'sqlId': official['sse_sql_id'], 'TITLE': '', 'SECURITY_CODE': '510300', 'BULLETIN_TYPE': '',
        'START_DATE': official['sse_query_start'], 'END_DATE': end, 'DATE_DESC': 1, 'DATE_ASC': '', 'CODE_DESC': '', 'CODE_ASC': ''}


def collect_and_admit(settings, source, directory, dates, attempt):
    official = read(location(settings['official_configuration']))
    cutoff = str(dates[-1].date())
    old_prices, old_features = pd.read_parquet(location(source['prices'])), pd.read_parquet(location(source['features']))
    start = str(old_prices.date.iloc[max(0, len(old_prices) - 10)].date())
    definitions = [
        ('sina', 'https://finance.sina.com.cn/realstock/company/sh510300/hisdata_klc2/klc_kl.js', None, 'https://finance.sina.com.cn/'),
        ('tencent', 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get',
            {'_var': 'kline_day2026', 'param': f'sh510300,day,{start},{cutoff},640,', 'r': '0.8205512681390605'}, 'https://gu.qq.com/'),
        ('manager', official['manager_url'], None, 'https://www.huatai-pb.com/'),
        ('sse_page_1', official['sse_query_url'], sse_params(official, cutoff, 1), 'https://www.sse.com.cn/'),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda definition: fetch_once(directory, *definition, attempt), definitions))
    sources = {r['source']: r for r in receipts}
    sina, tencent = [parse_prices(name, raw(sources[name]), start, cutoff) for name in ['sina', 'tencent']]
    comparison = compare_prices(sina, tencent, old_prices, dates)
    for name in ['sina', 'tencent']:
        accept_source(directory, sources[name])
    pages = [json.loads(raw(sources['sse_page_1']))]
    page_count = int(pages[0]['pageHelp']['pageCount'])
    require(1 <= page_count <= official['maximum_pages'], '官方公告页数超出原完整分页范围')
    for number in range(2, page_count + 1):
        name = 'sse_page_' + str(number)
        r = fetch_once(directory, name, official['sse_query_url'], sse_params(official, cutoff, number), 'https://www.sse.com.cn/', attempt)
        receipts.append(r)
        sources[name] = r
        pages.append(json.loads(raw(r)))
    for name, r in sources.items():
        if name == 'manager' or name.startswith('sse_page_'):
            check_http_clock(r['headers'], pd.Timestamp(r['retrieved_at']).to_pydatetime(), pd.Timestamp(cutoff).date(), official)
    manager = parse_manager(raw(sources['manager']), '510300')
    compare_manager(manager, ledger_rows(location(source['dividends'])))
    announcements = validate_announcements(pages, official, pd.Timestamp(cutoff).date(), read(location(source['coverage'])))
    for name, r in sources.items():
        if name == 'manager' or name.startswith('sse_page_'):
            accept_source(directory, r)
    prices, features = append_inputs(old_prices, old_features, sina, pd.read_csv(location(source['dividends'])), sources['sina']['retrieved_at'])
    for filename, frame in [('candidate_prices.parquet', prices), ('candidate_features.parquet', features)]:
        target = directory / filename
        if target.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(target), frame, check_exact=True)
        else:
            frame.to_parquet(target, index=False)
    comparison.to_csv(directory / 'source_precision_comparison.csv', index=False, encoding='utf-8-sig')
    coverage = read(location(source['coverage']))
    coverage.update(coverage_end=cutoff, retrieved_at=now(), event_ledger_changed=False,
        coverage_refresh_evidence={'manager_event_count': len(manager), **announcements})
    coverage_path = directory / 'candidate_dividend_coverage.json'
    if coverage_path.exists():
        saved_coverage = read(coverage_path)
        require(saved_coverage['coverage_end'] == cutoff and saved_coverage['coverage_refresh_evidence'] == coverage['coverage_refresh_evidence'], '已保存分红覆盖与本次核对不同')
    else:
        write_json(coverage_path, coverage, exclusive=True)
    receipt = {'accepted': True, 'completed_at': now(), 'cutoff': cutoff, 'dividend_coverage_through': cutoff,
        'features_sha256': digest(directory / 'candidate_features.parquet'), 'dividends_sha256': digest(location(source['dividends'])),
        'evidence_class': 'NEW_COMPLETE_DAILY_INPUTS', 'old_price_rows_preserved': len(old_prices),
        'old_feature_rows_preserved': len(old_features), 'new_trading_days': len(dates), 'old_prefix_exact': True,
        'new_model_fits': 0, 'raw_source_receipts': [r['receipt_file'] for r in receipts],
        'price_comparison': relative(directory / 'source_precision_comparison.csv'), 'source_cost_cny': 0}
    write_json(directory / 'admission_receipt.json', receipt, exclusive=True)
    return receipt


def forward_records(settings, source, account_folder, dates):
    """保留所有日期，迟到决定单列，绝不筛掉不利日拼接收益。"""
    origin = read(location(settings['research_origin']))
    initial = source['source_folder'] == settings['initial_source']['source_folder']
    if initial:
        for item in origin['accounts']:
            require(digest(location(item['checkpoint']['path'])) == item['checkpoint']['sha256'], '原研究起点状态改变')
    rows, day_status = [], []
    for date in dates:
        opening = date.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=9, minutes=30)
        timely = 0
        for folder in account_folder.glob('accounts/*/*'):
            if date == dates[0]:
                recorded = origin['recorded_at'] if initial else read(location(source['source_folder']) / 'accounts' / folder.parent.name / folder.name / 'recording.json')['actual_recorded_at']
            else:
                recorded = read(folder / 'recording.json')['actual_recorded_at']
            timely += pd.Timestamp(recorded) < opening
        day_status.append({'date': str(date.date()), 'timely_source_accounts': timely, 'all_22_incoming_decisions_saved_before_open': timely == 22})
    for cost in ['BASE', 'STRESS']:
        ledger = pd.read_parquet(account_folder / 'accounts' / cost / 'SELECTED_MIX_BAND10_SIMPLE2' / 'ledger.parquet')
        for item in ledger[ledger.date.isin(dates)].to_dict('records'):
            flag = next(r for r in day_status if r['date'] == str(item['date'].date()))
            rows.append({**flag, 'cost': cost, 'net_return': item['net_return'], 'equity': item['equity'], 'shares': item['shares'],
                'filled_quantity': item['filled_quantity'], 'commission': item['commission'], 'slippage_cost': item['slippage_cost']})
    frame = pd.DataFrame(rows)
    frame.to_csv(account_folder.parent / 'new_observed_daily_returns.csv', index=False, encoding='utf-8-sig')
    return {'new_observed_trading_days': len(dates), 'new_timely_incoming_decision_days': sum(r['all_22_incoming_decisions_saved_before_open'] for r in day_status),
        'daily_record': relative(account_folder.parent / 'new_observed_daily_returns.csv'), 'late_days_retained': True,
        'independent_stable_performance_established': False}


def run(settings_path, attempt=1):
    settings_path = Path(settings_path).resolve()
    settings = read(settings_path)
    root = location(settings['output_root'])
    with exclusive_run(root):
        completed_path = root / 'latest_completed.json'
        source = read(completed_path)['next_source'] if completed_path.exists() else settings['initial_source']
        prices = pd.read_parquet(location(source['prices']), columns=['date'])
        calendar = pd.to_datetime(pd.read_csv(location(settings['calendar'])).trade_date)
        clock = now()
        dates, last, next_day = completed_dates(calendar, prices.date.iloc[-1], clock)
        if len(dates) == 0:
            result = {'checked_at': clock, 'status': 'NO_NEW_COMPLETE_TRADING_DAY', 'accepted_price_cutoff': str(prices.date.iloc[-1].date()),
                'latest_complete_official_day': last, 'next_official_trading_day': next_day,
                'next_useful_check_after': next_day + 'T15:05:00+08:00', 'network_requests': 0,
                'new_account_rows': 0, 'new_model_fits': 0, 'new_observed_trading_days': 0, 'goal_achieved': False,
                'not_a_live_wait_handle': True}
            write_json(root / 'latest_check.json', result)
            print('暂无新的完整收盘日期，未请求来源，也未计算账户。', flush=True)
            return result
        cutoff = str(dates[-1].date())
        directory = root / cutoff
        directory.mkdir(parents=True, exist_ok=True)
        failure_path = directory / f'attempt{attempt}_failure.json'
        if failure_path.exists():
            print('本次尝试已有失败记录，保留原状态；失败来源可在规定间隔后使用下一尝试编号。', flush=True)
            return read(failure_path)
        request_path = directory / 'request.json'
        binding = {'settings_sha256': digest(settings_path), 'source': source, 'cutoff': cutoff,
            'adapter_sha256': digest(Path(__file__))}
        if request_path.exists():
            require(read(request_path)['binding'] == binding, '同日期的输入或程序设置改变')
        else:
            write_json(request_path, {'started_at': clock, 'binding': binding}, exclusive=True)
        try:
            admission_path = directory / 'admission_receipt.json'
            admission = read(admission_path) if admission_path.exists() else collect_and_admit(settings, source, directory, dates, attempt)
            require(admission['accepted'] and admission['cutoff'] == cutoff, '保存接纳回执不同')
            account_settings = {'configuration': settings['strategy_configuration'], 'source_folder': source['source_folder'],
                'output_folder': relative(directory / 'accounts_run'), 'source_features': source['features'],
                'features': relative(directory / 'candidate_features.parquet'), 'dividends': source['dividends'],
                'ridge_models': source['ridge_models'], 'within_models': source['within_models'],
                'cutoff': cutoff, 'admission_receipt': relative(admission_path), 'mode': 'FIXED_RESEARCH_CONTINUATION',
                'research_origin': settings['research_origin']}
            account_settings_path = directory / 'account_settings.json'
            if account_settings_path.exists():
                require(read(account_settings_path) == account_settings, '新增账户设置不同')
            else:
                write_json(account_settings_path, account_settings, exclusive=True)
            account_result = run_accounts(account_settings_path)
            observed = forward_records(settings, source, directory / 'accounts_run', dates)
            next_source = {**source, 'prices': relative(directory / 'candidate_prices.parquet'),
                'features': relative(directory / 'candidate_features.parquet'), 'coverage': relative(directory / 'candidate_dividend_coverage.json'),
                'source_folder': relative(directory / 'accounts_run')}
            result = {'completed_at': now(), 'status': 'COMPLETED_NEW_DAILY_INPUTS_AND_FIXED_CONTINUATION', 'cutoff': cutoff,
                'all_metrics': account_result['all_metrics'], 'account_result': relative(directory / 'accounts_run/result.json'),
                'admission_receipt': relative(admission_path), 'next_source': next_source, **observed,
                'new_model_fits': 0, 'new_candidates': 0, 'goal_achieved': False, 'position_impact': 0}
            write_json(directory / 'result.json', result)
            write_json(completed_path, result)
            return result
        except Exception as exc:
            result = {'failed_at': now(), 'status': 'NEW_DAILY_INPUT_OR_CONTINUATION_NOT_COMPLETED', 'cutoff': cutoff,
                'attempt': attempt, 'error_type': type(exc).__name__, 'error': str(exc), 'source_state_advanced': False,
                'goal_achieved': False, 'position_impact': 0, 'source_success_and_partial_accounts_retained': True}
            write_json(failure_path, result, exclusive=True)
            print('新日期未完成，保留来源和已有增量：' + str(exc), flush=True)
            return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='新增完整收盘才请求原免费来源，接纳后直接续算固定研究。')
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.settings, args.attempt), ensure_ascii=False), flush=True)
