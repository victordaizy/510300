"""取得固定十二家公司公开研报历史目录，保留全部分页和覆盖缺口。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now

OUT = ROOT / 'reports/research/510300_forward_eps_history_directory_v1'
RAW = ROOT / 'data/raw/510300_forward_eps_history_directory_v1'
MANIFEST = ROOT / 'config/510300_forward_eps_history_directory_v1_manifest.json'
SOURCE = ROOT / 'reports/research/510300_financial_quarter_history_v1/selected_before_download.parquet'
MEMBERS = ROOT / 'data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet'
URL = 'https://reportapi.eastmoney.com/report/list'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/report/'}
STOP = threading.Event()


def identity(p):
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(SOURCE)[['ts_code', 'sec_name']].drop_duplicates()
    universe = [{'ts_code': code, 'aliases': sorted(g.sec_name.unique().tolist())}
                for code, g in u.groupby('ts_code', sort=True)]
    if len(universe) != 12:
        raise ValueError('固定公司清单不是十二个代码')
    universe_path = OUT / 'companies_before_directory.json'
    save(universe_path, {'companies': universe}, exclusive=True)
    membership = pd.read_parquet(MEMBERS, columns=['membership_date', 'symbol'])
    membership = membership[membership.symbol.isin([x['ts_code'] for x in universe])]
    membership.to_parquet(OUT / 'fixed_company_historical_membership.parquet', index=False)
    inputs = [Path(__file__), SOURCE, MEMBERS, universe_path,
              OUT / 'fixed_company_historical_membership.parquet',
              ROOT / 'docs/510300_FORWARD_EPS_HISTORY_DIRECTORY_V1.md',
              ROOT / 'research/financial_annual_components_v1.py',
              ROOT / 'reports/research/510300_forward_eps_public_api_probe_v1/result.json']
    save(MANIFEST, {'registered_at': now(), 'budget_cny': 0, 'companies': universe,
                   'start': '2015-01-01', 'end': '2026-08-14', 'page_size': 50,
                   'directory_institutions': 'ALL_RETURNED', 'source_workers': 3,
                   'stock_return_read': False, 'source_api': URL,
                   'original_pdf_stage_institution_code': '80000007',
                   'files': [identity(p) for p in inputs]}, exclusive=True)
    print('已登记十二家公司、2015年至2026年8月的完整公开目录范围。', flush=True)


def fetch_page(code, year, page, end):
    target = RAW / f'{code}_{year}_{page:04d}.json'
    receipt = OUT / 'request_receipts' / f'{code}_{year}_{page:04d}.json'
    if target.exists() and receipt.exists():
        r = read(receipt)
        if hashlib.sha256(target.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('已保存目录内容变化')
        return read(target), r
    if STOP.is_set():
        raise RuntimeError('已出现访问限制，剩余请求停止')
    params = {'code': code, 'qType': 0, 'pageSize': 50, 'pageNo': page,
              'beginTime': f'{year}-01-01', 'endTime': end,
              'industryCode': '*', 'industry': '*', 'rating': '*',
              'ratingChange': '*', 'fields': ''}
    failures = []
    for attempt in range(2):
        try:
            response = requests.get(URL, params=params, headers=HEADERS, timeout=(12, 35))
            if response.status_code in (401, 403, 429):
                STOP.set()
                raise RuntimeError('公开接口访问限制，停止：' + str(response.status_code))
            response.raise_for_status()
            obj = response.json()
            if not isinstance(obj.get('data'), list) or 'hits' not in obj or 'TotalPage' not in obj:
                raise ValueError('公开目录返回结构不符')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            item = {'retrieved_at': now(), 'url': response.url, 'params': params,
                    'sha256': hashlib.sha256(response.content).hexdigest(),
                    'bytes': len(response.content), 'raw_path': target.relative_to(ROOT).as_posix(),
                    'prior_transport_errors': failures}
            save(receipt, item, exclusive=True)
            time.sleep(0.2)
            return obj, item
        except (requests.RequestException, ValueError) as exc:
            failures.append(type(exc).__name__ + ': ' + str(exc))
            if attempt == 1:
                raise
            time.sleep(1)
    raise RuntimeError('目录请求未返回')


def collect_year(company, year):
    code = company['ts_code'][:6]
    marker = OUT / 'year_receipts' / f'{code}_{year}.json'
    if marker.exists():
        return read(marker)
    end = '2026-08-14' if year == 2026 else f'{year}-12-31'
    records, page_paths, observed_hits, observed_total_pages = [], [], [], []
    first, first_receipt = fetch_page(code, year, 1, end)
    total_pages = max(1, int(first['TotalPage']))
    if total_pages > 200:
        raise ValueError('单公司年度目录超过登记运行上限，需检查返回范围')
    for page in range(1, total_pages + 1):
        obj, receipt = (first, first_receipt) if page == 1 else fetch_page(code, year, page, end)
        observed_hits.append(int(obj['hits']))
        observed_total_pages.append(int(obj['TotalPage']))
        if int(obj.get('pageNo', page)) != page:
            raise ValueError('返回页码与请求不符')
        page_paths.append(receipt['raw_path'])
        for row in obj['data']:
            if row['stockCode'] != code:
                raise ValueError('公开接口混入其他证券')
            pub = row['publishDate'][:10]
            if not (f'{year}-01-01' <= pub <= end):
                raise ValueError('目录日期超出查询区间')
            records.append({**row, 'ts_code': company['ts_code'], 'query_year': year,
                            'source_page': page, 'source_raw_path': receipt['raw_path'],
                            'directory_retrieved_at': receipt['retrieved_at']})
    ids = [x['infoCode'] for x in records]
    stable = len(set(observed_hits)) == 1 and len(set(observed_total_pages)) == 1
    matching = len(records) == int(first['hits']) and len(ids) == len(set(ids))
    result = {'ts_code': company['ts_code'], 'year': year, 'completed_at': now(),
              'returned_rows': len(records), 'unique_reports': len(set(ids)),
              'claimed_hits': observed_hits, 'claimed_total_pages': observed_total_pages,
              'pages': page_paths, 'pagination_complete': stable and matching,
              'status': 'DIRECTORY_PAGES_RECONCILED' if stable and matching else 'DIRECTORY_PAGINATION_DISCREPANCY',
              'rows': records}
    save(marker, result, exclusive=True)
    print('历史研报目录', company['ts_code'], year, len(records), result['status'], flush=True)
    return result


def run():
    manifest = read(MANIFEST)
    for record in manifest['files']:
        if hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('登记的来源或代码发生变化')
    if (OUT / 'result.json').exists():
        raise FileExistsError('目录已完成，禁止覆盖')
    tasks = [(c, y) for c in manifest['companies'] for y in range(2015, 2027)]
    done, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(collect_year, c, y): (c, y) for c, y in tasks}
        for future in as_completed(futures):
            c, y = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:
                error = {'ts_code': c['ts_code'], 'year': y, 'time': now(),
                         'error_type': type(exc).__name__, 'error': str(exc)}
                errors.append(error)
                print('目录未完成', c['ts_code'], y, error['error_type'], flush=True)
    rows = [r for d in done for r in d['rows']]
    rows.sort(key=lambda x: (x['ts_code'], x['publishDate'], x['infoCode'], x['source_page']))
    raw_records = OUT / 'all_directory_records.json'
    save(raw_records, {'rows': rows}, exclusive=True)
    columns = ['ts_code', 'stockName', 'infoCode', 'publishDate', 'orgCode', 'orgName',
               'orgSName', 'title', 'attachPages', 'source_raw_path', 'directory_retrieved_at']
    table = pd.DataFrame([{k: row.get(k) for k in columns} for row in rows])
    table.to_parquet(OUT / 'historical_report_directory.parquet', index=False)
    table.to_csv(OUT / '十二家公司_全机构历史研报目录.csv', index=False, encoding='utf-8-sig')
    fixed = table.loc[table.orgCode.eq('80000007')].drop_duplicates('infoCode').copy()
    fixed.to_parquet(OUT / 'guosen_original_pdf_queue.parquet', index=False)
    fixed.to_csv(OUT / '国信证券_全部历史原件队列.csv', index=False, encoding='utf-8-sig')
    yearly = pd.DataFrame([{'ts_code': x['ts_code'], 'year': x['year'],
                           'returned_rows': x['returned_rows'], 'unique_reports': x['unique_reports'],
                           'pagination_complete': x['pagination_complete']} for x in done])
    yearly.to_csv(OUT / '公司年度覆盖.csv', index=False, encoding='utf-8-sig')
    summary = table.groupby(['ts_code', 'orgCode', 'orgSName'], dropna=False).agg(
        reports=('infoCode', 'nunique'), first_report=('publishDate', 'min'),
        last_report=('publishDate', 'max')).reset_index()
    summary.to_csv(OUT / '公司机构覆盖.csv', index=False, encoding='utf-8-sig')
    result = {'completed_at': now(), 'status': 'PUBLIC_HISTORICAL_DIRECTORY_COLLECTED' if not errors else 'PUBLIC_HISTORICAL_DIRECTORY_PARTIAL',
              'scheduled_company_years': len(tasks), 'completed_company_years': len(done),
              'pagination_complete_company_years': sum(x['pagination_complete'] for x in done),
              'directory_occurrences': len(rows), 'unique_reports': int(table.infoCode.nunique()),
              'companies': int(table.ts_code.nunique()), 'institutions': int(table.orgCode.nunique()),
              'fixed_guosen_pdf_queue': len(fixed),
              'guosen_queue_by_company': fixed.groupby('ts_code').size().to_dict(),
              'errors': errors, 'access_restriction_stop': STOP.is_set(),
              'metadata_relative_eps_not_used_as_history': True,
              'historical_reports_outside_provider_not_proven_complete': True,
              'new_eps_facts': 0, 'new_accounts': 0, 'goal_achieved': False,
              'generated_files': [identity(p) for p in [raw_records, OUT / 'historical_report_directory.parquet', OUT / 'guosen_original_pdf_queue.parquet']]}
    save(OUT / 'result.json', result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False, default=int), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else run()
