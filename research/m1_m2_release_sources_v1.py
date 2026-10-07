"""重建人民银行逐次金融统计报告；只取来源，不读取市场收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_m1_m2_monthly_increment_v1'
RAW = ROOT / 'data/raw/macro/510300_m1_m2_monthly_increment_v1'
BASE = 'https://www.pbc.gov.cn'
LIST = BASE + '/diaochatongjisi/116219/116225/'
TZ = timezone(timedelta(hours=8))


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def fetch(url, attempt=1):
    """对同一个URL只留一份响应；失败保存回执，重试必须另存批次。"""
    key = hashlib.sha256((url if attempt == 1 else f'{url}|attempt={attempt}').encode()).hexdigest()[:24]
    receipt = RAW / f'{key}.json'
    if receipt.exists():
        return json.loads(receipt.read_text(encoding='utf-8'))
    started = now()
    row = {'url': url, 'attempt': attempt, 'started_at': started, 'market_data_read': False}
    try:
        r = requests.get(url, timeout=35, headers={'User-Agent': 'Mozilla/5.0 (compatible; historical-research)'})
        raw = RAW / f'{key}.html'
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open('xb') as f:
            f.write(r.content)
        row.update(status_code=r.status_code, final_url=r.url, retrieved_at=now(),
                   raw_path=raw.relative_to(ROOT).as_posix(), sha256=sha(raw), bytes=len(r.content),
                   content_type=r.headers.get('Content-Type'), server_date=r.headers.get('Date'),
                   tls_verified=True, status='FETCHED' if r.status_code == 200 else 'HTTP_FAILED')
    except requests.RequestException as exc:
        row.update(status='FETCH_FAILED', retrieved_at=now(), error=type(exc).__name__ + ': ' + str(exc))
    write_json(receipt, row)
    return row


def soup_from(row):
    if row['status'] != 'FETCHED':
        return None
    raw = (ROOT / row['raw_path']).read_bytes()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('gb18030')
    return BeautifulSoup(text, 'html.parser')


def report_month(title):
    title = re.sub(r'\s+', '', title)
    match = re.fullmatch(r'(20\d{2})年(\d{1,2}月|一季度|上半年|前三季度|三季度|全年)?金融统计数据报告', title)
    if not match:
        return None
    part = match.group(2) or '全年'
    month = int(part[:-1]) if part.endswith('月') else {'一季度': 3, '上半年': 6, '前三季度': 9, '三季度': 9, '全年': 12}[part]
    return f'{match.group(1)}-{month:02d}'


def parse_report(row, month, title):
    base = {'stat_month': month, 'title': title, 'source_url': row['url'],
            'retrieved_at': row['retrieved_at'], 'source_status': row['status'],
            'raw_path': row.get('raw_path'), 'source_sha256': row.get('sha256'),
            'historical_immutable_vintage_proven': False}
    soup = soup_from(row)
    if soup is None:
        return dict(base, admission='NO_VIEW_SOURCE')
    full = soup.get_text(' ', strip=True)
    content = soup.find(id='zoom')
    body = (content or soup).get_text(' ', strip=True)
    compact = re.sub(r'\s+', '', body).replace('（', '(').replace('）', ')')
    times = re.findall(r'文章来源[：:]?\s*.*?(20\d{2}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}(?::\d{2})?))?', full)
    if not times:
        return dict(base, admission='NO_VIEW_PUBLICATION_CLOCK', body_excerpt=body[:450])
    date, clock = times[0]
    precision = 'SECOND' if len(clock) == 8 else ('MINUTE' if clock else 'DATE')
    timestamp = date + 'T' + (clock if len(clock) == 8 else (clock + ':00' if clock else '23:59:59')) + '+08:00'
    values = {}
    evidence = {}
    for name in ['M1', 'M2']:
        found = re.search(r'\(' + name + r'\)余额.{0,65}?同比(增长|下降|减少|增加|持平)([-+\d.]+)?[%％]?', compact)
        if not found:
            return dict(base, admission='NO_VIEW_VALUE_PARSE', published_at=timestamp, body_excerpt=body[:450])
        sign = -1 if found.group(1) in ['下降', '减少'] else 1
        value = 0.0 if found.group(1) == '持平' else sign * float(found.group(2))
        values[name.lower() + '_yoy_pp'] = value
        evidence[name] = found.group(0)
    meta = soup.find('meta', attrs={'name': 'createDate'})
    base.update(**values, published_at=timestamp, publication_time_precision=precision,
                available_at_upper_bound=timestamp, cms_create_date=meta.get('content') if meta else None,
                evidence_json=json.dumps(evidence, ensure_ascii=False),
                notes=' '.join(re.findall(r'注\d?[：:].*?(?=注\d?[：:]|$)', body)),
                admission='DATED_REPORT_PARSED_PENDING_DEFINITION_ADMISSION')
    return base


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'config/510300_m1_m2_monthly_increment_v1.json').read_text(encoding='utf-8'))
    first = fetch(LIST + 'index.html')
    soup = soup_from(first)
    if soup is None:
        raise RuntimeError('官方目录不可用；已保存失败回执')
    last_links = [a.get('tagname', '') for a in soup.find_all('a') if a.get_text(strip=True) == '尾页']
    total = int(re.search(r'-(\d+)\.html', last_links[0]).group(1))
    lists = [first]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, LIST + f'11871-{n}.html') for n in range(2, total + 1)]
        for future in as_completed(futures):
            lists.append(future.result())
    refs = {}
    for row in lists:
        soup = soup_from(row)
        if soup is None:
            continue
        for a in soup.find_all('a', href=True):
            title = a.get_text(' ', strip=True)
            month = report_month(title)
            if month and cfg['source_start_month'] <= month <= cfg['source_end_month']:
                url = urljoin(BASE, a['href'])
                refs[url] = (month, title)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, url): (month, title) for url, (month, title) in refs.items()}
        for future in as_completed(futures):
            month, title = futures[future]
            row = parse_report(future.result(), month, title)
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(['stat_month', 'source_url'])
    dest = OUT / 'official_release_candidates.csv'
    if dest.exists():
        raise FileExistsError('候选来源表已生成，禁止覆盖')
    frame.to_csv(dest, index=False, encoding='utf-8-sig')
    receipt = {'study_id': cfg['study_id'], 'completed_at': now(), 'list_pages': len(lists),
               'list_failures': [r for r in lists if r['status'] != 'FETCHED'], 'report_rows': len(frame),
               'unique_months': int(frame.stat_month.nunique()), 'status_counts': frame.admission.value_counts().to_dict(),
               'source_table_sha256': sha(dest), 'market_data_reads': 0, 'new_fits': 0, 'new_accounts': 0,
               'source_evidence': '现时保存的带历史发布时间官方报告，不能证明发布当时已由本项目留存，不能计严格前向证据'}
    write_json(OUT / 'source_collection_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='仅收集人民银行原始报告，不读取收益')
    parser.add_argument('action', choices=['collect'])
    parser.parse_args()
    collect()
