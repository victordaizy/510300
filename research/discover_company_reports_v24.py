"""仅定位固定病例缺少的原公告，并保留查询响应。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import requests

OUT = Path(__file__).resolve().parents[1] / 'reports/research/510300_company_credit_realization_v24'
H = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.cninfo.com.cn/'}


def discover(code):
    dest = OUT / 'sources' / f'{code}_公告目录.json'
    if dest.exists():
        result = json.loads(dest.read_text('utf-8'))
    else:
        result = {'code': code, 'attempts': 1, 'responses': []}
        try:
            r = requests.get('https://www.cninfo.com.cn/new/information/topSearch/query', params={'keyWord': code, 'maxNum': 10}, headers=H, timeout=(8, 20))
            r.raise_for_status()
            entries = r.json()
            org = next(x['orgId'] for x in entries if str(x.get('code')) == code)
            for dates in ['2020-04-01~2020-04-30', '2021-04-01~2021-06-10']:
                payload = {'pageNum': '1', 'pageSize': '100', 'column': 'sse', 'tabName': 'fulltext', 'stock': f'{code},{org}', 'searchkey': '', 'secid': '', 'category': '', 'trade': '', 'seDate': dates, 'sortName': 'time', 'sortType': 'desc', 'isHLtitle': 'false'}
                r = requests.post('https://www.cninfo.com.cn/new/hisAnnouncement/query', data=payload, headers=H, timeout=(8, 20))
                r.raise_for_status()
                result['responses'].append({'payload': payload, 'data': r.json()})
            result['status'] = '目录已保存'
        except Exception as exc:
            result.update(status='本次失败不自动重试', error=str(exc))
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(code, result['status'], flush=True)
    for response in result['responses']:
        data = response['data']
        print('目录条数', data.get('totalAnnouncement'), flush=True)
        for row in data.get('announcements') or []:
            if any(k in row.get('announcementTitle', '') for k in ['季度报告', '更正']):
                print(json.dumps({k: row.get(k) for k in ['announcementTitle', 'announcementTime', 'adjunctUrl']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(discover, ['600519', '600276']))
