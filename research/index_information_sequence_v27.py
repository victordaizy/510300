"""按原观察窗口核对2019年货币、信用、消息与510300价格路径。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_index_information_sequence_v27'
PREVIOUS = ROOT / 'reports/research/510300_index_repricing_chronology_v26'
SOURCES = [
    {'id': 'money_march', 'name': '央行2019年一季度金融统计',
     'url': 'https://www.pbc.gov.cn/diaochatongjisi/116219/116225/4818fd3dab79452b88397496fdd6798f/index.html',
     'local': 'data/raw/macro/510300_m1_m2_monthly_increment_v1/3d9a185f6519d52bca95b5cc.html'},
    {'id': 'money_april', 'name': '央行2019年4月金融统计',
     'url': 'https://www.pbc.gov.cn/diaochatongjisi/116219/116225/3c2f9ef3241c41d08397b3b59249a315/index.html',
     'local': 'data/raw/macro/510300_m1_m2_monthly_increment_v1/605694121bf24dd4e0bb5a2c.html'},
    {'id': 'pmi_april', 'name': '统计局2019年4月PMI',
     'url': 'https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900300.html',
     'local': 'data/raw/macro/510300_macro_stress_2015_v2/sources_2015_2020/20260830T153000_+0800/nbs_pmi_2019-04_01_9f4bdc72223c.html'},
    {'id': 'economy_q1', 'name': '统计局2019年一季度经济发布',
     'url': 'https://www.stats.gov.cn/sj/xwfbh/fbhwd/202302/t20230203_1900276.html'},
    {'id': 'politburo', 'name': '新华社2019年4月19日政治局会议',
     'url': 'https://www.xinhuanet.com/politics/leaders/2019-04/19/c_1124390391.htm'},
    {'id': 'tariff_intent', 'name': '美国总统2019年5月5日公开推文存档',
     'url': 'https://www.presidency.ucsb.edu/documents/tweets-may-5-2019'},
    {'id': 'rrr', 'name': '中国政府网2019年5月6日定向降准报道',
     'url': 'https://www.gov.cn/xinwen/2019-05/06/content_5389151.htm'},
    {'id': 'tariff_notice', 'name': '美国贸易代表办公室关税法律通知',
     'url': 'https://ustr.gov/sites/default/files/enforcement/301Investigations/84_FR_20459.pdf'},
    {'id': 'ustr_may10', 'name': '美国贸易代表办公室2019年5月10日声明',
     'url': 'https://ustr.gov/about-us/policy-offices/press-office/press-releases/2019/may/statement-us-trade-representative',
     'local': 'reports/research/510300_historical_index_liquidity_transmission_v1/sources/ustr_20190510.html'},
    {'id': 'china_tariff', 'name': '国务院关税税则委员会2019年5月13日公告',
     'url': 'https://gss.mof.gov.cn/gzdt/zhengcefabu/201905/t20190513_3256787.htm'},
    {'id': 'economy_april', 'name': '统计局2019年4月经济发布',
     'url': 'https://www.stats.gov.cn/sj/xwfbh/fbhwd/202302/t20230203_1900315.html'},
]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def save(name, obj):
    (OUT / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(OUT / 'results' / name, index=False, encoding='utf-8-sig')


def freeze():
    if OUT.exists():
        raise RuntimeError('输出目录已存在，不能覆盖预先固定的范围。')
    for name in ['inputs', 'sources', 'results', 'figures', 'code']:
        (OUT / name).mkdir(parents=True)
    protocol = {
        'at': now(), 'study': '510300_INDEX_INFORMATION_SEQUENCE_V27',
        'previous_goal_turn_classification': 'PROGRESS',
        'question': '2019年3月货币改善后的原20日亏损，经历了哪些当时信息变化，哪些损失已经发生在五一假期关税消息之前？',
        'scope': '沿用六个既有病例中的2019-03；指数整体为研究单位；不扩展个股报告。全104月输入原样保留。',
        'historical_outcomes_already_seen': True,
        'not_independent_validation': True,
        'window': '保留E0为2019-04-15开盘至2019-05-15收盘，E1为2019-04-16开盘至2019-05-16收盘；不改变20交易日规则。',
        'anchors_fixed_before_path_calculation': ['原开盘', '2019-04-19收盘', '2019-04-30收盘', '2019-05-06开盘', '2019-05-06收盘', '2019-05-09收盘', '原末日收盘'],
        'anchor_reason': '按政策会议发布前收盘、PMI/节前收盘、复市开盘与收盘、新货币发布日和原终点分段；不按价格极值分段。',
        'cash_arithmetic': '各段固定股数价格加累计应享现金变化除以原入场价；分段贡献之和必须等于原收益。除息日严格晚于入场日才计现金，不复利重投资。',
        'volatility': '沿用20交易日每日含现金简单收益；RV为ddof=1标准差乘sqrt(252)；D为sqrt(252/20*负收益平方和)。五日D平方变化拆为新进入五日和退出五日；另看相邻最近两组五日。',
        'information_clock': '逐开盘09:30与收盘15:00保存最新已公布货币。5月9日央行网页时间为10:00:03，必须作盘中更新，不倒填开盘。',
        'source_uncertainty': '现今网页不是历史不可变快照。推文档案时间缺时区；保留UTC及美国东部、西部时区的候选时间，不当作认证首发时刻。日期级美国来源取当地日终上界。降准政府网19:29转发不冒充当天最早公告。',
        'attribution_limit': '分段是算术位置，不是新闻导致的收益；事件清单不保证完备，政策实施与首次宣布不同。不删除突发消息期间，不从反例中另调买点。',
        'new_models': 0, 'new_accounts': 0, 'orders_authorized': False, 'goal_achieved': False,
    }
    save('protocol.json', protocol)
    files = {
        'monthly.csv': PREVIOUS / 'results/104个月_宏观来源与指数定价前后全表.csv',
        'market.csv': PREVIOUS / 'inputs/market.csv',
    }
    records = []
    for name, source in files.items():
        dest = OUT / 'inputs' / name
        shutil.copy2(source, dest)
        records.append({'name': name, 'source': str(source), 'sha256': digest(dest)})
    save('freeze.json', {'at': now(), 'protocol_sha256': digest(OUT / 'protocol.json'), 'inputs': records})
    save('source_plan.json', SOURCES)
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    print('已固定2019年既有病例的信息顺序核对；原样保留全部104个月输入。')


def collect():
    receipt_path = OUT / 'source_receipts.json'
    existing = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else []
    attempted = {r['id'] for r in existing}
    for s in SOURCES:
        if s['id'] in attempted:
            continue
        item = {**s, 'retrieved_now_at': now(), 'historical_immutable_vintage_proven': False}
        suffix = '.pdf' if s['url'].endswith('.pdf') else '.html'
        dest = OUT / 'sources' / (s['id'] + suffix)
        try:
            if 'local' in s:
                shutil.copy2(ROOT / s['local'], dest)
                item['acquisition'] = '既有原文复制，非本轮网络下载'
            else:
                r = requests.get(s['url'], timeout=25, headers={'User-Agent': 'Mozilla/5.0'})
                item['http_status'] = r.status_code
                r.raise_for_status()
                dest.write_bytes(r.content)
                item['acquisition'] = '本轮公开原文下载'
            if suffix == '.pdf':
                import pdfplumber
                with pdfplumber.open(dest) as pdf:
                    source_text = '\n\n'.join(p.extract_text() or '' for p in pdf.pages)
            else:
                payload = dest.read_bytes()
                try:
                    html = payload.decode('utf-8')
                except UnicodeDecodeError:
                    html = payload.decode('gb18030', errors='replace')
                soup = BeautifulSoup(html, 'html.parser')
                for node in soup(['script', 'style', 'noscript']):
                    node.decompose()
                source_text = soup.get_text('\n', strip=True)
            text_path = dest.with_suffix('.txt')
            text_path.write_text(source_text, encoding='utf-8')
            item.update(status='SAVED_PENDING_CONTENT_REVIEW', path=str(dest.relative_to(ROOT)), sha256=digest(dest), text_chars=len(source_text))
        except Exception as error:
            item.update(status='SOURCE_NOT_SAVED', error=f'{type(error).__name__}: {error}')
        existing.append(item)
        save('source_receipts.json', existing)
        print(s['name'], item['status'], item.get('text_chars', ''), flush=True)


def build():
    frozen = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == frozen['protocol_sha256']
    for f in frozen['inputs']:
        assert digest(OUT / 'inputs' / f['name']) == f['sha256']
    monthly = pd.read_csv(OUT / 'inputs/monthly.csv')
    assert len(monthly) == 104 and monthly.stat_month.is_unique
    origin = monthly.loc[monthly.stat_month == '2019-03'].iloc[0]
    market = pd.read_csv(OUT / 'inputs/market.csv', parse_dates=['date']).set_index('date').sort_index()
    assert market.index.is_unique
    returns = market.close.add(market.dividend).div(market.close.shift()).sub(1)
    assert np.nanmax(np.abs(returns - market.total_simple)) < 1e-9
    negative_square = returns.clip(upper=0).pow(2)
    market['rv20_recomputed'] = returns.rolling(20).std(ddof=1) * np.sqrt(252)
    market['d20_recomputed'] = np.sqrt(negative_square.rolling(20).sum() * 252 / 20)
    market['negative_square_latest5'] = negative_square.rolling(5).sum()
    market['negative_square_prior5'] = negative_square.rolling(5).sum().shift(5)
    market['negative_square_exited5'] = negative_square.rolling(5).sum().shift(20)
    market['d20_square_change5'] = market.d20_recomputed.pow(2).diff(5)
    market['d20_square_change5_rebuilt'] = 252 / 20 * (market.negative_square_latest5 - market.negative_square_exited5)
    daily = market.loc['2019-04-12':'2019-05-16'].copy()
    assert abs(daily.loc['2019-04-12', 'rv20_recomputed'] - origin.rv20) < 1e-9
    assert abs(daily.loc['2019-04-12', 'd20_recomputed'] - origin.downside20) < 1e-9
    publication = pd.to_datetime(monthly.available_at_upper_bound, utc=True)
    known_rows = []
    for date, bar in daily.iterrows():
        for moment, time_str in [('open', '09:30:00'), ('close', '15:00:00')]:
            at = pd.Timestamp(f'{date:%Y-%m-%d} {time_str}', tz='Asia/Shanghai')
            allowed = monthly.loc[publication <= at.tz_convert('UTC')].copy()
            allowed['clock_for_sort'] = publication.loc[allowed.index]
            latest = allowed.sort_values('clock_for_sort').iloc[-1]
            known_rows.append({'date': date, 'moment': moment, 'at': at,
                               'known_stat_month': latest.stat_month, 'known_publication': latest.available_at_upper_bound,
                               'm1_yoy_pp': latest.m1_yoy_pp, 'm2_yoy_pp': latest.m2_yoy_pp,
                               'spread_pp': latest.spread_pp, 'availability_before_endpoint': True})
            for name, value in [('month', latest.stat_month), ('spread_pp', latest.spread_pp), ('publication', latest.available_at_upper_bound)]:
                daily.loc[date, f'{moment}_known_money_{name}'] = value
    clock_table = pd.DataFrame(known_rows)
    csv('逐开盘收盘_当时已公布货币.csv', clock_table)
    assert daily.loc['2019-05-09', 'open_known_money_month'] == '2019-03'
    assert daily.loc['2019-05-09', 'close_known_money_month'] == '2019-04'
    anchors = [('2019-04-19', 'close'), ('2019-04-30', 'close'), ('2019-05-06', 'open'),
               ('2019-05-06', 'close'), ('2019-05-09', 'close')]
    segments, endpoints, checks = [], [], []
    for e in ['E0', 'E1']:
        entry = pd.Timestamp(origin[f'{e}_20_entry_date'])
        end = pd.Timestamp(origin[f'{e}_20_exit_date'])
        window = market.loc[entry:end]
        assert len(window) == 20
        price0 = float(window.iloc[0].open)
        nodes = [(entry.strftime('%Y-%m-%d'), 'open')] + anchors + [(end.strftime('%Y-%m-%d'), 'close')]
        previous = None
        for date_str, side in nodes:
            date = pd.Timestamp(date_str)
            # 这一病例各节点日没有分红；保持原除息日严格晚于入场日的口径。
            assert market.loc[date, 'dividend'] == 0
            cash = market.loc[(market.index > entry) & (market.index <= date), 'dividend'].sum()
            value = market.loc[date, side] + cash
            node = {'clock': e, 'date': date, 'side': side, 'price': market.loc[date, side],
                    'cash_per_share': cash, 'fixed_share_value': value, 'original_entry_open': price0,
                    'cumulative_original_capital_return': value / price0 - 1}
            endpoints.append(node)
            if previous is not None:
                segments.append({'clock': e, 'start_date': previous['date'], 'start_side': previous['side'],
                                 'end_date': date, 'end_side': side,
                                 'start_price': previous['price'], 'end_price': node['price'],
                                 'cash_difference': cash - previous['cash_per_share'],
                                 'return_contribution_on_original_capital': (value - previous['fixed_share_value']) / price0,
                                 'standalone_segment_price_change': node['price'] / previous['price'] - 1,
                                 'role': 'EX_POST_LOCATION_NOT_NEWS_CAUSAL_EFFECT'})
            previous = node
        actual = previous['cumulative_original_capital_return']
        original = origin[f'{e}_20_return']
        total_contribution = sum(s['return_contribution_on_original_capital'] for s in segments if s['clock'] == e)
        assert abs(actual - original) < 1e-9
        assert abs(total_contribution - actual) < 1e-12
        checks.append({'clock': e, 'entry': entry, 'exit': end, 'sessions': len(window),
                       'cash_per_share': previous['cash_per_share'], 'original_return': original,
                       'recomputed_return': actual, 'sum_segments': total_contribution,
                       'original_error': actual - original, 'segment_error': total_contribution - actual})
        daily[f'{e}_cumulative_original_capital_return'] = np.nan
        for date in window.index:
            cash = window.loc[(window.index > entry) & (window.index <= date), 'dividend'].sum()
            daily.loc[date, f'{e}_cumulative_original_capital_return'] = (market.loc[date, 'close'] + cash) / price0 - 1
    csv('原窗口_固定股数分段贡献.csv', pd.DataFrame(segments))
    csv('原窗口_事件日期节点.csv', pd.DataFrame(endpoints))
    csv('原窗口_收益复算.csv', pd.DataFrame(checks))
    csv('完整逐日行情_波动来源与已知货币.csv', daily.reset_index())
    macro_cols = ['stat_month', 'published_at', 'm1_yoy_pp', 'm2_yoy_pp', 'spread_pp', 'delta3_spread_pp',
                  'd3_relative_current_log_pp', 'd3_relative_base_revision_log_pp',
                  'credit_3_corporate_total_yoy_change_yi', 'credit_3_corporate_long_yoy_change_yi',
                  'credit_3_corporate_short_yoy_change_yi', 'credit_3_bills_yoy_change_yi',
                  'credit_3_household_long_yoy_change_yi', 'orders_first_release_value',
                  'context_fdr_policy_gap_bp_mean_change', 'past_return60',
                  'rv20', 'downside20', 'downside_window_state']
    csv('两期货币_各自公布后才可用.csv', monthly.loc[monthly.stat_month.isin(['2019-03', '2019-04']), macro_cols])
    verification = {'at': now(), 'status': 'PASS_FIXED_WINDOW_ARITHMETIC_AND_CLOCKS',
                    'monthly_rows_retained': len(monthly), 'daily_rows': len(daily),
                    'known_money_endpoints': len(clock_table), 'origin_windows': checks,
                    'd20_square_change_max_error': np.nanmax(np.abs(daily.d20_square_change5 - daily.d20_square_change5_rebuilt)),
                    'money_may9_intraday_update_preserved': True,
                    'historical_immutable_vintage_proven': False,
                    'causal_identification': False, 'independent_validation': False, 'goal_achieved': False}
    assert verification['d20_square_change_max_error'] < 1e-12
    save('verification.json', verification)
    print(json.dumps(clean(verification), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    modes = {'freeze': freeze, 'collect': collect, 'build': build}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('用法：python research/index_information_sequence_v27.py freeze|collect|build')
    modes[sys.argv[1]]()
