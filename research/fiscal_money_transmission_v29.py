"""核对2025年连续分化阶段的财政用途、执行与货币资产负债表。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import shutil
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.money_balance_transmission_v8 import DCS, CB, ODCS, COUNTER, extract

OUT = ROOT / 'reports/research/510300_fiscal_money_transmission_v29'
V8 = ROOT / 'reports/research/510300_money_balance_transmission_v8'
FISCAL = ROOT / 'reports/research/510300_fiscal_execution_state_20d_v1'
MONTHS = [f'2025-{i:02d}' for i in range(6, 11)]
TABLE_NAMES = ['存款性公司概览_2025', '货币当局资产负债表_2025', '其他存款性公司资产负债表_2025']
NEW_SOURCES = [
    {'id': 'mof_h1_press', 'date': '2025-07-25', 'url': 'https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202507/t20250725_3968586.htm'},
    {'id': 'mof_q3_press', 'date': '2025-10-17', 'url': 'https://m.mof.gov.cn/czxw/202510/t20251017_3974415.htm'},
]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.isoformat()
    return x


def save(name, data):
    (OUT / name).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(OUT / 'results' / name, index=False, encoding='utf-8-sig')


def freeze():
    if OUT.exists():
        raise RuntimeError('本轮目录已存在，不覆盖既有范围。')
    for name in ['inputs', 'sources', 'results', 'code', 'figures']:
        (OUT / name).mkdir(parents=True)
    protocol = {
        'at': now(), 'study': '510300_FISCAL_MONEY_TRANSMISSION_V29',
        'previous_goal_turn_classification': 'PROGRESS',
        'question': '2025年6至10月剪刀差持续改善而企业中长期持续少增，财政融资、债务置换、执行支出与货币持有结构分别提供什么证据？',
        'selection': '阶段来自V28信用结构的连续分化，不按后续收益选取；全部五个月保留。原104个月741列底表完整复制。',
        'prior_work': '复用V8截至8月的银行资产负债测量与既有127期财政原文；不重跑已失败财政收益模型。新加9至10月余额、五个月财政实际执行与公布时钟、上半年及前三季度用途进展。',
        'historical_outcomes_already_seen': True,
        'source_roles': '2025年央行年度PDF为12月网址版本，对6至10月均是事后结构解释，不得进入原起点输入。财政月报按既有日期日终上界连接；原因发布会同样取日终，不把实施日期当作公开日期。',
        'balances': '读原表2025年5至10月，5月作为差分基点；只计算6至10月单月和五个月累计余额差。M1分项、M2资产负债对应项与政府净债权均保持恒等式及舍入残差。',
        'fiscal': '两本预算收入和支出分别记录，月度金额为相邻原累计公告差，不冒充单月首次原报值。收支差不称财政赤字，不与央行政府存款强行配平。',
        'use': '分开政府债发行、置换资金使用、预算下达、预算实际支出、银行资本补充和实体采购；同一资金在不同环节不能相加。未披露用途份额保持未知。',
        'clocks': '每个原货币21点快照只连接当时最近已公开财政月份；同统计月但后来公布的财政资料另列事后。财政执行和原货币可能面对不同统计月。',
        'market': '沿原五个月E0/E1及20交易日收益、此前60日与RV/D20，不新增账户、时间分组收益或解释回归。',
        'limits': '资产负债恒等式不识别独立因果，净余额不是逐笔资金流，基金支出含金融交易不能全部当新增需求。财政和政府债不是510300的净买入。',
        'new_models': 0, 'new_accounts': 0, 'orders_authorized': False,
        'independent_validation': False, 'goal_achieved': False,
    }
    save('protocol.json', protocol)
    files = {
        'monthly.csv': ROOT / 'reports/research/510300_index_money_information_lifetime_v28/inputs/monthly.csv',
        'fiscal.parquet': FISCAL / 'inputs/fiscal.parquet',
        'prior_balance.csv': V8 / 'results/20个月_货币与银行资产负债原金额.csv',
    }
    inputs = []
    for name, src in files.items():
        dest = OUT / 'inputs' / name
        shutil.copy2(src, dest)
        inputs.append({'name': name, 'source': str(src), 'sha256': digest(dest)})
    sources = []
    for name in TABLE_NAMES:
        for suffix in ['.pdf', '.txt', '.pdf.receipt.json']:
            src, dest = V8 / 'sources' / (name + suffix), OUT / 'sources' / (name + suffix)
            shutil.copy2(src, dest)
            sources.append({'name': dest.name, 'source': str(src), 'sha256': digest(dest), 'role': 'EX_POST_ANNUAL_TABLE_NOT_ORIGIN_INPUT'})
    fiscal = pd.read_parquet(OUT / 'inputs/fiscal.parquet')
    for r in fiscal[fiscal.reference_period.between('2025-05', '2025-10')].to_dict('records'):
        src = FISCAL / r['raw']
        assert digest(src) == r['raw_sha256']
        dest = OUT / 'sources' / ('fiscal_' + r['reference_period'] + '.html')
        shutil.copy2(src, dest)
        sources.append({'name': dest.name, 'source': str(src), 'url': r['source_url'], 'sha256': digest(dest), 'available_at': r['available_at']})
    save('freeze.json', {'at': now(), 'protocol_sha256': digest(OUT / 'protocol.json'), 'inputs': inputs, 'copied_sources': sources})
    save('new_source_plan.json', NEW_SOURCES)
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    print('已固定2025年6至10月完整财政货币传导阶段，保留原104个月和原收益。')


def collect():
    path = OUT / 'new_source_receipts.json'
    receipts = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    tried = {s['id'] for s in receipts}
    for s in NEW_SOURCES:
        if s['id'] in tried:
            continue
        r = {**s, 'retrieved_at': now(), 'first_historical_vintage_verified': False}
        try:
            response = requests.get(s['url'], timeout=25, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            dest = OUT / 'sources' / (s['id'] + '.html')
            dest.write_bytes(response.content)
            soup = BeautifulSoup(response.content.decode('utf-8'), 'html.parser')
            for el in soup(['script', 'style']):
                el.decompose()
            txt = soup.get_text('\n', strip=True)
            dest.with_suffix('.txt').write_text(txt, encoding='utf-8')
            r.update(status='SAVED_PENDING_CONTENT_REVIEW', sha256=digest(dest), chars=len(txt))
        except Exception as error:
            r.update(status='SOURCE_NOT_SAVED', error=str(error))
        receipts.append(r)
        save('new_source_receipts.json', receipts)
        print(s['id'], r['status'], r.get('chars', ''), flush=True)


def table(name, mapping):
    txt = (OUT / 'sources' / (name + '.txt')).read_text(encoding='utf-8').split('注：', 1)[0]
    data = {k: extract(txt, label, 11) for k, label in mapping.items()}
    return pd.DataFrame(data, index=[f'2025-{i:02d}' for i in range(1, 12)])


def build():
    freeze = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == freeze['protocol_sha256']
    for f in freeze['inputs']:
        assert digest(OUT / 'inputs' / f['name']) == f['sha256']
    for f in freeze['copied_sources']:
        assert digest(OUT / 'sources' / f['name']) == f['sha256']
    dm = dict(DCS, demand_total='活期存款', time_total='定期存款', payment_reserves='非银行支付机构客户备付金')
    balance = table('存款性公司概览_2025', dm).join(table('货币当局资产负债表_2025', CB)).join(table('其他存款性公司资产负债表_2025', ODCS)).loc['2025-05':'2025-10'].copy()
    balance.index.name = 'stat_month'
    balance['personal_demand_derived'] = balance.demand_total - balance.corporate_demand
    balance['personal_time_derived'] = balance.personal_total - balance.personal_demand_derived
    balance['government_identity_error'] = balance.government_net - balance.cb_gov_claims - balance.odcs_gov_claims + balance.cb_gov_deposits
    balance['m1_identity_error'] = balance.m1_table - balance.m0 - balance.demand_total - balance.payment_reserves
    balance['counterpart_rounding'] = balance.m2 - (balance.nfa + balance.government_net + balance.nonfinancial_claims + balance.other_financial_claims - balance.excluded_deposits - balance.bonds - balance.capital - balance.other_net)
    balance['other_deposits_residual'] = balance.other_deposits_table - balance.odcs_financial_deposits_m2
    assert balance[['government_identity_error', 'm1_identity_error', 'counterpart_rounding']].abs().max().max() < .051
    prior = pd.read_csv(OUT / 'inputs/prior_balance.csv').set_index('stat_month')
    comparable_cols = [c for c in balance.columns if c in prior.columns and pd.api.types.is_numeric_dtype(balance[c])]
    old_error = (balance.loc['2025-05':'2025-08', comparable_cols] - prior.loc['2025-05':'2025-08', comparable_cols]).abs().max().max()
    assert old_error < 1e-7
    csv('五月基点及六月至十月_事后原表余额.csv', balance.reset_index())
    differences = balance.diff().loc[MONTHS]
    differences['role'] = 'EX_POST_ANNUAL_TABLE_BALANCE_CHANGE_NOT_ORIGIN_INPUT'
    csv('六月至十月_逐月货币与政府债权存款变化.csv', differences.reset_index())
    counterpart = []
    for month in MONTHS:
        for key, sign in COUNTER.items():
            counterpart.append({'stat_month': month, 'item': key, 'balance_change_yi': differences.loc[month, key], 'sign': sign,
                                'm2_arithmetic_contribution_yi': differences.loc[month, key] * sign})
    csv('五个月_M2全部资产负债对应项.csv', pd.DataFrame(counterpart))
    f = pd.read_parquet(OUT / 'inputs/fiscal.parquet')
    selected = f[f.reference_period.between('2025-05', '2025-10')].copy().sort_values('reference_period')
    parsed = []
    for r in selected.to_dict('records'):
        path = OUT / 'sources' / ('fiscal_' + r['reference_period'] + '.html')
        text = BeautifulSoup(path.read_bytes().decode('utf-8'), 'html.parser').get_text(' ', strip=True)
        compact = re.sub(r'\s+', '', text).replace(',', '').replace('，', '')
        amounts = {}
        for key, label in [('general_revenue_ytd_yi', '全国一般公共预算收入'), ('general_expenditure_ytd_yi', '全国一般公共预算支出'),
                           ('fund_revenue_ytd_yi', '全国政府性基金预算收入'), ('fund_expenditure_ytd_yi', '全国政府性基金预算支出')]:
            values = re.findall(re.escape(label) + r'([0-9.]+)亿元', compact)
            assert len(set(values)) == 1, (r['reference_period'], key, values)
            amounts[key] = float(values[0])
        assert amounts['general_expenditure_ytd_yi'] == r['general_expenditure_ytd_yi']
        assert amounts['fund_expenditure_ytd_yi'] == r['fund_expenditure_ytd_yi']
        parsed.append({'stat_month': r['reference_period'], 'available_at': r['available_at'], 'source_url': r['source_url'],
                       'general_ytd_yoy_pp': r['general_headline_yoy_pp'], 'fund_ytd_yoy_pp': r['fund_headline_yoy_pp'], **amounts})
    fiscal = pd.DataFrame(parsed).set_index('stat_month')
    amounts = [c for c in fiscal.columns if c.endswith('_ytd_yi')]
    for key in amounts:
        fiscal[key.replace('_ytd_', '_month_')] = fiscal[key].diff()
    fiscal['general_month_expenditure_minus_revenue_yi'] = fiscal.general_expenditure_month_yi - fiscal.general_revenue_month_yi
    fiscal['fund_month_expenditure_minus_revenue_yi'] = fiscal.fund_expenditure_month_yi - fiscal.fund_revenue_month_yi
    csv('两本预算_累计及跨原公告推导单月.csv', fiscal.reset_index())
    original = pd.read_csv(OUT / 'inputs/monthly.csv')
    fiscal_clock = pd.to_datetime(f.available_at, utc=True)
    snapshots = []
    for a in original[original.stat_month.isin(MONTHS)].to_dict('records'):
        snap = pd.Timestamp(a['snapshot_at'])
        available = f.loc[fiscal_clock <= snap].sort_values('reference_period').iloc[-1]
        same = fiscal.loc[a['stat_month']]
        row = {'stat_month': a['stat_month'], 'snapshot_at': a['snapshot_at'], 'money_published_at': a['published_at'],
               'latest_fiscal_month_at_original_snapshot': available.reference_period,
               'latest_fiscal_available_at': available.available_at,
               'known_general_ytd_expenditure_yi': available.general_expenditure_ytd_yi,
               'known_general_ytd_yoy_pp': available.general_headline_yoy_pp,
               'known_fund_ytd_expenditure_yi': available.fund_expenditure_ytd_yi,
               'known_fund_ytd_yoy_pp': available.fund_headline_yoy_pp,
               'same_stat_month_fiscal_available_at': same.available_at,
               'same_stat_month_fiscal_known_at_origin': pd.Timestamp(same.available_at) <= snap,
               'same_stat_month_fiscal_role': 'EX_POST_IF_NOT_AVAILABLE_AT_ORIGINAL_SNAPSHOT',
               'annual_bank_table_role': 'EX_POST_VERSION_NOT_ORIGINAL_INPUT'}
        for key in ['spread_pp', 'delta3_spread_pp', 'd3_relative_current_log_pp', 'd3_relative_base_revision_log_pp',
                    'credit_3_corporate_long_yoy_change_yi', 'credit_3_household_long_yoy_change_yi',
                    'orders_first_release_value', 'past_return60', 'rv20', 'downside20', 'E0_20_return', 'E1_20_return']:
            row[key] = a[key]
        snapshots.append(row)
    snapshots = pd.DataFrame(snapshots)
    csv('五个原观察点_当时财政与后来同月资料.csv', snapshots)
    joint = snapshots.merge(fiscal.reset_index().add_prefix('same_month_'), left_on='stat_month', right_on='same_month_stat_month', validate='one_to_one')
    joint = joint.merge(differences.reset_index().add_prefix('expost_bank_delta_'), left_on='stat_month', right_on='expost_bank_delta_stat_month', validate='one_to_one')
    csv('五个月完整链_当时信用价格与事后财政货币.csv', joint)
    total = (balance.loc['2025-10'] - balance.loc['2025-05']).to_dict()
    save('results/五个月余额变化合计.json', total)
    verify = {'at': now(), 'status': 'PASS_FISCAL_AND_MONEY_MEASUREMENT', 'original_months_preserved': len(original),
              'original_columns_preserved': len(original.columns), 'phase_months': MONTHS, 'fiscal_reports_parsed': len(fiscal),
              'prior_v8_balance_max_error': old_error,
              'max_balance_identity_error_yi': float(balance[['government_identity_error', 'm1_identity_error', 'counterpart_rounding']].abs().max().max()),
              'same_month_fiscal_already_available_count': int(snapshots.same_stat_month_fiscal_known_at_origin.sum()),
              'all_fiscal_admitted_by_original_snapshot': bool((pd.to_datetime(snapshots.latest_fiscal_available_at, utc=True) <= pd.to_datetime(snapshots.snapshot_at, utc=True)).all()),
              'annual_tables_admitted_as_origin_inputs': False, 'historical_first_vintage_verified': False,
              'new_models': 0, 'new_accounts': 0, 'goal_achieved': False}
    save('verification.json', verify)
    print(json.dumps(clean(verify), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    modes = {'freeze': freeze, 'collect': collect, 'build': build}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('用法：python research/fiscal_money_transmission_v29.py freeze|collect|build')
    modes[sys.argv[1]]()
