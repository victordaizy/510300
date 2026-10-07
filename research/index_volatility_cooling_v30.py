"""沿用全部同公告周对，拆解指数总波动下降的来源与原剩余路径。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_index_volatility_cooling_v30'
V22 = ROOT / 'reports/research/510300_within_release_volatility_v22'
V29 = ROOT / 'reports/research/510300_fiscal_money_transmission_v29'
EPS = 1e-12
UP = '上涨平方项收缩主导'
DOWN = '下跌平方项收缩主导'
MEAN = '均值修正项主导'
TIE = '下降贡献并列'
OTHER = '总波动未下降'
STATES = [UP, DOWN, MEAN, TIE, OTHER]
PHASES = ['2018-2021', '2022-2024', '2025-2026']


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
    return x


def save(name, data):
    (OUT / name).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(OUT / 'results' / name, index=False, encoding='utf-8-sig')


def freeze():
    if OUT.exists():
        raise RuntimeError('本轮目录已存在，不覆盖冻结范围。')
    for folder in ['inputs', 'results', 'code', 'figures']:
        (OUT / folder).mkdir(parents=True)
    protocol = {
        'at': now(), 'previous_turn': 'V29_COMPLETE_BOUNDED_DIAGNOSTIC_PROGRESS',
        'question': '同一货币公告的原相邻周中，总波动下降由上涨平方、下跌平方或均值修正哪项主导？当时价格、信用及后续原路径分别怎样？',
        'prior_work': 'V3已经完成每日三项恒等式及总波动上升分组，未细分总波动未上升。V22完成D20下降的同公告周对及现金金额桥。本轮只细分原总波动下降，不重拟合已失败模型。',
        'members': '原V22全部336周对、445周与672条E0/E1路径不变，连同V29原104个月741列完整保存。',
        'identity': 'RV20平方=20/19*(U20平方+D20平方-252*日收益均值平方)。变化取两原观察点相减，间隔沿实际交易日数，不能强称五日。',
        'cooling_states': '总方差变化小于-1e-12时，三个贡献中最负者为主导；差在1e-12内并列。其余全部为未下降。主导是算术定义，非经济原因。',
        'window': '逐项记录两观察点之间进入与退出的收益平方，保留均值项。另保存全部20日原收益成员。',
        'macro': '同周对共享M1/M2公告，不意味着其他消息相同。复用原多层资料和更新标记；连接同公告最近三月信用，先验检查其公告已早于每对前点。',
        'outcomes': '仅复用原E0/E1完整20日、原终点剩余、尾段、途中最差收盘及未来D，不加期限和入场规则。所有待成熟保留。',
        'weighting': '每个状态内按货币发布周期等权，周期内周对等权；保留2018至2021、2022至2024、2025至2026三个固定时期。',
        'contrast': '仅比较下跌项收缩主导减上涨项收缩主导，E0的后20日收益和后D20。两组各至少24周期才用V22保存的六周期区块2000次计算描述区间；E1列点估计及组样本。',
        'background': '保留所有既有信用订单背景及三个月信用方向；只汇总背景覆盖和连续价格位置，不事后按牛熊或最好阈值筛选。',
        'nonoverlap': '直接复用V22按日期固定选定的78周对成员，不因新状态改选相位。',
        'cases': '原2019-03、2020-11、2021-01、2021-05、2024-08、2025-07六例，以及V29全部2025-06至10阶段；保存所有匹配周对，不挑最好与最差。',
        'historical_outcomes_seen': True, 'independent_validation': False, 'new_models': 0,
        'new_accounts': 0, 'goal_achieved': False, 'source_collection': '不新增行情或公司资料，使用冻结本地数据。',
    }
    save('protocol.json', protocol)
    sources = {
        'monthly.csv': V29 / 'inputs/monthly.csv', 'market.csv': V22 / 'inputs/market.csv',
        'weekly.csv': V22 / 'results/445周_原多层背景与利率利润补充.csv',
        'pairs.csv': V22 / 'results/336对_风险变化已实现价格与多层背景.csv',
        'paths.csv': V22 / 'results/672条_原终点剩余与延长尾段金额桥.csv',
        'nonoverlap.csv': V22 / 'results/按日期固定的不重叠周对_全部成员.csv',
        'block_draws.npz': V22 / 'results/固定六周期区块索引.npz',
        'policy_nodes.csv': V22 / 'results/周对之间_已有政策目录节点.csv',
    }
    receipts = []
    for name, source in sources.items():
        dest = OUT / 'inputs' / name
        shutil.copy2(source, dest)
        receipts.append({'name': name, 'source': str(source), 'sha256': digest(dest)})
    save('freeze.json', {'at': now(), 'protocol_sha256': digest(OUT / 'protocol.json'), 'inputs': receipts})
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    print('已固定原336周对及总波动下降的算术分解，不按新结果改变成员。')


def verify_inputs():
    frozen = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == frozen['protocol_sha256']
    for x in frozen['inputs']:
        assert digest(OUT / 'inputs' / x['name']) == x['sha256']


def features(market):
    z = pd.DataFrame({'date': market.date})
    z['return'] = (market.close + market.dividend) / market.close.shift(1) - 1
    z['up_square'] = z['return'].clip(lower=0) ** 2
    z['down_square'] = z['return'].clip(upper=0) ** 2
    z['up2'] = z.up_square.rolling(20).mean() * 252
    z['down2'] = z.down_square.rolling(20).mean() * 252
    z['mean_daily_return'] = z['return'].rolling(20).mean()
    z['mean_correction'] = -252 * z.mean_daily_return ** 2
    z['variance'] = z['return'].rolling(20).var(ddof=1) * 252
    z['identity_error'] = z.variance - (z.up2 + z.down2 + z.mean_correction) * 20 / 19
    assert z.identity_error.abs().max() < EPS
    return z


def label(row):
    if row['delta_variance'] >= -EPS:
        return OTHER
    contributions = np.array([row['change_' + k] for k in ['up2', 'down2', 'mean_correction']])
    lowest = contributions.min()
    winners = np.flatnonzero(np.abs(contributions - lowest) <= EPS)
    return [UP, DOWN, MEAN][winners[0]] if len(winners) == 1 else TIE


def build():
    verify_inputs()
    market = pd.read_csv(OUT / 'inputs/market.csv')
    weekly = pd.read_csv(OUT / 'inputs/weekly.csv')
    monthly = pd.read_csv(OUT / 'inputs/monthly.csv').set_index('stat_month')
    pairs = pd.read_csv(OUT / 'inputs/pairs.csv')
    f = features(market)
    locations = {d: i for i, d in enumerate(f.date)}
    csv('3476日_原收益与方差组成.csv', f)
    fields = ['credit_3_corporate_long_yoy_change_yi', 'credit_3_household_long_yoy_change_yi', 'credit_3_corporate_long_direction',
              'spread_pp', 'delta3_spread_pp', 'd3_relative_current_log_pp', 'd3_relative_base_revision_log_pp']
    assert all(c in monthly.columns for c in fields)
    rows, members = [], []
    for source in pairs.to_dict('records'):
        a, b = locations[source['early_observation_date']], locations[source['late_observation_date']]
        first, last = f.iloc[a], f.iloc[b]
        gap = b - a
        assert gap == source['snapshot_gap_sessions'] and 0 < gap < 20
        row = dict(source, delta_variance=float(last.variance - first.variance), delta_rv=float(np.sqrt(last.variance) - np.sqrt(first.variance)))
        for key in ['up2', 'down2', 'mean_correction', 'variance', 'mean_daily_return']:
            row['computed_early_' + key] = first[key]
            row['computed_late_' + key] = last[key]
        for key in ['up2', 'down2', 'mean_correction']:
            row['change_' + key] = (last[key] - first[key]) * 20 / 19
        for key in ['up', 'down']:
            entered = float(f.iloc[a+1:b+1][key + '_square'].sum())
            exited = float(f.iloc[a-19:b-19][key + '_square'].sum())
            row['entered_' + key + '_square'] = entered
            row['exited_' + key + '_square'] = exited
            row[key + '_rolling_error'] = row['change_' + key + '2'] - (entered - exited) * 252 / 19
        row['change_identity_error'] = row['delta_variance'] - sum(row['change_' + key] for key in ['up2', 'down2', 'mean_correction'])
        row['cooling_state'] = label(row)
        row['total_down_but_downside_increases'] = row['delta_variance'] < -EPS and row['change_down2'] > EPS
        row['rv_reuse_error'] = max(abs(np.sqrt(first.variance) - source['early_v_rv20']), abs(np.sqrt(last.variance) - source['late_v_rv20']))
        row['downside_reuse_error'] = max(abs(np.sqrt(first.down2) - source['early_v_downside20']), abs(np.sqrt(last.down2) - source['late_v_downside20']))
        m = monthly.loc[source['stat_month']]
        assert pd.Timestamp(m.available_at_upper_bound) <= pd.Timestamp(source['early_snapshot_at'])
        row['money_available_at'] = m.available_at_upper_bound
        for key in fields:
            row['money_' + key] = m[key]
        assert np.isclose(source['early_delta3_spread_pp'], m.delta3_spread_pp, equal_nan=True)
        rows.append(row)
        # 原20日成员全部保存，重叠成员明确标记，逐笔核对进入和退出。
        for point, end in [('early', a), ('late', b)]:
            for i in range(end-19, end+1):
                q = f.iloc[i]
                role = 'EXITED' if i <= b-20 else ('ENTERED' if i > a else 'COMMON')
                members.append({'pair_id': source['pair_id'], 'point': point, 'date': q.date,
                                'membership': role, 'daily_return': q['return'], 'up_square': q.up_square, 'down_square': q.down_square})
    data = pd.DataFrame(rows)
    csv('336对_总波动下降来源与全部原背景.csv', data)
    csv('13440条_两端原20日收益成员.csv', pd.DataFrame(members))
    path = pd.read_csv(OUT / 'inputs/paths.csv')
    extra = [c for c in data if c not in path or c == 'pair_id']
    full = path.merge(data[extra], on='pair_id', validate='many_to_one')
    csv('672条_原路径与降波来源完整连接.csv', full)
    original_nonoverlap = pd.read_csv(OUT / 'inputs/nonoverlap.csv')
    non = full[full.pair_id.isin(original_nonoverlap.pair_id) & full.clock.eq('E0')]
    assert set(non.pair_id) == set(original_nonoverlap.pair_id)
    csv('78对_沿用原日期不重叠成员.csv', non)
    cases = ['2019-03', '2020-11', '2021-01', '2021-05', '2024-08', '2025-07']
    csv('原六病例_全部相邻周.csv', full[full.stat_month.isin(cases)])
    csv('2025六月至十月_全部相邻周.csv', full[full.stat_month.between('2025-06', '2025-10')])
    check = {'at': now(), 'status': 'PASS_COOLING_DECOMPOSITION', 'monthly_rows': len(monthly), 'monthly_columns': len(monthly.columns) + 1,
             'weekly_rows': len(weekly), 'pairs': len(data), 'paths': len(full), 'mature_paths': int(full.status.eq('MATURE').sum()),
             'nonoverlap_pairs': len(non), 'source_members': len(members), 'cooling_state_counts': data.cooling_state.value_counts().to_dict(),
             'max_arithmetic_error': float(data[['up_rolling_error', 'down_rolling_error', 'change_identity_error']].abs().max().max()),
             'max_reuse_error': float(data[['rv_reuse_error', 'downside_reuse_error']].abs().max().max()),
             'total_down_but_downside_increases': int(data.total_down_but_downside_increases.sum()),
             'new_models': 0, 'new_accounts': 0, 'goal_achieved': False}
    assert check['max_arithmetic_error'] < EPS and check['max_reuse_error'] < EPS
    assert len(monthly) == 104 and len(weekly) == 445 and len(data) == 336 and len(full) == 672
    save('build_verification.json', check)
    print(json.dumps(clean(check), ensure_ascii=False, indent=2))


def weights(z):
    return (1 / z.groupby('stat_month').stat_month.transform('size')).to_numpy(float)


def weighted_stats(z, key):
    z = z[np.isfinite(pd.to_numeric(z[key], errors='coerce'))]
    if z.empty:
        return {key + suffix: np.nan for suffix in ['_mean', '_median', '_positive', '_min']}
    w, x = weights(z), z[key].to_numpy(float)
    order = np.argsort(x, kind='stable')
    median = np.interp(.5, (np.cumsum(w[order]) - w[order] / 2) / w.sum(), x[order])
    return {key + '_mean': np.average(x, weights=w), key + '_median': median,
            key + '_positive': np.average(x > 0, weights=w), key + '_min': x.min()}


def analyze():
    verify_inputs()
    full = pd.read_csv(OUT / 'results/672条_原路径与降波来源完整连接.csv')
    non = pd.read_csv(OUT / 'results/78对_沿用原日期不重叠成员.csv')
    metrics = ['observed_close_return', 'late_past_return20', 'late_past_return60', 'late_v_rv20', 'late_v_downside20',
               'late20_return', 'late20_worst', 'late20_downside', 'common_return', 'new_tail_contribution']
    distributions = []
    for sampling, base in [('原全部周对', full), ('原固定不重叠', non)]:
        for phase in PHASES:
            for clock in ['E0', 'E1']:
                if sampling == '原固定不重叠' and clock == 'E1':
                    continue
                pool = base[base.analysis_period.eq(phase) & base.clock.eq(clock) & base.status.eq('MATURE')]
                for state in ['全部状态', '全部总波动下降', *STATES]:
                    z = pool if state == '全部状态' else (pool[pool.delta_variance < -EPS] if state == '全部总波动下降' else pool[pool.cooling_state.eq(state)])
                    row = {'sampling': sampling, 'period': phase, 'clock': clock, 'state': state, 'pairs': len(z), 'cycles': z.stat_month.nunique(),
                           'negative_future_count': int(z.late20_return.lt(0).sum()), 'total_down_but_downside_increases': int(z.total_down_but_downside_increases.sum())}
                    for key in metrics:
                        row.update(weighted_stats(z, key))
                    distributions.append(row)
    csv('全部时期及下降来源_原路径周期等权分布.csv', pd.DataFrame(distributions))
    backgrounds = []
    for (phase, state, context, credit), z in full[full.clock.eq('E0')].groupby(
            ['analysis_period', 'cooling_state', 'late_joint_credit_orders_state', 'money_credit_3_corporate_long_direction'], dropna=False):
        row = {'period': phase, 'state': state, 'original_credit_orders_context': context, 'corporate_long_3month_direction': credit,
               'pairs': len(z), 'cycles': z.stat_month.nunique(), 'mature_pairs': int(z.status.eq('MATURE').sum())}
        for key in ['late_past_return60', 'money_delta3_spread_pp', 'money_d3_relative_current_log_pp', 'money_d3_relative_base_revision_log_pp',
                    'late_orders_first_release_value', 'money_credit_3_corporate_long_yoy_change_yi', 'late20_return']:
            row.update(weighted_stats(z, key))
        backgrounds.append(row)
    csv('全部信用订单背景及降波来源_完整覆盖.csv', pd.DataFrame(backgrounds))
    contrasts = []
    frozen_draws = np.load(OUT / 'inputs/block_draws.npz')
    for phase in PHASES:
        for clock in ['E0', 'E1']:
            pool = full[full.analysis_period.eq(phase) & full.clock.eq(clock) & full.status.eq('MATURE')]
            cycles = sorted(pool.stat_month.unique())
            a, b = pool[pool.cooling_state.eq(DOWN)], pool[pool.cooling_state.eq(UP)]
            ids = {m: i for i, m in enumerate(cycles)}
            for key in ['late20_return', 'late20_downside']:
                row = {'period': phase, 'clock': clock, 'comparison': DOWN + '减' + UP, 'outcome': key,
                       'down_pairs': len(a), 'up_pairs': len(b), 'down_cycles': a.stat_month.nunique(), 'up_cycles': b.stat_month.nunique(),
                       'mean_difference': np.nan, 'low': np.nan, 'high': np.nan, 'valid_draws': 0, 'missing_draws': 0,
                       'interval_status': 'NOT_COMPUTED_E1_OR_GROUP_UNDER_24_CYCLES'}
                if len(a) and len(b):
                    wa, wb = weights(a), weights(b)
                    xa, xb = a[key].to_numpy(float), b[key].to_numpy(float)
                    row['mean_difference'] = np.average(xa, weights=wa) - np.average(xb, weights=wb)
                    if clock == 'E0' and min(a.stat_month.nunique(), b.stat_month.nunique()) >= 24:
                        draws = frozen_draws[phase]
                        assert draws.shape == (2000, len(cycles))
                        ia, ib = a.stat_month.map(ids).to_numpy(), b.stat_month.map(ids).to_numpy()
                        values = []
                        for selected in draws:
                            count = np.bincount(selected, minlength=len(cycles))
                            u, v = wa * count[ia], wb * count[ib]
                            values.append(np.average(xa, weights=u) - np.average(xb, weights=v) if u.sum() and v.sum() else np.nan)
                        values = np.array(values)
                        valid = values[np.isfinite(values)]
                        row.update(low=np.quantile(valid, .025), high=np.quantile(valid, .975), valid_draws=len(valid), missing_draws=2000-len(valid),
                                   interval_status='DESCRIPTIVE_SAVED_6_CYCLE_BLOCKS')
                contrasts.append(row)
    csv('固定两来源对照_后收益与下行风险.csv', pd.DataFrame(contrasts))
    save('analysis_verification.json', {'at': now(), 'distribution_rows': len(distributions), 'background_rows': len(backgrounds),
                                       'fixed_contrasts': len(contrasts), 'saved_bootstrap_reused': True, 'independent_validation': False,
                                       'new_models': 0, 'new_accounts': 0, 'goal_achieved': False})
    print(pd.DataFrame(distributions).query("sampling == '原全部周对' and clock == 'E0'")[
        ['period', 'state', 'pairs', 'cycles', 'observed_close_return_mean', 'late20_return_mean', 'late20_downside_mean']].to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == '__main__':
    commands = {'freeze': freeze, 'build': build, 'analyze': analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit('用法：python research/index_volatility_cooling_v30.py freeze|build|analyze')
    commands[sys.argv[1]]()
