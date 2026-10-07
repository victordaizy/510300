"""核对原20交易日窗口内的货币信息更新，不改变任何原收益窗口。"""
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
OUT = ROOT / 'reports/research/510300_index_money_information_lifetime_v28'
PARENT = ROOT / 'reports/research/510300_index_information_sequence_v27'
CONTEXT = ['m1_yoy_pp', 'm2_yoy_pp', 'spread_pp', 'delta3_spread_pp',
           'd3_relative_current_log_pp', 'd3_relative_base_revision_log_pp',
           'credit_3_corporate_total_yoy_change_yi', 'credit_3_corporate_long_yoy_change_yi',
           'credit_3_corporate_short_yoy_change_yi', 'credit_3_bills_yoy_change_yi',
           'credit_3_household_long_yoy_change_yi', 'credit_3_corporate_long_direction',
           'credit_3_household_long_direction', 'credit_3_both_long',
           'orders_first_release_value', 'context_fdr_policy_gap_bp_mean_change',
           'past_return60', 'rv20', 'downside20', 'downside_window_state']


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


def period(stat_month):
    y = int(stat_month[:4])
    return '旧口径2018至2021' if y <= 2021 else ('旧口径2022至2024' if y <= 2024 else '新口径2025至2026')


def local_time(date, hour=0, minute=0):
    return pd.Timestamp(date).normalize().tz_localize('Asia/Shanghai') + pd.Timedelta(hours=hour, minutes=minute)


def freeze():
    if OUT.exists():
        raise RuntimeError('研究目录已存在，不能覆盖本轮固定范围。')
    for name in ['inputs', 'results', 'code', 'figures']:
        (OUT / name).mkdir(parents=True)
    protocol = {
        'at': now(), 'study': '510300_INDEX_MONEY_INFORMATION_LIFETIME_V28',
        'previous_goal_turn_classification': 'PROGRESS',
        'previous_turn_evidence': 'V27完成2019年原窗口的信用、信息顺序和价格波动核对；尚未独立验证方向优势。',
        'question': '全部原20交易日窗口有多少跨过后续货币公告，期间货币与信用结构如何更新，原完整收益在信息更新前后如何分布？',
        'novelty': 'V6核对有限政策目录，V22核对同一货币发布周期中的周对；本轮完整覆盖每月原未来20日中的所有后续货币公告。不是重跑V14条件拟合，也不营救既有失败模型。',
        'population': '全部104个月、E0及E1均保留，旧新M1口径分开；所有原已知列原样复制。',
        'historical_outcomes_already_seen': True,
        'future_information_role': '后续公告数值仅解释结果窗，严禁回填原入场。后续月份的全套价格、订单等列按其自身原21点快照标注，不能全部冒充货币公告瞬间已知。',
        'windows': '严格沿原E0/E1入场开盘和第20交易日收盘；未成熟维持原状态，不能填0或截短。',
        'clocks': '主时钟沿原网页公开上界。敏感性把每份后续货币的公开时间推到该北京时间日期23:59:59，不改变原入场退出日或选择优者。后续公告只有严格晚于入场开盘且不晚于原末日收盘才计入。',
        'partition': '对有后续公告的原窗口，用第一次后续公告前最后一个已发生收盘划分前段；从此收盘至首次不早于公告的交易日收盘为边界段；余下为后段。盘中公告时边界含公告前交易，不作新闻因果收益。无后续公告时整段归前段，仍不等于没有其他新信息。',
        'cash': '固定股数、应享现金不再投资，入场日除息不享权益。三段以同一原入场价为分母，贡献之和等于原20日收益。不是三段独立持有期或退出方案。',
        'credit_transitions': '全部103个相邻统计月对都记录，原舍入界和2023范围断点保留。仅同一M1制度可计算剪刀差水平变化；定义转换不相减。记录剪刀差三月仍改善而企业中长期由明确多增转明确少增的全部反例，不按其收益筛选。',
        'volatility_at_update': '只用公告前已发生的最近收盘计算RV20、D20和新五日/前五日/退出五日；不把盘中发布日的全日行情预先纳入。',
        'summaries': '按旧前期2018至2021、旧后期2022至2024及新口径2025至2026、E0/E1及两种公告时钟分别列示。统计跨公告覆盖、可用天数、三段对原投入的算术贡献，不比较未来标签分组策略，不筛阈值、不回归拟合。',
        'overlap': 'E0/E1高度重叠，相邻月窗口也可能重叠；不把208行当208次独立实验。',
        'missing': '目录未记录的下一条信息保持未知；原2026-08无行情快照，原2026-07的E1未成熟。无货币更新不等于其他宏观政策不变。',
        'new_models': 0, 'new_accounts': 0, 'orders_authorized': False,
        'independent_validation': False, 'causal_identification': False, 'goal_achieved': False,
    }
    save('protocol.json', protocol)
    records = []
    for name in ['monthly.csv', 'market.csv']:
        src, dest = PARENT / 'inputs' / name, OUT / 'inputs' / name
        shutil.copy2(src, dest)
        records.append({'name': name, 'source': str(src), 'sha256': digest(dest)})
    save('freeze.json', {'at': now(), 'protocol_sha256': digest(OUT / 'protocol.json'), 'inputs': records})
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    print('已固定104个月原窗口的信息更新核对；不改收益期限，不用未来数据分组挑选收益。')


def load_inputs():
    freeze = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
    assert digest(OUT / 'protocol.json') == freeze['protocol_sha256']
    for f in freeze['inputs']:
        assert digest(OUT / 'inputs' / f['name']) == f['sha256']
    m = pd.read_csv(OUT / 'inputs/monthly.csv').sort_values('stat_month').reset_index(drop=True).copy()
    assert len(m) == 104 and m.stat_month.is_unique
    m['clock_main'] = pd.to_datetime(m.available_at_upper_bound, utc=True).dt.tz_convert('Asia/Shanghai')
    m['clock_day_end'] = m.clock_main.dt.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    assert m.clock_main.is_monotonic_increasing
    p = pd.read_csv(OUT / 'inputs/market.csv', parse_dates=['date']).set_index('date').sort_index()
    assert p.index.is_unique
    ret = (p.close + p.dividend) / p.close.shift() - 1
    assert np.nanmax(np.abs(ret - p.total_simple)) < 1e-9
    negative = ret.clip(upper=0).pow(2)
    p['rv20'] = ret.rolling(20).std(ddof=1) * np.sqrt(252)
    p['d20'] = np.sqrt(negative.rolling(20).sum() * 252 / 20)
    p['negative_latest5'] = negative.rolling(5).sum()
    p['negative_prior5'] = negative.rolling(5).sum().shift(5)
    p['negative_exited5'] = negative.rolling(5).sum().shift(20)
    p['d20_change5'] = p.d20.diff(5)
    return m, p


def transitions(monthly):
    rows = []
    for i in range(len(monthly) - 1):
        a, b = monthly.iloc[i], monthly.iloc[i + 1]
        same = a.training_regime == b.training_regime
        assert pd.Period(b.stat_month, freq='M') == pd.Period(a.stat_month, freq='M') + 1
        row = {'stat_month': a.stat_month, 'next_stat_month': b.stat_month, 'period': period(a.stat_month),
               'training_regime': a.training_regime, 'next_training_regime': b.training_regime,
               'publication': a.available_at_upper_bound, 'next_publication': b.available_at_upper_bound,
               'next_original_snapshot_at': b.snapshot_at, 'same_m1_definition': same,
               'spread_level_change_pp': b.spread_pp - a.spread_pp if same else np.nan,
               'both_delta3_positive': bool(same and a.delta3_spread_pp > 0 and b.delta3_spread_pp > 0),
               'corp_long_from_more_to_less': bool(a.credit_3_corporate_long_direction == '多增' and b.credit_3_corporate_long_direction == '少增'),
               'household_long_from_more_to_less': bool(a.credit_3_household_long_direction == '多增' and b.credit_3_household_long_direction == '少增'),
               'next_information_role': 'EX_POST_NEXT_RELEASE_NOT_ORIGINAL_INPUT',
               'non_money_context_role': 'EACH_ROWS_OWN_ORIGINAL_21H_SNAPSHOT_NOT_ALL_KNOWN_AT_PUBLICATION'}
        for c in CONTEXT:
            row['origin_' + c], row['next_' + c] = a[c], b[c]
        row['continued_delta3_improvement_but_corp_long_reversed'] = row['both_delta3_positive'] and row['corp_long_from_more_to_less']
        rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 103
    csv('103个相邻公告_货币信用与各自快照背景.csv', result)
    csv('剪刀差仍改善而中长期转少增_全部相邻反例.csv', result[result.continued_delta3_improvement_but_corp_long_reversed])
    return result


def build():
    monthly, market = load_inputs()
    change = transitions(monthly)
    price_close_times = market.index.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
    window_rows, daily_rows, event_rows = [], [], []
    error_rows = []
    for a in monthly.to_dict('records'):
        for e in ['E0', 'E1']:
            original_status = a[e + '_20_status']
            for scenario in ['main', 'day_end']:
                rec = {'stat_month': a['stat_month'], 'training_regime': a['training_regime'],
                       'period': period(a['stat_month']), 'clock': e, 'release_clock_scenario': scenario,
                       'entry_date': a[e + '_20_entry_date'], 'exit_date': a[e + '_20_exit_date'],
                       'original_status': original_status, 'original_return': a[e + '_20_return'],
                       'status': original_status, 'new_information_role': 'EX_POST_ONLY'}
                if original_status != 'MATURE':
                    window_rows.append(rec)
                    continue
                entry, end = pd.Timestamp(rec['entry_date']), pd.Timestamp(rec['exit_date'])
                start_at, end_at = local_time(entry, 9, 30), local_time(end, 15)
                q = market.loc[entry:end].copy()
                assert len(q) == 20
                origin_open = float(q.iloc[0].open)
                entitlement = q.dividend.copy()
                entitlement.iloc[0] = 0.0
                q['entitled_cash'] = entitlement.cumsum()
                q['share_value'] = q.close + q.entitled_cash
                wealth_values = np.r_[origin_open, q.share_value.to_numpy()]
                pnl = np.diff(wealth_values) / origin_open
                total = float(q.iloc[-1].share_value / origin_open - 1)
                assert abs(total - rec['original_return']) < 1e-9
                event_clock = monthly['clock_' + scenario]
                active = monthly.loc[(event_clock > start_at) & (event_clock <= end_at)].copy()
                available_at_entry = monthly.loc[event_clock <= start_at].sort_values('clock_' + scenario).iloc[-1]
                rec.update(sessions=20, original_source_still_latest_at_entry=available_at_entry.stat_month == a['stat_month'],
                           latest_stat_month_at_entry=available_at_entry.stat_month, future_release_count=len(active),
                           future_release_months='|'.join(active.stat_month), original_entry_open=origin_open,
                           cash_per_share=float(q.iloc[-1].entitled_cash), recomputed_return=total,
                           status='MATURE_PARTITION_COMPLETE')
                assert rec['original_source_still_latest_at_entry']
                for order, (_, b) in enumerate(active.iterrows(), 1):
                    tau = b['clock_' + scenario]
                    known_close_mask = price_close_times < tau
                    known_price_row = market.loc[known_close_mask].iloc[-1]
                    known_price_day = market.loc[known_close_mask].index[-1]
                    er = {'stat_month': a['stat_month'], 'clock': e, 'release_clock_scenario': scenario,
                          'event_order': order, 'new_stat_month': b.stat_month, 'publication_used': tau,
                          'source_publication_main': b.clock_main, 'new_m1_yoy_pp': b.m1_yoy_pp,
                          'new_m2_yoy_pp': b.m2_yoy_pp, 'new_spread_pp': b.spread_pp,
                          'new_delta3_spread_pp': b.delta3_spread_pp,
                          'new_corp_long_direction': b.credit_3_corporate_long_direction,
                          'new_corp_long_yoy_change_yi': b.credit_3_corporate_long_yoy_change_yi,
                          'new_household_long_direction': b.credit_3_household_long_direction,
                          'last_close_known_before_publication': known_price_day,
                          'last_close_at': price_close_times[market.index.get_loc(known_price_day)],
                          'known_rv20': known_price_row.rv20, 'known_d20': known_price_row.d20,
                          'known_negative_latest5': known_price_row.negative_latest5,
                          'known_negative_prior5': known_price_row.negative_prior5,
                          'known_negative_exited5': known_price_row.negative_exited5,
                          'role': 'FUTURE_CONTEXT_UPDATE_NOT_ORIGINAL_ENTRY_INPUT'}
                    assert er['last_close_at'] < tau
                    event_rows.append(er)
                if len(active):
                    first = active.iloc[0]
                    first_at = first['clock_' + scenario]
                    qt = q.index.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
                    pre_n = int((qt < first_at).sum())
                    assert 0 <= pre_n < 20
                    stages = ['公告前'] * pre_n + ['跨公告边界'] + ['边界后'] * (19 - pre_n)
                    pre_contrib = float(pnl[:pre_n].sum())
                    boundary_contrib = float(pnl[pre_n])
                    post_contrib = float(pnl[pre_n + 1:].sum())
                    rec.update(first_future_stat_month=first.stat_month, first_future_publication=first_at,
                               boundary_trade_date=q.index[pre_n], pre_close_sessions=pre_n,
                               boundary_sessions=1, post_boundary_sessions=19 - pre_n)
                else:
                    stages = ['无目录内后续货币公告'] * 20
                    pre_contrib, boundary_contrib, post_contrib = total, 0.0, 0.0
                    rec.update(first_future_stat_month=None, first_future_publication=None,
                               boundary_trade_date=None, pre_close_sessions=20,
                               boundary_sessions=0, post_boundary_sessions=0)
                rec.update(pre_contribution=pre_contrib, boundary_contribution=boundary_contrib,
                           post_contribution=post_contrib,
                           boundary_and_after_contribution=boundary_contrib + post_contrib,
                           identity_error=pre_contrib + boundary_contrib + post_contrib - total)
                assert abs(rec['identity_error']) < 1e-12
                for k, (date, bar) in enumerate(q.iterrows()):
                    daily_rows.append({'stat_month': a['stat_month'], 'clock': e, 'release_clock_scenario': scenario,
                                       'date': date, 'stage': stages[k], 'open': bar.open, 'close': bar.close,
                                       'cash_entitled_cumulative': bar.entitled_cash, 'fixed_share_value': bar.share_value,
                                       'pnl_contribution_on_original_capital': pnl[k]})
                window_rows.append(rec)
                error_rows.append({'stat_month': a['stat_month'], 'clock': e, 'release_clock_scenario': scenario,
                                   'original_recompute_error': total - rec['original_return'],
                                   'partition_error': rec['identity_error']})
    windows = pd.DataFrame(window_rows)
    csv('416条原窗口与时钟敏感性_全部保留.csv', windows)
    csv('全部成熟原窗口_逐日贡献与信息区间.csv', pd.DataFrame(daily_rows))
    csv('原窗口内所有后续货币公告_当时已知价格波动.csv', pd.DataFrame(event_rows))
    csv('原完整收益与分段恒等式复算.csv', pd.DataFrame(error_rows))
    summaries = []
    for (p, e, scenario), g in windows.groupby(['period', 'clock', 'release_clock_scenario'], sort=False):
        mature = g[g.original_status == 'MATURE']
        updated = mature[mature.future_release_count > 0]
        summaries.append({'period': p, 'clock': e, 'release_clock_scenario': scenario,
                          'all_original_months': len(g), 'mature_original_months': len(mature),
                          'pending_months': len(g) - len(mature), 'windows_with_money_update': len(updated),
                          'update_fraction_of_mature': len(updated) / len(mature) if len(mature) else np.nan,
                          'windows_with_two_or_more_updates': int((mature.future_release_count >= 2).sum()),
                          'updated_median_pre_close_sessions': updated.pre_close_sessions.median(),
                          'updated_median_post_boundary_sessions': updated.post_boundary_sessions.median(),
                          'updated_mean_original_return': updated.original_return.mean(),
                          'updated_mean_pre_contribution': updated.pre_contribution.mean(),
                          'updated_mean_boundary_contribution': updated.boundary_contribution.mean(),
                          'updated_mean_post_contribution': updated.post_contribution.mean(),
                          'updated_median_abs_boundary_and_after_contribution': updated.boundary_and_after_contribution.abs().median(),
                          'updated_abs_total_cancelled_case_count': int((updated.boundary_and_after_contribution.abs() > updated.original_return.abs()).sum()),
                          'statistic_role': 'ARITHMETIC_DESCRIPTION_NOT_CAUSAL_EFFECT_OR_STRATEGY'})
    csv('固定时期_信息更新覆盖及原收益位置.csv', pd.DataFrame(summaries))
    keys = ['stat_month', 'clock']
    compare = windows[windows.release_clock_scenario == 'main'].merge(
        windows[windows.release_clock_scenario == 'day_end'], on=keys, suffixes=('_main', '_day_end'))
    mature = compare[compare.original_status_main == 'MATURE'].copy()
    mature['update_coverage_changed'] = mature.future_release_count_main != mature.future_release_count_day_end
    mature['boundary_changed'] = mature.boundary_trade_date_main.fillna('MISSING').astype(str) != mature.boundary_trade_date_day_end.fillna('MISSING').astype(str)
    csv('公告时钟敏感性_全部原窗口配对.csv', mature)
    credit_summary = []
    for p, g in change.groupby('period', sort=False):
        comparable = g[g.same_m1_definition]
        supported = comparable[comparable.origin_credit_3_corporate_long_direction.eq('多增')]
        credit_summary.append({'period': p, 'adjacent_pairs': len(g), 'same_m1_definition_pairs': len(comparable),
                               'origin_corp_long_confirmed_more': len(supported),
                               'corp_long_more_to_less': int(comparable.corp_long_from_more_to_less.sum()),
                               'both_delta3_positive': int(comparable.both_delta3_positive.sum()),
                               'both_delta3_positive_and_corp_long_more_to_less': int(comparable.continued_delta3_improvement_but_corp_long_reversed.sum())})
    csv('相邻公告_信用方向与货币改善持续性.csv', pd.DataFrame(credit_summary))
    verify = {'at': now(), 'status': 'PASS_ORIGINAL_WINDOW_INFORMATION_LIFETIME',
              'original_months': len(monthly), 'original_columns_retained_in_frozen_input': len(pd.read_csv(OUT / 'inputs/monthly.csv').columns),
              'windows_each_clock_scenario': 208, 'mature_windows_each_scenario': len(mature),
              'all_scenario_rows': len(windows), 'daily_contribution_rows': len(daily_rows),
              'adjacent_pairs': len(change), 'future_event_rows_both_scenarios': len(event_rows),
              'max_original_recompute_error': max(abs(r['original_recompute_error']) for r in error_rows),
              'max_partition_error': max(abs(r['partition_error']) for r in error_rows),
              'mature_original_same_source_at_entry': bool(windows.loc[windows.original_status.eq('MATURE'), 'original_source_still_latest_at_entry'].all()),
              'clock_sensitivity_update_coverage_changes': int(mature.update_coverage_changed.sum()),
              'clock_sensitivity_boundary_changes': int(mature.boundary_changed.sum()),
              'all_event_volatility_uses_prior_known_close': True,
              'historical_immutable_vintage_proven': False, 'causal_identification': False,
              'independent_validation': False, 'new_models': 0, 'goal_achieved': False}
    assert len(windows) == 416 and len(mature) == 205
    save('verification.json', verify)
    save('results/summary.json', {'coverage': summaries, 'credit_transitions': credit_summary,
                                'clock_sensitivity': {'coverage_changes': int(mature.update_coverage_changed.sum()), 'boundary_changes': int(mature.boundary_changed.sum())},
                                'verification': verify})
    print(json.dumps(clean(verify), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    modes = {'freeze': freeze, 'build': build}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('用法：python research/index_money_information_lifetime_v28.py freeze|build')
    modes[sys.argv[1]]()
