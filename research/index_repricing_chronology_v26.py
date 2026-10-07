"""连接货币公布前后的指数价格与估值，保持原ETF观察窗口。"""
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
OUT = ROOT / 'reports/research/510300_index_repricing_chronology_v26'
V5 = 'reports/research/510300_macro_earnings_pricing_bridge_v5'
SOURCES = {
    'monthly.csv': 'reports/research/510300_index_mainline_v25/results/104个月_指数主线与宏观来源全表.csv',
    'pre_pricing.csv': V5 + '/results/所有原点_同日期价格与PE恒等式.csv',
    'pe.parquet': V5 + '/inputs/pe.parquet',
    'index_price.parquet': V5 + '/inputs/index_price.parquet',
    'market.csv': V5 + '/inputs/market.csv',
    'prior_protocol.json': V5 + '/protocol.json',
}
CASES = ['2019-03', '2020-11', '2021-01', '2021-05', '2024-08', '2025-07']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(frame, name):
    frame.to_csv(OUT / 'results' / name, index=False, encoding='utf-8-sig')


def freeze():
    if OUT.exists():
        raise RuntimeError('研究目录已经存在，不覆盖原范围。')
    for folder in ['inputs', 'results', 'code', 'figures']:
        (OUT / folder).mkdir(parents=True)
    protocol = {
        'at': now(), 'study': '510300_INDEX_REPRICING_CHRONOLOGY_V26',
        'previous_goal_turn_classification': 'PROGRESS',
        'previous_progress': '104个月已重接指数主线；个股报告不再作为主结论。',
        'question': '同一货币观察前后的指数价格变化，分别伴随什么整体PE变化；信用改善是否足以代表指数价格所需的全部定价条件？',
        'population': '保留原104个月及旧新M1制度；只沿用原E0/E1的20交易日入场退出日期。',
        'previous_evidence': '原V5只分解观察前20/60日价格和PE。本轮复用其事前结果，新增原20日结果窗内的指数价格与PE分解。',
        'post_identity': '指数开盘至末日收盘的对数变化=首日开盘至收盘变化+首日收盘至末日的PE对数变化+同段P/PE隐含分母对数变化。后两项对应19个收盘间隔，不能冒充20个收盘间隔。',
        'asset_units': '价格与PE恒等式一律使用000300价格指数。510300原固定股数含现金分红收益另列，不把指数价格分解冒充ETF收益分解。',
        'availability': '公布前PE继续沿原下一交易日开盘可用的重建假设，并检查不晚于原观察时点；结果窗PE均标事后，不用作入场信息。历史首次版本未验证。',
        'pe_semantics': '沿用原indexCsiDsPe序列，未新增认证静态滚动及股本口径。P/PE只是隐含分母，不命名为已披露或预期EPS；PE变化不能唯一分配给利率、风险溢价或盈利预期。',
        'missing': '要求原窗口全部20个指数交易日及起止同日PE；缺一则保留NO_VIEW，不前向填充、不缩短期限或替换指数。原ETF已成熟结果仍保留。',
        'summaries': '仅按原M1制度及E0/E1分别报告覆盖、PE与同段价格方向一致数、价格下跌而隐含分母增加等事后构成。E0/E1高度重叠，不合并为独立样本。不按结果挑样本，不搜索阈值、不做方向预测拟合。',
        'cases_fixed_before_new_calculation': CASES,
        'macro_context': '保留原剪刀差及余额基数分解、近期企业与住户信用、订单、资金价格、波动来源和原后续路径；不打综合分。',
        'verification': '检查日期及交易日数、原结果保留、原ETF分红现金终值、指数对数恒等式与逐日链积。',
        'historical_outcomes_already_seen': True, 'new_models': 0, 'new_accounts': 0,
        'orders_authorized': False, 'goal_achieved': False,
    }
    save('protocol.json', protocol)
    sources = []
    for name, source in SOURCES.items():
        dest = OUT / 'inputs' / name
        shutil.copy2(ROOT / source, dest)
        sources.append({'name': name, 'source': source, 'sha256': digest(dest)})
    save('freeze.json', {'at': now(), 'protocol_sha256': digest(OUT / 'protocol.json'), 'inputs': sources})
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    print('已固定全部104个月的指数定价前后连接，不改变原收益窗口。')


def load_inputs():
    monthly = pd.read_csv(OUT / 'inputs/monthly.csv')
    prices = pd.read_parquet(OUT / 'inputs/index_price.parquet')
    prices['date'] = pd.to_datetime(prices.date).dt.normalize()
    pe = pd.read_parquet(OUT / 'inputs/pe.parquet')
    pe['observation_date'] = pd.to_datetime(pe.observation_date).dt.normalize()
    market = pd.read_csv(OUT / 'inputs/market.csv')
    market['date'] = pd.to_datetime(market.date)
    assert len(monthly) == 104 and not monthly.stat_month.duplicated().any()
    for data, column in [(prices, 'date'), (pe, 'observation_date'), (market, 'date')]:
        assert not data[column].duplicated().any(), column
    return monthly, prices.set_index('date').sort_index(), pe.set_index('observation_date').sort_index(), market.set_index('date').sort_index()


def build():
    monthly, prices, pe, market = load_inputs()
    records, daily, etf_checks = [], [], []
    for r in monthly.to_dict('records'):
        for clock in ['E0', 'E1']:
            p = clock + '_20_'
            row = {'stat_month': r['stat_month'], 'training_regime': r['training_regime'],
                   'clock': clock, 'observation_date': r['observation_date'],
                   'entry_date': r[p + 'entry_date'], 'exit_date': r[p + 'exit_date'],
                   'etf_original_status': r[p + 'status'], 'etf_original_return': r[p + 'return'],
                   'post_status': 'NO_VIEW_ORIGINAL_WINDOW_PENDING',
                   'post_information_role': 'EX_POST_DIAGNOSTIC_NOT_ENTRY_INPUT',
                   'post_denominator_role': 'IMPLIED_P_DIVIDED_BY_PE_NOT_VERIFIED_EPS'}
            if r[p + 'status'] != 'MATURE':
                records.append(row)
                continue
            entry, end = pd.Timestamp(row['entry_date']), pd.Timestamp(row['exit_date'])
            m = market.loc[entry:end]
            assert len(m) == 20
            cash = m.loc[m.index > entry, 'dividend'].sum()
            etf_return = (m.iloc[-1]['close'] + cash) / m.iloc[0]['open'] - 1
            assert abs(etf_return - row['etf_original_return']) < 1e-12
            etf_checks.append({'stat_month': r['stat_month'], 'clock': clock,
                               'entry_date': entry, 'exit_date': end, 'sessions': len(m),
                               'open': m.iloc[0]['open'], 'end_close': m.iloc[-1]['close'],
                               'entitled_cash_per_share': cash, 'recomputed_return': etf_return,
                               'original_return': row['etf_original_return'],
                               'error': etf_return - row['etf_original_return']})
            q = prices.loc[entry:end]
            row['post_index_sessions_available'] = len(q)
            if len(q) != 20 or not q.index.equals(m.index):
                row['post_status'] = 'NO_VIEW_INDEX_WINDOW_INCOMPLETE'
                records.append(row)
                continue
            if entry not in pe.index or end not in pe.index:
                row['post_status'] = 'NO_VIEW_EXACT_ENDPOINT_PE_MISSING'
                records.append(row)
                continue
            first, last = q.iloc[0], q.iloc[-1]
            e0, e1 = float(pe.loc[entry, 'original_pe_official']), float(pe.loc[end, 'original_pe_official'])
            if not (e0 > 0 and e1 > 0):
                row['post_status'] = 'NO_VIEW_NONPOSITIVE_PE'
                records.append(row)
                continue
            o, c0, c1 = float(first.open), float(first.close), float(last.close)
            lp = 100 * np.log(c1 / o)
            li = 100 * np.log(c0 / o)
            lv = 100 * np.log(e1 / e0)
            ld = 100 * np.log((c1 / e1) / (c0 / e0))
            row.update(post_status='COMPLETE_EX_POST_ALGEBRA_ONLY', post_close_intervals=19,
                       post_index_entry_open=o, post_index_entry_close=c0, post_index_exit_close=c1,
                       post_pe_entry=e0, post_pe_exit=e1,
                       post_pe_entry_available_at=pe.loc[entry, 'available_at'],
                       post_pe_exit_available_at=pe.loc[end, 'available_at'],
                       post_entry_implied_denominator=c0/e0, post_exit_implied_denominator=c1/e1,
                       post_index_open_to_close_return=c1/o-1, post_index_close_to_close_return=c1/c0-1,
                       post_index_log_return_pp=lp, post_first_intraday_log_pp=li,
                       post_close_to_close_log_price_pp=100*np.log(c1/c0),
                       post_log_pe_pp=lv, post_log_implied_denominator_pp=ld,
                       post_identity_error_pp=lp-li-lv-ld,
                       post_etf_minus_index_simple_return=etf_return-(c1/o-1))
            records.append(row)
            previous_close = None
            for date, bar in q.iterrows():
                daily.append({'stat_month': r['stat_month'], 'clock': clock, 'date': date,
                              'index_close': bar.close,
                              'index_day_gross_factor': bar.close/(o if previous_close is None else previous_close),
                              'same_date_pe': pe.loc[date, 'original_pe_official'] if date in pe.index else np.nan,
                              'information_role': 'EX_POST_DIAGNOSTIC_NOT_ENTRY_INPUT'})
                previous_close = bar.close
    result = pd.DataFrame(records)
    csv(result, '208条原窗口_指数价格与PE事后分解.csv')
    csv(pd.DataFrame(daily), '已覆盖原窗口_逐日指数价格.csv')
    csv(pd.DataFrame(etf_checks), '原ETF固定股数现金终值复算.csv')
    pre = pd.read_csv(OUT / 'inputs/pre_pricing.csv')
    pre = pre[pre.origin_key.str.startswith('104个月|')].copy()
    pre['stat_month'] = pre.origin_key.str.split('|', regex=False).str[-1]
    assert len(pre) == 104
    combined = monthly.merge(pre.drop(columns='origin_key'), on='stat_month', how='left', validate='one_to_one')
    for clock in ['E0', 'E1']:
        part = result[result.clock.eq(clock)]
        columns = [c for c in part if c.startswith('post_')]
        part = part[['stat_month'] + columns].rename(columns={c: clock + '_' + c for c in columns})
        combined = combined.merge(part, on='stat_month', how='left', validate='one_to_one')
    pd.testing.assert_frame_equal(monthly, combined[monthly.columns], check_dtype=False)
    csv(combined, '104个月_宏观来源与指数定价前后全表.csv')
    short = ['stat_month', 'observation_date', 'training_regime', 'spread_pp', 'delta3_spread_pp',
             'd3_relative_current_log_pp', 'd3_relative_base_revision_log_pp',
             'credit_3_corporate_total_yoy_change_yi', 'credit_3_corporate_long_yoy_change_yi',
             'credit_3_household_long_yoy_change_yi', 'orders_first_release_value',
             'context_fdr_policy_gap_bp_mean_change', 'past_return60', 'downside_window_state',
             'price_bridge_60_log_price_pp', 'price_bridge_60_log_pe_pp',
             'price_bridge_60_log_implied_denominator_pp', 'E0_20_return', 'E1_20_return',
             'E0_post_status', 'E0_post_index_log_return_pp', 'E0_post_first_intraday_log_pp',
             'E0_post_log_pe_pp', 'E0_post_log_implied_denominator_pp',
             'E1_post_status', 'E1_post_index_log_return_pp', 'E1_post_first_intraday_log_pp',
             'E1_post_log_pe_pp', 'E1_post_log_implied_denominator_pp']
    csv(combined[short], '104个月_定价前后便览.csv')
    csv(combined[combined.stat_month.isin(CASES)][short], '六个既有病例_货币信用与指数定价.csv')
    summaries = []
    for (regime, clock), group in result.groupby(['training_regime', 'clock'], sort=False):
        q = group[group.post_status.eq('COMPLETE_EX_POST_ALGEBRA_ONLY')]
        price, valuation, denom = q.post_close_to_close_log_price_pp, q.post_log_pe_pp, q.post_log_implied_denominator_pp
        summaries.append({'training_regime': regime, 'clock': clock, 'original_months': len(group),
                          'complete_months': len(q), 'missing_months': len(group)-len(q),
                          'same_sign_price_and_pe': ((price*valuation)>0).sum(),
                          'price_down_denominator_up': ((price<0)&(denom>0)).sum(),
                          'price_up_denominator_down': ((price>0)&(denom<0)).sum(),
                          'median_abs_log_price_pp': price.abs().median(),
                          'median_abs_log_pe_pp': valuation.abs().median(),
                          'median_abs_log_denominator_pp': denom.abs().median(),
                          'max_identity_error_pp': q.post_identity_error_pp.abs().max()})
    csv(pd.DataFrame(summaries), '旧新口径分别列示_事后算术构成.csv')
    save('results/summary.json', {'at': now(), 'observations': 104, 'windows': len(result),
                                 'original_columns_preserved': len(monthly.columns),
                                 'etf_mature_windows_recomputed': len(etf_checks),
                                 'coverage': result.groupby(['clock','post_status']).size().reset_index(name='windows').to_dict('records'),
                                 'summaries': summaries, 'new_predictive_claim': False})
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    print(json.dumps(clean(summaries), ensure_ascii=False, indent=2))


def complete():
    frozen = json.loads((OUT / 'freeze.json').read_text('utf-8'))
    assert digest(OUT / 'protocol.json') == frozen['protocol_sha256']
    for r in frozen['inputs']:
        assert digest(OUT / 'inputs' / r['name']) == r['sha256']
    original = pd.read_csv(OUT / 'inputs/monthly.csv')
    combined = pd.read_csv(OUT / 'results/104个月_宏观来源与指数定价前后全表.csv')
    pd.testing.assert_frame_equal(original, combined[original.columns], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    post = pd.read_csv(OUT / 'results/208条原窗口_指数价格与PE事后分解.csv')
    q = post[post.post_status.eq('COMPLETE_EX_POST_ALGEBRA_ONLY')]
    daily = pd.read_csv(OUT / 'results/已覆盖原窗口_逐日指数价格.csv')
    factors = daily.groupby(['stat_month', 'clock']).index_day_gross_factor.agg(['prod', 'size'])
    assert factors['size'].eq(20).all()
    pairs = q.set_index(['stat_month', 'clock'])
    chain_error = (factors['prod'] - (1+pairs.post_index_open_to_close_return)).abs().max()
    assert chain_error < 1e-12
    assert q.post_identity_error_pp.abs().max() < 1e-10
    start = pd.to_datetime(q.entry_date, utc=True) + pd.Timedelta(hours=7)
    assert (pd.to_datetime(q.post_pe_entry_available_at, utc=True)>start).all()
    pre = combined[combined.price_bridge_end_available_at.notna()]
    assert (pd.to_datetime(pre.price_bridge_end_available_at, utc=True)<=pd.to_datetime(pre.snapshot_at, utc=True)).all()
    for clock in ['E0','E1']:
        same = post[post.clock.eq(clock)].set_index('stat_month')
        expected = original.set_index('stat_month')
        assert np.allclose(same.etf_original_return, expected[clock+'_20_return'], equal_nan=True, rtol=1e-12, atol=1e-12)
    assert (OUT/'指数定价变化与货币信用传导.md').exists()
    supplement=json.loads((OUT/'membership_supplement.json').read_text('utf-8'))
    assert digest(OUT/'inputs/membership.parquet')==supplement['sha256']
    membership=pd.read_csv(OUT/'results/208条原窗口_隐含分母与成分变更日期.csv')
    intervals=pd.read_csv(OUT/'results/逐收盘间隔_价格PE与成分变更.csv')
    assert len(membership)==208 and not membership.duplicated(['stat_month','clock']).any()
    assert len(intervals)==len(q)*19
    assert membership.partition_error_pp.abs().max()<1e-10
    source=json.loads((OUT/'sources/调样公告事实与回执.json').read_text('utf-8'))
    assert digest(OUT/'sources/上交所_20210528_沪深300调样公告.html')==source['sha256']
    assert pd.Timestamp(source['assumed_known_at'])<pd.Timestamp(original.loc[original.stat_month.eq('2021-05'),'snapshot_at'].iloc[0])
    save('verification.json', {'at': now(), 'status': 'PASS_ORIGINAL_WINDOWS_AND_INDEX_REPRICING_IDENTITIES',
                               'original_months':104, 'original_columns_unchanged':len(original.columns),
                               'complete_index_windows':len(q), 'missing_windows_retained':len(post)-len(q),
                               'index_daily_rows':len(daily), 'max_daily_chain_error':chain_error,
                               'max_identity_error_pp':q.post_identity_error_pp.abs().max(),
                               'future_pe_not_used_as_entry_data':True, 'membership_close_intervals':len(intervals),
                               'max_membership_partition_error_pp':membership.partition_error_pp.abs().max(),
                               'rebalance_announcement_known_before_original_case':True,
                               'independent_validation':False})
    save('completion.json', {'at': now(), 'status':'COMPLETED_INDEX_REPRICING_CHRONOLOGY',
                            'previous_goal_turn_classification':'PROGRESS', 'current_goal_turn_classification':'PROGRESS',
                            'new_evidence':'新增原E0/E1窗口的整体指数价格和PE事后分解，保留全部104个月；全段成分变更日期核对显示2021年5月病例的隐含分母下降集中于已公告调样后的首个交易日，不能直接解释为盈利骤降。',
                            'company_financial_reports_added':0, 'new_models':0, 'new_accounts':0,
                            'goal_status':'active', 'goal_achieved':False, 'orders_authorized':False,
                            'global_mandate_modified':False})
    shutil.copy2(__file__, OUT / 'code' / Path(__file__).name)
    pd.DataFrame([{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
                  for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_index.csv']).to_csv(OUT/'file_index.csv',index=False,encoding='utf-8-sig')
    print('指数定价的前后分解已保存并核对；未来方向目标仍未完成。')


if __name__ == '__main__':
    {'freeze': freeze, 'build': build, 'complete': complete}[sys.argv[1]]()
