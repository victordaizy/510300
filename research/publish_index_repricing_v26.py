"""补足指数调样的口径解释，并发布指数定价分解。"""
from pathlib import Path
import json
import shutil
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from index_repricing_chronology_v26 import ROOT, OUT, CASES, now, digest, save, csv, load_inputs

MEMBERS = ROOT / 'data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet'
ANNOUNCEMENT = 'https://www.sse.com.cn/market/sseindex/diclosure/c/c_20210528_5476672.shtml'


def supplement():
    if (OUT / 'membership_supplement.json').exists():
        raise RuntimeError('调样补充已有记录，不重新设定范围。')
    shutil.copy2(MEMBERS, OUT / 'inputs/membership.parquet')
    save('membership_supplement.json', {
        'at': now(), 'reason': '既有2021年5月病例中，6月15日P/PE单日变化与本地成分变更同日；因此进一步核对全部原窗口的成分变更日期，而不只保留该病例。',
        'prior_observed_fact': '已经看到2021年6月15日价格下跌、PE上升及隐含分母约下降3.72%，这是结果后提出的口径核对，不是盲检。',
        'scope': '原104个月、E0/E1、原20交易日完全不变；用保存的日度300只名单标记集合变更。首日收盘之后的19个间隔按变更日和其余日做算术求和。',
        'identity': '隐含分母的全窗口对数变化=名单变更交易日上的对数变化之和+其他交易日上的对数变化之和。',
        'limits': '日期上的集中不等于调样因果贡献。名单未变也可能有调整股本、权重、财务或除数更新。本地名单为既有官方公告重放；并非逐日ETF真实持仓。没有全历史公告可得时钟，不将事后变更标记作为新入场信息。',
        'source': str(MEMBERS.relative_to(ROOT)), 'sha256': digest(OUT / 'inputs/membership.parquet'),
        'new_public_source': ANNOUNCEMENT, 'new_source_scope': '只核对2021年5月28日上交所调样公告的日期、25只替换及行业结构事实；不扩充公司报告。',
        'new_models': 0, 'new_accounts': 0,
    })
    (OUT / 'sources').mkdir(exist_ok=True)
    response = requests.get(ANNOUNCEMENT, timeout=30)
    response.raise_for_status()
    response.encoding = 'utf-8'
    assert '2021-05-28' in response.text and '25只样本' in response.text and '6月11日收盘后' in response.text
    target = OUT / 'sources/上交所_20210528_沪深300调样公告.html'
    target.write_bytes(response.content)
    save('sources/调样公告事实与回执.json', {
        'at': now(), 'url': ANNOUNCEMENT, 'sha256': digest(target), 'publication_date': '2021-05-28',
        'assumed_known_at': '2021-05-28T23:59:59+08:00', 'effective': '2021-06-11收盘后',
        'first_index_session_after_effective': '2021-06-15', 'replaced_members': 25,
        'csi300_sectors': {'医药卫生样本净变化': 7, '工业样本净变化': 2, '金融地产样本净变化': -9, '电信业务样本净变化': -2},
        'prospective_adjusted_pe_using_20210526_prices': 16.0,
        'historical_first_vintage_verified': False,
        'source_limit': '公告未提供保持其他因素不变时的全部估值分解；不能据此给6月15日PE变化确定因果份额。',
    })
    print('已保存上交所调样原公告，并固定全部原窗口的名单变更核对。')


def membership_calculation():
    supplement = json.loads((OUT / 'membership_supplement.json').read_text('utf-8'))
    assert digest(OUT / 'inputs/membership.parquet') == supplement['sha256']
    members = pd.read_parquet(OUT / 'inputs/membership.parquet')
    assert members.index_code.eq('000300').all()
    members['membership_date'] = pd.to_datetime(members.membership_date)
    assert not members.duplicated(['membership_date', 'symbol']).any()
    states = members.groupby('membership_date').symbol.agg(frozenset).sort_index()
    assert states.map(len).eq(300).all()
    transitions = []
    previous = None
    for date, symbols in states.items():
        transitions.append({'date': date, 'known_set_change': previous is not None,
                            'membership_set_changed': previous is not None and symbols != previous,
                            'added_members': len(symbols-previous) if previous is not None else np.nan,
                            'removed_members': len(previous-symbols) if previous is not None else np.nan})
        previous = symbols
    transitions = pd.DataFrame(transitions).set_index('date')
    csv(transitions.reset_index(), '原名单重建_全部成分集合变更日期.csv')
    monthly, prices, pe, market = load_inputs()
    post = pd.read_csv(OUT / 'results/208条原窗口_指数价格与PE事后分解.csv')
    records, intervals = [], []
    for r in post.to_dict('records'):
        row = {'stat_month': r['stat_month'], 'training_regime': r['training_regime'], 'clock': r['clock'],
               'post_status': r['post_status'], 'membership_decomposition_status': 'NO_VIEW_PARENT_PRICING_WINDOW'}
        if r['post_status'] != 'COMPLETE_EX_POST_ALGEBRA_ONLY':
            records.append(row)
            continue
        entry, end = pd.Timestamp(r['entry_date']), pd.Timestamp(r['exit_date'])
        dates = prices.loc[entry:end].index
        if not dates.isin(pe.index).all() or not dates[1:].isin(transitions.index).all():
            row['membership_decomposition_status'] = 'NO_VIEW_DAILY_PE_OR_MEMBERSHIP_MISSING'
            records.append(row)
            continue
        d = prices.loc[dates, ['close']].join(pe[['original_pe_official']])
        d['log_price_pp'] = 100*np.log(d.close/d.close.shift())
        d['log_pe_pp'] = 100*np.log(d.original_pe_official/d.original_pe_official.shift())
        denominator = d.close/d.original_pe_official
        d['log_denominator_pp'] = 100*np.log(denominator/denominator.shift())
        d = d.iloc[1:].join(transitions)
        assert len(d) == 19 and d.known_set_change.all()
        selected = d.membership_set_changed
        change_sum = d.loc[selected, 'log_denominator_pp'].sum()
        other_sum = d.loc[~selected, 'log_denominator_pp'].sum()
        error = change_sum+other_sum-r['post_log_implied_denominator_pp']
        assert abs(error) < 1e-10
        row.update(membership_decomposition_status='COMPLETE_TIMING_PARTITION_NOT_CAUSAL',
                   member_change_intervals=int(selected.sum()), other_intervals=int((~selected).sum()),
                   member_change_dates='|'.join(d.index[selected].strftime('%Y-%m-%d')),
                   denominator_log_pp_on_change_dates=change_sum,
                   denominator_log_pp_other_dates=other_sum,
                   denominator_log_pp_whole_window=r['post_log_implied_denominator_pp'],
                   denominator_abs_log_sum_on_change_dates=d.loc[selected, 'log_denominator_pp'].abs().sum(),
                   denominator_abs_log_sum_all_dates=d.log_denominator_pp.abs().sum(),
                   partition_error_pp=error)
        records.append(row)
        for date, z in d.iterrows():
            intervals.append({'stat_month':r['stat_month'], 'training_regime':r['training_regime'],
                              'clock':r['clock'], 'date':date, **z.to_dict()})
    result = pd.DataFrame(records)
    csv(result, '208条原窗口_隐含分母与成分变更日期.csv')
    csv(pd.DataFrame(intervals), '逐收盘间隔_价格PE与成分变更.csv')
    summaries = []
    for (regime, clock), group in result.groupby(['training_regime','clock'], sort=False):
        good = group[group.membership_decomposition_status.eq('COMPLETE_TIMING_PARTITION_NOT_CAUSAL')]
        absolute_total = good.denominator_abs_log_sum_all_dates.sum()
        summaries.append({'training_regime':regime, 'clock':clock, 'complete_windows':len(good),
                          'windows_cross_member_change':int(good.member_change_intervals.gt(0).sum()),
                          'all_intervals':int(good.member_change_intervals.sum()+good.other_intervals.sum()),
                          'member_change_intervals':int(good.member_change_intervals.sum()),
                          'abs_denominator_movement_fraction_on_change_dates':good.denominator_abs_log_sum_on_change_dates.sum()/absolute_total if absolute_total else None,
                          'max_partition_error_pp':good.partition_error_pp.abs().max()})
    save('results/membership_summary.json', {'at':now(),'summaries':summaries,'not_causal_attribution':True,
                                            'original_rows_preserved':len(result),
                                            'case_202105':result[result.stat_month.eq('2021-05')].to_dict('records')})
    print(json.dumps(json.loads((OUT/'results/membership_summary.json').read_text('utf-8')),ensure_ascii=False,indent=2))


def chart():
    data = pd.read_csv(OUT/'results/208条原窗口_指数价格与PE事后分解.csv')
    data = data[data.clock.eq('E0')].set_index('stat_month').loc[CASES]
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(13.4,6.4), dpi=180)
    fig.patch.set_facecolor('#f6f5f0'); ax.set_facecolor('#f6f5f0')
    positions = np.arange(len(data)); positive = np.zeros(len(data)); negative = np.zeros(len(data))
    components = [('post_first_intraday_log_pp','首日开盘至收盘','#83958f'),
                  ('post_log_pe_pp','随后19个收盘间隔的PE变化','#337b9c'),
                  ('post_log_implied_denominator_pp','同段隐含分母变化','#bd9065')]
    for column,label,color in components:
        values = data[column].to_numpy()
        bottom = np.where(values>=0,positive,negative)
        ax.bar(positions,values,bottom=bottom,width=.58,color=color,label=label)
        positive += np.maximum(values,0); negative += np.minimum(values,0)
    ax.scatter(positions,data.post_index_log_return_pp,color='#172a33',marker='D',s=44,zorder=5,label='指数开盘至期末收盘总变化')
    for i,value in enumerate(data.post_index_log_return_pp):
        location=positive[i]+1.3 if value>=0 else negative[i]-1.4
        ax.text(i,location,f'{value:+.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=11,color='#172a33')
    ax.set_xticks(positions,[f'{m}\n观察日 {d}' for m,d in zip(data.index,data.observation_date)],fontsize=10)
    ax.set_ylabel('对数百分点（100 × 对数变化）',fontsize=11)
    ax.axhline(0,color='#7b8589',lw=.9)
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#b5bdba')
    ax.set_ylim(min(negative)-6,max(positive)+6)
    fig.suptitle('指数价格变化与整体PE变化需要同时解释',x=.075,ha='left',fontsize=19,fontweight='bold',y=.98)
    ax.set_title('六个既有病例 · 原E0的20个交易日 · 所有分解项均为事后观察',loc='left',fontsize=11,pad=19,color='#52615c')
    fig.legend(loc='lower center',bbox_to_anchor=(.52,.065),ncol=2,frameon=False,fontsize=10)
    fig.text(.075,.023,'这是价格与PE的算术关系。隐含分母不是已经核实的指数盈利，柱形不是各类宏观冲击的因果贡献。',fontsize=10,color='#52615c')
    fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.24)
    fig.savefig(OUT/'figures/指数价格与PE变化_六个既有病例.png',facecolor=fig.get_facecolor())
    fig.savefig(OUT/'figures/指数价格与PE变化_六个既有病例.svg',facecolor=fig.get_facecolor())
    plt.close(fig)
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    print('指数定价分解图已生成。')


if __name__ == '__main__':
    {'supplement':supplement,'membership':membership_calculation,'chart':chart}[sys.argv[1]]()
