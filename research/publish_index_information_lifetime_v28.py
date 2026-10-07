"""核对延迟收益差，交付完整样本的信息更新范围与信用变化。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_index_money_information_lifetime_v28'
CASES = ['2019-03', '2020-11', '2021-01', '2021-05', '2024-08', '2025-07']


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def csv(name, data):
    data.to_csv(OUT / 'results' / name, index=False, encoding='utf-8-sig')


def bridge():
    m = pd.read_csv(OUT / 'inputs/monthly.csv')
    p = pd.read_csv(OUT / 'inputs/market.csv', parse_dates=['date']).set_index('date')
    w = pd.read_csv(OUT / 'results/416条原窗口与时钟敏感性_全部保留.csv')
    rows = []
    for a in m.to_dict('records'):
        row = {'stat_month': a['stat_month'], 'training_regime': a['training_regime'],
               'original_E0_status': a['E0_20_status'], 'original_E1_status': a['E1_20_status'],
               'status': 'NO_VIEW_ORIGINAL_PAIR_NOT_MATURE'}
        if a['E0_20_status'] != 'MATURE' or a['E1_20_status'] != 'MATURE':
            rows.append(row)
            continue
        t0, t1 = pd.Timestamp(a['E0_20_entry_date']), pd.Timestamp(a['E1_20_entry_date'])
        z0, z1 = pd.Timestamp(a['E0_20_exit_date']), pd.Timestamp(a['E1_20_exit_date'])
        assert p.index.get_loc(t1) - p.index.get_loc(t0) == 1
        assert p.index.get_loc(z1) - p.index.get_loc(z0) == 1
        p0, p1 = float(p.loc[t0, 'open']), float(p.loc[t1, 'open'])
        waiting_div = p.loc[(p.index > t0) & (p.index <= t1), 'dividend'].sum()
        common_div = p.loc[(p.index > t1) & (p.index <= z0), 'dividend'].sum()
        tail_div = p.loc[(p.index > z0) & (p.index <= z1), 'dividend'].sum()
        waiting = (p1 + waiting_div) / p0 - 1
        common = (p.loc[z0, 'close'] + common_div) / p1 - 1
        numerator = -waiting
        denominator = (1 - p1 / p0) * common
        tail = (p.loc[z1, 'close'] - p.loc[z0, 'close'] + tail_div) / p1
        diff = a['E1_20_return'] - a['E0_20_return']
        error = numerator + denominator + tail - diff
        assert abs(error) < 1e-12
        assert abs(waiting + p1 / p0 * common - a['E0_20_return']) < 1e-12
        assert abs(common + tail - a['E1_20_return']) < 1e-12
        row.update(status='COMPLETE_ARITHMETIC_ONLY', E0_entry=t0, E1_entry=t1, E0_exit=z0, E1_exit=z1,
                   original_E0_return=a['E0_20_return'], original_E1_return=a['E1_20_return'],
                   original_E1_minus_E0=diff, removed_waiting_contribution=numerator,
                   entry_denominator_contribution=denominator, added_tail_contribution=tail,
                   common_endpoint_remaining_return=common, waiting_dividend_per_share=waiting_div,
                   common_dividend_per_share=common_div, tail_dividend_per_share=tail_div,
                   identity_error=error)
        for scenario in ['main', 'day_end']:
            a0 = w[w.stat_month.eq(a['stat_month']) & w.clock.eq('E0') & w.release_clock_scenario.eq(scenario)].iloc[0]
            a1 = w[w.stat_month.eq(a['stat_month']) & w.clock.eq('E1') & w.release_clock_scenario.eq(scenario)].iloc[0]
            extra = int(a1.future_release_count - a0.future_release_count)
            assert extra in [0, 1]
            row[scenario + '_E0_updates'] = int(a0.future_release_count)
            row[scenario + '_E1_updates'] = int(a1.future_release_count)
            row[scenario + '_extra_money_release_in_tail'] = bool(extra)
            if extra:
                tau = pd.Timestamp(a1.first_future_publication)
                start = z0.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
                end = z1.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
                assert start < tau <= end
                row[scenario + '_extra_release_at'] = tau
        rows.append(row)
    data = pd.DataFrame(rows)
    csv('104个月_E1与E0差额的等待分母及新增末日.csv', data)
    csv('六个既有病例_延迟收益差与新公告.csv', data[data.stat_month.isin(CASES)])
    done = data[data.status.eq('COMPLETE_ARITHMETIC_ONLY')]
    assert len(done) == 102
    prior = pd.read_csv(ROOT / 'reports/research/510300_information_change_transmission_v6/results/原E0二十日_每日收盘已经公开的货币状态.csv')
    prior['was_updated'] = prior.original_stat_month != prior.latest_published_stat_month
    old_counts = prior.groupby('original_stat_month').was_updated.any()
    current = w[w.clock.eq('E0') & w.release_clock_scenario.eq('main') & w.original_status.eq('MATURE')].set_index('stat_month')
    assert len(old_counts) == 103
    assert all(bool(old_counts.loc[k]) == bool(current.loc[k, 'future_release_count']) for k in old_counts.index)
    # 对原逐日可知货币逐值核对，不能仅核对32这个汇总数。
    times = pd.to_datetime(m.available_at_upper_bound, utc=True)
    for r in prior.to_dict('records'):
        at = pd.Timestamp(r['close_information_cutoff'])
        latest = m.loc[times <= at].iloc[-1]
        assert latest.stat_month == r['latest_published_stat_month']
        assert abs(latest.spread_pp - r['spread_pp']) < 1e-10
    transitions = pd.read_csv(OUT / 'results/103个相邻公告_货币信用与各自快照背景.csv')
    comparable = transitions[transitions.same_m1_definition]
    credit_known = comparable.origin_credit_3_corporate_long_direction.isin(['多增', '少增', '显示舍入界内']) & comparable.next_credit_3_corporate_long_direction.isin(['多增', '少增', '显示舍入界内'])
    check = {
        'at': now(), 'status': 'PASS_E0_PRIOR_RECONCILIATION_AND_E1_BRIDGE',
        'original_rows_retained': len(data), 'complete_paired_windows': len(done),
        'max_timing_bridge_error': float(done.identity_error.abs().max()),
        'old_E0_windows_reconciled': len(old_counts), 'old_E0_daily_context_rows_reconciled': len(prior),
        'old_E0_updated_count': int(old_counts.sum()),
        'extra_release_due_to_E1_main': int(done.main_extra_money_release_in_tail.sum()),
        'extra_release_due_to_E1_day_end': int(done.day_end_extra_money_release_in_tail.sum()),
        'all_extra_releases_in_added_tail': True,
        'credit_same_m1_definition_pairs': len(comparable),
        'credit_both_directions_available_pairs': int(credit_known.sum()),
        'both_delta3_positive_pairs': int(comparable.both_delta3_positive.sum()),
        'both_delta3_positive_credit_available_pairs': int((credit_known & comparable.both_delta3_positive).sum()),
        'continued_improvement_credit_reverse_count': int(comparable.continued_delta3_improvement_but_corp_long_reversed.sum()),
        'source_vintage_formally_verified': False, 'causal_identification': False, 'goal_achieved': False,
    }
    save('timing_bridge_verification.json', check)
    print(json.dumps(check, ensure_ascii=False, indent=2))
    print(data[data.stat_month.eq('2021-05')].to_string(index=False))


def figure():
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    s = pd.read_csv(OUT / 'results/固定时期_信息更新覆盖及原收益位置.csv')
    group_order = ['旧口径2018至2021', '旧口径2022至2024', '新口径2025至2026']
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FAFBFC')
    colors = {'E0': '#355A79', 'E1': '#BC7441'}
    x = np.arange(3)
    for e, offset in [('E0', -.19), ('E1', .19)]:
        rows = s[s.clock.eq(e) & s.release_clock_scenario.eq('main')].set_index('period').loc[group_order]
        fractions = rows.update_fraction_of_mature * 100
        ax.bar(x + offset, fractions, width=.33, color=colors[e], label='原E0' if e == 'E0' else '延迟一天E1', zorder=3)
        for i, (_, r) in enumerate(rows.iterrows()):
            ax.text(i + offset, r.update_fraction_of_mature * 100 + 1.4,
                    f'{int(r.windows_with_money_update)}/{int(r.mature_original_months)}\n{r.update_fraction_of_mature:.1%}',
                    ha='center', va='bottom', fontsize=12, color=colors[e])
    ax.hlines(50, 1.025, 1.355, color='#172D3D', linewidth=2, zorder=5)
    ax.annotate('该组按公告日终核对为18/36（50%）', xy=(1.19, 50), xytext=(1.35, 67),
                arrowprops={'arrowstyle': '->', 'color': '#718394'}, fontsize=10, ha='center')
    ax.set_ylim(0, 74)
    ax.set_xticks(x, ['旧口径\n2018—2021', '旧口径\n2022—2024', '新口径\n2025—2026'])
    ax.set_ylabel('原20交易日窗口内出现新货币公告的比例（%）')
    ax.grid(axis='y', color='#DCE3EA', linewidth=.8, zorder=0)
    ax.legend(loc='upper left', frameon=False)
    fig.suptitle('延迟一天，部分原窗口会多经历一次货币信息更新', x=.08, y=.964, ha='left', fontsize=19, color='#233D59')
    fig.text(.08, .904, 'E0合计32/103；E1合计50/102。采用公告日终时钟时，E1为49/102。', fontsize=12, color='#576779')
    fig.text(.08, .095, '102个已成熟配对中，18个E1新增末日包含下一次公告；保守时钟下为17个。', fontsize=11, color='#344D63')
    fig.text(.08, .058, '这是原窗口的信息范围变化，不是新公告造成收益的比例，也不表示延迟一天更好或更差。', fontsize=10, color='#647487')
    fig.text(.08, .030, '保留所有原月份与待成熟状态；旧新M1分开；两套窗口重叠，不是独立样本。', fontsize=10, color='#647487')
    fig.subplots_adjust(left=.095, right=.965, top=.85, bottom=.19)
    fig.savefig(OUT / 'figures/原20日窗口_下一次货币信息覆盖.png', dpi=180)
    fig.savefig(OUT / 'figures/原20日窗口_下一次货币信息覆盖.svg')
    plt.close(fig)
    print('已生成原两套窗口的消息覆盖比较图。')


def report():
    m = pd.read_csv(OUT / 'inputs/monthly.csv')
    check = json.loads((OUT / 'timing_bridge_verification.json').read_text(encoding='utf-8'))
    s = pd.read_csv(OUT / 'results/固定时期_信息更新覆盖及原收益位置.csv')
    br = pd.read_csv(OUT / 'results/104个月_E1与E0差额的等待分母及新增末日.csv')
    a = br[br.stat_month.eq('2021-05')].iloc[0]
    transition = pd.read_csv(OUT / 'results/103个相邻公告_货币信用与各自快照背景.csv')
    continuing = transition[transition.both_delta3_positive].copy()
    categories = []
    for r in continuing.to_dict('records'):
        first = r['origin_credit_3_corporate_long_direction']
        last = r['next_credit_3_corporate_long_direction']
        if first == 'NO_VIEW' or last == 'NO_VIEW':
            category = '资料缺失'
        elif first == '显示舍入界内' or last == '显示舍入界内':
            category = '至少一期在显示舍入界内'
        else:
            category = first + '→' + last
        categories.append(category)
    continuing['credit_transition_category'] = categories
    csv('两期剪刀差仍改善_全部信用状态与月份.csv', continuing)
    directions = continuing.groupby(['period', 'origin_credit_3_corporate_long_direction', 'next_credit_3_corporate_long_direction'], dropna=False).size().reset_index(name='相邻公告对数')
    csv('两期剪刀差仍改善_完整信用方向变化.csv', directions)
    state_table = continuing.groupby(['period', 'credit_transition_category']).size().unstack(fill_value=0)
    state_table = state_table.reindex(index=['旧口径2018至2021', '旧口径2022至2024', '新口径2025至2026'], columns=['多增→多增', '少增→少增', '多增→少增', '至少一期在显示舍入界内', '资料缺失'], fill_value=0)
    assert int(state_table.to_numpy().sum()) == 25
    assert int(state_table['资料缺失'].sum()) == 3
    state_rows = ['| ' + label + ' | ' + ' | '.join(str(int(v)) for v in values) + ' |' for label, values in state_table.iterrows()]
    july_url = m.loc[m.stat_month.eq('2021-06'), 'source_url'].iloc[0]
    table = []
    for p in ['旧口径2018至2021', '旧口径2022至2024', '新口径2025至2026']:
        z = s[s.period.eq(p)].set_index(['clock', 'release_clock_scenario'])
        e0, e1, e1c = z.loc[('E0', 'main')], z.loc[('E1', 'main')], z.loc[('E1', 'day_end')]
        table.append(f'| {p} | {int(e0.windows_with_money_update)}/{int(e0.mature_original_months)} | {int(e1.windows_with_money_update)}/{int(e1.mature_original_months)} | {int(e1c.windows_with_money_update)}/{int(e1c.mature_original_months)} |')
    body = f'''# 510300原观察窗口中的货币更新与信用变化

**这轮新核对的重点是：延迟一天并继续持有20交易日，既改变入场价格，也改变最后一天可能遇到的信息。在原102个已成熟E0/E1配对中，18个延迟窗口新增了下一次货币公告；按更保守的公告日终时钟，仍有17个。两种窗口的收益差不能全部归因于入场早晚，也不能全部归给新公告。**

信用这一层得到的更一般结论是，连续两期剪刀差三个月变化均改善，既可以伴随企业中长期持续同比多增，也可以伴随持续同比少增。22个信用资料可比较的相邻公告对中，前者9对、后者10对，另有1对从多增转少增、2对至少一期在显示舍入界内。它们的时间分布不同，且相邻区间重叠，不能当作22个独立经济周期。

上轮2019年“剪刀差三个月仍改善、企业中长期由明确多增转明确少增”的直接转折，在当前可比相邻公告中只有这一例。它是有效的机制反例，却没有证明这是一种普遍重复的转折模式。分化更多表现为信用原本已经少增，而剪刀差还在改善。

本轮保持全部104个月、原E0/E1及20交易日期限不变，以指数整体价格、当时货币与最近三月信用结构为研究单位。没有扩展个股、拟合模型、设仓位或使用后来公告筛掉亏损。

## 已有结论与本轮新增的区分

核对时发现，V6主协议之外的补充文件已经记录103个E0窗口的2060个逐日货币状态，并报告32个窗口跨新公告。本轮逐值复核这2060行与32个覆盖，结果相同。E0的32个覆盖不是新发现，已在补充范围纠正中明确。

本轮新增完整E1、公告日终时钟敏感性、公告前与边界及边界后的同本金贡献，并把V23的最近三月信用结构接入相邻公告。已有多变量条件检验和冻结失败不重跑。

## 同样是20个交易日，经历的信息不同

| 原统计月所属时期 | E0跨新公告／成熟窗口 | E1跨新公告／成熟窗口 | E1按公告日终核对 |
|---|---:|---:|---:|
{chr(10).join(table)}
| 全部，合计仅表示覆盖 | 32/103 | 50/102 | 49/102 |

旧新口径分别保留，合计没有把两种M1数值拼成预测样本。E0与E1高度重叠，相邻月份也可能重叠，不能将205个成熟窗口当成205个独立试验。尚未成熟的3个原窗口保持待成熟，不填零。

全部18个新增公告都位于E0末日收盘之后、E1末日收盘之前，恰好进入E1额外增加的末日区间。只有2023年7月病例的E1覆盖，取决于9月11日的盘中时间是否采用：原网页标注12:41，原窗口当日收盘结束；若统一推到当天23:59:59，公告便在窗口外。两种结果都保存，不选择有利时钟。还有4个窗口的分段边界变化，但跨公告与否不变。

![原20日窗口与下一次货币公告](figures/原20日窗口_下一次货币信息覆盖.png)

在已跨公告的旧口径窗口中，第一次新货币公告前通常已有约18个收盘观察，新信息多出现在原20日的后部。这个位置解释了为什么推迟一天可能改变窗口内信息集；它不表示末端收益都由货币数据造成。没有跨下一次货币公告，也不表示期间没有PMI、经济数据、外部新闻或政策变化。

## 一个沿用的指数病例

2021年5月数据对应的原E0是6月11日开盘至7月9日收盘，毛收益{a.original_E0_return:.2%}；E1是6月15日开盘至7月12日收盘，毛收益{a.original_E1_return:.2%}。端午休市使延迟一个交易日跨过了四个日历日。下一期金融统计的网页时间为7月9日17:00:29，晚于E0终点，早于E1新增末日。[央行2021年上半年金融统计]({july_url})

两种原收益相差{a.original_E1_minus_E0 * 100:+.3f}个百分点。固定份额现金口径的算术分解为：

| 对E1减E0的贡献 | 百分点 |
|---|---:|
| 移除等待至E1开盘的原本金变化 | {a.removed_waiting_contribution * 100:+.3f} |
| 同一共同终点因入场分母不同形成的差额 | {a.entry_denominator_contribution * 100:+.3f} |
| E1新增最后一个交易日区间 | {a.added_tail_contribution * 100:+.3f} |
| 合计 | {a.original_E1_minus_E0 * 100:+.3f} |

新增末日区间包含新的货币信息，但其贡献不能当作该公告的因果影响；同日可能还有其他变化。等待贡献包含E0开盘到E1实际开盘的变化，不能提前当作E0起点已经知道的价格。全部102个配对都按同一恒等式完成，没有从中挑选最漂亮的延迟效果。

## 信用结构转折的覆盖比单一病例更窄

全部103个相邻统计月对中，102对处于同一M1口径；2024年12月至2025年1月跨定义，不计算剪刀差水平差。在这102对中，73对有两期可比较的企业中长期方向，其余保留历史缺失、区间缺失或2023年统计范围断点。

两期剪刀差三个月变化均大于零的有25对，其中22对的企业中长期方向资料完整。2019年3月至4月是这22对中唯一出现企业中长期从明确多增转为明确少增的一对：剪刀差三个月变化从+2.6到+2.4个百分点，近期企业中长期同比多增从+1200亿元到−1345亿元。

保留其余所有方向后，完整构成如下；“多增／少增”均指最近三个月贷款净增加额相对上年同期的差，不是贷款余额增减：

| 两期剪刀差都改善的相邻公告对 | 持续多增 | 持续少增 | 多增转少增 | 至少一期舍入界内 | 资料缺失 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(state_rows)}
| 合计，仅表示当前覆盖 | 9 | 10 | 1 | 2 | 3 |

这说明“剪刀差持续改善”不能作为企业中长期信用持续增强的替代变量。旧前期8对持续多增，旧后期和新口径持续少增更常见；由于经济阶段、政策、M1定义和样本长度同时不同，不能把这种分布差异归因于某一个因素，或据此生成看多看空规则。

新口径5对持续少增中，4对连在2025年6月至10月，属于连续区间中的重复背景，不能当作4次独立验证。沿用既有2025年7月病例，原E0/E1随后20日毛收益仍为+6.40%/+9.85%，所以信用持续少增也不能直接推出指数下跌。还须连接财政及置换融资、需求、折现条件与价格已反映的预期；本轮没有靠这一个上涨病例反向定义“弱信贷利多”。

这个“1/22”是当前历史资料的描述，不是未来发生概率。条件包括连续两期三个月改善，信用本身又采用重叠的最近三月区间；它不能代表所有改善月份，也不能遗漏那些在起点就已经同比少增的情况。方向在显示舍入界内时不强行视为多增或少增。本轮没有按这种事后转折分组计算买卖优势。

因此，上轮关于2019年的解释可以保留，但“此类转折很普遍”的扩展没有证据。更一般的命题仍是：剪刀差增速、贷款总量、期限结构和价格反应分别回答不同问题，需要持续更新，而不是彼此代替。

## 如何把波动放进信息更新过程

每一次后续公告都同时保存了发布前最近一个已经发生的收盘、对应RV20与下行尺度，以及新进入五日、紧邻前五日和退出五日的负收益平方和。盘中公告只使用前一个已完成收盘，不能提前使用当日全日波动。新货币和新信用资料是原持有期中的后来信息，不进入原入场表。

收益分三段：公告前最后收盘以前；跨公告的边界交易区间；随后到原终点。三段均以原入场金额为分母，能够相加。盘中公告的边界包含一部分公告前交易，日线资料无法识别分钟级响应；没有新公告时全段仍保留，也不会被命名为无消息期。各段长度不同，平均贡献不用于宣称哪一段持有更有优势。

已有研究支持波动有风险大小的延续性，但未建立稳定的方向优势。本轮新增的是原结果对信息更新和执行日历的暴露范围，尚未估计新信用消息如何独立改变沪深300的盈利预期或要求回报。因此不能把“及时更新背景”直接写成一条已经有效的交易规则。

## 文件与结果状态

104个月741个原字段完整保存在输入副本；两套公告时钟共416行原窗口、8200行逐日贡献；205个成熟原窗口的现金终值和分段和均匹配原收益。另保留全部103对相邻公告及全部102个成熟延迟配对。逐值核对旧E0信息账、固定份额分红权益、时间戳和收益恒等式通过。

这些检查支持测量一致，不认证历史不可变首版，也不是独立的方向验证。新旧M1、未知、显示舍入界、2023统计范围断点和未成熟结果均保留。研究目标继续进行中，没有推广新的买卖候选。

- [全部原窗口与时钟敏感性](results/416条原窗口与时钟敏感性_全部保留.csv)
- [全部相邻公告及完整信用背景](results/103个相邻公告_货币信用与各自快照背景.csv)
- [E1与E0收益差的全部102个成熟配对](results/104个月_E1与E0差额的等待分母及新增末日.csv)
- [公告时刻已经可知的指数波动](results/原窗口内所有后续货币公告_当时已知价格波动.csv)
- [原E0复核与E1差额核对](timing_bridge_verification.json)
'''
    (OUT / '指数观察窗口的信息更新与信用变化.md').write_text(body, encoding='utf-8')
    print('已保存完整样本结论；既有E0发现与本轮新增范围明确分开。')


def complete():
    v = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    b = json.loads((OUT / 'timing_bridge_verification.json').read_text(encoding='utf-8'))
    assert v['status'].startswith('PASS_') and b['status'].startswith('PASS_')
    for script in ['index_money_information_lifetime_v28.py', 'publish_index_information_lifetime_v28.py']:
        shutil.copy2(ROOT / 'research' / script, OUT / 'code' / script)
    report_path = OUT / '指数观察窗口的信息更新与信用变化.md'
    assert report_path.is_file() and (OUT / 'figures/原20日窗口_下一次货币信息覆盖.png').is_file()
    save('completion.json', {'at': now(), 'status': 'COMPLETED_INDEX_MONEY_INFORMATION_LIFETIME',
         'previous_turn_classification': 'PROGRESS', 'goal_turn_classification': 'PROGRESS',
         'actual_increment': 'E1新增信息范围、公告时间精度、原收益差的等待与末日构成，以及近期信用转折的全样本覆盖。',
         'existing_result_reused': 'V6原E0有32个窗口跨新公告，逐值复核一致，未宣称新发现。',
         'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
         'visual_review': '已查看PNG，文字及柱形数字可读',
         'new_models': 0, 'new_accounts': 0, 'causal_identification': False,
         'independent_validation': False, 'goal_achieved': False})
    print('本轮信息更新与延迟窗口核对完成；总研究目标保持进行中。')


if __name__ == '__main__':
    modes = {'bridge': bridge, 'figure': figure, 'report': report, 'complete': complete}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('用法：python research/publish_index_information_lifetime_v28.py bridge|figure|report|complete')
    modes[sys.argv[1]]()
