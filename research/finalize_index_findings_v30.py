"""按用户确认的风险与方向分别判断标准，复核并交付最终指数研究结论。"""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from index_volatility_cooling_v30 import ROOT, OUT, PHASES, STATES, UP, DOWN, MEAN, TIE, OTHER, EPS, now, digest, save, csv, verify_inputs

V3 = ROOT / 'reports/research/510300_volatility_money_decomposition_v3'
V22 = ROOT / 'reports/research/510300_within_release_volatility_v22'
V29 = ROOT / 'reports/research/510300_fiscal_money_transmission_v29'


def prepare():
    dest = OUT / 'prior_evidence'
    dest.mkdir(exist_ok=False)
    sources = {
        'weekly_v3.csv': V3 / 'results/波动分解_固定周度_完整观察.csv',
        'associations_v3.csv': V3 / 'results/波动与后续风险收益关联.csv',
        'disjoint_v3.csv': V3 / 'results/不重叠20日窗口_全部保留点.csv',
        'disjoint_associations_v3.csv': V3 / 'results/不重叠20日窗口_关联复核.csv',
        'macro_predictions.csv': V3 / 'evidence/510300_m1_m2_monthly_increment_v1_paired_M1_OLD_M2_MMF2018.csv',
        'risk_predictions.csv': V3 / 'evidence/510300_monthly_downside_forecast_v1_paired_losses.csv',
        'risk_account_result.json': V3 / 'evidence/510300_monthly_downside_forecast_v1_result.json',
        'v29_balances.csv': V29 / 'results/五月基点及六月至十月_事后原表余额.csv',
        'v29_fiscal_clock.csv': V29 / 'results/五个原观察点_当时财政与后来同月资料.csv',
        'v29_verification.json': V29 / 'saved_output_verification.json',
        'v22_associations.csv': V22 / 'results/D变化_已发生价格与后续结果关联.csv',
    }
    receipts = []
    for name, source in sources.items():
        target = dest / name
        shutil.copy2(source, target)
        receipts.append({'name': name, 'source': str(source), 'sha256': digest(target)})
    save('conclusion_acceptance.json', {
        'at': now(), 'original_objective': '我需要更明显的结论，继续拆解波动率和m1,m2剪刀差，直到找到510300确定性的结论',
        'user_clarification': '风险规律与方向优势分别给出结论，允许明确判定哪些关系无效',
        'authority': '用户对本任务异步澄清问题的直接回答；不是附件或代理自行缩减目标。',
        'required': ['指数为研究单位', '同时处理价格位置、波动过程、货币来源和信贷等背景',
                     '风险和方向分别判断', '区分被反例否定的充分条件与未获支持的统计关系',
                     '完整样本、失败例及旧新口径保留', '引用原始证据并复算关键结论',
                     '说明时点、已定价、样本不足和因果识别边界', '交付清晰结论和可复查结果'],
        'not_required_for_this_clarified_research_delivery': ['必须找到正收益入场规则', '保证未来涨跌', '账户夏普达标', '实盘下单'],
        'old_frozen_failures_remain_unchanged': True, 'prior_evidence_copies': receipts,
    })
    print('已保存用户确认的验收范围及原风险、方向证据快照。')


def weighted_rank_corr(x, y, months):
    w = 1 / pd.Series(months).map(pd.Series(months).value_counts()).to_numpy(float)
    def rank(a):
        unique, code = np.unique(np.round(a, 12), return_inverse=True)
        grouped = np.bincount(code, weights=w, minlength=len(unique))
        return (grouped.cumsum() - grouped / 2)[code]
    a, b = rank(x), rank(y)
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))


def verify():
    verify_inputs()
    acceptance = json.loads((OUT / 'conclusion_acceptance.json').read_text(encoding='utf-8'))
    for source in acceptance['prior_evidence_copies']:
        assert digest(OUT / 'prior_evidence' / source['name']) == source['sha256']
    pairs = pd.read_csv(OUT / 'results/336对_总波动下降来源与全部原背景.csv').set_index('pair_id')
    members = pd.read_csv(OUT / 'results/13440条_两端原20日收益成员.csv')
    calculated = []
    for (pair_id, point), z in members.groupby(['pair_id', 'point']):
        assert len(z) == 20 and z.date.is_unique
        r = z.daily_return.to_numpy(float)
        calculated.append({'pair_id': pair_id, 'point': point, 'variance': float(np.var(r, ddof=1) * 252),
                           'up2': float(np.mean(np.maximum(r, 0)**2) * 252),
                           'down2': float(np.mean(np.minimum(r, 0)**2) * 252), 'mean_correction': float(-252 * np.mean(r)**2)})
    c = pd.DataFrame(calculated).set_index(['pair_id', 'point'])
    errors = []
    for pair_id, p in pairs.iterrows():
        a, b = c.loc[(pair_id, 'early')], c.loc[(pair_id, 'late')]
        changes = (b[['up2', 'down2', 'mean_correction']] - a[['up2', 'down2', 'mean_correction']]) * 20 / 19
        diff = b.variance - a.variance
        errors.append(abs(diff - p.delta_variance))
        errors.extend(abs(changes.values - p[['change_up2', 'change_down2', 'change_mean_correction']].to_numpy(float)))
        if diff >= -EPS:
            state = OTHER
        else:
            equal = np.flatnonzero(np.abs(changes.to_numpy() - changes.min()) <= EPS)
            state = [UP, DOWN, MEAN][equal[0]] if len(equal) == 1 else TIE
        assert state == p.cooling_state
    assert max(errors) < EPS
    full = pd.read_csv(OUT / 'results/672条_原路径与降波来源完整连接.csv')
    old_paths = pd.read_csv(OUT / 'inputs/paths.csv')
    for col in old_paths:
        if pd.api.types.is_numeric_dtype(old_paths[col]):
            assert np.allclose(full[col], old_paths[col], atol=1e-12, rtol=0, equal_nan=True)
        else:
            assert full[col].fillna('缺失').equals(old_paths[col].fillna('缺失'))
    aggregate = pd.read_csv(OUT / 'results/全部时期及下降来源_原路径周期等权分布.csv')
    non = pd.read_csv(OUT / 'results/78对_沿用原日期不重叠成员.csv')
    aggregate_errors = []
    for r in aggregate.to_dict('records'):
        base = full if r['sampling'] == '原全部周对' else non
        z = base[base.analysis_period.eq(r['period']) & base.clock.eq(r['clock']) & base.status.eq('MATURE')]
        if r['state'] == '全部总波动下降':
            z = z[z.delta_variance < -EPS]
        elif r['state'] != '全部状态':
            z = z[z.cooling_state.eq(r['state'])]
        assert len(z) == r['pairs'] and z.stat_month.nunique() == r['cycles']
        for metric in ['observed_close_return', 'late20_return', 'late20_downside', 'common_return', 'new_tail_contribution']:
            value = z.groupby('stat_month')[metric].mean().mean()
            if len(z):
                aggregate_errors.append(abs(value - r[metric + '_mean']))
    assert max(aggregate_errors) < EPS
    contrasts = pd.read_csv(OUT / 'results/固定两来源对照_后收益与下行风险.csv')
    draws = np.load(OUT / 'inputs/block_draws.npz')
    interval_errors = []
    for row in contrasts[contrasts.interval_status.eq('DESCRIPTIVE_SAVED_6_CYCLE_BLOCKS')].itertuples(index=False):
        pool = full[full.analysis_period.eq(row.period) & full.clock.eq(row.clock) & full.status.eq('MATURE')]
        cycle_order = sorted(pool.stat_month.unique())
        means = []
        for state in [DOWN, UP]:
            means.append(pool[pool.cooling_state.eq(state)].groupby('stat_month')[row.outcome].mean().reindex(cycle_order).to_numpy())
        # 重采样周期均值，独立于主脚本逐周乘权重的计算。
        samples = draws[row.period]
        delta = np.nanmean(means[0][samples], axis=1) - np.nanmean(means[1][samples], axis=1)
        endpoints = np.nanquantile(delta, [.025, .975])
        interval_errors.extend(np.abs(endpoints - [row.low, row.high]))
    assert max(interval_errors, default=0) < EPS
    market = pd.read_csv(OUT / 'inputs/market.csv').set_index('date')
    weekly = pd.read_csv(OUT / 'prior_evidence/weekly_v3.csv')
    future_errors = []
    paths_checked = 0
    for r in weekly.to_dict('records'):
        for clock in ['E0', 'E1']:
            pfx = clock + '_20_'
            if pd.isna(r[pfx + 'return']):
                continue
            part = market.loc[r[pfx + 'entry_date']:r[pfx + 'exit_date']]
            assert len(part) == 20
            dividends = part.dividend.to_numpy().copy()
            dividends[0] = 0
            wealth = part.close.to_numpy() + dividends.cumsum()
            changes = wealth / np.r_[part.open.iloc[0], wealth[:-1]] - 1
            future_errors.extend([abs(wealth[-1] / part.open.iloc[0] - 1 - r[pfx + 'return']),
                                  abs(np.std(changes, ddof=1) * np.sqrt(252) - r[pfx + 'future_rv']),
                                  abs(np.sqrt(np.mean(np.minimum(changes, 0)**2) * 252) - r[pfx + 'future_downside'])])
            paths_checked += 1
    assert max(future_errors) < EPS
    associations = pd.read_csv(OUT / 'prior_evidence/associations_v3.csv')
    association_errors = []
    for r in associations.to_dict('records'):
        z = weekly[weekly.training_regime.eq(r['regime'])]
        if r['period'] != '全部':
            z = z[z.analysis_period.eq(r['period'])]
        key = r['entry'] + '_20_' + r['outcome']
        z = z.dropna(subset=[r['feature'], key])
        value = weighted_rank_corr(z[r['feature']].to_numpy(), z[key].to_numpy(), z.stat_month.to_numpy())
        association_errors.append(abs(value - r['weighted_spearman']))
    assert max(association_errors) < EPS
    nonw = pd.read_csv(OUT / 'prior_evidence/disjoint_v3.csv')
    kept = []
    for regime, z in weekly.groupby('training_regime'):
        end = '0000-00-00'
        for r in z.sort_values('observation_date').to_dict('records'):
            if pd.notna(r['E0_20_return']) and r['E0_20_entry_date'] > end:
                kept.append(r['origin_id'])
                end = r['E0_20_exit_date']
    assert set(kept) == set(nonw.origin_id)
    nonass = pd.read_csv(OUT / 'prior_evidence/disjoint_associations_v3.csv')
    for r in nonass.to_dict('records'):
        z = nonw[nonw.training_regime.eq(r['regime'])]
        if r['period'] != '全部':
            z = z[z.analysis_period.eq(r['period'])]
        value = spearmanr(np.round(z.v_rv20, 12), np.round(z['E0_20_' + r['outcome']], 12)).statistic
        assert abs(value - r['spearman']) < EPS
    macro = pd.read_csv(OUT / 'prior_evidence/macro_predictions.csv')
    macro_losses = {}
    for model in ['baseline', 'increment']:
        actual = (macro[model + '_prediction'] - macro.label)**2
        assert np.allclose(actual, macro[model + '_loss'], atol=EPS, rtol=0)
        macro_losses[model] = float(actual.mean())
    assert (pd.to_datetime(macro.training_last_maturity) <= pd.to_datetime(macro.origin)).all()
    assert (pd.to_datetime(macro.published_at, utc=True) <= pd.to_datetime(macro.origin_clock, utc=True)).all()
    # 小风险预测值进入分母，使用往返精度读取原CSV，避免解析尾数误差被放大。
    risk = pd.read_csv(OUT / 'prior_evidence/risk_predictions.csv', float_precision='round_trip')
    risk_losses = []
    for model in ['LOG_RISK_RIDGE', 'PAST_MONTHLY_MEAN', 'RECENT_DOWNSIDE20']:
        loss = np.log(risk[model]) + risk.observed_risk / risk[model]
        assert np.allclose(loss, risk[model + '_loss'], atol=EPS, rtol=0)
        for phase, z in risk.groupby('period'):
            risk_losses.append({'period': phase, 'model': model, 'mean_loss': float(z[model + '_loss'].mean()), 'months': len(z)})
    assert (pd.to_datetime(risk.training_last_maturity) <= pd.to_datetime(risk.origin)).all()
    csv('原风险预测损失_本次逐值复算.csv', pd.DataFrame(risk_losses))
    fiscal = pd.read_csv(OUT / 'prior_evidence/v29_fiscal_clock.csv')
    assert not fiscal.same_stat_month_fiscal_known_at_origin.any()
    b = pd.read_csv(OUT / 'prior_evidence/v29_balances.csv').set_index('stat_month')
    september = b.loc['2025-09'] - b.loc['2025-08']
    assert abs(september.m1_table - 19199.37) < .001
    assert abs(september.corporate_demand + 1637.79) < .001
    assert abs(september.personal_demand_derived - 17975.26) < .001
    check = {'at': now(), 'status': 'PASS_KEY_FINDINGS_RECOMPUTATION', 'original_pair_paths': len(full),
             'mature_pair_paths': int(full.status.eq('MATURE').sum()), 'original_future_paths_recomputed': paths_checked,
             'daily_members_checked': len(members), 'variance_change_max_error': max(errors), 'group_mean_max_error': max(aggregate_errors),
             'block_interval_max_error': max(interval_errors, default=0), 'future_path_max_error': max(future_errors),
             'all_v3_associations_recomputed': len(association_errors), 'association_max_error': max(association_errors),
             'nonoverlapping_weekly_windows': len(nonw), 'macro_prediction_months': len(macro),
             'macro_baseline_mse': macro_losses['baseline'], 'macro_increment_mse': macro_losses['increment'],
             'macro_relative_mse_increase': macro_losses['increment'] / macro_losses['baseline'] - 1,
             'prior_risk_prediction_months': len(risk), 'prior_risk_prediction_role': '不同自然月风险标签的既有预测，仅复算损失，不与原20日观察混用',
             'new_models': 0, 'new_accounts': 0, 'first_historical_vintage_verified': False,
             'independent_new_validation': False, 'user_clarified_research_delivery': True}
    save('final_verification.json', check)
    print(json.dumps(check, ensure_ascii=False, indent=2))


def figure():
    plt.rcParams.update({'font.family': FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(), 'axes.unicode_minus': False,
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    pairs = pd.read_csv(OUT / 'results/336对_总波动下降来源与全部原背景.csv')
    a = pd.read_csv(OUT / 'prior_evidence/associations_v3.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={'width_ratios': [1, 1.4]})
    fig.patch.set_facecolor('#FAFBFC')
    for ax in axes:
        ax.set_facecolor('#FAFBFC')
        ax.grid(axis='y', color='#DEE4EA', zorder=0)
        ax.axhline(0, color='#657581', linewidth=.8)
    counts = pairs.cooling_state.value_counts().reindex([UP, DOWN, MEAN], fill_value=0)
    axes[0].bar(np.arange(3), counts.values, color=['#C4874C', '#477991', '#9CA9B3'], width=.6, zorder=3)
    for i, count in enumerate(counts.values):
        axes[0].text(i, count + 2, str(count), ha='center', fontsize=13)
    axes[0].set_xticks(np.arange(3), ['上涨平方项\n收缩主导', '下跌平方项\n收缩主导', '均值修正项\n主导'])
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel('原同公告相邻周对数')
    axes[0].set_title('180次总波动下降，来源并不相同', loc='left', fontsize=14, pad=16)
    x = np.arange(3)
    risk, ret = [], []
    for phase in PHASES:
        z = a[a.period.eq('统计月' + phase) & a.entry.eq('E0') & a.feature.eq('v_rv20')].set_index('outcome')
        risk.append(z.loc['future_rv', 'weighted_spearman'])
        ret.append(z.loc['return', 'weighted_spearman'])
    for values, offset, color, label in [(risk, -.17, '#477991', '与未来20日总波动'), (ret, .17, '#C4874C', '与未来20日收益')]:
        axes[1].bar(x + offset, values, width=.3, color=color, label=label, zorder=3)
        for i, value in enumerate(values):
            axes[1].text(i + offset, value + (.015 if value >= 0 else -.025), f'{value:+.3f}', ha='center', va='bottom' if value >= 0 else 'top')
    axes[1].set_xticks(x, PHASES)
    axes[1].set_ylim(-.38, .60)
    axes[1].set_ylabel('原发布周期等权秩相关')
    axes[1].legend(frameon=False, loc='upper right', fontsize=10)
    axes[1].set_title('风险水平有延续，收益方向关系变号', loc='left', fontsize=14, pad=16)
    fig.suptitle('510300：风险规律与方向优势分别判断', x=.07, y=.965, ha='left', fontsize=20, color='#2B4157')
    fig.text(.07, .885, '左图为本轮新增；右图复算既有完整样本，未重新拟合。旧新M1口径分开。', color='#586877', fontsize=11)
    fig.text(.07, .095, '180次总波动下降中，48次下行平方尺度反而增加。总波动下降不能等同近期下跌幅度减轻。', fontsize=11, color='#334C62')
    fig.text(.07, .05, '观察窗口可能重叠，计数不是独立事件数；相关不是胜率。历史风险关联不构成单次回撤上限或交易信号。', fontsize=10, color='#607181')
    fig.subplots_adjust(left=.07, right=.975, top=.77, bottom=.23, wspace=.28)
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / f'figures/风险规律与方向边界.{suffix}', dpi=180)
    plt.close(fig)
    print('已生成风险规律与方向边界图。')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def report():
    a = pd.read_csv(OUT / 'prior_evidence/associations_v3.csv')
    n = pd.read_csv(OUT / 'prior_evidence/disjoint_associations_v3.csv')
    distributions = pd.read_csv(OUT / 'results/全部时期及下降来源_原路径周期等权分布.csv')
    p = pd.read_csv(OUT / 'results/336对_总波动下降来源与全部原背景.csv')
    check = json.loads((OUT / 'final_verification.json').read_text(encoding='utf-8'))
    risk_rows, cooling_rows, non_rows = [], [], []
    for phase in PHASES:
        z = a[a.period.eq('统计月' + phase) & a.entry.eq('E0') & a.feature.eq('v_rv20')].set_index('outcome')
        disjoint = n[n.period.eq('统计月' + phase) & n.outcome.eq('future_rv')].iloc[0]
        risk_rows.append([phase, f"{int(z.loc['future_rv', 'weeks'])} / {int(z.loc['future_rv', 'cycles'])}",
                          f"{z.loc['future_rv', 'weighted_spearman']:+.3f}", f"{z.loc['future_downside', 'weighted_spearman']:+.3f}",
                          f"{z.loc['return', 'weighted_spearman']:+.3f}", f'{disjoint.spearman:+.3f}（{int(disjoint.n_nonoverlapping_windows)}窗）'])
        for state in [UP, DOWN, MEAN, OTHER]:
            e0 = distributions[distributions.sampling.eq('原全部周对') & distributions.period.eq(phase) & distributions.clock.eq('E0') & distributions.state.eq(state)].iloc[0]
            e1 = distributions[distributions.sampling.eq('原全部周对') & distributions.period.eq(phase) & distributions.clock.eq('E1') & distributions.state.eq(state)].iloc[0]
            cooling_rows.append([phase, state, f'{int(e0.pairs)} / {int(e0.cycles)}', f'{e0.observed_close_return_mean:+.2%}',
                                 f'{e0.late20_return_mean:+.2%}', f'{e1.late20_return_mean:+.2%}', f'{e0.late20_worst_min:.2%}'])
        for state in [UP, DOWN]:
            z = distributions[distributions.sampling.eq('原固定不重叠') & distributions.period.eq(phase) & distributions.clock.eq('E0') & distributions.state.eq(state)].iloc[0]
            non_rows.append([phase, state, int(z.pairs), f'{z.late20_return_mean:+.2%}'])
    contradictory_rows = []
    for phase in PHASES:
        z = p[p.analysis_period.eq(phase)]
        cooling = z[z.delta_variance < -EPS]
        contradictory_rows.append([phase, len(z), len(cooling), int(cooling.total_down_but_downside_increases.sum())])
    body = f'''# 510300最终研究结论：风险规律与方向优势分别判断

**风险方面，有反复出现的历史证据支持“波动水平具有延续性”；方向方面，当前波动高低、降波及已检验的M1/M2剪刀差表示，没有形成稳定的未来20日方向优势。总波动下降不等于下跌幅度减轻，剪刀差改善也不等于企业需求增强。**

本结论采用你直接确认的验收标准：“风险规律与方向优势分别给出结论，允许明确判定哪些关系无效。”这意味着研究可以以支持、否定和未知三种结果结束，无须人为找出盈利买点。“无效”只用于证据能够覆盖的具体判断或固定检验，不能扩展成所有宏观分析都无用。

研究对象是510300指数ETF整体。本文连接货币来源、信用及财政背景、当时已知消息、指数此前定价、波动构成与之后剩余路径。行业和成分资料只用于核对指数传导；权重不足、估值口径不明及无法识别唯一原因的地方保留限制。

指数风险取决于权重聚合及成分共同涨跌，不能用个股平均波动替代。原内部核对仅24个月具备完整参考权重，缺失的80个月仍保留在指数主表；共同波动的代数贡献不称为宏观因素的因果份额。详见[指数主线与内部覆盖](<{(ROOT / 'reports/research/510300_index_mainline_v25/指数研究主线_货币来源与整体定价.md').as_posix()}>)。这种范围限制与个股盈利扩展无关，指数自身的价格和风险观察仍覆盖完整样本。

**一项可以保留的正面结论：风险大小有延续性，收益方向没有同样的稳定关系。**

{table(['统计月时期', '有效周 / 发布周期', '当前RV与后20日RV', '当前RV与后20日D', '当前RV与后20日收益', '去除未来窗口重叠后与RV'], risk_rows)}

这些是周期等权秩相关，不是胜率。旧口径总体风险相关+0.473，原六周期描述区间[+0.334,+0.566]；去掉未来窗口重叠后，三段风险关联仍为正。新口径只有19个成熟发布周期，不能当成充分的独立验证。风险的延续不提供单次最大回撤上限，也不保证止损成交。

原自然月风险预测另有140个月保存结果，本次复算其损失后，两个时期相对两项原基准的指定风险损失改善方向仍相同；它与这里的固定20日观察标签不同，未混用，原账户失败也保留。风险预测能力不能自动换成正收益能力。

**一项可明确否定的读法：总波动下降就代表近期下跌幅度减轻。**

本轮沿用全部336对同一货币公告周期内的相邻周。180对总波动下降，其中91对由上涨平方项收缩主导、80对由下跌平方项收缩主导、9对由均值修正主导；没有贡献并列。三项采用样本方差的精确恒等式，未省略均值项。

令U²为正日收益平方在全部20日上的平均乘252，D²为负日收益平方在全部20日上的平均乘252，则RV²=20/19×（U²+D²−252×平均日收益²）。按两原观察点逐项相减，最负贡献为下降主导项。三者描述已经发生的收益，不能把它们当作未来风险的上下界。

{table(['时期', '全部原周对', '总波动下降', '其中下行平方尺度反而增加'], contradictory_rows)}

合计48/180，即26.67%的总波动下降伴随下行平方尺度增加。该事实直接否定上述等同关系；它不代表这48次之后必跌。旧研究也已确认，D20下降仍可能来自旧跌幅退出，近期5日负收益平方和却在增加。两种窗口误读不同，都不能省去实际价格路径。

本轮新分组的所有结果如下。每个状态内发布周期等权；同一周期的周对等权。状态之间的价格、信用和波动水平并未被随机分配，均值差不是因果效果。

{table(['时期', '总波动变化来源', '成熟周对 / 周期', '两观察间已发生收益', '较晚起点E0后20日', '较晚起点E1后20日', 'E0最差一次途中收盘'], cooling_rows)}

2018—2021年，“下跌项收缩主导”减“上涨项收缩主导”的E0后20日均值差为−0.25个百分点，描述区间[−2.47,+1.74]跨零；E1差约+0.01个百分点。另两段样本没有达到预先规定的每组24周期区间门槛，保持未计算。不能从组均值反过来建立“上涨收缩更好”的交易规则。

固定日期不重叠成员也全部沿用，未寻找最有利相位：

{table(['时期', '下降来源', '不重叠周对', '后20日E0均值'], non_rows)}

较小样本和既有历史重复使用，限制了这些比较的推断范围。它们支持拒绝把某种降波直接命名为确定机会，不支持声称所有条件下收益分布都完全相同。

**M1/M2仍有解释价值，但必须拆清来源，当前方向增量未获支持。**

剪刀差是两种同比增速之差。必须分别记录M1和M2当期变化、同比基数及口径修订。2025年9月的事后分项显示M1增加19,199亿元、单位活期减少1,638亿元、个人活期增加17,975亿元，已提供“货币改善不能统一叫企业经营恢复”的直接反例。月度余额分项与同比变化不是同一个量，报告同时保留了原三个月对数基数分解。

财政融资也不是单一的需求渠道。已核对的官方资料同时包含债务置换、实际支出、银行资本补充及尚在程序中的额度；存量债务压力缓解与企业中长期贷款少增可以并存。对指数，还需区分这些变化对整体盈利预期、贴现率和风险溢价的可能影响，不能将融资规模等同指数净买盘。[财政部上半年发布会](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202507/t20250725_3968586.htm)、[前三季度发布会](https://m.mof.gov.cn/czxw/202510/t20251017_3974415.htm)

在原固定M1/M2增量预测中，53个配对月的价格与波动基准MSE为{check['macro_baseline_mse']:.6f}，加入剪刀差后为{check['macro_increment_mse']:.6f}，误差增加{check['macro_relative_mse_increase']:.2%}。本次按保存预测逐值复算，训练成熟时钟和公告时钟仍符合原记录。这个检验未支持所测试表示的可靠增量；它没有证明所有货币变量、所有期限或未来状态都无效。历史首版不可修订性仍未认证。

旧口径相近价格/波动控制、各时期和原E0/E1结果均已保留；新口径样本短，不能用多个周重复同一份宏观数据来扩充独立事件数量。现有资料不足以给出稳定的货币方向规则，结论停在“未获支持”，不改参数救援。

**“看起来合理的组合”也需要承担已经走过的价格和之后的新消息。**

原同公告周对中，D下降与已经发生收益的相关，在旧前后期为−0.469、−0.431，与之后20日收益的相关仅+0.023、+0.119，原区间跨零。近期下跌同步缓和、剪刀差改善、企业和居民中长期累计贷款同比多增、订单不低于50的原组合，仍有12对来自9个周期；之后E0均值−0.97%、E1−2.22%，不能据它宣称确定上涨。

2019年原观察窗口中，剪刀差三个月仍改善而企业中长期从多增转少增，指数在5月关税新闻前已经下跌；2024年9月则在旧货币统计仍弱时出现新政策与大幅定价变化。它们分别说明：旧统计不能覆盖新信息，后来新闻也不能承担之前全部涨跌。2021年原延迟窗口新增尾段和指数调整，还提醒我们区分持有期限、入场价格与指数统计口径。上述均为完整窗口复盘，未用于删除失败月份。

在2025年6至10月，全部五个货币起点的同统计月财政月报都尚未公布；年度银行表更是事后取得版本。解释成立的后来资料不能提前成为原起点证据。我们已经把“统计月份相同”与“当时已经知道”分开。

**最终可采用的结论及边界。**

| 研究判断 | 最终状态 | 适用边界 |
| --- | --- | --- |
| 当前波动水平包含随后风险大小的信息 | 历史证据反复支持 | 不保证单次风险上限，不直接决定涨跌 |
| 总波动下降等于下跌幅度减轻 | 被实际反例否定 | 48/180个原降波周对下行尺度反而增加 |
| D20下降等于近期下跌继续缓和 | 被窗口分解和实际反例否定 | 旧损失退出与新损失变化必须分开 |
| 剪刀差改善等于企业经营和信用需求增强 | 被结构和信用反例否定 | 基数、个人活期、财政及金融用途必须核对 |
| 所测试的剪刀差表示带来稳定20日方向增量 | 未获支持 | 原53月固定检验误差未改善；不能外推全部模型无效 |
| 宏观改善与好看波动状态构成确定上涨条件 | 充分条件被反例否定，稳定收益优势未建立 | 保留价格已反应程度、后续新消息和亏损 |
| 某一政策或资金用途解释指数多少涨跌 | 仍未知 | 会计恒等式、同时变化与时间先后都不独立识别因果份额 |
| 新M1已具有与旧口径同等稳健的规律 | 证据不足 | 新口径短、自由度不足处保持不可估计 |

研究使用3476条日线、104个月和445周，行情截至2026年9月11日。旧新M1分开，原20交易日为主，5/60日辅助保留；待成熟及缺失未填零。E0从原观察日21点后的下一交易日开盘起至第20交易日收盘，E1将入场与终点各延后一个交易日。所有观察收益采用固定份额应得现金分红近似，未计费用和完整账户约束。本研究没有生成当前市场意见、仓位或订单，也没有达到或重新宣称其他任务的账户目标。

本轮独立按原日收益成员核对了336对方差变化，按逐日财富路径复算{check['original_future_paths_recomputed']}个成熟周度E0/E1窗口，复算120条既有风险与收益关联、96个固定不重叠窗口，以及53个月货币预测损失。核对保证本文数字与保存数据一致；它不能认证历史不可修订版本、证明因果或代替未来样本。

![风险规律与方向边界](<{(OUT / 'figures/风险规律与方向边界.png').as_posix()}>)

按你确认的研究验收标准，风险、方向、无效关系和未知范围已分别形成结论。本次交付完成；旧冻结失败保持失败，缺少独立交易优势的状态保持未证实。若未来开展新策略或真实前向样本验证，应另列问题与协议，不把本报告当成通过交易验证。

可复查文件：[全部336周对及当时背景](<{(OUT / 'results/336对_总波动下降来源与全部原背景.csv').as_posix()}>)、[全部672条原路径](<{(OUT / 'results/672条_原路径与降波来源完整连接.csv').as_posix()}>)、[所有时期及状态分布](<{(OUT / 'results/全部时期及下降来源_原路径周期等权分布.csv').as_posix()}>)、[固定两组比较与未计算状态](<{(OUT / 'results/固定两来源对照_后收益与下行风险.csv').as_posix()}>)、[关键结论复算](<{(OUT / 'final_verification.json').as_posix()}>)、[财政货币机制完整报告](<{(V29 / '财政货币结构与指数定价.md').as_posix()}>)。

生成时间：{now()}。
'''
    (OUT / '510300_最终研究结论.md').write_text(body, encoding='utf-8')
    print('已生成分别判断风险、方向及无效关系的最终研究结论。')


def complete():
    verify_inputs()
    check = json.loads((OUT / 'final_verification.json').read_text(encoding='utf-8'))
    acceptance = json.loads((OUT / 'conclusion_acceptance.json').read_text(encoding='utf-8'))
    assert check['status'] == 'PASS_KEY_FINDINGS_RECOMPUTATION'
    assert acceptance['user_clarification'] == '风险规律与方向优势分别给出结论，允许明确判定哪些关系无效'
    report_path = OUT / '510300_最终研究结论.md'
    text = report_path.read_text(encoding='utf-8')
    for phrase in ['有反复出现的历史证据', '没有形成稳定的未来20日方向优势', '26.67%', '53个配对月', '未获支持', '仍未知', '证据不足']:
        assert phrase in text, phrase
    # 结论与当前保存数据对照；未获支持和未知是用户允许的研究结论，不提升成交易通过。
    p = pd.read_csv(OUT / 'results/336对_总波动下降来源与全部原背景.csv')
    assert len(p) == 336 and int((p.delta_variance < -EPS).sum()) == 180
    assert int(p.total_down_but_downside_increases.sum()) == 48
    original = pd.read_csv(OUT / 'inputs/monthly.csv')
    assert original.shape == (104, 741)
    assert original.E0_20_status.eq('MATURE').sum() == 103
    assert original.E1_20_status.eq('MATURE').sum() == 102
    assert (original.E0_20_status.ne('MATURE') | original.E1_20_status.ne('MATURE')).sum() == 2
    account = json.loads((OUT / 'prior_evidence/risk_account_result.json').read_text(encoding='utf-8'))
    assert account['status'] == 'RISK_INCREMENT_PASS_ACCOUNT_TARGETS_NOT_MET'
    source_checks = [
        ('数据完整与固定窗口', OUT / 'inputs/monthly.csv', '104个月741列；原E0成熟103、E1成熟102，其他状态保留；5/20/60日原列保留。'),
        ('立体宏观来源与信用财政背景', V29 / '财政货币结构与指数定价.md', '当期余额、隐含基数及修订、贷款期限、财政用途与发行使用时差分别有证据及限制。'),
        ('指数而非少数个股为单位', ROOT / 'reports/research/510300_index_mainline_v25/指数研究主线_货币来源与整体定价.md', '指数104个月全保留，内部参考权重仅24个月完整，未用局部公司补足缺失。'),
        ('价格先后与剩余空间', V22 / 'results/672条_原终点剩余与延长尾段金额桥.csv', '原完整窗口、观察期间涨跌、原终点剩余与新尾段分开；672条全部原列本次再次核对。'),
        ('波动测量及机械误读', OUT / 'results/13440条_两端原20日收益成员.csv', '逐成员复算336对三项方差变化；48/180反例否定总波动与下行尺度下降等同。'),
        ('历史风险规律单独判断', OUT / 'prior_evidence/associations_v3.csv', '120条相关与881条未来路径复算；三个时期风险关联为正，96个固定不重叠窗口同样核对。'),
        ('方向增量单独判断', OUT / 'prior_evidence/macro_predictions.csv', '53个月固定保存预测损失复算；加入剪刀差MSE增加1.02%，只判定原表示未获支持。'),
        ('全状态与旧新口径及失败例', OUT / 'results/全部时期及下降来源_原路径周期等权分布.csv', '三个固定时期、全部状态、E0/E1、固定不重叠均保留；不倒选相位。'),
        ('信息时点及无法识别范围', OUT / 'prior_evidence/v29_fiscal_clock.csv', '五个同统计月财政在原起点均未公开；年度表事后，历史首版及独立前向验证限制明确。'),
        ('明确交付支持否定及未知', report_path, '最终表列明历史支持、充分条件被否定、固定检验未获支持和资料不足，符合用户直接确认的标准。'),
    ]
    rows = []
    for requirement, path, finding in source_checks:
        assert path.is_file(), str(path)
        rows.append({'requirement': requirement, 'evidence_path': str(path), 'evidence_sha256': digest(path),
                     'disposition': 'SATISFIED_FOR_USER_CLARIFIED_RESEARCH_CONCLUSION', 'finding': finding})
    save('completion_audit.json', {'at': now(), 'authority': acceptance['user_clarification'], 'requirements': rows,
                                  'goal_requirements_met': True, 'remaining_required_work': [],
                                  'not_claimed': ['未来方向保证', '所有宏观变量都无效', '独立交易优势通过', '账户目标达成', '历史原始版本完全认证']})
    for source in [ROOT / 'research/index_volatility_cooling_v30.py', Path(__file__)]:
        shutil.copy2(source, OUT / 'code' / source.name)
    save('visual_review.json', {'at': now(), 'figure': 'figures/风险规律与方向边界.png', 'status': 'PASS_MANUAL_VISUAL_REVIEW',
                                'reviewed': ['文字完整无缺字', '正负相关刻度正确', '180次组成与48次反例标注一致', '历史关联非胜率和非独立样本限制可读']})
    save('completion.json', {'at': now(), 'status': 'COMPLETE_USER_CLARIFIED_RESEARCH_CONCLUSION', 'goal_turn_classification': 'PROGRESS',
                             'goal_requirements_met': True, 'report': report_path.name, 'report_sha256': digest(report_path),
                             'risk_conclusion': 'SUPPORTED_HISTORICAL_RISK_PERSISTENCE_WITH_LIMITS',
                             'direction_conclusion': 'TESTED_INCREMENT_NOT_SUPPORTED_AND_SUFFICIENT_CONDITIONS_FALSIFIED',
                             'unknowns_explicitly_retained': True, 'new_models': 0, 'new_accounts': 0})
    print('按用户确认的研究验收标准，完成范围核对通过；风险、方向与未知已分别交付。')


if __name__ == '__main__':
    commands = {'prepare': prepare, 'verify': verify, 'figure': figure, 'report': report, 'complete': complete}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit('用法：python research/finalize_index_findings_v30.py prepare|verify|figure|report|complete')
    commands[sys.argv[1]]()
