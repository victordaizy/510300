"""保存失效说明及附带发现的旧申赎日期口径问题，不改变旧研究。"""
import numpy as np
import pandas as pd

from research.direct_stump_failure_explanation_v1 import ROOT, SOURCE, OUT, INDEX, STUDY, PAIRS
from research.intraday_overnight_increment_v1 import write_json, digest, now
from research.monthly_single_factor_walkforward_v1 import read


DELIVERY = ROOT / 'deliverables/510300有限历史挖掘_失效原因与因子口径_20260914'
REPORT = DELIVERY / '策略为何失效_空仓选择分界规则与费用.md'
FACTOR_REPORT = DELIVERY / '旧日频申赎因子的日期口径说明.md'
NAMES = {'CONFIRMED_STUMP': '五日分界主规则', 'NO_FACTOR_LEARNER': '无因子学习', 'FIXED_WEEKLY_VOL10': '简单周度波动控制'}
PERIODS = {'early': '2015—2019年', 'main': '2020年至2026年9月11日'}
COSTS = {'BASE': '基础', 'STRESS': '压力'}


def link(path, title):
    return f'[{title}](<{path.as_posix()}>)'


def supplemental_share_note():
    source = ROOT / 'data/raw/flow/510300_fund_share_daily_tushare.parquet'
    downloader = ROOT / 'scripts/download_daily_flow_data_tushare.py'
    feature_code = ROOT / 'research/daily_etf_flow_direction_challenger.py'
    old_report = ROOT / 'reports/research/510300_daily_etf_flow_direction_report.json'
    shares = pd.read_parquet(source)
    old = read(old_report)
    disclosure_columns = [c for c in shares if any(s in c.lower() for s in ['publish', 'available', 'announce', 'receipt', 'disclos'])]
    assert not disclosure_columns
    note = {'status': 'NO_VIEW_HISTORICAL_PUBLICATION_TIME_NOT_ESTABLISHED_IN_READ_SOURCE', 'checked_at': now(),
        'scope': '完成账户解释时附带检查旧日频申赎资料；非本轮固定绩效分析的输入',
        'source': str(source.relative_to(ROOT)), 'source_rows': len(shares),
        'first_date': str(pd.Timestamp(shares.date.min()).date()), 'last_date': str(pd.Timestamp(shares.date.max()).date()),
        'columns': list(shares), 'historical_publication_time_columns': disclosure_columns,
        'batch_retrieved_at_values': [str(x) for x in shares.retrieved_at.unique()],
        'source_date_meaning': '下载程序将原trade_date复制为date；原接口定义为交易变动日期',
        'consumer_behavior': '原特征程序把date改名为share_snapshot_date，按相同日期允许合并，注释称真实披露日',
        'official_document': 'https://tushare.pro/document/2?doc_id=207',
        'official_document_checked_in_browser': True,
        'official_document_output_fields': ['ts_code', 'trade_date', 'fd_share'],
        'existing_rejection': old['status'], 'existing_evaluation': old['evaluation'],
        'finding_limit': '证明本次所读文件与代码未建立历史披露时点；没有证明每一行实际发生前视，也不证明全项目没有其他时间证据',
        'disposition': '保留原失败；不给本文件同日可用保证，不随意加一日延迟后当作已证明，不为改善成绩倒转因子方向',
        'new_source_downloads': 0, 'new_accounts': 0, 'new_model_fits': 0,
        'files': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in [source, downloader, feature_code, old_report]]}
    write_json(OUT / 'supplemental_fund_share_date_semantics.json', note, exclusive=True)
    FACTOR_REPORT.write_text('\n'.join([
        '# 旧日频申赎因子的日期口径说明', '',
        '这项问题属于旧的“510300日频申赎与融资行为”研究。此次重新读取的是其缓存数据、下载程序和因子程序，没有重跑旧预测或生成交易账户。', '',
        '## 已确认的事实', '',
        f"缓存有{len(shares)}行，覆盖{note['first_date']}至{note['last_date']}。确实是日频份额资料，不能说它只有季度频率。", '',
        '下载程序将接口的交易变动日期复制为本地日期列。官方文档给出的输出为基金代码、交易变动日期和基金份额；该日期不是文档定义的公告日。来源：[Tushare基金规模官方文档](https://tushare.pro/document/2?doc_id=207)。', '',
        '本地文件另外保存整批取得时间，均为2026年8月13日17时01分16秒附近；没有每一笔历史数据的实际发布时间或当时取得回执。后来下载历史，只证明下载时已经可见。', '',
        '旧因子程序注释称“从真实披露日向后合并份额”，但实际使用上述交易变动日期，并允许同一天合并。注释的含义超过了本次所见数据支持的范围。', '',
        '## 如何处理', '',
        '当前结论是“历史披露时点尚未建立”。这不等于证明所有记录都使用了未来信息，也不等于证明全项目其他材料中没有时间证据；但这份缓存自身不足以支持旧代码的同日已知声明。', '',
        '原日频模型已经失败，663个成熟预测的区分指标约0.470，概率误差相对基准恶化约5.93%。原失败保留；不能把它倒过来使用便宣称发现新方向，也不能因为发现时点问题就猜测修正后必然赚钱。', '',
        '日频ETF份额变化也不是全市场公募净现金申购。份额交易变动日期、公告可用日期、整批下载时间以及现金流口径都需要分别说明。', '',
        '在已有原始披露证据能够建立可用时点前，不将此缓存直接当作已证明历史实时可用的新信号。随意延迟一天只能是另一个假设，不能代替原始证据。依用户使用现有免费来源的要求，本次只读取现成缓存和免费网页，没有新调用付费数据。', '',
        '## 直接证据', '',
        '- ' + link(downloader, '下载和日期转换程序，日期转换在第51行'),
        '- ' + link(feature_code, '旧因子使用日期的程序，第47—69行'),
        '- ' + link(source, '本次实际读取的历史份额缓存'),
        '- ' + link(old_report, '原日频预测失败记录'),
        '- ' + link(OUT / 'supplemental_fund_share_date_semantics.json', '此次日期语义检查及文件身份'), '',
    ]), encoding='utf-8')
    return note


def verify_saved_tables(result):
    daily = pd.read_csv(OUT / 'daily_paired_components.csv')
    windows = pd.read_csv(OUT / 'subsequent_active_rule_windows.csv')
    transitions = pd.read_csv(OUT / 'all_training_confirmation_transitions.csv')
    aggregates = pd.read_csv(OUT / 'paired_annual_arithmetic_components.csv')
    components = ['average_weight_part', 'timing_part', 'execution_rights_part', 'friction_part']
    np.testing.assert_allclose(daily[components].sum(axis=1), daily.net_return, atol=1e-12, rtol=0)
    for r in aggregates.itertuples():
        group = daily[(daily.period == r.period) & (daily.cost == r.cost) & (daily.left == r.left) & (daily.right == r.right)]
        window = windows[(windows.period == r.period) & (windows.cost == r.cost) & (windows.left == r.left) & (windows.right == r.right)]
        assert len(group) == (1219 if r.period == 'early' else 1624)
        assert len(group) == window.days.sum()
        assert not group.date.duplicated().any()
        for key in components + ['net_return']:
            np.testing.assert_allclose(group[key].mean() * 242, getattr(r, key), atol=1e-11, rtol=0)
        np.testing.assert_allclose(window.annual_arithmetic_contribution.sum(), r.net_return, atol=1e-11, rtol=0)
    assert len(transitions) == result['training_updates']
    assert transitions.training_feature_winner.sum() == result['training_feature_winners']
    assert transitions.accepted.sum() == result['feature_winners_accepted']
    receipt = {'status': 'PASS_SAVED_DAILY_WINDOW_AND_ANNUAL_IDENTITIES', 'checked_at': now(),
        'pair_daily_rows': len(daily), 'subsequent_windows': len(windows), 'training_updates': len(transitions),
        'annual_comparisons': len(aggregates), 'new_accounts': 0, 'new_model_fits': 0,
        'scope': '保存表再求和、时间窗口覆盖、指标一致性；不重复上一轮完整成交模型核对'}
    write_json(OUT / 'saved_verification.json', receipt, exclusive=True)
    return windows, receipt


def main():
    result = read(OUT / 'result.json')
    windows, verification = verify_saved_tables(result)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    share_note = supplemental_share_note()
    cmp = {(r['period'], r['cost'], r['left'], r['right']): r for r in result['comparisons']}
    total = cmp['main', 'STRESS', 'CONFIRMED_STUMP', 'FIXED_WEEKLY_VOL10']
    blank = cmp['main', 'STRESS', 'NO_FACTOR_LEARNER', 'FIXED_WEEKLY_VOL10']
    signal = cmp['main', 'STRESS', 'CONFIRMED_STUMP', 'NO_FACTOR_LEARNER']
    no_factor_fraction = blank['net_return'] / total['net_return']
    lines = ['# 510300策略为何失效：空仓选择、五日分界和费用', '',
        '**目标仍未达到。此次已经把上一轮失败定位到具体环节：根据过去表现学习空仓这一层，是该轮账户落后的主要部分；五日涨跌分界的训练优势也没有持续到随后交易。费用差额占比较小。**', '',
        '本次只读取已保存的账户和月度学习记录。未增加候选、训练模型或回测账户，未改旧参数。', '',
        '## 一、给老板的一段话', '',
        '我们把这套策略拆开检查后发现，亏损差距主要出在择时判断。模型在过去一年半里挑出的买卖规则，到了随后半年往往不再有效；即使通过那半年的确认，再往后的交易也可能失效。另一个更大的问题是，模型根据过去表现决定空仓，经常错过了后续收益。交易成本有影响，但不足以解释主要差距。因此这套方法不值得继续调参数，下一步应寻找有新证据支持的交易信息。', '',
        '## 二、差距主要来自哪一层', '',
        '对照顺序固定：简单周度波动控制 → 加入空仓学习 → 再加入五日涨跌分界。每一层使用上一轮已经独立运行的完整账户。下表是年化算术平均日收益差，单位为百分点；它可相加，不能与复合年化收益混用。', '',
        '| 时期 | 费用 | 空仓学习相对简单基准 | 五日分界相对无因子学习 | 主规则相对简单基准合计 |',
        '|---|---|---:|---:|---:|']
    for period in ['early', 'main']:
        for cost in ['BASE', 'STRESS']:
            row = [cmp[period, cost, left, right]['net_return'] * 100 for left, right in PAIRS]
            lines.append(f'| {PERIODS[period]} | {COSTS[cost]} | {row[1]:+.2f} | {row[0]:+.2f} | {row[2]:+.2f} |')
    lines += ['', f"主时期压力情景，空仓选择这层的差额占总落后约{no_factor_fraction:.1%}。它是两条已保存账户之间的记账比较，包含各自长期持仓和现金路径，不能把占比理解为独立因果效应。", '',
        '五日分界在早期略有帮助，到了2020年以来反而拖累。不能只截取早期给它认定有效，也不能把这个负结果倒转为“反着交易就赚钱”。', '',
        '## 三、成本到底影响了多少', '',
        '以下拆分主时期压力下，主规则相对简单周度基准的差额。正数表示抵消部分落后，负数表示增加落后。', '',
        '| 来源 | 年化算术收益贡献差，百分点 |', '|---|---:|']
    for key, label in [('average_weight_part', '平均仓位较低对应的差额'), ('timing_part', '仓位变化与市场涨跌配合的差额'),
                       ('execution_rights_part', '交易时点及分红权益等剩余差额'), ('friction_part', '佣金和滑点差额'), ('net_return', '合计')]:
        lines.append(f'| {label} | {total[key] * 100:+.2f} |')
    lines += ['', '计算先以每个账户的前日净值为分母，分开账面费用，再把前日持仓乘ETF当日含分红收益这一项拆成平均仓位与仓位变化两部分。每个日账和整个时期的加总关系均已复核。', '',
        '费用差额约−0.19个百分点，主规则总落后约−4.44个百分点。因此本次没有证据支持靠修改费用假设来挽救方法。把原路径账面费用加回，只是记账拆分；如果实际没有费用，未来整手数量和持仓会改变，不能将简单加回称为零费用策略回测。', '',
        '## 四、规则是在什么时候失效', '',
        f"全部{result['training_updates']}次更新中，训练阶段有{result['training_feature_winners']}次选择了五日分界为第一名。这68次记录在训练段相对当次无因子基准的平均风险调整评分优势为{result['training_feature_winner_mean_utility_gap'] * 100:+.2f}个百分点；在随后确认段，平均差额转为{result['confirmation_feature_winner_mean_utility_gap'] * 100:+.2f}个百分点。评分是事前确定的均值减方差惩罚，不是复合年化或夏普。", '',
        f"其中{result['feature_winners_rejected_in_confirmation']}次在确认阶段被拒绝，{result['feature_winners_accepted']}次得到接受。月度训练和确认窗口大量重叠，68次记录不能当作68个独立实验。训练优势衰减可以来自选择噪声和市场关系变化，本次没有计算两者各占多少。", '',
        '进一步检查确认之后的真实研究账户窗口。窗口从该月规则首次在初始化或周五收盘参与判断后的下一交易日开始，到下一月规则实际生效前结束；模型月初更新但尚未到周五时，仍归入旧规则。账户仓位和资金连续保留。', '',
        '| 压力费用下的时期 | 通过确认且后续完整的窗口 | 随后胜过无因子学习 | 随后落后 |', '|---|---:|---:|---:|']
    for period in ['early', 'main']:
        r = next(x for x in result['subsequent_summaries'] if x['period'] == period and x['cost'] == 'STRESS' and x['left'] == 'CONFIRMED_STUMP' and x['right'] == 'NO_FACTOR_LEARNER')
        lines.append(f"| {PERIODS[period]} | {r['accepted_complete_windows']} | {r['accepted_window_wins']} | {r['accepted_window_losses']} |")
    lines += ['', '主时期15个窗口中有10个落后，说明这套确认机制仍不能保证下一阶段有效。胜负按收益差与万亿分之一的数值容差比较，其中包含非常微小的差额；胜出次数不等于具有经济意义的胜出次数。表格四舍五入为零时，以原始明细为准。末端被评价期截断的窗口单独标记，不混入上表；完整原表也保留未通过确认时采用常数规则的所有窗口。', '',
        '## 五、全部20次已确认分界的后续表现', '',
        '下表为压力费用下主规则减无因子学习的实际窗口收益差，未作年化。它来自两条原账户路径，因此包含窗口开始时继承的份数差异。', '',
        '| 模型更新日 | 后续交易观察起日 | 观察止日 | 交易日数 | 窗口收益差 |', '|---|---|---|---:|---:|']
    accepted = windows[(windows.cost == 'STRESS') & (windows.left == 'CONFIRMED_STUMP') & (windows.right == 'NO_FACTOR_LEARNER') & windows.accepted_feature]
    for r in accepted.itertuples():
        lines.append(f"| {str(r.fit_origin)[:10]} | {str(r.window_start)[:10]} | {str(r.window_end)[:10]} | {r.days} | {r.window_return_gap:+.2%} |")
    accepted.rename(columns={'period': '时期', 'cost': '费用', 'left': '主规则', 'right': '比较对象',
        'fit_origin': '更新日', 'window_start': '后续开始', 'window_end': '后续结束', 'days': '交易日数',
        'window_return_gap': '后续收益差', 'end_censored': '末端截断'}).to_csv(DELIVERY / '通过确认的20次规则后续表现.csv', index=False, encoding='utf-8-sig')
    lines += ['', '## 六、顺带确认一个旧因子的口径问题', '',
        '重新查看旧的日频ETF申赎研究时，发现缓存日期来自交易变动日期，而因子程序把它称为真实披露日并允许同日使用。缓存有1216行日频份额，但没有逐笔历史发布时间。官方文档也将该字段定义为交易变动日期。详见：' + link(FACTOR_REPORT, '旧日频申赎因子的日期口径说明') + '。', '',
        '这属于旧申赎研究的历史可用性证据缺口；不证明全部记录必然前视，也不证明改一个延迟就能提高收益。原失败保留，未重新计算。', '',
        '本次也查到了既有月初月末日历交易、日频申赎融资、ETF历史现金分配与资金利率比较的研究记录。这些方向已有对应方法，不能换个名称便算新的发现。此处是有限覆盖检查，不是全项目方法完整性审计。', '',
        '## 七、下一步及目标状态', '',
        '本轮完成了失效阶段和费用的解释，没有足够证据继续深化这套直接学习规则。继续允许历史挖掘，但只有出现尚未等价测试、含义清楚、当时可知的新信息或新经济机制，才登记下一套有限方法。当前尚未确定新的合格候选。对已知失败，不改窗口、倒转方向或挑盈利月份重新包装。', '',
        '下一步先核对现成资料中是否已有旧日频份额的原始披露时点证据，限定在已有源文件和直接关联回执；如果没有，就结束这一输入的局部补充，另寻可用机制。不会启动长期补数或凭空假定一天延迟后重跑收益。', '',
        '夏普1.2、年化10%的原目标保持，尚未达到；独立未来验证样本仍为零。原固定每日观察继续按既定规则等待下一完整交易日。', '',
        '## 八、完整结果入口', '',
        f"本次核对12份已保存账户的{result['verified_account_rows']:,}行日账和{result['verified_decision_rows']:,}条决策；两个针对性测试通过，保存表的日度、窗口与全期加总再次核对通过。未重复运行上一轮账户。", '',
        '- ' + link(ROOT / 'docs/510300_DIRECT_STUMP_FAILURE_EXPLANATION_V1.md', '计算前写明的分析范围与口径'),
        '- ' + link(OUT / 'all_training_confirmation_transitions.csv', '142次训练至确认变化'),
        '- ' + link(OUT / 'subsequent_active_rule_windows.csv', '全部随后观察窗口，包括拒绝与截尾窗口'),
        '- ' + link(OUT / 'paired_annual_arithmetic_components.csv', '两时期两费用全部差额拆分'),
        '- ' + link(OUT / 'daily_paired_components.csv', '逐日差额及全部贡献项'),
        '- ' + link(OUT / 'saved_verification.json', '保存表核对结果'), '',
        f"研究编号：{STUDY}。完成时间：{result['completed_at']}。", '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    record = {'study_id': STUDY, 'status': result['status'], 'completed_at': result['completed_at'],
        'result': str((OUT / 'result.json').relative_to(ROOT)).replace('\\', '/'),
        'report': str(REPORT.relative_to(ROOT)).replace('\\', '/'), 'report_sha256': digest(REPORT),
        'result_sha256': digest(OUT / 'result.json'), 'training_feature_winners': 68, 'confirmation_rejections': 48,
        'subsequent_main_accepted_wins': 5, 'subsequent_main_accepted_windows': 15,
        'main_stress_annual_arithmetic_total_gap': total['net_return'],
        'main_stress_no_factor_gap': blank['net_return'], 'main_stress_feature_gap': signal['net_return'],
        'main_stress_friction_gap': total['friction_part'], 'new_accounts': 0, 'new_model_fits': 0,
        'new_candidates': 0, 'goal_achieved': False, 'independent_validation': False, 'position_impact': 0}
    index = read(INDEX)
    preserved = {key: index[key] for key in ['registered_configurations_in_this_resumption', 'evaluation_accounts_in_this_resumption', 'finite_historical_mining_totals']}
    index.update(updated_at=now(), status='SAVED_DIRECT_LEARNING_FAILURE_EXPLAINED_NO_NEW_ELIGIBLE_CANDIDATE',
        latest_direct_stump_failure_explanation=record, latest_continuation_note=record['report'],
        last_goal_turn_classification='PROGRESS_SAVED_PHASE_FAILURE_ATTRIBUTION_AND_OLD_FACTOR_DATE_SEMANTICS_CHECK',
        consecutive_external_data_blocked_goal_turns=0, goal_achieved=False)
    index['running_studies'] = [r for r in index['running_studies'] if r.get('study', r.get('study_id')) != STUDY]
    index['latest_fund_share_date_semantics_check'] = {
        'status': share_note['status'], 'source_rows': share_note['source_rows'],
        'record': str((OUT / 'supplemental_fund_share_date_semantics.json').relative_to(ROOT)).replace('\\', '/'),
        'report': str(FACTOR_REPORT.relative_to(ROOT)).replace('\\', '/'),
        'old_rejection_preserved': True, 'historical_same_day_availability_not_established': True}
    index['next_work'].update(status='BOUNDED_EXISTING_FUND_SHARE_PUBLICATION_EVIDENCE_CHECK_AND_FIXED_DAILY_OBSERVATION',
        source=record['report'], registered=False, planned_settings=0, planned_new_accounts=0, planned_new_model_fits=0,
        historical_mining_allowed=True, external_data_required=False, implementation_remaining=True,
        implementation_remaining_scope='失效解释已完成；仅核对已有基金份额直接来源回执有无历史披露时点，不重训，不启动长期补数',
        next_historical_question_status='EXISTING_FUND_SHARE_PUBLICATION_RECEIPTS_BOUNDED_CHECK_PENDING',
        next_historical_question='旧日频份额直接关联的现成原始记录能否证明每期披露可用时点？若无则停止该局部补充',
        focus='避免把交易变动日当成披露日；只查直接相关已有回执，已覆盖日历、日频申赎及ETF现金分配机制不换名重做')
    assert all(index[key] == value for key, value in preserved.items())
    write_json(INDEX, index)
    write_json(OUT / 'delivery_receipt.json', {'status': 'CHINESE_EXPLANATIONS_AND_INDEX_UPDATED',
        'completed_at': now(), 'report': record['report'], 'report_sha256': digest(REPORT),
        'factor_report': str(FACTOR_REPORT.relative_to(ROOT)), 'factor_report_sha256': digest(FACTOR_REPORT),
        'new_accounts': 0, 'new_model_fits': 0, 'counters_preserved': True, 'goal_achieved': False}, exclusive=True)
    print('失效原因报告、旧因子日期口径说明及研究索引已保存。', flush=True)


if __name__ == '__main__':
    main()
