"""交付单因素直接交易学习的中文规则、全部结果与模拟买卖，维护独立计数。"""
import numpy as np
import pandas as pd

from research.single_factor_direct_stump_v1 import ROOT, OUT, INDEX, read, inputs
from research.intraday_overnight_increment_v1 import write_json, digest, now


DELIVERY = ROOT / 'deliverables/510300有限历史挖掘_单因素直接买卖规则_20260914'
REPORT = DELIVERY / '单因素直接买卖规则_完整检验结果.md'
NAMES = {'CONFIRMED_STUMP': '五日涨跌分界学习', 'NO_FACTOR_LEARNER': '无因子学习基准',
         'FIXED_WEEKLY_VOL10': '简单周度波动控制', 'BUY_HOLD': '买入持有', 'ETF_VOL10': '简单日度波动控制'}
PERIODS = {'early': '2015—2019年', 'main': '2020年至2026年9月11日'}
COSTS = {'BASE': '基础费用', 'STRESS': '压力费用'}


def link(path, title):
    return f'[{title}](<{path.as_posix()}>)'


def rule_text(rule):
    if rule['kind'] == 'CASH':
        return '保持空仓；已有仓位在下次计划交易时申请全部卖出'
    if rule['kind'] == 'PASSIVE':
        return '持续允许持仓，仓位按最近20日波动率调整'
    side = '小于' if rule['kind'] == 'LOW' else '大于或等于'
    return f"五日含分红累计收益{side}{rule['threshold']:.6%}时允许持仓，否则申请全部退出"


def export_chinese_tables(cfg, learned, result):
    data, _ = inputs(cfg)
    origins = np.array([r['fit_origin_index'] for r in learned['records']])
    monthly = []
    for r in learned['records']:
        monthly.append({'更新判断日': r['fit_origin'][:10], '训练开始': r['training_start'][:10],
                        '训练结束': r['training_end'][:10], '确认开始': r['confirmation_start'][:10],
                        '确认结束': r['confirmation_end'][:10], '训练优胜规则': rule_text(r['policies'][r['training_winner']]),
                        '实际采用规则': rule_text(r['primary_spec']), '无因子对照规则': rule_text(r['no_factor_spec']),
                        '五日因素通过确认': '是' if r['feature_policy_accepted'] else '否',
                        '训练比较账户数': r['internal_training_accounts'], '确认账户数': r['internal_confirmation_accounts'],
                        '说明': r['reason']})
    pd.DataFrame(monthly).to_csv(DELIVERY / '每月采用的具体中文规则.csv', index=False, encoding='utf-8-sig')
    trades = []
    for item in result['metrics']:
        if item['reused']:
            continue
        folder = OUT / 'evaluation' / item['period'] / item['cost'] / item['model']
        ledger = pd.read_parquet(folder / 'ledger.parquet')
        decisions = pd.read_parquet(folder / 'decisions.parquet').set_index('execution_date')
        for row in ledger[ledger.filled_quantity.ne(0)].itertuples():
            decision = decisions.loc[row.date]
            source = data.iloc[int(decision.origin_index)]
            record = learned['records'][int(np.searchsorted(origins, decision.origin_index, side='right') - 1)]
            rule = {'kind': 'PASSIVE'} if item['model'] == 'FIXED_WEEKLY_VOL10' else record['primary_spec' if item['model'] == 'CONFIRMED_STUMP' else 'no_factor_spec']
            trades.append({'时期': PERIODS[item['period']], '费用情景': COSTS[item['cost']], '策略': NAMES[item['model']],
                           '判断日': decision.origin.strftime('%Y-%m-%d'), '成交日': row.date.strftime('%Y-%m-%d'),
                           '方向': '买入' if row.filled_quantity > 0 else '全部退出' if row.shares == 0 else '减仓',
                           '请求份数': abs(int(row.requested_quantity)), '实际成交份数': abs(int(row.filled_quantity)),
                           '模拟成交价': row.fill_price, '佣金元': row.commission, '滑点成本元': row.slippage_cost,
                           '判断时五日累计收益': source.five_day_total_return,
                           '判断时二十日年化波动率': source.vol20, '判断目标仓位': decision.reference_weight,
                           '当时具体规则': rule_text(rule), '收盘剩余份数': int(row.shares), '收盘账户总值元': row.equity,
                           '执行说明': '历史模型模拟成交，非实际订单或实际成交'})
    pd.DataFrame(trades).to_csv(DELIVERY / '全部新账户每一次模拟买卖.csv', index=False, encoding='utf-8-sig')
    return len(trades)


def main():
    cfg = read(OUT / 'protocol.json')
    result = read(OUT / 'result.json')
    learned = read(OUT / 'learning_completed.json')
    verification = read(OUT / 'saved_verification.json')
    assert verification['status'] == 'PASS_SAVED_ACCOUNTS_AND_PAST_ONLY_SELECTION'
    assert not result['all_four_historical_point_goals_pass'] and not result['continuation_criteria_pass']
    DELIVERY.mkdir(parents=True, exist_ok=True)
    trade_count = export_chinese_tables(cfg, learned, result)
    counts = pd.Series([r['primary_policy'] for r in learned['records']]).value_counts()
    lines = [
        '# 510300单因素直接买卖规则：完整检验结果', '',
        '可以继续挖掘过去的数据。本轮实际完成了一个低复杂度方法：只看过去五天涨跌，直接用扣除交易费用后的账户表现学习一个买卖分界，并用随后一段历史确认。结果没有达到夏普1.2、年化10%，也没有获得继续深化这套固定规则的依据。', '',
        '**当前结论：方法更简单、时间顺序更严格，仍然不等于已经得到可靠盈利策略。** 研究目标保持；本轮失败完整保留。', '',
        '## 一、给老板的一段话', '',
        '我们尝试每月用过去约两年的数据，判断沪深300ETF最近五天上涨或下跌到什么程度时值得持有。先用较早的一年半选择买卖规则，再用随后半年确认，确认不通过就退回不依赖这个因素的基础规则。每周五收盘判断，下一交易日开盘买入、减仓或全部退出；行情波动越大，投入仓位越低。交易费和滑点都计入。最终发现，这种简单做法在2020年以来仍然亏损，说明历史里选出的漂亮分界并没有形成稳定交易优势，暂不能拿来宣称夏普1.2已经实现。', '',
        '## 二、到底使用什么因素', '',
        '唯一择时因素是最近5个交易日含现金分红的累计涨跌幅。每天的收益同时考虑当日价格变化和除息现金，再将五天收益连续复合。它不是五天涨幅的简单相加，也不是前瞻EPS或逆回购数据。', '',
        '唯一风险层是ETF本身最近20个交易日的含分红日收益波动率，使用样本标准差并按每年242个交易日年化。目标持仓比例为10%除以该波动率，最高100%。例如波动率20%时目标持仓50%；波动率10%或更低时最多100%。现金和无风险收益率均假定为零。', '',
        '这个方法检验短期涨跌究竟体现延续还是反转。方向由过去训练期选择，不能预先声称某一方向必然有效。', '',
        '## 三、如何学习，而且不偷看随后数据', '',
        '首次更新日为2014年12月31日，其后每月第一个交易日收盘更新。每次只用截至当时收盘的过去484个完整交易日：较早363日训练，后面121日确认。两段各自从20万元现金开始，不把训练段持仓带入确认段。期末按收盘价计账，不强制卖出。', '',
        '训练段只在初始化判断日和可在训练段内执行的周五判断日提取五日涨跌幅，至少52个有效起点。对这些起点计算25%、50%、75%分位数，得到三个分界。确认段不参加分界计算。', '',
        '每个分界对应两个方向：低于分界时允许持仓，或大于等于分界时允许持仓；其余时候空仓，共六条分界规则。另外比较始终空仓、始终允许持仓，共八条训练规则。训练评分是账户日收益的年化算术平均值，减去年化日收益样本方差的五倍。训练统一使用压力费用。', '',
        '训练评分最高者胜出；差距不超过万亿分之一时，先选成交次数少者，再按事前固定顺序选择。主规则只允许这一位训练第一名进入确认，不允许读完确认表现后改选训练第二名。完整固定顺序和数值约定见附后的原始中文方案。', '',
        '无因子对照只在“空仓”和“允许持仓”之间按训练评分选择；若训练选空仓，只有随后121日空仓评分确实高于持续持仓，才继续用空仓，否则允许持仓。主规则只有训练第一名属于六条分界规则之一、且其确认评分严格胜过这个无因子对照，才采用分界；否则退回无因子对照。确认后不重新拟合分界。', '',
        '每次最多跑三个确认账户，重复规则只算一次。月度更新完成后，参数保持至下一次更新；非周五更新不会立即交易。这里的“确认”仍然是历史开发程序的一部分，不是没见过数据上的独立验证。', '',
        '## 四、完整进场、减仓和退出规则', '',
        '1. 每周五收盘读取最近一次月度规则。周五休市，该周不增加替代判断日；两个独立评价账户开设时，额外做一次初始化判断。',
        '2. 空仓且规则允许持仓时，根据上述风险预算计算目标份数，按100份向下取整，在下一交易日开盘申请买入。',
        '3. 已持仓且规则仍允许持仓时，目标比例与实际收盘仓位相差不足10个百分点就保持份数，否则下一开盘申请增减仓。',
        '4. 规则不允许持仓时，下一开盘申请全部退出，退出不受10个百分点带宽限制。目标取整后不足100份也全部退出。',
        '5. 因素或风险资料不足时，不产生新交易判断，已有份额保留。周内不另加止盈、止损、追单或最长持有期；未成交申请到下一周五重新判断。', '',
        '涨停方向限制、跌停方向限制、当日买入份额不能当日卖出、现金是否足够、最低佣金、价格档0.001元均沿用完整账户模型。现金分红的权益登记、除息应收、现金到账分别计入。因只有日线资料，成交属于约定的历史执行模拟，并未验证真实盘口中的成交概率。', '',
        '## 五、候选全部四种情景', '',
        '两段分别从20万元开始计算，不把它们拼成一个账户。基础费用为佣金万2、单边滑点0.05%；压力费用为佣金万4、单边滑点0.10%；最低佣金均为5元。夏普用全部账户每日净收益计算，包含空仓日；年化收益为净值的复合年化收益。', '',
        '| 时期 | 费用 | 净复合年化 | 净夏普 | 最大回撤 | 成交次数 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for r in result['metrics']:
        if r['model'] == 'CONFIRMED_STUMP':
            lines.append(f"| {PERIODS[r['period']]} | {COSTS[r['cost']]} | {r['annual_return']:.2%} | {r['sharpe']:.3f} | {r['max_drawdown']:.2%} | {r['fills']} |")
    lines += ['', '**四种情景均未达到夏普1.2、年化10%。** 最新一期规则是2026年9月1日更新的“持续允许持仓，按最近20日波动率调整”，当次五日收益分界没有通过确认；这只是研究模型状态。', '',
              '## 六、所有对照结果', '',
              '无因子学习基准与主规则采用完全相同的训练、确认、周五执行及风险预算。简单周度波动控制持续允许持仓，不学习空仓。买入持有和日度波动控制复用已完成且日期、资本、成本一致的对照日账。', '',
              '| 时期 | 费用 | 对照 | 净复合年化 | 净夏普 | 最大回撤 |', '|---|---|---|---:|---:|---:|']
    for r in result['metrics']:
        if r['model'] != 'CONFIRMED_STUMP':
            lines.append(f"| {PERIODS[r['period']]} | {COSTS[r['cost']]} | {NAMES[r['model']]} | {r['annual_return']:.2%} | {r['sharpe']:.3f} | {r['max_drawdown']:.2%} |")
    lines += ['', '候选在早期略好于无因子学习，但在2020年以来更差；两个时期都没有胜过简单周度波动控制的年化收益和夏普。主时期压力下，候选年化−1.98%、夏普−0.191；简单周度波动控制年化2.13%、夏普0.238。后者也未达标。', '',
              '## 七、跨年是否重复有效', '',
              '下表直接列每个自然年实际净收益。2026年只到9月11日，不年化成全年收益；年度夏普和完整指标另有原始表。', '',
              '| 年份 | 候选净收益 | 无因子学习净收益 | 周度波动控制净收益 |', '|---|---:|---:|---:|']
    yearly = pd.read_csv(OUT / 'yearly_metrics.csv')
    for year in range(2015, 2027):
        returns = []
        for model in ['CONFIRMED_STUMP', 'NO_FACTOR_LEARNER', 'FIXED_WEEKLY_VOL10']:
            row = yearly[(yearly.year == year) & (yearly.cost == 'STRESS') & (yearly.model == model)].iloc[0]
            initial = row.end_equity - row.profit
            returns.append(row.profit / initial)
        lines.append(f"| {year}{'年截至9月11日' if year == 2026 else '年'} | {returns[0]:+.2%} | {returns[1]:+.2%} | {returns[2]:+.2%} |")
    lines += ['', '2015—2025年的11个完整年度里，候选严格胜过无因子学习6年，胜过简单周度波动控制4年；事前要求分别至少8年，未通过。相等年份不算胜出。微小收益差也会被计为严格胜出，因此这些次数并不表示经济差异显著。', '',
              '## 八、内部究竟尝试了多少次', '',
              f"本轮固定一个完整主学习方法，完成142次月度更新，每次产生一个主规则和一个无因子规则，共284个规则学习输出。内部实际计算1136份训练账户、286份确认账户，合计1422份；另计算12份正式评价账户，复用8份参考账户。不同窗口大量重叠，它们不是1422个独立样本。", '',
              f"142次更新中，五日涨跌分界通过确认20次，占{20 / 142:.2%}；其余{int(counts.get('CASH', 0))}次采用空仓、{int(counts.get('PASSIVE', 0))}次采用持续允许持仓。通过历史确认的20次仍没有产生全期可靠收益，这提示单次确认也会失效。它不支持事后放宽确认条件。", '',
              '训练的五日窗口、363／121日分段、三个分位点、风险评分、月度更新和周五执行等，都是提前确定但仍由研究者选择的设计。必须把这些选择和旧研究累计尝试一起理解，不能因只有一个因素就称无过拟合。', '',
              '## 九、必要验证和研究边界', '',
              f"五项针对性测试通过；从保存文件重新核对{verification['internal_ledger_rows']:,}行内部训练及确认日账、{verification['external_ledger_rows']:,}行正式及参考日账、{verification['new_account_decisions']:,}条新账户决策。核对现金、份数、分红、费用、净值和收益关系，训练与确认日期、实际分界、被选规则、周五判断及次日执行顺序。该复核没有重训或产生新账户。", '',
              '正式历史学习及账户运行约154秒。测试阶段修正过一个数据数组只读问题，发生在方法固定和读取本轮实际表现之前；原测试失败及修正记录已保存。', '',
              '全部历史此前已被项目使用，滚动训练只是防止每次计算偷看其后行情，无法抹掉研究者曾看过整段历史这一事实。当前没有独立未来收益验证。本轮也未声称计算了整个项目的过拟合概率或经过全部多重试验校正的夏普。', '',
              '反复尝试并挑选最高回测表现会增加选择偏差，这是回测过拟合研究的核心问题。本轮因此保留全部比较次数和失败，而不能用“找到了最高分”代替可靠性判断。参考：[The Probability of Backtest Overfitting，作者论文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。', '',
              '## 十、后续怎样继续历史挖掘', '',
              '继续允许有限历史研究，但本轮方法按原结果结束，不调整分位点、训练长度、确认期、交易日、风险预算来救回成绩。下一步可以从已经保存的训练、确认和随后账户记录定位规则失效发生在哪一阶段，区分五日分界的增量与空仓选择本身的损益；先做已保存结果的解释，不据此挑出表现较好的月份拼接策略。', '',
              '旧数据仍可用于排除错误、厘清原因和形成新的经济假设；任何新完整方法都需要在计算前限定试验数量、写清买入退出、保留失败，并最终等待独立未来验证。目标仍是夏普1.2、年化10%，目前尚未达到。', '',
              '## 十一、中文规则与每次买卖下载', '',
              '- ' + link(ROOT / cfg['rules'], '计算前固定的完整中文方案'),
              '- ' + link(DELIVERY / '每月采用的具体中文规则.csv', '142次更新实际采用的中文规则和数值分界'),
              '- ' + link(DELIVERY / '全部新账户每一次模拟买卖.csv', f'12份新账户全部{trade_count}次模拟买卖明细'),
              '- ' + link(OUT / 'evaluation_metrics.csv', '全部20份账户指标'),
              '- ' + link(OUT / 'yearly_metrics.csv', '全部账户逐年指标'),
              '- ' + link(OUT / 'evaluation', '正式账户每日净值、每次决策及成交原始文件'),
              '- ' + link(OUT / 'monthly_learning', '每月训练八规则评分、确认评分和完整日账'),
              '- ' + link(OUT / 'completed_cycles.csv', '已完成持仓周期'),
              '- ' + link(OUT / 'cycle_concentration.csv', '盈利集中度及未结束持仓周期损益'),
              '- ' + link(OUT / 'saved_verification.json', '保存结果核对记录'), '',
              '模拟买卖表中的不同费用情景和时期是分别计算的研究账户，不能把它们当成一个账户的连续订单。表内包含买入、减仓和全部退出，每笔给出判断日、成交日、实际规则、五日收益、风险值、份数、价格、费用及剩余份额。', '',
              f"方案固定时间：{cfg['registered_at']}。结果完成时间：{result['completed_at']}。研究编号：{cfg['study_id']}。", '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    record = {'study_id': result['study_id'], 'status': result['status'], 'completed_at': result['completed_at'],
              'rules': cfg['rules'], 'result': str((OUT / 'result.json').relative_to(ROOT)).replace('\\', '/'),
              'report': str(REPORT.relative_to(ROOT)).replace('\\', '/'), 'report_sha256': digest(REPORT),
              'result_sha256': digest(OUT / 'result.json'), 'new_candidates': 1, 'new_accounts': 12,
              'reused_accounts': 8, 'new_model_fits': 284, 'model_fit_count_scope': result['model_fit_count_scope'],
              'internal_training_accounts': 1136, 'internal_confirmation_accounts': 286,
              'accepted_feature_updates': 20, 'all_four_historical_point_targets_pass': False,
              'continuation_criteria_pass': False, 'selected_trading_model': None,
              'independent_validation': False, 'strict_forward_evidence_days': 0, 'goal_achieved': False, 'position_impact': 0}
    index = read(INDEX)
    count_keys = ['registered_configurations_in_this_resumption', 'evaluation_accounts_in_this_resumption']
    old_counts = {key: index[key] for key in count_keys}
    index.update(updated_at=now(), status='FINITE_DIRECT_STUMP_COMPLETED_NO_CONSISTENT_ACCOUNT_INCREMENT',
                 goal_achieved=False, latest_finite_anti_overfitting_historical_study=record,
                 latest_single_factor_direct_stump=record, latest_continuation_note=record['report'],
                 last_goal_turn_classification='PROGRESS_SINGLE_FACTOR_FULL_ACCOUNT_LEARNING_AND_VERIFICATION_COMPLETED',
                 consecutive_external_data_blocked_goal_turns=0)
    index['running_studies'] = [r for r in index['running_studies'] if r.get('study', r.get('study_id')) != result['study_id']]
    completed = index.setdefault('completed_historical_mining_studies', [])
    assert not any(r.get('study_id') == result['study_id'] for r in completed), '本轮已交付，禁止重复累计'
    completed.append(record)
    index['finite_historical_mining_totals'] = {
        'scope': '仅本列表；new_accounts是新增正式评价账户，内部训练确认账户另计；拟合数包含不同算法拟合或规则学习输出，复用不是唯一账户数',
        'completed_studies': len(completed), 'new_candidate_methods': sum(r['new_candidates'] for r in completed),
        'new_accounts': sum(r['new_accounts'] for r in completed), 'new_model_fits': sum(r['new_model_fits'] for r in completed),
        'reused_account_references': sum(r['reused_accounts'] for r in completed),
        'internal_training_accounts': sum(r.get('internal_training_accounts', 0) for r in completed),
        'internal_confirmation_accounts': sum(r.get('internal_confirmation_accounts', 0) for r in completed)}
    index['next_work'].update(status='SAVED_DIRECT_LEARNING_FAILURE_EXPLANATION_AND_FIXED_DAILY_OBSERVATION',
        source=record['report'], registered=False, planned_settings=0, planned_new_accounts=0, planned_new_model_fits=0,
        historical_mining_allowed=True, external_data_required=False, implementation_remaining=True,
        implementation_remaining_scope='本轮完整方法已完成；下一步仅解释保存的训练确认及随后账户失效阶段，尚未登记新策略',
        next_historical_question_status='SAVED_DIRECT_STUMP_SELECTION_FAILURE_EXPLANATION_PENDING',
        next_historical_question='训练胜出到确认、再到随后交易，五日分界增量与空仓选择分别在哪一阶段失效？',
        focus='从已保存日账解释跨阶段失效，不重训、不选择赢家月份拼接，不修改本轮固定规则')
    index['historical_mining_authorization_20260914'].update(latest_completed_study=result['study_id'], latest_result_report=record['report'])
    assert all(index[key] == value for key, value in old_counts.items())
    write_json(INDEX, index)
    write_json(OUT / 'delivery_receipt.json', {'status': 'CHINESE_REPORT_TRADE_TABLES_AND_INDEX_UPDATED',
               'completed_at': now(), 'report': record['report'], 'report_sha256': record['report_sha256'],
               'chinese_trade_rows': trade_count, 'monthly_rule_rows': 142,
               'new_accounts_generated_by_delivery': 0, 'new_model_fits_generated_by_delivery': 0,
               'old_counters_preserved': old_counts, 'goal_achieved': False}, exclusive=True)
    print('中文报告、每月规则、全部模拟买卖及研究索引已保存：' + str(REPORT), flush=True)


if __name__ == '__main__':
    main()
