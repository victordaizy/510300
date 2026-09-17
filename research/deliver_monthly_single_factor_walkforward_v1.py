"""从已完成结果生成中文说明，并将有限历史研究的完成状态接入索引。"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_monthly_single_factor_walkforward_v1'
DELIVERY = ROOT / 'deliverables/510300有限历史挖掘_单因子月度检验_20260914'
REPORT = DELIVERY / '历史仍可挖掘_三种单因子实际检验结果.md'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
STUDY = '510300_MONTHLY_SINGLE_FACTOR_WALKFORWARD_V1'
NAMES = {
    'MONTHLY_MOM20': '最近20日涨跌',
    'MONTHLY_Z20': '价格偏离20日均值',
    'MONTHLY_LOGVOL20': '最近20日波动',
    'MONTHLY_MEAN': '过去月收益平均值基准',
    'MONTHLY_VOL10': '月度波动控制基准',
    'BUY_HOLD': '买入持有基准',
    'ETF_VOL10': '日度波动控制基准',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def link(path, label):
    return f'[{label}](<{path.resolve().as_posix()}>)'


def write_json(path, value):
    temporary = path.with_name(path.name + '.monthly_delivery.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    result = read(OUT / 'result.json')
    verification = read(OUT / 'saved_verification_receipt.json')
    tests = read(OUT / 'tests_receipt.json')
    outcomes = read(OUT / 'candidate_outcomes.json')['candidates']
    protocol = read(OUT / 'protocol.json')
    assert result['new_accounts'] == 20 and result['new_model_fits'] == 423
    assert verification['status'] == 'PASS_SAVED_TIMING_AND_FULL_ACCOUNT_RECOMPUTATION'
    assert tests['passed'] == 5 and not result['goal_achieved']
    with (OUT / 'prediction_quality.csv').open(encoding='utf-8-sig', newline='') as handle:
        quality = {(r['model'], r['period']): r for r in csv.DictReader(handle)}
    rows = {(r['period'], r['cost'], r['model']): r for r in result['metrics']}
    text = [
        '# 历史仍可挖掘：三种单因子的实际检验结果',
        '',
        '研究完成日期：2026年9月14日。行情截止：2026年9月11日。',
        '',
        '**可以继续挖掘历史数据，同时减少过拟合。此前把后续工作收窄为只能等待新交易日，限制过头了。本轮已经实际完成三种简单方法的历史检验；三种均未通过预先规定的条件，也没有实现夏普1.2、年化10%的目标。**',
        '',
        '这次新增20份完整模拟账户，包括12份候选账户和8份同频基准账户；另复用8份已有基准。实际完成423次单因子拟合。单次拟合只看当时已经结束、收益已经可知的月份，三种方法全部报告，没有挑最好的一种拼入旧策略。',
        '',
        '## 一、历史数据怎样继续用',
        '',
        '先提出一个有明确含义的问题，写好因子、训练方式、买入规则、卖出规则和淘汰条件，再计算结果。每轮只比较少量完整方法；没有稳定帮助的方法就结束这一固定用法，不继续改窗口、门槛和配比来抬高旧回测。',
        '',
        '本轮的问题是：只凭一个简单行情因素学习下个月收益，能否比“不看因子、只用过去平均收益”更有用？一个因素若连这个基准都不能稳定改善，就没有理由马上叠加更多规则。',
        '',
        '历史数据已经被此前研究看过。按时间顺序学习可以避免本次训练直接偷看未来，但不能把已看过的历史重新变成独立样本。多次试验后选择好成绩仍有选择偏差，参见[Bailey与López de Prado的原论文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。本轮没有计算正式的过拟合概率，也不声称已经消除过拟合。',
        '',
        '## 二、三个因子分别是什么',
        '',
        '| 方法 | 具体计算含义 | 它尝试回答的问题 |',
        '|---|---|---|',
        '| 最近20日涨跌 | 将最近20个交易日、包含分红的每日收益转成对数后相加，表示这段时间的累计涨跌 | 最近涨跌是否与下个月收益有关？ |',
        '| 价格偏离20日均值 | 先构造包含分红的累计财富值；当天财富值减去最近20日平均值，再除以这20日的样本标准差 | 当前相对最近均值的位置，是否与下个月收益有关？ |',
        '| 最近20日波动 | 最近20日含分红日收益的样本标准差乘以242的平方根，得到年化波动率，再取自然对数作为输入 | 最近波动大小是否与下个月收益有关？ |',
        '',
        '每种方法单独运行，只有一个预测因子。三个因素不投票、不加权组合。因子与未来收益之间的正负关系，由事先确定的同一个学习方法从过去样本中估计；不在看结果后人为反转方向。除预测因子外，三种方法均使用20日波动率来限制仓位，这是共同的风险控制规则。',
        '',
        '## 三、训练、进入和退出的完整规则',
        '',
        '1. 每个月最后一个交易日收盘后判断一次。只使用当天及之前的完整价格和分红数据。',
        '2. 一条历史学习样本对应一个月末因子，以及从下个月首个交易日开盘买入、到再下个月首个交易日开盘卖出的收益。收益包括持有期间获得权利的分红。登记日必须在买入日至卖出日前一日之间；退出日和相关除息日都已发生后，该样本才能进入训练。尚未结束的月份不能参与。',
        '3. 取最近60个已经成熟的有效月度样本，至少24个才训练。相邻月份的持有区间不重叠，可以共用一个开盘边界。',
        '4. 只拟合一个系数和一个截距。先用训练样本的均值和样本标准差标准化，将标准化后的数值限制在负3至正3；用固定的岭回归学习关系，惩罚强度为1。这个惩罚会抑制过大的系数。没有搜索哪个训练窗口或惩罚强度回测更好。',
        '5. 若预测下个月的未扣费收益严格大于0.4%，允许持仓；目标仓位等于10%除以ETF的20日年化波动率，最高100%。例如波动率为20%，目标仓位为50%；波动率为10%或更低，最高满仓。0.4%是统一的事前缓冲，不保证精确覆盖所有实际费用。',
        '6. 原本空仓而产生正目标时，下一交易日开盘申请买入；已经持仓且仍有正目标时，按目标调整。正目标与实际收盘仓位相差不足10个百分点，保持原有份额。',
        '7. 预测收益小于或等于0.4%时，目标变为零，下一交易日开盘申请全部卖出。这个退出不受10个百分点调仓门槛限制。',
        '8. 月内保持份额，没有额外止盈、止损或最长持有期。连续数月满足条件可以一直持有。资料不足时不产生新判断：空仓保持空仓，已有份额继续保留。',
        '9. 因涨跌停或资金限制未完全成交的申请，不在月内额外追单，下个月末重新判断。全部失败申请和部分成交记录保留。',
        '',
        '20日窗口、60个月训练、24个月最低样本、截断3、惩罚1、0.4%门槛、10%风险目标、10个百分点调仓带宽，都是本轮计算前固定的选择。这些选择仍有研究者自由度，不能因为模型简单就当作没有过拟合。',
        '',
        '## 四、完整账户的计算口径',
        '',
        '每个账户以20万元现金开始，只使用510300和现金，不借贷。2015年1月5日至2019年12月31日、2020年1月2日至2026年9月11日分开计算，各自从20万元开始；两个账户不拼接。期末按收盘估值，不强制末日卖出。',
        '',
        '| 项目 | 基础成本 | 压力成本 |',
        '|---|---:|---:|',
        '| 每次成交佣金 | 成交金额万分之二，最低5元 | 成交金额万分之四，最低5元 |',
        '| 单边滑点 | 0.05% | 0.10% |',
        '',
        '模拟执行考虑整百份、0.001元价格档、当日买入不能当日卖出，以及原成交引擎的方向性涨跌停限制。分红登记、除息、到账分别记账。现金收益和无风险收益均假定为零；年化使用242个交易日。表中的年化为净值复合增长率，夏普按完整账户扣费后的日收益计算。最大回撤表示期间从高点跌到低点的最大幅度。',
        '',
        '## 五、三种候选的全部结果',
        '',
        '| 时期 | 成本 | 方法 | 净年化 | 净夏普 | 最大回撤 | 成交次数 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    candidates = list(NAMES)[:3]
    periods = {'early': '2015—2019', 'main': '2020—2026年9月11日'}
    costs = {'BASE': '基础', 'STRESS': '压力'}
    for period in periods:
        for cost in costs:
            for model in candidates:
                r = rows[period, cost, model]
                text.append(f"| {periods[period]} | {costs[cost]} | {NAMES[model]} | {r['annual_return']:.2%} | {r['sharpe']:.3f} | {r['max_drawdown']:.2%} | {r['fills']} |")
    text += [
        '',
        '三个方法在较早时期都有正收益，但在2020年以来的主时期全部亏损；两种成本下均未达到目标。这不能证明这些因子的任何用法都没有价值，但足以淘汰本轮事先固定的“单因子预测月收益并按门槛进出场”用法。',
        '',
        '## 六、对照基准的全部结果',
        '',
        '过去月收益平均值基准与候选使用相同成熟训练月份、相同0.4%门槛和相同仓位规则，只是不使用预测因子。月度波动控制基准完全不判断下个月涨跌，只在月末按相同风险目标和调仓带宽配置仓位。买入持有及日度波动控制复用已经完成、同样按期末收盘估值的历史账户。',
        '',
        '| 时期 | 成本 | 基准 | 净年化 | 净夏普 | 最大回撤 |',
        '|---|---|---|---:|---:|---:|',
    ]
    for period in periods:
        for cost in costs:
            for model in list(NAMES)[3:]:
                r = rows[period, cost, model]
                text.append(f"| {periods[period]} | {costs[cost]} | {NAMES[model]} | {r['annual_return']:.2%} | {r['sharpe']:.3f} | {r['max_drawdown']:.2%} |")
    text += [
        '',
        '较早时期，均值预测基准与月度波动控制基准相同，是因为当时的均值预测都满足入场门槛。2020年以来，压力情景的月度波动控制基准年化5.34%、夏普0.471；均值预测基准年化负3.98%、夏普负0.490。这个同规则对照表明，本轮收益判断与进出场门槛并未产生稳定帮助；不能把它泛化为所有择时都无效。月度波动控制基准本身也没有达到目标。',
        '',
        '## 七、有没有稳定的预测增量',
        '',
        '计算前规定：候选必须在两个时期、两种成本的全部四种情景中，净年化和净夏普均高于均值预测基准；同时在2015—2025年的11个完整年度中，至少8年的压力净收益高于该基准；两段时期的预测平方误差也都应低于该基准。2026年尚未结束，单列展示，不计入完整年度数量。',
        '',
        '下表的预测误差改善为相对均值预测基准的比例：正数代表误差减少、预测较好，负数代表误差增加、预测较差。较早时期有59个已成熟、完整落在评价范围的预测月份，主时期有80个。',
        '',
        '| 方法 | 早期预测误差改善 | 主时期预测误差改善 | 压力年度胜出数 | 四场景收益及夏普全部改善 | 获得继续加参数的依据 |',
        '|---|---:|---:|---:|---|---|',
    ]
    for outcome in outcomes:
        model = outcome['model']
        early = float(quality[model, 'early']['out_of_fit_r_squared'])
        main_period = float(quality[model, 'main']['out_of_fit_r_squared'])
        text.append(f"| {NAMES[model]} | {early:+.2%} | {main_period:+.2%} | {outcome['pressure_full_year_wins_over_mean']}/11 | 否 | 否 |")
    text += [
        '',
        '**三种方法均未通过。没有入选方法，也没有通过改变0.4%门槛、20日窗口、60个月训练长度或模型惩罚来挽救本轮结果。**',
        '',
        '## 八、做过哪些必要验证',
        '',
        f"五项有针对性的测试全部通过：改变未来数据不影响此前结果、分红权利与标签成熟时间、训练样本成熟且月份不重叠、拟合公式与独立求解一致、入场门槛与仓位边界。另从保存文件复核{verification['accounts']}份账户、{verification['ledger_rows']:,}行日账和{verification['new_account_decision_rows']:,}行新账户决策，核对现金、份额、分红应收、净值、费用及信号和执行日期。",
        '',
        '这些是程序、时间顺序和账户计算的验证，不能证明策略未来会赚钱。新增独立观察收益样本为零。本轮模拟没有形成实盘买卖。',
        '',
        '## 九、后续历史挖掘怎样推进',
        '',
        '下一项优先问题拟定为：已有历史中的单一风险信息，能否比只用过去平均风险，更稳定地预判下个月的下行风险？先登记风险标签、可用时间、一个固定预测方法和淘汰条件，再检验预测是否有增量；只有通过才另行写明完整进出场规则并计算账户。这个问题尚未运行，不能把它写成已经有效的新策略。',
        '',
        '这样的安排仍然使用过去数据推进研究，重点从凑高历史夏普转为寻找可重复的帮助。本轮失败结果保留，下一问题单独计入研究次数。原策略的高过拟合风险结论和固定日常观察继续保留。最终目标仍是完整账户成本后的夏普1.2与年化10%，目前尚未实现具有独立证据的达标。',
        '',
        '## 十、完整规则及结果文件',
        '',
        '- ' + link(ROOT / 'docs/510300_MONTHLY_SINGLE_FACTOR_WALKFORWARD_V1.md', '计算前写明的中文研究规则'),
        '- ' + link(OUT / 'account_metrics.csv', '28份账户的完整指标'),
        '- ' + link(OUT / 'yearly_metrics.csv', '全部逐年结果，含单列的2026部分年度'),
        '- ' + link(OUT / 'monthly_forecasts.csv', '全部按月预测和训练参数'),
        '- ' + link(OUT / 'training_members.csv', '各次训练实际使用的成熟月份'),
        '- ' + link(OUT / 'completed_cycles.csv', '完整持仓周期'),
        '- ' + link(OUT / 'cycle_concentration.csv', '持仓周期盈亏集中度'),
        '- ' + link(OUT / 'accounts', '20份新模拟账户的日账、决策及逐笔成交目录'),
        '- ' + link(OUT / 'saved_verification_receipt.json', '保存账户和时序的复核结果'),
        '',
        f"研究编号：{STUDY}。方案登记时间：{protocol['registered_at']}。结果完成时间：{result['completed_at']}。",
        '',
    ]
    DELIVERY.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(text), encoding='utf-8')
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    record = {
        'study_id': STUDY,
        'status': 'COMPLETED_ALL_THREE_FIXED_METHODS_REJECTED',
        'completed_at': result['completed_at'],
        'configuration': 'config/510300_monthly_single_factor_walkforward_v1.json',
        'rules': 'docs/510300_MONTHLY_SINGLE_FACTOR_WALKFORWARD_V1.md',
        'result': str((OUT / 'result.json').relative_to(ROOT)).replace('\\', '/'),
        'report': str(REPORT.relative_to(ROOT)).replace('\\', '/'),
        'verification': str((OUT / 'saved_verification_receipt.json').relative_to(ROOT)).replace('\\', '/'),
        'result_sha256': hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        'new_candidates': 3,
        'new_accounts': 20,
        'reused_accounts': 8,
        'new_model_fits': 423,
        'selected_model': None,
        'eligible_candidates': 0,
        'independent_validation': False,
        'strict_forward_evidence_days': 0,
        'goal_achieved': False,
        'position_impact': 0,
        'legacy_round_and_account_counters_not_redefined': True,
        'post_result_parameter_rescue': False,
    }
    index = read(INDEX)
    index['updated_at'] = stamp
    index['status'] = 'FINITE_HISTORICAL_STUDY_COMPLETED_CONTINUE_LOW_COMPLEXITY_RESEARCH'
    index['goal_achieved'] = False
    index['running_studies'] = [r for r in index.get('running_studies', []) if r.get('study', r.get('study_id')) != STUDY]
    index['latest_finite_anti_overfitting_historical_study'] = record
    completed = index.setdefault('completed_historical_mining_studies', [])
    if not any(r.get('study_id') == STUDY for r in completed):
        completed.append(record)
    index['research_primary_focus'] = '按用户授权继续有限历史挖掘；新方法先登记、少量比较、完整进出场与成本；失败固定用法不改参挽救'
    index['latest_continuation_note'] = str(REPORT.relative_to(ROOT)).replace('\\', '/')
    index['last_goal_turn_classification'] = 'PROGRESS_20_NEW_HISTORICAL_ACCOUNTS_423_FITS_COMPLETED'
    index['consecutive_external_data_blocked_goal_turns'] = 0
    if 'current_goal_blocked_audit' in index:
        index['current_goal_blocked_audit'].update({
            'current': False,
            'resolved_at': stamp,
            'resolved_by': 'USER_AUTHORIZED_FINITE_HISTORICAL_MINING_AND_REAL_STUDY_COMPLETED',
            'resolution_scope': '不再以等待新日期阻断历史研究；没有宣称新增独立日期已经到达',
        })
    next_work = index['next_work']
    next_work.update({
        'status': 'FINITE_HISTORICAL_MINING_AND_FIXED_DAILY_OBSERVATION',
        'source': str(REPORT.relative_to(ROOT)).replace('\\', '/'),
        'focus': '先登记并检验单一历史风险信息对下月下行风险的增量；通过后再登记完整账户；原每日固定观察独立继续',
        'research_class': 'FINITE_PREDECLARED_HISTORICAL_RESEARCH_WITH_SEPARATE_FIXED_DAILY_OBSERVATION',
        'historical_mining_allowed': True,
        'no_strategy_optimization': False,
        'no_existing_failed_strategy_retuning': True,
        'finite_research_requires_prior_method_and_rejection_rules': True,
        'external_data_required': False,
        'external_data_required_for_fixed_daily_continuation': True,
        'implementation_remaining': True,
        'implementation_remaining_scope': '下一历史风险信息问题的事前方案及有限检验；本轮20份账户已全部完成',
        'registered': False,
        'next_historical_question_status': 'PROPOSED_NOT_REGISTERED_NOT_RUN',
        'next_historical_question': '单一风险信息能否稳定改善下一月下行风险预测；先预测增量，后账户',
        'completed_monthly_return_methods_may_not_be_retuned_for_rescue': True,
    })
    index['historical_mining_authorization_20260914']['latest_completed_study'] = STUDY
    index['historical_mining_authorization_20260914']['latest_result_report'] = record['report']
    write_json(INDEX, index)
    saved = read(INDEX)
    assert saved['registered_configurations_in_this_resumption'] == index['registered_configurations_in_this_resumption']
    assert saved['latest_finite_anti_overfitting_historical_study']['new_accounts'] == 20
    assert not any(r.get('study', r.get('study_id')) == STUDY for r in saved['running_studies'])
    write_json(OUT / 'delivery_receipt.json', {
        'completed_at': stamp, 'status': 'CHINESE_REPORT_AND_RESEARCH_INDEX_UPDATED',
        'report': record['report'], 'report_sha256': record['report_sha256'],
        'new_accounts_generated_by_delivery': 0, 'new_model_fits_generated_by_delivery': 0,
        'old_counters_preserved': True, 'independent_validation': False, 'goal_achieved': False,
    })
    print('中文结果报告已生成，有限历史研究已标记完成，后续历史挖掘继续允许。')
    print(REPORT)


if __name__ == '__main__':
    main()
