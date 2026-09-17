"""交付已完成的月度风险预测、完整账户及保存收益归因。"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_monthly_downside_forecast_v1'
DELIVERY = ROOT / 'deliverables/510300有限历史挖掘_风险预测与账户检验_20260914'
REPORT = DELIVERY / '历史风险预测有增量_但完整账户尚未达标.md'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
NAMES = {
    'LOG_RISK_RIDGE': '单因子风险预测',
    'PAST_MONTHLY_MEAN': '过去平均下行风险',
    'RECENT_DOWNSIDE20': '最近20日下行风险',
    'BUY_HOLD': '买入持有',
    'MONTHLY_VOL10': '简单月度波动控制',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def link(path, label):
    return f'[{label}](<{path.resolve().as_posix()}>)'


def save(path, value):
    temporary = path.with_name(path.name + '.risk_delivery.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    result = read(OUT / 'result.json')
    prediction = read(OUT / 'prediction_result.json')
    verification = read(OUT / 'account_verification_receipt.json')
    attribution = read(OUT / 'saved_arithmetic_return_attribution.json')
    protocol = read(OUT / 'protocol.json')
    account_protocol = read(OUT / 'account_protocol.json')
    assert result['prediction_gate_pass'] and result['new_accounts'] == 12
    assert verification['status'] == 'PASS_SAVED_20_FULL_ACCOUNTS_AND_MONTHLY_EXECUTION'
    assert not result['goal_achieved']
    period_names = {'early': '2015—2019年', 'main': '2020年至2026年9月11日'}
    cost_names = {'BASE': '基础', 'STRESS': '压力'}
    content = [
        '# 历史风险预测有增量，但完整账户尚未达到目标',
        '',
        '完成日期：2026年9月14日。使用的行情截止2026年9月11日。',
        '',
        '**这轮获得了一个有限但明确的结果：简单模型对下个月下行风险的预测，通过了计算前写好的检验；将预测转为实际仓位后，完整账户没有达到夏普1.2、年化10%。因此保留风险信息增量的证据，同时保留交易账户未达标的结论。**',
        '',
        '本轮实际完成141次单因子拟合，检验140个成熟月份；预测门通过后才新增12份完整账户，并复用8份基准。没有借助新数据、改变风险预算或挑选参数抬高结果。',
        '',
        '## 一、给不熟悉量化研究的人解释',
        '',
        '我们用最近20天的波动大小，结合当时已经结束的历史月份，估计下个月每天可能跌得有多剧烈。预计风险大时少持有510300，预计风险小时多持有，最多满仓。结果显示，这种方式确实比两种简单办法更准确地估计了本轮定义的风险，但它没有充分判断什么时候值得买、什么时候应当卖。降低风险和提高每一份风险对应的收益，是两个需要分别检验的问题。',
        '',
        '## 二、因子、训练与风险目标',
        '',
        '唯一预测因子是最近20日含分红日收益的年化方差的自然对数。年化方差由日收益样本方差乘242得到。',
        '',
        '目标是下一个完整自然月的下行二阶矩：将每天收益中的正数改为零，负数平方，对该月全部交易日取平均，再乘242。分母包含上涨和平盘日。它描述下跌幅度形成的风险，不是该月总收益，也不是最大回撤。',
        '',
        '每个月最后一个交易日收盘，只使用当时已成熟的最近60个月度样本，至少24个月才训练。一个样本必须等目标月份完整结束才成熟；尚未结束的2026年9月不进入本轮检验。不同月份的标签不共用日收益行。',
        '',
        '训练只估计一个系数和一个截距，因子按训练样本标准化并限制在负3至正3，固定岭惩罚为1。对数风险预测还原为原始风险时，使用训练残差作固定方式的转换修正。训练窗口、惩罚、转换方式和所有数值下限均在计算前规定。',
        '',
        '两个基准分别是相同历史月份的平均下行风险，以及最近20日已经观察到的下行风险。三个方法都用同样的正风险数值下限。',
        '',
        '## 三、预测检验结果',
        '',
        '预先规定的主要评分为“预测风险的自然对数，加实际风险除以预测风险”，越小越好。下表给出基准损失减候选损失，正数代表候选更好；这些数值不是收益率，也不是改善百分比。评分方法背景参见[Patton的原论文](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)。本轮未证明该风险目标满足论文全部统计条件。',
        '',
        '| 时期 | 基准 | 成熟月份 | 候选的平均损失改善 |',
        '|---|---|---:|---:|',
    ]
    for item in prediction['period_metrics']:
        content.append(f"| {period_names[item['period']]} | {NAMES[item['baseline']]} | {item['months']} | {item['mean_loss_improvement']:+.4f} |")
    content += [
        '',
        '前后两个时期、两个基准的四项比较全部改善。2015—2025年的11个完整年度里，对过去平均风险基准有9年胜出，对最近20日风险基准有10年胜出，均达到事前至少8年的要求。2026部分年度单列，没有计入完整年度数量。',
        '',
        '| 基准 | 全部月份平均损失改善 | 连续6个月区块重采样的第5百分位 |',
        '|---|---:|---:|',
    ]
    for item in prediction['bootstrap']:
        content.append(f"| {NAMES[item['baseline']]} | {item['mean_loss_improvement']:+.4f} | {item['lower_5pct']:+.4f} |")
    content += [
        '',
        '重采样次数固定为5000，随机种子固定为20260914。两项下界均为正，因此通过事前预测门。这只是对已见历史序列的检验，没有消除此前选择研究问题的偏差。',
        '',
        '**结果也有局限：按另一项只作解释的平方误差衡量，较早时期候选不如“最近20日下行风险”基准。** 候选平方误差为0.003571，基准为0.003226。因此结论限于事先指定的主要评分及稳定性要求，不能写成任何角度都更准。',
        '',
        '## 四、完整进入、调整和退出规则',
        '',
        '1. 每月最后交易日收盘形成风险预测。目标仓位为10%除以“预测风险两倍的平方根”，最多100%。例如预测年化下行二阶矩为0.02，目标仓位为50%。系数2是固定风险预算换算约定，不保证实际收益上下对称。',
        '2. 原来空仓时，下一交易日开盘申请按正目标买入。已经持仓时，按新目标申请增减仓。',
        '3. 已有持仓且目标与实际收盘仓位相差不足10个百分点时，保持份额。若按目标计算的份额不足一手、整百份取整为零，则申请全部卖出，不受这个带宽限制。',
        '4. 有效有限风险一般产生正仓位，因此这套方法通常一直持有部分ETF；没有额外的高风险全空仓阈值，没有预测收益门槛，也没有额外止盈、止损或最长持有期。',
        '5. 月内保持份额。未知风险不产生新调仓指令；原有份额保留。未成交或部分成交的申请不在月内追加追单，下一月末重新判断。',
        '',
        '每个账户20万元起步，仅510300与现金，不借贷。2015年1月5日至2019年12月31日、2020年1月2日至2026年9月11日分别独立运行，期末按收盘估值，不强制清仓。现金收益和无风险收益均假定为零，按242个交易日年化。',
        '',
        '基础成本为每次成交佣金万分之二、最低5元，单边滑点0.05%；压力成本为佣金万分之四、最低5元，单边滑点0.10%。考虑整百份、价格档0.001、原方向性涨跌停模型、当日买入不能当日卖出，以及分红登记、除息和到账的不同时间。全部12份新账户实际没有完全未成交的申请。',
        '',
        '## 五、候选账户四种情景的实际表现',
        '',
        '| 时期 | 成本 | 净复合年化 | 净夏普 | 最大回撤 | 成交次数 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for item in result['metrics']:
        if item['model'] == 'LOG_RISK_RIDGE':
            content.append(f"| {period_names[item['period']]} | {cost_names[item['cost']]} | {item['annual_return']:.2%} | {item['sharpe']:.3f} | {item['max_drawdown']:.2%} | {item['fills']} |")
    content += [
        '',
        '**四种情景全部没有达到夏普1.2、年化10%。风险预测通过，不能代替账户达标。**',
        '',
        '## 六、全部基准账户',
        '',
        '过去平均下行风险、最近20日下行风险两个基准使用相同的风险到仓位映射及成本，因此可检验学习模型是否增加交易价值。简单月度波动控制直接使用最近20日全波动率配置10%风险预算，复用上一轮已完成账户。',
        '',
        '| 时期 | 成本 | 基准 | 净复合年化 | 净夏普 | 最大回撤 |',
        '|---|---|---|---:|---:|---:|',
    ]
    for item in result['metrics']:
        if item['model'] != 'LOG_RISK_RIDGE':
            content.append(f"| {period_names[item['period']]} | {cost_names[item['cost']]} | {NAMES[item['model']]} | {item['annual_return']:.2%} | {item['sharpe']:.3f} | {item['max_drawdown']:.2%} |")
    content += [
        '',
        '候选在四种情景中都优于“过去平均下行风险”账户，但收益和夏普都低于“最近20日下行风险”及“简单月度波动控制”账户。主时期压力情景候选年化2.25%、夏普0.274；简单月度波动控制为5.34%、0.471。后者也没有达到目标，不能因为它更好就认定任务完成。',
        '',
        '## 七、差距主要体现在哪里',
        '',
        '使用保存账户做了一次固定的收益恒等式拆分，没有重新拟合，也没有生成任何反事实账户。把日收益拆成平均前日仓位对应的市场收益、仓位偏离平均值与市场涨跌的配合、以及调仓时点和费用等剩余项。下表比较主时期压力情景的候选与简单月度波动控制基准。',
        '',
        '| 项目 | 候选减基准，年化算术平均收益差 |',
        '|---|---:|',
    ]
    selected = next(r for r in attribution['comparisons'] if r['period'] == 'main' and r['cost'] == 'STRESS' and r['comparator'] == 'MONTHLY_VOL10')
    for key, label in [('exposure_component', '平均仓位差对应的部分'),
                       ('timing_component', '仓位变化与市场涨跌配合的部分'),
                       ('execution_and_rights_residual', '调仓时点、费用及分红权利等剩余部分'),
                       ('annual_arithmetic_mean', '合计')]:
        content.append(f"| {label} | {selected['difference'][key] * 100:+.2f}个百分点 |")
    content += [
        '',
        '这个拆分针对年化算术平均日收益，不是复合年化收益，因此合计与上面的复合年化差不必相同。它是描述性的账户恒等式，不证明某一个因素造成了全部差距。',
        '',
        '候选平均前日仓位约52.91%，基准约65.41%；差距中较大一部分体现在仓位变化与市场涨跌的配合上。只因为预测风险更准，就继续增加仓位或细调门槛，没有得到本轮结果的支持。',
        '',
        '## 八、验证与边界',
        '',
        f"预测阶段5项测试、账户映射2项测试通过。保存结果核对了月度标签、成熟训练时间和主要预测损失，并核对{verification['accounts']}份账户、{verification['ledger_rows']:,}行日账的现金、份额、分红、费用、净值和月末判断次日执行关系。必要验证没有生成新模型或新账户。",
        '',
        '本轮共1种新候选方法、141次拟合、12份新账户和8份复用基准。独立未来收益样本仍为零。没有实盘买卖，没有将研究模拟持仓作为实际持仓。',
        '',
        '## 九、结论及下一步',
        '',
        '保留这个已经固定的风险预测作为信息比较基准；结束本轮风险到仓位的固定用法，不修改训练长度、惩罚、风险预算、调仓带或再添加空仓阈值来挽救成绩。旧的三个收益预测失败方法及旧策略的过拟合结论也保留。',
        '',
        '下一步优先检查既有研究中收益预测信息的证据，按实际训练记录和跨时期增量去重，寻找能解释可执行买卖时机的单一机制。先看它是否增加收益信息，再决定是否开展新账户；每个新方法仍须事前登记和全部报告。已经确认成交额价格冲击因子存在于旧的流动性增量退出研究中，不能把它换名当作全新发现。这个记录检查尚未形成新候选或新绩效结果。',
        '',
        '夏普1.2、年化10%的目标保持，目前尚未达到。',
        '',
        '## 十、完整证据入口',
        '',
        '- ' + link(ROOT / 'docs/510300_MONTHLY_DOWNSIDE_FORECAST_V1.md', '本轮计算前固定的全部中文规则'),
        '- ' + link(OUT / 'prediction_result.json', '全部预测门及140个月的检验汇总'),
        '- ' + link(OUT / 'forecasts.csv', '每个月预测与学习参数'),
        '- ' + link(OUT / 'yearly_metrics.csv', '逐年预测增量'),
        '- ' + link(OUT / 'account_metrics.csv', '全部20份账户的完整指标'),
        '- ' + link(OUT / 'account_yearly_metrics.csv', '全部账户逐年收益与风险'),
        '- ' + link(OUT / 'accounts', '12份新账户日账、每次决策及全部成交'),
        '- ' + link(OUT / 'cycle_concentration.csv', '完整与未结束持仓周期的记录'),
        '- ' + link(OUT / 'saved_arithmetic_return_attribution.json', '全部四种情景、四个基准的保存收益拆分'),
        '- ' + link(OUT / 'account_verification_receipt.json', '保存账户复核结果'),
        '',
        f"预测方案登记时间：{protocol['registered_at']}。账户方案登记时间：{account_protocol['registered_at']}。账户结果完成时间：{result['completed_at']}。",
        '',
    ]
    DELIVERY.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(content), encoding='utf-8')
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    record = {
        'study_id': result['study_id'], 'status': result['status'], 'completed_at': result['completed_at'],
        'rules': 'docs/510300_MONTHLY_DOWNSIDE_FORECAST_V1.md',
        'result': str((OUT / 'result.json').relative_to(ROOT)).replace('\\', '/'),
        'report': str(REPORT.relative_to(ROOT)).replace('\\', '/'),
        'result_sha256': hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        'new_candidates': 1, 'new_accounts': 12, 'reused_accounts': 8, 'new_model_fits': 141,
        'prediction_gate_pass': True, 'mature_prediction_months': 140,
        'all_four_historical_point_targets_pass': False, 'selected_trading_model': None,
        'information_benchmark_retained': 'LOG_RISK_RIDGE',
        'saved_return_attribution': str((OUT / 'saved_arithmetic_return_attribution.json').relative_to(ROOT)).replace('\\', '/'),
        'independent_validation': False, 'strict_forward_evidence_days': 0,
        'legacy_round_and_account_counters_not_redefined': True, 'goal_achieved': False, 'position_impact': 0,
    }
    index = read(INDEX)
    preserved_counts = {key: index[key] for key in ['registered_configurations_in_this_resumption', 'evaluation_accounts_in_this_resumption']}
    index['updated_at'] = stamp
    index['status'] = 'FINITE_RISK_INFORMATION_STUDY_COMPLETED_ACCOUNT_TARGETS_NOT_MET'
    index['goal_achieved'] = False
    index['running_studies'] = [r for r in index.get('running_studies', []) if r.get('study', r.get('study_id')) != result['study_id']]
    index['latest_finite_anti_overfitting_historical_study'] = record
    completed = index.setdefault('completed_historical_mining_studies', [])
    if not any(r.get('study_id') == result['study_id'] for r in completed):
        completed.append(record)
    index['finite_historical_mining_totals'] = {
        'scope': '仅本列表所列有限历史研究，不覆盖旧轮次计数；复用账户次数不是唯一账户数量',
        'completed_studies': len(completed),
        'new_candidate_methods': sum(r['new_candidates'] for r in completed),
        'new_accounts': sum(r['new_accounts'] for r in completed),
        'new_model_fits': sum(r['new_model_fits'] for r in completed),
        'reused_account_references': sum(r['reused_accounts'] for r in completed),
    }
    index['last_goal_turn_classification'] = 'PROGRESS_RISK_PREDICTION_PASS_12_ACCOUNTS_AND_RETURN_ATTRIBUTION_COMPLETED'
    index['consecutive_external_data_blocked_goal_turns'] = 0
    index['research_primary_focus'] = '继续有限历史研究，优先验证可执行买卖时机的收益信息；风险预测通过不代表账户达标'
    index['latest_continuation_note'] = record['report']
    index['next_work'].update({
        'status': 'FINITE_HISTORICAL_RETURN_INFORMATION_REVIEW_AND_FIXED_DAILY_OBSERVATION',
        'source': record['report'],
        'focus': '从已完成收益预测研究的训练与跨期增量记录中去重，确定有经济解释的单一买卖时机问题；事前登记后再执行新检验',
        'registered': False, 'planned_settings': 0, 'planned_new_accounts': 0,
        'planned_new_model_fits': 0, 'conditional_new_accounts_if_prediction_passes': 0,
        'implementation_remaining': True,
        'implementation_remaining_scope': '收益信息既有证据去重和下一有限问题的事前登记；本轮141次风险拟合、12份账户及收益归因已全部完成',
        'historical_mining_allowed': True, 'external_data_required': False,
        'next_historical_question_status': 'RETURN_INFORMATION_EVIDENCE_REVIEW_PENDING_NO_NEW_CANDIDATE',
        'next_historical_question': '哪些已有单一收益机制有可重复的跨期信息增量，且仍需要检验其可执行买卖价值？',
        'risk_forecast_fixed_benchmark': record['result'],
        'risk_to_position_mapping_post_result_retuning_allowed': False,
        'known_existing_price_impact_factor': 'config/510300_liquidity_increment_exit_v1.json',
    })
    index['historical_mining_authorization_20260914']['latest_completed_study'] = result['study_id']
    index['historical_mining_authorization_20260914']['latest_result_report'] = record['report']
    assert all(index[key] == value for key, value in preserved_counts.items())
    save(INDEX, index)
    save(OUT / 'delivery_receipt.json', {
        'completed_at': stamp, 'status': 'CHINESE_REPORT_AND_RESEARCH_INDEX_UPDATED',
        'report': record['report'], 'report_sha256': record['report_sha256'],
        'new_accounts_generated_by_delivery': 0, 'new_model_fits_generated_by_delivery': 0,
        'old_round_counters_preserved': True, 'goal_achieved': False,
    })
    print(json.dumps({'status': 'DELIVERED', 'report': record['report'],
                      'historical_mining_allowed': True, 'goal_achieved': False}, ensure_ascii=True))


if __name__ == '__main__':
    main()
