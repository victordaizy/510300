"""交付两档无融资资金预算的双门槛结果。"""
import json

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.unlevered_exposure_expansion_v1 import CONFIG, OUT, ROOT
from research.unlevered_exposure_expansion_inputs_v1 import CANDIDATES


def main():
    r = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    a = json.loads((OUT / 'joint_target_assessment.json').read_text(encoding='utf-8'))
    v = json.loads((OUT / 'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not x['four_scenario_joint_pass'] and not x['main_two_cost_joint_pass'] for x in a['candidates'].values()), '关闭结论与双门槛计算不同')
    decision = ('第180轮两档预算均未达到夏普1.2与复合年化10%的双门槛。'
                '一倍半预算主历史基础／压力夏普1.439／1.368、年化8.89%／8.42%；'
                '两倍预算主历史夏普1.323／1.250、年化9.38%／8.83%。'
                '两倍预算较早历史年化10.47%／10.51%，但夏普0.978／0.976。'
                '固定加仓不能解决当前收益与风险同时达标的问题，关闭两项设置。')
    lines = [
        '本轮根据用户新增年化10%目标检验资金使用量。前序“趋势、回撤、波动直接定仓位”与第161轮重复，因此没有新增这一重复账户。',
        '',
        '|预算|历史段|费用|净夏普|复合年化|最大回撤|平均股票仓位|双门槛|',
        '|---|---|---|---:|---:|---:|---:|---|',
    ]
    for period, key in [('主历史', 'all_metrics'), ('较早历史', 'earlier_diagnostics')]:
        for row in r[key]:
            if row['model'] in CANDIDATES:
                passed = row['net_sharpe'] >= 1.2 and row['annualized_return'] >= .10
                lines.append(f"|{CANDIDATES[row['model']]}|{period}|{'基础' if row['cost']=='BASE' else '压力'}|{row['net_sharpe']:.3f}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.2%}|{'通过' if passed else '未通过'}|")
    lines += ['', f"四项必要测试通过；八个模拟账户核心计算{r['run_seconds']:.2f}秒。该时间不含开发、测试和核对。",
              f"另一套目标重建和保存账户核对覆盖{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段完整持仓及{v['simulated_fills_checked']}次模拟成交。目标误差为零，费用、实际开盘、资金约束、分红和终点清算核对通过。",
              '',
              '两倍预算在主历史有174个判断日触及满仓上限。它使平均仓位从原约12.50%提高到18.10%，没有翻倍；高目标触顶、十个百分点调仓带、整手和自身净值共同影响实际结果。不能将原净收益乘二或推断收益会随倍率线性增加。',
              '',
              '主历史两倍预算最大回撤扩大到5.48%／5.96%，较早历史为9.87%／9.88%。较早历史风险调整表现反而弱于原方案，未形成跨时期改善。',
              '',
              '保存主结果中的旧字段“meets_point_target”只检查夏普；当前双门槛请看joint_target_metrics.csv和joint_target_assessment.json，其中全部八项均未通过。',
              '',
              '独立核对代码和完整账户流水不等于独立收益验证。两个历史段都已反复研究，未建立新的独立样本。较早压力费用收益略高来自不同费用父目标与实际份额路径，不表示更高费用创造收益。',
              '',
              '第179轮的失败只否定它的固定选择规则。该轮以收益实现日状态分组，学习的是当日状态与当日收益的关系，并非按前一天状态检验下一日可交易收益；不足样本和零波动也会选择现金。仅凭该轮结果不能证明所有按状态学习的方法无效。',
              '',
              '下一步只检验固定六十日完整账户波动预算，窗口含所有现金日；避免继续逐档增加第180轮的固定倍率。']
    write_json(OUT / 'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
               **{m: 'CLOSED_NO_SCENARIO_SIMULTANEOUS_SHARPE12_CAGR10' for m in CANDIDATES}}, exclusive=True)
    next_path = ROOT / 'docs/510300_ACCOUNT_VOLATILITY_EXPOSURE_NEXT_20260913.md'
    doc = ROOT / 'deliverables/510300资金预算双门槛_第180轮_20260913/资金预算_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '一倍半和两倍股票资金预算、最高满仓',
        'CLOSED_EXPOSURE_EXPANSION_JOINT_TARGET_NOT_MET', decision, '\n'.join(lines), next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'ACCOUNT_VOLATILITY_EXPOSURE_PREPARED',
        '按过去六十日完整账户净收益波动分配股票资金，固定10%风险预算')
    p = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(p.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=181, registered=False, planned_settings=1, planned_new_accounts=4,
                              planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['joint_acceptance_requirements'] = {'net_sharpe_minimum': 1.2, 'compound_annual_return_minimum': .10,
        'required_costs': ['BASE', 'STRESS'], 'cash_annual_return': 0., 'independent_validation_required': True,
        'all_previously_observed_history_is_research_replay': True}
    index['latest_joint_target_assessment'] = str((OUT / 'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 180
    index['latest_main_return_improvement_candidate'] = {
        'round': 180, 'model': 'EXPOSURE_EXPANSION_200', 'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'selection': 'POST_SELECTED_MAXIMUM_MAIN_CAGR_OF_TWO_BUDGETS',
        'status': 'MAIN_CAGR_BELOW10_EARLIER_SHARPE_BELOW12', 'goal_achieved': False}
    write_json(p, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
