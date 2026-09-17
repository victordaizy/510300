"""交付第200轮完整账户及节费归因，接续有限调仓门槛批次。"""
import json

import pandas as pd

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.three_source_order_intent_mix_v1 import CANDIDATES, CONFIG, OUT, PRIMARY, ROOT


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    require(joint['candidates'][PRIMARY]['earlier_two_cost_joint_pass'], '简单组合较早联合结果不同')
    require(not any(r['four_scenario_joint_pass'] for r in joint['candidates'].values()), '本轮完整目标状态不同')
    ranked = []
    for model in [*CANDIDATES, 'TWO_CLOSE_ZERO_EXIT']:
        row = {'model': model}
        ratios = []
        for period, key in [('main', 'all_metrics'), ('earlier', 'earlier_diagnostics')]:
            for cost in ['BASE', 'STRESS']:
                metric = next(r for r in result[key] if r['model'] == model and r['cost'] == cost)
                row.update({period+'_'+cost.lower()+'_'+k: metric[k] for k in ['net_sharpe', 'annualized_return', 'max_drawdown']})
                ratios.extend([metric['net_sharpe']/1.2, metric['annualized_return']/.1])
        row['minimum_joint_ratio'] = min(ratios)
        ranked.append(row)
    ranking = pd.DataFrame(ranked).sort_values('minimum_joint_ratio', ascending=False)
    require(ranking.iloc[0].model == 'TWO_CLOSE_ZERO_EXIT', '联合比较最接近者变化')
    ranking.to_csv(OUT/'three_candidate_joint_comparison.csv', index=False, encoding='utf-8-sig')
    diffs = pd.read_csv(OUT/'saved_comparison_differences.csv')
    main = diffs[(diffs.model == PRIMARY) & (diffs.comparison == 'TWO_CLOSE_ZERO_EXIT') &
                 (diffs.period == 'evaluation') & (diffs.cost == 'STRESS')].iloc[0]
    saved_cost = -float(main.commission_difference+main.slippage_cost_difference)
    gross = float(main.price_pnl_difference+main.dividend_recognized_difference)
    decision = ('第200轮两套三来源计划份额合并、八条完整账户已完成。80%／15%／5%简单权重在較早历史两档费用同时达到夏普1.2和年化10%，'
        '但主历史压力情景仍未通过；保存筛选权重也未通过全部条件。原198继续作为四场景联合最接近候选，目标保持进行中。').replace('較早', '较早')
    detail = [
        f"五项必要测试通过。八条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试、核对；无新预测模型、参考账户或外部来源。",
        f"已核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期及{checked['simulated_fills_checked']}次模拟成交。来源首次明确零仍持仓的计划已保留，只使用来源当时的份额、净值和申请，未从下一开盘成交反推。",
        '简单组合主基础／压力夏普1.225883／1.158570、年化10.4729%／9.8308%；较早基础／压力夏普1.203419／1.204050、年化11.0609%／11.1421%。较早通过只是历史点值，独立验证未建立。',
        '保存筛选权重的较早压力夏普为1.1998741367，严格低于1.2。表格若四舍五入显示1.200，仍不能判定为通过。主压力夏普1.160820、年化9.8601%，也未通过。',
        f"相对原198，简单组合主压力佣金与滑点合计节省{saved_cost:.2f}元，但价格与分红毛利润变化为{gross:.2f}元，最终净资产少{abs(main.terminal_equity_difference):.2f}元。交易次数从113增至143，单次规模缩小使总费用下降；不能把节费解释为交易次数减少。",
        '简单组合较早最大回撤从原198的7.89%／8.04%降至7.41%／7.56%，夏普提高，但年化从11.21%／11.33%降至11.06%／11.14%。主历史回撤扩大至8.04%／8.57%，因此不能把较早改善扩展为全面改善。',
        '按四场景夏普与年化各自距离门槛的最弱比例，原198为0.967607，保存权重合并为0.967350，简单合并为0.965475。该比例只用于比较离目标的差距，不是收益、胜率或独立证据。',
        '本轮不再追加权重。下一批一次检验两个来源的零、五、十五、二十个百分点调仓门槛，共八套设置、三十二条账户，原十个百分点直接复用。下一批目前仅准备方案，尚未实现、测试、冻结或回测。全部历史已反复使用，目标尚未完成。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'candidates': {PRIMARY: 'EARLIER_TWO_COST_JOINT_PASS_MAIN_STRESS_FAIL',
            'INTENT_MIX_DIAGNOSTIC_WEIGHTS': 'MAIN_STRESS_AND_EARLIER_STRESS_SHARPE_FAIL'},
        'retained_joint_candidate': 'TWO_CLOSE_ZERO_EXIT', 'cost_saved_main_stress_vs198': saved_cost,
        'gross_profit_change_main_stress_vs198': gross}, exclusive=True)
    nxt = ROOT/'docs/510300_FINITE_REBALANCE_BAND_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300三来源计划份额合并_第200轮_20260913/三来源合并账户_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '三来源买卖意图合并为一个账户',
        'COMPLETED_SIMPLE_MIX_EARLIER_JOINT_PASS_MAIN_STRESS_FAIL', decision, '\n\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'FINITE_REBALANCE_BAND_BATCH_PREPARED',
        '两套现有目标一次比较四档调仓门槛，复用原十个百分点账户')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=201, registered=False, planned_settings=8, planned_new_accounts=32,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_FINITE_REBALANCE_BAND_CALIBRATION')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_intent_mix_comparison'] = str((OUT/'three_candidate_joint_comparison.csv').relative_to(ROOT))
    index['latest_earlier_joint_pass_intent_mix'] = {'round': 200, 'model': PRIMARY, 'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED', 'status': 'EARLIER_TWO_COST_JOINT_PASS_MAIN_STRESS_FAIL'}
    for key in ['current_best_joint_comparison_candidate', 'latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 200
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
