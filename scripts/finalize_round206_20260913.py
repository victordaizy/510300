"""交付四档投入结果，接续正目标减仓比较并保留原联合候选。"""
import json
import pandas as pd
from research.addition_gate_exposure_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'eight_setting_joint_comparison.csv')
    require(ranking.iloc[0].model == 'ADD_GATE_BAND20_LOWER01', '保留比较首位不同')
    require(all(r['earlier_two_cost_joint_pass'] and not r['main_two_cost_joint_pass'] for r in joint['candidates'].values()), '四档联合结果不同')
    rows = []
    for folder, model in [('510300_addition_only_return_gate_batch_v1','ADD_GATE_BAND20_LOWER01'),
                          ('510300_addition_gate_exposure_batch_v1','ADD_GATE_EXPOSURE_105')]:
        for period in ['evaluation','earlier_diagnostic']:
            for cost in ['BASE','STRESS']:
                d = pd.read_parquet(ROOT/'reports/research'/folder/period/cost/f'{model}_decisions.parquet')
                ordinary = d.requested_quantity.lt(0)&d.reference_weight.gt(0)
                rows.append({'model': model, 'period': period, 'cost': cost,
                    'positive_target_sell_requests': int(ordinary.sum()),
                    'nonnegative_return_sell_requests': int((ordinary&d.observed_daily_return.ge(0)).sum()),
                    'zero_target_sell_requests': int((d.requested_quantity.lt(0)&d.reference_weight.eq(0)).sum())})
    pd.DataFrame(rows).to_csv(OUT/'saved_reduction_request_inventory.csv', index=False, encoding='utf-8-sig')
    decision = ('第206轮四档目标投入、十六条完整账户完成。四档较早历史均通过，主历史压力年化均超过10%，'
        '但夏普均低于1.2且低于原205；投入倍率批次结束，当前最接近联合目标的方案仍为原205。')
    detail = [f"六项必要测试通过，核心计算{result['run_seconds']:.2f}秒；核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期、{checked['simulated_fills_checked']}次模拟成交和{checked['suppressed_buy_requests_checked']}次被拦加仓。均为自己的实际账户，目标上限100%，首次买入没有被拦。",
        '预定主方案1.10倍的主压力夏普1.1696、年化10.5656%，未达标；四个倍率全列出，没有追加倍率或挑年份。1.05倍是本批最接近联合目标的新方案，但最弱门槛比值0.982141，仍低于原205的0.983572。',
        '1.05倍把主压力年化从9.8357%提高到10.5479%，夏普由1.1862降至1.1786。其余三档都通过联合目标，单独主压力夏普仍失败。该方案可保留为有年化余量的次级研究对照，不能取代原205的联合最优身份。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|']
    for row in ranking.to_dict('records'):
        cells = [f"{row[t+'_net_sharpe']:.4f}／{row[t+'_annualized_return']:.4%}" for t in ['main_base','main_stress','earlier_base','earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.6f}|")
    detail += ['', '下一批只改变正目标期间的普通减仓，明确零和终点清仓完全沿用。原倍率及1.05倍各比较非正日减仓、非负日减仓、暂不普通减仓、普通减仓一半，共八套三十二条账户。当前只准备规则。保存申请盘点不生成新账户、不代表成交数量或因果结论。所有历史依然是研究回放，独立验证未建立。']
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'retained_joint_candidate': 'ADD_GATE_BAND20_LOWER01', 'secondary_return_margin_candidate': 'ADD_GATE_EXPOSURE_105',
        'candidates': {m: 'EARLIER_PASS_MAIN_STRESS_SHARPE_FAIL_CLOSED_EXPOSURE_BATCH' for m in CANDIDATES}}, exclusive=True)
    nxt = ROOT/'docs/510300_POSITIVE_TARGET_REDUCTION_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300固定加仓投入批量比较_第206轮_20260913/四档目标投入_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '固定加仓条件的四档目标投入',
        'COMPLETED_EXPOSURE_RETURN_IMPROVED_SHARPE_FAILED_RETAIN_ROUND205', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'POSITIVE_TARGET_REDUCTION_BATCH_PREPARED',
        '清仓照常，比较正目标减仓条件、暂缓和半量，两个固定投入各四种')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=207, registered=False, planned_settings=8, planned_new_accounts=32,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_POSITIVE_TARGET_REDUCTION_WITH_ZERO_EXIT_PRESERVED')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_addition_gate_exposure_comparison'] = str((OUT/'eight_setting_joint_comparison.csv').relative_to(ROOT))
    index['latest_saved_reduction_request_inventory'] = str((OUT/'saved_reduction_request_inventory.csv').relative_to(ROOT))
    selected = ranking[ranking.model.eq('ADD_GATE_EXPOSURE_105')].iloc[0].to_dict()
    index['latest_secondary_return_margin_candidate'] = {'round': 206, 'model': selected['model'],
        'source_result': str((OUT/'result.json').relative_to(ROOT)), 'goal_achieved': False,
        'independent_validation': 'NOT_ESTABLISHED', 'last_compared_completed_round': 206,
        **{k: selected[k] for k in selected if k.endswith(('_net_sharpe','_annualized_return','_max_drawdown'))}}
    for key in ['current_best_joint_comparison_candidate','latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 206
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
