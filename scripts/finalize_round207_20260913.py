"""交付普通减仓批次并明确结束，接续已有来源的有限删减。"""
import json
import pandas as pd
from research.positive_target_reduction_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'thirteen_setting_joint_comparison.csv')
    require(ranking.iloc[0].model == 'ADD_GATE_BAND20_LOWER01', '原205首位不同')
    require(not any(r['main_two_cost_joint_pass'] for r in joint['candidates'].values()), '主历史联合结果不同')
    decision = ('第207轮八套普通减仓、三十二条完整账户完成。全部未改善原205的联合差距，主历史压力夏普均下降；'
        '减仓过滤、暂缓及半量批次结束，原205仍为当前联合比较候选，完整目标未达到。')
    detail = [f"七项必要测试通过，核心计算{result['run_seconds']:.2f}秒；核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期、{checked['simulated_fills_checked']}次模拟成交。独立检查{checked['modified_reduction_requests_checked']}次改动的普通减仓请求、{checked['zero_exit_requests_checked']}次明确零退出申请以及{checked['suppressed_buy_requests_checked']}次被拦加仓。申请次数可能包含等待，不能等同独立成交。",
        '预定主方案为1.05倍、普通减仓一半：主压力夏普1.1477、年化10.3351%，失败。实际成交从原206同倍率的110次增至123次，半量使后续收盘再次调整，不能简单假定减半数量会节省总费用。',
        '本批最接近联合目标的新设置为原倍率、非负日才普通减仓，主压力夏普1.1761、年化9.7926%，均低于原205；较早两档有改善，仍不足以抵消主历史的失败。',
        '暂不普通减仓的原倍率方案主压力年化提高至10.5617%，但夏普降至1.1391；较早两档夏普也低于1.2。提高持有程度仍没有同时解决收益与风险。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|']
    for row in ranking.to_dict('records'):
        cells = [f"{row[t+'_net_sharpe']:.4f}／{row[t+'_annualized_return']:.4%}" for t in ['main_base','main_stress','earlier_base','earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.6f}|")
    detail += ['', '下一批恢复原普通减仓，固定已有1%加仓条件，分别去掉三来源中的一个辅助来源，以及只保留核心来源；每种比较原倍率与1.05倍，共六套二十四条账户。使用收盘计划份额合并，不能把两次归零确认的等待误当空仓。现阶段仅准备规则。全部历史仍是研究回放，独立验证未建立。']
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'retained_joint_candidate': 'ADD_GATE_BAND20_LOWER01',
        'candidates': {m: 'POSITIVE_TARGET_REDUCTION_NO_JOINT_IMPROVEMENT_CLOSED' for m in CANDIDATES}}, exclusive=True)
    nxt = ROOT/'docs/510300_GATED_SOURCE_ABLATION_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300普通减仓批量比较_第207轮_20260913/八套普通减仓_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '八套正目标普通减仓',
        'COMPLETED_ORDINARY_REDUCTION_VARIANTS_FAILED_RETAIN_ROUND205', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'GATED_SOURCE_ABLATION_BATCH_PREPARED',
        '固定已有加仓规则，去掉一个辅助来源或只保留核心，三个简化各两档投入')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=208, registered=False, planned_settings=6, planned_new_accounts=24,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_GATED_SOURCE_ABLATION_WITH_PLANNED_EXPOSURE')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_positive_target_reduction_comparison'] = str((OUT/'thirteen_setting_joint_comparison.csv').relative_to(ROOT))
    for key in ['current_best_joint_comparison_candidate','latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate',
                'latest_all_four_scenario_excess_candidate','latest_secondary_return_margin_candidate']:
        index[key]['last_compared_completed_round'] = 207
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
