"""交付来源删减结果和新增量证据，接续实际计划合并账户。"""
import json
import shutil
import pandas as pd
from research.gated_source_ablation_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'eleven_setting_joint_comparison.csv')
    mix_path = ROOT/'reports/research/510300_incremental_saved_mix_through208/result.json'
    mix = json.loads(mix_path.read_text(encoding='utf-8'))
    require(ranking.iloc[0].model == 'ADD_GATE_BAND20_LOWER01', '原205首位不同')
    require(all(r['earlier_two_cost_joint_pass'] and not r['main_two_cost_joint_pass'] for r in joint['candidates'].values()), '六套联合结果不同')
    require(mix['total_paths'] == 308 and mix['new_candidates_read'] == 50 and mix['new_accounts'] == 0, '增量诊断范围不同')
    decision = ('第208轮六套来源删减、二十四条完整账户完成，全部未改善原205。另完成308条保存路径的增量组合筛选，'
        '找到四场景虚拟组合越过两项目标的新线索；尚未形成真实组合账户，不能宣布目标完成，下一批直接验证实际合并。')
    detail = [f"六项必要测试通过，核心计算{result['run_seconds']:.2f}秒；核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期、{checked['simulated_fills_checked']}次模拟成交与{checked['suppressed_buy_requests_checked']}次被拦加仓。来源计划从旧实际份额、当时净值和申请独立重建。",
        '预定主方案为去掉区间固定预算、目标1.05倍，主压力夏普1.1649、年化9.9156%，未达标。原倍率去掉该来源是本批最佳新设置，主压力夏普1.1809、年化9.7754%，仍弱于原205；较早历史两档改善不能替代主历史失败。',
        '去掉反弹辅助的原倍率主压力夏普1.1309，只保留核心为1.1366，均明显低于原205的1.1862。因此不能把来源数量减少直接理解为风险调整表现提高。六套删减批次结束，不追加权重。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|']
    for row in ranking.to_dict('records'):
        cells = [f"{row[t+'_net_sharpe']:.4f}／{row[t+'_annualized_return']:.4%}" for t in ['main_base','main_stress','earlier_base','earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.6f}|")
    detail += ['', '## 增量保存收益组合：新线索，不是可交易账户成绩', '',
        f"复用258条旧矩阵，只新增读取五十候选的200份已核对账本，累计308条不同完整路径。两个固定起点、{mix['optimizer_objective_evaluations']}次目标函数评价，耗时{mix['run_seconds']:.2f}秒，零新增交易账户。没有重复原矩阵优化；加入的是后来实际产生的新结果。",
        f"完整虚拟组合最弱门槛比值{mix['best_virtual_minimum_joint_ratio']:.6f}；前三来源归一化为{mix['top_three_virtual_minimum_joint_ratio']:.6f}。两者均大于1，但这是理想日收益混合，未模拟组合自己的资金、整手和成交，不等同目标完成。",
        '|虚拟组合|场景|夏普|复合年化|','|---|---|---:|---:|']
    for row in mix['metrics']:
        detail.append(f"|{row['variant']}|{row['scenario']}|{row['net_sharpe']:.4f}|{row['annualized_return']:.4%}|")
    detail += ['', '下一批直接把五来源、前三来源、70/15/15简单三来源和85/15简单两来源，分别按零与十个百分点外层门槛形成新账户，共八套三十二条。各来源原过滤和等待已经体现在收盘计划，不在新账户重复叠加。168原来源与179保存对照的四份账本已确认完全相同，使用168原决定取得计划。现阶段只准备规则。']
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'retained_joint_candidate': 'ADD_GATE_BAND20_LOWER01', 'incremental_virtual_point_pass': True,
        'incremental_virtual_source': str(mix_path.relative_to(ROOT)),
        'candidates': {m: 'SOURCE_ABLATION_NO_JOINT_IMPROVEMENT_CLOSED' for m in CANDIDATES}}, exclusive=True)
    nxt = ROOT/'docs/510300_INCREMENTAL_SELECTED_INTENT_MIX_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300来源删减与增量组合_第208轮_20260913/六套来源删减及新组合线索_全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '六套来源删减与增量组合新线索',
        'COMPLETED_ABLATIONS_FAILED_NEW_VIRTUAL_MIX_POINT_PASS_REQUIRES_ACCOUNT', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'INCREMENTAL_SELECTED_INTENT_MIX_PREPARED',
        '对增量筛选出的新来源，用四种固定权重和两个外层门槛实际生成完整单账户')
    for name in ['result.json','virtual_metrics.csv','增量组合筛选说明.md']:
        shutil.copy2(mix_path.parent/name, doc.parent/('增量组合_'+name))
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=209, registered=False, planned_settings=8, planned_new_accounts=32,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_INCREMENTAL_SELECTED_PLANNED_INTENT_MIX', source_preflight=str(mix_path.relative_to(ROOT)))
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_gated_source_ablation_comparison'] = str((OUT/'eleven_setting_joint_comparison.csv').relative_to(ROOT))
    index['previous_incremental_saved_mix_through199'] = index['latest_incremental_saved_mix']
    index['latest_incremental_saved_mix'] = str(mix_path.relative_to(ROOT))
    index['latest_incremental_virtual_point_pass'] = {'source':str(mix_path.relative_to(ROOT)), 'total_paths':308,
        'new_candidates':50, 'new_accounts':0, 'goal_achieved':False, 'independent_validation':False,
        'status':'VIRTUAL_MIX_ONLY_AWAITING_ACTUAL_ACCOUNT', 'best_virtual_minimum_joint_ratio':mix['best_virtual_minimum_joint_ratio']}
    for key in ['current_best_joint_comparison_candidate','latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate',
                'latest_all_four_scenario_excess_candidate','latest_secondary_return_margin_candidate']:
        index[key]['last_compared_completed_round'] = 208
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
