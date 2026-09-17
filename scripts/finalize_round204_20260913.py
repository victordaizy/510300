"""交付强势准入四账户场景，合并呈现本回合两批进展并接续仅加仓过滤。"""
import json

import pandas as pd

from research.close_return_buy_strength_gate_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'seven_setting_joint_comparison.csv')
    checks = pd.read_csv(OUT/'saved_buy_gate_checks.csv')
    require(not any(r['main_two_cost_joint_pass'] or r['earlier_two_cost_joint_pass'] for r in joint['candidates'].values()), '本批联合结果不同')
    require(ranking.iloc[0].model == 'INTENT_MIX_BAND_00', '原201比较首位不同')
    previous = json.loads((ROOT/'reports/research/510300_close_return_buy_gate_batch_v1/result.json').read_text(encoding='utf-8'))
    decision = ('第204轮四套收盘强势买入准入、十六条完整账户已完成，全部未通过联合目标。零门槛且当日收益非负才买入能略升主历史夏普，'
        '但主压力年化下降、较早历史明显变弱；原201零门槛合并继续保留，目标尚未完成。结束本次对初次买入和加仓一起过滤的强势方向。')
    detail = [
        f"本回合连续完成第203和204两批，共八套新设置、三十二条新账户；两批核心计算分别为{previous['run_seconds']:.2f}和{result['run_seconds']:.2f}秒，合计{previous['run_seconds']+result['run_seconds']:.2f}秒，不包括开发、测试与核对。",
        f"本批五项必要测试通过，保存核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期、{checked['simulated_fills_checked']}次模拟成交和{checked['suppressed_buy_requests_checked']}次被拦正申请。下限方向与含等号边界独立重算，旧上限文件未修改。",
        '本批主方案为零调仓门槛、收益至少1%才新增买入，主基础／压力夏普1.0846／1.0262、年化8.3596%／7.8656%；较早夏普0.8944／0.9043、年化6.9470%／7.0993%，未达标。',
        '较宽松的零门槛、收益非负才新增买入，主基础／压力夏普1.2541／1.1857，比原201略高；年化10.2925%／9.6687%，反而下降。较早夏普1.0081／1.0146、年化8.7908%／8.9289%，明显弱于原201。不能只挑主历史夏普局部提高来宣称有效。',
        '二十个百分点调仓、收益非负才买入是本批最弱门槛比值最高的新方案，为0.8714，仍明显低于原201的0.9773；没有新的联合比较候选提升。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|',
    ]
    for row in ranking.to_dict('records'):
        cells = [f"{row[tag+'_net_sharpe']:.4f}／{row[tag+'_annualized_return']:.3%}" for tag in ['main_base', 'main_stress', 'earlier_base', 'earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.4f}|")
    detail += ['', '## 被拦请求实际发生于入场还是加仓', '', '|策略|区间|费用|被拦正申请|其中空仓买入|其中加仓|', '|---|---|---|---:|---:|---:|']
    for row in checks.to_dict('records'):
        detail.append(f"|{CANDIDATES[row['model']]}|{'主历史' if row['period']=='evaluation' else '较早历史'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['suppressed_buy_requests']}|{row['suppressed_initial_entry_requests']}|{row['suppressed_addition_requests']}|")
    detail += ['', '## 下一项局部研究', '',
        '零门槛、收益非负才买入在主压力拦64次申请，其中22次是空仓买入、42次是加仓；较早压力34次中分别为14次与20次。第203轮收益不超过1%的零门槛方案，主压力27次被拦有19次发生于空仓、8次为加仓。次数可能包含同一等待过程中多次申请，不能直接当作独立机会。',
        '这说明过去过滤同时作用于初次入场和已有持仓加仓，尚未单独证明仅过滤加仓的效果。下一批保留原规则的初次进入，只在新账户实际已持股且正常申请为正时应用收益条件。两种调仓方式各比较上下方向的零和1%边界，八套、三十二条账户。',
        '下一批目前仅准备规则，尚未实现、测试、冻结或计算。该局部范围也是观察过历史后提出，旧失败完整保留，不宣称独立验证。原201主压力夏普1.1728、年化9.8216%，仍未完整达标，持续目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'candidates': {m: 'LOWER_RETURN_BUY_GATE_FULL_JOINT_FAIL_CLOSED' for m in CANDIDATES},
        'retained_joint_candidate': 'INTENT_MIX_BAND_00',
        'post_selected_local_main_sharpe_improvement': 'BUY_STRENGTH_BAND00_R00'}, exclusive=True)
    nxt = ROOT/'docs/510300_ADDITION_ONLY_RETURN_GATE_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300强势买入过滤批量比较_第204轮_20260913/四套强势买入过滤_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '四套收盘强势买入准入',
        'COMPLETED_LOWER_RETURN_BUY_GATES_FAILED_RETAIN_ROUND201', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'ADDITION_ONLY_RETURN_GATE_BATCH_PREPARED',
        '初次入场完整沿用原策略，只对实际已有持仓的正加仓申请应用四种收益准入')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=205, registered=False, planned_settings=8, planned_new_accounts=32,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_ENTRY_PRESERVED_ADDITION_ONLY_RETURN_FILTERS')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_close_return_buy_strength_comparison'] = str((OUT/'seven_setting_joint_comparison.csv').relative_to(ROOT))
    index['entry_vs_addition_suppression_evidence'] = {'source': str((OUT/'saved_buy_gate_checks.csv').relative_to(ROOT)),
        'model': 'BUY_STRENGTH_BAND00_R00', 'main_stress_suppressed': 64, 'main_stress_initial': 22, 'main_stress_additions': 42,
        'earlier_stress_suppressed': 34, 'earlier_stress_initial': 14, 'earlier_stress_additions': 20,
        'new_accounts_in_this_comparison': 0, 'independent_validation': False}
    for key in ['current_best_joint_comparison_candidate', 'latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 204
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
