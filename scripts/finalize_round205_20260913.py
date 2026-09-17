"""交付仅加仓结果，保留联合差距改善并接续固定条件的仓位投入批次。"""
import json
import pandas as pd
from research.addition_only_return_gate_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'eleven_setting_joint_comparison.csv')
    checks = pd.read_csv(OUT/'saved_buy_gate_checks.csv')
    selected = ranking.iloc[0].to_dict()
    require(selected['model'] == 'ADD_GATE_BAND20_LOWER01', '联合首位不同')
    require(not any(r['four_scenario_joint_pass'] for r in joint['candidates'].values()), '完整通过状态不同')
    require(checks.suppressed_initial_entry_requests.eq(0).all(), '存在误拦初次入场')
    decision = ('第205轮八套仅加仓过滤、三十二条完整账户完成。二十个百分点调仓、当天收益至少1%才允许已有持仓加仓的方案，'
        '成为目前联合差距最小的历史比较候选；主压力夏普1.1862、年化9.8357%，较早两档已通过，但完整目标未达到。')
    detail = [f"五项必要测试通过，核心计算{result['run_seconds']:.2f}秒，不包括开发、测试与核对。共核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期、{checked['simulated_fills_checked']}次模拟成交；{checked['suppressed_buy_requests_checked']}次被拦请求全部发生于已有实际持仓，没有误拦首次买入。",
        '原定主方案是零门槛、收益非负才加仓，未达标；当前保留的是八套结果中的事后选择。最弱门槛比值由原201的0.977332提高到0.983572，仍小于1。不能把较早历史或主基础单独通过等同完整成功。',
        '相对原201零门槛，选中方案四档夏普都提高、回撤都减小；主压力年化略升，其他三档年化下降。主压力夏普从1.1728升至1.1862、年化从9.8216%升至9.8357%、最大回撤由8.4170%降至7.1576%。较早压力夏普1.2310、年化10.4425%，已越过两项门槛。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|']
    for row in ranking.to_dict('records'):
        cells = [f"{row[t+'_net_sharpe']:.4f}／{row[t+'_annualized_return']:.4%}" for t in ['main_base','main_stress','earlier_base','earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.6f}|")
    detail += ['', '本批上下方向、零与1%边界的八套过滤比较结束，不追加相邻收益阈值。下一研究固定选中方案的全部进出场与加仓条件，只检验1.05、1.10、1.15、1.20倍目标投入，上限100%，四套十六条新账户。现阶段仅准备规则。所有已经观察过的历史仍为研究回放，独立验证未建立。']
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'post_selected_joint_candidate': selected['model'], 'minimum_joint_ratio': selected['minimum_joint_ratio'],
        'candidates': {m: 'EARLIER_TWO_COST_PASS_MAIN_FAIL' if joint['candidates'][m]['earlier_two_cost_joint_pass'] else 'FULL_JOINT_TARGET_NOT_MET' for m in CANDIDATES}}, exclusive=True)
    nxt = ROOT/'docs/510300_ADDITION_GATE_EXPOSURE_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300仅加仓过滤批量比较_第205轮_20260913/八套仅加仓过滤_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '八套仅加仓过滤',
        'COMPLETED_ADDITION_ONLY_JOINT_GAP_IMPROVED_FULL_TARGET_NOT_MET', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'ADDITION_GATE_EXPOSURE_BATCH_PREPARED',
        '固定二十个百分点调仓和1%加仓准入，检验四档目标投入倍数并限制100%')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    old = index['current_best_joint_comparison_candidate']
    require(selected['minimum_joint_ratio'] > old['minimum_joint_ratio'], '最弱差距没有改善')
    index['previous_best_joint_comparison_candidate_before205'] = old
    index['current_best_joint_comparison_candidate'] = {'round': 205, 'study': result['study_id'], 'model': selected['model'],
        'source_result': str((OUT/'result.json').relative_to(ROOT)), 'minimum_joint_ratio': selected['minimum_joint_ratio'],
        'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED',
        'status': 'POST_SELECTED_JOINT_GAP_IMPROVEMENT_MAIN_STRESS_FAIL_EARLIER_PASS', 'last_compared_completed_round': 205,
        **{k: selected[k] for k in selected if k.endswith(('_net_sharpe','_annualized_return','_max_drawdown'))}}
    index['next_work'].update(candidate_round=206, registered=False, planned_settings=4, planned_new_accounts=16,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_FIXED_ADDITION_GATE_CAPPED_EXPOSURE_MULTIPLIERS')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_addition_only_return_gate_comparison'] = str((OUT/'eleven_setting_joint_comparison.csv').relative_to(ROOT))
    for key in ['latest_main_return_improvement_candidate','current_best_four_scenario_comparison_candidate',
                'latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 205
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
