"""呈现更接近双目标的新候选，完整保留压力和较早夏普缺口。"""
import json
import math

import pandas as pd

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.two_close_zero_exit_v1 import CONFIG, OUT, PRIMARY, ROOT


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    require(not joint['candidates'][PRIMARY]['four_scenario_joint_pass'], '本轮联合结论不同')
    index_path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    ranking = pd.read_csv(ROOT/'reports/research/510300_joint_saved_frontier_through186/joint_four_scenario_ranking.csv')
    old_best = ranking.loc[ranking.minimum_joint_ratio.idxmax()]
    comparisons = [{'through_round': 186, 'model': old_best['model'], 'minimum_joint_ratio': float(old_best.minimum_joint_ratio),
        'source': 'reports/research/510300_joint_saved_frontier_through186/joint_four_scenario_ranking.csv'}]
    observed = [(r['round'], ROOT/r['result']) for r in index['completed_rounds'] if r['round'] > 186]
    observed.append((198, OUT/'result.json'))
    for number, path in observed:
        saved = json.loads(path.read_text(encoding='utf-8'))
        for model in {r['model'] for r in saved.get('all_metrics', [])}:
            main_rows = [r for r in saved['all_metrics'] if r['model'] == model]
            early_rows = [r for r in saved.get('earlier_diagnostics', []) if r['model'] == model]
            if len(main_rows) != 2 or len(early_rows) != 2:
                continue
            rows = [next((r for r in group if r['cost'] == cost), None) for group in [main_rows, early_rows] for cost in ['BASE', 'STRESS']]
            if any(r is None for r in rows) or [r['trading_days'] for r in rows] != [1604, 1604, 1219, 1219]:
                continue
            if not all(r['net_sharpe'] is not None and math.isfinite(r['net_sharpe']) and math.isfinite(r['annualized_return']) for r in rows):
                continue
            comparisons.append({'through_round': number, 'model': model,
                'minimum_joint_ratio': min(min(r['net_sharpe']/1.2, r['annualized_return']/.1) for r in rows),
                'source': str(path.relative_to(ROOT))})
    frame = pd.DataFrame(comparisons).sort_values(['minimum_joint_ratio', 'through_round'], ascending=[False, False])
    frame.to_csv(OUT/'incremental_joint_comparison.csv', index=False, encoding='utf-8-sig')
    require(frame.iloc[0].model == PRIMARY and frame.iloc[0].through_round == 198, '联合比较最优身份不同')
    score = float(frame.iloc[0].minimum_joint_ratio)
    decision = ('第198轮两次明确零确认退出明显缩小联合目标缺口，但尚未达标。主历史基础／压力净夏普1.230／1.161，年化10.60%／9.93%；'
        '较早历史夏普1.183／1.187，年化11.21%／11.33%。四个场景的净利润均高于原181，主历史波动和回撤同时上升。'
        '保留为当前同一四场景口径下最接近联合门槛的历史比较候选，不能标为已完成或独立验证通过。')
    detail = [
        f"七项必要测试通过，四条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试与核对。保存核对覆盖{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期、{checked['simulated_fills_checked']}次模拟成交。",
        '相对原181，主历史基础／压力账户终值分别增加21365.13元／20674.99元；较早分别增加19961.44元／22613.65元。主压力价格与分红利润增加20281.30元，费用再节省393.69元；较早压力价格与分红利润增加23107.20元，费用增加493.55元。改善主要来自退出时点，不能归因为手续费减少。',
        '代价是主历史年化波动从原约7.26%升至8.47%／8.46%，最大回撤从6.72%／7.15%升至7.81%／8.34%。因此主历史年化提高，夏普却从1.306／1.223下降至1.230／1.161。较早回撤由9.68%／10.14%降至7.89%／8.04%，属于局部风险改善。',
        '尚缺：主压力夏普距1.2差约0.039，年化距10%差约0.07个百分点；较早基础／压力夏普分别差约0.017／0.013。主基础已通过两项点值门槛，其他场景不能用四舍五入算作通过。',
        f"复用截至186轮的标准完整四场景保存排名，并增量比较187至198轮齐全指标，最弱门槛比值从原181的{float(old_best.minimum_joint_ratio):.4f}提高到{score:.4f}。比值是八项指标相对对应门槛的最小值，只用于研究优先级，不是达标概率，也不是全部旧研究重新审计。比较范围及来源保存在incremental_joint_comparison.csv。",
        '这条退出规则是在查看一个与两个交易日的退出后价格诊断后选择，已经使用过历史。四条完整模拟账户修正了诊断中缺少资金、费用、分红和再进入的问题，但没有产生新的独立样本。',
        '本轮没有增加来源模型、外部数据或其他资产。下一批保持两次零确认，只把已有十八套预算中尚未使用这条退出规则的十七套一次组合，共六十八条新账户；原198直接复用。没有增加风险窗口、预算数值或确认天数。',
        '下一批目前只准备完整中文方案，尚未实现、冻结或回测；目标继续保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'RETAIN_CLOSEST_SAVED_FOUR_SCENARIO_JOINT_CANDIDATE_STRESS_AND_EARLIER_SHARPE_BELOW12',
        'minimum_joint_ratio': score, 'independent_validation': 'NOT_ESTABLISHED'}, exclusive=True)
    nxt = ROOT/'docs/510300_CONFIRMED_EXIT_EXISTING_BUDGET_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300两次归零确认退出_第198轮_20260913/两次归零确认退出_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '连续两个收盘明确归零后再退出',
        'RETAINED_CLOSEST_JOINT_CANDIDATE_FULL_TARGET_NOT_MET', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'CONFIRMED_EXIT_EXISTING_BUDGET_BATCH_PREPARED',
        '固定两次零确认，将已有预算中尚未采用新退出的十七套一次批量组合')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=199, registered=False, planned_settings=17, planned_new_accounts=68,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_FIXED_EXIT_AND_EXISTING_BUDGET_INTERACTION_BATCH', reused_completed_candidate='TWO_CLOSE_ZERO_EXIT')
    source_result = str((OUT/'result.json').relative_to(ROOT))
    values = {('main' if key == 'all_metrics' else 'earlier')+'_'+r['cost'].lower(): r
              for key in ['all_metrics', 'earlier_diagnostics'] for r in result[key] if r['model'] == PRIMARY}
    candidate = {'round': 198, 'study': result['study_id'], 'model': PRIMARY, 'source_result': source_result,
        'minimum_joint_ratio': score, 'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED',
        'status': 'CLOSEST_STANDARD_SAVED_JOINT_CANDIDATE_MAIN_STRESS_AND_EARLIER_SHARPE_NOT_MET',
        **{tag+'_'+field: row[field] for tag, row in values.items() for field in ['net_sharpe', 'annualized_return', 'max_drawdown']}}
    index['current_best_joint_comparison_candidate'] = candidate
    index['latest_main_return_improvement_candidate'] = candidate.copy()
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_incremental_joint_comparison'] = str((OUT/'incremental_joint_comparison.csv').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 198
    write_json(index_path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
