"""交付均值反弹有限组合，并将下一步改为复用账本的批量快筛。"""
import json

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.mean_rebound_auxiliary_v1 import CONFIG, OUT, ROOT


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    require(not joint['goal_achieved'], '联合验收状态不同')
    decision = ('第195轮四分之一反弹辅助在主历史基础费用下达到净夏普1.316、年化10.16%；压力费用下夏普1.223、年化9.37%，收益仍不足。'
        '二分之一辅助主基础夏普1.275、年化10.52%，压力夏普1.176、年化9.64%。两种组合均降低较早历史夏普，完整目标未完成。'
        '保留四分之一组合的主历史局部改善记录，不追加反弹权重；原181仍作为跨场景联合比较基准。')
    detail = [
        f"六项必要测试通过，八条新组合账户加两条原来缺失的较早辅助参考，核心计算共{result['run_seconds']:.2f}秒，不含开发、测试、核对与交付。原辅助主历史两条直接复用，没有重跑。",
        f"保存结果已核对：组合{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期、{checked['simulated_fills_checked']}次模拟成交；新增参考另有{checked['reference_decisions_checked']}个决定、{checked['complete_reference_cycles']}个周期、{checked['reference_fills_checked']}次模拟成交。核对包含实际份额、成本、分红、退出和终点结算，属于模拟结果核对。",
        '四分之一组合较早基础／压力净夏普1.017／1.001，年化9.67%／9.57%，回撤10.49%／10.64%；二分之一组合夏普0.926／0.902，年化9.29%／9.09%，回撤11.91%／12.37%。',
        '原反弹辅助单独在较早基础／压力场景的夏普为负0.112／负0.161、年化为负1.20%／负1.58%。主历史约0.136的低相关性没有带来稳定的跨历史改善；加大辅助会加重较早退化，因此结束这两档组合试验。',
        '主历史基础费用单项联合过线，不能抵消压力费用的收益不足和较早历史的夏普不足。全部结果来自已反复研究的历史，独立验证尚未建立。',
        '下一步先批量检查既有完整策略之间的互补性，再对少量组合做完整账户，减少逐个小改动和重复文档。虚拟收益组合只用于筛选，不能冒充可成交的策略成绩。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'MEAN_REBOUND_AUX_25': 'MAIN_BASE_JOINT_PASS_LOCAL_IMPROVEMENT_EARLIER_DEGRADATION_NO_PROMOTION',
        'MEAN_REBOUND_AUX_50': 'MAIN_BASE_JOINT_PASS_STRESS_AND_EARLIER_FAIL_CLOSED'}, exclusive=True)
    nxt = ROOT/'docs/510300_SAVED_COMBINATION_FAST_SCREEN_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300均值反弹辅助_第195轮_20260913/均值反弹辅助_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, doc, '原账户加四分之一或二分之一均值反弹',
        'COMPLETED_LOCAL_MAIN_IMPROVEMENT_FULL_JOINT_TARGET_NOT_MET', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'SAVED_COMBINATION_FAST_SCREEN_PREPARED',
        '先批量筛选保存收益的互补组合，再准备少量完整账户，暂停逐项追加小参数')
    p = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(p.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=196, registered=False, planned_settings=0,
        planned_new_accounts=0, planned_new_model_fits=0, planned_new_reference_accounts=0,
        planned_weight_optimizer_starts=2, external_data_required=False,
        research_class='RETROSPECTIVE_SAVED_RETURN_FEASIBILITY_SCREEN_BEFORE_NEXT_ACCOUNT_ROUND')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_main_return_improvement_candidate'] = {'round': 195, 'model': 'MEAN_REBOUND_AUX_25',
        'source_result': str((OUT/'result.json').relative_to(ROOT)), 'goal_achieved': False,
        'status': 'MAIN_BASE_JOINT_PASS_STRESS_CAGR_AND_EARLIER_SHARPE_FAIL',
        'selection': 'LOCAL_MAIN_RETURN_IMPROVEMENT_ONLY_EARLIER_DEGRADATION'}
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 195
    write_json(p, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
