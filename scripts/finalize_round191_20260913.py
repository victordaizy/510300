"""完成方向确认日历补充的中文交付，接续有限的持仓与市场联合作用检验。"""
import json

from research.confirmed_month_edge_idle_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT / 'joint_target_assessment.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT / 'saved_verification_receipt.json').read_text(encoding='utf-8'))
    verdict = joint['candidates'][PRIMARY]
    require(not verdict['main_two_cost_joint_pass'] and not verdict['earlier_two_cost_joint_pass'], '第191轮联合结果不同')
    decision = ('第191轮二十日上涨确认减少了无方向日历补充的损害，但仍未完成目标。'
        '主历史基础／压力净夏普1.080／0.936，复合年化10.72%／9.18%；'
        '较早历史净夏普0.731／0.585，年化8.23%／6.40%。暂停日历补充方向，保留第181轮为既有联合目标的比较候选。')
    detail = [
        f"五项必要测试通过，四条新模拟账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试、核对和交付。保存结果已核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}段完整持仓和{checked['simulated_fills_checked']}次模拟成交。",
        '',
        '主历史新增日历机会从第190轮388个原空仓决定日减为190日；较早基础从249日减至135日、压力从244日减至130日。原第181轮已有正目标完全保留。筛掉弱方向机会改善了第190轮，但仍不足以作为完整目标方案。',
        '主历史基础与压力最大回撤分别为10.94%和11.86%，较早为22.21%和23.64%。主基础年化超过10%仅为一个指标通过，不能掩盖基础夏普不足、压力两项不足以及较早的明显弱表现。',
        '独立按原始收盘与分红重建二十日复合涨跌，并按已发生的开市日期重建日历条件。仅使用上一收盘信息，原信号时点15时05分与执行日9时最终决定分别保留。万亿分之一的正涨跌比较容差已在运行前列明，用于浮点舍入，不在看到收益后调整。',
        '这一轮结果是历史完整模拟账户，未发生实盘交易。实现核对不能充当独立收益验证。所有费用情景、现金日、未成交和分年结果保留。',
        '',
        '下一项直接检验持仓状态与市场环境的十二个联合作用，只用原八项数据，预期25组不同成熟输入各拟合一次，其余月份向前复用。先核实相同输入，减少重复训练；只算一个固定方法的四条新账户，不扫描参数。',
    ]
    write_json(OUT / 'candidate_outcomes.json', {
        'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_CONFIRMED_MONTH_EDGE_IDLE_JOINT_TARGET_NOT_MET',
        'calendar_supplement_direction': 'PAUSED_AFTER_UNCONFIRMED_AND_DIRECTION_CONFIRMED_FAILURES',
    }, exclusive=True)
    next_path = ROOT / 'docs/510300_HOLDING_MARKET_COUPLING_EXIT_NEXT_20260913.md'
    document = ROOT / 'deliverables/510300方向确认日历补充_第191轮_20260913/方向确认日历补充_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, document, '方向确认后的原空仓日历补充',
        'CLOSED_CONFIRMED_MONTH_EDGE_IDLE_JOINT_TARGET_NOT_MET', decision, '\n'.join(detail),
        next_path, next_path.read_text(encoding='utf-8').splitlines(), 'HOLDING_MARKET_COUPLING_EXIT_PREPARED',
        '原八项加十二个持仓与市场乘积，单一周期内退出模型，相同成熟输入向前复用')
    path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=192, registered=False, planned_settings=1,
        planned_new_accounts=4, planned_new_model_fits=25, planned_monthly_eligible_models=114,
        planned_reused_monthly_fits=89, planned_new_reference_accounts=0, external_data_required=False)
    index['latest_joint_target_assessment'] = str((OUT / 'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate',
                'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 191
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
