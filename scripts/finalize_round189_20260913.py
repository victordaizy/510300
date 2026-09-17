"""关闭固定周度清仓，登记月内两端的两种完整账户组合。"""
import json

from research.thursday_weekly_reduction_v1 import CONFIG,OUT,PRIMARY,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))['candidates'][PRIMARY]
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not a['main_two_cost_joint_pass'] and not a['earlier_two_cost_joint_pass'],'第189轮结论不同')
    decision=('第189轮周四收盘决定下一开盘清仓。主历史基础／压力净夏普0.990／0.727、复合年化5.94%／4.27%；'
        '较早净夏普0.871／0.606、年化7.06%／4.77%。新增周度清仓显著增加费用，四场景均未达标，关闭这个固定星期规则。')
    detail=[f"四项必要测试通过，四条新模拟账户核心计算{r['run_seconds']:.2f}秒。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段模拟持仓和{v['simulated_fills_checked']}次模拟成交。以上核心耗时不含开发、核对和交付。",'',
        '主历史清除了90个原正目标决定日，较早基础89日、压力90日。主历史成交228次，较早208／211次；压力费用主历史累计佣金及滑点约5.55万元，较早约5.62万元。相较原第181轮主历史约2.56万元、较早约1.39万元的压力费用，成本明显增加。',
        '原模拟账户日内隔夜拆分不能替代新周度清仓账户的结果：清仓同时放弃部分日内收益，恢复会增加成本并改变净资产与份额。四条新账户已完整包含这些影响。',
        '规则只读取已经结束的交易日星期，星期四后休市不会提前退出，星期四没开市不会自动转到星期三。核对保留了这些节假日边界，没有根据未来休市长度或未来价格改变指令。',
        '下一轮使用已保存的月初前三个交易日或月末五个自然日状态，比较只补原空仓与日历期间用满预算。来源目标在前一收盘已知，日历判断在执行日9时进行，同日开盘成交；两个时钟分开记录。',
        '历史双门槛比较仍以第181轮作为主要收益风险对照。目标未完成，已有历史均为研究回放，实现核对不等于独立收益验证。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        PRIMARY:'CLOSED_FIXED_WEEKLY_REDUCTION_HIGHER_COSTS_JOINT_FAILED'},exclusive=True)
    nxt=ROOT/'docs/510300_MONTH_EDGE_OPPORTUNITY_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300周度清仓_第189轮_20260913/周度清仓_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'周四收盘决定下一开盘清仓',
        'CLOSED_THURSDAY_WEEKLY_REDUCTION_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'MONTH_EDGE_OPPORTUNITY_PREPARED',
        '按执行日九点已知日历，比较原空仓补充月内两端及月内两端使用全部预算')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=190,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=189
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
