"""关闭无方向确认的日历补充，报告收益过线但风险不合格。"""
import json

from research.month_edge_opportunity_v1 import CONFIG,OUT,PRIMARY,FULL,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not x['main_two_cost_joint_pass'] and not x['earlier_two_cost_joint_pass'] for x in a['candidates'].values()),'第190轮结论不同')
    decision=('第190轮原空仓时补充月内两端，主历史基础／压力夏普0.973／0.820、复合年化11.52%／9.52%；'
        '较早夏普0.355／0.245、年化4.61%／2.71%。月内两端满预算比较方案更弱。主基础年化单项过线不代表目标通过，两个固定组合均关闭。')
    detail=[f"五项必要测试通过，八条新模拟账户核心计算{r['run_seconds']:.2f}秒，不含开发、核对与交付。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段模拟持仓及{v['simulated_fills_checked']}次模拟成交。",'',
        '主方案补入主历史388个原空仓决定日，较早基础249日、压力244日，主历史成交228次。基础费用年化达到11.52%，但最大回撤14.91%、夏普0.973；压力回撤17.46%，年化与夏普都不足。较早回撤27.82%／29.06%，不能以主历史的单项收益掩盖。',
        '比较方案在日历有效且原已有部分仓位时也用满预算，主历史基础／压力年化10.59%／8.63%、夏普0.848／0.706；较早年化2.94%／1.08%、夏普0.254／0.149。进一步增加暴露没有改善收益风险效率。',
        '独立按已发生日期重建月内已开市序号和自然月末条件。原目标时点为前一收盘15时05分，本轮最终决定为执行日9时，两个时间分别保留；没有用执行日行情形成信号。全部份额仍按前一收盘净值和价格请求，再由开盘实际模拟费用及现金约束决定成交。',
        '下一轮单独检验此前二十日含分红财富上涨确认：只约束新增日历机会，原第181轮正目标幅度不变。月初三日和月末五个自然日不变，只有一个预先列明候选。',
        '目标尚未完成。完整基础、压力、年度及较早结果均保留，实现核对不构成独立收益验证。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        **{m:'CLOSED_UNCONFIRMED_MONTH_EDGE_OPPORTUNITY_LOW_SHARPE_EARLY_LOSS' for m in [PRIMARY,FULL]}},exclusive=True)
    nxt=ROOT/'docs/510300_CONFIRMED_MONTH_EDGE_IDLE_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300月内两端补充_第190轮_20260913/月内两端补充_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'月初月末日历机会与原账户风险预算',
        'CLOSED_MONTH_EDGE_OPPORTUNITY_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'CONFIRMED_MONTH_EDGE_IDLE_PREPARED',
        '原空仓的日历补充需要前一收盘二十日上涨确认，日历天数与原正目标幅度不变')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=191,registered=False,planned_settings=1,planned_new_accounts=4,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=190
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
