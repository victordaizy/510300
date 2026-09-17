"""关闭无效反弹补充，交付全部结果及下一项有限方法。"""
import json

from research.idle_reversal_opportunity_v1 import CONFIG,OUT,PRIMARY,TREND,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not x['main_two_cost_joint_pass'] and not x['earlier_two_cost_joint_pass'] for x in a['candidates'].values()),'第186轮双门槛结论不同')
    decision=('第186轮在原策略明确空仓时加入已有两日反弹。普通超跌主方案主历史基础／压力夏普0.455／0.204、复合年化5.37%／1.87%；'
        '带长期趋势限制的比较方案夏普0.623／0.415、年化6.65%／4.17%。两个方案的较早历史也未达标，新增持仓和交易没有改善全账户结果，关闭这两项固定补充方式。')
    detail=[f"四项必要测试通过，八条新模拟账户核心计算{r['run_seconds']:.2f}秒，不含开发、核对与交付。零次模型训练和外部数据等待。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段模拟持仓、{v['simulated_fills_checked']}次模拟成交。",'',
        '|方案|历史段|费用|净夏普|复合年化|最大回撤|成交次数|', '|---|---|---|---:|---:|---:|---:|']
    for period,key in [('主','all_metrics'),('较早','earlier_diagnostics')]:
        for m in r[key]:
            if m['model'] in [PRIMARY,TREND]:
                detail.append(f"|{'普通超跌主方案' if m['model']==PRIMARY else '趋势内超跌比较方案'}|{period}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{m['max_drawdown']:.2%}|{m['trade_count']}|")
    detail += ['',
        '普通反弹在主历史补入540个原策略空仓决定日，趋势内反弹补入262日，分别把成交次数提高到376和280次。普通反弹基础费用的佣金加滑点约6.16万元，压力约10.20万元；趋势内反弹相应约4.62万元和7.77万元。这些是模拟路径累计实际费用，不能直接除以初始资金当作年化损耗。',
        '主历史普通反弹最大回撤由原预算约6.72%／7.15%扩大到27.92%／35.36%；趋势内反弹也扩大至11.97%／12.90%。较早历史普通反弹回撤29.23%／31.34%，趋势内23.77%／23.82%。增加进入机会未形成有效风险分散。',
        '独立核对从含分红财富变化重建两日强弱及二百日趋势，以状态转换重建完整历史反弹路径，核对已保存信号、原策略优先级、未知处理、次日开盘和实际费用。没有重新训练、重跑旧来源账户，核对不等于独立收益验证。',
        '下一轮保留原策略进入退出资格，直接用六十日市场波动确定仓位，固定比较每日调整和整段固定目标。使用同样10%风险预算，不把包含空仓日的来源账户波动作为分母。',
        '全部历史已经观察，双门槛以joint_target_assessment.json及joint_target_metrics.csv为准。目标保持未完成。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        **{m:'CLOSED_IDLE_REVERSAL_HIGHER_RISK_AND_COSTS_JOINT_FAILED' for m in [PRIMARY,TREND]}},exclusive=True)
    nxt=ROOT/'docs/510300_SIGNAL_MARKET_RISK_BUDGET_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300空仓反弹补充_第186轮_20260913/空仓反弹补充_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'原策略明确空仓时补充已有两日反弹',
        'CLOSED_IDLE_REVERSAL_OPPORTUNITY_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'SIGNAL_MARKET_RISK_BUDGET_PREPARED',
        '保留原进入退出资格，用市场波动直接给仓位，比较每日调整和区间固定')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=187,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=186
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
