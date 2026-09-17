"""交付市场风险预算的失败证据与下一项策略自身收益条件。"""
import json

from research.signal_market_risk_budget_v1 import CONFIG,OUT,PRIMARY,EPISODE,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not x['main_two_cost_joint_pass'] and not x['earlier_two_cost_joint_pass'] for x in a['candidates'].values()),'第187轮结论不同')
    decision=('第187轮保留原正信号资格并按六十日市场波动直接分配10%风险预算。每日调整主方案主历史基础／压力夏普1.111／1.039、复合年化6.47%／6.02%；'
        '较早夏普0.831／0.812、年化5.08%／4.97%。区间固定方案同样未达标。两方案降低部分回撤，却明显削弱收益，关闭这两项直接市场波动仓位方案。')
    detail=[f"五项必要测试通过，八条新模拟账户核心计算{r['run_seconds']:.2f}秒，不含开发、核对与交付。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段模拟持仓和{v['simulated_fills_checked']}次模拟成交。",'',
        '|方案|历史段|费用|净夏普|复合年化|最大回撤|成交次数|','|---|---|---|---:|---:|---:|---:|']
    for period,key in [('主','all_metrics'),('较早','earlier_diagnostics')]:
        for m in r[key]:
            if m['model'] in [PRIMARY,EPISODE]:
                detail.append(f"|{'每日调整主方案' if m['model']==PRIMARY else '区间固定比较方案'}|{period}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{m['max_drawdown']:.2%}|{m['trade_count']}|")
    detail += ['',
        '来源进入资格保持主历史436个正目标决定日、较早基础446日和压力452日。每日市场预算的主历史平均实际股票占比约17.23%，区间固定约16.51%；较早约21%左右。收益减少并非来自新增退出，而是不同的仓位分配及其账户路径。',
        '主历史每日方案最大回撤6.28%／6.55%，区间固定6.10%／6.34%；较早约6.82%至7.20%。较第181轮部分回撤更低，但年化与夏普的双门槛距离更大，不能将降低回撤写成目标达成。',
        '从原始收盘和现金股息重建六十日市场风险，独立核对区间起点、固定目标、来源未知不重置及下一开盘现金成交。核对不重新生成旧账户，不代表独立收益验证。',
        '下一轮保留第181轮完整仓位，检验原策略自身六十日净收益或净资产均线转弱时暂停、恢复时重入，两个条件固定比较，不调整已失败市场风险规则。',
        '继续使用截至186轮已保存双门槛清单，不重复遍历旧研究。目标仍未完成，所有历史均为已观察研究回放。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        **{m:'CLOSED_DIRECT_MARKET_RISK_BUDGET_RETURN_SHORTFALL_JOINT_FAILED' for m in [PRIMARY,EPISODE]}},exclusive=True)
    nxt=ROOT/'docs/510300_REFERENCE_ACCOUNT_HEALTH_GATE_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300市场风险仓位_第187轮_20260913/市场风险仓位_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'按市场波动直接确定已批准信号的仓位',
        'CLOSED_SIGNAL_MARKET_RISK_BUDGET_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'REFERENCE_ACCOUNT_HEALTH_GATE_PREPARED',
        '保留第181轮目标幅度，按来源自身近期净收益和净资产均线决定暂停与恢复')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=188,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=187
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
