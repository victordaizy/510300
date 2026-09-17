"""第182轮固定区间风险倍率结果交付。"""
import json

from research.episode_account_risk_budget_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))['candidates'][PRIMARY]
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not a['main_two_cost_joint_pass'] and not a['earlier_two_cost_joint_pass'],'联合目标结论不同')
    decision=('第182轮在每个连续正目标区间开始时固定风险倍率。主历史基础／压力净夏普1.255／1.186、'
              '复合年化8.56%／8.05%；较早历史净夏普1.160／1.151、复合年化9.48%／9.48%。'
              '四场景均未同时达到净夏普1.2和复合年化10%，关闭本固定结构。')
    detail=[f"四项必要测试通过。四个新模拟账户核心计算{r['run_seconds']:.2f}秒，不含开发、测试及核对。",
            f"独立从原账户收益重算六十日风险，以零目标划分区间重建固定倍率，核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段完整模拟持仓、{v['simulated_fills_checked']}次模拟成交。",'',
            '|历史段|费用|净夏普|复合年化|最大回撤|成交次数|', '|---|---|---:|---:|---:|---:|']
    for period,key in [('主历史','all_metrics'),('较早历史','earlier_diagnostics')]:
        for row in r[key]:
            if row['model']==PRIMARY:
                detail.append(f"|{period}|{'基础' if row['cost']=='BASE' else '压力'}|{row['net_sharpe']:.3f}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|{row['trade_count']}|")
    detail += ['', '主历史交易由每日预算的115次减至89次，但成本节约没有抵消价格与分红收益减少。固定区间倍率因此不能作为有效收益改进。',
               '较早历史压力费用夏普升至1.151、最大回撤降至6.45%，优于每日预算对应1.056和10.14%；但年化降至9.48%。这是取舍，不是所有要求都改善。',
               '区间来自原目标的零与正转换。主历史每档30段，较早每档16段；未知不终止区间，不能按事后盈利周期选择倍率。',
               '联合门槛以joint_target_assessment.json及joint_target_metrics.csv为准，旧meets_point_target只检查夏普。实现核对不代表独立收益验证，已研究的历史不能重新作为独立样本。',
               '下一项检查正目标直接满仓、零目标退出，不增加风险预算或修改窗口。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,PRIMARY:'CLOSED_NO_SCENARIO_JOINT_SHARPE12_CAGR10'},exclusive=True)
    nxt=ROOT/'docs/510300_BINARY_SIGNAL_EXPOSURE_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300区间固定风险预算_第182轮_20260913/区间固定风险预算_结果及全部中文规则.md'
    result=deliver_round(ROOT,OUT,CONFIG,doc,'连续正目标区间固定账户风险倍率',
        'CLOSED_EPISODE_ACCOUNT_RISK_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'BINARY_SIGNAL_EXPOSURE_PREPARED','有明确正信号时目标满仓，零信号退出')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=183,registered=False,planned_settings=1,planned_new_accounts=4,
                          planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[key]['last_compared_completed_round']=182
    i['latest_four_scenario_sharpe_budget_comparison']={'round':182,'model':PRIMARY,
        'status':'MINIMUM_FOUR_SHARPE_1_151_ALL_FOUR_CAGR_BELOW10_NOT_INDEPENDENT',
        'source_result':str((OUT/'result.json').relative_to(ROOT)),'goal_achieved':False}
    write_json(p,i)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
