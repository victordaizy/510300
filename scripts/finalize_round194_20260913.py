"""完成两档风险校准的公开比较，保留原十个百分点预算和失败记录。"""
import json

from research.finite_account_risk_calibration_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    joint=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    verified=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not any(v.values()) for v in joint['candidates'].values()),'两档校准联合结论不同')
    decision=('第194轮两档提高账户风险预算均未达到完整目标。12%预算主历史基础／压力夏普1.250／1.168，年化9.67%／8.98%；'
        '15%预算夏普1.199／1.119，年化9.48%／8.82%。较早年化虽超过10%，两档夏普仍只有约1.02至1.04。结束本次有限校准，原181仍作为较佳联合比较候选。')
    detail=[
        f"五项必要测试通过，八条新模拟账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试、核对与交付。独立重建六十日风险与两档目标，并核对{verified['actual_decisions_checked']}个决定、{verified['complete_actual_cycles']}个完整周期、{verified['simulated_fills_checked']}次模拟成交；零新模型。",
        '',
        '12%预算较早基础／压力净夏普1.043／1.037，年化10.27%／10.27%；15%预算较早夏普1.021／1.015，年化10.49%／10.51%。收益单项过线不等于联合目标通过。',
        '主历史原181有436个正目标判断，其中184个满仓；12%预算有213／214个满仓，15%预算246个。平均实际股票比例从原约18.9%升到约19.8%和20.7%至20.9%，规模提高受到满仓上限限制，不会按预算数字等比例增加全账户收益。',
        '相对181，12%预算主基础终值只增加58.05元、压力少234.39元，夏普下降约0.057／0.055。15%预算主基础／压力终值分别少4058.60／3793.67元，夏普下降约0.108／0.104。提高预算没有获得所需的收益风险改善。',
        '本轮是已观察历史后的两档参数校准，所有结果已计入研究次数；没有将其写成新因子，也没有修改旧181冻结文件或旧成绩。八条核对通过的模拟账户不构成独立未来验证。',
        '',
        '下一项检查均值反弹来源的互补性：原23辅助主历史与181日收益相关约0.136，独立表现较弱，不能据此宣布组合有效。已保存来源诊断；两个新组合预算预先固定为辅助四分之一与二分之一，另明确补算原缺失的两条较早辅助参考，共十条新路径，零外部数据。',
        '目标尚未完成，停止这次风险预算加码，继续已列明的有限组合研究。',
    ]
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        'ACCOUNT_RISK_12':'CLOSED_FINITE_CALIBRATION_NO_JOINT_PASS',
        'ACCOUNT_RISK_15':'CLOSED_FINITE_CALIBRATION_MAIN_DEGRADATION_NO_JOINT_PASS'},exclusive=True)
    nxt=ROOT/'docs/510300_MEAN_REBOUND_AUXILIARY_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300两档风险预算校准_第194轮_20260913/两档风险预算校准_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'12%和15%账户风险预算有限校准',
        'CLOSED_FINITE_ACCOUNT_RISK_CALIBRATION_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),
        nxt,nxt.read_text(encoding='utf-8').splitlines(),'MEAN_REBOUND_AUXILIARY_PREPARED',
        '181原目标加原23均值反弹的25%或50%，补齐两条已明确缺失的较早辅助参考')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(p.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=195,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=2,planned_total_new_simulated_paths=10,
        external_data_required=False,research_class='RETROSPECTIVE_FINITE_DIVERSIFICATION_COMBINATION',
        source_preflight='reports/research/510300_saved_rebound_diversification_preflight_20260913/result.json')
    index['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round']=194
    write_json(p,index)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
