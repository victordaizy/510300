"""交付保持份额的完整结果，接续公开登记的有限风险预算校准。"""
import json

from research.entry_shares_preservation_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    joint=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    checked=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not any(joint['candidates'][PRIMARY].values()),'第193轮联合目标结论不同')
    decision=('第193轮保持初始实际份额降低交易和回撤，但整体收益与夏普下降。'
        '主历史基础／压力夏普1.130／1.070，年化7.84%／7.39%；'
        '较早夏普0.999／1.005，年化7.51%／7.61%。未达到联合目标，结束这套固定份额执行结构。')
    detail=[
        f"六项必要测试通过，四条新模拟账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试、核对与交付。没有新模型拟合或外部数据。核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期和{checked['simulated_fills_checked']}次模拟成交。",
        '',
        '主历史每档由第181轮115次成交降至60次，较早由63／64次降至32次。本轮四条账户均没有持仓中途追加或部分卖出，来源收盘目标逐条不变；已知正目标保持份额、零目标退出和未知状态均已核对。',
        '相对第181轮，主基础节省费用5109.94元，但股票价格及分红损益少43863.70元，终值净少38753.76元；主压力节省9276.08元，价格及分红损益少42354.80元，终值净少33078.72元。减少调仓有成本收益，也改变承担股票风险的时段与份额；本次后者损失更大。',
        '较早基础／压力终值分别少33465.34／31385.40元。最大回撤从第181轮9.68%／10.14%减至6.15%／6.45%，但两档年化均降至约7.5%。更小回撤不足以替代完整收益和夏普门槛。',
        '主历史本轮最大回撤6.72%／6.86%。较早压力收益略高于基础，来源两档账户目标路径原本不同，本轮也独立计算费用、份额和全部净值，不强行将两费用路径改为相同交易。',
        '',
        '下一步公开登记一个有限历史参数校准：在第174轮原目标和完整账户六十日波动基础上，一次比较12%与15%风险预算。它属于使用过历史后的参数选择，不称为新因子或独立验证；第181轮10%预算及原失败记录保留。新八账户的收益、费用、回撤和较早表现全部报告，不看一档结果后追加数值。',
        '本轮没有实盘交易，保存核对证明模拟符合规则，不能当成未来高夏普已验证。目标继续。',
    ]
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        PRIMARY:'CLOSED_ENTRY_SHARES_PRESERVATION_LOWER_COST_BUT_LOWER_RETURN_AND_SHARPE'},exclusive=True)
    nxt=ROOT/'docs/510300_FINITE_ACCOUNT_RISK_CALIBRATION_NEXT_20260913.md'
    document=ROOT/'deliverables/510300正目标保持份额_第193轮_20260913/正目标保持份额_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,document,'正目标期间保持初始实际份额',
        'CLOSED_ENTRY_SHARES_PRESERVATION_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),
        nxt,nxt.read_text(encoding='utf-8').splitlines(),'FINITE_ACCOUNT_RISK_CALIBRATION_PREPARED',
        '明确标记历史参数校准，一次比较12%与15%风险预算，保留原10%结果')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(p.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=194,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False,
        research_class='EXPLICIT_RETROSPECTIVE_FINITE_PARAMETER_CALIBRATION',
        authority_basis='USER_REQUEST_FLEXIBLE_FAST_STRATEGY_CHANGES_EXISTING_V6_FINITE_CANDIDATES_AUTHORIZATION')
    index['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round']=193
    write_json(p,index)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
