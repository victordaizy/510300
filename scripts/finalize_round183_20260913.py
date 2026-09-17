"""交付满仓信号检验及阶段归因，转向有明确时钟的退出研究。"""
import json
import shutil

from research.binary_signal_exposure_v1 import CONFIG,OUT,PRIMARY,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))['candidates'][PRIMARY]
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not a['main_two_cost_joint_pass'] and not a['earlier_two_cost_joint_pass'],'本轮联合结论不同')
    attribution=ROOT/'reports/research/510300_signal_stage_attribution_through183'
    ar=json.loads((attribution/'result.json').read_text(encoding='utf-8'))
    require(ar['accounts']==20 and ar['all_group_profits_reconcile_to_account'],'阶段归因不完整')
    decision=('第183轮原目标严格为正时目标满仓，归零时退出。主历史基础／压力净夏普0.979／0.913、'
        '复合年化9.29%／8.61%；较早净夏普0.817／0.816、复合年化10.36%／10.39%。'
        '四个场景均没有同时满足净夏普1.2和复合年化10%，关闭满仓映射方案。')
    detail=[f"三项必要测试通过，四个新模拟账户核心计算{r['run_seconds']:.2f}秒，不含开发、测试和核对。",
        f"独立从来源目标重建{v['actual_decisions_checked']}个决定，核对{v['complete_actual_cycles']}段完整模拟持仓与{v['simulated_fills_checked']}次模拟成交；目标误差为零，资金、费用、股息和实际下一开盘核对通过。",'',
        '|历史段|费用|净夏普|复合年化|最大回撤|成交次数|平均股票仓位|',
        '|---|---|---:|---:|---:|---:|---:|']
    for period,key in [('主历史','all_metrics'),('较早历史','earlier_diagnostics')]:
        for row in r[key]:
            if row['model']==PRIMARY:
                detail.append(f"|{period}|{'基础' if row['cost']=='BASE' else '压力'}|{row['net_sharpe']:.3f}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|{row['trade_count']}|{row['mean_exposure']:.2%}|")
    detail += ['',
        '主历史每档30段实际持仓、60次成交，较早每档16段、32次成交。交易减少但单次资金使用量更大，不能把交易次数降低等同于总费用降低。',
        '主历史平均股票仓位约27%，较早约36%—37%；只有来源信号为正时才持股，目标满仓不等于全程持股。满仓增加了承担的波动，却没有实现双门槛。',
        '',
        '补充归因读取原174、两倍180、每日181、区间182和满仓183共20条既有账户，无新增账户或拟合。按收益发生前一收盘的核心独有、辅助独有、双方同时、零目标或退出过渡分组，各组金额加总与完整账户净利润一致。',
        '核心和辅助对应阶段的整段净利润均为正。满仓方案两个历史段的亏损日金额都明显扩大，回撤约10.5%和13.9%；归因不支持直接删除某一模块，也不能证明任一保护阈值有效。',
        '零目标或退出过渡包含收到退出信号后到下一开盘的旧份额损益。其他各组也受实际旧份额、成交费用和分红影响，分组不是纯粹模块增量贡献，不应据此拼接年度收益。',
        '',
        '联合门槛查看joint_target_metrics.csv和joint_target_assessment.json；旧meets_point_target只表示夏普检查。已有历史均被观察过，模拟计算核对不能建立独立收益证据。',
        '下一项固定以信号区间开始时的二十日日波动为尺度，检验明确的收盘保护退出、下一开盘执行及退出后等待原信号归零再启动。该方案尚未计算，不代表保护退出有效。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        PRIMARY:'CLOSED_NO_SCENARIO_JOINT_SHARPE12_CAGR10'},exclusive=True)
    nxt=ROOT/'docs/510300_SIGNAL_EPISODE_PROTECTIVE_EXIT_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300正信号满仓_第183轮_20260913/正信号满仓_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'明确正信号目标满仓、归零退出',
        'CLOSED_BINARY_SIGNAL_EXPOSURE_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'SIGNAL_EPISODE_PROTECTIVE_EXIT_PREPARED',
        '信号区间高点回撤触发收盘保护退出，下一开盘执行并锁定至来源归零')
    for name in ['信号阶段损益归因.md','stage_pnl.csv']:
        shutil.copy2(attribution/name,doc.parent/name)
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=184,registered=False,planned_settings=1,planned_new_accounts=4,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    i['latest_signal_stage_attribution']=str((attribution/'result.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[key]['last_compared_completed_round']=183
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
