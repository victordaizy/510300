"""第184轮保护退出失败及全部受影响区间交付。"""
import json
import pandas as pd

from research.signal_episode_protective_exit_v1 import CONFIG,OUT,PRIMARY,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))['candidates'][PRIMARY]
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not a['main_two_cost_joint_pass'] and not a['earlier_two_cost_joint_pass'],'保护退出联合结论不同')
    pairs=pd.read_csv(OUT/'protected_episode_comparisons.csv')
    require(len(pairs)==36,'受影响区间不完整')
    decision=('第184轮正信号满仓加三倍日波动的收盘回撤保护。主历史基础／压力夏普0.901／0.828、'
        '复合年化7.63%／6.96%；较早夏普0.257／0.258、复合年化2.07%／2.09%。'
        '四个场景均未达到双门槛，较早最大回撤反而扩大至19.67%／19.91%，关闭这项固定保护退出。')
    detail=[f"五项必要测试通过，四条新模拟账户核心计算{r['run_seconds']:.2f}秒。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段实际模拟持仓和{v['simulated_fills_checked']}次模拟成交；全部成交按下一开盘及实际费用。",'',
        '|历史段|费用|被保护区间|区间收益改善|区间收益恶化|', '|---|---|---:|---:|---:|']
    for (period,cost),g in pairs.groupby(['period','cost']):
        detail.append(f"|{'主' if period=='evaluation' else '较早'}|{'基础' if cost=='BASE' else '压力'}|{len(g)}|{int(g.interval_return_difference.gt(1e-10).sum())}|{int(g.interval_return_difference.lt(-1e-10).sum())}|")
    detail += ['',
        '区间对比采用两个完整账户在同一原信号区间的实际净收益，各自相对区间开始前的净资产计算，包含佣金、滑点、股息和开盘跳空。不同账户的金额规模、此前路径和费用取整不同，因此它不是纯粹的止损价收益，也不能把各段差额直接相加当作全账户复合收益差。',
        '较早首个信号起点为2014年12月31日，账户从2015年1月5日开盘开始；2015年1月19日收盘触发保护后一直等到原信号归零。在这个完整比较区间，基础账户保护版本净收益-7.32%，无保护满仓版本+30.26%，差额约-37.58个百分点。压力费用对应约-37.57个百分点。',
        '主历史也有未被保护改善的区间。例如2026年6月4日开始的区间，在6月8日收盘触发退出，基础保护账户该区间-3.10%，无保护+3.29%。全部36行区间对比均保留在protected_episode_comparisons.csv，不隐藏改善或恶化的案例。',
        '原信号的主历史30段中11段、较早16段中7段触发保护；主历史正目标日从436减少到327，较早基础从446减至238，压力从452减至244。保护削减了持仓机会，但没有形成有效净收益改善。',
        '保护距离按信号开始时固定，收盘回撤触发后下一开盘卖出；真实跳空可能使损失超过保护距离。报告中的回撤是账户净值回撤，不能用信号门槛替代实际最大回撤。',
        '下一轮检验退出后恢复到退出前高点时可重新进入，不修改二十日窗口或三倍距离；主方案使用已有第181轮资金预算，比较方案使用满仓。规则先登记，收益之后检验。',
        '旧meets_point_target仅检查夏普，当前双门槛以joint_target_assessment.json及joint_target_metrics.csv为准。实现核对不是独立收益验证，目标保持未完成。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        PRIMARY:'CLOSED_PROTECTIVE_EXIT_MISSED_RECOVERIES_JOINT_TARGET_FAILED'},exclusive=True)
    nxt=ROOT/'docs/510300_PROTECTED_RECOVERY_REENTRY_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300信号保护退出_第184轮_20260913/信号保护退出_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'信号区间高点回撤的收盘保护退出',
        'CLOSED_SIGNAL_PROTECTIVE_EXIT_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'PROTECTED_RECOVERY_REENTRY_PREPARED',
        '保护退出后价格恢复到退出前高点可重入，比较原账户风险预算和满仓')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=185,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=184
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
