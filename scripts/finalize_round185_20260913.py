"""交付第185轮已核对结果，登记空仓反弹补充研究。"""
import json

from research.protected_recovery_reentry_v1 import CONFIG,OUT,PRIMARY,FULL,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not x['main_two_cost_joint_pass'] and not x['earlier_two_cost_joint_pass'] for x in a['candidates'].values()),'第185轮双门槛结论不同')
    decision=('第185轮固定保护后恢复到退出前高点可重入。原账户风险预算主方案，主历史基础／压力夏普1.233／1.146、复合年化8.63%／7.97%；'
        '较早夏普0.868／0.848、年化6.95%／6.84%。满仓比较方案更弱。两个方案四个场景均未同时达到双门槛，关闭这两项附加保护与恢复重入组合。')
    detail=[f"七项必要测试通过，八条新模拟账户核心计算{r['run_seconds']:.2f}秒。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段模拟持仓和{v['simulated_fills_checked']}次模拟成交；来源时点、仓位、开盘价格和实际费用一致。",'',
        '|方案|历史段|费用|净夏普|复合年化|最大回撤|', '|---|---|---|---:|---:|---:|']
    for period,key in [('主','all_metrics'),('较早','earlier_diagnostics')]:
        for m in r[key]:
            if m['model'] in [PRIMARY,FULL]:
                detail.append(f"|{'原预算主方案' if m['model']==PRIMARY else '满仓比较方案'}|{period}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{m['max_drawdown']:.2%}|")
    detail += ['',
        '主历史每种费用有11次保护退出、2次高点恢复重入；较早历史有8次保护退出、4次恢复重入。恢复机制挽回部分被第184轮永久退出丢失的收益，但无法恢复原策略完整的机会。',
        '主方案主历史基础／压力年化8.63%／7.97%，低于未加保护的第181轮9.66%／8.99%；较早年化6.95%／6.84%也低于第181轮9.88%／9.84%。满仓恢复方案同样弱于第183轮无保护满仓的对应历史收益。',
        '复核独立重建了固定保护距离、观察高点、冻结恢复高点、暂停和多次重入状态，没有使用未来价格，也没有用保护线替代下一开盘真实模拟成交价格。实现一致性核对不是独立收益验证。',
        '下一轮将不再调保护阈值。直接使用第一轮已保存的两日超跌及趋势内超跌两种状态，只在第181轮原策略明确空仓时补充机会；两个固定候选、八个账户，零次训练和外部数据等待。',
        '全部旧历史均属于研究回放。双门槛以joint_target_assessment.json和joint_target_metrics.csv为准，旧meets_point_target字段仅检查夏普。目标保持未完成。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        **{m:'CLOSED_RECOVERY_PROTECTION_WEAKER_THAN_UNPROTECTED_PARENT_JOINT_FAILED' for m in [PRIMARY,FULL]}},exclusive=True)
    nxt=ROOT/'docs/510300_IDLE_REVERSAL_OPPORTUNITY_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300保护恢复重入_第185轮_20260913/保护恢复重入_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'固定保护后高点恢复重入，比较原预算与满仓',
        'CLOSED_PROTECTED_RECOVERY_REENTRY_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'IDLE_REVERSAL_OPPORTUNITY_PREPARED',
        '原策略空仓时使用已保存两日超跌反弹，固定比较有无长期趋势限制')
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=186,registered=False,planned_settings=2,planned_new_accounts=8,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=185
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
