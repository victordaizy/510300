"""关闭来源净值暂停条件，保存时段归因及周度减仓事前规则。"""
import json
import shutil

from research.reference_account_health_gate_v1 import CONFIG,OUT,PRIMARY,MEAN,ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now,require,write_json


def main():
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    v=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(all(not x['main_two_cost_joint_pass'] and not x['earlier_two_cost_joint_pass'] for x in a['candidates'].values()),'第188轮结论不同')
    diagnostic=ROOT/'reports/research/510300_saved_session_attribution_through188'
    d=json.loads((diagnostic/'result.json').read_text(encoding='utf-8'))
    require(d['saved_accounts']==8 and d['new_accounts']==0,'保存时段归因范围不同')
    decision=('第188轮按来源自身净收益暂停恢复。六十日净收益主方案主历史基础／压力夏普1.114／0.968、复合年化6.95%／5.69%；'
        '较早夏普0.767／0.769、年化6.11%／6.05%。净资产均线比较方案四场景也未达标。保留原仓位加暂停条件仍削弱收益，关闭这两项固定健康条件。')
    detail=[f"五项必要测试通过，八条新模拟账户核心计算{r['run_seconds']:.2f}秒，不含开发、核对和交付。核对{v['actual_decisions_checked']}个决定、{v['complete_actual_cycles']}段模拟持仓和{v['simulated_fills_checked']}次模拟成交。",'',
        '|方案|历史段|费用|净夏普|复合年化|最大回撤|成交次数|','|---|---|---|---:|---:|---:|---:|']
    for period,key in [('主','all_metrics'),('较早','earlier_diagnostics')]:
        for m in r[key]:
            if m['model'] in [PRIMARY,MEAN]:
                detail.append(f"|{'净收益主方案' if m['model']==PRIMARY else '净资产均线比较方案'}|{period}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{m['max_drawdown']:.2%}|{m['trade_count']}|")
    detail += ['',
        '主历史原正目标436日，净收益条件减少到基础345日和压力341日，均线条件减少到325日和321日。较早净收益条件减少71／79个原正目标日，均线条件减少87／90日。暂停并不等于有效避损，失去的恢复机会也会影响完整账户。',
        '基础与压力费用使用各自独立来源账户，因此净值健康日期可以不同。个别压力指标略高于基础指标不表示费用能改善收益，而是交易路径不同；两种费用结果都完整保留。',
        '核对使用已保存来源收盘净值，从账户锚点计算六十日窗口并独立重建允许、暂停、恢复和未知，不读取终点开盘清仓后的净值作为收盘信号。暂停账户没有反馈改变背景来源。',
        '另读取第181、182轮八条旧账户完成日内隔夜归因，新增账户为零。第181轮主历史基础日内价格损益约19.89万元，隔夜价格加股息约-1.62万元，费用约1.41万元，合成净利润约16.86万元。较早第182轮含股息隔夜为正，不能简单把隔夜统一删除。',
        '原方案新建仓首日日内损益合计为正，本次证据不支持统一推迟买入。收盘成交必须有适用的执行机制依据；没有把股票收盘集合竞价假设或日线收盘价直接用于全部ETF历史。详细归因、金额口径和历史官方公告链接附在时段归因说明。',
        '下一轮仅检验周四收盘决定下一开盘清仓，其他日期按第181轮原目标。一个候选、四个账户，全部仍使用下一开盘成交，同时计算失去的日内收益和增加的交易费用。',
        '全部历史已观察，保存核对及损益归因均非独立收益验证。目标仍未完成。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        **{m:'CLOSED_REFERENCE_ACCOUNT_HEALTH_GATE_RETURN_AND_SHARPE_SHORTFALL' for m in [PRIMARY,MEAN]}},exclusive=True)
    nxt=ROOT/'docs/510300_THURSDAY_WEEKLY_REDUCTION_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300账户收益暂停恢复_第188轮_20260913/账户收益暂停恢复_结果及全部中文规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'按原策略自身近期净收益决定暂停与恢复',
        'CLOSED_REFERENCE_ACCOUNT_HEALTH_GATE_JOINT_TARGET_NOT_MET',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'THURSDAY_WEEKLY_REDUCTION_PREPARED',
        '周四收盘决定下一开盘清仓，其他交易日按原风险预算，完整计算放弃日内收益及新增成本')
    destination=doc.parent/'保存账户时段归因'
    destination.mkdir()
    for name in ['result.json','session_attribution.csv','日内隔夜损益说明.md','source_files.json']:
        shutil.copy2(diagnostic/name,destination/name)
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=189,registered=False,planned_settings=1,planned_new_accounts=4,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False)
    i['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    i['latest_saved_session_attribution']=str((diagnostic/'result.json').relative_to(ROOT))
    for k in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[k]['last_compared_completed_round']=188
    write_json(p,i)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
