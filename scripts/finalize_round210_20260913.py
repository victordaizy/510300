"""交付固定历史通过方案的稳定性诊断，接续时间边界和免费数据前提。"""
import json
import pandas as pd
from research.point_pass_fixed_diagnostic_v1 import CONFIG, MODELS, NAMES, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    intervals=pd.read_csv(OUT/'bootstrap_intervals.csv')
    pairs=pd.read_csv(OUT/'bootstrap_two_cost_threshold_fractions.csv')
    concentrated=pd.read_csv(OUT/'cycle_concentration.csv')
    annual=pd.read_csv(OUT/'yearly_metrics.csv')
    excluded=pd.read_csv(OUT/'leave_one_year_statistics.csv')
    require(result['candidate_configurations']==0 and result['new_accounts_generated']==0 and result['historical_point_target_met'],'诊断范围或历史点值状态不同')
    require(checked['bootstrap_metric_rows_checked']==72000,'重采样核对范围不同')
    simple='SELECTED_MIX_BAND10_SIMPLE2'
    decision=('第210轮三个固定候选、十二份保存账本诊断完成，历史通过数值全部复算一致。区块重采样区间明显跨过目标下方，'
        '盈利集中且部分年份影响较大，因此只能保留历史点值通过，尚不能认定稳定或独立实现夏普1.2、年化10%。固定策略，下一项核实免费后续数据与前瞻起点。')
    detail=[f"本轮零新候选、零新拟合、零新账户。三项必要统计测试通过，诊断核心耗时{result['run_seconds']:.2f}秒；独立以逐条区块索引和二阶矩重建{checked['bootstrap_metric_rows_checked']}条重采样指标，核对36组区间和18个两费用共同过线比例。",
        '三个对象分别是209事前主方案、简单两来源十个百分点方案和五来源数值首位；所有观察前身份保留。重采样不是重新选策略，不能消除之前583套不同设置及308路径筛选造成的选优偏差。',
        '简单两来源的主压力20日区块重采样，夏普中位数1.2228，2.5%至97.5%分位数0.5458至1.7773；年化中位数9.9688%，分位数2.9375%至20.9359%。区间不是已校正选优的置信保证。',
        '简单两来源在主历史三个区块长度下，两档费用同时过两项数值门槛的重采样比例为42.85%至43.55%；较早为47.15%至49.65%。这些是固定历史样本重新拼接的比例，不是未来成功概率。',
        '主压力43个完整持仓周期中，最大盈利周期占总净利润42.97%，前五盈利周期占86.83%；较早压力24周期对应29.97%和89.33%。分母保留所有亏损周期，比例可能高于100%，不是按盈利周期利润重新归一化。',
        '只作年份敏感性统计时，主压力拿掉2020年，剩余夏普1.0453、年化8.5343%；拿掉2024年，夏普1.2494但年化7.6656%。较早拿掉2015年，剩余夏普0.9596、年化6.5495%。这些拼接统计不可执行，不是删除年份后的新策略回测。',
        '', '## 固定三个方案的区块重采样区间', '',
        '|方案|区间|费用|区块日数|夏普2.5%分位|夏普中位|夏普97.5%分位|年化2.5%分位|年化中位|年化97.5%分位|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in intervals.to_dict('records'):
        detail.append(f"|{NAMES[r['model']]}|{'主历史' if r['period']=='evaluation' else '较早历史'}|{'基础' if r['cost']=='BASE' else '压力'}|{r['block_length']}|{r['sharpe_lower_2_5']:.4f}|{r['sharpe_median']:.4f}|{r['sharpe_upper_97_5']:.4f}|{r['annual_lower_2_5']:.3%}|{r['annual_median']:.3%}|{r['annual_upper_97_5']:.3%}|")
    detail+=['','## 两档费用同时越过数值门槛的比例','','每种区块2000次，使用同一抽样日期比较基础与压力费用。比例不作未来概率解释。','',
        '|方案|区间|区块日数|两费用同时越过夏普与年化门槛|','|---|---|---:|---:|']
    for r in pairs.to_dict('records'):
        detail.append(f"|{NAMES[r['model']]}|{'主历史' if r['period']=='evaluation' else '较早历史'}|{r['block_length']}|{r['two_cost_joint_threshold_fraction']:.2%}|")
    detail+=['','## 完整持仓周期盈利集中度','','|方案|区间|费用|周期数|净利润|最大盈利周期占净利润|前五盈利周期占净利润|','|---|---|---|---:|---:|---:|---:|']
    for r in concentrated.to_dict('records'):
        detail.append(f"|{NAMES[r['model']]}|{'主历史' if r['period']=='evaluation' else '较早历史'}|{'基础' if r['cost']=='BASE' else '压力'}|{r['complete_cycles']}|{r['total_net_profit']:.2f}|{r['largest_cycle_share_of_net_profit']:.2%}|{r['top_five_share_of_net_profit']:.2%}|")
    detail+=['','## 简单两来源压力费用逐年实际收益','','2026年只到8月14日开盘，列实际收益，不当作全年收益。全区间通过不等于每年通过。','',
        '|区间|年份|该年实际收益|该年夏普|该年最大回撤|交易次数|','|---|---:|---:|---:|---:|---:|']
    for r in annual[(annual.model==simple)&(annual.cost=='STRESS')].to_dict('records'):
        detail.append(f"|{'主历史' if r['period']=='evaluation' else '较早历史'}|{r['year']}|{r['cumulative_return']:.3%}|{r['net_sharpe']:.3f}|{r['max_drawdown']:.3%}|{r['trade_count']}|")
    detail+=['','## 简单两来源压力费用的逐年留出敏感性','','下列是拿掉对应年份后的剩余日收益统计，非可执行账户，不据此选年份。','',
        '|区间|去掉年份|剩余夏普|剩余复合年化|两项数值均通过|','|---|---:|---:|---:|---|']
    for r in excluded[(excluded.model==simple)&(excluded.cost=='STRESS')].to_dict('records'):
        detail.append(f"|{'主历史' if r['period']=='evaluation' else '较早历史'}|{r['excluded_year']}|{r['net_sharpe']:.4f}|{r['annualized_return']:.3%}|{'是' if r['remaining_joint_point_pass'] else '否'}|")
    detail+=['','## 下一步','','保持209已冻结设置，核实8月14日后免费行情与分红覆盖、来源连续状态和正式冻结时刻。冻结以前已发生的新读行情只能列作补充历史候选区间，不能自动称为前瞻；真正前瞻还需要事先形成、可复核的决定记录。下一项只检查数据与实施前提，不启动交易账户或服务。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,'historical_joint_point_pass_preserved':True,
        'independent_validation':'NOT_ESTABLISHED','statistical_stability':'NOT_ESTABLISHED',
        'new_candidates':0,'new_accounts':0,'candidates':{m:'HISTORICAL_POINT_PASS_PRESERVED_WIDE_INTERVALS_SELECTION_UNADJUSTED' for m in MODELS}},exclusive=True)
    nxt=ROOT/'docs/510300_POST_SELECTION_DATA_FEASIBILITY_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300达标方案稳定性诊断_第210轮_20260913/固定方案稳定性_逐年周期及重采样结果.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'固定历史达标方案的稳定性诊断',
        'COMPLETED_POINT_PASS_PRESERVED_STABILITY_AND_INDEPENDENCE_NOT_ESTABLISHED',decision,'\n'.join(detail),
        nxt,nxt.read_text(encoding='utf-8').splitlines(),'POST_SELECTION_DATA_FEASIBILITY_PREPARED',
        '不改设置，核实既有免费后续行情分红、正式冻结前后时间边界与来源状态恢复前提')
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=211,registered=False,planned_settings=0,planned_new_accounts=0,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=True,
        research_class='FIXED_STRATEGY_POST_SELECTION_FREE_DATA_AND_TIME_BOUNDARY_FEASIBILITY')
    index['status']='ROUND210_HISTORICAL_POINT_PASS_PRESERVED_STABILITY_UNCERTAIN'
    index['latest_fixed_point_pass_diagnostic']=str((OUT/'result.json').relative_to(ROOT))
    index['historical_stability_status']='NOT_ESTABLISHED_WIDE_RESAMPLED_INTERVALS_AND_PROFIT_CONCENTRATION'
    index['independent_validation']='NOT_ESTABLISHED'
    index['current_best_joint_comparison_candidate']['last_saved_diagnostic_round']=210
    index['preferred_simpler_joint_point_pass_candidate']['last_saved_diagnostic_round']=210
    write_json(path,index)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
