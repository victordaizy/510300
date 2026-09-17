"""交付已经核对的八套历史联合达标账户，固定简单方案并接续稳定性诊断。"""
import json
from pathlib import Path
import pandas as pd
from research.incremental_selected_intent_mix_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def candidate_record(row,result,status):
    return {'round':209,'study':result['study_id'],'model':row['model'],
        'source_result':str((OUT/'result.json').relative_to(ROOT)),
        'minimum_joint_ratio':row['minimum_joint_ratio'],'historical_four_scenario_joint_point_pass':True,
        'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','status':status,'last_compared_completed_round':209,
        **{k:row[k] for k in row if k.endswith(('_net_sharpe','_annualized_return','_max_drawdown'))}}


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked=json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint=json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking=pd.read_csv(OUT/'thirteen_setting_joint_comparison.csv')
    require(all(r['four_scenario_joint_pass'] for r in joint['candidates'].values()),'八套未全部历史通过')
    require(checked['actual_accounts']==32 and checked['actual_decisions_checked']==45168 and checked['suppressed_buy_requests_checked']==0,'核对范围或重复过滤不同')
    best=ranking.iloc[0].to_dict()
    simple=ranking[ranking.model.eq('SELECTED_MIX_BAND10_SIMPLE2')].iloc[0].to_dict()
    require(best['model']=='SELECTED_MIX_BAND00_FULL','最弱门槛首位不同')
    decision=('第209轮八套组合、三十二条完整账户已完成并核对。全部八套在主历史和较早历史、基础和压力费用四场景均达到净夏普至少1.2、复合年化至少10%。'
        '历史回测目标已有可核查的通过结果；独立验证尚未建立，完整持续目标保持进行中，接下来固定设置检查稳定性。')
    detail=[f"六项必要测试通过，核心计算{result['run_seconds']:.2f}秒；核对{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期和{checked['simulated_fills_checked']}次模拟成交。五个来源的收盘计划从其原账本独立重建，新账户按自己的净资产和份额计算申请，未重复施加原加仓过滤。",
        '事前主方案为零门槛、70%／15%／15%三来源，其四场景均通过；没有在计算后更改主方案。数值首位为五来源零门槛，外层较简单的方案为85%／15%两来源十个百分点门槛，两者是八套结果中的不同事后选择，身份分别记录。',
        '原205主压力夏普1.1862、年化9.8357%；新两来源十个百分点方案为1.2164、10.3233%，最大回撤由7.1576%降到6.5086%。该方案主历史基础夏普1.2812、年化10.9281%；较早基础1.2287、10.8809%；较早压力1.2395、11.0310%。',
        '两来源十个百分点方案在主历史基础和压力均119次模拟成交，较早两档均66次；对应零门槛两来源主压力200次、较早压力124次。选择十个百分点便于减少调整，仍须保留两方案全部结果，不能把它称为所有指标最优。',
        '五来源零门槛的四场景最弱门槛比值最高，为1.019477；主压力夏普1.2234、年化10.2054%。两来源十个百分点比值1.013640，最弱指标同样高于门槛。该比值不是胜率，也不能表示未来收益保证。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|']
    for row in ranking.to_dict('records'):
        cells=[f"{row[t+'_net_sharpe']:.4f}／{row[t+'_annualized_return']:.4%}" for t in ['main_base','main_stress','earlier_base','earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.6f}|")
    detail+=['','## 较简单两来源方案的逐年实际收益','','不完整年度只列该段实际收益，不把2026年截至8月14日的收益当作完整年度收益。全区间达标不代表每个自然年都达到相同夏普。',
        '|历史区间|费用|年份|该年实际收益|该年夏普|该年最大回撤|交易次数|','|---|---|---:|---:|---:|---:|---:|']
    annual=pd.read_csv(OUT/'yearly_metrics.csv')
    for row in annual[annual.model.eq(simple['model'])].sort_values(['period','cost','year']).to_dict('records'):
        detail.append(f"|{'主历史' if row['period']=='evaluation' else '较早历史'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['year']}|{row['cumulative_return']:.3%}|{row['net_sharpe']:.3f}|{row['max_drawdown']:.3%}|{row['trade_count']}|")
    detail+=['','## 保持设置，继续验证','','截至本轮历史已被多次选择使用；209前的增量筛选涉及308条路径和65次目标函数评价，本轮八套也全部计数。账本核对证明计算与规则对应，不能消除选优偏差或证明未来仍有夏普1.2。',
        '下一项固定事前主方案、简单两来源和五来源首位，读取已保存账本做逐年、完整周期集中度、逐年留出敏感性与固定区块重采样诊断；不新增候选、不再调整权重、不生成交易账户，也不把重采样当独立验证。当前仅准备规则。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'goal_achieved':False,
        'historical_four_scenario_joint_point_pass':True,'independent_validation':'NOT_ESTABLISHED',
        'primary_preserved':'SELECTED_MIX_BAND00_SIMPLE3','post_selected_joint_best':best['model'],
        'preferred_simpler_candidate':simple['model'],
        'candidates':{m:'FOUR_SCENARIO_HISTORICAL_JOINT_POINT_PASS_INDEPENDENT_NOT_ESTABLISHED' for m in CANDIDATES}},exclusive=True)
    nxt=ROOT/'docs/510300_POINT_PASS_FIXED_DIAGNOSTIC_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300历史目标通过_第209轮_20260913/八套达标组合_全部结果与中文进出场规则.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'八套实际组合通过历史夏普与年化目标',
        'COMPLETED_EIGHT_ACTUAL_COMBINATIONS_HISTORICAL_POINT_PASS_INDEPENDENT_NOT_ESTABLISHED',decision,'\n'.join(detail),
        nxt,nxt.read_text(encoding='utf-8').splitlines(),'POINT_PASS_FIXED_DIAGNOSTIC_PREPARED',
        '固定三个已通过历史点值的候选，检查年份与周期集中、固定区块重采样及独立证据边界')
    short=doc.parent/'两来源85比15方案_中文执行说明.md'
    lines=['# 两来源85比15：历史回测目标已通过','',
        '本说明对应第209轮SELECTED_MIX_BAND10_SIMPLE2。四场景均通过净夏普至少1.2、复合年化至少10%，完整账本和买卖申请已经核对；这是反复研究过的历史结果，独立验证尚未建立。','',
        '## 老板先看结果','', '|区间|费用|净夏普|年化收益|最大回撤|模拟成交|','|---|---|---:|---:|---:|---:|']
    for period,key in [('主历史2020—2026年','all_metrics'),('较早2015—2019年','earlier_diagnostics')]:
        for row in result[key]:
            if row['model']==simple['model']:
                lines.append(f"|{period}|{'基础' if row['cost']=='BASE' else '压力'}|{row['net_sharpe']:.4f}|{row['annualized_return']:.4%}|{abs(row['max_drawdown']):.4%}|{row['trade_count']}|")
    lines+=['','## 策略怎样形成仓位','',
        '外层只合并两个来源：85%权重给“原三来源合并、目标1.15倍、强势时才允许持仓加仓”的参考计划；15%权重给“趋势波动连续预算与收益强弱连续段相加封顶”的参考计划。85%和15%是两个计划的权重，并不等于始终买入85%或15%的股票。两个来源内部包含多项因素，不能把本方案称为只有两个因子。',
        '每个来源先用当日收盘实际份额加已提出买卖申请，算出计划份额，再按该来源自己的净资产换成仓位比例。合成股票目标等于第一个比例乘85%加第二个比例乘15%，剩余为现金，最高100%，不借款。来源原加仓过滤、持仓等待和退出条件已包含在计划内，不在外层重复加一次。',
        '## 什么时候进入、加仓、减仓和退出','',
        '空仓时：两个来源都给出已知计划，且合成目标为正，就按合成目标和本账户资金计算买入份额，向下取整百份；下一交易日开盘尝试买入。取整为零或资金不足则按原执行约束处理。',
        '已有持仓时：实际股票市值占账户净资产的比例，与合成目标相差不到十个百分点，保持当前份额；达到十个百分点，就按目标中心重新计算整百份。目标份额高于实有份额则申请加仓，低于实有份额则申请减仓。',
        '完全退出：合成目标明确为零，申请下一开盘卖完；固定研究终点开盘也全部清算。任何有权重的来源计划未知时保持实际份额，不把缺失当清仓。受阻申请在下个收盘按最新信息重算，不把未成交当成交。卖完后重新出现已知正目标，按同样的进入规则处理。',
        '所有决定使用当日收盘资料，在十五时零五分形成，下一真实开盘执行。新账户依据自己的现金、份额和净资产成交，只对自己的交易扣费。',
        '## 数据和费用口径','',
        '主历史2020年1月2日至2026年8月14日终点开盘；较早历史2015年1月5日至2019年12月31日终点开盘。每账户二十万元，二百四十二日年化，现金和无风险收益按零。基础佣金万分之二、最低五元、滑点万分之五；压力佣金万分之四、最低五元、滑点千分之一。整百份、千分之一元价位、次日可卖、方向涨跌停，分红登记、除息和付款分别记账。',
        '## 解释边界','',
        '历史点值已经通过，不能据此保证未来稳定达到1.2夏普或10%年化。现在固定该设置，继续检查年份依赖与统计稳定性；没有因此连接券商或下单。',
        '两来源内部全部因素、窗口、训练时点与原进出场规则，以及八套完整比较和逐年结果，见同目录“八套达标组合_全部结果与中文进出场规则.md”。']
    short.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    index['previous_best_joint_comparison_candidate_before209']=index['current_best_joint_comparison_candidate']
    index['current_best_joint_comparison_candidate']=candidate_record(best,result,'POST_SELECTED_FIVE_SOURCE_FULL_HISTORICAL_JOINT_POINT_PASS')
    index['preferred_simpler_joint_point_pass_candidate']=candidate_record(simple,result,'POST_SELECTED_TWO_SOURCE_BAND10_HISTORICAL_JOINT_POINT_PASS')
    index['historical_joint_point_target_met']=True
    index['historical_point_pass_round']=209
    index['historical_point_pass_candidates']=list(CANDIDATES)
    index['independent_validation']='NOT_ESTABLISHED'
    index['status']='ROUND209_HISTORICAL_JOINT_POINT_PASS_CONTINUE_INDEPENDENT_VALIDATION'
    index['next_work'].update(candidate_round=210,registered=False,planned_settings=0,planned_new_accounts=0,
        planned_new_model_fits=0,planned_new_reference_accounts=0,external_data_required=False,
        research_class='FIXED_POINT_PASS_CANDIDATES_SAVED_STABILITY_DIAGNOSTIC',
        fixed_existing_candidates=['SELECTED_MIX_BAND00_SIMPLE3',simple['model'],best['model']])
    index['latest_joint_target_assessment']=str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_actual_selected_intent_mix_comparison']=str((OUT/'thirteen_setting_joint_comparison.csv').relative_to(ROOT))
    index['latest_incremental_virtual_point_pass'].update(status='ACTUAL_COMBINATIONS_TESTED_ROUND209_ALL_EIGHT_POINT_PASS',actual_account_round=209)
    index['latest_simpler_strategy_chinese_explanation']=str(short.relative_to(ROOT))
    for key in ['latest_main_return_improvement_candidate','current_best_four_scenario_comparison_candidate',
                'latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate','latest_secondary_return_margin_candidate']:
        index[key]['last_compared_completed_round']=209
    write_json(path,index)
    print(json.dumps({**delivered,'简单方案说明':str(short),'历史四场景通过':True,'独立验证':'尚未建立'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
