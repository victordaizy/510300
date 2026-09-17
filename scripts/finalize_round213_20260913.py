"""交付连续账户实际结果和退出过程，转入一次月首模型更新及增量续接。"""
import json
import shutil

import pandas as pd

from research.fast_round_delivery_v1 import deliver_round
from research.post_selection_continuous_replay_v1 import ROOT,OUT,CONFIG,PRIMARY,read
from research.intraday_overnight_increment_v1 import now,write_json,require


def main():
    result=read(OUT/'result.json');verified=read(OUT/'saved_verification_receipt.json')
    require(verified['accounts']==22 and result['continuous_cutoff']=='2026-08-31','连续账户结果范围不同')
    decision=('固定85%与15%方案已按正常进出场连续到8月31日。基础净夏普1.2884、年化10.97%；压力净夏普1.2239、年化10.37%。'
        '22条必要账户的旧正常区间全部一致；原终点清仓没有进入新正常路径。下一项只补九月原定月首模型并增量接到9月11日。')
    detail=['上面主历史表为本轮完整连续账户：2020年1月2日至2026年8月31日收盘，共1615个真实交易日，以最后完整收盘计价，不人工终点清仓。较早历史表仍复用209原记录，本轮没有重跑较早历史。',
        '', '## 新增延续段实际结果', '',
        '延续段以2026年8月13日收盘净值为起点，正常处理8月14日至8月31日的12个交易日。212新增20日行情中的8月17日至31日有11日，另外本轮恢复了旧终点8月14日的正常收盘。',
        '', '|费用|起点净值|末日净值|净利润|区间净收益|实际成交次数|末日持仓|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['extension_segment_results']:
        detail.append(f"|{'基础' if r['cost']=='BASE' else '压力'}|{r['start_equity']:,.2f}元|{r['end_equity']:,.2f}元|{r['net_profit_cny']:,.2f}元|{r['net_return']:.4%}|{r['trades']}|{r['ending_shares']}份|")
    detail+=['','区间收益已经包含当期实际佣金与滑点，未另行年化。两档费用的旧起点资产不同，源自此前已经发生的费用差异；没有重新投入二十万元或使用终点清仓现金重置。',
        '', '## 压力费用账户实际怎样退出', '',
        '|时点|判断或执行|实际持仓|','|---|---|---:|',
        '|8月13日收盘|原目标约76.23%，未达到新增调仓需求，下一开盘申请为零|61800份|',
        '|8月14日开盘及收盘|按正常申请保持，恢复当日完整收盘；未采用原研究终点卖出|61800份|',
        '|8月17日收盘|来源合成目标降到约5.42%，请求下一开盘减仓57500份|61800份|',
        '|8月18日开盘|按原费用和份额规则卖出57500份|4300份|',
        '|8月20日收盘|合成目标明确归零，请求下一开盘卖出剩余4300份|4300份|',
        '|8月21日开盘|全部退出成交|0份|',
        '|8月31日收盘|目标仍为零，下一交易日9月1日无买卖申请|0份|',
        '', '这是固定规则自然生成的退出路径，没有事后指定卖出日。十个百分点门槛用于普通正目标调仓，明确零目标退出不受该门槛阻拦。',
        '', '## 进出场及因子保持哪些定义', '',
        '主方案按206来源的计划持有比例占85%、168来源占15%合成。计划持有比例取来源当日实际份额加上已提出的下一开盘申请，再按当日收盘价格和自身净值换算；因此首次归零后仍等待确认的持仓不会被当成空仓。',
        '主方案仅用原十个百分点门槛调仓，明确零全部退出；下单规模按原二十万元起点衍生的实际净值、整百份、T+1与下一开盘价格规则计算。206内部1%加仓条件保留在来源内，主方案不重复叠加。原趋势、收益连续段、相邻收益相关、涨跌日数、普通及下行波动、实际账户风险预算、均值反弹和持仓退出模型均未增加参数。',
        '全部内部因子、门槛、进入和退出规则见本目录复制的“固定两来源方案_完整中文执行说明.md”及“必要来源因素及规则.md”。规则用中文说明，程序及配置仅作为复算依据。',
        '', '## 验证及提速结果', '',
        '三类账户连续引擎和目标适配共10项必要测试通过。保存22条账户、38742条账户日记录和38764条决定；每条旧正常区间的净值、份额、成交和申请核对一致，所有保存账户财富恒等式通过。',
        '另外选取真实学习退出压力来源在8月13日仍持有64200份的状态，保存为JSON后恢复。恢复后的账户、决定和最终控制器状态与一次连续运行逐值精确相同。无成交价格空字段在拼接时曾出现对象列与浮点列的类型差异，规范化为空浮点值后核对通过；没有修改任何交易或收益。原中止与接续记录均保留。',
        '本轮零新策略参数、零新拟合、零网络请求。22条必要账户只进行一次完整批量重放；真实断点验证额外运行四段，其中两段属于先前类型检查中止。当前报告耗时字段包含该次中止与接续开发，不把它冒充纯策略计算速度。',
        '整条依赖图的增量恢复尚未验完：本轮证明原区间一致、三类引擎基本恢复和一条真实周期来源恢复；下一项实际从全部8月31日状态只接九月新增日期，补齐整图连续性证据。',
        '', '## 下一项与结论边界', '',
        '直接按31和114的原规则，在9月1日只补两条必要月首模型，保留原八因素、最近二十个已成熟周期及原十周期／一百行门槛。先补一条原训练基准参考的自然结束周期，再从已保存的22条账户状态增量接到9月11日；不重跑此前全部研究或重复整批22条旧账户。',
        '截至8月31日的历史连续账户仍达到夏普1.2与年化10%。这段补充历史发生在方案正式冻结之前，严格前瞻证据仍为零，不能据此宣称未来稳定达标。目标继续保持进行中。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'status':'FIXED_STRATEGY_CONTINUOUS_HISTORICAL_POINT_PASS',
        'new_candidates':0,'new_model_fits':0,'continuous_two_cost_point_pass':all(r['net_sharpe']>=1.2 and r['annualized_return']>=.1 for r in result['all_metrics']),
        'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','next':'原定九月月首更新和整图增量接续'},exclusive=True)
    nxt=ROOT/'docs/510300_SEPTEMBER_MONTHLY_CONTINUATION_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300连续续算实际结果_第213轮_20260913/固定策略至8月31日_历史结果与自然退出说明.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'固定策略连续至八月末的实际结果',
        'COMPLETED_22_CONTINUOUS_ACCOUNTS_HISTORICAL_POINT_PASS',decision,'\n'.join(detail),nxt,nxt.read_text(encoding='utf-8').splitlines(),
        'SEPTEMBER_ORIGINAL_MONTHLY_UPDATE_AND_DELTA_CONTINUATION_PREPARED','只补9月1日原定两条月度模型，再从22条八月末账户状态增量接至9月11日')
    content=doc.read_text(encoding='utf-8').replace('## 主历史：2020年1月2日至2026年8月14日开盘','## 主历史连续账户：2020年1月2日至2026年8月31日收盘')
    doc.write_text(content,encoding='utf-8')
    shutil.copy2(ROOT/'deliverables/510300历史目标通过_第209轮_20260913/两来源85比15方案_中文执行说明.md',doc.parent/'固定两来源方案_完整中文执行说明.md')
    shutil.copy2(ROOT/'deliverables/510300策略续算提速_第212轮_20260913/补充因素及最小计算范围_中文说明.md',doc.parent/'必要来源因素及规则.md')
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json';index=read(path)
    index['latest_completed_round']['configuration']=str(CONFIG.relative_to(ROOT));index['completed_rounds'][-1]['configuration']=str(CONFIG.relative_to(ROOT))
    index['latest_completed_round']['metric_period_note']='主历史连续至2026-08-31收盘；不同于旧2026-08-14终点开盘口径，不覆盖209四情景联合比较'
    index['completed_rounds'][-1]['metric_period_note']=index['latest_completed_round']['metric_period_note']
    index['next_work'].update(candidate_round=214,registered=False,planned_settings=0,planned_new_accounts=23,
        planned_new_model_fits=2,planned_new_reference_accounts=0,planned_existing_training_reference_replays=1,
        planned_existing_account_incremental_continuations=22,external_data_required=False,
        research_class='FIXED_ORIGINAL_MONTHLY_FITS_AND_FULL_GRAPH_INCREMENTAL_CONTINUATION',continuation_cutoff='2026-09-11')
    index['status']='ROUND213_CONTINUOUS_HISTORICAL_POINT_PASS_TO_AUGUST31_MONTHLY_DELTA_NEXT'
    index['latest_continuous_fixed_strategy_result']=str((OUT/'result.json').relative_to(ROOT))
    index['latest_continuous_account_cutoff']='2026-08-31'
    index['supplemental_data_accepted_into_strategy']=True
    index['supplemental_strategy_used_cutoff']='2026-08-31'
    index['full_pipeline_incremental_resume_verified']=False
    index['strict_forward_evidence_days']=0;index['independent_validation']='NOT_ESTABLISHED'
    write_json(path,index)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
