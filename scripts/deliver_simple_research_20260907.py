"""汇总四轮已保存的简单策略结果，交付中文说明并更新持续研究索引。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from research.intraday_overnight_increment_v1 import now,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/510300简单策略快速研究_20260907'
STUDIES=[(23,'510300_simple_price_entry_exit_v1','简单趋势、反弹及状态切换'),
         (24,'510300_simple_regime_local_v1','切换策略局部改动及季度选择'),
         (25,'510300_simple_session_divergence_v1','日内与隔夜涨跌分歧'),
         (26,'510300_simple_volume_reversal_v1','量价条件与急跌回升')]

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def pct(value):return '未定义' if pd.isna(value) else f'{value:.2%}'

def dec(value):return '未定义' if pd.isna(value) else f'{value:.3f}'

def local_name(item):
    regime={'FAST':'快速趋势','SLOW':'慢速趋势'}[item['regime']]
    exit_rule={'QUICK':'快速退出','PATIENT':'耐心退出'}[item['exit_mode']]
    gate={'ALL':'反弹不另限','DRAWDOWN':'反弹限回撤','VOLATILITY':'反弹限波动'}[item['rebound_gate']]
    return f'{regime}／{exit_rule}／{gate}／偏离{item["z"]:g}'

def names_for(folder,cfg):
    if 'candidate_names' in cfg:return {**cfg['candidate_names'],'BUY_HOLD':'买入持有'}
    if folder.endswith('local_v1'):
        return {**{item['id']:local_name(item) for item in cfg['candidates']},'A1_PAST756':'每季度按过去756日选一种','A2_PAST504':'每季度按过去504日选一种','BUY_HOLD':'买入持有'}
    return {**{item['id']:item['name'] for item in cfg['candidates']},'BUY_HOLD':'买入持有'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=[];all_metrics=[];all_yearly=[];all_eras=[];all_cycles=[];loaded=[]
    for number,folder,title in STUDIES:
        p=ROOT/'reports/research'/folder
        result=read(str((p/'result.json').relative_to(ROOT)))
        cfg=read('config/'+folder+'.json');names=names_for(folder,cfg)
        metric=pd.DataFrame(result['all_metrics']);metric['策略']=metric.model.map(names);metric['研究轮次']=number
        all_metrics.append(metric)
        for filename,dest in [('yearly_metrics.csv',all_yearly),('era_metrics.csv',all_eras)]:
            frame=pd.read_csv(p/filename);frame['策略']=frame.model.map(names);frame['研究轮次']=number;dest.append(frame)
        best=result['post_selected_best_base'];stress=next(m for m in result['all_metrics'] if m['cost']=='STRESS' and m['model']==best['model'])
        summary.append({'轮次':number,'研究':title,'本轮最高的策略':names[best['model']],'基础夏普':best['net_sharpe'],'压力夏普':stress['net_sharpe'],
                        '基础年化收益':best['annualized_return'],'基础最大回撤':best['max_drawdown'],'基础交易次数':best['trade_count']})
        for cost in ['BASE','STRESS']:
            for key in names:
                cp=p/'evaluation'/cost/f'{key}_cycles.csv'
                if cp.exists():
                    cycles=pd.read_csv(cp);cycles['策略']=names[key];cycles['研究轮次']=number;cycles['费用']=cost;all_cycles.append(cycles)
        loaded.append((number,p,cfg,result,names))
        outcome={'recorded_at':now(),'status':'COMPLETED_TARGET_NOT_MET','historical_point_target_met':result['historical_point_target_met'],
                 'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY','new_gpt_review_archive_created':False,'position_impact':0}
        if number==26:outcome['earlier_history_counterevidence']='reports/research/510300_simple_panic_earlier_history_v1/result.json'
        write_json(p/'acceptance_outcome.json',outcome)
    m=pd.concat(all_metrics,ignore_index=True);y=pd.concat(all_yearly,ignore_index=True);e=pd.concat(all_eras,ignore_index=True)
    m.to_csv(OUT/'全部策略完整账户.csv',index=False,encoding='utf-8-sig')
    y.to_csv(OUT/'全部策略逐年表现.csv',index=False,encoding='utf-8-sig')
    e.to_csv(OUT/'全部策略三个阶段.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_cycles,ignore_index=True).to_csv(OUT/'全部实际买入卖出周期.csv',index=False,encoding='utf-8-sig')
    earlier=read('reports/research/510300_simple_panic_earlier_history_v1/result.json')
    pd.DataFrame(earlier['metrics']).to_csv(OUT/'恐慌回升策略更早五年对照.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'四轮最快结果概览.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(14,10),dpi=170)
    for ax,(number,p,cfg,result,names) in zip(axes.ravel(),loaded):
        best=result['post_selected_best_base'];key=best['model']
        ledger=pd.read_parquet(p/'evaluation/BASE'/f'{key}_ledger.parquet')
        bh=pd.read_parquet(ROOT/'reports/research/510300_simple_price_entry_exit_v1/evaluation/BASE/BUY_HOLD_ledger.parquet')
        ax.plot(ledger.date,ledger.equity/200000,color='#006F87',lw=1.8,label='本轮历史最高策略')
        ax.plot(bh.date,bh.equity/200000,color='#A7ADB3',lw=1,label='买入持有')
        title=f'第{number}轮 · 最高基础夏普 {best["net_sharpe"]:.3f}'
        if number==26:title+='（仅3次买卖周期）'
        ax.set_title(title,fontsize=12,pad=12)
        ax.set_ylabel('20万元起点归一化净值')
        ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('简单策略四轮历史筛选 · 同一完整账户与基础费用',fontsize=17,y=.98)
    fig.text(.5,.014,'2020-01-02—2026-08-14 开盘终点；各轮最高均为事后筛选，尚未达到稳定夏普1.2。',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'四轮简单策略完整净值.png');plt.close(fig)

    lines=['# 510300：放弃补齐EPS后的简单策略快速检验','',f'结果整理时间：{now()}。',
           '', '已经停止继续补齐前瞻每股收益、研报日期和股数口径，优先使用已有行情直接改进策略。新完成四轮，合计59个新策略或参数设置；另有1个原策略重复对照。每个策略都有进入、退出、重新进入规则，规则均用中文列在后文。',
           '', '**目前仍未实现稳定夏普1.2。** 这四轮最高的完整时期基础夏普为1.118、压力费用下1.093，来自“短期高波动急跌后回升”，但只有3次买卖周期；原参数在更早2015至2019年基础夏普为负0.107，构成明显反证。不能把这一历史点估计当成已找到稳定策略。',
           '', '所有主比较统一为2020年1月2日至2026年8月14日开盘终点、20万元、510300与现金、含分红、真实份额及费用的连续账户。基础费用为单边佣金万分之二、最低5元、滑点万分之五；压力费用为单边佣金万分之四、最低5元、滑点千分之一。现金与无风险收益为零，年化242日。股票ETF买入后次日才能卖出，参见[上交所说明](https://big5.sse.com.cn/site/cht/www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)。',
           '', '## 四轮结果', '', '|研究轮次|本轮历史最高策略|基础夏普|压力夏普|基础年化收益|基础最大回撤|实际成交次数|', '|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f'|第{s["轮次"]}轮|{s["本轮最高的策略"]}|{dec(s["基础夏普"])}|{dec(s["压力夏普"])}|{pct(s["基础年化收益"])}|{pct(s["基础最大回撤"])}|{s["基础交易次数"]}|')
    lines.extend(['','买入持有在同一时期的基础夏普0.288、年化收益3.63%、最大回撤40.95%。最高夏普的恐慌回升策略年化收益只有2.38%，没有高于买入持有；夏普上升不能替代稳定超额收益的要求。',
                  '', '![四轮完整净值](四轮简单策略完整净值.png)', '', '## 本次决定和原因', '',
                  '前瞻每股收益来源修正后的六种方法，最好基础夏普0.480；各自成熟预测的均方误差均高于当时训练均值基线。继续补更多研报尚未显示收益，因此停止来源补齐。保留已有结果，只有出现可以用现成数据快速检验的明确新假设才重新考虑该因子。', '',
                  '普通均线、简单反弹并非加上止损就能达标。趋势突破加震荡反弹的基础夏普0.481，局部改变趋势状态、退出速度和反弹限制后，最好只提高到0.512。按过去756日或504日夏普每季度选一种，基础夏普分别为负0.245、负0.094，暂不继续这两个追逐近期表现的选择器。', '',
                  '换用日内与隔夜涨跌分歧后，最好基础夏普0.465，尚不足以保留为独立主线。量价与急跌收复找到更高点估计，但高波动反弹只在少数近年交易中成功，换到更早历史立即失败。下一步继续新的简单交易机制，优先检验同样入场信号下的预定限价等待、订单到期与退出时钟；若日线不能充分判断成交，必须保留条件成交和保守假设，不能把触及最低价直接当成必然成交。', '',
                  '## 最高历史点估计来自哪三次交易', '', '|买入日期|卖出日期|实际持有交易间隔|完整周期净利润|退出原因|', '|---|---|---:|---:|---|'])
    cycles=pd.read_csv(ROOT/'reports/research/510300_simple_volume_reversal_v1/evaluation/BASE/V6_PANIC_RECOVERY_cycles.csv')
    for r in cycles.itertuples():
        lines.append(f'|{r.entry_date[:10]}|{r.exit_date[:10]}|{r.holding_intervals}|{r.net_profit_cny:,.2f}元|{r.exit_reasons}|')
    lines.extend(['','2022至2023年这一策略全程现金，收益为零、夏普因波动为零而未定义。不能写成该阶段夏普为零，也不能将其描述为该阶段稳定盈利。更早2015至2019年发生8次完整买卖周期，基础年化收益负1.02%、最大回撤14.41%，说明近年3次盈利不足以代表普遍有效。', '', '## 全部策略的历史表现', '',
                  '以下两种费用逐项配对。成交次数包含买入和卖出，不能直接当作完整持仓周期次数。第一轮的切换主方案在第二轮作为原样对照重复出现；没有把它算成两个新策略。'])
    for number,p,cfg,result,names in loaded:
        lines.extend(['',f'### 第{number}轮','', '|策略或完整参数设置|基础夏普|压力夏普|基础年化收益|基础最大回撤|成交次数|','|---|---:|---:|---:|---:|---:|'])
        for b in (r for r in result['all_metrics'] if r['cost']=='BASE' and r['model']!='BUY_HOLD'):
            s=next(r for r in result['all_metrics'] if r['cost']=='STRESS' and r['model']==b['model'])
            lines.append(f'|{names[b["model"]]}|{dec(b["net_sharpe"])}|{dec(s["net_sharpe"])}|{pct(b["annualized_return"])}|{pct(b["max_drawdown"])}|{b["trade_count"]}|')
    lines.extend(['','## 每个因子与全部中文进出场规则','', '下列内容直接收录各轮账户计算前的中文规则。所有条件都用于研究账户，不代表实际持仓或交易订单。'])
    for number,p,cfg,result,names in loaded:
        protocol=(ROOT/cfg['rules']).read_text(encoding='utf-8')
        for line in protocol.splitlines():
            lines.append('#'+line if line.startswith('#') else line)
        lines.append('')
    lines.extend(['## 数值文件','', '- [全部策略完整账户](全部策略完整账户.csv)', '- [全部策略逐年表现](全部策略逐年表现.csv)',
                  '- [全部策略三个阶段](全部策略三个阶段.csv)', '- [全部实际买入卖出周期](全部实际买入卖出周期.csv)', '- [更早五年的原参数对照](恐慌回升策略更早五年对照.csv)',
                  '', '本次四轮主评价共128个记录，其中120个新增评价账户、8个复用对照；另保存24个季度选择训练账户和4个更早历史诊断账户。未制作GPT审阅数值包，未启动交易。'])
    document=OUT/'简单策略快速检验_全部因子进出场与历史表现.md'
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')

    index=read('reports/research/510300_sharpe_1_2_latest_research.json')
    for number,p,cfg,result,names in loaded:
        if any(r['round']==number for r in index['completed_rounds']):continue
        registered=any(r['round']==number for r in index['partial_rounds'])
        count=result['candidate_configurations']
        record={'round':number,'study':result['study_id'],'title':dict((n,t) for n,_,t in STUDIES)[number],
                'status':'COMPLETED_TARGET_NOT_MET','result':str((p/'result.json').relative_to(ROOT)),
                'candidate_configurations':count,'evaluated_candidate_source_runs':count,'evaluation_accounts':result['evaluation_accounts'],
                'new_accounts_generated':result['new_accounts_generated'],'reused_control_accounts':result.get('reused_control_accounts',0),
                'post_selected_best_base':result['post_selected_best_base'],'primary_base':result['primary'][0],'primary_stress':result['primary'][1]}
        if number==26:record['earlier_history_counterevidence']='reports/research/510300_simple_panic_earlier_history_v1/result.json'
        index['completed_rounds'].append(record)
        index['evaluation_accounts_in_this_resumption']+=result['evaluation_accounts']
        index['evaluated_configurations_in_this_resumption']+=count
        index['evaluated_candidate_source_runs_including_corrected_replays']+=count
        if not registered:
            index['registered_configurations_in_this_resumption']+=count
            index['registered_candidate_source_runs_including_unrun_legacy_bindings']+=count
        index['partial_rounds']=[r for r in index['partial_rounds'] if r['round']!=number]
    index['latest_completed_round']=next(r for r in index['completed_rounds'] if r['round']==26)
    best=loaded[-1][3]['post_selected_best_base']
    if best['net_sharpe']>index['post_selected_best_base']['net_sharpe']:
        index['post_selected_best_base']={'round':26,'title':'短期高波动急跌后回升','source_result':'reports/research/510300_simple_volume_reversal_v1/result.json',
                                         **best,'status':'HISTORICAL_POST_SELECTED_BEST_EARLIER_PERIOD_FAILED','completed_round_trips':3,
                                         'earlier_period_net_sharpe':-.10683567142232424,'stable_excess_established':False,'goal_achieved':False}
    index.update({'updated_at':now(),'status':'ROUNDS23_TO26_COMPLETE_CONTINUE_FAST_SIMPLE_RESEARCH','running_studies':[],
                  'process_state_note':'第23至26轮和恐慌回升更早历史诊断均已完成，无研究进程仍运行。持续任务保持有效，下一轮需新登记后执行。',
                  'next_work':['停止EPS和来源补齐，保持简单方法优先。已完成第23至26轮；不要重复运行已保存账户。',
                               '下一步优先检验预定限价等待入场、订单到期与退出时钟，保留日线无法确认成交的条件状态，使用保守成交假设及T+1。',
                               '对出现改善的简单策略优先用原参数跨时期检验；第26轮V6近年1.118只有3次交易，更早历史为负0.107，不做围绕这3次交易的细密调参。'],
                  'count_warning':'完成26轮，253个不同配置或范围、265个已评价来源版本、614个主评价账户。登记来源版本270，另有5个旧未运行绑定。第24轮另24训练账户、原参数更早历史另4诊断账户不计入614。'})
    entry={'created_at':now(),'type':'FOUR_FAST_SIMPLE_STRATEGY_ROUNDS_CHINESE_REPORT','directory':str(OUT),'main_document':str(document),'new_gpt_review_archive_created':False}
    if not any(d.get('type')==entry['type'] for d in index['deliveries']):index['deliveries'].append(entry)
    write_json(ROOT/'reports/research/510300_sharpe_1_2_latest_research.json',index)
    write_json(OUT/'delivery_receipt.json',{'generated_at':now(),'status':'CREATED_VISUAL_CHECK_PENDING','main_document':str(document),
               'main_evaluation_rows':len(m),'yearly_rows':len(y),'era_rows':len(e),'new_candidate_configurations':59,'completed_rounds':[23,24,25,26],
               'visual_check_pending':True,'goal_achieved':False,'new_gpt_review_archive_created':False})
    print(json.dumps({'中文说明':str(document),'图表':str(OUT/'四轮简单策略完整净值.png'),'主评价记录':len(m),'完成轮数':len(index['completed_rounds'])},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
