"""交付第27、28轮及静态风险归因，更新持续研究状态。"""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from research.intraday_overnight_increment_v1 import now,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/510300限价与多信号组合研究_20260907'

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def number(x):return '未定义' if pd.isna(x) else f'{x:.3f}'
def pct(x):return '未定义' if pd.isna(x) else f'{x:.2%}'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    r27=read('reports/research/510300_simple_limit_execution_v1/result.json')
    r28=read('reports/research/510300_simple_signal_blend_v1/result.json')
    cfg27=read('config/510300_simple_limit_execution_v1.json');cfg28=read('config/510300_simple_signal_blend_v1.json')
    uncertainty=read('reports/research/510300_simple_pair_saved_uncertainty_v1/result.json')
    attribution=read('reports/research/510300_simple_pair_static_risk_attribution_v1/result.json')
    copied=[]
    for prefix,folder in [('限价','510300_simple_limit_execution_v1'),('组合','510300_simple_signal_blend_v1')]:
        for file in ['metrics.csv','yearly_metrics.csv','era_metrics.csv','earlier_diagnostics.csv']:
            target=OUT/f'{prefix}_{file}'
            target.write_bytes((ROOT/'reports/research'/folder/file).read_bytes());copied.append(target.name)
    (OUT/'限价订单统计.csv').write_bytes((ROOT/'reports/research/510300_simple_limit_execution_v1/order_statistics.csv').read_bytes())
    pd.DataFrame(attribution['static_accounts']).to_csv(OUT/'固定半仓完整账户.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(attribution['increments']).to_csv(OUT/'双信号相对半仓的择时增量区间.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(uncertainty['results']).to_csv(OUT/'双信号相对买入持有的不确定性.csv',index=False,encoding='utf-8-sig')
    lines=['# 510300：限价与多信号组合的结果及失败原因','',f'整理时间：{now()}。','',
           '**目标仍未达到：没有确认稳定超额和完整账户净夏普1.2。** 本次新增12种限价设置、8种固定组合，完成74个主评价记录，其中64个新增账户、10个复用对照；另24个更早历史诊断账户、3个更早基础信号账户、4个固定半仓诊断账户。没有继续补齐EPS，没有制作GPT数值审阅包。',
           '', '## 本次最有用的结果','',
           '限价没有解决收益来源问题。只按开盘条件成交的最高基础夏普0.617，仅1次完整交易；加入盘中明显跌穿后的条件成交，最高0.942，仅2次完整交易。二者都是“急跌后放量收强、比信号收盘低1%、等待1日”。原参数在2015至2019年基础夏普分别为负0.260和负0.229。盘中版本只有日线成交假设，不是实际排队和成交证据。',
           '', '固定组合改善了部分历史结果，但共识本身不等于稳定。至少两种状态同意才进入的基础夏普为0.755，更早五年只有0.028；预定等权主方案为0.605，更早为0.380。',
           '', '**后续保留的简单候选是“趋势与日内强弱各半”**：基础夏普主评价0.534、更早0.517；压力费用下分别0.467、0.446。两种费用下，两段历史的复合年化收益均高于各自买入持有。不过相对固定半仓的择时增量区间仍跨过零，尚不足以确认稳定超额，距离1.2也很远。',
           '', '## 为什么没有达到目标','',
           '1. 较高点估计反复来自极少数交易。上一轮高波动回升的1.118只有3次买卖，本轮限价0.942只有2次；同参数换到更早历史均亏损。这类结果容易受事件样本与筛选影响。',
           '2. 买得更便宜也会错过交易。限价改变了能成交的日期和持仓周期，不能只看成交价改善而忽略未成交与长期空仓。本轮限价最好设置年化收益很低。',
           '3. 多因子同意未能稳定识别适用时期。共识规则在近年表现较好，在更早历史几乎没有夏普；此前按近期夏普选规则也失败。切换规则本身需要可重复的信息优势。',
           '4. 降低波动、提高复合收益与稳定择时超额是不同问题。双信号组合相对买入持有的复合年化收益较高，但较早期间的算术平均超额为负。进一步对比固定半仓后，择时增量点估计虽为正，其区间仍跨零。不能用一个正的年化差替代稳定性证据。',
           '', '## 十二种限价设置的完整时期表现','',
           '|基础信号与订单设置|只开盘：基础夏普|只开盘：压力夏普|盘中条件：基础夏普|盘中条件：压力夏普|',
           '|---|---:|---:|---:|---:|']
    for item in cfg27['candidates']:
        name=f'{cfg27["signal_names"][item["signal"]]}；低{item["discount"]:.1%}；等待{item["valid_days"]}日'
        values=[]
        for assumption in ['OPEN_ONLY','PENETRATION']:
            for cost in ['BASE','STRESS']:
                m=next(x for x in r27['all_metrics'] if x['model']==item['id'] and x['assumption']==assumption and x['cost']==cost)
                values.append(number(m['net_sharpe']))
        lines.append('|'+name+'|'+'|'.join(values)+'|')
    lines.extend(['','## 八种固定组合的两个时期对照','',
                  '|策略|主评价基础夏普|主评价压力夏普|更早基础夏普|更早压力夏普|主评价基础年化收益|更早基础年化收益|',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for key,name in cfg28['candidate_names'].items():
        a=next(x for x in r28['all_metrics'] if x['model']==key and x['cost']=='BASE')
        b=next(x for x in r28['all_metrics'] if x['model']==key and x['cost']=='STRESS')
        c=next(x for x in r28['earlier_diagnostics'] if x['model']==key and x['cost']=='BASE')
        d=next(x for x in r28['earlier_diagnostics'] if x['model']==key and x['cost']=='STRESS')
        lines.append(f'|{name}|{number(a["net_sharpe"])}|{number(b["net_sharpe"])}|{number(c["net_sharpe"])}|{number(d["net_sharpe"])}|{pct(a["annualized_return"])}|{pct(c["annualized_return"])}|')
    lines.extend(['','主评价为2020年1月2日至2026年8月14日开盘终点；更早诊断为2015年1月5日至2019年12月31日开盘终点。两段均从20万元重新建立账户，不能拼成一条实际连续净值。更早历史已经被研究过，仍不属于独立验证。','',
                  '![两个时期的完整账户](两个时期的组合与固定半仓.png)','', '## 双信号组合与固定半仓的比较','',
                  '固定半仓对照在每个收盘检查50%目标，沿用10个百分点免调仓区间；它不使用未来信息决定仓位。两段实际结果各只有初始买入与终点卖出，期间未触及需调仓的区间。双信号组合主评价复合年化收益5.28%，固定半仓1.92%；更早分别5.29%、2.04%。',
                  '', '采用20日连续区间抽样，相对固定半仓的年化算术收益增量及95%区间如下。这些区间未修正多次策略筛选，不能作为独立通过证明。','',
                  '|时期及费用|择时增量点估计|95%区间下限|95%区间上限|','|---|---:|---:|---:|'])
    for m in attribution['increments']:
        if m['block']!=20:continue
        name=('主评价' if m['period']=='evaluation' else '更早诊断')+('／基础费用' if m['cost']=='BASE' else '／压力费用')
        lo,hi=m['interval_95'];lines.append(f'|{name}|{pct(m["annualized_arithmetic_timing_increment"])}|{pct(lo)}|{pct(hi)}|')
    lines.extend(['','60日区间抽样也跨过零。因此当前证据支持“值得继续研究”，不支持“稳定超额已经成立”。后续优先比较双信号固定权重及预定风险缩减，不再围绕少数成功交易细调限价或反弹门槛。ETF组合是否可计入1.2验收范围仍待用户明确；未明确前继续保持510300和现金。','',
                  '## 限价单有效期的实际含义',''])
    lines.extend((ROOT/'docs/510300_LIMIT_ORDER_DAY_VALIDITY_NOTE_20260907.md').read_text(encoding='utf-8').splitlines()[2:])
    lines.extend(['','## 全部中文因子及进出场规则',''])
    for p in ['docs/510300_SIMPLE_LIMIT_EXECUTION_V1.md','docs/510300_SIMPLE_SIGNAL_BLEND_V1.md']:
        for line in (ROOT/p).read_text(encoding='utf-8').splitlines():lines.append('#'+line if line.startswith('#') else line)
        lines.append('')
    lines.extend(['## 数值文件',''])
    for f in copied+['限价订单统计.csv','固定半仓完整账户.csv','双信号相对半仓的择时增量区间.csv','双信号相对买入持有的不确定性.csv']:
        lines.append(f'- [{f}]({f})')
    document=OUT/'限价与多信号组合_全部规则结果及失败原因.md';document.write_text('\n'.join(lines)+'\n',encoding='utf-8')

    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(14,6),dpi=170)
    for ax,period,title in zip(axes,['earlier_diagnostic','evaluation'],['2015—2019：更早历史诊断','2020—2026：主评价']):
        for key,label,color in [('B2_TREND_SESSION','趋势与日内强弱各半','#006F87'),('B5_MAJORITY','至少两种同意才进入','#CA7D2C')]:
            df=pd.read_parquet(ROOT/'reports/research/510300_simple_signal_blend_v1'/period/'BASE'/f'{key}_ledger.parquet')
            ax.plot(df.date,df.equity/200000,label=label,color=color,lw=1.6)
        st=pd.read_parquet(ROOT/'reports/research/510300_simple_pair_static_risk_attribution_v1'/period/'BASE/STATIC_HALF_ledger.parquet')
        ax.plot(st.date,st.equity/200000,label='固定半仓对照',color='#888888',lw=1.1)
        ax.set_title(title,fontsize=13);ax.set_ylabel('20万元起点归一化净值');ax.legend(frameon=False,fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('同一规则在两个时期的完整账户表现',fontsize=17)
    fig.text(.5,.02,'基础费用；两段各自重新建立20万元账户。已有历史筛选结果，未确认稳定超额或夏普1.2。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.93));fig.savefig(OUT/'两个时期的组合与固定半仓.png');plt.close(fig)
    for folder,result in [('510300_simple_limit_execution_v1',r27),('510300_simple_signal_blend_v1',r28)]:
        write_json(ROOT/'reports/research'/folder/'acceptance_outcome.json',{'recorded_at':now(),'status':'COMPLETED_TARGET_NOT_MET',
                   'goal_achieved':False,'historical_point_target_met':result['historical_point_target_met'],'independent_validation':'NOT_ESTABLISHED','position_impact':0})
    index=read('reports/research/510300_sharpe_1_2_latest_research.json')
    record={'round':28,'study':r28['study_id'],'title':'固定信号组合与多数共识仓位','status':'COMPLETED_TARGET_NOT_MET',
            'result':'reports/research/510300_simple_signal_blend_v1/result.json','candidate_configurations':8,'evaluated_candidate_source_runs':8,
            'evaluation_accounts':18,'new_accounts_generated':16,'reused_control_accounts':2,'earlier_diagnostic_accounts':18,'earlier_expert_accounts':3,
            'post_selected_best_base':r28['post_selected_best_base'],'primary_base':r28['primary'][0],'primary_stress':r28['primary'][1]}
    if not any(r['round']==28 for r in index['completed_rounds']):
        index['completed_rounds'].append(record);index['evaluation_accounts_in_this_resumption']+=18
        index['evaluated_configurations_in_this_resumption']+=8;index['evaluated_candidate_source_runs_including_corrected_replays']+=8
    index['partial_rounds']=[r for r in index['partial_rounds'] if r['round']!=28]
    index.update({'updated_at':now(),'status':'ROUNDS27_28_COMPLETE_PAIR_RISK_RESEARCH_CONTINUES','latest_completed_round':record,'running_studies':[],
                  'process_state_note':'第27、28轮及双信号不确定性、固定半仓归因均已完成。没有上述任务的运行进程。',
                  'next_work':['不继续EPS补齐、稀疏事件或限价的细密调参。','优先研究趋势与日内强弱这两个固定状态的预定风险权重和波动缩减；新参数需要先登记，主时期与较早时期都完整比较。',
                               '资产验收范围问题已明确询问，用户尚未回复；未明确前仅510300与现金。'],
                  'count_warning':'完成28轮，273个不同配置或范围、285个已评价来源版本、688个主评价记录。登记来源版本290，包含5个旧未运行绑定。第27、28轮另24更早诊断、3基础专家及4固定半仓账户不计入688。',
                  'current_followup_candidate':{'study':'510300_SIMPLE_SIGNAL_BLEND_V1','model':'B2_TREND_SESSION','base_main_sharpe':.5339922400677327,
                                               'base_earlier_sharpe':.5170276208448453,'status':'RESEARCH_CANDIDATE_ONLY_INCREMENT_INTERVALS_CROSS_ZERO',
                                               'static_risk_attribution':'reports/research/510300_simple_pair_static_risk_attribution_v1/result.json'}})
    delivery={'created_at':now(),'type':'LIMIT_AND_FIXED_BLEND_CHINESE_RESULTS','directory':str(OUT),'main_document':str(document),'new_gpt_review_archive_created':False}
    if not any(d.get('type')==delivery['type'] for d in index['deliveries']):index['deliveries'].append(delivery)
    write_json(ROOT/'reports/research/510300_sharpe_1_2_latest_research.json',index)
    write_json(OUT/'delivery_receipt.json',{'created_at':now(),'status':'CREATED_VISUAL_CHECK_PENDING','main_document':str(document),
               'new_candidate_configurations':20,'main_evaluation_records':74,'earlier_diagnostic_accounts':24,'static_diagnostic_accounts':4,
               'visual_check_pending':True,'goal_achieved':False,'new_gpt_review_archive_created':False})
    print(json.dumps({'交付文件':str(document),'图表':str(OUT/'两个时期的组合与固定半仓.png'),'完成轮数':len(index['completed_rounds'])},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
