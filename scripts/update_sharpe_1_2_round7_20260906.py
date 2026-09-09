"""更新持续研究导航，保留全部既有轮次和用户新增经济因子要求。"""
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    latest=json.loads(path.read_text(encoding='utf-8-sig'))
    report=ROOT/'reports/research/510300_total_reverse_repo_v2'
    r=json.loads((report/'result.json').read_text(encoding='utf-8'))
    m=pd.read_csv(report/'metrics.csv')
    latest['completed_rounds']=[x for x in latest['completed_rounds'] if x['study']!=r['study_id']]
    latest['completed_rounds'].append({'round':7,'study':r['study_id'],'title':'全期限逆回购与因子口径修正','status':r['status'],
         'result':'reports/research/510300_total_reverse_repo_v2/result.json','candidate_configurations':15,'evaluation_accounts':len(m),
         'primary_base':next(x for x in r['primary'] if x['cost']=='BASE'),'primary_stress':next(x for x in r['primary'] if x['cost']=='STRESS'),
         'post_selected_best_base':r['post_selected_best_base']})
    rows=[]
    for item in latest['completed_rounds']:
        f=pd.read_csv((ROOT/item['result']).parent/'metrics.csv')
        rows.extend({'round':item['title'],**x} for x in f.to_dict('records'))
    frame=pd.DataFrame(rows)
    frame.to_csv(ROOT/'deliverables/510300夏普1.2持续研究_七轮完整指标_20260906.csv',index=False,encoding='utf-8-sig')
    latest['updated_at']=datetime.now().astimezone().isoformat()
    latest['status']='SEVEN_ROUNDS_COMPLETED_FACTOR_REBUILD_CONTINUES_TARGET_NOT_MET'
    latest['registered_configurations_in_this_resumption']=sum(x['candidate_configurations'] for x in latest['completed_rounds'])
    latest['evaluation_accounts_in_this_resumption']=len(frame)
    latest['post_selected_best_base']=frame.loc[(frame.cost=='BASE')&(frame.model!='BUY_HOLD')].sort_values('net_sharpe',ascending=False).iloc[0].to_dict()
    latest['running_studies']=[{'study':'510300_FUNDAMENTAL_AND_FUND_FLOW_REBUILD_V1','status':'SOURCE_RECONSTRUCTION_PARTIAL_CONTINUE_FROM_SAVED_RECORDS',
         'path':'reports/research/510300_fundamental_and_fund_flow_rebuild_v1','source_reports_archived':77,'structured_category_rows':154,
         'remaining_work':'恢复公募月报原始发布日期，区分2025年11月分类范围切换和5条同口径份额修订；核对510300份额原始时钟与拆分；恢复利润、股本、回购的历史公告版本，再预登记并运行独立的新版本账户。'}]
    latest['next_work']='用户明确要求继续直到找到夏普1.2，不因第七轮完成而停止。第七轮全期限公开量已完成，15候选均未达标，不重复运行。优先继续盈利、估值、股东回报、公募申赎的实质来源修复。已发现income/cashflow的update_date晚于notice_date而available_at取notice_date，必须检查旧实现是否过滤修订，不能把最新修订值当首次公告值；见结构性股权风险溢价引擎旧资料。公募77月报已结构化，25份目录日期疑似迁移，2025年11月口径切换。逐一解决或隔离有证据的问题，不能把所有因子一律判错，也不能仅做审计停在原地；可运行新机制完成完整账户。'
    latest['new_evidence']=['第七轮主方案基础夏普-0.180856，压力-0.157465，15个候选32账户均未达到1.2。',
         '各期限普通逆回购补出408555亿元非七天累计披露金额；2016年补出6.9万亿元，分期限合计与官方年报在舍入精度内吻合。',
         '37篇买断式公告按月报与预告分开，44条期限记录；不将公告量冒充每日实际净投放。',
         '六轮95项不同因子逐项中文说明完成；34项价格因子独立数值复算通过，旧七天数量范围不足已新版本纠正。',
         '公募77份官方月报形成154条分类记录，25份需查原始发布日期，2025年11月分类范围改变，5条同口径相邻记录有份额修订。']
    zipped=ROOT/'deliverables/510300夏普1.2持续研究_第七轮因子修正_GPT审阅_20260906.zip'
    delivery=json.loads(zipped.with_suffix('.delivery.json').read_text(encoding='utf-8'))
    latest['deliveries']=[d for d in latest['deliveries'] if 7 not in d['rounds']]
    latest['deliveries'].append({'rounds':[7],'zip':zipped.relative_to(ROOT).as_posix(),
             'md':'deliverables/510300第七轮_因子修正_20260906/第七轮结果与全部中文规则.md',
             'bytes':delivery['bytes'],'sha256':delivery['sha256'],'members':delivery['members']})
    path.write_text(json.dumps(latest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 510300夏普1.2持续研究：截至第七轮','',
           f"当前完成{len(latest['completed_rounds'])}轮、{latest['registered_configurations_in_this_resumption']}个登记配置和{len(frame)}个完整评价账户。全部候选均未达到全期成本后夏普1.2。登记数包含重复对照，不是独立试验数。",'',
           f"本次持续研究最高基础历史夏普仍为{latest['post_selected_best_base']['net_sharpe']:.6f}，来自第六轮月末与月初条件。第七轮补全逆回购后，主方案夏普-0.180856；尚无达标策略。",'',
           '| 轮次 | 研究 | 配置数 | 评价账户数 | 主方案基础夏普 | 主方案压力夏普 |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for x in latest['completed_rounds']:lines.append(f"| {x['round']} | {x['title']} | {x['candidate_configurations']} | {x['evaluation_accounts']} | {x['primary_base']['net_sharpe']:.4f} | {x['primary_stress']['net_sharpe']:.4f} |")
    lines += ['', '账户均保留二十万元、完整日期、真实分红、整手、次日卖出限制、佣金与滑点。旧结果保留；因子口径问题以新版本修正并明确影响范围。', '',
              '本次新增95项因子逐项中文说明、全期限与买断式公告重建、公募77份月报及154条分类记录。继续按用户要求研究盈利、估值、股东回报与公募申赎，不把净值上涨或二级市场成交量冒充净申购。', '',
              '持续研究安排保持启用。下一步承接已保存的来源与修正事项，完成可运行的新机制，不重复失败候选。']
    (ROOT/'deliverables/510300夏普1.2持续研究_截至第七轮_20260906.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'轮次':len(latest['completed_rounds']),'配置':latest['registered_configurations_in_this_resumption'],'账户':len(frame),'总目标达到':False,'继续研究':latest['running_studies'][0]['study']},ensure_ascii=False))

if __name__=='__main__':main()
