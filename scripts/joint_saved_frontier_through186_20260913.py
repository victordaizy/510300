"""增量读取旧结果，按双门槛定位薄弱项，不重跑旧账户。"""
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
from scripts.saved_candidate_frontier_20260909 import FIELDS,NORMAL,clean,signature

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_joint_saved_frontier_through186'
OLD=ROOT/'reports/research/510300_saved_candidate_frontier_20260909/candidate_sources.json'
TAGS=['main_base','main_stress','earlier_base','earlier_stress']


def main():
    began=time.perf_counter()
    index_path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==186 and not OUT.exists(),'增量诊断前序或输出状态不同')
    OUT.mkdir()
    sources=[OLD,index_path,Path(__file__)]
    versions=defaultdict(dict)
    for old in json.loads(OLD.read_text(encoding='utf-8')):
        c=copy.deepcopy(old)
        c.pop('candidate_id',None)
        versions[(c['model'],c['assumption'],signature(c['main']))][signature(c['earlier']) if c['earlier'] else 'UNKNOWN']=c
    coverage=[]
    for rec in index['completed_rounds']:
        if rec['round']<=130:
            continue
        path=ROOT/rec['result']
        result=json.loads(path.read_text(encoding='utf-8'))
        sources.append(path)
        main_rows=result.get('all_metrics',[])
        early_rows=result.get('earlier_diagnostics',[])
        coverage.append({'round':rec['round'],'source':str(path.relative_to(ROOT)),'main_rows':len(main_rows),'earlier_rows':len(early_rows)})
        group=defaultdict(lambda:defaultdict(list))
        for phase,rows in [('main',main_rows),('earlier',early_rows)]:
            for row in rows:
                assumption=row.get('assumption') or ''
                normalized='NEXT_OPEN' if assumption in NORMAL else assumption
                group[(row['model'],normalized)][phase].append(row)
        for (model,assumption),phases in group.items():
            pairs={phase:[next((x for x in rows if x['cost']==cost),None) for cost in ['BASE','STRESS']]
                   for phase,rows in phases.items()}
            m=pairs.get('main',[])
            e=pairs.get('earlier',[])
            if len(m)!=2 or any(x is None for x in m):
                continue
            require(all(sum(x['cost']==cost for x in phases['main'])==1 for cost in ['BASE','STRESS']),'主费用对重复')
            if len(e)!=2 or any(x is None for x in e):
                e=[]
            else:
                require(all(sum(x['cost']==cost for x in phases['earlier'])==1 for cost in ['BASE','STRESS']),'较早费用对重复')
            key=(model,assumption,signature(m))
            early_key=signature(e) if e else 'UNKNOWN'
            c=versions[key].setdefault(early_key,{'model':model,'name':m[0].get('name') or model,'assumption':assumption,
                'main':clean(m),'earlier':clean(e),'sources':[]})
            c['sources'].append({'round':rec['round'],'title':rec['title'],'folder':str(path.parent.relative_to(ROOT)),
                'main_source':str(path.relative_to(ROOT)),'earlier_source':str(path.relative_to(ROOT)) if e else None})
    candidates=[]
    for alternatives in versions.values():
        known=[k for k in alternatives if k!='UNKNOWN']
        if 'UNKNOWN' in alternatives and len(known)==1:
            alternatives[known[0]]['sources'].extend(alternatives.pop('UNKNOWN')['sources'])
        candidates.extend(alternatives.values())
    rows=[]
    for number,c in enumerate(candidates,1):
        c['candidate_id']=f'JOINT_{number:04d}'
        values=c['main']+c['earlier']
        complete=len(values)==4 and [x.get('trading_days') for x in values]==[1604,1604,1219,1219]
        finite=complete and all(x.get(k) is not None and math.isfinite(x[k]) for x in values for k in ['net_sharpe','annualized_return'])
        row={'candidate_id':c['candidate_id'],'model':c['model'],'name':c['name'],'assumption':c['assumption'],
             'source_rounds':','.join(map(str,sorted({s['round'] for s in c['sources']}))),
             'complete_finite_same_length':finite,'current_top_ledger_verified':False}
        for tag,m in zip(TAGS,values):
            row.update({tag+'_'+k:m.get(k) for k in FIELDS})
        if finite:
            ratios={tag+'净夏普':m['net_sharpe']/1.2 for tag,m in zip(TAGS,values)}
            ratios.update({tag+'复合年化':m['annualized_return']/.1 for tag,m in zip(TAGS,values)})
            bottleneck=min(ratios,key=ratios.get)
            row.update(minimum_joint_ratio=ratios[bottleneck],bottleneck=bottleneck,
                       all_four_joint_point_pass=all(x>=1 for x in ratios.values()))
        rows.append(row)
    frame=pd.DataFrame(rows)
    ranked=frame[frame.complete_finite_same_length & frame.assumption.eq('NEXT_OPEN')].sort_values(
        ['minimum_joint_ratio','candidate_id'],ascending=[False,True]).copy()
    top_checks=[]
    market=pd.read_parquet(ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet')
    for candidate_id in ranked.head(5).candidate_id:
        c=next(x for x in candidates if x['candidate_id']==candidate_id)
        source=max((s for s in c['sources'] if s.get('earlier_source')),key=lambda s:s['round'])
        folder=ROOT/source['folder']
        cfg_path=ROOT/'config'/f'{folder.name}.json'
        cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
        require(cfg['initial_capital']==200000 and cfg['annual_days']==242 and cfg['cash_annual_rate_assumption']==0,'靠前版本账户口径不一致')
        require(cfg['costs']=={'BASE':{'commission':.0002,'minimum':5.,'slippage':.0005},'STRESS':{'commission':.0004,'minimum':5.,'slippage':.001}},'靠前版本费用不同')
        sources.append(cfg_path)
        for period,metrics in [('evaluation',c['main']),('earlier_diagnostic',c['earlier'])]:
            left,right=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
            expected=market.loc[market.date.between(left,right)]
            for m in metrics:
                ledger_path=folder/period/m['cost']/f"{c['model']}_ledger.parquet"
                ledger=pd.read_parquet(ledger_path)
                sources.append(ledger_path)
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(expected.date)),'靠前账户完整日期不同')
                np.testing.assert_allclose(ledger.open,expected.open,atol=0,rtol=0)
                measured=summarize(ledger,cfg)
                for key in ['net_sharpe','annualized_return','max_drawdown','commission','slippage_cost']:
                    require(abs(measured[key]-m[key])<1e-7,'靠前保存指标不能复算')
                top_checks.append({'candidate_id':candidate_id,'model':c['model'],'period':period,'cost':m['cost'],'source_round':source['round'],'ledger':str(ledger_path.relative_to(ROOT))})
        frame.loc[frame.candidate_id.eq(candidate_id),'current_top_ledger_verified']=True
        ranked.loc[ranked.candidate_id.eq(candidate_id),'current_top_ledger_verified']=True
    for name,table in [('all_saved_candidates.csv',frame),('joint_four_scenario_ranking.csv',ranked),('incremental_source_coverage.csv',pd.DataFrame(coverage)),('top_saved_ledger_checks.csv',pd.DataFrame(top_checks))]:
        table.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    write_json(OUT/'candidate_sources.json',clean(candidates),exclusive=True)
    write_json(OUT/'source_files.json',[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(sources))],exclusive=True)
    summary={'completed_at':now(),'status':'INCREMENTAL_JOINT_RANKING_TOP_FIVE_SAVED_LEDGERS_CHECKED',
        'through_round':186,'reused_saved_frontier_through_round':130,'new_result_documents_read':len(coverage),
        'distinct_named_performance_versions':len(frame),'standard_complete_four_scenario_versions':len(ranked),
        'all_four_joint_point_pass':int(ranked.all_four_joint_point_pass.sum()),'top_saved_ledgers_checked':len(top_checks),
        'new_accounts':0,'new_models':0,'goal_achieved':False,'independent_performance_validation':False,
        'ranking_meaning':'八项指标相对门槛比值的最小值，仅为研究优先级；不是胜率或置信度',
        'run_seconds':time.perf_counter()-began,'top_five':clean(ranked.head(5).to_dict('records'))}
    write_json(OUT/'result.json',summary,exclusive=True)
    lines=['# 已保存策略的双门槛比较：截至第186轮','',
        f"复用截至第130轮的保存清单，只增量读取后续{len(coverage)}份结果，没有重跑任何旧账户或训练模型。清单共{len(frame)}个不同名称及绩效版本，其中{len(ranked)}个有标准下一开盘口径及四个历史费用场景的完整有限指标。四场景同时达到夏普1.2与年化10%的版本数为{summary['all_four_joint_point_pass']}。",'',
        '排序使用八项指标相对目标比值中的最小值，识别最明显的缺口；这是描述性研究优先级，不是达标概率，也不是独立验证。前五名对应二十条已保存账户在本次复算了日期、开盘、费用口径及收益指标，其余沿用来源保存数值。', '',
        '|方案|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱比值|',
        '|---|---|---|---|---|---:|']
    for _,row in ranked.head(10).iterrows():
        cells=[f"{row[tag+'_net_sharpe']:.3f}／{row[tag+'_annualized_return']:.2%}" for tag in TAGS]
        lines.append('|'+str(row['name'])+'|'+'|'.join(cells)+f"|{row.minimum_joint_ratio:.3f}|")
    lines += ['', '排序属于对已经观察过的历史再次比较。不同历史条件或缺少较早结果的版本仍保留在全部候选表中，不能混入标准排名。名称不同但历史路径相同的版本可能仍分别出现，不把版本数当作独立试验数量。']
    (OUT/'双门槛优先级说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    index['latest_joint_saved_frontier']=str((OUT/'result.json').relative_to(ROOT))
    write_json(index_path,index)
    print(json.dumps({'覆盖':{k:v for k,v in summary.items() if k!='top_five'},'前五':ranked.head(5)[['model','minimum_joint_ratio','bottleneck']].to_dict('records')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
