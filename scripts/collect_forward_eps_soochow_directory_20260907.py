"""把固定机构公开研报目录扩展到全部历史沪深300证券。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now

OUT=ROOT/'reports/research/510300_forward_eps_soochow_directory_v1'
RAW=ROOT/'data/raw/510300_forward_eps_soochow_directory_v1'
MANIFEST=ROOT/'config/510300_forward_eps_soochow_directory_v1_manifest.json'
MEMBERS=ROOT/'data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet'
STOP=threading.Event()
HEADERS={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/report/'}


def ident(p):
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    m=pd.read_parquet(MEMBERS,columns=['membership_date','symbol'])
    m.to_parquet(OUT/'historical_membership.parquet',index=False)
    universe=sorted(m.symbol.unique().tolist())
    save(OUT/'historical_union_before_reports.json',{'securities':universe,'count':len(universe)},exclusive=True)
    paths=[Path(__file__),ROOT/'docs/510300_FORWARD_EPS_SOOCHOW_HISTORY_V1.md',MEMBERS,
           OUT/'historical_membership.parquet',OUT/'historical_union_before_reports.json',
           ROOT/'research/financial_annual_components_v1.py',
           ROOT/'reports/research/510300_forward_eps_multi_institution_probe_v1/result.json',
           ROOT/'reports/research/510300_forward_eps_multi_institution_probe_v1/protocol.json']
    save(MANIFEST,{'registered_at':now(),'institution':'东吴证券','institution_code':'80000031',
                   'years':list(range(2015,2027)),'start':'2015-01-01','end':'2026-08-14','budget_cny':0,
                   'security_union_count':len(universe),'return_values_read_for_selection':False,
                   'files':[ident(p) for p in paths]},exclusive=True)
    print('已登记历史沪深300并集',len(universe),'个证券的东吴研报目录。',flush=True)


def page(year,number):
    raw=RAW/f'{year}_{number:04d}.json';receipt=OUT/'request_receipts'/f'{year}_{number:04d}.json'
    if raw.exists() and receipt.exists():
        rec=read(receipt)
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=rec['sha256']:raise ValueError('目录缓存变化')
        return read(raw),rec
    if STOP.is_set():raise RuntimeError('公开接口访问限制后的请求停止')
    params={'code':'*','orgCode':'80000031','qType':0,'pageSize':50,'pageNo':number,
            'beginTime':f'{year}-01-01','endTime':'2026-08-14' if year==2026 else f'{year}-12-31',
            'industryCode':'*','industry':'*','rating':'*','ratingChange':'*','fields':''}
    errors=[]
    for attempt in range(2):
        try:
            r=requests.get('https://reportapi.eastmoney.com/report/list',params=params,headers=HEADERS,timeout=(12,35))
            if r.status_code in (401,403,429):STOP.set();raise RuntimeError('公开接口访问限制：'+str(r.status_code))
            r.raise_for_status();d=r.json()
            if not isinstance(d.get('data'),list):raise ValueError('目录数据结构不符')
            raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(r.content)
            rec={'retrieved_at':now(),'params':params,'url':r.url,'raw_path':raw.relative_to(ROOT).as_posix(),
                 'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'prior_transport_errors':errors}
            save(receipt,rec,exclusive=True);time.sleep(.2)
            return d,rec
        except (requests.RequestException,ValueError) as exc:
            errors.append(type(exc).__name__+': '+str(exc))
            if attempt==1:raise
            time.sleep(1)
    raise RuntimeError('公开目录请求未返回')


def year_run(year):
    marker=OUT/'year_receipts'/f'{year}.json'
    if marker.exists():return read(marker)
    first,rec=page(year,1);total=max(1,int(first['TotalPage']))
    if total>500:raise ValueError('单年超过五百页运行上限，需要检查查询范围')
    rows=[];counts=[];pages=[]
    for n in range(1,total+1):
        d,rr=(first,rec) if n==1 else page(year,n)
        if int(d.get('pageNo',n))!=n:raise ValueError('目录返回页码不符')
        counts.append([int(d['hits']),int(d['TotalPage'])]);pages.append(rr['raw_path'])
        for x in d['data']:
            if x['orgCode']!='80000031':raise ValueError('机构过滤未生效')
            if not f'{year}-01-01'<=x['publishDate'][:10]<=rr['params']['endTime']:raise ValueError('研报日期超出查询范围')
            rows.append({**x,'query_year':year,'source_page':n,'source_raw_path':rr['raw_path'],'directory_retrieved_at':rr['retrieved_at']})
        if n%10==0:print('全行业东吴目录',year,n,'/',total,'页',flush=True)
    unique=len({x['infoCode'] for x in rows})
    complete=len(set(tuple(x) for x in counts))==1 and len(rows)==int(first['hits']) and unique==len(rows)
    result={'year':year,'completed_at':now(),'rows':rows,'pages':pages,'counts':counts,
            'unique_reports':unique,'pagination_complete':complete}
    save(marker,result,exclusive=True);print('全行业年度目录完成',year,len(rows),'分页一致',complete,flush=True)
    return result


def run():
    manifest=read(MANIFEST)
    for x in manifest['files']:
        if ident(ROOT/x['path'])['sha256']!=x['sha256']:raise ValueError('登记输入变化')
    if (OUT/'result.json').exists():raise FileExistsError('本轮目录已完成')
    done=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(year_run,y):y for y in manifest['years']}
        for f in as_completed(futures):
            try:done.append(f.result())
            except Exception as e:
                errors.append({'year':futures[f],'error_type':type(e).__name__,'error':str(e),'at':now()})
                print('全行业目录未完成',futures[f],type(e).__name__,flush=True)
    rows=sorted([r for d in done for r in d['rows']],key=lambda x:(x['publishDate'],x['infoCode']))
    save(OUT/'all_directory_records.json',{'rows':rows},exclusive=True)
    cols=['stockCode','stockName','infoCode','publishDate','orgCode','orgName','orgSName','title','attachPages','source_raw_path','directory_retrieved_at']
    d=pd.DataFrame([{k:x.get(k) for k in cols} for x in rows])
    d['ts_code']=d.stockCode.map(lambda x:x+('.SH' if x.startswith(('5','6','9')) else '.SZ'))
    d.to_parquet(OUT/'all_soochow_stock_reports.parquet',index=False)
    universe=set(read(OUT/'historical_union_before_reports.json')['securities'])
    q=d.loc[d.ts_code.isin(universe)].drop_duplicates('infoCode').copy()
    q.to_parquet(OUT/'historical_member_original_pdf_queue.parquet',index=False)
    q.to_csv(OUT/'历史成分股_东吴原始研报队列.csv',index=False,encoding='utf-8-sig')
    q.groupby('ts_code').agg(reports=('infoCode','nunique'),first=('publishDate','min'),last=('publishDate','max')).to_csv(OUT/'证券覆盖.csv',encoding='utf-8-sig')
    result={'completed_at':now(),'status':'HISTORICAL_MEMBER_ANALYST_DIRECTORY_COMPLETE' if not errors else 'HISTORICAL_MEMBER_ANALYST_DIRECTORY_PARTIAL',
            'scheduled_years':12,'completed_years':len(done),'pagination_complete_years':sum(x['pagination_complete'] for x in done),
            'all_stock_report_occurrences':len(rows),'all_stock_reports_unique':int(d.infoCode.nunique()),
            'historical_member_pdf_queue':len(q),'covered_historical_member_securities':int(q.ts_code.nunique()),
            'historical_member_union':len(universe),'year_report_counts':{str(x['year']):x['unique_reports'] for x in done},
            'queue_year_counts':q.publishDate.str[:4].value_counts().sort_index().to_dict(),
            'errors':errors,'access_restriction_stop':STOP.is_set(),'market_consensus':False,
            'new_eps_facts':0,'new_accounts':0,'goal_achieved':False,
            'outputs':[ident(OUT/'all_soochow_stock_reports.parquet'),ident(OUT/'historical_member_original_pdf_queue.parquet')]}
    save(OUT/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    freeze() if a.freeze else run()
