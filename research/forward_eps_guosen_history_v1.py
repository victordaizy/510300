"""读取国信原始研报的绝对年度预测，保留原文口径、日期和缺口。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now,norm,compact,decimal

NUM=r'\(?[-+]?\d[\d,]*(?:\.\d+)?\)?'


def exact_row(text,labels,width,unit):
    matches=[]
    for label in labels:
        pattern=r'(?m)^\s*'+re.escape(label)+r'\s*\('+unit+r'\)\s*'+r'\s+'.join('('+NUM+')' for _ in range(width))+r'(?=\s*(?:[^\d\s.,%+\-()]|$))'
        for m in re.finditer(pattern,text):
            matches.append({'label':label,'raw':m.group(0).strip(),'cells':list(m.groups()),
                            'values':[str(decimal(v)) for v in m.groups()]})
    if len(matches)!=1:raise ValueError('明确年度数值行不唯一或未识别：'+str(labels))
    return matches[0]


def internal_date(first):
    candidates=re.findall(r'(20\d{2})年(\d{1,2})月(\d{1,2})日',compact(first)[:650])
    if not candidates:raise ValueError('报告首页明确落款日未识别')
    y,m,d=map(int,candidates[0])
    return date(y,m,d).isoformat()


def parse(pages,metadata,row):
    if metadata['info_code']!=row['infoCode'] or str(metadata['company_code'])!='80000007':raise ValueError('证券目录或机构不符')
    if row['ts_code'][:6] not in [str(s.get('stock')) for s in metadata['security']]:raise ValueError('目录证券代码不符')
    front=compact('\n'.join(pages[:2]))
    if row['ts_code'][:6] not in front or ('guosen.com.cn' not in front.lower() and '国信证券' not in front):raise ValueError('原件首页证券或机构身份未确认')
    report_date=internal_date(pages[0])
    dates=[report_date,str(metadata['notice_date'])[:10],str(metadata['eitime'])[:10],str(row['publishDate'])[:10]]
    for v in dates:date.fromisoformat(v)
    info_date=max(dates)
    tables=[]
    for page_number,original in enumerate(pages[:3],start=1):
        s=norm(original)
        for marker in re.finditer(r'盈利预测和财务指标',s):
            tail=s[marker.end():]
            stop=re.search(r'(?:营业收入|总营业收入|归母净利润|净利润)\s*\(',tail)
            if not stop:continue
            head=tail[:stop.start()]
            header=re.findall(r'(?<!\d)(20\d{2})\s*([AEae]?)(?!\d)',head)
            if not 3<=len(header)<=6:continue
            yrs=[int(y) for y,f in header]
            if yrs!=list(range(yrs[0],yrs[0]+len(yrs))):continue
            predicted=[i for i,(y,f) in enumerate(header) if f.upper()=='E']
            if not predicted or predicted!=list(range(predicted[0],len(header))):continue
            end=re.search(r'资料来源',tail)
            body=tail[:end.start()] if end else tail
            eps=exact_row(body,['摊薄每股收益'],len(header),'元')
            note_match=re.search(r'摊薄每股收益按最新总股本计算',compact(tail))
            if not note_match:raise ValueError('最新总股本口径附注缺失')
            optional={};optional_errors={}
            for key,labels,unit in [('net_profit',['归母净利润','净利润'],'百万元'),('pe',['市盈率'],'PE')]:
                try:optional[key]=exact_row(body,labels,len(header),unit)
                except ValueError as exc:optional_errors[key]=str(exc)
            tables.append({'page':page_number,'header':[[int(y),f.upper()] for y,f in header],
                           'eps':eps,'optional':optional,'optional_errors':optional_errors,
                           'basis_note':'摊薄每股收益按最新总股本计算','header_raw':head.strip()})
    if len(tables)!=1:raise ValueError('前三页明确预测表不是唯一一组：'+str(len(tables)))
    t=tables[0]
    snapshot=None
    for m in re.finditer(r'总股本\s*/\s*流通(?:股本)?\s*\(百万股\)\s*('+NUM+r')\s*/\s*('+NUM+r')',norm('\n'.join(pages[:2]))):
        item={'total_shares_million_exact':str(decimal(m.group(1))),'raw':m.group(0)}
        if snapshot is not None:raise ValueError('首页最新总股本出现多个数值')
        snapshot=item
    facts=[]
    for i,(year,flag) in enumerate(t['header']):
        if flag!='E':continue
        eps=t['eps']['values'][i]
        f={'report_id':row['infoCode'],'ts_code':row['ts_code'],'sec_name':row['stockName'],
           'institution_code':'80000007','institution':'国信证券','target_fiscal_year':year,
           'eps_value_exact':eps,'eps_unit_as_reported':'元/股','eps_currency_iso_independently_proven':False,
           'eps_definition':'摊薄每股收益，按报告当时最新总股本计算',
           'source_page':t['page'],'source_eps_row':t['eps']['raw'],'source_eps_cell':t['eps']['cells'][i],
           'header':t['header'],'header_raw':t['header_raw'],'selected_column_one_based':i+1,
           'report_internal_date':report_date,'provider_notice_date':metadata['notice_date'],
           'provider_eitime':metadata['eitime'],'directory_publish_date':row['publishDate'],
           'conservative_information_date':info_date,'historical_immutable_snapshot_proven':False,
           'target_fiscal_year_already_ended':year<int(report_date[:4]),
           'share_snapshot_million':snapshot['total_shares_million_exact'] if snapshot else None,
           'share_snapshot_raw':snapshot['raw'] if snapshot else None,
           'share_snapshot_is_rounded_report_value':True,
           'actual_future_eps_used_as_predictor':False}
        for key in ['net_profit','pe']:
            if key in t['optional']:
                q=t['optional'][key]
                f[key+'_value_exact']=q['values'][i]
                f[key+'_source_raw']=q['raw']
                f[key+'_source_label']=q['label']
            else:f[key+'_value_exact']=None
        facts.append(f)
    return {'facts':facts,'table':t,'report_internal_date':report_date,'conservative_information_date':info_date,
            'date_fields_differ':len(set(dates))>1,'share_snapshot':snapshot}


def get_paths(scope):
    source=ROOT/('reports/research/510300_forward_eps_'+scope+'_originals_v1')
    out=ROOT/('reports/research/510300_forward_eps_'+scope+'_facts_v1')
    manifest=ROOT/('config/510300_forward_eps_'+scope+'_facts_v1_manifest.json')
    return source,out,manifest


def identity(p):
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def preflight():
    cases=[('financial','AP201903071303507274','2019-03-06',2019,'1.49'),
           ('financial','AP202203101551739416','2022-03-10',2022,'2.01'),
           ('csi','AP201811011229685729','2018-11-01',2018,'0.92')]
    for scope,aid,expected_date,year,value in cases:
        source,_,_=get_paths(scope)
        d=read(source/'document_records'/(aid+'.json'));p=read(source/'page_texts'/(aid+'.json'))
        r=parse(p['pages'],d['provider_metadata'],d['directory_record'])
        assert r['report_internal_date']==expected_date
        f=[x for x in r['facts'] if x['target_fiscal_year']==year]
        assert len(f)==1 and Decimal(f[0]['eps_value_exact'])==Decimal(value)
        assert all(compact(x['source_eps_row']) in compact(p['pages'][x['source_page']-1]) for x in r['facts'])
        print('原件前瞻年度核对通过',aid,len(r['facts']),flush=True)


def freeze(scope):
    source,out,manifest=get_paths(scope);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'research/financial_annual_components_v1.py',source/'selected_before_originals.parquet',
           ROOT/('config/510300_forward_eps_'+scope+'_originals_v1_manifest.json'),
           ROOT/'docs/510300_FORWARD_EPS_ABSOLUTE_YEAR_FACTS_V1.md',ROOT/'tests/test_forward_eps_guosen_history_v1.py']
    save(manifest,{'registered_at':now(),'scope':scope,'table_rule':'前三页唯一明确年度预测表；按预测标记和原列读取，不用接口相对年度数值',
                   'selection_by_returns':False,'budget_cny':0,'files':[identity(p) for p in paths]},exclusive=True)
    print('年度EPS提取规则已登记',scope,flush=True)


def run(scope):
    source,out,manifest=get_paths(scope)
    for x in read(manifest)['files']:
        if identity(ROOT/x['path'])['sha256']!=x['sha256']:raise ValueError('登记读取器或队列变化')
    if not (source/'result.json').exists():raise RuntimeError('原件采集仍未结束，沿用原进程等待')
    if (out/'result.json').exists():raise FileExistsError('预测事实已完成，不覆盖')
    rows=pd.read_parquet(source/'selected_before_originals.parquet').to_dict('records')
    facts=[];records=[]
    for row in rows:
        aid=row['infoCode'];p=source/'document_records'/(aid+'.json')
        if not p.exists():
            records.append({'report_id':aid,'ts_code':row['ts_code'],'status':'NO_VIEW_ORIGINAL_UNAVAILABLE'})
            continue
        d=read(p);pages=read(source/'page_texts'/(aid+'.json'))['pages']
        try:
            parsed=parse(pages,d['provider_metadata'],row)
            for f in parsed['facts']:
                if compact(f['source_eps_row']) not in compact(pages[f['source_page']-1]):raise ValueError('EPS原行不在指定原页')
                f['raw_pdf_path']=d['source']['raw_pdf_path'];f['pdf_sha256']=d['source']['pdf_sha256']
            record={'report_id':aid,'ts_code':row['ts_code'],'status':'ANNUAL_FORECAST_EPS_EXTRACTED',
                    'source_record':p.relative_to(ROOT).as_posix(),**parsed}
            facts.extend(parsed['facts'])
        except ValueError as exc:
            record={'report_id':aid,'ts_code':row['ts_code'],'status':'NO_VIEW_LAYOUT_OR_BASIS_UNRESOLVED',
                    'source_record':p.relative_to(ROOT).as_posix(),'reason':str(exc)}
        save(out/'document_facts'/(aid+'.json'),record,exclusive=True);records.append(record)
    df=pd.DataFrame(facts)
    if not df.empty:
        if df.duplicated(['report_id','target_fiscal_year']).any():raise ValueError('同报告预测年度不唯一')
        df.to_parquet(out/'annual_eps_forecast_vintages.parquet',index=False)
        names={'report_id':'研报编号','ts_code':'证券代码','sec_name':'公司名称','institution':'预测机构','target_fiscal_year':'预测目标年度',
               'eps_value_exact':'预测每股收益_原文元每股','report_internal_date':'研报落款日','conservative_information_date':'较晚公开日期',
               'share_snapshot_million':'原文最新总股本_百万股','source_page':'原件页码','source_eps_row':'原文预测行'}
        df[list(names)].rename(columns=names).to_csv(out/'年度前瞻每股收益_中文明细.csv',index=False,encoding='utf-8-sig')
    save(out/'document_outcomes.json',{'rows':records},exclusive=True)
    result={'completed_at':now(),'status':'ABSOLUTE_YEAR_EPS_FACTS_EXTRACTED_WITH_EXPLICIT_GAPS','scope':scope,
            'selected_reports':len(rows),'parsed_reports':sum(r['status']=='ANNUAL_FORECAST_EPS_EXTRACTED' for r in records),
            'forecast_eps_facts':len(facts),'companies_with_eps':int(df.ts_code.nunique()) if len(df) else 0,
            'facts_with_share_snapshot':int(df.share_snapshot_million.notna().sum()) if len(df) else 0,
            'reports_with_date_differences':sum(bool(r.get('date_fields_differ')) for r in records),
            'status_counts':dict(Counter(r['status'] for r in records)),
            'unresolved_reason_counts':dict(Counter(r.get('reason') for r in records if r.get('reason'))),
            'earliest_information_date':df.conservative_information_date.min() if len(df) else None,
            'latest_information_date':df.conservative_information_date.max() if len(df) else None,
            'exact_next_twelve_month_eps_constructed':False,'market_consensus':False,'new_accounts':0,'goal_achieved':False}
    save(out/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['financial','csi'],default='financial')
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.preflight:preflight()
    elif a.freeze:freeze(a.scope)
    else:run(a.scope)
