"""只修正年度表头的Parquet存储，保持原件读取规则及逐条预测不变。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research import forward_eps_guosen_history_v1 as base
from research.financial_annual_components_v1 import read,save,now,compact


def get_paths(scope):
    source=ROOT/('reports/research/510300_forward_eps_'+scope+'_originals_v1')
    out=ROOT/('reports/research/510300_forward_eps_'+scope+'_facts_v1_1')
    manifest=ROOT/('config/510300_forward_eps_'+scope+'_facts_v1_1_manifest.json')
    return source,out,manifest


def parquet_frame(facts):
    rows=[]
    for f in facts:
        row=dict(f)
        row['header_json']=json.dumps(row.pop('header'),ensure_ascii=False,separators=(',',':'))
        rows.append(row)
    return pd.DataFrame(rows)


def freeze(scope):
    source,out,manifest=get_paths(scope);out.mkdir(parents=True,exist_ok=True)
    inputs=[Path(__file__),ROOT/'research/forward_eps_guosen_history_v1.py',
            ROOT/'research/financial_annual_components_v1.py',source/'selected_before_originals.parquet',
            ROOT/('config/510300_forward_eps_'+scope+'_facts_v1_manifest.json'),
            ROOT/'tests/test_forward_eps_guosen_history_v1.py',ROOT/'tests/test_forward_eps_guosen_history_storage_v1_1.py']
    old=ROOT/('reports/research/510300_forward_eps_'+scope+'_facts_v1/document_facts')
    if old.exists():inputs.extend(sorted(old.glob('*.json')))
    save(manifest,{'registered_at':now(),'scope':scope,'amendment':'表头数组包含整数年度与字符串标记，Arrow无法作为同质列表存储；只在Parquet列中编码为可逆JSON，逐原件JSON和预测数字不变。',
                   'original_parser_unchanged':True,'prior_records_preserved':True,
                   'files':[base.identity(p) for p in inputs]},exclusive=True)
    print('前瞻表头存储修正已登记',scope,flush=True)


def run(scope):
    source,out,manifest=get_paths(scope)
    for x in read(manifest)['files']:
        if base.identity(ROOT/x['path'])['sha256']!=x['sha256']:raise ValueError('登记输入变化')
    if not (source/'result.json').exists():raise RuntimeError('原件采集仍未结束，等待原进程')
    if (out/'result.json').exists():raise FileExistsError('本版预测事实已经完成')
    rows=pd.read_parquet(source/'selected_before_originals.parquet').to_dict('records')
    facts=[];records=[];reused=0
    for row in rows:
        aid=row['infoCode'];p=source/'document_records'/(aid+'.json')
        previous=ROOT/('reports/research/510300_forward_eps_'+scope+'_facts_v1/document_facts')/(aid+'.json')
        if previous.exists():
            record=read(previous);reused+=1
        elif not p.exists():
            record={'report_id':aid,'ts_code':row['ts_code'],'status':'NO_VIEW_ORIGINAL_UNAVAILABLE'}
        else:
            d=read(p);pages=read(source/'page_texts'/(aid+'.json'))['pages']
            try:
                parsed=base.parse(pages,d['provider_metadata'],row)
                for f in parsed['facts']:
                    if compact(f['source_eps_row']) not in compact(pages[f['source_page']-1]):raise ValueError('EPS原行与原页不符')
                    f['raw_pdf_path']=d['source']['raw_pdf_path'];f['pdf_sha256']=d['source']['pdf_sha256']
                record={'report_id':aid,'ts_code':row['ts_code'],'status':'ANNUAL_FORECAST_EPS_EXTRACTED',
                        'source_record':p.relative_to(ROOT).as_posix(),**parsed}
            except ValueError as exc:
                record={'report_id':aid,'ts_code':row['ts_code'],'status':'NO_VIEW_LAYOUT_OR_BASIS_UNRESOLVED',
                        'source_record':p.relative_to(ROOT).as_posix(),'reason':str(exc)}
        save(out/'document_facts'/(aid+'.json'),record,exclusive=True)
        facts.extend(record.get('facts',[]));records.append(record)
    df=parquet_frame(facts)
    if not df.empty:
        if df.duplicated(['report_id','target_fiscal_year']).any():raise ValueError('预测目标年度重复')
        df.to_parquet(out/'annual_eps_forecast_vintages.parquet',index=False)
        saved=pd.read_parquet(out/'annual_eps_forecast_vintages.parquet')
        if len(saved)!=len(facts):raise ValueError('预测明细存储行数变化')
        for a,b in zip(facts,saved.to_dict('records')):
            if a['header']!=json.loads(b['header_json']) or a['eps_value_exact']!=b['eps_value_exact']:raise ValueError('表头或精确数值存储往返变化')
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
            'exact_next_twelve_month_eps_constructed':False,'market_consensus':False,
            'reused_prior_document_records':reused,'parquet_roundtrip_header_and_exact_value_checked':True,
            'new_accounts':0,'goal_achieved':False}
    save(out/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['financial','csi'],required=True)
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    freeze(a.scope) if a.freeze else run(a.scope)
