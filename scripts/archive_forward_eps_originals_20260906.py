"""归档登记队列的券商原始研报与日期，复用已经保存的原件。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now,compact

HEADERS={'User-Agent':'Mozilla/5.0','Referer':'https://data.eastmoney.com/report/'}


def paths(scope):
    label='510300_forward_eps_'+scope+'_originals_v1'
    queue=ROOT/('reports/research/510300_forward_eps_history_directory_v1/guosen_original_pdf_queue.parquet' if scope=='financial' else 'reports/research/510300_forward_eps_csi_directory_v1/historical_member_original_pdf_queue.parquet')
    return ROOT/('reports/research/'+label),ROOT/('data/raw/'+label),ROOT/('config/'+label+'_manifest.json'),queue


def identity(p):
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def freeze(scope):
    out,raw,manifest,queue=paths(scope)
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(queue)
    if df.infoCode.duplicated().any() or not df.orgCode.eq('80000007').all():raise ValueError('原件队列机构或唯一性不符')
    df.to_parquet(out/'selected_before_originals.parquet',index=False)
    p=[Path(__file__),queue,out/'selected_before_originals.parquet',ROOT/'research/financial_annual_components_v1.py',
       ROOT/('docs/510300_FORWARD_EPS_HISTORY_DIRECTORY_V1.md' if scope=='financial' else 'docs/510300_FORWARD_EPS_CSI_HISTORY_V1.md')]
    save(manifest,{'registered_at':now(),'scope':scope,'reports':len(df),'companies':int(df.ts_code.nunique()),
                   'budget_cny':0,'maximum_processes':3,'selection':'登记目录内全部国信原件，包括无预测表和失效链接',
                   'files':[identity(x) for x in p]},exclusive=True)
    print('原始研报队列已登记',scope,len(df),'份',flush=True)


def request_bytes(url):
    failures=[]
    for n in range(2):
        try:
            r=requests.get(url,headers=HEADERS,timeout=(12,35))
            if r.status_code in (401,403,429):raise PermissionError('公开来源访问限制：'+str(r.status_code))
            r.raise_for_status()
            return r.content,failures
        except requests.RequestException as exc:
            failures.append(type(exc).__name__+': '+str(exc))
            if n==1:raise
            time.sleep(1)
    raise RuntimeError('请求没有返回')


def reuse(identifier,scope):
    candidates=[]
    if scope=='csi':candidates.append(paths('financial')[0])
    candidates.append(ROOT/'reports/research/510300_forward_eps_source_pilot_v2')
    for old in candidates:
        p=old/'document_records'/(identifier+'.json')
        t=old/'page_texts'/(identifier+'.json')
        if p.exists() and t.exists():
            d=read(p)
            if 'source' in d and 'provider_metadata' in d:
                source=d['source']
                for key,hashkey in [('raw_pdf_path','pdf_sha256'),('raw_html_path','html_sha256')]:
                    if hashlib.sha256((ROOT/source[key]).read_bytes()).hexdigest()!=source[hashkey]:raise ValueError('原有原件缓存指纹变化')
                return source,d['provider_metadata'],read(t)['pages'],p.relative_to(ROOT).as_posix()
    return None


def one(scope,row):
    out,raw,manifest,queue=paths(scope)
    aid=row['infoCode'];receipt=out/'document_records'/(aid+'.json')
    if receipt.exists():return read(receipt)
    reused=reuse(aid,scope)
    try:
        if reused:
            source,meta,pages,reused_from=reused
        else:
            raw.mkdir(parents=True,exist_ok=True)
            url='https://data.eastmoney.com/report/zw_stock.jshtml?infocode='+aid
            hp=raw/(aid+'.html')
            html,herrors=(hp.read_bytes(),[]) if hp.exists() else request_bytes(url)
            hp.write_bytes(html)
            text=html.decode('utf-8-sig')
            match=re.search(r'var\s+zwinfo\s*=\s*',text)
            if not match:raise ValueError('公开页面没有研报元数据')
            meta=json.JSONDecoder().raw_decode(text[match.end():])[0]
            pdfurl=meta['attach_url']
            if not pdfurl.startswith('https://pdf.dfcfw.com/pdf/'):raise ValueError('附件域名不在公开来源范围')
            pp=raw/(aid+'.pdf')
            pdf,perrors=(pp.read_bytes(),[]) if pp.exists() else request_bytes(pdfurl)
            if not pdf.startswith(b'%PDF-'):raise ValueError('附件不是PDF原件')
            pp.write_bytes(pdf)
            doc=pdfium.PdfDocument(pdf);pages=[]
            try:
                for i in range(len(doc)):
                    page=doc[i];tp=page.get_textpage()
                    try:pages.append(tp.get_text_range())
                    finally:tp.close();page.close()
            finally:doc.close()
            source={'report_id':aid,'provider_detail_url':url,'pdf_url':pdfurl,'retrieved_at':now(),
                    'raw_html_path':hp.relative_to(ROOT).as_posix(),'raw_pdf_path':pp.relative_to(ROOT).as_posix(),
                    'pdf_sha256':hashlib.sha256(pdf).hexdigest(),'html_sha256':hashlib.sha256(html).hexdigest(),
                    'pdf_pages':len(pages),'prior_transport_errors':herrors+perrors}
            reused_from=None
        if meta['info_code']!=aid or str(meta['company_code'])!='80000007':raise ValueError('目录与附件编号或机构不符')
        stocks=[str(x.get('stock')) for x in meta['security']]
        if row['ts_code'][:6] not in stocks:raise ValueError('详情证券代码与登记对象不符')
        front=compact('\n'.join(pages[:2]))
        front_identity=row['ts_code'][:6] in front and ('guosen.com.cn' in front.lower() or '国信证券' in front)
        d={'report_id':aid,'ts_code':row['ts_code'],'directory_record':row,'source':source,
           'provider_metadata':meta,'reused_from':reused_from,'first_two_page_identity_match':front_identity,
           'status':'ARCHIVED_FRONT_IDENTITY_MATCH' if front_identity else 'ARCHIVED_FRONT_IDENTITY_PENDING',
           'new_eps_facts':0}
        save(out/'page_texts'/(aid+'.json'),{'source':source,'pages':pages},exclusive=True)
        save(receipt,d,exclusive=True)
        return d
    except Exception as exc:
        err={'report_id':aid,'ts_code':row['ts_code'],'directory_record':row,'at':now(),
             'status':'ORIGINAL_ARCHIVE_FAILED','error_type':type(exc).__name__,'error':str(exc)}
        save(out/'errors'/(aid+'.json'),err)
        if isinstance(exc,PermissionError):raise
        return err


def run(scope):
    out,raw,manifest,queue=paths(scope)
    for x in read(manifest)['files']:
        if identity(ROOT/x['path'])['sha256']!=x['sha256']:raise ValueError('登记范围或代码变化')
    if (out/'result.json').exists():raise FileExistsError('原件阶段已完成，不覆盖')
    rows=pd.read_parquet(out/'selected_before_originals.parquet').to_dict('records')
    done=[];restricted=False
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(one,scope,row):row for row in rows}
        for future in as_completed(futures):
            row=futures[future]
            try:
                d=future.result();done.append(d)
            except PermissionError as exc:
                restricted=True
                for f in futures:f.cancel()
                done.append({'report_id':row['infoCode'],'status':'ORIGINAL_ARCHIVE_FAILED','error_type':'PermissionError','error':str(exc)})
            except CancelledError:
                done.append({'report_id':row['infoCode'],'status':'SKIPPED_ACCESS_RESTRICTION'})
            except Exception as exc:
                done.append({'report_id':row['infoCode'],'status':'WORKER_FAILED','error_type':type(exc).__name__,'error':str(exc)})
            if len(done)%10==0:print('原始研报归档',scope,len(done),'/',len(rows),'已保存',sum('source' in x for x in done),flush=True)
    good=[x for x in done if 'source' in x]
    result={'completed_at':now(),'status':'ORIGINAL_REPORTS_ARCHIVED' if len(good)==len(rows) else 'ORIGINAL_REPORT_ARCHIVE_PARTIAL',
            'scope':scope,'selected_reports':len(rows),'archived_reports':len(good),
            'reused_reports':sum(bool(x.get('reused_from')) for x in good),
            'first_two_page_identity_matches':sum(x['first_two_page_identity_match'] for x in good),
            'pdf_pages':sum(x['source']['pdf_pages'] for x in good),
            'pdf_bytes':sum((ROOT/x['source']['raw_pdf_path']).stat().st_size for x in good),
            'archived_companies':len({x['ts_code'] for x in good}),'access_restriction_stop':restricted,
            'failures':[x for x in done if 'source' not in x],
            'new_eps_facts':0,'new_accounts':0,'goal_achieved':False}
    save(out/'result.json',result,exclusive=True);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['financial','csi'],required=True)
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');a=ap.parse_args()
    freeze(a.scope) if a.freeze else run(a.scope)
