"""收集公开的私募操作意向，区分所属月、调查仓位和公开时间。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_private_manager_intent_v1'

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def fetch_one(url):
    key=sha256(url.encode()).hexdigest()[:16]
    receipt=OUT/'receipts'/f'{key}.json'
    if receipt.exists():
        return read(receipt)
    rec={'url':url,'key':key,'retrieved_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'attempts':[]}
    for attempt in range(2):
        try:
            r=requests.get(url,timeout=20,headers={'User-Agent':'Mozilla/5.0'})
            rec['attempts'].append({'http_status':r.status_code})
            if r.status_code!=200:
                rec['status']='ACCESS_FAILED';break
            pdf=r.content.startswith(b'%PDF')
            fn='raw_local_only/'+key+('.pdf' if pdf else '.html')
            (OUT/fn).write_bytes(r.content)
            rec.update(status='SAVED',raw_path=fn,sha256=sha256(r.content).hexdigest(),bytes=len(r.content),kind='pdf' if pdf else 'html')
            if not pdf:
                s=BeautifulSoup(r.content,'html.parser',from_encoding=r.apparent_encoding)
                title=s.title.get_text(' ',strip=True) if s.title else ''
                rec['title']=title
                for bad in s(['script','style','nav','footer']):bad.decompose()
                full=s.get_text(' ',strip=True)
                article=s.select_one('#artibody') or s.select_one('#article')
                body=article.get_text(' ',strip=True) if article else full
                rec['text_path']='raw_local_only/'+key+'.txt'
                (OUT/rec['text_path']).write_text(body,encoding='utf-8')
                rec['header_text_local_only']=full[:1800]
                dates=[]
                for y,m,d in re.findall(r'(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})(?:日)?',full[:2500]):
                    try: dates.append(str(pd.Timestamp(int(y),int(m),int(d)).date()))
                    except ValueError:continue
                rec['header_date_candidates']=list(dict.fromkeys(dates))[:8]
            else:
                doc=pdfium.PdfDocument(OUT/fn)
                texts=[]
                for page in doc:
                    textpage=page.get_textpage();texts.append(textpage.get_text_range());textpage.close();page.close()
                rec['metadata']=doc.get_metadata_dict();doc.close()
                rec['text_path']='raw_local_only/'+key+'_pages.json'
                dump(OUT/rec['text_path'],texts)
            break
        except Exception as exc:
            rec['attempts'].append({'error':str(exc)});rec['status']='ACCESS_FAILED'
    dump(receipt,rec)
    return rec

def fetch():
    (OUT/'receipts').mkdir(exist_ok=True)
    urls=sorted(set(u for p in OUT.glob('*candidate_urls.json') for u in read(p)))
    records=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs={pool.submit(fetch_one,u):u for u in urls}
        for job in as_completed(jobs):
            rec=job.result();records.append(rec)
            if len(records)%20==0 or rec['status']!='SAVED' or len(records)==len(urls):
                print(f"来源进度 {len(records)}/{len(urls)}：{rec['status']}",flush=True)
    dump(OUT/'receipts/source_downloads.json',records)
    print('公开来源保存完成',flush=True)

def extract():
    rows=[]
    for rec in read(OUT/'receipts/source_downloads.json'):
        if rec['status']!='SAVED':continue
        if rec['kind']=='pdf':
            pages=read(OUT/rec['text_path']);text='\n'.join(pages);head=pages[0][:2300]
        else:
            text=(OUT/rec['text_path']).read_text(encoding='utf-8');head=rec.get('header_text_local_only','')
        compact=re.sub(r'\s+','',text)
        plans=list(re.finditer(r'仓位增减(?:持)?投资计划(?:指标|指数)(?:值)?(?:为|是|[:：])?([\d]+(?:\.\d+)?)',compact))
        plan_values=list(dict.fromkeys(float(m[1]) for m in plans))
        avg=list(re.finditer(r'(?:平均仓位|整体平均仓位)(?:为|在|处于|达到)([\d]+(?:\.\d+)?)%',compact))
        avg_values=list(dict.fromkeys(float(m[1]) for m in avg))
        confidence=list(re.finditer(r'A股信心指数(?:值)?(?:为|是)([\d]+(?:\.\d+)?)',compact))
        trend=list(re.finditer(r'市场(?:市场)?趋势预期信心指标(?:值)?(?:为|是)([\d]+(?:\.\d+)?)',compact))
        month_matches=re.findall(r'(20\d{2})年(\d{1,2})月',re.sub(r'\s+','',head))
        month_guesses=list(dict.fromkeys(f'{int(y):04d}-{int(m):02d}' for y,m in month_matches if 1<=int(m)<=12))[:10]
        snippets=[]
        for m in plans+avg:
            snippets.append(compact[max(0,m.start()-70):m.end()+35])
        rows.append({'key':rec['key'],'url':rec['url'],'kind':rec['kind'],'title':rec.get('title',''),
                     'header_dates':'|'.join(rec.get('header_date_candidates',[])),
                     'month_candidates':'|'.join(month_guesses),'plan_values':'|'.join(map(str,plan_values)),
                     'average_exposure_values':'|'.join(map(str,avg_values)),
                     'confidence_values':'|'.join(dict.fromkeys(m[1] for m in confidence)),
                     'trend_values':'|'.join(dict.fromkeys(m[1] for m in trend)),
                     'evidence_snippets':' || '.join(snippets[:5]),'manual_admission':'PENDING_SOURCE_REVIEW'})
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'字段候选.csv',index=False,encoding='utf-8-sig')
    dump(OUT/'field_candidates.json',rows)
    catalogue=read(OUT/'raw_local_only/api_search.json')['data']['data']
    pd.DataFrame([{k:r[k] for k in ['id','report_title','post_time','cover_date','read_auth']} for r in catalogue]).to_csv(OUT/'原发布方公开目录.csv',index=False,encoding='utf-8-sig')
    print('已保存来源',len(frame),'含计划数值',frame.plan_values.ne('').sum(),'含调查仓位',frame.average_exposure_values.ne('').sum(),flush=True)
    print(frame.loc[frame.plan_values.ne(''),['key','month_candidates','header_dates','plan_values','average_exposure_values']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='私募操作意向公开资料');p.add_argument('action',choices=['fetch','extract'])
    globals()[p.parse_args().action]()
