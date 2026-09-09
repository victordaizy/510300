"""修正国信明确年度表头及每股收益行别名，保留预测、股本与日期证据。"""
from __future__ import annotations
from datetime import date
import re
from research.forward_eps_guosen_history_v1 import NUM,exact_row,internal_date,identity
from research.financial_annual_components_v1 import norm,compact,decimal


def annual_header(head):
    candidates=[]
    for match in re.finditer(r'(?m)^\s*((?:20\d{2}\s*[AEae]?\s+){2,5}20\d{2}\s*[AEae]?)\s*$',head):
        header=re.findall(r'(20\d{2})\s*([AEae]?)',match.group(1))
        candidates.append((header,match.group(1)))
    if len(candidates)!=1:raise ValueError('独立年度列标题未唯一识别：'+str(len(candidates)))
    return candidates[0]


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
            header,year_line=annual_header(head)
            if not 3<=len(header)<=6:continue
            yrs=[int(y) for y,f in header]
            if yrs!=list(range(yrs[0],yrs[0]+len(yrs))):continue
            predicted=[i for i,(y,f) in enumerate(header) if f.upper()=='E']
            if not predicted or predicted!=list(range(predicted[0],len(header))):continue
            end=re.search(r'资料来源',tail)
            body=tail[:end.start()] if end else tail
            eps=exact_row(body,['摊薄每股收益','每股收益'],len(header),'(?:元|摊薄)')
            eps['unit_as_reported']='元/股' if re.search(r'\(元\)',eps['raw']) else '原行仅标注摊薄，金额单位未明确'
            note_match=re.search(r'摊薄每股收益按最新总股本计算',compact(tail))
            if not note_match:raise ValueError('最新总股本口径附注缺失')
            optional={};optional_errors={}
            for key,labels,unit in [('net_profit',['归母净利润','净利润'],'百万元'),('pe',['市盈率'],'PE')]:
                try:optional[key]=exact_row(body,labels,len(header),unit)
                except ValueError as exc:optional_errors[key]=str(exc)
            tables.append({'page':page_number,'header':[[int(y),f.upper()] for y,f in header],
                           'eps':eps,'optional':optional,'optional_errors':optional_errors,
                           'basis_note':'摊薄每股收益按最新总股本计算','header_raw':year_line.strip(),'caption_before_header':head.strip()})
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
           'eps_value_exact':eps,'eps_unit_as_reported':t['eps']['unit_as_reported'],'source_eps_label':t['eps']['label'],'eps_currency_iso_independently_proven':False,
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


