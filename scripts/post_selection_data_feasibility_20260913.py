"""一次取得既有免费来源并盘点时间边界，不修改冻结数据或计算策略收益。"""
import concurrent.futures
import hashlib
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.official_dividend_coverage_refresh_v1 import calendar_dates, check_http_clock, compare_manager, completed_trade_day, ledger_rows, parse_manager, validate_announcements

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_post_selection_data_feasibility_v1'
CONFIG=ROOT/'config/510300_post_selection_data_feasibility_v1.json'
PRIMARY='SELECTED_MIX_BAND10_SIMPLE2'
SOURCES={'SELECTED_MIX_BAND10_SIMPLE2':'510300_incremental_selected_intent_mix_v1',
    'ADD_GATE_EXPOSURE_115':'510300_addition_gate_exposure_batch_v1','RUNS_OPPORTUNITY_CAPPED_SUM':'510300_runs_opportunity_union_v1'}
TZ=ZoneInfo('Asia/Shanghai')


def fetch(name,url,params,referer):
    start=now()
    receipt={'source':name,'url':url,'params':params,'started_at':start,'request_count':1,'cost_cny':0,'tls_verified':True}
    try:
        response=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0','Referer':referer},timeout=(8,20),allow_redirects=False)
        body=response.content
        require(len(body)<=4_000_000,'来源响应过大')
        path=OUT/(name+'.raw')
        with path.open('xb') as stream:
            stream.write(body)
        receipt.update(retrieved_at=now(),http_status=response.status_code,final_url=response.url,server_date=response.headers.get('Date'),
            headers=dict(response.headers),raw_file=str(path.relative_to(ROOT)),raw_sha256=digest(path),raw_bytes=len(body))
        response.raise_for_status()
        require(response.status_code==200,'来源没有直接成功响应')
        receipt['status']='HTTP_OK_UNPARSED'
    except Exception as exc:
        receipt.update(status='EXTERNAL_FREE_SOURCE_FAILED',error_type=type(exc).__name__,error=str(exc),finished_at=now())
    write_json(OUT/(name+'_receipt.json'),receipt,exclusive=True)
    return receipt


def parse_prices(name,body,end):
    if name=='sina':
        globals_=ak.fund_etf_hist_sina.__globals__
        vm=globals_['py_mini_racer'].MiniRacer()
        vm.eval(globals_['hk_js_decode'])
        encoded=body.decode('utf-8').split('=')[1].split(';')[0].replace('"','')
        data=pd.DataFrame(vm.call('d',encoded))
    else:
        text=body.decode('utf-8')
        value=ak.stock_zh_a_hist_tx.__globals__['demjson'].decode(text[text.find('{'):])
        rows=value['data']['sh510300']['day']
        data=pd.DataFrame([{'date':r[0],'open':r[1],'close':r[2],'high':r[3],'low':r[4],
            'volume':float(r[5])*100,'amount':float(r[8])*10000 if len(r)>8 and r[8] not in ['',None] else np.nan} for r in rows])
    require(all(c in data for c in ['date','open','close','high','low','volume']),'行情必要字段缺失')
    data['date']=pd.to_datetime(data.date,errors='raise').dt.tz_localize(None).dt.normalize()
    if 'amount' not in data:
        data['amount']=np.nan
    for column in ['open','close','high','low','volume','amount']:
        data[column]=pd.to_numeric(data[column],errors='coerce')
    data=data[data.date.between('2026-08-01',end)].sort_values('date').reset_index(drop=True)
    require(len(data)>0 and not data.date.duplicated().any(),'行情为空或交易日重复')
    required=data[['open','close','high','low','volume']]
    require(np.isfinite(required.to_numpy()).all() and (required.iloc[:,:4]>0).all().all() and (data.volume>=0).all(),'价格或成交量非法')
    require((data.high>=data[['open','close','low']].max(axis=1)).all() and (data.low<=data[['open','close','high']].min(axis=1)).all(),'最高最低关系错误')
    data['symbol']='510300.SH'
    data['source']=name
    data.to_parquet(OUT/(name+'_daily.parquet'),index=False)
    return data


def main():
    require(not OUT.exists() and not CONFIG.exists(),'数据前提检查已开始，请接续保存输出而非重新请求')
    index_path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==210 and not index['running_studies'],'前序状态不同')
    now_dt=datetime.now(TZ)
    cal_path=ROOT/'data/reference/sse_trade_calendar_2026.csv'
    calendar=calendar_dates(cal_path)
    completed=completed_trade_day(calendar,now_dt)
    old_path=ROOT/'config/510300_incremental_selected_intent_mix_v1.json'
    old=json.loads(old_path.read_text(encoding='utf-8'))
    official_path=ROOT/'config/510300_official_dividend_coverage_refresh_v1.json'
    official=json.loads(official_path.read_text(encoding='utf-8'))
    coverage_path=ROOT/'data/reference/510300_dividends_coverage.json'
    coverage=json.loads(coverage_path.read_text(encoding='utf-8'))
    frozen_at=pd.Timestamp(old['registered_at'])
    OUT.mkdir(parents=True)
    cfg={k:old[k] for k in ['evaluation_start','data_cutoff','initial_capital','annual_days','cash_annual_rate_assumption',
        'high_sharpe_target','annual_return_target','costs','features','dividends','earlier_start','earlier_terminal']}
    cfg.update(study_id='510300_POST_SELECTION_DATA_FEASIBILITY_V1',round=211,registered_at=now(),primary=PRIMARY,
        candidate_models=[],candidate_configurations=0,rules='docs/510300_POST_SELECTION_DATA_FEASIBILITY_V1.md',
        strategy_freeze_time=old['registered_at'],last_completed_market_date=completed.isoformat(),
        new_model_fits=0,new_reference_accounts=0,new_trading_accounts=0,source_requests_per_endpoint=1,
        goal_achieved=False,independent_validation='NOT_ESTABLISHED',position_impact=0,
        prior_goal_turn_classification='PROGRESS_ROUND210_COMPLETED_FIXED_STABILITY_DIAGNOSTIC')
    text=(ROOT/'docs/510300_POST_SELECTION_DATA_FEASIBILITY_NEXT_20260913.md').read_text(encoding='utf-8')
    text=text.replace('本项目前只准备规则，尚未实现、请求新来源或计算后续收益。','本项在首次外部请求前登记；只做数据前提检查，不计算后续策略收益。')
    (ROOT/cfg['rules']).write_text(text,encoding='utf-8')
    paths=[Path(__file__),ROOT/cfg['rules'],old_path,official_path,coverage_path,ROOT/cfg['dividends'],cal_path,
        Path(inspect.getfile(ak.fund_etf_hist_sina)),Path(inspect.getfile(ak.stock_zh_a_hist_tx)),ROOT/'research/official_dividend_coverage_refresh_v1.py']
    for model,folder in SOURCES.items():
        paths += [ROOT/'reports/research'/folder/p/c/f'{model}_{kind}.parquet' for p in ['evaluation','earlier_diagnostic'] for c in ['BASE','STRESS'] for kind in ['ledger','decisions']]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG),'request_limit':4,'new_accounts':0},exclusive=True)
    write_json(OUT/'tests_receipt.json',{'recorded_at':now(),'exit_code':0,'passed':0,'method':'只读资料检查，不新增单元测试；使用既有解析器和运行时数据校验'},exclusive=True)
    index['running_studies']=[{'round':211,'study':cfg['study_id'],'status':'SOURCE_PREFLIGHT_RUNNING','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='POST_SELECTION_DATA_FEASIBILITY_RUNNING',source=cfg['rules'])
    write_json(index_path,index)
    began=time.perf_counter()
    end=now_dt.date().isoformat()
    params={'isPagination':'true','pageHelp.pageSize':25,'pageHelp.pageNo':1,'pageHelp.beginPage':1,'pageHelp.cacheSize':1,
        'pageHelp.endPage':1,'type':'inParams','sqlId':official['sse_sql_id'],'TITLE':'','SECURITY_CODE':'510300',
        'BULLETIN_TYPE':'','START_DATE':official['sse_query_start'],'END_DATE':end,'DATE_DESC':1,'DATE_ASC':'','CODE_DESC':'','CODE_ASC':''}
    requests_=[('sina','https://finance.sina.com.cn/realstock/company/sh510300/hisdata_klc2/klc_kl.js',None,'https://finance.sina.com.cn/'),
        ('tencent','https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get',
         {'_var':'kline_day2026','param':f'sh510300,day,2026-08-01,{end},640,','r':'0.8205512681390605'},'https://gu.qq.com/'),
        ('manager',official['manager_url'],None,'https://www.huatai-pb.com/'),
        ('sse',official['sse_query_url'],params,'https://www.sse.com.cn/')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(lambda args:fetch(*args),requests_))
    prices,parsing={},[]
    for receipt in receipts:
        if receipt['source'] not in ['sina','tencent']:
            continue
        name=receipt['source']
        try:
            require(receipt['status']=='HTTP_OK_UNPARSED','来源请求失败')
            prices[name]=parse_prices(name,(ROOT/receipt['raw_file']).read_bytes(),completed.isoformat())
            d=prices[name]
            parsing.append({'source':name,'status':'PARSED_VALID_PRICES','rows':len(d),'last_date':str(d.date.max().date()),
                'rows_after_old_cutoff':int(d.date.gt('2026-08-14').sum()),'amount_missing_rows':int(d.amount.isna().sum())})
        except Exception as exc:
            parsing.append({'source':name,'status':'NO_VALID_NEW_PRICE_TABLE','error_type':type(exc).__name__,'error':str(exc)})
    expected=[pd.Timestamp(d) for d in calendar if pd.Timestamp('2026-08-14').date()<d<=completed]
    comparison={'status':'NOT_COMPUTED_MISSING_SOURCE','expected_new_dates':[str(x.date()) for x in expected]}
    if len(prices)==2:
        left=prices['sina'].set_index('date')
        right=prices['tencent'].set_index('date')
        common=left.index.intersection(right.index)
        records=[]
        for day in common:
            for field in ['open','high','low','close']:
                records.append({'date':day,'field':field,'sina':left.loc[day,field],'tencent':right.loc[day,field],
                    'absolute_difference':abs(left.loc[day,field]-right.loc[day,field])})
        differences=pd.DataFrame(records)
        differences.to_csv(OUT/'price_crosscheck.csv',index=False,encoding='utf-8-sig')
        missing={name:[str(d.date()) for d in expected if d not in frame.date.values] for name,frame in prices.items()}
        exact=bool(differences.absolute_difference.le(1e-9).all())
        comparison={'status':'PASS_OHLC_AND_NEW_DATE_COVERAGE' if exact and not any(missing.values()) else 'PRICE_DIFFERENCE_OR_DATE_GAP',
            'common_dates':len(common),'maximum_ohlc_difference':float(differences.absolute_difference.max()),
            'missing_new_dates':missing,'expected_new_dates':[str(x.date()) for x in expected]}
    dividend={'status':'NOT_CONFIRMED','previous_coverage_end':coverage['coverage_end'],'requested_coverage_end':completed.isoformat()}
    try:
        official_receipts={r['source']:r for r in receipts if r['source'] in ['manager','sse']}
        for r in official_receipts.values():
            require(r['status']=='HTTP_OK_UNPARSED','官方来源请求失败')
            check_http_clock(r['headers'],datetime.fromisoformat(r['retrieved_at']),completed,official)
        rows=parse_manager((ROOT/official_receipts['manager']['raw_file']).read_bytes(),'510300')
        compare_manager(rows,ledger_rows(ROOT/cfg['dividends']))
        page=json.loads((ROOT/official_receipts['sse']['raw_file']).read_bytes())
        checked=validate_announcements([page],official,now_dt.date(),coverage)
        dividend.update(status='OFFICIAL_CANDIDATE_COVERAGE_CONFIRMED_NO_LEDGER_CHANGE',manager_events=len(rows),**checked)
    except Exception as exc:
        dividend.update(status='NOT_CONFIRMED',error_type=type(exc).__name__,error=str(exc))
    write_json(OUT/'dividend_candidate_coverage.json',dividend,exclusive=True)
    inventory=[]
    for model,folder in SOURCES.items():
        for period in ['evaluation','earlier_diagnostic']:
            for cost in ['BASE','STRESS']:
                base=ROOT/'reports/research'/folder/period/cost
                ledger=pd.read_parquet(base/f'{model}_ledger.parquet')
                decisions=pd.read_parquet(base/f'{model}_decisions.parquet')
                previous,last,request=ledger.iloc[-2],ledger.iloc[-1],decisions.iloc[-1]
                inventory.append({'model':model,'period':period,'cost':cost,'last_normal_close':str(previous.date.date()),
                    'last_normal_cash':float(previous.cash),'last_normal_shares':int(previous.shares),'last_normal_equity':float(previous.equity),
                    'last_normal_dividend_receivable':float(previous.dividend_receivable),'terminal_date':str(last.date.date()),
                    'terminal_mark_clock':str(last.mark_clock),'source_normal_target':float(request.reference_weight),
                    'source_normal_request':int(request.requested_quantity),'terminal_actual_fill':int(last.filled_quantity),
                    'terminal_ending_shares':int(last.shares),'terminal_fill_differs_from_normal_request':int(last.filled_quantity)!=int(request.requested_quantity),
                    'complete_recursive_source_checkpoint_verified':False})
    pd.DataFrame(inventory).to_csv(OUT/'terminal_state_inventory.csv',index=False,encoding='utf-8-sig')
    times={'strategy_frozen_at':frozen_at.isoformat(),'old_history_cutoff':'2026-08-14','latest_completed_day':completed.isoformat(),
        'expected_gap_dates_before_strategy_freeze':[str(d.date()) for d in expected if d.date()<frozen_at.date()],
        'expected_completed_dates_after_strategy_freeze':[str(d.date()) for d in expected if d.date()>frozen_at.date()],
        'strict_forward_evidence_days':0,'decisions_precommitted_for_new_dates':0,
        'gap_class':'PRE_SELECTION_HISTORICAL_GAP_NOT_STRICT_FORWARD','first_future_calendar_date':next(d.isoformat() for d in calendar if d>now_dt.date())}
    write_json(OUT/'time_boundaries.json',times,exclusive=True)
    old_result=json.loads((ROOT/'reports/research/510300_incremental_selected_intent_mix_v1/result.json').read_text(encoding='utf-8'))
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'COMPLETED_FREE_SOURCE_AND_TIME_BOUNDARY_PREFLIGHT',
        'candidate_configurations':0,'candidate_models':[],'evaluation_accounts':0,'new_accounts_generated':0,'reused_control_accounts':0,
        'earlier_diagnostic_accounts':0,'new_earlier_diagnostic_accounts':0,'new_model_fits':0,'new_reference_accounts':0,
        'all_metrics':[r for r in old_result['all_metrics'] if r['model']==PRIMARY],
        'earlier_diagnostics':[r for r in old_result['earlier_diagnostics'] if r['model']==PRIMARY],
        'metric_context_only_reused_from_round209':True,'primary':PRIMARY,
        'post_selected_best_base':next(r for r in old_result['all_metrics'] if r['model']==PRIMARY and r['cost']=='BASE'),
        'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','position_impact':0,'run_seconds':time.perf_counter()-began,
        'requests':[{k:v for k,v in r.items() if k!='headers'} for r in receipts],'price_parsing':parsing,'price_comparison':comparison,
        'dividend_coverage':dividend,'time_boundaries':times,'source_state_inventory_rows':len(inventory),
        'data_accepted_into_strategy':False,'new_strategy_performance_computed':False,'source_cost_cny':0}
    write_json(OUT/'result.json',result,exclusive=True)
    for item in cfg['frozen_files']:
        path=Path(item['path'])
        require(digest(path if path.is_absolute() else ROOT/path)==item['sha256'],'原冻结输入发生改变')
    write_json(OUT/'saved_verification_receipt.json',{'verified_at':now(),'status':'PASS_SAVED_SOURCE_RECEIPTS_TIME_SPLIT_AND_TERMINAL_STATE_INVENTORY',
        'requests_attempted':4,'source_state_inventory_rows':12,'new_accounts':0,'new_performance_metrics':0,
        'old_frozen_inputs_unchanged':True,'source_success_is_reported_separately':True,'independent_performance_validation':False},exclusive=True)
    print(json.dumps({k:result[k] for k in ['status','run_seconds','price_parsing','price_comparison','dividend_coverage','time_boundaries']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
