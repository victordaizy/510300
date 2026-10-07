"""私募操作意向的固定历史诊断；意向、仓位和真实资金流分别记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_private_manager_intent_v1'
MARKET = ROOT / 'reports/research/510300_participant_identity_clock_v1/inputs/market.parquet'

# 月份、可用日期上界、计划指数、同一月度调查仓位、来源键、时钟类别、备注。
# 原发布方目录的日期只支持当前版本历史重建，不证明历史首版内容。
PLAN_ROWS = [
    ('2014-10','2014-09-30',100.98,None,'f527a100cd3e3822','DATED_ARTICLE','正文明确询问接下来的十月'),
    ('2015-12','2015-12-03',106.94,None,'b994ab53cbc7d3f8','DATED_ARTICLE','不以文内十二月二日讯提前转载可用时间'),
    ('2016-12','2016-12-08',97.,None,'fd8f27c28da3ef86','DATED_ARTICLE','未报告同口径平均仓位'),
    ('2018-07','2018-07-04',119.3,56.21,'926e4ffdd04dd574','DATED_ARTICLE',''),
    ('2018-08',None,108.55,57.82,'c694140019bc46c2','NO_PUBLICATION_CLOCK','封面日期不作实际公布日'),
    ('2019-05','2019-05-07',106.03,74.09,'d8160a97b8230060','DATED_ARTICLE',''),
    ('2019-09','2019-09-09',112.88,71.98,'f98817573a887954','DATED_ARTICLE','九月三日原发布方文章没有计划指数，不提前使用'),
    ('2019-10','2019-10-09',106.31,64.54,'58110ffbb2c93d46','DATED_ARTICLE',''),
    ('2020-02','2020-02-05',115.54,71.83,'89deb40d80f62720','DATED_ARTICLE','重复转载合并为一个月'),
    ('2020-08','2020-08-03',106.89,79.11,'b0f1f9cc4cdb476e','DATED_ARTICLE',''),
    ('2021-05','2021-05-06',109.43,78.,'50b22889ce570541','DATED_ARTICLE',''),
    ('2021-07','2021-07-02',101.28,83.,'7bd0ced372a450c3','DATED_ARTICLE',''),
    ('2021-08','2021-08-02',110.24,80.,'f929fd0b65c5e131','DATED_ARTICLE',''),
    ('2021-11','2021-11-01',107.69,82.,'4e3d671d57211f33','DATED_ARTICLE',''),
    ('2022-01','2022-01-25',108.99,83.,'9d7bfe578adb9ccb','PUBLIC_CATALOGUE_CURRENT_PDF','采用目录上架日，不能提前到封面日期'),
    ('2022-05','2022-05-22',120.27,70.,'9a69142f4a115f0b','PUBLIC_CATALOGUE_CURRENT_PDF','五月报告位于七月命名路径；首版内容未认证'),
    ('2023-04','2023-04-04',112.5,None,'4bd2fd511d1cbcfd','DATED_ARTICLE','文内周频测算仓位不混入月度调查仓位'),
    ('2023-05','2023-05-28',100.,76.,'edd19f1768b416e8','PUBLIC_CATALOGUE_CURRENT_PDF','采用目录上架日'),
    ('2023-06','2023-06-04',108.11,79.,'95f0b53e98156691','DATED_ARTICLE',''),
    ('2023-08','2023-08-01',118.98,None,'08218b4fa1fd0a08','DATED_ARTICLE','未报告同口径平均仓位'),
    ('2023-09','2023-09-05',116.19,None,'b2ef652de2d01f68','DATED_ARTICLE','按数值记录，不沿用正文多数人减仓这一矛盾概括'),
    ('2024-03','2024-03-20',115.5,77.,'5feb94d4da85bb17','PUBLIC_CATALOGUE_CURRENT_PDF','三月二日报道只有总信心和仓位，不提前计划指数'),
    ('2024-07','2024-08-01',112.89,78.,'c283fa644e6b8941','PUBLIC_CATALOGUE_CURRENT_PDF','发布上界已晚于意向月份，只作验证'),
    ('2024-08','2024-08-15',111.64,73.,'adc8c93add062262','PUBLIC_CATALOGUE_CURRENT_PDF','采用目录上架日'),
    ('2024-09','2024-09-04',107.64,73.,'61f21c6e990061c3','DATED_ARTICLE','九月四日已有具体数字，不用九月二十九日目录时间'),
    ('2024-10','2024-10-09',118.63,78.,'d9e8e2cf00ee378c','DATED_ARTICLE','页面署期十月九日，晚于URL十月八日'),
    ('2024-11','2024-11-01',113.73,78.,'0380dc61c7bbd54a','PUBLIC_CATALOGUE_CURRENT_PDF','采用正文当期原值，不依照后期环比修正'),
    ('2024-12','2024-12-02',113.75,78.,'a491fb81fa6cd8ee','PUBLIC_CATALOGUE_CURRENT_PDF','保留原始小数差异'),
    ('2025-01','2025-01-10',113.67,76.,'9b74b1cfbdb96271','DATED_ARTICLE',''),
    ('2025-03','2025-03-03',114.56,79.,'f08973a6cc50eac0','DATED_ARTICLE',''),
    ('2025-07','2025-07-07',113.73,77.,'05ab02abe7d84d6a','DATED_ARTICLE',''),
    ('2025-08','2025-08-01',112.72,77.,'ea756fcb993501f8','PUBLIC_CATALOGUE_CURRENT_PDF','标题八月但计划段写七月，计划数值不进入主诊断'),
    ('2025-09','2025-09-02',112.95,78.,'68ccfbb7610b7097','DATED_ARTICLE',''),
    ('2026-01','2026-01-12',111.97,78.,'6c6443d8effade18','PUBLIC_CATALOGUE_CURRENT_PDF',''),
    ('2026-02','2026-02-03',111.34,79.,'84c3fd39c51b30e9','PUBLIC_CATALOGUE_CURRENT_PDF',''),
    ('2026-03','2026-03-04',111.86,78.87,'8adcacc913c3e295','PUBLIC_CATALOGUE_CURRENT_PDF',''),
]

# 补充同一月度调查的仓位；仅供下一期报告仓位变化验证，绝不作为早期输入。
EXPOSURE_ROWS = [
    ('2018-10','2018-10-09',62.5,'9d8a6b66e8890cf5'),
    ('2019-07','2019-07-02',72.46,'457ebfbacfc5ba90'),
    ('2019-08','2019-08-05',71.92,'cff9c0c49d9439ba'),
    ('2019-11','2019-11-04',66.55,'18f29da55e16d6a9'),
    ('2020-03','2020-03-03',73.26,'7c3634ac7e96fe54'),
    ('2020-05','2020-05-06',71.35,'2f7ccc41a4de6804'),
    ('2020-07','2020-07-02',79.11,'bb9cc826c37824cd'),
    ('2020-10','2020-10-12',82.59,'ca651884a5ec2d82'),
    ('2021-01','2021-01-06',83.,'aa011b4bed7b60d6'),
    ('2021-12','2021-12-03',81.,'a123fea9d3811f50'),
    ('2022-04','2022-04-08',67.,'8dad3a413d72b3dd'),
    ('2022-09','2022-09-03',78.,'c9dd32baa4b99c79'),
    ('2023-10','2023-10-10',80.,'f55ce6627bdc4650'),
    ('2025-02','2025-02-10',77.,'92ac54fe2d9028fb'),
]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate():
    if (OUT / 'protocol.json').exists():
        raise ValueError('数值协议已冻结，禁止改写来源准入清单')
    rows=[]
    for month, date, plan, exposure, key, clock, note in PLAN_ROWS:
        rec=read(OUT/'receipts'/f'{key}.json')
        cand=next(r for r in read(OUT/'field_candidates.json') if r['key']==key)
        assert plan in [float(x) for x in cand['plan_values'].split('|')], key
        if exposure is not None:
            assert exposure in [float(x) for x in cand['average_exposure_values'].split('|')], key
        status='ADMITTED_HISTORICAL_RECONSTRUCTION'
        if date is None: status='VALIDATION_ONLY_NO_PUBLICATION_CLOCK'
        elif date[:7]>month: status='VALIDATION_ONLY_STALE_AFTER_INTENT_MONTH'
        elif month=='2025-08': status='VALIDATION_ONLY_PLAN_MONTH_CONFLICT'
        rows.append(dict(month=month, available_date=date, plan_index=plan, survey_exposure_pct=exposure, key=key,
                         url=rec['url'], source_sha256=rec['sha256'], clock=clock, status=status, note=note,
                         archived_first_version=False, actual_cash_flow=False))
    df=pd.DataFrame(rows)
    assert df.month.is_unique
    df.to_csv(OUT/'月度意向人工准入.csv',index=False,encoding='utf-8-sig')
    dump(OUT/'admitted_sources.json',rows)
    exposure_rows=[dict(month=r['month'],available_date=r['available_date'],survey_exposure_pct=r['survey_exposure_pct'],key=r['key'])
                   for r in rows if r['survey_exposure_pct'] is not None]
    for month,date,value,key in EXPOSURE_ROWS:
        rec=read(OUT/'receipts'/f'{key}.json')
        exposure_rows.append(dict(month=month,available_date=date,survey_exposure_pct=value,key=key))
    e=pd.DataFrame(exposure_rows).sort_values('month')
    assert e.month.is_unique
    e.to_csv(OUT/'月度调查仓位.csv',index=False,encoding='utf-8-sig')
    months=pd.DataFrame({'month':pd.period_range('2014-01','2026-08',freq='M').astype(str)})
    months=months.merge(df[['month','available_date','plan_index','survey_exposure_pct','status']],on='month',how='left',validate='one_to_one')
    months.status=months.status.fillna('MISSING_NOT_ZERO')
    months.to_csv(OUT/'全部月份覆盖与缺口.csv',index=False,encoding='utf-8-sig')
    keys=set(df.key)|set(e.key)
    safe=[]
    for r in read(OUT/'receipts/source_downloads.json'):
        safe.append({k:v for k,v in r.items() if k not in ['header_text_local_only','metadata']})
        safe[-1]['used_in_curated_fields']=r['key'] in keys
    dump(OUT/'来源链接与下载记录.json',safe)
    print('独立意向月份',len(df),'历史诊断可用',int(df.status.str.startswith('ADMITTED').sum()),'调查仓位月份',len(e),'全期月份',len(months),flush=True)


def freeze():
    protocol=OUT/'protocol.json'
    if protocol.exists():
        raise ValueError('协议已经冻结')
    (OUT/'inputs').mkdir(exist_ok=True)
    # 原样复制文件，不读市场值；此时才固定标签和模型。
    (OUT/'inputs/market.parquet').write_bytes(MARKET.read_bytes())
    spec={
        'study_id':'510300_PRIVATE_MANAGER_INTENT_V1','frozen_at':now(),
        'role':'有限历史机制诊断和小样本预测检查；调查意向不是成交指令，仓位变化不等于真实净买入',
        'research_selection':'在多年已研究历史上新增调查意向来源。属于开发记录，不是未见市场或严格前向。继承整个研究项目的选择偏差。',
        'frozen_source_files':{n:sha(OUT/n) for n in ['admitted_sources.json','月度调查仓位.csv','全部月份覆盖与缺口.csv','inputs/market.parquet']},
        'availability':'公开日期一律按当日结束，下一交易日开盘开始评价。目录加当前PDF仅为历史重建；同时固定仅有日期公开正文的子样本。',
        'publication_month':'无日期、晚于计划所属月、计划月份矛盾不进入主诊断；保留验证栏。缺失月份不插值。',
        'market_label':{'horizon_trading_days':20,'entry':'strictly_next_open','exit':'20th_trading_day_close','dividends':'入场日除息不享有，之后到退出日按每份现金分红计入应收；不再投资','gross_only':True},
        'features':{'prior_return20':'最近已收盘日含分红财富/此前20日财富-1','logvol20':'此前20个日总回报标准差乘sqrt(242)取对数','intent':'(计划指数-100)/100','headroom':'1-调查仓位/100，仅为可加仓空间粗略代理','intent_headroom':'intent*headroom'},
        'descriptive_tests':['intent对之前20日收益','intent对随后20日收益','intent_headroom对随后20日收益','intent对下期月度调查仓位变化','intent_headroom对下期月度调查仓位变化'],
        'descriptive_method':'Spearman秩相关；全体可用和有日期正文子集均报告；逐年剔除范围仅作集中度诊断，不选子样本',
        'behavior_label':'严格相邻报告月份的调查平均仓位之差；比较的下次报告所属仓位日必须不早于当前可用日期。不是资金净流量，可能受价格和样本构成影响。',
        'fixed_models':{'A':['prior_return20','logvol20'],'B':['prior_return20','logvol20','intent'],'AC':['prior_return20','logvol20'],'C':['prior_return20','logvol20','intent','headroom','intent_headroom']},
        'estimator':{'type':'Ridge','alpha':10.,'scaler':'仅当时成熟训练样本的均值及标准差','fit_intercept':True,'minimum_mature_training_events':24,'no_tuning':True},
        'evaluation':'逐时点扩展训练，仅标签结束日早于决策日者入训练。A/B使用相同计划可用训练及评价月份；AC/C使用相同意向和仓位完整训练及评价月份。AC是为C重新匹配训练样本的价格基准，不是另一个候选。同期成熟均值和恒零收益基准也报告。',
        'promotion_gate':{'minimum_evaluation_events':24,'minimum_evaluation_span_years':3.,'required':'相对A和成熟训练均值的MSE均降低，按月份顺序3事件移动区块5000次重采样的成对改进95%区间下界均大于0；至少三个评价年，去掉任一年两项改进仍为正；有日期正文子集方向一致。未达到样本门槛时，不计算显著性或宣布正负作用。'},
        'accounts':'通过上述门槛后才能另立账户协议。本协议不授权账户回测。','capital_main':200000,'capital_comparison':20000,'annualization':242,
        'stop':'固定本轮来源、变量、模型、期限和方向；不反号或换窗口救援。新增资料另登记版本。',
        'new_labels_read_before_freeze':0,'new_model_fits_before_freeze':0,'strict_forward_days':0,
    }
    dump(protocol,spec)
    dump(OUT/'freeze_receipt.json',{'frozen_at':now(),'protocol_sha256':sha(protocol),'analysis_code_sha256':sha(Path(__file__))})
    print('数值协议已冻结；未来收益尚未读取；不运行账户',flush=True)


def check_frozen():
    spec=read(OUT/'protocol.json'); receipt=read(OUT/'freeze_receipt.json')
    assert sha(OUT/'protocol.json')==receipt['protocol_sha256']
    for name,digest in spec['frozen_source_files'].items():assert sha(OUT/name)==digest,name
    return spec


def build_panel():
    spec=check_frozen()
    m=pd.read_parquet(OUT/'inputs/market.parquet').sort_values('date').reset_index(drop=True)
    m.date=pd.to_datetime(m.date)
    assert m.date.is_unique and m.date.is_monotonic_increasing
    m['prior_return20']=m.wealth/m.wealth.shift(20)-1
    m['logvol20']=np.log(m.total_simple.rolling(20).std(ddof=1)*np.sqrt(242))
    e=pd.read_csv(OUT/'月度调查仓位.csv').set_index('month')
    rows=[]
    for r in read(OUT/'admitted_sources.json'):
        if not r['status'].startswith('ADMITTED'):continue
        r=r.copy();pub=pd.Timestamp(r['available_date']);positions=np.flatnonzero(m.date.gt(pub))
        if not len(positions):continue
        entry=int(positions[0]); end=entry+spec['market_label']['horizon_trading_days']-1
        if entry<21 or end>=len(m):continue
        last=m.iloc[entry-1]; close=m.iloc[end]; first=m.iloc[entry]
        r.update(decision_date=str(last.date.date()),entry_date=str(first.date.date()),exit_date=str(close.date.date()),
                 entry_open=float(first.open),exit_close=float(close.close),prior_return20=float(last.prior_return20),logvol20=float(last.logvol20),
                 holding_dividend=float(m.iloc[entry+1:end+1].dividend.sum()))
        r['forward_return20']=(r['exit_close']+r['holding_dividend'])/r['entry_open']-1
        r['intent']=(r['plan_index']-100)/100
        w=r['survey_exposure_pct'];r['headroom']=None if w is None else 1-w/100
        r['intent_headroom']=None if w is None else r['intent']*r['headroom']
        next_month=str(pd.Period(r['month'],freq='M')+1)
        r['next_survey_exposure_change_pp']=None;r['next_survey_publication']=None
        # 下一份月报通常报告本月末的平均仓位。
        economic_end=pd.Period(r['month'],freq='M').end_time.normalize()
        if w is not None and next_month in e.index and economic_end>=pub:
            er=e.loc[next_month]
            if pd.notna(er.available_date):
                r['next_survey_exposure_change_pp']=float(er.survey_exposure_pct-w)
                r['next_survey_publication']=str(er.available_date)
        rows.append(r)
    panel=pd.DataFrame(rows).sort_values(['available_date','month']).reset_index(drop=True)
    panel.to_csv(OUT/'逐事件特征与随后收益.csv',index=False,encoding='utf-8-sig')
    panel.to_parquet(OUT/'逐事件特征与随后收益.parquet',index=False)
    return panel,m


def rank_corr(d,x,y):
    a=d[[x,y]].dropna()
    if len(a)<3 or a[x].nunique()<2 or a[y].nunique()<2:return None
    return float(spearmanr(a[x],a[y]).statistic)


def analyze():
    spec=check_frozen(); d,m=build_panel()
    checks=[('意向与此前收益','intent','prior_return20'),('意向与随后收益','intent','forward_return20'),
            ('意向乘空间与随后收益','intent_headroom','forward_return20'),('意向与下期调查仓位变化','intent','next_survey_exposure_change_pp'),
            ('意向乘空间与下期调查仓位变化','intent_headroom','next_survey_exposure_change_pp')]
    correlations=[]
    for subset,frame in [('全部历史重建',d),('有日期公开正文',d[d.clock.eq('DATED_ARTICLE')])]:
        for name,x,y in checks:
            a=frame.dropna(subset=[x,y]); omit=[]
            for year in a.month.str[:4].unique():
                z=rank_corr(a[a.month.str[:4]!=year],x,y)
                if z is not None:omit.append(z)
            correlations.append(dict(subset=subset,comparison=name,n=len(a),spearman=rank_corr(a,x,y),
                                     omit_one_year_min=min(omit) if omit else None,omit_one_year_max=max(omit) if omit else None))
    pd.DataFrame(correlations).to_csv(OUT/'固定相关诊断.csv',index=False,encoding='utf-8-sig')
    predictions=[];model_records=[]
    for _,row in d.iterrows():
        hist=d[(d.exit_date<row.decision_date)&(d.available_date<row.available_date)]
        for name,cols in spec['fixed_models'].items():
            required=spec['fixed_models']['C'] if name in ['AC','C'] else spec['fixed_models']['B']
            train=hist.dropna(subset=required+['forward_return20'])
            if len(train)<spec['estimator']['minimum_mature_training_events'] or row[required].isna().any():continue
            scaler=StandardScaler().fit(train[cols]); X=scaler.transform(train[cols])
            model=Ridge(alpha=10.).fit(X,train.forward_return20)
            pred=float(model.predict(scaler.transform(pd.DataFrame([row[cols]])))[0])
            predictions.append(dict(month=row.month,decision_date=row.decision_date,entry_date=row.entry_date,exit_date=row.exit_date,
                                    model=name,n_train=len(train),max_train_exit=train.exit_date.max(),prediction=pred,
                                    training_mean=float(train.forward_return20.mean()),actual=float(row.forward_return20)))
            model_records.append(dict(month=row.month,model=name,features=cols,train_months=train.month.tolist(),
                                      means=scaler.mean_.tolist(),scales=scaler.scale_.tolist(),coefficients=model.coef_.tolist(),intercept=float(model.intercept_)))
    p=pd.DataFrame(predictions,columns=['month','decision_date','entry_date','exit_date','model','n_train','max_train_exit','prediction','training_mean','actual'])
    p.to_csv(OUT/'固定逐时点预测.csv',index=False,encoding='utf-8-sig'); dump(OUT/'固定模型记录.json',model_records)
    evals=[]
    for name in ['B','C']:
        baseline='A' if name=='B' else 'AC'
        z=p[p.model.eq(name)].merge(p[p.model.eq(baseline)],on='month',suffixes=('_added','_base'))
        record={'model':name,'evaluation_events':len(z),'status':'INSUFFICIENT_EVALUATION_NOT_ESTABLISHED','accounts':'NOT_RUN_PREDICTIVE_GATE'}
        if len(z):
            y=z.actual_added.to_numpy(); eb=(y-z.prediction_base.to_numpy())**2; ea=(y-z.prediction_added.to_numpy())**2
            em=(y-z.training_mean_added.to_numpy())**2
            record.update(rmse_base_pp=float(np.sqrt(eb.mean())*100),rmse_added_pp=float(np.sqrt(ea.mean())*100),
                          rmse_mature_mean_pp=float(np.sqrt(em.mean())*100),rmse_zero_pp=float(np.sqrt((y*y).mean())*100),
                          mse_skill_vs_base=float(1-ea.mean()/eb.mean()),mse_skill_vs_mean=float(1-ea.mean()/em.mean()),
                          evaluation_start=str(z.entry_date_added.min()),evaluation_end=str(z.exit_date_added.max()))
        # 本轮源覆盖最多33个可用月份，24成熟训练门槛后必然不足24评价。
        assert len(z)<spec['promotion_gate']['minimum_evaluation_events']
        evals.append(record)
    summary={
        'study_id':spec['study_id'],'completed_at':now(),'status':'COMPLETED_FIXED_HISTORICAL_DIAGNOSTIC_INSUFFICIENT_PREDICTIVE_EVIDENCE',
        'independent_plan_months':len(PLAN_ROWS),'admitted_event_count':len(d),'dated_article_count':int(d.clock.eq('DATED_ARTICLE').sum()),
        'intent_and_exposure_events':int(d.intent_headroom.notna().sum()),'consecutive_exposure_label_pairs':int(d.next_survey_exposure_change_pp.notna().sum()),
        'plan_above100':int(d.plan_index.gt(100).sum()),'plan_equal100':int(d.plan_index.eq(100).sum()),'plan_below100':int(d.plan_index.lt(100).sum()),
        'first_event':d.month.min(),'last_event':d.month.max(),'source_search_leads':len(read(OUT/'receipts/source_downloads.json')),
        'source_saved':sum(r['status']=='SAVED' for r in read(OUT/'receipts/source_downloads.json')),
        'market_daily_points':len(m),'market_last_date':str(m.date.max().date()),'new_model_fits':len(model_records),
        'descriptive_results':correlations,'predictive_comparisons':evals,'accounts_run':0,'strict_forward_days':0,
        'whole_goal_achieved':False,'actual_manager_trade_prediction_tested':False,
        'source_boundary':'非随机缺失、调查样本未知、首版未认证；意向覆盖全部A股主观私募，并非专门针对510300；下期调查仓位只作弱验证。',
    }
    dump(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def plots():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,
                         'figure.facecolor':'#f7f9fc','axes.spines.top':False,'axes.spines.right':False})
    d=pd.read_parquet(OUT/'逐事件特征与随后收益.parquet');m=pd.read_parquet(OUT/'inputs/market.parquet')
    m.date=pd.to_datetime(m.date);m=m[m.date.ge('2014-01-01')]
    a=pd.read_csv(OUT/'全部月份覆盖与缺口.csv');a['date']=pd.to_datetime(a.month+'-01')
    a.loc[~a.status.str.startswith('ADMITTED'),['plan_index','survey_exposure_pct']]=np.nan
    (OUT/'figures').mkdir(exist_ok=True)
    fig,axs=plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[1.2,1,1]})
    fig.subplots_adjust(left=.095,right=.97,top=.89,bottom=.11,hspace=.13)
    fig.suptitle('私募想买、还有空间买，是否能预示后续行情？',x=.095,ha='left',y=.967,fontsize=20,fontweight='bold')
    fig.text(.095,.926,'510300完整日线 × 月度增减仓意向 × 调查仓位｜空白就是缺失，不插值；点按公开日期放置',color='#536176')
    axs[0].plot(m.date,100*m.wealth/m.wealth.iloc[0],color='#2457bf',lw=1.5)
    axs[0].set_ylabel('含分红总回报\n2014起点=100');axs[0].set_title('价格轨迹',loc='left',fontsize=12)
    x=pd.to_datetime(d.available_date)
    for clock,marker,color,label in [('DATED_ARTICLE','o','#0f766e','有日期公开正文'),('PUBLIC_CATALOGUE_CURRENT_PDF','^','#b7791f','目录日期＋当前报告')]:
        z=d[d.clock.eq(clock)];axs[1].scatter(pd.to_datetime(z.available_date),z.plan_index,s=37,marker=marker,color=color,label=label)
    axs[1].axhline(100,color='#8a93a3',ls='--',lw=1);axs[1].set_ylabel('增减仓计划指数');axs[1].legend(loc='upper left',ncol=2,fontsize=9)
    axs[1].set_ylim(91,130)
    e=pd.read_csv(OUT/'月度调查仓位.csv').dropna(subset=['available_date'])
    axs[2].scatter(pd.to_datetime(e.available_date),e.survey_exposure_pct,s=30,color='#8762b2')
    axs[2].set_ylabel('调查平均仓位（%）');axs[2].set_ylim(45,90)
    axs[2].xaxis.set_major_locator(mdates.YearLocator(2));axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for ax in axs:ax.grid(axis='y',alpha=.19);ax.set_axisbelow(True)
    fig.text(.095,.045,'历史资料重建，尚未建立预测价值。调查意愿不是实际订单；平均仓位变化可能来自价格或样本变化。\n20日收益从公开日之后的开盘计算；此图没有策略净值、最佳卖点或资金身份认定。',fontsize=10,color='#536176')
    fig.savefig(OUT/'figures/私募意向仓位与510300走势.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    fig.subplots_adjust(top=.79,bottom=.2,left=.08,right=.97,wspace=.22)
    fig.suptitle('看多意愿与随后收益：逐月观察，不挑成功案例',x=.08,ha='left',y=.97,fontsize=19,fontweight='bold')
    for ax,xcol,title in [(axs[0],'intent','增减仓意向'),(axs[1],'intent_headroom','增减仓意向 × 可加仓空间')]:
        z=d.dropna(subset=[xcol,'forward_return20']);c=rank_corr(z,xcol,'forward_return20')
        ax.scatter(z[xcol]*100,z.forward_return20*100,c=pd.to_datetime(z.available_date).dt.year,cmap='viridis',s=45,alpha=.85)
        ax.axhline(0,color='#a1a8b4',ls='--',lw=1);ax.axvline(0,color='#a1a8b4',ls='--',lw=1)
        ax.set_title(f'{title}\n{len(z)}次观察；秩相关 {c:+.3f}',loc='left',fontsize=12)
        ax.set_xlabel('计划指数 − 100' if xcol=='intent' else '(计划指数 − 100) × (1 − 调查仓位)')
        ax.set_ylabel('下一开盘起20日毛收益（%）');ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.text(.08,.065,'包含缺失和来源限制下的历史诊断；横轴不是实际流入金额。未校正搜集选择偏差，不能据此反号交易。\n固定逐时点预测另有记录；有效评价少于24次，账户和夏普检验未运行。',fontsize=10,color='#536176')
    fig.savefig(OUT/'figures/私募意向与随后20日收益.png',dpi=170);plt.close(fig)
    print('已生成完整走势对照和逐月散点图',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='私募意向固定历史诊断')
    parser.add_argument('action',choices=['curate','freeze','analyze','plots'])
    globals()[parser.parse_args().action]()
