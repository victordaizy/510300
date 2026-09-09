"""独立只读复核前瞻EPS原页、月度特征、成熟标签和已保存账户，不重新拟合。"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import pypdfium2 as pdfium


def read(p):return json.loads(p.read_text('utf-8'))


def compact(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))


def close(a,b,tol=1e-10):
    if a is None or (isinstance(a,float) and np.isnan(a)):
        assert b is None or (isinstance(b,float) and np.isnan(b));return
    assert np.isclose(float(a),float(b),rtol=tol,atol=tol),(a,b)


def verify(root,scope='financial'):
    b=root/'reports/research';fact=b/f'510300_forward_eps_{scope}_facts_v1_1';out=b/f'510300_forward_eps_monthly_policy_v1_{scope}'
    config=read(root/'config/510300_forward_eps_monthly_policy_v1.json');result=read(out/'result.json')
    records=read(fact/'document_outcomes.json')['rows'];df=pd.read_parquet(fact/'annual_eps_forecast_vintages.parquet')
    source_checks=0;fact_checks=0;pdf_seen={}
    for record in records:
        if not record.get('facts'):continue
        source=read(root/record['source_record'])['source'];p=root/source['raw_pdf_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==source['pdf_sha256']
        html=root/source['raw_html_path'];assert hashlib.sha256(html.read_bytes()).hexdigest()==source['html_sha256']
        doc=pdfium.PdfDocument(p);pages=[]
        try:
            for i in range(min(3,len(doc))):
                page=doc[i];tp=page.get_textpage()
                try:pages.append(tp.get_text_range())
                finally:tp.close();page.close()
        finally:doc.close()
        pdf_seen[source['raw_pdf_path']]=source['pdf_sha256'];source_checks+=1
        for f in record['facts']:
            page=pages[f['source_page']-1];column=f['selected_column_one_based']-1
            assert compact(f['source_eps_row']) in compact(page)
            assert compact(f['header_raw']) in compact(page)
            assert f['header'][column]==[f['target_fiscal_year'],'E']
            assert Decimal(f['source_eps_cell'])==Decimal(f['eps_value_exact'])
            saved=df.loc[df.report_id.eq(f['report_id'])&df.target_fiscal_year.eq(f['target_fiscal_year'])]
            assert len(saved)==1 and saved.iloc[0].eps_value_exact==f['eps_value_exact']
            assert json.loads(saved.iloc[0].header_json)==f['header']
            dates=[f['report_internal_date'],f['provider_notice_date'][:10],f['provider_eitime'][:10],f['directory_publish_date'][:10]]
            assert f['conservative_information_date']==max(dates)
            for metric in ['net_profit','pe']:
                if f.get(metric+'_value_exact') is not None:
                    rr=record['table']['optional'][metric]
                    assert compact(rr['raw']) in compact(page)
                    assert Decimal(rr['values'][column])==Decimal(f[metric+'_value_exact'])
            fact_checks+=1
    assert fact_checks==len(df)==read(fact/'result.json')['forecast_eps_facts']
    monthly=pd.read_parquet(out/'monthly_forward_eps_features.parquet');company=pd.read_parquet(out/'company_month_feature_evidence.parquet')
    membership=pd.read_parquet(b/'510300_forward_eps_csi_directory_v1/historical_membership.parquet')
    members={(pd.Timestamp(d),s) for d,s in membership[['membership_date','symbol']].itertuples(index=False,name=None)}
    assert all((pd.Timestamp(d),s) in members for d,s in company[['origin','ts_code']].itertuples(index=False,name=None))
    for row in company.loc[company.information_date.notna()].to_dict('records'):
        assert pd.Timestamp(row['information_date'])<row['origin']
        assert (row['origin']-pd.Timestamp(row['information_date'])).days<=180
        if pd.notna(row.get('prior_information_date')):assert pd.Timestamp(row['prior_information_date'])<row['origin']-pd.Timedelta(days=90)
        assert int(row['target_fiscal_year'])==row['origin'].year+1
    mapping={'eps_growth':'forward_eps_growth_median','profit_revision':'same_year_profit_revision_90_median',
             'reported_earnings_yield':'reported_forward_earnings_yield_median','raw_eps_revision_unadjusted':'raw_eps_revision_90_unadjusted_median'}
    for row in monthly.to_dict('records'):
        g=company.loc[company.origin.eq(row['origin'])]
        for source,key in mapping.items():
            v=pd.to_numeric(g[source],errors='coerce');v=v[np.isfinite(v)]
            assert len(v)==row[source+'_company_count'];close(float(v.median()) if len(v) else None,row[key])
    labels=pd.read_parquet(out/'monthly_features_and_mature_labels.parquet');price=pd.read_parquet(b/'510300_adaptive_allocation_v1/features.parquet')
    dividends=pd.read_csv(root/config['inputs']['dividends'])
    for col in ['record_date','ex_date','payment_date']:dividends[col]=pd.to_datetime(dividends[col])
    label_count=0
    for row in labels.loc[labels.Y60.notna()].itertuples():
        a,bidx=int(row.origin_index)+1,int(row.origin_index)+61
        d1,d2=price.date.iloc[a],price.date.iloc[bidx]
        cash=dividends.loc[(dividends.record_date>=d1)&(dividends.record_date<d2)&(dividends.ex_date<=d2),'cash_dividend_per_share'].sum()
        close(row.Y60,(price.open.iloc[bidx]+cash)/price.open.iloc[a]-1)
        assert row.label_exit_date==d2;label_count+=1
    trained=0
    for r in read(out/'training_receipts.json')['rows']:
        origin=pd.Timestamp(r['origin']);eligible=labels.loc[labels.all_features_valid&(labels.origin<origin)&labels.label_exit_date.notna()&(labels.label_exit_date<=origin)&labels.Y60.notna()]
        expected=eligible.origin.dt.strftime('%Y-%m-%d').tolist()
        assert r['training_months']==expected and r['training_samples']==len(expected) and len(expected)==len(set(expected))
        if r['status']=='TRAINED_MONTHLY_MODEL':
            assert len(expected)>=12 and pd.Timestamp(r['latest_label_exit_date'])<=origin
            p=root/r['model_file']['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['model_file']['sha256'];trained+=1
    assert trained==result['trained_model_count']
    checked=0
    for r in result['all_metrics']:
        ledger=pd.read_parquet(out/'evaluation'/r['cost']/(r['model']+'_ledger.parquet'))
        decisions=pd.read_parquet(out/'evaluation'/r['cost']/(r['model']+'_decisions.parquet'))
        ret=ledger.net_return.to_numpy(float);wealth=np.r_[1,np.cumprod(1+ret)];n=len(ret);years=242
        mean=ret.mean()*years;vol=ret.std(ddof=1)*np.sqrt(years)
        for name,value in [('cumulative_return',wealth[-1]-1),('annualized_return',wealth[-1]**(years/n)-1),
                           ('annualized_arithmetic_mean',mean),('annualized_volatility',vol),
                           ('net_sharpe',mean/vol if vol>1e-15 else None),('max_drawdown',(wealth/np.maximum.accumulate(wealth)-1).min()),
                           ('commission',ledger.commission.sum()),('slippage_cost',ledger.slippage_cost.sum())]:close(value,r[name])
        np.testing.assert_allclose(ledger.equity,ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,atol=1e-6,rtol=0)
        np.testing.assert_allclose(ledger.equity,200000*wealth[1:],atol=1e-6,rtol=0)
        np.testing.assert_allclose(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,atol=1e-6,rtol=0)
        assert ledger.terminal_unliquidated.iloc[-1]==False
        assert decisions.loc[decisions.signal_state.eq('NO_VIEW_KEEP_EXISTING_SHARES'),'requested_quantity'].eq(0).all()
        allowed=set(labels.origin_index)
        assert decisions.loc[~decisions.origin_index.isin(allowed),'requested_quantity'].eq(0).all()
        if r['model']=='BUY_HOLD':
            baseline=pd.read_parquet(root/'reports/research/510300_adaptive_allocation_v1/evaluation'/r['cost']/'BUY_HOLD_ledger.parquet')
            np.testing.assert_array_equal(ledger.equity,baseline.equity)
        checked+=1
    assert checked==result['evaluation_accounts']
    for cid in config['costs']:
        stats=pd.read_parquet(out/f'{cid}_saved_bootstrap_statistics.parquet')
        assert len(stats)==2000
        for col in stats:
            v=stats[col].dropna().to_numpy();got=np.quantile(v,[.025,.975]).tolist() if len(v) else [None,None]
            for a,bv in zip(got,result['uncertainty'][cid][col+'_95_interval']):close(a,bv)
    return {'status':'PASS_READ_ONLY_SAVED_SOURCE_FEATURE_LABEL_ACCOUNT_RECOMPUTATION','scope':scope,
            'pdf_reports_reextracted':source_checks,'eps_facts_checked':fact_checks,'company_month_rows_checked':len(company),
            'monthly_origins_checked':len(monthly),'saved_labels_checked':label_count,'saved_model_receipts_checked':trained,
            'complete_accounts_checked':checked,'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,'random_samples_regenerated':0,
            'security_audit_performed':False,'external_review_performed':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--scope',default='financial',choices=['financial','csi']);ap.add_argument('--output',type=Path);a=ap.parse_args()
    result=verify(a.root,a.scope)
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
