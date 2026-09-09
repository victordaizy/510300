"""从固定EPS报告原文提取发布时估值空间，保留缺失和原始时钟。"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_report_valuation_fields_v1 import parse_front_page

FACTS=ROOT/'reports/research/510300_forward_eps_csi_facts_v2'
ORIGINALS=ROOT/'reports/research/510300_forward_eps_csi_originals_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
OUT=ROOT/'reports/research/510300_forward_eps_report_valuation_source_v1'


def aggregate_valuation(company,monthly,minimum_companies=10):
    if company.duplicated(['origin','ts_code']).any():
        raise ValueError('公司月度记录重复')
    rows=[]
    for row in monthly.itertuples():
        group=company.loc[company.origin.eq(row.origin)]
        values=group.loc[np.isfinite(group.eps_growth)&np.isfinite(group.report_target_midpoint_upside)]
        count=len(values)
        rows.append({'origin':row.origin,'valuation_company_count':count,
                     'report_target_midpoint_upside_median':float(values.report_target_midpoint_upside.median()) if count>=minimum_companies else np.nan,
                     'report_target_age_days_median':float(values.report_age_days.median()) if count else np.nan,
                     'report_target_company_coverage_of_index':count/int(row.actual_index_members) if row.actual_index_members else np.nan})
    output=monthly.merge(pd.DataFrame(rows),on='origin',validate='one_to_one')
    output['all_valuation_features_valid']=output.all_eps_features_valid & output.valuation_company_count.ge(minimum_companies) & np.isfinite(output.report_target_midpoint_upside_median)
    return output


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    config={'study_id':'510300_FORWARD_EPS_REPORT_VALUATION_SOURCE_V1','minimum_valuation_companies':10,
            'only_existing_fixed_eps_reports':True,'target_quote_same_original_report':True,
            'not_current_month_end_fair_value':True,'new_return_reads':0}
    paths=[Path(__file__),ROOT/'research/forward_eps_report_valuation_fields_v1.py',
           ROOT/'tests/test_forward_eps_report_valuation_fields_v1.py',FACTS/'result.json',FACTS/'annual_eps_forecast_vintages.parquet',
           SOURCE/'company_month_feature_evidence.parquet',SOURCE/'monthly_forward_eps_features.parquet']
    save(OUT/'source_protocol.json',{'registered_at':now(),'config':config,'files':[identity(p) for p in paths]},exclusive=True)
    facts=pd.read_parquet(FACTS/'annual_eps_forecast_vintages.parquet').drop_duplicates('report_id').sort_values('report_id')
    receipts=[];rows=[]
    for r in facts.itertuples():
        pagefile=ORIGINALS/'page_texts'/(r.report_id+'.json')
        text=read(pagefile)['pages'][0]
        parsed=parse_front_page(text)
        record={'report_id':r.report_id,'ts_code':r.ts_code,'report_internal_date':r.report_internal_date,
                'information_date':r.conservative_information_date,'original_pdf_path':r.raw_pdf_path,
                'saved_page_text_identity':identity(pagefile),**parsed}
        save(OUT/'document_fields'/(r.report_id+'.json'),record,exclusive=True)
        receipts.append(record['saved_page_text_identity'])
        upside=parsed['target_upside']
        rows.append({'report_id':r.report_id,'ts_code':r.ts_code,'information_date':r.conservative_information_date,
                     'report_reference_price':float(parsed['quote']['value']) if parsed['quote']['value'] is not None else np.nan,
                     'reference_price_source_status':parsed['quote']['status'],
                     'target_lower':float(parsed['target']['lower']) if parsed['target']['status']=='EXPLICIT_TARGET_INTERVAL' else np.nan,
                     'target_upper':float(parsed['target']['upper']) if parsed['target']['status']=='EXPLICIT_TARGET_INTERVAL' else np.nan,
                     'report_target_midpoint_upside':float(upside['midpoint']) if upside is not None else np.nan,
                     'report_total_market_cap_million':float(parsed['market_cap']['value']) if parsed['market_cap']['value'] is not None else np.nan,
                     'target_field_status':parsed['target']['status']})
    frame=pd.DataFrame(rows);frame.to_parquet(OUT/'report_valuation_fields.parquet',index=False)
    save(OUT/'saved_page_text_receipts.json',{'files':receipts},exclusive=True)
    company=pd.read_parquet(SOURCE/'company_month_feature_evidence.parquet')
    joined=company.merge(frame[['report_id','ts_code','report_target_midpoint_upside']],on=['report_id','ts_code'],how='left',validate='many_to_one')
    assert len(joined)==len(company)
    joined.to_parquet(OUT/'company_month_valuation_features.parquet',index=False)
    monthly=pd.read_parquet(SOURCE/'monthly_forward_eps_features.parquet')
    combined=aggregate_valuation(joined,monthly,config['minimum_valuation_companies'])
    combined.to_parquet(OUT/'monthly_valuation_features.parquet',index=False)
    combined.to_csv(OUT/'每月前瞻EPS与研报估值空间.csv',index=False,encoding='utf-8-sig')
    result={**config,'completed_at':now(),'status':'EXPLICIT_REPORT_VALUATION_SOURCE_COMPLETE_WITH_GAPS',
            'reports_examined':len(frame),'reports_with_explicit_quote':int(frame.report_reference_price.notna().sum()),
            'reports_with_explicit_target':int(frame.target_lower.notna().sum()),
            'reports_with_quote_and_target':int(frame.report_target_midpoint_upside.notna().sum()),
            'negative_upside_report_count':int(frame.report_target_midpoint_upside.lt(0).sum()),
            'reports_with_reported_market_cap':int(frame.report_total_market_cap_million.notna().sum()),
            'company_month_rows':len(joined),'monthly_origins':len(combined),
            'valid_valuation_months':int(combined.all_valuation_features_valid.sum()),
            'minimum_company_count_in_valid_months':int(combined.loc[combined.all_valuation_features_valid,'valuation_company_count'].min()) if combined.all_valuation_features_valid.any() else None,
            'first_valid_month':str(combined.loc[combined.all_valuation_features_valid,'origin'].min()),
            'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,'exact_daily_shares_constructed':False,
            'goal_achieved':False}
    save(OUT/'result.json',result,exclusive=True)
    print('同报告报价与明确估值区间提取完成：',result,flush=True)


if __name__=='__main__':
    run()
