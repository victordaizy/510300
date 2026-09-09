"""将原件股本快照与当时更早公开的财报配对，保留差异而不前填股数。"""
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity

FACTS=ROOT/'reports/research/510300_forward_eps_csi_facts_v2/annual_eps_forecast_vintages.parquet'
FINANCIALS=ROOT/'data/raw/fundamentals/csi300_financials_point_in_time.parquet'
OUT=ROOT/'reports/research/510300_forward_eps_share_snapshot_reconciliation_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'input_receipt.json',{'started_at':now(),'files':[identity(p) for p in [Path(__file__),FACTS,FINANCIALS]],
                                 'reconstruction_not_immutable_historical_snapshot':True},exclusive=True)
    facts=pd.read_parquet(FACTS).drop_duplicates('report_id')
    reports=facts.loc[facts.share_snapshot_million.notna()].copy()
    reports['report_internal_date']=pd.to_datetime(reports.report_internal_date)
    financials=pd.read_parquet(FINANCIALS)
    financials=financials.loc[financials.total_shares.gt(0)]
    rows=[]
    for r in reports.itertuples():
        eligible=financials.loc[financials.con_code.eq(r.ts_code)&financials.available_at.lt(r.report_internal_date)
                                &financials.report_period.le(r.report_internal_date)].sort_values(['report_period','available_at'])
        item={'report_id':r.report_id,'ts_code':r.ts_code,'report_internal_date':r.report_internal_date,
              'original_snapshot_million_shares':r.share_snapshot_million,'original_snapshot_raw':r.share_snapshot_raw,
              'daily_share_basis_admitted':False,'original_pdf_path':r.raw_pdf_path}
        if eligible.empty:
            item['status']='NO_VIEW_PRIOR_PUBLISHED_FINANCIAL_SHARES_MISSING'
        else:
            selected=eligible.iloc[-1]
            tied=eligible.loc[eligible.report_period.eq(selected.report_period)&eligible.available_at.eq(selected.available_at)]
            if tied.total_shares.nunique()!=1:
                item['status']='NO_VIEW_PRIOR_FINANCIAL_SHARES_CONFLICT'
            else:
                snapshot=float(r.share_snapshot_million)*1e6
                delta=snapshot-float(selected.total_shares)
                item.update({'financial_period':selected.report_period,'financial_available_at':selected.available_at,
                             'balance_ann_date':selected.balance_ann_date,'balance_f_ann_date':selected.balance_f_ann_date,
                             'financial_total_shares':float(selected.total_shares),'difference_shares':delta,
                             'difference_ratio':snapshot/selected.total_shares-1,
                             'status':'WITHIN_ONE_MILLION_SHARE_DISPLAY_HALF_UNIT' if abs(delta)<=500000 else 'DIFFERENCE_REQUIRES_CORPORATE_ACTION_OR_SOURCE_EXPLANATION',
                             'not_proof_no_intervening_share_changes':True})
        rows.append(item)
    frame=pd.DataFrame(rows);frame.to_parquet(OUT/'snapshot_pairs.parquet',index=False)
    frame.to_csv(OUT/'研报股本与更早财报配对.csv',index=False,encoding='utf-8-sig')
    result={'study_id':'510300_FORWARD_EPS_SHARE_SNAPSHOT_RECONCILIATION_V1','completed_at':now(),
            'status':'SHARE_SNAPSHOTS_COMPARED_NO_DAILY_SHARE_IMPUTATION','explicit_report_snapshots':len(frame),
            'companies':int(frame.ts_code.nunique()),'status_counts':frame.status.value_counts().to_dict(),
            'first_snapshot_report_date':str(frame.report_internal_date.min().date()),
            'last_snapshot_report_date':str(frame.report_internal_date.max().date()),
            'financial_panel_share_unit':'股','report_snapshot_unit':'百万股，显示精度有限',
            'ordinary_common_equity_share_definition_verified_for_all_reports':False,
            'daily_shares_created':0,'new_accounts_generated':0,'new_models_fit':0,'new_downloads':0,
            'goal_achieved':False}
    save(OUT/'result.json',result,exclusive=True)
    print('研报股本快照配对已完成：',result['status_counts'],flush=True)


if __name__=='__main__':
    run()
