"""确认现有前瞻EPS对应的历史价格和股数资料覆盖，不构造未校正估值。"""
from pathlib import Path
import sys
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import save,now
from research.forward_eps_guosen_history_v1 import identity

OUT=ROOT/'reports/research/510300_forward_eps_price_share_alignment_inventory_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi/company_month_feature_evidence.parquet'
PRICES=ROOT/'data/raw/constituents/000300_constituent_daily.parquet'
FINANCIALS=ROOT/'data/raw/fundamentals/csi300_financials_point_in_time.parquet'


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'input_receipt.json',{'recorded_at':now(),'files':[identity(p) for p in [Path(__file__),SOURCE,PRICES,FINANCIALS]],
                                 'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0},exclusive=True)
    company=pd.read_parquet(SOURCE)
    selected=company.loc[company.status.eq('CURRENT_FORWARD_EPS_AVAILABLE')].copy()
    fields=['date','con_code','raw_close','outstanding_share','total_market_cap_cny','market_cap_asof_date']
    prices=pd.read_parquet(PRICES,columns=fields)
    assert not prices.duplicated(['date','con_code']).any()
    joined=selected.merge(prices,left_on=['origin','ts_code'],right_on=['date','con_code'],how='left',validate='one_to_one')
    joined['unadjusted_close_available']=joined.raw_close.gt(0)
    joined['historical_share_field_present']=joined.outstanding_share.notna()
    joined['historical_market_cap_field_present']=joined.total_market_cap_cny.notna()
    joined['share_basis_admitted']=False
    joined['admission_note']='股数字段与研报口径、公司行为日期尚未对齐，不构造已校正前瞻估值'
    joined.to_parquet(OUT/'company_month_price_share_coverage.parquet',index=False)
    summaries=[]
    for origin,g in joined.groupby('origin'):
        summaries.append({'origin':origin,'forward_eps_company_count':len(g),
                          'unadjusted_close_company_count':int(g.unadjusted_close_available.sum()),
                          'historical_share_field_count':int(g.historical_share_field_present.sum()),
                          'historical_market_cap_field_count':int(g.historical_market_cap_field_present.sum())})
    pd.DataFrame(summaries).to_csv(OUT/'前瞻EPS对应当月价格与股数覆盖.csv',index=False,encoding='utf-8-sig')
    meta=pq.ParquetFile(FINANCIALS)
    result={'study_id':'510300_FORWARD_EPS_PRICE_SHARE_ALIGNMENT_INVENTORY_V1','completed_at':now(),
            'status':'PRICE_AVAILABLE_SHARE_BASIS_ALIGNMENT_PENDING','forward_eps_company_month_rows':len(joined),
            'unadjusted_month_end_close_matches':int(joined.unadjusted_close_available.sum()),
            'missing_month_end_close_rows':int((~joined.unadjusted_close_available).sum()),
            'historical_share_field_nonnull_rows':int(joined.historical_share_field_present.sum()),
            'historical_market_cap_field_nonnull_rows':int(joined.historical_market_cap_field_present.sum()),
            'price_input_unique_symbols':int(prices.con_code.nunique()),
            'historical_financial_panel_candidate_rows':meta.metadata.num_rows,
            'historical_financial_panel_candidate_columns':meta.schema_arrow.names,
            'financial_panel_total_shares_not_yet_admitted_as_daily_shares':True,
            'why_not_substitute':'财报期末总股本不自动等于研报基准日及交易日股本，需要公告可见时点和增发、转增、回购注销等变化对齐。',
            'corrected_current_forward_pe_created':False,'new_models_fit':0,'new_accounts_generated':0,
            'new_downloads':0,'future_returns_computed':0,'goal_achieved':False}
    save(OUT/'result.json',result,exclusive=True)
    print('现有价格覆盖7092条EPS公司月份，83条缺失；股数字段均空，已记录旧财报股本候选，未构造假定已校正的估值。',flush=True)


if __name__=='__main__':
    run()
