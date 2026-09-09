"""只读公司月度预测资料，记录修正中位数退化原因，不读取账户收益。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_revision_distribution_features_v1 import build_features,revision_statistics

SOURCE = ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
FACTS = ROOT/'reports/research/510300_forward_eps_csi_facts_v2'
OUT = ROOT/'reports/research/510300_forward_eps_revision_diagnostic_v1'


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    paths = [Path(__file__),ROOT/'research/forward_eps_revision_distribution_features_v1.py',
             SOURCE/'company_month_feature_evidence.parquet',SOURCE/'monthly_forward_eps_features.parquet',
             FACTS/'annual_eps_forecast_vintages.parquet',FACTS/'result.json']
    save(OUT/'input_receipt.json',{'started_at':now(),'files':[identity(p) for p in paths],
                                 'account_returns_read':False,'new_accounts':0},exclusive=True)
    company = pd.read_parquet(SOURCE/'company_month_feature_evidence.parquet')
    monthly = pd.read_parquet(SOURCE/'monthly_forward_eps_features.parquet')
    features = build_features(company,monthly)
    facts = pd.read_parquet(FACTS/'annual_eps_forecast_vintages.parquet')
    lookup = {(r.report_id,int(r.target_fiscal_year)):r for r in facts.itertuples()}
    paired = company.loc[np.isfinite(company.profit_revision)].copy()
    for row in paired.itertuples():
        year = int(row.origin.year)+1
        a,b = lookup[(row.report_id,year)],lookup[(row.prior_report_id,year)]
        assert a.ts_code==b.ts_code==row.ts_code
        assert a.net_profit_source_label==b.net_profit_source_label
        assert '百万元' in a.net_profit_source_raw and '百万元' in b.net_profit_source_raw
        assert pd.Timestamp(a.conservative_information_date)<row.origin
        assert pd.Timestamp(b.conservative_information_date)<row.origin-pd.Timedelta(days=90)
        av,bv=float(a.net_profit_value_exact),float(b.net_profit_value_exact)
        expected=2*(av-bv)/(abs(av)+abs(bv))
        np.testing.assert_allclose(row.profit_revision,expected,atol=1e-14,rtol=0)
    company['revision_information_state'] = '比较资料缺失，不作为零修正'
    finite = np.isfinite(company.profit_revision)
    same = company.report_id.eq(company.prior_report_id)
    company.loc[finite & same,'revision_information_state'] = '同一报告延用，没有新报告信息'
    company.loc[finite & ~same & company.profit_revision.eq(0),'revision_information_state'] = '已有新报告，但原文利润预测数字未变'
    company.loc[finite & company.profit_revision.gt(0),'revision_information_state'] = '同年度利润预测上修'
    company.loc[finite & company.profit_revision.lt(0),'revision_information_state'] = '同年度利润预测下修'
    selected = company.loc[company.origin.isin(monthly.loc[monthly.all_eps_features_valid,'origin'])]
    all_stats = revision_statistics(company)
    selected_stats = revision_statistics(selected)
    new_count = selected_stats['comparable_company_count']-selected_stats['same_report_count']
    company.to_parquet(OUT/'classified_company_months.parquet',index=False)
    features.to_parquet(OUT/'monthly_distribution_features.parquet',index=False)
    features.to_csv(OUT/'各月盈利修正分布与更新情况.csv',index=False,encoding='utf-8-sig')
    selected.groupby('revision_information_state').size().rename('公司月份记录数').to_csv(OUT/'有效月份修正状态统计.csv',encoding='utf-8-sig')
    result={'study_id':'510300_FORWARD_EPS_REVISION_DIAGNOSTIC_V1','completed_at':now(),
            'status':'SAVED_COMPANY_MONTH_REVISION_DECOMPOSITION_COMPLETE','company_month_rows':len(company),
            'all_comparable_company_months':all_stats,'valid_model_month_comparable_company_months':selected_stats,
            'valid_model_months':int(monthly.all_eps_features_valid.sum()),
            'revision_median_positive_months':int(features.loc[features.distribution_features_valid,'median_revision'].gt(0).sum()),
            'revision_median_zero_months':int(features.loc[features.distribution_features_valid,'median_revision'].eq(0).sum()),
            'revision_median_negative_months':int(features.loc[features.distribution_features_valid,'median_revision'].lt(0).sum()),
            'new_report_pair_rows_in_valid_months':new_count,'all_5500_source_pairs_year_label_unit_and_clock_checked':True,
            'month_and_company_rows_not_independent_events':True,
            'duplicate_report_pairs_across_months':int(paired.duplicated(['ts_code','report_id','prior_report_id','target_fiscal_year']).sum()),
            'no_absolute_target_year_company_months':int(company.status.eq('NO_VIEW_ABSOLUTE_TARGET_YEAR_MISSING').sum()),
            'account_returns_read':False,'new_models_fit':0,'new_accounts_generated':0,'new_downloads':0,'goal_achieved':False}
    save(OUT/'result.json',result,exclusive=True)
    print('公司月度修正分解完成：',selected_stats,flush=True)


if __name__=='__main__':
    run()
