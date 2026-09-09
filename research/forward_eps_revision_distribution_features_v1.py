"""将同绝对年度利润修正拆为分布、幅度和报告更新率，保留缺失。"""
from __future__ import annotations

import numpy as np
import pandas as pd

REVISION_FEATURES = ['profit_revision_breadth', 'profit_revision_mean', 'profit_report_renewal_fraction']
EPS_FEATURES = ['forward_eps_growth_median', 'reported_forward_earnings_yield_median', *REVISION_FEATURES]


def revision_statistics(group):
    values = pd.to_numeric(group.profit_revision, errors='coerce')
    valid = group.loc[np.isfinite(values)].copy()
    n = len(valid)
    if not n:
        return {'comparable_company_count':0, 'same_report_count':0, 'new_report_unchanged_count':0,
                'up_revision_count':0, 'down_revision_count':0, 'unchanged_count':0,
                'median_revision':np.nan, **{name:np.nan for name in REVISION_FEATURES}}
    if valid.report_id.isna().any() or valid.prior_report_id.isna().any():
        raise ValueError('有限修正值必须有当前和原比较报告')
    same = valid.report_id.eq(valid.prior_report_id)
    if not valid.loc[same,'profit_revision'].eq(0).all():
        raise ValueError('同一报告同一年度出现非零修正')
    up = int(valid.profit_revision.gt(0).sum())
    down = int(valid.profit_revision.lt(0).sum())
    zero = int(valid.profit_revision.eq(0).sum())
    return {'comparable_company_count':n, 'same_report_count':int(same.sum()),
            'new_report_unchanged_count':int((~same & valid.profit_revision.eq(0)).sum()),
            'up_revision_count':up, 'down_revision_count':down, 'unchanged_count':zero,
            'median_revision':float(valid.profit_revision.median()),
            'profit_revision_breadth':(up-down)/n,
            'profit_revision_mean':float(valid.profit_revision.mean()),
            'profit_report_renewal_fraction':float((~same).mean())}


def build_features(company, monthly):
    if company.duplicated(['origin','ts_code']).any() or monthly.origin.duplicated().any():
        raise ValueError('每家公司每月及月末汇总必须唯一')
    available = company.loc[company.target_fiscal_year.notna()]
    if not available.target_fiscal_year.eq(available.origin.dt.year+1).all():
        raise ValueError('前瞻比较混入非下一绝对年度')
    groupmap = {origin:g for origin,g in company.groupby('origin')}
    rows = []
    for row in monthly.itertuples():
        stats = revision_statistics(groupmap.get(row.origin, company.iloc[:0]))
        if stats['comparable_company_count'] != row.profit_revision_company_count:
            raise ValueError('公司修正数与保存月末覆盖数不一致')
        if stats['comparable_company_count']:
            if stats['median_revision'] != row.same_year_profit_revision_90_median:
                raise ValueError('原修正中位数不能从公司记录复现')
        rows.append({'origin':row.origin, **stats})
    out = monthly.merge(pd.DataFrame(rows), on='origin', validate='one_to_one')
    out['distribution_features_valid'] = out.all_eps_features_valid & np.isfinite(out[EPS_FEATURES]).all(axis=1)
    for name in REVISION_FEATURES:
        out.loc[~out.distribution_features_valid,name] = np.nan
    return out
