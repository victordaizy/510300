"""按用户纠偏，把已有货币、信用与风险证据重接到指数主表。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_index_mainline_v25'
SOURCES={
 'monthly.csv':'reports/research/510300_credit_recency_structure_v23/results/104个月_信用近期性与完整经营定价背景.csv',
 'basket_risk.csv':'reports/research/510300_constituent_risk_transmission_v10/results/111个观察点_篮子风险与原后续路径.csv',
 'rates.csv':'reports/research/510300_rate_context_generalization_v20/results/104个月_原宏观价格背景与事前事后利率.csv',
 'basket_protocol.json':'reports/research/510300_constituent_risk_transmission_v10/protocol.json',
 'rate_protocol.json':'reports/research/510300_rate_context_generalization_v20/protocol.json',
}


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def freeze():
    if OUT.exists():raise RuntimeError('目录已存在，不覆盖协议。')
    for d in ['inputs','results','code']:(OUT/d).mkdir(parents=True)
    protocol={'at':now(),'study':'510300_INDEX_MAINLINE_V25','user_correction':'你太关注个股了，指数和个股还是不太一样','previous_goal_turn':'PROGRESS：V24补充了局部经营证据，但用户明确要求恢复指数主线。','objective':'在全部原104个月中连接货币来源、近期信用期限和用途、资金价格与整体估值、指数实际价格波动，以及可得的完整篮子共同涨跌。','main_unit':'每次货币发布对应的510300实际走势；任何内部缺失不得删除该指数观察。','scope':'只重接已保存结果，不新增公司、不采集财报、不重新运行旧预测模型或账户。','levels':['货币余额、同比基数和统计口径','信用总量、期限和资金用途','资金价格、整体估值和政策信息时钟','指数价格路径、涨跌波动、共同涨跌程度与参与广度','原E0/E1后20日路径及原辅助期限'],'macro_chain':'M2增加不等于股市净流入；M1变化不等于经营需求的唯一来源；传导需逐层核实，不能用指标投票替代。','aggregate_premium':'整体PE、国债利率、资金利率是观察量；未经预期现金流识别，不命名为已测得风险溢价。','risk_identity':'沿原V10：固定参考权重篮子方差=A+B*rho；新增展示的共同变动项份额=(Var-A)/Var。该份额只是协方差的算术占比，不是宏观解释率或预测R平方。','coverage':'原V10完整300只、25日收益与事前权重准入不变。104行中参考权重与等权结果分别并列，原7个LPR病例不混入。','rates':'连接V20原观察时点两套时钟的利率，保留来源已标注的重建假设。持有期利率仅保留在原输入，不加入本轮事前列。','comparison':'列全部可用期的共同变动份额分布、方差变化两项分解和覆盖状态；不以事后收益筛选或排名。保留原6个既有展示病例2019-03、2020-11、2021-01、2021-05、2024-08、2025-07用于索引说明，不将其当新检验。','frozen_failures':'原离散度和M1/M2预测失败不重跑、不改名复活。','new_models':0,'new_accounts':0,'independent_validation':False,'goal_achieved':False}
    save('protocol.json',protocol)
    inputs=[]
    for name,relative in SOURCES.items():
        source=ROOT/relative; dest=OUT/'inputs'/name;shutil.copy2(source,dest)
        inputs.append(dict(name=name,source=relative,sha256=sha(dest)))
    save('freeze.json',dict(at=now(),protocol_sha256=sha(OUT/'protocol.json'),inputs=inputs))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    print('已按指数主线固定104个月的连接范围。')


def build():
    m=pd.read_csv(OUT/'inputs/monthly.csv'); original=m.copy()
    b=pd.read_csv(OUT/'inputs/basket_risk.csv'); b=b[b.origin_kind.eq('MONTHLY')].copy()
    rates=pd.read_csv(OUT/'inputs/rates.csv')
    ratecols=[c for c in rates if c.startswith(('main_origin_','delay1_origin_'))]
    assert len(m)==104 and len(b)==208 and not m.stat_month.duplicated().any()
    assert not set(ratecols)&set(m)
    m=m.merge(rates[['stat_month']+ratecols],on='stat_month',how='left',validate='one_to_one')
    chosen=['status','member_count','weight_date','current_variance','previous_variance','current_rv','previous_rv','current_net_down','previous_net_down','current_weighted_mean_correlation','previous_weighted_mean_correlation','current_variance_diagonal_A','previous_variance_diagonal_A','current_variance_pair_scale_B','previous_variance_pair_scale_B','variance_change_stock_vol_component','variance_change_correlation_component','stock_fraction_downside_improved','stock_weight_downside_improved','net_downside_falls','individual_downside_energy_rises','current_positive_20d_fraction','current_positive_20d_weight','return_correlation_with_etf20']
    summaries=[]
    for basket,prefix in [('REFERENCE_WEIGHT','index_structure'),('EQUAL','equal_structure')]:
        x=b[b.basket.eq(basket)][['origin_month']+chosen].rename(columns={'origin_month':'stat_month',**{c:prefix+'_'+c for c in chosen}})
        assert len(x)==104
        m=m.merge(x,on='stat_month',how='left',validate='one_to_one').copy()
        var=m[prefix+'_current_variance'];a=m[prefix+'_current_variance_diagonal_A']
        m[prefix+'_cross_covariance_fraction']=(var-a)/var
        valid=m[m[prefix+'_status'].eq('COMPLETE_DIAGNOSTIC_ONLY')]
        share=valid[prefix+'_cross_covariance_fraction']
        change=valid[prefix+'_current_variance']-valid[prefix+'_previous_variance']
        errors=change-valid[prefix+'_variance_change_stock_vol_component']-valid[prefix+'_variance_change_correlation_component']
        assert errors.abs().max()<1e-10
        recovered=valid[prefix+'_current_variance_diagonal_A']+valid[prefix+'_current_variance_pair_scale_B']*valid[prefix+'_current_weighted_mean_correlation']
        assert (recovered-valid[prefix+'_current_variance']).abs().max()<1e-10
        falls=valid[valid[prefix+'_net_downside_falls'].eq(True)]
        summaries.append(dict(basket=basket,complete_months=len(valid),missing_months=104-len(valid),cross_covariance_fraction_min=float(share.min()),cross_covariance_fraction_median=float(share.median()),cross_covariance_fraction_max=float(share.max()),downside_falls=len(falls),falls_with_weighted_individual_energy_up=int(falls[prefix+'_individual_downside_energy_rises'].eq(True).sum()),falls_with_less_than_half_improved=int((falls[prefix+'_stock_fraction_downside_improved']<.5).sum()),max_variance_change_identity_error=float(errors.abs().max())))
    m['risk_premium_identification_status']='NO_VIEW_NOT_IDENTIFIED_BY_PE_MINUS_YIELD'
    m['equity_flow_identification_status']='NO_VIEW_MONEY_BALANCE_NOT_STOCK_NET_INFLOW'
    pd.testing.assert_frame_equal(original.reset_index(drop=True),m[original.columns].reset_index(drop=True),check_dtype=False)
    m.to_csv(OUT/'results/104个月_指数主线与宏观来源全表.csv',index=False,encoding='utf-8-sig')
    cases=['2019-03','2020-11','2021-01','2021-05','2024-08','2025-07']
    m[m.stat_month.isin(cases)].to_csv(OUT/'results/既有六个指数病例_完整背景索引.csv',index=False,encoding='utf-8-sig')
    cols=['stat_month','observation_date','training_regime','index_structure_status','equal_structure_status','index_structure_cross_covariance_fraction','index_structure_current_rv','index_structure_previous_rv','index_structure_current_weighted_mean_correlation','index_structure_previous_weighted_mean_correlation','index_structure_variance_change_stock_vol_component','index_structure_variance_change_correlation_component','past_return20','past_return60','internal_breadth20','spread_pp','delta3_spread_pp','credit_3_corporate_total_yoy_change_yi','credit_3_corporate_long_yoy_change_yi','credit_3_household_long_yoy_change_yi','orders_first_release_value','valuation_original_pe_official','E0_20_return','E1_20_return']
    m[cols].to_csv(OUT/'results/104个月_指数结构与信用背景便览.csv',index=False,encoding='utf-8-sig')
    coverage=m.groupby(['training_regime','index_structure_status','equal_structure_status'],dropna=False).size().reset_index(name='months')
    coverage.to_csv(OUT/'results/全部制度与内部证据覆盖.csv',index=False,encoding='utf-8-sig')
    save('results/aggregate_summary.json',dict(at=now(),observations=104,original_columns_preserved=len(original.columns),known_before_origin_rate_columns=len(ratecols),summaries=summaries,new_predictive_claim=False,individual_company_examples_in_mainline=0))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


def complete():
    frozen=json.loads((OUT/'freeze.json').read_text('utf-8'))
    assert sha(OUT/'protocol.json')==frozen['protocol_sha256']
    for r in frozen['inputs']:assert sha(OUT/'inputs'/r['name'])==r['sha256']
    original=pd.read_csv(OUT/'inputs/monthly.csv')
    combined=pd.read_csv(OUT/'results/104个月_指数主线与宏观来源全表.csv')
    pd.testing.assert_frame_equal(original,combined[original.columns],check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    baskets=pd.read_csv(OUT/'inputs/basket_risk.csv')
    baskets=baskets[baskets.origin_kind.eq('MONTHLY')]
    dates=original.set_index('stat_month').observation_date
    expected_dates=baskets.origin_month.map(dates)
    same_dates=baskets.observation_date.eq(expected_dates)
    both_missing=baskets.observation_date.isna() & expected_dates.isna()
    # 原行情截止之后的观察，两侧日期均缺失；保留其不可观察状态。
    assert (same_dates | both_missing).all()
    assert baskets.loc[both_missing,'status'].eq('NO_VIEW_DATE_OR_MEMBERSHIP').all()
    clock_count=0
    for mode in ['main','delay1']:
        c=combined[mode+'_origin_known_at_assumed'].notna()
        assert (pd.to_datetime(combined.loc[c,mode+'_origin_known_at_assumed'],utc=True)<=pd.to_datetime(combined.loc[c,'snapshot_at'],utc=True)).all()
        clock_count+=int(c.sum())
    assert combined.index_structure_status.notna().all()
    assert len(combined)==104 and combined.index_structure_status.eq('COMPLETE_DIAGNOSTIC_ONLY').sum()==24
    assert (OUT/'指数研究主线_货币来源与整体定价.md').exists()
    save('verification.json',dict(at=now(),status='PASS_INDEX_LEVEL_JOIN_AND_PRESERVED_COVERAGE',index_months=104,original_columns_unchanged=len(original.columns),monthly_basket_date_matches=int(same_dates.sum()),monthly_basket_dates_both_missing_preserved=int(both_missing.sum()),rate_origin_clock_checks=clock_count,reference_weight_complete_months=24,reference_weight_missing_months=80,independent_prediction_validation=False))
    save('completion.json',dict(at=now(),status='COMPLETED_INDEX_MAINLINE_RECONNECTION',previous_goal_turn_classification='PROGRESS',current_goal_turn_classification='PROGRESS',user_correction_applied=True,main_unit='510300实际指数路径，104个月全部保留',company_reports_used_as_index_conclusion=False,new_evidence='将最新近期信用、全体月度指数路径、既有两套利率时钟与完整篮子共同变动重接；24个完整参考篮子协方差项占方差中位数96.48%，仅为算术结构。',goal_status='active',goal_achieved=False,new_models=0,new_accounts=0,orders_authorized=False,global_mandate_modified=False))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    pd.DataFrame([dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_index.csv']).to_csv(OUT/'file_index.csv',index=False,encoding='utf-8-sig')
    print('104个月指数主线已保存，个股病例不再作为主结论；原结果和缺失完整保留。')


if __name__=='__main__':{'freeze':freeze,'build':build,'complete':complete}[sys.argv[1]]()
