"""验证预测年度、原行边界和可用日期，防止把实际值或错期数据作为预测。"""
from copy import deepcopy
import pytest
from research.forward_eps_guosen_history_v1 import parse,exact_row,internal_date


def fixture():
    page='''证券研究报告 | 2024年03月15日
平安银行(000001.SZ) 国信证券 guosen.com.cn
总股本/流通(百万股) 19,406/19,406
盈利预测和财务指标 2022 2023 2024E 2025E 2026E
营业收入(百万元) 1 2 3 4 5
归母净利润(百万元) 10 20 30 40 50
摊薄每股收益(元) 1.49 1.73 2.24 2.40 2.67 总资产收益率 0.1% 0.2%
市盈率(PE) 9 8 7 6 5
资料来源：国信证券预测 注：摊薄每股收益按最新总股本计算'''
    meta={'info_code':'APTEST','company_code':'80000007','security':[{'stock':'000001'}],
          'notice_date':'2024-03-16 00:00:00','eitime':'2024-03-16 09:00:00'}
    row={'infoCode':'APTEST','ts_code':'000001.SZ','stockName':'平安银行','publishDate':'2024-03-15 00:00:00'}
    return [page],meta,row


def test_only_explicit_forecast_years():
    p,m,r=fixture();f=parse(p,m,r)['facts']
    assert [x['target_fiscal_year'] for x in f]==[2024,2025,2026]
    assert [x['eps_value_exact'] for x in f]==['2.24','2.40','2.67']


def test_later_publication_date_and_current_api_year_ignored():
    p,m,r=fixture();r['predictThisYearEps']='2.67';r['currentYear']=2026
    f=parse(p,m,r)['facts'][0]
    assert f['eps_value_exact']=='2.24'
    assert f['conservative_information_date']=='2024-03-16'


def test_share_snapshot_not_weighted_average():
    p,m,r=fixture();f=parse(p,m,r)['facts'][0]
    assert f['share_snapshot_million']=='19406'
    assert f['share_snapshot_is_rounded_report_value']


def test_eps_missing_cell_cannot_take_next_indicator():
    p,m,r=fixture();p[0]=p[0].replace('2.40 2.67','2.40')
    with pytest.raises(ValueError):parse(p,m,r)


def test_extra_cell_rejected():
    with pytest.raises(ValueError):exact_row('摊薄每股收益(元) 1 2 3 4 5 6',['摊薄每股收益'],5,'元')


def test_broken_year_sequence_rejected():
    p,m,r=fixture();p[0]=p[0].replace('2025E','2027E')
    with pytest.raises(ValueError):parse(p,m,r)


def test_prediction_marker_required():
    p,m,r=fixture();p[0]=p[0].replace('2024E','2024').replace('2025E','2025').replace('2026E','2026')
    with pytest.raises(ValueError):parse(p,m,r)


def test_subject_mismatch_rejected():
    p,m,r=fixture();m['security']=[{'stock':'601318'}]
    with pytest.raises(ValueError):parse(p,m,r)


def test_basis_note_required():
    p,m,r=fixture();p[0]=p[0].replace('摊薄每股收益按最新总股本计算','')
    with pytest.raises(ValueError):parse(p,m,r)


def test_two_duplicate_tables_rejected():
    p,m,r=fixture()
    with pytest.raises(ValueError):parse(p+p,m,r)


def test_old_date_spaces():
    assert internal_date('证券研究报告 2019 年 3 月 6 日')=='2019-03-06'


def test_negative_eps_preserved():
    p,m,r=fixture();p[0]=p[0].replace('2.24 2.40 2.67','-0.20 (0.10) 0.05')
    assert [x['eps_value_exact'] for x in parse(p,m,r)['facts']]==['-0.20','-0.10','0.05']
