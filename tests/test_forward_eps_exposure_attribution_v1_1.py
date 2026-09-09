"""验证未交易日期仍持有ETF，以及除息日的分红权益对账。"""
import numpy as np
import pandas as pd
from research.forward_eps_exposure_attribution_v1_1 import market_and_exposures


def test_no_trade_rows_use_daily_shares_and_keep_dividend_entitlement():
    price=pd.DataFrame({'date':pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03']),
                        'open':[10.,10.,10.5],'close':[10.,11.,10.8],'dividend':[0.,0.,.5]})
    ledger=pd.DataFrame({'date':price.date.iloc[1:].reset_index(drop=True),'open':[10.,10.5],'mark':[11.,10.8],
                         'shares_before':[0,100],'shares_after':[100.,np.nan],'shares':[100,100],
                         'dividend_recognized':[0.,50.],'commission':[5.,0.],'slippage_cost':[0.,0.],
                         'equity':[2095.,2125.],'net_return':[95/2000,30/2095],'pnl':[95.,30.]})
    actual=market_and_exposures(ledger,price,2000)
    np.testing.assert_allclose(actual.q_day,[100,100])
    np.testing.assert_allclose(actual.net_return,ledger.net_return,atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual.dividend_residual_cny,[0,0])


def test_explicit_post_trade_shares_must_match_daily_position():
    import pytest
    price=pd.DataFrame({'date':pd.to_datetime(['2025-01-01','2025-01-02']),
                        'open':[10.,10.],'close':[10.,11.],'dividend':[0.,0.]})
    ledger=pd.DataFrame({'date':[price.date.iloc[1]],'open':[10.],'mark':[11.],'shares_before':[0],
                         'shares_after':[100.],'shares':[200],'dividend_recognized':[0.],'commission':[5.],
                         'slippage_cost':[0.],'equity':[2095.],'net_return':[95/2000],'pnl':[95.]})
    with pytest.raises(AssertionError):market_and_exposures(ledger,price,2000)
