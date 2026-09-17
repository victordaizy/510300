"""覆盖诊断中零目标被错误转换为未知的真实失败边界。"""
import numpy as np
from research.strategy_review_diagnostics_v1 import capped_internal_targets


def test_risk_cap_keeps_explicit_zero_unknown_and_episode_clock():
    source = np.array([0., .4, .2, 0., np.nan, .5])
    daily, episode = capped_internal_targets(source, [np.nan, 2., .5, np.nan, 1., 3.],
                                              [np.nan, 2., .5, np.nan, 1., .4])
    np.testing.assert_allclose(daily, [0., .4, .1, 0., np.nan, .5], equal_nan=True)
    np.testing.assert_allclose(episode, [0., .4, .2, 0., np.nan, .2], equal_nan=True)


def test_real_saved_zero_targets_survive_risk_cap():
    import pandas as pd
    from research.strategy_review_diagnostics_v1 import MAIN
    for cost in ['BASE', 'STRESS']:
        saved = pd.read_parquet(MAIN / f'factors/account_risk_{cost}.parquet')
        daily, episode = capped_internal_targets(saved.source_target, saved.multiplier60, saved.multiplier30)
        zero = saved.source_target.eq(0).to_numpy()
        assert np.all(daily[zero] == 0.) and np.all(episode[zero] == 0.)
        unknown = saved.source_target.isna().to_numpy()
        assert np.isnan(daily[unknown]).all() and np.isnan(episode[unknown]).all()
