"""成分股L2分类资金流标准化测试。"""

from __future__ import annotations

import pandas as pd

from scripts.download_csi300_component_moneyflow_tushare import (
    AMOUNT_COLUMNS,
    VOLUME_COLUMNS,
    normalize_moneyflow,
)


def test_large_and_small_net_amounts_are_derived_in_source_units() -> None:
    row = {column: 0.0 for column in AMOUNT_COLUMNS + VOLUME_COLUMNS}
    row.update(
        {
            "ts_code": "000001.SZ",
            "trade_date": "20240102",
            "buy_lg_amount": 20.0,
            "buy_elg_amount": 30.0,
            "sell_lg_amount": 10.0,
            "sell_elg_amount": 15.0,
            "buy_sm_amount": 12.0,
            "sell_sm_amount": 5.0,
        }
    )
    result = normalize_moneyflow(pd.DataFrame([row]))
    assert result.loc[0, "large_extra_large_net_amount_10k_cny"] == 25.0
    assert result.loc[0, "small_net_amount_10k_cny"] == 7.0
