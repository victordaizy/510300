"""小账户整仓轮换引擎测试。"""

from __future__ import annotations

import pandas as pd

from research.small_account_cross_sectional import (
    SmallAccountCosts,
    run_small_account_open_backtest,
)


def _market() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2025-01-02", periods=6)
    rows = []
    for symbol, price in (("A", 10.0), ("B", 20.0), ("C", 30.0), ("D", 40.0)):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "con_code": symbol,
                    "raw_open": price,
                    "total_return_open": price + offset * 0.1,
                    "total_return_close": price + offset * 0.1 + 0.05,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows), dates


def test_next_open_t_plus_one_and_every_trade_at_least_5000() -> None:
    market, dates = _market()
    targets = pd.DataFrame(
        [
            {"signal_date": dates[0], "con_code": symbol, "selection_rank": rank, "regime": "BULL"}
            for rank, symbol in enumerate(("A", "B", "C"), start=1)
        ]
        + [
            {"signal_date": dates[2], "con_code": symbol, "selection_rank": rank, "regime": "BULL"}
            for rank, symbol in enumerate(("B", "C", "D"), start=1)
        ]
    )
    ledger, trades = run_small_account_open_backtest(
        market,
        targets,
        dates,
        20000.0,
        SmallAccountCosts(cash_annual_rate=0.0),
    )
    assert trades["notional"].ge(5000.0).all()
    assert trades.loc[trades["signal_date"].eq(dates[0]), "date"].eq(dates[1]).all()
    assert ledger["position_count"].max() <= 3


def test_subminimum_exit_is_not_executed() -> None:
    market, dates = _market()
    a_mask = market["con_code"].eq("A")
    a_path = pd.Series([10.0, 10.0, 9.2, 8.5, 7.8, 7.8], index=dates)
    market.loc[a_mask, "raw_open"] = market.loc[a_mask, "date"].map(a_path)
    market.loc[a_mask, "total_return_open"] = market.loc[a_mask, "date"].map(a_path)
    market.loc[a_mask, "total_return_close"] = market.loc[a_mask, "date"].map(a_path)
    targets = pd.DataFrame(
        [
            {"signal_date": dates[0], "con_code": symbol, "selection_rank": rank, "regime": "BULL"}
            for rank, symbol in enumerate(("A", "B", "C"), start=1)
        ]
        + [
            {"signal_date": dates[3], "con_code": symbol, "selection_rank": rank, "regime": "BULL"}
            for rank, symbol in enumerate(("B", "C", "D"), start=1)
        ]
    )
    ledger, trades = run_small_account_open_backtest(
        market,
        targets,
        dates,
        20000.0,
        SmallAccountCosts(cash_annual_rate=0.0),
    )
    sold_a = trades.loc[trades["con_code"].eq("A") & trades["side"].eq("卖出")]
    assert sold_a.empty
    assert ledger["blocked_small_sell_count"].sum() >= 1
