"""现有含分红价格构造22日低点恐慌距离，波动带回落后进入。"""
import numpy as np
import pandas as pd


def factor_frame(data):
    wealth = pd.Series(data.wealth.to_numpy(float), index=data.index)
    low = wealth*(data.low+data.dividend)/(data.close+data.dividend)
    valid_low = np.isfinite(data.low) & (data.low > 0) & np.isfinite(low) & (low > 0)
    low = low.where(valid_low)
    high = wealth.rolling(22, min_periods=22).max()
    fear = 100*(high-low)/high
    mean = fear.rolling(20, min_periods=20).mean()
    sd = fear.rolling(20, min_periods=20).std(ddof=1)
    upper = mean+2*sd
    price_mean = wealth.rolling(20, min_periods=20).mean()
    valid = np.isfinite(pd.concat([wealth, wealth.shift(1), fear, fear.shift(1), upper, upper.shift(1)], axis=1)).all(axis=1)
    return pd.DataFrame({"date": data.date.to_numpy(), "wealth": wealth.to_numpy(), "wealth_low": low.to_numpy(),
        "highest_close22": high.to_numpy(), "fear22": fear.to_numpy(), "fear_mean20": mean.to_numpy(),
        "fear_sd20": sd.to_numpy(), "fear_upper20": upper.to_numpy(), "price_mean20": price_mean.to_numpy(),
        "factor_valid": valid.to_numpy()})


def trading_rule(factors):
    f = factors
    entry = f.factor_valid & (f.fear22.shift(1) > f.fear_upper20.shift(1)) & (f.fear22 <= f.fear_upper20) & (f.wealth > f.wealth.shift(1))
    price_exit = np.isfinite(f.price_mean20) & np.isfinite(f.wealth) & (f.wealth >= f.price_mean20)
    return {"entry": entry.fillna(False).to_numpy(int), "exit": {1: price_exit.fillna(False).to_numpy(bool)}}
