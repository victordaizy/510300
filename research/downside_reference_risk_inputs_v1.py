"""同一参考下比较零目标下行幅度与全部平方幅度，完整保留未知窗口。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from research.vintage_reference_risk_inputs_v1 import vintage_risk_targets, MULTIPLIER, VOLATILITY

PRIMARY = "DOWNSIDE_REFERENCE_RISK"
CONTROL = "SECOND_MOMENT_RISK_CONTROL"
MODELS = {PRIMARY: "零目标下行风险与固定参考", CONTROL: "全部平方幅度与固定参考对照"}


def moment_risk_frame(data, cfg):
    """分母始终为完整二十个交易日；缺失不能通过条件筛选变成零。"""
    require(cfg["risk_window"] == 20 and cfg["downside_symmetric_scale"] == 2., "本轮平方风险定义不允许改变")
    require(cfg["annual_days"] == 242 and cfg["target_volatility"] == .10, "本轮风险尺度与年化口径不同")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "平方风险行情日历必须递增且无重复")
    for column in ["close", "previous_close", "dividend"]:
        values = data[column].to_numpy(float)
        require(not np.isinf(values).any(), "平方风险价格或分红包含无穷值")
        require(not np.any(np.isfinite(values) & (values < 0 if column == "dividend" else values <= 0)), "平方风险价格或分红不合法")
    returns = ((data.close+data.dividend)/data.previous_close-1).reset_index(drop=True)
    squares = returns.pow(2)
    negative_squares = pd.Series(np.minimum(returns.to_numpy(float), 0.)**2)
    require(not np.isinf(squares).any(), "平方风险数值溢出")
    full_mean = squares.rolling(20, min_periods=20).mean()
    downside_mean = negative_squares.rolling(20, min_periods=20).mean()
    frame = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)), "economic_simple_return": returns,
        "full_second_moment20": full_mean, "downside_second_moment20": downside_mean,
        "complete_return_count20": returns.rolling(20, min_periods=1).count()})
    risk_measures = {PRIMARY: np.sqrt(2.*cfg["annual_days"]*downside_mean), CONTROL: np.sqrt(cfg["annual_days"]*full_mean)}
    for model, risk in risk_measures.items():
        known_positive = np.isfinite(risk) & risk.gt(0)
        observations = pd.Series(np.where(known_positive, np.minimum(1., .10/risk), np.nan))
        frame[f"{model}_risk"] = risk
        frame[f"{model}_multiplier"] = observations.ffill().fillna(1.)
        frame[f"{model}_risk_status"] = np.where(known_positive, "COMPLETE_POSITIVE_RISK_AMPLITUDE", "NO_NEW_POSITIVE_RISK_CARRIED_MULTIPLIER")
    return frame


def moment_reference_targets(data, risk, references, cfg, start, cost_id, model):
    """复用已经核对的费用与收盘配对；只替换预先规定的风险定义。"""
    require(model in MODELS, "没有登记该平方风险设置")
    adapted = pd.DataFrame({"date": risk.date, MULTIPLIER: risk[f"{model}_multiplier"], VOLATILITY: risk[f"{model}_risk"]})
    frame, summary = vintage_risk_targets(data, adapted, references, cfg, start, cost_id)
    frame = frame.rename(columns={"realized_annual_volatility20": "risk_amplitude20"})
    frame["risk_view_status"] = risk[f"{model}_risk_status"].to_numpy()
    frame["model"] = model
    for column in ["economic_simple_return", "full_second_moment20", "downside_second_moment20", "complete_return_count20"]:
        frame[column] = risk[column].to_numpy()
    summary["model"] = model
    return frame, summary
