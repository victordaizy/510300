"""绝对验收的边界、未定义指标与独立性语义。"""
import math

from research.high_return_sharpe_goal_v1 import evaluate


def complete_metric():
    return {"net_cagr": .10, "net_sharpe": 1.5, "max_drawdown": .10,
            "p_times_b": 1.01, "standard_expectancy_loss_units": .1,
            "mean_cycle_net_return": .01}


def test_exact_user_thresholds_do_not_grant_independent_validation():
    result = evaluate(complete_metric())
    assert result["historical_economic_pass"]
    assert not result["validated_goal_pass"]
    assert result["independent_validation"] == "NOT_ESTABLISHED"
    assert not evaluate({**complete_metric(), "p_times_b": 1.})["historical_economic_pass"]


def test_zero_trade_and_all_winner_undefined_quality_cannot_pass():
    cash = {**complete_metric(), "net_cagr": 0., "net_sharpe": None,
            "p_times_b": None, "standard_expectancy_loss_units": None,
            "mean_cycle_net_return": None, "max_drawdown": 0.}
    assert not evaluate(cash)["historical_economic_pass"]
    # 只有赢家时盈亏比不可估，不能把无穷大当作合格收益证据。
    assert not evaluate({**complete_metric(), "p_times_b": math.inf})["historical_economic_pass"]


def test_one_failed_objective_or_invalid_drawdown_prevents_joint_pass():
    for key, value in (("net_cagr", .099999), ("net_sharpe", 1.499999),
                       ("max_drawdown", .100001), ("max_drawdown", -.01),
                       ("mean_cycle_net_return", 0.), ("standard_expectancy_loss_units", 0.),
                       ("net_sharpe", float("nan"))):
        assert not evaluate({**complete_metric(), key: value})["historical_economic_pass"]
